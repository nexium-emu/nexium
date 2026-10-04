use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU64, AtomicU8, AtomicUsize, Ordering};

use crate::sys::{self, Shm};

const CHUNK: usize = 1 << 20;
const MAX_REGIONS: usize = 64;
const SCRATCH_SLOTS: usize = 64;
const POOL_CAP: usize = 16384;
const EMPTY: u8 = 0;
const BUSY: u8 = 1;
const READY: u8 = 2;

struct Region {
    base: usize,
    span: usize,
    len: usize,
    fd: c_int,
    offset: u64,
    states: Box<[AtomicU8]>,
    direct: Box<[AtomicI64]>,
}

static REGIONS: [AtomicPtr<Region>; MAX_REGIONS] = [const { AtomicPtr::new(ptr::null_mut()) }; MAX_REGIONS];
static SCRATCH_BASE: AtomicUsize = AtomicUsize::new(0);
static SCRATCH_USED: [AtomicBool; SCRATCH_SLOTS] = [const { AtomicBool::new(false) }; SCRATCH_SLOTS];
static POOL_LOCK: AtomicBool = AtomicBool::new(false);
static POOL_LEN: AtomicUsize = AtomicUsize::new(0);
static POOL: [AtomicI64; POOL_CAP] = [const { AtomicI64::new(-1) }; POOL_CAP];
static RESIDENT: AtomicUsize = AtomicUsize::new(0);
static CAP: AtomicUsize = AtomicUsize::new(1536 << 20);
static CLOCK: AtomicUsize = AtomicUsize::new(0);
static LOADS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);
static LOAD_NANOS: AtomicU64 = AtomicU64::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);
static INSTALL_LOCK: AtomicBool = AtomicBool::new(false);
static mut OLD_SEGV: libc::sigaction = unsafe { std::mem::zeroed() };
static mut OLD_BUS: libc::sigaction = unsafe { std::mem::zeroed() };

pub struct Stats {
    pub regions: usize,
    pub resident_mib: usize,
    pub cap_mib: usize,
    pub loads: u64,
    pub evictions: u64,
    pub avg_load_us: f64,
}

pub fn stats() -> Stats {
    let loads = LOADS.load(Ordering::Relaxed);
    Stats {
        regions: REGIONS.iter().filter(|r| !r.load(Ordering::Acquire).is_null()).count(),
        resident_mib: RESIDENT.load(Ordering::Relaxed) >> 20,
        cap_mib: CAP.load(Ordering::Relaxed) >> 20,
        loads,
        evictions: EVICTIONS.load(Ordering::Relaxed),
        avg_load_us: if loads == 0 { 0.0 } else { LOAD_NANOS.load(Ordering::Relaxed) as f64 / loads as f64 / 1000.0 },
    }
}

pub fn set_cap_mib(mib: usize) {
    CAP.store(mib.max(64) << 20, Ordering::Relaxed);
}

fn lock(flag: &AtomicBool) {
    while flag.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
    }
}

fn unlock(flag: &AtomicBool) {
    flag.store(false, Ordering::Release);
}

fn now_nanos() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn pool_take() -> Option<i64> {
    lock(&POOL_LOCK);
    let n = POOL_LEN.load(Ordering::Relaxed);
    let out = if n > 0 {
        POOL_LEN.store(n - 1, Ordering::Relaxed);
        Some(POOL[n - 1].load(Ordering::Relaxed))
    } else {
        None
    };
    unlock(&POOL_LOCK);
    out
}

fn pool_give(direct: i64) {
    lock(&POOL_LOCK);
    let n = POOL_LEN.load(Ordering::Relaxed);
    if n < POOL_CAP {
        POOL[n].store(direct, Ordering::Relaxed);
        POOL_LEN.store(n + 1, Ordering::Relaxed);
        unlock(&POOL_LOCK);
    } else {
        unlock(&POOL_LOCK);
        let mut shm = Shm { direct_start: direct, bytes: CHUNK };
        unsafe { sys::ps5_shm_destroy(&mut shm) };
    }
}

fn scratch_acquire() -> usize {
    loop {
        for (i, used) in SCRATCH_USED.iter().enumerate() {
            if used.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return i;
            }
        }
        unsafe { libc::sched_yield() };
    }
}

fn find(addr: usize) -> Option<&'static Region> {
    for slot in &REGIONS {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let r = unsafe { &*p };
        if addr >= r.base && addr < r.base + r.span {
            return Some(r);
        }
    }
    None
}

unsafe fn load_chunk(r: &Region, chunk: usize) -> bool {
    unsafe {
        let started = now_nanos();
        let direct = match pool_take() {
            Some(d) => d,
            None => {
                let mut shm = Shm::default();
                let rc = sys::ps5_shm_create(CHUNK, &mut shm);
                if rc != 0 {
                    crate::klog!("filemap: chunk {chunk}: shm_create rc {rc:#x} resident {} MiB", RESIDENT.load(Ordering::Relaxed) >> 20);
                    return false;
                }
                shm.direct_start
            }
        };
        let shm = Shm { direct_start: direct, bytes: CHUNK };
        let slot = scratch_acquire();
        let scratch = (SCRATCH_BASE.load(Ordering::Relaxed) + slot * CHUNK) as *mut c_void;
        let mut view = ptr::null_mut();
        let rw = sys::SHM_READ | sys::SHM_WRITE;
        let flags = sys::SHM_FIXED | sys::SHM_KEEP_RESERVED;
        let rc = sys::ps5_shm_map(&shm, 0, CHUNK, scratch, rw, flags, &mut view);
        if rc != 0 {
            crate::klog!("filemap: chunk {chunk}: scratch map {scratch:p} rc {rc:#x}");
            SCRATCH_USED[slot].store(false, Ordering::Release);
            pool_give(direct);
            return false;
        }
        let start = chunk * CHUNK;
        let want = CHUNK.min(r.len.saturating_sub(start));
        let mut done = 0usize;
        while done < want {
            let n = libc::pread(
                r.fd,
                (scratch as *mut u8).add(done).cast(),
                want - done,
                (r.offset + (start + done) as u64) as libc::off_t,
            );
            if n <= 0 {
                if n < 0 && *libc::__error() == libc::EINTR {
                    continue;
                }
                crate::klog!("filemap: chunk {chunk}: pread fd {} returned {n} errno {} after {done}/{want}", r.fd, *libc::__error());
                break;
            }
            done += n as usize;
        }
        if done < CHUNK {
            ptr::write_bytes((scratch as *mut u8).add(done), 0, CHUNK - done);
        }
        sys::ps5_shm_unmap(scratch, CHUNK, sys::SHM_KEEP_RESERVED);
        SCRATCH_USED[slot].store(false, Ordering::Release);
        let target = (r.base + start) as *mut c_void;
        let rc = sys::ps5_shm_map(&shm, 0, CHUNK, target, sys::SHM_READ, flags, &mut view);
        if rc != 0 {
            crate::klog!("filemap: chunk {chunk}: target map {target:p} rc {rc:#x}");
            pool_give(direct);
            return false;
        }
        r.direct[chunk].store(direct, Ordering::Release);
        RESIDENT.fetch_add(CHUNK, Ordering::Relaxed);
        LOADS.fetch_add(1, Ordering::Relaxed);
        LOAD_NANOS.fetch_add(now_nanos() - started, Ordering::Relaxed);
        true
    }
}

static PROBE_DIRECT: AtomicI64 = AtomicI64::new(-1);

unsafe fn usable(at: usize) -> bool {
    unsafe {
        let mut direct = PROBE_DIRECT.load(Ordering::Acquire);
        if direct < 0 {
            let mut shm = Shm::default();
            if sys::ps5_shm_create(CHUNK, &mut shm) != 0 {
                return false;
            }
            direct = shm.direct_start;
            PROBE_DIRECT.store(direct, Ordering::Release);
        }
        let shm = Shm { direct_start: direct, bytes: CHUNK };
        let mut view = ptr::null_mut();
        let flags = sys::SHM_FIXED | sys::SHM_KEEP_RESERVED;
        if sys::ps5_shm_map(&shm, 0, CHUNK, at as *mut c_void, sys::SHM_READ, flags, &mut view) != 0 {
            return false;
        }
        sys::ps5_shm_unmap(at as *mut c_void, CHUNK, sys::SHM_KEEP_RESERVED) == 0
    }
}

unsafe fn reserve_usable(span: usize) -> Option<*mut c_void> {
    unsafe {
        let mut rejected: Vec<*mut c_void> = Vec::new();
        let mut found = None;
        let hints = std::iter::once(0usize).chain((0..60).map(|k| 0x10_0000_0000 + k * 0x4_0000_0000));
        for hint in hints {
            let mut base = ptr::null_mut();
            if sys::ps5_vrange_reserve(span, hint as *mut c_void, CHUNK, &mut base) != 0 {
                continue;
            }
            let at = base as usize;
            if usable(at) && usable(at + span / 2) && usable(at + span - CHUNK) {
                found = Some(base);
                break;
            }
            rejected.push(base);
        }
        if !rejected.is_empty() {
            crate::klog!("filemap: skipped {} unusable reservations of {span:#x} bytes (first {:p})", rejected.len(), rejected[0]);
        }
        for base in rejected {
            sys::ps5_vrange_release(base, span);
        }
        found
    }
}

unsafe fn evict_chunk(r: &Region, chunk: usize) {
    unsafe {
        sys::ps5_shm_unmap((r.base + chunk * CHUNK) as *mut c_void, CHUNK, sys::SHM_KEEP_RESERVED);
        let direct = r.direct[chunk].swap(-1, Ordering::AcqRel);
        if direct >= 0 {
            pool_give(direct);
        }
        RESIDENT.fetch_sub(CHUNK, Ordering::Relaxed);
    }
}

unsafe fn evict_until_under_cap(keep: (usize, usize)) {
    let cap = CAP.load(Ordering::Relaxed);
    let mut budget = 1usize << 20;
    while RESIDENT.load(Ordering::Relaxed) > cap && budget > 0 {
        budget -= 1;
        let hand = CLOCK.fetch_add(1, Ordering::Relaxed);
        let region_index = (hand >> 32) % MAX_REGIONS;
        let p = REGIONS[region_index].load(Ordering::Acquire);
        if p.is_null() {
            CLOCK.store(((region_index + 1) % MAX_REGIONS) << 32, Ordering::Relaxed);
            continue;
        }
        let r = unsafe { &*p };
        let chunk = hand & 0xffff_ffff;
        if chunk >= r.states.len() {
            CLOCK.store(((region_index + 1) % MAX_REGIONS) << 32, Ordering::Relaxed);
            continue;
        }
        if (r.base, chunk) == keep {
            continue;
        }
        if r.states[chunk].compare_exchange(READY, BUSY, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            unsafe { evict_chunk(r, chunk) };
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
            r.states[chunk].store(EMPTY, Ordering::Release);
        }
    }
}

pub fn handle_fault(addr: usize) -> bool {
    let Some(r) = find(addr) else { return false };
    let chunk = (addr - r.base) / CHUNK;
    loop {
        match r.states[chunk].load(Ordering::Acquire) {
            READY => return true,
            BUSY => unsafe {
                libc::sched_yield();
            },
            _ => {
                if r.states[chunk].compare_exchange(EMPTY, BUSY, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                    break;
                }
            }
        }
    }
    if !unsafe { load_chunk(r, chunk) } {
        r.states[chunk].store(EMPTY, Ordering::Release);
        return false;
    }
    r.states[chunk].store(READY, Ordering::Release);
    if RESIDENT.load(Ordering::Relaxed) > CAP.load(Ordering::Relaxed) {
        unsafe { evict_until_under_cap((r.base, chunk)) };
    }
    true
}

unsafe extern "C" fn on_fault(signo: c_int, info: *mut libc::siginfo_t, context: *mut c_void) {
    unsafe {
        let addr = *((info as *const u8).add(24) as *const usize);
        if handle_fault(addr) {
            return;
        }
        if find(addr).is_some() {
            crate::klog!("filemap: unserved fault at {addr:#x}");
        }
        let old = if signo == libc::SIGBUS { &raw const OLD_BUS } else { &raw const OLD_SEGV };
        let old = &*old;
        if old.sa_flags & libc::SA_SIGINFO != 0 && old.sa_sigaction > 1 {
            let f: unsafe extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) = std::mem::transmute(old.sa_sigaction);
            f(signo, info, context);
            return;
        }
        if old.sa_sigaction > 1 {
            let f: unsafe extern "C" fn(c_int) = std::mem::transmute(old.sa_sigaction);
            f(signo);
            return;
        }
        libc::sigaction(signo, old, ptr::null_mut());
    }
}

fn install() -> bool {
    if INSTALLED.load(Ordering::Acquire) {
        return true;
    }
    lock(&INSTALL_LOCK);
    if !INSTALLED.load(Ordering::Acquire) {
        unsafe {
            let mut base = ptr::null_mut();
            if sys::ps5_vrange_reserve(SCRATCH_SLOTS * CHUNK, ptr::null_mut(), CHUNK, &mut base) != 0 {
                unlock(&INSTALL_LOCK);
                return false;
            }
            SCRATCH_BASE.store(base as usize, Ordering::Relaxed);
            if let Some(mib) = std::env::var("NEXIUM_PS5_FILE_CACHE_MB").ok().and_then(|v| v.parse().ok()) {
                set_cap_mib(mib);
            }
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as unsafe extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(libc::SIGSEGV, &action, &raw mut OLD_SEGV);
            libc::sigaction(libc::SIGBUS, &action, &raw mut OLD_BUS);
        }
        INSTALLED.store(true, Ordering::Release);
    }
    unlock(&INSTALL_LOCK);
    true
}

const MAX_FDS: usize = 8192;
static PATHS: [AtomicPtr<CString>; MAX_FDS] = [const { AtomicPtr::new(ptr::null_mut()) }; MAX_FDS];
static HANDLE_ROUTE: AtomicU8 = AtomicU8::new(0);

unsafe extern "C" {
    fn __real_open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn __real_close(fd: c_int) -> c_int;
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_open(path: *const c_char, flags: c_int, mode: c_uint) -> c_int {
    unsafe {
        let fd = __real_open(path, flags, mode);
        if fd >= 0 && (fd as usize) < MAX_FDS && !path.is_null() {
            let entry = Box::into_raw(Box::new(CStr::from_ptr(path).to_owned()));
            let old = PATHS[fd as usize].swap(entry, Ordering::AcqRel);
            if !old.is_null() {
                drop(Box::from_raw(old));
            }
        }
        fd
    }
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_close(fd: c_int) -> c_int {
    unsafe {
        if fd >= 0 && (fd as usize) < MAX_FDS {
            let old = PATHS[fd as usize].swap(ptr::null_mut(), Ordering::AcqRel);
            if !old.is_null() {
                drop(Box::from_raw(old));
            }
        }
        __real_close(fd)
    }
}

unsafe fn same_file(a: c_int, b: c_int) -> bool {
    unsafe {
        let mut sa: libc::stat = std::mem::zeroed();
        let mut sb: libc::stat = std::mem::zeroed();
        libc::fstat(a, &mut sa) == 0
            && libc::fstat(b, &mut sb) == 0
            && sa.st_dev == sb.st_dev
            && sa.st_ino == sb.st_ino
            && sa.st_size == sb.st_size
    }
}

unsafe fn own_handle(fd: c_int) -> c_int {
    unsafe {
        let dup = libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0);
        if dup >= 0 {
            if HANDLE_ROUTE.swap(1, Ordering::Relaxed) != 1 {
                crate::klog!("filemap: own handles via fcntl(F_DUPFD_CLOEXEC)");
            }
            return dup;
        }
        let dup_errno = *libc::__error();
        if fd < 0 || fd as usize >= MAX_FDS {
            return -1;
        }
        let entry = PATHS[fd as usize].load(Ordering::Acquire);
        if entry.is_null() {
            crate::klog!("filemap: fd {fd}: F_DUPFD refused (errno {dup_errno}) and no recorded path");
            return -1;
        }
        let path = &*entry;
        let reopened = __real_open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC, 0 as c_uint);
        if reopened < 0 {
            crate::klog!("filemap: reopen {path:?} failed (errno {})", *libc::__error());
            return -1;
        }
        if !same_file(fd, reopened) {
            crate::klog!("filemap: reopened {path:?} is not the mapped file");
            __real_close(reopened);
            return -1;
        }
        if HANDLE_ROUTE.swap(2, Ordering::Relaxed) != 2 {
            crate::klog!("filemap: own handles by reopening the recorded path (F_DUPFD errno {dup_errno})");
        }
        reopened
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_ps5_file_map(fd: c_int, offset: u64, len: usize) -> *mut c_void {
    unsafe {
        if !install() {
            *libc::__error() = libc::ENOMEM;
            return ptr::null_mut();
        }
        let span = len.max(1).div_ceil(CHUNK) * CHUNK;
        let Some(base) = reserve_usable(span) else {
            *libc::__error() = libc::ENOMEM;
            return ptr::null_mut();
        };
        let dup = own_handle(fd);
        if dup < 0 {
            sys::ps5_vrange_release(base, span);
            *libc::__error() = libc::EPERM;
            return ptr::null_mut();
        }
        let chunks = span / CHUNK;
        let region = Box::new(Region {
            base: base as usize,
            span,
            len,
            fd: dup,
            offset,
            states: (0..chunks).map(|_| AtomicU8::new(EMPTY)).collect(),
            direct: (0..chunks).map(|_| AtomicI64::new(-1)).collect(),
        });
        let raw = Box::into_raw(region);
        for slot in &REGIONS {
            if slot.compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                crate::klog!("filemap: fd {fd} offset {offset} len {len} -> {base:p} ({chunks} x 1 MiB chunks, demand paged)");
                return base;
            }
        }
        drop(Box::from_raw(raw));
        __real_close(dup);
        sys::ps5_vrange_release(base, span);
        *libc::__error() = libc::EMFILE;
        ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_ps5_file_unmap(addr: *mut c_void, _len: usize) -> c_int {
    unsafe {
        for slot in &REGIONS {
            let p = slot.load(Ordering::Acquire);
            if p.is_null() || (*p).base != addr as usize {
                continue;
            }
            if slot.compare_exchange(p, ptr::null_mut(), Ordering::AcqRel, Ordering::Relaxed).is_err() {
                continue;
            }
            let r = Box::from_raw(p);
            for chunk in 0..r.states.len() {
                if r.states[chunk].swap(BUSY, Ordering::AcqRel) == READY {
                    evict_chunk(&r, chunk);
                }
            }
            sys::ps5_vrange_release(r.base as *mut c_void, r.span);
            __real_close(r.fd);
            return 0;
        }
        -1
    }
}
