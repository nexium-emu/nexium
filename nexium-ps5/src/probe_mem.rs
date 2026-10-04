use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::sys::{self, ExecRegion, ExecRequest, Shm};

pub type Check = fn() -> Result<String, String>;

pub const CHECKS: &[(&str, Check)] = &[
    ("exec-region", exec_region),
    ("flexible-exec", flexible_exec),
    ("vrange-reserve", vrange_reserve),
    ("shm-mirror", shm_mirror),
    ("fault-remap", fault_remap),
    ("heap-direct", heap_direct),
];

const PAGE: usize = 0x4000;

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

unsafe fn emit(code: *mut u8, bytes: &[u8]) {
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), code, bytes.len()) };
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

const ADD: [u8; 5] = [0x48, 0x8d, 0x04, 0x37, 0xc3];
const MUL: [u8; 8] = [0x48, 0x89, 0xf8, 0x48, 0x0f, 0xaf, 0xc6, 0xc3];

fn exec_region() -> Result<String, String> {
    let anchor = exec_region as fn() -> Result<String, String> as usize;
    let mut detail = Vec::new();
    for bytes in [64usize << 20, 512 << 20] {
        let request = ExecRequest { bytes, anchor, flags: sys::EXEC_NEAR, ..Default::default() };
        let mut region = ExecRegion::default();
        let started = Instant::now();
        let rc = unsafe { sys::ps5_exec_alloc(&request, &mut region) };
        if rc != 0 {
            return Err(format!("ps5_exec_alloc({} MiB) = {rc:#x}", bytes >> 20));
        }
        let alloc_ms = ms(started);
        let write = region.write_view as *mut u8;
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(region.base) };
        unsafe { emit(write, &ADD) };
        let add = f(40, 2);
        unsafe { emit(write, &MUL) };
        let mul = f(6, 9);
        let tail = region.bytes - 16;
        unsafe { emit(write.add(tail), &ADD) };
        let g: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute((region.base as *mut u8).add(tail)) };
        let tail_result = g(1, 2);
        let started = Instant::now();
        let mut acc = 0u64;
        for i in 0..10_000u64 {
            unsafe { emit(write, if i & 1 == 0 { &ADD[..] } else { &MUL[..] }) };
            acc = acc.wrapping_add(f(i, 3));
        }
        let smc_us = started.elapsed().as_secs_f64() * 1e6 / 10_000.0;
        let expected: u64 = (0..10_000u64).map(|i| if i & 1 == 0 { i + 3 } else { i * 3 }).fold(0, u64::wrapping_add);
        let distance = region.base as i64 - anchor as i64;
        let dual = region.base != region.write_view;
        unsafe { sys::ps5_exec_free(&mut region) };
        if add != 42 || mul != 54 || tail_result != 3 || acc != expected {
            return Err(format!("results add={add} mul={mul} tail={tail_result} smc={acc}/{expected}"));
        }
        detail.push(format!(
            "{}MiB alloc={alloc_ms:.1}ms base-anchor={:+}MiB dual_view={dual} rewrite+call={smc_us:.2}us",
            bytes >> 20,
            distance >> 20
        ));
    }
    let (mut regions, mut live) = (0u64, 0u64);
    unsafe { sys::ps5_exec_live(&mut regions, &mut live) };
    detail.push(format!("live after free: {regions} regions {live} bytes"));
    Ok(detail.join("; "))
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

fn flexible_exec() -> Result<String, String> {
    unsafe {
        let p = libc::mmap(ptr::null_mut(), PAGE, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
        if p == libc::MAP_FAILED {
            return Err(format!("mmap rw errno {}", errno()));
        }
        let rwx = libc::mprotect(p, PAGE, libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC);
        let rwx_errno = if rwx != 0 { errno() } else { 0 };
        let rx = libc::mprotect(p, PAGE, libc::PROT_READ | libc::PROT_EXEC);
        let rx_errno = if rx != 0 { errno() } else { 0 };
        libc::munmap(p, PAGE);
        let q = libc::mmap(
            ptr::null_mut(),
            PAGE,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        );
        let map_errno = if q == libc::MAP_FAILED { errno() } else { 0 };
        if q != libc::MAP_FAILED {
            libc::munmap(q, PAGE);
        }
        let none = libc::mmap(ptr::null_mut(), 64 << 30, libc::PROT_NONE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0);
        let none_ok = none != libc::MAP_FAILED;
        let none_errno = if none_ok { 0 } else { errno() };
        if none_ok {
            libc::munmap(none, 64 << 30);
        }
        Ok(format!(
            "flexible mprotect RWX rc={rwx} errno={rwx_errno}; RX rc={rx} errno={rx_errno}; mmap RWX errno={map_errno}; mmap PROT_NONE 64GiB ok={none_ok} errno={none_errno}"
        ))
    }
}

fn vrange_reserve() -> Result<String, String> {
    let mut detail = Vec::new();
    let mut largest = 0usize;
    for shift in [40u32, 38, 37, 36, 35, 34, 33] {
        let bytes = 1usize << shift;
        let mut base = ptr::null_mut();
        let rc = unsafe { sys::ps5_vrange_reserve(bytes, ptr::null_mut(), 1 << 21, &mut base) };
        if rc == 0 {
            largest = largest.max(bytes);
            detail.push(format!("{}GiB@{base:p}", bytes >> 30));
            unsafe { sys::ps5_vrange_release(base, bytes) };
        } else {
            detail.push(format!("{}GiB:{rc:#x}", bytes >> 30));
        }
    }
    if largest < (16 << 30) {
        return Err(format!("no reservation of 16 GiB or more: {}", detail.join(" ")));
    }
    Ok(detail.join(" "))
}

fn shm_mirror() -> Result<String, String> {
    unsafe {
        let span = 1usize << 30;
        let mut base = ptr::null_mut();
        let rc = sys::ps5_vrange_reserve(span, ptr::null_mut(), 1 << 21, &mut base);
        if rc != 0 {
            return Err(format!("vrange {rc:#x}"));
        }
        let mut shm = Shm::default();
        let size = 64usize << 20;
        let rc = sys::ps5_shm_create(size, &mut shm);
        if rc != 0 {
            sys::ps5_vrange_release(base, span);
            return Err(format!("shm_create {rc:#x}"));
        }
        let rw = sys::SHM_READ | sys::SHM_WRITE;
        let flags = sys::SHM_FIXED | sys::SHM_KEEP_RESERVED;
        let a_addr = base;
        let b_addr = (base as *mut u8).add(512 << 20) as *mut c_void;
        let (mut a, mut b) = (ptr::null_mut(), ptr::null_mut());
        let ra = sys::ps5_shm_map(&shm, 0, size, a_addr, rw, flags, &mut a);
        let rb = sys::ps5_shm_map(&shm, 0, size, b_addr, rw, flags, &mut b);
        let result;
        if ra != 0 || rb != 0 || a != a_addr || b != b_addr {
            result = Err(format!("map a={ra:#x}@{a:p} b={rb:#x}@{b:p}"));
        } else {
            let offsets = [0usize, 8, PAGE - 8, PAGE, 0x123450, size - 8];
            for (i, &off) in offsets.iter().enumerate() {
                ptr::write_volatile((a as *mut u8).add(off) as *mut u64, 0xA5A5_0000 + i as u64);
            }
            let mismatch = offsets.iter().enumerate().find(|(i, &off)| {
                ptr::read_volatile((b as *mut u8).add(off) as *const u64) != 0xA5A5_0000 + *i as u64
            });
            let c_addr = (base as *mut u8).add(256 << 20) as *mut c_void;
            let mut c = ptr::null_mut();
            let r4k = sys::ps5_shm_map(&shm, 0x1000, PAGE, c_addr, rw, flags, &mut c);
            if r4k == 0 {
                sys::ps5_shm_unmap(c, PAGE, sys::SHM_KEEP_RESERVED);
            }
            let r16k = sys::ps5_shm_map(&shm, PAGE, PAGE, c_addr, rw, flags, &mut c);
            let page_ok = r16k == 0 && ptr::read_volatile((c as *mut u8) as *const u64) == 0xA5A5_0003;
            if r16k == 0 {
                sys::ps5_shm_unmap(c, PAGE, sys::SHM_KEEP_RESERVED);
            }
            let d_addr = (base as *mut u8).add(256 << 20).add(0x1000) as *mut c_void;
            let mut d = ptr::null_mut();
            let rmis = sys::ps5_shm_map(&shm, 0, PAGE, d_addr, rw, flags, &mut d);
            if rmis == 0 {
                sys::ps5_shm_unmap(d, PAGE, sys::SHM_KEEP_RESERVED);
            }
            result = match mismatch {
                Some((i, _)) => Err(format!("mirror mismatch at index {i}")),
                None if !page_ok => Err(format!("16 KiB view map={r16k:#x} content wrong")),
                None => Ok(format!(
                    "64MiB object mirrored at +0 and +512MiB; 16KiB view ok; 4KiB-offset view rc={r4k:#x}; 4KiB-misaligned address rc={rmis:#x}; direct_start={:#x}",
                    shm.direct_start
                )),
            };
        }
        if !a.is_null() {
            sys::ps5_shm_unmap(a, size, sys::SHM_KEEP_RESERVED);
        }
        if !b.is_null() {
            sys::ps5_shm_unmap(b, size, sys::SHM_KEEP_RESERVED);
        }
        sys::ps5_shm_destroy(&mut shm);
        sys::ps5_vrange_release(base, span);
        result
    }
}

static FAULT_BASE: AtomicUsize = AtomicUsize::new(0);
static FAULT_LEN: AtomicUsize = AtomicUsize::new(0);
static FAULT_SHM_START: AtomicU64 = AtomicU64::new(0);
static FAULT_SHM_BYTES: AtomicUsize = AtomicUsize::new(0);
static FAULT_HITS: AtomicUsize = AtomicUsize::new(0);
static FAULT_SIGNO: AtomicI32 = AtomicI32::new(0);
static FAULT_RIP: AtomicU64 = AtomicU64::new(0);
static FAULT_RSP: AtomicU64 = AtomicU64::new(0);
static FAULT_SI_ADDR: AtomicU64 = AtomicU64::new(0);
static FAULT_MC_ADDR: AtomicU64 = AtomicU64::new(0);
static FAULT_MAP_RC: AtomicI32 = AtomicI32::new(0);
static mut OLD_SEGV: libc::sigaction = unsafe { std::mem::zeroed() };
static mut OLD_BUS: libc::sigaction = unsafe { std::mem::zeroed() };

unsafe extern "C" fn on_fault(signo: c_int, info: *mut libc::siginfo_t, context: *mut c_void) {
    unsafe {
        let addr = *((info as *const u8).add(24) as *const usize);
        let base = FAULT_BASE.load(Ordering::Relaxed);
        let len = FAULT_LEN.load(Ordering::Relaxed);
        if base != 0 && addr >= base && addr < base + len {
            let page = addr & !(PAGE - 1);
            let shm = Shm {
                direct_start: FAULT_SHM_START.load(Ordering::Relaxed) as i64,
                bytes: FAULT_SHM_BYTES.load(Ordering::Relaxed),
            };
            let mut view = ptr::null_mut();
            let rc = sys::ps5_shm_map(
                &shm,
                page - base,
                PAGE,
                page as *mut c_void,
                sys::SHM_READ | sys::SHM_WRITE,
                sys::SHM_FIXED | sys::SHM_KEEP_RESERVED,
                &mut view,
            );
            FAULT_MAP_RC.store(rc, Ordering::Relaxed);
            FAULT_SIGNO.store(signo, Ordering::Relaxed);
            FAULT_RIP.store(*sys::context_reg(context, sys::MC_RIP), Ordering::Relaxed);
            FAULT_RSP.store(*sys::context_reg(context, sys::MC_RSP), Ordering::Relaxed);
            FAULT_MC_ADDR.store(*sys::context_reg(context, sys::MC_ADDR), Ordering::Relaxed);
            FAULT_SI_ADDR.store(addr as u64, Ordering::Relaxed);
            FAULT_HITS.fetch_add(1, Ordering::Relaxed);
            if rc == 0 {
                return;
            }
        }
        let old = if signo == libc::SIGBUS { &raw const OLD_BUS } else { &raw const OLD_SEGV };
        libc::sigaction(signo, old, ptr::null_mut());
    }
}

#[inline(never)]
fn faulting_read(p: *const u64) -> u64 {
    unsafe { ptr::read_volatile(p) }
}

fn fault_remap() -> Result<String, String> {
    unsafe {
        let span = 64usize << 20;
        let mut base = ptr::null_mut();
        if sys::ps5_vrange_reserve(span, ptr::null_mut(), 1 << 21, &mut base) != 0 {
            return Err("vrange".into());
        }
        let mut shm = Shm::default();
        if sys::ps5_shm_create(span, &mut shm) != 0 {
            sys::ps5_vrange_release(base, span);
            return Err("shm_create".into());
        }
        let mut writer_base = ptr::null_mut();
        sys::ps5_vrange_reserve(span, ptr::null_mut(), 1 << 21, &mut writer_base);
        let mut writer = ptr::null_mut();
        let rc = sys::ps5_shm_map(
            &shm,
            0,
            span,
            writer_base,
            sys::SHM_READ | sys::SHM_WRITE,
            sys::SHM_FIXED | sys::SHM_KEEP_RESERVED,
            &mut writer,
        );
        if rc != 0 {
            return Err(format!("writer map {rc:#x}"));
        }
        let offsets = [0usize, 3 * PAGE + 8, 0x123450, span - 8];
        for (i, &off) in offsets.iter().enumerate() {
            ptr::write_volatile((writer as *mut u8).add(off) as *mut u64, 0xFEED_0000 + i as u64);
        }
        FAULT_BASE.store(base as usize, Ordering::SeqCst);
        FAULT_LEN.store(span, Ordering::SeqCst);
        FAULT_SHM_START.store(shm.direct_start as u64, Ordering::SeqCst);
        FAULT_SHM_BYTES.store(shm.bytes, Ordering::SeqCst);
        FAULT_HITS.store(0, Ordering::SeqCst);
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = on_fault as unsafe extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) as usize;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(libc::SIGSEGV, &action, &raw mut OLD_SEGV);
        libc::sigaction(libc::SIGBUS, &action, &raw mut OLD_BUS);
        let started = Instant::now();
        let mut values = Vec::new();
        for &off in &offsets {
            values.push(faulting_read((base as *mut u8).add(off) as *const u64));
        }
        let again = faulting_read((base as *mut u8).add(offsets[1]) as *const u64);
        let elapsed_us = started.elapsed().as_secs_f64() * 1e6;
        libc::sigaction(libc::SIGSEGV, &raw const OLD_SEGV, ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &raw const OLD_BUS, ptr::null_mut());
        FAULT_BASE.store(0, Ordering::SeqCst);
        let hits = FAULT_HITS.load(Ordering::SeqCst);
        let rip = FAULT_RIP.load(Ordering::SeqCst) as usize;
        let reader = faulting_read as fn(*const u64) -> u64 as usize;
        let detail = format!(
            "hits={hits} signo={} rip={rip:#x} (reader {reader:#x}, +{}) rsp={:#x} si_addr={:#x} mc_addr={:#x} map_rc={:#x} total={elapsed_us:.0}us",
            FAULT_SIGNO.load(Ordering::SeqCst),
            rip.wrapping_sub(reader),
            FAULT_RSP.load(Ordering::SeqCst),
            FAULT_SI_ADDR.load(Ordering::SeqCst),
            FAULT_MC_ADDR.load(Ordering::SeqCst),
            FAULT_MAP_RC.load(Ordering::SeqCst)
        );
        sys::ps5_shm_unmap(writer, span, sys::SHM_KEEP_RESERVED);
        sys::ps5_vrange_release(writer_base, span);
        for &off in &offsets {
            sys::ps5_shm_unmap((base as *mut u8).add(off & !(PAGE - 1)) as *mut c_void, PAGE, sys::SHM_KEEP_RESERVED);
        }
        sys::ps5_shm_destroy(&mut shm);
        sys::ps5_vrange_release(base, span);
        let expected: Vec<u64> = (0..offsets.len() as u64).map(|i| 0xFEED_0000 + i).collect();
        if values != expected || again != expected[1] {
            return Err(format!("values {values:x?} again {again:#x}; {detail}"));
        }
        if hits != offsets.len() || rip < reader || rip > reader + 64 {
            return Err(format!("unexpected context: {detail}"));
        }
        Ok(detail)
    }
}

fn heap_direct() -> Result<String, String> {
    let before_flex = sys::flexible_available().unwrap_or(0);
    let mut before = sys::HeapStats::default();
    unsafe { sys::ps5_heap_stats(&mut before) };
    let block: Vec<u8> = vec![1u8; 2 << 30];
    let touched: u64 = block.iter().step_by(PAGE).map(|&b| b as u64).sum();
    let mut during = sys::HeapStats::default();
    unsafe { sys::ps5_heap_stats(&mut during) };
    let during_flex = sys::flexible_available().unwrap_or(0);
    drop(block);
    if touched != (2u64 << 30) / PAGE as u64 {
        return Err(format!("touch sum {touched}"));
    }
    Ok(format!(
        "2GiB Vec: heap mapped {}->{} MiB (range {:#x}+{}GiB, arenas {}, libc fallbacks {}); flexible free {}->{} MiB",
        before.mapped_bytes >> 20,
        during.mapped_bytes >> 20,
        during.range_base,
        during.range_bytes >> 30,
        during.arenas,
        during.libc_fallbacks,
        before_flex >> 20,
        during_flex >> 20
    ))
}
