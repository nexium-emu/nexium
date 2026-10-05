use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};

const DIRTY: u8 = 0;
const CLEAN: u8 = 1;
const GUEST_PAGE: u64 = 4096;

struct Entry {
    guest_base: u64,
    host_base: usize,
    len: usize,
    owner: bool,
    page_lo: usize,
    page_hi: usize,
    states: Box<[AtomicU8]>,
}

struct Registry {
    entries: UnsafeCell<Vec<Entry>>,
}

unsafe impl Sync for Registry {}

static REGISTRY: Registry = Registry {
    entries: UnsafeCell::new(Vec::new()),
};
static LOCK: AtomicBool = AtomicBool::new(false);
static ENABLED: AtomicU8 = AtomicU8::new(0);
static HOST_PAGE: AtomicUsize = AtomicUsize::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);
static INSTALL_LOCK: AtomicBool = AtomicBool::new(false);
static FAULTS: AtomicU64 = AtomicU64::new(0);
static PROTECTS: AtomicU64 = AtomicU64::new(0);
static PROTECT_FAILURES: AtomicU64 = AtomicU64::new(0);
static mut OLD_SEGV: libc::sigaction = unsafe { std::mem::zeroed() };
static mut OLD_BUS: libc::sigaction = unsafe { std::mem::zeroed() };

pub struct Stats {
    pub regions: usize,
    pub faults: u64,
    pub protects: u64,
    pub protect_failures: u64,
}

pub fn stats() -> Stats {
    lock(&LOCK);
    let regions = unsafe { (*REGISTRY.entries.get()).len() };
    unlock(&LOCK);
    Stats {
        regions,
        faults: FAULTS.load(Ordering::Relaxed),
        protects: PROTECTS.load(Ordering::Relaxed),
        protect_failures: PROTECT_FAILURES.load(Ordering::Relaxed),
    }
}

pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Acquire) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("NEXIUM_SOFT_WRITE_WATCH")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            ENABLED.store(if on { 1 } else { 2 }, Ordering::Release);
            on
        }
    }
}

fn host_page() -> usize {
    let cached = HOST_PAGE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    HOST_PAGE.store(page, Ordering::Relaxed);
    page
}

fn lock(flag: &AtomicBool) {
    while flag
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }
}

fn unlock(flag: &AtomicBool) {
    flag.store(false, Ordering::Release);
}

fn protect(page: usize, len: usize, writable: bool) -> bool {
    let prot = if writable {
        libc::PROT_READ | libc::PROT_WRITE
    } else {
        libc::PROT_READ
    };
    unsafe { libc::mprotect(page as *mut libc::c_void, len, prot) == 0 }
}

pub fn register(guest_base: u64, host: *mut u8, len: usize, owner: bool) {
    if !enabled() || len == 0 {
        return;
    }
    let page = host_page();
    let host_base = host as usize;
    let page_lo = host_base.div_ceil(page) * page;
    let page_hi = ((host_base + len) / page * page).max(page_lo);
    let pages = if owner { (page_hi - page_lo) / page } else { 0 };
    let states: Box<[AtomicU8]> = (0..pages).map(|_| AtomicU8::new(DIRTY)).collect();
    let entry = Entry { guest_base, host_base, len, owner, page_lo, page_hi, states };
    lock(&LOCK);
    unsafe { (*REGISTRY.entries.get()).push(entry) };
    unlock(&LOCK);
}

pub fn unregister(guest_base: u64, host: *mut u8, len: usize) {
    if !enabled() {
        return;
    }
    let page = host_page();
    lock(&LOCK);
    let entries = unsafe { &mut *REGISTRY.entries.get() };
    let removed = entries
        .iter()
        .position(|e| e.guest_base == guest_base && e.host_base == host as usize && e.len == len)
        .map(|index| entries.swap_remove(index));
    if let Some(entry) = &removed {
        for (index, state) in entry.states.iter().enumerate() {
            if state.load(Ordering::Acquire) == CLEAN {
                protect(entry.page_lo + index * page, page, true);
                state.store(DIRTY, Ordering::Release);
            }
        }
    }
    unlock(&LOCK);
    drop(removed);
}

fn owner_of(entries: &[Entry], addr: usize, page: usize) -> Option<&Entry> {
    entries
        .iter()
        .find(|e| e.owner && addr >= e.page_lo && addr + page <= e.page_hi)
}

fn push_guest_pages(out: &mut Vec<u64>, lo: u64, hi: u64) {
    let mut page = lo & !(GUEST_PAGE - 1);
    while page < hi {
        if out.last() != Some(&page) {
            out.push(page);
        }
        page += GUEST_PAGE;
    }
}

pub fn take(va: u64, len: usize, dirty: &mut Vec<u64>) -> Option<bool> {
    if len == 0 || !enabled() {
        return None;
    }
    let end = va.checked_add(len as u64)?;
    install();
    let page = host_page();
    let lo = va & !(GUEST_PAGE - 1);
    let hi = end.checked_add(GUEST_PAGE - 1)? & !(GUEST_PAGE - 1);
    let start = dirty.len();
    lock(&LOCK);
    let entries = unsafe { &*REGISTRY.entries.get() };
    let Some(entry) = entries
        .iter()
        .find(|e| e.guest_base <= lo && hi <= e.guest_base + e.len as u64)
    else {
        unlock(&LOCK);
        return None;
    };
    let host_lo = entry.host_base + (lo - entry.guest_base) as usize;
    let host_hi = host_lo + (hi - lo) as usize;
    let to_guest = |host: usize| entry.guest_base + (host - entry.host_base) as u64;
    let mut owner: Option<&Entry> = None;
    let mut batch = ProtectBatch::new(page);
    let mut at = host_lo / page * page;
    while at < host_hi {
        if !owner.is_some_and(|o| at >= o.page_lo && at + page <= o.page_hi) {
            owner = owner_of(entries, at, page);
        }
        let was_dirty = match owner {
            Some(o) => {
                let state = &o.states[(at - o.page_lo) / page];
                if state.load(Ordering::Acquire) == CLEAN {
                    false
                } else {
                    batch.add(at, state);
                    true
                }
            }
            None => true,
        };
        if was_dirty {
            let a = at.max(host_lo);
            let b = (at + page).min(host_hi);
            push_guest_pages(dirty, to_guest(a), to_guest(b));
        }
        at += page;
    }
    batch.flush();
    unlock(&LOCK);
    Some(dirty.len() > start)
}

struct ProtectBatch<'a> {
    page: usize,
    start: usize,
    states: [Option<&'a AtomicU8>; 64],
    count: usize,
}

impl<'a> ProtectBatch<'a> {
    fn new(page: usize) -> Self {
        Self { page, start: 0, states: [None; 64], count: 0 }
    }

    fn add(&mut self, at: usize, state: &'a AtomicU8) {
        if self.count == self.states.len() || (self.count > 0 && self.start + self.count * self.page != at) {
            self.flush();
        }
        if self.count == 0 {
            self.start = at;
        }
        self.states[self.count] = Some(state);
        self.count += 1;
    }

    fn flush(&mut self) {
        if self.count == 0 {
            return;
        }
        if protect(self.start, self.count * self.page, false) {
            for state in self.states[..self.count].iter().flatten() {
                state.store(CLEAN, Ordering::Release);
            }
            PROTECTS.fetch_add(1, Ordering::Relaxed);
        } else {
            PROTECT_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
        self.count = 0;
    }
}

fn handle(addr: usize) -> bool {
    let page = host_page();
    let at = addr / page * page;
    lock(&LOCK);
    let entries = unsafe { &*REGISTRY.entries.get() };
    let handled = match owner_of(entries, at, page) {
        Some(o) => {
            let index = (at - o.page_lo) / page;
            let state = &o.states[index];
            if state.load(Ordering::Acquire) == CLEAN {
                protect(at, page, true);
                state.store(DIRTY, Ordering::Release);
                FAULTS.fetch_add(1, Ordering::Relaxed);
            }
            true
        }
        None => false,
    };
    unlock(&LOCK);
    handled
}

unsafe extern "C" fn on_fault(signo: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    unsafe {
        let addr = *((info as *const u8).add(24) as *const usize);
        if handle(addr) {
            return;
        }
        let old = if signo == libc::SIGBUS { &raw const OLD_BUS } else { &raw const OLD_SEGV };
        let old = &*old;
        if old.sa_flags & libc::SA_SIGINFO != 0 && old.sa_sigaction > 1 {
            let f: unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                std::mem::transmute(old.sa_sigaction);
            f(signo, info, context);
            return;
        }
        if old.sa_sigaction > 1 {
            let f: unsafe extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
            f(signo);
            return;
        }
        libc::sigaction(signo, old, std::ptr::null_mut());
    }
}

fn install() {
    if INSTALLED.load(Ordering::Acquire) {
        return;
    }
    lock(&INSTALL_LOCK);
    if !INSTALLED.load(Ordering::Acquire) {
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault
                as unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void)
                as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(libc::SIGSEGV, &action, &raw mut OLD_SEGV);
            libc::sigaction(libc::SIGBUS, &action, &raw mut OLD_BUS);
        }
        log::info!("soft write-watch: fault handler installed (host page {} bytes)", host_page());
        INSTALLED.store(true, Ordering::Release);
    }
    unlock(&INSTALL_LOCK);
}
