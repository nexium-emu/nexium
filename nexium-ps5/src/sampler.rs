use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const SLOTS: usize = 512;
const FRAMES: usize = 32;
const SCAN_WORDS: usize = 384;
const SCAN_LIMIT: usize = 64 << 10;
const IMAGE_REACH: usize = 64 << 20;

static NAMED: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
static NAMES: [[AtomicU64; 2]; SLOTS] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; SLOTS];
static NAMED_COUNT: AtomicUsize = AtomicUsize::new(0);
static TOPS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
static TARGET_TOP: AtomicUsize = AtomicUsize::new(0);
static TARGET_LO: AtomicUsize = AtomicUsize::new(0);
static TARGET_HI: AtomicUsize = AtomicUsize::new(0);
static RAX: AtomicU64 = AtomicU64::new(0);
static RET: AtomicU64 = AtomicU64::new(0);
static REQUEST: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
static RIP: AtomicU64 = AtomicU64::new(0);
static STACK: [AtomicU64; FRAMES] = [const { AtomicU64::new(0) }; FRAMES];

pub fn note_name(thread: usize, name: *const c_char) {
    if name.is_null() {
        return;
    }
    let slot = NAMED_COUNT.fetch_add(1, Ordering::AcqRel);
    if slot >= SLOTS {
        return;
    }
    let mut bytes = [0u8; 16];
    let src = unsafe { CStr::from_ptr(name) }.to_bytes();
    let n = src.len().min(15);
    bytes[..n].copy_from_slice(&src[..n]);
    NAMES[slot][0].store(u64::from_le_bytes(bytes[..8].try_into().unwrap()), Ordering::Relaxed);
    NAMES[slot][1].store(u64::from_le_bytes(bytes[8..].try_into().unwrap()), Ordering::Relaxed);
    let marker = 0u8;
    if thread == unsafe { pthread_self() } {
        TOPS[slot].store(&marker as *const u8 as usize & !7, Ordering::Relaxed);
    }
    NAMED[slot].store(thread, Ordering::Release);
}

unsafe extern "C" {
    static __ehdr_start: u8;
}

fn name_of(thread: usize) -> String {
    let count = NAMED_COUNT.load(Ordering::Acquire).min(SLOTS);
    for slot in (0..count).rev() {
        if NAMED[slot].load(Ordering::Acquire) == thread {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&NAMES[slot][0].load(Ordering::Relaxed).to_le_bytes());
            bytes[8..].copy_from_slice(&NAMES[slot][1].load(Ordering::Relaxed).to_le_bytes());
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(16);
            return String::from_utf8_lossy(&bytes[..end]).into_owned();
        }
    }
    format!("{thread:#x}")
}

#[no_mangle]
pub extern "C" fn nexium_ps5_sample_anchor() -> usize {
    nexium_ps5_sample_anchor as usize
}

unsafe extern "C" fn on_sample(_signo: c_int, _info: *mut libc::siginfo_t, context: *mut c_void) {
    unsafe {
        let lo = &raw const __ehdr_start as usize;
        let hi = lo + IMAGE_REACH;
        let rip = *crate::sys::context_reg(context, crate::sys::MC_RIP);
        let rsp = *crate::sys::context_reg(context, crate::sys::MC_RSP) as usize;
        RIP.store(rip, Ordering::Relaxed);
        RAX.store(*crate::sys::context_reg(context, crate::sys::MC_RAX), Ordering::Relaxed);
        RET.store(*(rsp as *const u64), Ordering::Relaxed);
        let top = TARGET_TOP.load(Ordering::Relaxed);
        let words = if top > rsp { ((top - rsp) / 8).min(SCAN_LIMIT / 8) } else { SCAN_WORDS };
        let mut found = 0;
        let (stack_lo, stack_hi) = (TARGET_LO.load(Ordering::Relaxed).max(rsp), TARGET_HI.load(Ordering::Relaxed));
        let mut rbp = *crate::sys::context_reg(context, crate::sys::MC_RBP) as usize;
        while found < FRAMES && rbp >= stack_lo && rbp + 16 <= stack_hi && rbp & 7 == 0 {
            let ret = *((rbp + 8) as *const usize);
            if ret <= lo || ret >= hi {
                break;
            }
            STACK[found].store(ret as u64, Ordering::Relaxed);
            found += 1;
            let next = *(rbp as *const usize);
            if next <= rbp {
                break;
            }
            rbp = next;
        }
        let words = if found > 0 { 0 } else { words };
        for i in 0..words {
            if found == FRAMES {
                break;
            }
            let word = *((rsp + i * 8) as *const usize);
            if word > lo && word < hi {
                STACK[found].store(word as u64, Ordering::Relaxed);
                found += 1;
            }
        }
        for slot in &STACK[found..] {
            slot.store(0, Ordering::Relaxed);
        }
        DONE.store(REQUEST.load(Ordering::Acquire), Ordering::Release);
    }
}

fn stack_bounds(thread: usize) -> (usize, usize) {
    unsafe {
        let mut attr: *mut c_void = std::ptr::null_mut();
        if pthread_attr_init(&mut attr) != 0 {
            return (0, 0);
        }
        let mut addr: *mut c_void = std::ptr::null_mut();
        let mut size = 0usize;
        let ok = pthread_attr_get_np(thread, &mut attr) == 0 && pthread_attr_getstack(&attr, &mut addr, &mut size) == 0;
        pthread_attr_destroy(&mut attr);
        if ok && !addr.is_null() {
            (addr as usize, addr as usize + size)
        } else {
            (0, 0)
        }
    }
}

unsafe extern "C" {
    fn pthread_attr_init(attr: *mut *mut c_void) -> c_int;
    fn pthread_attr_destroy(attr: *mut *mut c_void) -> c_int;
    fn pthread_attr_get_np(thread: usize, attr: *mut *mut c_void) -> c_int;
    fn pthread_attr_getstack(attr: *const *mut c_void, addr: *mut *mut c_void, size: *mut usize) -> c_int;
    fn pthread_kill(thread: usize, sig: c_int) -> c_int;
    fn pthread_self() -> usize;
}

pub fn start(interval_ms: u64) {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = on_sample as unsafe extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(libc::SIGUSR2, &action, std::ptr::null_mut());
    }
    crate::klog!(
        "sample: anchor {:#x} every {interval_ms} ms, image {:p}",
        nexium_ps5_sample_anchor as usize,
        &raw const __ehdr_start
    );
    let filter = std::env::var("NEXIUM_PS5_SAMPLE_THREADS").ok();
    let _ = std::thread::Builder::new().name("nexium-sampler".into()).spawn(move || {
        let me = unsafe { pthread_self() };
        let mut seq = 0u64;
        loop {
            std::thread::sleep(Duration::from_millis(interval_ms));
            let count = NAMED_COUNT.load(Ordering::Acquire).min(SLOTS);
            for slot in 0..count {
                let thread = NAMED[slot].load(Ordering::Acquire);
                if thread == 0 || thread == me {
                    continue;
                }
                if let Some(filter) = &filter {
                    let name = name_of(thread);
                    if !filter.split(',').any(|prefix| name.starts_with(prefix)) {
                        continue;
                    }
                }
                seq += 1;
                TARGET_TOP.store(TOPS[slot].load(Ordering::Relaxed), Ordering::Relaxed);
                let (stack_lo, stack_hi) = stack_bounds(thread);
                TARGET_LO.store(stack_lo, Ordering::Relaxed);
                TARGET_HI.store(stack_hi, Ordering::Relaxed);
                REQUEST.store(seq, Ordering::Release);
                if unsafe { pthread_kill(thread, libc::SIGUSR2) } != 0 {
                    NAMED[slot].store(0, Ordering::Release);
                    continue;
                }
                let asked = Instant::now();
                while DONE.load(Ordering::Acquire) != seq && asked.elapsed() < Duration::from_millis(100) {
                    std::thread::yield_now();
                }
                if DONE.load(Ordering::Acquire) != seq {
                    crate::klog!("sample: {} no reply", name_of(thread));
                    continue;
                }
                let frames: Vec<String> = STACK
                    .iter()
                    .map(|s| s.load(Ordering::Relaxed))
                    .take_while(|&v| v != 0)
                    .map(|v| format!("{v:x}"))
                    .collect();
                crate::klog!(
                    "sample: {} rax {:x} ret {:x} rip {:x} stack {}",
                    name_of(thread),
                    RAX.load(Ordering::Relaxed),
                    RET.load(Ordering::Relaxed),
                    RIP.load(Ordering::Relaxed),
                    frames.join(" ")
                );
            }
        }
    });
}
