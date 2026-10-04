use std::ffi::c_int;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

const UNSET: u8 = 0;
const BUSY: u8 = 1;
const READY: u8 = 2;
const OFF: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(UNSET);
static FREQ: AtomicU64 = AtomicU64::new(0);
static BASE_TSC: AtomicU64 = AtomicU64::new(0);
static BASE_MONO_NS: AtomicU64 = AtomicU64::new(0);
static BASE_REAL_NS: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" {
    fn __real_clock_gettime(clock: c_int, ts: *mut libc::timespec) -> c_int;
    fn sceKernelGetTscFrequency() -> u64;
}

fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn to_ns(ts: &libc::timespec) -> u64 {
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

unsafe fn init() {
    unsafe {
        let disabled = {
            let v = libc::getenv(c"NEXIUM_PS5_TSC_CLOCK".as_ptr());
            !v.is_null() && *v == b'0' as libc::c_char
        };
        let freq = if disabled { 0 } else { sceKernelGetTscFrequency() };
        if freq == 0 {
            STATE.store(OFF, Ordering::Release);
            return;
        }
        let mut mono: libc::timespec = std::mem::zeroed();
        let mut real: libc::timespec = std::mem::zeroed();
        let tsc = rdtsc();
        if __real_clock_gettime(libc::CLOCK_MONOTONIC, &mut mono) != 0
            || __real_clock_gettime(libc::CLOCK_REALTIME, &mut real) != 0
        {
            STATE.store(OFF, Ordering::Release);
            return;
        }
        FREQ.store(freq, Ordering::Relaxed);
        BASE_TSC.store(tsc, Ordering::Relaxed);
        BASE_MONO_NS.store(to_ns(&mono), Ordering::Relaxed);
        BASE_REAL_NS.store(to_ns(&real), Ordering::Relaxed);
        STATE.store(READY, Ordering::Release);
    }
}

fn ready() -> bool {
    match STATE.load(Ordering::Acquire) {
        READY => true,
        UNSET => {
            if STATE.compare_exchange(UNSET, BUSY, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                unsafe { init() };
            }
            STATE.load(Ordering::Acquire) == READY
        }
        _ => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_clock_gettime(clock: c_int, ts: *mut libc::timespec) -> c_int {
    unsafe {
        let base = match clock {
            libc::CLOCK_MONOTONIC
            | libc::CLOCK_MONOTONIC_PRECISE
            | libc::CLOCK_MONOTONIC_FAST
            | libc::CLOCK_UPTIME
            | libc::CLOCK_UPTIME_PRECISE
            | libc::CLOCK_UPTIME_FAST => Some(&BASE_MONO_NS),
            libc::CLOCK_REALTIME | libc::CLOCK_REALTIME_PRECISE | libc::CLOCK_REALTIME_FAST => {
                Some(&BASE_REAL_NS)
            }
            _ => None,
        };
        if let Some(base) = base {
            if !ts.is_null() && ready() {
                let delta = rdtsc().wrapping_sub(BASE_TSC.load(Ordering::Relaxed));
                let ns = base.load(Ordering::Relaxed)
                    + (delta as u128 * 1_000_000_000 / FREQ.load(Ordering::Relaxed) as u128) as u64;
                (*ts).tv_sec = (ns / 1_000_000_000) as libc::time_t;
                (*ts).tv_nsec = (ns % 1_000_000_000) as libc::c_long;
                return 0;
            }
        }
        __real_clock_gettime(clock, ts)
    }
}
