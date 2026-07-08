use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static IN_FLIGHT: AtomicI64 = AtomicI64::new(0);
static TOTAL_BUILT: AtomicU64 = AtomicU64::new(0);
static BURST_BUILT: AtomicU64 = AtomicU64::new(0);
static LAST_END_MS: AtomicU64 = AtomicU64::new(0);

const LINGER_MS: u64 = 1500;

fn now_ms() -> u64 {
    static BASE: OnceLock<Instant> = OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        end();
    }
}

pub fn guard() -> Guard {
    begin();
    Guard
}

pub fn begin() {
    let idle = IN_FLIGHT.load(Ordering::Relaxed) <= 0
        && now_ms().saturating_sub(LAST_END_MS.load(Ordering::Relaxed)) > LINGER_MS;
    if idle {
        BURST_BUILT.store(0, Ordering::Relaxed);
    }
    IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
}

pub fn end() {
    IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    TOTAL_BUILT.fetch_add(1, Ordering::Relaxed);
    BURST_BUILT.fetch_add(1, Ordering::Relaxed);
    LAST_END_MS.store(now_ms(), Ordering::Relaxed);
}

pub fn in_flight() -> i64 {
    IN_FLIGHT.load(Ordering::Relaxed).max(0)
}

pub fn total_built() -> u64 {
    TOTAL_BUILT.load(Ordering::Relaxed)
}

pub fn burst_built() -> u64 {
    BURST_BUILT.load(Ordering::Relaxed)
}

pub fn recently_active() -> bool {
    let last = LAST_END_MS.load(Ordering::Relaxed);
    last != 0 && now_ms().saturating_sub(last) < LINGER_MS
}
