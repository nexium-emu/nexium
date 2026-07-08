use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

static IN_FLIGHT: AtomicI64 = AtomicI64::new(0);
static TOTAL_BUILT: AtomicU64 = AtomicU64::new(0);

pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
        TOTAL_BUILT.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn guard() -> Guard {
    IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
    Guard
}

pub fn begin() {
    IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
}

pub fn end() {
    IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    TOTAL_BUILT.fetch_add(1, Ordering::Relaxed);
}

pub fn in_flight() -> i64 {
    IN_FLIGHT.load(Ordering::Relaxed).max(0)
}

pub fn total_built() -> u64 {
    TOTAL_BUILT.load(Ordering::Relaxed)
}
