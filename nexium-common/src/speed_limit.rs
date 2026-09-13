use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn reset() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn toggle() -> bool {
    !ENABLED.fetch_xor(true, Ordering::Relaxed)
}
