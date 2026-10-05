use std::sync::atomic::{AtomicBool, Ordering};

static NORMAL: AtomicBool = AtomicBool::new(false);

pub fn set_normal(v: bool) {
    NORMAL.store(v, Ordering::Relaxed);
}

pub fn normal() -> bool {
    NORMAL.load(Ordering::Relaxed)
}
