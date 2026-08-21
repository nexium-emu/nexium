use std::sync::atomic::{AtomicBool, Ordering};

static BIG_WARP: AtomicBool = AtomicBool::new(false);

pub fn set_big_warp(v: bool) {
    BIG_WARP.store(v, Ordering::Relaxed);
}

pub fn big_warp() -> bool {
    BIG_WARP.load(Ordering::Relaxed)
}
