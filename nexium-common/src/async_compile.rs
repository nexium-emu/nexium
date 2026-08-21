use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static SHADERS_BUILT: AtomicU64 = AtomicU64::new(0);

pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn note_shader_built() {
    SHADERS_BUILT.fetch_add(1, Ordering::Relaxed);
}

pub fn shaders_built() -> u64 {
    SHADERS_BUILT.load(Ordering::Relaxed)
}
