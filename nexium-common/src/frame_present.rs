use std::sync::{Mutex, OnceLock};

pub struct FramePresent {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

static SLOT: OnceLock<Mutex<Option<FramePresent>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<FramePresent>> {
    SLOT.get_or_init(|| Mutex::new(None))
}

pub fn set_last_presented(width: u32, height: u32, pixels: Vec<u8>) {
    if let Ok(mut s) = slot().lock() {
        *s = Some(FramePresent { width, height, pixels });
    }
}

pub fn take_last_presented() -> Option<(u32, u32, Vec<u8>)> {
    slot().lock().ok().and_then(|mut s| s.take().map(|f| (f.width, f.height, f.pixels)))
}

pub fn peek_last_presented_clone() -> Option<(u32, u32, Vec<u8>)> {
    slot().lock().ok().and_then(|s| s.as_ref().map(|f| (f.width, f.height, f.pixels.clone())))
}
