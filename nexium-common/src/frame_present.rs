use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static FRAME_N: AtomicU64 = AtomicU64::new(0);
static DIAGNOSTICS_ENABLED: OnceLock<bool> = OnceLock::new();

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
    let diagnostics_enabled =
        *DIAGNOSTICS_ENABLED.get_or_init(|| std::env::var_os("NEXIUM_DIAG").is_some());
    if diagnostics_enabled {
        let n = FRAME_N.fetch_add(1, Ordering::Relaxed);
        if n % 60 == 0 && pixels.len() >= 4 {
            let mut mx = 0u8;
            let mut sum: u64 = 0;
            let mut cnt: u64 = 0;
            for px in pixels.chunks_exact(4) {
                let m = px[0].max(px[1]).max(px[2]);
                if m > mx {
                    mx = m;
                }
                sum += m as u64;
                cnt += 1;
            }
            let avg = if cnt > 0 { sum / cnt } else { 0 };
            let (mut br, mut bg, mut bb, mut bc): (u64, u64, u64, u64) = (0, 0, 0, 0);
            let row = (width as usize) * 4;
            let start = (height as usize / 2) * row;
            for px in pixels.get(start..).unwrap_or(&[]).chunks_exact(4) {
                br += px[0] as u64;
                bg += px[1] as u64;
                bb += px[2] as u64;
                bc += 1;
            }
            let d = bc.max(1);
            log::info!(
                "[brightness] frame#{} {}x{} max_rgb={} avg_rgb={} bottomRGB=({},{},{})",
                n,
                width,
                height,
                mx,
                avg,
                br / d,
                bg / d,
                bb / d
            );
        }
    }
    if let Ok(mut s) = slot().lock() {
        *s = Some(FramePresent {
            width,
            height,
            pixels,
        });
    }
}

pub fn take_last_presented() -> Option<(u32, u32, Vec<u8>)> {
    slot()
        .lock()
        .ok()
        .and_then(|mut s| s.take().map(|f| (f.width, f.height, f.pixels)))
}

pub fn peek_last_presented_clone() -> Option<(u32, u32, Vec<u8>)> {
    slot()
        .lock()
        .ok()
        .and_then(|s| s.as_ref().map(|f| (f.width, f.height, f.pixels.clone())))
}
