use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

pub const MIN_FACTOR: f32 = 0.25;
pub const MAX_FACTOR: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Setting {
    Factor(f32),
    Height(u32),
}

impl Setting {
    pub const NATIVE: Setting = Setting::Factor(1.0);

    pub fn resolve(self, reference_height: u32) -> f32 {
        let factor = match self {
            Setting::Factor(factor) => factor,
            Setting::Height(_) if reference_height == 0 => 1.0,
            Setting::Height(height) => height as f32 / reference_height as f32,
        };
        clamp_factor(factor)
    }
}

pub fn clamp_factor(factor: f32) -> f32 {
    if !factor.is_finite() || (factor - 1.0).abs() < 1e-3 {
        return 1.0;
    }
    factor.clamp(MIN_FACTOR, MAX_FACTOR)
}

struct Config {
    global: Setting,
    titles: HashMap<u64, Setting>,
}

struct Pending {
    title_id: u64,
    setting: Setting,
}

static CONFIG: Mutex<Option<Config>> = Mutex::new(None);
static PENDING: Mutex<Option<Pending>> = Mutex::new(None);
static REFERENCE: AtomicU64 = AtomicU64::new(0);
static LATCHED: AtomicBool = AtomicBool::new(false);
static FACTOR: AtomicU32 = AtomicU32::new(0x3F80_0000);

pub fn configure(global: Setting, titles: HashMap<u64, Setting>) {
    *CONFIG.lock().unwrap() = Some(Config { global, titles });
}

pub fn setting_for(title_id: u64) -> Setting {
    if let Ok(value) = std::env::var("NEXIUM_RENDER_SCALE") {
        if let Some(height) = value.strip_suffix('p').and_then(|height| height.parse().ok()) {
            return Setting::Height(height);
        }
        if let Ok(factor) = value.parse() {
            return Setting::Factor(factor);
        }
    }
    let config = CONFIG.lock().unwrap();
    let Some(config) = config.as_ref() else {
        return Setting::NATIVE;
    };
    config.titles.get(&title_id).copied().unwrap_or(config.global)
}

pub fn begin_title(title_id: u64, reference: Option<(u32, u32)>) {
    let setting = setting_for(title_id);
    LATCHED.store(false, Ordering::Release);
    FACTOR.store(1.0f32.to_bits(), Ordering::Relaxed);
    REFERENCE.store(reference.map_or(0, pack), Ordering::Relaxed);
    *PENDING.lock().unwrap() = Some(Pending { title_id, setting });
    if reference.is_some() || matches!(setting, Setting::Factor(_)) {
        latch();
    }
}

pub fn set_reference(width: u32, height: u32) {
    if !LATCHED.load(Ordering::Acquire) {
        REFERENCE.store(pack((width, height)), Ordering::Relaxed);
    }
}

pub fn factor() -> f32 {
    if !LATCHED.load(Ordering::Acquire) {
        latch();
    }
    f32::from_bits(FACTOR.load(Ordering::Relaxed))
}

pub fn is_native() -> bool {
    factor() == 1.0
}

pub fn scale_with(factor: f32, length: u32) -> u32 {
    if factor == 1.0 || length == 0 {
        return length;
    }
    ((length as f32 * factor).round() as u32).max(1)
}

pub fn scale(length: u32) -> u32 {
    scale_with(factor(), length)
}

pub fn scale_offset(offset: i32) -> i32 {
    let factor = factor();
    if factor == 1.0 {
        return offset;
    }
    (offset as f32 * factor).round() as i32
}

fn pack((width, height): (u32, u32)) -> u64 {
    (u64::from(width) << 32) | u64::from(height)
}

fn latch() {
    let mut pending = PENDING.lock().unwrap();
    if LATCHED.load(Ordering::Acquire) {
        return;
    }
    let Some(active) = pending.take() else {
        return;
    };
    let reference = REFERENCE.load(Ordering::Relaxed);
    let (width, height) = ((reference >> 32) as u32, reference as u32);
    let factor = active.setting.resolve(height);
    FACTOR.store(factor.to_bits(), Ordering::Relaxed);
    LATCHED.store(true, Ordering::Release);
    if reference == 0 {
        log::info!(
            "[render-scale] title={:016X} setting={:?} scale={factor:.2}",
            active.title_id,
            active.setting
        );
    } else {
        log::info!(
            "[render-scale] title={:016X} setting={:?} scale={factor:.2} internal={}x{} guest={width}x{height}",
            active.title_id,
            active.setting,
            scale_with(factor, width),
            scale_with(factor, height)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_targets_resolve_against_the_reference() {
        assert_eq!(Setting::Height(720).resolve(1440), 0.5);
        assert_eq!(Setting::Height(1080).resolve(1080), 1.0);
        assert_eq!(Setting::Height(720).resolve(0), 1.0);
        assert_eq!(Setting::Factor(9.0).resolve(0), MAX_FACTOR);
        assert_eq!(Setting::Factor(f32::NAN).resolve(0), 1.0);
    }

    #[test]
    fn scaled_lengths_round_and_stay_nonzero() {
        assert_eq!(scale_with(0.5, 2560), 1280);
        assert_eq!(scale_with(0.75, 1), 1);
        assert_eq!(scale_with(0.75, 0), 0);
        assert_eq!(scale_with(1.0, 1234), 1234);
        assert_eq!(scale_with(0.6667, 1080), 720);
    }
}
