use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(true);
static SYNC_TO_VIDEO: AtomicBool = AtomicBool::new(true);
static LAST_VIDEO_FRAME_US: AtomicU64 = AtomicU64::new(0);

const VIDEO_PLAYBACK_WINDOW_US: u64 = 500_000;

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn reset() {
    let requested = std::env::var("NEXIUM_UNLOCKED").ok();
    ENABLED.store(!start_unlocked(requested.as_deref()), Ordering::Relaxed);
    LAST_VIDEO_FRAME_US.store(0, Ordering::Relaxed);
}

pub fn toggle() -> bool {
    !ENABLED.fetch_xor(true, Ordering::Relaxed)
}

pub fn set_sync_to_video(enabled: bool) {
    SYNC_TO_VIDEO.store(enabled, Ordering::Relaxed);
}

pub fn sync_to_video() -> bool {
    SYNC_TO_VIDEO.load(Ordering::Relaxed)
}

pub fn note_video_frame() {
    LAST_VIDEO_FRAME_US.store(now_us(), Ordering::Relaxed);
}

pub fn video_playing() -> bool {
    video_recent(LAST_VIDEO_FRAME_US.load(Ordering::Relaxed), now_us())
}

pub fn pacing() -> bool {
    enabled() || (sync_to_video() && video_playing())
}

fn video_recent(last_video_us: u64, now_us: u64) -> bool {
    last_video_us != 0 && now_us.saturating_sub(last_video_us) <= VIDEO_PLAYBACK_WINDOW_US
}

fn now_us() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64 + 1
}

fn start_unlocked(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

#[cfg(test)]
mod tests {
    use super::{start_unlocked, video_recent, VIDEO_PLAYBACK_WINDOW_US};

    #[test]
    fn unlocked_launch_requires_explicit_opt_in() {
        for value in [None, Some(""), Some("0"), Some("false"), Some("invalid")] {
            assert!(!start_unlocked(value));
        }
        for value in ["1", "true", "TRUE", " true "] {
            assert!(start_unlocked(Some(value)));
        }
    }

    #[test]
    fn video_counts_as_playing_only_while_frames_arrive() {
        let now = 10_000_000;
        assert!(!video_recent(0, now));
        assert!(video_recent(now - 1_000, now));
        assert!(video_recent(now - VIDEO_PLAYBACK_WINDOW_US, now));
        assert!(!video_recent(now - VIDEO_PLAYBACK_WINDOW_US - 1, now));
        assert!(video_recent(now, now));
    }
}
