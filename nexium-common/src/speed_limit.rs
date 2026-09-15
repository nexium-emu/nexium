use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn reset() {
    let requested = std::env::var("NEXIUM_UNLOCKED").ok();
    ENABLED.store(!start_unlocked(requested.as_deref()), Ordering::Relaxed);
}

pub fn toggle() -> bool {
    !ENABLED.fetch_xor(true, Ordering::Relaxed)
}

fn start_unlocked(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

#[cfg(test)]
mod tests {
    use super::start_unlocked;

    #[test]
    fn unlocked_launch_requires_explicit_opt_in() {
        for value in [None, Some(""), Some("0"), Some("false"), Some("invalid")] {
            assert!(!start_unlocked(value));
        }
        for value in ["1", "true", "TRUE", " true "] {
            assert!(start_unlocked(Some(value)));
        }
    }
}
