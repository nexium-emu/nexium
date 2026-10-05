use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

const BOOT_MODE_UNSET: u8 = u8::MAX;

static BOOT_NORMAL_ACCURACY: AtomicU8 = AtomicU8::new(BOOT_MODE_UNSET);

fn parse_gpu_accuracy(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "normal" => Some(true),
        "high" => Some(false),
        _ => None,
    }
}

fn env_normal_accuracy() -> Option<bool> {
    static VALUE: OnceLock<Option<bool>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NEXIUM_GPU_ACCURACY")
            .ok()
            .and_then(|value| parse_gpu_accuracy(&value))
    })
}

fn resolve_normal_accuracy(env: Option<bool>, preference: bool) -> bool {
    env.unwrap_or(preference)
}

pub fn configure_gpu_accuracy() {
    let normal = resolve_normal_accuracy(
        env_normal_accuracy(),
        nexium_common::gpu_accuracy::normal(),
    );
    BOOT_NORMAL_ACCURACY.store(u8::from(normal), Ordering::Relaxed);
    if normal {
        log::info!("gpu: accuracy normal; GPFIFO fences and payload semaphores signal at submission");
    } else {
        log::info!("gpu: accuracy high; GPFIFO fences and payload semaphores signal after the host GPU completes");
    }
}

pub(crate) fn normal_accuracy() -> bool {
    match BOOT_NORMAL_ACCURACY.load(Ordering::Relaxed) {
        BOOT_MODE_UNSET => resolve_normal_accuracy(
            env_normal_accuracy(),
            nexium_common::gpu_accuracy::normal(),
        ),
        mode => mode != 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_accuracy_env_override_wins_over_preference() {
        assert_eq!(parse_gpu_accuracy("Normal"), Some(true));
        assert_eq!(parse_gpu_accuracy(" high "), Some(false));
        assert_eq!(parse_gpu_accuracy("fast"), None);
        assert!(resolve_normal_accuracy(Some(true), false));
        assert!(!resolve_normal_accuracy(Some(false), true));
        assert!(resolve_normal_accuracy(None, true));
        assert!(!resolve_normal_accuracy(None, false));
    }
}
