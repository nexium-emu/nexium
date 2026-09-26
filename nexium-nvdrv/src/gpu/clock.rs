use std::sync::OnceLock;
use std::time::Instant;

const REPORT_TICKS_PER_SECOND: u64 = 614_400_000;

fn fast_gpu_time_value_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

fn fast_gpu_time_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = fast_gpu_time_value_enabled(std::env::var("NEXIUM_FAST_GPU_TIME").ok().as_deref());
        if enabled {
            log::info!("gpu: fast GPU time compatibility mode enabled; guest GPU report timestamps scaled by 1/256; CPU and ioctl clocks unchanged");
        }
        enabled
    })
}

pub(crate) fn nanoseconds() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn ticks_from_nanoseconds(ns: u64) -> u64 {
    ((u128::from(ns) * u128::from(REPORT_TICKS_PER_SECOND)) / 1_000_000_000) as u64
}

fn report_ticks_from_nanoseconds(ns: u64, fast_gpu_time: bool) -> u64 {
    let ticks = ticks_from_nanoseconds(ns);
    if fast_gpu_time { ticks / 256 } else { ticks }
}

pub(crate) fn report_timestamp() -> u64 {
    let fast_gpu_time = fast_gpu_time_enabled();
    report_ticks_from_nanoseconds(nanoseconds(), fast_gpu_time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_clock_has_hardware_rate_and_preserves_long_uptimes() {
        assert_eq!(ticks_from_nanoseconds(0), 0);
        assert_eq!(ticks_from_nanoseconds(625), 384);
        assert_eq!(ticks_from_nanoseconds(1_000_000_000), 614_400_000);
        assert_eq!(ticks_from_nanoseconds(86_400_000_000_000), 53_084_160_000_000);
        assert_eq!(ticks_from_nanoseconds(u64::MAX), 11_333_679_558_887_148_512);
    }

    #[test]
    fn fast_gpu_time_requires_explicit_one() {
        for value in [None, Some(""), Some("0"), Some("false"), Some("off"), Some("true"), Some("2")] {
            assert!(!fast_gpu_time_value_enabled(value));
        }
        assert!(fast_gpu_time_value_enabled(Some("1")));
    }

    #[test]
    fn report_modes_preserve_hardware_and_reference_rates() {
        for (ns, hardware, fast) in [
            (0, 0, 0),
            (625, 384, 1),
            (1_000_000_000, 614_400_000, 2_400_000),
            (86_400_000_000_000, 53_084_160_000_000, 207_360_000_000),
            (u64::MAX, 11_333_679_558_887_148_512, 44_272_185_776_902_923),
        ] {
            assert_eq!(report_ticks_from_nanoseconds(ns, false), hardware);
            assert_eq!(report_ticks_from_nanoseconds(ns, true), fast);
            assert_eq!(ticks_from_nanoseconds(ns), hardware);
        }
    }

    #[test]
    fn report_modes_round_down_across_fraction_boundaries() {
        for ns in 0..=2500 {
            assert_eq!(report_ticks_from_nanoseconds(ns, false), ns * 384 / 625);
            assert_eq!(report_ticks_from_nanoseconds(ns, true), ns * 3 / 1250);
        }
        assert_eq!(report_ticks_from_nanoseconds(416, true), 0);
        assert_eq!(report_ticks_from_nanoseconds(417, true), 1);
    }

    #[test]
    fn report_modes_remain_monotonic_at_long_uptime_boundaries() {
        let samples = [
            0, 1, 416, 417, 624, 625, 833, 834, 1249, 1250,
            1_000_000_000, 86_400_000_000_000, u64::MAX - 625, u64::MAX - 1, u64::MAX,
        ];
        for fast_gpu_time in [false, true] {
            for pair in samples.windows(2) {
                assert!(report_ticks_from_nanoseconds(pair[0], fast_gpu_time)
                    <= report_ticks_from_nanoseconds(pair[1], fast_gpu_time));
            }
        }
    }

    #[test]
    fn report_and_ioctl_time_share_a_monotonic_epoch() {
        let fast_gpu_time = fast_gpu_time_enabled();
        let before = nanoseconds();
        let report = report_timestamp();
        let after = nanoseconds();
        assert!(after >= before);
        assert!(report >= report_ticks_from_nanoseconds(before, fast_gpu_time));
        assert!(report <= report_ticks_from_nanoseconds(after, fast_gpu_time));
    }
}
