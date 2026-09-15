use std::sync::OnceLock;
use std::time::Instant;

const REPORT_TICKS_PER_SECOND: u64 = 614_400_000;

pub(crate) fn nanoseconds() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn ticks_from_nanoseconds(ns: u64) -> u64 {
    ((u128::from(ns) * u128::from(REPORT_TICKS_PER_SECOND)) / 1_000_000_000) as u64
}

pub(crate) fn report_timestamp() -> u64 {
    ticks_from_nanoseconds(nanoseconds())
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
    fn report_and_ioctl_time_share_a_monotonic_epoch() {
        let before = nanoseconds();
        let report = report_timestamp();
        let after = nanoseconds();
        assert!(after >= before);
        assert!(report >= ticks_from_nanoseconds(before));
        assert!(report <= ticks_from_nanoseconds(after));
    }
}
