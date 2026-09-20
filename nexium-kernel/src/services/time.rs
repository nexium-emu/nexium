use nexium_common::result::SUCCESS;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CLOCK_SOURCE_ID: [u8; 16] = *b"NeXiumClockSrc01";

struct ClockAnchor {
    unix_seconds: u64,
    started: Instant,
}

static CLOCK_ANCHOR: OnceLock<ClockAnchor> = OnceLock::new();

pub fn unix_time_seconds() -> i64 {
    let anchor = CLOCK_ANCHOR.get_or_init(|| ClockAnchor {
        unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        started: Instant::now(),
    });
    anchor
        .unix_seconds
        .saturating_add(anchor.started.elapsed().as_secs()) as i64
}

pub fn steady_clock_time_point() -> Vec<u8> {
    let mut out = Vec::with_capacity(0x18);
    out.extend_from_slice(&unix_time_seconds().to_le_bytes());
    out.extend_from_slice(&CLOCK_SOURCE_ID);
    out
}

pub fn system_clock_context() -> Vec<u8> {
    let mut out = Vec::with_capacity(0x20);
    out.extend_from_slice(&0i64.to_le_bytes());
    out.extend_from_slice(&steady_clock_time_point());
    out
}

pub struct TimeService {
    start_time: u64,
}

impl TimeService {
    pub fn new() -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self { start_time }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::info!("time cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_get_system_time(),
            1 => self.cmd_get_posix_time(),
            2 => self.cmd_get_steady_clock_time_point(),
            _ => {
                log::info!("time stub command: {}", cmd_id);
                0
            }
        }
    }

    pub fn get_system_time(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    fn cmd_get_system_time(&self) -> u32 {
        let now = self.get_system_time();
        let switch_epoch = now.saturating_sub(946_684_800 * 1_000_000_000);
        log::info!("Time::GetSystemTime -> {} ns (switch epoch)", switch_epoch);
        SUCCESS
    }

    fn cmd_get_posix_time(&self) -> u32 {
        let now = self.get_system_time() / 1_000_000_000;
        log::info!("Time::GetPosixTime -> {} s", now);
        SUCCESS
    }

    fn cmd_get_steady_clock_time_point(&self) -> u32 {
        let now = self.get_system_time();
        log::info!("Time::GetSteadyClockTimePoint -> {} ns", now);
        SUCCESS
    }
}

impl Default for TimeService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_clock_context_reconstructs_current_posix_time() {
        let before = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let context = system_clock_context();
        let offset = i64::from_le_bytes(context[0..8].try_into().unwrap());
        let steady = i64::from_le_bytes(context[8..16].try_into().unwrap());
        let current = unix_time_seconds();
        let after = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        assert!((before.saturating_sub(1)..=after).contains(&(offset + steady)));
        assert!((before.saturating_sub(1)..=after).contains(&current));
        assert_eq!(&context[16..32], &CLOCK_SOURCE_ID);
    }
}
