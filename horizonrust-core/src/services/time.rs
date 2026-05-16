use std::time::{SystemTime, UNIX_EPOCH};
use crate::common::result::SUCCESS;

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
                0  // Return success for unknown commands
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
        let switch_epoch = now.saturating_sub(946_684_800 * 1_000_000_000);  // Switch epoch: 2000-01-01
        log::info!("Time::GetSystemTime -> {} ns (switch epoch)", switch_epoch);
        SUCCESS
    }

    fn cmd_get_posix_time(&self) -> u32 {
        let now = self.get_system_time() / 1_000_000_000;  // Convert to seconds
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
