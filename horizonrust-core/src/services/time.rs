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
        log::debug!("time cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_get_system_time(),
            1 => self.cmd_get_posix_time(),
            2 => self.cmd_get_steady_clock_time_point(),
            _ => {
                log::warn!("unknown time command: {}", cmd_id);
                1
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
        log::debug!("Time::GetSystemTime");
        SUCCESS
    }

    fn cmd_get_posix_time(&self) -> u32 {
        log::debug!("Time::GetPosixTime");
        SUCCESS
    }

    fn cmd_get_steady_clock_time_point(&self) -> u32 {
        log::debug!("Time::GetSteadyClockTimePoint");
        SUCCESS
    }
}

impl Default for TimeService {
    fn default() -> Self {
        Self::new()
    }
}
