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
            0 => SUCCESS,
            1 => SUCCESS,
            2 => SUCCESS,
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
}

impl Default for TimeService {
    fn default() -> Self {
        Self::new()
    }
}
