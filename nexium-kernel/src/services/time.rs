use nexium_common::result::SUCCESS;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SWITCH_EPOCH_UNIX_SECONDS: u64 = 946_684_800;
const CLOCK_SOURCE_ID: [u8; 16] = *b"NeXiumClockSrc01";

struct ClockAnchor {
    switch_seconds: u64,
    started: Instant,
}

static CLOCK_ANCHOR: OnceLock<ClockAnchor> = OnceLock::new();

pub fn switch_time_seconds() -> i64 {
    let anchor = CLOCK_ANCHOR.get_or_init(|| ClockAnchor {
        switch_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(SWITCH_EPOCH_UNIX_SECONDS),
        started: Instant::now(),
    });
    anchor
        .switch_seconds
        .saturating_add(anchor.started.elapsed().as_secs()) as i64
}

pub fn steady_clock_time_point() -> Vec<u8> {
    let mut out = Vec::with_capacity(0x18);
    out.extend_from_slice(&switch_time_seconds().to_le_bytes());
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
