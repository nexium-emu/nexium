pub struct RtcService;

impl RtcService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("rtc cmd: {}", cmd_id);
        0
    }
}

impl Default for RtcService {
    fn default() -> Self {
        Self::new()
    }
}
