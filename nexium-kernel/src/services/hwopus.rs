pub struct HwOpusService;

impl HwOpusService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("hwopus cmd: {}", cmd_id);
        0
    }
}

impl Default for HwOpusService {
    fn default() -> Self {
        Self::new()
    }
}
