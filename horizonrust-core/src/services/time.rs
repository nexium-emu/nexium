pub struct TimeService;

impl TimeService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("time cmd: {}", cmd_id);
        0
    }
}

impl Default for TimeService {
    fn default() -> Self {
        Self::new()
    }
}
