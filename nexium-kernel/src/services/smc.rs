pub struct SmcService;

impl SmcService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("smc cmd: {}", cmd_id);
        0
    }
}

impl Default for SmcService {
    fn default() -> Self {
        Self::new()
    }
}
