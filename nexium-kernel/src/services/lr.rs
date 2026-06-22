pub struct LrService;

impl LrService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("lr cmd: {}", cmd_id);
        0
    }
}

impl Default for LrService {
    fn default() -> Self {
        Self::new()
    }
}
