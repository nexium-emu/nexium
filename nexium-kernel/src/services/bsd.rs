pub struct BsdService;

impl BsdService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("bsd cmd: {}", cmd_id);
        0
    }
}

impl Default for BsdService {
    fn default() -> Self {
        Self::new()
    }
}
