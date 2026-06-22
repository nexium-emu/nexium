pub struct PrepoService;

impl PrepoService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("prepo cmd: {}", cmd_id);
        0
    }
}

impl Default for PrepoService {
    fn default() -> Self {
        Self::new()
    }
}
