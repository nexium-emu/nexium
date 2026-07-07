pub struct NimService;

impl NimService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("nim cmd: {}", cmd_id);
        0
    }
}

impl Default for NimService {
    fn default() -> Self {
        Self::new()
    }
}
