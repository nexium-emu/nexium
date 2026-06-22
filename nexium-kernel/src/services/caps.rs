pub struct CapsService;

impl CapsService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("caps cmd: {}", cmd_id);
        0
    }
}

impl Default for CapsService {
    fn default() -> Self {
        Self::new()
    }
}
