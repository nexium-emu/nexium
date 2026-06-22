pub struct PscService;

impl PscService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("psc cmd: {}", cmd_id);
        0
    }
}

impl Default for PscService {
    fn default() -> Self {
        Self::new()
    }
}
