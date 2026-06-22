pub struct ApmService;

impl ApmService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("apm cmd: {}", cmd_id);
        0
    }
}

impl Default for ApmService {
    fn default() -> Self {
        Self::new()
    }
}
