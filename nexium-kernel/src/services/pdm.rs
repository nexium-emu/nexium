pub struct PdmService;

impl PdmService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("pdm cmd: {}", cmd_id);
        0
    }
}

impl Default for PdmService {
    fn default() -> Self {
        Self::new()
    }
}
