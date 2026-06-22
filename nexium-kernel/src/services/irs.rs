pub struct InfraredService;

impl InfraredService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("irs cmd: {}", cmd_id);
        0
    }
}

impl Default for InfraredService {
    fn default() -> Self {
        Self::new()
    }
}
