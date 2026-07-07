pub struct PcieLmService;

impl PcieLmService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("pcielm cmd: {}", cmd_id);
        0
    }
}

impl Default for PcieLmService {
    fn default() -> Self {
        Self::new()
    }
}
