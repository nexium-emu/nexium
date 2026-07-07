pub struct FspService;

impl FspService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("fsp cmd: {}", cmd_id);
        0
    }
}

impl Default for FspService {
    fn default() -> Self {
        Self::new()
    }
}
