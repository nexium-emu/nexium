pub struct SplService;

impl SplService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("spl cmd: {}", cmd_id);
        0
    }
}

impl Default for SplService {
    fn default() -> Self {
        Self::new()
    }
}
