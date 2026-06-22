pub struct SslService;

impl SslService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("ssl cmd: {}", cmd_id);
        0
    }
}

impl Default for SslService {
    fn default() -> Self {
        Self::new()
    }
}
