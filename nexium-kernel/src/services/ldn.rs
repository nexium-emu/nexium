pub struct LdnService;

impl LdnService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("ldn cmd: {}", cmd_id);
        0
    }
}

impl Default for LdnService {
    fn default() -> Self {
        Self::new()
    }
}
