pub struct OlscService;

impl OlscService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("olsc cmd: {}", cmd_id);
        0
    }
}

impl Default for OlscService {
    fn default() -> Self {
        Self::new()
    }
}
