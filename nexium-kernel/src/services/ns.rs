pub struct ContentService;

impl ContentService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("ns cmd: {}", cmd_id);
        0
    }
}

impl Default for ContentService {
    fn default() -> Self {
        Self::new()
    }
}
