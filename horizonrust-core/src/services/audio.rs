pub struct AudioService;

impl AudioService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("audio cmd: {}", cmd_id);
        0
    }
}

impl Default for AudioService {
    fn default() -> Self {
        Self::new()
    }
}
