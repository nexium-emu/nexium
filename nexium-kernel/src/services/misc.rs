pub struct MiscService;

impl MiscService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("misc cmd: {}", cmd_id);
        0
    }
}

impl Default for MiscService {
    fn default() -> Self {
        Self::new()
    }
}
