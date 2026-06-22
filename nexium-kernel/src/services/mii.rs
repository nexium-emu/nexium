pub struct MiiService;

impl MiiService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("mii cmd: {}", cmd_id);
        0
    }
}

impl Default for MiiService {
    fn default() -> Self {
        Self::new()
    }
}
