pub struct DisplayService;

impl DisplayService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("vi cmd: {}", cmd_id);
        0
    }
}

impl Default for DisplayService {
    fn default() -> Self {
        Self::new()
    }
}
