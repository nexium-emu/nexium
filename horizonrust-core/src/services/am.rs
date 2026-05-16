pub struct AppletService;

impl AppletService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("am cmd: {}", cmd_id);
        0
    }
}

impl Default for AppletService {
    fn default() -> Self {
        Self::new()
    }
}
