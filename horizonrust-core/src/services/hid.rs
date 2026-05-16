pub struct HidService;

impl HidService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("hid cmd: {}", cmd_id);
        0
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
