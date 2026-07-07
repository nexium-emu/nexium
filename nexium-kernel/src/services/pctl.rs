pub struct ParentalControlService;

impl ParentalControlService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("pctl cmd: {}", cmd_id);
        0
    }
}

impl Default for ParentalControlService {
    fn default() -> Self {
        Self::new()
    }
}
