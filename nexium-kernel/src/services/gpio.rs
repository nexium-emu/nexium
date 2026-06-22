pub struct GpioService;

impl GpioService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("gpio cmd: {}", cmd_id);
        0
    }
}

impl Default for GpioService {
    fn default() -> Self {
        Self::new()
    }
}
