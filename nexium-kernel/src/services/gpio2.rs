pub struct Gpio2Service;

impl Gpio2Service {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("gpio2 cmd: {}", cmd_id); 0 }
}

impl Default for Gpio2Service {
    fn default() -> Self { Self::new() }
}
