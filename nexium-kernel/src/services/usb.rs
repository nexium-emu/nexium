pub struct UsbService;

impl UsbService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("usb cmd: {}", cmd_id);
        0
    }
}

impl Default for UsbService {
    fn default() -> Self {
        Self::new()
    }
}
