pub struct FrameOut {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct Services {
}

impl Services {
    pub fn new() -> Self {
        Self {}
    }

    pub fn dispatch_service(&self, port_name: &str, cmd_id: u32) -> u32 {
        log::trace!("dispatch_service: port={} cmd_id={}", port_name, cmd_id);
        0
    }
}

impl Default for Services {
    fn default() -> Self {
        Self::new()
    }
}
