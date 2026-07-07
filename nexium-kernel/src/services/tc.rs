pub struct TcService;

impl TcService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("tc cmd: {}", cmd_id);
        0
    }
}

impl Default for TcService {
    fn default() -> Self {
        Self::new()
    }
}
