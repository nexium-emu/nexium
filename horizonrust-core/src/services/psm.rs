pub struct PsmService;

impl PsmService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("psm cmd: {}", cmd_id); 0 }
}

impl Default for PsmService {
    fn default() -> Self { Self::new() }
}
