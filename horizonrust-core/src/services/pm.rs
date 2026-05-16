pub struct PmService;

impl PmService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("pm cmd: {}", cmd_id); 0 }
}

impl Default for PmService {
    fn default() -> Self { Self::new() }
}
