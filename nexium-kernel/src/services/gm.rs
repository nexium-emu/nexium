pub struct GmService;

impl GmService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("gm cmd: {}", cmd_id); 0 }
}

impl Default for GmService {
    fn default() -> Self { Self::new() }
}
