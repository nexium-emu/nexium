pub struct OmmService;

impl OmmService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("omm cmd: {}", cmd_id); 0 }
}

impl Default for OmmService {
    fn default() -> Self { Self::new() }
}
