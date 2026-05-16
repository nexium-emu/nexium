pub struct LdrService;

impl LdrService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("ldr cmd: {}", cmd_id); 0 }
}

impl Default for LdrService {
    fn default() -> Self { Self::new() }
}
