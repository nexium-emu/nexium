pub struct SysService;

impl SysService {
    pub fn new() -> Self { Self }
    pub fn dispatch(&self, cmd_id: u32) -> u32 { log::trace!("sys cmd: {}", cmd_id); 0 }
}

impl Default for SysService {
    fn default() -> Self { Self::new() }
}
