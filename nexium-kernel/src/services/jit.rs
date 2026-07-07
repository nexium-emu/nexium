pub struct JitService;

impl JitService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("jit cmd: {}", cmd_id);
        0
    }
}

impl Default for JitService {
    fn default() -> Self {
        Self::new()
    }
}
