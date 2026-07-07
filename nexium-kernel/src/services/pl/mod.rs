pub mod handlers;

pub struct PlService;

impl PlService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("pl cmd: {} (legacy stub fallback)", cmd_id);
        0
    }
}

impl Default for PlService {
    fn default() -> Self {
        Self::new()
    }
}
