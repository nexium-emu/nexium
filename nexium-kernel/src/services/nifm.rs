pub struct NetworkService;

impl NetworkService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("nifm cmd: {}", cmd_id);
        0
    }
}

impl Default for NetworkService {
    fn default() -> Self {
        Self::new()
    }
}
