pub struct MountService;

impl MountService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("mount cmd: {}", cmd_id);
        0
    }
}

impl Default for MountService {
    fn default() -> Self {
        Self::new()
    }
}
