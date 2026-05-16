pub struct BufferQueueService;

impl BufferQueueService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("nvnflinger cmd: {}", cmd_id);
        0
    }
}

impl Default for BufferQueueService {
    fn default() -> Self {
        Self::new()
    }
}
