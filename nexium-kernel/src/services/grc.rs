pub struct GameRecordingService;

impl GameRecordingService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("grc cmd: {}", cmd_id);
        0
    }
}

impl Default for GameRecordingService {
    fn default() -> Self {
        Self::new()
    }
}
