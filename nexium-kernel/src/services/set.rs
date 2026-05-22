pub struct SettingsService;

impl SettingsService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("set cmd: {}", cmd_id);
        0
    }
}

impl Default for SettingsService {
    fn default() -> Self {
        Self::new()
    }
}
