use crate::common::result::SUCCESS;

pub struct DisplayService;

impl DisplayService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("vi cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_get_display_service(),
            1 => self.cmd_get_display_vsync_event(),
            2 => self.cmd_get_display_status(),
            _ => {
                log::warn!("unknown vi command: {}", cmd_id);
                1
            }
        }
    }

    fn cmd_get_display_service(&self) -> u32 {
        log::debug!("VI::GetDisplayService");
        SUCCESS
    }

    fn cmd_get_display_vsync_event(&self) -> u32 {
        log::debug!("VI::GetDisplayVsyncEvent");
        SUCCESS
    }

    fn cmd_get_display_status(&self) -> u32 {
        log::debug!("VI::GetDisplayStatus");
        SUCCESS
    }
}

impl Default for DisplayService {
    fn default() -> Self {
        Self::new()
    }
}
