use crate::common::result::SUCCESS;

pub struct DisplayService;

impl DisplayService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("vi cmd: {}", cmd_id);
        match cmd_id {
            1010 => self.open_display(),
            1020 => self.close_display(),
            2020 => self.open_layer(),
            2030 => self.create_stray_layer(),
            7000 => self.get_display_vsync_event(),
            _ => {
                log::warn!("unknown vi command: {}", cmd_id);
                SUCCESS
            }
        }
    }

    fn open_display(&self) -> u32 {
        log::debug!("VI::OpenDisplay");
        SUCCESS
    }

    fn close_display(&self) -> u32 {
        log::debug!("VI::CloseDisplay");
        SUCCESS
    }

    fn open_layer(&self) -> u32 {
        log::debug!("VI::OpenLayer");
        SUCCESS
    }

    fn create_stray_layer(&self) -> u32 {
        log::debug!("VI::CreateStrayLayer");
        SUCCESS
    }

    fn get_display_vsync_event(&self) -> u32 {
        log::debug!("VI::GetDisplayVsyncEvent");
        SUCCESS
    }
}

impl Default for DisplayService {
    fn default() -> Self {
        Self::new()
    }
}
