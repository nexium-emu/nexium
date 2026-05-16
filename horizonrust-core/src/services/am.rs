use crate::common::result::SUCCESS;

pub struct AppletService;

impl AppletService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("am cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_create_applet_manager(),
            1 => self.cmd_open_library_applet_layer_storage(),
            2 => self.cmd_create_managed_display_layer(),
            _ => {
                log::warn!("unknown am command: {}", cmd_id);
                1
            }
        }
    }

    fn cmd_create_applet_manager(&self) -> u32 {
        log::debug!("AM::CreateAppletManager");
        SUCCESS
    }

    fn cmd_open_library_applet_layer_storage(&self) -> u32 {
        log::debug!("AM::OpenLibraryAppletLayerStorage");
        SUCCESS
    }

    fn cmd_create_managed_display_layer(&self) -> u32 {
        log::debug!("AM::CreateManagedDisplayLayer");
        SUCCESS
    }
}

impl Default for AppletService {
    fn default() -> Self {
        Self::new()
    }
}
