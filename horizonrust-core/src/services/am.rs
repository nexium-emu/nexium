use crate::common::result::SUCCESS;

pub struct AppletService;

impl AppletService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("am cmd: {}", cmd_id);
        match cmd_id {
            0 => self.open_application_proxy(),
            40 => self.create_managed_display_layer(),
            100 => self.open_system_applet_proxy(),
            200 | 201 => self.open_library_applet_proxy(),
            300 => self.open_overlay_applet_proxy(),
            350 => self.open_system_application_proxy(),
            _ => {
                log::warn!("unknown am command: {}", cmd_id);
                SUCCESS
            }
        }
    }

    fn open_application_proxy(&self) -> u32 {
        log::debug!("AM::OpenApplicationProxy");
        SUCCESS
    }

    fn create_managed_display_layer(&self) -> u32 {
        log::debug!("AM::CreateManagedDisplayLayer");
        SUCCESS
    }

    fn open_system_applet_proxy(&self) -> u32 {
        log::debug!("AM::OpenSystemAppletProxy");
        SUCCESS
    }

    fn open_library_applet_proxy(&self) -> u32 {
        log::debug!("AM::OpenLibraryAppletProxy");
        SUCCESS
    }

    fn open_overlay_applet_proxy(&self) -> u32 {
        log::debug!("AM::OpenOverlayAppletProxy");
        SUCCESS
    }

    fn open_system_application_proxy(&self) -> u32 {
        log::debug!("AM::OpenSystemApplicationProxy");
        SUCCESS
    }
}

impl Default for AppletService {
    fn default() -> Self {
        Self::new()
    }
}
