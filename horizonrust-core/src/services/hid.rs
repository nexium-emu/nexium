use crate::common::result::SUCCESS;

pub struct HidService {
    input_event_handle: u32,
    standard_port: u32,
}

impl HidService {
    pub fn new() -> Self {
        Self {
            input_event_handle: 0x100,
            standard_port: 0x101,
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("hid cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_create_applet_resource(),
            1 => self.cmd_activate_touch_screen(),
            2 => self.cmd_activate_debug_pad(),
            _ => {
                log::warn!("unknown hid command: {}", cmd_id);
                1
            }
        }
    }

    pub fn get_input_event_handle(&self) -> u32 {
        self.input_event_handle
    }

    fn cmd_create_applet_resource(&self) -> u32 {
        log::debug!("HID::CreateAppletResource");
        SUCCESS
    }

    fn cmd_activate_touch_screen(&self) -> u32 {
        log::debug!("HID::ActivateTouchScreen");
        SUCCESS
    }

    fn cmd_activate_debug_pad(&self) -> u32 {
        log::debug!("HID::ActivateDebugPad");
        SUCCESS
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
