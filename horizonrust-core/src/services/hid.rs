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
        log::info!("hid cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_create_applet_resource(),
            1 => self.cmd_activate_touch_screen(),
            2 => self.cmd_activate_debug_pad(),
            100 => self.cmd_set_supported_npad_style_set(),
            101 => self.cmd_get_supported_npad_style_set(),
            102 => self.cmd_set_npad_joy_hold_type(),
            _ => {
                log::info!("hid stub command: {}", cmd_id);
                SUCCESS
            }
        }
    }

    pub fn get_input_event_handle(&self) -> u32 {
        self.input_event_handle
    }

    fn cmd_create_applet_resource(&self) -> u32 {
        log::info!("HID::CreateAppletResource");
        SUCCESS
    }

    fn cmd_activate_touch_screen(&self) -> u32 {
        log::info!("HID::ActivateTouchScreen");
        SUCCESS
    }

    fn cmd_activate_debug_pad(&self) -> u32 {
        log::info!("HID::ActivateDebugPad");
        SUCCESS
    }

    fn cmd_set_supported_npad_style_set(&self) -> u32 {
        log::info!("HID::SetSupportedNpadStyleSet");
        SUCCESS
    }

    fn cmd_get_supported_npad_style_set(&self) -> u32 {
        log::info!("HID::GetSupportedNpadStyleSet");
        SUCCESS
    }

    fn cmd_set_npad_joy_hold_type(&self) -> u32 {
        log::info!("HID::SetNpadJoyHoldType");
        SUCCESS
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
