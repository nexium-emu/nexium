use nexium_common::result::SUCCESS;

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
            100 => self.cmd_set_supported_npad_style_set(),
            101 => self.cmd_get_supported_npad_style_set(),
            102 => self.cmd_set_supported_npad_id_type(),
            103 => self.cmd_activate_npad(),
            120 => self.cmd_set_npad_joy_hold_type(),
            121 => self.cmd_get_npad_joy_hold_type(),
            122 => self.cmd_set_npad_joy_assignment_mode_single_by_default(),
            123 => self.cmd_set_npad_joy_assignment_mode_single(),
            124 => self.cmd_set_npad_joy_assignment_mode_dual(),
            125 => self.cmd_merge_single_joy_as_dual_joy(),
            128 => self.cmd_set_npad_handheld_activation_mode(),
            _ => {
                log::warn!("hid.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)", cmd_id);
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

    fn cmd_get_npad_joy_hold_type(&self) -> u32 {
        log::info!("HID::GetNpadJoyHoldType");
        SUCCESS
    }

    fn cmd_set_supported_npad_id_type(&self) -> u32 {
        log::info!("HID::SetSupportedNpadIdType");
        SUCCESS
    }

    fn cmd_activate_npad(&self) -> u32 {
        log::info!("HID::ActivateNpad");
        SUCCESS
    }

    fn cmd_set_npad_joy_assignment_mode_single_by_default(&self) -> u32 {
        log::info!("HID::SetNpadJoyAssignmentModeSingleByDefault");
        SUCCESS
    }

    fn cmd_set_npad_joy_assignment_mode_single(&self) -> u32 {
        log::info!("HID::SetNpadJoyAssignmentModeSingle");
        SUCCESS
    }

    fn cmd_set_npad_joy_assignment_mode_dual(&self) -> u32 {
        log::info!("HID::SetNpadJoyAssignmentModeDual");
        SUCCESS
    }

    fn cmd_merge_single_joy_as_dual_joy(&self) -> u32 {
        log::info!("HID::MergeSingleJoyAsDualJoy");
        SUCCESS
    }

    fn cmd_set_npad_handheld_activation_mode(&self) -> u32 {
        log::info!("HID::SetNpadHandheldActivationMode");
        SUCCESS
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
