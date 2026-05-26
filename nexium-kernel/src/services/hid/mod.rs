use nexium_common::result::SUCCESS;
use std::collections::HashMap;

pub mod handlers;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NpadAssignmentMode {
    Dual,
    Single,
}

#[derive(Copy, Clone, Debug)]
pub struct NpadAssignment {
    pub mode: NpadAssignmentMode,
    pub device_type: i64,
}

impl Default for NpadAssignment {
    fn default() -> Self {
        Self { mode: NpadAssignmentMode::Dual, device_type: 0 }
    }
}

pub struct HidService {
    input_event_handle: u32,
    standard_port: u32,
    pub npad_assignment: HashMap<u32, NpadAssignment>,
    pub npad_handheld_activation_mode: u64,
    pub npad_joy_hold_type: u64,
}

impl HidService {
    pub fn new() -> Self {
        Self {
            input_event_handle: 0x100,
            standard_port: 0x101,
            npad_assignment: HashMap::new(),
            npad_handheld_activation_mode: 0,
            npad_joy_hold_type: 0,
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("hid cmd: {}", cmd_id);
        match cmd_id {
            0 | 1 | 2 | 100 | 101 | 102 | 103 | 120 | 121 | 122 | 123 | 124 | 125 | 128 => SUCCESS,
            _ => {
                log::warn!("hid.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)", cmd_id);
                SUCCESS
            }
        }
    }

    pub fn get_input_event_handle(&self) -> u32 {
        self.input_event_handle
    }

    pub fn set_npad_assignment_single_by_default(&mut self, npad_id: u32, _aruid: u64) {
        let entry = self.npad_assignment.entry(npad_id).or_default();
        entry.mode = NpadAssignmentMode::Single;
        entry.device_type = 0;
        log::info!("HID::SetNpadJoyAssignmentModeSingleByDefault npad_id={:#x}", npad_id);
    }

    pub fn set_npad_assignment_single(&mut self, npad_id: u32, _aruid: u64, device_type: i64) {
        let entry = self.npad_assignment.entry(npad_id).or_default();
        entry.mode = NpadAssignmentMode::Single;
        entry.device_type = device_type;
        log::info!("HID::SetNpadJoyAssignmentModeSingle npad_id={:#x} device_type={}", npad_id, device_type);
    }

    pub fn set_npad_assignment_dual(&mut self, npad_id: u32, _aruid: u64) {
        let entry = self.npad_assignment.entry(npad_id).or_default();
        entry.mode = NpadAssignmentMode::Dual;
        entry.device_type = 0;
        log::info!("HID::SetNpadJoyAssignmentModeDual npad_id={:#x}", npad_id);
    }

    pub fn merge_single_joy_as_dual_joy(&mut self, npad_id_left: u32, npad_id_right: u32, _aruid: u64) {
        let l = self.npad_assignment.entry(npad_id_left).or_default();
        l.mode = NpadAssignmentMode::Dual;
        l.device_type = 0;
        let r = self.npad_assignment.entry(npad_id_right).or_default();
        r.mode = NpadAssignmentMode::Dual;
        r.device_type = 0;
        log::info!("HID::MergeSingleJoyAsDualJoy left={:#x} right={:#x}", npad_id_left, npad_id_right);
    }

    pub fn set_npad_handheld_activation_mode(&mut self, mode: u64) {
        self.npad_handheld_activation_mode = mode;
        log::info!("HID::SetNpadHandheldActivationMode mode={}", mode);
    }

    pub fn set_npad_joy_hold_type(&mut self, ty: u64) {
        self.npad_joy_hold_type = ty;
        log::info!("HID::SetNpadJoyHoldType type={}", ty);
    }

    pub fn get_npad_joy_hold_type(&self) -> u64 {
        self.npad_joy_hold_type
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
