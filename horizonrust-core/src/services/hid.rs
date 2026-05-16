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
            0 => SUCCESS,
            1 => SUCCESS,
            2 => SUCCESS,
            _ => {
                log::warn!("unknown hid command: {}", cmd_id);
                1
            }
        }
    }

    pub fn get_input_event_handle(&self) -> u32 {
        self.input_event_handle
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}
