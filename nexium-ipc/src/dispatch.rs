use nexium_common::result::SUCCESS;

pub struct IpcDispatcher;

impl IpcDispatcher {
    pub fn parse_request(buffer: &[u8]) -> Result<IpcRequestParsed, &'static str> {
        if buffer.len() < 16 {
            return Err("buffer too small");
        }

        let magic = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 0x49435346 {
            return Err("bad SFCI magic");
        }

        let _version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        let cmd_id = u32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
        let token = u32::from_le_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]);

        Ok(IpcRequestParsed {
            cmd_id,
            token,
            data_offset: 16,
        })
    }

    pub fn build_response(token: u32, result: u32, response_data: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 16 + response_data.len()];

        buf[0..4].copy_from_slice(&0x4F435346u32.to_le_bytes());
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&result.to_le_bytes());
        buf[12..16].copy_from_slice(&token.to_le_bytes());

        if !response_data.is_empty() {
            buf[16..].copy_from_slice(response_data);
        }

        buf
    }
}

pub struct IpcRequestParsed {
    pub cmd_id: u32,
    pub token: u32,
    pub data_offset: usize,
}

pub struct IpcCommandHandler {
    pub name: String,
    pub cmd_id: u32,
}

impl IpcCommandHandler {
    pub fn new(name: &str, cmd_id: u32) -> Self {
        Self {
            name: name.to_string(),
            cmd_id,
        }
    }

    pub fn handle_sm_command(cmd_id: u32) -> (u32, Vec<u8>) {
        match cmd_id {
            0 => {
                log::debug!("SM::RegisterService");
                (SUCCESS, vec![])
            }
            1 => {
                log::debug!("SM::UnregisterService");
                (SUCCESS, vec![])
            }
            2 => {
                log::debug!("SM::GetServiceHandle");
                (SUCCESS, vec![])
            }
            3 => {
                log::debug!("SM::RegisterServiceForDomain");
                (SUCCESS, vec![])
            }
            _ => {
                log::warn!("unknown SM command: {}", cmd_id);
                (1, vec![])
            }
        }
    }

    pub fn handle_hid_command(cmd_id: u32) -> (u32, Vec<u8>) {
        match cmd_id {
            0 => {
                log::debug!("HID::CreateAppletResource");
                (SUCCESS, vec![])
            }
            _ => {
                log::warn!("unknown HID command: {}", cmd_id);
                (1, vec![])
            }
        }
    }

    pub fn handle_time_command(cmd_id: u32) -> (u32, Vec<u8>) {
        match cmd_id {
            0 => {
                log::debug!("Time::GetSystemTime");
                let mut response = vec![0u8; 8];
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                response.copy_from_slice(&now.to_le_bytes());
                (SUCCESS, response)
            }
            _ => {
                log::warn!("unknown Time command: {}", cmd_id);
                (1, vec![])
            }
        }
    }
}
