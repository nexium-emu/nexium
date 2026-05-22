use crate::parcel::ParcelReader;
use nexium_common::result::SUCCESS;

pub struct IpcRequest {
    pub cmd_id: u32,
    pub token: u32,
    pub data: Vec<u8>,
}

impl IpcRequest {
    pub fn parse(data: Vec<u8>) -> Result<Self, &'static str> {
        if data.len() < 16 {
            return Err("request too small");
        }

        let mut reader = ParcelReader::new(data.clone());

        let magic = reader.read_u32()?;
        if magic != 0x49435346 {
            return Err("bad magic");
        }

        let _version = reader.read_u32()?;
        let cmd_id = reader.read_u32()?;
        let token = reader.read_u32()?;

        Ok(IpcRequest {
            cmd_id,
            token,
            data,
        })
    }

    pub fn get_reader(&self) -> ParcelReader {
        ParcelReader::new(self.data.clone())
    }
}

pub struct IpcResponse {
    pub result: u32,
    pub data: Vec<u8>,
}

impl IpcResponse {
    pub fn success(token: u32) -> Self {
        let mut data = vec![0u8; 16];
        let magic = 0x4F435346u32;
        let version = 1u32;
        let result = SUCCESS;

        data[0..4].copy_from_slice(&magic.to_le_bytes());
        data[4..8].copy_from_slice(&version.to_le_bytes());
        data[8..12].copy_from_slice(&result.to_le_bytes());
        data[12..16].copy_from_slice(&token.to_le_bytes());

        IpcResponse {
            result: SUCCESS,
            data,
        }
    }

    pub fn error(result: u32, token: u32) -> Self {
        let mut data = vec![0u8; 16];
        let magic = 0x4F435346u32;
        let version = 1u32;

        data[0..4].copy_from_slice(&magic.to_le_bytes());
        data[4..8].copy_from_slice(&version.to_le_bytes());
        data[8..12].copy_from_slice(&result.to_le_bytes());
        data[12..16].copy_from_slice(&token.to_le_bytes());

        IpcResponse {
            result,
            data,
        }
    }

    pub fn with_data(token: u32, response_data: &[u8]) -> Self {
        let mut data = vec![0u8; 16 + response_data.len()];
        let magic = 0x4F435346u32;
        let version = 1u32;
        let result = SUCCESS;

        data[0..4].copy_from_slice(&magic.to_le_bytes());
        data[4..8].copy_from_slice(&version.to_le_bytes());
        data[8..12].copy_from_slice(&result.to_le_bytes());
        data[12..16].copy_from_slice(&token.to_le_bytes());
        data[16..].copy_from_slice(response_data);

        IpcResponse {
            result: SUCCESS,
            data,
        }
    }
}
