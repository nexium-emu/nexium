pub mod hipc;
pub mod cmif;
pub mod parcel;
pub mod request;

pub use hipc::{HipcHeader, HipcSpecialHeader, HipcCommandType};
pub use hipc::{TLS_BUFFER_SIZE, TLS_REQUEST_OFFSET};
pub use cmif::{CmifInHeader, CmifOutHeader, CmifDomainInHeader, CmifDomainOutHeader};
pub use cmif::{CMIF_IN_MAGIC, CMIF_OUT_MAGIC, CMIF_DOMAIN_REQ_SEND, CMIF_DOMAIN_REQ_CLOSE};

#[derive(Debug)]
pub enum IpcError {
    BufferTooSmall,
    BadCmifMagic,
    BadCommandType,
    ShortInput,
}

pub type IpcResult<T> = Result<T, IpcError>;

#[derive(Copy, Clone, Debug)]
pub struct IpcBuffer {
    pub addr: u64,
    pub size: u64,
    pub mode: u32,
}

pub struct IpcCtx {
    pub buf: Vec<u8>,
    pub hipc: HipcHeader,
    pub cmif_in: CmifInHeader,
    pub cmd_id: u32,
    pub token: u32,
}

impl IpcCtx {
    pub fn new(buf: Vec<u8>) -> IpcResult<Self> {
        if buf.len() < 24 {
            return Err(IpcError::BufferTooSmall);
        }

        let mut hdr_bytes = [0u8; 8];
        hdr_bytes.copy_from_slice(&buf[..8]);
        let hipc = HipcHeader::from_le_bytes(hdr_bytes);

        let mut cmif_bytes = [0u8; 16];
        cmif_bytes.copy_from_slice(&buf[8..24]);
        let cmif_in: CmifInHeader = *bytemuck::from_bytes(&cmif_bytes[..16]);

        if cmif_in.magic != CMIF_IN_MAGIC {
            return Err(IpcError::BadCmifMagic);
        }

        Ok(IpcCtx {
            buf,
            hipc,
            cmif_in,
            cmd_id: cmif_in.cmd_id,
            token: cmif_in.token,
        })
    }

    pub fn build_response(&self, result: u32) -> Vec<u8> {
        let mut response = vec![0u8; 16];
        let cmif_out = if result == 0 {
            CmifOutHeader::success(self.token)
        } else {
            CmifOutHeader::error(result, self.token)
        };
        let bytes = bytemuck::bytes_of(&cmif_out);
        response[..16].copy_from_slice(bytes);
        response
    }
}
