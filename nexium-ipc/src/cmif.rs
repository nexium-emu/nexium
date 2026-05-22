use bytemuck::{Pod, Zeroable};

pub const CMIF_IN_MAGIC: u32 = u32::from_le_bytes(*b"SFCI");
pub const CMIF_OUT_MAGIC: u32 = u32::from_le_bytes(*b"SFCO");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CmifInHeader {
    pub magic: u32,
    pub version: u32,
    pub cmd_id: u32,
    pub token: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CmifOutHeader {
    pub magic: u32,
    pub version: u32,
    pub result: u32,
    pub token: u32,
}

impl CmifOutHeader {
    pub fn success(token: u32) -> Self {
        Self {
            magic: CMIF_OUT_MAGIC,
            version: 1,
            result: 0,
            token,
        }
    }

    pub fn error(result: u32, token: u32) -> Self {
        Self {
            magic: CMIF_OUT_MAGIC,
            version: 1,
            result,
            token,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CmifDomainInHeader {
    pub kind: u8,
    pub num_in_objects: u8,
    pub data_size: u16,
    pub object_id: u32,
    pub padding: u32,
    pub token: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CmifDomainOutHeader {
    pub num_out_objects: u32,
    pub padding: [u32; 3],
}

pub const CMIF_DOMAIN_REQ_SEND: u8 = 1;
pub const CMIF_DOMAIN_REQ_CLOSE: u8 = 2;
