#![allow(unused_parens)]

use modular_bitfield::{bitfield, prelude::*};
use num_enum::TryFromPrimitive;

pub const TLS_BUFFER_SIZE: usize = 0x100;
pub const TLS_REQUEST_OFFSET: u64 = 0;

#[derive(Copy, Clone, Debug, Eq, PartialEq, TryFromPrimitive)]
#[repr(u16)]
pub enum HipcCommandType {
    Invalid = 0,
    LegacyRequest = 1,
    Close = 2,
    LegacyControl = 3,
    Request = 4,
    Control = 5,
    RequestWithContext = 6,
    ControlWithContext = 7,
}

#[bitfield(bits = 64)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct HipcHeader {
    pub command_type: B16,
    pub num_send_statics: B4,
    pub num_send_buffers: B4,
    pub num_recv_buffers: B4,
    pub num_exch_buffers: B4,
    pub num_data_words: B10,
    pub recv_static_mode: B4,
    pub padding: B6,
    pub recv_list_offset: B11,
    pub has_special_header: bool,
}

impl HipcHeader {
    pub fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self::from_bytes(bytes)
    }

    pub fn to_le_bytes(self) -> [u8; 8] {
        self.into_bytes()
    }
}

#[bitfield(bits = 32)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct HipcSpecialHeader {
    pub send_pid: bool,
    pub num_copy_handles: B4,
    pub num_move_handles: B4,
    #[skip]
    __: B23,
}

impl HipcSpecialHeader {
    pub fn from_u32(val: u32) -> Self {
        Self::from_bytes(val.to_le_bytes())
    }

    pub fn to_u32(self) -> u32 {
        u32::from_le_bytes(self.into_bytes())
    }
}
