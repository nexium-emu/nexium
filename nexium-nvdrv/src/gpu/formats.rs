use ash::vk;

pub const FMT_R32G32B32A32_FLOAT: u32 = 0xC0;
pub const FMT_R32G32B32A32_SINT: u32 = 0xC1;
pub const FMT_R32G32B32A32_UINT: u32 = 0xC2;
pub const FMT_R32G32B32X32_FLOAT: u32 = 0xC3;
pub const FMT_R32G32B32X32_SINT: u32 = 0xC4;
pub const FMT_R32G32B32X32_UINT: u32 = 0xC5;
pub const FMT_R16G16B16A16_UNORM: u32 = 0xC6;
pub const FMT_R16G16B16A16_SNORM: u32 = 0xC7;
pub const FMT_R16G16B16A16_SINT: u32 = 0xC8;
pub const FMT_R16G16B16A16_UINT: u32 = 0xC9;
pub const FMT_R16G16B16A16_FLOAT: u32 = 0xCA;
pub const FMT_R32G32_FLOAT: u32 = 0xCB;
pub const FMT_R32G32_SINT: u32 = 0xCC;
pub const FMT_R32G32_UINT: u32 = 0xCD;
pub const FMT_R16G16B16X16_FLOAT: u32 = 0xCE;
pub const FMT_A8R8G8B8_UNORM: u32 = 0xCF;
pub const FMT_A8R8G8B8_SRGB: u32 = 0xD0;
pub const FMT_A2B10G10R10_UNORM: u32 = 0xD1;
pub const FMT_A2B10G10R10_UINT: u32 = 0xD2;
pub const FMT_A8B8G8R8_UNORM: u32 = 0xD5;
pub const FMT_A8B8G8R8_SRGB: u32 = 0xD6;
pub const FMT_A8B8G8R8_SNORM: u32 = 0xD7;
pub const FMT_A8B8G8R8_SINT: u32 = 0xD8;
pub const FMT_A8B8G8R8_UINT: u32 = 0xD9;
pub const FMT_R16G16_UNORM: u32 = 0xDA;
pub const FMT_R16G16_SNORM: u32 = 0xDB;
pub const FMT_R16G16_SINT: u32 = 0xDC;
pub const FMT_R16G16_UINT: u32 = 0xDD;
pub const FMT_R16G16_FLOAT: u32 = 0xDE;
pub const FMT_A2R10G10B10_UNORM: u32 = 0xDF;
pub const FMT_B10G11R11_FLOAT: u32 = 0xE0;
pub const FMT_R32_SINT: u32 = 0xE3;
pub const FMT_R32_UINT: u32 = 0xE4;
pub const FMT_R32_FLOAT: u32 = 0xE5;
pub const FMT_X8R8G8B8_UNORM: u32 = 0xE6;
pub const FMT_X8R8G8B8_SRGB: u32 = 0xE7;
pub const FMT_R5G6B5_UNORM: u32 = 0xE8;
pub const FMT_A1R5G5B5_UNORM: u32 = 0xE9;
pub const FMT_R8G8_UNORM: u32 = 0xEA;
pub const FMT_R8G8_SNORM: u32 = 0xEB;
pub const FMT_R8G8_SINT: u32 = 0xEC;
pub const FMT_R8G8_UINT: u32 = 0xED;
pub const FMT_R16_UNORM: u32 = 0xEE;
pub const FMT_R16_SNORM: u32 = 0xEF;
pub const FMT_R16_SINT: u32 = 0xF0;
pub const FMT_R16_UINT: u32 = 0xF1;
pub const FMT_R16_FLOAT: u32 = 0xF2;
pub const FMT_R8_UNORM: u32 = 0xF3;
pub const FMT_R8_SNORM: u32 = 0xF4;
pub const FMT_R8_SINT: u32 = 0xF5;
pub const FMT_R8_UINT: u32 = 0xF6;
pub const FMT_Y1_8X8: u32 = 0x1C;
pub const FMT_AY8: u32 = 0x1D;
pub const FMT_A8_UNORM: u32 = 0xF7;
pub const FMT_X1R5G5B5_UNORM: u32 = 0xF8;
pub const FMT_X8B8G8R8_UNORM: u32 = 0xF9;
pub const FMT_X8B8G8R8_SRGB: u32 = 0xFA;
pub const FMT_Z1R5G5B5: u32 = 0xFB;
pub const FMT_O1R5G5B5: u32 = 0xFC;
pub const FMT_Z8R8G8B8: u32 = 0xFD;
pub const FMT_O8R8G8B8: u32 = 0xFE;
pub const FMT_Y32: u32 = 0xFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceFormat {
    R32G32B32A32Float,
    R32G32B32A32Sint,
    R32G32B32A32Uint,
    R32G32B32X32Float,
    R32G32B32X32Sint,
    R32G32B32X32Uint,
    R16G16B16A16Unorm,
    R16G16B16A16Snorm,
    R16G16B16A16Sint,
    R16G16B16A16Uint,
    R16G16B16A16Float,
    R32G32Float,
    R32G32Sint,
    R32G32Uint,
    R16G16B16X16Float,
    A8R8G8B8Unorm,
    A8R8G8B8Srgb,
    A2B10G10R10Unorm,
    A2B10G10R10Uint,
    A8B8G8R8Unorm,
    A8B8G8R8Srgb,
    A8B8G8R8Snorm,
    A8B8G8R8Sint,
    A8B8G8R8Uint,
    R16G16Unorm,
    R16G16Snorm,
    R16G16Sint,
    R16G16Uint,
    R16G16Float,
    A2R10G10B10Unorm,
    B10G11R11Float,
    R32Sint,
    R32Uint,
    R32Float,
    X8R8G8B8Unorm,
    X8R8G8B8Srgb,
    R5G6B5Unorm,
    A1R5G5B5Unorm,
    R8G8Unorm,
    R8G8Snorm,
    R8G8Sint,
    R8G8Uint,
    R16Unorm,
    R16Snorm,
    R16Sint,
    R16Uint,
    R16Float,
    R8Unorm,
    R8Snorm,
    R8Sint,
    R8Uint,
    Y1_8X8,
    AY8,
    X8B8G8R8Unorm,
    X8B8G8R8Srgb,
    X1R5G5B5Unorm,
    A8Unorm,
    Y32,
    Z1R5G5B5,
    O1R5G5B5,
    Z8R8G8B8,
    O8R8G8B8,
}

impl SurfaceFormat {
    pub fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw & 0xff {
            FMT_R32G32B32A32_FLOAT => Self::R32G32B32A32Float,
            FMT_R32G32B32A32_SINT => Self::R32G32B32A32Sint,
            FMT_R32G32B32A32_UINT => Self::R32G32B32A32Uint,
            FMT_R32G32B32X32_FLOAT => Self::R32G32B32X32Float,
            FMT_R32G32B32X32_SINT => Self::R32G32B32X32Sint,
            FMT_R32G32B32X32_UINT => Self::R32G32B32X32Uint,
            FMT_R16G16B16A16_UNORM => Self::R16G16B16A16Unorm,
            FMT_R16G16B16A16_SNORM => Self::R16G16B16A16Snorm,
            FMT_R16G16B16A16_SINT => Self::R16G16B16A16Sint,
            FMT_R16G16B16A16_UINT => Self::R16G16B16A16Uint,
            FMT_R16G16B16A16_FLOAT => Self::R16G16B16A16Float,
            FMT_R32G32_FLOAT => Self::R32G32Float,
            FMT_R32G32_SINT => Self::R32G32Sint,
            FMT_R32G32_UINT => Self::R32G32Uint,
            FMT_R16G16B16X16_FLOAT => Self::R16G16B16X16Float,
            FMT_A8R8G8B8_UNORM => Self::A8R8G8B8Unorm,
            FMT_A8R8G8B8_SRGB => Self::A8R8G8B8Srgb,
            FMT_A2B10G10R10_UNORM => Self::A2B10G10R10Unorm,
            FMT_A2B10G10R10_UINT => Self::A2B10G10R10Uint,
            FMT_A8B8G8R8_UNORM => Self::A8B8G8R8Unorm,
            FMT_A8B8G8R8_SRGB => Self::A8B8G8R8Srgb,
            FMT_A8B8G8R8_SNORM => Self::A8B8G8R8Snorm,
            FMT_A8B8G8R8_SINT => Self::A8B8G8R8Sint,
            FMT_A8B8G8R8_UINT => Self::A8B8G8R8Uint,
            FMT_R16G16_UNORM => Self::R16G16Unorm,
            FMT_R16G16_SNORM => Self::R16G16Snorm,
            FMT_R16G16_SINT => Self::R16G16Sint,
            FMT_R16G16_UINT => Self::R16G16Uint,
            FMT_R16G16_FLOAT => Self::R16G16Float,
            FMT_A2R10G10B10_UNORM => Self::A2R10G10B10Unorm,
            FMT_B10G11R11_FLOAT => Self::B10G11R11Float,
            FMT_R32_SINT => Self::R32Sint,
            FMT_R32_UINT => Self::R32Uint,
            FMT_R32_FLOAT => Self::R32Float,
            FMT_X8R8G8B8_UNORM => Self::X8R8G8B8Unorm,
            FMT_X8R8G8B8_SRGB => Self::X8R8G8B8Srgb,
            FMT_R5G6B5_UNORM => Self::R5G6B5Unorm,
            FMT_A1R5G5B5_UNORM => Self::A1R5G5B5Unorm,
            FMT_R8G8_UNORM => Self::R8G8Unorm,
            FMT_R8G8_SNORM => Self::R8G8Snorm,
            FMT_R8G8_SINT => Self::R8G8Sint,
            FMT_R8G8_UINT => Self::R8G8Uint,
            FMT_R16_UNORM => Self::R16Unorm,
            FMT_R16_SNORM => Self::R16Snorm,
            FMT_R16_SINT => Self::R16Sint,
            FMT_R16_UINT => Self::R16Uint,
            FMT_R16_FLOAT => Self::R16Float,
            FMT_R8_UNORM => Self::R8Unorm,
            FMT_R8_SNORM => Self::R8Snorm,
            FMT_R8_SINT => Self::R8Sint,
            FMT_R8_UINT => Self::R8Uint,
            FMT_Y1_8X8 => Self::Y1_8X8,
            FMT_AY8 => Self::AY8,
            FMT_A8_UNORM => Self::A8Unorm,
            FMT_X1R5G5B5_UNORM => Self::X1R5G5B5Unorm,
            FMT_X8B8G8R8_UNORM => Self::X8B8G8R8Unorm,
            FMT_X8B8G8R8_SRGB => Self::X8B8G8R8Srgb,
            FMT_Z1R5G5B5 => Self::Z1R5G5B5,
            FMT_O1R5G5B5 => Self::O1R5G5B5,
            FMT_Z8R8G8B8 => Self::Z8R8G8B8,
            FMT_O8R8G8B8 => Self::O8R8G8B8,
            FMT_Y32 => Self::Y32,
            _ => return None,
        })
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R32G32B32A32Float
            | Self::R32G32B32A32Sint
            | Self::R32G32B32A32Uint
            | Self::R32G32B32X32Float
            | Self::R32G32B32X32Sint
            | Self::R32G32B32X32Uint => 16,
            Self::R16G16B16A16Unorm
            | Self::R16G16B16A16Snorm
            | Self::R16G16B16A16Sint
            | Self::R16G16B16A16Uint
            | Self::R16G16B16A16Float
            | Self::R32G32Float
            | Self::R32G32Sint
            | Self::R32G32Uint
            | Self::R16G16B16X16Float => 8,
            Self::A8R8G8B8Unorm
            | Self::A8R8G8B8Srgb
            | Self::A2B10G10R10Unorm
            | Self::A2B10G10R10Uint
            | Self::A8B8G8R8Unorm
            | Self::A8B8G8R8Srgb
            | Self::A8B8G8R8Snorm
            | Self::A8B8G8R8Sint
            | Self::A8B8G8R8Uint
            | Self::R16G16Unorm
            | Self::R16G16Snorm
            | Self::R16G16Sint
            | Self::R16G16Uint
            | Self::R16G16Float
            | Self::A2R10G10B10Unorm
            | Self::B10G11R11Float
            | Self::R32Sint
            | Self::R32Uint
            | Self::R32Float
            | Self::X8R8G8B8Unorm
            | Self::X8R8G8B8Srgb
            | Self::X8B8G8R8Unorm
            | Self::X8B8G8R8Srgb
            | Self::Z8R8G8B8
            | Self::O8R8G8B8
            | Self::Y32 => 4,
            Self::R5G6B5Unorm
            | Self::A1R5G5B5Unorm
            | Self::R8G8Unorm
            | Self::R8G8Snorm
            | Self::R8G8Sint
            | Self::R8G8Uint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Sint
            | Self::R16Uint
            | Self::R16Float
            | Self::Y1_8X8
            | Self::AY8
            | Self::X1R5G5B5Unorm
            | Self::Z1R5G5B5
            | Self::O1R5G5B5 => 2,
            Self::R8Unorm | Self::R8Snorm | Self::R8Sint | Self::R8Uint | Self::A8Unorm => 1,
        }
    }

    pub fn vk_format(self) -> Option<vk::Format> {
        Some(match self {
            Self::R32G32B32A32Float | Self::R32G32B32X32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::R32G32B32A32Sint | Self::R32G32B32X32Sint => vk::Format::R32G32B32A32_SINT,
            Self::R32G32B32A32Uint | Self::R32G32B32X32Uint => vk::Format::R32G32B32A32_UINT,
            Self::R16G16B16A16Unorm => vk::Format::R16G16B16A16_UNORM,
            Self::R16G16B16A16Snorm => vk::Format::R16G16B16A16_SNORM,
            Self::R16G16B16A16Sint => vk::Format::R16G16B16A16_SINT,
            Self::R16G16B16A16Uint => vk::Format::R16G16B16A16_UINT,
            Self::R16G16B16A16Float | Self::R16G16B16X16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::R32G32Float => vk::Format::R32G32_SFLOAT,
            Self::R32G32Sint => vk::Format::R32G32_SINT,
            Self::R32G32Uint => vk::Format::R32G32_UINT,
            Self::A8R8G8B8Unorm | Self::X8R8G8B8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::A8R8G8B8Srgb | Self::X8R8G8B8Srgb => vk::Format::B8G8R8A8_SRGB,
            Self::A2B10G10R10Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
            Self::A2B10G10R10Uint => vk::Format::A2B10G10R10_UINT_PACK32,
            Self::A8B8G8R8Unorm | Self::X8B8G8R8Unorm => vk::Format::A8B8G8R8_UNORM_PACK32,
            Self::A8B8G8R8Srgb | Self::X8B8G8R8Srgb => vk::Format::A8B8G8R8_SRGB_PACK32,
            Self::A8B8G8R8Snorm => vk::Format::A8B8G8R8_SNORM_PACK32,
            Self::A8B8G8R8Sint => vk::Format::A8B8G8R8_SINT_PACK32,
            Self::A8B8G8R8Uint => vk::Format::A8B8G8R8_UINT_PACK32,
            Self::R16G16Unorm => vk::Format::R16G16_UNORM,
            Self::R16G16Snorm => vk::Format::R16G16_SNORM,
            Self::R16G16Sint => vk::Format::R16G16_SINT,
            Self::R16G16Uint => vk::Format::R16G16_UINT,
            Self::R16G16Float => vk::Format::R16G16_SFLOAT,
            Self::A2R10G10B10Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
            Self::B10G11R11Float => vk::Format::B10G11R11_UFLOAT_PACK32,
            Self::R32Sint => vk::Format::R32_SINT,
            Self::R32Uint => vk::Format::R32_UINT,
            Self::R32Float | Self::Y32 => vk::Format::R32_SFLOAT,
            Self::R5G6B5Unorm => vk::Format::R5G6B5_UNORM_PACK16,
            Self::A1R5G5B5Unorm | Self::X1R5G5B5Unorm => vk::Format::B5G5R5A1_UNORM_PACK16,
            Self::R8G8Unorm => vk::Format::R8G8_UNORM,
            Self::R8G8Snorm => vk::Format::R8G8_SNORM,
            Self::R8G8Sint => vk::Format::R8G8_SINT,
            Self::R8G8Uint => vk::Format::R8G8_UINT,
            Self::R16Unorm => vk::Format::R16_UNORM,
            Self::R16Snorm => vk::Format::R16_SNORM,
            Self::R16Sint => vk::Format::R16_SINT,
            Self::R16Uint => vk::Format::R16_UINT,
            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::R8Unorm | Self::A8Unorm => vk::Format::R8_UNORM,
            Self::R8Snorm => vk::Format::R8_SNORM,
            Self::R8Sint => vk::Format::R8_SINT,
            Self::R8Uint => vk::Format::R8_UINT,
            Self::Y1_8X8 | Self::AY8 => vk::Format::R8G8_UNORM,
            Self::Z1R5G5B5 | Self::O1R5G5B5 => vk::Format::B5G5R5A1_UNORM_PACK16,
            Self::Z8R8G8B8 | Self::O8R8G8B8 => vk::Format::B8G8R8A8_UNORM,
        })
    }

    pub fn decode_rgba8(self, src: &[u8]) -> Option<[u8; 4]> {
        let get_u16 = || Some(u16::from_le_bytes(src.get(0..2)?.try_into().ok()?));
        let get_u32 = || Some(u32::from_le_bytes(src.get(0..4)?.try_into().ok()?));
        let unorm = |value: u32, max: u32| ((value * 255 + max / 2) / max) as u8;
        Some(match self {
            Self::A8R8G8B8Unorm | Self::A8R8G8B8Srgb | Self::X8R8G8B8Unorm | Self::X8R8G8B8Srgb => {
                [
                    src[2],
                    src[1],
                    src[0],
                    if src[3] == 0 && matches!(self, Self::X8R8G8B8Unorm | Self::X8R8G8B8Srgb) {
                        255
                    } else {
                        src[3]
                    },
                ]
            }
            Self::A8B8G8R8Unorm | Self::A8B8G8R8Srgb | Self::X8B8G8R8Unorm | Self::X8B8G8R8Srgb => {
                [
                    src[0],
                    src[1],
                    src[2],
                    if matches!(self, Self::X8B8G8R8Unorm | Self::X8B8G8R8Srgb) {
                        255
                    } else {
                        src[3]
                    },
                ]
            }
            Self::A2B10G10R10Unorm => {
                let value = get_u32()?;
                [
                    unorm(value & 0x3ff, 0x3ff),
                    unorm((value >> 10) & 0x3ff, 0x3ff),
                    unorm((value >> 20) & 0x3ff, 0x3ff),
                    unorm((value >> 30) & 0x3, 0x3),
                ]
            }
            Self::A2R10G10B10Unorm => {
                let value = get_u32()?;
                [
                    unorm((value >> 20) & 0x3ff, 0x3ff),
                    unorm((value >> 10) & 0x3ff, 0x3ff),
                    unorm(value & 0x3ff, 0x3ff),
                    unorm((value >> 30) & 0x3, 0x3),
                ]
            }
            Self::R5G6B5Unorm => {
                let value = get_u16()? as u32;
                [
                    unorm((value >> 11) & 0x1f, 0x1f),
                    unorm((value >> 5) & 0x3f, 0x3f),
                    unorm(value & 0x1f, 0x1f),
                    255,
                ]
            }
            Self::A1R5G5B5Unorm | Self::X1R5G5B5Unorm => {
                let value = get_u16()? as u32;
                [
                    unorm((value >> 10) & 0x1f, 0x1f),
                    unorm((value >> 5) & 0x1f, 0x1f),
                    unorm(value & 0x1f, 0x1f),
                    if matches!(self, Self::X1R5G5B5Unorm) {
                        255
                    } else {
                        unorm((value >> 15) & 1, 1)
                    },
                ]
            }
            Self::R8G8Unorm
            | Self::R8G8Snorm
            | Self::R8G8Sint
            | Self::R8G8Uint
            | Self::Y1_8X8
            | Self::AY8 => [src[0], src[1], 0, 255],
            Self::R16Unorm | Self::R16Snorm | Self::R16Sint | Self::R16Uint | Self::R16Float => {
                let value = get_u16()? as u32;
                let value = if matches!(self, Self::R16Unorm) {
                    unorm(value, u16::MAX as u32)
                } else {
                    (value >> 8) as u8
                };
                [value, value, value, 255]
            }
            Self::R8Unorm | Self::R8Snorm | Self::R8Sint | Self::R8Uint | Self::A8Unorm => [
                src[0],
                src[0],
                src[0],
                if matches!(self, Self::A8Unorm) {
                    src[0]
                } else {
                    255
                },
            ],
            _ => return None,
        })
    }

    pub fn encode_rgba8(self, rgba: [u8; 4], dst: &mut [u8]) -> bool {
        let pack = |value: u32| value.to_le_bytes();
        match self {
            Self::A8R8G8B8Unorm | Self::A8R8G8B8Srgb | Self::X8R8G8B8Unorm | Self::X8R8G8B8Srgb => {
                if dst.len() < 4 {
                    return false;
                }
                dst[..4].copy_from_slice(&[
                    rgba[2],
                    rgba[1],
                    rgba[0],
                    if matches!(self, Self::X8R8G8B8Unorm | Self::X8R8G8B8Srgb) {
                        255
                    } else {
                        rgba[3]
                    },
                ]);
            }
            Self::A8B8G8R8Unorm | Self::A8B8G8R8Srgb | Self::X8B8G8R8Unorm | Self::X8B8G8R8Srgb => {
                if dst.len() < 4 {
                    return false;
                }
                dst[..4].copy_from_slice(&[
                    rgba[0],
                    rgba[1],
                    rgba[2],
                    if matches!(self, Self::X8B8G8R8Unorm | Self::X8B8G8R8Srgb) {
                        255
                    } else {
                        rgba[3]
                    },
                ]);
            }
            Self::A2B10G10R10Unorm => {
                if dst.len() < 4 {
                    return false;
                }
                let value = ((rgba[0] as u32 * 0x3ff + 127) / 255)
                    | (((rgba[1] as u32 * 0x3ff + 127) / 255) << 10)
                    | (((rgba[2] as u32 * 0x3ff + 127) / 255) << 20)
                    | (((rgba[3] as u32 * 3 + 127) / 255) << 30);
                dst[..4].copy_from_slice(&pack(value));
            }
            Self::A2R10G10B10Unorm => {
                if dst.len() < 4 {
                    return false;
                }
                let value = (((rgba[0] as u32 * 0x3ff + 127) / 255) << 20)
                    | (((rgba[1] as u32 * 0x3ff + 127) / 255) << 10)
                    | ((rgba[2] as u32 * 0x3ff + 127) / 255)
                    | (((rgba[3] as u32 * 3 + 127) / 255) << 30);
                dst[..4].copy_from_slice(&pack(value));
            }
            Self::R5G6B5Unorm => {
                if dst.len() < 2 {
                    return false;
                }
                let value = (((rgba[0] as u16 * 31 + 127) / 255) << 11)
                    | (((rgba[1] as u16 * 63 + 127) / 255) << 5)
                    | ((rgba[2] as u16 * 31 + 127) / 255);
                dst[..2].copy_from_slice(&value.to_le_bytes());
            }
            Self::A1R5G5B5Unorm | Self::X1R5G5B5Unorm => {
                if dst.len() < 2 {
                    return false;
                }
                let alpha = if matches!(self, Self::X1R5G5B5Unorm) {
                    1
                } else {
                    (rgba[3] >= 128) as u16
                };
                let value = (((rgba[0] as u16 * 31 + 127) / 255) << 10)
                    | (((rgba[1] as u16 * 31 + 127) / 255) << 5)
                    | ((rgba[2] as u16 * 31 + 127) / 255)
                    | (alpha << 15);
                dst[..2].copy_from_slice(&value.to_le_bytes());
            }
            Self::R8G8Unorm
            | Self::R8G8Snorm
            | Self::R8G8Sint
            | Self::R8G8Uint
            | Self::Y1_8X8
            | Self::AY8 => {
                if dst.len() < 2 {
                    return false;
                }
                dst[..2].copy_from_slice(&rgba[..2]);
            }
            Self::R16Unorm | Self::R16Snorm | Self::R16Sint | Self::R16Uint | Self::R16Float => {
                if dst.len() < 2 {
                    return false;
                }
                let value = (rgba[0] as u16) * 257;
                dst[..2].copy_from_slice(&value.to_le_bytes());
            }
            Self::R8Unorm | Self::R8Snorm | Self::R8Sint | Self::R8Uint | Self::A8Unorm => {
                if dst.is_empty() {
                    return false;
                }
                dst[0] = if matches!(self, Self::A8Unorm) {
                    rgba[3]
                } else {
                    rgba[0]
                };
            }
            _ => return false,
        }
        true
    }

    pub fn is_presentable(self) -> bool {
        self.decode_rgba8(&[0; 16]).is_some()
    }
}

pub fn map_surface_format(raw: u32) -> vk::Format {
    SurfaceFormat::from_raw(raw)
        .and_then(SurfaceFormat::vk_format)
        .unwrap_or(vk::Format::R8G8B8A8_UNORM)
}

pub fn surface_bytes_per_pixel(raw: u32) -> usize {
    SurfaceFormat::from_raw(raw)
        .map(SurfaceFormat::bytes_per_pixel)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_have_hardware_sizes() {
        assert_eq!(surface_bytes_per_pixel(FMT_R32G32B32A32_FLOAT), 16);
        assert_eq!(surface_bytes_per_pixel(FMT_R16G16B16A16_FLOAT), 8);
        assert_eq!(surface_bytes_per_pixel(FMT_A8R8G8B8_UNORM), 4);
        assert_eq!(surface_bytes_per_pixel(FMT_R5G6B5_UNORM), 2);
        assert_eq!(surface_bytes_per_pixel(FMT_R8_UNORM), 1);
    }

    #[test]
    fn formats_share_vulkan_mapping() {
        assert_eq!(
            map_surface_format(FMT_A8R8G8B8_UNORM),
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            map_surface_format(FMT_A8B8G8R8_UNORM),
            vk::Format::A8B8G8R8_UNORM_PACK32
        );
        assert_eq!(
            map_surface_format(FMT_X8B8G8R8_UNORM),
            vk::Format::A8B8G8R8_UNORM_PACK32
        );
    }
}
