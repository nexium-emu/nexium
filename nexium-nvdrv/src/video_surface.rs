use std::fmt;

pub const VIC_CONFIG_SIZE: usize = 0x610;
pub const VIC_FORMAT_A8B8G8R8: u8 = 31;
pub const VIC_FORMAT_A8R8G8B8: u8 = 32;
pub const VIC_FORMAT_X8B8G8R8: u8 = 35;
pub const VIC_FORMAT_Y8_U8V8_420: u8 = 67;
pub const VIC_FORMAT_Y8_V8U8_420: u8 = 68;

const VIC_OUTPUT_CONFIG_OFFSET: usize = 0x10;
const VIC_OUTPUT_SURFACE_OFFSET: usize = 0x20;
const VIC_SLOT_ARRAY_OFFSET: usize = 0x90;
const VIC_SLOT_SIZE: usize = 0xb0;
const VIC_SLOT_COUNT: usize = 8;
const VIC_SLOT_SURFACE_OFFSET: usize = 0x40;
const MAX_BLOCK_HEIGHT_LOG2: u8 = 5;
const GOB_WIDTH_BYTES: usize = 64;
const GOB_HEIGHT: usize = 8;
const GOB_SIZE: usize = GOB_WIDTH_BYTES * GOB_HEIGHT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VicBlockKind {
    Pitch,
    BlockLinear,
    Other(u8),
}

impl VicBlockKind {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Pitch,
            1 => Self::BlockLinear,
            value => Self::Other(value),
        }
    }

    pub fn raw(self) -> u8 {
        match self {
            Self::Pitch => 0,
            Self::BlockLinear => 1,
            Self::Other(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaOrder {
    Uv,
    Vu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbaOrder {
    Rgba,
    Bgra,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VicColorMatrix {
    pub enabled: bool,
    pub coefficients: [[i32; 3]; 3],
    pub offsets: [i32; 3],
    pub shift: u8,
    pub clamp_min: u16,
    pub clamp_max: u16,
    pub alpha: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VicOutputSurface {
    pub pixel_format: u8,
    pub block_kind: VicBlockKind,
    pub block_height_log2: u8,
    pub surface: Dimensions,
    pub luma: Dimensions,
    pub chroma: Dimensions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VicInputSlot {
    pub index: usize,
    pub pixel_format: u8,
    pub block_kind: VicBlockKind,
    pub block_height_log2: u8,
    pub surface: Dimensions,
    pub luma: Dimensions,
    pub chroma: Dimensions,
    pub source_rect: Rect,
    pub destination_rect: Rect,
    pub color_matrix: VicColorMatrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VicConfigSummary {
    pub target_rect: Rect,
    pub output: VicOutputSurface,
    pub enabled_slots: Vec<VicInputSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputPlaneAddresses {
    pub luma: u64,
    pub chroma: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceLayout {
    Pitch {
        luma_pitch: usize,
        chroma_pitch: usize,
    },
    BlockLinear {
        block_height_log2: u8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nv12OutputConfig {
    pub addresses: OutputPlaneAddresses,
    pub visible: Dimensions,
    pub luma_storage: Dimensions,
    pub chroma_storage: Dimensions,
    pub layout: SurfaceLayout,
    pub chroma_order: ChromaOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbaOutputConfig {
    pub address: u64,
    pub visible: Dimensions,
    pub storage: Dimensions,
    pub layout: PlaneMemoryLayout,
    pub order: RgbaOrder,
    pub color_matrix: VicColorMatrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I420Frame {
    pub width: usize,
    pub height: usize,
    pub y_stride: usize,
    pub u_stride: usize,
    pub v_stride: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneMemoryLayout {
    Pitch {
        pitch: usize,
    },
    BlockLinear {
        width_bytes: usize,
        block_height_log2: u8,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneWrite {
    pub address: u64,
    pub logical_row_bytes: usize,
    pub logical_height: usize,
    pub layout: PlaneMemoryLayout,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nv12SurfaceWrites {
    pub luma: PlaneWrite,
    pub chroma: PlaneWrite,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoSurfaceError {
    ConfigTooShort {
        actual: usize,
        required: usize,
    },
    UnsupportedPixelFormat(u8),
    UnsupportedBlockKind(u8),
    InvalidBlockHeight(u8),
    InvalidClampRange {
        minimum: u16,
        maximum: u16,
    },
    ZeroDimension(&'static str),
    StrideTooSmall {
        plane: &'static str,
        stride: usize,
        required: usize,
    },
    PlaneTooShort {
        plane: &'static str,
        actual: usize,
        required: usize,
    },
    ArithmeticOverflow,
    AllocationFailed(usize),
}

impl fmt::Display for VideoSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigTooShort { actual, required } => {
                write!(
                    f,
                    "VIC config is {actual:#x} bytes, expected at least {required:#x}"
                )
            }
            Self::UnsupportedPixelFormat(format) => {
                write!(f, "unsupported VIC pixel format {format}")
            }
            Self::UnsupportedBlockKind(kind) => write!(f, "unsupported VIC block kind {kind}"),
            Self::InvalidBlockHeight(log2) => {
                write!(f, "invalid VIC block-height log2 {log2}")
            }
            Self::InvalidClampRange { minimum, maximum } => {
                write!(f, "invalid VIC clamp range {minimum}..={maximum}")
            }
            Self::ZeroDimension(name) => write!(f, "{name} has a zero dimension"),
            Self::StrideTooSmall {
                plane,
                stride,
                required,
            } => write!(
                f,
                "{plane} stride {stride} is smaller than the required {required} bytes"
            ),
            Self::PlaneTooShort {
                plane,
                actual,
                required,
            } => write!(
                f,
                "{plane} plane is {actual} bytes, expected at least {required}"
            ),
            Self::ArithmeticOverflow => write!(f, "video surface size overflow"),
            Self::AllocationFailed(size) => {
                write!(f, "failed to allocate a {size}-byte video surface")
            }
        }
    }
}

impl std::error::Error for VideoSurfaceError {}

impl VicOutputSurface {
    pub fn nv12_config(
        &self,
        addresses: OutputPlaneAddresses,
    ) -> Result<Nv12OutputConfig, VideoSurfaceError> {
        let chroma_order = match self.pixel_format {
            VIC_FORMAT_Y8_U8V8_420 => ChromaOrder::Vu,
            VIC_FORMAT_Y8_V8U8_420 => ChromaOrder::Uv,
            format => return Err(VideoSurfaceError::UnsupportedPixelFormat(format)),
        };
        let layout = match self.block_kind {
            VicBlockKind::Pitch => SurfaceLayout::Pitch {
                luma_pitch: align_up(self.luma.width, 16)?,
                chroma_pitch: align_up(
                    self.chroma
                        .width
                        .checked_mul(2)
                        .ok_or(VideoSurfaceError::ArithmeticOverflow)?,
                    16,
                )?,
            },
            VicBlockKind::BlockLinear => {
                validate_block_height(self.block_height_log2)?;
                SurfaceLayout::BlockLinear {
                    block_height_log2: self.block_height_log2,
                }
            }
            VicBlockKind::Other(kind) => {
                return Err(VideoSurfaceError::UnsupportedBlockKind(kind));
            }
        };
        Ok(Nv12OutputConfig {
            addresses,
            visible: self.surface,
            luma_storage: self.luma,
            chroma_storage: self.chroma,
            layout,
            chroma_order,
        })
    }

    pub fn rgba_config(
        &self,
        address: u64,
        color_matrix: VicColorMatrix,
    ) -> Result<RgbaOutputConfig, VideoSurfaceError> {
        let order = match self.pixel_format {
            VIC_FORMAT_A8B8G8R8 | VIC_FORMAT_X8B8G8R8 => RgbaOrder::Rgba,
            VIC_FORMAT_A8R8G8B8 => RgbaOrder::Bgra,
            format => return Err(VideoSurfaceError::UnsupportedPixelFormat(format)),
        };
        let width_bytes = self
            .luma
            .width
            .checked_mul(4)
            .ok_or(VideoSurfaceError::ArithmeticOverflow)?;
        let layout = match self.block_kind {
            VicBlockKind::Pitch => PlaneMemoryLayout::Pitch {
                pitch: align_up(width_bytes, 16)?,
            },
            VicBlockKind::BlockLinear => {
                validate_block_height(self.block_height_log2)?;
                PlaneMemoryLayout::BlockLinear {
                    width_bytes,
                    block_height_log2: self.block_height_log2,
                }
            }
            VicBlockKind::Other(kind) => {
                return Err(VideoSurfaceError::UnsupportedBlockKind(kind));
            }
        };
        Ok(RgbaOutputConfig {
            address,
            visible: self.surface,
            storage: self.luma,
            layout,
            order,
            color_matrix,
        })
    }
}

pub fn parse_vic_config(bytes: &[u8]) -> Result<VicConfigSummary, VideoSurfaceError> {
    if bytes.len() < VIC_CONFIG_SIZE {
        return Err(VideoSurfaceError::ConfigTooShort {
            actual: bytes.len(),
            required: VIC_CONFIG_SIZE,
        });
    }

    let target_x = read_u32(bytes, VIC_OUTPUT_CONFIG_OFFSET + 8);
    let target_y = read_u32(bytes, VIC_OUTPUT_CONFIG_OFFSET + 12);
    let target_rect = Rect {
        left: target_x & 0x3fff,
        right: (target_x >> 16) & 0x3fff,
        top: target_y & 0x3fff,
        bottom: (target_y >> 16) & 0x3fff,
    };

    let output_format = read_u32(bytes, VIC_OUTPUT_SURFACE_OFFSET);
    let output = VicOutputSurface {
        pixel_format: (output_format & 0x7f) as u8,
        block_kind: VicBlockKind::from_raw(((output_format >> 11) & 0xf) as u8),
        block_height_log2: ((output_format >> 15) & 0xf) as u8,
        surface: parse_dimensions(read_u32(bytes, VIC_OUTPUT_SURFACE_OFFSET + 4)),
        luma: parse_dimensions(read_u32(bytes, VIC_OUTPUT_SURFACE_OFFSET + 8)),
        chroma: parse_dimensions(read_u32(bytes, VIC_OUTPUT_SURFACE_OFFSET + 12)),
    };

    let mut enabled_slots = Vec::new();
    for index in 0..VIC_SLOT_COUNT {
        let base = VIC_SLOT_ARRAY_OFFSET + index * VIC_SLOT_SIZE;
        if read_u64(bytes, base) & 1 == 0 {
            continue;
        }
        let source_x = read_u64(bytes, base + 0x20);
        let source_y = read_u64(bytes, base + 0x28);
        let destination = read_u64(bytes, base + 0x30);
        let clamp_and_alpha = read_u64(bytes, base + 0x10);
        let surface_base = base + VIC_SLOT_SURFACE_OFFSET;
        let surface_format = read_u32(bytes, surface_base);
        let matrix_base = base + 0x60;
        let matrix0 = read_u64(bytes, matrix_base);
        let matrix1 = read_u64(bytes, matrix_base + 8);
        let matrix2 = read_u64(bytes, matrix_base + 16);
        let matrix3 = read_u64(bytes, matrix_base + 24);
        enabled_slots.push(VicInputSlot {
            index,
            pixel_format: (surface_format & 0x7f) as u8,
            block_kind: VicBlockKind::from_raw(((surface_format >> 11) & 0xf) as u8),
            block_height_log2: ((surface_format >> 15) & 0xf) as u8,
            surface: parse_dimensions(read_u32(bytes, surface_base + 4)),
            luma: parse_dimensions(read_u32(bytes, surface_base + 8)),
            chroma: parse_dimensions(read_u32(bytes, surface_base + 12)),
            source_rect: Rect {
                left: (source_x & 0x3fff_ffff) as u32,
                right: ((source_x >> 32) & 0x3fff_ffff) as u32,
                top: (source_y & 0x3fff_ffff) as u32,
                bottom: ((source_y >> 32) & 0x3fff_ffff) as u32,
            },
            destination_rect: Rect {
                left: (destination & 0x3fff) as u32,
                right: ((destination >> 16) & 0x3fff) as u32,
                top: ((destination >> 32) & 0x3fff) as u32,
                bottom: ((destination >> 48) & 0x3fff) as u32,
            },
            color_matrix: VicColorMatrix {
                enabled: matrix1 >> 63 != 0,
                coefficients: [
                    [signed_20(matrix0), signed_20(matrix1), signed_20(matrix2)],
                    [
                        signed_20(matrix0 >> 20),
                        signed_20(matrix1 >> 20),
                        signed_20(matrix2 >> 20),
                    ],
                    [
                        signed_20(matrix0 >> 40),
                        signed_20(matrix1 >> 40),
                        signed_20(matrix2 >> 40),
                    ],
                ],
                offsets: [
                    signed_20(matrix3),
                    signed_20(matrix3 >> 20),
                    signed_20(matrix3 >> 40),
                ],
                shift: ((matrix0 >> 60) & 0xf) as u8,
                clamp_min: (clamp_and_alpha & 0x3ff) as u16,
                clamp_max: ((clamp_and_alpha >> 10) & 0x3ff) as u16,
                alpha: ((clamp_and_alpha >> 32) & 0x3ff) as u16,
            },
        });
    }

    Ok(VicConfigSummary {
        target_rect,
        output,
        enabled_slots,
    })
}

pub fn write_i420_to_nv12(
    frame: I420Frame,
    config: Nv12OutputConfig,
) -> Result<Nv12SurfaceWrites, VideoSurfaceError> {
    validate_frame(&frame)?;
    validate_output(&config)?;

    let copy_width = frame
        .width
        .min(config.visible.width)
        .min(config.luma_storage.width);
    let copy_height = frame
        .height
        .min(config.visible.height)
        .min(config.luma_storage.height);
    let chroma_width = div_ceil(copy_width, 2)
        .min(div_ceil(frame.width, 2))
        .min(config.chroma_storage.width);
    let chroma_height = div_ceil(copy_height, 2)
        .min(div_ceil(frame.height, 2))
        .min(config.chroma_storage.height);
    let chroma_row_bytes = config
        .chroma_storage
        .width
        .checked_mul(2)
        .ok_or(VideoSurfaceError::ArithmeticOverflow)?;

    match config.layout {
        SurfaceLayout::Pitch {
            luma_pitch,
            chroma_pitch,
        } => {
            require_stride("output luma", luma_pitch, config.luma_storage.width)?;
            require_stride("output chroma", chroma_pitch, chroma_row_bytes)?;
            let mut luma = zeroed(checked_plane_size(luma_pitch, config.luma_storage.height)?)?;
            let mut chroma = zeroed(checked_plane_size(
                chroma_pitch,
                config.chroma_storage.height,
            )?)?;
            copy_luma(&frame, &mut luma, luma_pitch, copy_width, copy_height);
            interleave_chroma(
                &frame,
                &mut chroma,
                chroma_pitch,
                chroma_width,
                chroma_height,
                config.chroma_order,
            );
            Ok(Nv12SurfaceWrites {
                luma: PlaneWrite {
                    address: config.addresses.luma,
                    logical_row_bytes: config.luma_storage.width,
                    logical_height: config.luma_storage.height,
                    layout: PlaneMemoryLayout::Pitch { pitch: luma_pitch },
                    bytes: luma,
                },
                chroma: PlaneWrite {
                    address: config.addresses.chroma,
                    logical_row_bytes: chroma_row_bytes,
                    logical_height: config.chroma_storage.height,
                    layout: PlaneMemoryLayout::Pitch {
                        pitch: chroma_pitch,
                    },
                    bytes: chroma,
                },
            })
        }
        SurfaceLayout::BlockLinear { block_height_log2 } => {
            validate_block_height(block_height_log2)?;
            let luma_linear_pitch = config.luma_storage.width;
            let chroma_linear_pitch = chroma_row_bytes;
            let mut luma_linear = zeroed(checked_plane_size(
                luma_linear_pitch,
                config.luma_storage.height,
            )?)?;
            let mut chroma_linear = zeroed(checked_plane_size(
                chroma_linear_pitch,
                config.chroma_storage.height,
            )?)?;
            copy_luma(
                &frame,
                &mut luma_linear,
                luma_linear_pitch,
                copy_width,
                copy_height,
            );
            interleave_chroma(
                &frame,
                &mut chroma_linear,
                chroma_linear_pitch,
                chroma_width,
                chroma_height,
                config.chroma_order,
            );
            let luma = swizzle_block_linear(
                &luma_linear,
                luma_linear_pitch,
                config.luma_storage.width,
                config.luma_storage.height,
                block_height_log2,
            )?;
            let chroma = swizzle_block_linear(
                &chroma_linear,
                chroma_linear_pitch,
                chroma_row_bytes,
                config.chroma_storage.height,
                block_height_log2,
            )?;
            Ok(Nv12SurfaceWrites {
                luma: PlaneWrite {
                    address: config.addresses.luma,
                    logical_row_bytes: config.luma_storage.width,
                    logical_height: config.luma_storage.height,
                    layout: PlaneMemoryLayout::BlockLinear {
                        width_bytes: config.luma_storage.width,
                        block_height_log2,
                    },
                    bytes: luma,
                },
                chroma: PlaneWrite {
                    address: config.addresses.chroma,
                    logical_row_bytes: chroma_row_bytes,
                    logical_height: config.chroma_storage.height,
                    layout: PlaneMemoryLayout::BlockLinear {
                        width_bytes: chroma_row_bytes,
                        block_height_log2,
                    },
                    bytes: chroma,
                },
            })
        }
    }
}

pub fn write_i420_to_rgba(
    frame: I420Frame,
    config: RgbaOutputConfig,
) -> Result<PlaneWrite, VideoSurfaceError> {
    validate_frame(&frame)?;
    if config.visible.width == 0 || config.visible.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("visible output"));
    }
    if config.storage.width == 0 || config.storage.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("RGBA output"));
    }
    if config.color_matrix.enabled && config.color_matrix.clamp_min > config.color_matrix.clamp_max
    {
        return Err(VideoSurfaceError::InvalidClampRange {
            minimum: config.color_matrix.clamp_min,
            maximum: config.color_matrix.clamp_max,
        });
    }

    let width = frame
        .width
        .min(config.visible.width)
        .min(config.storage.width);
    let height = frame
        .height
        .min(config.visible.height)
        .min(config.storage.height);
    let row_bytes = config
        .storage
        .width
        .checked_mul(4)
        .ok_or(VideoSurfaceError::ArithmeticOverflow)?;

    let (linear_pitch, output_layout) = match config.layout {
        PlaneMemoryLayout::Pitch { pitch } => {
            require_stride("output RGBA", pitch, row_bytes)?;
            (pitch, PlaneMemoryLayout::Pitch { pitch })
        }
        PlaneMemoryLayout::BlockLinear {
            width_bytes,
            block_height_log2,
        } => {
            validate_block_height(block_height_log2)?;
            require_stride("output RGBA", width_bytes, row_bytes)?;
            (
                width_bytes,
                PlaneMemoryLayout::BlockLinear {
                    width_bytes,
                    block_height_log2,
                },
            )
        }
    };
    let mut linear = zeroed(checked_plane_size(linear_pitch, config.storage.height)?)?;
    for y in 0..height {
        let source_y = y * frame.y_stride;
        let source_u = (y / 2) * frame.u_stride;
        let source_v = (y / 2) * frame.v_stride;
        let destination = y * linear_pitch;
        for x in 0..width {
            let rgba = convert_pixel(
                frame.y[source_y + x],
                frame.u[source_u + x / 2],
                frame.v[source_v + x / 2],
                config.color_matrix,
            );
            let pixel = destination + x * 4;
            match config.order {
                RgbaOrder::Rgba => linear[pixel..pixel + 4].copy_from_slice(&rgba),
                RgbaOrder::Bgra => {
                    linear[pixel..pixel + 4].copy_from_slice(&[rgba[2], rgba[1], rgba[0], rgba[3]])
                }
            }
        }
    }

    let bytes = match output_layout {
        PlaneMemoryLayout::Pitch { .. } => linear,
        PlaneMemoryLayout::BlockLinear {
            width_bytes,
            block_height_log2,
        } => swizzle_block_linear(
            &linear,
            linear_pitch,
            width_bytes,
            config.storage.height,
            block_height_log2,
        )?,
    };
    Ok(PlaneWrite {
        address: config.address,
        logical_row_bytes: row_bytes,
        logical_height: config.storage.height,
        layout: output_layout,
        bytes,
    })
}

fn convert_pixel(y: u8, u: u8, v: u8, matrix: VicColorMatrix) -> [u8; 4] {
    if !matrix.enabled {
        return [y, u, v, (matrix.alpha.min(1023) >> 2) as u8];
    }
    let input = [i64::from(y) << 2, i64::from(u) << 2, i64::from(v) << 2];
    let mut output = [0i64; 3];
    for row in 0..3 {
        let value = input[0] * i64::from(matrix.coefficients[row][0])
            + input[1] * i64::from(matrix.coefficients[row][1])
            + input[2] * i64::from(matrix.coefficients[row][2]);
        output[row] = ((value >> matrix.shift) + i64::from(matrix.offsets[row])) >> 8;
    }
    let minimum = i64::from(matrix.clamp_min);
    let maximum = i64::from(matrix.clamp_max);
    let channel = |value: i64| value.clamp(minimum, maximum).clamp(0, 1023) as u16;
    [
        (channel(output[0]) >> 2) as u8,
        (channel(output[1]) >> 2) as u8,
        (channel(output[2]) >> 2) as u8,
        (channel(i64::from(matrix.alpha)) >> 2) as u8,
    ]
}

pub fn block_linear_plane_size(
    width_bytes: usize,
    height: usize,
    block_height_log2: u8,
) -> Option<usize> {
    if block_height_log2 > MAX_BLOCK_HEIGHT_LOG2 {
        return None;
    }
    let block_height = 1usize.checked_shl(block_height_log2.into())?;
    let rows_per_block = block_height.checked_mul(GOB_HEIGHT)?;
    let gobs_per_row = div_ceil_checked(width_bytes, GOB_WIDTH_BYTES)?;
    let block_rows = div_ceil_checked(height, rows_per_block)?;
    block_rows
        .checked_mul(gobs_per_row)?
        .checked_mul(block_height)?
        .checked_mul(GOB_SIZE)
}

pub fn block_linear_byte_offset(
    width_bytes: usize,
    block_height_log2: u8,
    x: usize,
    y: usize,
) -> Option<usize> {
    if block_height_log2 > MAX_BLOCK_HEIGHT_LOG2 || x >= width_bytes {
        return None;
    }
    let block_height = 1usize.checked_shl(block_height_log2.into())?;
    let rows_per_block = block_height.checked_mul(GOB_HEIGHT)?;
    let gobs_per_row = div_ceil_checked(width_bytes, GOB_WIDTH_BYTES)?;
    let block_row_stride = gobs_per_row
        .checked_mul(block_height)?
        .checked_mul(GOB_SIZE)?;
    let block_y = y / rows_per_block;
    let y_in_block = y % rows_per_block;
    let gob_row = y_in_block / GOB_HEIGHT;
    let y_in_gob = y_in_block % GOB_HEIGHT;
    let gob_column = x / GOB_WIDTH_BYTES;
    let x_in_gob = x % GOB_WIDTH_BYTES;
    let gob_offset = block_y
        .checked_mul(block_row_stride)?
        .checked_add(
            gob_column
                .checked_mul(block_height)?
                .checked_mul(GOB_SIZE)?,
        )?
        .checked_add(gob_row.checked_mul(GOB_SIZE)?)?;
    let in_gob = ((x_in_gob >> 5) & 1) * 256
        + ((y_in_gob >> 1) & 3) * 64
        + ((x_in_gob >> 4) & 1) * 32
        + (y_in_gob & 1) * 16
        + (x_in_gob & 15);
    gob_offset.checked_add(in_gob)
}

fn swizzle_block_linear(
    linear: &[u8],
    linear_pitch: usize,
    width_bytes: usize,
    height: usize,
    block_height_log2: u8,
) -> Result<Vec<u8>, VideoSurfaceError> {
    let size = block_linear_plane_size(width_bytes, height, block_height_log2)
        .ok_or(VideoSurfaceError::ArithmeticOverflow)?;
    let mut tiled = zeroed(size)?;
    for y in 0..height {
        let source_row = y
            .checked_mul(linear_pitch)
            .ok_or(VideoSurfaceError::ArithmeticOverflow)?;
        for x in 0..width_bytes {
            let source = source_row
                .checked_add(x)
                .ok_or(VideoSurfaceError::ArithmeticOverflow)?;
            let destination = block_linear_byte_offset(width_bytes, block_height_log2, x, y)
                .ok_or(VideoSurfaceError::ArithmeticOverflow)?;
            tiled[destination] = linear[source];
        }
    }
    Ok(tiled)
}

fn validate_frame(frame: &I420Frame) -> Result<(), VideoSurfaceError> {
    if frame.width == 0 || frame.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("I420 frame"));
    }
    let chroma_width = div_ceil(frame.width, 2);
    let chroma_height = div_ceil(frame.height, 2);
    require_stride("I420 Y", frame.y_stride, frame.width)?;
    require_stride("I420 U", frame.u_stride, chroma_width)?;
    require_stride("I420 V", frame.v_stride, chroma_width)?;
    require_plane(
        "I420 Y",
        frame.y.len(),
        checked_plane_size(frame.y_stride, frame.height)?,
    )?;
    require_plane(
        "I420 U",
        frame.u.len(),
        checked_plane_size(frame.u_stride, chroma_height)?,
    )?;
    require_plane(
        "I420 V",
        frame.v.len(),
        checked_plane_size(frame.v_stride, chroma_height)?,
    )?;
    Ok(())
}

fn validate_output(config: &Nv12OutputConfig) -> Result<(), VideoSurfaceError> {
    if config.visible.width == 0 || config.visible.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("visible output"));
    }
    if config.luma_storage.width == 0 || config.luma_storage.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("luma output"));
    }
    if config.chroma_storage.width == 0 || config.chroma_storage.height == 0 {
        return Err(VideoSurfaceError::ZeroDimension("chroma output"));
    }
    Ok(())
}

fn copy_luma(
    frame: &I420Frame,
    output: &mut [u8],
    output_pitch: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let source = y * frame.y_stride;
        let destination = y * output_pitch;
        output[destination..destination + width].copy_from_slice(&frame.y[source..source + width]);
    }
}

fn interleave_chroma(
    frame: &I420Frame,
    output: &mut [u8],
    output_pitch: usize,
    width: usize,
    height: usize,
    order: ChromaOrder,
) {
    for y in 0..height {
        let source_u = y * frame.u_stride;
        let source_v = y * frame.v_stride;
        let destination = y * output_pitch;
        for x in 0..width {
            let u = frame.u[source_u + x];
            let v = frame.v[source_v + x];
            let pair = destination + x * 2;
            match order {
                ChromaOrder::Uv => {
                    output[pair] = u;
                    output[pair + 1] = v;
                }
                ChromaOrder::Vu => {
                    output[pair] = v;
                    output[pair + 1] = u;
                }
            }
        }
    }
}

fn parse_dimensions(raw: u32) -> Dimensions {
    Dimensions {
        width: ((raw & 0x3fff) + 1) as usize,
        height: (((raw >> 14) & 0x3fff) + 1) as usize,
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn signed_20(value: u64) -> i32 {
    ((value as i32) << 12) >> 12
}

fn validate_block_height(block_height_log2: u8) -> Result<(), VideoSurfaceError> {
    if block_height_log2 > MAX_BLOCK_HEIGHT_LOG2 {
        Err(VideoSurfaceError::InvalidBlockHeight(block_height_log2))
    } else {
        Ok(())
    }
}

fn require_stride(
    plane: &'static str,
    stride: usize,
    required: usize,
) -> Result<(), VideoSurfaceError> {
    if stride < required {
        Err(VideoSurfaceError::StrideTooSmall {
            plane,
            stride,
            required,
        })
    } else {
        Ok(())
    }
}

fn require_plane(
    plane: &'static str,
    actual: usize,
    required: usize,
) -> Result<(), VideoSurfaceError> {
    if actual < required {
        Err(VideoSurfaceError::PlaneTooShort {
            plane,
            actual,
            required,
        })
    } else {
        Ok(())
    }
}

fn checked_plane_size(stride: usize, height: usize) -> Result<usize, VideoSurfaceError> {
    stride
        .checked_mul(height)
        .ok_or(VideoSurfaceError::ArithmeticOverflow)
}

fn zeroed(size: usize) -> Result<Vec<u8>, VideoSurfaceError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| VideoSurfaceError::AllocationFailed(size))?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, VideoSurfaceError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(VideoSurfaceError::ArithmeticOverflow)
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value / divisor + usize::from(value % divisor != 0)
}

fn div_ceil_checked(value: usize, divisor: usize) -> Option<usize> {
    value
        .checked_add(divisor.checked_sub(1)?)
        .map(|value| value / divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dimensions(width: usize, height: usize) -> Dimensions {
        Dimensions { width, height }
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn dimension_word(width: u32, height: u32) -> u32 {
        (width - 1) | ((height - 1) << 14)
    }

    fn frame_4x2() -> I420Frame {
        I420Frame {
            width: 4,
            height: 2,
            y_stride: 4,
            u_stride: 2,
            v_stride: 2,
            y: vec![1, 2, 3, 4, 5, 6, 7, 8],
            u: vec![10, 20],
            v: vec![30, 40],
        }
    }

    fn pitch_config(
        visible: Dimensions,
        luma_storage: Dimensions,
        chroma_storage: Dimensions,
        luma_pitch: usize,
        chroma_pitch: usize,
    ) -> Nv12OutputConfig {
        Nv12OutputConfig {
            addresses: OutputPlaneAddresses {
                luma: 0x1200,
                chroma: 0x3400,
            },
            visible,
            luma_storage,
            chroma_storage,
            layout: SurfaceLayout::Pitch {
                luma_pitch,
                chroma_pitch,
            },
            chroma_order: ChromaOrder::Uv,
        }
    }

    fn identity_color_matrix() -> VicColorMatrix {
        VicColorMatrix {
            enabled: true,
            coefficients: [[256, 0, 0], [0, 256, 0], [0, 0, 256]],
            offsets: [0; 3],
            shift: 0,
            clamp_min: 0,
            clamp_max: 1023,
            alpha: 1023,
        }
    }

    fn rgba_config(
        width: usize,
        height: usize,
        storage_width: usize,
        storage_height: usize,
        layout: PlaneMemoryLayout,
        order: RgbaOrder,
    ) -> RgbaOutputConfig {
        RgbaOutputConfig {
            address: 0x5600,
            visible: dimensions(width, height),
            storage: dimensions(storage_width, storage_height),
            layout,
            order,
            color_matrix: identity_color_matrix(),
        }
    }

    fn unswizzle(
        tiled: &[u8],
        width_bytes: usize,
        height: usize,
        block_height_log2: u8,
    ) -> Vec<u8> {
        let mut linear = vec![0; width_bytes * height];
        for y in 0..height {
            for x in 0..width_bytes {
                let offset =
                    block_linear_byte_offset(width_bytes, block_height_log2, x, y).unwrap();
                linear[y * width_bytes + x] = tiled[offset];
            }
        }
        linear
    }

    #[test]
    fn parses_output_and_enabled_slot_bitfields() {
        let mut bytes = vec![0u8; VIC_CONFIG_SIZE];
        put_u32(&mut bytes, 0x18, 13 | (1266 << 16));
        put_u32(&mut bytes, 0x1c, 7 | (712 << 16));
        put_u32(
            &mut bytes,
            VIC_OUTPUT_SURFACE_OFFSET,
            VIC_FORMAT_Y8_V8U8_420 as u32 | (1 << 11) | (4 << 15),
        );
        put_u32(
            &mut bytes,
            VIC_OUTPUT_SURFACE_OFFSET + 4,
            dimension_word(1280, 720),
        );
        put_u32(
            &mut bytes,
            VIC_OUTPUT_SURFACE_OFFSET + 8,
            dimension_word(1280, 768),
        );
        put_u32(
            &mut bytes,
            VIC_OUTPUT_SURFACE_OFFSET + 12,
            dimension_word(640, 384),
        );

        let slot = VIC_SLOT_ARRAY_OFFSET + 3 * VIC_SLOT_SIZE;
        put_u64(&mut bytes, slot, 1);
        put_u64(&mut bytes, slot + 0x20, 5 | (1274u64 << 32));
        put_u64(&mut bytes, slot + 0x28, 9 | (710u64 << 32));
        put_u64(
            &mut bytes,
            slot + 0x30,
            11 | (1268u64 << 16) | (13u64 << 32) | (706u64 << 48),
        );
        let slot_surface = slot + VIC_SLOT_SURFACE_OFFSET;
        put_u32(
            &mut bytes,
            slot_surface,
            VIC_FORMAT_Y8_U8V8_420 as u32 | (1 << 11) | (3 << 15),
        );
        put_u32(&mut bytes, slot_surface + 4, dimension_word(1280, 720));
        put_u32(&mut bytes, slot_surface + 8, dimension_word(1280, 768));
        put_u32(&mut bytes, slot_surface + 12, dimension_word(640, 384));

        let parsed = parse_vic_config(&bytes).unwrap();
        assert_eq!(
            parsed.target_rect,
            Rect {
                left: 13,
                right: 1266,
                top: 7,
                bottom: 712,
            }
        );
        assert_eq!(parsed.output.pixel_format, VIC_FORMAT_Y8_V8U8_420);
        assert_eq!(parsed.output.block_kind, VicBlockKind::BlockLinear);
        assert_eq!(parsed.output.block_height_log2, 4);
        assert_eq!(parsed.output.surface, dimensions(1280, 720));
        assert_eq!(parsed.output.luma, dimensions(1280, 768));
        assert_eq!(parsed.output.chroma, dimensions(640, 384));
        assert_eq!(parsed.enabled_slots.len(), 1);
        let parsed_slot = &parsed.enabled_slots[0];
        assert_eq!(parsed_slot.index, 3);
        assert_eq!(parsed_slot.pixel_format, VIC_FORMAT_Y8_U8V8_420);
        assert_eq!(parsed_slot.block_kind, VicBlockKind::BlockLinear);
        assert_eq!(parsed_slot.block_height_log2, 3);
        assert_eq!(parsed_slot.surface, dimensions(1280, 720));
        assert_eq!(
            parsed_slot.source_rect,
            Rect {
                left: 5,
                right: 1274,
                top: 9,
                bottom: 710,
            }
        );
        assert_eq!(
            parsed_slot.destination_rect,
            Rect {
                left: 11,
                right: 1268,
                top: 13,
                bottom: 706,
            }
        );
    }

    #[test]
    fn rejects_short_config() {
        assert_eq!(
            parse_vic_config(&vec![0; VIC_CONFIG_SIZE - 1]),
            Err(VideoSurfaceError::ConfigTooShort {
                actual: VIC_CONFIG_SIZE - 1,
                required: VIC_CONFIG_SIZE,
            })
        );
    }

    #[test]
    fn interleaves_uv_and_preserves_addresses() {
        let output = write_i420_to_nv12(
            frame_4x2(),
            pitch_config(dimensions(4, 2), dimensions(4, 2), dimensions(2, 1), 4, 4),
        )
        .unwrap();
        assert_eq!(output.luma.address, 0x1200);
        assert_eq!(output.chroma.address, 0x3400);
        assert_eq!(output.luma.bytes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(output.chroma.bytes, vec![10, 30, 20, 40]);
    }

    #[test]
    fn pitch_layout_leaves_padding_zeroed() {
        let output = write_i420_to_nv12(
            frame_4x2(),
            pitch_config(dimensions(4, 2), dimensions(4, 3), dimensions(2, 2), 7, 6),
        )
        .unwrap();
        assert_eq!(
            output.luma.bytes,
            vec![1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            output.chroma.bytes,
            vec![10, 30, 20, 40, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn block_linear_offsets_match_tegra_gob_layout() {
        assert_eq!(block_linear_byte_offset(65, 1, 0, 0), Some(0));
        assert_eq!(block_linear_byte_offset(65, 1, 15, 0), Some(15));
        assert_eq!(block_linear_byte_offset(65, 1, 16, 0), Some(32));
        assert_eq!(block_linear_byte_offset(65, 1, 32, 0), Some(256));
        assert_eq!(block_linear_byte_offset(65, 1, 0, 1), Some(16));
        assert_eq!(block_linear_byte_offset(65, 1, 0, 2), Some(64));
        assert_eq!(block_linear_byte_offset(65, 1, 0, 8), Some(512));
        assert_eq!(block_linear_byte_offset(65, 1, 64, 0), Some(1024));
        assert_eq!(block_linear_byte_offset(65, 1, 0, 16), Some(2048));
    }

    #[test]
    fn block_linear_planes_roundtrip_to_nv12() {
        let width = 70;
        let height = 18;
        let chroma_width = div_ceil(width, 2);
        let chroma_height = div_ceil(height, 2);
        let y = (0..width * height)
            .map(|value| value.wrapping_mul(17) as u8)
            .collect::<Vec<_>>();
        let u = (0..chroma_width * chroma_height)
            .map(|value| value.wrapping_mul(5) as u8)
            .collect::<Vec<_>>();
        let v = (0..chroma_width * chroma_height)
            .map(|value| 255u8.wrapping_sub(value.wrapping_mul(3) as u8))
            .collect::<Vec<_>>();
        let frame = I420Frame {
            width,
            height,
            y_stride: width,
            u_stride: chroma_width,
            v_stride: chroma_width,
            y: y.clone(),
            u: u.clone(),
            v: v.clone(),
        };
        let output = write_i420_to_nv12(
            frame,
            Nv12OutputConfig {
                addresses: OutputPlaneAddresses { luma: 0, chroma: 0 },
                visible: dimensions(width, height),
                luma_storage: dimensions(width, height),
                chroma_storage: dimensions(chroma_width, chroma_height),
                layout: SurfaceLayout::BlockLinear {
                    block_height_log2: 1,
                },
                chroma_order: ChromaOrder::Uv,
            },
        )
        .unwrap();

        assert_eq!(
            output.luma.bytes.len(),
            block_linear_plane_size(width, height, 1).unwrap()
        );
        assert_eq!(unswizzle(&output.luma.bytes, width, height, 1), y);
        let chroma = unswizzle(&output.chroma.bytes, chroma_width * 2, chroma_height, 1);
        for row in 0..chroma_height {
            for column in 0..chroma_width {
                let source = row * chroma_width + column;
                let destination = row * chroma_width * 2 + column * 2;
                assert_eq!(chroma[destination], u[source]);
                assert_eq!(chroma[destination + 1], v[source]);
            }
        }
    }

    #[test]
    fn vic_output_builds_default_pitch_and_chroma_order() {
        let output = VicOutputSurface {
            pixel_format: VIC_FORMAT_Y8_V8U8_420,
            block_kind: VicBlockKind::Pitch,
            block_height_log2: 0,
            surface: dimensions(1280, 720),
            luma: dimensions(1279, 720),
            chroma: dimensions(639, 360),
        }
        .nv12_config(OutputPlaneAddresses { luma: 1, chroma: 2 })
        .unwrap();
        assert_eq!(output.chroma_order, ChromaOrder::Uv);
        assert_eq!(
            output.layout,
            SurfaceLayout::Pitch {
                luma_pitch: 1280,
                chroma_pitch: 1280,
            }
        );

        let output = VicOutputSurface {
            pixel_format: VIC_FORMAT_Y8_U8V8_420,
            block_kind: VicBlockKind::Pitch,
            block_height_log2: 0,
            surface: dimensions(1280, 720),
            luma: dimensions(1279, 720),
            chroma: dimensions(639, 360),
        }
        .nv12_config(OutputPlaneAddresses { luma: 1, chroma: 2 })
        .unwrap();
        assert_eq!(output.chroma_order, ChromaOrder::Vu);
    }

    #[test]
    fn packed_output_applies_matrix_and_channel_order() {
        let frame = I420Frame {
            width: 2,
            height: 2,
            y_stride: 2,
            u_stride: 1,
            v_stride: 1,
            y: vec![100, 120, 140, 160],
            u: vec![50],
            v: vec![200],
        };
        let rgba = write_i420_to_rgba(
            frame.clone(),
            rgba_config(
                2,
                2,
                2,
                2,
                PlaneMemoryLayout::Pitch { pitch: 8 },
                RgbaOrder::Rgba,
            ),
        )
        .unwrap();
        assert_eq!(rgba.address, 0x5600);
        assert_eq!(rgba.bytes[0..8], [100, 50, 200, 255, 120, 50, 200, 255]);

        let bgra = write_i420_to_rgba(
            frame,
            rgba_config(
                2,
                2,
                2,
                2,
                PlaneMemoryLayout::Pitch { pitch: 8 },
                RgbaOrder::Bgra,
            ),
        )
        .unwrap();
        assert_eq!(bgra.bytes[0..8], [200, 50, 100, 255, 200, 50, 120, 255]);
    }

    #[test]
    fn packed_pitch_uses_byte_alignment_and_zeroes_storage_padding() {
        let width = 5;
        let frame = I420Frame {
            width,
            height: 2,
            y_stride: width,
            u_stride: 3,
            v_stride: 3,
            y: vec![64; width * 2],
            u: vec![96; 3],
            v: vec![192; 3],
        };
        let surface = VicOutputSurface {
            pixel_format: VIC_FORMAT_A8B8G8R8,
            block_kind: VicBlockKind::Pitch,
            block_height_log2: 0,
            surface: dimensions(width, 2),
            luma: dimensions(width, 3),
            chroma: dimensions(1, 1),
        };
        let config = surface
            .rgba_config(0x5600, identity_color_matrix())
            .unwrap();
        assert_eq!(config.layout, PlaneMemoryLayout::Pitch { pitch: 32 });
        let output = write_i420_to_rgba(frame, config).unwrap();
        assert_eq!(output.bytes.len(), 96);
        assert_eq!(&output.bytes[20..32], &[0; 12]);
        assert_eq!(&output.bytes[52..], &[0; 44]);
    }

    #[test]
    fn packed_block_linear_roundtrips_across_gobs() {
        let width = 17;
        let height = 9;
        let frame = I420Frame {
            width,
            height,
            y_stride: width,
            u_stride: 9,
            v_stride: 9,
            y: (0..width * height).map(|value| value as u8).collect(),
            u: vec![40; 9 * 5],
            v: vec![180; 9 * 5],
        };
        let output = write_i420_to_rgba(
            frame,
            rgba_config(
                width,
                height,
                width,
                height,
                PlaneMemoryLayout::BlockLinear {
                    width_bytes: width * 4,
                    block_height_log2: 1,
                },
                RgbaOrder::Rgba,
            ),
        )
        .unwrap();
        assert_eq!(output.bytes.len(), 2048);
        let linear = unswizzle(&output.bytes, width * 4, height, 1);
        assert_eq!(&linear[0..8], &[0, 40, 180, 255, 1, 40, 180, 255]);
        let last = (width * height - 1) * 4;
        assert_eq!(&linear[last..last + 4], &[152, 40, 180, 255]);
    }

    #[test]
    fn parses_signed_slot_color_matrix() {
        let mut bytes = vec![0u8; VIC_CONFIG_SIZE];
        let slot = VIC_SLOT_ARRAY_OFFSET;
        put_u64(&mut bytes, slot, 1);
        put_u64(&mut bytes, slot + 0x10, 7 | (900 << 10) | (777u64 << 32));
        let encode = |value: i32| u64::from((value as u32) & 0x000f_ffff);
        put_u64(
            &mut bytes,
            slot + 0x60,
            encode(256) | (encode(-3) << 20) | (encode(4) << 40) | (2 << 60),
        );
        put_u64(
            &mut bytes,
            slot + 0x68,
            encode(-5) | (encode(6) << 20) | (encode(-7) << 40) | (1 << 63),
        );
        put_u64(
            &mut bytes,
            slot + 0x70,
            encode(8) | (encode(-9) << 20) | (encode(10) << 40),
        );
        put_u64(
            &mut bytes,
            slot + 0x78,
            encode(-11) | (encode(12) << 20) | (encode(-13) << 40),
        );

        let matrix = parse_vic_config(&bytes).unwrap().enabled_slots[0].color_matrix;
        assert!(matrix.enabled);
        assert_eq!(
            matrix.coefficients,
            [[256, -5, 8], [-3, 6, -9], [4, -7, 10]]
        );
        assert_eq!(matrix.offsets, [-11, 12, -13]);
        assert_eq!(matrix.shift, 2);
        assert_eq!(
            (matrix.clamp_min, matrix.clamp_max, matrix.alpha),
            (7, 900, 777)
        );
    }
}
