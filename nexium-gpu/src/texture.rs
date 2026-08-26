#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TicFormat {
    R32G32B32A32,
    R32G32,
    R16G16B16A16,
    A8B8G8R8,
    A2B10G10R10,
    R8G8B8A8,
    R5G6B5,
    A1R5G5B5,
    A4R4G4B4,
    A5B5G5R1,
    A1B5G5R5,
    B5G6R5,
    A4B4G4R4,
    R8,
    R8G8,
    R16,
    R16G16,
    R32,
    G24R8,
    Z32,
    Z24S8,
    X8Z24,
    S8Z24,
    B10G11R11,
    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6S,
    BC6U,
    BC7,
    Etc2Rgb,
    Etc2RgbA1,
    Etc2Rgba,
    Astc(u8, u8),
    Unknown(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero,
    R,
    G,
    B,
    A,
    One,
    Unknown(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentType {
    Snorm,
    Unorm,
    Sint,
    Uint,
    SnormForceFp16,
    UnormForceFp16,
    Float,
    Unknown(u32),
}

impl ComponentType {
    pub fn from_raw(raw: u32) -> Self {
        match raw & 0x7 {
            1 => ComponentType::Snorm,
            2 => ComponentType::Unorm,
            3 => ComponentType::Sint,
            4 => ComponentType::Uint,
            5 => ComponentType::SnormForceFp16,
            6 => ComponentType::UnormForceFp16,
            7 => ComponentType::Float,
            other => ComponentType::Unknown(other),
        }
    }
}

impl SwizzleSource {
    pub fn from_raw(raw: u32) -> Self {
        match raw & 0x7 {
            0 => SwizzleSource::Zero,
            2 => SwizzleSource::R,
            3 => SwizzleSource::G,
            4 => SwizzleSource::B,
            5 => SwizzleSource::A,
            6 | 7 => SwizzleSource::One,
            other => SwizzleSource::Unknown(other),
        }
    }
}

impl TicFormat {
    pub fn from_raw(format_word: u32) -> Self {
        match format_word & 0x7F {
            0x01 => TicFormat::R32G32B32A32,
            0x04 => TicFormat::R32G32,
            0x03 => TicFormat::R16G16B16A16,
            0x06 => TicFormat::Etc2Rgb,
            0x08 => TicFormat::A8B8G8R8,
            0x09 => TicFormat::A2B10G10R10,
            0x0A => TicFormat::Etc2RgbA1,
            0x0B => TicFormat::Etc2Rgba,
            0x0C => TicFormat::R16G16,
            0x0F => TicFormat::R32,
            0x0E => TicFormat::G24R8,
            0x13 => TicFormat::A5B5G5R1,
            0x14 => TicFormat::A1B5G5R5,
            0x15 => TicFormat::B5G6R5,
            0x12 => TicFormat::A4B4G4R4,
            0x1B => TicFormat::R16,
            0x18 => TicFormat::R8G8,
            0x1C | 0x1D => TicFormat::R8,
            0x21 => TicFormat::B10G11R11,
            0x24 => TicFormat::BC1,
            0x25 => TicFormat::BC2,
            0x26 => TicFormat::BC3,
            0x27 => TicFormat::BC4,
            0x28 => TicFormat::BC5,
            0x10 => TicFormat::BC6S,
            0x11 => TicFormat::BC6U,
            0x17 => TicFormat::BC7,
            0x29 => TicFormat::Z24S8,
            0x2A => TicFormat::X8Z24,
            0x2B => TicFormat::S8Z24,
            0x2F => TicFormat::Z32,
            0x2D => TicFormat::Unknown(0x2D),
            0x40 => TicFormat::Astc(4, 4),
            0x50 => TicFormat::Astc(5, 4),
            0x41 => TicFormat::Astc(5, 5),
            0x51 => TicFormat::Astc(6, 5),
            0x42 => TicFormat::Astc(6, 6),
            0x55 => TicFormat::Astc(8, 5),
            0x52 => TicFormat::Astc(8, 6),
            0x44 => TicFormat::Astc(8, 8),
            0x56 => TicFormat::Astc(10, 5),
            0x57 => TicFormat::Astc(10, 6),
            0x53 => TicFormat::Astc(10, 8),
            0x45 => TicFormat::Astc(10, 10),
            0x54 => TicFormat::Astc(12, 10),
            0x46 => TicFormat::Astc(12, 12),
            other => TicFormat::Unknown(other),
        }
    }

    pub fn src_bpp(&self) -> usize {
        match self {
            TicFormat::R32G32B32A32 => 16,
            TicFormat::R32G32 | TicFormat::R16G16B16A16 => 8,
            TicFormat::A8B8G8R8
            | TicFormat::A2B10G10R10
            | TicFormat::R8G8B8A8
            | TicFormat::R32
            | TicFormat::G24R8
            | TicFormat::Z32
            | TicFormat::Z24S8
            | TicFormat::X8Z24
            | TicFormat::S8Z24
            | TicFormat::B10G11R11 => 4,
            TicFormat::R5G6B5
            | TicFormat::A1R5G5B5
            | TicFormat::A4R4G4B4
            | TicFormat::A5B5G5R1
            | TicFormat::A1B5G5R5
            | TicFormat::B5G6R5
            | TicFormat::A4B4G4R4 => 2,
            TicFormat::R16 | TicFormat::R8G8 => 2,
            TicFormat::R16G16 => 4,
            TicFormat::R8 => 1,
            TicFormat::BC1 | TicFormat::BC4 | TicFormat::Etc2Rgb | TicFormat::Etc2RgbA1 => 8,
            TicFormat::BC2
            | TicFormat::BC3
            | TicFormat::BC5
            | TicFormat::BC6S
            | TicFormat::BC6U
            | TicFormat::BC7
            | TicFormat::Etc2Rgba => 16,
            TicFormat::Astc(_, _) => 16,
            TicFormat::Unknown(_) => 4,
        }
    }

    pub fn storage_extent(&self, width: u32, height: u32) -> (u32, u32, usize) {
        let (block_width, block_height) = self.block_extent();
        (
            (width + block_width - 1) / block_width,
            (height + block_height - 1) / block_height,
            self.src_bpp(),
        )
    }

    pub fn block_extent(&self) -> (u32, u32) {
        match self {
            TicFormat::BC1
            | TicFormat::BC2
            | TicFormat::BC3
            | TicFormat::BC4
            | TicFormat::BC5
            | TicFormat::BC6S
            | TicFormat::BC6U
            | TicFormat::BC7
            | TicFormat::Etc2Rgb
            | TicFormat::Etc2RgbA1
            | TicFormat::Etc2Rgba => (4, 4),
            TicFormat::Astc(bw, bh) => (*bw as u32, *bh as u32),
            _ => (1, 1),
        }
    }

    pub fn linear_size(&self, width: u32, height: u32) -> usize {
        let (storage_width, storage_height, bpp) = self.storage_extent(width, height);
        storage_width as usize * storage_height as usize * bpp
    }

    pub fn block_linear_size(&self, width: u32, height: u32, block_height_log2: u32) -> usize {
        let (storage_width, storage_height, bpp) = self.storage_extent(width, height);
        block_linear_byte_size(storage_width, storage_height, bpp, block_height_log2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleGrid {
    pub width: u32,
    pub height: u32,
}

impl SampleGrid {
    pub const SINGLE: Self = Self {
        width: 1,
        height: 1,
    };

    pub fn count(self) -> u32 {
        self.width.saturating_mul(self.height)
    }
}

pub fn maxwell_sample_grid(msaa_mode: u32) -> Option<SampleGrid> {
    let (width, height) = match msaa_mode {
        0 => (1, 1),
        1 | 5 => (2, 1),
        2 | 8 | 9 => (2, 2),
        3 | 4 | 10 | 11 => (4, 2),
        6 => (4, 4),
        _ => return None,
    };
    Some(SampleGrid { width, height })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TicEntry {
    pub format: TicFormat,
    pub component_types: [ComponentType; 4],
    pub swizzle: [SwizzleSource; 4],
    pub gpu_va: u64,
    pub width: u32,
    pub height: u32,
    pub block_width_log2: u32,
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
    pub tile_width_spacing: u32,
    pub pitch_bytes: u32,
    pub is_block_linear: bool,
    pub texture_type: u32,
    pub depth: u32,
    pub base_layer: u32,
    pub normalized_coords: bool,
    pub is_srgb: bool,
    pub is_sparse: bool,
    pub msaa_mode: u32,
    pub max_mip_level: u32,
    pub res_min_mip_level: u32,
    pub res_max_mip_level: u32,
}

impl TicEntry {
    pub fn parse(raw: &[u8]) -> Option<TicEntry> {
        if raw.len() < 32 {
            return None;
        }
        let w0 = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let w1 = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let w2 = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
        let w3 = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
        let w4 = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]);

        let format = TicFormat::from_raw(w0);
        let component_types = [
            ComponentType::from_raw((w0 >> 7) & 0x7),
            ComponentType::from_raw((w0 >> 10) & 0x7),
            ComponentType::from_raw((w0 >> 13) & 0x7),
            ComponentType::from_raw((w0 >> 16) & 0x7),
        ];
        let swizzle = [
            SwizzleSource::from_raw((w0 >> 19) & 0x7),
            SwizzleSource::from_raw((w0 >> 22) & 0x7),
            SwizzleSource::from_raw((w0 >> 25) & 0x7),
            SwizzleSource::from_raw((w0 >> 28) & 0x7),
        ];

        let addr_lo = w1 as u64;
        let addr_hi = (w2 & 0xFFFF) as u64;
        let gpu_va = (addr_hi << 32) | addr_lo;

        let header_version = (w2 >> 21) & 0x7;
        let is_buffer_header = header_version == 0;
        let is_block_linear = matches!(header_version, 3 | 4);
        let pitch_bytes = if matches!(header_version, 1 | 2) {
            (w3 & 0xFFFF) << 5
        } else {
            0
        };

        let block_width_log2 = if is_block_linear { w3 & 0x7 } else { 0 };
        let block_height_log2 = if is_block_linear { (w3 >> 3) & 0x7 } else { 0 };
        let block_depth_log2 = if is_block_linear { (w3 >> 6) & 0x7 } else { 0 };
        let tile_width_spacing = if is_block_linear { (w3 >> 10) & 0x7 } else { 0 };

        let width = if is_buffer_header {
            ((w3 & 0xFFFF) << 16 | (w4 & 0xFFFF)).checked_add(1)?
        } else {
            (w4 & 0xFFFF) + 1
        };
        let layer_base_0_2 = (w4 >> 16) & 0x7;
        let layer_base_3_7 = (w2 >> 16) & 0x1F;
        let layer_base_8_10 = (w2 >> 29) & 0x7;
        let base_layer = layer_base_0_2 | (layer_base_3_7 << 3) | (layer_base_8_10 << 8);
        let texture_type = if is_buffer_header {
            6
        } else {
            (w4 >> 23) & 0xF
        };
        let is_srgb = (w4 >> 22) & 1 != 0;
        let w5 = u32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]);
        let height = (w5 & 0xFFFF) + 1;
        let depth = ((w5 >> 16) & 0x3FFF) + 1;
        let is_sparse = (w5 >> 30) & 1 != 0;
        let normalized_coords = (w5 >> 31) & 1 != 0;
        let w7 = u32::from_le_bytes([raw[28], raw[29], raw[30], raw[31]]);
        let max_mip_level = (w3 >> 28) & 0xF;
        let res_min_mip_level = w7 & 0xF;
        let res_max_mip_level = (w7 >> 4) & 0xF;
        let msaa_mode = (w7 >> 8) & 0xF;

        if gpu_va == 0
            || width == 0
            || height == 0
            || (!is_buffer_header && width > 16384)
            || (!is_buffer_header && height > 16384)
        {
            return None;
        }

        Some(TicEntry {
            format,
            component_types,
            swizzle,
            gpu_va,
            width,
            height,
            block_width_log2,
            block_height_log2,
            block_depth_log2,
            tile_width_spacing,
            pitch_bytes,
            is_block_linear,
            texture_type,
            depth,
            base_layer,
            normalized_coords,
            is_srgb,
            is_sparse,
            msaa_mode,
            max_mip_level,
            res_min_mip_level,
            res_max_mip_level,
        })
    }

    pub fn mip_levels(&self) -> u32 {
        if self.is_buffer() {
            1
        } else {
            self.max_mip_level.saturating_add(1)
        }
    }

    pub fn is_buffer(&self) -> bool {
        self.texture_type == 6
    }

    pub fn view_base_mip(&self) -> u32 {
        self.res_min_mip_level.min(self.max_mip_level)
    }

    pub fn view_mip_levels(&self) -> u32 {
        let base = self.view_base_mip();
        self.res_max_mip_level
            .min(self.max_mip_level)
            .saturating_sub(base)
            .saturating_add(1)
    }

    pub fn sample_grid(&self) -> Option<SampleGrid> {
        if self.is_buffer() {
            Some(SampleGrid::SINGLE)
        } else {
            maxwell_sample_grid(self.msaa_mode)
        }
    }

    pub fn sample_count(&self) -> Option<u32> {
        self.sample_grid().map(SampleGrid::count)
    }

    pub fn logical_mip_extent(&self, level: u32) -> (u32, u32) {
        (
            self.width.checked_shr(level).unwrap_or(0).max(1),
            self.height.checked_shr(level).unwrap_or(0).max(1),
        )
    }

    pub fn physical_mip_extent(&self, level: u32) -> Option<(u32, u32)> {
        let (width, height) = self.logical_mip_extent(level);
        let samples = self.sample_grid()?;
        Some((
            width.checked_mul(samples.width)?,
            height.checked_mul(samples.height)?,
        ))
    }

    pub fn physical_storage_extent(&self, level: u32) -> Option<(u32, u32, usize)> {
        if self.is_sparse {
            return None;
        }
        let (width, height) = self.physical_mip_extent(level)?;
        Some(self.format.storage_extent(width, height))
    }

    pub fn physical_linear_size(&self, level: u32) -> Option<usize> {
        let (storage_width, storage_height, bpp) = self.physical_storage_extent(level)?;
        (storage_width as usize)
            .checked_mul(storage_height as usize)?
            .checked_mul(bpp)
    }

    pub fn pitch_linear_layer_size(&self) -> Option<usize> {
        if self.pitch_bytes == 0 {
            return None;
        }
        let (storage_width, storage_height, bpp) = self.physical_storage_extent(0)?;
        let row_size = (storage_width as usize).checked_mul(bpp)?;
        let pitch = self.pitch_bytes as usize;
        if pitch < row_size {
            return None;
        }
        pitch.checked_mul(storage_height as usize)
    }

    pub fn pitch_linear_size(&self, layers: u32) -> Option<usize> {
        self.pitch_linear_layer_size()?
            .checked_mul(layers.max(1) as usize)
    }

    pub fn layer_stride_bytes(&self) -> Option<usize> {
        if self.is_buffer() || self.texture_type == 2 {
            return None;
        }
        self.pitch_linear_layer_size()
            .or_else(|| block_linear_mip_layout(self).map(|layout| layout.layer_stride))
            .or_else(|| self.physical_linear_size(0))
    }

    pub fn backing_gpu_va(&self) -> Option<u64> {
        if self.is_buffer() || self.texture_type == 2 || self.base_layer == 0 {
            return Some(self.gpu_va);
        }
        let layer_offset = self
            .layer_stride_bytes()?
            .checked_mul(self.base_layer as usize)?;
        self.gpu_va.checked_sub(layer_offset as u64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLinearMipLevel {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub storage_width: u32,
    pub storage_height: u32,
    pub block_height_log2: u32,
    pub stride_alignment_log2: u32,
    pub guest_offset: usize,
    pub guest_size: usize,
    pub linear_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLinearMipLayout {
    pub levels: Vec<BlockLinearMipLevel>,
    pub layer_size: usize,
    pub layer_stride: usize,
}

impl BlockLinearMipLayout {
    pub fn guest_size_bytes(&self, layers: u32) -> usize {
        if layers > 1 {
            self.layer_stride.saturating_mul(layers as usize)
        } else {
            self.layer_size
        }
    }
}

pub fn block_linear_mip_layout(tic: &TicEntry) -> Option<BlockLinearMipLayout> {
    if !tic.is_block_linear || tic.texture_type == 2 || tic.is_sparse {
        return None;
    }

    let mip_levels = tic.mip_levels();
    let bpp = tic.format.src_bpp();
    let bpp_log2 = bpp.checked_ilog2()?;
    if (1usize << bpp_log2) != bpp {
        return None;
    }

    let mut levels = Vec::with_capacity(mip_levels as usize);
    let mut layer_size = 0usize;
    for level in 0..mip_levels {
        let (width, height) = tic.logical_mip_extent(level);
        let (storage_width, storage_height, _) = tic.physical_storage_extent(level)?;
        let width_bytes = storage_width.saturating_mul(bpp as u32);

        let single_base_level = level == 0 && mip_levels == 1;
        let block_width_log2 = if single_base_level {
            tic.block_width_log2
        } else {
            adjusted_mip_block_log2(width_bytes, tic.block_width_log2, 64)
        };
        let block_height_log2 = if single_base_level {
            tic.block_height_log2
        } else {
            adjusted_mip_block_log2(storage_height, tic.block_height_log2, 8)
        };
        let block_depth_log2 = if single_base_level {
            tic.block_depth_log2
        } else {
            adjusted_mip_block_log2(1, tic.block_depth_log2, 1)
        };

        let stride_gob_width_log2 = 6u32
            .saturating_sub(bpp_log2)
            .saturating_add(tic.tile_width_spacing);
        let stride_gob_height_log2 = 3u32.saturating_add(tic.block_height_log2);
        let stride_is_small = storage_width <= (1u32 << stride_gob_width_log2.min(31))
            || storage_height <= (1u32 << stride_gob_height_log2.min(31))
            || 1 < (1u32 << tic.block_depth_log2.min(31));
        let stride_alignment_log2 =
            6u32.saturating_sub(bpp_log2)
                .saturating_add(if stride_is_small {
                    0
                } else {
                    tic.tile_width_spacing
                });

        let mut gobs_w = div_ceil_u32(width_bytes, 64);
        let gobs_h = div_ceil_u32(storage_height, 8);
        let gob_width_log2 = 6u32
            .saturating_sub(bpp_log2)
            .saturating_add(tic.tile_width_spacing);
        let gob_height_log2 = 3u32.saturating_add(tic.block_height_log2);
        let is_small = width_bytes <= (1u32 << gob_width_log2.min(31))
            || storage_height <= (1u32 << gob_height_log2.min(31))
            || 1 < (1u32 << tic.block_depth_log2.min(31));
        if !is_small && tic.tile_width_spacing != 0 {
            gobs_w = align_up_log2_u32(gobs_w, tic.tile_width_spacing);
        }

        let tiles_w = div_ceil_pow2_u32(gobs_w, block_width_log2);
        let tiles_h = div_ceil_pow2_u32(gobs_h, block_height_log2);
        let tiles_d = div_ceil_pow2_u32(1, block_depth_log2);
        let tile_count = (tiles_w as usize)
            .saturating_mul(tiles_h as usize)
            .saturating_mul(tiles_d as usize);
        let guest_size = tile_count
            .checked_shl(
                9u32.saturating_add(block_width_log2)
                    .saturating_add(block_height_log2)
                    .saturating_add(block_depth_log2),
            )
            .unwrap_or(usize::MAX);
        let linear_size = tic.physical_linear_size(level)?;
        levels.push(BlockLinearMipLevel {
            level,
            width,
            height,
            storage_width,
            storage_height,
            block_height_log2,
            stride_alignment_log2,
            guest_offset: layer_size,
            guest_size,
            linear_size,
        });
        layer_size = layer_size.saturating_add(guest_size);
    }

    let alignment_log2 = if tic.tile_width_spacing != 0 {
        9u32.saturating_add(tic.tile_width_spacing)
            .saturating_add(tic.block_height_log2)
            .saturating_add(tic.block_depth_log2)
    } else {
        let (_, tile_height) = tic.format.block_extent();
        let (_, physical_height) = tic.physical_mip_extent(0)?;
        let aligned_height = align_up_u32(physical_height, tile_height.max(1));
        let block_height_log2 = adjusted_mip_block_log2(aligned_height, tic.block_height_log2, 8);
        let block_depth_log2 = adjusted_mip_block_log2(1, tic.block_depth_log2, 1);
        9u32.saturating_add(block_height_log2)
            .saturating_add(block_depth_log2)
    };
    let layer_stride = align_up_log2_usize(layer_size, alignment_log2);
    Some(BlockLinearMipLayout {
        levels,
        layer_size,
        layer_stride,
    })
}

pub fn texture_guest_size_bytes(tic: &TicEntry, layers: u32) -> Option<usize> {
    tic.pitch_linear_size(layers).or_else(|| {
        block_linear_mip_layout(tic).map(|layout| layout.guest_size_bytes(layers.max(1)))
    })
}

pub fn unpack_pitch_linear(raw: &[u8], tic: &TicEntry, layers: u32) -> Option<Vec<u8>> {
    let layer_guest_size = tic.pitch_linear_layer_size()?;
    let (storage_width, storage_height, bpp) = tic.physical_storage_extent(0)?;
    let row_size = (storage_width as usize).checked_mul(bpp)?;
    let layer_linear_size = row_size.checked_mul(storage_height as usize)?;
    let layer_count = layers.max(1) as usize;
    let guest_size = layer_guest_size.checked_mul(layer_count)?;
    let linear_size = layer_linear_size.checked_mul(layer_count)?;
    if raw.len() < guest_size {
        return None;
    }
    let mut linear = vec![0; linear_size];
    let pitch = tic.pitch_bytes as usize;
    for layer in 0..layer_count {
        let guest_layer = layer.checked_mul(layer_guest_size)?;
        let linear_layer = layer.checked_mul(layer_linear_size)?;
        for row in 0..storage_height as usize {
            let guest_start = guest_layer.checked_add(row.checked_mul(pitch)?)?;
            let linear_start = linear_layer.checked_add(row.checked_mul(row_size)?)?;
            linear[linear_start..linear_start + row_size]
                .copy_from_slice(&raw[guest_start..guest_start + row_size]);
        }
    }
    Some(linear)
}

fn adjusted_mip_block_log2(size: u32, mut block_log2: u32, gob_extent: u32) -> u32 {
    while block_log2 > 0 && size <= (1u32 << (block_log2 - 1).min(31)).saturating_mul(gob_extent) {
        block_log2 -= 1;
    }
    block_log2
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value / divisor + u32::from(value % divisor != 0)
}

fn div_ceil_pow2_u32(value: u32, shift: u32) -> u32 {
    let divisor = 1u32 << shift.min(31);
    div_ceil_u32(value, divisor)
}

fn align_up_u32(value: u32, alignment: u32) -> u32 {
    div_ceil_u32(value, alignment).saturating_mul(alignment)
}

fn align_up_log2_u32(value: u32, shift: u32) -> u32 {
    let alignment = 1u32 << shift.min(31);
    align_up_u32(value, alignment)
}

fn align_up_log2_usize(value: usize, shift: u32) -> usize {
    let Some(alignment) = 1usize.checked_shl(shift) else {
        return usize::MAX;
    };
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .unwrap_or(usize::MAX)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapMode {
    Wrap,
    Mirror,
    ClampToEdge,
    Border,
    Clamp,
    MirrorOnceClampToEdge,
    MirrorOnceBorder,
    MirrorOnceClampOgl,
    Unknown,
}
impl WrapMode {
    pub fn from_raw(v: u32) -> Self {
        match v & 0x7 {
            0 => WrapMode::Wrap,
            1 => WrapMode::Mirror,
            2 => WrapMode::ClampToEdge,
            3 => WrapMode::Border,
            4 => WrapMode::Clamp,
            5 => WrapMode::MirrorOnceClampToEdge,
            6 => WrapMode::MirrorOnceBorder,
            7 => WrapMode::MirrorOnceClampOgl,
            _ => WrapMode::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexFilter {
    None,
    Nearest,
    Linear,
}
impl TexFilter {
    pub fn from_filter_raw(v: u32) -> Self {
        match v & 0x3 {
            2 => TexFilter::Linear,
            _ => TexFilter::Nearest,
        }
    }

    pub fn from_mipmap_raw(v: u32) -> Self {
        match v & 0x3 {
            2 => TexFilter::Nearest,
            3 => TexFilter::Linear,
            _ => TexFilter::None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerReduction {
    WeightedAverage,
    Min,
    Max,
}

impl SamplerReduction {
    pub fn from_raw(v: u32) -> Self {
        match v & 0x3 {
            1 => SamplerReduction::Min,
            2 => SamplerReduction::Max,
            _ => SamplerReduction::WeightedAverage,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthCompareFunc {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl DepthCompareFunc {
    pub fn from_raw(v: u32) -> Self {
        match v & 0x7 {
            1 => DepthCompareFunc::Less,
            2 => DepthCompareFunc::Equal,
            3 => DepthCompareFunc::LessEqual,
            4 => DepthCompareFunc::Greater,
            5 => DepthCompareFunc::NotEqual,
            6 => DepthCompareFunc::GreaterEqual,
            7 => DepthCompareFunc::Always,
            _ => DepthCompareFunc::Never,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TscEntry {
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub wrap_p: WrapMode,
    pub depth_compare_enabled: bool,
    pub depth_compare_func: DepthCompareFunc,
    pub max_anisotropy: u32,
    pub mag_filter: TexFilter,
    pub min_filter: TexFilter,
    pub mip_filter: TexFilter,
    pub reduction: SamplerReduction,
    pub mip_lod_bias: i32,
    pub min_lod_clamp: u32,
    pub max_lod_clamp: u32,
    pub border_color_bits: [u32; 4],
}
impl TscEntry {
    pub fn parse(raw: &[u8]) -> Option<TscEntry> {
        if raw.len() < 32 {
            return None;
        }
        let w0 = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let w1 = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let w2 = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
        let border_color_bits = [
            u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]),
            u32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]),
            u32::from_le_bytes([raw[24], raw[25], raw[26], raw[27]]),
            u32::from_le_bytes([raw[28], raw[29], raw[30], raw[31]]),
        ];
        let raw_bias = ((w1 >> 12) & 0x1FFF) as i32;
        let mip_lod_bias = if (raw_bias & 0x1000) != 0 {
            raw_bias - 0x2000
        } else {
            raw_bias
        };
        Some(TscEntry {
            wrap_u: WrapMode::from_raw(w0),
            wrap_v: WrapMode::from_raw(w0 >> 3),
            wrap_p: WrapMode::from_raw(w0 >> 6),
            depth_compare_enabled: ((w0 >> 9) & 1) != 0,
            depth_compare_func: DepthCompareFunc::from_raw(w0 >> 10),
            max_anisotropy: (w0 >> 20) & 0x7,
            mag_filter: TexFilter::from_filter_raw(w1),
            min_filter: TexFilter::from_filter_raw(w1 >> 4),
            mip_filter: TexFilter::from_mipmap_raw(w1 >> 6),
            reduction: SamplerReduction::from_raw(w1 >> 10),
            mip_lod_bias,
            min_lod_clamp: w2 & 0xFFF,
            max_lod_clamp: (w2 >> 12) & 0xFFF,
            border_color_bits,
        })
    }

    pub fn lod_bias(&self) -> f32 {
        self.mip_lod_bias as f32 / 256.0
    }

    pub fn min_lod(&self) -> f32 {
        self.min_lod_clamp as f32 / 256.0
    }

    pub fn max_lod(&self) -> f32 {
        self.max_lod_clamp as f32 / 256.0
    }

    pub fn max_anisotropy(&self) -> f32 {
        (1u32 << self.max_anisotropy.min(4)) as f32
    }
}

const GOB_W: usize = 64;
const GOB_H: usize = 8;
const GOB_SIZE: usize = 512;

pub fn unswizzle_block_linear(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    bpp: usize,
    block_height_log2: u32,
) -> Vec<u8> {
    let stride_alignment_log2 = 6u32.saturating_sub(bpp.checked_ilog2().unwrap_or(0));
    unswizzle_block_linear_strided(
        src,
        width_px,
        height_px,
        bpp,
        block_height_log2,
        stride_alignment_log2,
    )
}

pub fn unswizzle_block_linear_strided(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    bpp: usize,
    block_height_log2: u32,
    stride_alignment_log2: u32,
) -> Vec<u8> {
    let width = width_px as usize;
    let height = height_px as usize;
    let dst_stride = width * bpp;
    let mut dst = vec![0u8; dst_stride * height];

    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let aligned_width = align_up_pow2_usize(width, stride_alignment_log2);
    let aligned_width_bytes = aligned_width.saturating_mul(bpp);
    let gobs_per_row = aligned_width_bytes.div_ceil(GOB_W);
    let block_row_stride_bytes = gobs_per_row * block_height * GOB_SIZE;

    for y in 0..height {
        let block_y = y / rows_per_block;
        let y_in_block = y - block_y * rows_per_block;
        let gob_row_in_block = y_in_block / GOB_H;
        let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
        let block_row_offset = block_y * block_row_stride_bytes;
        for x in 0..width {
            let byte_x = x * bpp;
            let gob_col = byte_x / GOB_W;
            let x_in_gob = byte_x - gob_col * GOB_W;
            let gob_offset =
                block_row_offset + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
            let in_gob = ((x_in_gob >> 5) & 1) * 256
                + ((y_in_gob >> 1) & 3) * 64
                + ((x_in_gob >> 4) & 1) * 32
                + (y_in_gob & 1) * 16
                + (x_in_gob & 15);
            let src_off = gob_offset + in_gob;
            let dst_off = y * dst_stride + byte_x;
            if src_off + bpp <= src.len() && dst_off + bpp <= dst.len() {
                dst[dst_off..dst_off + bpp].copy_from_slice(&src[src_off..src_off + bpp]);
            }
        }
    }

    dst
}

pub fn swizzle_block_linear_strided(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    bpp: usize,
    block_height_log2: u32,
    stride_alignment_log2: u32,
) -> Vec<u8> {
    let width = width_px as usize;
    let height = height_px as usize;
    let src_stride = width.saturating_mul(bpp);

    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height.saturating_mul(GOB_H);
    let aligned_width = align_up_pow2_usize(width, stride_alignment_log2);
    let aligned_width_bytes = aligned_width.saturating_mul(bpp);
    let gobs_per_row = aligned_width_bytes.div_ceil(GOB_W);
    let block_row_stride_bytes = gobs_per_row
        .saturating_mul(block_height)
        .saturating_mul(GOB_SIZE);
    let block_rows = height.div_ceil(rows_per_block);
    let mut dst = vec![0u8; block_rows.saturating_mul(block_row_stride_bytes)];

    for y in 0..height {
        let block_y = y / rows_per_block;
        let y_in_block = y - block_y * rows_per_block;
        let gob_row_in_block = y_in_block / GOB_H;
        let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
        let block_row_offset = block_y * block_row_stride_bytes;
        for x in 0..width {
            let byte_x = x * bpp;
            let gob_col = byte_x / GOB_W;
            let x_in_gob = byte_x - gob_col * GOB_W;
            let gob_offset =
                block_row_offset + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
            let in_gob = ((x_in_gob >> 5) & 1) * 256
                + ((y_in_gob >> 1) & 3) * 64
                + ((x_in_gob >> 4) & 1) * 32
                + (y_in_gob & 1) * 16
                + (x_in_gob & 15);
            let dst_off = gob_offset + in_gob;
            let src_off = y * src_stride + byte_x;
            if src_off + bpp <= src.len() && dst_off + bpp <= dst.len() {
                dst[dst_off..dst_off + bpp].copy_from_slice(&src[src_off..src_off + bpp]);
            }
        }
    }

    dst
}

pub fn unswizzle_block_linear_3d(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> Vec<u8> {
    unswizzle_block_linear_3d_with_block_width(
        src,
        width_px,
        height_px,
        depth_px,
        bpp,
        0,
        block_height_log2,
        block_depth_log2,
        tile_width_spacing,
    )
}

pub fn unswizzle_block_linear_3d_with_block_width(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    block_width_log2: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> Vec<u8> {
    let width = width_px as usize;
    let height = height_px as usize;
    let depth = depth_px as usize;
    let dst_stride = width.saturating_mul(bpp);
    let mut dst = vec![0u8; dst_stride.saturating_mul(height).saturating_mul(depth)];
    let gobs_in_x = align_up_pow2_usize(
        block_linear_gobs_in_x(
            width,
            height,
            depth,
            bpp,
            block_height_log2,
            block_depth_log2,
            tile_width_spacing,
        ),
        block_width_log2,
    );
    let block_height = 1usize << block_height_log2 as usize;
    let block_depth = 1usize << block_depth_log2 as usize;
    let block_size = gobs_in_x << (9 + block_height_log2 as usize + block_depth_log2 as usize);
    let slice_size = ((height + block_height * GOB_H - 1) / (block_height * GOB_H)) * block_size;
    let block_height_mask = block_height - 1;
    let block_depth_mask = block_depth - 1;
    let x_shift = 9usize + block_height_log2 as usize + block_depth_log2 as usize;

    for z in 0..depth {
        let offset_z = (z / block_depth) * slice_size
            + (z & block_depth_mask) * (GOB_SIZE << block_height_log2 as usize);
        for y in 0..height {
            let block_y = y / GOB_H;
            let offset_y =
                (block_y / block_height) * block_size + (block_y & block_height_mask) * GOB_SIZE;
            let y_in_gob = y & (GOB_H - 1);
            for x in 0..width {
                let byte_x = x.saturating_mul(bpp);
                let offset_x = (byte_x / GOB_W) << x_shift;
                let x_in_gob = byte_x & (GOB_W - 1);
                let in_gob = ((x_in_gob >> 5) & 1) * 256
                    + ((y_in_gob >> 1) & 3) * 64
                    + ((x_in_gob >> 4) & 1) * 32
                    + (y_in_gob & 1) * 16
                    + (x_in_gob & 15);
                let src_off = offset_z + offset_y + offset_x + in_gob;
                let dst_off = (z * height * dst_stride) + y * dst_stride + byte_x;
                if src_off + bpp <= src.len() && dst_off + bpp <= dst.len() {
                    dst[dst_off..dst_off + bpp].copy_from_slice(&src[src_off..src_off + bpp]);
                }
            }
        }
    }

    dst
}

pub fn swizzle_block_linear_3d(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> Vec<u8> {
    let width = width_px as usize;
    let height = height_px as usize;
    let depth = depth_px as usize;
    let src_stride = width.saturating_mul(bpp);
    let mut dst = vec![
        0u8;
        block_linear_byte_size_3d(
            width_px,
            height_px,
            depth_px,
            bpp,
            block_height_log2,
            block_depth_log2,
            tile_width_spacing,
        )
    ];
    let gobs_in_x = block_linear_gobs_in_x(
        width,
        height,
        depth,
        bpp,
        block_height_log2,
        block_depth_log2,
        tile_width_spacing,
    );
    let block_height = 1usize << block_height_log2 as usize;
    let block_depth = 1usize << block_depth_log2 as usize;
    let block_size = gobs_in_x << (9 + block_height_log2 as usize + block_depth_log2 as usize);
    let slice_size = ((height + block_height * GOB_H - 1) / (block_height * GOB_H)) * block_size;
    let block_height_mask = block_height - 1;
    let block_depth_mask = block_depth - 1;
    let x_shift = 9usize + block_height_log2 as usize + block_depth_log2 as usize;

    for z in 0..depth {
        let offset_z = (z / block_depth) * slice_size
            + (z & block_depth_mask) * (GOB_SIZE << block_height_log2 as usize);
        for y in 0..height {
            let block_y = y / GOB_H;
            let offset_y =
                (block_y / block_height) * block_size + (block_y & block_height_mask) * GOB_SIZE;
            let y_in_gob = y & (GOB_H - 1);
            for x in 0..width {
                let byte_x = x.saturating_mul(bpp);
                let offset_x = (byte_x / GOB_W) << x_shift;
                let x_in_gob = byte_x & (GOB_W - 1);
                let in_gob = ((x_in_gob >> 5) & 1) * 256
                    + ((y_in_gob >> 1) & 3) * 64
                    + ((x_in_gob >> 4) & 1) * 32
                    + (y_in_gob & 1) * 16
                    + (x_in_gob & 15);
                let dst_off = offset_z + offset_y + offset_x + in_gob;
                let src_off = (z * height * src_stride) + y * src_stride + byte_x;
                if src_off + bpp <= src.len() && dst_off + bpp <= dst.len() {
                    dst[dst_off..dst_off + bpp].copy_from_slice(&src[src_off..src_off + bpp]);
                }
            }
        }
    }

    dst
}

pub fn native_render_target_source_size(
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    layout_signature: u64,
) -> Option<usize> {
    let width = usize::try_from(width_px).ok()?;
    let height = usize::try_from(height_px).ok()?;
    let depth = usize::try_from(depth_px).ok()?;
    if width == 0 || height == 0 || depth == 0 || bpp == 0 {
        return None;
    }
    let row_bytes = width.checked_mul(bpp)?;
    let tight_size = row_bytes.checked_mul(height)?.checked_mul(depth)?;

    match layout_signature & 0xff {
        0 if layout_signature == 0 => Some(tight_size),
        1 if layout_signature >> 40 == 0 => {
            let block_width_log2 = ((layout_signature >> 8) & 0xff) as u32;
            let block_height_log2 = ((layout_signature >> 16) & 0xff) as u32;
            let block_depth_log2 = ((layout_signature >> 24) & 0xff) as u32;
            let tile_width_spacing = ((layout_signature >> 32) & 0xff) as u32;
            checked_block_linear_byte_size_3d_with_block_width(
                width,
                height,
                depth,
                bpp,
                block_width_log2,
                block_height_log2,
                block_depth_log2,
                tile_width_spacing,
            )
        }
        2 if layout_signature >> 40 == 0 => {
            let pitch = usize::try_from(layout_signature >> 8).ok()?;
            (pitch >= row_bytes)
                .then(|| pitch.checked_mul(height)?.checked_mul(depth))
                .flatten()
        }
        _ => None,
    }
}

pub fn unpack_native_render_target(
    src: &[u8],
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    layout_signature: u64,
) -> Option<Vec<u8>> {
    let guest_size =
        native_render_target_source_size(width_px, height_px, depth_px, bpp, layout_signature)?;
    if src.len() < guest_size {
        return None;
    }
    let width = usize::try_from(width_px).ok()?;
    let height = usize::try_from(height_px).ok()?;
    let depth = usize::try_from(depth_px).ok()?;
    let row_bytes = width.checked_mul(bpp)?;
    let tight_size = row_bytes.checked_mul(height)?.checked_mul(depth)?;

    match layout_signature & 0xff {
        0 if layout_signature == 0 => Some(src[..tight_size].to_vec()),
        1 if layout_signature >> 40 == 0 => {
            let block_width_log2 = ((layout_signature >> 8) & 0xff) as u32;
            let block_height_log2 = ((layout_signature >> 16) & 0xff) as u32;
            let block_depth_log2 = ((layout_signature >> 24) & 0xff) as u32;
            let tile_width_spacing = ((layout_signature >> 32) & 0xff) as u32;
            let unit_bpp_log2 = row_bytes.trailing_zeros().min(4);
            let unit_bpp = 1usize.checked_shl(unit_bpp_log2)?;
            let unit_width = u32::try_from(row_bytes / unit_bpp).ok()?;
            let linear = unswizzle_block_linear_3d_with_block_width(
                &src[..guest_size],
                unit_width,
                height_px,
                depth_px,
                unit_bpp,
                block_width_log2,
                block_height_log2,
                block_depth_log2,
                tile_width_spacing,
            );
            (linear.len() == tight_size).then_some(linear)
        }
        2 if layout_signature >> 40 == 0 => {
            let pitch = usize::try_from(layout_signature >> 8).ok()?;
            if pitch < row_bytes {
                return None;
            }
            let slice_size = pitch.checked_mul(height)?;
            let mut linear = Vec::with_capacity(tight_size);
            for z in 0..depth {
                let slice_offset = z.checked_mul(slice_size)?;
                for y in 0..height {
                    let row_offset = slice_offset.checked_add(y.checked_mul(pitch)?)?;
                    let row_end = row_offset.checked_add(row_bytes)?;
                    linear.extend_from_slice(src.get(row_offset..row_end)?);
                }
            }
            Some(linear)
        }
        _ => None,
    }
}

pub fn block_linear_byte_size_3d(
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> usize {
    block_linear_byte_size_3d_with_block_width(
        width_px,
        height_px,
        depth_px,
        bpp,
        0,
        block_height_log2,
        block_depth_log2,
        tile_width_spacing,
    )
}

pub fn block_linear_byte_size_3d_with_block_width(
    width_px: u32,
    height_px: u32,
    depth_px: u32,
    bpp: usize,
    block_width_log2: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> usize {
    let gobs_in_x = align_up_pow2_usize(
        block_linear_gobs_in_x(
            width_px as usize,
            height_px as usize,
            depth_px as usize,
            bpp,
            block_height_log2,
            block_depth_log2,
            tile_width_spacing,
        ),
        block_width_log2,
    );
    let block_height = 1usize << block_height_log2 as usize;
    let block_depth = 1usize << block_depth_log2 as usize;
    let block_size = gobs_in_x << (9 + block_height_log2 as usize + block_depth_log2 as usize);
    let slice_size =
        ((height_px as usize + block_height * GOB_H - 1) / (block_height * GOB_H)) * block_size;
    ((depth_px as usize + block_depth - 1) / block_depth).saturating_mul(slice_size)
}

fn block_linear_gobs_in_x(
    width: usize,
    height: usize,
    depth: usize,
    bpp: usize,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> usize {
    let width_bytes = width.saturating_mul(bpp);
    let raw_gobs = (width_bytes + GOB_W - 1) / GOB_W;
    let gob_width_bytes = GOB_W.checked_shl(tile_width_spacing).unwrap_or(usize::MAX);
    let gob_height = GOB_H.checked_shl(block_height_log2).unwrap_or(usize::MAX);
    let block_depth = 1usize.checked_shl(block_depth_log2).unwrap_or(usize::MAX);
    let small = width_bytes <= gob_width_bytes || height <= gob_height || depth < block_depth;
    if small {
        raw_gobs
    } else {
        align_up_pow2_usize(raw_gobs, tile_width_spacing)
    }
}

fn checked_block_linear_byte_size_3d_with_block_width(
    width: usize,
    height: usize,
    depth: usize,
    bpp: usize,
    block_width_log2: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
) -> Option<usize> {
    let width_bytes = width.checked_mul(bpp)?;
    let raw_gobs = width_bytes.div_ceil(GOB_W);
    let gob_width_bytes = GOB_W.checked_shl(tile_width_spacing)?;
    let gob_height = GOB_H.checked_shl(block_height_log2)?;
    let block_depth = 1usize.checked_shl(block_depth_log2)?;
    let small = width_bytes <= gob_width_bytes || height <= gob_height || depth < block_depth;
    let spaced_gobs = if small {
        raw_gobs
    } else {
        checked_align_up_pow2_usize(raw_gobs, tile_width_spacing)?
    };
    let gobs_in_x = checked_align_up_pow2_usize(spaced_gobs, block_width_log2)?;
    let block_height = 1usize.checked_shl(block_height_log2)?;
    let rows_per_block = block_height.checked_mul(GOB_H)?;
    let x_shift = 9u32
        .checked_add(block_height_log2)?
        .checked_add(block_depth_log2)?;
    let block_size = gobs_in_x.checked_shl(x_shift)?;
    let slice_size = height.div_ceil(rows_per_block).checked_mul(block_size)?;
    depth.div_ceil(block_depth).checked_mul(slice_size)
}

fn checked_align_up_pow2_usize(value: usize, shift: u32) -> Option<usize> {
    let alignment = 1usize.checked_shl(shift)?;
    let mask = alignment.checked_sub(1)?;
    Some(value.checked_add(mask)? & !mask)
}

fn align_up_pow2_usize(value: usize, shift: u32) -> usize {
    if shift == 0 {
        value
    } else {
        let mask = (1usize << shift) - 1;
        (value + mask) & !mask
    }
}

pub fn block_linear_byte_size(
    width_px: u32,
    height_px: u32,
    bpp: usize,
    block_height_log2: u32,
) -> usize {
    let width_bytes = (width_px as usize) * bpp;
    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let block_rows = (height_px as usize + rows_per_block - 1) / rows_per_block;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    block_rows * gobs_per_row * block_height * GOB_SIZE
}

fn expand_5(v: u16) -> u8 {
    let v = v as u8;
    (v << 3) | (v >> 2)
}

fn expand_6(v: u16) -> u8 {
    let v = v as u8;
    (v << 2) | (v >> 4)
}

fn rgb565(v: u16) -> [u8; 3] {
    [
        expand_5((v >> 11) & 0x1F),
        expand_6((v >> 5) & 0x3F),
        expand_5(v & 0x1F),
    ]
}

fn bc_color_palette(block: &[u8], bc1_alpha: bool) -> [[u8; 4]; 4] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let p0 = rgb565(c0);
    let p1 = rgb565(c1);
    let mut p = [[0u8; 4]; 4];
    p[0] = [p0[0], p0[1], p0[2], 255];
    p[1] = [p1[0], p1[1], p1[2], 255];
    if c0 > c1 || !bc1_alpha {
        for i in 0..3 {
            p[2][i] = ((2 * p0[i] as u16 + p1[i] as u16) / 3) as u8;
            p[3][i] = ((p0[i] as u16 + 2 * p1[i] as u16) / 3) as u8;
        }
        p[2][3] = 255;
        p[3][3] = 255;
    } else {
        for i in 0..3 {
            p[2][i] = ((p0[i] as u16 + p1[i] as u16) / 2) as u8;
        }
        p[2][3] = 255;
    }
    p
}

fn put_rgba(out: &mut [u8], width: usize, height: usize, x: usize, y: usize, rgba: [u8; 4]) {
    if x >= width || y >= height {
        return;
    }
    let off = (y * width + x) * 4;
    out[off..off + 4].copy_from_slice(&rgba);
}

fn decode_bc1(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let width = width as usize;
    let height = height as usize;
    let blocks_w = (width + 3) / 4;
    let blocks_h = (height + 3) / 4;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let off = (by * blocks_w + bx) * 8;
            if off + 8 > src.len() {
                continue;
            }
            let block = &src[off..off + 8];
            let palette = bc_color_palette(block, true);
            let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            for py in 0..4 {
                for px in 0..4 {
                    let idx = ((bits >> (2 * (py * 4 + px))) & 3) as usize;
                    put_rgba(out, width, height, bx * 4 + px, by * 4 + py, palette[idx]);
                }
            }
        }
    }
}

fn decode_bc2(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let width = width as usize;
    let height = height as usize;
    let blocks_w = (width + 3) / 4;
    let blocks_h = (height + 3) / 4;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let off = (by * blocks_w + bx) * 16;
            if off + 16 > src.len() {
                continue;
            }
            let block = &src[off..off + 16];
            let alpha = u64::from_le_bytes([
                block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7],
            ]);
            let palette = bc_color_palette(&block[8..16], false);
            let bits = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
            for py in 0..4 {
                for px in 0..4 {
                    let pos = py * 4 + px;
                    let idx = ((bits >> (2 * pos)) & 3) as usize;
                    let mut rgba = palette[idx];
                    rgba[3] = (((alpha >> (4 * pos)) & 0xF) as u8) * 17;
                    put_rgba(out, width, height, bx * 4 + px, by * 4 + py, rgba);
                }
            }
        }
    }
}

fn decode_bc3(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let width = width as usize;
    let height = height as usize;
    let blocks_w = (width + 3) / 4;
    let blocks_h = (height + 3) / 4;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let off = (by * blocks_w + bx) * 16;
            if off + 16 > src.len() {
                continue;
            }
            let block = &src[off..off + 16];
            let a0 = block[0];
            let a1 = block[1];
            let mut alpha_palette = [0u8; 8];
            alpha_palette[0] = a0;
            alpha_palette[1] = a1;
            if a0 > a1 {
                alpha_palette[2] = ((6 * a0 as u16 + a1 as u16) / 7) as u8;
                alpha_palette[3] = ((5 * a0 as u16 + 2 * a1 as u16) / 7) as u8;
                alpha_palette[4] = ((4 * a0 as u16 + 3 * a1 as u16) / 7) as u8;
                alpha_palette[5] = ((3 * a0 as u16 + 4 * a1 as u16) / 7) as u8;
                alpha_palette[6] = ((2 * a0 as u16 + 5 * a1 as u16) / 7) as u8;
                alpha_palette[7] = ((a0 as u16 + 6 * a1 as u16) / 7) as u8;
            } else {
                alpha_palette[2] = ((4 * a0 as u16 + a1 as u16) / 5) as u8;
                alpha_palette[3] = ((3 * a0 as u16 + 2 * a1 as u16) / 5) as u8;
                alpha_palette[4] = ((2 * a0 as u16 + 3 * a1 as u16) / 5) as u8;
                alpha_palette[5] = ((a0 as u16 + 4 * a1 as u16) / 5) as u8;
                alpha_palette[6] = 0;
                alpha_palette[7] = 255;
            }
            let mut alpha_bits = 0u64;
            for i in 0..6 {
                alpha_bits |= (block[2 + i] as u64) << (8 * i);
            }
            let palette = bc_color_palette(&block[8..16], false);
            let bits = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
            for py in 0..4 {
                for px in 0..4 {
                    let pos = py * 4 + px;
                    let idx = ((bits >> (2 * pos)) & 3) as usize;
                    let alpha_idx = ((alpha_bits >> (3 * pos)) & 7) as usize;
                    let mut rgba = palette[idx];
                    rgba[3] = alpha_palette[alpha_idx];
                    put_rgba(out, width, height, bx * 4 + px, by * 4 + py, rgba);
                }
            }
        }
    }
}

fn decode_astc(src: &[u8], width: u32, height: u32, bw: usize, bh: usize, out: &mut [u8]) {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u32; w * h];
    if texture2ddecoder::decode_astc(src, w, h, bw, bh, &mut buf).is_err() {
        for px in out.chunks_exact_mut(4) {
            px[0] = 0xFF;
            px[1] = 0x00;
            px[2] = 0xFF;
            px[3] = 0xFF;
        }
        return;
    }
    for (i, c) in buf.iter().enumerate() {
        let o = i * 4;
        out[o] = ((c >> 16) & 0xFF) as u8;
        out[o + 1] = ((c >> 8) & 0xFF) as u8;
        out[o + 2] = (c & 0xFF) as u8;
        out[o + 3] = ((c >> 24) & 0xFF) as u8;
    }
}

fn unpack_bcn_u32(buf: &[u32], out: &mut [u8]) {
    for (i, c) in buf.iter().enumerate() {
        let o = i * 4;
        if o + 3 < out.len() {
            out[o] = ((c >> 16) & 0xFF) as u8;
            out[o + 1] = ((c >> 8) & 0xFF) as u8;
            out[o + 2] = (c & 0xFF) as u8;
            out[o + 3] = ((c >> 24) & 0xFF) as u8;
        }
    }
}

fn fill_magenta(out: &mut [u8]) {
    for px in out.chunks_exact_mut(4) {
        px[0] = 0xFF;
        px[1] = 0x00;
        px[2] = 0xFF;
        px[3] = 0xFF;
    }
}

fn decode_ufloat(v: u32, mantissa_bits: u32) -> f32 {
    let exponent_bits = 5;
    let mantissa_mask = (1u32 << mantissa_bits) - 1;
    let exponent_mask = (1u32 << exponent_bits) - 1;
    let mantissa = v & mantissa_mask;
    let exponent = (v >> mantissa_bits) & exponent_mask;
    if exponent == 0 {
        (mantissa as f32) * 2f32.powi(-14 - mantissa_bits as i32)
    } else if exponent == exponent_mask {
        if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + (mantissa as f32) / ((1u32 << mantissa_bits) as f32))
            * 2f32.powi(exponent as i32 - 15)
    }
}

fn float_to_u8(v: f32) -> u8 {
    if !v.is_finite() {
        return if v.is_sign_positive() { 255 } else { 0 };
    }
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    match exponent {
        0 => sign * 2f32.powi(-14) * (mantissa as f32 / 1024.0),
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * 2f32.powi(exponent as i32 - 15) * (1.0 + mantissa as f32 / 1024.0),
    }
}

fn decode_b10g11r11(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let pixels = width as usize * height as usize;
    for i in 0..pixels.min(src.len() / 4) {
        let p = u32::from_le_bytes([src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]]);
        let r = decode_ufloat(p & 0x7ff, 6);
        let g = decode_ufloat((p >> 11) & 0x7ff, 6);
        let b = decode_ufloat((p >> 22) & 0x3ff, 5);
        out[i * 4] = float_to_u8(r);
        out[i * 4 + 1] = float_to_u8(g);
        out[i * 4 + 2] = float_to_u8(b);
        out[i * 4 + 3] = 0xff;
    }
}

fn decode_bc4(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    if texture2ddecoder::decode_bc4(src, w, h, &mut buf).is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

fn decode_bc5(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    if texture2ddecoder::decode_bc5(src, w, h, &mut buf).is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

fn decode_bc6(src: &[u8], width: u32, height: u32, signed: bool, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    let result = if signed {
        texture2ddecoder::decode_bc6_signed(src, w, h, &mut buf)
    } else {
        texture2ddecoder::decode_bc6_unsigned(src, w, h, &mut buf)
    };
    if result.is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

fn bc_snorm_palette(block: &[u8]) -> ([u8; 8], u64) {
    let raw_e0 = block[0] as i8;
    let raw_e1 = block[1] as i8;
    let e0 = (raw_e0 as i16).max(-127);
    let e1 = (raw_e1 as i16).max(-127);
    let mut values = [0i16; 8];
    values[0] = e0;
    values[1] = e1;
    if raw_e0 > raw_e1 {
        values[2] = (6 * e0 + e1) / 7;
        values[3] = (5 * e0 + 2 * e1) / 7;
        values[4] = (4 * e0 + 3 * e1) / 7;
        values[5] = (3 * e0 + 4 * e1) / 7;
        values[6] = (2 * e0 + 5 * e1) / 7;
        values[7] = (e0 + 6 * e1) / 7;
    } else {
        values[2] = (4 * e0 + e1) / 5;
        values[3] = (3 * e0 + 2 * e1) / 5;
        values[4] = (2 * e0 + 3 * e1) / 5;
        values[5] = (e0 + 4 * e1) / 5;
        values[6] = -127;
        values[7] = 127;
    }
    let mut palette = [0u8; 8];
    for (dst, value) in palette.iter_mut().zip(values) {
        *dst = value as i8 as u8;
    }
    let mut indices = 0u64;
    for (i, byte) in block[2..8].iter().enumerate() {
        indices |= (*byte as u64) << (i * 8);
    }
    (palette, indices)
}

fn decode_bc4_snorm(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let width = width as usize;
    let height = height as usize;
    let blocks_w = (width + 3) / 4;
    let blocks_h = (height + 3) / 4;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let off = (by * blocks_w + bx) * 8;
            if off + 8 > src.len() {
                continue;
            }
            let (palette, indices) = bc_snorm_palette(&src[off..off + 8]);
            for py in 0..4 {
                for px in 0..4 {
                    let pos = py * 4 + px;
                    let index = ((indices >> (pos * 3)) & 7) as usize;
                    put_rgba(
                        out,
                        width,
                        height,
                        bx * 4 + px,
                        by * 4 + py,
                        [palette[index], 0, 0, 127],
                    );
                }
            }
        }
    }
}

fn decode_bc5_snorm(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let width = width as usize;
    let height = height as usize;
    let blocks_w = (width + 3) / 4;
    let blocks_h = (height + 3) / 4;
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let off = (by * blocks_w + bx) * 16;
            if off + 16 > src.len() {
                continue;
            }
            let (red, red_indices) = bc_snorm_palette(&src[off..off + 8]);
            let (green, green_indices) = bc_snorm_palette(&src[off + 8..off + 16]);
            for py in 0..4 {
                for px in 0..4 {
                    let pos = py * 4 + px;
                    let red_index = ((red_indices >> (pos * 3)) & 7) as usize;
                    let green_index = ((green_indices >> (pos * 3)) & 7) as usize;
                    put_rgba(
                        out,
                        width,
                        height,
                        bx * 4 + px,
                        by * 4 + py,
                        [red[red_index], green[green_index], 0, 127],
                    );
                }
            }
        }
    }
}

fn decode_bc7(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    if texture2ddecoder::decode_bc7(src, w, h, &mut buf).is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

fn decode_etc2(src: &[u8], width: u32, height: u32, alpha_bits: u8, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    let result = match alpha_bits {
        0 => texture2ddecoder::decode_etc2_rgb(src, w, h, &mut buf),
        1 => texture2ddecoder::decode_etc2_rgba1(src, w, h, &mut buf),
        _ => texture2ddecoder::decode_etc2_rgba8(src, w, h, &mut buf),
    };
    if result.is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

pub fn decode_to_rgba8_typed(
    src: &[u8],
    width: u32,
    height: u32,
    format: TicFormat,
    component_type: ComponentType,
) -> Vec<u8> {
    let pixels = (width as usize) * (height as usize);
    let mut out = vec![0u8; pixels * 4];

    match format {
        TicFormat::A8B8G8R8 => {
            let n = pixels.min(src.len() / 4);
            for i in 0..n {
                let off = i * 4;
                out[off] = src[off];
                out[off + 1] = src[off + 1];
                out[off + 2] = src[off + 2];
                out[off + 3] = src[off + 3];
            }
        }
        TicFormat::R8G8B8A8 => {
            let n = (pixels * 4).min(src.len());
            out[..n].copy_from_slice(&src[..n]);
        }
        TicFormat::A2B10G10R10 => {
            for i in 0..pixels.min(src.len() / 4) {
                let v = u32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                let r = v & 0x3ff;
                let g = (v >> 10) & 0x3ff;
                let b = (v >> 20) & 0x3ff;
                let a = (v >> 30) & 0x3;
                out[i * 4] = ((r * 255 + 511) / 1023) as u8;
                out[i * 4 + 1] = ((g * 255 + 511) / 1023) as u8;
                out[i * 4 + 2] = ((b * 255 + 511) / 1023) as u8;
                out[i * 4 + 3] = ((a * 255 + 1) / 3) as u8;
            }
        }
        TicFormat::R5G6B5 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = ((v >> 11) & 0x1F) as u8;
                let g = ((v >> 5) & 0x3F) as u8;
                let b = (v & 0x1F) as u8;
                out[i * 4] = (r << 3) | (r >> 2);
                out[i * 4 + 1] = (g << 2) | (g >> 4);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::A1R5G5B5 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let a = if (v >> 15) & 1 == 1 { 0xFFu8 } else { 0 };
                let r = ((v >> 10) & 0x1F) as u8;
                let g = ((v >> 5) & 0x1F) as u8;
                let b = (v & 0x1F) as u8;
                out[i * 4] = (r << 3) | (r >> 2);
                out[i * 4 + 1] = (g << 3) | (g >> 2);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = a;
            }
        }
        TicFormat::A4R4G4B4 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let a = ((v >> 12) & 0xF) as u8;
                let r = ((v >> 8) & 0xF) as u8;
                let g = ((v >> 4) & 0xF) as u8;
                let b = (v & 0xF) as u8;
                out[i * 4] = (r << 4) | r;
                out[i * 4 + 1] = (g << 4) | g;
                out[i * 4 + 2] = (b << 4) | b;
                out[i * 4 + 3] = (a << 4) | a;
            }
        }
        TicFormat::A5B5G5R1 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = if v & 1 == 1 { 0xFF } else { 0 };
                let g = ((v >> 1) & 0x1F) as u8;
                let b = ((v >> 6) & 0x1F) as u8;
                let a = ((v >> 11) & 0x1F) as u8;
                out[i * 4] = r;
                out[i * 4 + 1] = (g << 3) | (g >> 2);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = (a << 3) | (a >> 2);
            }
        }
        TicFormat::A1B5G5R5 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = (v & 0x1F) as u8;
                let g = ((v >> 5) & 0x1F) as u8;
                let b = ((v >> 10) & 0x1F) as u8;
                let a = if (v >> 15) & 1 == 1 { 0xFF } else { 0 };
                out[i * 4] = (r << 3) | (r >> 2);
                out[i * 4 + 1] = (g << 3) | (g >> 2);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = a;
            }
        }
        TicFormat::B5G6R5 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = (v & 0x1F) as u8;
                let g = ((v >> 5) & 0x3F) as u8;
                let b = ((v >> 11) & 0x1F) as u8;
                out[i * 4] = (r << 3) | (r >> 2);
                out[i * 4 + 1] = (g << 2) | (g >> 4);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::A4B4G4R4 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = (v & 0xF) as u8;
                let g = ((v >> 4) & 0xF) as u8;
                let b = ((v >> 8) & 0xF) as u8;
                let a = ((v >> 12) & 0xF) as u8;
                out[i * 4] = (r << 4) | r;
                out[i * 4 + 1] = (g << 4) | g;
                out[i * 4 + 2] = (b << 4) | b;
                out[i * 4 + 3] = (a << 4) | a;
            }
        }
        TicFormat::R8 => {
            for i in 0..pixels.min(src.len()) {
                let v = src[i];
                out[i * 4] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R8G8 => {
            for i in 0..pixels.min(src.len() / 2) {
                out[i * 4] = src[i * 2];
                out[i * 4 + 1] = src[i * 2 + 1];
                out[i * 4 + 2] = 0;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R16 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = src[i * 2 + 1];
                out[i * 4] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R16G16 => {
            for i in 0..pixels.min(src.len() / 4) {
                let r = src[i * 4 + 1];
                let g = src[i * 4 + 3];
                out[i * 4] = r;
                out[i * 4 + 1] = g;
                out[i * 4 + 2] = 0;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R32G32B32A32 => {
            for i in 0..pixels.min(src.len() / 16) {
                let off = i * 16;
                out[i * 4] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off],
                    src[off + 1],
                    src[off + 2],
                    src[off + 3],
                ])));
                out[i * 4 + 1] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off + 4],
                    src[off + 5],
                    src[off + 6],
                    src[off + 7],
                ])));
                out[i * 4 + 2] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off + 8],
                    src[off + 9],
                    src[off + 10],
                    src[off + 11],
                ])));
                out[i * 4 + 3] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off + 12],
                    src[off + 13],
                    src[off + 14],
                    src[off + 15],
                ])));
            }
        }
        TicFormat::R32G32 => {
            for i in 0..pixels.min(src.len() / 8) {
                let off = i * 8;
                out[i * 4] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off],
                    src[off + 1],
                    src[off + 2],
                    src[off + 3],
                ])));
                out[i * 4 + 1] = float_to_u8(f32::from_bits(u32::from_le_bytes([
                    src[off + 4],
                    src[off + 5],
                    src[off + 6],
                    src[off + 7],
                ])));
                out[i * 4 + 2] = 0;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R32 | TicFormat::Z32 => {
            for i in 0..pixels.min(src.len() / 4) {
                let raw = u32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                let v = float_to_u8(f32::from_bits(raw));
                out[i * 4] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::G24R8 => {
            for i in 0..pixels.min(src.len() / 4) {
                let raw = u32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                out[i * 4] = (raw & 0xff) as u8;
                out[i * 4 + 1] = (raw >> 24) as u8;
                out[i * 4 + 2] = 0;
                out[i * 4 + 3] = 0xff;
            }
        }
        TicFormat::R16G16B16A16 => {
            for i in 0..pixels.min(src.len() / 8) {
                let src_off = i * 8;
                let dst_off = i * 4;
                for component in 0..4 {
                    let off = src_off + component * 2;
                    let value = u16::from_le_bytes([src[off], src[off + 1]]);
                    out[dst_off + component] = float_to_u8(f16_to_f32(value));
                }
            }
        }
        TicFormat::Z24S8 | TicFormat::X8Z24 | TicFormat::S8Z24 => {
            let depth_high = matches!(format, TicFormat::Z24S8);
            for i in 0..pixels.min(src.len() / 4) {
                let raw = u32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                let v = if depth_high {
                    (raw >> 24) as u8
                } else {
                    (raw >> 16) as u8
                };
                out[i * 4] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::B10G11R11 => decode_b10g11r11(src, width, height, &mut out),
        TicFormat::BC1 => decode_bc1(src, width, height, &mut out),
        TicFormat::BC2 => decode_bc2(src, width, height, &mut out),
        TicFormat::BC3 => decode_bc3(src, width, height, &mut out),
        TicFormat::BC4
            if matches!(
                component_type,
                ComponentType::Snorm | ComponentType::SnormForceFp16
            ) =>
        {
            decode_bc4_snorm(src, width, height, &mut out)
        }
        TicFormat::BC4 => decode_bc4(src, width, height, &mut out),
        TicFormat::BC5
            if matches!(
                component_type,
                ComponentType::Snorm | ComponentType::SnormForceFp16
            ) =>
        {
            decode_bc5_snorm(src, width, height, &mut out)
        }
        TicFormat::BC5 => decode_bc5(src, width, height, &mut out),
        TicFormat::BC6S => decode_bc6(src, width, height, true, &mut out),
        TicFormat::BC6U => decode_bc6(src, width, height, false, &mut out),
        TicFormat::BC7 => decode_bc7(src, width, height, &mut out),
        TicFormat::Etc2Rgb => decode_etc2(src, width, height, 0, &mut out),
        TicFormat::Etc2RgbA1 => decode_etc2(src, width, height, 1, &mut out),
        TicFormat::Etc2Rgba => decode_etc2(src, width, height, 8, &mut out),
        TicFormat::Astc(bw, bh) => {
            decode_astc(src, width, height, bw as usize, bh as usize, &mut out)
        }
        TicFormat::Unknown(_) => {
            for i in 0..pixels {
                out[i * 4] = 0xFF;
                out[i * 4 + 1] = 0x00;
                out[i * 4 + 2] = 0xFF;
                out[i * 4 + 3] = 0xFF;
            }
        }
    }

    if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        if k < 400 {
            let (mut mr, mut mg, mut mb, mut ma, mut na) = (0u8, 0u8, 0u8, 0u8, 255u8);
            for px in out.chunks_exact(4) {
                mr = mr.max(px[0]);
                mg = mg.max(px[1]);
                mb = mb.max(px[2]);
                ma = ma.max(px[3]);
                na = na.min(px[3]);
            }
            log::warn!(
                "[texdecode] #{} {:?} {}x{} maxR={} maxG={} maxB={} alpha={}..{}",
                k,
                format,
                width,
                height,
                mr,
                mg,
                mb,
                na,
                ma
            );
        }
    }

    out
}

pub fn decode_to_rgba8(src: &[u8], width: u32, height: u32, format: TicFormat) -> Vec<u8> {
    decode_to_rgba8_typed(src, width, height, format, ComponentType::Unorm)
}

#[cfg(test)]
mod tests {
    use super::{
        block_linear_byte_size_3d, block_linear_byte_size_3d_with_block_width,
        block_linear_mip_layout, decode_to_rgba8, decode_to_rgba8_typed, maxwell_sample_grid,
        swizzle_block_linear_3d, swizzle_block_linear_strided, texture_guest_size_bytes,
        unpack_native_render_target, unpack_pitch_linear, unswizzle_block_linear_3d,
        unswizzle_block_linear_3d_with_block_width, unswizzle_block_linear_strided, ComponentType,
        SampleGrid, SwizzleSource, TicEntry, TicFormat,
    };

    #[test]
    fn parses_named_g24r8_target_descriptor() {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&0x2492_4a0eu32.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        let tic = TicEntry::parse(&raw).unwrap();
        assert_eq!(tic.format, TicFormat::G24R8);
        assert_eq!(tic.format.src_bpp(), 4);
        assert_eq!(
            tic.component_types,
            [
                ComponentType::Uint,
                ComponentType::Unorm,
                ComponentType::Unorm,
                ComponentType::Unorm,
            ]
        );
        assert_eq!(tic.swizzle, [SwizzleSource::R; 4]);
    }

    #[test]
    fn decodes_g24r8_logical_stencil_and_depth_channels() {
        let rgba = decode_to_rgba8(&0x1234_56abu32.to_le_bytes(), 1, 1, TicFormat::G24R8);
        assert_eq!(rgba, [0xab, 0x12, 0, 0xff]);
    }

    #[test]
    fn decodes_r8g8_as_independent_red_and_green_channels() {
        let rgba = decode_to_rgba8(&[0x21, 0xe3, 0x7f, 0x80], 2, 1, TicFormat::R8G8);
        assert_eq!(rgba, [0x21, 0xe3, 0, 0xff, 0x7f, 0x80, 0, 0xff]);
    }

    #[test]
    fn decodes_bc5_snorm_as_signed_channels() {
        let block = [0xc0, 0x40, 0, 0, 0, 0, 0, 0, 0x20, 0xe0, 0, 0, 0, 0, 0, 0];
        let signed = decode_to_rgba8_typed(&block, 4, 4, TicFormat::BC5, ComponentType::Snorm);
        assert!(signed
            .chunks_exact(4)
            .all(|pixel| pixel == [0xc0, 0x20, 0, 0x7f]));

        let unsigned = decode_to_rgba8(&block, 4, 4, TicFormat::BC5);
        assert!(unsigned
            .chunks_exact(4)
            .all(|pixel| pixel == [0xc0, 0x20, 0, 0xff]));
    }

    #[test]
    fn bc_snorm_selects_mode_before_clamping_endpoints() {
        let block = [0x81, 0x80, 0x1f, 0, 0, 0, 0, 0];
        let signed = decode_to_rgba8_typed(&block, 4, 4, TicFormat::BC4, ComponentType::Snorm);
        assert_eq!(&signed[..8], &[0x81, 0, 0, 0x7f, 0x81, 0, 0, 0x7f]);
    }

    #[test]
    fn parses_maxwell_g8r8_and_maps_video_luma_formats() {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&0x18u32.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        let tic = TicEntry::parse(&raw).unwrap();
        assert_eq!(tic.format, TicFormat::R8G8);
        assert_eq!(tic.format.src_bpp(), 2);
        assert_eq!(TicFormat::from_raw(0x1c), TicFormat::R8);
        assert_eq!(TicFormat::from_raw(0x1d), TicFormat::R8);
    }

    #[test]
    fn parses_pitch_linear_row_stride_and_guest_size() {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&0x1cu32.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        raw[8..12].copy_from_slice(&(1u32 << 21).to_le_bytes());
        raw[12..16].copy_from_slice(&(512u32 >> 5).to_le_bytes());
        raw[16..20].copy_from_slice(&(479u32 | (1 << 23)).to_le_bytes());
        raw[20..24].copy_from_slice(&271u32.to_le_bytes());
        let tic = TicEntry::parse(&raw).unwrap();
        assert!(!tic.is_block_linear);
        assert_eq!(tic.pitch_bytes, 512);
        assert_eq!(tic.pitch_linear_layer_size(), Some(512 * 272));
        assert_eq!(texture_guest_size_bytes(&tic, 1), Some(512 * 272));
    }

    #[test]
    fn maxwell_msaa_modes_decode_to_physical_sample_grids() {
        let expected = [
            (0, 1, 1),
            (1, 2, 1),
            (2, 2, 2),
            (3, 4, 2),
            (4, 4, 2),
            (5, 2, 1),
            (6, 4, 4),
            (8, 2, 2),
            (9, 2, 2),
            (10, 4, 2),
            (11, 4, 2),
        ];
        for (mode, width, height) in expected {
            let grid = maxwell_sample_grid(mode).unwrap();
            assert_eq!(grid, SampleGrid { width, height });
            assert_eq!(grid.count(), width * height);
        }
        for reserved in [7, 12, 13, 14, 15, 16, u32::MAX] {
            assert_eq!(maxwell_sample_grid(reserved), None);
        }
    }

    #[test]
    fn parses_msaa_and_sparse_tic_metadata() {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&0x1cu32.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        raw[8..12].copy_from_slice(&(3u32 << 21).to_le_bytes());
        raw[16..20].copy_from_slice(&(1u32 << 23).to_le_bytes());
        raw[20..24].copy_from_slice(&(1u32 << 30).to_le_bytes());
        raw[28..32].copy_from_slice(&(11u32 << 8).to_le_bytes());

        let tic = TicEntry::parse(&raw).unwrap();
        assert_eq!(tic.msaa_mode, 11);
        assert_eq!(
            tic.sample_grid(),
            Some(SampleGrid {
                width: 4,
                height: 2
            })
        );
        assert_eq!(tic.sample_count(), Some(8));
        assert!(tic.is_sparse);
        assert_eq!(tic.depth, 1);
        assert!(!tic.normalized_coords);
        assert_eq!((tic.res_min_mip_level, tic.res_max_mip_level), (0, 0));
        assert_eq!(tic.physical_mip_extent(0), Some((4, 2)));
        assert_eq!(tic.physical_storage_extent(0), None);
        assert_eq!(tic.physical_linear_size(0), None);
        assert_eq!(block_linear_mip_layout(&tic), None);
        assert_eq!(texture_guest_size_bytes(&tic, 1), None);
    }

    #[test]
    fn msaa_expands_physical_footprint_but_not_logical_extent() {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&0x1cu32.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        raw[8..12].copy_from_slice(&(3u32 << 21).to_le_bytes());
        raw[16..20].copy_from_slice(&(16u32 | (1 << 23)).to_le_bytes());
        raw[20..24].copy_from_slice(&8u32.to_le_bytes());
        raw[28..32].copy_from_slice(&(3u32 << 8).to_le_bytes());

        let tic = TicEntry::parse(&raw).unwrap();
        assert_eq!((tic.width, tic.height), (17, 9));
        assert_eq!(tic.logical_mip_extent(0), (17, 9));
        assert_eq!(tic.physical_mip_extent(0), Some((68, 18)));
        assert_eq!(tic.physical_storage_extent(0), Some((68, 18, 1)));
        assert_eq!(tic.physical_linear_size(0), Some(68 * 18));

        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!((layout.levels[0].width, layout.levels[0].height), (17, 9));
        assert_eq!(
            (
                layout.levels[0].storage_width,
                layout.levels[0].storage_height,
            ),
            (68, 18)
        );
        assert_eq!(layout.levels[0].linear_size, 68 * 18);
        assert_eq!(layout.levels[0].guest_size, 6 * 512);
        assert_eq!(texture_guest_size_bytes(&tic, 1), Some(6 * 512));
    }

    #[test]
    fn msaa_expansion_precedes_compression_block_rounding() {
        let tic = TicEntry {
            format: TicFormat::BC1,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 1,
            width: 7,
            height: 5,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 2,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        assert_eq!(tic.logical_mip_extent(0), (7, 5));
        assert_eq!(tic.physical_mip_extent(0), Some((14, 10)));
        assert_eq!(tic.physical_storage_extent(0), Some((4, 3, 8)));
        assert_eq!(tic.physical_linear_size(0), Some(96));
    }

    #[test]
    fn pitch_linear_rows_unpack_without_padding() {
        let tic = TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 1,
            width: 3,
            height: 2,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 32,
            is_block_linear: false,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let mut guest = vec![0xcc; 64];
        guest[0..3].copy_from_slice(&[1, 2, 3]);
        guest[32..35].copy_from_slice(&[4, 5, 6]);
        let linear = unpack_pitch_linear(&guest, &tic, 1).unwrap();
        assert_eq!(linear, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn parses_r16g16b16a16_float_and_mip_ranges() {
        let mut raw = [0u8; 32];
        let w0: u32 = 0x03 | (7 << 7) | (7 << 10) | (7 << 13) | (7 << 16);
        raw[0..4].copy_from_slice(&w0.to_le_bytes());
        raw[4..8].copy_from_slice(&1u32.to_le_bytes());
        raw[8..12].copy_from_slice(&(3u32 << 21).to_le_bytes());
        raw[12..16].copy_from_slice(&(7u32 << 28).to_le_bytes());
        raw[28..32].copy_from_slice(&(2u32 | (6 << 4)).to_le_bytes());
        let tic = TicEntry::parse(&raw).unwrap();
        assert_eq!(tic.format, TicFormat::R16G16B16A16);
        assert_eq!(tic.format.src_bpp(), 8);
        assert_eq!(tic.component_types, [ComponentType::Float; 4]);
        assert_eq!(tic.mip_levels(), 8);
        assert_eq!(tic.view_base_mip(), 2);
        assert_eq!(tic.view_mip_levels(), 5);
    }

    #[test]
    fn recognizes_bc6h_formats_and_block_sizes() {
        assert_eq!(TicFormat::from_raw(0x10), TicFormat::BC6S);
        assert_eq!(TicFormat::from_raw(0x11), TicFormat::BC6U);
        assert_eq!(TicFormat::BC6U.src_bpp(), 16);
        assert_eq!(TicFormat::BC6U.block_extent(), (4, 4));
        assert_eq!(TicFormat::BC6U.linear_size(4096, 2048), 8 * 1024 * 1024);
    }

    #[test]
    fn maxwell_compressed_and_packed_ids_preserve_channel_order() {
        assert_eq!(TicFormat::from_raw(0x06), TicFormat::Etc2Rgb);
        assert_eq!(TicFormat::from_raw(0x0a), TicFormat::Etc2RgbA1);
        assert_eq!(TicFormat::from_raw(0x0b), TicFormat::Etc2Rgba);
        assert_eq!(TicFormat::Etc2Rgb.storage_extent(7, 5), (2, 2, 8));
        assert_eq!(TicFormat::Etc2RgbA1.storage_extent(7, 5), (2, 2, 8));
        assert_eq!(TicFormat::Etc2Rgba.storage_extent(7, 5), (2, 2, 16));
        assert_eq!(TicFormat::from_raw(0x12), TicFormat::A4B4G4R4);
        assert_eq!(
            decode_to_rgba8(&0x4321u16.to_le_bytes(), 1, 1, TicFormat::A4B4G4R4),
            [0x11, 0x22, 0x33, 0x44]
        );
        assert_eq!(TicFormat::from_raw(0x13), TicFormat::A5B5G5R1);
        assert_eq!(
            decode_to_rgba8(&0xf801u16.to_le_bytes(), 1, 1, TicFormat::A5B5G5R1),
            [0xff, 0, 0, 0xff]
        );
        assert_eq!(TicFormat::from_raw(0x14), TicFormat::A1B5G5R5);
        assert_eq!(
            decode_to_rgba8(&0xfc00u16.to_le_bytes(), 1, 1, TicFormat::A1B5G5R5),
            [0, 0, 0xff, 0xff]
        );
        assert_eq!(TicFormat::from_raw(0x15), TicFormat::B5G6R5);
        assert_eq!(
            decode_to_rgba8(&0x001fu16.to_le_bytes(), 1, 1, TicFormat::B5G6R5),
            [0xff, 0, 0, 0xff]
        );
        assert_eq!(TicFormat::from_raw(0x2d), TicFormat::Unknown(0x2d));
    }

    #[test]
    fn parses_one_d_buffer_width_from_both_descriptor_words() {
        let mut header_only = [0u8; 32];
        header_only[0..4].copy_from_slice(&0x1du32.to_le_bytes());
        header_only[4..8].copy_from_slice(&1u32.to_le_bytes());
        header_only[16..20].copy_from_slice(&(1u32 << 23).to_le_bytes());
        assert!(TicEntry::parse(&header_only).unwrap().is_buffer());

        let slot8 = [
            0x1b, 0x92, 0x14, 0x60, 0x00, 0x00, 0x77, 0x03, 0x04, 0x00, 0x00, 0x00, 0x0b, 0x00,
            0x00, 0x00, 0xff, 0xb7, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let tic = TicEntry::parse(&slot8).unwrap();
        assert!(tic.is_buffer());
        assert_eq!(tic.texture_type, 6);
        assert_eq!(tic.format, TicFormat::R16);
        assert_eq!(tic.component_types, [ComponentType::Uint; 4]);
        assert_eq!(
            tic.swizzle,
            [
                SwizzleSource::R,
                SwizzleSource::Zero,
                SwizzleSource::Zero,
                SwizzleSource::One,
            ]
        );
        assert_eq!(tic.gpu_va, 0x403770000);
        assert_eq!(tic.width, 768_000);
        assert_eq!(tic.mip_levels(), 1);
        assert_eq!(tic.format.linear_size(tic.width, 1), 1_536_000);

        let slot9 = [
            0x0f, 0x92, 0x14, 0x60, 0x00, 0x00, 0x6d, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x7f, 0xbb, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let tic = TicEntry::parse(&slot9).unwrap();
        assert!(tic.is_buffer());
        assert_eq!(tic.format, TicFormat::R32);
        assert_eq!(tic.gpu_va, 0x4056d0000);
        assert_eq!(tic.width, 48_000);
        assert_eq!(tic.format.linear_size(tic.width, 1), 192_000);
    }

    #[test]
    fn decodes_r16g16b16a16_float_for_diagnostics() {
        let raw = [0x00, 0x00, 0x00, 0x38, 0x00, 0x3c, 0x00, 0x3c];
        let rgba = decode_to_rgba8(&raw, 1, 1, TicFormat::R16G16B16A16);
        assert_eq!(rgba, [0, 128, 255, 255]);
    }

    #[test]
    fn r16g16b16a16_cube_mips_match_maxwell_layer_stride() {
        let tic = TicEntry {
            format: TicFormat::R16G16B16A16,
            component_types: [ComponentType::Float; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x5a4fd0000,
            width: 128,
            height: 128,
            block_width_log2: 0,
            block_height_log2: 4,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 3,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 7,
            res_min_mip_level: 0,
            res_max_mip_level: 7,
        };
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!(
            layout
                .levels
                .iter()
                .map(|level| level.guest_size)
                .collect::<Vec<_>>(),
            [0x20000, 0x8000, 0x2000, 0x800, 0x200, 0x200, 0x200, 0x200]
        );
        assert_eq!(layout.layer_size, 0x2b000);
        assert_eq!(layout.layer_stride, 0x2c000);
        assert_eq!(texture_guest_size_bytes(&tic, 6), Some(0x108000));
        assert_eq!(tic.layer_stride_bytes(), Some(0x2c000));

        let layer_view = TicEntry {
            gpu_va: tic.gpu_va + 4 * 0x2c000,
            base_layer: 4,
            ..tic
        };
        assert_eq!(layer_view.backing_gpu_va(), Some(tic.gpu_va));
    }

    #[test]
    fn pitch_linear_base_layer_uses_guest_row_stride() {
        let tic = TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 0x20_6000,
            width: 3,
            height: 2,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0x1000,
            is_block_linear: false,
            texture_type: 5,
            depth: 2,
            base_layer: 3,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        assert_eq!(tic.layer_stride_bytes(), Some(0x2000));
        assert_eq!(tic.backing_gpu_va(), Some(0x20_0000));
    }

    #[test]
    fn tile_width_spacing_controls_large_mip_row_stride() {
        let tic = TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 1,
            width: 1025,
            height: 9,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 4,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 1,
            res_min_mip_level: 0,
            res_max_mip_level: 1,
        };
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!(layout.levels[0].stride_alignment_log2, 10);
        assert_eq!(layout.levels[1].stride_alignment_log2, 6);

        let mut guest = vec![0u8; 12_289];
        guest[12_288] = 0xa5;
        let linear = unswizzle_block_linear_strided(&guest, 513, 9, 1, 0, 10);
        assert_eq!(linear[8 * 513 + 512], 0xa5);
    }

    #[test]
    fn strided_block_linear_mip_swizzle_round_trips() {
        let linear: Vec<u8> = (0..64usize * 32 * 2)
            .map(|index| index.wrapping_mul(37) as u8)
            .collect();
        let guest = swizzle_block_linear_strided(&linear, 64, 32, 2, 2, 5);
        assert_eq!(guest.len(), 0x1000);
        assert_eq!(
            unswizzle_block_linear_strided(&guest, 64, 32, 2, 2, 5),
            linear
        );
    }

    #[test]
    fn single_mip_preserves_maxwell_base_block_height() {
        let tic = TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 1,
            width: 1,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 4,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!(layout.levels[0].block_height_log2, 4);
        assert_eq!(layout.levels[0].guest_size, 0x2000);
        assert_eq!(texture_guest_size_bytes(&tic, 1), Some(0x2000));
    }

    #[test]
    fn block_linear_3d_round_trip_preserves_tight_texels() {
        let (width, height, depth, bpp) = (13, 11, 5, 4usize);
        let linear: Vec<u8> = (0..width * height * depth * bpp as u32)
            .map(|index| index.wrapping_mul(37).wrapping_add(11) as u8)
            .collect();
        let tiled = swizzle_block_linear_3d(&linear, width, height, depth, bpp, 1, 1, 2);
        assert_eq!(
            tiled.len(),
            block_linear_byte_size_3d(width, height, depth, bpp, 1, 1, 2)
        );
        assert_eq!(
            unswizzle_block_linear_3d(&tiled, width, height, depth, bpp, 1, 1, 2),
            linear
        );
    }

    #[test]
    fn block_width_padding_expands_guest_footprint() {
        let unpadded = block_linear_byte_size_3d_with_block_width(17, 8, 1, 4, 0, 0, 0, 0);
        let padded = block_linear_byte_size_3d_with_block_width(17, 8, 1, 4, 2, 0, 0, 0);

        assert_eq!(unpadded, 1024);
        assert_eq!(padded, 2048);
    }

    #[test]
    fn native_render_target_unpack_handles_tight_and_pitch_layouts() {
        let tight: Vec<u8> = (0..24).collect();
        assert_eq!(
            unpack_native_render_target(&tight, 3, 2, 2, 2, 0),
            Some(tight.clone())
        );

        let mut pitched = vec![0xcc; 32];
        pitched[0..6].copy_from_slice(&tight[0..6]);
        pitched[8..14].copy_from_slice(&tight[6..12]);
        pitched[16..22].copy_from_slice(&tight[12..18]);
        pitched[24..30].copy_from_slice(&tight[18..24]);
        assert_eq!(
            unpack_native_render_target(&pitched, 3, 2, 2, 2, 2 | (8 << 8)),
            Some(tight)
        );
    }

    #[test]
    fn native_render_target_unpack_decodes_block_linear_3d_layout() {
        let (width, height, depth, bpp) = (13, 11, 5, 4usize);
        let linear: Vec<u8> = (0..width * height * depth * bpp as u32)
            .map(|index| index.wrapping_mul(29).wrapping_add(7) as u8)
            .collect();
        let guest = swizzle_block_linear_3d(&linear, width, height, depth, bpp, 1, 1, 2);
        let signature = 1 | (1 << 16) | (1 << 24) | (2 << 32);

        assert_eq!(
            unpack_native_render_target(&guest, width, height, depth, bpp, signature),
            Some(linear)
        );
    }

    #[test]
    fn block_width_aware_unswizzle_uses_padded_block_rows() {
        let (width, height, depth, bpp) = (17, 9, 1, 4usize);
        let guest_size =
            block_linear_byte_size_3d_with_block_width(width, height, depth, bpp, 2, 0, 0, 0);
        let mut guest = vec![0u8; guest_size];
        let marker = [0x12, 0x34, 0x56, 0x78];
        guest[0..4].copy_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
        guest[0xa00..0xa04].copy_from_slice(&marker);

        let linear = unswizzle_block_linear_3d_with_block_width(
            &guest, width, height, depth, bpp, 2, 0, 0, 0,
        );
        assert_eq!(&linear[0..4], &[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(&linear[0x260..0x264], &marker);
        assert_eq!(
            unpack_native_render_target(&guest, width, height, depth, bpp, 1 | (2 << 8)),
            Some(linear)
        );
    }

    #[test]
    fn native_render_target_unpack_rejects_invalid_or_truncated_layouts() {
        assert_eq!(unpack_native_render_target(&[0; 7], 2, 1, 1, 4, 0), None);
        assert_eq!(
            unpack_native_render_target(&[0; 8], 2, 1, 1, 4, 2 | (4 << 8)),
            None
        );
        assert_eq!(unpack_native_render_target(&[0; 8], 2, 1, 1, 4, 3), None);
        assert_eq!(
            unpack_native_render_target(&[0; 8], 2, 1, 1, 4, 1 | (1 << 40)),
            None
        );
        assert_eq!(
            unpack_native_render_target(&[], u32::MAX, u32::MAX, u32::MAX, usize::MAX, 0),
            None
        );

        let required = block_linear_byte_size_3d_with_block_width(17, 9, 1, 4, 2, 0, 0, 0);
        let truncated = vec![0u8; required - 1];
        assert_eq!(
            unpack_native_render_target(&truncated, 17, 9, 1, 4, 1 | (2 << 8)),
            None
        );
    }

    #[test]
    fn block_linear_3d_large_level_uses_maxwell_tile_spacing_stride() {
        let (width, height, depth, bpp) = (1025, 17, 2, 4usize);
        let mut linear = vec![0u8; width as usize * height as usize * depth as usize * bpp];
        let tight_offset = 0x22084;
        let marker = [0x12, 0x34, 0x56, 0x78];
        linear[tight_offset..tight_offset + marker.len()].copy_from_slice(&marker);

        let tiled = swizzle_block_linear_3d(&linear, width, height, depth, bpp, 1, 1, 4);
        assert_eq!(tiled.len(), 0x50000);
        assert_eq!(&tiled[0x48400..0x48404], &marker);
    }
}
