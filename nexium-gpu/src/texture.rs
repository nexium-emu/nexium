#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TicFormat {
    A8B8G8R8,
    R8G8B8A8,
    R5G6B5,
    A1R5G5B5,
    A4R4G4B4,
    R8,
    R8G8,
    R16,
    Unknown(u32),
}

impl TicFormat {
    pub fn from_raw(format_word: u32) -> Self {
        match format_word & 0x7F {
            0x08 => TicFormat::A8B8G8R8,
            0x09 => TicFormat::R5G6B5,
            0x0A => TicFormat::A1R5G5B5,
            0x0B => TicFormat::A4R4G4B4,
            0x12 => TicFormat::R16,
            0x1B => TicFormat::R16,
            0x1C => TicFormat::R8G8,
            0x1D => TicFormat::R8,
            0x2D => TicFormat::R8G8B8A8,
            other => TicFormat::Unknown(other),
        }
    }

    pub fn src_bpp(&self) -> usize {
        match self {
            TicFormat::A8B8G8R8 | TicFormat::R8G8B8A8 => 4,
            TicFormat::R5G6B5 | TicFormat::A1R5G5B5 | TicFormat::A4R4G4B4 => 2,
            TicFormat::R16 | TicFormat::R8G8 => 2,
            TicFormat::R8 => 1,
            TicFormat::Unknown(_) => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TicEntry {
    pub format: TicFormat,
    pub gpu_va: u64,
    pub width: u32,
    pub height: u32,
    pub block_height_log2: u32,
    pub is_block_linear: bool,
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

        let addr_lo = w1 as u64;
        let addr_hi = (w2 & 0xFFFF) as u64;
        let gpu_va = (addr_hi << 32) | addr_lo;

        let header_version = (w2 >> 21) & 0x7;
        let is_block_linear = header_version == 3;

        let block_height_log2 = if is_block_linear {
            (w3 >> 3) & 0x7
        } else {
            0
        };

        let width = (w4 & 0xFFFF) + 1;
        let w5 = u32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]);
        let height = (w5 & 0xFFFF) + 1;

        if gpu_va == 0 || width == 0 || height == 0 || width > 16384 || height > 16384 {
            return None;
        }

        Some(TicEntry {
            format,
            gpu_va,
            width,
            height,
            block_height_log2,
            is_block_linear,
        })
    }
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
    Nearest,
    Linear,
}
impl TexFilter {
    pub fn from_raw(v: u32) -> Self {
        if (v & 0x3) == 2 {
            TexFilter::Linear
        } else {
            TexFilter::Nearest
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TscEntry {
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub wrap_p: WrapMode,
    pub mag_filter: TexFilter,
    pub min_filter: TexFilter,
    pub mip_filter: TexFilter,
}
impl TscEntry {
    pub fn parse(raw: &[u8]) -> Option<TscEntry> {
        if raw.len() < 32 {
            return None;
        }
        let w0 = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let w1 = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        Some(TscEntry {
            wrap_u: WrapMode::from_raw(w0),
            wrap_v: WrapMode::from_raw(w0 >> 3),
            wrap_p: WrapMode::from_raw(w0 >> 6),
            mag_filter: TexFilter::from_raw(w1),
            min_filter: TexFilter::from_raw(w1 >> 4),
            mip_filter: TexFilter::from_raw(w1 >> 6),
        })
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
    let width = width_px as usize;
    let height = height_px as usize;
    let dst_stride = width * bpp;
    let mut dst = vec![0u8; dst_stride * height];

    let width_bytes = width * bpp;
    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
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
            let gob_offset = block_row_offset
                + gob_col * block_height * GOB_SIZE
                + gob_row_in_block * GOB_SIZE;
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

pub fn decode_to_rgba8(
    src: &[u8],
    width: u32,
    height: u32,
    format: TicFormat,
) -> Vec<u8> {
    let pixels = (width as usize) * (height as usize);
    let mut out = vec![0u8; pixels * 4];

    match format {
        TicFormat::A8B8G8R8 => {
            let n = pixels.min(src.len() / 4);
            for i in 0..n {
                let off = i * 4;
                out[off]     = src[off];
                out[off + 1] = src[off + 1];
                out[off + 2] = src[off + 2];
                out[off + 3] = src[off + 3];
            }
        }
        TicFormat::R8G8B8A8 => {
            let n = (pixels * 4).min(src.len());
            out[..n].copy_from_slice(&src[..n]);
        }
        TicFormat::R5G6B5 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let r = ((v >> 11) & 0x1F) as u8;
                let g = ((v >> 5)  & 0x3F) as u8;
                let b = (v         & 0x1F) as u8;
                out[i * 4    ] = (r << 3) | (r >> 2);
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
                let g = ((v >> 5)  & 0x1F) as u8;
                let b = (v         & 0x1F) as u8;
                out[i * 4    ] = (r << 3) | (r >> 2);
                out[i * 4 + 1] = (g << 3) | (g >> 2);
                out[i * 4 + 2] = (b << 3) | (b >> 2);
                out[i * 4 + 3] = a;
            }
        }
        TicFormat::A4R4G4B4 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let a = ((v >> 12) & 0xF) as u8;
                let r = ((v >> 8)  & 0xF) as u8;
                let g = ((v >> 4)  & 0xF) as u8;
                let b = (v         & 0xF) as u8;
                out[i * 4    ] = (r << 4) | r;
                out[i * 4 + 1] = (g << 4) | g;
                out[i * 4 + 2] = (b << 4) | b;
                out[i * 4 + 3] = (a << 4) | a;
            }
        }
        TicFormat::R8 => {
            for i in 0..pixels.min(src.len()) {
                let v = src[i];
                out[i * 4    ] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::R8G8 => {
            for i in 0..pixels.min(src.len() / 2) {
                let intensity = src[i * 2];
                let alpha = src[i * 2 + 1];
                out[i * 4    ] = intensity;
                out[i * 4 + 1] = intensity;
                out[i * 4 + 2] = intensity;
                out[i * 4 + 3] = alpha;
            }
        }
        TicFormat::R16 => {
            for i in 0..pixels.min(src.len() / 2) {
                let v = src[i * 2 + 1];
                out[i * 4    ] = v;
                out[i * 4 + 1] = v;
                out[i * 4 + 2] = v;
                out[i * 4 + 3] = 0xFF;
            }
        }
        TicFormat::Unknown(_) => {
            for i in 0..pixels {
                out[i * 4    ] = 0xFF;
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
            let (mut mr, mut mg, mut mb) = (0u8, 0u8, 0u8);
            for px in out.chunks_exact(4) {
                mr = mr.max(px[0]);
                mg = mg.max(px[1]);
                mb = mb.max(px[2]);
            }
            log::warn!(
                "[texdecode] #{} {:?} {}x{} maxR={} maxG={} maxB={}",
                k, format, width, height, mr, mg, mb
            );
        }
    }

    out
}
