#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TicFormat {
    A8B8G8R8,
    R8G8B8A8,
    R5G6B5,
    A1R5G5B5,
    A4R4G4B4,
    R8,
    R8G8,
    R16,
    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC7,
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
            0x08 => TicFormat::A8B8G8R8,
            0x09 => TicFormat::R5G6B5,
            0x0A => TicFormat::A1R5G5B5,
            0x0B => TicFormat::A4R4G4B4,
            0x12 => TicFormat::R16,
            0x1B => TicFormat::R16,
            0x1C => TicFormat::R8G8,
            0x1D => TicFormat::R8,
            0x24 => TicFormat::BC1,
            0x25 => TicFormat::BC2,
            0x26 => TicFormat::BC3,
            0x27 => TicFormat::BC4,
            0x28 => TicFormat::BC5,
            0x17 => TicFormat::BC7,
            0x2D => TicFormat::R8G8B8A8,
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
            TicFormat::A8B8G8R8 | TicFormat::R8G8B8A8 => 4,
            TicFormat::R5G6B5 | TicFormat::A1R5G5B5 | TicFormat::A4R4G4B4 => 2,
            TicFormat::R16 | TicFormat::R8G8 => 2,
            TicFormat::R8 => 1,
            TicFormat::BC1 | TicFormat::BC4 => 8,
            TicFormat::BC2 | TicFormat::BC3 | TicFormat::BC5 | TicFormat::BC7 => 16,
            TicFormat::Astc(_, _) => 16,
            TicFormat::Unknown(_) => 4,
        }
    }

    pub fn storage_extent(&self, width: u32, height: u32) -> (u32, u32, usize) {
        match self {
            TicFormat::BC1
            | TicFormat::BC2
            | TicFormat::BC3
            | TicFormat::BC4
            | TicFormat::BC5
            | TicFormat::BC7 => ((width + 3) / 4, (height + 3) / 4, self.src_bpp()),
            TicFormat::Astc(bw, bh) => {
                let bw = *bw as u32;
                let bh = *bh as u32;
                ((width + bw - 1) / bw, (height + bh - 1) / bh, self.src_bpp())
            }
            _ => (width, height, self.src_bpp()),
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

#[derive(Clone, Copy, Debug)]
pub struct TicEntry {
    pub format: TicFormat,
    pub swizzle: [SwizzleSource; 4],
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
        let is_block_linear = header_version == 3;

        let block_height_log2 = if is_block_linear { (w3 >> 3) & 0x7 } else { 0 };

        let width = (w4 & 0xFFFF) + 1;
        let w5 = u32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]);
        let height = (w5 & 0xFFFF) + 1;

        if gpu_va == 0 || width == 0 || height == 0 || width > 16384 || height > 16384 {
            return None;
        }

        Some(TicEntry {
            format,
            swizzle,
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

fn decode_bc7(src: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let (w, h) = (width as usize, height as usize);
    let mut buf = vec![0u32; w * h];
    if texture2ddecoder::decode_bc7(src, w, h, &mut buf).is_err() {
        fill_magenta(out);
        return;
    }
    unpack_bcn_u32(&buf, out);
}

pub fn decode_to_rgba8(src: &[u8], width: u32, height: u32, format: TicFormat) -> Vec<u8> {
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
                let intensity = src[i * 2];
                let alpha = src[i * 2 + 1];
                out[i * 4] = intensity;
                out[i * 4 + 1] = intensity;
                out[i * 4 + 2] = intensity;
                out[i * 4 + 3] = alpha;
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
        TicFormat::BC1 => decode_bc1(src, width, height, &mut out),
        TicFormat::BC2 => decode_bc2(src, width, height, &mut out),
        TicFormat::BC3 => decode_bc3(src, width, height, &mut out),
        TicFormat::BC4 => decode_bc4(src, width, height, &mut out),
        TicFormat::BC5 => decode_bc5(src, width, height, &mut out),
        TicFormat::BC7 => decode_bc7(src, width, height, &mut out),
        TicFormat::Astc(bw, bh) => decode_astc(src, width, height, bw as usize, bh as usize, &mut out),
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
