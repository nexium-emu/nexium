use super::super::GpuMappings;
use crate::QueuedFrame;
use std::sync::{Arc, Mutex};

pub const FERMI_2D_CLASS: u32 = 0x902D;

const FMT_A8R8G8B8: u32 = 0xCF;
const FMT_A8B8G8R8: u32 = 0xD5;
const FMT_X8R8G8B8: u32 = 0xE6;
const FMT_X8B8G8R8: u32 = 0xCB;

const DST_BASE: u32 = 0x80;
const SRC_BASE: u32 = 0x9C;
const SURFACE_FORMAT: u32 = 0x0;
const SURFACE_MEMORY_LAYOUT: u32 = 0x1;
const SURFACE_BLOCK_SIZE: u32 = 0x2;
const SURFACE_DEPTH: u32 = 0x3;
const SURFACE_LAYER: u32 = 0x4;
const SURFACE_PITCH: u32 = 0x5;
const SURFACE_WIDTH: u32 = 0x6;
const SURFACE_HEIGHT: u32 = 0x7;
const SURFACE_OFFSET_HIGH: u32 = 0x8;
const SURFACE_OFFSET_LOW: u32 = 0x9;

const M_OPERATION: u32 = 0x222;
const M_PIXELS_DST_X0: u32 = 0x22C;
const M_PIXELS_DST_Y0: u32 = 0x22D;
const M_PIXELS_DST_WIDTH: u32 = 0x22E;
const M_PIXELS_DST_HEIGHT: u32 = 0x22F;
const M_PIXELS_DU_DX_LOW: u32 = 0x230;
const M_PIXELS_DU_DX_HIGH: u32 = 0x231;
const M_PIXELS_DV_DY_LOW: u32 = 0x232;
const M_PIXELS_DV_DY_HIGH: u32 = 0x233;
const M_PIXELS_SRC_X0_LOW: u32 = 0x234;
const M_PIXELS_SRC_X0_HIGH: u32 = 0x235;
const M_PIXELS_SRC_Y0_LOW: u32 = 0x236;
const M_PIXELS_SRC_Y0_HIGH: u32 = 0x237;

const MEMORY_LAYOUT_BLOCK_LINEAR: u32 = 0;
const MEMORY_LAYOUT_PITCH: u32 = 1;

#[derive(Default, Clone, Copy)]
struct Surface {
    format: u32,
    memory_layout: u32,
    block_size: u32,
    depth: u32,
    _layer: u32,
    pitch: u32,
    width: u32,
    height: u32,
    offset_high: u32,
    offset_low: u32,
}

impl Surface {
    fn gpu_va(&self) -> u64 {
        ((self.offset_high as u64) << 32) | self.offset_low as u64
    }
    fn block_height_log2(&self) -> u32 {
        (self.block_size >> 4) & 0xF
    }
    fn bytes_per_pixel(&self) -> usize {
        match self.format {
            0xCF | 0xCB | 0xCA | 0xC3 | 0xC1 | 0xC0 => 4,
            0xE5 | 0xE6 => 2,
            0x1D => 1,
            _ => 4,
        }
    }
}

#[derive(Default)]
pub struct Fermi2D {
    dst: Surface,
    src: Surface,
    operation: u32,
    dst_x0: i32,
    dst_y0: i32,
    dst_width_blit: i32,
    dst_height_blit: i32,
    du_dx_low: u32,
    du_dx_high: u32,
    dv_dy_low: u32,
    dv_dy_high: u32,
    src_x0_low: u32,
    src_x0_high: u32,
    src_y0_low: u32,
    pub blit_count: u64,
    pub captured_frames: Arc<Mutex<Vec<QueuedFrame>>>,
}

impl Fermi2D {
    pub fn new() -> Self { Self::default() }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if (DST_BASE..DST_BASE + 16).contains(&method) {
            Self::write_surface(&mut self.dst, method - DST_BASE, arg);
            return;
        }
        if (SRC_BASE..SRC_BASE + 16).contains(&method) {
            Self::write_surface(&mut self.src, method - SRC_BASE, arg);
            return;
        }
        match method {
            M_OPERATION => self.operation = arg,
            M_PIXELS_DST_X0 => self.dst_x0 = arg as i32,
            M_PIXELS_DST_Y0 => self.dst_y0 = arg as i32,
            M_PIXELS_DST_WIDTH => self.dst_width_blit = arg as i32,
            M_PIXELS_DST_HEIGHT => self.dst_height_blit = arg as i32,
            M_PIXELS_DU_DX_LOW => self.du_dx_low = arg,
            M_PIXELS_DU_DX_HIGH => self.du_dx_high = arg,
            M_PIXELS_DV_DY_LOW => self.dv_dy_low = arg,
            M_PIXELS_DV_DY_HIGH => self.dv_dy_high = arg,
            M_PIXELS_SRC_X0_LOW => self.src_x0_low = arg,
            M_PIXELS_SRC_X0_HIGH => self.src_x0_high = arg,
            M_PIXELS_SRC_Y0_LOW => self.src_y0_low = arg,
            M_PIXELS_SRC_Y0_HIGH => {
                self.execute_blit(arg, mappings, mem_read, mem_write);
            }
            _ => {
                log::trace!("Fermi2D: unhandled method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn write_surface(s: &mut Surface, off: u32, arg: u32) {
        match off {
            SURFACE_FORMAT => s.format = arg,
            SURFACE_MEMORY_LAYOUT => s.memory_layout = arg,
            SURFACE_BLOCK_SIZE => s.block_size = arg,
            SURFACE_DEPTH => s.depth = arg,
            SURFACE_LAYER => s._layer = arg,
            SURFACE_PITCH => s.pitch = arg,
            SURFACE_WIDTH => s.width = arg,
            SURFACE_HEIGHT => s.height = arg,
            SURFACE_OFFSET_HIGH => s.offset_high = arg,
            SURFACE_OFFSET_LOW => s.offset_low = arg,
            _ => {}
        }
    }

    fn execute_blit(
        &mut self,
        src_y0_high: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let _ = src_y0_high;
        let src_va = self.src.gpu_va();
        let dst_va = self.dst.gpu_va();
        let Some(src_cpu) = mappings.cpu_address_for(src_va) else {
            log::trace!("Fermi2D: src gpu_va {:#x} not mapped", src_va);
            return;
        };
        let Some(dst_cpu) = mappings.cpu_address_for(dst_va) else {
            log::trace!("Fermi2D: dst gpu_va {:#x} not mapped", dst_va);
            return;
        };

        let bpp = self.dst.bytes_per_pixel().max(self.src.bytes_per_pixel());
        let dst_x0 = self.dst_x0.max(0) as usize;
        let dst_y0 = self.dst_y0.max(0) as usize;
        let blit_w = self.dst_width_blit.max(0) as usize;
        let blit_h = self.dst_height_blit.max(0) as usize;
        if blit_w == 0 || blit_h == 0 { return; }

        let src_layout = self.src.memory_layout;
        let dst_layout = self.dst.memory_layout;

        match (src_layout, dst_layout) {
            (MEMORY_LAYOUT_PITCH, MEMORY_LAYOUT_BLOCK_LINEAR) => {
                self.blit_pitch_to_block(src_cpu, dst_cpu, bpp, dst_x0, dst_y0, blit_w, blit_h, mem_read, mem_write);
            }
            (MEMORY_LAYOUT_BLOCK_LINEAR, MEMORY_LAYOUT_PITCH) => {
                self.blit_block_to_pitch(src_cpu, dst_cpu, bpp, dst_x0, dst_y0, blit_w, blit_h, mem_read, mem_write);
            }
            (MEMORY_LAYOUT_PITCH, MEMORY_LAYOUT_PITCH) => {
                self.blit_pitch_to_pitch(src_cpu, dst_cpu, bpp, dst_x0, dst_y0, blit_w, blit_h, mem_read, mem_write);
            }
            (MEMORY_LAYOUT_BLOCK_LINEAR, MEMORY_LAYOUT_BLOCK_LINEAR) => {
                log::trace!("Fermi2D: block→block blit not supported (skipping)");
            }
            _ => {
                log::trace!("Fermi2D: unknown layout combo src={} dst={}", src_layout, dst_layout);
            }
        }
        self.blit_count = self.blit_count.wrapping_add(1);

        let framebuffer_like = self.dst.memory_layout == MEMORY_LAYOUT_PITCH
            && self.dst.width >= 320
            && self.dst.height >= 240
            && self.dst.pitch >= self.dst.width * (bpp as u32)
            && matches!(self.dst.format, FMT_A8R8G8B8 | FMT_A8B8G8R8 | FMT_X8R8G8B8 | FMT_X8B8G8R8);
        if framebuffer_like {
            self.try_publish_frame(dst_cpu, bpp, mem_read);
        }
    }

    fn try_publish_frame(
        &self,
        dst_cpu: u64,
        bpp: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) {
        let w = self.dst.width as usize;
        let h = self.dst.height as usize;
        let row_bytes = w * bpp;
        let total = h * row_bytes;
        if total == 0 { return; }
        let mut pixels = vec![0u8; total];
        let pitch = self.dst.pitch as usize;
        for y in 0..h {
            let g_off = dst_cpu + (y * pitch) as u64;
            let p_off = y * row_bytes;
            if !mem_read(g_off, &mut pixels[p_off..p_off + row_bytes]) {
                return;
            }
        }
        let rgba = match self.dst.format {
            FMT_A8B8G8R8 | FMT_X8B8G8R8 => pixels,
            FMT_A8R8G8B8 | FMT_X8R8G8B8 => {
                let mut out = pixels;
                for px in out.chunks_exact_mut(4) {
                    px.swap(0, 2);
                }
                out
            }
            _ => return,
        };
        let mut rgba = rgba;
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
        let rgb_nz = rgba.chunks_exact(4).filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0).count();
        log::info!(
            "Fermi2D::publish_frame dst_cpu={:#x} {}x{} pitch={} fmt={:#x} rgb_nz={}",
            dst_cpu, w, h, pitch, self.dst.format, rgb_nz
        );
        let mut q = self.captured_frames.lock().unwrap();
        if q.len() >= 2 { q.remove(0); }
        q.push(QueuedFrame { width: w as u32, height: h as u32, pixels: rgba });
    }

    fn blit_pitch_to_pitch(
        &self,
        src_cpu: u64,
        dst_cpu: u64,
        bpp: usize,
        dst_x0: usize,
        dst_y0: usize,
        w: usize,
        h: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let row_bytes = w * bpp;
        let src_pitch = self.src.pitch.max(row_bytes as u32) as usize;
        let dst_pitch = self.dst.pitch.max((dst_x0 as u32 + w as u32).saturating_mul(bpp as u32)) as usize;
        let mut row = vec![0u8; row_bytes];
        for y in 0..h {
            let src_off = src_cpu + (y * src_pitch) as u64;
            let dst_off = dst_cpu + ((dst_y0 + y) * dst_pitch + dst_x0 * bpp) as u64;
            if !mem_read(src_off, &mut row) { break; }
            mem_write(dst_off, &row);
        }
    }

    fn blit_pitch_to_block(
        &self,
        src_cpu: u64,
        dst_cpu: u64,
        bpp: usize,
        dst_x0: usize,
        dst_y0: usize,
        w: usize,
        h: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_pitch = self.src.pitch.max((w * bpp) as u32) as usize;
        let block_height_log2 = self.dst.block_height_log2();
        let dst_width_bytes = (self.dst.width as usize) * bpp;
        let dst_height = self.dst.height as usize;
        let tiled_size = tiled_size_bytes(dst_width_bytes, dst_height, block_height_log2);
        let mut tiled = vec![0u8; tiled_size];
        mem_read(dst_cpu, &mut tiled);
        let mut row = vec![0u8; w * bpp];
        for y in 0..h {
            let src_off = src_cpu + (y * src_pitch) as u64;
            if !mem_read(src_off, &mut row) { break; }
            let dst_y = dst_y0 + y;
            for x in 0..w {
                let dst_byte_x = (dst_x0 + x) * bpp;
                let off = block_linear_offset(dst_byte_x, dst_y, dst_width_bytes, block_height_log2);
                if off + bpp <= tiled.len() {
                    tiled[off..off + bpp].copy_from_slice(&row[x * bpp..x * bpp + bpp]);
                }
            }
        }
        mem_write(dst_cpu, &tiled);
    }

    fn blit_block_to_pitch(
        &self,
        src_cpu: u64,
        dst_cpu: u64,
        bpp: usize,
        dst_x0: usize,
        dst_y0: usize,
        w: usize,
        h: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let block_height_log2 = self.src.block_height_log2();
        let src_width_bytes = (self.src.width as usize) * bpp;
        let src_height = self.src.height as usize;
        let tiled_size = tiled_size_bytes(src_width_bytes, src_height, block_height_log2);
        let mut tiled = vec![0u8; tiled_size];
        mem_read(src_cpu, &mut tiled);
        let dst_pitch = self.dst.pitch.max(((dst_x0 + w) * bpp) as u32) as usize;
        let mut row = vec![0u8; w * bpp];
        for y in 0..h {
            for x in 0..w {
                let src_byte_x = x * bpp;
                let off = block_linear_offset(src_byte_x, y, src_width_bytes, block_height_log2);
                if off + bpp <= tiled.len() {
                    row[x * bpp..x * bpp + bpp].copy_from_slice(&tiled[off..off + bpp]);
                }
            }
            let dst_off = dst_cpu + ((dst_y0 + y) * dst_pitch + dst_x0 * bpp) as u64;
            mem_write(dst_off, &row);
        }
    }
}

const GOB_W: usize = 64;
const GOB_H: usize = 8;
const GOB_SIZE: usize = 512;

fn tiled_size_bytes(width_bytes: usize, height: usize, block_height_log2: u32) -> usize {
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_rows = (height + rows_per_block - 1) / rows_per_block;
    gobs_per_row * block_rows * block_height * GOB_SIZE
}

fn block_linear_offset(byte_x: usize, y: usize, width_bytes: usize, block_height_log2: u32) -> usize {
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_row_stride = gobs_per_row * block_height * GOB_SIZE;
    let block_y = y / rows_per_block;
    let y_in_block = y - block_y * rows_per_block;
    let gob_row_in_block = y_in_block / GOB_H;
    let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
    let gob_col = byte_x / GOB_W;
    let x_in_gob = byte_x - gob_col * GOB_W;
    let in_gob = (x_in_gob & 0x0F)
        | ((y_in_gob & 0x01) << 4)
        | ((x_in_gob & 0x30) << 1)
        | ((y_in_gob & 0x06) << 6);
    block_y * block_row_stride + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE + in_gob
}
