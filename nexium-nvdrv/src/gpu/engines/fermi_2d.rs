use super::super::GpuMappings;
use crate::gpu::formats::{surface_bytes_per_pixel, SurfaceFormat};
use crate::QueuedFrame;
use std::sync::{Arc, Mutex};

pub const FERMI_2D_CLASS: u32 = 0x902D;

const DST_BASE: u32 = 0x80;
const SRC_BASE: u32 = 0x8C;
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

const M_OPERATION: u32 = 0xAB;
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
        ((self.block_size >> 4) & 0xF).min(5)
    }
    fn block_width_log2(&self) -> u32 {
        (self.block_size & 0xF).min(5)
    }
    fn format_info(&self) -> Option<SurfaceFormat> {
        SurfaceFormat::from_raw(self.format)
    }
    fn bytes_per_pixel(&self) -> usize {
        surface_bytes_per_pixel(self.format)
    }
    fn pitch(&self) -> usize {
        self.pitch
            .max((self.width as usize).saturating_mul(self.bytes_per_pixel()) as u32)
            as usize
    }
    fn pixel_offset(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.width as usize || y >= self.height as usize {
            return None;
        }
        let bpp = self.bytes_per_pixel();
        if self.memory_layout == MEMORY_LAYOUT_PITCH {
            Some(y * self.pitch() + x * bpp)
        } else if self.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR {
            Some(block_linear_offset(
                x * bpp,
                y,
                self.width as usize * bpp,
                self.block_width_log2(),
                self.block_height_log2(),
            ))
        } else {
            None
        }
    }
    fn storage_size(&self) -> usize {
        if self.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR {
            tiled_size_bytes(
                self.width as usize * self.bytes_per_pixel(),
                self.height as usize,
                self.block_width_log2(),
                self.block_height_log2(),
            )
        } else {
            self.pitch().saturating_mul(self.height as usize)
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
    src_y0_high: u32,
    pub blit_count: u64,
    pub captured_frames: Arc<Mutex<Vec<QueuedFrame>>>,
}

impl Fermi2D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if (DST_BASE..DST_BASE + 10).contains(&method) {
            Self::write_surface(&mut self.dst, method - DST_BASE, arg);
            return;
        }
        if (SRC_BASE..SRC_BASE + 10).contains(&method) {
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
                self.src_y0_high = arg;
                self.execute_blit(arg, mappings, renderer, mem_read, mem_write);
            }
            _ => {
                log::trace!("Fermi2D: unhandled method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn try_rt_resolve(
        &self,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mappings: &GpuMappings,
        src_va: u64,
        dst_va: u64,
        width: usize,
        height: usize,
        dst_x_step: i64,
        dst_y_step: i64,
        src_x0: i64,
        src_y0: i64,
        du_dx: i64,
        dv_dy: i64,
    ) -> bool {
        let Some(renderer) = renderer else {
            return false;
        };
        if std::env::var_os("NEXIUM_NO_RT_RESOLVE").is_some() {
            return false;
        }
        let Some(src_nv) = mappings.nvmap_id_for(src_va) else {
            return false;
        };
        let Some(dst_nv) = mappings.nvmap_id_for(dst_va) else {
            return false;
        };
        let Some((kw, kh)) = renderer.rt_key_at_va(src_nv, self.src.width, self.src.height, src_va)
        else {
            return false;
        };
        if kw == 0 || kh == 0 || self.src.width <= kw {
            return false;
        }
        if dst_x_step <= 0 || dst_y_step <= 0 {
            return false;
        }
        let sx_ratio = (self.src.width / kw).max(1) as i64;
        let sy_ratio = (self.src.height / kh).max(1) as i64;
        let sx0 = ((src_x0 >> 32) / sx_ratio) as i32;
        let sy0 = ((src_y0 >> 32) / sy_ratio) as i32;
        let sx1 = (((src_x0 + du_dx.saturating_mul(width as i64)) >> 32) / sx_ratio) as i32;
        let sy1 = (((src_y0 + dv_dy.saturating_mul(height as i64)) >> 32) / sy_ratio) as i32;
        let dx0 = self.dst_x0;
        let dy0 = self.dst_y0;
        let dx1 = self.dst_x0.saturating_add(width as i32);
        let dy1 = self.dst_y0.saturating_add(height as i32);
        match renderer.resolve_rt_copy(
            src_nv,
            kw,
            kh,
            src_va,
            dst_nv,
            self.dst.width,
            self.dst.height,
            dst_va,
            [sx0, sy0, sx1, sy1],
            [dx0, dy0, dx1, dy1],
        ) {
            Ok(done) => {
                if done {
                    log::debug!(
                        "Fermi2D: rt-resolve src={}:{}x{} (surface {}x{}) @{:#x} -> dst={}:{}x{}@{:#x}",
                        src_nv,
                        kw,
                        kh,
                        self.src.width,
                        self.src.height,
                        src_va,
                        dst_nv,
                        self.dst.width,
                        self.dst.height,
                        dst_va
                    );
                }
                done
            }
            Err(e) => {
                log::warn!("Fermi2D: rt-resolve failed: {}", e);
                false
            }
        }
    }

    fn stage_blit_rt_source(
        &self,
        mappings: &GpuMappings,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let Some(renderer) = renderer else {
            return;
        };
        if self.src.memory_layout != MEMORY_LAYOUT_BLOCK_LINEAR {
            return;
        }
        {
            use std::sync::OnceLock;
            static OFF: OnceLock<bool> = OnceLock::new();
            if *OFF.get_or_init(|| std::env::var_os("NEXIUM_NO_F2D_SRC_STAGE").is_some()) {
                return;
            }
        }
        let src_va = self.src.gpu_va();
        let Some(nvmap) = mappings.nvmap_id_for(src_va) else {
            return;
        };
        let Some((src_cpu, limit)) = mappings.cpu_range_for(src_va) else {
            return;
        };
        let Some((kw, kh, bpp, mut raw)) = renderer.readback_target_raw(nvmap, src_va) else {
            return;
        };
        let width_bytes = (kw as usize) * bpp;
        if kh >= 2 && raw.len() >= width_bytes * kh as usize {
            let h = kh as usize;
            for y in 0..h / 2 {
                let (top, bot) = raw.split_at_mut((h - 1 - y) * width_bytes);
                top[y * width_bytes..(y + 1) * width_bytes]
                    .swap_with_slice(&mut bot[..width_bytes]);
            }
        }
        let tiled = super::maxwell_dma::swizzle_block_linear(
            &raw,
            width_bytes,
            kh as usize,
            width_bytes,
            width_bytes,
            kh as usize,
            self.src.block_height_log2(),
            0,
            0,
        );
        let n = tiled.len().min(limit as usize);
        mem_write(src_cpu, &tiled[..n]);
        if std::env::var_os("NEXIUM_JUMBO_DBG").is_some() {
            let nz = tiled[..n].iter().filter(|b| **b != 0).count();
            log::warn!(
                "[f2d-stage] src={:#x} dst={:#x} nvmap={} {}x{} bpp={} nz={}/{}",
                src_va,
                self.dst.gpu_va(),
                nvmap,
                kw,
                kh,
                bpp,
                nz,
                n
            );
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
        _src_y0_high: u32,
        mappings: &GpuMappings,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_va = self.src.gpu_va();
        let dst_va = self.dst.gpu_va();
        let Some((src_cpu, src_limit)) = mappings.cpu_range_for(src_va) else {
            log::trace!("Fermi2D: src gpu_va {:#x} not mapped", src_va);
            return;
        };
        let Some((dst_cpu, dst_limit)) = mappings.cpu_range_for(dst_va) else {
            log::trace!("Fermi2D: dst gpu_va {:#x} not mapped", dst_va);
            return;
        };
        if super::maxwell_compute::has_pending_writebacks() {
            let src_size = self.src.storage_size();
            let dst_size = self.dst.storage_size();
            if super::maxwell_compute::pending_writeback_overlaps(src_va, src_cpu, src_size)
                || super::maxwell_compute::pending_writeback_overlaps(dst_va, dst_cpu, dst_size)
            {
                if let Some(renderer) = renderer {
                    super::maxwell_compute::resolve_pending_writebacks(
                        renderer, mappings, mem_write,
                    );
                }
            }
        }

        let operation = self.operation & 0x7;
        if !matches!(operation, 0 | 3 | 5) {
            log::trace!("Fermi2D: unsupported operation {}", operation);
            return;
        }

        let Some(src_format) = self.src.format_info() else {
            log::trace!("Fermi2D: unknown source format {:#x}", self.src.format);
            return;
        };
        let Some(dst_format) = self.dst.format_info() else {
            log::trace!("Fermi2D: unknown destination format {:#x}", self.dst.format);
            return;
        };
        let width = self.dst_width_blit.unsigned_abs() as usize;
        let height = self.dst_height_blit.unsigned_abs() as usize;
        if width == 0 || height == 0 || self.dst.width == 0 || self.dst.height == 0 {
            return;
        }
        let dst_x_step = if self.dst_width_blit < 0 { -1i64 } else { 1 };
        let dst_y_step = if self.dst_height_blit < 0 { -1i64 } else { 1 };
        let src_x0 = fixed_32_32(self.src_x0_low, self.src_x0_high);
        let src_y0 = fixed_32_32(self.src_y0_low, self.src_y0_high);
        let du_dx = fixed_or_one(self.du_dx_low, self.du_dx_high);
        let dv_dy = fixed_or_one(self.dv_dy_low, self.dv_dy_high);
        let src_bpp = src_format.bytes_per_pixel();
        let dst_bpp = dst_format.bytes_per_pixel();
        if self.try_rt_resolve(
            renderer, mappings, src_va, dst_va, width, height, dst_x_step, dst_y_step, src_x0,
            src_y0, du_dx, dv_dy,
        ) {
            self.blit_count = self.blit_count.wrapping_add(1);
            nexium_gpu::tex_invalidate::bump_region(dst_va, self.dst.storage_size() as u64);
            return;
        }
        self.stage_blit_rt_source(mappings, renderer, mem_write);
        let mut src_pixel = vec![0u8; src_bpp];
        let mut dst_pixel = vec![0u8; dst_bpp];
        let src_size = self.src.storage_size().min(src_limit as usize);
        let dst_size = self.dst.storage_size().min(dst_limit as usize);
        let bulk_size_ok = src_size <= 256 * 1024 * 1024 && dst_size <= 256 * 1024 * 1024;
        let mut src_buf = if bulk_size_ok {
            vec![0u8; src_size]
        } else {
            Vec::new()
        };
        let mut dst_buf = if bulk_size_ok {
            vec![0u8; dst_size]
        } else {
            Vec::new()
        };
        let bulk = bulk_size_ok
            && src_size != 0
            && dst_size != 0
            && mem_read(src_cpu, &mut src_buf)
            && mem_read(dst_cpu, &mut dst_buf);

        let src_end = src_cpu.saturating_add(self.src.storage_size() as u64);
        let dst_end = dst_cpu.saturating_add(self.dst.storage_size() as u64);
        let overlaps = src_cpu < dst_end && dst_cpu < src_end;
        let mut deferred_writes = (!bulk && overlaps).then(Vec::new);

        let fast_downsample = bulk
            && src_format == dst_format
            && src_bpp == 4
            && self.src.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR
            && self.dst.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR
            && self.src.block_width_log2() == 0
            && self.dst.block_width_log2() == 0
            && self.src.block_height_log2() == self.dst.block_height_log2()
            && self.dst_x0 == 0
            && self.dst_y0 == 0
            && width == self.dst.width as usize
            && height == self.dst.height as usize
            && width % 16 == 0
            && self.src.width as usize >= width * 2
            && self.src.height as usize >= height
            && src_x0 == 1i64 << 31
            && src_y0 == 0
            && du_dx == 2i64 << 32
            && dv_dy == 1i64 << 32
            && downsample_block_linear_2x_rgba(
                &src_buf,
                &mut dst_buf,
                self.src.width as usize,
                self.dst.width as usize,
                height,
                self.src.block_height_log2(),
            );

        if !fast_downsample {
            for y in 0..height {
                let dst_y = self.dst_y0 as i64 + y as i64 * dst_y_step;
                let src_y = (src_y0 + dv_dy.saturating_mul(y as i64)) >> 32;
                if dst_y < 0 || src_y < 0 {
                    continue;
                }
                for x in 0..width {
                    let dst_x = self.dst_x0 as i64 + x as i64 * dst_x_step;
                    let src_x = (src_x0 + du_dx.saturating_mul(x as i64)) >> 32;
                    if dst_x < 0 || src_x < 0 {
                        continue;
                    }
                    let Some(src_offset) = self.src.pixel_offset(src_x as usize, src_y as usize)
                    else {
                        continue;
                    };
                    let Some(dst_offset) = self.dst.pixel_offset(dst_x as usize, dst_y as usize)
                    else {
                        continue;
                    };
                    if bulk {
                        if src_offset + src_bpp > src_buf.len() {
                            continue;
                        }
                        src_pixel.copy_from_slice(&src_buf[src_offset..src_offset + src_bpp]);
                    } else if !mem_read(src_cpu + src_offset as u64, &mut src_pixel) {
                        continue;
                    }
                    if src_format == dst_format {
                        dst_pixel.copy_from_slice(&src_pixel);
                    } else {
                        let Some(rgba) = src_format.decode_rgba8(&src_pixel) else {
                            continue;
                        };
                        if !dst_format.encode_rgba8(rgba, &mut dst_pixel) {
                            continue;
                        }
                    }
                    if bulk {
                        if dst_offset + dst_bpp <= dst_buf.len() {
                            dst_buf[dst_offset..dst_offset + dst_bpp].copy_from_slice(&dst_pixel);
                        }
                    } else {
                        let dst_address = dst_cpu + dst_offset as u64;
                        if let Some(writes) = deferred_writes.as_mut() {
                            writes.push((dst_address, dst_pixel.clone()));
                        } else {
                            mem_write(dst_address, &dst_pixel);
                        }
                    }
                }
            }
        }
        if bulk {
            mem_write(dst_cpu, &dst_buf);
        } else if let Some(writes) = deferred_writes {
            for (address, pixel) in writes {
                mem_write(address, &pixel);
            }
        }
        self.blit_count = self.blit_count.wrapping_add(1);
        nexium_gpu::tex_invalidate::bump_region(dst_va, self.dst.storage_size() as u64);

        let framebuffer_like = self.dst.memory_layout == MEMORY_LAYOUT_PITCH
            && self.dst.width >= 320
            && self.dst.height >= 240
            && self.dst.pitch() >= self.dst.width as usize * self.dst.bytes_per_pixel()
            && dst_format.is_presentable();
        if framebuffer_like {
            self.try_publish_frame(dst_cpu, mem_read);
        }
    }

    fn try_publish_frame(&self, dst_cpu: u64, mem_read: &dyn Fn(u64, &mut [u8]) -> bool) {
        let w = self.dst.width as usize;
        let h = self.dst.height as usize;
        let Some(format) = self.dst.format_info() else {
            return;
        };
        let bpp = format.bytes_per_pixel();
        let row_bytes = w.saturating_mul(4);
        let total = h.saturating_mul(row_bytes);
        if total == 0 {
            return;
        }
        let storage_size = self.dst.storage_size();
        let mut storage = vec![0u8; storage_size];
        if !mem_read(dst_cpu, &mut storage) {
            return;
        }
        let mut rgba = vec![0u8; total];
        let pitch = self.dst.pitch();
        for y in 0..h {
            for x in 0..w {
                let Some(offset) = self.dst.pixel_offset(x, y) else {
                    return;
                };
                let Some(raw) = storage.get(offset..offset + bpp) else {
                    return;
                };
                let Some(pixel) = format.decode_rgba8(raw) else {
                    return;
                };
                let out = (y * w + x) * 4;
                rgba[out..out + 4].copy_from_slice(&pixel);
            }
        }
        let rgb_nz = rgba
            .chunks_exact(4)
            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
            .count();
        log::info!(
            "Fermi2D::publish_frame dst_cpu={:#x} {}x{} pitch={} fmt={:#x} rgb_nz={}",
            dst_cpu,
            w,
            h,
            pitch,
            self.dst.format,
            rgb_nz
        );
        let mut q = self.captured_frames.lock().unwrap();
        if q.len() >= 2 {
            q.remove(0);
        }
        q.push(QueuedFrame {
            width: w as u32,
            height: h as u32,
            pixels: rgba,
        });
    }
}

const GOB_W: usize = 64;
const GOB_H: usize = 8;
const GOB_SIZE: usize = 512;

fn downsample_block_linear_2x_rgba(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    dst_width: usize,
    height: usize,
    block_height_log2: u32,
) -> bool {
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let block_size = block_height * GOB_SIZE;
    let src_columns = (src_width * 4).div_ceil(GOB_W);
    let dst_columns = (dst_width * 4).div_ceil(GOB_W);
    let src_block_row_size = src_columns * block_size;
    let dst_block_row_size = dst_columns * block_size;

    for y in 0..height {
        let block_y = y / rows_per_block;
        let y_in_block = y % rows_per_block;
        let gob_row = y_in_block / GOB_H;
        let y_in_gob = y_in_block % GOB_H;
        let y_offset = ((y_in_gob & 1) << 4) | ((y_in_gob & 6) << 6);
        let src_row = block_y * src_block_row_size + gob_row * GOB_SIZE + y_offset;
        let dst_row = block_y * dst_block_row_size + gob_row * GOB_SIZE + y_offset;

        for column in 0..dst_columns {
            let src_a = src_row + column * 2 * block_size;
            let src_b = src_a + block_size;
            let dst_base = dst_row + column * block_size;
            if !copy_even_rgba(dst, dst_base, src, src_a, src_a + 32)
                || !copy_even_rgba(dst, dst_base + 32, src, src_a + 64, src_a + 96)
                || !copy_even_rgba(dst, dst_base + 64, src, src_b, src_b + 32)
                || !copy_even_rgba(dst, dst_base + 96, src, src_b + 64, src_b + 96)
            {
                return false;
            }
        }
    }
    true
}

fn copy_even_rgba(dst: &mut [u8], dst_offset: usize, src: &[u8], a: usize, b: usize) -> bool {
    let Some(a) = src.get(a..a + 16) else {
        return false;
    };
    let Some(b) = src.get(b..b + 16) else {
        return false;
    };
    let Some(dst) = dst.get_mut(dst_offset..dst_offset + 16) else {
        return false;
    };
    dst[0..4].copy_from_slice(&a[0..4]);
    dst[4..8].copy_from_slice(&a[8..12]);
    dst[8..12].copy_from_slice(&b[0..4]);
    dst[12..16].copy_from_slice(&b[8..12]);
    true
}

fn fixed_32_32(low: u32, high: u32) -> i64 {
    (((high as u64) << 32) | low as u64) as i64
}

fn fixed_or_one(low: u32, high: u32) -> i64 {
    let value = fixed_32_32(low, high);
    if value == 0 {
        1i64 << 32
    } else {
        value
    }
}

fn tiled_size_bytes(
    width_bytes: usize,
    height: usize,
    block_width_log2: u32,
    block_height_log2: u32,
) -> usize {
    let block_width = 1usize << block_width_log2;
    let block_height = 1usize << block_height_log2;
    let block_width_bytes = block_width * GOB_W;
    let rows_per_block = block_height * GOB_H;
    let blocks_per_row = width_bytes.div_ceil(block_width_bytes);
    let block_rows = height.div_ceil(rows_per_block);
    blocks_per_row
        .saturating_mul(block_rows)
        .saturating_mul(block_width)
        .saturating_mul(block_height)
        .saturating_mul(GOB_SIZE)
}

fn block_linear_offset(
    byte_x: usize,
    y: usize,
    width_bytes: usize,
    block_width_log2: u32,
    block_height_log2: u32,
) -> usize {
    let block_width = 1usize << block_width_log2;
    let block_height = 1usize << block_height_log2;
    let block_width_bytes = block_width * GOB_W;
    let rows_per_block = block_height * GOB_H;
    let blocks_per_row = width_bytes.div_ceil(block_width_bytes);
    let block_y = y / rows_per_block;
    let y_in_block = y % rows_per_block;
    let block_x = byte_x / block_width_bytes;
    let x_in_block = byte_x % block_width_bytes;
    let gob_row = y_in_block / GOB_H;
    let y_in_gob = y_in_block % GOB_H;
    let gob_col = x_in_block / GOB_W;
    let x_in_gob = x_in_block % GOB_W;
    let in_gob = (x_in_gob & 0x0F)
        | ((y_in_gob & 0x01) << 4)
        | ((x_in_gob & 0x30) << 1)
        | ((y_in_gob & 0x06) << 6);
    let block_size = block_width * block_height * GOB_SIZE;
    let block_row_stride = blocks_per_row * block_size;
    block_y * block_row_stride
        + block_x * block_size
        + gob_row * block_width * GOB_SIZE
        + gob_col * block_height * GOB_SIZE
        + in_gob
}
