use super::super::GpuMappings;
use crate::gpu::formats::{surface_bytes_per_pixel, SurfaceFormat};
use crate::QueuedFrame;
use nexium_gpu::rt_cache::RtKey;
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

pub(crate) fn method_executes_blit(method: u32) -> bool {
    method == M_PIXELS_SRC_Y0_HIGH
}

const MEMORY_LAYOUT_BLOCK_LINEAR: u32 = 0;
const MEMORY_LAYOUT_PITCH: u32 = 1;
const EXACT_RT_COPY_PROVENANCE_CAPACITY: usize = 64;

fn exact_rt_copy_provenance_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("NEXIUM_KICKOFF_PROFILE").is_some()
            || matches!(
                std::env::var("NEXIUM_FERMI_RT_SNAPSHOT_LEASE")
                    .ok()
                    .as_deref(),
                Some("1") | Some("true") | Some("on") | Some("yes")
            )
    })
}

fn rt_resolve_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_NO_RT_RESOLVE").is_some())
}

fn jumbo_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_JUMBO_DBG").is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactRtCopyProvenance {
    pub destination: RtKey,
    pub stamp: u64,
    pub destination_va: u64,
    pub destination_size: u64,
    pub destination_generation: u64,
    pub bytes_per_pixel: usize,
    pub block_size: u32,
}

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
    guest_write_range: Option<(u64, u64)>,
    logical_guest_write_span: Option<(u64, usize)>,
    exact_rt_copies: Vec<ExactRtCopyProvenance>,
}

impl Fermi2D {
    pub(crate) fn blit_touches_live_rt(&self, renderer: Option<&nexium_gpu::Renderer>) -> bool {
        let Some(renderer) = renderer else {
            return false;
        };
        renderer.has_color_target_touching(self.src.gpu_va())
            || renderer.has_color_target_touching(self.dst.gpu_va())
    }
}

impl Fermi2D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take_guest_write_range(&mut self) -> Option<(u64, u64)> {
        self.guest_write_range.take()
    }

    pub(crate) fn take_logical_guest_write_span(&mut self) -> Option<(u64, usize)> {
        self.logical_guest_write_span.take()
    }

    fn record_logical_guest_write(&mut self, address: u64, size: usize) {
        if size != 0 {
            self.logical_guest_write_span = Some((address, size));
        }
    }

    fn record_guest_write(&mut self, address: u64, size: usize) {
        if size == 0 {
            return;
        }
        let end = address.saturating_add(size as u64);
        self.guest_write_range = Some(match self.guest_write_range {
            Some((start, current_size)) => {
                let current_end = start.saturating_add(current_size);
                let merged_start = start.min(address);
                let merged_end = current_end.max(end);
                (merged_start, merged_end.saturating_sub(merged_start))
            }
            None => (address, end.saturating_sub(address)),
        });
    }

    fn record_exact_rt_copy(
        &mut self,
        destination: RtKey,
        stamp: u64,
        destination_va: u64,
        destination_size: u64,
        bytes_per_pixel: usize,
        block_size: u32,
    ) {
        if stamp == 0 || destination_size == 0 || bytes_per_pixel == 0 {
            return;
        }
        let provenance = ExactRtCopyProvenance {
            destination,
            stamp,
            destination_va,
            destination_size,
            destination_generation: nexium_gpu::tex_invalidate::region_gen_range(
                destination_va,
                destination_size,
            ),
            bytes_per_pixel,
            block_size,
        };
        if let Some(index) = self
            .exact_rt_copies
            .iter()
            .position(|entry| entry.destination_va == destination_va)
        {
            self.exact_rt_copies.remove(index);
        }
        if self.exact_rt_copies.len() >= EXACT_RT_COPY_PROVENANCE_CAPACITY {
            self.exact_rt_copies.remove(0);
        }
        self.exact_rt_copies.push(provenance);
    }

    pub fn exact_rt_copy_for_tic(
        &self,
        tic: &nexium_gpu::texture::TicEntry,
        nvmap_id: u32,
    ) -> Option<ExactRtCopyProvenance> {
        if !tic.is_block_linear
            || !matches!(tic.texture_type, 1 | 5)
            || tic.depth != 1
            || tic.base_layer != 0
            || tic.max_mip_level != 0
            || tic.res_min_mip_level != 0
            || tic.res_max_mip_level != 0
        {
            return None;
        }
        let provenance = *self
            .exact_rt_copies
            .iter()
            .rev()
            .find(|entry| entry.destination_va == tic.gpu_va)?;
        let block_width_log2 = provenance.block_size & 0xF;
        let block_height_log2 = (provenance.block_size >> 4) & 0xF;
        if provenance.destination.nvmap_id != nvmap_id
            || provenance.destination.gpu_va != tic.gpu_va
            || provenance.destination.width != tic.width
            || provenance.destination.height != tic.height
            || provenance.destination.depth != 1
            || provenance.destination.is_3d
            || tic.format.src_bpp() != provenance.bytes_per_pixel
            || provenance.block_size & !0xFF != 0
            || block_width_log2 > 5
            || block_height_log2 > 5
            || tic.block_width_log2 != block_width_log2
            || tic.block_height_log2 != block_height_log2
            || tic.block_depth_log2 != 0
            || tic.tile_width_spacing != 0
            || nexium_gpu::tex_invalidate::region_gen_range(
                provenance.destination_va,
                provenance.destination_size,
            ) != provenance.destination_generation
        {
            return None;
        }
        Some(provenance)
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        renderer: Option<&std::sync::Arc<nexium_gpu::renderer::Renderer>>,
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
        if rt_resolve_disabled() {
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
        if kw == 0 || kh == 0 {
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

    #[allow(clippy::too_many_arguments)]
    fn exact_copy_candidate(
        &self,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mappings: &GpuMappings,
        src_va: u64,
        dst_va: u64,
        src_limit: u64,
        dst_limit: u64,
        width: usize,
        height: usize,
        dst_x_step: i64,
        dst_y_step: i64,
        src_x0: i64,
        src_y0: i64,
        du_dx: i64,
        dv_dy: i64,
        src_bpp: usize,
        dst_bpp: usize,
    ) -> Option<(RtKey, u64, u32)> {
        let renderer = renderer?;
        if !exact_rt_identity_copy_compatible(
            &self.src,
            &self.dst,
            self.dst_x0,
            self.dst_y0,
            width,
            height,
            dst_x_step,
            dst_y_step,
            src_x0,
            src_y0,
            du_dx,
            dv_dy,
            src_bpp,
            dst_bpp,
        ) {
            return None;
        }
        let src_size = self.src.storage_size();
        let dst_size = self.dst.storage_size();
        if src_size == 0
            || dst_size == 0
            || src_size as u64 > src_limit
            || dst_size as u64 > dst_limit
        {
            return None;
        }
        let src_nvmap = mappings.nvmap_id_for(src_va)?;
        let dst_nvmap = mappings.nvmap_id_for(dst_va)?;
        let (source, source_stamp) = renderer.render_target_at_va(src_nvmap, src_va)?;
        if source.nvmap_id != src_nvmap
            || source.gpu_va != src_va
            || source.width != self.src.width
            || source.height != self.src.height
            || source.width != self.dst.width
            || source.height != self.dst.height
            || source.depth != 1
            || source.is_3d
            || source_stamp == 0
        {
            return None;
        }
        Some((source, source_stamp, dst_nvmap))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_exact_rt_identity_copy(
        &self,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mappings: &GpuMappings,
        src_va: u64,
        dst_va: u64,
        src_limit: u64,
        dst_limit: u64,
        width: usize,
        height: usize,
        dst_x_step: i64,
        dst_y_step: i64,
        src_x0: i64,
        src_y0: i64,
        du_dx: i64,
        dv_dy: i64,
        src_bpp: usize,
        dst_bpp: usize,
    ) -> Option<(RtKey, u64)> {
        let (source, source_stamp, dst_nvmap) = self.exact_copy_candidate(
            renderer, mappings, src_va, dst_va, src_limit, dst_limit, width, height, dst_x_step,
            dst_y_step, src_x0, src_y0, du_dx, dv_dy, src_bpp, dst_bpp,
        )?;
        let renderer = renderer?;
        match renderer.resolve_rt_copy_exact_guest(source, source_stamp, src_bpp, dst_nvmap, dst_va)
        {
            Ok(provenance) => provenance,
            Err(error) => {
                log::warn!(
                    "Fermi2D: exact RT copy {:#x}->{:#x} failed: {}",
                    src_va,
                    dst_va,
                    error
                );
                None
            }
        }
    }

    fn blit_geometry(&self) -> Option<(usize, usize, i64, i64, i64, i64, i64, i64, usize, usize)> {
        if !matches!(self.operation & 0x7, 0 | 3 | 5) {
            return None;
        }
        let src_format = self.src.format_info()?;
        let dst_format = self.dst.format_info()?;
        let width = self.dst_width_blit.unsigned_abs() as usize;
        let height = self.dst_height_blit.unsigned_abs() as usize;
        if width == 0 || height == 0 || self.dst.width == 0 || self.dst.height == 0 {
            return None;
        }
        let dst_x_step = if self.dst_width_blit < 0 { -1i64 } else { 1 };
        let dst_y_step = if self.dst_height_blit < 0 { -1i64 } else { 1 };
        let src_x0 = fixed_32_32(self.src_x0_low, self.src_x0_high);
        let src_y0 = fixed_32_32(self.src_y0_low, self.src_y0_high);
        let du_dx = fixed_or_one(self.du_dx_low, self.du_dx_high);
        let dv_dy = fixed_or_one(self.dv_dy_low, self.dv_dy_high);
        Some((
            width,
            height,
            dst_x_step,
            dst_y_step,
            src_x0,
            src_y0,
            du_dx,
            dv_dy,
            src_format.bytes_per_pixel(),
            dst_format.bytes_per_pixel(),
        ))
    }

    pub(crate) fn blit_exact_async_candidate(
        &self,
        renderer: Option<&nexium_gpu::renderer::Renderer>,
        mappings: &GpuMappings,
    ) -> bool {
        if !async_blit_enabled() {
            return false;
        }
        let src_va = self.src.gpu_va();
        let dst_va = self.dst.gpu_va();
        let Some((_, src_limit)) = mappings.cpu_range_for(src_va) else {
            return false;
        };
        let Some((_, dst_limit)) = mappings.cpu_range_for(dst_va) else {
            return false;
        };
        let Some((
            width,
            height,
            dst_x_step,
            dst_y_step,
            src_x0,
            src_y0,
            du_dx,
            dv_dy,
            src_bpp,
            dst_bpp,
        )) = self.blit_geometry()
        else {
            return false;
        };
        self.exact_copy_candidate(
            renderer, mappings, src_va, dst_va, src_limit, dst_limit, width, height, dst_x_step,
            dst_y_step, src_x0, src_y0, du_dx, dv_dy, src_bpp, dst_bpp,
        )
        .is_some()
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
        if jumbo_debug_enabled() {
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
        renderer_arc: Option<&std::sync::Arc<nexium_gpu::renderer::Renderer>>,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let renderer = renderer_arc.map(|arc| arc.as_ref());
        self.guest_write_range = None;
        self.logical_guest_write_span = None;
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
        if async_blit_enabled() {
            if let Some((source, source_stamp, dst_nvmap)) = self.exact_copy_candidate(
                renderer, mappings, src_va, dst_va, src_limit, dst_limit, width, height,
                dst_x_step, dst_y_step, src_x0, src_y0, du_dx, dv_dy, src_bpp, dst_bpp,
            ) {
                if let (Some(arc), Some(rt)) =
                    (renderer_arc, crate::render_thread::maybe_render_thread())
                {
                    match arc.plan_rt_copy_exact_guest(
                        source,
                        source_stamp,
                        src_bpp,
                        dst_nvmap,
                        dst_va,
                    ) {
                        Ok(Some((destination, stamp))) => {
                            let job_renderer = std::sync::Arc::clone(arc);
                            rt.submit_named(
                                "fermi-exact-copy",
                                Box::new(move || {
                                    if let Err(error) =
                                        job_renderer.execute_rt_copy_exact(source, destination)
                                    {
                                        log::warn!(
                                            "Fermi2D: async exact RT copy failed: {}",
                                            error
                                        );
                                    }
                                }),
                            );
                            self.blit_count = self.blit_count.wrapping_add(1);
                            let destination_size = self.dst.storage_size();
                            self.record_logical_guest_write(dst_va, destination_size);
                            nexium_gpu::tex_invalidate::bump_region(
                                dst_va,
                                destination_size as u64,
                            );
                            if exact_rt_copy_provenance_enabled() {
                                self.record_exact_rt_copy(
                                    destination,
                                    stamp,
                                    dst_va,
                                    destination_size as u64,
                                    dst_bpp,
                                    self.dst.block_size,
                                );
                            }
                            return;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            log::warn!(
                                "Fermi2D: async exact RT copy plan {:#x}->{:#x} failed: {}",
                                src_va,
                                dst_va,
                                error
                            );
                        }
                    }
                }
            }
        }
        if let Some((destination, stamp)) = self.try_exact_rt_identity_copy(
            renderer, mappings, src_va, dst_va, src_limit, dst_limit, width, height, dst_x_step,
            dst_y_step, src_x0, src_y0, du_dx, dv_dy, src_bpp, dst_bpp,
        ) {
            self.blit_count = self.blit_count.wrapping_add(1);
            let destination_size = self.dst.storage_size();
            self.record_logical_guest_write(dst_va, destination_size);
            nexium_gpu::tex_invalidate::bump_region(dst_va, destination_size as u64);
            if exact_rt_copy_provenance_enabled() {
                self.record_exact_rt_copy(
                    destination,
                    stamp,
                    dst_va,
                    destination_size as u64,
                    dst_bpp,
                    self.dst.block_size,
                );
            }
            return;
        }
        if self.try_rt_resolve(
            renderer, mappings, src_va, dst_va, width, height, dst_x_step, dst_y_step, src_x0,
            src_y0, du_dx, dv_dy,
        ) {
            self.blit_count = self.blit_count.wrapping_add(1);
            let destination_size = self.dst.storage_size();
            self.record_logical_guest_write(dst_va, destination_size);
            nexium_gpu::tex_invalidate::bump_region(dst_va, destination_size as u64);
            return;
        }
        self.stage_blit_rt_source(mappings, renderer, mem_write);
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

        let fast_identity = bulk
            && copy_identical_surface_storage(
                &self.src,
                &self.dst,
                &src_buf,
                &mut dst_buf,
                self.dst_x0,
                self.dst_y0,
                width,
                height,
                dst_x_step,
                dst_y_step,
                src_x0,
                src_y0,
                du_dx,
                dv_dy,
            );
        let fast_downsample = !fast_identity
            && bulk
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
        let fast_block_linear = !fast_identity
            && !fast_downsample
            && bulk
            && src_format == dst_format
            && copy_same_format_block_linear(
                &self.src,
                &self.dst,
                &src_buf,
                &mut dst_buf,
                self.dst_x0,
                self.dst_y0,
                width,
                height,
                dst_x_step,
                dst_y_step,
                src_x0,
                src_y0,
                du_dx,
                dv_dy,
            );

        if !fast_identity && !fast_downsample && !fast_block_linear {
            let mut src_pixel = vec![0u8; src_bpp];
            let mut dst_pixel = vec![0u8; dst_bpp];
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
                        } else if mem_write(dst_address, &dst_pixel) {
                            self.record_guest_write(dst_address, dst_pixel.len());
                        }
                    }
                }
            }
        }
        if bulk {
            if mem_write(dst_cpu, &dst_buf) {
                self.record_guest_write(dst_cpu, dst_buf.len());
            }
        } else if let Some(writes) = deferred_writes {
            for (address, pixel) in writes {
                if mem_write(address, &pixel) {
                    self.record_guest_write(address, pixel.len());
                }
            }
        }
        if self.guest_write_range.is_some() {
            self.record_logical_guest_write(dst_va, self.dst.storage_size());
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
const GOB_X_SECTORS: [usize; 4] = [0, 32, 256, 288];

fn in_gob_x_offset(x_in_gob: usize) -> usize {
    (x_in_gob & 0x0F) + ((x_in_gob & 0x10) << 1) + ((x_in_gob & 0x20) << 3)
}

fn in_gob_y_offset(y_in_gob: usize) -> usize {
    ((y_in_gob & 0x01) << 4) + ((y_in_gob & 0x06) << 5)
}

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
        let y_offset = in_gob_y_offset(y_in_gob);
        let src_row = block_y * src_block_row_size + gob_row * GOB_SIZE + y_offset;
        let dst_row = block_y * dst_block_row_size + gob_row * GOB_SIZE + y_offset;

        for column in 0..dst_columns {
            let src_a = src_row + column * 2 * block_size;
            let src_b = src_a + block_size;
            let dst_base = dst_row + column * block_size;
            let [s0, s1, s2, s3] = GOB_X_SECTORS;
            if !copy_even_rgba(dst, dst_base + s0, src, src_a + s0, src_a + s1)
                || !copy_even_rgba(dst, dst_base + s1, src, src_a + s2, src_a + s3)
                || !copy_even_rgba(dst, dst_base + s2, src, src_b + s0, src_b + s1)
                || !copy_even_rgba(dst, dst_base + s3, src, src_b + s2, src_b + s3)
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

pub(crate) fn async_blit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_FERMI_ASYNC_BLIT").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
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

fn exact_rt_identity_copy_compatible(
    src_surface: &Surface,
    dst_surface: &Surface,
    dst_x0: i32,
    dst_y0: i32,
    width: usize,
    height: usize,
    dst_x_step: i64,
    dst_y_step: i64,
    src_x0: i64,
    src_y0: i64,
    du_dx: i64,
    dv_dy: i64,
    src_bpp: usize,
    dst_bpp: usize,
) -> bool {
    src_surface.format == dst_surface.format
        && src_surface.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR
        && dst_surface.memory_layout == MEMORY_LAYOUT_BLOCK_LINEAR
        && src_surface.block_size == dst_surface.block_size
        && src_surface.width == dst_surface.width
        && src_surface.height == dst_surface.height
        && src_surface.pitch() == dst_surface.pitch()
        && src_surface.depth <= 1
        && dst_surface.depth <= 1
        && src_surface._layer == 0
        && dst_surface._layer == 0
        && src_surface.storage_size() == dst_surface.storage_size()
        && src_bpp == dst_bpp
        && src_bpp == src_surface.bytes_per_pixel()
        && dst_bpp == dst_surface.bytes_per_pixel()
        && matches!(src_bpp, 1 | 2 | 4 | 8 | 16)
        && dst_x_step == 1
        && dst_y_step == 1
        && dst_x0 == 0
        && dst_y0 == 0
        && width == dst_surface.width as usize
        && height == dst_surface.height as usize
        && src_x0 == 1i64 << 31
        && src_y0 == 1i64 << 31
        && du_dx == 1i64 << 32
        && dv_dy == 1i64 << 32
}

fn copy_identical_surface_storage(
    src_surface: &Surface,
    dst_surface: &Surface,
    src: &[u8],
    dst: &mut [u8],
    dst_x0: i32,
    dst_y0: i32,
    width: usize,
    height: usize,
    dst_x_step: i64,
    dst_y_step: i64,
    src_x0: i64,
    src_y0: i64,
    du_dx: i64,
    dv_dy: i64,
) -> bool {
    let one = 1i64 << 32;
    let Some(tightly_covered_size) = (src_surface.width as usize)
        .checked_mul(src_surface.height as usize)
        .and_then(|pixels| pixels.checked_mul(src_surface.bytes_per_pixel()))
    else {
        return false;
    };
    let layout_compatible = src_surface.memory_layout == dst_surface.memory_layout
        && match src_surface.memory_layout {
            MEMORY_LAYOUT_BLOCK_LINEAR => {
                src_surface.block_width_log2() == dst_surface.block_width_log2()
                    && src_surface.block_height_log2() == dst_surface.block_height_log2()
            }
            MEMORY_LAYOUT_PITCH => src_surface.pitch() == dst_surface.pitch(),
            _ => false,
        };
    if src_surface.format != dst_surface.format
        || src_surface.width != dst_surface.width
        || src_surface.height != dst_surface.height
        || !layout_compatible
        || src_surface.storage_size() != tightly_covered_size
        || dst_surface.storage_size() != tightly_covered_size
        || src.len() != tightly_covered_size
        || dst.len() != tightly_covered_size
        || dst_x0 != 0
        || dst_y0 != 0
        || width != dst_surface.width as usize
        || height != dst_surface.height as usize
        || dst_x_step != 1
        || dst_y_step != 1
        || !(0..one).contains(&src_x0)
        || !(0..one).contains(&src_y0)
        || du_dx != one
        || dv_dy != one
    {
        return false;
    }
    dst.copy_from_slice(src);
    true
}

fn copy_same_format_block_linear(
    src_surface: &Surface,
    dst_surface: &Surface,
    src: &[u8],
    dst: &mut [u8],
    dst_x0: i32,
    dst_y0: i32,
    width: usize,
    height: usize,
    dst_x_step: i64,
    dst_y_step: i64,
    src_x0: i64,
    src_y0: i64,
    du_dx: i64,
    dv_dy: i64,
) -> bool {
    if src_surface.memory_layout != MEMORY_LAYOUT_BLOCK_LINEAR
        || dst_surface.memory_layout != MEMORY_LAYOUT_BLOCK_LINEAR
        || src_surface.bytes_per_pixel() != dst_surface.bytes_per_pixel()
    {
        return false;
    }
    let bpp = src_surface.bytes_per_pixel();
    let src_width = src_surface.width as usize;
    let src_height = src_surface.height as usize;
    let dst_width = dst_surface.width as usize;
    let dst_height = dst_surface.height as usize;
    let src_width_bytes = src_width.saturating_mul(bpp);
    let dst_width_bytes = dst_width.saturating_mul(bpp);
    let column_indices = reachable_unit_step_range(dst_x0, dst_x_step, dst_width, width);
    let mut columns = Vec::with_capacity(column_indices.len());
    for x in column_indices {
        let src_x = (src_x0 + du_dx.saturating_mul(x as i64)) >> 32;
        let dst_x = dst_x0 as i64 + x as i64 * dst_x_step;
        if src_x < 0 || dst_x < 0 || src_x >= src_width as i64 || dst_x >= dst_width as i64 {
            continue;
        }
        columns.push((
            block_linear_x_offset(
                src_x as usize * bpp,
                src_surface.block_width_log2(),
                src_surface.block_height_log2(),
            ),
            block_linear_x_offset(
                dst_x as usize * bpp,
                dst_surface.block_width_log2(),
                dst_surface.block_height_log2(),
            ),
        ));
    }
    for y in 0..height {
        let src_y = (src_y0 + dv_dy.saturating_mul(y as i64)) >> 32;
        let dst_y = dst_y0 as i64 + y as i64 * dst_y_step;
        if src_y < 0 || dst_y < 0 || src_y >= src_height as i64 || dst_y >= dst_height as i64 {
            continue;
        }
        let src_y_offset = block_linear_y_offset(
            src_y as usize,
            src_width_bytes,
            src_surface.block_width_log2(),
            src_surface.block_height_log2(),
        );
        let dst_y_offset = block_linear_y_offset(
            dst_y as usize,
            dst_width_bytes,
            dst_surface.block_width_log2(),
            dst_surface.block_height_log2(),
        );
        for &(src_x_offset, dst_x_offset) in &columns {
            let src_offset = src_y_offset + src_x_offset;
            let dst_offset = dst_y_offset + dst_x_offset;
            if src_offset + bpp <= src.len() && dst_offset + bpp <= dst.len() {
                dst[dst_offset..dst_offset + bpp]
                    .copy_from_slice(&src[src_offset..src_offset + bpp]);
            }
        }
    }
    true
}

fn reachable_unit_step_range(
    origin: i32,
    step: i64,
    extent: usize,
    count: usize,
) -> std::ops::Range<usize> {
    if extent == 0 || count == 0 {
        return 0..0;
    }
    let origin = origin as i64;
    let extent = extent.min(i64::MAX as usize) as i64;
    let count = count.min(i64::MAX as usize) as i64;
    let (start, end) = match step {
        1 => ((-origin).max(0), extent.saturating_sub(origin)),
        -1 => (
            origin.saturating_sub(extent).saturating_add(1).max(0),
            origin.saturating_add(1),
        ),
        _ => return 0..0,
    };
    let start = start.clamp(0, count);
    let end = end.clamp(0, count);
    if start >= end {
        start as usize..start as usize
    } else {
        start as usize..end as usize
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

fn block_linear_x_offset(byte_x: usize, block_width_log2: u32, block_height_log2: u32) -> usize {
    let block_width = 1usize << block_width_log2;
    let block_height = 1usize << block_height_log2;
    let block_width_bytes = block_width * GOB_W;
    let block_x = byte_x / block_width_bytes;
    let x_in_block = byte_x % block_width_bytes;
    let gob_col = x_in_block / GOB_W;
    let x_in_gob = x_in_block % GOB_W;
    let block_size = block_width * block_height * GOB_SIZE;
    block_x * block_size + gob_col * block_height * GOB_SIZE + in_gob_x_offset(x_in_gob)
}

fn block_linear_y_offset(
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
    let gob_row = y_in_block / GOB_H;
    let y_in_gob = y_in_block % GOB_H;
    let block_size = block_width * block_height * GOB_SIZE;
    block_y * blocks_per_row * block_size
        + gob_row * block_width * GOB_SIZE
        + in_gob_y_offset(y_in_gob)
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
    let in_gob = in_gob_x_offset(x_in_gob) + in_gob_y_offset(y_in_gob);
    let block_size = block_width * block_height * GOB_SIZE;
    let block_row_stride = blocks_per_row * block_size;
    block_y * block_row_stride
        + block_x * block_size
        + gob_row * block_width * GOB_SIZE
        + gob_col * block_height * GOB_SIZE
        + in_gob
}

#[cfg(test)]
mod tests {
    use super::{
        block_linear_offset, block_linear_x_offset, block_linear_y_offset,
        copy_identical_surface_storage, copy_same_format_block_linear,
        exact_rt_identity_copy_compatible, method_executes_blit, Fermi2D, Surface,
        MEMORY_LAYOUT_BLOCK_LINEAR, MEMORY_LAYOUT_PITCH, M_PIXELS_SRC_Y0_HIGH,
    };
    use crate::gpu::formats::FMT_A8B8G8R8_UNORM;
    use crate::gpu::GpuMappings;
    use nexium_gpu::rt_cache::RtKey;
    use nexium_gpu::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
    use std::cell::Cell;

    #[test]
    fn only_final_pixel_method_executes_blit() {
        assert!(method_executes_blit(M_PIXELS_SRC_Y0_HIGH));
        assert!(!method_executes_blit(M_PIXELS_SRC_Y0_HIGH - 1));
        assert!(!method_executes_blit(0));
    }

    #[test]
    fn exact_rt_copy_provenance_requires_a_current_direct_tic() {
        const VA: u64 = 0x7fff_1000_0000;
        const SIZE: u64 = 0x4000;
        let key = RtKey::new(7, 64, 32, VA);
        let tic = TicEntry {
            format: TicFormat::R8G8B8A8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: VA,
            width: 64,
            height: 32,
            block_width_log2: 0,
            block_height_log2: 2,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        nexium_gpu::tex_invalidate::bump_region(VA, SIZE);
        let mut fermi = Fermi2D::new();
        fermi.record_exact_rt_copy(key, 9, VA, SIZE, 4, 0x20);

        assert_eq!(
            fermi
                .exact_rt_copy_for_tic(&tic, 7)
                .map(|entry| entry.stamp),
            Some(9)
        );
        assert!(fermi.exact_rt_copy_for_tic(&tic, 8).is_none());

        let mut mipmapped = tic;
        mipmapped.max_mip_level = 1;
        assert!(fermi.exact_rt_copy_for_tic(&mipmapped, 7).is_none());

        let mut wrong_bpp = tic;
        wrong_bpp.format = TicFormat::R8;
        assert!(fermi.exact_rt_copy_for_tic(&wrong_bpp, 7).is_none());

        let mut wrong_block_width = tic;
        wrong_block_width.block_width_log2 = 1;
        assert!(fermi.exact_rt_copy_for_tic(&wrong_block_width, 7).is_none());

        let mut wrong_block_height = tic;
        wrong_block_height.block_height_log2 = 1;
        assert!(fermi
            .exact_rt_copy_for_tic(&wrong_block_height, 7)
            .is_none());

        let mut wrong_block_depth = tic;
        wrong_block_depth.block_depth_log2 = 1;
        assert!(fermi.exact_rt_copy_for_tic(&wrong_block_depth, 7).is_none());

        let mut tiled = tic;
        tiled.tile_width_spacing = 1;
        assert!(fermi.exact_rt_copy_for_tic(&tiled, 7).is_none());

        nexium_gpu::tex_invalidate::bump_region(VA, SIZE);
        assert!(fermi.exact_rt_copy_for_tic(&tic, 7).is_none());
    }

    #[test]
    fn fallback_guest_write_range_is_successful_and_one_shot() {
        const SRC_GPU: u64 = 0x10_0000;
        const DST_GPU: u64 = 0x20_0000;
        const SRC_CPU: u64 = 0x30_0000;
        const DST_CPU: u64 = 0x40_0000;
        const BYTE_COUNT: usize = 8;

        let mut mappings = GpuMappings::new();
        mappings.add(SRC_GPU, 0x1000, SRC_CPU, 1);
        mappings.add(DST_GPU, 0x1000, DST_CPU, 2);
        let surface = |gpu_va: u64| Surface {
            format: FMT_A8B8G8R8_UNORM,
            memory_layout: MEMORY_LAYOUT_PITCH,
            depth: 1,
            pitch: BYTE_COUNT as u32,
            width: 2,
            height: 1,
            offset_high: (gpu_va >> 32) as u32,
            offset_low: gpu_va as u32,
            ..Surface::default()
        };
        let mut fermi = Fermi2D {
            src: surface(SRC_GPU),
            dst: surface(DST_GPU),
            dst_width_blit: 2,
            dst_height_blit: 1,
            du_dx_high: 1,
            dv_dy_high: 1,
            src_x0_low: 1 << 31,
            src_y0_low: 1 << 31,
            ..Fermi2D::default()
        };
        let mem_read = |address: u64, output: &mut [u8]| {
            assert!(address == SRC_CPU || address == DST_CPU);
            output.fill(if address == SRC_CPU { 0x5a } else { 0 });
            true
        };
        let writes = Cell::new(0usize);
        let mem_write = |address: u64, input: &[u8]| {
            assert_eq!(address, DST_CPU);
            assert_eq!(input.len(), BYTE_COUNT);
            writes.set(writes.get() + 1);
            true
        };

        fermi.dispatch_method(
            M_PIXELS_SRC_Y0_HIGH,
            0,
            &mappings,
            None,
            &mem_read,
            &mem_write,
        );

        assert_eq!(writes.get(), 1);
        assert_eq!(
            fermi.take_guest_write_range(),
            Some((DST_CPU, BYTE_COUNT as u64))
        );
        assert_eq!(fermi.take_guest_write_range(), None);
        assert_eq!(
            fermi.take_logical_guest_write_span(),
            Some((DST_GPU, BYTE_COUNT))
        );
        assert_eq!(fermi.take_logical_guest_write_span(), None);

        fermi.dispatch_method(
            M_PIXELS_SRC_Y0_HIGH,
            0,
            &mappings,
            None,
            &mem_read,
            &|_, _| false,
        );
        assert_eq!(fermi.take_guest_write_range(), None);
        assert_eq!(fermi.take_logical_guest_write_span(), None);
    }

    #[test]
    fn separated_block_linear_offsets_match_reference() {
        for block_width_log2 in 0..=5 {
            for block_height_log2 in 0..=5 {
                for width_bytes in [1, 15, 16, 63, 64, 65, 127, 128, 129, 257] {
                    let rows = (1usize << block_height_log2) * 8;
                    for y in 0..rows * 2 + 3 {
                        for byte_x in 0..width_bytes {
                            assert_eq!(
                                block_linear_x_offset(byte_x, block_width_log2, block_height_log2,)
                                    + block_linear_y_offset(
                                        y,
                                        width_bytes,
                                        block_width_log2,
                                        block_height_log2,
                                    ),
                                block_linear_offset(
                                    byte_x,
                                    y,
                                    width_bytes,
                                    block_width_log2,
                                    block_height_log2,
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn in_gob_offsets_match_tegra_swizzle_table() {
        for y in 0..8usize {
            for x in 0..64usize {
                let expected = ((x % 64) / 32) * 256
                    + ((y % 8) / 2) * 64
                    + ((x % 32) / 16) * 32
                    + (y % 2) * 16
                    + (x % 16);
                assert_eq!(
                    super::in_gob_x_offset(x) + super::in_gob_y_offset(y),
                    expected,
                    "x={x} y={y}"
                );
            }
        }
        for (sector, base) in super::GOB_X_SECTORS.iter().enumerate() {
            assert_eq!(*base, super::in_gob_x_offset(sector * 16));
        }
    }

    #[test]
    fn accelerated_block_linear_scale_matches_pixel_copy() {
        for block_width_log2 in 0..=2 {
            for block_height_log2 in 0..=2 {
                let block_size = block_width_log2 | (block_height_log2 << 4);
                let src_surface = Surface {
                    format: 0xcf,
                    memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
                    block_size,
                    width: 19,
                    height: 15,
                    ..Surface::default()
                };
                let dst_surface = Surface {
                    format: 0xcf,
                    memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
                    block_size,
                    width: 13,
                    height: 11,
                    ..Surface::default()
                };
                let src = (0..src_surface.storage_size())
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                    .collect::<Vec<_>>();
                let mut reference = vec![0x5a; dst_surface.storage_size()];
                let mut accelerated = reference.clone();
                let width = 11;
                let height = 8;
                let src_x0 = 1i64 << 31;
                let src_y0 = 1i64 << 30;
                let du_dx = 3i64 << 31;
                let dv_dy = 5i64 << 30;
                for y in 0..height {
                    let src_y = (src_y0 + dv_dy * y as i64) >> 32;
                    let dst_y = 1 + y;
                    for x in 0..width {
                        let src_x = (src_x0 + du_dx * x as i64) >> 32;
                        let dst_x = 1 + x;
                        let src_offset = src_surface
                            .pixel_offset(src_x as usize, src_y as usize)
                            .unwrap();
                        let dst_offset = dst_surface.pixel_offset(dst_x, dst_y).unwrap();
                        reference[dst_offset..dst_offset + 4]
                            .copy_from_slice(&src[src_offset..src_offset + 4]);
                    }
                }
                assert!(copy_same_format_block_linear(
                    &src_surface,
                    &dst_surface,
                    &src,
                    &mut accelerated,
                    1,
                    1,
                    width,
                    height,
                    1,
                    1,
                    src_x0,
                    src_y0,
                    du_dx,
                    dv_dy,
                ));
                assert_eq!(accelerated, reference);
            }
        }
    }

    #[test]
    fn identity_copy_uses_snapshot_for_overlapping_storage() {
        let surface = Surface {
            format: 0xcf,
            memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
            block_size: 0,
            width: 16,
            height: 8,
            ..Surface::default()
        };
        let size = surface.storage_size();
        let shift = 37;
        let mut guest = (0..size + shift)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let source_snapshot = guest[..size].to_vec();
        let mut destination_snapshot = guest[shift..shift + size].to_vec();
        let mut pixel_reference = destination_snapshot.clone();
        let bpp = surface.bytes_per_pixel();
        for y in 0..surface.height as usize {
            for x in 0..surface.width as usize {
                let offset = surface.pixel_offset(x, y).unwrap();
                pixel_reference[offset..offset + bpp]
                    .copy_from_slice(&source_snapshot[offset..offset + bpp]);
            }
        }
        let mut expected = guest.clone();
        expected[shift..shift + size].copy_from_slice(&pixel_reference);

        assert!(copy_identical_surface_storage(
            &surface,
            &surface,
            &source_snapshot,
            &mut destination_snapshot,
            0,
            0,
            surface.width as usize,
            surface.height as usize,
            1,
            1,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
        ));
        assert_eq!(destination_snapshot, pixel_reference);
        guest[shift..shift + size].copy_from_slice(&destination_snapshot);
        assert_eq!(guest, expected);
    }

    #[test]
    fn padded_identity_copy_preserves_destination_padding() {
        let surface = Surface {
            format: 0xcf,
            memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
            block_size: 0x20,
            width: 19,
            height: 15,
            ..Surface::default()
        };
        let src = (0..surface.storage_size())
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(9))
            .collect::<Vec<_>>();
        let initial = vec![0x5a; surface.storage_size()];
        let mut pixel_reference = initial.clone();
        let mut touched = vec![false; surface.storage_size()];
        let bpp = surface.bytes_per_pixel();
        for y in 0..surface.height as usize {
            for x in 0..surface.width as usize {
                let offset = surface.pixel_offset(x, y).unwrap();
                pixel_reference[offset..offset + bpp].copy_from_slice(&src[offset..offset + bpp]);
                touched[offset..offset + bpp].fill(true);
            }
        }
        assert!(touched.iter().any(|touched| !touched));

        let mut raw_attempt = initial.clone();
        assert!(!copy_identical_surface_storage(
            &surface,
            &surface,
            &src,
            &mut raw_attempt,
            0,
            0,
            surface.width as usize,
            surface.height as usize,
            1,
            1,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
        ));
        assert_eq!(raw_attempt, initial);

        let mut accelerated = initial.clone();
        assert!(copy_same_format_block_linear(
            &surface,
            &surface,
            &src,
            &mut accelerated,
            0,
            0,
            surface.width as usize,
            surface.height as usize,
            1,
            1,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
        ));
        assert_eq!(accelerated, pixel_reference);
        assert!(accelerated
            .iter()
            .zip(&touched)
            .all(|(&byte, &touched)| touched || byte == 0x5a));
    }

    #[test]
    fn block_linear_copy_caps_signed_guest_width() {
        let surface = Surface {
            format: 0xcf,
            memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
            block_size: 0,
            width: 8,
            height: 8,
            ..Surface::default()
        };
        let src = (0..surface.storage_size())
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        let mut reference = vec![0x5a; surface.storage_size()];
        let mut accelerated = reference.clone();
        let bpp = surface.bytes_per_pixel();
        for x in 0..surface.width as usize {
            let src_offset = surface.pixel_offset(x, 0).unwrap();
            let dst_offset = surface
                .pixel_offset(surface.width as usize - 1 - x, 0)
                .unwrap();
            reference[dst_offset..dst_offset + bpp]
                .copy_from_slice(&src[src_offset..src_offset + bpp]);
        }

        assert!(copy_same_format_block_linear(
            &surface,
            &surface,
            &src,
            &mut accelerated,
            surface.width as i32 - 1,
            0,
            i32::MIN.unsigned_abs() as usize,
            1,
            -1,
            1,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
        ));
        assert_eq!(accelerated, reference);
    }

    #[test]
    fn exact_rt_identity_copy_requires_full_2d_surface() {
        let surface = Surface {
            format: 0xe0,
            memory_layout: MEMORY_LAYOUT_BLOCK_LINEAR,
            block_size: 0x40,
            depth: 1,
            width: 1600,
            height: 900,
            ..Surface::default()
        };
        let compatible = |src: &Surface,
                          dst: &Surface,
                          dst_x0,
                          dst_y0,
                          width,
                          height,
                          src_x0,
                          src_y0,
                          du_dx,
                          dv_dy,
                          src_bpp,
                          dst_bpp| {
            exact_rt_identity_copy_compatible(
                src, dst, dst_x0, dst_y0, width, height, 1, 1, src_x0, src_y0, du_dx, dv_dy,
                src_bpp, dst_bpp,
            )
        };
        assert!(compatible(
            &surface,
            &surface,
            0,
            0,
            1600,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));

        let mut incompatible = surface;
        incompatible.block_size = 0x30;
        assert!(!compatible(
            &surface,
            &incompatible,
            0,
            0,
            1600,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));
        incompatible = surface;
        incompatible._layer = 1;
        assert!(!compatible(
            &surface,
            &incompatible,
            0,
            0,
            1600,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));
        assert!(!compatible(
            &surface,
            &surface,
            1,
            0,
            1600,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));
        assert!(!compatible(
            &surface,
            &surface,
            0,
            0,
            1599,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));
        assert!(!compatible(
            &surface,
            &surface,
            0,
            0,
            1600,
            900,
            0,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            4,
        ));
        assert!(!compatible(
            &surface,
            &surface,
            0,
            0,
            1600,
            900,
            1i64 << 31,
            1i64 << 31,
            1i64 << 32,
            1i64 << 32,
            4,
            8,
        ));
    }
}
