use super::super::GpuMappings;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, OnceLock,
};

fn jumbo_dbg() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("NEXIUM_JUMBO_DBG").map_or(false, |v| v == "1"))
}

fn video_dma_trace() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("NEXIUM_VIDEO_DMA_TRACE").is_some())
}

fn trace_video_sample(
    phase: &str,
    sequence: u64,
    gpu_va: u64,
    cpu_va: u64,
    sample_len: usize,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> (bool, usize, u64) {
    let mut sample = vec![0u8; sample_len.min(4096)];
    let read = mem_read(cpu_va, &mut sample);
    let (nonzero, checksum) = if read { sample_stats(&sample) } else { (0, 0) };
    log::warn!(
        "[video-dma] {} #{} gpu={:#x} cpu={:#x} bytes={} read={} nz={} ck={:#x}",
        phase,
        sequence,
        gpu_va,
        cpu_va,
        sample.len(),
        read,
        nonzero,
        checksum
    );
    (read, nonzero, checksum)
}

fn sample_stats(buf: &[u8]) -> (usize, u64) {
    let n = buf.len().min(65536);
    let mut nonzero = 0usize;
    let mut sum = 0u64;
    for &b in &buf[..n] {
        if b != 0 {
            nonzero += 1;
        }
        sum = sum.wrapping_add(b as u64).wrapping_mul(31);
    }
    (nonzero, sum)
}

pub const MAXWELL_DMA_CLASS: u32 = 0xB0B5;

const M_OFFSET_IN_UPPER: u32 = 0x100;
const M_OFFSET_IN_LOWER: u32 = 0x101;
const M_OFFSET_OUT_UPPER: u32 = 0x102;
const M_OFFSET_OUT_LOWER: u32 = 0x103;
const M_PITCH_IN: u32 = 0x104;
const M_PITCH_OUT: u32 = 0x105;
const M_LINE_LENGTH_IN: u32 = 0x106;
const M_LINE_COUNT: u32 = 0x107;
pub const M_LAUNCH_DMA: u32 = 0xC0;
const M_SET_REMAP_CONST_A: u32 = 0x1C0;
const M_SET_REMAP_CONST_B: u32 = 0x1C1;
const M_SET_REMAP_COMPONENTS: u32 = 0x1C2;
const M_SET_DST_BLOCK_SIZE: u32 = 0x1C3;
const M_SET_DST_WIDTH: u32 = 0x1C4;
const M_SET_DST_HEIGHT: u32 = 0x1C5;
const M_SET_DST_DEPTH: u32 = 0x1C6;
const M_SET_DST_LAYER: u32 = 0x1C7;
const M_SET_DST_ORIGIN: u32 = 0x1C8;
const M_SET_SRC_BLOCK_SIZE: u32 = 0x1CA;
const M_SET_SRC_WIDTH: u32 = 0x1CB;
const M_SET_SRC_HEIGHT: u32 = 0x1CC;
const M_SET_SRC_DEPTH: u32 = 0x1CD;
const M_SET_SRC_LAYER: u32 = 0x1CE;
const M_SET_SRC_ORIGIN: u32 = 0x1CF;

const LAUNCH_SRC_LAYOUT_BIT: u32 = 7;
const LAUNCH_DST_LAYOUT_BIT: u32 = 8;
const LAUNCH_MULTI_LINE_BIT: u32 = 9;
const LAUNCH_REMAP_ENABLE_BIT: u32 = 10;
const LAYOUT_BLOCK_LINEAR: u32 = 0;
const LAYOUT_PITCH: u32 = 1;

#[derive(Default)]
pub struct MaxwellDma {
    offset_in_upper: u32,
    offset_in_lower: u32,
    offset_out_upper: u32,
    offset_out_lower: u32,
    pitch_in: u32,
    pitch_out: u32,
    line_length_in: u32,
    line_count: u32,
    remap_const_a: u32,
    remap_const_b: u32,
    remap_components: u32,
    src_block_size: u32,
    src_width: u32,
    src_height: u32,
    src_origin_x: u32,
    src_origin_y: u32,
    dst_block_size: u32,
    dst_width: u32,
    dst_height: u32,
    dst_origin_x: u32,
    dst_origin_y: u32,
    pub blit_count: u64,

    pub blit_dst_by_src: HashMap<u64, u64>,

    pub last_tiled_dst_cpu: u64,

    pub last_tiled_dst_bh_log2: u32,

    pub last_tiled_dst_stride: u32,

    pub last_tiled_dst_height: u32,

    pub draw_texture_blits: u64,

    clamp_log_count: u64,
}

impl MaxwellDma {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        match method {
            M_OFFSET_IN_UPPER => self.offset_in_upper = arg,
            M_OFFSET_IN_LOWER => self.offset_in_lower = arg,
            M_OFFSET_OUT_UPPER => self.offset_out_upper = arg,
            M_OFFSET_OUT_LOWER => self.offset_out_lower = arg,
            M_PITCH_IN => self.pitch_in = arg,
            M_PITCH_OUT => self.pitch_out = arg,
            M_LINE_LENGTH_IN => self.line_length_in = arg,
            M_LINE_COUNT => self.line_count = arg,
            M_SET_REMAP_CONST_A => self.remap_const_a = arg,
            M_SET_REMAP_CONST_B => self.remap_const_b = arg,
            M_SET_REMAP_COMPONENTS => self.remap_components = arg,
            M_SET_SRC_BLOCK_SIZE => self.src_block_size = arg,
            M_SET_SRC_WIDTH => self.src_width = arg,
            M_SET_SRC_HEIGHT => self.src_height = arg,
            M_SET_SRC_DEPTH | M_SET_SRC_LAYER => {}
            M_SET_SRC_ORIGIN => {
                self.src_origin_x = arg & 0xFFFF;
                self.src_origin_y = (arg >> 16) & 0xFFFF;
            }
            M_SET_DST_BLOCK_SIZE => self.dst_block_size = arg,
            M_SET_DST_WIDTH => self.dst_width = arg,
            M_SET_DST_HEIGHT => self.dst_height = arg,
            M_SET_DST_DEPTH | M_SET_DST_LAYER => {}
            M_SET_DST_ORIGIN => {
                self.dst_origin_x = arg & 0xFFFF;
                self.dst_origin_y = (arg >> 16) & 0xFFFF;
            }
            M_LAUNCH_DMA => self.launch_dma(arg, mappings, mem_read, mem_write),
            _ => {
                log::trace!("MaxwellDma: unhandled method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn src_addr(&self) -> u64 {
        ((self.offset_in_upper as u64) << 32) | self.offset_in_lower as u64
    }

    pub fn transfer_spans(&self) -> [(u64, usize); 2] {
        let lines = self.line_count.max(1) as usize;
        let src_stride = self.line_length_in.max(self.pitch_in) as usize;
        let dst_stride = self.line_length_in.max(self.pitch_out) as usize;
        let dst_addr = ((self.offset_out_upper as u64) << 32) | self.offset_out_lower as u64;
        [
            (self.src_addr(), src_stride.saturating_mul(lines)),
            (dst_addr, dst_stride.saturating_mul(lines)),
        ]
    }

    pub fn stage_rt_source(
        &self,
        flags: u32,
        mappings: &GpuMappings,
        renderer: &nexium_gpu::renderer::Renderer,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_layout = (flags >> LAUNCH_SRC_LAYOUT_BIT) & 1;
        let dst_layout = (flags >> LAUNCH_DST_LAYOUT_BIT) & 1;
        if src_layout != LAYOUT_BLOCK_LINEAR {
            return;
        }
        if dst_layout != LAYOUT_PITCH {
            static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *OFF.get_or_init(|| std::env::var_os("NEXIUM_NO_DMA_BLOCK_STAGE").is_some()) {
                return;
            }
        }
        let src_gpu = self.src_addr();
        let Some(nvmap) = mappings.nvmap_id_for(src_gpu) else {
            if jumbo_dbg() {
                log::warn!("[jumbo] stage MISS no-nvmap src={:#x}", self.src_addr());
            }
            return;
        };
        let Some((src_cpu, limit)) = mappings.cpu_range_for(src_gpu) else {
            if jumbo_dbg() {
                log::warn!("[jumbo] stage MISS no-cpu-range src={:#x}", src_gpu);
            }
            return;
        };
        let bh_log2 = ((self.src_block_size >> 4) & 0xF) as u32;
        if let Some((kw, kh, bpp, mut raw)) = renderer.readback_target_raw(nvmap, src_gpu) {
            if jumbo_dbg() {
                let (nz, ck) = sample_stats(&raw);
                log::warn!(
                    "[jumbo] stage RAW src={:#x} nvmap={} {}x{} bpp={} nz={} ck={:#x}",
                    src_gpu,
                    nvmap,
                    kw,
                    kh,
                    bpp,
                    nz,
                    ck
                );
            }
            let width_bytes = (kw as usize) * bpp;
            if kh >= 2 && raw.len() >= width_bytes * kh as usize {
                let h = kh as usize;
                for y in 0..h / 2 {
                    let (top, bot) = raw.split_at_mut((h - 1 - y) * width_bytes);
                    top[y * width_bytes..(y + 1) * width_bytes]
                        .swap_with_slice(&mut bot[..width_bytes]);
                }
            }
            let tiled = swizzle_block_linear(
                &raw,
                width_bytes,
                kh as usize,
                width_bytes,
                width_bytes,
                kh as usize,
                bh_log2,
                0,
                0,
            );
            let n = tiled.len().min(limit as usize);
            mem_write(src_cpu, &tiled[..n]);
            log::debug!(
                "MaxwellDma::stage_rt_source raw va={:#x} nvmap={} {}x{} bpp={} bytes={}",
                src_gpu,
                nvmap,
                kw,
                kh,
                bpp,
                n
            );
            return;
        }
        let Some((kw, kh)) = renderer.rt_key_for_nvmap(nvmap, self.src_width, self.src_height)
        else {
            if jumbo_dbg() {
                log::warn!(
                    "[jumbo] stage MISS no-rt-key src={:#x} nvmap={} {}x{}",
                    src_gpu,
                    nvmap,
                    self.src_width,
                    self.src_height
                );
            }
            return;
        };
        let Some(mut rgba) = renderer.readback_target(nvmap, kw, kh) else {
            if jumbo_dbg() {
                log::warn!(
                    "[jumbo] stage MISS readback-none src={:#x} nvmap={} {}x{}",
                    src_gpu,
                    nvmap,
                    kw,
                    kh
                );
            }
            return;
        };
        if jumbo_dbg() {
            let (nz, ck) = sample_stats(&rgba);
            log::warn!(
                "[jumbo] stage FUZZY src={:#x} nvmap={} {}x{} nz={} ck={:#x}",
                src_gpu,
                nvmap,
                kw,
                kh,
                nz,
                ck
            );
        }
        let width_bytes = (kw as usize) * 4;
        if kh >= 2 && rgba.len() >= width_bytes * kh as usize {
            let h = kh as usize;
            for y in 0..h / 2 {
                let (top, bot) = rgba.split_at_mut((h - 1 - y) * width_bytes);
                top[y * width_bytes..(y + 1) * width_bytes]
                    .swap_with_slice(&mut bot[..width_bytes]);
            }
        }
        let tiled = swizzle_block_linear(
            &rgba,
            width_bytes,
            kh as usize,
            width_bytes,
            width_bytes,
            kh as usize,
            bh_log2,
            0,
            0,
        );
        let n = tiled.len().min(limit as usize);
        mem_write(src_cpu, &tiled[..n]);
    }

    fn dst_addr(&self) -> u64 {
        ((self.offset_out_upper as u64) << 32) | self.offset_out_lower as u64
    }

    fn launch_dma(
        &mut self,
        flags: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_layout = (flags >> LAUNCH_SRC_LAYOUT_BIT) & 1;
        let dst_layout = (flags >> LAUNCH_DST_LAYOUT_BIT) & 1;
        let multi_line = (flags >> LAUNCH_MULTI_LINE_BIT) & 1 != 0;
        let remap_enable = (flags >> LAUNCH_REMAP_ENABLE_BIT) & 1 != 0;

        let src_gpu = self.src_addr();
        let dst_gpu = self.dst_addr();
        static DMA_DROPPED: AtomicU64 = AtomicU64::new(0);
        let Some(src_cpu) = mappings.cpu_address_for(src_gpu) else {
            let n = DMA_DROPPED.fetch_add(1, Ordering::Relaxed);
            if n < 32 || n % 1024 == 0 {
                log::warn!(
                    "[dma-drop] #{} src gpu_va={:#x} dst={:#x} units={} lines={}",
                    n,
                    src_gpu,
                    dst_gpu,
                    self.line_length_in,
                    self.line_count
                );
            }
            return;
        };
        let Some((dst_cpu, dst_limit)) = mappings.cpu_range_for(dst_gpu) else {
            let n = DMA_DROPPED.fetch_add(1, Ordering::Relaxed);
            if n < 32 || n % 1024 == 0 {
                log::warn!(
                    "[dma-drop] #{} dst gpu_va={:#x} src={:#x} units={} lines={}",
                    n,
                    dst_gpu,
                    src_gpu,
                    self.line_length_in,
                    self.line_count
                );
            }
            return;
        };
        let dst_limit = dst_limit as usize;

        let line_length_units = self.line_length_in as usize;
        let line_count = if multi_line {
            self.line_count.max(1) as usize
        } else {
            1
        };

        if line_length_units == 0 {
            return;
        }

        let (
            component_size,
            num_src_components,
            num_dst_components,
            dst_x_sel,
            dst_y_sel,
            dst_z_sel,
            dst_w_sel,
        ) = if remap_enable {
            (
                ((self.remap_components >> 16) & 0x3) as usize + 1,
                ((self.remap_components >> 20) & 0x3) as usize + 1,
                ((self.remap_components >> 24) & 0x3) as usize + 1,
                (self.remap_components >> 0) & 0x7,
                (self.remap_components >> 4) & 0x7,
                (self.remap_components >> 8) & 0x7,
                (self.remap_components >> 12) & 0x7,
            )
        } else {
            (1, 1, 1, 0, 1, 2, 3)
        };
        let src_bytes_per_group = component_size * num_src_components;
        let dst_bytes_per_group = component_size * num_dst_components;
        let line_length_src = line_length_units * src_bytes_per_group;
        let line_length_dst = line_length_units * dst_bytes_per_group;

        static VIDEO_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        static VIDEO_GEOMETRIES: OnceLock<Mutex<std::collections::HashSet<[u64; 18]>>> =
            OnceLock::new();
        let movie_sized = line_count >= 100 && (line_length_src >= 256 || line_length_dst >= 256);
        let video_sequence = if video_dma_trace() && movie_sized {
            let geometry = [
                src_layout as u64,
                dst_layout as u64,
                line_length_units as u64,
                line_count as u64,
                line_length_src as u64,
                line_length_dst as u64,
                self.pitch_in as u64,
                self.pitch_out as u64,
                self.src_width as u64,
                self.src_height as u64,
                self.dst_width as u64,
                self.dst_height as u64,
                self.src_block_size as u64,
                self.dst_block_size as u64,
                remap_enable as u64,
                component_size as u64,
                num_src_components as u64,
                num_dst_components as u64,
            ];
            VIDEO_GEOMETRIES
                .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
                .lock()
                .ok()
                .and_then(|mut set| {
                    set.insert(geometry)
                        .then(|| VIDEO_SEQUENCE.fetch_add(1, Ordering::Relaxed))
                })
        } else {
            None
        };
        if let Some(sequence) = video_sequence.filter(|sequence| *sequence < 512) {
            log::warn!(
                "[video-dma] launch #{} blit={} src={:#x} dst={:#x} layout={}->{} flags={:#x} units={} lines={} src_bytes={} dst_bytes={} pitch={}->{} size={}x{}->{:}x{} block={:#x}->{:#x} remap={} comp={} nsrc={} ndst={} sel={}{}{}{}",
                sequence,
                self.blit_count,
                src_gpu,
                dst_gpu,
                src_layout,
                dst_layout,
                flags,
                line_length_units,
                line_count,
                line_length_src,
                line_length_dst,
                self.pitch_in,
                self.pitch_out,
                self.src_width,
                self.src_height,
                self.dst_width,
                self.dst_height,
                self.src_block_size,
                self.dst_block_size,
                remap_enable,
                component_size,
                num_src_components,
                num_dst_components,
                dst_x_sel,
                dst_y_sel,
                dst_z_sel,
                dst_w_sel
            );
            trace_video_sample("src", sequence, src_gpu, src_cpu, line_length_src, mem_read);
        }

        let src_bytes_per_element = if remap_enable {
            component_size.max(1) * num_src_components.max(1)
        } else {
            1
        };
        let dst_bytes_per_element = if remap_enable {
            component_size.max(1) * num_dst_components.max(1)
        } else {
            1
        };

        if self.blit_count < 64 {
            log::info!(
                "MaxwellDma::launch[{}]: src_gpu={:#x} cpu={:#x} dst_gpu={:#x} cpu={:#x} \
                 src_layout={} dst_layout={} flags={:#x} line_units={} line_count={} \
                 remap={} comp_size={} n_src={} n_dst={} sel={}{}{}{} \
                 src_w={} src_h={} src_blk={:#x} dst_w={} dst_h={} dst_blk={:#x} pitch_in={} pitch_out={}",
                self.blit_count,
                src_gpu,
                src_cpu,
                dst_gpu,
                dst_cpu,
                src_layout,
                dst_layout,
                flags,
                line_length_units,
                line_count,
                remap_enable,
                component_size,
                num_src_components,
                num_dst_components,
                dst_x_sel,
                dst_y_sel,
                dst_z_sel,
                dst_w_sel,
                self.src_width,
                self.src_height,
                self.src_block_size,
                self.dst_width,
                self.dst_height,
                self.dst_block_size,
                self.pitch_in,
                self.pitch_out,
            );
        }

        match (src_layout, dst_layout) {
            (LAYOUT_PITCH, LAYOUT_BLOCK_LINEAR) => {
                self.blit_pitch_to_block(
                    src_cpu,
                    dst_cpu,
                    dst_limit,
                    dst_gpu,
                    mappings,
                    line_length_src,
                    line_length_dst,
                    line_count,
                    remap_enable,
                    component_size,
                    num_src_components,
                    num_dst_components,
                    [dst_x_sel, dst_y_sel, dst_z_sel, dst_w_sel],
                    dst_bytes_per_element,
                    mem_read,
                    mem_write,
                );
                let dst_w = if self.dst_width != 0 {
                    (self.dst_width as usize) * dst_bytes_per_element.max(1)
                } else {
                    line_length_dst
                };
                let dst_h = if self.dst_height != 0 {
                    self.dst_height as usize
                } else {
                    line_count
                };
                let bh = ((self.dst_block_size >> 4) & 0xF) as u32;
                nexium_gpu::tex_invalidate::bump_region(
                    dst_gpu,
                    tiled_size_bytes(dst_w, dst_h, bh) as u64,
                );
            }
            (LAYOUT_BLOCK_LINEAR, LAYOUT_PITCH) => {
                let dst_pitch = self.pitch_out.max(line_length_src as u32) as usize;
                nexium_gpu::pitch_oracle::record_pitch_dst(
                    dst_gpu,
                    (dst_pitch * line_count) as u64,
                );
                self.blit_block_to_pitch(
                    src_cpu,
                    dst_cpu,
                    dst_limit,
                    line_length_src,
                    line_count,
                    src_bytes_per_element,
                    mem_read,
                    mem_write,
                );
                nexium_gpu::tex_invalidate::bump_region(dst_gpu, (dst_pitch * line_count) as u64);
            }
            (LAYOUT_PITCH, LAYOUT_PITCH) => {
                self.blit_pitch_to_pitch(
                    src_cpu,
                    dst_cpu,
                    dst_limit,
                    line_length_src,
                    line_count,
                    mem_read,
                    mem_write,
                );
                let dst_pitch = self.pitch_out.max(line_length_src as u32) as usize;
                nexium_gpu::tex_invalidate::bump_region(dst_gpu, (dst_pitch * line_count) as u64);
            }
            (LAYOUT_BLOCK_LINEAR, LAYOUT_BLOCK_LINEAR) => {
                self.blit_block_to_block(
                    src_cpu,
                    dst_cpu,
                    dst_limit,
                    dst_gpu,
                    line_length_src,
                    line_length_dst,
                    line_count,
                    src_bytes_per_element,
                    dst_bytes_per_element,
                    mem_read,
                    mem_write,
                );
            }
            _ => {
                if self.clamp_log_count < 24 {
                    self.clamp_log_count += 1;
                    log::warn!(
                        "MaxwellDma::launch unsupported combo src_layout={} dst_layout={} \
                         src_gpu={:#x} dst_gpu={:#x} src_w={} src_h={} dst_w={} dst_h={} line_len={} lines={}",
                        src_layout,
                        dst_layout,
                        src_gpu,
                        dst_gpu,
                        self.src_width,
                        self.src_height,
                        self.dst_width,
                        self.dst_height,
                        self.line_length_in,
                        line_count,
                    );
                }
            }
        }
        if let Some(sequence) = video_sequence.filter(|sequence| *sequence < 512) {
            trace_video_sample("dst", sequence, dst_gpu, dst_cpu, line_length_dst, mem_read);
            log::warn!(
                "[video-dma] done #{} gen={} range={}",
                sequence,
                nexium_gpu::tex_invalidate::region_gen_range(
                    dst_gpu,
                    line_length_dst.saturating_mul(line_count) as u64
                ),
                line_length_dst.saturating_mul(line_count)
            );
        }
        self.blit_count = self.blit_count.wrapping_add(1);
    }

    fn apply_remap(
        &self,
        src: &[u8],
        line_length_units: usize,
        line_count: usize,
        component_size: usize,
        num_src_components: usize,
        num_dst_components: usize,
        sel: [u32; 4],
    ) -> Vec<u8> {
        let dst_bytes_per_group = component_size * num_dst_components;
        let src_bytes_per_group = component_size * num_src_components;
        let mut out = vec![0u8; line_length_units * line_count * dst_bytes_per_group];
        let const_a = self.remap_const_a.to_le_bytes();
        let const_b = self.remap_const_b.to_le_bytes();
        for y in 0..line_count {
            let src_row_off = y * line_length_units * src_bytes_per_group;
            let dst_row_off = y * line_length_units * dst_bytes_per_group;
            for x in 0..line_length_units {
                let src_group_off = src_row_off + x * src_bytes_per_group;
                let dst_group_off = dst_row_off + x * dst_bytes_per_group;
                for dst_c in 0..num_dst_components {
                    let dst_byte_off = dst_group_off + dst_c * component_size;
                    let s = sel[dst_c.min(3)];
                    let src_slice: &[u8] = match s {
                        0..=3 if (s as usize) < num_src_components => {
                            let off = src_group_off + (s as usize) * component_size;
                            &src[off..off + component_size]
                        }
                        4 => &const_a[..component_size.min(4)],
                        5 => &const_b[..component_size.min(4)],
                        _ => continue,
                    };
                    out[dst_byte_off..dst_byte_off + component_size]
                        .copy_from_slice(&src_slice[..component_size]);
                }
            }
        }
        out
    }

    fn blit_pitch_to_pitch(
        &self,
        src_cpu: u64,
        dst_cpu: u64,
        dst_limit: usize,
        line_length: usize,
        line_count: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_pitch = self.pitch_in.max(line_length as u32) as usize;
        let dst_pitch = self.pitch_out.max(line_length as u32) as usize;
        let mut row = vec![0u8; line_length];
        for y in 0..line_count {
            let dst_row_off = y * dst_pitch;
            if dst_row_off >= dst_limit {
                break;
            }
            let n = line_length.min(dst_limit - dst_row_off);
            let src_off = src_cpu + (y * src_pitch) as u64;
            let dst_off = dst_cpu + dst_row_off as u64;
            if !mem_read(src_off, &mut row) {
                break;
            }
            mem_write(dst_off, &row[..n]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn blit_pitch_to_block(
        &mut self,
        src_cpu: u64,
        dst_cpu: u64,
        dst_limit: usize,
        dst_gpu: u64,
        mappings: &GpuMappings,
        line_length_src: usize,
        line_length_dst: usize,
        line_count: usize,
        remap_enable: bool,
        component_size: usize,
        num_src_components: usize,
        num_dst_components: usize,
        sel: [u32; 4],
        bytes_per_element: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.last_tiled_dst_cpu = dst_cpu;

        let block_height_log2 = ((self.dst_block_size >> 4) & 0xF) as u32;
        let src_pitch = self.pitch_in.max(line_length_src as u32) as usize;

        let dst_width_bytes = if self.dst_width != 0 {
            (self.dst_width as usize) * bytes_per_element.max(1)
        } else {
            line_length_dst
        };
        let dst_height = if self.dst_height != 0 {
            self.dst_height as usize
        } else {
            line_count
        };

        let mut linear_src = vec![0u8; src_pitch * line_count];
        for y in 0..line_count {
            let src_off = src_cpu + (y * src_pitch) as u64;
            let row_off = y * src_pitch;
            mem_read(src_off, &mut linear_src[row_off..row_off + line_length_src]);
        }
        let post_remap = if remap_enable {
            let line_length_units = line_length_src / (component_size * num_src_components).max(1);
            self.apply_remap(
                &linear_src,
                line_length_units,
                line_count,
                component_size,
                num_src_components,
                num_dst_components,
                sel,
            )
        } else {
            linear_src
        };
        let post_remap_pitch = if remap_enable {
            line_length_dst
        } else {
            src_pitch
        };
        if jumbo_dbg() && dst_width_bytes <= 512 && line_count <= 64 {
            let (nz, ck) = sample_stats(&post_remap);
            log::warn!(
                "[jumbo] p2b dst={:#x} w={}B h={} nz={} ck={:#x}",
                dst_gpu,
                dst_width_bytes,
                line_count,
                nz,
                ck
            );
        }
        let tiled_size = tiled_size_bytes(dst_width_bytes, dst_height, block_height_log2);
        let mut tiled = vec![0u8; tiled_size];
        let n = tiled_size.min(dst_limit);
        mem_read(dst_cpu, &mut tiled[..n]);
        swizzle_block_linear_into(
            &mut tiled,
            &post_remap,
            line_length_dst,
            line_count,
            post_remap_pitch,
            dst_width_bytes,
            block_height_log2,
            (self.dst_origin_x as usize) * bytes_per_element.max(1),
            self.dst_origin_y as usize,
        );
        if tiled_size > dst_limit && self.clamp_log_count < 16 {
            self.clamp_log_count += 1;
            log::warn!(
                "MaxwellDma::blit_pitch_to_block clamp: tiled={} > dst_limit={} \
                 dst_gpu={:#x} dst_cpu={:#x} dst_w={} dst_h={} bh={} line_count={} line_len_dst={} | {}",
                tiled_size,
                dst_limit,
                dst_gpu,
                dst_cpu,
                self.dst_width,
                self.dst_height,
                block_height_log2,
                line_count,
                line_length_dst,
                mappings.describe_around(dst_gpu),
            );
        }
        mem_write(dst_cpu, &tiled[..n]);
        nexium_gpu::pitch_oracle::clear_pitch_range(dst_gpu, n as u64);
        self.last_tiled_dst_bh_log2 = block_height_log2;
        self.last_tiled_dst_stride = dst_width_bytes as u32;
        self.last_tiled_dst_height = dst_height as u32;
    }

    #[allow(clippy::too_many_arguments)]
    fn blit_block_to_block(
        &mut self,
        src_cpu: u64,
        dst_cpu: u64,
        dst_limit: usize,
        dst_gpu: u64,
        line_length_src: usize,
        line_length_dst: usize,
        line_count: usize,
        src_bytes_per_element: usize,
        dst_bytes_per_element: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let src_bh = ((self.src_block_size >> 4) & 0xF) as u32;
        let src_width_bytes = if self.src_width != 0 {
            (self.src_width as usize) * src_bytes_per_element.max(1)
        } else {
            line_length_src
        };
        let src_height = if self.src_height != 0 {
            self.src_height as usize
        } else {
            line_count
        };
        let mut src_tiled = vec![0u8; tiled_size_bytes(src_width_bytes, src_height, src_bh)];
        mem_read(src_cpu, &mut src_tiled);

        let inter_pitch = line_length_src.max(1);
        let linear = unswizzle_block_linear_bytes(
            &src_tiled,
            line_length_src,
            line_count,
            inter_pitch,
            src_width_bytes,
            src_height,
            src_bh,
            (self.src_origin_x as usize) * src_bytes_per_element.max(1),
            self.src_origin_y as usize,
        );

        let dst_bh = ((self.dst_block_size >> 4) & 0xF) as u32;
        let dst_width_bytes = if self.dst_width != 0 {
            (self.dst_width as usize) * dst_bytes_per_element.max(1)
        } else {
            line_length_dst
        };
        let dst_height = if self.dst_height != 0 {
            self.dst_height as usize
        } else {
            line_count
        };
        let tiled_size = tiled_size_bytes(dst_width_bytes, dst_height, dst_bh);
        let mut tiled = vec![0u8; tiled_size];
        let n = tiled_size.min(dst_limit);
        mem_read(dst_cpu, &mut tiled[..n]);
        swizzle_block_linear_into(
            &mut tiled,
            &linear,
            line_length_dst,
            line_count,
            inter_pitch,
            dst_width_bytes,
            dst_bh,
            (self.dst_origin_x as usize) * dst_bytes_per_element.max(1),
            self.dst_origin_y as usize,
        );
        mem_write(dst_cpu, &tiled[..n]);
        nexium_gpu::pitch_oracle::clear_pitch_range(dst_gpu, n as u64);
        nexium_gpu::tex_invalidate::bump_region(dst_gpu, tiled_size as u64);
    }

    fn blit_block_to_pitch(
        &mut self,
        src_cpu: u64,
        dst_cpu: u64,
        dst_limit: usize,
        line_length: usize,
        line_count: usize,
        bytes_per_element: usize,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        {
            let block_height_log2_dbg = ((self.src_block_size >> 4) & 0xF) as u32;
            let src_width_bytes_dbg = if self.src_width != 0 {
                (self.src_width as usize) * bytes_per_element.max(1)
            } else {
                line_length
            };
            let src_height_dbg = if self.src_height != 0 {
                self.src_height as usize
            } else {
                line_count
            };
            let tiled_size_dbg =
                tiled_size_bytes(src_width_bytes_dbg, src_height_dbg, block_height_log2_dbg);
            let remap_comp_size = ((self.remap_components >> 16) & 0x3) as usize + 1;
            let remap_n_src = ((self.remap_components >> 20) & 0x3) as usize + 1;
            let remap_n_dst = ((self.remap_components >> 24) & 0x3) as usize + 1;
            log::debug!(
                "MaxwellDma::blit_block_to_pitch DIAG: src_width={} src_height={} \
                 line_length={} line_count={} src_block_size={:#x} \
                 remap_components={:#x} remap_comp_size={} remap_n_src={} remap_n_dst={} \
                 src_width_bytes_used={} src_height_used={} bh_log2={} tiled_size_bytes={}",
                self.src_width,
                self.src_height,
                line_length,
                line_count,
                self.src_block_size,
                self.remap_components,
                remap_comp_size,
                remap_n_src,
                remap_n_dst,
                src_width_bytes_dbg,
                src_height_dbg,
                block_height_log2_dbg,
                tiled_size_dbg,
            );
        }
        self.blit_dst_by_src.insert(src_cpu, dst_cpu);
        let dst_pitch = self.pitch_out.max(line_length as u32) as usize;
        let block_height_log2 = ((self.src_block_size >> 4) & 0xF) as u32;
        let src_width_bytes = if self.src_width != 0 {
            (self.src_width as usize) * bytes_per_element.max(1)
        } else {
            line_length
        };
        let src_height = if self.src_height != 0 {
            self.src_height as usize
        } else {
            line_count
        };
        let tiled_size = tiled_size_bytes(src_width_bytes, src_height, block_height_log2);
        let mut tiled = vec![0u8; tiled_size];
        mem_read(src_cpu, &mut tiled);
        let linear = unswizzle_block_linear_bytes(
            &tiled,
            line_length,
            line_count,
            dst_pitch,
            src_width_bytes,
            src_height,
            block_height_log2,
            (self.src_origin_x as usize) * bytes_per_element.max(1),
            self.src_origin_y as usize,
        );
        if jumbo_dbg() {
            let (nz, ck) = sample_stats(&linear);
            log::warn!(
                "[jumbo] b2p dst={:#x} {}x{} pitch={} nz={} ck={:#x}",
                dst_cpu,
                line_length,
                line_count,
                dst_pitch,
                nz,
                ck
            );
        }
        for y in 0..line_count {
            let off = y * dst_pitch;
            if off >= dst_limit {
                break;
            }
            let n = line_length.min(dst_limit - off);
            mem_write(dst_cpu + off as u64, &linear[off..off + n]);
        }
    }
}

const GOB_W: usize = 64;
const GOB_H: usize = 8;
const GOB_SIZE: usize = 512;

pub(super) fn tiled_size_bytes(width_bytes: usize, height: usize, block_height_log2: u32) -> usize {
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_rows = (height + rows_per_block - 1) / rows_per_block;
    gobs_per_row * block_rows * block_height * GOB_SIZE
}

fn in_gob_offset(x: usize, y: usize) -> usize {
    ((x >> 5) & 1) * 256 + ((y >> 1) & 3) * 64 + ((x >> 4) & 1) * 32 + (y & 1) * 16 + (x & 15)
}

pub(crate) fn swizzle_block_linear(
    src_linear: &[u8],
    copy_width_bytes: usize,
    copy_height: usize,
    src_pitch: usize,
    dst_width_bytes: usize,
    dst_height: usize,
    block_height_log2: u32,
    origin_x: usize,
    origin_y: usize,
) -> Vec<u8> {
    let dst_size = tiled_size_bytes(dst_width_bytes, dst_height, block_height_log2);
    let mut dst = vec![0u8; dst_size];
    swizzle_block_linear_into(
        &mut dst,
        src_linear,
        copy_width_bytes,
        copy_height,
        src_pitch,
        dst_width_bytes,
        block_height_log2,
        origin_x,
        origin_y,
    );
    dst
}

pub(crate) fn swizzle_block_linear_into(
    dst: &mut [u8],
    src_linear: &[u8],
    copy_width_bytes: usize,
    copy_height: usize,
    src_pitch: usize,
    dst_width_bytes: usize,
    block_height_log2: u32,
    origin_x: usize,
    origin_y: usize,
) {
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (dst_width_bytes + GOB_W - 1) / GOB_W;
    let block_row_stride = gobs_per_row * block_height * GOB_SIZE;
    for y in 0..copy_height {
        let dst_y = origin_y + y;
        let block_y = dst_y / rows_per_block;
        let y_in_block = dst_y - block_y * rows_per_block;
        let gob_row_in_block = y_in_block / GOB_H;
        let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
        let block_row_off = block_y * block_row_stride;
        let src_row_off = y * src_pitch;
        let mut x = 0usize;
        while x < copy_width_bytes {
            let dst_byte_x = origin_x + x;
            let gob_col = dst_byte_x / GOB_W;
            let x_in_gob = dst_byte_x - gob_col * GOB_W;
            let gob_offset =
                block_row_off + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
            let dst_off = gob_offset + in_gob_offset(x_in_gob, y_in_gob);
            let src_off = src_row_off + x;
            let span = (16 - (dst_byte_x & 15)).min(copy_width_bytes - x);
            let copy_len = span
                .min(dst.len().saturating_sub(dst_off))
                .min(src_linear.len().saturating_sub(src_off));
            if copy_len != 0 {
                dst[dst_off..dst_off + copy_len]
                    .copy_from_slice(&src_linear[src_off..src_off + copy_len]);
            }
            x += span;
        }
    }
}

fn unswizzle_block_linear_bytes(
    src_tiled: &[u8],
    copy_width_bytes: usize,
    copy_height: usize,
    dst_pitch: usize,
    src_width_bytes: usize,
    _src_height: usize,
    block_height_log2: u32,
    origin_x: usize,
    origin_y: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_pitch * copy_height];
    let block_height = 1usize << block_height_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (src_width_bytes + GOB_W - 1) / GOB_W;
    let block_row_stride = gobs_per_row * block_height * GOB_SIZE;
    for y in 0..copy_height {
        let src_y = origin_y + y;
        let block_y = src_y / rows_per_block;
        let y_in_block = src_y - block_y * rows_per_block;
        let gob_row_in_block = y_in_block / GOB_H;
        let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
        let block_row_off = block_y * block_row_stride;
        let dst_row_off = y * dst_pitch;
        for x in 0..copy_width_bytes {
            let src_byte_x = origin_x + x;
            let gob_col = src_byte_x / GOB_W;
            let x_in_gob = src_byte_x - gob_col * GOB_W;
            let gob_offset =
                block_row_off + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
            let src_off = gob_offset + in_gob_offset(x_in_gob, y_in_gob);
            let dst_off = dst_row_off + x;
            if src_off < src_tiled.len() && dst_off < dst.len() {
                dst[dst_off] = src_tiled[src_off];
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::{
        in_gob_offset, swizzle_block_linear_into, tiled_size_bytes, GOB_H, GOB_SIZE, GOB_W,
    };

    fn swizzle_reference(
        dst: &mut [u8],
        src_linear: &[u8],
        copy_width_bytes: usize,
        copy_height: usize,
        src_pitch: usize,
        dst_width_bytes: usize,
        block_height_log2: u32,
        origin_x: usize,
        origin_y: usize,
    ) {
        let block_height = 1usize << block_height_log2;
        let rows_per_block = block_height * GOB_H;
        let gobs_per_row = dst_width_bytes.div_ceil(GOB_W);
        let block_row_stride = gobs_per_row * block_height * GOB_SIZE;
        for y in 0..copy_height {
            let dst_y = origin_y + y;
            let block_y = dst_y / rows_per_block;
            let y_in_block = dst_y - block_y * rows_per_block;
            let gob_row_in_block = y_in_block / GOB_H;
            let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
            let block_row_off = block_y * block_row_stride;
            let src_row_off = y * src_pitch;
            for x in 0..copy_width_bytes {
                let dst_byte_x = origin_x + x;
                let gob_col = dst_byte_x / GOB_W;
                let x_in_gob = dst_byte_x - gob_col * GOB_W;
                let gob_offset =
                    block_row_off + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
                let dst_off = gob_offset + in_gob_offset(x_in_gob, y_in_gob);
                let src_off = src_row_off + x;
                if dst_off < dst.len() && src_off < src_linear.len() {
                    dst[dst_off] = src_linear[src_off];
                }
            }
        }
    }

    #[test]
    fn chunked_swizzle_matches_byte_reference() {
        for block_height_log2 in 0..=5 {
            for (dst_width, dst_height, origin_x, origin_y, copy_width, copy_height) in [
                (17, 9, 0, 0, 17, 9),
                (65, 33, 1, 3, 61, 27),
                (137, 73, 15, 7, 119, 61),
                (320, 180, 32, 16, 257, 129),
            ] {
                let src_pitch = copy_width + 13;
                let mut src = vec![0u8; src_pitch * copy_height];
                for (index, byte) in src.iter_mut().enumerate() {
                    *byte = index.wrapping_mul(37).wrapping_add(11) as u8;
                }
                let dst_size = tiled_size_bytes(dst_width, dst_height, block_height_log2);
                let mut expected = vec![0xa5; dst_size];
                let mut actual = expected.clone();
                swizzle_reference(
                    &mut expected,
                    &src,
                    copy_width,
                    copy_height,
                    src_pitch,
                    dst_width,
                    block_height_log2,
                    origin_x,
                    origin_y,
                );
                swizzle_block_linear_into(
                    &mut actual,
                    &src,
                    copy_width,
                    copy_height,
                    src_pitch,
                    dst_width,
                    block_height_log2,
                    origin_x,
                    origin_y,
                );
                assert_eq!(actual, expected);
            }
        }
    }
}
