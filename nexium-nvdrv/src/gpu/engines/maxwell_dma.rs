use super::super::GpuMappings;
use nexium_gpu::rt_cache::RtKey;
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

fn dma_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_DMA_TRACE").is_some())
}

fn dma_trace_cpu_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::env::var("NEXIUM_DMA_TRACE_CPU")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok())
    })
}

fn dma_trace_gpu_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::env::var("NEXIUM_DMA_TRACE_GPU")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok())
    })
}

fn dma_semaphore_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_DMA_SEMAPHORE_TRACE").is_some())
}

fn virtual_rt_dma_disabled() -> bool {
    #[cfg(test)]
    {
        false
    }
    #[cfg(not(test))]
    {
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("NEXIUM_VIRTUAL_RT_DMA").is_none()
                || std::env::var_os("NEXIUM_NO_VIRTUAL_RT_DMA").is_some()
        })
    }
}

fn exact_rt_copy_subresource_supported(
    src_depth: u32,
    src_layer: u32,
    dst_depth: u32,
    dst_layer: u32,
) -> bool {
    src_depth <= 1 && dst_depth <= 1 && src_layer == 0 && dst_layer == 0
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

fn invalidate_block_linear_aliases(
    mappings: &GpuMappings,
    dst_gpu: u64,
    dst_cpu: u64,
    written_span: usize,
) {
    if written_span == 0 {
        return;
    }
    let written_span = written_span as u64;
    let mut aliases = mappings.gpu_regions_for_cpu_range(dst_cpu, written_span);
    if !aliases
        .iter()
        .any(|(alias, available)| *alias == dst_gpu && *available >= written_span)
    {
        aliases.push((dst_gpu, written_span));
    }
    aliases.sort_unstable();
    aliases.dedup();
    for (alias, available) in aliases {
        nexium_gpu::pitch_oracle::clear_pitch_range(alias, available);
        nexium_gpu::tex_invalidate::bump_region(alias, available);
    }
}

pub const MAXWELL_DMA_CLASS: u32 = 0xB0B5;

const M_SEMAPHORE_ADDRESS_UPPER: u32 = 0x90;
const M_SEMAPHORE_ADDRESS_LOWER: u32 = 0x91;
const M_SEMAPHORE_PAYLOAD: u32 = 0x92;
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

pub(crate) fn method_requires_hard_boundary(method: u32) -> bool {
    method == M_LAUNCH_DMA
}

const LAUNCH_SRC_LAYOUT_BIT: u32 = 7;
const LAUNCH_DST_LAYOUT_BIT: u32 = 8;
const LAUNCH_MULTI_LINE_BIT: u32 = 9;
const LAUNCH_REMAP_ENABLE_BIT: u32 = 10;
const LAYOUT_BLOCK_LINEAR: u32 = 0;
const LAYOUT_PITCH: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct RtCopyRecord {
    pub source: RtKey,
    pub source_stamp: u64,
    pub size: u64,
    pub generation: u64,
    pub dst_layout: u32,
    pub dst_block_size: u32,
    pub bpp: usize,
    pub gpu_resolved_block: bool,
    pub source_may_advance: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactPresentSourceToken {
    pub source: RtKey,
    pub source_stamp: u64,
    pub source_may_advance: bool,
    destination_va: u64,
    destination_size: u64,
    destination_generation: u64,
}

impl ExactPresentSourceToken {
    pub fn destination_is_current(self) -> bool {
        self.destination_size != 0
            && nexium_gpu::tex_invalidate::region_gen_range(
                self.destination_va,
                self.destination_size,
            ) == self.destination_generation
    }
}

struct PendingRtSource {
    provenance: Option<RtSourceProvenance>,
    src_gpu: u64,
    src_block_size: u32,
    virtual_present: bool,
    virtual_block: bool,
    gpu_resolved_block: bool,
    linear: Option<LinearRtSource>,
}

#[derive(Clone, Copy)]
struct RtSourceProvenance {
    source: RtKey,
    source_stamp: u64,
    bpp: usize,
}

struct LinearRtSource {
    bytes: Vec<u8>,
    width_bytes: usize,
    height: usize,
}

#[derive(Default)]
pub struct MaxwellDma {
    semaphore_address_upper: u32,
    semaphore_address_lower: u32,
    semaphore_payload: u32,
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
    src_depth: u32,
    src_layer: u32,
    src_origin_x: u32,
    src_origin_y: u32,
    dst_block_size: u32,
    dst_width: u32,
    dst_height: u32,
    dst_depth: u32,
    dst_layer: u32,
    dst_origin_x: u32,
    dst_origin_y: u32,
    pub blit_count: u64,

    rt_copy_sources: HashMap<u64, RtCopyRecord>,
    pending_rt_source: Option<PendingRtSource>,
    present_epoch: u64,
    present_sources: HashMap<(u32, u64, u32, u32), u64>,
    present_destinations: HashMap<(u32, u64, u32, u32), u64>,

    pub last_tiled_dst_cpu: u64,

    pub last_tiled_dst_bh_log2: u32,

    pub last_tiled_dst_stride: u32,

    pub last_tiled_dst_height: u32,

    pub draw_texture_blits: u64,

    clamp_log_count: u64,
    guest_write_range: Option<(u64, u64)>,
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
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        match method {
            M_SEMAPHORE_ADDRESS_UPPER => self.semaphore_address_upper = arg,
            M_SEMAPHORE_ADDRESS_LOWER => self.semaphore_address_lower = arg,
            M_SEMAPHORE_PAYLOAD => self.semaphore_payload = arg,
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
            M_SET_SRC_DEPTH => self.src_depth = arg,
            M_SET_SRC_LAYER => self.src_layer = arg,
            M_SET_SRC_ORIGIN => {
                self.src_origin_x = arg & 0xFFFF;
                self.src_origin_y = (arg >> 16) & 0xFFFF;
            }
            M_SET_DST_BLOCK_SIZE => self.dst_block_size = arg,
            M_SET_DST_WIDTH => self.dst_width = arg,
            M_SET_DST_HEIGHT => self.dst_height = arg,
            M_SET_DST_DEPTH => self.dst_depth = arg,
            M_SET_DST_LAYER => self.dst_layer = arg,
            M_SET_DST_ORIGIN => {
                self.dst_origin_x = arg & 0xFFFF;
                self.dst_origin_y = (arg >> 16) & 0xFFFF;
            }
            M_LAUNCH_DMA => {
                self.trace_semaphore_launch(arg, mappings);
                self.launch_dma(arg, mappings, mem_read, mem_write, mem_copy);
            }
            _ => {
                log::trace!("MaxwellDma: unhandled method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn trace_semaphore_launch(&self, flags: u32, mappings: &GpuMappings) {
        let semaphore_type = (flags >> 3) & 3;
        if semaphore_type == 0 || !dma_semaphore_trace_enabled() {
            return;
        }
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let count = COUNT.fetch_add(1, Ordering::Relaxed);
        if count >= 32 && count % 128 != 0 {
            return;
        }
        let gpu_va = ((u64::from(self.semaphore_address_upper) & 0xff) << 32)
            | u64::from(self.semaphore_address_lower);
        log::warn!(
            "[dma-semaphore] #{} flags={:#x} type={} regs=[{:#x},{:#x},{:#x}] gpu_va={:#x} cpu={:#x?} src={:#x} dst={:#x} line_units={} line_count={}",
            count,
            flags,
            semaphore_type,
            self.semaphore_address_upper,
            self.semaphore_address_lower,
            self.semaphore_payload,
            gpu_va,
            mappings.cpu_address_for(gpu_va),
            self.src_addr(),
            self.dst_addr(),
            self.line_length_in,
            self.line_count,
        );
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

    pub fn take_guest_write_range(&mut self) -> Option<(u64, u64)> {
        self.guest_write_range.take()
    }

    fn rt_copy_record(&self, gpu_va: u64) -> Option<RtCopyRecord> {
        let record = self.rt_copy_sources.get(&gpu_va)?;
        let generation = nexium_gpu::tex_invalidate::region_gen_range(gpu_va, record.size);
        (generation == record.generation).then_some(*record)
    }

    fn invalidate_rt_copy_aliases(
        &mut self,
        mappings: &GpuMappings,
        destination_va: u64,
        size: u64,
    ) {
        let Some((destination_cpu, destination_available)) = mappings.cpu_range_for(destination_va)
        else {
            self.rt_copy_sources.remove(&destination_va);
            return;
        };
        let destination_size = size.min(destination_available);
        if destination_size == 0 {
            self.rt_copy_sources.remove(&destination_va);
            return;
        }
        let destination_end = destination_cpu.saturating_add(destination_size);
        self.rt_copy_sources.retain(|&record_va, record| {
            if record_va == destination_va {
                return false;
            }
            let Some((record_cpu, record_available)) = mappings.cpu_range_for(record_va) else {
                nexium_gpu::tex_invalidate::bump_region(record_va, record.size);
                return false;
            };
            let record_size = record.size.min(record_available);
            let record_end = record_cpu.saturating_add(record_size);
            let disjoint = destination_cpu >= record_end || record_cpu >= destination_end;
            if !disjoint {
                nexium_gpu::tex_invalidate::bump_region(record_va, record.size);
            }
            disjoint
        });
    }

    fn pending_copy_source(&self, gpu_va: u64) -> Option<PendingRtSource> {
        let record = self.rt_copy_record(gpu_va)?;
        Some(PendingRtSource {
            provenance: Some(RtSourceProvenance {
                source: record.source,
                source_stamp: record.source_stamp,
                bpp: record.bpp,
            }),
            src_gpu: gpu_va,
            src_block_size: record.dst_block_size,
            virtual_present: false,
            virtual_block: record.dst_layout == LAYOUT_BLOCK_LINEAR
                && (record.gpu_resolved_block || !virtual_rt_dma_disabled()),
            gpu_resolved_block: record.gpu_resolved_block,
            linear: None,
        })
    }

    pub fn rt_copy_source(&self, gpu_va: u64) -> Option<(RtKey, u64)> {
        let record = self.rt_copy_record(gpu_va)?;
        Some((record.source, record.source_stamp))
    }

    pub fn exact_present_source(
        &self,
        gpu_va: u64,
        width: u32,
        height: u32,
    ) -> Option<(RtKey, u64, bool)> {
        self.exact_present_source_token(gpu_va, width, height)
            .map(|token| (token.source, token.source_stamp, token.source_may_advance))
    }

    pub fn exact_present_source_token(
        &self,
        gpu_va: u64,
        width: u32,
        height: u32,
    ) -> Option<ExactPresentSourceToken> {
        let record = self.rt_copy_record(gpu_va)?;
        let source = record.source;
        (record.bpp == 4
            && record.source_stamp != 0
            && source.width == width
            && source.height == height
            && source.depth == 1
            && !source.is_3d)
            .then_some(ExactPresentSourceToken {
                source,
                source_stamp: record.source_stamp,
                source_may_advance: record.source_may_advance,
                destination_va: gpu_va,
                destination_size: record.size,
                destination_generation: record.generation,
            })
    }

    pub fn stage_rt_source(
        &mut self,
        flags: u32,
        mappings: &GpuMappings,
        renderer: &nexium_gpu::renderer::Renderer,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.pending_rt_source = None;
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
        if !crate::gpu::vk_dispatch::sync_render_thread() {
            log::warn!(
                "MaxwellDma: could not drain render thread before staging RT {:#x}",
                src_gpu
            );
            return;
        }
        let inherited_rt_source = self.pending_copy_source(src_gpu);
        let exact_source = renderer.render_target_at_va(nvmap, src_gpu);
        let inherited_gpu_source = inherited_rt_source.as_ref().and_then(|pending| {
            let provenance = pending.provenance?;
            (pending.virtual_block
                && pending.gpu_resolved_block
                && renderer.render_target_stamp(provenance.source) == Some(provenance.source_stamp))
            .then_some((provenance.source, provenance.source_stamp))
        });
        let block_source = exact_source.or(inherited_gpu_source);
        let use_inherited_metadata = block_source.is_none()
            && inherited_rt_source.as_ref().is_some_and(|pending| {
                pending.virtual_block && (!pending.gpu_resolved_block || !virtual_rt_dma_disabled())
            });
        if use_inherited_metadata {
            self.pending_rt_source = inherited_rt_source;
            return;
        }
        if dst_layout == LAYOUT_BLOCK_LINEAR {
            let dst_gpu = self.dst_addr();
            let remap_enable = ((flags >> LAUNCH_REMAP_ENABLE_BIT) & 1) != 0;
            let multi_line = ((flags >> LAUNCH_MULTI_LINE_BIT) & 1) != 0;
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
            let selectors = [dst_x_sel, dst_y_sel, dst_z_sel, dst_w_sel];
            let identity_remap = !remap_enable
                || (num_src_components == num_dst_components
                    && selectors[..num_dst_components]
                        .iter()
                        .enumerate()
                        .all(|(index, &selector)| selector == index as u32));
            let source_row_bytes = (self.line_length_in as usize)
                .checked_mul(component_size.checked_mul(num_src_components).unwrap_or(0));
            let destination_row_bytes = (self.line_length_in as usize)
                .checked_mul(component_size.checked_mul(num_dst_components).unwrap_or(0));
            let virtual_source = block_source.and_then(|(source, source_stamp)| {
                let width = source.width as usize;
                let height = source.height as usize;
                let source_row_bytes = source_row_bytes?;
                let destination_row_bytes = destination_row_bytes?;
                let bpp = source_row_bytes.checked_div(width)?;
                let plan = active_gob_copy_plan(
                    source_row_bytes,
                    height,
                    (self.src_block_size >> 4) & 0xF,
                )?;
                let direct_source_matches =
                    exact_source.is_some_and(|candidate| candidate == (source, source_stamp));
                let inherited_source_matches =
                    inherited_rt_source.as_ref().is_some_and(|pending| {
                        pending.virtual_block
                            && pending.gpu_resolved_block
                            && pending.src_gpu == src_gpu
                            && pending.src_block_size == self.src_block_size
                            && pending.provenance.is_some_and(|provenance| {
                                provenance.source == source
                                    && provenance.source_stamp == source_stamp
                                    && provenance.bpp == bpp
                            })
                    });
                ((direct_source_matches || inherited_source_matches)
                    && source.depth == 1
                    && !source.is_3d
                    && exact_rt_copy_subresource_supported(
                        self.src_depth,
                        self.src_layer,
                        self.dst_depth,
                        self.dst_layer,
                    )
                    && width != 0
                    && height != 0
                    && matches!(bpp, 1 | 2 | 4 | 8 | 16)
                    && source_row_bytes == width.checked_mul(bpp)?
                    && destination_row_bytes == source_row_bytes
                    && multi_line
                    && identity_remap
                    && self.src_block_size == self.dst_block_size
                    && self.src_width == source.width
                    && self.dst_width == source.width
                    && self.src_height == source.height
                    && self.dst_height == source.height
                    && self.src_origin_x == 0
                    && self.src_origin_y == 0
                    && self.dst_origin_x == 0
                    && self.dst_origin_y == 0
                    && self.line_count as usize == height
                    && mappings
                        .cpu_range_for(dst_gpu)
                        .is_some_and(|(_, limit)| limit >= plan.mapped_extent as u64))
                .then_some((source, source_stamp, bpp, plan.mapped_extent))
            });
            if let Some((source, source_stamp, bpp, mapped_extent)) = virtual_source {
                let resolved = mappings.nvmap_id_for(dst_gpu).and_then(|_| {
                    match renderer.resolve_rt_copy_exact(source, source_stamp, bpp, dst_gpu) {
                        Ok(copy) => copy,
                        Err(error) => {
                            log::warn!(
                                "MaxwellDma: exact RT copy {:#x}->{:#x} failed: {}",
                                src_gpu,
                                dst_gpu,
                                error
                            );
                            None
                        }
                    }
                });
                let virtual_provenance = resolved
                    .map(|(destination, destination_stamp)| (destination, destination_stamp, true))
                    .or_else(|| {
                        (!virtual_rt_dma_disabled()).then_some((source, source_stamp, false))
                    });
                if let Some((provenance_source, provenance_stamp, gpu_resolved_block)) =
                    virtual_provenance
                {
                    self.pending_rt_source = Some(PendingRtSource {
                        provenance: Some(RtSourceProvenance {
                            source: provenance_source,
                            source_stamp: provenance_stamp,
                            bpp,
                        }),
                        src_gpu,
                        src_block_size: self.src_block_size,
                        virtual_present: false,
                        virtual_block: true,
                        gpu_resolved_block,
                        linear: None,
                    });
                    if crate::gpu::pusher::kickprof::enabled() {
                        static VIRTUAL_BLOCK_LOGS: AtomicU64 = AtomicU64::new(0);
                        if VIRTUAL_BLOCK_LOGS.fetch_add(1, Ordering::Relaxed) < 12 {
                            log::warn!(
                                "[dma-rt] {} block src={:#x} dst={:#x} nvmap={} {}x{} bpp={} extent={}",
                                if gpu_resolved_block { "resolved" } else { "metadata" },
                                src_gpu,
                                dst_gpu,
                                nvmap,
                                source.width,
                                source.height,
                                bpp,
                                mapped_extent,
                            );
                        }
                    }
                    return;
                }
            }
        }
        let virtual_present_disabled = {
            static DISABLED: OnceLock<bool> = OnceLock::new();
            *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_NO_VIRTUAL_PRESENT_DMA").is_some())
        };
        if !virtual_present_disabled && dst_layout == LAYOUT_PITCH {
            let dst_gpu = self.dst_addr();
            let remap_enable = ((flags >> LAUNCH_REMAP_ENABLE_BIT) & 1) != 0;
            let multi_line = ((flags >> LAUNCH_MULTI_LINE_BIT) & 1) != 0;
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
            let selectors = [dst_x_sel, dst_y_sel, dst_z_sel, dst_w_sel];
            let identity_remap = !remap_enable
                || (num_src_components == num_dst_components
                    && selectors[..num_dst_components]
                        .iter()
                        .enumerate()
                        .all(|(index, &selector)| selector == index as u32));
            let src_group_bytes = component_size.saturating_mul(num_src_components);
            let dst_group_bytes = component_size.saturating_mul(num_dst_components);
            let source_row_bytes = (self.line_length_in as usize).saturating_mul(src_group_bytes);
            let destination_row_bytes =
                (self.line_length_in as usize).saturating_mul(dst_group_bytes);
            if let Some((source, source_stamp)) = exact_source.filter(|(source, _)| {
                let width = source.width as usize;
                let height = source.height as usize;
                let bpp = source_row_bytes.checked_div(width).unwrap_or(0);
                source.nvmap_id == nvmap
                    && source.gpu_va == src_gpu
                    && width != 0
                    && bpp == 4
                    && source_row_bytes == width.saturating_mul(bpp)
                    && destination_row_bytes == source_row_bytes
                    && multi_line
                    && identity_remap
                    && self.src_origin_x == 0
                    && self.src_origin_y == 0
                    && self.dst_origin_x == 0
                    && self.dst_origin_y == 0
                    && self.src_width == source.width
                    && self.src_height == source.height
                    && self.line_count as usize == height
                    && self.pitch_out as usize >= destination_row_bytes
                    && mappings.nvmap_id_for(dst_gpu).is_some()
                    && self.is_recent_present_copy(
                        nvmap,
                        src_gpu,
                        dst_gpu,
                        source.width,
                        source.height,
                    )
            }) {
                let bpp = source_row_bytes / source.width as usize;
                self.pending_rt_source = Some(PendingRtSource {
                    provenance: Some(RtSourceProvenance {
                        source,
                        source_stamp,
                        bpp,
                    }),
                    src_gpu,
                    src_block_size: self.src_block_size,
                    virtual_present: true,
                    virtual_block: false,
                    gpu_resolved_block: false,
                    linear: None,
                });
                if crate::gpu::pusher::kickprof::enabled() {
                    static VIRTUAL_LOGS: AtomicU64 = AtomicU64::new(0);
                    if VIRTUAL_LOGS.fetch_add(1, Ordering::Relaxed) < 12 {
                        log::warn!(
                            "[dma-present] live src={:#x} dst={:#x} nvmap={} {}x{} row={} pitch={} floor_stamp={}",
                            src_gpu,
                            dst_gpu,
                            nvmap,
                            source.width,
                            source.height,
                            source_row_bytes,
                            self.pitch_out,
                            source_stamp,
                        );
                    }
                }
                return;
            }
        }
        let Some((src_cpu, limit)) = mappings.cpu_range_for(src_gpu) else {
            if jumbo_dbg() {
                log::warn!("[jumbo] stage MISS no-cpu-range src={:#x}", src_gpu);
            }
            return;
        };
        let bh_log2 = ((self.src_block_size >> 4) & 0xF) as u32;
        if let Some((kw, kh, bpp, raw)) = renderer.readback_target_raw(nvmap, src_gpu) {
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
            let staged = mem_write(src_cpu, &tiled[..n]);
            if crate::gpu::pusher::kickprof::enabled() {
                static STAGE_GATE_LOGS: AtomicU64 = AtomicU64::new(0);
                if STAGE_GATE_LOGS.fetch_add(1, Ordering::Relaxed) < 12 {
                    log::warn!(
                        "[dma-rt-gate] stage src={:#x} dst_layout={} raw={}x{}x{} src_regs={}x{} block={:#x} staged={} bytes={}/{} limit={}",
                        src_gpu,
                        dst_layout,
                        kw,
                        kh,
                        bpp,
                        self.src_width,
                        self.src_height,
                        self.src_block_size,
                        staged,
                        n,
                        tiled.len(),
                        limit,
                    );
                }
            }
            if staged {
                let provenance = exact_source.map(|(source, source_stamp)| RtSourceProvenance {
                    source,
                    source_stamp,
                    bpp,
                });
                let linear = if n == tiled.len() {
                    Some(LinearRtSource {
                        bytes: raw,
                        width_bytes,
                        height: kh as usize,
                    })
                } else {
                    None
                };
                if provenance.is_some() || linear.is_some() {
                    self.pending_rt_source = Some(PendingRtSource {
                        provenance,
                        src_gpu,
                        src_block_size: self.src_block_size,
                        virtual_present: false,
                        virtual_block: false,
                        gpu_resolved_block: false,
                        linear,
                    });
                }
            }
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
        let Some(rgba) = renderer.readback_target(nvmap, kw, kh) else {
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

    pub fn register_present_surface(
        &mut self,
        nvmap_id: u32,
        destination_vas: &[u64],
        width: u32,
        height: u32,
        source_vas: &[u64],
    ) {
        const WINDOW: u64 = 8;

        self.present_epoch = self.present_epoch.saturating_add(1);
        let epoch = self.present_epoch;
        for &destination_va in destination_vas {
            if destination_va != 0 && width != 0 && height != 0 {
                self.present_destinations
                    .insert((nvmap_id, destination_va, width, height), epoch);
            }
        }
        for &source_va in source_vas {
            if source_va != 0 && width != 0 && height != 0 {
                self.present_sources
                    .insert((nvmap_id, source_va, width, height), epoch);
            }
        }
        if crate::gpu::pusher::kickprof::enabled() {
            static REGISTRATION_LOGS: AtomicU64 = AtomicU64::new(0);
            if REGISTRATION_LOGS.fetch_add(1, Ordering::Relaxed) < 12 {
                log::warn!(
                    "[dma-present] register epoch={} nvmap={} dst={:x?} {}x{} sources={:x?}",
                    epoch,
                    nvmap_id,
                    destination_vas,
                    width,
                    height,
                    source_vas,
                );
            }
        }
        let oldest = epoch.saturating_sub(WINDOW);
        self.present_sources.retain(|_, seen| *seen >= oldest);
        self.present_destinations.retain(|_, seen| *seen >= oldest);
    }

    fn is_recent_present_copy(
        &self,
        nvmap_id: u32,
        source_va: u64,
        destination_va: u64,
        width: u32,
        height: u32,
    ) -> bool {
        const WINDOW: u64 = 8;

        let recent = |seen: Option<&u64>| {
            seen.is_some_and(|seen| self.present_epoch.saturating_sub(*seen) <= WINDOW)
        };
        recent(
            self.present_sources
                .get(&(nvmap_id, source_va, width, height)),
        ) && recent(
            self.present_destinations
                .get(&(nvmap_id, destination_va, width, height)),
        )
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
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        self.guest_write_range = None;
        let src_layout = (flags >> LAUNCH_SRC_LAYOUT_BIT) & 1;
        let dst_layout = (flags >> LAUNCH_DST_LAYOUT_BIT) & 1;
        let multi_line = (flags >> LAUNCH_MULTI_LINE_BIT) & 1 != 0;
        let remap_enable = (flags >> LAUNCH_REMAP_ENABLE_BIT) & 1 != 0;
        let pending_rt_source = self.pending_rt_source.take();

        let src_gpu = self.src_addr();
        let dst_gpu = self.dst_addr();
        let map_started = crate::gpu::pusher::kickprof::start();
        static DMA_DROPPED: AtomicU64 = AtomicU64::new(0);
        let Some((src_cpu, src_limit)) = mappings.cpu_range_for(src_gpu) else {
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
        let src_limit = src_limit as usize;
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
        let gpu_watch = dma_trace_gpu_address();
        if let Some(watch) = dma_trace_cpu_address().or(gpu_watch) {
            let covers = |base: u64, limit: usize| watch >= base && watch < base.wrapping_add(limit as u64);
            let watched = if gpu_watch.is_some() && dma_trace_cpu_address().is_none() {
                covers(src_gpu, src_limit) || covers(dst_gpu, dst_limit)
            } else {
                covers(src_cpu, src_limit) || covers(dst_cpu, dst_limit)
            };
            if watched {
                log::warn!(
                    "[dma-trace-cpu] #{} src_gpu={:#x} cpu={:#x} limit={:#x} dst_gpu={:#x} cpu={:#x} limit={:#x} flags={:#x} units={} lines={} pitch={}->{} size={}x{}->{}x{} block={:#x}->{:#x} rt_source={}",
                    self.blit_count,
                    src_gpu,
                    src_cpu,
                    src_limit,
                    dst_gpu,
                    dst_cpu,
                    dst_limit,
                    flags,
                    self.line_length_in,
                    self.line_count,
                    self.pitch_in,
                    self.pitch_out,
                    self.src_width,
                    self.src_height,
                    self.dst_width,
                    self.dst_height,
                    self.src_block_size,
                    self.dst_block_size,
                    pending_rt_source.is_some()
                );
            }
        }
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::DMA_MAP, map_started);

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
        let remap_selectors = [dst_x_sel, dst_y_sel, dst_z_sel, dst_w_sel];
        let identity_remap = !remap_enable
            || (num_src_components == num_dst_components
                && remap_selectors[..num_dst_components]
                    .iter()
                    .enumerate()
                    .all(|(index, &selector)| selector == index as u32));

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
        let source_compatible = pending_rt_source.as_ref().is_some_and(|pending| {
            pending.provenance.is_some_and(|provenance| {
                identity_remap
                    && src_layout == LAYOUT_BLOCK_LINEAR
                    && pending.src_gpu == src_gpu
                    && pending.src_block_size == self.src_block_size
                    && self.src_origin_x == 0
                    && self.src_origin_y == 0
                    && line_length_src == self.src_width.max(1) as usize * provenance.bpp
                    && line_count == self.src_height.max(1) as usize
            })
        });
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
        let linear_rt_source_compatible = pending_rt_source.as_ref().is_some_and(|pending| {
            pending.linear.as_ref().is_some_and(|linear| {
                src_layout == LAYOUT_BLOCK_LINEAR
                    && pending.src_gpu == src_gpu
                    && pending.src_block_size == self.src_block_size
                    && identity_remap
                    && self.src_origin_x == 0
                    && self.src_origin_y == 0
                    && line_length_src == linear.width_bytes
                    && line_count == linear.height
                    && src_width_bytes == linear.width_bytes
                    && src_height == linear.height
                    && linear
                        .width_bytes
                        .checked_mul(linear.height)
                        .is_some_and(|required| linear.bytes.len() >= required)
            })
        });
        if crate::gpu::pusher::kickprof::enabled()
            && pending_rt_source
                .as_ref()
                .and_then(|pending| pending.linear.as_ref())
                .is_some()
            && !linear_rt_source_compatible
        {
            static REJECT_LOGS: AtomicU64 = AtomicU64::new(0);
            if REJECT_LOGS.fetch_add(1, Ordering::Relaxed) < 12 {
                let pending = pending_rt_source.as_ref().unwrap();
                let linear = pending.linear.as_ref().unwrap();
                log::warn!(
                    "[dma-rt-gate] reject src={:#x}/{:#x} block={:#x}/{:#x} layout={}->{} remap={} origin={},{} line={}x{} src={}x{} raw={}x{}",
                    pending.src_gpu,
                    src_gpu,
                    pending.src_block_size,
                    self.src_block_size,
                    src_layout,
                    dst_layout,
                    remap_enable,
                    self.src_origin_x,
                    self.src_origin_y,
                    line_length_src,
                    line_count,
                    src_width_bytes,
                    src_height,
                    linear.width_bytes,
                    linear.height,
                );
            }
        }
        let mut copied_size = 0u64;

        if dma_trace_enabled() && self.blit_count < 64 {
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
                let written_span = self.blit_pitch_to_block(
                    src_cpu,
                    src_limit,
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
                if written_span != 0 {
                    invalidate_block_linear_aliases(mappings, dst_gpu, dst_cpu, written_span);
                    self.guest_write_range = Some((dst_cpu, written_span as u64));
                    copied_size = written_span as u64;
                }
            }
            (LAYOUT_BLOCK_LINEAR, LAYOUT_PITCH) => {
                let dst_pitch = (self.pitch_out as usize).max(line_length_src);
                let copy_size = dst_pitch.saturating_mul(line_count);
                let virtual_present = source_compatible
                    && pending_rt_source
                        .as_ref()
                        .is_some_and(|pending| pending.virtual_present)
                    && identity_remap
                    && line_length_dst == line_length_src
                    && self.dst_origin_x == 0
                    && self.dst_origin_y == 0
                    && copy_size <= dst_limit;
                let meta_started = crate::gpu::pusher::kickprof::start();
                nexium_gpu::pitch_oracle::record_pitch_dst(dst_gpu, copy_size as u64);
                crate::gpu::pusher::kickprof::add(
                    crate::gpu::pusher::kickprof::DMA_META,
                    meta_started,
                );
                let used_linear_rt = if virtual_present {
                    let started = crate::gpu::pusher::kickprof::start();
                    crate::gpu::pusher::kickprof::add_sized(
                        crate::gpu::pusher::kickprof::DMA_VIRTUAL,
                        started,
                        line_length_src.saturating_mul(line_count),
                    );
                    true
                } else if linear_rt_source_compatible {
                    pending_rt_source
                        .as_ref()
                        .and_then(|pending| pending.linear.as_ref())
                        .is_some_and(|linear| {
                            let started = crate::gpu::pusher::kickprof::start();
                            let copied = self.blit_linear_to_pitch(
                                &linear.bytes,
                                dst_cpu,
                                dst_limit,
                                line_length_src,
                                line_count,
                                mem_write,
                            );
                            if copied {
                                crate::gpu::pusher::kickprof::add_sized(
                                    crate::gpu::pusher::kickprof::DMA_RT_LINEAR,
                                    started,
                                    line_length_src.saturating_mul(line_count),
                                );
                            }
                            copied
                        })
                } else {
                    false
                };
                if !used_linear_rt {
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
                }
                let meta_started = crate::gpu::pusher::kickprof::start();
                nexium_gpu::tex_invalidate::bump_region(dst_gpu, copy_size as u64);
                crate::gpu::pusher::kickprof::add(
                    crate::gpu::pusher::kickprof::DMA_META,
                    meta_started,
                );
                copied_size = copy_size as u64;
            }
            (LAYOUT_PITCH, LAYOUT_PITCH) => {
                self.blit_pitch_to_pitch(
                    src_cpu,
                    dst_cpu,
                    dst_limit,
                    line_length_src,
                    line_count,
                    !remap_enable,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
                let dst_pitch = (self.pitch_out as usize).max(line_length_src);
                let meta_started = crate::gpu::pusher::kickprof::start();
                nexium_gpu::tex_invalidate::bump_region(dst_gpu, (dst_pitch * line_count) as u64);
                crate::gpu::pusher::kickprof::add_sized(
                    crate::gpu::pusher::kickprof::DMA_META,
                    meta_started,
                    dst_pitch.saturating_mul(line_count),
                );
                copied_size = (dst_pitch * line_count) as u64;
            }
            (LAYOUT_BLOCK_LINEAR, LAYOUT_BLOCK_LINEAR) => {
                let layout_plan = (identity_remap
                    && self.src_block_size == self.dst_block_size
                    && self.src_width != 0
                    && self.src_width == self.dst_width
                    && self.src_height != 0
                    && self.src_height == self.dst_height
                    && self.src_origin_x == 0
                    && self.src_origin_y == 0
                    && self.dst_origin_x == 0
                    && self.dst_origin_y == 0
                    && src_bytes_per_element == dst_bytes_per_element)
                    .then(|| {
                        (self.src_width as usize)
                            .checked_mul(src_bytes_per_element)
                            .filter(|&width_bytes| {
                                line_length_src == width_bytes
                                    && line_length_dst == width_bytes
                                    && line_count == self.src_height as usize
                            })
                            .and_then(|width_bytes| {
                                active_gob_copy_plan(
                                    width_bytes,
                                    self.src_height as usize,
                                    (self.src_block_size >> 4) & 0xF,
                                )
                            })
                    })
                    .flatten();
                let virtual_extent = (source_compatible
                    && pending_rt_source.as_ref().is_some_and(|pending| {
                        pending.virtual_block
                            && (pending.gpu_resolved_block || !virtual_rt_dma_disabled())
                            && pending.provenance.is_some_and(|provenance| {
                                provenance.bpp == src_bytes_per_element
                                    && provenance.source.width == self.src_width
                                    && provenance.source.height == self.src_height
                                    && provenance.source.depth == 1
                                    && !provenance.source.is_3d
                            })
                    }))
                .then(|| layout_plan.as_ref())
                .flatten()
                .filter(|plan| dst_limit >= plan.mapped_extent)
                .map(|plan| (plan.mapped_extent, plan.active_bytes));
                let used_virtual = virtual_extent.is_some();
                if let Some((mapped_extent, active_bytes)) = virtual_extent {
                    let started = crate::gpu::pusher::kickprof::start();
                    nexium_gpu::pitch_oracle::clear_pitch_range(dst_gpu, mapped_extent as u64);
                    nexium_gpu::tex_invalidate::bump_region(dst_gpu, mapped_extent as u64);
                    crate::gpu::pusher::kickprof::add_sized(
                        crate::gpu::pusher::kickprof::DMA_VIRTUAL,
                        started,
                        active_bytes,
                    );
                }
                let direct_plan = if used_virtual {
                    None
                } else {
                    layout_plan.filter(|plan| {
                        active_gob_copy_ranges_valid(plan, src_cpu, src_limit, dst_cpu, dst_limit)
                    })
                };
                let direct_started = direct_plan
                    .as_ref()
                    .map(|_| crate::gpu::pusher::kickprof::start());
                let used_direct = direct_plan
                    .as_ref()
                    .is_some_and(|plan| copy_active_gobs(plan, src_cpu, dst_cpu, mem_copy));
                if used_direct {
                    let plan = direct_plan.as_ref().unwrap();
                    crate::gpu::pusher::kickprof::add_sized(
                        crate::gpu::pusher::kickprof::DMA_COPY,
                        direct_started.flatten(),
                        plan.active_bytes,
                    );
                    nexium_gpu::pitch_oracle::clear_pitch_range(dst_gpu, plan.mapped_extent as u64);
                    nexium_gpu::tex_invalidate::bump_region(dst_gpu, plan.mapped_extent as u64);
                }
                let linear_rt = linear_rt_source_compatible
                    .then(|| {
                        pending_rt_source
                            .as_ref()
                            .and_then(|pending| pending.linear.as_ref())
                    })
                    .flatten()
                    .filter(|linear| line_length_dst == linear.width_bytes);
                if !used_virtual && !used_direct {
                    let started = linear_rt.map(|_| crate::gpu::pusher::kickprof::start());
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
                        linear_rt.map(|linear| linear.bytes.as_slice()),
                        mem_read,
                        mem_write,
                    );
                    if let Some(started) = started.flatten() {
                        crate::gpu::pusher::kickprof::add_sized(
                            crate::gpu::pusher::kickprof::DMA_RT_LINEAR,
                            Some(started),
                            line_length_src.saturating_mul(line_count),
                        );
                    }
                }
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
                copied_size = tiled_size_bytes(dst_width_bytes, dst_height, dst_bh) as u64;
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
        if copied_size != 0 {
            self.invalidate_rt_copy_aliases(mappings, dst_gpu, copied_size);
        }
        if source_compatible && copied_size != 0 && copied_size <= dst_limit as u64 {
            if let Some(pending) = pending_rt_source {
                if let Some(provenance) = pending.provenance {
                    self.rt_copy_sources.insert(
                        dst_gpu,
                        RtCopyRecord {
                            source: provenance.source,
                            source_stamp: provenance.source_stamp,
                            size: copied_size,
                            generation: nexium_gpu::tex_invalidate::region_gen_range(
                                dst_gpu,
                                copied_size,
                            ),
                            dst_layout,
                            dst_block_size: self.dst_block_size,
                            bpp: dst_bytes_per_element,
                            gpu_resolved_block: pending.gpu_resolved_block,
                            source_may_advance: pending.virtual_present,
                        },
                    );
                } else {
                    self.rt_copy_sources.remove(&dst_gpu);
                }
            } else {
                self.rt_copy_sources.remove(&dst_gpu);
            }
        } else {
            self.rt_copy_sources.remove(&dst_gpu);
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
        src_pitch: usize,
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
            let src_row_off = y * src_pitch;
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
        allow_direct: bool,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        let src_pitch = (self.pitch_in as usize).max(line_length);
        let dst_pitch = (self.pitch_out as usize).max(line_length);

        let mut direct_available = allow_direct;
        let mut row = Vec::new();
        for y in 0..line_count {
            let dst_row_off = y * dst_pitch;
            if dst_row_off >= dst_limit {
                break;
            }
            let n = line_length.min(dst_limit - dst_row_off);
            let src_off = src_cpu + (y * src_pitch) as u64;
            let dst_off = dst_cpu + dst_row_off as u64;
            if direct_available && n == line_length {
                let started = crate::gpu::pusher::kickprof::start();
                let copied = mem_copy(src_off, dst_off, n);
                crate::gpu::pusher::kickprof::add_sized(
                    crate::gpu::pusher::kickprof::DMA_COPY,
                    started,
                    n,
                );
                if copied {
                    continue;
                }
                direct_available = false;
            }
            row.resize(line_length, 0);
            let started = crate::gpu::pusher::kickprof::start();
            if !mem_read(src_off, &mut row) {
                crate::gpu::pusher::kickprof::add_sized(
                    crate::gpu::pusher::kickprof::DMA_FALLBACK,
                    started,
                    n,
                );
                break;
            }
            mem_write(dst_off, &row[..n]);
            crate::gpu::pusher::kickprof::add_sized(
                crate::gpu::pusher::kickprof::DMA_FALLBACK,
                started,
                n,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn blit_pitch_to_block(
        &mut self,
        src_cpu: u64,
        src_limit: usize,
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
    ) -> usize {
        self.last_tiled_dst_cpu = dst_cpu;

        let block_height_log2 = ((self.dst_block_size >> 4) & 0xF) as u32;
        let src_pitch = (self.pitch_in as usize).max(line_length_src);

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

        let Some(source_extent) = line_count
            .checked_sub(1)
            .and_then(|last_row| last_row.checked_mul(src_pitch))
            .and_then(|last_row_off| last_row_off.checked_add(line_length_src))
        else {
            return 0;
        };
        if source_extent > src_limit {
            return 0;
        }

        let mut linear_src = vec![0u8; source_extent];
        for y in 0..line_count {
            let src_off = src_cpu + (y * src_pitch) as u64;
            let row_off = y * src_pitch;
            if !mem_read(src_off, &mut linear_src[row_off..row_off + line_length_src]) {
                return 0;
            }
        }
        let post_remap = if remap_enable {
            let line_length_units = line_length_src / (component_size * num_src_components).max(1);
            self.apply_remap(
                &linear_src,
                src_pitch,
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
        if !mem_write(dst_cpu, &tiled[..n]) {
            return 0;
        }
        self.last_tiled_dst_bh_log2 = block_height_log2;
        self.last_tiled_dst_stride = dst_width_bytes as u32;
        self.last_tiled_dst_height = dst_height as u32;
        n
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
        linear_source: Option<&[u8]>,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let owned_linear;
        let linear = if let Some(linear) = linear_source {
            linear
        } else {
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
            owned_linear = unswizzle_block_linear_bytes(
                &src_tiled,
                line_length_src,
                line_count,
                line_length_src.max(1),
                src_width_bytes,
                src_height,
                src_bh,
                (self.src_origin_x as usize) * src_bytes_per_element.max(1),
                self.src_origin_y as usize,
            );
            &owned_linear
        };

        let inter_pitch = line_length_src.max(1);

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
            linear,
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
        let dst_pitch = (self.pitch_out as usize).max(line_length);
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

    fn blit_linear_to_pitch(
        &self,
        linear: &[u8],
        dst_cpu: u64,
        dst_limit: usize,
        line_length: usize,
        line_count: usize,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> bool {
        let Some(required) = line_length.checked_mul(line_count) else {
            return false;
        };
        if linear.len() < required {
            return false;
        }

        let dst_pitch = (self.pitch_out as usize).max(line_length);
        if dst_pitch == line_length && required <= dst_limit {
            return required == 0 || mem_write(dst_cpu, &linear[..required]);
        }

        for y in 0..line_count {
            let dst_off = y.saturating_mul(dst_pitch);
            if dst_off >= dst_limit {
                break;
            }
            let src_off = y * line_length;
            let n = line_length.min(dst_limit - dst_off);
            mem_write(dst_cpu + dst_off as u64, &linear[src_off..src_off + n]);
        }
        true
    }
}

const GOB_W: usize = 64;
const GOB_H: usize = 8;
const GOB_SIZE: usize = 512;

#[derive(Debug)]
struct ActiveGobCopyPlan {
    mapped_extent: usize,
    active_bytes: usize,
    segments: Vec<(usize, usize)>,
}

fn active_gob_copy_plan(
    width_bytes: usize,
    height: usize,
    block_height_log2: u32,
) -> Option<ActiveGobCopyPlan> {
    if width_bytes == 0 || height == 0 || width_bytes % GOB_W != 0 || block_height_log2 > 5 {
        return None;
    }

    let block_height = 1usize.checked_shl(block_height_log2)?;
    let rows_per_block = block_height.checked_mul(GOB_H)?;
    let gob_columns = width_bytes / GOB_W;
    let gob_column_stride = block_height.checked_mul(GOB_SIZE)?;
    let block_row_stride = gob_columns.checked_mul(gob_column_stride)?;
    let full_block_rows = height / rows_per_block;
    let remaining_rows = height % rows_per_block;
    let full_prefix = full_block_rows.checked_mul(block_row_stride)?;
    let block_rows = full_block_rows.checked_add(usize::from(remaining_rows != 0))?;
    let mapped_extent = block_rows.checked_mul(block_row_stride)?;
    let active_bytes = width_bytes.checked_mul(height)?;
    let mut segments = Vec::new();
    if full_prefix != 0 {
        segments.push((0, full_prefix));
    }

    if remaining_rows != 0 {
        let full_tail_gob_bytes = (remaining_rows / GOB_H).checked_mul(GOB_SIZE)?;
        let partial_gob_rows = remaining_rows % GOB_H;
        let mut partial_sectors = Vec::with_capacity(partial_gob_rows * (GOB_W / 16));
        for y in 0..partial_gob_rows {
            for x in (0..GOB_W).step_by(16) {
                partial_sectors.push(in_gob_offset(x, y));
            }
        }
        partial_sectors.sort_unstable();
        let mut partial_spans: Vec<(usize, usize)> = Vec::new();
        for offset in partial_sectors {
            if let Some((start, len)) = partial_spans.last_mut() {
                if start.checked_add(*len) == Some(offset) {
                    *len = len.checked_add(16)?;
                    continue;
                }
            }
            partial_spans.push((offset, 16));
        }

        for column in 0..gob_columns {
            let column_base = full_prefix.checked_add(column.checked_mul(gob_column_stride)?)?;
            if full_tail_gob_bytes != 0 {
                segments.push((column_base, full_tail_gob_bytes));
            }
            let partial_base = column_base.checked_add(full_tail_gob_bytes)?;
            for &(offset, len) in &partial_spans {
                segments.push((partial_base.checked_add(offset)?, len));
            }
        }
    }

    if segments.iter().any(|&(offset, len)| {
        offset
            .checked_add(len)
            .is_none_or(|end| end > mapped_extent)
    }) || segments
        .iter()
        .try_fold(0usize, |sum, &(_, len)| sum.checked_add(len))?
        != active_bytes
    {
        return None;
    }

    Some(ActiveGobCopyPlan {
        mapped_extent,
        active_bytes,
        segments,
    })
}

fn active_gob_copy_ranges_valid(
    plan: &ActiveGobCopyPlan,
    src_cpu: u64,
    src_limit: usize,
    dst_cpu: u64,
    dst_limit: usize,
) -> bool {
    if src_limit < plan.mapped_extent || dst_limit < plan.mapped_extent {
        return false;
    }
    let Some(src_end) = src_cpu.checked_add(plan.mapped_extent as u64) else {
        return false;
    };
    let Some(dst_end) = dst_cpu.checked_add(plan.mapped_extent as u64) else {
        return false;
    };
    src_cpu == dst_cpu || src_end <= dst_cpu || dst_end <= src_cpu
}

fn copy_active_gobs(
    plan: &ActiveGobCopyPlan,
    src_cpu: u64,
    dst_cpu: u64,
    mem_copy: &dyn Fn(u64, u64, usize) -> bool,
) -> bool {
    if src_cpu == dst_cpu {
        return true;
    }
    plan.segments
        .iter()
        .all(|&(offset, len)| mem_copy(src_cpu + offset as u64, dst_cpu + offset as u64, len))
}

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
        active_gob_copy_plan, active_gob_copy_ranges_valid, copy_active_gobs,
        exact_rt_copy_subresource_supported, in_gob_offset, method_requires_hard_boundary,
        swizzle_block_linear, swizzle_block_linear_into, tiled_size_bytes,
        unswizzle_block_linear_bytes, LinearRtSource, MaxwellDma, PendingRtSource, RtKey,
        RtSourceProvenance, GOB_H, GOB_SIZE, GOB_W, LAUNCH_DST_LAYOUT_BIT, LAUNCH_MULTI_LINE_BIT,
        LAUNCH_REMAP_ENABLE_BIT, LAUNCH_SRC_LAYOUT_BIT, M_LAUNCH_DMA, M_LINE_COUNT,
        M_LINE_LENGTH_IN, M_OFFSET_IN_LOWER, M_OFFSET_IN_UPPER, M_OFFSET_OUT_LOWER,
        M_OFFSET_OUT_UPPER, M_PITCH_IN, M_PITCH_OUT,
    };
    use crate::gpu::GpuMappings;
    use std::cell::{Cell, RefCell};

    const SRC_GPU: u64 = 0x1000;
    const DST_GPU: u64 = 0x3000;
    const SRC_CPU: u64 = 0x1_0000;
    const DST_CPU: u64 = 0x1_1000;

    #[test]
    fn only_dma_launch_requires_hard_boundary() {
        assert!(method_requires_hard_boundary(M_LAUNCH_DMA));
        assert!(!method_requires_hard_boundary(M_OFFSET_IN_UPPER));
        assert!(!method_requires_hard_boundary(M_LINE_LENGTH_IN));
        assert!(!method_requires_hard_boundary(M_LAUNCH_DMA + 1));
    }

    #[test]
    fn exact_rt_copy_accepts_only_single_layer_2d_subresources() {
        assert!(exact_rt_copy_subresource_supported(0, 0, 0, 0));
        assert!(exact_rt_copy_subresource_supported(1, 0, 1, 0));
        assert!(!exact_rt_copy_subresource_supported(2, 0, 1, 0));
        assert!(!exact_rt_copy_subresource_supported(1, 0, 2, 0));
        assert!(!exact_rt_copy_subresource_supported(1, 1, 1, 0));
        assert!(!exact_rt_copy_subresource_supported(1, 0, 1, 1));
    }

    fn launch_linear_copy(
        dma: &mut MaxwellDma,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        for (method, value) in [
            (M_OFFSET_IN_UPPER, 0),
            (M_OFFSET_IN_LOWER, SRC_GPU as u32),
            (M_OFFSET_OUT_UPPER, 0),
            (M_OFFSET_OUT_LOWER, DST_GPU as u32),
            (M_PITCH_IN, 64),
            (M_PITCH_OUT, 64),
            (M_LINE_LENGTH_IN, 64),
            (M_LINE_COUNT, 1),
            (
                M_LAUNCH_DMA,
                (1 << LAUNCH_SRC_LAYOUT_BIT) | (1 << LAUNCH_DST_LAYOUT_BIT),
            ),
        ] {
            dma.dispatch_method(method, value, mappings, mem_read, mem_write, mem_copy);
        }
    }

    fn linear_copy_fixture() -> (GpuMappings, RefCell<Vec<u8>>, Vec<u8>) {
        let mut mappings = GpuMappings::new();
        mappings.add(SRC_GPU, 0x1000, SRC_CPU, 1);
        mappings.add(DST_GPU, 0x1000, DST_CPU, 2);
        let expected: Vec<u8> = (0..64).map(|value| (value * 17) as u8).collect();
        let mut memory = vec![0u8; 0x2000];
        memory[..expected.len()].copy_from_slice(&expected);
        (mappings, RefCell::new(memory), expected)
    }

    #[test]
    fn identity_remap_honors_padded_source_pitch() {
        const LINE_LENGTH_UNITS: usize = 3;
        const LINE_COUNT: usize = 3;
        const COMPONENT_SIZE: usize = 4;
        const ACTIVE_ROW_BYTES: usize = LINE_LENGTH_UNITS * COMPONENT_SIZE;
        const SRC_PITCH: usize = ACTIVE_ROW_BYTES + 4;

        let mut src = vec![0xee; SRC_PITCH * LINE_COUNT];
        let mut expected = Vec::with_capacity(ACTIVE_ROW_BYTES * LINE_COUNT);
        for y in 0..LINE_COUNT {
            let row: Vec<u8> = (0..ACTIVE_ROW_BYTES)
                .map(|x| 0x10 + (y * 0x20 + x) as u8)
                .collect();
            src[y * SRC_PITCH..y * SRC_PITCH + ACTIVE_ROW_BYTES].copy_from_slice(&row);
            expected.extend_from_slice(&row);
        }

        let remapped = MaxwellDma::new().apply_remap(
            &src,
            SRC_PITCH,
            LINE_LENGTH_UNITS,
            LINE_COUNT,
            COMPONENT_SIZE,
            1,
            1,
            [0, 1, 2, 3],
        );

        assert_eq!(remapped, expected);
    }

    #[test]
    fn linear_dma_uses_direct_guest_copy() {
        let (mappings, memory, expected) = linear_copy_fixture();
        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        let copies = Cell::new(0usize);
        let mem_read = |address: u64, output: &mut [u8]| {
            reads.set(reads.get() + 1);
            let offset = (address - SRC_CPU) as usize;
            output.copy_from_slice(&memory.borrow()[offset..offset + output.len()]);
            true
        };
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            let offset = (address - SRC_CPU) as usize;
            memory.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };
        let mem_copy = |src: u64, dst: u64, len: usize| {
            copies.set(copies.get() + 1);
            assert_eq!((src, dst, len), (SRC_CPU, DST_CPU, 64));
            let src_offset = (src - SRC_CPU) as usize;
            let dst_offset = (dst - SRC_CPU) as usize;
            memory
                .borrow_mut()
                .copy_within(src_offset..src_offset + len, dst_offset);
            true
        };

        launch_linear_copy(
            &mut MaxwellDma::new(),
            &mappings,
            &mem_read,
            &mem_write,
            &mem_copy,
        );

        assert_eq!(copies.get(), 1);
        assert_eq!(reads.get(), 0);
        assert_eq!(writes.get(), 0);
        assert_eq!(&memory.borrow()[0x1000..0x1040], expected.as_slice());
    }

    #[test]
    fn linear_dma_falls_back_when_direct_copy_is_unavailable() {
        let (mappings, memory, expected) = linear_copy_fixture();
        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        let copies = Cell::new(0usize);
        let mem_read = |address: u64, output: &mut [u8]| {
            reads.set(reads.get() + 1);
            let offset = (address - SRC_CPU) as usize;
            output.copy_from_slice(&memory.borrow()[offset..offset + output.len()]);
            true
        };
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            let offset = (address - SRC_CPU) as usize;
            memory.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };
        let mem_copy = |_: u64, _: u64, _: usize| {
            copies.set(copies.get() + 1);
            false
        };

        launch_linear_copy(
            &mut MaxwellDma::new(),
            &mappings,
            &mem_read,
            &mem_write,
            &mem_copy,
        );

        assert_eq!(copies.get(), 1);
        assert_eq!(reads.get(), 1);
        assert_eq!(writes.get(), 1);
        assert_eq!(&memory.borrow()[0x1000..0x1040], expected.as_slice());
    }

    #[test]
    fn staged_rt_linear_copy_matches_block_linear_round_trip() {
        let width = 137usize;
        let height = 73usize;
        let dst_pitch = 160usize;
        let block_height_log2 = 3;
        let raw: Vec<u8> = (0..width * height)
            .map(|index| index.wrapping_mul(37).wrapping_add(11) as u8)
            .collect();
        let tiled = swizzle_block_linear(
            &raw,
            width,
            height,
            width,
            width,
            height,
            block_height_log2,
            0,
            0,
        );
        let expected = unswizzle_block_linear_bytes(
            &tiled,
            width,
            height,
            dst_pitch,
            width,
            height,
            block_height_log2,
            0,
            0,
        );

        let mut dma = MaxwellDma::new();
        dma.pitch_out = dst_pitch as u32;
        let output = RefCell::new(vec![0u8; dst_pitch * height]);
        let writes = Cell::new(0usize);
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            let offset = (address - DST_CPU) as usize;
            output.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };

        assert!(dma.blit_linear_to_pitch(
            &raw,
            DST_CPU,
            dst_pitch * height,
            width,
            height,
            &mem_write,
        ));
        assert_eq!(writes.get(), height);
        assert_eq!(*output.borrow(), expected);
    }

    #[test]
    fn staged_rt_linear_copy_coalesces_tightly_pitched_rows() {
        let width = 128usize;
        let height = 64usize;
        let raw: Vec<u8> = (0..width * height).map(|index| index as u8).collect();
        let mut dma = MaxwellDma::new();
        dma.pitch_out = width as u32;
        let output = RefCell::new(vec![0u8; raw.len()]);
        let writes = Cell::new(0usize);
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            let offset = (address - DST_CPU) as usize;
            output.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };

        assert!(dma.blit_linear_to_pitch(&raw, DST_CPU, raw.len(), width, height, &mem_write,));
        assert_eq!(writes.get(), 1);
        assert_eq!(*output.borrow(), raw);

        let writes_before = writes.get();
        assert!(!dma.blit_linear_to_pitch(
            &raw[..raw.len() - 1],
            DST_CPU,
            raw.len(),
            width,
            height,
            &mem_write,
        ));
        assert_eq!(writes.get(), writes_before);
    }

    #[test]
    fn staged_rt_launch_uses_carried_bytes_without_rereading_source() {
        let width = 64usize;
        let height = 8usize;
        let raw: Vec<u8> = (0..width * height)
            .map(|index| index.wrapping_mul(29).wrapping_add(7) as u8)
            .collect();
        let mut mappings = GpuMappings::new();
        mappings.add(SRC_GPU, 0x1000, SRC_CPU, 1);
        mappings.add(DST_GPU, 0x1000, DST_CPU, 2);

        let mut dma = MaxwellDma::new();
        dma.offset_in_lower = SRC_GPU as u32;
        dma.offset_out_lower = DST_GPU as u32;
        dma.pitch_out = width as u32;
        dma.line_length_in = (width / 4) as u32;
        dma.line_count = height as u32;
        dma.src_width = (width / 4) as u32;
        dma.src_height = height as u32;
        dma.remap_components = (3 << 16) | 0x3210;
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: None,
            src_gpu: SRC_GPU,
            src_block_size: dma.src_block_size,
            virtual_present: false,
            virtual_block: false,
            gpu_resolved_block: false,
            linear: Some(LinearRtSource {
                bytes: raw.clone(),
                width_bytes: width,
                height,
            }),
        });

        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        let output = RefCell::new(vec![0u8; raw.len()]);
        let mem_read = |_: u64, output: &mut [u8]| {
            reads.set(reads.get() + 1);
            output.fill(0xee);
            true
        };
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            let offset = (address - DST_CPU) as usize;
            output.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };

        dma.launch_dma(
            (1 << LAUNCH_DST_LAYOUT_BIT)
                | (1 << LAUNCH_MULTI_LINE_BIT)
                | (1 << LAUNCH_REMAP_ENABLE_BIT),
            &mappings,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );

        assert_eq!(reads.get(), 0);
        assert_eq!(writes.get(), 1);
        assert_eq!(*output.borrow(), raw);
        assert!(dma.pending_rt_source.is_none());
        assert_eq!(dma.blit_count, 1);
    }

    #[test]
    fn staged_rt_block_copy_skips_source_unswizzle() {
        let width = 64usize;
        let height = 8usize;
        let raw: Vec<u8> = (0..width * height)
            .map(|index| index.wrapping_mul(43).wrapping_add(5) as u8)
            .collect();
        let expected = swizzle_block_linear(&raw, width, height, width, width, height, 0, 0, 0);
        let mut mappings = GpuMappings::new();
        mappings.add(SRC_GPU, 0x1000, SRC_CPU, 1);
        mappings.add(DST_GPU, 0x1000, DST_CPU, 2);

        let mut dma = MaxwellDma::new();
        dma.offset_in_lower = SRC_GPU as u32;
        dma.offset_out_lower = DST_GPU as u32;
        dma.line_length_in = (width / 4) as u32;
        dma.line_count = height as u32;
        dma.src_width = (width / 4) as u32;
        dma.src_height = height as u32;
        dma.dst_width = (width / 4) as u32;
        dma.dst_height = height as u32;
        dma.remap_components = (3 << 16) | 0x3210;
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: None,
            src_gpu: SRC_GPU,
            src_block_size: dma.src_block_size,
            virtual_present: false,
            virtual_block: false,
            gpu_resolved_block: false,
            linear: Some(LinearRtSource {
                bytes: raw,
                width_bytes: width,
                height,
            }),
        });

        let source_reads = Cell::new(0usize);
        let destination_reads = Cell::new(0usize);
        let output = RefCell::new(vec![0u8; 0x1000]);
        let mem_read = |address: u64, bytes: &mut [u8]| {
            if address == SRC_CPU {
                source_reads.set(source_reads.get() + 1);
            } else if address == DST_CPU {
                destination_reads.set(destination_reads.get() + 1);
            }
            bytes.fill(0);
            true
        };
        let mem_write = |address: u64, input: &[u8]| {
            let offset = (address - DST_CPU) as usize;
            output.borrow_mut()[offset..offset + input.len()].copy_from_slice(input);
            true
        };

        dma.launch_dma(
            (1 << LAUNCH_MULTI_LINE_BIT) | (1 << LAUNCH_REMAP_ENABLE_BIT),
            &mappings,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );

        assert_eq!(source_reads.get(), 0);
        assert_eq!(destination_reads.get(), 1);
        assert_eq!(&output.borrow()[..expected.len()], expected.as_slice());
        assert!(dma.pending_rt_source.is_none());
    }

    #[test]
    fn present_surface_registration_expires() {
        let mut dma = MaxwellDma::new();
        dma.register_present_surface(7, &[DST_GPU], 1280, 720, &[SRC_GPU]);
        assert!(dma.is_recent_present_copy(7, SRC_GPU, DST_GPU, 1280, 720));
        assert!(!dma.is_recent_present_copy(7, SRC_GPU + 1, DST_GPU, 1280, 720));

        for index in 0..9 {
            dma.register_present_surface(
                8,
                &[DST_GPU + 0x10_000 + index],
                64,
                64,
                &[SRC_GPU + 0x10_000 + index],
            );
        }
        assert!(!dma.is_recent_present_copy(7, SRC_GPU, DST_GPU, 1280, 720));
    }

    fn configure_virtual_present_source(
        dma: &mut MaxwellDma,
        src_gpu: u64,
        dst_gpu: u64,
        width: u32,
        height: u32,
        bpp: usize,
        source: RtKey,
        source_stamp: u64,
    ) {
        debug_assert!((1..=4).contains(&bpp));
        dma.offset_in_upper = (src_gpu >> 32) as u32;
        dma.offset_in_lower = src_gpu as u32;
        dma.offset_out_upper = (dst_gpu >> 32) as u32;
        dma.offset_out_lower = dst_gpu as u32;
        dma.pitch_out = width.saturating_mul(bpp as u32);
        dma.line_length_in = width;
        dma.line_count = height;
        dma.src_width = width;
        dma.src_height = height;
        dma.src_origin_x = 0;
        dma.src_origin_y = 0;
        dma.dst_origin_x = 0;
        dma.dst_origin_y = 0;
        dma.remap_components = ((bpp as u32 - 1) << 16) | 0x3210;
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: Some(RtSourceProvenance {
                source,
                source_stamp,
                bpp,
            }),
            src_gpu,
            src_block_size: dma.src_block_size,
            virtual_present: true,
            virtual_block: false,
            gpu_resolved_block: false,
            linear: None,
        });
    }

    #[test]
    fn virtual_present_live_source_preserves_floor_until_destination_write() {
        const SOURCE_GPU: u64 = 0xb1_0000;
        const DEST_GPU: u64 = 0xb3_0000;
        const SOURCE_CPU: u64 = 0x51_0000;
        const DEST_CPU: u64 = 0x53_0000;
        let width_pixels = 16usize;
        let width_bytes = width_pixels * 4;
        let height = 8usize;
        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 41);
        mappings.add(DEST_GPU, 0x1000, DEST_CPU, 42);

        let mut dma = MaxwellDma::new();
        let source = RtKey::new(41, width_pixels as u32, height as u32, SOURCE_GPU);
        configure_virtual_present_source(
            &mut dma,
            SOURCE_GPU,
            DEST_GPU,
            width_pixels as u32,
            height as u32,
            4,
            source,
            42,
        );
        assert_eq!(dma.pitch_out, width_bytes as u32);

        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        dma.launch_dma(
            (1 << LAUNCH_DST_LAYOUT_BIT)
                | (1 << LAUNCH_MULTI_LINE_BIT)
                | (1 << LAUNCH_REMAP_ENABLE_BIT),
            &mappings,
            &|_, bytes| {
                reads.set(reads.get() + 1);
                bytes.fill(0);
                true
            },
            &|_, _| {
                writes.set(writes.get() + 1);
                true
            },
            &|_, _, _| false,
        );

        assert_eq!(reads.get(), 0);
        assert_eq!(writes.get(), 0);
        assert_eq!(dma.rt_copy_source(DEST_GPU), Some((source, 42)));
        assert_eq!(
            dma.exact_present_source(DEST_GPU, width_pixels as u32, height as u32),
            Some((source, 42, true))
        );
        let carried = dma.pending_copy_source(DEST_GPU).unwrap();
        assert!(!carried.gpu_resolved_block);
        assert!(!carried.virtual_block);
        assert_eq!(
            dma.exact_present_source(DEST_GPU, width_pixels as u32, height as u32 + 1),
            None
        );
        let token = dma
            .exact_present_source_token(DEST_GPU, width_pixels as u32, height as u32)
            .unwrap();
        assert!(token.destination_is_current());
        nexium_gpu::tex_invalidate::bump_region(DEST_GPU, width_bytes as u64 * height as u64);
        assert!(!token.destination_is_current());
        assert_eq!(dma.rt_copy_source(DEST_GPU), None);
        assert_eq!(
            dma.exact_present_source(DEST_GPU, width_pixels as u32, height as u32),
            None
        );
        assert!(dma.pending_rt_source.is_none());
    }

    #[test]
    fn exact_present_rejects_non_rgba8_live_source() {
        const SOURCE_GPU: u64 = 0xc1_0000;
        const DEST_GPU: u64 = 0xc3_0000;
        const SOURCE_CPU: u64 = 0x61_0000;
        const DEST_CPU: u64 = 0x63_0000;
        let width = 16;
        let height = 8;
        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 51);
        mappings.add(DEST_GPU, 0x1000, DEST_CPU, 52);

        let source = RtKey::new(51, width, height, SOURCE_GPU);
        let mut dma = MaxwellDma::new();
        configure_virtual_present_source(
            &mut dma, SOURCE_GPU, DEST_GPU, width, height, 2, source, 77,
        );
        dma.launch_dma(
            (1 << LAUNCH_DST_LAYOUT_BIT)
                | (1 << LAUNCH_MULTI_LINE_BIT)
                | (1 << LAUNCH_REMAP_ENABLE_BIT),
            &mappings,
            &|_, _| panic!("virtual present must not read guest memory"),
            &|_, _| panic!("virtual present must not write guest memory"),
            &|_, _, _| false,
        );

        assert_eq!(dma.rt_copy_source(DEST_GPU), Some((source, 77)));
        assert_eq!(dma.exact_present_source(DEST_GPU, width, height), None);
    }

    #[test]
    fn dma_write_through_cpu_alias_invalidates_old_live_present_record() {
        const FIRST_SOURCE_GPU: u64 = 0xd1_0000;
        const SECOND_SOURCE_GPU: u64 = 0xd2_0000;
        const FIRST_DEST_GPU: u64 = 0xd3_0000;
        const ALIAS_DEST_GPU: u64 = 0xd4_0000;
        const FIRST_SOURCE_CPU: u64 = 0x71_0000;
        const SECOND_SOURCE_CPU: u64 = 0x72_0000;
        const SHARED_DEST_CPU: u64 = 0x73_0000;
        let width = 16;
        let height = 8;
        let mut mappings = GpuMappings::new();
        mappings.add(FIRST_SOURCE_GPU, 0x1000, FIRST_SOURCE_CPU, 61);
        mappings.add(SECOND_SOURCE_GPU, 0x1000, SECOND_SOURCE_CPU, 62);
        mappings.add(FIRST_DEST_GPU, 0x1000, SHARED_DEST_CPU, 63);
        mappings.add(ALIAS_DEST_GPU, 0x1000, SHARED_DEST_CPU, 64);

        let mut dma = MaxwellDma::new();
        let first_source = RtKey::new(61, width, height, FIRST_SOURCE_GPU);
        configure_virtual_present_source(
            &mut dma,
            FIRST_SOURCE_GPU,
            FIRST_DEST_GPU,
            width,
            height,
            4,
            first_source,
            81,
        );
        let flags = (1 << LAUNCH_DST_LAYOUT_BIT)
            | (1 << LAUNCH_MULTI_LINE_BIT)
            | (1 << LAUNCH_REMAP_ENABLE_BIT);
        dma.launch_dma(
            flags,
            &mappings,
            &|_, _| panic!("virtual present must not read guest memory"),
            &|_, _| panic!("virtual present must not write guest memory"),
            &|_, _, _| false,
        );
        assert_eq!(
            dma.exact_present_source(FIRST_DEST_GPU, width, height),
            Some((first_source, 81, true))
        );
        let first_token = dma
            .exact_present_source_token(FIRST_DEST_GPU, width, height)
            .unwrap();
        assert!(first_token.destination_is_current());

        let second_source = RtKey::new(62, width, height, SECOND_SOURCE_GPU);
        configure_virtual_present_source(
            &mut dma,
            SECOND_SOURCE_GPU,
            ALIAS_DEST_GPU,
            width,
            height,
            4,
            second_source,
            82,
        );
        dma.launch_dma(
            flags,
            &mappings,
            &|_, _| panic!("virtual present must not read guest memory"),
            &|_, _| panic!("virtual present must not write guest memory"),
            &|_, _, _| false,
        );

        assert_eq!(dma.rt_copy_source(FIRST_DEST_GPU), None);
        assert!(!first_token.destination_is_current());
        assert_eq!(
            dma.exact_present_source(ALIAS_DEST_GPU, width, height),
            Some((second_source, 82, true))
        );
    }

    fn configure_virtual_block_copy(
        dma: &mut MaxwellDma,
        src_gpu: u64,
        dst_gpu: u64,
        width: u32,
        height: u32,
        block_height_log2: u32,
    ) {
        dma.offset_in_upper = (src_gpu >> 32) as u32;
        dma.offset_in_lower = src_gpu as u32;
        dma.offset_out_upper = (dst_gpu >> 32) as u32;
        dma.offset_out_lower = dst_gpu as u32;
        dma.line_length_in = width;
        dma.line_count = height;
        dma.src_width = width;
        dma.src_height = height;
        dma.dst_width = width;
        dma.dst_height = height;
        dma.src_block_size = block_height_log2 << 4;
        dma.dst_block_size = block_height_log2 << 4;
        dma.src_origin_x = 0;
        dma.src_origin_y = 0;
        dma.dst_origin_x = 0;
        dma.dst_origin_y = 0;
        dma.remap_components = (3 << 16) | 0x3210;
    }

    #[test]
    fn virtual_block_copy_and_compatible_chain_use_zero_guest_io() {
        const FIRST_GPU: u64 = 0x81_0000;
        const SECOND_GPU: u64 = 0x83_0000;
        const THIRD_GPU: u64 = 0x85_0000;
        const FIRST_CPU: u64 = 0x21_0000;
        const SECOND_CPU: u64 = 0x23_0000;
        const THIRD_CPU: u64 = 0x25_0000;
        let width = 16;
        let height = 8;
        let mut mappings = GpuMappings::new();
        mappings.add(FIRST_GPU, 0x1000, FIRST_CPU, 11);
        mappings.add(SECOND_GPU, 0x1000, SECOND_CPU, 12);
        mappings.add(THIRD_GPU, 0x1000, THIRD_CPU, 13);

        let source = RtKey::new(11, width, height, FIRST_GPU);
        let mut dma = MaxwellDma::new();
        configure_virtual_block_copy(&mut dma, FIRST_GPU, SECOND_GPU, width, height, 0);
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: Some(RtSourceProvenance {
                source,
                source_stamp: 42,
                bpp: 4,
            }),
            src_gpu: FIRST_GPU,
            src_block_size: dma.src_block_size,
            virtual_present: false,
            virtual_block: true,
            gpu_resolved_block: false,
            linear: None,
        });

        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        let copies = Cell::new(0usize);
        let flags = (1 << LAUNCH_MULTI_LINE_BIT) | (1 << LAUNCH_REMAP_ENABLE_BIT);
        let mem_read = |_: u64, bytes: &mut [u8]| {
            reads.set(reads.get() + 1);
            bytes.fill(0);
            true
        };
        let mem_write = |_: u64, _: &[u8]| {
            writes.set(writes.get() + 1);
            true
        };
        let mem_copy = |_: u64, _: u64, _: usize| {
            copies.set(copies.get() + 1);
            true
        };
        dma.launch_dma(flags, &mappings, &mem_read, &mem_write, &mem_copy);
        assert_eq!(dma.rt_copy_source(SECOND_GPU), Some((source, 42)));
        assert_eq!((reads.get(), writes.get(), copies.get()), (0, 0, 0));
        assert_eq!(dma.take_guest_write_range(), None);

        let chained = dma.pending_copy_source(SECOND_GPU).unwrap();
        assert!(chained.virtual_block);
        assert_eq!(chained.src_block_size, 0);
        dma.pending_rt_source = Some(chained);
        configure_virtual_block_copy(&mut dma, SECOND_GPU, THIRD_GPU, width, height, 0);
        dma.launch_dma(flags, &mappings, &mem_read, &mem_write, &mem_copy);
        assert_eq!(dma.rt_copy_source(THIRD_GPU), Some((source, 42)));
        assert_eq!((reads.get(), writes.get(), copies.get()), (0, 0, 0));
        assert_eq!(dma.take_guest_write_range(), None);
    }

    #[test]
    fn gpu_resolved_block_provenance_expires_after_guest_write() {
        const SOURCE_GPU: u64 = 0xa1_0000;
        const DEST_GPU: u64 = 0xa3_0000;
        const SOURCE_CPU: u64 = 0x41_0000;
        const DEST_CPU: u64 = 0x43_0000;
        let width = 16;
        let height = 8;
        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 31);
        mappings.add(DEST_GPU, 0x1000, DEST_CPU, 32);

        let resolved = RtKey::new(u32::MAX, width, height, DEST_GPU);
        let mut dma = MaxwellDma::new();
        configure_virtual_block_copy(&mut dma, SOURCE_GPU, DEST_GPU, width, height, 0);
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: Some(RtSourceProvenance {
                source: resolved,
                source_stamp: 99,
                bpp: 4,
            }),
            src_gpu: SOURCE_GPU,
            src_block_size: dma.src_block_size,
            virtual_present: false,
            virtual_block: true,
            gpu_resolved_block: true,
            linear: None,
        });

        let flags = (1 << LAUNCH_MULTI_LINE_BIT) | (1 << LAUNCH_REMAP_ENABLE_BIT);
        dma.launch_dma(
            flags,
            &mappings,
            &|_, _| panic!("GPU-resolved copy must not read guest memory"),
            &|_, _| panic!("GPU-resolved copy must not write guest memory"),
            &|_, _, _| panic!("GPU-resolved copy must not use guest memcpy"),
        );
        assert_eq!(dma.rt_copy_source(DEST_GPU), Some((resolved, 99)));
        assert_eq!(
            dma.exact_present_source(DEST_GPU, width, height),
            Some((resolved, 99, false))
        );
        assert_eq!(dma.exact_present_source(DEST_GPU, width + 1, height), None);
        assert!(dma
            .pending_copy_source(DEST_GPU)
            .is_some_and(|pending| pending.gpu_resolved_block));

        nexium_gpu::tex_invalidate::bump_region(DEST_GPU, 0x1000);
        assert_eq!(dma.rt_copy_source(DEST_GPU), None);
        assert_eq!(dma.exact_present_source(DEST_GPU, width, height), None);
        assert!(dma.pending_copy_source(DEST_GPU).is_none());
    }

    #[test]
    fn virtual_block_rejection_falls_back_to_guest_copy() {
        const SOURCE_GPU: u64 = 0x91_0000;
        const DEST_GPU: u64 = 0x93_0000;
        const SOURCE_CPU: u64 = 0x31_0000;
        const DEST_CPU: u64 = 0x31_1000;
        let width = 16;
        let height = 8;
        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 21);
        mappings.add(DEST_GPU, 0x1000, DEST_CPU, 22);
        let mut dma = MaxwellDma::new();
        configure_virtual_block_copy(&mut dma, SOURCE_GPU, DEST_GPU, width, height, 0);
        dma.dst_block_size = 1 << 4;
        let source = RtKey::new(21, width, height, SOURCE_GPU);
        dma.pending_rt_source = Some(PendingRtSource {
            provenance: Some(RtSourceProvenance {
                source,
                source_stamp: 7,
                bpp: 4,
            }),
            src_gpu: SOURCE_GPU,
            src_block_size: dma.src_block_size,
            virtual_present: false,
            virtual_block: true,
            gpu_resolved_block: false,
            linear: None,
        });

        let memory = RefCell::new(vec![0x5a; 0x2000]);
        let reads = Cell::new(0usize);
        let writes = Cell::new(0usize);
        let copies = Cell::new(0usize);
        dma.launch_dma(
            (1 << LAUNCH_MULTI_LINE_BIT) | (1 << LAUNCH_REMAP_ENABLE_BIT),
            &mappings,
            &|address, bytes| {
                reads.set(reads.get() + 1);
                let offset = (address - SOURCE_CPU) as usize;
                bytes.copy_from_slice(&memory.borrow()[offset..offset + bytes.len()]);
                true
            },
            &|address, bytes| {
                writes.set(writes.get() + 1);
                let offset = (address - SOURCE_CPU) as usize;
                memory.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
                true
            },
            &|_, _, _| {
                copies.set(copies.get() + 1);
                true
            },
        );

        assert_eq!(reads.get(), 2);
        assert_eq!(writes.get(), 1);
        assert_eq!(copies.get(), 0);
    }

    #[test]
    fn pitch_to_block_invalidates_all_written_cpu_aliases() {
        const SOURCE_GPU: u64 = 0xe1_0000;
        const DEST_GPU: u64 = 0xe3_0000;
        const ALIAS_GPU: u64 = 0xe5_0000;
        const SOURCE_CPU: u64 = 0x81_0000;
        const DEST_CPU: u64 = 0x83_0000;
        const WIDTH: usize = 64;
        const HEIGHT: usize = 8;
        const WRITTEN_SPAN: usize = 0x180;

        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 71);
        mappings.add(DEST_GPU, WRITTEN_SPAN as u64, DEST_CPU, 72);
        mappings.add(ALIAS_GPU, 0x1000, DEST_CPU, 73);

        let source = (0..WIDTH * HEIGHT)
            .map(|index| index.wrapping_mul(29).wrapping_add(7) as u8)
            .collect::<Vec<_>>();
        let destination = RefCell::new(vec![0u8; 0x1000]);
        let writes = Cell::new(0usize);
        let mem_read = |address: u64, output: &mut [u8]| {
            if let Some(offset) = address.checked_sub(SOURCE_CPU) {
                let offset = offset as usize;
                if let Some(input) = source.get(offset..offset.saturating_add(output.len())) {
                    output.copy_from_slice(input);
                    return true;
                }
            }
            if let Some(offset) = address.checked_sub(DEST_CPU) {
                let offset = offset as usize;
                if let Some(input) = destination
                    .borrow()
                    .get(offset..offset.saturating_add(output.len()))
                {
                    output.copy_from_slice(input);
                    return true;
                }
            }
            false
        };
        let mem_write = |address: u64, input: &[u8]| {
            writes.set(writes.get() + 1);
            assert_eq!(address, DEST_CPU);
            assert_eq!(input.len(), WRITTEN_SPAN);
            destination.borrow_mut()[..input.len()].copy_from_slice(input);
            true
        };

        let mut dma = MaxwellDma::new();
        dma.offset_in_lower = SOURCE_GPU as u32;
        dma.offset_out_lower = DEST_GPU as u32;
        dma.pitch_in = WIDTH as u32;
        dma.line_length_in = WIDTH as u32;
        dma.line_count = HEIGHT as u32;
        dma.dst_width = WIDTH as u32;
        dma.dst_height = HEIGHT as u32;

        nexium_gpu::pitch_oracle::record_pitch_dst(DEST_GPU, 0x1000);
        nexium_gpu::pitch_oracle::record_pitch_dst(ALIAS_GPU, 0x1000);
        let exact_generation =
            nexium_gpu::tex_invalidate::region_gen_range(DEST_GPU, WRITTEN_SPAN as u64);
        let alias_generation =
            nexium_gpu::tex_invalidate::region_gen_range(ALIAS_GPU, WRITTEN_SPAN as u64);

        dma.launch_dma(
            (1 << LAUNCH_SRC_LAYOUT_BIT) | (1 << LAUNCH_MULTI_LINE_BIT),
            &mappings,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );

        assert_eq!(writes.get(), 1);
        assert_eq!(
            dma.take_guest_write_range(),
            Some((DEST_CPU, WRITTEN_SPAN as u64))
        );
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen_range(DEST_GPU, WRITTEN_SPAN as u64),
            exact_generation
        );
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen_range(ALIAS_GPU, WRITTEN_SPAN as u64),
            alias_generation
        );
        assert!(!nexium_gpu::pitch_oracle::is_pitch_dst(DEST_GPU));
        assert!(!nexium_gpu::pitch_oracle::is_pitch_dst(ALIAS_GPU));
        assert!(nexium_gpu::pitch_oracle::is_pitch_dst(
            DEST_GPU + WRITTEN_SPAN as u64
        ));
        assert!(nexium_gpu::pitch_oracle::is_pitch_dst(
            ALIAS_GPU + WRITTEN_SPAN as u64
        ));
    }

    #[test]
    fn failed_pitch_to_block_write_preserves_alias_metadata() {
        const SOURCE_GPU: u64 = 0xf1_0000;
        const DEST_GPU: u64 = 0xf3_0000;
        const ALIAS_GPU: u64 = 0xf5_0000;
        const SOURCE_CPU: u64 = 0x91_0000;
        const DEST_CPU: u64 = 0x93_0000;
        const WIDTH: usize = 64;
        const HEIGHT: usize = 8;

        let mut mappings = GpuMappings::new();
        mappings.add(SOURCE_GPU, 0x1000, SOURCE_CPU, 81);
        mappings.add(DEST_GPU, 0x1000, DEST_CPU, 82);
        mappings.add(ALIAS_GPU, 0x1000, DEST_CPU, 83);

        let mut dma = MaxwellDma::new();
        dma.offset_in_lower = SOURCE_GPU as u32;
        dma.offset_out_lower = DEST_GPU as u32;
        dma.pitch_in = WIDTH as u32;
        dma.line_length_in = WIDTH as u32;
        dma.line_count = HEIGHT as u32;
        dma.dst_width = WIDTH as u32;
        dma.dst_height = HEIGHT as u32;

        nexium_gpu::pitch_oracle::record_pitch_dst(DEST_GPU, 0x1000);
        nexium_gpu::pitch_oracle::record_pitch_dst(ALIAS_GPU, 0x1000);
        let exact_generation = nexium_gpu::tex_invalidate::region_gen_range(DEST_GPU, 0x1000);
        let alias_generation = nexium_gpu::tex_invalidate::region_gen_range(ALIAS_GPU, 0x1000);

        dma.launch_dma(
            (1 << LAUNCH_SRC_LAYOUT_BIT) | (1 << LAUNCH_MULTI_LINE_BIT),
            &mappings,
            &|_, output| {
                output.fill(0x5a);
                true
            },
            &|address, input| {
                assert_eq!(address, DEST_CPU);
                assert_eq!(input.len(), WIDTH * HEIGHT);
                false
            },
            &|_, _, _| false,
        );

        assert_eq!(dma.take_guest_write_range(), None);
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen_range(DEST_GPU, 0x1000),
            exact_generation
        );
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen_range(ALIAS_GPU, 0x1000),
            alias_generation
        );
        assert!(nexium_gpu::pitch_oracle::is_pitch_dst(DEST_GPU));
        assert!(nexium_gpu::pitch_oracle::is_pitch_dst(ALIAS_GPU));
    }

    fn assert_active_gob_copy_matches_reference(
        width_bytes: usize,
        height: usize,
        block_height_log2: u32,
    ) {
        let plan = active_gob_copy_plan(width_bytes, height, block_height_log2).unwrap();
        let source: Vec<u8> = (0..plan.mapped_extent)
            .map(|index| index.wrapping_mul(37).wrapping_add(11) as u8)
            .collect();
        let linear = unswizzle_block_linear_bytes(
            &source,
            width_bytes,
            height,
            width_bytes,
            width_bytes,
            height,
            block_height_log2,
            0,
            0,
        );
        let mut expected = vec![0xa5; plan.mapped_extent];
        swizzle_block_linear_into(
            &mut expected,
            &linear,
            width_bytes,
            height,
            width_bytes,
            width_bytes,
            block_height_log2,
            0,
            0,
        );

        let dst_base = plan.mapped_extent + 0x1000;
        let mut bytes = vec![0xa5; dst_base + plan.mapped_extent];
        bytes[..plan.mapped_extent].copy_from_slice(&source);
        let bytes = RefCell::new(bytes);
        let copies = Cell::new(0usize);
        assert!(copy_active_gobs(
            &plan,
            0,
            dst_base as u64,
            &|src, dst, len| {
                copies.set(copies.get() + 1);
                bytes
                    .borrow_mut()
                    .copy_within(src as usize..src as usize + len, dst as usize);
                true
            },
        ));
        assert_ne!(copies.get(), 0);
        assert_eq!(
            &bytes.borrow()[dst_base..dst_base + plan.mapped_extent],
            expected.as_slice()
        );
    }

    #[test]
    fn active_gob_copy_matches_reference_and_preserves_partial_gob_padding() {
        assert_active_gob_copy_matches_reference(128, 19, 2);
    }

    #[test]
    fn active_gob_copy_handles_7680x1080_bh5_partial_block_row() {
        let plan = active_gob_copy_plan(7680, 1080, 5).unwrap();
        assert_eq!(plan.segments.len(), 121);
        assert_eq!(plan.active_bytes, 7680 * 1080);
        assert!(plan.mapped_extent > plan.active_bytes);
        assert_active_gob_copy_matches_reference(7680, 1080, 5);
    }

    #[test]
    fn active_gob_copy_preflight_rejects_bad_width_extent_and_overlap() {
        assert!(active_gob_copy_plan(65, 64, 2).is_none());
        let plan = active_gob_copy_plan(128, 19, 2).unwrap();
        let extent = plan.mapped_extent;
        assert!(active_gob_copy_ranges_valid(
            &plan, 0x10_000, extent, 0x10_000, extent,
        ));
        assert!(active_gob_copy_ranges_valid(
            &plan,
            0x10_000,
            extent,
            0x10_000 + extent as u64,
            extent,
        ));
        assert!(!active_gob_copy_ranges_valid(
            &plan,
            0x10_000,
            extent,
            0x10_000 + extent as u64 - 1,
            extent,
        ));
        assert!(!active_gob_copy_ranges_valid(
            &plan,
            0x10_000,
            extent - 1,
            0x20_000,
            extent,
        ));
    }

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
