use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use super::engines::maxwell3d::{DrawCall, GsDebugRegs, PendingSemaphoreWrite};
use super::engines::{Fermi2D, KeplerCompute, MaxwellDma};
use super::engines::{KeplerMemory, KeplerMemoryWriteOutcome};
use super::pusher::{
    contiguous_constbuf_write_run_end, gpu_profile_enabled, semrel_legacy, semrel_verify,
};
use super::vk_dispatch::{PendingDrawBatch, PreparedDrawPacketizer, SsboSnapshotCache};
use super::GpuMappings;
use crate::PipelineStats;
use std::sync::atomic::Ordering as AtomicOrdering;

pub(crate) enum PrepEvent {
    EngineMethod {
        class: u32,
        method: u32,
        arg: u32,
        is_last: bool,
    },
    EngineMethods {
        class: u32,
        methods: Vec<(u32, u32, bool)>,
    },
    InlineUploadMethods(Vec<(u32, u32)>),
    ConstbufWrites {
        writes: Vec<(u64, u32)>,
        trace: Option<(u64, u32)>,
    },
    Draws {
        draws: Vec<DrawCall>,
        gs_debug: Option<GsDebugRegs>,
        replay_constbuf_writes: Vec<(u64, u32)>,
        constbuf_trace: Option<(u64, u32)>,
    },
    SemRelease(Vec<PendingSemaphoreWrite>),
    Barrier {
        barrier_flushes: u32,
        fragment_barriers: u32,
        tiled_cache_barriers: u32,
        texture_invalidates: u32,
    },
    PullerSemWrite {
        gpu_va: u64,
        payload: u32,
        long: bool,
    },
    KickBegin {
        swept: Vec<(u64, usize)>,
        dirty: Vec<(u64, usize)>,
    },
    EntryBegin,
    HardFlush {
        reason: usize,
        clear_ssbo: bool,
    },
    KickEnd {
        hard_after: bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    },
    Present {
        job: crate::render_thread::RenderJob,
        flush_small_rts: bool,
        on_prepared: Option<crate::PresentPrepared>,
    },
    DrainBarrier {
        done: crossbeam::channel::Sender<bool>,
        flush_small_rts: bool,
    },
    SemAcquire {
        gpu_va: u64,
        payload: u32,
        ack: crossbeam::channel::Sender<()>,
    },
    SetRenderer(Option<Arc<nexium_gpu::Renderer>>),
    SetGuestMemory(Option<super::GuestMemoryAccess>),
    CompleteAfterGuestWrites(Box<dyn FnOnce() + Send>),
}

pub(crate) enum DrawVecRecycle<'a> {
    Inline(&'a mut Vec<DrawCall>),
    Threaded(&'a crossbeam::channel::Sender<Vec<DrawCall>>),
    Discard,
}

impl DrawVecRecycle<'_> {
    fn recycle(self, mut draws: Vec<DrawCall>) -> bool {
        match self {
            Self::Inline(destination) => {
                draws.clear();
                *destination = draws;
                true
            }
            Self::Threaded(tx) => recycle_processed_draw_vec(Some(tx), draws),
            Self::Discard => recycle_processed_draw_vec(None, draws),
        }
    }
}

fn eager_clear_resolve_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_EAGER_CLEAR_RESOLVE").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

const DRAW_TARGET_RANGE_SLOTS: usize = 9;

fn draw_target_ranges_array(
    draw: &super::engines::maxwell3d::DrawCall,
) -> ([(u64, u64); DRAW_TARGET_RANGE_SLOTS], usize) {
    const CONSERVATIVE_BYTES_PER_PIXEL: u64 = 16;
    let mut ranges = [(0u64, 0u64); DRAW_TARGET_RANGE_SLOTS];
    let mut count = 0usize;
    for rt in &draw.rt {
        let base = (u64::from(rt.address_hi) << 32) | u64::from(rt.address_lo);
        if base == 0 || count == ranges.len() {
            continue;
        }
        let size = u64::from(rt.width)
            .saturating_mul(u64::from(rt.height))
            .saturating_mul(u64::from(rt.depth.max(1)))
            .saturating_mul(CONSERVATIVE_BYTES_PER_PIXEL)
            .max(u64::from(rt.layer_stride).saturating_mul(u64::from(rt.depth.max(1))));
        ranges[count] = (base, size);
        count += 1;
    }
    if draw.zeta_enable && count < ranges.len() {
        let base = (u64::from(draw.zeta.address_hi) << 32) | u64::from(draw.zeta.address_lo);
        if base != 0 {
            let size = u64::from(draw.zeta.width)
                .saturating_mul(u64::from(draw.zeta.height))
                .saturating_mul(CONSERVATIVE_BYTES_PER_PIXEL)
                .max(u64::from(draw.zeta.array_pitch));
            ranges[count] = (base, size);
            count += 1;
        }
    }
    (ranges, count)
}

fn draw_target_ranges(draw: &super::engines::maxwell3d::DrawCall) -> Vec<(u64, u64)> {
    let (ranges, count) = draw_target_ranges_array(draw);
    ranges[..count].to_vec()
}

fn draws_write_pending_compute(draws: &[DrawCall], ranges: &[(u64, u64)]) -> bool {
    draws.iter().any(|draw| {
        let (targets, count) = draw_target_ranges_array(draw);
        targets[..count].iter().any(|&(base, size)| {
            ranges.iter().any(|&(address, length)| ranges_overlap(base, size, address, length))
        })
    })
}

fn draws_write_pending_compute_sorted(draws: &[DrawCall], merged: &[(u64, u64)]) -> bool {
    if merged.is_empty() {
        return false;
    }
    draws.iter().any(|draw| {
        let (targets, count) = draw_target_ranges_array(draw);
        targets[..count]
            .iter()
            .any(|&(base, size)| sorted_ranges_overlap(merged, base, size))
    })
}

fn draws_write_pending_compute_any(draws: &[DrawCall], barrier: &ComputeMemoryBarrier) -> bool {
    if compute_target_fast_enabled() {
        draws_write_pending_compute_sorted(draws, &barrier.gpu_ranges_merged)
    } else {
        draws_write_pending_compute(draws, &barrier.gpu_ranges)
    }
}

fn sorted_ranges_overlap(merged: &[(u64, u64)], base: u64, size: u64) -> bool {
    if size == 0 {
        return false;
    }
    let end = base.saturating_add(size);
    let index = merged.partition_point(|&(range_base, _)| range_base < end);
    index != 0 && {
        let (range_base, range_size) = merged[index - 1];
        range_size != 0 && range_base.saturating_add(range_size) > base
    }
}

fn coalesce_sorted_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for &(base, size) in ranges {
        if size == 0 {
            continue;
        }
        let end = base.saturating_add(size);
        if let Some(last) = merged.last_mut() {
            let last_end = last.0.saturating_add(last.1);
            if base <= last_end {
                last.1 = end.max(last_end) - last.0;
                continue;
            }
        }
        merged.push((base, size));
    }
    merged
}

fn compute_target_fast_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_COMPUTE_TARGET_FAST").map_or(true, |value| value != "0")
    })
}

fn barrier_span_reuse_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_BARRIER_SPAN_REUSE").map_or(true, |value| value != "0")
    })
}

fn ranges_overlap(a: u64, a_len: u64, b: u64, b_len: u64) -> bool {
    a_len != 0 && b_len != 0 && a < b.saturating_add(b_len) && b < a.saturating_add(a_len)
}

thread_local! {
    static COMPUTE_BARRIER_BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct ComputeBarrierBypassGuard(bool);

impl Drop for ComputeBarrierBypassGuard {
    fn drop(&mut self) {
        COMPUTE_BARRIER_BYPASS.with(|bypass| bypass.set(self.0));
    }
}

pub(crate) fn compute_barrier_bypass() -> ComputeBarrierBypassGuard {
    ComputeBarrierBypassGuard(COMPUTE_BARRIER_BYPASS.with(|bypass| bypass.replace(true)))
}

fn kc_barrier_backtrace_filter() -> Option<Option<u64>> {
    static FILTER: std::sync::OnceLock<Option<Option<u64>>> = std::sync::OnceLock::new();
    *FILTER.get_or_init(|| {
        let value = std::env::var("NEXIUM_KC_BARRIER_BACKTRACE").ok()?;
        let value = value.trim();
        let parsed = value
            .strip_prefix("0x")
            .and_then(|hex| u64::from_str_radix(hex, 16).ok())
            .or_else(|| value.parse::<u64>().ok());
        Some(parsed.filter(|len| *len > 1))
    })
}

fn compute_barrier_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_KC_BARRIER_DISABLE").is_some())
}

fn deferred_compute_draw_resolve_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_DEFER_COMPUTE_DRAW_RESOLVE").ok().as_deref(),
            Some("0" | "false" | "off" | "no")
        )
    })
}

#[derive(Clone, Default)]
struct ComputeMemoryBarrier {
    cpu_ranges: Arc<[(u64, u64)]>,
    gpu_ranges: Arc<[(u64, u64)]>,
    gpu_ranges_merged: Arc<[(u64, u64)]>,
    spans: Arc<[super::engines::maxwell_compute::PendingComputeWritebackSpan]>,
    revision: u64,
    armed: std::cell::Cell<bool>,
}

impl ComputeMemoryBarrier {
    fn new(
        spans: &[super::engines::maxwell_compute::PendingComputeWritebackSpan],
        mappings: &GpuMappings,
    ) -> Self {
        let mut gpu_ranges = Vec::new();
        let mut cpu_ranges = Vec::new();
        for span in spans {
            gpu_ranges.push((span.gpu_va, span.len as u64));
            cpu_ranges.push((span.cpu_addr, span.len as u64));
            gpu_ranges.extend(mappings.gpu_regions_for_cpu_range(span.cpu_addr, span.len as u64));
        }
        gpu_ranges.sort_unstable();
        gpu_ranges.dedup();
        let gpu_ranges_merged = coalesce_sorted_ranges(&gpu_ranges);
        Self {
            cpu_ranges: cpu_ranges.into(),
            gpu_ranges: gpu_ranges.into(),
            gpu_ranges_merged: gpu_ranges_merged.into(),
            spans: spans.to_vec().into(),
            revision: 0,
            armed: std::cell::Cell::new(!spans.is_empty()),
        }
    }

    fn before_access(&self, address: u64, len: usize, resolve: impl FnOnce()) {
        if !self.armed.get()
            || COMPUTE_BARRIER_BYPASS.with(|bypass| bypass.get())
            || compute_barrier_disabled()
        {
            return;
        }
        let Some(&(base, size)) = self
            .cpu_ranges
            .iter()
            .find(|&&(base, size)| ranges_overlap(address, len as u64, base, size))
        else {
            return;
        };
        if crate::env_flag_set!("NEXIUM_KC_BARRIER_TRACE") {
            use std::sync::atomic::{AtomicU64, Ordering};
            static TRIGGERS: AtomicU64 = AtomicU64::new(0);
            let trigger = TRIGGERS.fetch_add(1, Ordering::Relaxed);
            log::warn!(
                "[kc-barrier] access={:#x}+{:#x} pending={:#x}+{:#x} ranges={} trigger={}",
                address,
                len,
                base,
                size,
                self.cpu_ranges.len(),
                trigger
            );
            let capture = match kc_barrier_backtrace_filter() {
                Some(Some(filter)) => {
                    static CAPTURED: AtomicU64 = AtomicU64::new(0);
                    len as u64 == filter
                        && super::engines::maxwell_compute::resolve_depth() == 0
                        && CAPTURED.fetch_add(1, Ordering::Relaxed) < 24
                }
                Some(None) => trigger % 400 == 0 && trigger < 24_000,
                None => false,
            };
            if capture {
                log::warn!(
                    "[kc-barrier-bt] trigger={} len={:#x} {}",
                    trigger,
                    len,
                    std::backtrace::Backtrace::force_capture()
                );
            }
        }
        self.armed.set(false);
        resolve();
    }

    fn invalidate_snapshots(&self, cache: &mut SsboSnapshotCache, mappings: &GpuMappings) {
        let writes: Vec<_> = self.gpu_ranges.iter()
            .map(|&(address, size)| (address, size as usize)).collect();
        cache.invalidate_gpu_writes(mappings, &writes);
        for &(address, size) in self.gpu_ranges.iter() {
            nexium_gpu::tex_invalidate::bump_region(address, size);
        }
    }
}

#[derive(Default)]
struct ComputeBarrierCache {
    revision: Option<(u64, u64)>,
    barrier: ComputeMemoryBarrier,
}

impl ComputeBarrierCache {
    fn get(
        &mut self,
        revision: u64,
        spans: &[super::engines::maxwell_compute::PendingComputeWritebackSpan],
        mappings: &GpuMappings,
    ) -> ComputeMemoryBarrier {
        let revision = (revision, mappings.generation());
        if self.revision != Some(revision) {
            self.barrier = ComputeMemoryBarrier::new(spans, mappings);
            self.barrier.revision = revision.0;
            self.revision = Some(revision);
        }
        self.barrier.clone()
    }

    fn get_current(&mut self, mappings: &GpuMappings) -> ComputeMemoryBarrier {
        let revision = super::engines::maxwell_compute::pending_writeback_revision();
        if barrier_span_reuse_enabled()
            && self.revision == Some((revision, mappings.generation()))
        {
            return self.barrier.clone();
        }
        let spans = super::engines::maxwell_compute::pending_writeback_spans_snapshot();
        self.get(revision, &spans, mappings)
    }
}

fn dma_narrow_resolve_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_DMA_NARROW_RESOLVE").ok().as_deref(),
            Some("0") | Some("false") | Some("off") | Some("no")
        )
    })
}

fn mirror_sweep_per_entry_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_MIRROR_SWEEP_PER_ENTRY")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

fn engb_prof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_PREP_PROFILE").is_some())
}

fn bulk_compute_upload_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_BULK_COMPUTE_UPLOAD").ok().as_deref(),
            Some("1")
                | Some("true")
                | Some("TRUE")
                | Some("on")
                | Some("ON")
                | Some("yes")
                | Some("YES")
        )
    })
}

fn texture_cache_invalidate_clear_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        texture_cache_invalidate_clear_value_enabled(
            std::env::var_os("NEXIUM_TIC_INVALIDATE_CLEAR").as_deref(),
        )
    })
}

fn texture_cache_invalidate_clear_value_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| {
        !matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )
    })
}

fn nonterminal_inline_data_run_end(methods: &[(u32, u32, bool)], start: usize) -> usize {
    let mut end = start;
    while let Some(&(method, _, is_last)) = methods.get(end) {
        if !KeplerCompute::is_nonterminal_inline_data(method, is_last) {
            break;
        }
        end += 1;
    }
    end
}

fn engb_prof_record(class: u32, methods: usize, elapsed: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static METHODS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static BATCHES: AtomicU64 = AtomicU64::new(0);
    let slot = match class {
        super::engines::MAXWELL_DMA_CLASS => 0,
        super::engines::FERMI_2D_CLASS => 1,
        super::engines::KEPLER_MEMORY_CLASS => 2,
        super::engines::KEPLER_COMPUTE_CLASS => 3,
        _ => 4,
    };
    NS[slot].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    METHODS[slot].fetch_add(methods as u64, Ordering::Relaxed);
    let batches = BATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    if batches % 8192 == 0 {
        let mut parts = Vec::with_capacity(5);
        for (slot, name) in ["dma", "fermi", "kmem", "kcomp", "other"]
            .iter()
            .enumerate()
        {
            let ms = NS[slot].swap(0, Ordering::Relaxed) as f64 / 1_000_000.0;
            let methods = METHODS[slot].swap(0, Ordering::Relaxed);
            parts.push(format!("{}={:.1}ms/n{}", name, ms, methods));
        }
        log::warn!("[engb-prof] window {}", parts.join(" "));
    }
}

fn draw_prof_start() -> Option<std::time::Instant> {
    engb_prof_enabled().then(std::time::Instant::now)
}

fn draw_prof_record(flushes: u64, draws: u64, started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::{AtomicU64, Ordering};
    static NS: AtomicU64 = AtomicU64::new(0);
    static FLUSHES: AtomicU64 = AtomicU64::new(0);
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    DRAWS.fetch_add(draws, Ordering::Relaxed);
    let flushes_total = FLUSHES.fetch_add(flushes, Ordering::Relaxed) + flushes;
    if flushes_total % 4096 < flushes {
        log::warn!(
            "[draw-prof] window flush_ms={:.1} flushes={} draws={}",
            NS.swap(0, Ordering::Relaxed) as f64 / 1_000_000.0,
            flushes_total,
            DRAWS.swap(0, Ordering::Relaxed)
        );
    }
}

pub(crate) struct PrepEngines<'a> {
    pub(crate) maxwell_dma: &'a mut MaxwellDma,
    pub(crate) fermi_2d: &'a mut Fermi2D,
    pub(crate) kepler_compute: &'a mut KeplerCompute,
    pub(crate) kepler_memory: &'a mut KeplerMemory,
}

fn rt_invalidation_prelude_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_RT_INVALIDATION_PRELUDE").map_or(true, |value| value != "0")
    })
}

fn ssbo_refresh_interval() -> Duration {
    static INTERVAL: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        Duration::from_micros(
            std::env::var("NEXIUM_SSBO_REFRESH_US")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(1000),
        )
    })
}

pub(crate) struct PrepState {
    pub(crate) renderer: Option<Arc<nexium_gpu::Renderer>>,
    pub(crate) guest_memory: Option<super::GuestMemoryAccess>,
    pending_small_rt_wb:
        Option<std::sync::mpsc::Receiver<(bool, Vec<super::vk_dispatch::GuestWriteChunk>)>>,
    pending_cpu_rt_wb:
        Vec<std::sync::mpsc::Receiver<(bool, Vec<super::vk_dispatch::GuestWriteChunk>)>>,
    pub(crate) vk_batch: PendingDrawBatch,
    pub(crate) prepared_draw_packets: PreparedDrawPacketizer,
    vk_flush_completed: bool,
    pub(crate) ssbo_snapshot_cache: SsboSnapshotCache,
    pending_storage_readbacks: Vec<super::vk_dispatch::PendingStorageReadback>,
    pub(crate) inline_upload: KeplerMemory,
    pub(crate) constbuf_invalidation_scratch: Vec<(u64, usize)>,
    pub(crate) constbuf_patched_scratch: Vec<(u64, u64, usize)>,
    pub(crate) constbuf_bytes_scratch: Vec<u8>,
    compute_snapshot_revision: Option<(u64, u64)>,
    compute_barrier_cache: ComputeBarrierCache,
    renderer_guest_writes_pending: bool,
    ssbo_refreshed_at: Option<std::time::Instant>,
    ssbo_refreshed_epoch: u64,
    mirror_acquire_epoch: u64,
    #[cfg(test)]
    pub(crate) prepared_packet_drain_counts: PreparedPacketDrainCounts,
}

impl PrepState {
    fn invalidate_compute_snapshots(&mut self, barrier: &ComputeMemoryBarrier, mappings: &GpuMappings) {
        let revision = (
            super::engines::maxwell_compute::pending_writeback_revision(),
            mappings.generation(),
        );
        if self.compute_snapshot_revision != Some(revision) {
            barrier.invalidate_snapshots(&mut self.ssbo_snapshot_cache, mappings);
            self.compute_snapshot_revision = Some(revision);
        }
    }

    pub(crate) fn post_compute_overlay_retire(&self) {
        let retired = super::engines::maxwell_compute::take_retired_compute_serials();
        if retired.is_empty() {
            return;
        }
        let (Some(render_thread), Some(renderer)) = (
            crate::render_thread::maybe_render_thread(),
            self.renderer.clone(),
        ) else {
            return;
        };
        render_thread.submit_named(
            "compute-overlay-retire",
            Box::new(move || renderer.retire_compute_overlay_serials(&retired)),
        );
    }

    pub(crate) fn schedule_kick_completion(&self, on_complete: Option<Box<dyn FnOnce() + Send>>) {
        let Some(on_complete) = on_complete else {
            return;
        };
        if super::accuracy::normal_accuracy() && !super::completion::guest_writes_pending() {
            on_complete();
            return;
        }
        let Some(renderer) = self.renderer.clone() else {
            on_complete();
            return;
        };
        if !super::completion::submit_renderer_completion(renderer, on_complete) {
            log::error!(
                "[gpu-sync] failed to enqueue GPFIFO completion behind the renderer timeline"
            );
        }
    }

    pub(crate) fn complete_after_guest_writes(&mut self, callback: Box<dyn FnOnce() + Send>) {
        if std::mem::take(&mut self.renderer_guest_writes_pending) {
            if let Some(renderer) = self.renderer.clone() {
                super::completion::submit_renderer_completion(renderer, callback);
                return;
            }
        }
        callback();
    }

    pub(crate) fn run_event(
        &mut self,
        event: PrepEvent,
        draw_vec_recycle: DrawVecRecycle<'_>,
        engines: &mut PrepEngines,
        mappings: &GpuMappings,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> bool {
        use super::pusher::kickprof;
        match event {
            PrepEvent::InlineUploadMethods(methods) => {
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                self.process_inline_upload_methods(methods, mappings, mem_read, mem_write);
                true
            }
            PrepEvent::KickBegin { swept, dirty } => {
                self.drain_landed_compute(mappings);
                self.publish_pending_storage_readbacks(false, mappings, mem_read, mem_write);
                self.begin_ssbo_snapshot_epoch_with(Some((swept, dirty)));
                true
            }
            PrepEvent::EntryBegin => {
                self.begin_ssbo_snapshot_entry();
                true
            }
            PrepEvent::HardFlush { reason, clear_ssbo } => {
                self.record_flush_reason(reason);
                self.flush_vk(mappings, mem_read, mem_write);
                if clear_ssbo {
                    self.ssbo_snapshot_cache.retain_watchable_full_aurora_snapshots();
                }
                true
            }
            PrepEvent::KickEnd {
                hard_after,
                writeback_small_rts,
                on_complete,
            } => {
                let joined = self.join_small_rt_writeback(mappings);
                super::watchdog::phase(super::watchdog::Phase::KickEnd, u64::from(hard_after));
                let kp_tail = kickprof::start();
                self.land_or_resolve_pending_compute(mappings, mem_write);
                kickprof::add(kickprof::RESOLVE_TAIL, kp_tail);
                if hard_after {
                    self.record_flush_reason(kickprof::FLUSH_HARD_TAIL);
                }
                self.flush_vk_with_boundary(mappings, mem_read, mem_write, hard_after);
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                let submitted = self.finish_prepared_draw_packet_tail(hard_after);
                self.post_compute_overlay_retire();
                let writeback_completed = if !joined {
                    false
                } else if writeback_small_rts
                    && super::vk_dispatch::has_pending_small_rt_writebacks()
                {
                    if let Some(r) = self.renderer.clone() {
                        let kp = kickprof::start();
                        let completed = self.writeback_small_rts(&r, mappings, mem_write);
                        kickprof::add(kickprof::SMALLRT, kp);
                        completed
                    } else {
                        false
                    }
                } else {
                    true
                };
                if joined
                    && !writeback_small_rts
                    && super::vk_dispatch::cpu_readable_rt_writeback_mode()
                        == super::vk_dispatch::CpuReadableRtWritebackMode::Kick
                    && super::vk_dispatch::has_pending_cpu_readable_rt_writebacks()
                {
                    if let Some(r) = self.renderer.clone() {
                        let kp = kickprof::start();
                        self.writeback_cpu_readable_rts(&r, mappings, mem_write);
                        kickprof::add(kickprof::SMALLRT, kp);
                    }
                }
                self.end_ssbo_snapshot_epoch();
                let completed = joined && submitted && writeback_completed;
                if completed {
                    self.schedule_kick_completion(on_complete);
                }
                if crate::kick_timeline_enabled() {
                    log::warn!(
                        "[ktl] us={} kick end stages: {}",
                        crate::timeline_us(),
                        kickprof::take_kick_stage_report()
                    );
                }
                completed
            }
            PrepEvent::Present {
                job,
                flush_small_rts,
                on_prepared,
            } => {
                let kp_wb = super::pusher::kickprof::start();
                let joined = self.join_small_rt_writeback(mappings);
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                let kp_flush = super::pusher::kickprof::start();
                let flushed = self.flush_prepared_draw_packets();
                super::pusher::kickprof::add(super::pusher::kickprof::PRES_WB_FLUSH, kp_flush);
                let completed = if !joined || !flushed {
                    false
                } else if flush_small_rts && super::vk_dispatch::has_pending_small_rt_writebacks() {
                    if let Some(renderer) = self.renderer.clone() {
                        let async_memory = super::vk_dispatch::async_small_rt_writeback_enabled()
                            .then(|| self.guest_memory.clone())
                            .flatten()
                            .filter(super::GuestMemoryAccess::is_available);
                        if let Some(memory) = async_memory {
                            if let Some(pending) =
                                super::vk_dispatch::spawn_small_rt_writeback_async(
                                    &renderer, memory,
                                )
                            {
                                self.pending_small_rt_wb = Some(pending);
                                true
                            } else {
                                super::vk_dispatch::writeback_small_rts(
                                    &renderer,
                                    mappings,
                                    mem_write,
                                    &mut self.ssbo_snapshot_cache,
                                )
                            }
                        } else {
                            super::vk_dispatch::writeback_small_rts(
                                &renderer,
                                mappings,
                                mem_write,
                                &mut self.ssbo_snapshot_cache,
                            )
                        }
                    } else {
                        false
                    }
                } else {
                    true
                };
                if completed
                    && super::vk_dispatch::cpu_readable_rt_writeback_mode()
                        == super::vk_dispatch::CpuReadableRtWritebackMode::Present
                    && super::vk_dispatch::has_pending_cpu_readable_rt_writebacks()
                {
                    if let Some(renderer) = self.renderer.clone() {
                        let kp = super::pusher::kickprof::start();
                        self.writeback_cpu_readable_rts(&renderer, mappings, mem_write);
                        super::pusher::kickprof::add(super::pusher::kickprof::SMALLRT, kp);
                    }
                }
                super::pusher::kickprof::add(super::pusher::kickprof::PRES_WB, kp_wb);
                if completed {
                    let kp_submit = super::pusher::kickprof::start();
                    if let Some(on_prepared) = on_prepared {
                        on_prepared();
                    }
                    if let Some(rt) = crate::render_thread::maybe_render_thread() {
                        rt.submit_named("async-present-readback", job);
                    } else {
                        job();
                    }
                    super::pusher::kickprof::add(super::pusher::kickprof::PRES_SUBMIT, kp_submit);
                }
                completed
            }
            PrepEvent::DrainBarrier {
                done,
                flush_small_rts,
            } => {
                let joined = self.join_small_rt_writeback(mappings);
                let flushed = self.flush_prepared_draw_packets();
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                let writeback_completed = if !joined || !flushed {
                    false
                } else if flush_small_rts && super::vk_dispatch::has_pending_small_rt_writebacks() {
                    if let Some(renderer) = self.renderer.clone() {
                        super::vk_dispatch::writeback_small_rts(
                            &renderer,
                            mappings,
                            mem_write,
                            &mut self.ssbo_snapshot_cache,
                        )
                    } else {
                        false
                    }
                } else {
                    true
                };
                let completed = joined && flushed && writeback_completed;
                let _ = done.send(completed);
                completed
            }
            PrepEvent::SemAcquire {
                gpu_va,
                payload,
                ack,
            } => {
                if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                    let start = std::time::Instant::now();
                    let mut last = 0u32;
                    loop {
                        let mut buf = [0u8; 4];
                        if mem_read(cpu, &mut buf) {
                            last = u32::from_le_bytes(buf);
                            if last == payload || (last.wrapping_sub(payload) as i32) >= 0 {
                                break;
                            }
                        }
                        if start.elapsed() >= Duration::from_secs(3) {
                            log::warn!(
                                "[sem-acquire] prep timeout gpu_va={:#x} payload={:#x} last={:#x}",
                                gpu_va,
                                payload,
                                last
                            );
                            break;
                        }
                        std::thread::sleep(Duration::from_micros(100));
                    }
                } else {
                    log::warn!(
                        "[sem-acquire] prep unmapped gpu_va={:#x} payload={:#x}",
                        gpu_va,
                        payload
                    );
                }
                self.ssbo_snapshot_cache.mirror_bump_sweep();
                self.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
                let _ = ack.send(());
                true
            }
            PrepEvent::SetRenderer(renderer) => {
                self.set_renderer(renderer);
                true
            }
            PrepEvent::CompleteAfterGuestWrites(callback) => {
                self.complete_after_guest_writes(callback);
                true
            }
            PrepEvent::SetGuestMemory(memory) => {
                self.set_guest_memory_access(memory);
                true
            }
            PrepEvent::ConstbufWrites { writes, trace } => {
                let kp = kickprof::start();
                self.sync_semaphore_acquire_sweep();
                self.commit_constbuf_writes(&writes, trace, mappings, mem_write);
                kickprof::add(kickprof::CBUFWB, kp);
                true
            }
            PrepEvent::PullerSemWrite {
                gpu_va,
                payload,
                long,
            } => {
                super::watchdog::phase(super::watchdog::Phase::Semaphore, gpu_va);
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                if super::accuracy::normal_accuracy() {
                    self.land_or_resolve_pending_compute(mappings, mem_write);
                } else {
                    self.resolve_pending_compute(mappings, mem_write);
                }
                self.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
                self.ssbo_snapshot_cache.invalidate_gpu_write(
                    mappings,
                    gpu_va,
                    if long { 16 } else { 4 },
                );
                if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                    if long {
                        let ts = super::clock::report_timestamp();
                        let mut buf = [0u8; 16];
                        buf[0..8].copy_from_slice(&(payload as u64).to_le_bytes());
                        buf[8..16].copy_from_slice(&ts.to_le_bytes());
                        mem_write(cpu, &buf);
                    } else {
                        mem_write(cpu, &payload.to_le_bytes());
                    }
                    log::trace!(
                        "puller: semaphore write gpu_va={:#x} cpu={:#x} payload={:#x} long={}",
                        gpu_va,
                        cpu,
                        payload,
                        long
                    );
                } else {
                    log::warn!(
                        "puller: semaphore write gpu_va={:#x} not mapped — payload={:#x} dropped",
                        gpu_va,
                        payload
                    );
                }
                true
            }
            PrepEvent::Barrier {
                mut barrier_flushes,
                fragment_barriers,
                tiled_cache_barriers,
                mut texture_invalidates,
            } => {
                if super::pusher::async_barrier_segments_enabled() {
                    if tiled_cache_barriers != 0 {
                        barrier_flushes = barrier_flushes.saturating_sub(tiled_cache_barriers);
                        kickprof::count(kickprof::BARRIER_TILED_NOOP, tiled_cache_barriers as u64);
                    }
                    if !self.vk_batch.is_empty()
                        && (fragment_barriers != 0 || texture_invalidates != 0)
                    {
                        let call = self.vk_batch.last_mut().unwrap();
                        if fragment_barriers != 0 {
                            call.fragment_barrier_after = true;
                            barrier_flushes = barrier_flushes.saturating_sub(fragment_barriers);
                            kickprof::count(
                                kickprof::BARRIER_SEGMENT_FRAGMENT,
                                fragment_barriers as u64,
                            );
                        }
                        if texture_invalidates != 0 {
                            call.fragment_barrier_after = true;
                            call.texture_cache_invalidate_after = true;
                            barrier_flushes = barrier_flushes.saturating_sub(texture_invalidates);
                            kickprof::count(
                                kickprof::BARRIER_SEGMENT_TEXTURE_INVALIDATE,
                                texture_invalidates as u64,
                            );
                            texture_invalidates = 0;
                        }
                    }
                }
                if barrier_flushes != 0 || texture_invalidates != 0 {
                    if barrier_flushes != 0 {
                        self.record_flush_reason(kickprof::FLUSH_SOFT_BARRIER);
                    }
                    if texture_invalidates != 0 {
                        self.record_flush_reason(kickprof::FLUSH_SOFT_TEXTURE_INVALIDATE);
                    }
                    self.flush_vk_soft(mappings, mem_read, mem_write);
                    let kp = kickprof::start();
                    if texture_invalidates != 0 {
                        if texture_cache_invalidate_clear_enabled() {
                            if let Some(r) = self.renderer.clone() {
                                if let Some(rt) = crate::render_thread::maybe_render_thread() {
                                    let flushed = self.flush_prepared_draw_packets();
                                    let submitted = rt.submit_timeout_named(
                                        "texture-cache-invalidate",
                                        Box::new(move || r.clear_texture_cache()),
                                        std::time::Duration::from_secs(3),
                                    );
                                    self.vk_flush_completed &= flushed && submitted;
                                } else {
                                    r.clear_texture_cache();
                                }
                            }
                        }
                    }
                    kickprof::add(kickprof::BARRIER, kp);
                    if super::pusher::maxwell_sync_debug_enabled() {
                        log::warn!(
                            "[gpu-sync] maxwell barriers={} texture_invalidates={}",
                            barrier_flushes,
                            texture_invalidates
                        );
                    }
                }
                true
            }
            PrepEvent::SemRelease(writes) => {
                super::watchdog::phase(super::watchdog::Phase::Semaphore, writes.len() as u64);
                self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                let can_complete_asynchronously =
                    super::completion::async_semaphore_completion_enabled()
                        && self
                            .renderer
                            .as_ref()
                            .is_some_and(|renderer| renderer.timeline_sync_available())
                        && self
                            .guest_memory
                            .as_ref()
                            .is_some_and(super::GuestMemoryAccess::is_available)
                        && writes
                            .iter()
                            .all(|write| write.can_complete_asynchronously());
                let synthetic_only = !writes.is_empty()
                    && writes
                        .iter()
                        .all(|write| !write.requires_renderer_completion());
                let immediate_synthetic_writes =
                    synthetic_only && super::accuracy::normal_accuracy();
                let defer_synthetic_writes = !can_complete_asynchronously
                    && !immediate_synthetic_writes
                    && synthetic_only
                    && super::engines::maxwell_compute::has_pending_writebacks()
                    && self.compute_landing_possible();
                if can_complete_asynchronously
                    || defer_synthetic_writes
                    || immediate_synthetic_writes
                {
                    self.land_or_resolve_pending_compute(mappings, mem_write);
                } else {
                    self.resolve_pending_compute(mappings, mem_write);
                }
                self.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
                if defer_synthetic_writes {
                    for write in &writes {
                        self.ssbo_snapshot_cache.invalidate_gpu_write(
                            mappings,
                            write.gpu_va,
                            if write.long { 16 } else { 4 },
                        );
                    }
                    let renderer = self.renderer.as_ref().unwrap().clone();
                    let memory = self.guest_memory.as_ref().unwrap().clone();
                    let count = writes.len();
                    let scheduled = super::completion::submit_renderer_completion(
                        renderer,
                        move || {
                            for write in writes {
                                let written = if write.long {
                                    let mut buf = [0u8; 16];
                                    buf[0..8].copy_from_slice(&u64::from(write.payload).to_le_bytes());
                                    buf[8..16]
                                        .copy_from_slice(&super::clock::report_timestamp().to_le_bytes());
                                    memory.write_gpu(write.gpu_va, &buf)
                                } else {
                                    memory.write_gpu(write.gpu_va, &write.payload.to_le_bytes())
                                };
                                if written.is_none() {
                                    log::warn!(
                                        "pusher: deferred report gpu_va={:#x} not mapped; payload={:#x} dropped",
                                        write.gpu_va,
                                        write.payload
                                    );
                                }
                            }
                        },
                    );
                    if scheduled {
                        self.renderer_guest_writes_pending = true;
                        stats
                            .fence_releases
                            .fetch_add(count as u64, AtomicOrdering::Relaxed);
                    } else {
                        log::error!("[gpu-sync] deferred report scheduling failed; payloads dropped");
                    }
                } else if can_complete_asynchronously && super::accuracy::normal_accuracy() {
                    let memory = self.guest_memory.as_ref().unwrap().clone();
                    let mut pending = Vec::with_capacity(writes.len());
                    for write in writes {
                        self.ssbo_snapshot_cache.invalidate_gpu_write(
                            mappings,
                            write.gpu_va,
                            if write.long { 16 } else { 4 },
                        );
                        pending.push((write.gpu_va, write.payload));
                    }
                    super::pusher::write_payload_fences(&pending, |gpu_va, bytes| {
                        memory.write_gpu_with_mappings(mappings, gpu_va, bytes)
                    });
                    stats
                        .fence_releases
                        .fetch_add(pending.len() as u64, AtomicOrdering::Relaxed);
                } else if can_complete_asynchronously {
                    let mut pending = Vec::with_capacity(writes.len());
                    for write in writes {
                        self.ssbo_snapshot_cache.invalidate_gpu_write(
                            mappings,
                            write.gpu_va,
                            if write.long { 16 } else { 4 },
                        );
                        pending.push((write.gpu_va, write.payload));
                    }

                    if !pending.is_empty() {
                        self.record_flush_reason(kickprof::FLUSH_HARD_SEMREL_ASYNC);
                        self.flush_vk(mappings, mem_read, mem_write);
                        let pending = Arc::new(pending);
                        let renderer = self.renderer.as_ref().unwrap().clone();
                        let memory = self.guest_memory.as_ref().unwrap().clone();
                        let async_pending = Arc::clone(&pending);
                        let async_memory = memory.clone();
                        let verify_renderer = semrel_verify().then(|| renderer.clone());
                        let kp = kickprof::start();
                        let scheduled = super::completion::submit_renderer_completion(
                            renderer,
                            move || {
                                if let Some(renderer) = verify_renderer {
                                    if !renderer.wait_idle_checked() {
                                        log::error!(
                                            "[gpu-sync] semaphore verification idle wait failed; payloads dropped"
                                        );
                                        return;
                                    }
                                }
                                super::pusher::write_payload_fences(
                                    &async_pending,
                                    |gpu_va, bytes| async_memory.write_gpu(gpu_va, bytes),
                                );
                            },
                        );
                        if scheduled {
                            self.renderer_guest_writes_pending = true;
                        } else {
                            if self.sync_renderer_idle("report-semaphore-fallback") {
                                super::pusher::write_payload_fences(&pending, |gpu_va, bytes| {
                                    memory.write_gpu_with_mappings(mappings, gpu_va, bytes)
                                });
                            } else {
                                log::error!(
                                    "[gpu-sync] semaphore fallback ordering failed; {} payloads dropped",
                                    pending.len()
                                );
                            }
                        }
                        kickprof::add(kickprof::SEMREL, kp);
                        stats
                            .fence_releases
                            .fetch_add(pending.len() as u64, AtomicOrdering::Relaxed);
                    }
                } else {
                    let mut renderer_completion = None;
                    for write in writes {
                        if write.requires_renderer_completion() {
                            let completed = if let Some(completed) = renderer_completion {
                                completed
                            } else {
                                self.record_flush_reason(kickprof::FLUSH_HARD_SEMREL_SYNC);
                                self.flush_vk(mappings, mem_read, mem_write);
                                let kp = kickprof::start();
                                let completed = self.sync_renderer_idle("report-semaphore");
                                kickprof::add(kickprof::SEMREL, kp);
                                renderer_completion = Some(completed);
                                completed
                            };
                            if !completed {
                                log::error!(
                                    "[gpu-sync] semaphore ordering failed gpu_va={:#x}; payload={:#x} dropped",
                                    write.gpu_va,
                                    write.payload
                                );
                                continue;
                            }
                        }
                        self.ssbo_snapshot_cache.invalidate_gpu_write(
                            mappings,
                            write.gpu_va,
                            if write.long { 16 } else { 4 },
                        );
                        if let Some(cpu) = mappings.cpu_address_for(write.gpu_va) {
                            let ok = if write.long {
                                let ts = super::clock::report_timestamp();
                                let mut buf = [0u8; 16];
                                buf[0..8].copy_from_slice(&(write.payload as u64).to_le_bytes());
                                buf[8..16].copy_from_slice(&ts.to_le_bytes());
                                mem_write(cpu, &buf)
                            } else {
                                mem_write(cpu, &write.payload.to_le_bytes())
                            };
                            if crate::syncpt_debug_enabled() {
                                log::info!(
                                    "[syncpt] fence release gpu_va={:#x} cpu={:#x} payload={:#x} long={} write_ok={}",
                                    write.gpu_va,
                                    cpu,
                                    write.payload,
                                    write.long,
                                    ok
                                );
                            }
                            stats.fence_releases.fetch_add(1, AtomicOrdering::Relaxed);
                        } else {
                            log::warn!(
                            "pusher: fence release gpu_va={:#x} not mapped — payload={:#x} dropped",
                            write.gpu_va,
                            write.payload
                        );
                        }
                    }
                }
                true
            }
            PrepEvent::Draws {
                draws,
                gs_debug,
                mut replay_constbuf_writes,
                constbuf_trace,
            } => {
                if !super::pusher::Pusher::entry_begin_events_enabled() {
                    self.begin_ssbo_snapshot_entry();
                }
                self.sync_semaphore_acquire_sweep();
                self.drain_landed_compute(mappings);
                self.join_cpu_rt_writebacks(mappings, 8);
                self.publish_pending_storage_readbacks(false, mappings, mem_read, mem_write);
                let kp_pre = kickprof::start();
                let memory_barrier = self.compute_barrier_cache.get_current(mappings);
                let compute_spans = memory_barrier.spans.clone();
                if !compute_spans.is_empty() {
                    let deferred = deferred_compute_draw_resolve_enabled()
                        && crate::render_thread::maybe_render_thread().is_some();
                    if eager_clear_resolve_enabled()
                        || (!deferred && draws.iter().any(|draw| !draw.is_clear))
                        || draws_write_pending_compute_any(&draws, &memory_barrier)
                    {
                        self.resolve_pending_compute(mappings, mem_write);
                        memory_barrier.armed.set(false);
                    } else {
                        self.invalidate_compute_snapshots(&memory_barrier, mappings);
                    }
                }
                self.ssbo_snapshot_cache.set_compute_pending(
                    memory_barrier.cpu_ranges.clone(),
                    memory_barrier.spans.clone(),
                    memory_barrier.revision,
                );
                let barrier_renderer = self.renderer.clone();
                let resolve_memory = || {
                    if let Some(renderer) = &barrier_renderer {
                        super::engines::maxwell_compute::resolve_pending_writebacks(
                            renderer, mappings, mem_write,
                        );
                    }
                };
                let guarded_read = |address, bytes: &mut [u8]| {
                    memory_barrier.before_access(address, bytes.len(), &resolve_memory);
                    mem_read(address, bytes)
                };
                let guarded_write = |address, bytes: &[u8]| {
                    memory_barrier.before_access(address, bytes.len(), &resolve_memory);
                    mem_write(address, bytes)
                };
                let mem_read: &dyn Fn(u64, &mut [u8]) -> bool = &guarded_read;
                let mem_write: &dyn Fn(u64, &[u8]) -> bool = &guarded_write;
                let mut compute_probe =
                    super::vk_dispatch::ComputeGraphicsProbe::new(&compute_spans);
                let mut committed_constbuf_writes = 0;
                let mut draws_completed = true;
                kickprof::add(kickprof::DRAWS_PRE, kp_pre);
                if let Some(r) = self.renderer.clone() {
                    if replay_constbuf_writes.is_empty() {
                        let kp = kickprof::start();
                        let completed = super::vk_dispatch::enqueue_draws(
                            &draws,
                            &mut self.vk_batch,
                            &mut self.prepared_draw_packets,
                            &mut self.ssbo_snapshot_cache,
                            &mut self.pending_storage_readbacks,
                            mappings,
                            gs_debug.as_ref(),
                            &r,
                            engines.maxwell_dma,
                            &*engines.fermi_2d,
                            mem_read,
                            mem_write,
                            compute_probe.as_mut(),
                        );
                        self.vk_flush_completed &= completed;
                        draws_completed &= completed;
                        kickprof::add(kickprof::ENQ, kp);
                    } else {
                        let mut draw_start = 0;
                        while draw_start < draws.len() {
                            let (draw_end, end) = super::pusher::constbuf_replay_group_end_by(
                                &draws,
                                draw_start,
                                committed_constbuf_writes,
                                replay_constbuf_writes.len(),
                                |draw| draw.constbuf_write_count,
                            );
                            if end != committed_constbuf_writes {
                                let kp = kickprof::start();
                                self.commit_constbuf_writes(
                                    &replay_constbuf_writes[committed_constbuf_writes..end],
                                    constbuf_trace,
                                    mappings,
                                    mem_write,
                                );
                                kickprof::add(kickprof::CBUFWB, kp);
                            }
                            committed_constbuf_writes = end;
                            let kp = kickprof::start();
                            let completed = super::vk_dispatch::enqueue_draws(
                                &draws[draw_start..draw_end],
                                &mut self.vk_batch,
                                &mut self.prepared_draw_packets,
                                &mut self.ssbo_snapshot_cache,
                                &mut self.pending_storage_readbacks,
                                mappings,
                                gs_debug.as_ref(),
                                &r,
                                engines.maxwell_dma,
                                &*engines.fermi_2d,
                                mem_read,
                                mem_write,
                                compute_probe.as_mut(),
                            );
                            self.vk_flush_completed &= completed;
                            draws_completed &= completed;
                            kickprof::add(kickprof::ENQ, kp);
                            if !completed {
                                break;
                            }
                            draw_start = draw_end;
                        }
                    }
                } else if replay_constbuf_writes.is_empty() {
                    super::engines::sw_renderer::execute_draws(
                        &draws,
                        mappings,
                        engines.maxwell_dma,
                        mem_read,
                        mem_write,
                    );
                } else {
                    let mut draw_start = 0;
                    while draw_start < draws.len() {
                        let (draw_end, end) = super::pusher::constbuf_replay_group_end_by(
                            &draws,
                            draw_start,
                            committed_constbuf_writes,
                            replay_constbuf_writes.len(),
                            |draw| draw.constbuf_write_count,
                        );
                        if end != committed_constbuf_writes {
                            let kp = kickprof::start();
                            self.commit_constbuf_writes(
                                &replay_constbuf_writes[committed_constbuf_writes..end],
                                constbuf_trace,
                                mappings,
                                mem_write,
                            );
                            kickprof::add(kickprof::CBUFWB, kp);
                        }
                        committed_constbuf_writes = end;
                        super::engines::sw_renderer::execute_draws(
                            &draws[draw_start..draw_end],
                            mappings,
                            engines.maxwell_dma,
                            mem_read,
                            mem_write,
                        );
                        draw_start = draw_end;
                    }
                }
                if draws_completed && committed_constbuf_writes < replay_constbuf_writes.len() {
                    let kp = kickprof::start();
                    self.commit_constbuf_writes(
                        &replay_constbuf_writes[committed_constbuf_writes..],
                        constbuf_trace,
                        mappings,
                        mem_write,
                    );
                    kickprof::add(kickprof::CBUFWB, kp);
                }
                replay_constbuf_writes.clear();
                self.invalidate_resolved_compute_writebacks(mappings);
                if let Some(probe) = compute_probe {
                    probe.finish();
                }
                draw_vec_recycle.recycle(draws);
                draws_completed
            }
            PrepEvent::EngineMethods { class, methods } => {
                super::watchdog::phase(
                    super::watchdog::Phase::Methods,
                    (u64::from(class) << 32) | methods.first().map_or(0, |m| u64::from(m.0)),
                );
                if class != super::engines::maxwell3d::MAXWELL3D_CLASS {
                    self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                }
                let started = engb_prof_enabled().then(std::time::Instant::now);
                let count = methods.len();
                if class == super::engines::KEPLER_COMPUTE_CLASS {
                    let bulk_upload = bulk_compute_upload_enabled();
                    let mut cursor = 0;
                    while cursor < methods.len() {
                        if bulk_upload {
                            let end = nonterminal_inline_data_run_end(&methods, cursor);
                            if end != cursor {
                                engines.kepler_compute.dispatch_inline_data_bulk(
                                    methods[cursor..end].iter().map(|&(_, arg, _)| arg),
                                );
                                cursor = end;
                                continue;
                            }
                        }

                        let (method, arg, is_last) = methods[cursor];
                        if !engines
                            .kepler_compute
                            .dispatch_method_fast(method, arg, is_last)
                        {
                            let _ = self.run_engine_method(
                                class, method, arg, is_last, engines, mappings, stats, mem_read,
                                mem_write, mem_copy,
                            );
                        }
                        cursor += 1;
                    }
                    self.invalidate_resolved_compute_writebacks(mappings);
                } else {
                    for (method, arg, is_last) in methods {
                        let _ = self.run_engine_method(
                            class, method, arg, is_last, engines, mappings, stats, mem_read,
                            mem_write, mem_copy,
                        );
                    }
                }
                if let Some(started) = started {
                    engb_prof_record(class, count, started.elapsed());
                }
                true
            }
            PrepEvent::EngineMethod {
                class,
                method,
                arg,
                is_last,
            } => {
                if class != super::engines::maxwell3d::MAXWELL3D_CLASS {
                    self.publish_pending_storage_readbacks(true, mappings, mem_read, mem_write);
                }
                self.run_engine_method(
                    class, method, arg, is_last, engines, mappings, stats, mem_read, mem_write,
                    mem_copy,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_engine_method(
        &mut self,
        class: u32,
        method: u32,
        arg: u32,
        is_last: bool,
        engines: &mut PrepEngines,
        mappings: &GpuMappings,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> bool {
        use super::pusher::kickprof;
        {
            match class {
                super::engines::MAXWELL_DMA_CLASS => {
                    let maxwell_dma = &mut *engines.maxwell_dma;
                    let requires_hard_boundary =
                        super::engines::maxwell_dma::method_requires_hard_boundary(method);
                    if requires_hard_boundary {
                        self.record_flush_reason(kickprof::FLUSH_HARD_DMA);
                        self.flush_vk(mappings, mem_read, mem_write);
                    }
                    let kp = kickprof::start();
                    let mut deferred = false;
                    if requires_hard_boundary {
                        let spans = maxwell_dma.transfer_spans(arg);
                        let (dst_gpu, dst_size) = spans[1];
                        self.ssbo_snapshot_cache
                            .invalidate_gpu_write(mappings, dst_gpu, dst_size);
                        if super::engines::maxwell_compute::has_pending_writebacks() {
                            let span_overlaps = |&(gpu_va, size): &(u64, usize)| {
                                let cpu_addr = mappings
                                    .cpu_range_for(gpu_va)
                                    .map(|(cpu, _)| cpu)
                                    .unwrap_or(0);
                                super::engines::maxwell_compute::pending_writeback_overlaps(
                                    gpu_va, cpu_addr, size,
                                )
                            };
                            let overlaps = spans.iter().any(span_overlaps);
                            if overlaps {
                                let trace = super::engines::maxwell_compute::compute_sync_trace_enabled()
                                    .then(|| {
                                        (
                                            std::time::Instant::now(),
                                            span_overlaps(&spans[0]),
                                            span_overlaps(&spans[1]),
                                            super::engines::maxwell_compute::pending_writeback_count(),
                                        )
                                    });
                                deferred = self.try_defer_dma(maxwell_dma, arg, mappings);
                                if !deferred && dma_narrow_resolve_enabled() {
                                    let ranges: Vec<(u64, u64, usize)> = spans
                                        .iter()
                                        .map(|&(gpu_va, size)| {
                                            let cpu_addr = mappings
                                                .cpu_range_for(gpu_va)
                                                .map(|(cpu, _)| cpu)
                                                .unwrap_or(0);
                                            (gpu_va, cpu_addr, size)
                                        })
                                        .collect();
                                    self.resolve_pending_compute_overlapping(mappings, mem_write, &ranges);
                                } else if !deferred {
                                    self.resolve_pending_compute(mappings, mem_write);
                                }
                                if let Some((started, source, destination, before)) = trace {
                                    log::warn!(
                                        "[ktl] us={} csync kind=dma arg={:#x} src_overlap={} dst_overlap={} src={:#x}+{:#x} dst={:#x}+{:#x} pending={}->{} deferred={} ms={:.3}",
                                        crate::timeline_us(),
                                        arg,
                                        source,
                                        destination,
                                        spans[0].0,
                                        spans[0].1,
                                        spans[1].0,
                                        spans[1].1,
                                        before,
                                        super::engines::maxwell_compute::pending_writeback_count(),
                                        deferred,
                                        started.elapsed().as_secs_f64() * 1000.0
                                    );
                                }
                            }
                        }
                        if let Some(r) = self.renderer.clone().filter(|_| !deferred) {
                            let stage_started = kickprof::start();
                            let recorded: std::cell::RefCell<Vec<(u64, usize)>> =
                                std::cell::RefCell::new(Vec::new());
                            let recording_write = |addr: u64, bytes: &[u8]| {
                                let written = mem_write(addr, bytes);
                                if written && !bytes.is_empty() {
                                    recorded.borrow_mut().push((addr, bytes.len()));
                                }
                                written
                            };
                            maxwell_dma.stage_rt_source(arg, mappings, &r, &recording_write);
                            for (addr, len) in recorded.into_inner() {
                                self.ssbo_snapshot_cache.mirror_mark_cpu(addr, len);
                            }
                            kickprof::add(kickprof::DMA_STAGE, stage_started);
                        }
                    }
                    let pre = maxwell_dma.blit_count;
                    if deferred {
                        maxwell_dma.commit_deferred_pitch_launch(arg, mappings);
                    } else {
                        maxwell_dma
                            .dispatch_method(method, arg, mappings, mem_read, mem_write, mem_copy);
                    }
                    if requires_hard_boundary {
                        if let Some((cpu_addr, size)) = maxwell_dma.take_guest_write_range() {
                            if let Some(r) = self.renderer.clone() {
                                let gpu_ranges = mappings.gpu_regions_for_cpu_range(cpu_addr, size);
                                if let Some(rt) = crate::render_thread::maybe_render_thread() {
                                    rt.submit_named(
                                        "dma-guest-write-invalidate",
                                        Box::new(move || {
                                            r.invalidate_render_target_range(
                                                cpu_addr,
                                                size,
                                                &gpu_ranges,
                                            )
                                        }),
                                    );
                                } else {
                                    r.invalidate_render_target_range(cpu_addr, size, &gpu_ranges);
                                }
                            }
                        }
                    }
                    let n = maxwell_dma.blit_count - pre;
                    if n > 0 {
                        stats
                            .maxwell_dma_blits
                            .fetch_add(n, AtomicOrdering::Relaxed);
                    }
                    kickprof::add(kickprof::DMA, kp);
                    true
                }
                super::engines::FERMI_2D_CLASS => {
                    let fermi_2d = &mut *engines.fermi_2d;
                    if super::engines::fermi_2d::method_executes_blit(method) {
                        if fermi_ordered_exact_enabled() {
                            if let Some(preview) =
                                fermi_2d.preview_exact_identity_blit(arg, mappings)
                            {
                                let copy_size = usize::try_from(preview.destination_size).ok();
                                let compute_overlap = copy_size.is_none_or(|size| {
                                    super::engines::maxwell_compute::has_pending_writebacks()
                                        && (super::engines::maxwell_compute::pending_writeback_overlaps(
                                            preview.source.gpu_va,
                                            preview.source.cpu_addr,
                                            size,
                                        ) || super::engines::maxwell_compute::pending_writeback_overlaps(
                                            preview.destination.gpu_va,
                                            preview.destination.cpu_addr,
                                            size,
                                        ))
                                });
                                if !compute_overlap {
                                    self.record_flush_reason(kickprof::FLUSH_HARD_FERMI);
                                    self.flush_vk_soft(mappings, mem_read, mem_write);
                                    self.ssbo_snapshot_cache.clear_ssbo_snapshots();
                                    if let (Some(source), Some(renderer)) = (
                                        self.prepared_draw_packets
                                            .latest_pending_exact_color_source(
                                                preview.source,
                                                preview.expected_bpp,
                                            ),
                                        self.renderer.clone(),
                                    ) {
                                        let destination = preview.destination;
                                        let expected_bpp = preview.expected_bpp;
                                        let job_renderer = Arc::clone(&renderer);
                                        let job = Box::new(move || {
                                            match job_renderer.execute_ordered_rt_copy_exact(
                                                source,
                                                expected_bpp,
                                                destination,
                                            ) {
                                                Ok(Some(_)) => {}
                                                Ok(None) => log::warn!(
                                                    "Fermi2D: ordered exact RT copy source vanished"
                                                ),
                                                Err(error) => log::warn!(
                                                    "Fermi2D: ordered exact RT copy failed: {}",
                                                    error
                                                ),
                                            }
                                        });
                                        if !self
                                            .prepared_draw_packets
                                            .drain_hard_then_job("fermi-ordered-exact-copy", job)
                                        {
                                            log::warn!(
                                                "Fermi2D: ordered exact RT copy enqueue failed"
                                            );
                                            return false;
                                        }
                                        fermi_2d.trace_blit("ordered-exact", arg);
                                        let kp = kickprof::start();
                                        let pre = fermi_2d.blit_count;
                                        if !fermi_2d.commit_ordered_exact_blit(arg, &preview) {
                                            log::error!(
                                                "Fermi2D: ordered exact RT copy commit rejected"
                                            );
                                            return false;
                                        }
                                        if let Some((gpu_addr, size)) =
                                            fermi_2d.take_logical_guest_write_span()
                                        {
                                            self.ssbo_snapshot_cache
                                                .invalidate_gpu_write(mappings, gpu_addr, size);
                                        }
                                        self.invalidate_resolved_compute_writebacks(mappings);
                                        let n = fermi_2d.blit_count - pre;
                                        if n > 0 {
                                            stats
                                                .fermi_2d_blits
                                                .fetch_add(n, AtomicOrdering::Relaxed);
                                        }
                                        kickprof::add(kickprof::FERMI, kp);
                                        kickprof::count(kickprof::FERMI_DRAIN_SKIPPED, 1);
                                        return true;
                                    }
                                }
                            }
                        }
                        if fermi_ordered_exact_enabled() && fermi_ordered_region_enabled() {
                            if let (Some(preview), Some(renderer), Some(render_thread)) = (
                                fermi_2d.preview_ordered_region_blit(
                                    arg,
                                    mappings,
                                    self.renderer.as_deref(),
                                ),
                                self.renderer.clone(),
                                crate::render_thread::maybe_render_thread(),
                            ) {
                                let compute_overlap = match (
                                    usize::try_from(preview.source_size),
                                    usize::try_from(preview.destination_size),
                                ) {
                                    (Ok(source_size), Ok(destination_size)) => {
                                        super::engines::maxwell_compute::has_pending_writebacks()
                                            && (super::engines::maxwell_compute::pending_writeback_overlaps(
                                                preview.source.gpu_va,
                                                preview.source_cpu,
                                                source_size,
                                            ) || super::engines::maxwell_compute::pending_writeback_overlaps(
                                                preview.destination.gpu_va,
                                                preview.destination_cpu,
                                                destination_size,
                                            ))
                                    }
                                    _ => true,
                                };
                                if !compute_overlap {
                                    self.record_flush_reason(kickprof::FLUSH_HARD_FERMI);
                                    self.flush_vk(mappings, mem_read, mem_write);
                                    self.ssbo_snapshot_cache.clear_ssbo_snapshots();
                                    let source = preview.source;
                                    let destination = preview.destination;
                                    let region = preview.region;
                                    render_thread.submit_named(
                                        "fermi-ordered-region-copy",
                                        Box::new(move || {
                                            match renderer.execute_ordered_rt_copy_region(
                                                source,
                                                destination,
                                                region,
                                            ) {
                                                Ok(true) => {}
                                                Ok(false) => log::warn!(
                                                    "Fermi2D: ordered RT region copy {} -> {} skipped",
                                                    source.label(),
                                                    destination.label()
                                                ),
                                                Err(error) => log::warn!(
                                                    "Fermi2D: ordered RT region copy failed: {}",
                                                    error
                                                ),
                                            }
                                        }),
                                    );
                                    fermi_2d.trace_blit("ordered-region", arg);
                                    let kp = kickprof::start();
                                    let pre = fermi_2d.blit_count;
                                    fermi_2d.commit_ordered_region_blit(arg, &preview);
                                    if let Some((gpu_addr, size)) =
                                        fermi_2d.take_logical_guest_write_span()
                                    {
                                        self.ssbo_snapshot_cache
                                            .invalidate_gpu_write(mappings, gpu_addr, size);
                                    }
                                    self.invalidate_resolved_compute_writebacks(mappings);
                                    let n = fermi_2d.blit_count - pre;
                                    if n > 0 {
                                        stats
                                            .fermi_2d_blits
                                            .fetch_add(n, AtomicOrdering::Relaxed);
                                    }
                                    kickprof::add(kickprof::FERMI, kp);
                                    kickprof::count(kickprof::FERMI_DRAIN_SKIPPED, 1);
                                    return true;
                                }
                            }
                        }
                        self.record_flush_reason(kickprof::FLUSH_HARD_FERMI);
                        self.flush_vk(mappings, mem_read, mem_write);
                        self.ssbo_snapshot_cache.clear_ssbo_snapshots();
                        let async_candidate =
                            fermi_2d.blit_exact_async_candidate(self.renderer.as_deref(), mappings);
                        let needs_drain = !async_candidate
                            && (!fermi_lazy_drain_enabled()
                                || fermi_2d.blit_touches_live_rt(self.renderer.as_deref()));
                        if needs_drain {
                            let kp_drain = kickprof::start();
                            let drained = super::vk_dispatch::sync_render_thread();
                            kickprof::add(kickprof::FERMI_DRAIN, kp_drain);
                            if !drained {
                                log::warn!("Fermi2D: render-thread drain failed; skipping blit");
                                return false;
                            }
                        } else {
                            kickprof::count(kickprof::FERMI_DRAIN_SKIPPED, 1);
                        }
                    }
                    let kp = kickprof::start();
                    let r = self.renderer.clone();
                    let pre = fermi_2d.blit_count;
                    fermi_2d.dispatch_method(
                        method,
                        arg,
                        mappings,
                        r.as_ref(),
                        mem_read,
                        mem_write,
                    );
                    if let Some((gpu_addr, size)) = fermi_2d.take_logical_guest_write_span() {
                        self.ssbo_snapshot_cache
                            .invalidate_gpu_write(mappings, gpu_addr, size);
                    }
                    if let Some((cpu_addr, size)) = fermi_2d.take_guest_write_range() {
                        if let Some(r) = self.renderer.clone() {
                            let gpu_ranges = mappings.gpu_regions_for_cpu_range(cpu_addr, size);
                            if let Some(rt) = crate::render_thread::maybe_render_thread() {
                                rt.submit_named(
                                    "fermi-guest-write-invalidate",
                                    Box::new(move || {
                                        r.invalidate_render_target_range(
                                            cpu_addr,
                                            size,
                                            &gpu_ranges,
                                        )
                                    }),
                                );
                            } else {
                                r.invalidate_render_target_range(cpu_addr, size, &gpu_ranges);
                            }
                        }
                    }
                    self.invalidate_resolved_compute_writebacks(mappings);
                    let n = fermi_2d.blit_count - pre;
                    if n > 0 {
                        stats.fermi_2d_blits.fetch_add(n, AtomicOrdering::Relaxed);
                    }
                    kickprof::add(kickprof::FERMI, kp);
                    true
                }
                super::engines::KEPLER_MEMORY_CLASS => {
                    let kepler_memory = &mut *engines.kepler_memory;
                    if kepler_memory.method_requires_hard_boundary(method) {
                        self.record_flush_reason(kickprof::FLUSH_HARD_KEPLER_MEMORY);
                        self.flush_vk(mappings, mem_read, mem_write);
                        self.ssbo_snapshot_cache.retain_watchable_full_aurora_snapshots();
                    }
                    let kp = kickprof::start();
                    let outcome =
                        kepler_memory.dispatch_method(method, arg, mappings, mem_read, mem_write);
                    self.apply_kepler_memory_write(mappings, mem_read, mem_write, outcome);
                    kickprof::add(kickprof::KEPLER, kp);
                    true
                }
                super::engines::KEPLER_COMPUTE_CLASS => {
                    let kepler_compute = &mut *engines.kepler_compute;
                    if kepler_compute.method_requires_hard_boundary(method, is_last) {
                        self.record_flush_reason(kickprof::FLUSH_HARD_KEPLER_COMPUTE);
                        self.flush_vk(mappings, mem_read, mem_write);
                        if kepler_compute.method_writes_guest_memory(method, is_last) {
                            kickprof::count(kickprof::KC_CACHE_CLEAR, 1);
                            self.ssbo_snapshot_cache.retain_watchable_full_aurora_snapshots();
                        }
                    }
                    let kp = kickprof::start();
                    let outcome = {
                        let cache_cell = std::cell::RefCell::new(&mut self.ssbo_snapshot_cache);
                        let content_key = |gpu_va: u64, len: usize| -> Option<u64> {
                            cache_cell
                                .borrow_mut()
                                .mirror_content_key(mappings, gpu_va, len, mem_read)
                        };
                        kepler_compute.dispatch_method(
                            method,
                            arg,
                            is_last,
                            self.renderer.as_ref(),
                            mappings,
                            mem_read,
                            mem_write,
                            &content_key,
                        )
                    };
                    self.apply_kepler_memory_write(mappings, mem_read, mem_write, outcome);
                    self.invalidate_resolved_compute_writebacks(mappings);
                    kickprof::add(kickprof::KEPLER, kp);
                    true
                }
                _ => true,
            }
        }
    }

    pub fn set_renderer(&mut self, r: Option<Arc<nexium_gpu::Renderer>>) {
        let renderer_changed = match (&self.renderer, &r) {
            (Some(current), Some(next)) => !Arc::ptr_eq(current, next),
            (None, None) => false,
            _ => true,
        };
        if renderer_changed {
            self.ssbo_snapshot_cache.clear_prepared_texture_snapshots();
            self.pending_small_rt_wb = None;
            let cleared = super::vk_dispatch::clear_pending_small_rt_writebacks();
            if cleared != 0 {
                log::debug!(
                    "[rt-writeback] discarded {} stale target(s) after renderer change",
                    cleared
                );
            }
        }
        self.renderer = r;
    }

    pub(crate) fn set_guest_memory_access(&mut self, memory: Option<super::GuestMemoryAccess>) {
        self.guest_memory = memory;
    }

    pub(crate) fn begin_ssbo_snapshot_epoch(&mut self) {
        self.begin_ssbo_snapshot_epoch_with(None);
    }

    pub(crate) fn begin_ssbo_snapshot_epoch_with(
        &mut self,
        sweep: Option<(Vec<(u64, usize)>, Vec<(u64, usize)>)>,
    ) {
        self.ssbo_refreshed_at = None;
        self.ssbo_snapshot_cache.mirror_bump_sweep();
        self.ssbo_snapshot_cache.mirror_begin_kick();
        self.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
        self.ssbo_snapshot_cache.reset_epoch();
        if let Some((swept, dirty)) = sweep {
            self.ssbo_snapshot_cache.adopt_decode_sweep(&swept, &dirty);
        }
        self.ssbo_snapshot_cache.refresh_input_guest_writes();
    }

    pub(crate) fn end_ssbo_snapshot_epoch(&mut self) {
        let kp = super::pusher::kickprof::start();
        self.ssbo_snapshot_cache.profile_epoch();
        self.ssbo_snapshot_cache.clear_ssbo_snapshots();
        self.ssbo_snapshot_cache.clear_prepared_texture_snapshots();
        if nexium_memory::fastmem::decode_write_watch_sweep_enabled() {
            publish_decode_sweep_plan(nexium_memory::fastmem::write_watch_memo_used_ranges());
        }
        super::pusher::kickprof::add(super::pusher::kickprof::EPOCH_END, kp);
    }

    fn join_small_rt_writeback(&mut self, mappings: &GpuMappings) -> bool {
        let Some(rx) = self.pending_small_rt_wb.take() else {
            return true;
        };
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok((completed, chunks)) => {
                if !chunks.is_empty() {
                    super::vk_dispatch::apply_small_rt_writeback_chunks(
                        &mut self.ssbo_snapshot_cache,
                        mappings,
                        &chunks,
                    );
                }
                completed
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                log::warn!("[rt-writeback] async join timed out");
                self.pending_small_rt_wb = Some(rx);
                false
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                log::warn!("[rt-writeback] async join disconnected");
                false
            }
        }
    }

    pub(crate) fn writeback_small_rts(
        &mut self,
        renderer: &Arc<nexium_gpu::Renderer>,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> bool {
        if !super::vk_dispatch::has_pending_small_rt_writebacks() {
            return false;
        }
        let kp_flush = super::pusher::kickprof::start();
        let flushed = self.flush_prepared_draw_packets();
        super::pusher::kickprof::add(super::pusher::kickprof::PRES_WB_FLUSH, kp_flush);
        if !flushed {
            log::warn!("[rt-writeback] skipped after prepared draw drain failure");
            return false;
        }
        super::vk_dispatch::writeback_small_rts(
            renderer,
            mappings,
            mem_write,
            &mut self.ssbo_snapshot_cache,
        )
    }

    pub(crate) fn writeback_cpu_readable_rts(
        &mut self,
        renderer: &Arc<nexium_gpu::Renderer>,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> bool {
        if !self.flush_prepared_draw_packets() {
            log::warn!("[rt-writeback] cpu-readable writeback skipped after prepared draw drain failure");
            return false;
        }
        self.join_cpu_rt_writebacks(mappings, 4);
        let async_memory = super::vk_dispatch::cpu_rt_writeback_async_enabled()
            .then(|| self.guest_memory.clone())
            .flatten()
            .filter(super::GuestMemoryAccess::is_available);
        if let Some(memory) = async_memory {
            if let Some(pending) =
                super::vk_dispatch::spawn_cpu_readable_rt_writeback_async(renderer, memory)
            {
                self.pending_cpu_rt_wb.push(pending);
                return true;
            }
        }
        super::vk_dispatch::writeback_cpu_readable_rts(
            renderer,
            mappings,
            mem_write,
            &mut self.ssbo_snapshot_cache,
        )
    }

    fn join_cpu_rt_writebacks(&mut self, mappings: &GpuMappings, keep_pending: usize) {
        let mut index = 0;
        while index < self.pending_cpu_rt_wb.len() {
            let must_block = self.pending_cpu_rt_wb.len() - index > keep_pending;
            let received = if must_block {
                self.pending_cpu_rt_wb[index]
                    .recv_timeout(std::time::Duration::from_millis(500))
                    .map_err(|_| true)
            } else {
                self.pending_cpu_rt_wb[index].try_recv().map_err(|error| {
                    matches!(error, std::sync::mpsc::TryRecvError::Disconnected)
                })
            };
            match received {
                Ok((_, chunks)) => {
                    if !chunks.is_empty() {
                        super::vk_dispatch::apply_small_rt_writeback_chunks(
                            &mut self.ssbo_snapshot_cache,
                            mappings,
                            &chunks,
                        );
                    }
                    self.pending_cpu_rt_wb.remove(index);
                }
                Err(true) => {
                    if must_block {
                        log::warn!("[rt-writeback] async cpu-readable writeback join timed out");
                    }
                    self.pending_cpu_rt_wb.remove(index);
                }
                Err(false) => index += 1,
            }
        }
    }

    pub(crate) fn process_inline_upload_methods(
        &mut self,
        methods: Vec<(u32, u32)>,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        for (method, arg) in methods {
            let outcome = self
                .inline_upload
                .dispatch_method(method, arg, mappings, mem_read, mem_write);
            if !matches!(outcome, KeplerMemoryWriteOutcome::NoWrite) {
                self.apply_kepler_memory_write(mappings, mem_read, mem_write, outcome);
            }
        }
    }

    pub(crate) fn begin_ssbo_snapshot_entry(&mut self) {
        let watch_started = super::pusher::kickprof::start();
        if mirror_sweep_per_entry_enabled() {
            self.ssbo_snapshot_cache.mirror_bump_sweep();
        }
        if self.ssbo_snapshot_cache.has_watched_ssbo_state() && self.ssbo_refresh_due() {
            self.ssbo_snapshot_cache.refresh_ssbo_guest_writes();
        }
        super::pusher::kickprof::add(super::pusher::kickprof::ENQ_WATCH, watch_started);
        let retain_started = super::pusher::kickprof::start();
        self.ssbo_snapshot_cache
            .retain_watchable_full_aurora_snapshots();
        super::pusher::kickprof::add(super::pusher::kickprof::ENTRY_PREP, retain_started);
    }

    fn ssbo_refresh_due(&mut self) -> bool {
        if !super::accuracy::normal_accuracy() {
            return true;
        }
        let now = std::time::Instant::now();
        let epoch = SEMAPHORE_ACQUIRE_EPOCH.load(std::sync::atomic::Ordering::Acquire);
        if self.ssbo_refreshed_epoch == epoch
            && self
                .ssbo_refreshed_at
                .is_some_and(|at| now.duration_since(at) < ssbo_refresh_interval())
        {
            return false;
        }
        self.ssbo_refreshed_at = Some(now);
        self.ssbo_refreshed_epoch = epoch;
        true
    }

    fn sync_semaphore_acquire_sweep(&mut self) {
        let epoch = SEMAPHORE_ACQUIRE_EPOCH.load(std::sync::atomic::Ordering::Acquire);
        if self.mirror_acquire_epoch != epoch {
            self.mirror_acquire_epoch = epoch;
            nexium_memory::fastmem::bump_write_watch_epoch();
            self.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
        }
    }

    pub(crate) fn record_flush_reason(&self, phase: usize) {
        if !self.vk_batch.is_empty() {
            super::pusher::kickprof::count(phase, 1);
        }
    }

    pub(crate) fn has_prepared_draw_packets(&self) -> bool {
        self.prepared_draw_packets.has_pending()
    }

    pub(crate) fn has_pending_storage_readbacks(&self) -> bool {
        !self.pending_storage_readbacks.is_empty()
    }

    pub(crate) fn pending_storage_readbacks_overlap(&self, gpu_va: u64, len: u64) -> bool {
        super::vk_dispatch::pending_storage_readbacks_overlap(&self.pending_storage_readbacks, gpu_va, len)
    }

    pub(crate) fn publish_pending_storage_readbacks(
        &mut self,
        blocking: bool,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> bool {
        if self.pending_storage_readbacks.is_empty() {
            return true;
        }
        let Some(renderer) = self.renderer.clone() else {
            self.pending_storage_readbacks.clear();
            return false;
        };
        let flush_started = std::time::Instant::now();
        if blocking {
            self.flush_vk_with_boundary(mappings, mem_read, mem_write, false);
            self.flush_prepared_draw_packets();
        }
        let flush_ms = flush_started.elapsed().as_secs_f64() * 1000.0;
        if blocking && flush_ms > 2.0 {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 || n % 512 == 0 {
                log::info!("[graphics-ssbo-writeback] blocking publish flush #{n} took {flush_ms:.3} ms");
            }
        }
        super::vk_dispatch::publish_pending_storage_readbacks(
            &mut self.pending_storage_readbacks,
            blocking,
            &renderer,
            mappings,
            &mut self.ssbo_snapshot_cache,
            mem_read,
            mem_write,
        )
    }

    pub(crate) fn flush_prepared_draw_packets(&mut self) -> bool {
        #[cfg(test)]
        {
            self.prepared_packet_drain_counts.hard += 1;
        }
        let drained = self.prepared_draw_packets.drain_hard();
        let submitted = self.prepared_draw_packets.take_submit_completed();
        std::mem::replace(&mut self.vk_flush_completed, true) && drained && submitted
    }

    pub(crate) fn flush_prepared_draw_packets_soft(&mut self) -> bool {
        #[cfg(test)]
        {
            self.prepared_packet_drain_counts.soft += 1;
        }
        let drained = self.prepared_draw_packets.drain_soft();
        let submitted = self.prepared_draw_packets.take_submit_completed();
        std::mem::replace(&mut self.vk_flush_completed, true) && drained && submitted
    }

    pub(crate) fn finish_prepared_draw_packet_tail(&mut self, hard_after: bool) -> bool {
        if hard_after {
            let submitted = self.prepared_draw_packets.take_submit_completed();
            std::mem::replace(&mut self.vk_flush_completed, true) && submitted
        } else {
            self.flush_prepared_draw_packets_soft()
        }
    }

    #[cfg(test)]
    pub(crate) fn prepared_packet_drain_counts(&self) -> (usize, usize) {
        (
            self.prepared_packet_drain_counts.hard,
            self.prepared_packet_drain_counts.soft,
        )
    }

    fn apply_kepler_memory_write(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        outcome: KeplerMemoryWriteOutcome,
    ) {
        if let Some(spans) = super::pusher::kepler_rt_invalidation_spans(mappings, &outcome) {
            self.route_rt_invalidation(spans, mappings, mem_read, mem_write);
        }
        match outcome {
            KeplerMemoryWriteOutcome::NoWrite => {}
            KeplerMemoryWriteOutcome::Exact(spans) => {
                self.ssbo_snapshot_cache.invalidate_gpu_writes(mappings, &spans);
            }
            KeplerMemoryWriteOutcome::Unknown(_) => self.ssbo_snapshot_cache.clear(),
        }
    }

    fn route_rt_invalidation(
        &mut self,
        mut spans: Vec<(u64, u64, u64)>,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let Some(renderer) = self.renderer.clone() else {
            return;
        };
        let Some(render_thread) = crate::render_thread::maybe_render_thread() else {
            renderer.invalidate_render_target_ranges(&spans);
            return;
        };
        if !rt_invalidation_prelude_enabled() {
            render_thread.submit_named(
                "kepler-guest-write-invalidate",
                Box::new(move || renderer.invalidate_render_target_ranges(&spans)),
            );
            return;
        }
        if super::vk_dispatch::draws_touch_rt_spans(&self.vk_batch, &spans) {
            self.flush_vk_soft(mappings, mem_read, mem_write);
        }
        if self.prepared_draw_packets.attach_rt_invalidations(&mut spans) {
            return;
        }
        if !render_thread.submit_rt_invalidation(renderer, spans) {
            log::warn!("render target invalidation dropped: render thread unavailable");
        }
    }

    pub(crate) fn flush_vk(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.flush_vk_with_boundary(mappings, mem_read, mem_write, true);
    }

    pub(crate) fn flush_vk_soft(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.flush_vk_with_boundary(mappings, mem_read, mem_write, false);
    }

    pub(crate) fn flush_vk_with_boundary(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        hard_after: bool,
    ) {
        if self.vk_batch.is_empty() {
            if hard_after {
                let completed = self.prepared_draw_packets.drain_hard();
                self.vk_flush_completed &= completed;
            }
            return;
        }
        let memory_barrier = if deferred_compute_draw_resolve_enabled()
            && crate::render_thread::maybe_render_thread().is_some()
        {
            self.compute_barrier_cache.get_current(mappings)
        } else {
            ComputeMemoryBarrier::default()
        };
        if memory_barrier.armed.get() {
            self.invalidate_compute_snapshots(&memory_barrier, mappings);
        }
        self.ssbo_snapshot_cache.set_compute_pending(
            memory_barrier.cpu_ranges.clone(),
            memory_barrier.spans.clone(),
            memory_barrier.revision,
        );
        let barrier_renderer = self.renderer.clone();
        let resolve_memory = || {
            if let Some(renderer) = &barrier_renderer {
                super::engines::maxwell_compute::resolve_pending_writebacks(
                    renderer, mappings, mem_write,
                );
            }
        };
        let guarded_read = |address, bytes: &mut [u8]| {
            memory_barrier.before_access(address, bytes.len(), &resolve_memory);
            mem_read(address, bytes)
        };
        let guarded_write = |address, bytes: &[u8]| {
            memory_barrier.before_access(address, bytes.len(), &resolve_memory);
            mem_write(address, bytes)
        };
        let mem_read: &dyn Fn(u64, &mut [u8]) -> bool = &guarded_read;
        let mem_write: &dyn Fn(u64, &[u8]) -> bool = &guarded_write;
        let dp = draw_prof_start();
        let batch_len = self.vk_batch.len();
        let kp = super::pusher::kickprof::start();
        let completed = if let Some(r) = self.renderer.clone() {
            if hard_after {
                super::vk_dispatch::flush_accum(
                    &mut self.vk_batch,
                    &r,
                    mappings,
                    mem_read,
                    mem_write,
                    &mut self.ssbo_snapshot_cache,
                    &mut self.prepared_draw_packets,
                )
            } else {
                super::vk_dispatch::flush_accum_soft(
                    &mut self.vk_batch,
                    &r,
                    mappings,
                    mem_read,
                    mem_write,
                    &mut self.ssbo_snapshot_cache,
                    &mut self.prepared_draw_packets,
                )
            }
        } else {
            if hard_after {
                let _ = self.prepared_draw_packets.drain_hard();
            }
            self.vk_batch.clear();
            false
        };
        self.vk_flush_completed &= completed;
        self.invalidate_resolved_compute_writebacks(mappings);
        super::pusher::kickprof::add(super::pusher::kickprof::FLUSHP, kp);
        draw_prof_record(1, batch_len as u64, dp);
    }

    #[track_caller]
    pub(crate) fn resolve_pending_compute(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let _ = self.resolve_pending_compute_report(mappings, mem_write);
    }

    pub(crate) fn drain_landed_compute(&mut self, mappings: &GpuMappings) {
        for span in super::engines::maxwell_compute::drain_landed_writebacks() {
            self.ssbo_snapshot_cache
                .invalidate_gpu_write(mappings, span.gpu_va, span.len);
        }
    }

    #[track_caller]
    pub(crate) fn compute_landing_possible(&self) -> bool {
        super::engines::maxwell_compute::compute_landing_enabled()
            && crate::render_thread::maybe_render_thread().is_some()
            && self
                .renderer
                .as_ref()
                .is_some_and(|renderer| renderer.timeline_sync_available())
            && self
                .guest_memory
                .as_ref()
                .is_some_and(super::GuestMemoryAccess::is_available)
    }

    #[track_caller]
    pub(crate) fn land_or_resolve_pending_compute(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if !super::engines::maxwell_compute::has_pending_writebacks() {
            return;
        }
        if self.compute_landing_possible() {
            let renderer = self.renderer.clone().unwrap();
            let memory = self.guest_memory.clone().unwrap();
            super::engines::maxwell_compute::schedule_pending_landings(renderer, memory);
        } else {
            self.resolve_pending_compute(mappings, mem_write);
        }
    }

    #[track_caller]
    pub(crate) fn resolve_pending_compute_report(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> Vec<super::engines::maxwell_compute::PendingComputeWritebackSpan> {
        if !super::engines::maxwell_compute::has_pending_writebacks() {
            return Vec::new();
        }
        super::watchdog::phase(super::watchdog::Phase::ComputeResolve, 0);
        let Some(renderer) = self.renderer.as_deref() else {
            return Vec::new();
        };
        let spans = super::engines::maxwell_compute::resolve_pending_writebacks_report(
            renderer, mappings, mem_write,
        );
        for span in &spans {
            self.ssbo_snapshot_cache
                .invalidate_gpu_write(mappings, span.gpu_va, span.len);
        }
        spans
    }

    #[track_caller]
    pub(crate) fn resolve_pending_compute_overlapping(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        ranges: &[(u64, u64, usize)],
    ) {
        if !super::engines::maxwell_compute::has_pending_writebacks() {
            return;
        }
        super::watchdog::phase(super::watchdog::Phase::ComputeResolve, 0);
        let Some(renderer) = self.renderer.as_deref() else {
            return;
        };
        let spans = super::engines::maxwell_compute::resolve_pending_writebacks_overlapping_report(
            renderer, mappings, mem_write, ranges,
        );
        for span in &spans {
            self.ssbo_snapshot_cache
                .invalidate_gpu_write(mappings, span.gpu_va, span.len);
        }
    }

    fn try_defer_dma(
        &self,
        maxwell_dma: &super::engines::MaxwellDma,
        flags: u32,
        mappings: &GpuMappings,
    ) -> bool {
        if !super::engines::maxwell_compute::dma_defer_enabled() || self.renderer.is_none() {
            return false;
        }
        let Some(memory) = self
            .guest_memory
            .as_ref()
            .filter(|memory| memory.is_available() && memory.reader().is_some())
        else {
            return false;
        };
        let Some(mut copies) = maxwell_dma.deferrable_pitch_copies(flags, mappings) else {
            return false;
        };
        if !copies.iter().all(|copy| {
            super::engines::maxwell_compute::pending_writeback_covers(copy.src_cpu, copy.len)
        }) {
            return false;
        }
        if super::vk_dispatch::cbuf_compute_overlay_enabled() {
            let spans = super::engines::maxwell_compute::pending_writeback_spans_snapshot();
            for copy in &mut copies {
                copy.source = super::vk_dispatch::compute_overlay_source_for_range(
                    &spans,
                    copy.src_cpu,
                    copy.len,
                );
            }
        }
        super::engines::maxwell_compute::push_deferred_dma(copies, memory.clone());
        true
    }

    pub(crate) fn invalidate_resolved_compute_writebacks(&mut self, mappings: &GpuMappings) {
        for span in super::engines::maxwell_compute::take_resolved_writeback_spans() {
            self.ssbo_snapshot_cache
                .invalidate_gpu_write(mappings, span.gpu_va, span.len);
        }
    }

    pub(crate) fn sync_renderer_idle(&mut self, reason: &str) -> bool {
        super::watchdog::phase(super::watchdog::Phase::RenderWait, reason.len() as u64);
        let profile = gpu_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let Some(renderer) = self.renderer.clone() else {
            return true;
        };
        if !self.flush_prepared_draw_packets() {
            log::warn!("[gpu-sync] {} prepared draw drain failed", reason);
            return false;
        }
        if semrel_legacy() || !renderer.timeline_sync_available() {
            return self.sync_renderer_idle_legacy(reason);
        }
        if !super::vk_dispatch::sync_render_thread() {
            log::warn!("[gpu-sync] {} render drain timeout", reason);
            return false;
        }
        let target = renderer.submitted_generation();
        let blocked_started = super::pusher::kickprof::rate_start();
        let mut completed = renderer.wait_submit_generation_patiently(target, reason);
        super::pusher::kickprof::add_blocked(blocked_started);
        if !completed {
            log::error!(
                "[gpu-sync] {} timeline wait failed target={}; device-idle fallback suppressed",
                reason,
                target
            );
        }
        if completed && semrel_verify() {
            completed = renderer.wait_idle_checked();
        }
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(1) {
                log::warn!(
                    "[gpu-sync] {} elapsed_ms={:.3}",
                    reason,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
        completed
    }

    pub(crate) fn sync_renderer_idle_legacy(&self, reason: &str) -> bool {
        let profile = gpu_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let Some(renderer) = self.renderer.clone() else {
            return true;
        };
        let completed = if let Some(rt) = crate::render_thread::maybe_render_thread() {
            let (tx, rx) = mpsc::sync_channel(1);
            let r = renderer.clone();
            let job = Box::new(move || {
                let _ = tx.send(r.wait_idle_if_dirty_checked());
            }) as crate::render_thread::RenderJob;
            if !rt.submit_timeout(job, Duration::from_secs(3)) {
                log::warn!("[gpu-sync] {} render thread submit timeout", reason);
                return false;
            }
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(completed) => completed,
                Err(_) => {
                    log::warn!("[gpu-sync] {} render thread idle timeout", reason);
                    false
                }
            }
        } else {
            renderer.wait_idle_if_dirty_checked()
        };
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(1) {
                log::warn!(
                    "[gpu-sync] {} elapsed_ms={:.3}",
                    reason,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
        completed
    }

    pub(crate) fn commit_constbuf_writes(
        &mut self,
        writes: &[(u64, u32)],
        trace: Option<(u64, u32)>,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let mut invalidation_ranges = std::mem::take(&mut self.constbuf_invalidation_scratch);
        invalidation_ranges.clear();
        let mut patched_ranges = std::mem::take(&mut self.constbuf_patched_scratch);
        patched_ranges.clear();
        let mut run_bytes = std::mem::take(&mut self.constbuf_bytes_scratch);
        run_bytes.clear();
        let mut run_start = 0usize;
        while run_start < writes.len() {
            let run_end = contiguous_constbuf_write_run_end(writes, run_start);
            let run = &writes[run_start..run_end];
            let gpu_va = run[0].0;
            let run_len = run.len().saturating_mul(std::mem::size_of::<u32>());
            let contiguous_mapping = mappings
                .cpu_range_for(gpu_va)
                .filter(|(_, remaining)| *remaining >= run_len as u64);

            if let Some((cpu, _)) = contiguous_mapping {
                run_bytes.clear();
                run_bytes.reserve(run_len);
                for (index, &(write_gpu_va, dword)) in run.iter().enumerate() {
                    trace_constbuf_upload(
                        trace,
                        write_gpu_va,
                        cpu + (index * std::mem::size_of::<u32>()) as u64,
                        dword,
                    );
                    run_bytes.extend_from_slice(&dword.to_le_bytes());
                }
                let written = mem_write(cpu, &run_bytes);
                if written
                    && self.renderer.is_some()
                    && self
                        .ssbo_snapshot_cache
                        .mirror_write_through(cpu, &run_bytes)
                {
                    patched_ranges.push((gpu_va, cpu, run_len));
                } else {
                    invalidation_ranges.push((gpu_va, run_len));
                }
            } else {
                invalidation_ranges.extend(
                    run.iter()
                        .map(|&(gpu_va, _)| (gpu_va, std::mem::size_of::<u32>())),
                );
                for &(write_gpu_va, dword) in run {
                    if let Some(cpu) = mappings.cpu_address_for(write_gpu_va) {
                        trace_constbuf_upload(trace, write_gpu_va, cpu, dword);
                        mem_write(cpu, &dword.to_le_bytes());
                    } else {
                        static DROPPED: AtomicU64 = AtomicU64::new(0);
                        let n = DROPPED.fetch_add(1, Ordering::Relaxed);
                        if n < 32 || n % 4096 == 0 {
                            log::warn!(
                                "[cbuf-upload-drop] #{} gpu_va={:#x} dword={:#010x}",
                                n,
                                write_gpu_va,
                                dword
                            );
                        }
                    }
                }
            }
            run_start = run_end;
        }
        if self.renderer.is_some() {
            self.ssbo_snapshot_cache
                .defer_gpu_writes(&invalidation_ranges);
            self.ssbo_snapshot_cache
                .defer_patched_gpu_writes(&patched_ranges);
        } else {
            self.ssbo_snapshot_cache
                .invalidate_gpu_writes(mappings, &invalidation_ranges);
        }
        self.constbuf_invalidation_scratch = invalidation_ranges;
        self.constbuf_patched_scratch = patched_ranges;
        self.constbuf_bytes_scratch = run_bytes;
    }

    pub(crate) fn new() -> Self {
        Self {
            renderer: None,
            guest_memory: None,
            pending_small_rt_wb: None,
            pending_cpu_rt_wb: Vec::new(),
            vk_batch: PendingDrawBatch::default(),
            prepared_draw_packets: PreparedDrawPacketizer::default(),
            vk_flush_completed: true,
            ssbo_snapshot_cache: SsboSnapshotCache::default(),
            pending_storage_readbacks: Vec::new(),
            inline_upload: KeplerMemory::new(),
            constbuf_invalidation_scratch: Vec::new(),
            constbuf_patched_scratch: Vec::new(),
            constbuf_bytes_scratch: Vec::new(),
            compute_snapshot_revision: None,
            compute_barrier_cache: ComputeBarrierCache::default(),
            renderer_guest_writes_pending: false,
            ssbo_refreshed_at: None,
            ssbo_refreshed_epoch: 0,
            mirror_acquire_epoch: SEMAPHORE_ACQUIRE_EPOCH.load(std::sync::atomic::Ordering::Acquire),
            #[cfg(test)]
            prepared_packet_drain_counts: PreparedPacketDrainCounts::default(),
        }
    }
}

static SEMAPHORE_ACQUIRE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn note_semaphore_acquire() {
    SEMAPHORE_ACQUIRE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

fn fermi_lazy_drain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_FERMI_LAZY_DRAIN").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        ) && super::experimental_gpu_scheduling_enabled()
    })
}

fn fermi_ordered_exact_value_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0"
            || value.eq_ignore_ascii_case("false")
            || value.eq_ignore_ascii_case("off")
            || value.eq_ignore_ascii_case("no")
    })
}

fn fermi_ordered_region_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        fermi_ordered_exact_value_enabled(
            std::env::var("NEXIUM_FERMI_ORDERED_REGION").ok().as_deref(),
        )
    })
}

fn fermi_ordered_exact_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        fermi_ordered_exact_value_enabled(
            std::env::var("NEXIUM_FERMI_ORDERED_EXACT").ok().as_deref(),
        )
    }) && crate::render_thread::maybe_render_thread().is_some()
}

fn trace_constbuf_upload(trace: Option<(u64, u32)>, gpu_va: u64, cpu: u64, dword: u32) {
    let Some((watch_va, watch_len)) = super::pusher::constbuf_upload_watch() else {
        return;
    };
    if gpu_va >= watch_va.saturating_add(watch_len) || gpu_va.saturating_add(4) <= watch_va {
        return;
    }
    let (cb_addr, cb_size) = trace.unwrap_or_default();
    log::warn!(
        "[cbuf-upload] target={:#x} cpu={:#x} cb={:#x} size={:#x} dword={:#010x} float={:.6}",
        gpu_va,
        cpu,
        cb_addr,
        cb_size,
        dword,
        f32::from_bits(dword),
    );
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct PreparedPacketDrainCounts {
    pub(crate) hard: usize,
    pub(crate) soft: usize,
}

pub(crate) enum PrepLane {
    Inline(PrepState),
    Threaded(PrepThreadHandle),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrepThreadBehavior {
    Pipeline,
    DrainEachKick,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrepBarrierDispatch {
    Queued,
    Inline,
    Disconnected,
}

pub(crate) enum PrepThreadShutdown {
    Joined { state: PrepState, drained: bool },
    Panicked { drained: bool },
}

impl PrepLane {
    pub(crate) fn take_pending_draws(&self, draws: &mut Vec<DrawCall>) -> Vec<DrawCall> {
        let replacement = match self {
            Self::Inline(_) => Vec::new(),
            Self::Threaded(handle) => handle
                .try_take_recycled_draw_vec()
                .unwrap_or_else(|| Vec::with_capacity(draws.len())),
        };
        std::mem::replace(draws, replacement)
    }

    pub(crate) fn inline_state(&mut self) -> Option<&mut PrepState> {
        match self {
            PrepLane::Inline(state) => Some(state),
            PrepLane::Threaded(_) => None,
        }
    }

    pub(crate) fn is_threaded(&self) -> bool {
        matches!(self, PrepLane::Threaded(_))
    }
}

pub(crate) struct PrepThreadHandle {
    tx: crossbeam::channel::Sender<PrepEvent>,
    worker: std::thread::JoinHandle<PrepState>,
    draw_vec_recycle_rx: Option<crossbeam::channel::Receiver<Vec<DrawCall>>>,
    inflight_kicks: Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
    failed: Arc<AtomicBool>,
    behavior: PrepThreadBehavior,
}

struct PrepWorkerExitGuard {
    failed: Arc<AtomicBool>,
    inflight_kicks: Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
    clean: bool,
}

impl Drop for PrepWorkerExitGuard {
    fn drop(&mut self) {
        if self.clean {
            return;
        }
        self.failed.store(true, Ordering::Release);
        let (lock, condvar) = &*self.inflight_kicks;
        let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
        *inflight = 0;
        condvar.notify_all();
    }
}

fn record_prep_event_completion(failed: &AtomicBool, completed: bool) {
    if !completed {
        failed.store(true, Ordering::Release);
    }
}

const PREP_EVENT_QUEUE_CAPACITY: usize = 4096;

static DECODE_SWEEP_PLAN: std::sync::Mutex<Option<std::sync::Arc<[(u64, usize)]>>> =
    std::sync::Mutex::new(None);

pub(crate) fn publish_decode_sweep_plan(ranges: Vec<(u64, usize)>) {
    let mut plan = DECODE_SWEEP_PLAN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *plan = Some(std::sync::Arc::from(ranges));
}

pub(crate) fn decode_sweep_plan() -> Option<std::sync::Arc<[(u64, usize)]>> {
    DECODE_SWEEP_PLAN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}
const PREP_EVENT_BURST: usize = 256;
const DRAW_VEC_RECYCLE_QUEUE_CAPACITY: usize = 256;

static DRAW_VEC_RECYCLE_HITS: AtomicU64 = AtomicU64::new(0);
static DRAW_VEC_RECYCLE_MISSES: AtomicU64 = AtomicU64::new(0);
static DRAW_VEC_RECYCLE_RETURNS: AtomicU64 = AtomicU64::new(0);
static DRAW_VEC_RECYCLE_DROPS: AtomicU64 = AtomicU64::new(0);
static DRAW_VEC_RECYCLE_CAPACITY_SUM: AtomicU64 = AtomicU64::new(0);
static DRAW_VEC_RECYCLE_CAPACITY_MAX: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Default)]
struct DrawVecRecycleStats {
    hits: u64,
    misses: u64,
    returns: u64,
    drops: u64,
    capacity_sum: u64,
    capacity_max: u64,
}

impl DrawVecRecycleStats {
    fn snapshot() -> Self {
        Self {
            hits: DRAW_VEC_RECYCLE_HITS.load(Ordering::Relaxed),
            misses: DRAW_VEC_RECYCLE_MISSES.load(Ordering::Relaxed),
            returns: DRAW_VEC_RECYCLE_RETURNS.load(Ordering::Relaxed),
            drops: DRAW_VEC_RECYCLE_DROPS.load(Ordering::Relaxed),
            capacity_sum: DRAW_VEC_RECYCLE_CAPACITY_SUM.load(Ordering::Relaxed),
            capacity_max: DRAW_VEC_RECYCLE_CAPACITY_MAX.load(Ordering::Relaxed),
        }
    }

    fn since(self, previous: Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(previous.hits),
            misses: self.misses.saturating_sub(previous.misses),
            returns: self.returns.saturating_sub(previous.returns),
            drops: self.drops.saturating_sub(previous.drops),
            capacity_sum: self.capacity_sum.saturating_sub(previous.capacity_sum),
            capacity_max: self.capacity_max,
        }
    }

    fn average_returned_capacity(self) -> f64 {
        if self.returns == 0 {
            0.0
        } else {
            self.capacity_sum as f64 / self.returns as f64
        }
    }
}

fn record_draw_vec_recycle_result(hit: bool) {
    if !engb_prof_enabled() {
        return;
    }
    if hit {
        DRAW_VEC_RECYCLE_HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        DRAW_VEC_RECYCLE_MISSES.fetch_add(1, Ordering::Relaxed);
    }
}

fn record_draw_vec_recycle_return(returned: bool, capacity: usize) {
    if !engb_prof_enabled() {
        return;
    }
    if returned {
        DRAW_VEC_RECYCLE_RETURNS.fetch_add(1, Ordering::Relaxed);
        DRAW_VEC_RECYCLE_CAPACITY_SUM.fetch_add(capacity as u64, Ordering::Relaxed);
        DRAW_VEC_RECYCLE_CAPACITY_MAX.fetch_max(capacity as u64, Ordering::Relaxed);
    } else {
        DRAW_VEC_RECYCLE_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

fn draw_vec_recycling_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_RECYCLE_DRAW_VECS").ok().as_deref(),
            Some("1")
                | Some("true")
                | Some("TRUE")
                | Some("on")
                | Some("ON")
                | Some("yes")
                | Some("YES")
        )
    })
}

fn recycle_processed_draw_vec(
    tx: Option<&crossbeam::channel::Sender<Vec<DrawCall>>>,
    mut draws: Vec<DrawCall>,
) -> bool {
    let Some(tx) = tx else {
        record_draw_vec_recycle_return(false, draws.capacity());
        return false;
    };
    draws.clear();
    let capacity = draws.capacity();
    let returned = tx.try_send(draws).is_ok();
    record_draw_vec_recycle_return(returned, capacity);
    returned
}

fn try_receive_recycled_draw_vec(
    rx: Option<&crossbeam::channel::Receiver<Vec<DrawCall>>>,
) -> Option<Vec<DrawCall>> {
    let Some(rx) = rx else {
        record_draw_vec_recycle_result(false);
        return None;
    };
    let Ok(mut draws) = rx.try_recv() else {
        record_draw_vec_recycle_result(false);
        return None;
    };
    draws.clear();
    record_draw_vec_recycle_result(true);
    Some(draws)
}

fn prep_kicks_in_flight() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_PREP_KICKS_IN_FLIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|limit| (1..=4).contains(limit))
            .unwrap_or(2)
    })
}

impl PrepThreadHandle {
    pub(crate) fn send(&self, event: PrepEvent) {
        if self.send_recover(event).is_err() {
            log::error!("[gpu-prep] event dropped after prep thread exit");
        }
    }

    pub(crate) fn send_recover(&self, event: PrepEvent) -> Result<(), PrepEvent> {
        if self.failed.load(Ordering::Acquire) {
            return Err(event);
        }
        self.tx.send(event).map_err(|error| {
            self.failed.store(true, Ordering::Release);
            error.0
        })
    }

    pub(crate) fn behavior(&self) -> PrepThreadBehavior {
        self.behavior
    }

    pub(crate) fn failure_latch(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.failed)
    }

    pub(crate) fn try_take_recycled_draw_vec(&self) -> Option<Vec<DrawCall>> {
        try_receive_recycled_draw_vec(self.draw_vec_recycle_rx.as_ref())
    }

    pub(crate) fn begin_kick(&self) {
        if self.failed.load(Ordering::Acquire) {
            return;
        }
        let (lock, condvar) = &*self.inflight_kicks;
        let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
        while *inflight >= prep_kicks_in_flight() {
            if self.failed.load(Ordering::Acquire) {
                return;
            }
            let (next, _) = condvar
                .wait_timeout(inflight, Duration::from_millis(50))
                .unwrap_or_else(|error| error.into_inner());
            inflight = next;
        }
        if self.failed.load(Ordering::Acquire) {
            return;
        }
        *inflight += 1;
    }

    fn finish_kick(inflight_kicks: &(std::sync::Mutex<usize>, std::sync::Condvar)) {
        let (lock, condvar) = inflight_kicks;
        let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
        *inflight = inflight.saturating_sub(1);
        condvar.notify_all();
    }

    pub(crate) fn cancel_kick(&self) {
        Self::finish_kick(&self.inflight_kicks);
    }

    pub(crate) fn shutdown(self, flush_small_rts: bool) -> PrepThreadShutdown {
        let Self {
            tx,
            worker,
            draw_vec_recycle_rx: _,
            inflight_kicks: _,
            failed,
            behavior: _,
        } = self;
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        let queued = tx
            .send(PrepEvent::DrainBarrier {
                done: done_tx,
                flush_small_rts,
            })
            .is_ok();
        let drained = queued && done_rx.recv().unwrap_or(false) && !failed.load(Ordering::Acquire);
        drop(tx);
        match worker.join() {
            Ok(state) => PrepThreadShutdown::Joined { state, drained },
            Err(_) => PrepThreadShutdown::Panicked { drained },
        }
    }
}

pub(crate) struct PrepThreadResources {
    pub(crate) maxwell_dma: Arc<parking_lot::Mutex<MaxwellDma>>,
    pub(crate) fermi_2d: Arc<parking_lot::Mutex<Fermi2D>>,
    pub(crate) kepler_compute: Arc<parking_lot::Mutex<KeplerCompute>>,
    pub(crate) kepler_memory: Arc<parking_lot::Mutex<KeplerMemory>>,
    pub(crate) mappings: Arc<parking_lot::RwLock<GpuMappings>>,
    pub(crate) stats: Arc<PipelineStats>,
    pub(crate) mem_read: crate::AsyncMemoryRead,
    pub(crate) mem_write: crate::AsyncMemoryWrite,
    pub(crate) mem_copy: crate::AsyncMemoryCopy,
}

pub(crate) fn spawn_prep_thread(
    mut state: PrepState,
    resources: PrepThreadResources,
    behavior: PrepThreadBehavior,
) -> PrepThreadHandle {
    let (tx, rx) = crossbeam::channel::bounded::<PrepEvent>(PREP_EVENT_QUEUE_CAPACITY);
    let (draw_vec_recycle_tx, draw_vec_recycle_rx) = if draw_vec_recycling_enabled() {
        let (tx, rx) =
            crossbeam::channel::bounded::<Vec<DrawCall>>(DRAW_VEC_RECYCLE_QUEUE_CAPACITY);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let inflight_kicks = Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));
    let worker_inflight = Arc::clone(&inflight_kicks);
    let failed = Arc::new(AtomicBool::new(false));
    let worker_failed = Arc::clone(&failed);
    let worker = std::thread::Builder::new()
        .name("nexium-gpu-prep".to_string())
        .spawn(move || {
            let mut exit_guard = PrepWorkerExitGuard {
                failed: Arc::clone(&worker_failed),
                inflight_kicks: Arc::clone(&worker_inflight),
                clean: false,
            };
            nexium_common::thread_cpu_set::apply_current_thread_cpu_set(
                nexium_common::thread_cpu_set::ThreadCpuSetTarget::GpuPrep,
            );
            #[cfg(windows)]
            unsafe {
                #[link(name = "kernel32")]
                extern "system" {
                    fn GetCurrentThread() -> *mut std::ffi::c_void;
                    fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
                }
                let _ = SetThreadPriority(GetCurrentThread(), 2);
            }
            super::vk_dispatch::set_prep_sync_guest_read(resources.mem_read.clone());
            if nexium_memory::fastmem::decode_write_watch_sweep_enabled() {
                nexium_memory::fastmem::enable_write_watch_memo_for_thread();
            }
            let mem_read = move |addr: u64, buf: &mut [u8]| (resources.mem_read)(addr, buf);
            let mem_write = move |addr: u64, buf: &[u8]| (resources.mem_write)(addr, buf);
            let mem_copy =
                move |src: u64, dst: u64, len: usize| (resources.mem_copy)(src, dst, len);
            let mut prof_busy_ns = 0u64;
            let mut prof_events = 0u64;
            let mut prof_kicks = 0u64;
            let mut prof_class_ns = [0u64; 16];
            let mut prof_recycle_last = DrawVecRecycleStats::snapshot();
            fn event_class_index(event: &PrepEvent) -> usize {
                match event {
                    PrepEvent::EngineMethod { .. } => 0,
                    PrepEvent::EngineMethods { .. } => 1,
                    PrepEvent::InlineUploadMethods(_) => 2,
                    PrepEvent::ConstbufWrites { .. } => 3,
                    PrepEvent::Draws { .. } => 4,
                    PrepEvent::SemRelease(_) => 5,
                    PrepEvent::Barrier { .. } => 6,
                    PrepEvent::PullerSemWrite { .. } => 7,
                    PrepEvent::KickBegin { .. } => 8,
                    PrepEvent::EntryBegin => 9,
                    PrepEvent::HardFlush { .. } => 10,
                    PrepEvent::KickEnd { .. } => 11,
                    PrepEvent::Present { .. } => 12,
                    PrepEvent::DrainBarrier { .. } => 13,
                    PrepEvent::SemAcquire { .. } => 14,
                    _ => 15,
                }
            }
            let prof_enabled = std::env::var_os("NEXIUM_PREP_PROFILE").is_some();
            let mut burst: Vec<PrepEvent> = Vec::with_capacity(PREP_EVENT_BURST);
            while let Ok(first) = rx.recv() {
                let burst_started = prof_enabled.then(std::time::Instant::now);
                burst.push(first);
                while burst.len() < PREP_EVENT_BURST {
                    match rx.try_recv() {
                        Ok(event) => burst.push(event),
                        Err(_) => break,
                    }
                }
                let mut maxwell_dma = resources.maxwell_dma.lock();
                let mut fermi_2d = resources.fermi_2d.lock();
                let mut kepler_compute = resources.kepler_compute.lock();
                let mut kepler_memory = resources.kepler_memory.lock();
                let mappings = resources.mappings.read_recursive();
                for event in burst.drain(..) {
                    if worker_failed.load(Ordering::Acquire) {
                        match event {
                            PrepEvent::KickEnd { on_complete, .. } => {
                                drop(on_complete);
                                PrepThreadHandle::finish_kick(&worker_inflight);
                            }
                            PrepEvent::DrainBarrier { done, .. } => {
                                let _ = done.send(false);
                            }
                            _ => {}
                        }
                        continue;
                    }
                    let class_index = event_class_index(&event);
                    let class_started = burst_started.map(|_| std::time::Instant::now());
                    let (event, deferred_completion, is_kick_end) = match event {
                        PrepEvent::KickEnd {
                            hard_after,
                            writeback_small_rts,
                            on_complete,
                        } => (
                            PrepEvent::KickEnd {
                                hard_after,
                                writeback_small_rts,
                                on_complete: None,
                            },
                            on_complete,
                            true,
                        ),
                        other => (other, None, false),
                    };
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut engines = PrepEngines {
                            maxwell_dma: &mut maxwell_dma,
                            fermi_2d: &mut fermi_2d,
                            kepler_compute: &mut kepler_compute,
                            kepler_memory: &mut kepler_memory,
                        };
                        state.run_event(
                            event,
                            draw_vec_recycle_tx
                                .as_ref()
                                .map(DrawVecRecycle::Threaded)
                                .unwrap_or(DrawVecRecycle::Discard),
                            &mut engines,
                            &mappings,
                            &resources.stats,
                            &mem_read,
                            &mem_write,
                            &mem_copy,
                        )
                    }));
                    if outcome.is_err() {
                        log::error!("[gpu-prep] event handler panicked; clearing prep caches");
                        state.vk_batch.clear();
                        state.ssbo_snapshot_cache.clear();
                    }
                    let completed = matches!(outcome, Ok(true));
                    record_prep_event_completion(&worker_failed, completed);
                    if completed {
                        state.schedule_kick_completion(deferred_completion);
                    }
                    if is_kick_end {
                        PrepThreadHandle::finish_kick(&worker_inflight);
                        prof_kicks += 1;
                    }
                    if let Some(started) = class_started {
                        prof_class_ns[class_index] += started.elapsed().as_nanos() as u64;
                    }
                    prof_events += 1;
                }
                if let Some(started) = burst_started {
                    prof_busy_ns += started.elapsed().as_nanos() as u64;
                    if prof_kicks >= 64 {
                        const CLASS_NAMES: [&str; 16] = [
                            "engm", "engb", "inup", "cbwr", "draw", "srel", "barr", "psem", "kbeg",
                            "ebeg", "hflv", "kend", "pres", "drnb", "sacq", "othr",
                        ];
                        let mut classes = String::new();
                        for (name, ns) in CLASS_NAMES.iter().zip(prof_class_ns.iter()) {
                            if *ns != 0 {
                                classes.push_str(&format!(
                                    " {}={:.2}",
                                    name,
                                    *ns as f64 / prof_kicks as f64 / 1_000_000.0
                                ));
                            }
                        }
                        let recycle_total = DrawVecRecycleStats::snapshot();
                        let recycle_window = recycle_total.since(prof_recycle_last);
                        log::warn!(
                            "[prep-prof] kicks=64 busy_ms_per_kick={:.2} events_per_kick={:.1} |{} | draw_vec_recycle window=h{}/m{}/r{}/d{} cap_avg={:.1} total=h{}/m{}/r{}/d{} cap_avg={:.1} cap_max={} queue_cap={}",
                            prof_busy_ns as f64 / prof_kicks as f64 / 1_000_000.0,
                            prof_events as f64 / prof_kicks as f64,
                            classes,
                            recycle_window.hits,
                            recycle_window.misses,
                            recycle_window.returns,
                            recycle_window.drops,
                            recycle_window.average_returned_capacity(),
                            recycle_total.hits,
                            recycle_total.misses,
                            recycle_total.returns,
                            recycle_total.drops,
                            recycle_total.average_returned_capacity(),
                            recycle_total.capacity_max,
                            DRAW_VEC_RECYCLE_QUEUE_CAPACITY,
                        );
                        prof_recycle_last = recycle_total;
                        prof_busy_ns = 0;
                        prof_events = 0;
                        prof_kicks = 0;
                        prof_class_ns = [0u64; 16];
                    }
                }
            }

            let (lock, condvar) = &*worker_inflight;
            let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
            *inflight = 0;
            condvar.notify_all();
            drop(inflight);
            exit_guard.clean = true;
            state
        })
        .expect("spawn GPU prep thread");
    PrepThreadHandle {
        tx,
        worker,
        draw_vec_recycle_rx,
        inflight_kicks,
        failed,
        behavior,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        fermi_ordered_exact_value_enabled, nonterminal_inline_data_run_end,
        record_prep_event_completion, recycle_processed_draw_vec,
        texture_cache_invalidate_clear_value_enabled, try_receive_recycled_draw_vec, DrawVecRecycle,
        PrepEvent, PrepLane, PrepState, PrepThreadBehavior, PrepThreadHandle, PrepThreadShutdown,
    };
    use crate::gpu::engines::maxwell3d::DrawCall;
    use crate::gpu::GpuMappings;

    fn compute_span() -> crate::gpu::engines::maxwell_compute::PendingComputeWritebackSpan {
        crate::gpu::engines::maxwell_compute::PendingComputeWritebackSpan {
            dispatch_id: 1,
            serial: 1,
            resource_index: 0,
            binding: 0,
            raw: true,
            gpu_va: 0x1000,
            cpu_addr: 0x8000,
            len: 0x100,
            source_offset: 0,
        }
    }

    #[test]
    fn compute_memory_barrier_resolves_before_overlapping_access_only_once() {
        let barrier = super::ComputeMemoryBarrier::new(&[compute_span()], &GpuMappings::default());
        let value = std::cell::Cell::new(0);
        barrier.before_access(0x8100, 16, || panic!("adjacent range is independent"));
        barrier.before_access(0x8000, 0, || panic!("empty range is independent"));
        barrier.before_access(0x7ff0, 16, || panic!("adjacent range is independent"));
        barrier.before_access(0x7fff, 2, || value.set(42));
        assert_eq!(value.get(), 42);
        barrier.before_access(0x8000, 16, || panic!("already resolved"));
        assert!(!barrier.armed.get());
    }

    #[test]
    fn compute_memory_barrier_orders_cpu_overwrites_after_old_results() {
        let barrier = super::ComputeMemoryBarrier::new(&[compute_span()], &GpuMappings::default());
        let memory = std::cell::Cell::new(0);
        barrier.before_access(0x8000, 4, || memory.set(10));
        memory.set(20);
        barrier.before_access(0x8000, 4, || memory.set(10));
        assert_eq!(memory.get(), 20);
        assert!(super::ranges_overlap(u64::MAX - 8, 32, u64::MAX - 4, 4));
    }

    #[test]
    fn compute_draw_dependency_covers_aliases_and_ordinary_render_targets() {
        let mut mappings = GpuMappings::default();
        mappings.add(0x1000, 0x1000, 0x8000, 1);
        mappings.add(0x4000, 0x1000, 0x8000, 1);
        let barrier = super::ComputeMemoryBarrier::new(&[compute_span()], &mappings);
        assert!(barrier.gpu_ranges.contains(&(0x4000, 0x100)));
        let mut draw = DrawCall::default();
        draw.rt[0].address_lo = 0x4000;
        draw.rt[0].width = 4;
        draw.rt[0].height = 4;
        draw.rt[0].depth = 1;
        assert!(super::draws_write_pending_compute(&[draw.clone()], &barrier.gpu_ranges));
        draw.is_clear = true;
        assert!(super::draws_write_pending_compute(&[draw.clone()], &barrier.gpu_ranges));
        draw.rt[0].address_lo = 0x9000;
        assert!(!super::draws_write_pending_compute(&[draw], &barrier.gpu_ranges));
    }

    #[test]
    fn sorted_pending_compute_overlap_matches_linear_scan() {
        let ranges = [
            (0x1000u64, 0x100u64),
            (0x1100, 0x80),
            (0x3000, 0x10),
            (0x5000, 0x1000),
            (0x5800, 0x100),
            (0x7000, 0),
        ];
        let merged = super::coalesce_sorted_ranges(&ranges);
        assert_eq!(merged, vec![(0x1000, 0x180), (0x3000, 0x10), (0x5000, 0x1000)]);
        for base in (0x0..0x8000u64).step_by(0x40) {
            for size in [0u64, 0x1, 0x40, 0x200, 0x1000, u64::MAX] {
                let linear = ranges
                    .iter()
                    .any(|&(address, length)| super::ranges_overlap(base, size, address, length));
                assert_eq!(
                    super::sorted_ranges_overlap(&merged, base, size),
                    linear,
                    "base={base:#x} size={size:#x}"
                );
            }
        }
        let mut draw = DrawCall::default();
        draw.rt[0].address_lo = 0x5f00;
        draw.rt[0].width = 4;
        draw.rt[0].height = 4;
        draw.rt[0].depth = 1;
        assert!(super::draws_write_pending_compute_sorted(&[draw.clone()], &merged));
        draw.rt[0].address_lo = 0x6000;
        assert!(!super::draws_write_pending_compute_sorted(&[draw], &merged));
    }

    #[test]
    fn compute_barrier_cache_rearms_and_tracks_dispatches_and_remappings() {
        let mut cache = super::ComputeBarrierCache::default();
        let mut mappings = GpuMappings::default();
        mappings.add(0x1000, 0x1000, 0x8000, 1);
        let first = cache.get(1, &[compute_span()], &mappings);
        first.before_access(0x8000, 4, || {});
        let second = cache.get(1, &[compute_span()], &mappings);
        assert!(!first.armed.get());
        assert!(second.armed.get());
        assert!(std::sync::Arc::ptr_eq(&first.gpu_ranges, &second.gpu_ranges));
        mappings.add(0x4000, 0x1000, 0x8000, 1);
        let remapped = cache.get(1, &[compute_span()], &mappings);
        assert!(remapped.gpu_ranges.contains(&(0x4000, 0x100)));
        let mut next_span = compute_span();
        next_span.cpu_addr = 0xa000;
        let next = cache.get(2, &[next_span], &mappings);
        next.before_access(0x8000, 4, || panic!("old dispatch no longer pending"));
        let resolved = std::cell::Cell::new(false);
        next.before_access(0xa000, 4, || resolved.set(true));
        assert!(resolved.get());
        let drained = cache.get(3, &[], &mappings);
        assert!(!drained.armed.get());
        assert!(drained.gpu_ranges.is_empty());
    }

    fn test_prep_handle(
        tx: crossbeam::channel::Sender<PrepEvent>,
        worker: std::thread::JoinHandle<PrepState>,
    ) -> PrepThreadHandle {
        PrepThreadHandle {
            tx,
            worker,
            draw_vec_recycle_rx: None,
            inflight_kicks: std::sync::Arc::new((
                std::sync::Mutex::new(0),
                std::sync::Condvar::new(),
            )),
            failed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            behavior: PrepThreadBehavior::Pipeline,
        }
    }

    #[test]
    fn bulk_compute_upload_scanner_stops_before_terminal_word() {
        let methods = [
            (0x6C, 0, false),
            (0x6D, 1, false),
            (0x6D, 2, false),
            (0x6D, 3, true),
            (0x6D, 4, false),
            (0xAF, 5, false),
        ];

        assert_eq!(nonterminal_inline_data_run_end(&methods, 0), 0);
        assert_eq!(nonterminal_inline_data_run_end(&methods, 1), 3);
        assert_eq!(nonterminal_inline_data_run_end(&methods, 3), 3);
        assert_eq!(nonterminal_inline_data_run_end(&methods, 4), 5);
        assert_eq!(
            nonterminal_inline_data_run_end(&methods, methods.len()),
            methods.len()
        );
    }

    #[test]
    fn texture_invalidates_do_not_destroy_persistent_images_by_default() {
        use std::ffi::OsStr;

        assert!(!texture_cache_invalidate_clear_value_enabled(None));
        for disabled in ["", "0", "false", "OFF", " no "] {
            assert!(!texture_cache_invalidate_clear_value_enabled(Some(
                OsStr::new(disabled)
            )));
        }
        for enabled in ["1", "true", "on", "yes"] {
            assert!(texture_cache_invalidate_clear_value_enabled(Some(
                OsStr::new(enabled)
            )));
        }
    }

    #[test]
    fn ordered_exact_fermi_copies_default_on_with_explicit_rollback() {
        assert!(fermi_ordered_exact_value_enabled(None));
        for enabled in ["", "1", "true", "ON", " yes "] {
            assert!(fermi_ordered_exact_value_enabled(Some(enabled)));
        }
        for disabled in ["0", "false", "OFF", " no "] {
            assert!(!fermi_ordered_exact_value_enabled(Some(disabled)));
        }
    }

    #[test]
    fn kick_and_barrier_boundaries_advance_resident_mirror_sweep_but_entries_do_not() {
        let mut state = PrepState::new();
        let before = state.ssbo_snapshot_cache.mirror_sweep_generation();

        state.begin_ssbo_snapshot_entry();
        assert_eq!(state.ssbo_snapshot_cache.mirror_sweep_generation(), before);

        state.ssbo_snapshot_cache.mirror_cbuf_barrier_bump();
        let after_barrier = state.ssbo_snapshot_cache.mirror_sweep_generation();
        assert_ne!(after_barrier, before);

        state.begin_ssbo_snapshot_epoch();
        assert_ne!(
            state.ssbo_snapshot_cache.mirror_sweep_generation(),
            after_barrier
        );
    }

    #[test]
    fn semaphore_acquire_advances_resident_mirror_sweep() {
        let mut state = PrepState::new();
        state.sync_semaphore_acquire_sweep();
        let before = state.ssbo_snapshot_cache.mirror_sweep_generation();

        super::note_semaphore_acquire();
        state.sync_semaphore_acquire_sweep();

        assert_ne!(state.ssbo_snapshot_cache.mirror_sweep_generation(), before);
    }

    #[test]
    fn draw_vec_recycle_roundtrips_outer_capacity_and_returns_empty() {
        let (tx, rx) = crossbeam::channel::bounded(1);
        let mut draws = Vec::with_capacity(19);
        let mut draw = DrawCall::default();
        draw.inline_indices = vec![1, 2, 3, 4];
        draws.push(draw);
        let capacity = draws.capacity();

        assert!(recycle_processed_draw_vec(Some(&tx), draws));
        let recycled = try_receive_recycled_draw_vec(Some(&rx)).unwrap();

        assert!(recycled.is_empty());
        assert_eq!(recycled.capacity(), capacity);
    }

    #[test]
    fn draw_vec_recycle_unavailable_and_full_paths_never_wait() {
        assert!(!recycle_processed_draw_vec(None, vec![DrawCall::default()]));
        assert!(try_receive_recycled_draw_vec(None).is_none());

        let (tx, rx) = crossbeam::channel::bounded(1);
        let queued = Vec::<DrawCall>::with_capacity(7);
        tx.try_send(queued).unwrap();
        assert!(!recycle_processed_draw_vec(
            Some(&tx),
            vec![DrawCall::default()]
        ));
        assert_eq!(rx.try_recv().unwrap().capacity(), 7);

        let (disconnected_tx, disconnected_rx) = crossbeam::channel::bounded(1);
        drop(disconnected_rx);
        assert!(!recycle_processed_draw_vec(
            Some(&disconnected_tx),
            vec![DrawCall::default()]
        ));
    }

    #[test]
    fn inline_draw_vec_recycle_handles_empty_and_failed_events() {
        let mut state = PrepState::new();
        let mappings = GpuMappings::new();
        let stats = crate::PipelineStats::default();
        let mut maxwell_dma = super::MaxwellDma::new();
        let mut fermi_2d = super::Fermi2D::new();
        let mut kepler_compute = super::KeplerCompute::new();
        let mut kepler_memory = super::KeplerMemory::new();
        let mut engines = super::PrepEngines {
            maxwell_dma: &mut maxwell_dma,
            fermi_2d: &mut fermi_2d,
            kepler_compute: &mut kepler_compute,
            kepler_memory: &mut kepler_memory,
        };
        let mut pending = Vec::<DrawCall>::with_capacity(13);
        let pointer = pending.as_ptr();
        let capacity = pending.capacity();
        let draws = std::mem::take(&mut pending);
        assert!(state.run_event(
            PrepEvent::Draws {
                draws,
                gs_debug: None,
                replay_constbuf_writes: Vec::new(),
                constbuf_trace: None,
            },
            DrawVecRecycle::Inline(&mut pending),
            &mut engines,
            &mappings,
            &stats,
            &|_, _| panic!("empty draws do not read guest memory"),
            &|_, _| panic!("empty draws do not write guest memory"),
            &|_, _, _| false,
        ));
        assert!(pending.is_empty());
        assert_eq!(pending.as_ptr(), pointer);
        assert_eq!(pending.capacity(), capacity);

        state.vk_flush_completed = false;
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        assert!(!state.run_event(
            PrepEvent::DrainBarrier {
                done: done_tx,
                flush_small_rts: false,
            },
            DrawVecRecycle::Inline(&mut pending),
            &mut engines,
            &mappings,
            &stats,
            &|_, _| false,
            &|_, _| false,
            &|_, _, _| false,
        ));
        assert!(!done_rx.try_recv().unwrap());
        assert!(pending.is_empty());
        assert_eq!(pending.as_ptr(), pointer);
        assert_eq!(pending.capacity(), capacity);
    }

    #[test]
    fn threaded_draw_vec_recycle_uses_returned_capacity_before_allocating() {
        let (event_tx, _event_rx) = crossbeam::channel::bounded(1);
        let worker = std::thread::spawn(PrepState::new);
        let mut handle = test_prep_handle(event_tx, worker);
        let (recycle_tx, recycle_rx) = crossbeam::channel::bounded(1);
        handle.draw_vec_recycle_rx = Some(recycle_rx);
        let lane = PrepLane::Threaded(handle);

        let recycled = Vec::<DrawCall>::with_capacity(17);
        let recycled_pointer = recycled.as_ptr();
        let recycled_capacity = recycled.capacity();
        assert!(DrawVecRecycle::Threaded(&recycle_tx).recycle(recycled));
        let mut pending = vec![DrawCall::default(), DrawCall::default()];
        pending[0].first_vertex = 3;
        pending[1].first_vertex = 9;
        let pending_pointer = pending.as_ptr();
        let draws = lane.take_pending_draws(&mut pending);

        assert_eq!(draws.as_ptr(), pending_pointer);
        assert_eq!(draws[0].first_vertex, 3);
        assert_eq!(draws[1].first_vertex, 9);
        assert!(pending.is_empty());
        assert_eq!(pending.as_ptr(), recycled_pointer);
        assert_eq!(pending.capacity(), recycled_capacity);

        pending.push(DrawCall::default());
        let next = lane.take_pending_draws(&mut pending);
        assert_eq!(next.as_ptr(), recycled_pointer);
        assert_eq!(next.len(), 1);
        assert!(pending.is_empty());
        assert!(pending.capacity() >= next.len());
        let PrepLane::Threaded(handle) = lane else {
            unreachable!()
        };
        handle.worker.join().unwrap();
    }

    #[test]
    fn hard_tail_propagates_buffered_flush_failure_once() {
        let mut state = PrepState::new();
        state.vk_flush_completed = false;

        assert!(!state.finish_prepared_draw_packet_tail(true));
        assert!(state.finish_prepared_draw_packet_tail(true));
    }

    #[test]
    fn async_small_rt_join_propagates_worker_failure() {
        let mut state = PrepState::new();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send((false, Vec::new())).unwrap();
        state.pending_small_rt_wb = Some(rx);

        assert!(!state.join_small_rt_writeback(&GpuMappings::new()));
    }

    #[test]
    fn prep_thread_shutdown_drains_and_joins() {
        let (tx, rx) = crossbeam::channel::bounded(2);
        let worker = std::thread::spawn(move || {
            let state = PrepState::new();
            while let Ok(event) = rx.recv() {
                if let PrepEvent::DrainBarrier { done, .. } = event {
                    let _ = done.send(true);
                }
            }
            state
        });
        let handle = test_prep_handle(tx, worker);

        assert!(matches!(
            handle.shutdown(false),
            PrepThreadShutdown::Joined { drained: true, .. }
        ));
    }

    #[test]
    fn prep_thread_shutdown_propagates_failed_barrier() {
        let (tx, rx) = crossbeam::channel::bounded(2);
        let worker = std::thread::spawn(move || {
            let state = PrepState::new();
            while let Ok(event) = rx.recv() {
                if let PrepEvent::DrainBarrier { done, .. } = event {
                    let _ = done.send(false);
                }
            }
            state
        });
        let handle = test_prep_handle(tx, worker);

        assert!(matches!(
            handle.shutdown(false),
            PrepThreadShutdown::Joined { drained: false, .. }
        ));
    }

    #[test]
    fn prep_thread_shutdown_reports_disconnected_barrier() {
        let (tx, rx) = crossbeam::channel::bounded(1);
        drop(rx);
        let worker = std::thread::spawn(PrepState::new);
        let handle = test_prep_handle(tx, worker);

        assert!(matches!(
            handle.shutdown(false),
            PrepThreadShutdown::Joined { drained: false, .. }
        ));
    }

    #[test]
    fn prep_event_failure_latch_never_recovers() {
        let failed = std::sync::atomic::AtomicBool::new(false);

        record_prep_event_completion(&failed, false);
        assert!(failed.load(std::sync::atomic::Ordering::Acquire));

        record_prep_event_completion(&failed, true);
        assert!(failed.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn failed_prep_thread_rejects_events_and_shutdown_success() {
        let (tx, rx) = crossbeam::channel::bounded(2);
        let worker = std::thread::spawn(move || {
            let state = PrepState::new();
            while let Ok(event) = rx.recv() {
                if let PrepEvent::DrainBarrier { done, .. } = event {
                    let _ = done.send(true);
                }
            }
            state
        });
        let handle = test_prep_handle(tx, worker);
        handle
            .failure_latch()
            .store(true, std::sync::atomic::Ordering::Release);

        assert!(handle
            .send_recover(PrepEvent::KickBegin {
                swept: Vec::new(),
                dirty: Vec::new(),
            })
            .is_err());
        assert!(matches!(
            handle.shutdown(false),
            PrepThreadShutdown::Joined { drained: false, .. }
        ));
    }
}
