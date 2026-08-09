use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use super::engines::maxwell3d::{DrawCall, GsDebugRegs, PendingSemaphoreWrite};
use super::engines::{Fermi2D, KeplerCompute, MaxwellDma};
use super::engines::{KeplerMemory, KeplerMemoryWriteOutcome};
use super::pusher::{
    contiguous_constbuf_write_run_end, gpu_profile_enabled, semrel_legacy, semrel_verify,
};
use super::vk_dispatch::{PreparedDrawPacketizer, SsboSnapshotCache};
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
    KickBegin,
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
    },
    DrainBarrier {
        done: crossbeam::channel::Sender<()>,
        flush_small_rts: bool,
    },
    SemAcquire {
        gpu_va: u64,
        payload: u32,
        ack: crossbeam::channel::Sender<()>,
    },
    SetRenderer(Option<Arc<nexium_gpu::Renderer>>),
    SetGuestMemory(Option<super::GuestMemoryAccess>),
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

pub(crate) struct PrepState {
    pub(crate) renderer: Option<Arc<nexium_gpu::Renderer>>,
    pub(crate) guest_memory: Option<super::GuestMemoryAccess>,
    pending_small_rt_wb:
        Option<std::sync::mpsc::Receiver<Vec<super::vk_dispatch::GuestWriteChunk>>>,
    pub(crate) vk_batch: Vec<nexium_gpu::draw::Maxwell3dDrawCall>,
    pub(crate) prepared_draw_packets: PreparedDrawPacketizer,
    pub(crate) ssbo_snapshot_cache: SsboSnapshotCache,
    pub(crate) inline_upload: KeplerMemory,
    pub(crate) constbuf_invalidation_scratch: Vec<(u64, usize)>,
    pub(crate) constbuf_patched_scratch: Vec<(u64, u64, usize)>,
    pub(crate) constbuf_bytes_scratch: Vec<u8>,
    #[cfg(test)]
    pub(crate) prepared_packet_drain_counts: PreparedPacketDrainCounts,
}

impl PrepState {
    pub(crate) fn run_event(
        &mut self,
        event: PrepEvent,
        draw_vec_recycle_tx: Option<&crossbeam::channel::Sender<Vec<DrawCall>>>,
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
                self.process_inline_upload_methods(methods, mappings, mem_read, mem_write);
                true
            }
            PrepEvent::KickBegin => {
                self.begin_ssbo_snapshot_epoch();
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
                    self.ssbo_snapshot_cache.clear_ssbo_snapshots();
                }
                true
            }
            PrepEvent::KickEnd {
                hard_after,
                writeback_small_rts,
                on_complete,
            } => {
                self.join_small_rt_writeback(mappings);
                let kp_tail = kickprof::start();
                self.resolve_pending_compute(mappings, mem_write);
                kickprof::add(kickprof::RESOLVE_TAIL, kp_tail);
                if hard_after {
                    self.record_flush_reason(kickprof::FLUSH_HARD_TAIL);
                }
                self.flush_vk_with_boundary(mappings, mem_read, mem_write, hard_after);
                self.finish_prepared_draw_packet_tail(hard_after, writeback_small_rts);
                if writeback_small_rts {
                    if let Some(r) = self.renderer.clone() {
                        let kp = kickprof::start();
                        self.writeback_small_rts(&r, mappings, mem_write);
                        kickprof::add(kickprof::SMALLRT, kp);
                    }
                }
                self.end_ssbo_snapshot_epoch();
                if let Some(on_complete) = on_complete {
                    on_complete();
                }
                true
            }
            PrepEvent::Present {
                job,
                flush_small_rts,
            } => {
                let kp_wb = super::pusher::kickprof::start();
                if flush_small_rts
                    && self.renderer.is_some()
                    && super::vk_dispatch::has_pending_small_rt_writebacks()
                {
                    let renderer = self.renderer.clone().unwrap();
                    let async_memory = super::vk_dispatch::async_small_rt_writeback_enabled()
                        .then(|| self.guest_memory.clone())
                        .flatten()
                        .filter(super::GuestMemoryAccess::is_available);
                    if let Some(memory) = async_memory {
                        self.join_small_rt_writeback(mappings);
                        let kp_flush = super::pusher::kickprof::start();
                        let flushed = self.flush_prepared_draw_packets();
                        super::pusher::kickprof::add(
                            super::pusher::kickprof::PRES_WB_FLUSH,
                            kp_flush,
                        );
                        if flushed {
                            self.pending_small_rt_wb =
                                super::vk_dispatch::spawn_small_rt_writeback_async(
                                    &renderer, memory,
                                );
                        } else {
                            log::warn!("[rt-writeback] skipped after prepared draw drain failure");
                        }
                    } else {
                        let _ = self.writeback_small_rts(&renderer, mappings, mem_write);
                    }
                } else {
                    self.flush_prepared_draw_packets();
                }
                super::pusher::kickprof::add(super::pusher::kickprof::PRES_WB, kp_wb);
                let kp_submit = super::pusher::kickprof::start();
                if let Some(rt) = crate::render_thread::maybe_render_thread() {
                    rt.submit_named("async-present-readback", job);
                } else {
                    job();
                }
                super::pusher::kickprof::add(super::pusher::kickprof::PRES_SUBMIT, kp_submit);
                true
            }
            PrepEvent::DrainBarrier {
                done,
                flush_small_rts,
            } => {
                self.join_small_rt_writeback(mappings);
                if flush_small_rts
                    && self.renderer.is_some()
                    && super::vk_dispatch::has_pending_small_rt_writebacks()
                {
                    let renderer = self.renderer.clone().unwrap();
                    let _ = self.writeback_small_rts(&renderer, mappings, mem_write);
                } else {
                    self.flush_prepared_draw_packets();
                }
                let _ = done.send(());
                true
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
                let _ = ack.send(());
                true
            }
            PrepEvent::SetRenderer(renderer) => {
                self.set_renderer(renderer);
                true
            }
            PrepEvent::SetGuestMemory(memory) => {
                self.set_guest_memory_access(memory);
                true
            }
            PrepEvent::ConstbufWrites { writes, trace } => {
                let kp = kickprof::start();
                self.commit_constbuf_writes(&writes, trace, mappings, mem_write);
                kickprof::add(kickprof::CBUFWB, kp);
                true
            }
            PrepEvent::PullerSemWrite {
                gpu_va,
                payload,
                long,
            } => {
                self.resolve_pending_compute(mappings, mem_write);
                self.ssbo_snapshot_cache.invalidate_gpu_write(
                    mappings,
                    gpu_va,
                    if long { 16 } else { 4 },
                );
                if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                    if long {
                        let ts = super::pusher::GPU_SEM_TICK.fetch_add(1, AtomicOrdering::Relaxed);
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
                        static CLEAR_ON_TIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let clear_on_tic = *CLEAR_ON_TIC.get_or_init(|| {
                            std::env::var_os("NEXIUM_TIC_INVALIDATE_CLEAR").is_some()
                        });
                        if clear_on_tic {
                            if let Some(r) = self.renderer.clone() {
                                if let Some(rt) = crate::render_thread::maybe_render_thread() {
                                    self.flush_prepared_draw_packets();
                                    rt.submit_named(
                                        "texture-cache-invalidate",
                                        Box::new(move || r.clear_texture_cache()),
                                    );
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
                self.resolve_pending_compute(mappings, mem_write);
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
                if can_complete_asynchronously {
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
                        if !scheduled {
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
                                let ts = super::pusher::GPU_SEM_TICK
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                let mut buf = [0u8; 16];
                                buf[0..8].copy_from_slice(&(write.payload as u64).to_le_bytes());
                                buf[8..16].copy_from_slice(&ts.to_le_bytes());
                                mem_write(cpu, &buf)
                            } else {
                                mem_write(cpu, &write.payload.to_le_bytes())
                            };
                            log::trace!(
                            "pusher: fence release gpu_va={:#x} cpu={:#x} payload={:#x} long={} write_ok={}",
                            write.gpu_va,
                            cpu,
                            write.payload,
                            write.long,
                            ok
                        );
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
                let compute_spans = self.resolve_pending_compute_report(mappings, mem_write);
                let mut compute_probe =
                    super::vk_dispatch::ComputeGraphicsProbe::new(&compute_spans);
                let mut committed_constbuf_writes = 0;
                if let Some(r) = self.renderer.clone() {
                    if replay_constbuf_writes.is_empty() {
                        let kp = kickprof::start();
                        super::vk_dispatch::enqueue_draws(
                            &draws,
                            &mut self.vk_batch,
                            &mut self.prepared_draw_packets,
                            &mut self.ssbo_snapshot_cache,
                            mappings,
                            gs_debug.as_ref(),
                            &r,
                            engines.maxwell_dma,
                            &*engines.fermi_2d,
                            mem_read,
                            mem_write,
                            compute_probe.as_mut(),
                        );
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
                            super::vk_dispatch::enqueue_draws(
                                &draws[draw_start..draw_end],
                                &mut self.vk_batch,
                                &mut self.prepared_draw_packets,
                                &mut self.ssbo_snapshot_cache,
                                mappings,
                                gs_debug.as_ref(),
                                &r,
                                engines.maxwell_dma,
                                &*engines.fermi_2d,
                                mem_read,
                                mem_write,
                                compute_probe.as_mut(),
                            );
                            kickprof::add(kickprof::ENQ, kp);
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
                if committed_constbuf_writes < replay_constbuf_writes.len() {
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
                if let Some(probe) = compute_probe {
                    probe.finish();
                }
                recycle_processed_draw_vec(draw_vec_recycle_tx, draws);
                true
            }
            PrepEvent::EngineMethods { class, methods } => {
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
            } => self.run_engine_method(
                class, method, arg, is_last, engines, mappings, stats, mem_read, mem_write,
                mem_copy,
            ),
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
                    if requires_hard_boundary {
                        let spans = maxwell_dma.transfer_spans();
                        let (dst_gpu, dst_size) = spans[1];
                        self.ssbo_snapshot_cache
                            .invalidate_gpu_write(mappings, dst_gpu, dst_size);
                        if super::engines::maxwell_compute::has_pending_writebacks() {
                            let overlaps = spans.iter().any(|&(gpu_va, size)| {
                                let cpu_addr = mappings
                                    .cpu_range_for(gpu_va)
                                    .map(|(cpu, _)| cpu)
                                    .unwrap_or(0);
                                super::engines::maxwell_compute::pending_writeback_overlaps(
                                    gpu_va, cpu_addr, size,
                                )
                            });
                            if overlaps {
                                self.resolve_pending_compute(mappings, mem_write);
                            }
                        }
                        if let Some(r) = self.renderer.clone() {
                            let stage_started = kickprof::start();
                            if super::vk_dispatch::input_mirror_enabled() {
                                let recorded: std::cell::RefCell<Vec<(u64, usize)>> =
                                    std::cell::RefCell::new(Vec::new());
                                let recording_write = |addr: u64, bytes: &[u8]| {
                                    recorded.borrow_mut().push((addr, bytes.len()));
                                    mem_write(addr, bytes)
                                };
                                maxwell_dma.stage_rt_source(arg, mappings, &r, &recording_write);
                                for (addr, len) in recorded.into_inner() {
                                    self.ssbo_snapshot_cache.mirror_mark_cpu(addr, len);
                                }
                            } else {
                                maxwell_dma.stage_rt_source(arg, mappings, &r, mem_write);
                            }
                            kickprof::add(kickprof::DMA_STAGE, stage_started);
                        }
                    }
                    let pre = maxwell_dma.blit_count;
                    maxwell_dma
                        .dispatch_method(method, arg, mappings, mem_read, mem_write, mem_copy);
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
                        self.ssbo_snapshot_cache.clear_ssbo_snapshots();
                    }
                    let kp = kickprof::start();
                    let outcome =
                        kepler_memory.dispatch_method(method, arg, mappings, mem_read, mem_write);
                    super::pusher::apply_kepler_memory_write(
                        &mut self.ssbo_snapshot_cache,
                        mappings,
                        outcome,
                    );
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
                            self.ssbo_snapshot_cache.clear_ssbo_snapshots();
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
                    super::pusher::apply_kepler_memory_write(
                        &mut self.ssbo_snapshot_cache,
                        mappings,
                        outcome,
                    );
                    self.invalidate_resolved_compute_writebacks(mappings);
                    kickprof::add(kickprof::KEPLER, kp);
                    true
                }
                _ => true,
            }
        }
    }

    pub fn set_renderer(&mut self, r: Option<Arc<nexium_gpu::Renderer>>) {
        self.renderer = r;
    }

    pub(crate) fn set_guest_memory_access(&mut self, memory: Option<super::GuestMemoryAccess>) {
        self.guest_memory = memory;
    }

    pub(crate) fn begin_ssbo_snapshot_epoch(&mut self) {
        self.ssbo_snapshot_cache.mirror_begin_kick();
        self.ssbo_snapshot_cache.reset_epoch();
        self.ssbo_snapshot_cache.refresh_input_guest_writes();
    }

    pub(crate) fn end_ssbo_snapshot_epoch(&mut self) {
        let kp = super::pusher::kickprof::start();
        self.ssbo_snapshot_cache.profile_epoch();
        self.ssbo_snapshot_cache.clear_ssbo_snapshots();
        super::pusher::kickprof::add(super::pusher::kickprof::EPOCH_END, kp);
    }

    fn join_small_rt_writeback(&mut self, mappings: &GpuMappings) {
        let Some(rx) = self.pending_small_rt_wb.take() else {
            return;
        };
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(chunks) => {
                if !chunks.is_empty() {
                    super::vk_dispatch::apply_small_rt_writeback_chunks(
                        &mut self.ssbo_snapshot_cache,
                        mappings,
                        &chunks,
                    );
                }
            }
            Err(_) => log::warn!("[rt-writeback] async join timed out"),
        }
    }

    pub(crate) fn writeback_small_rts(
        &mut self,
        renderer: &Arc<nexium_gpu::Renderer>,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> bool {
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
                super::pusher::apply_kepler_memory_write(
                    &mut self.ssbo_snapshot_cache,
                    mappings,
                    outcome,
                );
            }
        }
    }

    pub(crate) fn begin_ssbo_snapshot_entry(&mut self) {
        let watch_started = super::pusher::kickprof::start();
        self.ssbo_snapshot_cache.refresh_ssbo_guest_writes();
        super::pusher::kickprof::add(super::pusher::kickprof::ENQ_WATCH, watch_started);
        let retain_started = super::pusher::kickprof::start();
        self.ssbo_snapshot_cache
            .retain_watchable_full_aurora_snapshots();
        super::pusher::kickprof::add(super::pusher::kickprof::ENTRY_PREP, retain_started);
    }

    pub(crate) fn record_flush_reason(&self, phase: usize) {
        if !self.vk_batch.is_empty() {
            super::pusher::kickprof::count(phase, 1);
        }
    }

    pub(crate) fn has_prepared_draw_packets(&self) -> bool {
        self.prepared_draw_packets.has_pending()
    }

    pub(crate) fn flush_prepared_draw_packets(&mut self) -> bool {
        #[cfg(test)]
        {
            self.prepared_packet_drain_counts.hard += 1;
        }
        self.prepared_draw_packets.drain_hard()
    }

    pub(crate) fn flush_prepared_draw_packets_soft(&mut self) -> bool {
        #[cfg(test)]
        {
            self.prepared_packet_drain_counts.soft += 1;
        }
        self.prepared_draw_packets.drain_soft()
    }

    pub(crate) fn finish_prepared_draw_packet_tail(
        &mut self,
        hard_after: bool,
        writeback_small_rts: bool,
    ) {
        if !hard_after && !writeback_small_rts {
            self.flush_prepared_draw_packets_soft();
        }
    }

    #[cfg(test)]
    pub(crate) fn prepared_packet_drain_counts(&self) -> (usize, usize) {
        (
            self.prepared_packet_drain_counts.hard,
            self.prepared_packet_drain_counts.soft,
        )
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
                self.flush_prepared_draw_packets();
            }
            return;
        }
        let dp = draw_prof_start();
        let batch_len = self.vk_batch.len();
        let kp = super::pusher::kickprof::start();
        if let Some(r) = self.renderer.clone() {
            if hard_after {
                super::vk_dispatch::flush_accum(
                    &mut self.vk_batch,
                    &r,
                    mappings,
                    mem_read,
                    mem_write,
                    &mut self.ssbo_snapshot_cache,
                    &mut self.prepared_draw_packets,
                );
            } else {
                super::vk_dispatch::flush_accum_soft(
                    &mut self.vk_batch,
                    &r,
                    mappings,
                    mem_read,
                    mem_write,
                    &mut self.ssbo_snapshot_cache,
                    &mut self.prepared_draw_packets,
                );
            }
        } else {
            if hard_after {
                self.flush_prepared_draw_packets();
            }
            self.vk_batch.clear();
        }
        super::pusher::kickprof::add(super::pusher::kickprof::FLUSHP, kp);
        draw_prof_record(1, batch_len as u64, dp);
    }

    pub(crate) fn resolve_pending_compute(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let _ = self.resolve_pending_compute_report(mappings, mem_write);
    }

    pub(crate) fn resolve_pending_compute_report(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> Vec<super::engines::maxwell_compute::PendingComputeWritebackSpan> {
        if !super::engines::maxwell_compute::has_pending_writebacks() {
            return Vec::new();
        }
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

    pub(crate) fn invalidate_resolved_compute_writebacks(&mut self, mappings: &GpuMappings) {
        for span in super::engines::maxwell_compute::take_resolved_writeback_spans() {
            self.ssbo_snapshot_cache
                .invalidate_gpu_write(mappings, span.gpu_va, span.len);
        }
    }

    pub(crate) fn sync_renderer_idle(&mut self, reason: &str) -> bool {
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
        let mut completed = renderer.wait_submit_generation(target, Duration::from_secs(3));
        if !completed {
            completed = renderer.wait_idle_checked();
            if !completed {
                log::error!(
                    "[gpu-sync] {} timeline and device-idle waits failed target={}",
                    reason,
                    target
                );
            }
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
                mem_write(cpu, &run_bytes);
                if self.renderer.is_some()
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
            vk_batch: Vec::new(),
            prepared_draw_packets: PreparedDrawPacketizer::default(),
            ssbo_snapshot_cache: SsboSnapshotCache::default(),
            inline_upload: KeplerMemory::new(),
            constbuf_invalidation_scratch: Vec::new(),
            constbuf_patched_scratch: Vec::new(),
            constbuf_bytes_scratch: Vec::new(),
            #[cfg(test)]
            prepared_packet_drain_counts: PreparedPacketDrainCounts::default(),
        }
    }
}

fn fermi_lazy_drain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_FERMI_LAZY_DRAIN").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
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

impl PrepLane {
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

pub(crate) enum PrepEngineAccess<'a, 'b> {
    Direct(&'a mut PrepEngines<'b>),
    Remote,
}

pub(crate) struct PrepThreadHandle {
    tx: crossbeam::channel::Sender<PrepEvent>,
    draw_vec_recycle_rx: Option<crossbeam::channel::Receiver<Vec<DrawCall>>>,
    inflight_kicks: Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
}

const PREP_EVENT_QUEUE_CAPACITY: usize = 4096;
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
        if self.tx.send(event).is_err() {
            log::error!("[gpu-prep] event dropped after prep thread exit");
        }
    }

    pub(crate) fn try_take_recycled_draw_vec(&self) -> Option<Vec<DrawCall>> {
        try_receive_recycled_draw_vec(self.draw_vec_recycle_rx.as_ref())
    }

    pub(crate) fn begin_kick(&self) {
        let (lock, condvar) = &*self.inflight_kicks;
        let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
        while *inflight >= prep_kicks_in_flight() {
            inflight = condvar
                .wait(inflight)
                .unwrap_or_else(|error| error.into_inner());
        }
        *inflight += 1;
    }

    fn finish_kick(inflight_kicks: &(std::sync::Mutex<usize>, std::sync::Condvar)) {
        let (lock, condvar) = inflight_kicks;
        let mut inflight = lock.lock().unwrap_or_else(|error| error.into_inner());
        *inflight = inflight.saturating_sub(1);
        condvar.notify_all();
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
    std::thread::Builder::new()
        .name("nexium-gpu-prep".to_string())
        .spawn(move || {
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
                    PrepEvent::KickBegin => 8,
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
            while let Ok(first) = rx.recv() {
                let burst_started = prof_enabled.then(std::time::Instant::now);
                let mut burst = vec![first];
                while burst.len() < 256 {
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
                for event in burst {
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
                            draw_vec_recycle_tx.as_ref(),
                            &mut engines,
                            &mappings,
                            &resources.stats,
                            &mem_read,
                            &mem_write,
                            &mem_copy,
                        );
                    }));
                    if outcome.is_err() {
                        log::error!("[gpu-prep] event handler panicked; clearing prep caches");
                        state.vk_batch.clear();
                        state.ssbo_snapshot_cache.clear();
                    }
                    if let Some(on_complete) = deferred_completion {
                        on_complete();
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
        })
        .expect("spawn GPU prep thread");
    PrepThreadHandle {
        tx,
        draw_vec_recycle_rx,
        inflight_kicks,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        nonterminal_inline_data_run_end, recycle_processed_draw_vec, try_receive_recycled_draw_vec,
    };
    use crate::gpu::engines::maxwell3d::DrawCall;

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
}
