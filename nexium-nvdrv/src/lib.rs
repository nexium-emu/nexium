use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub(crate) fn syncpoint_reached(current: u32, threshold: u32) -> bool {
    current.wrapping_sub(threshold) < 0x8000_0000
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FenceWaitDisposition {
    Reached,
    OrderedPredecessor,
    Strict,
}

fn classify_submit_fence_wait(
    current: u32,
    ordered_reserved_max: Option<u32>,
    threshold: u32,
) -> FenceWaitDisposition {
    if syncpoint_reached(current, threshold) {
        FenceWaitDisposition::Reached
    } else if ordered_reserved_max.is_some_and(|reserved| syncpoint_reached(reserved, threshold)) {
        FenceWaitDisposition::OrderedPredecessor
    } else {
        FenceWaitDisposition::Strict
    }
}

fn gpfifo_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_GPFIFO_TRACE").is_some())
}

fn video_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_VIDEO_TRACE").is_some())
}

fn trace_video_ioctl(device: NvDevice, req: &IoctlRequest, cmd: u16) {
    if !video_trace_enabled() || !matches!(device, NvDevice::NvhostNvdec | NvDevice::NvhostVic) {
        return;
    }
    static TRACES: AtomicU64 = AtomicU64::new(0);
    let sequence = TRACES.fetch_add(1, Ordering::Relaxed);
    if sequence >= 512 {
        return;
    }
    let in_head = &req.in_data[..req.in_data.len().min(128)];
    let inline_head = &req.inline_in_data[..req.inline_in_data.len().min(128)];
    log::info!(
        "[video-ioctl] seq={} fd={} device={:?} ioctl={:#010x} cmd={:#06x} in_size={} inline_in_size={} out_size={} in_head={:02x?} inline_head={:02x?}",
        sequence,
        req.fd,
        device,
        req.ioctl_id,
        cmd,
        req.in_data.len(),
        req.inline_in_data.len(),
        req.out_size,
        in_head,
        inline_head,
    );
}

#[allow(clippy::too_many_arguments)]
fn trace_gpfifo_submit(
    warning: bool,
    cmd: u16,
    fd: u32,
    flags: u32,
    fence: Option<(u32, u32, u32)>,
    address: Option<u64>,
    num_entries: Option<u32>,
    in_size: usize,
    inline_in_size: usize,
    out_size: usize,
    branch: &str,
) {
    if !gpfifo_trace_enabled() {
        return;
    }
    static TRACES: AtomicU64 = AtomicU64::new(0);
    static WARNINGS: AtomicU64 = AtomicU64::new(0);
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let trace_limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_GPFIFO_TRACE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 1)
            .unwrap_or(256)
    });
    let (counter, limit) = if warning {
        (&WARNINGS, 64)
    } else {
        (&TRACES, trace_limit)
    };
    let sequence = counter.fetch_add(1, Ordering::Relaxed);
    if sequence >= limit {
        return;
    }
    let fence_id = fence
        .map(|value| value.0.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let fence_value = fence
        .map(|value| value.1.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let fence_current = fence
        .map(|value| value.2.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let address = address
        .map(|value| format!("{value:#x}"))
        .unwrap_or_else(|| "n/a".to_string());
    let num_entries = num_entries
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let level = if warning {
        log::Level::Warn
    } else {
        log::Level::Info
    };
    log::log!(
        level,
        "[gpfifo-trace] seq={} cmd={:#06x} fd={} flags={:#x} fence_id={} fence_value={} fence_current={} addr={} num_entries={} in_size={} inline_in_size={} out_size={} branch={}",
        sequence,
        cmd,
        fd,
        flags,
        fence_id,
        fence_value,
        fence_current,
        address,
        num_entries,
        in_size,
        inline_in_size,
        out_size,
        branch,
    );
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let bytes: [u8; 4] = data.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) -> bool {
    let Some(end) = offset.checked_add(4) else {
        return false;
    };
    let Some(bytes) = data.get_mut(offset..end) else {
        return false;
    };
    bytes.copy_from_slice(&value.to_le_bytes());
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChannelSubmitLayout {
    command_buffer_count: u32,
    relocation_count: u32,
    syncpoint_count: u32,
    fence_count: u32,
    command_buffers_offset: usize,
    relocations_offset: usize,
    relocation_shifts_offset: usize,
    syncpoints_offset: usize,
    fences_offset: usize,
    total_size: usize,
}

impl ChannelSubmitLayout {
    fn parse(data: &[u8]) -> Option<Self> {
        let command_buffer_count = read_u32(data, 0)?;
        let relocation_count = read_u32(data, 4)?;
        let syncpoint_count = read_u32(data, 8)?;
        let fence_count = read_u32(data, 12)?;
        let command_buffers_offset = 0x10usize;
        let relocations_offset = command_buffers_offset.checked_add(
            usize::try_from(command_buffer_count)
                .ok()?
                .checked_mul(0x0c)?,
        )?;
        let relocation_shifts_offset = relocations_offset
            .checked_add(usize::try_from(relocation_count).ok()?.checked_mul(0x10)?)?;
        let syncpoints_offset = relocation_shifts_offset
            .checked_add(usize::try_from(relocation_count).ok()?.checked_mul(4)?)?;
        let fences_offset = syncpoints_offset
            .checked_add(usize::try_from(syncpoint_count).ok()?.checked_mul(0x14)?)?;
        let total_size =
            fences_offset.checked_add(usize::try_from(fence_count).ok()?.checked_mul(4)?)?;
        if total_size > data.len() {
            return None;
        }
        Some(Self {
            command_buffer_count,
            relocation_count,
            syncpoint_count,
            fence_count,
            command_buffers_offset,
            relocations_offset,
            relocation_shifts_offset,
            syncpoints_offset,
            fences_offset,
            total_size,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Host1xMethodTrace {
    word_index: usize,
    class_id: u32,
    method: u32,
    argument: u32,
}

fn decode_host1x_methods(
    words: &[u32],
    initial_class: u32,
    max_methods: usize,
) -> Vec<Host1xMethodTrace> {
    let mut traces = Vec::new();
    let mut current_class = initial_class;
    let mut method_offset = 0u32;
    let mut mask = 0u32;
    let mut count = 0u32;
    let mut incrementing = false;

    for (word_index, &raw) in words.iter().enumerate() {
        if traces.len() >= max_methods {
            break;
        }
        let method = if mask != 0 {
            let bit = mask.trailing_zeros();
            mask &= !(1u32 << bit);
            Some(method_offset.wrapping_add(bit))
        } else if count != 0 {
            count -= 1;
            let method = method_offset;
            if incrementing {
                method_offset = method_offset.wrapping_add(1);
            }
            Some(method)
        } else {
            let value = raw & 0xffff;
            method_offset = (raw >> 16) & 0x0fff;
            match raw >> 28 {
                0 => {
                    mask = value & 0x3f;
                    current_class = (value >> 6) & 0x03ff;
                    None
                }
                1 | 2 => {
                    count = value;
                    incrementing = raw >> 28 == 1;
                    None
                }
                3 => {
                    mask = value;
                    None
                }
                4 => {
                    traces.push(Host1xMethodTrace {
                        word_index,
                        class_id: current_class,
                        method: method_offset,
                        argument: value & 0x0fff,
                    });
                    None
                }
                _ => None,
            }
        };
        if let Some(method) = method {
            traces.push(Host1xMethodTrace {
                word_index,
                class_id: current_class,
                method,
                argument: raw,
            });
        }
    }
    traces
}

pub mod bufferqueue;
pub mod gpu;
pub mod render_thread;
pub mod video_decode;
pub mod video_ffmpeg;
pub mod video_host1x;
pub mod video_surface;
pub use bufferqueue::{BufferQueue, GraphicBuffer, QueuedFrame};
pub use gpu::GpuContext;

struct VideoChannelRuntime {
    parser: video_host1x::VideoHost1xParser,
    decoder: Option<openh264::decoder::Decoder>,
    ffmpeg: Option<video_ffmpeg::FfmpegDecoder>,
    ffmpeg_failed: bool,
    composer: video_decode::H264AnnexBComposer,
}

impl VideoChannelRuntime {
    fn new(device: NvDevice) -> Self {
        let initial_class = match device {
            NvDevice::NvhostNvdec => video_host1x::NVDEC_CLASS_ID,
            NvDevice::NvhostVic => video_host1x::VIC_CLASS_ID,
            _ => 0,
        };
        Self {
            parser: video_host1x::VideoHost1xParser::new(initial_class),
            decoder: None,
            ffmpeg: None,
            ffmpeg_failed: false,
            composer: video_decode::H264AnnexBComposer::new(),
        }
    }
}

fn debug_giant_entries(
    cmd: u16,
    num_entries: u32,
    entries: &[gpu::CommandListHeader],
    raw: &[u8],
    raw_len_total: usize,
) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_MME_FORENSICS").is_some()) {
        return;
    }
    let Some(gi) = entries.iter().position(|e| e.entry_count() > 16384) else {
        return;
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return;
    }
    let lo = gi.saturating_sub(2);
    let hi = (gi + 3).min(entries.len());
    let ctx: Vec<String> = entries[lo..hi]
        .iter()
        .enumerate()
        .map(|(k, e)| format!("[{}]gpu={:#x} sz={}", lo + k, e.address(), e.entry_count()))
        .collect();
    let rlo = gi.saturating_sub(1) * 8;
    let rhi = ((gi + 2) * 8).min(raw.len());
    let rawhex: Vec<String> = raw
        .get(rlo..rhi)
        .unwrap_or(&[])
        .chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            format!("{:08x}", u32::from_le_bytes(b))
        })
        .collect();
    log::warn!(
        "[giant-entry] cmd={:#x} num_entries={} entries.len={} raw_total={} giant_idx={} ctx=[{}] raw[{}..{}]=[{}]",
        cmd,
        num_entries,
        entries.len(),
        raw_len_total,
        gi,
        ctx.join(" "),
        rlo,
        rhi,
        rawhex.join(" ")
    );
}

#[derive(Default)]
pub struct PipelineStats {
    pub gpfifo_submits: AtomicU64,
    pub gpfifo_entries: AtomicU64,
    pub methods_dispatched: AtomicU64,
    pub maxwell3d_draws: AtomicU64,
    pub maxwell3d_clears: AtomicU64,
    pub fermi_2d_blits: AtomicU64,
    pub maxwell_dma_blits: AtomicU64,
    pub nvmap_creates: AtomicU64,
    pub nvmap_allocs: AtomicU64,
    pub queue_buffer_calls: AtomicU64,
    pub dequeue_buffer_calls: AtomicU64,
    pub request_buffer_calls: AtomicU64,
    pub vsync_signals: AtomicU64,
    pub frames_submitted: AtomicU64,
    pub frames_drained: AtomicU64,
    pub fence_releases: AtomicU64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct PipelineStatsSnapshot {
    pub gpfifo_submits: u64,
    pub gpfifo_entries: u64,
    pub methods_dispatched: u64,
    pub maxwell3d_draws: u64,
    pub maxwell3d_clears: u64,
    pub fermi_2d_blits: u64,
    pub maxwell_dma_blits: u64,
    pub nvmap_creates: u64,
    pub nvmap_allocs: u64,
    pub queue_buffer_calls: u64,
    pub dequeue_buffer_calls: u64,
    pub request_buffer_calls: u64,
    pub vsync_signals: u64,
    pub frames_submitted: u64,
    pub frames_drained: u64,
    pub fence_releases: u64,
}

impl PipelineStats {
    pub fn snapshot(&self) -> PipelineStatsSnapshot {
        PipelineStatsSnapshot {
            gpfifo_submits: self.gpfifo_submits.load(Ordering::Relaxed),
            gpfifo_entries: self.gpfifo_entries.load(Ordering::Relaxed),
            methods_dispatched: self.methods_dispatched.load(Ordering::Relaxed),
            maxwell3d_draws: self.maxwell3d_draws.load(Ordering::Relaxed),
            maxwell3d_clears: self.maxwell3d_clears.load(Ordering::Relaxed),
            fermi_2d_blits: self.fermi_2d_blits.load(Ordering::Relaxed),
            maxwell_dma_blits: self.maxwell_dma_blits.load(Ordering::Relaxed),
            nvmap_creates: self.nvmap_creates.load(Ordering::Relaxed),
            nvmap_allocs: self.nvmap_allocs.load(Ordering::Relaxed),
            queue_buffer_calls: self.queue_buffer_calls.load(Ordering::Relaxed),
            dequeue_buffer_calls: self.dequeue_buffer_calls.load(Ordering::Relaxed),
            request_buffer_calls: self.request_buffer_calls.load(Ordering::Relaxed),
            vsync_signals: self.vsync_signals.load(Ordering::Relaxed),
            frames_submitted: self.frames_submitted.load(Ordering::Relaxed),
            frames_drained: self.frames_drained.load(Ordering::Relaxed),
            fence_releases: self.fence_releases.load(Ordering::Relaxed),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NvDevice {
    Nvmap,
    NvhostCtrl,
    NvhostCtrlGpu,
    NvhostAsGpu,
    NvhostGpu,
    NvhostDbgGpu,
    NvhostProfGpu,
    NvhostNvdec,
    NvhostNvjpg,
    NvhostVic,
    NvhostNvenc,
    Other,
}

impl NvDevice {
    pub fn from_path(path: &str) -> Option<Self> {
        let trimmed = path.trim_end_matches('\0').trim_end_matches('/');
        Some(match trimmed {
            "/dev/nvmap" => NvDevice::Nvmap,
            "/dev/nvhost-ctrl" => NvDevice::NvhostCtrl,
            "/dev/nvhost-ctrl-gpu" => NvDevice::NvhostCtrlGpu,
            "/dev/nvhost-as-gpu" => NvDevice::NvhostAsGpu,
            "/dev/nvhost-gpu" => NvDevice::NvhostGpu,
            "/dev/nvhost-dbg-gpu" => NvDevice::NvhostDbgGpu,
            "/dev/nvhost-prof-gpu" => NvDevice::NvhostProfGpu,
            "/dev/nvhost-nvdec" => NvDevice::NvhostNvdec,
            "/dev/nvhost-nvjpg" => NvDevice::NvhostNvjpg,
            "/dev/nvhost-vic" => NvDevice::NvhostVic,
            "/dev/nvhost-nvenc" => NvDevice::NvhostNvenc,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct NvFile {
    pub device: NvDevice,
    pub nvmap_fd: Option<u32>,
    pub submit_timeout: u32,
}

pub struct NvmapHandle {
    pub id: u32,
    pub size: u32,
    pub address: u64,
    pub kind: u32,
    pub align: u32,
    pub channel_map_address: u32,
    pub channel_pin_count: u32,
}

pub struct IoctlRequest {
    pub fd: u32,
    pub ioctl_id: u32,
    pub in_data: Vec<u8>,
    pub inline_in_data: Vec<u8>,
    pub out_size: usize,
}

pub struct IoctlOutcome {
    pub result: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct CtrlEventWait {
    pub syncpt_id: u32,
    pub threshold: u32,
}

pub const NVRESULT_NOT_IMPLEMENTED: u32 = 1;

pub type AsyncMemoryRead = Arc<dyn Fn(u64, &mut [u8]) -> bool + Send + Sync>;
pub type AsyncMemoryWrite = Arc<dyn Fn(u64, &[u8]) -> bool + Send + Sync>;
pub type AsyncMemoryCopy = Arc<dyn Fn(u64, u64, usize) -> bool + Send + Sync>;

#[derive(Clone, Copy)]
struct AsyncGpuCompletion {
    fd: u32,
    syncpt_id: u32,
    threshold: u32,
}

enum AsyncGpuSubmission {
    Inline {
        entries: Vec<gpu::CommandListHeader>,
        completion: AsyncGpuCompletion,
    },
    Gpfifo {
        address: u64,
        num_entries: u32,
        completion: AsyncGpuCompletion,
    },
    Present(crate::render_thread::RenderJob),
    Barrier(crossbeam::channel::Sender<()>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncPresentSubmit {
    Enqueued,
    Coalesced,
    Unavailable,
}

struct AsyncGpuQueue {
    gpu: Arc<GpuContext>,
    tx: crossbeam::channel::Sender<AsyncGpuSubmission>,
    completions: Arc<Mutex<Vec<AsyncGpuCompletion>>>,
    pending: Arc<std::sync::atomic::AtomicUsize>,
    capacity: usize,
    profile: Option<AsyncGpuQueueProfile>,
    defer_small_rts: bool,
}

struct AsyncPresentPendingGuard {
    pending: Arc<std::sync::atomic::AtomicUsize>,
}

impl Drop for AsyncPresentPendingGuard {
    fn drop(&mut self) {
        self.pending
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

fn guarded_present_job<F>(
    pending: Arc<std::sync::atomic::AtomicUsize>,
    present: F,
) -> crate::render_thread::RenderJob
where
    F: FnOnce() + Send + 'static,
{
    let pending_guard = AsyncPresentPendingGuard { pending };
    Box::new(move || {
        let _pending_guard = pending_guard;
        present();
    })
}

fn async_present_inflight_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_PRESENT_INFLIGHT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| (1..=3).contains(count))
            .unwrap_or(2)
    })
}

struct AsyncGpuPendingGuard {
    pending: Arc<std::sync::atomic::AtomicUsize>,
}

impl Drop for AsyncGpuPendingGuard {
    fn drop(&mut self) {
        self.pending
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

struct AsyncGpuQueueProfile {
    submissions: AtomicU64,
    full: AtomicU64,
    blocked_ns: AtomicU64,
    max_inflight: std::sync::atomic::AtomicUsize,
    idle_calls: AtomicU64,
    idle_queue_ns: AtomicU64,
    idle_render_ns: AtomicU64,
    idle_barrier_send_ns: AtomicU64,
    idle_barrier_wait_ns: AtomicU64,
    max_idle_queue_ns: AtomicU64,
    max_idle_render_ns: AtomicU64,
}

struct AsyncGpuDrain {
    completed: bool,
    barrier_send_ns: u64,
    barrier_wait_ns: u64,
}

impl AsyncGpuQueueProfile {
    fn new() -> Self {
        Self {
            submissions: AtomicU64::new(0),
            full: AtomicU64::new(0),
            blocked_ns: AtomicU64::new(0),
            max_inflight: std::sync::atomic::AtomicUsize::new(0),
            idle_calls: AtomicU64::new(0),
            idle_queue_ns: AtomicU64::new(0),
            idle_render_ns: AtomicU64::new(0),
            idle_barrier_send_ns: AtomicU64::new(0),
            idle_barrier_wait_ns: AtomicU64::new(0),
            max_idle_queue_ns: AtomicU64::new(0),
            max_idle_render_ns: AtomicU64::new(0),
        }
    }

    fn submitted(&self, was_full: bool, blocked_ns: u64, inflight: usize, capacity: usize) {
        self.max_inflight.fetch_max(inflight, Ordering::Relaxed);
        if was_full {
            self.full.fetch_add(1, Ordering::Relaxed);
            self.blocked_ns.fetch_add(blocked_ns, Ordering::Relaxed);
        }
        let submits = self.submissions.fetch_add(1, Ordering::Relaxed) + 1;
        if submits % 128 != 0 {
            return;
        }
        let full = self.full.swap(0, Ordering::Relaxed);
        let blocked_ns = self.blocked_ns.swap(0, Ordering::Relaxed);
        let max_inflight = self.max_inflight.swap(inflight, Ordering::Relaxed);
        log::warn!(
            "[async-gpu-prof] submits=128 full={} blocked_ms={:.2} max_inflight={} inflight_at_submit={} queue_depth={}",
            full,
            blocked_ns as f64 / 1_000_000.0,
            max_inflight,
            inflight,
            capacity,
        );
    }

    fn idle_wait(&self, queue_ns: u64, render_ns: u64, barrier_send_ns: u64, barrier_wait_ns: u64) {
        self.idle_queue_ns.fetch_add(queue_ns, Ordering::Relaxed);
        self.idle_render_ns.fetch_add(render_ns, Ordering::Relaxed);
        self.idle_barrier_send_ns
            .fetch_add(barrier_send_ns, Ordering::Relaxed);
        self.idle_barrier_wait_ns
            .fetch_add(barrier_wait_ns, Ordering::Relaxed);
        self.max_idle_queue_ns
            .fetch_max(queue_ns, Ordering::Relaxed);
        self.max_idle_render_ns
            .fetch_max(render_ns, Ordering::Relaxed);
        let calls = self.idle_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if calls % 60 != 0 {
            return;
        }
        let queue_ns = self.idle_queue_ns.swap(0, Ordering::Relaxed);
        let render_ns = self.idle_render_ns.swap(0, Ordering::Relaxed);
        let barrier_send_ns = self.idle_barrier_send_ns.swap(0, Ordering::Relaxed);
        let barrier_wait_ns = self.idle_barrier_wait_ns.swap(0, Ordering::Relaxed);
        let max_queue_ns = self.max_idle_queue_ns.swap(0, Ordering::Relaxed);
        let max_render_ns = self.max_idle_render_ns.swap(0, Ordering::Relaxed);
        log::warn!(
            "[async-gpu-prof] idle_calls=60 queue_avg_ms={:.2} queue_max_ms={:.2} render_avg_ms={:.2} render_max_ms={:.2} barrier_send_avg_ms={:.2} barrier_wait_avg_ms={:.2}",
            queue_ns as f64 / 60_000_000.0,
            max_queue_ns as f64 / 1_000_000.0,
            render_ns as f64 / 60_000_000.0,
            max_render_ns as f64 / 1_000_000.0,
            barrier_send_ns as f64 / 60_000_000.0,
            barrier_wait_ns as f64 / 60_000_000.0,
        );
    }
}

impl AsyncGpuQueue {
    fn new(
        gpu: Arc<GpuContext>,
        mem_read: AsyncMemoryRead,
        mem_write: AsyncMemoryWrite,
        mem_copy: AsyncMemoryCopy,
    ) -> Self {
        let capacity = async_gpu_queue_depth();
        if gpu::gpu_pipeline_enabled() {
            gpu.install_prep_thread(gpu::prep::PrepThreadResources {
                maxwell_dma: Arc::clone(&gpu.maxwell_dma),
                fermi_2d: Arc::clone(&gpu.fermi_2d),
                kepler_compute: Arc::clone(&gpu.kepler_compute),
                kepler_memory: Arc::clone(&gpu.kepler_memory),
                mappings: Arc::clone(&gpu.mappings),
                stats: Arc::clone(&gpu.stats),
                mem_read: Arc::clone(&mem_read),
                mem_write: Arc::clone(&mem_write),
                mem_copy: Arc::clone(&mem_copy),
            });
        }
        let (tx, rx) = crossbeam::channel::bounded(capacity);
        let completions = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::clone(&completions);
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pending_worker = Arc::clone(&pending);
        let worker_gpu = Arc::clone(&gpu);
        let profile = async_gpu_queue_profile_enabled().then(AsyncGpuQueueProfile::new);
        let defer_small_rts = matches!(
            std::env::var("NEXIUM_ASYNC_GPU_DEFER_SMALLRT")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        if defer_small_rts {
            log::info!("nexium-nvdrv: async GPU small-RT writeback deferred to queue barriers");
        }
        log::info!("nexium-nvdrv: async GPU queue depth={capacity}");
        std::thread::Builder::new()
            .name("nexium-gpu-submit".to_string())
            .spawn(move || {
                nexium_common::thread_cpu_set::apply_current_thread_cpu_set(
                    nexium_common::thread_cpu_set::ThreadCpuSetTarget::GpuSubmit,
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
                let pipeline = gpu::gpu_pipeline_enabled();
                let make_on_complete =
                    |completion: AsyncGpuCompletion| -> Box<dyn FnOnce() + Send> {
                        let completed = Arc::clone(&completed);
                        let pending = Arc::clone(&pending_worker);
                        Box::new(move || {
                            completed.lock().push(completion);
                            pending.fetch_sub(1, std::sync::atomic::Ordering::Release);
                        })
                    };
                while let Ok(submission) = rx.recv() {
                    match submission {
                        AsyncGpuSubmission::Inline {
                            entries,
                            completion,
                        } => {
                            let on_complete = Some(make_on_complete(completion));
                            if defer_small_rts {
                                worker_gpu.process_inline_gpfifo_soft_deferred(
                                    &entries,
                                    |addr, buf| mem_read(addr, buf),
                                    |addr, buf| mem_write(addr, buf),
                                    |src, dst, len| mem_copy(src, dst, len),
                                    on_complete,
                                );
                            } else {
                                worker_gpu.process_inline_gpfifo_soft(
                                    &entries,
                                    |addr, buf| mem_read(addr, buf),
                                    |addr, buf| mem_write(addr, buf),
                                    |src, dst, len| mem_copy(src, dst, len),
                                    on_complete,
                                );
                            }
                        }
                        AsyncGpuSubmission::Gpfifo {
                            address,
                            num_entries,
                            completion,
                        } => {
                            let on_complete = Some(make_on_complete(completion));
                            if defer_small_rts {
                                worker_gpu.submit_gpfifo_soft_deferred(
                                    address,
                                    num_entries,
                                    |addr, buf| mem_read(addr, buf),
                                    |addr, buf| mem_write(addr, buf),
                                    |src, dst, len| mem_copy(src, dst, len),
                                    on_complete,
                                );
                            } else {
                                worker_gpu.submit_gpfifo_soft(
                                    address,
                                    num_entries,
                                    |addr, buf| mem_read(addr, buf),
                                    |addr, buf| mem_write(addr, buf),
                                    |src, dst, len| mem_copy(src, dst, len),
                                    on_complete,
                                );
                            }
                        }
                        AsyncGpuSubmission::Present(job) => {
                            let pending_guard = AsyncGpuPendingGuard {
                                pending: Arc::clone(&pending_worker),
                            };
                            let job: crate::render_thread::RenderJob = Box::new(move || {
                                let _pending_guard = pending_guard;
                                job();
                            });
                            let job = if pipeline {
                                match worker_gpu.prep_present(job, defer_small_rts) {
                                    Ok(()) => continue,
                                    Err(job) => {
                                        log::error!(
                                            "[gpu-prep] prep lane unavailable; preserving present on render FIFO"
                                        );
                                        job
                                    }
                                }
                            } else {
                                job
                            };
                            let _ = worker_gpu.flush_prepared_draw_packets();
                            if defer_small_rts {
                                let _ = worker_gpu
                                    .flush_small_rt_writebacks(|addr, buf| mem_write(addr, buf));
                            }
                            if let Some(render_thread) = crate::render_thread::maybe_render_thread()
                            {
                                render_thread.submit_named("async-present-readback", job);
                            } else {
                                job();
                            }
                        }
                        AsyncGpuSubmission::Barrier(done) => {
                            if pipeline
                                && worker_gpu.prep_drain_barrier(done.clone(), defer_small_rts)
                            {
                            } else {
                                let _ = worker_gpu.flush_prepared_draw_packets();
                                if defer_small_rts {
                                    let _ = worker_gpu.flush_small_rt_writebacks(|addr, buf| {
                                        mem_write(addr, buf)
                                    });
                                }
                                let _ = done.send(());
                            }
                        }
                    }
                }
            })
            .expect("spawn GPU submit thread");
        Self {
            gpu,
            tx,
            completions,
            pending,
            capacity,
            profile,
            defer_small_rts,
        }
    }

    fn submit(&self, submission: AsyncGpuSubmission) -> bool {
        let inflight = self.pending.fetch_add(1, Ordering::Relaxed) + 1;
        let (queued, was_full, blocked_ns) = if self.profile.is_some() {
            match self.tx.try_send(submission) {
                Ok(()) => (true, false, 0),
                Err(crossbeam::channel::TrySendError::Full(submission)) => {
                    let started = std::time::Instant::now();
                    let queued = self.tx.send(submission).is_ok();
                    (queued, true, started.elapsed().as_nanos() as u64)
                }
                Err(crossbeam::channel::TrySendError::Disconnected(_)) => (false, false, 0),
            }
        } else {
            (self.tx.send(submission).is_ok(), false, 0)
        };
        if let Some(profile) = &self.profile {
            profile.submitted(was_full, blocked_ns, inflight, self.capacity);
        }
        if queued {
            true
        } else {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            false
        }
    }

    fn drain(&self) -> AsyncGpuDrain {
        let pending = self.pending.load(std::sync::atomic::Ordering::Acquire);
        if pending == 0
            && !gpu::gpu_pipeline_enabled()
            && (!self.defer_small_rts || !gpu::vk_dispatch::has_pending_small_rt_writebacks())
            && !self.gpu.has_prepared_draw_packets()
        {
            return AsyncGpuDrain {
                completed: true,
                barrier_send_ns: 0,
                barrier_wait_ns: 0,
            };
        }
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        let send_started = std::time::Instant::now();
        if self.tx.send(AsyncGpuSubmission::Barrier(done_tx)).is_err() {
            return AsyncGpuDrain {
                completed: false,
                barrier_send_ns: send_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                barrier_wait_ns: 0,
            };
        }
        let barrier_send_ns = send_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let wait_started = std::time::Instant::now();
        AsyncGpuDrain {
            completed: done_rx.recv().is_ok(),
            barrier_send_ns,
            barrier_wait_ns: wait_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        }
    }

    fn profile_idle_wait(
        &self,
        queue_ns: u64,
        render_ns: u64,
        barrier_send_ns: u64,
        barrier_wait_ns: u64,
    ) {
        if let Some(profile) = &self.profile {
            profile.idle_wait(queue_ns, render_ns, barrier_send_ns, barrier_wait_ns);
        }
    }
}

fn async_gpu_queue_depth() -> usize {
    std::env::var("NEXIUM_ASYNC_GPU_QUEUE_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|depth| (1..=64).contains(depth))
        .unwrap_or(8)
}

fn async_gpu_queue_profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NEXIUM_ASYNC_GPU_PROFILE").is_some())
}

fn log_unknown_ioctl(device: &str, cmd: u16) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(String, u16)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert((device.to_string(), cmd)) {
        log::warn!(
            "{}: UNHANDLED ioctl cmd={:#x} → returning fake SUCCESS with zeroed output",
            device,
            cmd
        );
    }
}

impl IoctlOutcome {
    pub fn ok(data: Vec<u8>) -> Self {
        Self { result: 0, data }
    }

    pub fn error(result: u32) -> Self {
        Self {
            result,
            data: Vec::new(),
        }
    }
}

pub struct Nvdrv {
    pub files: HashMap<u32, NvFile>,
    pub next_fd: u32,
    pub nvmap_handles: HashMap<u32, NvmapHandle>,
    pub next_nvmap_id: u32,
    pub bufferqueues: Arc<Mutex<HashMap<u32, BufferQueue>>>,
    pub bufferqueue_state_generation: Arc<AtomicU64>,
    pub frame_queue: Arc<Mutex<Vec<QueuedFrame>>>,
    pub next_event_id: u32,
    pub next_syncpoint_id: u32,
    pub retired_syncpts: Arc<Mutex<HashMap<u32, (u32, u32)>>>,
    ordered_submit_max: HashMap<u32, u32>,
    pub next_ctrl_event_slot: u32,
    pub ctrl_event_waits: HashMap<(u32, u32), CtrlEventWait>,
    pub gpu: Arc<GpuContext>,
    pub last_swap_return: Arc<Mutex<Option<std::time::Instant>>>,
    pub queue_buffer_active: Arc<std::sync::atomic::AtomicBool>,
    pub stats: Arc<PipelineStats>,
    pub channel_client_data: u64,
    video_channels: HashMap<u32, VideoChannelRuntime>,
    video_frames: HashMap<u64, video_decode::OwnedI420Frame>,
    video_frame_order: VecDeque<u64>,
    pub legacy_gfx: std::sync::atomic::AtomicBool,
    pub renderer: std::sync::OnceLock<Option<Arc<nexium_gpu::Renderer>>>,
    gpu_async: Option<AsyncGpuQueue>,
    async_present_pending: Arc<std::sync::atomic::AtomicUsize>,
}

impl Nvdrv {
    pub fn new() -> Self {
        let stats = Arc::new(PipelineStats::default());
        let bufferqueue_state_generation = Arc::new(AtomicU64::new(0));
        Self {
            files: HashMap::new(),
            next_fd: 1,
            nvmap_handles: HashMap::new(),
            next_nvmap_id: 1,
            bufferqueues: Arc::new(Mutex::new(HashMap::new())),
            bufferqueue_state_generation,
            frame_queue: Arc::new(Mutex::new(Vec::new())),
            next_event_id: 1,
            next_syncpoint_id: 1,
            retired_syncpts: Arc::new(Mutex::new(HashMap::new())),
            ordered_submit_max: HashMap::new(),
            next_ctrl_event_slot: 0,
            ctrl_event_waits: HashMap::new(),
            gpu: Arc::new(GpuContext::with_stats(stats.clone())),
            last_swap_return: Arc::new(Mutex::new(None)),
            queue_buffer_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats,
            channel_client_data: 0,
            video_channels: HashMap::new(),
            video_frames: HashMap::new(),
            video_frame_order: VecDeque::new(),
            legacy_gfx: std::sync::atomic::AtomicBool::new(false),
            renderer: std::sync::OnceLock::new(),
            gpu_async: None,
            async_present_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn frame_queue_depth(&self) -> usize {
        self.frame_queue.lock().len()
    }

    pub fn renderer(&self) -> Option<&Arc<nexium_gpu::Renderer>> {
        let slot = self
            .renderer
            .get_or_init(|| match nexium_gpu::Renderer::new() {
                Ok(r) => {
                    log::info!("nexium-nvdrv: Vulkan Renderer initialized");
                    self.gpu.pusher.lock().set_renderer(Some(r.clone()));
                    Some(r)
                }
                Err(e) => {
                    log::warn!(
                        "nexium-nvdrv: Vulkan Renderer init failed: {} (falling back to CPU)",
                        e
                    );
                    None
                }
            });
        slot.as_ref()
    }

    pub fn set_guest_memory_writer<F>(&self, writer: F)
    where
        F: Fn(u64, &[u8]) -> bool + Send + Sync + 'static,
    {
        self.gpu.set_guest_memory_writer(Arc::new(writer));
    }

    pub fn set_gpu_async_memory(
        &mut self,
        mem_read: AsyncMemoryRead,
        mem_write: AsyncMemoryWrite,
        mem_copy: AsyncMemoryCopy,
    ) {
        let requested = matches!(
            std::env::var("NEXIUM_ASYNC_GPU").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        if !requested {
            return;
        }
        if !gpu::experimental_gpu_scheduling_enabled() {
            log::warn!(
                "nexium-nvdrv: asynchronous GPU submission quarantined; developer opt-in requires NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1"
            );
            return;
        }
        if self.gpu_async.is_none() {
            log::info!("nexium-nvdrv: async GPU submit thread ENABLED");
            self.gpu_async = Some(AsyncGpuQueue::new(
                Arc::clone(&self.gpu),
                mem_read,
                mem_write,
                mem_copy,
            ));
        }
    }

    pub fn wait_gpu_idle(&self) {
        let mut queue_profile = None;
        if let Some(queue) = &self.gpu_async {
            let queue_started = std::time::Instant::now();
            let drain = queue.drain();
            let queue_ns = queue_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            queue_profile = Some((
                queue,
                queue_ns,
                drain.completed,
                drain.barrier_send_ns,
                drain.barrier_wait_ns,
            ));
            self.poll_gpu_completions();
        }
        let render_started = std::time::Instant::now();
        let _ = gpu::vk_dispatch::sync_render_thread();
        let render_ns = render_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if let Some((queue, ..)) = queue_profile {
            let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while queue.pending.load(Ordering::Acquire) != 0
                && std::time::Instant::now() < completion_deadline
            {
                self.poll_gpu_completions();
                std::thread::yield_now();
            }
            if queue.pending.load(Ordering::Acquire) != 0 {
                log::error!("[gpu-sync] timed out draining renderer-backed GPU completions");
            }
            self.poll_gpu_completions();
        }
        if let Some((queue, queue_ns, queue_completed, barrier_send_ns, barrier_wait_ns)) =
            queue_profile
        {
            queue.profile_idle_wait(
                queue_ns,
                queue_completed.then_some(render_ns).unwrap_or(0),
                barrier_send_ns,
                barrier_wait_ns,
            );
        }
    }

    pub fn try_queue_ordered_present<F>(&self, present: F) -> AsyncPresentSubmit
    where
        F: FnOnce() + Send + 'static,
    {
        let limit = async_present_inflight_limit();
        if self
            .async_present_pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |inflight| {
                (inflight < limit).then_some(inflight + 1)
            })
            .is_err()
        {
            return AsyncPresentSubmit::Coalesced;
        }
        let present = guarded_present_job(Arc::clone(&self.async_present_pending), present);
        let queued = if let Some(queue) = &self.gpu_async {
            queue.submit(AsyncGpuSubmission::Present(present))
        } else if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
            render_thread.submit_timeout_named(
                "ordered-present-readback",
                present,
                std::time::Duration::from_secs(3),
            )
        } else {
            present();
            true
        };
        if queued {
            AsyncPresentSubmit::Enqueued
        } else {
            AsyncPresentSubmit::Unavailable
        }
    }

    fn poll_gpu_completions(&self) {
        let Some(queue) = &self.gpu_async else {
            return;
        };
        let mut completions = queue.completions.lock();
        if completions.is_empty() {
            return;
        }
        let mut orphaned: Vec<(u32, u32)> = Vec::new();
        {
            let mut channels = self.gpu.channels.lock();
            for completion in completions.drain(..) {
                if let Some(channel) = channels.get_mut(&completion.fd) {
                    if channel.syncpt_id == completion.syncpt_id {
                        if !syncpoint_reached(channel.syncpt_min, completion.threshold) {
                            channel.syncpt_min = completion.threshold;
                        }
                    }
                } else {
                    orphaned.push((completion.syncpt_id, completion.threshold));
                }
            }
        }
        if !orphaned.is_empty() {
            let mut retired = self.retired_syncpts.lock();
            for (id, threshold) in orphaned {
                let entry = retired.entry(id).or_insert((0, threshold));
                if !syncpoint_reached(entry.0, threshold) {
                    entry.0 = threshold;
                }
                entry.1 = entry.1.max(threshold);
            }
        }
    }

    pub fn pace_swap(&self, swap_interval: i32) {
        const VSYNC_NS: u64 = 16_666_667;
        let n = swap_interval.clamp(1, 4) as u64;
        let target = std::time::Duration::from_nanos(VSYNC_NS.saturating_mul(n));
        let mut slot = self.last_swap_return.lock();
        if let Some(prev) = *slot {
            let elapsed = prev.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
        *slot = Some(std::time::Instant::now());
    }

    pub fn open(&mut self, path: &str) -> Result<u32, ()> {
        let device = NvDevice::from_path(path).ok_or(())?;
        let fd = self.next_fd;
        self.next_fd = self.next_fd.wrapping_add(1);
        self.files.insert(
            fd,
            NvFile {
                device,
                nvmap_fd: None,
                submit_timeout: 0,
            },
        );
        if matches!(device, NvDevice::NvhostNvdec | NvDevice::NvhostVic) {
            let (syncpt_id, _) = self.ensure_channel_syncpoint(fd);
            self.video_channels
                .insert(fd, VideoChannelRuntime::new(device));
            log::debug!(
                "nvdrv:Open channel device={:?} fd={} syncpt_id={}",
                device,
                fd,
                syncpt_id
            );
        }
        log::debug!("nvdrv:Open '{}' → fd={}", path, fd);
        Ok(fd)
    }

    pub fn close(&mut self, fd: u32) {
        self.files.remove(&fd);
        self.video_channels.remove(&fd);
        if let Some(channel) = self.gpu.channels.lock().remove(&fd) {
            if channel.syncpt_id != 0 {
                self.retired_syncpts
                    .lock()
                    .insert(channel.syncpt_id, (channel.syncpt_min, channel.syncpt_max));
            }
        }
        log::debug!("nvdrv:Close fd={}", fd);
    }

    fn ensure_channel_syncpoint(&mut self, fd: u32) -> (u32, u32) {
        let mut channels = self.gpu.channels.lock();
        let channel = channels.entry(fd).or_default();
        if channel.syncpt_id == 0 {
            channel.syncpt_id = self.next_syncpoint_id;
            self.next_syncpoint_id = self.next_syncpoint_id.wrapping_add(1).max(1);
        }
        (channel.syncpt_id, channel.syncpt_max)
    }

    fn reserve_channel_submit(&mut self, fd: u32, flags: u32, increment_value: u32) -> (u32, u32) {
        let (syncpt_id, _) = self.ensure_channel_syncpoint(fd);
        let mut channels = self.gpu.channels.lock();
        let channel = channels.get_mut(&fd).unwrap();
        let mut increment: u32 = if flags & (1 << 1) != 0 { 2 } else { 0 };
        if flags & (1 << 8) != 0 {
            increment = increment.wrapping_add(increment_value);
        }
        channel.syncpt_max = channel.syncpt_max.wrapping_add(increment);
        let threshold = channel.syncpt_max;
        drop(channels);
        self.ordered_submit_max.insert(syncpt_id, threshold);
        (syncpt_id, threshold)
    }

    fn syncpoint_value(&self, id: u32) -> u32 {
        self.poll_gpu_completions();
        self.gpu
            .channels
            .lock()
            .values()
            .find(|channel| channel.syncpt_id == id)
            .map(|channel| channel.syncpt_min)
            .or_else(|| self.retired_syncpts.lock().get(&id).map(|(min, _)| *min))
            .unwrap_or(0)
    }

    fn syncpoint_max(&self, id: u32) -> u32 {
        self.poll_gpu_completions();
        self.gpu
            .channels
            .lock()
            .values()
            .find(|channel| channel.syncpt_id == id)
            .map(|channel| channel.syncpt_max)
            .or_else(|| self.retired_syncpts.lock().get(&id).map(|(_, max)| *max))
            .unwrap_or(0)
    }

    fn increment_syncpoint(&mut self, id: u32, amount: u32) -> u32 {
        let mut channels = self.gpu.channels.lock();
        let Some(channel) = channels
            .values_mut()
            .find(|channel| channel.syncpt_id == id)
        else {
            drop(channels);
            let mut retired = self.retired_syncpts.lock();
            let e = retired.entry(id).or_insert((0, 0));
            e.1 = e.1.wrapping_add(amount);
            e.0 = e.1;
            return e.0;
        };
        channel.syncpt_max = channel.syncpt_max.wrapping_add(amount);
        channel.syncpt_min = channel.syncpt_max;
        channel.syncpt_min
    }

    fn reserve_syncpoint_max(&mut self, id: u32, amount: u32) -> u32 {
        let mut channels = self.gpu.channels.lock();
        let Some(channel) = channels
            .values_mut()
            .find(|channel| channel.syncpt_id == id)
        else {
            drop(channels);
            let mut retired = self.retired_syncpts.lock();
            let entry = retired.entry(id).or_insert((0, 0));
            entry.1 = entry.1.wrapping_add(amount);
            return entry.1;
        };
        channel.syncpt_max = channel.syncpt_max.wrapping_add(amount);
        channel.syncpt_max
    }

    fn complete_syncpoint_to(&mut self, id: u32, threshold: u32) {
        let mut channels = self.gpu.channels.lock();
        let Some(channel) = channels
            .values_mut()
            .find(|channel| channel.syncpt_id == id)
        else {
            drop(channels);
            let mut retired = self.retired_syncpts.lock();
            let entry = retired.entry(id).or_insert((0, threshold));
            entry.0 = threshold;
            entry.1 = entry.1.max(threshold);
            return;
        };
        channel.syncpt_min = threshold;
    }

    fn channel_submit_completion(
        &self,
        fd: u32,
        syncpt_id: u32,
        threshold: u32,
    ) -> Box<dyn FnOnce() + Send> {
        let gpu = Arc::clone(&self.gpu);
        let retired_syncpts = Arc::clone(&self.retired_syncpts);
        Box::new(move || {
            let mut channels = gpu.channels.lock();
            if let Some(channel) = channels
                .get_mut(&fd)
                .filter(|channel| channel.syncpt_id == syncpt_id)
            {
                if !syncpoint_reached(channel.syncpt_min, threshold) {
                    channel.syncpt_min = threshold;
                }
                drop(channels);
                nexium_common::host_wake::signal();
                return;
            }
            if let Some(channel) = channels.get(&fd) {
                log::warn!(
                    "[gpu-sync] ignored mismatched channel completion fd={} expected_syncpt={} got_syncpt={}",
                    fd,
                    channel.syncpt_id,
                    syncpt_id
                );
            }
            drop(channels);
            let mut retired = retired_syncpts.lock();
            let entry = retired.entry(syncpt_id).or_insert((0, threshold));
            if !syncpoint_reached(entry.0, threshold) {
                entry.0 = threshold;
            }
            entry.1 = entry.1.max(threshold);
            drop(retired);
            nexium_common::host_wake::signal();
        })
    }

    fn wait_for_strict_submit_fence(
        &self,
        syncpt_id: u32,
        threshold: u32,
        timeout: std::time::Duration,
    ) -> bool {
        let started = std::time::Instant::now();
        loop {
            if syncpoint_reached(self.syncpoint_value(syncpt_id), threshold) {
                return true;
            }
            if started.elapsed() >= timeout {
                log::error!(
                    "[gpu-sync] strict submit fence timed out syncpt={} threshold={}; dependent GPFIFO rejected",
                    syncpt_id,
                    threshold
                );
                return false;
            }
            nexium_common::host_wake::micro_pause();
        }
    }

    pub fn is_syncpoint_reached(&self, id: u32, threshold: u32) -> bool {
        self.poll_gpu_completions();
        syncpoint_reached(self.syncpoint_value(id), threshold)
    }

    pub fn queue_buffer_fence_disposition(&self, id: u32, threshold: u32) -> FenceWaitDisposition {
        let disposition = classify_submit_fence_wait(
            self.syncpoint_value(id),
            self.ordered_submit_max.get(&id).copied(),
            threshold,
        );
        Self::queue_buffer_fence_stat(disposition);
        disposition
    }

    fn queue_buffer_fence_stat(disposition: FenceWaitDisposition) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_FENCE_PROFILE").is_some()) {
            return;
        }
        static REACHED: AtomicU64 = AtomicU64::new(0);
        static ORDERED: AtomicU64 = AtomicU64::new(0);
        static STRICT: AtomicU64 = AtomicU64::new(0);
        static TOTAL: AtomicU64 = AtomicU64::new(0);
        match disposition {
            FenceWaitDisposition::Reached => {
                REACHED.fetch_add(1, Ordering::Relaxed);
            }
            FenceWaitDisposition::OrderedPredecessor => {
                ORDERED.fetch_add(1, Ordering::Relaxed);
            }
            FenceWaitDisposition::Strict => {
                STRICT.fetch_add(1, Ordering::Relaxed);
            }
        }
        let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if total % 256 == 0 {
            log::warn!(
                "[queue-fence] total={} reached={} ordered={} strict={}",
                total,
                REACHED.load(Ordering::Relaxed),
                ORDERED.load(Ordering::Relaxed),
                STRICT.load(Ordering::Relaxed),
            );
        }
    }

    pub fn ctrl_event_wait(&self, fd: u32, event_id: u32) -> Option<CtrlEventWait> {
        self.ctrl_event_waits.get(&(fd, event_id & 0xFF)).copied()
    }

    fn fence_wait_stat(deferred: bool) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_FENCE_PROFILE").is_some()) {
            return;
        }
        static INSTANT: AtomicU64 = AtomicU64::new(0);
        static DEFERRED: AtomicU64 = AtomicU64::new(0);
        static TOTAL: AtomicU64 = AtomicU64::new(0);
        if deferred {
            DEFERRED.fetch_add(1, Ordering::Relaxed);
        } else {
            INSTANT.fetch_add(1, Ordering::Relaxed);
        }
        let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if total % 512 == 0 {
            log::warn!(
                "[fence-wait] total={} instant={} deferred={}",
                total,
                INSTANT.load(Ordering::Relaxed),
                DEFERRED.load(Ordering::Relaxed)
            );
        }
    }

    pub fn device_for_fd(&self, fd: u32) -> Option<NvDevice> {
        self.files.get(&fd).map(|f| f.device)
    }

    pub fn dispatch_ioctl(&mut self, req: IoctlRequest) -> IoctlOutcome {
        self.dispatch_ioctl_with_mem(req, &|_, _| false, &|_, _| false)
    }

    pub fn dispatch_ioctl_with_mem(
        &mut self,
        req: IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> IoctlOutcome {
        self.dispatch_ioctl_with_mem_and_copy(req, mem_read, mem_write, &|_, _, _| false)
    }

    pub fn dispatch_ioctl_with_mem_and_copy(
        &mut self,
        req: IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> IoctlOutcome {
        self.poll_gpu_completions();
        let device = match self.files.get(&req.fd) {
            Some(f) => f.device,
            None => {
                log::warn!("nvdrv:Ioctl on invalid fd={}", req.fd);
                return IoctlOutcome::error(0xCE01);
            }
        };
        let cmd = (req.ioctl_id & 0xFFFF) as u16;
        log::trace!(
            "nvdrv:Ioctl fd={} device={:?} ioctl={:#010x} cmd={:#06x} in_size={} inline_in_size={} out_size={}",
            req.fd,
            device,
            req.ioctl_id,
            cmd,
            req.in_data.len(),
            req.inline_in_data.len(),
            req.out_size
        );
        trace_video_ioctl(device, &req, cmd);

        match device {
            NvDevice::Nvmap => self.nvmap_ioctl(cmd, &req),
            NvDevice::NvhostCtrlGpu => self.nvhost_ctrl_gpu_ioctl(cmd, &req),
            NvDevice::NvhostAsGpu => self.nvhost_as_gpu_ioctl(cmd, &req),
            NvDevice::NvhostGpu => {
                self.nvhost_gpu_ioctl_with_mem(cmd, &req, mem_read, mem_write, mem_copy)
            }
            NvDevice::NvhostCtrl => self.nvhost_ctrl_ioctl(cmd, &req),
            NvDevice::NvhostNvdec | NvDevice::NvhostVic => {
                self.nvhost_channel_ioctl_with_mem(device, cmd, &req, mem_read, mem_write)
            }
            _ => IoctlOutcome::ok(vec![0u8; req.out_size]),
        }
    }

    fn pin_channel_buffer(&mut self, handle_id: u32) -> u32 {
        let Some(handle) = self.nvmap_handles.get_mut(&handle_id) else {
            return 0;
        };
        if handle.channel_map_address != 0 {
            handle.channel_pin_count = handle.channel_pin_count.saturating_add(1);
            return handle.channel_map_address;
        }
        let size = handle.size;
        let cpu_address = handle.address;
        if size == 0 || cpu_address == 0 {
            return 0;
        }

        let allocation_size = u64::from(size).max(0x1000);
        let map_address = self.gpu.alloc_gpu_va(allocation_size);
        let Ok(map_address_u32) = u32::try_from(map_address) else {
            if map_address != 0 {
                self.gpu.free_va(map_address, allocation_size);
            }
            return 0;
        };
        if map_address_u32 == 0 {
            return 0;
        }

        self.gpu
            .mappings
            .write()
            .add(map_address, u64::from(size), cpu_address, handle_id);
        handle.channel_map_address = map_address_u32;
        handle.channel_pin_count = 1;
        map_address_u32
    }

    fn unpin_channel_buffer(&mut self, handle_id: u32) {
        if let Some(handle) = self.nvmap_handles.get_mut(&handle_id) {
            handle.channel_pin_count = handle.channel_pin_count.saturating_sub(1);
        }
    }

    fn video_cpu_address(&self, gpu_va: u64) -> Option<u64> {
        let mappings = self.gpu.mappings.read();
        mappings.cpu_address_for(gpu_va).or_else(|| {
            mappings
                .cpu_address_for_any32(gpu_va)
                .map(|(_, cpu, _)| cpu)
        })
    }

    fn gpu_regions_for_cpu_range(&self, cpu_address: u64, size: u64) -> Vec<(u64, u64)> {
        self.gpu
            .mappings
            .read()
            .gpu_regions_for_cpu_range(cpu_address, size)
    }

    fn read_channel_command_buffer(
        &self,
        memory_id: u32,
        offset: u32,
        word_count: i32,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) -> Option<Vec<u32>> {
        const MAX_VIDEO_COMMAND_WORDS: usize = 0x1_0000;
        let handle = self.nvmap_handles.get(&memory_id)?;
        let word_count = usize::try_from(word_count).ok()?;
        if word_count == 0 || word_count > MAX_VIDEO_COMMAND_WORDS {
            log::warn!(
                "video channel rejected command buffer nvmap={} words={}",
                memory_id,
                word_count
            );
            return None;
        }
        let byte_count = word_count.checked_mul(4)?;
        let end = usize::try_from(offset).ok()?.checked_add(byte_count)?;
        if end > usize::try_from(handle.size).ok()? {
            log::warn!(
                "video channel command buffer exceeds nvmap={} offset={:#x} bytes={:#x} size={:#x}",
                memory_id,
                offset,
                byte_count,
                handle.size
            );
            return None;
        }
        let cpu_address = handle.address.checked_add(u64::from(offset))?;
        let mut bytes = vec![0u8; byte_count];
        if !mem_read(cpu_address, &mut bytes) {
            log::warn!(
                "video channel failed to read command buffer nvmap={} cpu={:#x} bytes={:#x}",
                memory_id,
                cpu_address,
                byte_count
            );
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        )
    }

    fn process_video_command_buffer(
        &mut self,
        device: NvDevice,
        fd: u32,
        memory_id: u32,
        offset: u32,
        word_count: i32,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let Some(words) = self.read_channel_command_buffer(memory_id, offset, word_count, mem_read)
        else {
            return;
        };
        let mut runtime = self
            .video_channels
            .remove(&fd)
            .unwrap_or_else(|| VideoChannelRuntime::new(device));

        for word in words {
            let writes = runtime.parser.feed(std::slice::from_ref(&word));
            for write in writes {
                if !write.is_execute() {
                    continue;
                }
                let registers = *runtime.parser.registers(write.engine);
                match write.engine {
                    video_host1x::VideoEngine::Nvdec => {
                        self.process_nvdec_execute(fd, &registers, &mut runtime, mem_read);
                    }
                    video_host1x::VideoEngine::Vic => {
                        self.process_vic_execute(fd, &registers, mem_read, mem_write);
                    }
                }
            }
        }

        self.video_channels.insert(fd, runtime);
    }

    fn process_nvdec_execute(
        &mut self,
        fd: u32,
        registers: &[u32; video_host1x::ENGINE_REGISTER_COUNT],
        runtime: &mut VideoChannelRuntime,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) {
        const CODEC_METHOD: usize = 0x80;
        const PICTURE_INFO_METHOD: usize = 0x101;
        const BITSTREAM_METHOD: usize = 0x102;
        const SURFACE_LUMA_BASE_METHOD: usize = 0x10c;
        const MAX_BITSTREAM_SIZE: usize = 32 * 1024 * 1024;

        if registers[CODEC_METHOD] != 3 {
            log::warn!(
                "[video-decode] fd={} unsupported NVDEC codec {}",
                fd,
                registers[CODEC_METHOD]
            );
            return;
        }

        let context_iova = u64::from(registers[PICTURE_INFO_METHOD]) << 8;
        let bitstream_iova = u64::from(registers[BITSTREAM_METHOD]) << 8;
        let Some(context_cpu) = self.video_cpu_address(context_iova) else {
            log::warn!(
                "[video-decode] fd={} unmapped context iova={:#x}",
                fd,
                context_iova
            );
            return;
        };
        let Some(bitstream_cpu) = self.video_cpu_address(bitstream_iova) else {
            log::warn!(
                "[video-decode] fd={} unmapped bitstream iova={:#x}",
                fd,
                bitstream_iova
            );
            return;
        };

        let mut context_bytes = vec![0u8; video_decode::H264_DECODER_CONTEXT_SIZE];
        if !mem_read(context_cpu, &mut context_bytes) {
            log::warn!(
                "[video-decode] fd={} failed context read cpu={:#x}",
                fd,
                context_cpu
            );
            return;
        }
        let context = match video_decode::H264DecoderContext::parse(&context_bytes) {
            Ok(context) => context,
            Err(error) => {
                log::warn!("[video-decode] fd={} invalid H.264 context: {}", fd, error);
                return;
            }
        };
        let bitstream_len = context.stream_len as usize;
        if bitstream_len == 0 || bitstream_len > MAX_BITSTREAM_SIZE {
            log::warn!(
                "[video-decode] fd={} invalid bitstream size {}",
                fd,
                bitstream_len
            );
            return;
        }
        let mut bitstream = vec![0u8; bitstream_len];
        if !mem_read(bitstream_cpu, &mut bitstream) {
            log::warn!(
                "[video-decode] fd={} failed bitstream read cpu={:#x} bytes={}",
                fd,
                bitstream_cpu,
                bitstream_len
            );
            return;
        }
        let packet = match runtime.composer.compose(&context_bytes, &bitstream) {
            Ok(packet) => packet,
            Err(error) => {
                log::warn!("[video-decode] fd={} compose failed: {}", fd, error);
                return;
            }
        };

        let mut ffmpeg_frame: Option<video_decode::OwnedI420Frame> = None;
        if video_ffmpeg::enabled() && !runtime.ffmpeg_failed {
            if runtime.ffmpeg.is_none() {
                let dims = context
                    .frame_width()
                    .and_then(|width| context.frame_height().map(|height| (width, height)));
                match dims {
                    Ok((width, height)) => match video_ffmpeg::FfmpegDecoder::new(width, height) {
                        Ok(decoder) => {
                            log::info!(
                                "[video-decode] fd={} ffmpeg software decoder {}x{}",
                                fd,
                                width,
                                height
                            );
                            runtime.ffmpeg = Some(decoder);
                        }
                        Err(error) => {
                            log::warn!(
                                "[video-decode] fd={} ffmpeg init failed ({}), using OpenH264",
                                fd,
                                error
                            );
                            runtime.ffmpeg_failed = true;
                        }
                    },
                    Err(error) => {
                        log::warn!(
                            "[video-decode] fd={} ffmpeg dims unavailable ({}), using OpenH264",
                            fd,
                            error
                        );
                        runtime.ffmpeg_failed = true;
                    }
                }
            }
            if let Some(decoder) = runtime.ffmpeg.as_mut() {
                match decoder.decode(&packet) {
                    Ok(Some(raw)) => {
                        match video_ffmpeg::i420_frame(decoder.width(), decoder.height(), &raw) {
                            Ok(frame) => ffmpeg_frame = Some(frame),
                            Err(error) => {
                                log::warn!(
                                    "[video-decode] fd={} ffmpeg frame invalid ({}), using OpenH264",
                                    fd,
                                    error
                                );
                                runtime.ffmpeg = None;
                                runtime.ffmpeg_failed = true;
                            }
                        }
                    }
                    Ok(None) => {
                        log::debug!("[video-decode] fd={} ffmpeg needs more data", fd);
                        return;
                    }
                    Err(error) => {
                        log::warn!(
                            "[video-decode] fd={} ffmpeg decode failed ({}), using OpenH264",
                            fd,
                            error
                        );
                        runtime.ffmpeg = None;
                        runtime.ffmpeg_failed = true;
                    }
                }
            }
        }
        let frame = if let Some(frame) = ffmpeg_frame {
            frame
        } else {
            if runtime.decoder.is_none() {
                match openh264::decoder::Decoder::new(openh264::OpenH264API::from_source()) {
                    Ok(decoder) => runtime.decoder = Some(decoder),
                    Err(error) => {
                        log::warn!("[video-decode] fd={} OpenH264 init failed: {}", fd, error);
                        return;
                    }
                }
            }
            let decoded = match runtime.decoder.as_mut().unwrap().decode(&packet) {
                Ok(Some(decoded)) => decoded,
                Ok(None) => {
                    log::debug!("[video-decode] fd={} decoder needs more data", fd);
                    return;
                }
                Err(error) => {
                    log::warn!("[video-decode] fd={} OpenH264 decode failed: {}", fd, error);
                    return;
                }
            };
            let (width, height) = decoded.dimension_rgb();
            match video_decode::OwnedI420Frame::from_strided_planes(
                width,
                height,
                decoded.strides_yuv(),
                decoded.y_with_stride(),
                decoded.u_with_stride(),
                decoded.v_with_stride(),
            ) {
                Ok(frame) => frame,
                Err(error) => {
                    log::warn!(
                        "[video-decode] fd={} decoded frame copy failed: {}",
                        fd,
                        error
                    );
                    return;
                }
            }
        };
        let (width, height) = (frame.width(), frame.height());

        let picture_index = context.parameter_set.current_picture_index as usize;
        let Some(surface_register) = registers.get(SURFACE_LUMA_BASE_METHOD + picture_index) else {
            log::warn!(
                "[video-decode] fd={} invalid picture index {}",
                fd,
                picture_index
            );
            return;
        };
        let luma_iova = (u64::from(*surface_register) << 8)
            .wrapping_add(u64::from(context.parameter_set.luma_frame_offset));
        self.video_frames.insert(luma_iova, frame);
        self.video_frame_order.retain(|key| *key != luma_iova);
        self.video_frame_order.push_back(luma_iova);
        while self.video_frame_order.len() > 32 {
            if let Some(old_key) = self.video_frame_order.pop_front() {
                self.video_frames.remove(&old_key);
            }
        }

        static DECODED_FRAMES: AtomicU64 = AtomicU64::new(0);
        let frame_index = DECODED_FRAMES.fetch_add(1, Ordering::Relaxed);
        if frame_index < 32 || frame_index % 300 == 0 {
            log::info!(
                "[video-decode] frame={} fd={} {}x{} bytes={} luma_iova={:#x} picture={} guest_frame={}",
                frame_index,
                fd,
                width,
                height,
                bitstream_len,
                luma_iova,
                picture_index,
                context.parameter_set.frame_number
            );
        }
    }

    fn process_vic_execute(
        &mut self,
        fd: u32,
        registers: &[u32; video_host1x::ENGINE_REGISTER_COUNT],
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        const SURFACE_BASE_METHOD: usize = 0x100;
        const SURFACE_REGISTERS_PER_SLOT: usize = 8 * 3;
        const CONFIG_METHOD: usize = 0x1c2;
        const OUTPUT_LUMA_METHOD: usize = 0x1c8;
        const OUTPUT_CHROMA_METHOD: usize = 0x1c9;

        let config_iova = u64::from(registers[CONFIG_METHOD]) << 8;
        let Some(config_cpu) = self.video_cpu_address(config_iova) else {
            log::warn!(
                "[video-vic] fd={} unmapped config iova={:#x}",
                fd,
                config_iova
            );
            return;
        };
        let mut config_bytes = vec![0u8; video_surface::VIC_CONFIG_SIZE];
        if !mem_read(config_cpu, &mut config_bytes) {
            log::warn!(
                "[video-vic] fd={} failed config read cpu={:#x}",
                fd,
                config_cpu
            );
            return;
        }
        let summary = match video_surface::parse_vic_config(&config_bytes) {
            Ok(summary) => summary,
            Err(error) => {
                log::warn!("[video-vic] fd={} invalid config: {}", fd, error);
                return;
            }
        };
        let Some(input_slot) = summary.enabled_slots.first() else {
            log::debug!("[video-vic] fd={} has no enabled input slot", fd);
            return;
        };
        let input_method = SURFACE_BASE_METHOD
            .saturating_add(input_slot.index.saturating_mul(SURFACE_REGISTERS_PER_SLOT));
        let Some(input_luma_register) = registers.get(input_method) else {
            return;
        };
        let input_luma_iova = u64::from(*input_luma_register) << 8;
        if video_trace_enabled() {
            static VIC_CONFIG_TRACES: AtomicU64 = AtomicU64::new(0);
            let sequence = VIC_CONFIG_TRACES.fetch_add(1, Ordering::Relaxed);
            if sequence < 64 {
                log::info!(
                    "[video-vic-config] seq={} fd={} input={:#x} input_format={} input_block={:?}/{} input_size={}x{} output=[{:#x},{:#x}] output_format={} output_block={:?}/{} output_size={}x{} matrix={} shift={} clamp={}..{} alpha={} target={:?} source={:?} dest={:?}",
                    sequence,
                    fd,
                    input_luma_iova,
                    input_slot.pixel_format,
                    input_slot.block_kind,
                    input_slot.block_height_log2,
                    input_slot.surface.width,
                    input_slot.surface.height,
                    u64::from(registers[OUTPUT_LUMA_METHOD]) << 8,
                    u64::from(registers[OUTPUT_CHROMA_METHOD]) << 8,
                    summary.output.pixel_format,
                    summary.output.block_kind,
                    summary.output.block_height_log2,
                    summary.output.surface.width,
                    summary.output.surface.height,
                    input_slot.color_matrix.enabled,
                    input_slot.color_matrix.shift,
                    input_slot.color_matrix.clamp_min,
                    input_slot.color_matrix.clamp_max,
                    input_slot.color_matrix.alpha,
                    summary.target_rect,
                    input_slot.source_rect,
                    input_slot.destination_rect,
                );
            }
        }

        let (frame_key, exact_frame, frame) =
            if let Some(frame) = self.video_frames.get(&input_luma_iova).cloned() {
                (input_luma_iova, true, frame)
            } else {
                let fallback_key = self
                    .video_frame_order
                    .iter()
                    .rev()
                    .find(|key| self.video_frames.contains_key(key))
                    .copied();
                let Some(fallback_key) = fallback_key else {
                    log::warn!(
                        "[video-vic] fd={} no decoded frame for input luma={:#x}",
                        fd,
                        input_luma_iova
                    );
                    return;
                };
                (
                    fallback_key,
                    false,
                    self.video_frames[&fallback_key].clone(),
                )
            };

        let strides = frame.strides();
        let surface_frame = video_surface::I420Frame {
            width: frame.width(),
            height: frame.height(),
            y_stride: strides.0,
            u_stride: strides.1,
            v_stride: strides.2,
            y: frame.y().to_vec(),
            u: frame.u().to_vec(),
            v: frame.v().to_vec(),
        };
        let output_luma_iova = u64::from(registers[OUTPUT_LUMA_METHOD]) << 8;
        let output_chroma_iova = u64::from(registers[OUTPUT_CHROMA_METHOD]) << 8;
        let output_is_nv12 = matches!(
            summary.output.pixel_format,
            video_surface::VIC_FORMAT_Y8_U8V8_420 | video_surface::VIC_FORMAT_Y8_V8U8_420
        );
        let writes = if output_is_nv12 {
            let output_config =
                match summary
                    .output
                    .nv12_config(video_surface::OutputPlaneAddresses {
                        luma: output_luma_iova,
                        chroma: output_chroma_iova,
                    }) {
                    Ok(config) => config,
                    Err(error) => {
                        log::warn!("[video-vic] fd={} unsupported output: {}", fd, error);
                        return;
                    }
                };
            match video_surface::write_i420_to_nv12(surface_frame, output_config) {
                Ok(writes) => vec![writes.luma, writes.chroma],
                Err(error) => {
                    log::warn!("[video-vic] fd={} conversion failed: {}", fd, error);
                    return;
                }
            }
        } else {
            let output_config = match summary
                .output
                .rgba_config(output_luma_iova, input_slot.color_matrix)
            {
                Ok(config) => config,
                Err(error) => {
                    log::warn!("[video-vic] fd={} unsupported output: {}", fd, error);
                    return;
                }
            };
            match video_surface::write_i420_to_rgba(surface_frame, output_config) {
                Ok(write) => vec![write],
                Err(error) => {
                    log::warn!("[video-vic] fd={} conversion failed: {}", fd, error);
                    return;
                }
            }
        };

        let mut mapped_writes = Vec::with_capacity(writes.len());
        for write in writes {
            let Some(cpu_address) = self.video_cpu_address(write.address) else {
                log::warn!(
                    "[video-vic] fd={} unmapped output plane iova={:#x}",
                    fd,
                    write.address
                );
                return;
            };
            let mut aliases = self.gpu_regions_for_cpu_range(cpu_address, write.bytes.len() as u64);
            aliases.push((write.address, write.bytes.len() as u64));
            aliases.sort_unstable();
            aliases.dedup();
            mapped_writes.push((write, cpu_address, aliases));
        }
        for (write, cpu_address, _) in &mapped_writes {
            gpu::vk_dispatch::register_video_tic_cpu_target(*cpu_address, write.bytes.len() as u64);
            if !mem_write(*cpu_address, &write.bytes) {
                log::warn!(
                    "[video-vic] fd={} failed output write iova={:#x} cpu={:#x} bytes={}",
                    fd,
                    write.address,
                    cpu_address,
                    write.bytes.len()
                );
                return;
            }
        }
        for (_, _, aliases) in &mapped_writes {
            for &(alias, size) in aliases {
                nexium_gpu::tex_invalidate::bump_region(alias, size);
            }
        }
        if let Some(renderer) = self.renderer.get().and_then(|renderer| renderer.as_ref()) {
            for (_, _, aliases) in &mapped_writes {
                for &(alias, _) in aliases {
                    renderer.invalidate_texture_address(alias);
                }
            }
        }

        if output_is_nv12 {
            if exact_frame {
                self.video_frames.remove(&frame_key);
                self.video_frame_order.retain(|key| *key != frame_key);
            }
            self.video_frames.insert(output_luma_iova, frame);
            self.video_frame_order
                .retain(|key| *key != output_luma_iova);
            self.video_frame_order.push_back(output_luma_iova);
            while self.video_frame_order.len() > 32 {
                if let Some(old_key) = self.video_frame_order.pop_front() {
                    self.video_frames.remove(&old_key);
                }
            }
        } else if exact_frame {
            self.video_frames.remove(&frame_key);
            self.video_frame_order.retain(|key| *key != frame_key);
        }

        let first_output = mapped_writes[0].0.address;
        let second_output = mapped_writes
            .get(1)
            .map(|write| write.0.address)
            .unwrap_or(0);
        let first_aliases = &mapped_writes[0].2;
        let second_aliases = mapped_writes
            .get(1)
            .map(|write| write.2.as_slice())
            .unwrap_or(&[]);
        static CONVERTED_FRAMES: AtomicU64 = AtomicU64::new(0);
        let frame_index = CONVERTED_FRAMES.fetch_add(1, Ordering::Relaxed);
        if frame_index < 32 || frame_index % 300 == 0 {
            log::info!(
                "[video-vic] frame={} fd={} input={:#x} exact={} output=[{:#x},{:#x}] aliases={:?}/{:?} format={} block={:?}/{} {}x{} target={:?} source={:?} dest={:?}",
                frame_index,
                fd,
                input_luma_iova,
                exact_frame,
                first_output,
                second_output,
                first_aliases,
                second_aliases,
                summary.output.pixel_format,
                summary.output.block_kind,
                summary.output.block_height_log2,
                summary.output.surface.width,
                summary.output.surface.height,
                summary.target_rect,
                input_slot.source_rect,
                input_slot.destination_rect,
            );
        }
    }

    fn trace_channel_command_buffer(
        &self,
        device: NvDevice,
        fd: u32,
        buffer_index: usize,
        memory_id: u32,
        offset: u32,
        word_count: i32,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) {
        if !video_trace_enabled() {
            return;
        }
        static TRACES: AtomicU64 = AtomicU64::new(0);
        let sequence = TRACES.fetch_add(1, Ordering::Relaxed);
        if sequence >= 128 {
            return;
        }
        let Some(handle) = self.nvmap_handles.get(&memory_id) else {
            log::warn!(
                "[video-cmdbuf] seq={} device={:?} fd={} index={} nvmap={} missing",
                sequence,
                device,
                fd,
                buffer_index,
                memory_id
            );
            return;
        };
        if word_count <= 0 {
            log::warn!(
                "[video-cmdbuf] seq={} device={:?} fd={} index={} nvmap={} offset={:#x} invalid_word_count={}",
                sequence,
                device,
                fd,
                buffer_index,
                memory_id,
                offset,
                word_count
            );
            return;
        }
        let requested_words = word_count as usize;
        let captured_words = requested_words.min(256);
        let Some(cpu_address) = handle.address.checked_add(u64::from(offset)) else {
            return;
        };
        let mut bytes = vec![0u8; captured_words.saturating_mul(4)];
        let read_ok = mem_read(cpu_address, &mut bytes);
        let in_bounds = u64::from(offset)
            .checked_add((requested_words as u64).saturating_mul(4))
            .map(|end| end <= u64::from(handle.size))
            .unwrap_or(false);
        if !read_ok {
            log::warn!(
                "[video-cmdbuf] seq={} device={:?} fd={} index={} nvmap={} map={:#x} cpu={:#x} offset={:#x} words={} in_bounds={} read=false",
                sequence,
                device,
                fd,
                buffer_index,
                memory_id,
                handle.channel_map_address,
                cpu_address,
                offset,
                requested_words,
                in_bounds
            );
            return;
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let initial_class = match device {
            NvDevice::NvhostNvdec => 0xf0,
            NvDevice::NvhostVic => 0x5d,
            _ => 0,
        };
        let methods = decode_host1x_methods(&words, initial_class, 512);
        let word_dump = words
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect::<Vec<_>>()
            .join(" ");
        let method_dump = methods
            .iter()
            .map(|trace| {
                format!(
                    "@{}:c={:#x},m={:#x},a={:#x}",
                    trace.word_index, trace.class_id, trace.method, trace.argument
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        log::info!(
            "[video-cmdbuf] seq={} device={:?} fd={} index={} nvmap={} map={:#x} cpu={:#x} offset={:#x} words={} captured={} in_bounds={} raw=[{}] methods=[{}]",
            sequence,
            device,
            fd,
            buffer_index,
            memory_id,
            handle.channel_map_address,
            cpu_address,
            offset,
            requested_words,
            captured_words,
            in_bounds,
            word_dump,
            method_dump
        );
    }

    fn nvhost_channel_ioctl_with_mem(
        &mut self,
        device: NvDevice,
        cmd: u16,
        req: &IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let copy_size = req.in_data.len().min(out.len());
        out[..copy_size].copy_from_slice(&req.in_data[..copy_size]);

        match cmd {
            0x4801 => {
                let nvmap_fd = read_u32(&req.in_data, 0).unwrap_or(0);
                if let Some(file) = self.files.get_mut(&req.fd) {
                    file.nvmap_fd = Some(nvmap_fd);
                }
                log::debug!(
                    "nvhost-channel:SetNVMAPfd device={:?} fd={} nvmap_fd={}",
                    device,
                    req.fd,
                    nvmap_fd
                );
            }
            0x0002 => {
                let (syncpoint_id, _) = self.ensure_channel_syncpoint(req.fd);
                write_u32(&mut out, 4, syncpoint_id);
                log::debug!(
                    "nvhost-channel:GetSyncpoint device={:?} fd={} param={} syncpt_id={}",
                    device,
                    req.fd,
                    read_u32(&req.in_data, 0).unwrap_or(0),
                    syncpoint_id
                );
            }
            0x0003 => {
                write_u32(&mut out, 4, 0);
                log::debug!(
                    "nvhost-channel:GetWaitbase device={:?} fd={} value=0",
                    device,
                    req.fd
                );
            }
            0x0007 if device == NvDevice::NvhostNvdec => {
                let timeout = read_u32(&req.in_data, 0).unwrap_or(0);
                if let Some(file) = self.files.get_mut(&req.fd) {
                    file.submit_timeout = timeout;
                }
                log::debug!(
                    "nvhost-channel:SetSubmitTimeout fd={} timeout={}",
                    req.fd,
                    timeout
                );
            }
            0x0009 => {
                let num_entries = read_u32(&req.in_data, 0).unwrap_or(0) as usize;
                let parsed_entries = req.in_data.len().saturating_sub(0x0c) / 8;
                let mapped_entries = num_entries.min(parsed_entries);
                for index in 0..mapped_entries {
                    let entry_offset = 0x0c + index * 8;
                    let handle_id = read_u32(&req.in_data, entry_offset).unwrap_or(0);
                    let map_address = self.pin_channel_buffer(handle_id);
                    write_u32(&mut out, entry_offset + 4, map_address);
                    if video_trace_enabled() {
                        let cpu_address = self
                            .nvmap_handles
                            .get(&handle_id)
                            .map(|handle| handle.address)
                            .unwrap_or(0);
                        log::info!(
                            "[video-map] device={:?} fd={} index={} nvmap={} map={:#x} cpu={:#x}",
                            device,
                            req.fd,
                            index,
                            handle_id,
                            map_address,
                            cpu_address
                        );
                    }
                }
                log::debug!(
                    "nvhost-channel:MapBuffer device={:?} fd={} requested={} parsed={} mapped={}",
                    device,
                    req.fd,
                    num_entries,
                    parsed_entries,
                    mapped_entries
                );
            }
            0x000a => {
                let num_entries = read_u32(&req.in_data, 0).unwrap_or(0) as usize;
                let parsed_entries = req.in_data.len().saturating_sub(0x0c) / 8;
                let unmapped_entries = num_entries.min(parsed_entries);
                let header_size = out.len().min(0x0c);
                out[..header_size].fill(0);
                for index in 0..unmapped_entries {
                    let entry_offset = 0x0c + index * 8;
                    let handle_id = read_u32(&req.in_data, entry_offset).unwrap_or(0);
                    self.unpin_channel_buffer(handle_id);
                    if let Some(entry) = out.get_mut(entry_offset..entry_offset + 8) {
                        entry.fill(0);
                    }
                }
                log::debug!(
                    "nvhost-channel:UnmapBuffer device={:?} fd={} requested={} parsed={} unmapped={}",
                    device,
                    req.fd,
                    num_entries,
                    parsed_entries,
                    unmapped_entries
                );
            }
            0x0001 => {
                let Some(layout) = ChannelSubmitLayout::parse(&req.in_data) else {
                    log::warn!(
                        "nvhost-channel:Submit device={:?} fd={} malformed_size={}",
                        device,
                        req.fd,
                        req.in_data.len()
                    );
                    return IoctlOutcome::ok(out);
                };
                log::debug!(
                    "nvhost-channel:Submit device={:?} fd={} command_buffers={} relocations={} syncpoints={} fences={} offsets=[cb:{:#x} reloc:{:#x} shifts:{:#x} syncpt:{:#x} fence:{:#x}] total={:#x}",
                    device,
                    req.fd,
                    layout.command_buffer_count,
                    layout.relocation_count,
                    layout.syncpoint_count,
                    layout.fence_count,
                    layout.command_buffers_offset,
                    layout.relocations_offset,
                    layout.relocation_shifts_offset,
                    layout.syncpoints_offset,
                    layout.fences_offset,
                    layout.total_size
                );
                for index in 0..layout.command_buffer_count as usize {
                    let offset = layout.command_buffers_offset + index * 0x0c;
                    let memory_id = read_u32(&req.in_data, offset).unwrap_or(0);
                    let memory_offset = read_u32(&req.in_data, offset + 4).unwrap_or(0);
                    let word_count = read_u32(&req.in_data, offset + 8).unwrap_or(0) as i32;
                    self.trace_channel_command_buffer(
                        device,
                        req.fd,
                        index,
                        memory_id,
                        memory_offset,
                        word_count,
                        mem_read,
                    );
                    self.process_video_command_buffer(
                        device,
                        req.fd,
                        memory_id,
                        memory_offset,
                        word_count,
                        mem_read,
                        mem_write,
                    );
                }
                for index in 0..layout.syncpoint_count as usize {
                    let offset = layout.syncpoints_offset + index * 0x14;
                    let syncpoint_id = read_u32(&req.in_data, offset).unwrap_or(0);
                    let increments = read_u32(&req.in_data, offset + 4).unwrap_or(0);
                    let threshold = self.reserve_syncpoint_max(syncpoint_id, increments);
                    if index < layout.fence_count as usize {
                        write_u32(&mut out, layout.fences_offset + index * 4, threshold);
                    }
                    self.complete_syncpoint_to(syncpoint_id, threshold);
                    log::debug!(
                        "nvhost-channel:Submit syncpt_index={} id={} increments={} threshold={}",
                        index,
                        syncpoint_id,
                        increments,
                        threshold
                    );
                }
            }
            other => {
                out.fill(0);
                log::debug!(
                    "nvhost-channel: unknown ioctl device={:?} cmd={:#x}",
                    device,
                    other
                );
            }
        }

        IoctlOutcome::ok(out)
    }

    fn nvmap_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);
        match cmd {
            0x0101 => {
                let raw_size = if req.in_data.len() >= 4 {
                    u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ])
                } else {
                    0
                };
                let size = (raw_size + 0xFFF) & !0xFFF;
                let id = self.next_nvmap_id;
                self.next_nvmap_id = self.next_nvmap_id.wrapping_add(1);
                self.nvmap_handles.insert(
                    id,
                    NvmapHandle {
                        id,
                        size,
                        address: 0,
                        kind: 0,
                        align: 0,
                        channel_map_address: 0,
                        channel_pin_count: 0,
                    },
                );
                self.stats.nvmap_creates.fetch_add(1, Ordering::Relaxed);
                if out.len() < 8 {
                    out.resize(8, 0);
                }
                out[0..4].copy_from_slice(&size.to_le_bytes());
                out[4..8].copy_from_slice(&id.to_le_bytes());
                log::debug!(
                    "nvmap:Create in_data={:02x?} → size={} id={}",
                    &req.in_data[..req.in_data.len().min(16)],
                    size,
                    id
                );
            }
            0x0103 => {
                if req.in_data.len() >= 4 && out.len() >= 8 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    out[4..8].copy_from_slice(&id.to_le_bytes());
                    log::debug!("nvmap:FromId id={} → handle={}", id, id);
                }
            }
            0x0104 => {
                if req.in_data.len() >= 32 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let address = u64::from_le_bytes([
                        req.in_data[24],
                        req.in_data[25],
                        req.in_data[26],
                        req.in_data[27],
                        req.in_data[28],
                        req.in_data[29],
                        req.in_data[30],
                        req.in_data[31],
                    ]);
                    let align = u32::from_le_bytes([
                        req.in_data[12],
                        req.in_data[13],
                        req.in_data[14],
                        req.in_data[15],
                    ]);
                    if let Some(h) = self.nvmap_handles.get_mut(&id) {
                        h.address = address;
                        h.align = align;
                    }
                    self.stats.nvmap_allocs.fetch_add(1, Ordering::Relaxed);
                    log::debug!(
                        "nvmap:Alloc id={} addr={:#x} align={:#x}",
                        id,
                        address,
                        align
                    );
                }
            }
            0x0105 => {
                if req.in_data.len() >= 4 && out.len() >= 24 {
                    let handle = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let size = self.nvmap_handles.get(&handle).map(|h| h.size).unwrap_or(0);
                    self.nvmap_handles.remove(&handle);
                    out[8..16].copy_from_slice(&0u64.to_le_bytes());
                    out[16..20].copy_from_slice(&size.to_le_bytes());
                    out[20..24].copy_from_slice(&0u32.to_le_bytes());
                    log::debug!("nvmap:Free handle={} size={}", handle, size);
                }
            }
            0x0109 => {
                if req.in_data.len() >= 8 && out.len() >= 12 {
                    let handle = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let param = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let result = match param {
                        1 => self.nvmap_handles.get(&handle).map(|h| h.size).unwrap_or(0),
                        2 => 0x10000,
                        3 => 0,
                        4 => 0x40000000,
                        5 => self
                            .nvmap_handles
                            .get(&handle)
                            .map(|h| h.kind as u32)
                            .unwrap_or(0),
                        _ => 0,
                    };
                    out[8..12].copy_from_slice(&result.to_le_bytes());
                    log::debug!("nvmap:Param handle={} param={} → {}", handle, param, result);
                }
            }
            0x010E => {
                if req.in_data.len() >= 8 && out.len() >= 4 {
                    let handle = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    out[0..4].copy_from_slice(&handle.to_le_bytes());
                    log::debug!("nvmap:GetId handle={} → id={}", handle, handle);
                }
            }
            other => {
                log_unknown_ioctl("nvmap", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_ctrl_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4701 => {
                if out.len() < 4 {
                    out.resize(4, 0);
                }
                out[0..4].copy_from_slice(&1u32.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:ZCullGetCtxSize → 1");
            }
            0x4702 => {
                if out.len() < 40 {
                    out.resize(40, 0);
                }
                let words: [u32; 10] =
                    [0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10];
                for (i, w) in words.iter().enumerate() {
                    out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                log::debug!("nvhost-ctrl-gpu:ZCullGetInfo");
            }
            0x4705 => {
                if out.len() < 0xB0 {
                    out.resize(0xB0, 0);
                }
                out[0..8].copy_from_slice(&0xA0u64.to_le_bytes());
                out[8..16].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
                let gc_off = 16usize;
                let gc: &[(usize, u32)] = &[
                    (0x00, 0x120),
                    (0x04, 0x0b),
                    (0x08, 0xa1),
                    (0x0c, 0x01),
                    (0x10, 0x40000),
                    (0x14, 0x0),
                    (0x18, 0),
                    (0x1c, 0),
                    (0x20, 0x02),
                    (0x24, 0x20),
                    (0x28, 0x20000),
                    (0x2c, 0x20000),
                    (0x30, 0x1b),
                    (0x34, 0x30000),
                    (0x38, 0x01),
                    (0x3c, 0x503),
                    (0x40, 0x503),
                    (0x44, 0x80),
                    (0x48, 0x28),
                    (0x4c, 0x0),
                    (0x50, 0x55),
                    (0x54, 0x0),
                    (0x58, 0x902d),
                    (0x5c, 0xb197),
                    (0x60, 0xb1c0),
                    (0x64, 0xb06f),
                    (0x68, 0xa140),
                    (0x6c, 0xb0b5),
                    (0x70, 0x01),
                    (0x74, 0x0),
                    (0x78, 0x02),
                    (0x7c, 0x01),
                    (0x80, 0x0),
                    (0x84, 0x01),
                    (0x88, 0x21d70),
                    (0x8c, 0x0),
                ];
                for (off, val) in gc {
                    let pos = gc_off + off;
                    out[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
                let chipname: u64 = 0x6230326d67;
                out[gc_off + 0x90..gc_off + 0x98].copy_from_slice(&chipname.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:GetCharacteristics → GM20B");
            }
            0x4703 => {
                log::debug!("nvhost-ctrl-gpu:ZbcSetTable (ack)");
            }
            0x4704 => {
                log::debug!("nvhost-ctrl-gpu:ZbcQueryTable");
            }
            0x4706 => {
                if out.len() < 24 {
                    out.resize(24, 0);
                }
                if req.in_data.len() >= 4 {
                    let mask_buf_size = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    if mask_buf_size != 0 {
                        out[16..20].copy_from_slice(&3u32.to_le_bytes());
                    }
                    out[0..4].copy_from_slice(&mask_buf_size.to_le_bytes());
                }
                log::debug!("nvhost-ctrl-gpu:GetTpcMasks → 3");
            }
            0x4714 => {
                if out.len() >= 8 {
                    out[0..4].copy_from_slice(&0x07u32.to_le_bytes());
                    out[4..8].copy_from_slice(&0x01u32.to_le_bytes());
                }
                self.legacy_gfx.store(true, Ordering::Relaxed);
                log::debug!(
                    "nvhost-ctrl-gpu:GetActiveSlotMask → slot=7 mask=1 (legacy_gfx detected)"
                );
            }
            0x471c => {
                if out.len() < 16 {
                    out.resize(16, 0);
                }
                let ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                out[0..8].copy_from_slice(&ns.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:GetGpuTime → {}ns", ns);
            }
            other => {
                log_unknown_ioctl("nvhost-ctrl-gpu", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_as_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4101 => {
                log::debug!("nvhost-as-gpu:BindChannel");
            }
            0x4102 => {
                let pages = if req.in_data.len() >= 4 {
                    u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ])
                } else {
                    0
                };
                let page_size = if req.in_data.len() >= 8 {
                    u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ])
                } else {
                    0x1000
                };
                let flags = if req.in_data.len() >= 12 {
                    u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ])
                } else {
                    0
                };
                let total_size = (pages as u64) * (page_size as u64);
                let offset_in: u64 = if req.in_data.len() >= 24 {
                    u64::from_le_bytes([
                        req.in_data[16],
                        req.in_data[17],
                        req.in_data[18],
                        req.in_data[19],
                        req.in_data[20],
                        req.in_data[21],
                        req.in_data[22],
                        req.in_data[23],
                    ])
                } else {
                    0
                };
                let alloc = if (flags & 0x1) != 0 && offset_in != 0 {
                    self.gpu.alloc_va_fixed(offset_in, total_size.max(0x1000));
                    offset_in
                } else {
                    self.gpu.alloc_gpu_va_aligned(
                        total_size.max(0x1000),
                        (page_size as u64).max(0x1000),
                    )
                };
                if out.len() >= 24 {
                    out[0..4].copy_from_slice(&pages.to_le_bytes());
                    out[4..8].copy_from_slice(&page_size.to_le_bytes());
                }
                log::debug!(
                    "nvhost-as-gpu:AllocSpace pages={} page_size={:#x} flags={:#x} offset_in={:#x} → gpu_va={:#x}",
                    pages,
                    page_size,
                    flags,
                    offset_in,
                    alloc
                );
                if out.len() >= 24 {
                    out[16..24].copy_from_slice(&alloc.to_le_bytes());
                }
            }
            0x4105 => {
                if req.in_data.len() >= 8 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let removed = self.gpu.mappings.write().remove(gpu_va);
                    if let Some(size) = removed {
                        self.gpu.free_va(gpu_va, size);
                    }
                    log::debug!("nvhost-as-gpu:UnmapBuffer gpu_va={:#x}", gpu_va);
                }
            }
            0x4106 => {
                if req.in_data.len() >= 40 {
                    let flags = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let _kind = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let nvmap_id = u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ]);
                    let _page_size = u32::from_le_bytes([
                        req.in_data[12],
                        req.in_data[13],
                        req.in_data[14],
                        req.in_data[15],
                    ]);
                    let buffer_offset = u64::from_le_bytes([
                        req.in_data[16],
                        req.in_data[17],
                        req.in_data[18],
                        req.in_data[19],
                        req.in_data[20],
                        req.in_data[21],
                        req.in_data[22],
                        req.in_data[23],
                    ]);
                    let mapping_size_in = u64::from_le_bytes([
                        req.in_data[24],
                        req.in_data[25],
                        req.in_data[26],
                        req.in_data[27],
                        req.in_data[28],
                        req.in_data[29],
                        req.in_data[30],
                        req.in_data[31],
                    ]);
                    let requested_offset: u64 = u64::from_le_bytes([
                        req.in_data[32],
                        req.in_data[33],
                        req.in_data[34],
                        req.in_data[35],
                        req.in_data[36],
                        req.in_data[37],
                        req.in_data[38],
                        req.in_data[39],
                    ]);

                    if (flags & 0x100) != 0 {
                        let valid = self
                            .gpu
                            .mappings
                            .read()
                            .mapping_starting_at(requested_offset)
                            .is_some_and(|mapping| mapping.size >= mapping_size_in);
                        if !valid {
                            log::warn!(
                                "nvhost-as-gpu:MapBufferEx remap rejected base={:#x} buffer_offset={:#x} size={:#x}",
                                requested_offset,
                                buffer_offset,
                                mapping_size_in,
                            );
                            return IoctlOutcome::error(0xB);
                        }
                        if out.len() >= 40 {
                            out[32..40].copy_from_slice(&requested_offset.to_le_bytes());
                        }
                        log::debug!(
                            "nvhost-as-gpu:MapBufferEx remap base={:#x} buffer_offset={:#x} size={:#x}",
                            requested_offset,
                            buffer_offset,
                            mapping_size_in,
                        );
                        return IoctlOutcome::ok(out);
                    }

                    let mapping_size = if mapping_size_in == 0 {
                        self.nvmap_handles
                            .get(&nvmap_id)
                            .map(|h| (h.size as u64).saturating_sub(buffer_offset))
                            .unwrap_or(0x1000)
                    } else {
                        mapping_size_in
                    };
                    let handle_cpu = self
                        .nvmap_handles
                        .get(&nvmap_id)
                        .map(|h| h.address.wrapping_add(buffer_offset))
                        .unwrap_or(0);
                    let (gpu_va, cpu_addr, final_nvmap) = if (flags & 0x1) != 0
                        && requested_offset != 0
                    {
                        self.gpu
                            .alloc_va_fixed(requested_offset, mapping_size.max(0x1000));
                        (requested_offset, handle_cpu, nvmap_id)
                    } else if (flags & 0x100) != 0 && requested_offset != 0 {
                        let remap_va = requested_offset.wrapping_add(buffer_offset);
                        self.gpu.alloc_va_fixed(remap_va, mapping_size.max(0x1000));
                        let handle_valid =
                            nvmap_id != 0 && self.nvmap_handles.contains_key(&nvmap_id);
                        let (cpu, nv) = if handle_valid {
                            (handle_cpu, nvmap_id)
                        } else {
                            let m = self.gpu.mappings.read();
                            match m.cpu_address_for(remap_va) {
                                Some(cpu) => (cpu, m.nvmap_id_for(remap_va).unwrap_or(nvmap_id)),
                                None => (handle_cpu, nvmap_id),
                            }
                        };
                        log::debug!(
                                "nvhost-as-gpu:MapBufferEx REMAP offset={:#x} buffer_offset={:#x} → gpu_va={:#x} cpu={:#x} nvmap={} handle_valid={}",
                                requested_offset,
                                buffer_offset,
                                remap_va,
                                cpu,
                                nv,
                                handle_valid
                            );
                        (remap_va, cpu, nv)
                    } else if requested_offset != 0 {
                        self.gpu
                            .alloc_va_fixed(requested_offset, mapping_size.max(0x1000));
                        (requested_offset, handle_cpu, nvmap_id)
                    } else {
                        let big = self
                            .nvmap_handles
                            .get(&nvmap_id)
                            .map(|h| h.align >= 0x10000)
                            .unwrap_or(false);
                        (
                            self.gpu.alloc_va(mapping_size.max(0x1000), big),
                            handle_cpu,
                            nvmap_id,
                        )
                    };
                    let nvmap_id = final_nvmap;
                    log::debug!(
                        "nvhost-as-gpu:MapBufferEx flags={:#x} nvmap_id={} req_off={:#x} cpu_addr={:#x} size={:#x} → gpu_va={:#x}",
                        flags,
                        nvmap_id,
                        requested_offset,
                        cpu_addr,
                        mapping_size,
                        gpu_va
                    );

                    self.gpu
                        .mappings
                        .write()
                        .add(gpu_va, mapping_size, cpu_addr, nvmap_id);

                    if out.len() >= 40 {
                        out[32..40].copy_from_slice(&gpu_va.to_le_bytes());
                    }
                }
            }
            0x4108 => {
                let small_offset: u64 = 0x0400_0000;
                let small_page: u32 = 0x1000;
                let small_pages: u64 = ((1u64 << 34) - small_offset) / small_page as u64;
                let big_offset: u64 = 1u64 << 34;
                let big_page: u32 = 0x10000;
                let big_pages: u64 = ((1u64 << 37) - big_offset) / big_page as u64;
                if out.len() < 64 {
                    out.resize(64, 0);
                }
                out[16..24].copy_from_slice(&small_offset.to_le_bytes());
                out[24..28].copy_from_slice(&small_page.to_le_bytes());
                out[32..40].copy_from_slice(&small_pages.to_le_bytes());
                out[40..48].copy_from_slice(&big_offset.to_le_bytes());
                out[48..52].copy_from_slice(&big_page.to_le_bytes());
                out[56..64].copy_from_slice(&big_pages.to_le_bytes());
                log::debug!(
                    "nvhost-as-gpu:GetVARegions small_pages={} big_pages={}",
                    small_pages,
                    big_pages
                );
            }
            0x4109 => {
                let big_page_size = if req.in_data.len() >= 12 {
                    u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ])
                } else {
                    0
                };
                let va_start = if req.in_data.len() >= 24 {
                    u64::from_le_bytes([
                        req.in_data[16],
                        req.in_data[17],
                        req.in_data[18],
                        req.in_data[19],
                        req.in_data[20],
                        req.in_data[21],
                        req.in_data[22],
                        req.in_data[23],
                    ])
                } else {
                    0
                };
                log::debug!(
                    "nvhost-as-gpu:AllocAsEx big_page_size={:#x} va_start={:#x} in_len={}",
                    big_page_size,
                    va_start,
                    req.in_data.len()
                );
            }
            0x4103 => {
                if req.in_data.len() >= 16 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let pages = u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ]);
                    let page_size = u32::from_le_bytes([
                        req.in_data[12],
                        req.in_data[13],
                        req.in_data[14],
                        req.in_data[15],
                    ]);
                    let size = ((pages as u64) * (page_size as u64)).max(0x1000);
                    self.gpu.free_va(gpu_va, size);
                    log::debug!(
                        "nvhost-as-gpu:FreeSpace gpu_va={:#x} size={:#x}",
                        gpu_va,
                        size
                    );
                }
            }
            0x4114 => {
                let num_entries = req.in_data.len() / 20;
                for i in 0..num_entries {
                    let off = i * 20;
                    if req.in_data.len() < off + 20 {
                        break;
                    }
                    let _flags = u16::from_le_bytes([req.in_data[off], req.in_data[off + 1]]);
                    let _kind = u16::from_le_bytes([req.in_data[off + 2], req.in_data[off + 3]]);
                    let nvmap_handle = u32::from_le_bytes([
                        req.in_data[off + 4],
                        req.in_data[off + 5],
                        req.in_data[off + 6],
                        req.in_data[off + 7],
                    ]);
                    let handle_offset_big_pages = u32::from_le_bytes([
                        req.in_data[off + 8],
                        req.in_data[off + 9],
                        req.in_data[off + 10],
                        req.in_data[off + 11],
                    ]);
                    let as_offset_big_pages = u32::from_le_bytes([
                        req.in_data[off + 12],
                        req.in_data[off + 13],
                        req.in_data[off + 14],
                        req.in_data[off + 15],
                    ]);
                    let big_pages = u32::from_le_bytes([
                        req.in_data[off + 16],
                        req.in_data[off + 17],
                        req.in_data[off + 18],
                        req.in_data[off + 19],
                    ]);
                    let big_page_size: u64 = 0x10000;
                    let gpu_va = (as_offset_big_pages as u64) * big_page_size;
                    let size = (big_pages as u64) * big_page_size;
                    let handle_off = (handle_offset_big_pages as u64) * big_page_size;
                    let cpu_addr = self
                        .nvmap_handles
                        .get(&nvmap_handle)
                        .map(|h| h.address.wrapping_add(handle_off))
                        .unwrap_or(0);
                    log::debug!(
                        "nvhost-as-gpu:Remap[{}/{}] nvmap_id={} cpu={:#x} → gpu_va={:#x} size={:#x}",
                        i,
                        num_entries,
                        nvmap_handle,
                        cpu_addr,
                        gpu_va,
                        size
                    );
                    if cpu_addr != 0 {
                        self.gpu.alloc_va_fixed(gpu_va, size);
                        self.gpu
                            .mappings
                            .write()
                            .add(gpu_va, size, cpu_addr, nvmap_handle);
                    }
                }
            }
            0x4118 => {
                if std::env::var_os("NEXIUM_SYNC_RO_MAP_FAKE_OK").is_some() {
                    log::warn!(
                        "nvhost-as-gpu:GetSyncPointRoMap → fake SUCCESS (base=0) [NEXIUM_SYNC_RO_MAP_FAKE_OK]"
                    );
                } else {
                    log::warn!(
                        "nvhost-as-gpu:GetSyncPointRoMap → NotImplemented (guest must use ioctl syncpt waits)"
                    );
                    return IoctlOutcome::error(NVRESULT_NOT_IMPLEMENTED);
                }
            }
            other => {
                log_unknown_ioctl("nvhost-as-gpu", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_gpu_ioctl_with_mem(
        &mut self,
        cmd: u16,
        req: &IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);
        let submit_flags = req
            .in_data
            .get(12..16)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0);
        let submit_fence_value = req
            .in_data
            .get(20..24)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0);
        let submit_fence_id = req
            .in_data
            .get(16..20)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0);

        match cmd {
            0x4801 => {
                log::debug!("nvhost-gpu:SetNvmapFd");
            }
            0x4803 => {
                log::debug!("nvhost-gpu:ChannelSetTimeout");
            }
            0x4808 | 0x481b => {
                let submit_address = req
                    .in_data
                    .get(0..8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()));
                let submit_num_entries = req
                    .in_data
                    .get(8..12)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
                let trace_fence_current = if gpfifo_trace_enabled() && req.in_data.len() >= 24 {
                    Some(self.syncpoint_value(submit_fence_id))
                } else {
                    None
                };
                let trace_fence = trace_fence_current
                    .map(|current| (submit_fence_id, submit_fence_value, current));
                let trace_submit = |warning, branch| {
                    trace_gpfifo_submit(
                        warning,
                        cmd,
                        req.fd,
                        submit_flags,
                        trace_fence,
                        submit_address,
                        submit_num_entries,
                        req.in_data.len(),
                        req.inline_in_data.len(),
                        req.out_size,
                        branch,
                    );
                };
                let submit_payload_present = submit_address.unwrap_or(0) != 0
                    || submit_num_entries.unwrap_or(0) != 0
                    || !req.inline_in_data.is_empty()
                    || req.in_data.len() > 24
                    || (!req.in_data.is_empty() && req.in_data.len() < 16);
                if submit_flags & 1 != 0 && submit_flags & (1 << 8) != 0 {
                    trace_submit(false, "reject-error4-conflicting-flags");
                    return IoctlOutcome::error(4);
                }
                if submit_flags & 1 != 0 {
                    let current = trace_fence_current
                        .unwrap_or_else(|| self.syncpoint_value(submit_fence_id));
                    match classify_submit_fence_wait(
                        current,
                        self.ordered_submit_max.get(&submit_fence_id).copied(),
                        submit_fence_value,
                    ) {
                        FenceWaitDisposition::Reached => {}
                        FenceWaitDisposition::OrderedPredecessor => {
                            trace_submit(false, "ordered-reserved-fence-wait-elided");
                        }
                        FenceWaitDisposition::Strict => {
                            trace_submit(false, "strict-external-fence-wait");
                            if !self.wait_for_strict_submit_fence(
                                submit_fence_id,
                                submit_fence_value,
                                std::time::Duration::from_secs(3),
                            ) {
                                trace_submit(true, "reject-error5-fence-wait-timeout");
                                return IoctlOutcome::error(5);
                            }
                        }
                    }
                }
                let _ = self.renderer();
                if req.in_data.len() >= 16 {
                    let address = u64::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let num_entries = u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ]);
                    log::trace!(
                        "nvhost-gpu:SubmitGPFIFO addr={:#x} entries={}",
                        address,
                        num_entries
                    );

                    if cmd == 0x481b && req.inline_in_data.len() >= (num_entries as usize) * 8 {
                        trace_submit(false, "inline-481b");
                        let entries: Vec<gpu::CommandListHeader> = (0..num_entries as usize)
                            .map(|i| {
                                let off = i * 8;
                                gpu::CommandListHeader {
                                    address_lo: u32::from_le_bytes([
                                        req.inline_in_data[off],
                                        req.inline_in_data[off + 1],
                                        req.inline_in_data[off + 2],
                                        req.inline_in_data[off + 3],
                                    ]),
                                    address_hi_and_count: u32::from_le_bytes([
                                        req.inline_in_data[off + 4],
                                        req.inline_in_data[off + 5],
                                        req.inline_in_data[off + 6],
                                        req.inline_in_data[off + 7],
                                    ]),
                                }
                            })
                            .collect();
                        debug_giant_entries(
                            cmd,
                            num_entries,
                            &entries,
                            &req.inline_in_data,
                            req.inline_in_data.len(),
                        );
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .gpfifo_entries
                            .fetch_add(entries.len() as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) =
                            self.reserve_channel_submit(req.fd, submit_flags, submit_fence_value);
                        let queued = self.gpu_async.as_ref().is_some_and(|queue| {
                            queue.submit(AsyncGpuSubmission::Inline {
                                entries: entries.clone(),
                                completion: AsyncGpuCompletion {
                                    fd: req.fd,
                                    syncpt_id,
                                    threshold: syncpt_value,
                                },
                            })
                        });
                        if !queued {
                            let on_complete = Some(self.channel_submit_completion(
                                req.fd,
                                syncpt_id,
                                syncpt_value,
                            ));
                            let _ = self.gpu.process_inline_gpfifo(
                                &entries,
                                mem_read,
                                mem_write,
                                mem_copy,
                                on_complete,
                            );
                        }
                        log::trace!(
                            "nvhost-gpu:SubmitGPFIFO (inline) entries={} draws={}",
                            entries.len(),
                            self.gpu.maxwell3d.lock().draw_count()
                        );
                        if out.len() >= 24 {
                            out[12..16].copy_from_slice(&0u32.to_le_bytes());
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else if cmd == 0x4808 && req.in_data.len() > 24 {
                        let available = ((req.in_data.len() - 24) / 8).min(num_entries as usize);
                        if available < num_entries as usize {
                            log::warn!(
                                "nvhost-gpu:SubmitGpfifo rejected truncated payload: num_entries={} available={} in_size={}",
                                num_entries,
                                available,
                                req.in_data.len()
                            );
                            trace_submit(true, "reject-error10-truncated-embedded");
                            return IoctlOutcome::error(0xA);
                        }
                        trace_submit(false, "embedded-4808");
                        let entries: Vec<gpu::CommandListHeader> = (0..num_entries as usize)
                            .map(|i| {
                                let off = 24 + i * 8;
                                gpu::CommandListHeader {
                                    address_lo: u32::from_le_bytes([
                                        req.in_data[off],
                                        req.in_data[off + 1],
                                        req.in_data[off + 2],
                                        req.in_data[off + 3],
                                    ]),
                                    address_hi_and_count: u32::from_le_bytes([
                                        req.in_data[off + 4],
                                        req.in_data[off + 5],
                                        req.in_data[off + 6],
                                        req.in_data[off + 7],
                                    ]),
                                }
                            })
                            .collect();
                        debug_giant_entries(
                            cmd,
                            num_entries,
                            &entries,
                            req.in_data.get(24..).unwrap_or(&[]),
                            req.in_data.len(),
                        );
                        if crate::gpu::pusher::direct_forensics()
                            && entries.iter().any(|e| e.entry_count() > 4096)
                        {
                            use std::sync::atomic::{AtomicU32, Ordering as AO};
                            static N: AtomicU32 = AtomicU32::new(0);
                            if N.fetch_add(1, AO::Relaxed) < 4 {
                                log::warn!(
                                    "[el-raw] in_size={} num_entries={} in_data={:02x?}",
                                    req.in_data.len(),
                                    num_entries,
                                    &req.in_data[..req.in_data.len().min(192)]
                                );
                            }
                        }
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .gpfifo_entries
                            .fetch_add(entries.len() as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) =
                            self.reserve_channel_submit(req.fd, submit_flags, submit_fence_value);
                        let queued = self.gpu_async.as_ref().is_some_and(|queue| {
                            queue.submit(AsyncGpuSubmission::Inline {
                                entries: entries.clone(),
                                completion: AsyncGpuCompletion {
                                    fd: req.fd,
                                    syncpt_id,
                                    threshold: syncpt_value,
                                },
                            })
                        });
                        if !queued {
                            let on_complete = Some(self.channel_submit_completion(
                                req.fd,
                                syncpt_id,
                                syncpt_value,
                            ));
                            let _ = self.gpu.process_inline_gpfifo(
                                &entries,
                                mem_read,
                                mem_write,
                                mem_copy,
                                on_complete,
                            );
                        }
                        if log::log_enabled!(log::Level::Trace) {
                            let (dc, cc) = {
                                let m = self.gpu.maxwell3d.lock();
                                (m.draw_count(), m.clear_count())
                            };
                            log::trace!(
                                "nvhost-gpu:SubmitGPFIFO processed {} entries (draws={}, clears={})",
                                entries.len(),
                                dc,
                                cc
                            );
                        }
                        if out.len() >= 24 {
                            out[12..16].copy_from_slice(&0u32.to_le_bytes());
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else if cmd == 0x481b && address != 0 {
                        trace_submit(false, "kickoff-481b");
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .gpfifo_entries
                            .fetch_add(num_entries as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) =
                            self.reserve_channel_submit(req.fd, submit_flags, submit_fence_value);
                        let queued = self.gpu_async.as_ref().is_some_and(|queue| {
                            queue.submit(AsyncGpuSubmission::Gpfifo {
                                address,
                                num_entries,
                                completion: AsyncGpuCompletion {
                                    fd: req.fd,
                                    syncpt_id,
                                    threshold: syncpt_value,
                                },
                            })
                        });
                        if !queued {
                            let on_complete = Some(self.channel_submit_completion(
                                req.fd,
                                syncpt_id,
                                syncpt_value,
                            ));
                            let _ = self.gpu.submit_gpfifo(
                                address,
                                num_entries,
                                mem_read,
                                mem_write,
                                mem_copy,
                                on_complete,
                            );
                        }
                        log::trace!(
                            "nvhost-gpu:SubmitGPFIFO (kickoff) addr={:#x} entries={} draws={}",
                            address,
                            num_entries,
                            self.gpu.maxwell3d.lock().draw_count()
                        );
                        if out.len() >= 24 {
                            out[12..16].copy_from_slice(&0u32.to_le_bytes());
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else {
                        trace_submit(
                            submit_payload_present,
                            if submit_payload_present {
                                "payload-no-processing"
                            } else {
                                "empty-no-processing"
                            },
                        );
                        let (syncpt_id, syncpt_value) = self.ensure_channel_syncpoint(req.fd);
                        if out.len() >= 24 {
                            out[12..16].copy_from_slice(&0u32.to_le_bytes());
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    }
                } else {
                    trace_submit(
                        submit_payload_present,
                        if submit_payload_present {
                            "short-header-payload-no-processing"
                        } else {
                            "short-header-empty-no-processing"
                        },
                    );
                }
            }
            0x4809 => {
                if req.in_data.len() >= 8 && out.len() >= 16 {
                    let class_num = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    out[8..16].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
                    log::debug!(
                        "nvhost-gpu:AllocObjCtx class={:#x} → obj_id=0xDEADBEEF",
                        class_num
                    );
                }
            }
            0x480b => {
                if req.in_data.len() >= 12 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let mode = u32::from_le_bytes([
                        req.in_data[8],
                        req.in_data[9],
                        req.in_data[10],
                        req.in_data[11],
                    ]);
                    log::debug!("nvhost-gpu:ZCullBind gpu_va={:#x} mode={}", gpu_va, mode);
                }
            }
            0x480c => {
                if req.in_data.len() >= 20 {
                    let enable = u32::from_le_bytes([
                        req.in_data[16],
                        req.in_data[17],
                        req.in_data[18],
                        req.in_data[19],
                    ]);
                    log::debug!("nvhost-gpu:SetErrorNotifier enable={}", enable);
                }
            }
            0x480d => {
                if req.in_data.len() >= 4 {
                    let prio = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    log::debug!("nvhost-gpu:SetPriority prio={:#x}", prio);
                }
            }
            0x4816 => {
                log::debug!("nvhost-gpu:GetErrorInfo");
            }
            0x4817 => {
                if out.len() >= 16 {
                    for b in out[..16].iter_mut() {
                        *b = 0;
                    }
                    out[14] = 0xFF;
                    out[15] = 0xFF;
                }
                log::trace!("nvhost-gpu:GetErrorNotification → status=0xFFFF (no error)");
            }
            0x481a => {
                if req.in_data.len() >= 32 && out.len() >= 32 {
                    let num_entries = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let flags = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let (syncpt_id, syncpt_value) = self.ensure_channel_syncpoint(req.fd);
                    out[12..16].copy_from_slice(&syncpt_id.to_le_bytes());
                    out[16..20].copy_from_slice(&syncpt_value.to_le_bytes());
                    log::info!(
                        "nvhost-gpu:AllocGpfifoEx2 num_entries={} flags={:#x} → fence_id={}",
                        num_entries,
                        flags,
                        syncpt_id
                    );
                }
            }
            0x4714 => {
                if req.in_data.len() >= 8 {
                    let data = u64::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    self.channel_client_data = data;
                    log::debug!("nvhost-gpu:SetClientData data={:#x}", data);
                }
            }
            0x4715 => {
                if out.len() >= 8 {
                    out[0..8].copy_from_slice(&self.channel_client_data.to_le_bytes());
                }
                log::debug!("nvhost-gpu:GetClientData → {:#x}", self.channel_client_data);
            }
            0x481d => {
                log::debug!("nvhost-gpu:ChannelSetTimeslice");
            }
            other => {
                log_unknown_ioctl("nvhost-gpu", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_ctrl_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x0014 => {
                if req.in_data.len() >= 4 && out.len() >= 8 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let value = self.syncpoint_value(id);
                    out[4..8].copy_from_slice(&value.to_le_bytes());
                    log::debug!("nvhost-ctrl:SyncptRead syncpt_id={} → {}", id, value);
                }
            }
            0x0015 => {
                if req.in_data.len() >= 4 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let value = self.increment_syncpoint(id, 1);
                    log::debug!("nvhost-ctrl:SyncptIncr syncpt_id={} → {}", id, value);
                }
            }
            0x0016 => {
                if req.in_data.len() >= 12 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let threshold = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let current = self.syncpoint_value(id);
                    if !syncpoint_reached(current, threshold) {
                        return IoctlOutcome::error(5);
                    }
                }
            }
            0x0019 => {
                if req.in_data.len() >= 12 && out.len() >= 16 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let threshold = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let current = self.syncpoint_value(id);
                    out[12..16].copy_from_slice(&current.to_le_bytes());
                    if !syncpoint_reached(current, threshold) {
                        return IoctlOutcome {
                            result: 5,
                            data: out,
                        };
                    }
                    log::debug!(
                        "nvhost-ctrl:SyncptWaitEx syncpt={} threshold={:#x} current={}",
                        id,
                        threshold,
                        current
                    );
                }
            }
            0x001a => {
                if req.in_data.len() >= 4 && out.len() >= 8 {
                    let id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let value = self.syncpoint_max(id);
                    out[4..8].copy_from_slice(&value.to_le_bytes());
                    log::debug!("nvhost-ctrl:SyncptReadMax syncpt={} → {}", id, value);
                }
            }
            0x001c => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    self.ctrl_event_waits.remove(&(req.fd, event_id & 0xFF));
                    log::debug!("nvhost-ctrl:EventSignal event_id={}", event_id);
                }
            }
            0x001d => {
                if req.in_data.len() >= 16 && out.len() >= 16 {
                    let syncpt_id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let threshold = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let current_val = self.syncpoint_value(syncpt_id);
                    if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
                        log::info!(
                            "[syncpt] event-wait id={} threshold={} current={}",
                            syncpt_id,
                            threshold,
                            current_val
                        );
                    }
                    if syncpoint_reached(current_val, threshold) {
                        Self::fence_wait_stat(false);
                        out[12..16].copy_from_slice(&current_val.to_le_bytes());
                        log::debug!(
                            "nvhost-ctrl:EventWait syncpt={} threshold={:#x} current={} → Success (already reached)",
                            syncpt_id,
                            threshold,
                            current_val
                        );
                    } else {
                        Self::fence_wait_stat(true);
                        let slot = self.next_ctrl_event_slot & 63;
                        self.next_ctrl_event_slot = self.next_ctrl_event_slot.wrapping_add(1);
                        let event_val: u32 = slot | ((syncpt_id & 0xFFF) << 16) | (1 << 28);
                        self.ctrl_event_waits.insert(
                            (req.fd, slot),
                            CtrlEventWait {
                                syncpt_id,
                                threshold,
                            },
                        );
                        out[12..16].copy_from_slice(&event_val.to_le_bytes());
                        log::debug!(
                            "nvhost-ctrl:EventWait syncpt={} threshold={:#x} current={} → Timeout (deferred, slot={}, event_val={:#x})",
                            syncpt_id,
                            threshold,
                            current_val,
                            slot,
                            event_val
                        );
                        return IoctlOutcome {
                            result: 5,
                            data: out,
                        };
                    }
                }
            }
            0x001e => {
                if req.in_data.len() >= 16 {
                    let syncpt_id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let threshold = u32::from_le_bytes([
                        req.in_data[4],
                        req.in_data[5],
                        req.in_data[6],
                        req.in_data[7],
                    ]);
                    let event_id = u32::from_le_bytes([
                        req.in_data[12],
                        req.in_data[13],
                        req.in_data[14],
                        req.in_data[15],
                    ]);
                    let current_val = self.syncpoint_value(syncpt_id);
                    if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
                        log::info!(
                            "[syncpt] event-wait-async event={:#x} id={} threshold={} current={}",
                            event_id,
                            syncpt_id,
                            threshold,
                            current_val
                        );
                    }
                    if syncpoint_reached(current_val, threshold) {
                        Self::fence_wait_stat(false);
                        self.ctrl_event_waits.remove(&(req.fd, event_id & 0xFF));
                        if out.len() >= 16 {
                            out[12..16].copy_from_slice(&current_val.to_le_bytes());
                        }
                        log::debug!(
                            "nvhost-ctrl:EventWaitAsync syncpt={} threshold={:#x} current={} event_id={} → Success",
                            syncpt_id,
                            threshold,
                            current_val,
                            event_id
                        );
                    } else {
                        Self::fence_wait_stat(true);
                        if out.len() >= 16 {
                            out[12..16].copy_from_slice(&event_id.to_le_bytes());
                        }
                        self.ctrl_event_waits.insert(
                            (req.fd, event_id & 0xFF),
                            CtrlEventWait {
                                syncpt_id,
                                threshold,
                            },
                        );
                        log::debug!(
                            "nvhost-ctrl:EventWaitAsync syncpt={} threshold={:#x} current={} event_id={} → Timeout",
                            syncpt_id,
                            threshold,
                            current_val,
                            event_id
                        );
                        return IoctlOutcome {
                            result: 5,
                            data: out,
                        };
                    }
                }
            }
            0x001f => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    log::debug!("nvhost-ctrl:EventRegister event_id={}", event_id);
                }
            }
            0x0020 => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([
                        req.in_data[0],
                        req.in_data[1],
                        req.in_data[2],
                        req.in_data[3],
                    ]);
                    let slot = event_id & 0xFF;
                    self.ctrl_event_waits
                        .retain(|(fd, id), _| *fd != req.fd || *id != slot);
                    log::debug!("nvhost-ctrl:EventUnregister event_id={}", event_id);
                }
            }
            0x001b => {
                let cstr = |b: &[u8]| -> String {
                    String::from_utf8_lossy(b.split(|&c| c == 0).next().unwrap_or(&[])).into_owned()
                };
                let domain = cstr(req.in_data.get(0..0x41).unwrap_or(&[]));
                let param = cstr(req.in_data.get(0x41..0x82).unwrap_or(&[]));
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 12 {
                    log::info!(
                        "nvhost-ctrl:NvOsGetConfigU32 domain='{}' param='{}' → ConfigVarNotFound",
                        domain,
                        param
                    );
                }
                return IoctlOutcome::error(0x0003_0006);
            }
            other => {
                log_unknown_ioctl("nvhost-ctrl", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    pub fn with_bufferqueue<R>(&self, binder_id: u32, f: impl FnOnce(&mut BufferQueue) -> R) -> R {
        let mut bqs = self.bufferqueues.lock();
        let generation = Arc::clone(&self.bufferqueue_state_generation);
        let bq = bqs
            .entry(binder_id)
            .or_insert_with(|| BufferQueue::with_generation(binder_id, generation));
        f(bq)
    }

    pub fn bufferqueue_state_generation(&self) -> u64 {
        self.bufferqueue_state_generation.load(Ordering::Acquire)
    }

    pub fn drain_frames(&self) -> Vec<QueuedFrame> {
        let frames = std::mem::take(&mut *self.frame_queue.lock());
        self.stats
            .frames_drained
            .fetch_add(frames.len() as u64, Ordering::Relaxed);
        frames
    }

    pub fn drain_latest_frame(&self) -> Option<QueuedFrame> {
        let mut queue = self.frame_queue.lock();
        let len = queue.len();
        let latest = queue.pop();
        queue.clear();
        self.stats
            .frames_drained
            .fetch_add(len as u64, Ordering::Relaxed);
        latest
    }

    pub fn submit_frame(&self, frame: QueuedFrame) {
        self.queue_buffer_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut queue = self.frame_queue.lock();
        queue.clear();
        queue.push(frame);
        self.stats.frames_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn capture_gpu_frame(
        &self,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
    ) -> Option<QueuedFrame> {
        self.wait_gpu_idle();
        let maxwell = self.gpu.maxwell3d.lock();
        let (w, h) = maxwell.primary_rt_size()?;
        let gpu_va = maxwell.primary_rt_gpu_va()?;
        drop(maxwell);

        let mappings = self.gpu.mappings.read();
        let cpu = mappings.cpu_address_for(gpu_va)?;
        drop(mappings);

        let size = (w as usize) * (h as usize) * 4;
        let mut pixels = vec![0u8; size];
        if mem_read(cpu, &mut pixels) {
            Some(QueuedFrame {
                width: w,
                height: h,
                pixels,
            })
        } else {
            None
        }
    }

    pub fn try_capture_sdl_surface(
        &self,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
    ) -> Option<QueuedFrame> {
        const CANDIDATES: &[(u32, u32, u32)] = &[
            (1280, 720, 1280),
            (1280, 768, 1280),
            (1920, 1080, 1920),
            (640, 360, 640),
            (854, 480, 854),
            (427, 240, 427),
            (480, 270, 480),
        ];

        let mut best: Option<(u32, u32, u32, u64, u32, Vec<u8>)> = None;
        for handle in self.nvmap_handles.values() {
            if handle.address == 0 || handle.size == 0 {
                continue;
            }
            let Some(&(w, h, stride)) = CANDIDATES
                .iter()
                .find(|(_w, hh, stride)| (*stride as u32) * (*hh as u32) * 4 == handle.size as u32)
            else {
                continue;
            };

            let mut linear = vec![0u8; handle.size as usize];
            if !mem_read(handle.address, &mut linear) {
                continue;
            }

            let nz = linear.iter().filter(|b| **b != 0).count();
            if nz < 256 {
                continue;
            }

            for px in linear.chunks_exact_mut(4) {
                px[3] = 0xFF;
            }

            if best.as_ref().map(|b| nz > b.4 as usize).unwrap_or(true) {
                best = Some((w, h, stride, handle.address, nz as u32, linear));
            }
        }

        let (w, h, stride, addr, nz, linear) = best?;
        log::debug!(
            "captured SDL surface addr={:#x} {}x{} stride={} (nz={})",
            addr,
            w,
            h,
            stride,
            nz
        );

        let dst_w = 1280u32;
        let dst_h = 720u32;
        let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
        for dy in 0..dst_h {
            let sy = dy * h / dst_h;
            for dx in 0..dst_w {
                let sx = dx * w / dst_w;
                let s = ((sy * stride + sx) * 4) as usize;
                let d = ((dy * dst_w + dx) * 4) as usize;
                if s + 4 <= linear.len() {
                    out[d..d + 4].copy_from_slice(&linear[s..s + 4]);
                }
            }
        }
        Some(QueuedFrame {
            width: dst_w,
            height: dst_h,
            pixels: out,
        })
    }

    pub fn gpu_draw_count(&self) -> u64 {
        self.wait_gpu_idle();
        self.gpu.maxwell3d.lock().draw_count()
    }

    pub fn last_clear_color(&self) -> [f32; 4] {
        self.wait_gpu_idle();
        let m = self.gpu.maxwell3d.lock();
        let c = m.regs.clear_color;
        [c.r, c.g, c.b, c.a]
    }

    pub fn last_clear_count(&self) -> u64 {
        self.wait_gpu_idle();
        self.gpu.maxwell3d.lock().clear_count()
    }

    pub fn drain_fermi2d_frame(&self) -> Option<QueuedFrame> {
        self.wait_gpu_idle();
        let f2d = self.gpu.fermi_2d.lock();
        let mut q = f2d.captured_frames.lock().unwrap();
        q.pop()
    }
}

impl Default for Nvdrv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_fence_wait_only_elides_an_ordered_reserved_predecessor() {
        assert_eq!(
            classify_submit_fence_wait(7, Some(9), 7),
            FenceWaitDisposition::Reached
        );
        assert_eq!(
            classify_submit_fence_wait(7, Some(9), 9),
            FenceWaitDisposition::OrderedPredecessor
        );
        assert_eq!(
            classify_submit_fence_wait(7, Some(8), 9),
            FenceWaitDisposition::Strict
        );
        assert_eq!(
            classify_submit_fence_wait(7, None, 9),
            FenceWaitDisposition::Strict
        );

        assert_eq!(
            classify_submit_fence_wait(u32::MAX - 1, Some(1), 1),
            FenceWaitDisposition::OrderedPredecessor
        );
    }

    #[test]
    fn only_gpfifo_reservations_are_recorded_as_ordered() {
        let mut nvdrv = Nvdrv::new();
        let gpu_fd = nvdrv.open("/dev/nvhost-gpu").unwrap();
        let (gpu_syncpt, threshold) = nvdrv.reserve_channel_submit(gpu_fd, 1 << 1, 0);
        assert_eq!(threshold, 2);
        assert_eq!(nvdrv.ordered_submit_max.get(&gpu_syncpt), Some(&threshold));
        assert_eq!(
            nvdrv.queue_buffer_fence_disposition(gpu_syncpt, threshold),
            FenceWaitDisposition::OrderedPredecessor
        );

        let video_fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        let video_syncpt = nvdrv.ensure_channel_syncpoint(video_fd).0;
        let video_threshold = nvdrv.reserve_syncpoint_max(video_syncpt, 1);
        assert_eq!(video_threshold, 1);
        assert!(!nvdrv.ordered_submit_max.contains_key(&video_syncpt));
        assert_eq!(
            nvdrv.queue_buffer_fence_disposition(video_syncpt, video_threshold),
            FenceWaitDisposition::Strict
        );

        nvdrv
            .gpu
            .channels
            .lock()
            .get_mut(&gpu_fd)
            .unwrap()
            .syncpt_min = threshold;
        assert_eq!(
            nvdrv.queue_buffer_fence_disposition(gpu_syncpt, threshold),
            FenceWaitDisposition::Reached
        );
    }

    #[test]
    fn strict_submit_fence_timeout_rejects_without_vulkan_work() {
        let nvdrv = Nvdrv::new();
        assert!(!nvdrv.wait_for_strict_submit_fence(0xfeed, 1, std::time::Duration::ZERO));
    }

    #[test]
    fn ordered_present_guard_releases_inflight_once_on_run_or_rejection() {
        let rejected_pending = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let rejected = guarded_present_job(Arc::clone(&rejected_pending), || {});
        drop(rejected);
        assert_eq!(rejected_pending.load(Ordering::Acquire), 0);

        let executed_pending = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let executed = guarded_present_job(Arc::clone(&executed_pending), || {});
        executed();
        assert_eq!(executed_pending.load(Ordering::Acquire), 0);
    }

    fn request(fd: u32, ioctl_id: u32, in_data: Vec<u8>, out_size: usize) -> IoctlRequest {
        IoctlRequest {
            fd,
            ioctl_id,
            in_data,
            inline_in_data: Vec::new(),
            out_size,
        }
    }

    fn test_nvmap_handle(id: u32, size: u32, address: u64) -> NvmapHandle {
        NvmapHandle {
            id,
            size,
            address,
            kind: 0,
            align: 0x1000,
            channel_map_address: 0,
            channel_pin_count: 0,
        }
    }

    #[test]
    fn video_channels_get_distinct_nonzero_syncpoints() {
        let mut nvdrv = Nvdrv::new();
        let nvdec_fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        let vic_fd = nvdrv.open("/dev/nvhost-vic").unwrap();

        let mut nvdec_input = vec![0u8; 8];
        write_u32(&mut nvdec_input, 0, 7);
        let nvdec = nvdrv.dispatch_ioctl(request(nvdec_fd, 0xc008_0002, nvdec_input.clone(), 8));
        let vic = nvdrv.dispatch_ioctl(request(vic_fd, 0xc008_0002, vec![0u8; 8], 8));
        let nvdec_syncpoint = read_u32(&nvdec.data, 4).unwrap();
        let vic_syncpoint = read_u32(&vic.data, 4).unwrap();

        assert_eq!(nvdec.result, 0);
        assert_eq!(vic.result, 0);
        assert_eq!(read_u32(&nvdec.data, 0), Some(7));
        assert_ne!(nvdec_syncpoint, 0);
        assert_ne!(vic_syncpoint, 0);
        assert_ne!(nvdec_syncpoint, vic_syncpoint);

        let again = nvdrv.dispatch_ioctl(request(nvdec_fd, 0xc008_0002, nvdec_input, 8));
        assert_eq!(read_u32(&again.data, 4), Some(nvdec_syncpoint));
    }

    #[test]
    fn map_buffer_ex_remap_reuses_exact_base_without_new_mapping() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x5_04d3_0000u64;
        let cpu = 0x4a_0200_0000u64;
        nvdrv.gpu.mappings.write().add(base, 0x400000, cpu, 77);

        let mut input = vec![0u8; 40];
        input[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        input[16..24].copy_from_slice(&0x2f0000u64.to_le_bytes());
        input[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        input[32..40].copy_from_slice(&base.to_le_bytes());
        let before = nvdrv.gpu.mappings.read().iter().count();

        let first = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input.clone(), 40));
        let second = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input, 40));

        assert_eq!(first.result, 0);
        assert_eq!(second.result, 0);
        assert_eq!(
            u64::from_le_bytes(first.data[32..40].try_into().unwrap()),
            base
        );
        let mappings = nvdrv.gpu.mappings.read();
        assert_eq!(mappings.iter().count(), before);
        assert_eq!(
            mappings.cpu_address_for(base + 0x2f0000),
            Some(cpu + 0x2f0000)
        );
        assert!(mappings.iter().all(|mapping| mapping.nvmap_id != 0));
    }

    #[test]
    fn map_buffer_ex_remap_rejects_missing_or_oversized_base() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x5_04d3_0000u64;
        nvdrv
            .gpu
            .mappings
            .write()
            .add(base, 0x10000, 0x4a_0200_0000, 77);

        let remap = |offset: u64, size: u64| {
            let mut input = vec![0u8; 40];
            input[0..4].copy_from_slice(&0x100u32.to_le_bytes());
            input[24..32].copy_from_slice(&size.to_le_bytes());
            input[32..40].copy_from_slice(&offset.to_le_bytes());
            input
        };

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, remap(base + 0x1000, 0x1000), 40))
                .result,
            0xB
        );
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, remap(base, 0x20000), 40))
                .result,
            0xB
        );
    }

    #[test]
    fn gpu_submit_rejects_truncated_embedded_gpfifo_entries() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-gpu").unwrap();
        let mut input = vec![0u8; 24 + 8];
        write_u32(&mut input, 8, 2);
        write_u32(&mut input, 24, 0x1000);
        write_u32(&mut input, 28, 1 << 10);

        let submitted = nvdrv.dispatch_ioctl(request(fd, 0xc020_4808, input, 24));

        assert_eq!(submitted.result, 0xA);
        assert_eq!(nvdrv.stats.gpfifo_submits.load(Ordering::Relaxed), 0);
        assert_eq!(nvdrv.stats.gpfifo_entries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn video_channel_stores_nvmap_fd_and_submit_timeout() {
        let mut nvdrv = Nvdrv::new();
        let nvdec_fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        let nvmap_fd = nvdrv.open("/dev/nvmap").unwrap();

        let nvmap = nvdrv.dispatch_ioctl(request(
            nvdec_fd,
            0x4004_4801,
            nvmap_fd.to_le_bytes().to_vec(),
            4,
        ));
        let timeout = nvdrv.dispatch_ioctl(request(
            nvdec_fd,
            0x4004_0007,
            1000u32.to_le_bytes().to_vec(),
            4,
        ));

        assert_eq!(nvmap.result, 0);
        assert_eq!(timeout.result, 0);
        assert_eq!(nvdrv.files[&nvdec_fd].nvmap_fd, Some(nvmap_fd));
        assert_eq!(nvdrv.files[&nvdec_fd].submit_timeout, 1000);
    }

    #[test]
    fn video_channel_map_and_unmap_match_variable_wire_layout() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        let cpu_address = 0x4a07_f000_00u64;
        nvdrv
            .nvmap_handles
            .insert(497, test_nvmap_handle(497, 0x8000, cpu_address));

        let mut input = vec![0u8; 0x0c + 2 * 8];
        write_u32(&mut input, 0, 1);
        write_u32(&mut input, 4, 0x1122_3344);
        write_u32(&mut input, 8, 0x5566_7788);
        write_u32(&mut input, 0x0c, 497);
        write_u32(&mut input, 0x10, 0xffff_ffff);
        write_u32(&mut input, 0x14, 999);
        write_u32(&mut input, 0x18, 0xaabb_ccdd);

        let mapped = nvdrv.dispatch_ioctl(request(fd, 0xc01c_0009, input.clone(), input.len()));
        let map_address = read_u32(&mapped.data, 0x10).unwrap();
        assert_ne!(map_address, 0);
        assert_eq!(&mapped.data[..0x10], &input[..0x10]);
        assert_eq!(&mapped.data[0x14..], &input[0x14..]);
        assert_eq!(
            nvdrv
                .gpu
                .mappings
                .read()
                .cpu_address_for(u64::from(map_address)),
            Some(cpu_address)
        );
        assert_eq!(nvdrv.nvmap_handles[&497].channel_pin_count, 1);

        let unmapped = nvdrv.dispatch_ioctl(request(fd, 0xc01c_000a, mapped.data, input.len()));
        assert_eq!(&unmapped.data[..0x14], &[0u8; 0x14]);
        assert_eq!(&unmapped.data[0x14..], &input[0x14..]);
        assert_eq!(nvdrv.nvmap_handles[&497].channel_pin_count, 0);
    }

    #[test]
    fn video_channel_submit_preserves_payload_and_reserves_fence() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        let syncpoint = nvdrv.ensure_channel_syncpoint(fd).0;
        let cpu_address = 0x4a07_f000_00u64;
        nvdrv
            .nvmap_handles
            .insert(497, test_nvmap_handle(497, 0x8000, cpu_address));

        let mut input = vec![0u8; 0x40];
        write_u32(&mut input, 0, 2);
        write_u32(&mut input, 4, 0);
        write_u32(&mut input, 8, 1);
        write_u32(&mut input, 12, 1);
        write_u32(&mut input, 0x10, 497);
        write_u32(&mut input, 0x14, 0);
        write_u32(&mut input, 0x18, 139);
        write_u32(&mut input, 0x1c, 497);
        write_u32(&mut input, 0x20, 0x22c);
        write_u32(&mut input, 0x24, 2);
        write_u32(&mut input, 0x28, syncpoint);
        write_u32(&mut input, 0x2c, 1);
        write_u32(&mut input, 0x30, 0x1111_1111);
        write_u32(&mut input, 0x34, 0x2222_2222);
        write_u32(&mut input, 0x38, 0x3333_3333);
        write_u32(&mut input, 0x3c, 0xffff_ffff);

        let memory = vec![0u8; 0x234];
        let mem_read = |address: u64, output: &mut [u8]| {
            let Some(offset) = address.checked_sub(cpu_address) else {
                return false;
            };
            let offset = offset as usize;
            let Some(source) = memory.get(offset..offset.saturating_add(output.len())) else {
                return false;
            };
            output.copy_from_slice(source);
            true
        };
        let submitted = nvdrv.dispatch_ioctl_with_mem(
            request(fd, 0xc040_0001, input.clone(), input.len()),
            &mem_read,
            &|_, _| false,
        );

        assert_eq!(submitted.result, 0);
        assert_eq!(&submitted.data[..0x3c], &input[..0x3c]);
        assert_eq!(read_u32(&submitted.data, 0x3c), Some(1));
        assert_eq!(nvdrv.syncpoint_max(syncpoint), 1);
        assert_eq!(nvdrv.syncpoint_value(syncpoint), 1);
    }

    #[test]
    fn host1x_trace_decodes_incrementing_and_immediate_methods() {
        let words = [
            (0x10 << 16) | (0xf0 << 6),
            (1 << 28) | (0x20 << 16) | 2,
            0xaaaa_0001,
            0xbbbb_0002,
            (4 << 28) | (0x12 << 16) | 0x123,
        ];
        let methods = decode_host1x_methods(&words, 0, 8);

        assert_eq!(
            methods,
            vec![
                Host1xMethodTrace {
                    word_index: 2,
                    class_id: 0xf0,
                    method: 0x20,
                    argument: 0xaaaa_0001,
                },
                Host1xMethodTrace {
                    word_index: 3,
                    class_id: 0xf0,
                    method: 0x21,
                    argument: 0xbbbb_0002,
                },
                Host1xMethodTrace {
                    word_index: 4,
                    class_id: 0xf0,
                    method: 0x12,
                    argument: 0x123,
                },
            ]
        );
    }
}
