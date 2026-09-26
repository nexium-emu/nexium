use parking_lot::{Condvar, Mutex};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub(crate) fn syncpoint_reached(current: u32, threshold: u32) -> bool {
    current.wrapping_sub(threshold) < 0x8000_0000
}

pub(crate) fn syncpoint_expired(min: u32, max: u32, threshold: u32) -> bool {
    max.wrapping_sub(threshold) >= min.wrapping_sub(threshold)
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

pub fn kick_timeline_enabled() -> bool {
    nexium_common::timeline::enabled()
}

pub fn timeline_us() -> u64 {
    nexium_common::timeline::us()
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
pub mod video_decode_thread;
pub mod video_ffmpeg;
pub mod video_host1x;
pub mod video_surface;
pub mod video_vp9;
pub use bufferqueue::{BufferQueue, GraphicBuffer, QueuedFrame};
pub use gpu::GpuContext;
pub use nexium_gpu::{
    PipelinedPresentCompletion, PipelinedPresentFrame, PipelinedPresentReadback,
    PipelinedPresentSubmission, PresentDepth,
};

pub const FRAME_QUEUE_CAPACITY: usize = 4;
const FRAME_QUEUE_STALL_GRACE: std::time::Duration = std::time::Duration::from_millis(250);

fn note_frame_replaced_after_stall(waited: std::time::Duration) {
    static REPLACED: AtomicU64 = AtomicU64::new(0);
    let n = REPLACED.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        log::warn!(
            "[frame-queue] presenter stalled {:.0} ms with {} frames queued; replaced the oldest frame (#{})",
            waited.as_secs_f64() * 1000.0,
            FRAME_QUEUE_CAPACITY,
            n
        );
    }
    if gpu::watchdog::armed() && (n == 4 || n % 64 == 0) {
        gpu::stackdump::dump_all_threads("frame-queue presenter stall");
    }
}

pub struct FrameQueueState {
    frames: Mutex<FrameQueueContents>,
    frame_available: Condvar,
    space_available: Condvar,
    closed: AtomicBool,
    presenter_stalled: AtomicBool,
}

#[derive(Default)]
struct FrameQueueContents {
    pending: VecDeque<QueuedFrame>,
    deadline_shift: std::time::Duration,
    last_source_deadline: Option<std::time::Instant>,
}

impl FrameQueueContents {
    fn push(&mut self, mut frame: QueuedFrame, stalled: bool, now: std::time::Instant) {
        let source_deadline = frame.present_at;
        if source_deadline.is_none()
            || source_deadline
                .zip(self.last_source_deadline)
                .is_some_and(|(deadline, previous)| deadline < previous)
        {
            self.deadline_shift = std::time::Duration::ZERO;
        }
        self.last_source_deadline = source_deadline;
        frame.present_at =
            source_deadline.and_then(|deadline| deadline.checked_sub(self.deadline_shift));
        let mut retained_correction = std::time::Duration::ZERO;
        if let Some(deadline) = frame
            .present_at
            .filter(|deadline| stalled && *deadline > now)
        {
            let correction = deadline.duration_since(now);
            self.deadline_shift = self.deadline_shift.saturating_add(correction);
            retained_correction = correction;
            frame.present_at = Some(now);
        }
        if stalled {
            let retained_lead = self
                .pending
                .iter()
                .filter_map(|frame| frame.present_at)
                .max()
                .map_or(std::time::Duration::ZERO, |deadline| {
                    deadline.saturating_duration_since(now)
                });
            retained_correction = retained_correction.max(retained_lead);
        }
        if retained_correction != std::time::Duration::ZERO {
            for retained in &mut self.pending {
                retained.present_at = retained
                    .present_at
                    .and_then(|deadline| deadline.checked_sub(retained_correction));
            }
            note_frame_deadlines_rebased(retained_correction, self.deadline_shift);
        }
        self.pending.push_back(frame);
    }
}

fn note_frame_deadlines_rebased(correction: std::time::Duration, total: std::time::Duration) {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*TRACE.get_or_init(|| std::env::var_os("NEXIUM_FRAME_QUEUE_TRACE").is_some()) {
        return;
    }
    static REBASED: AtomicU64 = AtomicU64::new(0);
    let count = REBASED.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 4 || count.is_power_of_two() {
        log::warn!(
            "[frame-queue] rebased future deadlines correction_ms={:.3} total_ms={:.3} count={}",
            correction.as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            count,
        );
    }
}

impl FrameQueueState {
    pub fn new() -> Self {
        Self {
            frames: Mutex::new(FrameQueueContents::default()),
            frame_available: Condvar::new(),
            space_available: Condvar::new(),
            closed: AtomicBool::new(false),
            presenter_stalled: AtomicBool::new(false),
        }
    }

    fn note_consumer_progress(resumed: bool) {
        if resumed {
            log::info!("[frame-queue] presenter resumed; FIFO backpressure restored");
        }
    }

    fn enqueue(&self, frame: QueuedFrame) -> bool {
        let mut frames = self.frames.lock();
        let mut stalled_since = None;
        while frames.pending.len() >= FRAME_QUEUE_CAPACITY {
            if self.closed.load(Ordering::Acquire) {
                return false;
            }
            if self.presenter_stalled.load(Ordering::Acquire) {
                frames.pending.pop_front();
                break;
            }
            let started = *stalled_since.get_or_insert_with(std::time::Instant::now);
            let waited = started.elapsed();
            if waited >= FRAME_QUEUE_STALL_GRACE {
                frames.pending.pop_front();
                self.presenter_stalled.store(true, Ordering::Release);
                note_frame_replaced_after_stall(waited);
                break;
            }
            self.space_available
                .wait_for(&mut frames, FRAME_QUEUE_STALL_GRACE - waited);
        }
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        frames.push(
            frame,
            self.presenter_stalled.load(Ordering::Acquire),
            std::time::Instant::now(),
        );
        drop(frames);
        self.frame_available.notify_one();
        true
    }

    fn pop_front_due(&self, now: std::time::Instant) -> Option<QueuedFrame> {
        let mut frames = self.frames.lock();
        let is_due = frames
            .pending
            .front()
            .and_then(|frame| frame.present_at)
            .is_none_or(|deadline| deadline <= now);
        let frame = (is_due || !nexium_common::speed_limit::enabled())
            .then(|| frames.pending.pop_front()).flatten();
        let resumed = frame.is_some() && self.presenter_stalled.swap(false, Ordering::AcqRel);
        drop(frames);
        if frame.is_some() {
            Self::note_consumer_progress(resumed);
            self.space_available.notify_one();
        }
        frame
    }

    pub fn wait_pop_front_due(&self, stopping: impl Fn() -> bool) -> Option<QueuedFrame> {
        const STOP_POLL: std::time::Duration = std::time::Duration::from_millis(10);
        let mut frames = self.frames.lock();
        loop {
            if stopping() || self.closed.load(Ordering::Acquire) {
                return None;
            }
            let Some(front) = frames.pending.front() else {
                self.frame_available.wait_for(&mut frames, STOP_POLL);
                continue;
            };
            let now = std::time::Instant::now();
            if let Some(deadline) = front.present_at
                .filter(|deadline| nexium_common::speed_limit::enabled() && *deadline > now)
            {
                self.frame_available
                    .wait_for(&mut frames, (deadline - now).min(STOP_POLL));
                continue;
            }
            let frame = frames.pending.pop_front();
            let resumed = self.presenter_stalled.swap(false, Ordering::AcqRel);
            drop(frames);
            Self::note_consumer_progress(resumed);
            self.space_available.notify_one();
            return frame;
        }
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.frame_available.notify_all();
        self.space_available.notify_all();
    }

    fn drain(&self) -> Vec<QueuedFrame> {
        let mut queue = self.frames.lock();
        let frames: Vec<_> = std::mem::take(&mut queue.pending).into_iter().collect();
        let resumed = !frames.is_empty() && self.presenter_stalled.swap(false, Ordering::AcqRel);
        drop(queue);
        if !frames.is_empty() {
            Self::note_consumer_progress(resumed);
            self.space_available.notify_all();
        }
        frames
    }

    fn len(&self) -> usize {
        self.frames.lock().pending.len()
    }
}

pub type FrameQueue = Arc<FrameQueueState>;

pub fn enqueue_bounded_frame(frame_queue: &FrameQueue, stats: &PipelineStats, frame: QueuedFrame) {
    if frame_queue.enqueue(frame) {
        stats.frames_submitted.fetch_add(1, Ordering::Relaxed);
    }
}

struct VideoChannelRuntime {
    parser: video_host1x::VideoHost1xParser,
    decoder: Option<openh264::decoder::Decoder>,
    ffmpeg_config: Option<(video_ffmpeg::FfmpegCodec, u32, u32)>,
    ffmpeg_failed: Arc<std::sync::atomic::AtomicBool>,
    composer: video_decode::H264AnnexBComposer,
    vp9_composer: video_vp9::Vp9FrameComposer,
    vp9_packet_target: Option<u64>,
    vp9_unavailable_logged: bool,
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
            ffmpeg_config: None,
            ffmpeg_failed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            composer: video_decode::H264AnnexBComposer::new(),
            vp9_composer: video_vp9::Vp9FrameComposer::new(),
            vp9_packet_target: None,
            vp9_unavailable_logged: false,
        }
    }
}

fn next_vp9_packet_target(pending_target: &mut Option<u64>, current_target: u64) -> u64 {
    pending_target
        .replace(current_target)
        .unwrap_or(current_target)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
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
    pub user_refcount: u32,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CtrlEventWait {
    pub syncpt_id: u32,
    pub threshold: u32,
}

pub const NVRESULT_NOT_IMPLEMENTED: u32 = 1;

const MAX_VSMS: usize = 128;
const AS_GPU_SMALL_PAGE_SIZE: u32 = 0x1000;
const AS_GPU_DEFAULT_BIG_PAGE_SIZE: u32 = 0x10000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AsGpuAllocation {
    base: u64,
    size: u64,
    page_size: u32,
    sparse: bool,
    big_pages: bool,
}

struct AsGpuState {
    initialized: bool,
    big_page_size: u32,
    allocations: BTreeMap<u64, AsGpuAllocation>,
}

impl Default for AsGpuState {
    fn default() -> Self {
        Self {
            initialized: false,
            big_page_size: AS_GPU_DEFAULT_BIG_PAGE_SIZE,
            allocations: BTreeMap::new(),
        }
    }
}

const CTRL_EVENT_WAIT_FAIL_LIMIT: u32 = 3;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CtrlEventWaitFailure {
    failures: u32,
    logged: bool,
    drain_attempted: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CtrlEventWaitFailureAction {
    log: bool,
    drain: bool,
}

fn ctrl_event_wait_failure_action(
    state: &mut CtrlEventWaitFailure,
    submitted: bool,
    escape_enabled: bool,
) -> CtrlEventWaitFailureAction {
    state.failures = state.failures.saturating_add(1);
    if state.failures <= CTRL_EVENT_WAIT_FAIL_LIMIT {
        return CtrlEventWaitFailureAction::default();
    }
    let log = !state.logged;
    state.logged = true;
    let drain = submitted && escape_enabled && !state.drain_attempted;
    if drain {
        state.drain_attempted = true;
    }
    CtrlEventWaitFailureAction { log, drain }
}

fn syncpoint_escape_drain_value_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn syncpoint_escape_drain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        syncpoint_escape_drain_value_enabled(std::env::var_os("NEXIUM_SYNCPT_ESCAPE").as_deref())
    })
}

fn ioctl_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_IOCTL_PROFILE").is_some())
}

type IoctlProfileTable = HashMap<(NvDevice, u16), (u64, u64, HashMap<u32, u64>)>;

fn ioctl_profile_state() -> &'static Mutex<(IoctlProfileTable, Option<std::time::Instant>)> {
    static STATE: std::sync::OnceLock<Mutex<(IoctlProfileTable, Option<std::time::Instant>)>> =
        std::sync::OnceLock::new();
    STATE.get_or_init(|| Mutex::new((HashMap::new(), None)))
}

fn ioctl_profile_record(device: NvDevice, cmd: u16, result: u32, elapsed_ns: u64) {
    let now = std::time::Instant::now();
    let mut guard = ioctl_profile_state().lock();
    let (table, last_dump) = &mut *guard;
    let entry = table
        .entry((device, cmd))
        .or_insert_with(|| (0, 0, HashMap::new()));
    entry.0 = entry.0.saturating_add(1);
    entry.1 = entry.1.saturating_add(elapsed_ns);
    *entry.2.entry(result).or_insert(0) += 1;

    let due = match last_dump {
        Some(previous) => now.duration_since(*previous) >= std::time::Duration::from_secs(1),
        None => true,
    };
    if !due {
        return;
    }
    *last_dump = Some(now);
    let mut rows: Vec<_> = table.drain().collect();
    rows.sort_by_key(|(_, (count, _, _))| std::cmp::Reverse(*count));
    log::warn!("[ioctl-profile] top nvdrv ioctls (last interval):");
    for ((device, cmd), (count, ns, results)) in rows.into_iter().take(10) {
        let mut codes: Vec<_> = results.into_iter().collect();
        codes.sort_by_key(|(_, hits)| std::cmp::Reverse(*hits));
        let codes: Vec<String> = codes
            .into_iter()
            .take(3)
            .map(|(code, hits)| format!("r{}={}", code, hits))
            .collect();
        log::warn!(
            "  {:?} cmd={:#06x} count={} total_ms={:.2} avg_us={:.2} [{}]",
            device,
            cmd,
            count,
            ns as f64 / 1.0e6,
            ns as f64 / 1000.0 / count.max(1) as f64,
            codes.join(" ")
        );
    }
}

pub type AsyncMemoryRead = Arc<dyn Fn(u64, &mut [u8]) -> bool + Send + Sync>;
pub type AsyncMemoryWrite = Arc<dyn Fn(u64, &[u8]) -> bool + Send + Sync>;
pub type PresentPrepared = Box<dyn FnOnce() + Send>;
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
        completion: Option<AsyncGpuCompletion>,
    },
    Present {
        job: crate::render_thread::RenderJob,
        pending: Arc<std::sync::atomic::AtomicUsize>,
        limit: usize,
        on_prepared: Option<PresentPrepared>,
    },
    Barrier(crossbeam::channel::Sender<bool>),
    Shutdown,
}

struct QueuedAsyncGpuSubmission {
    submission: AsyncGpuSubmission,
    pending: Option<AsyncGpuPendingGuard>,
}

impl QueuedAsyncGpuSubmission {
    fn tracked(submission: AsyncGpuSubmission, pending: AsyncGpuPendingGuard) -> Self {
        Self {
            submission,
            pending: Some(pending),
        }
    }

    fn untracked(submission: AsyncGpuSubmission) -> Self {
        Self {
            submission,
            pending: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncPresentSubmit {
    Enqueued,
    Coalesced,
    Unavailable,
}

struct AsyncGpuQueue {
    gpu: Arc<GpuContext>,
    frame_queue: FrameQueue,
    tx: crossbeam::channel::Sender<QueuedAsyncGpuSubmission>,
    pending: Arc<std::sync::atomic::AtomicUsize>,
    capacity: usize,
    profile: Option<AsyncGpuQueueProfile>,
    defer_small_rts: bool,
    #[cfg(test)]
    hard_kicks: bool,
    failed: Arc<std::sync::atomic::AtomicBool>,
    stopping: Arc<std::sync::atomic::AtomicBool>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

struct AsyncGpuWait<'a> {
    failed: &'a AtomicBool,
    stopping: &'a AtomicBool,
    closed: &'a AtomicBool,
    poll_interval: std::time::Duration,
    report_interval: std::time::Duration,
}

impl AsyncGpuWait<'_> {
    fn cancelled(&self) -> bool {
        self.stopping.load(Ordering::Acquire) || self.closed.load(Ordering::Acquire)
    }

    fn interrupted(&self) -> bool {
        self.failed.load(Ordering::Acquire) || self.cancelled()
    }

    fn send<T>(
        &self,
        tx: &crossbeam::channel::Sender<T>,
        mut value: T,
        label: &str,
    ) -> Result<(), T> {
        let mut report = AsyncGpuWaitReport::new(self.report_interval);
        loop {
            if self.interrupted() {
                return Err(value);
            }
            match tx.send_timeout(value, self.poll_interval) {
                Ok(()) => return Ok(()),
                Err(crossbeam::channel::SendTimeoutError::Timeout(returned)) => {
                    value = returned;
                    report.waiting(label);
                }
                Err(crossbeam::channel::SendTimeoutError::Disconnected(returned)) => {
                    log::error!("[async-gpu] {label} disconnected");
                    self.failed.store(true, Ordering::Release);
                    return Err(returned);
                }
            }
        }
    }

    fn receive_barrier(&self, rx: &crossbeam::channel::Receiver<bool>) -> bool {
        let mut report = AsyncGpuWaitReport::new(self.report_interval);
        loop {
            if self.interrupted() {
                return false;
            }
            match rx.recv_timeout(self.poll_interval) {
                Ok(true) => return true,
                Ok(false) => {
                    log::error!("[async-gpu] drain barrier reported failure");
                    self.failed.store(true, Ordering::Release);
                    return false;
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    report.waiting("drain barrier");
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    log::error!("[async-gpu] drain barrier disconnected");
                    self.failed.store(true, Ordering::Release);
                    return false;
                }
            }
        }
    }
}

struct AsyncGpuWaitReport {
    started: std::time::Instant,
    next_report: std::time::Duration,
    interval: std::time::Duration,
    progress: (u64, u64),
}

impl AsyncGpuWaitReport {
    fn new(interval: std::time::Duration) -> Self {
        Self {
            started: std::time::Instant::now(),
            next_report: interval,
            interval,
            progress: gpu::watchdog::progress_snapshot(),
        }
    }

    fn waiting(&mut self, label: &str) {
        let waited = self.started.elapsed();
        if waited < self.next_report {
            return;
        }
        let progress = gpu::watchdog::progress_snapshot();
        log::warn!(
            "[async-gpu] {label} pending waited_ms={:.1} worker_progress={} render_progress={}",
            waited.as_secs_f64() * 1000.0,
            progress.0.wrapping_sub(self.progress.0),
            progress.1.wrapping_sub(self.progress.1),
        );
        self.progress = progress;
        self.next_report = waited.saturating_add(self.interval);
    }
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

#[cfg(test)]
fn reserve_ordered_present_slot(pending: &std::sync::atomic::AtomicUsize, limit: usize) {
    let _ = reserve_ordered_present_slot_until(pending, limit, || false);
}

fn reserve_ordered_present_slot_until(
    pending: &std::sync::atomic::AtomicUsize,
    limit: usize,
    stopping: impl Fn() -> bool,
) -> bool {
    let waited_from = std::time::Instant::now();
    let mut next_report = std::time::Duration::from_millis(250);
    loop {
        if stopping() {
            return false;
        }
        if pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |inflight| {
                (inflight < limit).then_some(inflight + 1)
            })
            .is_ok()
        {
            return true;
        }
        let waited = waited_from.elapsed();
        if waited >= next_report {
            log::warn!(
                "[ordered-present] FIFO backpressure pending={} limit={} waited_ms={:.1}",
                pending.load(Ordering::Acquire),
                limit,
                waited.as_secs_f64() * 1000.0,
            );
            next_report = next_report.saturating_add(std::time::Duration::from_secs(1));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

struct AsyncGpuPendingGuard {
    pending: Arc<std::sync::atomic::AtomicUsize>,
    failed: Arc<std::sync::atomic::AtomicBool>,
    completed: bool,
}

impl AsyncGpuPendingGuard {
    fn reserve(
        pending: Arc<std::sync::atomic::AtomicUsize>,
        failed: Arc<std::sync::atomic::AtomicBool>,
    ) -> (Self, usize) {
        let inflight = pending.fetch_add(1, Ordering::Relaxed) + 1;
        (
            Self {
                pending,
                failed,
                completed: false,
            },
            inflight,
        )
    }

    fn complete(mut self) {
        self.completed = true;
    }
}

impl Drop for AsyncGpuPendingGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.failed.store(true, Ordering::Release);
        }
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
        frame_queue: FrameQueue,
        mem_read: AsyncMemoryRead,
        mem_write: AsyncMemoryWrite,
        mem_copy: AsyncMemoryCopy,
    ) -> Self {
        let hard_kicks =
            async_gpu_hard_kicks_value(std::env::var("NEXIUM_ASYNC_GPU").ok().as_deref())
                && !gpu::gpu_pipeline_enabled();
        Self::new_with_mode(gpu, frame_queue, mem_read, mem_write, mem_copy, hard_kicks)
    }

    fn new_with_mode(
        gpu: Arc<GpuContext>,
        frame_queue: FrameQueue,
        mem_read: AsyncMemoryRead,
        mem_write: AsyncMemoryWrite,
        mem_copy: AsyncMemoryCopy,
        hard_kicks: bool,
    ) -> Self {
        let capacity = async_gpu_queue_depth();
        if gpu::gpu_pipeline_enabled() {
            gpu.install_prep_thread(
                gpu::prep::PrepThreadResources {
                    maxwell_dma: Arc::clone(&gpu.maxwell_dma),
                    fermi_2d: Arc::clone(&gpu.fermi_2d),
                    kepler_compute: Arc::clone(&gpu.kepler_compute),
                    kepler_memory: Arc::clone(&gpu.kepler_memory),
                    mappings: Arc::clone(&gpu.mappings),
                    stats: Arc::clone(&gpu.stats),
                    mem_read: Arc::clone(&mem_read),
                    mem_write: Arc::clone(&mem_write),
                    mem_copy: Arc::clone(&mem_copy),
                },
                gpu::prep::PrepThreadBehavior::Pipeline,
            );
        }
        let (tx, rx) = crossbeam::channel::bounded::<QueuedAsyncGpuSubmission>(capacity);
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let failed_worker = Arc::clone(&failed);
        let stopping = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopping_worker = Arc::clone(&stopping);
        let worker_frame_queue = Arc::clone(&frame_queue);
        let worker_gpu = Arc::clone(&gpu);
        let profile = async_gpu_queue_profile_enabled().then(AsyncGpuQueueProfile::new);
        let defer_small_rts = !hard_kicks
            && (!gpu::eager_small_rt_writeback_enabled()
                || matches!(
                    std::env::var("NEXIUM_ASYNC_GPU_DEFER_SMALLRT")
                        .ok()
                        .as_deref(),
                    Some("1") | Some("true") | Some("on") | Some("yes")
                ));
        let flush_small_rts = defer_small_rts && gpu::eager_small_rt_writeback_enabled();
        if defer_small_rts {
            log::info!("nexium-nvdrv: async GPU small-RT writeback deferred to queue barriers");
        }
        if hard_kicks {
            log::info!(
                "nexium-nvdrv: async GPU queue uses hard kick boundaries (sync-path semantics per kick)"
            );
        }
        log::info!("nexium-nvdrv: async GPU queue depth={capacity}");
        let worker = std::thread::Builder::new()
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
                gpu::watchdog::register_worker_thread();
                gpu::watchdog::install();
                let pipeline = gpu::gpu_pipeline_enabled();
                let make_on_complete =
                    |completion: Option<AsyncGpuCompletion>,
                     pending_guard: AsyncGpuPendingGuard|
                     -> Box<dyn FnOnce() + Send> {
                        let completion_gpu = Arc::clone(&worker_gpu);
                        Box::new(move || {
                            if let Some(completion) = completion {
                                completion_gpu.record_syncpoint_completion(
                                    completion.fd,
                                    completion.syncpt_id,
                                    completion.threshold,
                                );
                            }
                            pending_guard.complete();
                        })
                    };
                gpu::watchdog::phase(gpu::watchdog::Phase::Idle, 0);
                while let Ok(queued) = rx.recv() {
                    if matches!(&queued.submission, AsyncGpuSubmission::Shutdown) {
                        break;
                    }
                    if failed_worker.load(Ordering::Acquire) {
                        drop(queued);
                        continue;
                    }
                    let QueuedAsyncGpuSubmission {
                        submission,
                        pending,
                    } = queued;
                    let processed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || match (submission, pending) {
                        (
                            AsyncGpuSubmission::Inline {
                                entries,
                                completion,
                            },
                            Some(pending_guard),
                        ) => {
                            let on_complete =
                                Some(make_on_complete(completion, pending_guard));
                            gpu::watchdog::phase(gpu::watchdog::Phase::Kick, entries.len() as u64);
                            if hard_kicks {
                                worker_gpu.process_inline_gpfifo(
                                    &entries,
                                    |addr, buf| mem_read(addr, buf),
                                    |addr, buf| mem_write(addr, buf),
                                    |src, dst, len| mem_copy(src, dst, len),
                                    on_complete,
                                );
                            } else if defer_small_rts {
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
                        (
                            AsyncGpuSubmission::Present {
                                job,
                                pending,
                                limit,
                                on_prepared,
                            },
                            Some(pending_guard),
                        ) => {
                            gpu::watchdog::phase(gpu::watchdog::Phase::Present, 0);
                            if !reserve_ordered_present_slot_until(&pending, limit, || {
                                stopping_worker.load(Ordering::Acquire)
                                    || worker_frame_queue.closed.load(Ordering::Acquire)
                                    || failed_worker.load(Ordering::Acquire)
                            }) {
                                if stopping_worker.load(Ordering::Acquire)
                                    || worker_frame_queue.closed.load(Ordering::Acquire)
                                {
                                    pending_guard.complete();
                                }
                                return;
                            }
                            let job = guarded_present_job(pending, job);
                            let job: crate::render_thread::RenderJob = Box::new(move || {
                                job();
                                pending_guard.complete();
                            });
                            let job = if pipeline {
                                match worker_gpu.prep_present(job, flush_small_rts, on_prepared) {
                                    Ok(()) => None,
                                    Err((job, on_prepared)) => {
                                        log::error!(
                                            "[gpu-prep] prep lane unavailable; preserving present on render FIFO"
                                        );
                                        Some((job, on_prepared))
                                    }
                                }
                            } else {
                                Some((job, on_prepared))
                            };
                            if let Some((job, on_prepared)) = job {
                                let mut completed = worker_gpu.flush_prepared_draw_packets();
                                if completed
                                    && flush_small_rts
                                    && gpu::vk_dispatch::has_pending_small_rt_writebacks()
                                {
                                    completed = worker_gpu
                                        .flush_small_rt_writebacks(|addr, buf| mem_write(addr, buf));
                                }
                                if completed
                                    && gpu::vk_dispatch::cpu_readable_rt_writeback_mode()
                                        == gpu::vk_dispatch::CpuReadableRtWritebackMode::Present
                                    && gpu::vk_dispatch::has_pending_cpu_readable_rt_writebacks()
                                {
                                    worker_gpu.flush_cpu_readable_rt_writebacks(|addr, buf| mem_write(addr, buf));
                                }
                                if completed {
                                    if let Some(on_prepared) = on_prepared {
                                        on_prepared();
                                    }
                                    if let Some(render_thread) =
                                        crate::render_thread::maybe_render_thread()
                                    {
                                        render_thread
                                            .submit_named("async-present-readback", job);
                                    } else {
                                        job();
                                    }
                                } else {
                                    log::error!(
                                        "[ordered-present] flush failed before render submission"
                                    );
                                    failed_worker.store(true, Ordering::Release);
                                }
                            }
                        }
                        (AsyncGpuSubmission::Barrier(done), None) => {
                            gpu::watchdog::phase(gpu::watchdog::Phase::Barrier, 0);
                            if pipeline {
                                match worker_gpu.prep_drain_barrier(done, flush_small_rts) {
                                    gpu::prep::PrepBarrierDispatch::Queued => {}
                                    gpu::prep::PrepBarrierDispatch::Inline => {
                                        log::error!(
                                            "[gpu-prep] drain barrier reached an inline lane"
                                        );
                                        failed_worker.store(true, Ordering::Release);
                                    }
                                    gpu::prep::PrepBarrierDispatch::Disconnected => {
                                        log::error!(
                                            "[gpu-prep] drain barrier reached a disconnected lane"
                                        );
                                        failed_worker.store(true, Ordering::Release);
                                    }
                                }
                            } else {
                                let mut completed = worker_gpu.flush_prepared_draw_packets();
                                if completed
                                    && flush_small_rts
                                    && gpu::vk_dispatch::has_pending_small_rt_writebacks()
                                {
                                    completed = worker_gpu
                                        .flush_small_rt_writebacks(|addr, buf| mem_write(addr, buf));
                                }
                                if completed
                                    && gpu::vk_dispatch::cpu_readable_rt_writeback_mode()
                                        == gpu::vk_dispatch::CpuReadableRtWritebackMode::Present
                                    && gpu::vk_dispatch::has_pending_cpu_readable_rt_writebacks()
                                {
                                    worker_gpu.flush_cpu_readable_rt_writebacks(|addr, buf| mem_write(addr, buf));
                                }
                                let _ = done.send(completed);
                            }
                        }
                        (AsyncGpuSubmission::Shutdown, None) => unreachable!(),
                        (_, _) => {
                            log::error!("[async-gpu] malformed pending ownership");
                            failed_worker.store(true, Ordering::Release);
                        }
                    },
                    ));
                    if processed.is_err() {
                        log::error!("[async-gpu] submission panicked; queue failed");
                        failed_worker.store(true, Ordering::Release);
                    }
                    gpu::watchdog::phase(gpu::watchdog::Phase::Idle, 0);
                }
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                let _ = gpu.shutdown_prep_thread(flush_small_rts);
                panic!("spawn GPU submit thread: {error}");
            }
        };
        Self {
            gpu,
            frame_queue,
            tx,
            pending,
            capacity,
            profile,
            defer_small_rts: flush_small_rts,
            #[cfg(test)]
            hard_kicks,
            failed,
            stopping,
            worker: Mutex::new(Some(worker)),
        }
    }

    fn wait_context(&self) -> AsyncGpuWait<'_> {
        AsyncGpuWait {
            failed: &self.failed,
            stopping: &self.stopping,
            closed: &self.frame_queue.closed,
            poll_interval: std::time::Duration::from_millis(10),
            report_interval: std::time::Duration::from_secs(3),
        }
    }

    fn submit(&self, submission: AsyncGpuSubmission) -> bool {
        let wait = self.wait_context();
        if wait.interrupted() {
            return false;
        }
        let (pending_guard, inflight) =
            AsyncGpuPendingGuard::reserve(Arc::clone(&self.pending), Arc::clone(&self.failed));
        let queued_submission = QueuedAsyncGpuSubmission::tracked(submission, pending_guard);
        let (result, was_full, blocked_ns) = match self.tx.try_send(queued_submission) {
            Ok(()) => (Ok(()), false, 0),
            Err(crossbeam::channel::TrySendError::Full(queued_submission)) => {
                let started = std::time::Instant::now();
                let result = wait.send(&self.tx, queued_submission, "submission enqueue");
                (result, true, started.elapsed().as_nanos() as u64)
            }
            Err(crossbeam::channel::TrySendError::Disconnected(queued_submission)) => {
                self.failed.store(true, Ordering::Release);
                (Err(queued_submission), false, 0)
            }
        };
        let queued = match result {
            Ok(()) => true,
            Err(mut queued_submission) => {
                if wait.cancelled() {
                    if let Some(pending) = queued_submission.pending.take() {
                        pending.complete();
                    }
                }
                false
            }
        };
        if let Some(profile) = &self.profile {
            profile.submitted(was_full, blocked_ns, inflight, self.capacity);
        }
        queued
    }

    fn drain(&self) -> AsyncGpuDrain {
        let wait = self.wait_context();
        if wait.interrupted() {
            return AsyncGpuDrain {
                completed: false,
                barrier_send_ns: 0,
                barrier_wait_ns: 0,
            };
        }
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
        if wait
            .send(
                &self.tx,
                QueuedAsyncGpuSubmission::untracked(AsyncGpuSubmission::Barrier(done_tx)),
                "drain barrier enqueue",
            )
            .is_err()
        {
            return AsyncGpuDrain {
                completed: false,
                barrier_send_ns: send_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                barrier_wait_ns: 0,
            };
        }
        let barrier_send_ns = send_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let wait_started = std::time::Instant::now();
        let completed = wait.receive_barrier(&done_rx);
        AsyncGpuDrain {
            completed,
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

impl Drop for AsyncGpuQueue {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        let (replacement_tx, replacement_rx) = crossbeam::channel::bounded(0);
        drop(replacement_rx);
        let tx = std::mem::replace(&mut self.tx, replacement_tx);
        if tx
            .send(QueuedAsyncGpuSubmission::untracked(
                AsyncGpuSubmission::Shutdown,
            ))
            .is_err()
        {
            self.failed.store(true, Ordering::Release);
        }
        drop(tx);
        if let Some(worker) = self.worker.lock().take() {
            if worker.join().is_err() {
                log::error!("[async-gpu] submit worker panicked during shutdown");
                self.failed.store(true, Ordering::Release);
            }
        }
        if !self.gpu.shutdown_prep_thread(self.defer_small_rts) {
            self.failed.store(true, Ordering::Release);
        }
        if !gpu::vk_dispatch::sync_render_thread() {
            log::error!("[async-gpu] render worker drain failed during shutdown");
            self.failed.store(true, Ordering::Release);
        }
        if self.pending.load(Ordering::Acquire) != 0 {
            log::error!(
                "[async-gpu] pending work remained after shutdown: {}",
                self.pending.load(Ordering::Acquire)
            );
            self.failed.store(true, Ordering::Release);
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

fn gpu_thread_flag_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes" | "hard" | "soft"
        )
    })
}

fn async_gpu_soft_kicks_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim().eq_ignore_ascii_case("soft"))
}

fn async_gpu_hard_kicks_value(value: Option<&str>) -> bool {
    !async_gpu_soft_kicks_value(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuThreadModeRequest {
    Inline,
    SyncPrepThread,
    AsyncGpu,
    Conflict,
}

fn gpu_thread_mode_request(
    sync_prep: Option<&str>,
    async_gpu: Option<&str>,
    gpu_pipeline: Option<&str>,
) -> GpuThreadModeRequest {
    let sync_prep = gpu_thread_flag_enabled(sync_prep);
    let async_default = async_gpu.is_none();
    let async_gpu = gpu_thread_flag_enabled(async_gpu);
    let gpu_pipeline = gpu_thread_flag_enabled(gpu_pipeline);
    if sync_prep && (async_gpu || gpu_pipeline) {
        GpuThreadModeRequest::Conflict
    } else if sync_prep {
        GpuThreadModeRequest::SyncPrepThread
    } else if async_gpu || async_default || gpu_pipeline {
        GpuThreadModeRequest::AsyncGpu
    } else {
        GpuThreadModeRequest::Inline
    }
}

fn async_gpu_requires_quarantine(async_gpu: Option<&str>, gpu_pipeline: Option<&str>) -> bool {
    async_gpu_soft_kicks_value(async_gpu) || gpu_thread_flag_enabled(gpu_pipeline)
}

fn ioctl_requires_async_gpu_drain(device: NvDevice, cmd: u16) -> bool {
    match device {
        NvDevice::NvhostAsGpu => matches!(cmd, 0x4102 | 0x4103 | 0x4105 | 0x4106 | 0x4114),
        NvDevice::NvhostNvdec => cmd == 0x0009,
        NvDevice::NvhostVic => matches!(cmd, 0x0001 | 0x0009),
        NvDevice::Nvmap => cmd == 0x0105,
        _ => false,
    }
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
    pub frame_queue: FrameQueue,
    pub next_event_id: u32,
    pub next_syncpoint_id: u32,
    pub retired_syncpts: Arc<Mutex<HashMap<u32, (u32, u32)>>>,
    ordered_submit_max: HashMap<u32, u32>,
    pub next_ctrl_event_slot: u32,
    pub ctrl_event_waits: HashMap<(u32, u32), CtrlEventWait>,
    ctrl_event_wait_failures: Arc<Mutex<HashMap<(u32, u32), CtrlEventWaitFailure>>>,
    pub gpu: Arc<GpuContext>,
    pub last_swap_return: Arc<Mutex<Option<std::time::Instant>>>,
    pub queue_buffer_active: Arc<std::sync::atomic::AtomicBool>,
    pub stats: Arc<PipelineStats>,
    pub channel_client_data: u64,
    video_channels: HashMap<u32, VideoChannelRuntime>,
    video_decoder: video_decode_thread::VideoDecoder,
    pub legacy_gfx: std::sync::atomic::AtomicBool,
    pub renderer: std::sync::OnceLock<Option<Arc<nexium_gpu::Renderer>>>,
    pub presentation_target: Option<Arc<nexium_gpu::presentation::PresentationTarget>>,
    as_gpu_states: HashMap<u32, AsGpuState>,
    gpu_async: Option<Arc<AsyncGpuQueue>>,
    sync_prep_thread: bool,
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
            frame_queue: Arc::new(FrameQueueState::new()),
            next_event_id: 1,
            next_syncpoint_id: 1,
            retired_syncpts: Arc::new(Mutex::new(HashMap::new())),
            ordered_submit_max: HashMap::new(),
            next_ctrl_event_slot: 0,
            ctrl_event_waits: HashMap::new(),
            ctrl_event_wait_failures: Arc::new(Mutex::new(HashMap::new())),
            gpu: Arc::new(GpuContext::with_stats(stats.clone())),
            last_swap_return: Arc::new(Mutex::new(None)),
            queue_buffer_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats,
            channel_client_data: 0,
            video_channels: HashMap::new(),
            video_decoder: video_decode_thread::VideoDecoder::new(),
            legacy_gfx: std::sync::atomic::AtomicBool::new(false),
            renderer: std::sync::OnceLock::new(),
            presentation_target: None,
            as_gpu_states: HashMap::new(),
            gpu_async: None,
            sync_prep_thread: false,
            async_present_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn frame_queue_depth(&self) -> usize {
        self.frame_queue.len()
    }

    pub fn renderer(&self) -> Option<&Arc<nexium_gpu::Renderer>> {
        let slot = self
            .renderer
            .get_or_init(|| match nexium_gpu::Renderer::new_with_presentation(self.presentation_target.clone()) {
                Ok(r) => {
                    log::info!("nexium-nvdrv: Vulkan Renderer initialized");
                    self.gpu
                        .lock_pusher("lib.rs:set_renderer")
                        .set_renderer(Some(r.clone()));
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

    fn invalidate_texture_mapping_update(&self, update: &gpu::GpuMappingUpdate) {
        if update.changed_gpu_ranges.is_empty() && update.epoch_transitions.is_empty() {
            return;
        }
        let Some(renderer) = self
            .renderer
            .get()
            .and_then(|renderer| renderer.as_ref())
            .cloned()
        else {
            return;
        };
        let changed_gpu_ranges = update.changed_gpu_ranges.clone();
        let transitions = update
            .epoch_transitions
            .iter()
            .map(
                |transition| nexium_gpu::rt_cache::RtMappingEpochTransition {
                    gpu_va: transition.gpu_va,
                    size: transition.size,
                    old_epoch: transition.old_epoch,
                    new_epoch: transition.new_epoch,
                },
            )
            .collect::<Vec<_>>();
        let mut texture_ranges = changed_gpu_ranges.clone();
        texture_ranges.extend(
            update
                .epoch_transitions
                .iter()
                .map(|transition| (transition.gpu_va, transition.size)),
        );
        texture_ranges.sort_unstable();
        let job = move || {
            renderer.apply_render_target_mapping_update(&transitions, &changed_gpu_ranges);
            for &(gpu_va, size) in &texture_ranges {
                renderer.invalidate_texture_range(gpu_va, size);
            }
        };
        if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
            render_thread.submit_named("gpu-map-invalidate", Box::new(job));
        } else {
            job();
        }
    }

    fn as_gpu_allocation_containing(
        &self,
        fd: u32,
        gpu_va: u64,
        size: u64,
    ) -> Option<AsGpuAllocation> {
        let gpu_end = gpu_va.checked_add(size).filter(|_| size != 0)?;
        let state = self.as_gpu_states.get(&fd)?;
        let allocation = state
            .allocations
            .range(..=gpu_va)
            .next_back()
            .map(|(_, allocation)| *allocation)?;
        let allocation_end = allocation.base.checked_add(allocation.size)?;
        (gpu_va >= allocation.base && gpu_end <= allocation_end).then_some(allocation)
    }

    fn release_as_gpu_allocation(
        &mut self,
        fd: u32,
        base: u64,
        expected: Option<(u64, u32)>,
    ) -> bool {
        let Some(allocation) = self
            .as_gpu_states
            .get(&fd)
            .and_then(|state| state.allocations.get(&base))
            .copied()
        else {
            return false;
        };
        if expected.is_some_and(|(size, page_size)| {
            size != allocation.size || page_size != allocation.page_size
        }) {
            return false;
        }
        let removed = if allocation.sparse {
            let Ok(removed) = self
                .gpu
                .mappings
                .write()
                .remove_all_contained_with_metadata(fd, allocation.base, allocation.size)
            else {
                return false;
            };
            removed
        } else {
            self.gpu
                .mappings
                .write()
                .remove_all_for_allocation_with_metadata(fd, allocation.base)
        };
        for (owned_gpu_va, owned_size) in removed.owned_va_ranges {
            let freed = self.gpu.free_va(owned_gpu_va, owned_size);
            debug_assert!(freed);
        }
        let freed = self.gpu.free_va(allocation.base, allocation.size);
        debug_assert!(freed);
        if let Some(state) = self.as_gpu_states.get_mut(&fd) {
            state.allocations.remove(&base);
        }
        self.invalidate_texture_mapping_update(&removed.update);
        true
    }

    fn release_as_gpu_fd_mappings(&mut self, fd: u32) -> bool {
        let Ok(removed) = self
            .gpu
            .mappings
            .write()
            .remove_all_contained_with_metadata(fd, 0, u64::MAX)
        else {
            return false;
        };
        for (owned_gpu_va, owned_size) in removed.owned_va_ranges {
            let freed = self.gpu.free_va(owned_gpu_va, owned_size);
            debug_assert!(freed);
        }
        self.invalidate_texture_mapping_update(&removed.update);
        true
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
        let sync_prep = std::env::var("NEXIUM_SYNC_PREP_THREAD").ok();
        let async_gpu = std::env::var("NEXIUM_ASYNC_GPU").ok();
        let gpu_pipeline = std::env::var("NEXIUM_GPU_PIPELINE").ok();
        match gpu_thread_mode_request(
            sync_prep.as_deref(),
            async_gpu.as_deref(),
            gpu_pipeline.as_deref(),
        ) {
            GpuThreadModeRequest::Inline => return,
            GpuThreadModeRequest::Conflict => {
                log::error!(
                    "nexium-nvdrv: NEXIUM_SYNC_PREP_THREAD is mutually exclusive with asynchronous GPU modes"
                );
                return;
            }
            GpuThreadModeRequest::SyncPrepThread => {
                if self.gpu_async.is_some() {
                    log::error!(
                        "nexium-nvdrv: synchronous prep thread rejected while async GPU queue is active"
                    );
                    return;
                }
                if !self.sync_prep_thread {
                    self.sync_prep_thread = self.gpu.install_prep_thread(
                        gpu::prep::PrepThreadResources {
                            maxwell_dma: Arc::clone(&self.gpu.maxwell_dma),
                            fermi_2d: Arc::clone(&self.gpu.fermi_2d),
                            kepler_compute: Arc::clone(&self.gpu.kepler_compute),
                            kepler_memory: Arc::clone(&self.gpu.kepler_memory),
                            mappings: Arc::clone(&self.gpu.mappings),
                            stats: Arc::clone(&self.gpu.stats),
                            mem_read,
                            mem_write,
                            mem_copy,
                        },
                        gpu::prep::PrepThreadBehavior::DrainEachKick,
                    );
                    if self.sync_prep_thread {
                        log::info!("nexium-nvdrv: synchronous drained GPU prep thread ENABLED");
                    } else {
                        log::error!(
                            "nexium-nvdrv: synchronous GPU prep thread installation failed"
                        );
                    }
                }
                return;
            }
            GpuThreadModeRequest::AsyncGpu => {}
        }
        if self.sync_prep_thread {
            log::error!("nexium-nvdrv: async GPU queue rejected while sync prep thread is active");
            return;
        }
        if async_gpu_requires_quarantine(async_gpu.as_deref(), gpu_pipeline.as_deref())
            && !gpu::experimental_gpu_scheduling_enabled()
        {
            log::warn!(
                "nexium-nvdrv: soft-boundary asynchronous GPU submission quarantined; developer opt-in requires NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1"
            );
            return;
        }
        if self.gpu_async.is_none() {
            log::info!(
                "nexium-nvdrv: async GPU submit thread ENABLED (set NEXIUM_ASYNC_GPU=0 for synchronous submission)"
            );
            self.gpu_async = Some(Arc::new(AsyncGpuQueue::new(
                Arc::clone(&self.gpu),
                Arc::clone(&self.frame_queue),
                mem_read,
                mem_write,
                mem_copy,
            )));
        }
    }

    fn finish_sync_prep_submit(&self, label: &str) -> bool {
        if !self.sync_prep_thread {
            return true;
        }
        if self.gpu.drain_prep_after_kick() {
            true
        } else {
            log::error!("nvhost-gpu: {label} failed while draining GPU prep");
            false
        }
    }

    pub fn wait_gpu_idle_checked(&self) -> bool {
        let interrupted = || {
            self.frame_queue.closed.load(Ordering::Acquire)
                || self
                    .gpu_async
                    .as_ref()
                    .is_some_and(|queue| queue.wait_context().interrupted())
        };
        if interrupted() {
            return false;
        }
        let prep_completed = if self.sync_prep_thread {
            self.gpu.drain_prep_thread(false)
        } else {
            true
        };
        let mut queue_profile = None;
        let mut queue_completed = true;
        if let Some(queue) = &self.gpu_async {
            let queue_started = std::time::Instant::now();
            let drain = queue.drain();
            if !drain.completed {
                self.poll_gpu_completions();
                return false;
            }
            queue_completed = drain.completed;
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
        let render_completed = gpu::vk_dispatch::sync_render_thread_until(&interrupted);
        let render_ns = render_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.poll_gpu_completions();
        if !render_completed {
            return false;
        }
        let mut completions_completed = true;
        if let Some((queue, ..)) = queue_profile {
            let mut report = AsyncGpuWaitReport::new(std::time::Duration::from_secs(3));
            while queue.pending.load(Ordering::Acquire) != 0 {
                if interrupted() {
                    return false;
                }
                self.poll_gpu_completions();
                report.waiting("renderer-backed completions");
                std::thread::yield_now();
            }
            if queue.failed.load(Ordering::Acquire) {
                log::error!("[gpu-sync] asynchronous GPU queue failed");
                completions_completed = false;
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
        prep_completed && queue_completed && render_completed && completions_completed
    }

    pub fn wait_gpu_idle(&self) {
        if !self.wait_gpu_idle_checked() {
            log::error!("[gpu-sync] GPU idle wait failed");
        }
    }

    pub fn try_queue_ordered_present<F>(
        &self,
        present: F,
        on_prepared: Option<PresentPrepared>,
    ) -> AsyncPresentSubmit
    where
        F: FnOnce() + Send + 'static,
    {
        let limit = async_present_inflight_limit();
        if let Some(queue) = &self.gpu_async {
            return if queue.submit(AsyncGpuSubmission::Present {
                job: Box::new(present),
                pending: Arc::clone(&self.async_present_pending),
                limit,
                on_prepared,
            }) {
                AsyncPresentSubmit::Enqueued
            } else {
                AsyncPresentSubmit::Unavailable
            };
        }
        let submit: Box<dyn FnOnce(&'static str, crate::render_thread::RenderJob) + Send> =
            Box::new(move |label, job| {
                if let Some(on_prepared) = on_prepared {
                    on_prepared();
                }
                if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
                    render_thread.submit_named(label, job);
                } else if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                    log::error!("[ordered-present] job panicked; dispatcher continuing");
                }
            });
        let queued = crate::render_thread::present_thread().submit_ordered_named(
            Arc::clone(&self.async_present_pending),
            limit,
            "ordered-present-readback",
            Box::new(present),
            submit,
        );
        if queued {
            AsyncPresentSubmit::Enqueued
        } else {
            AsyncPresentSubmit::Unavailable
        }
    }

    fn poll_gpu_completions(&self) {
        {
            let mut events = self.gpu.syncpoint_events();
            if !events.is_empty() {
                let mut channels = self.gpu.channels.lock();
                let mut retired = self.retired_syncpts.lock();
                while let Some(event) = events.pop_front() {
                    match event {
                        gpu::PendingSyncpointEvent::Increment { syncpt_id, count } => {
                            if let Some(channel) = channels
                                .values_mut()
                                .find(|channel| channel.syncpt_id == syncpt_id)
                            {
                                channel.syncpt_min = channel.syncpt_min.wrapping_add(count);
                                if !syncpoint_reached(channel.syncpt_max, channel.syncpt_min) {
                                    channel.syncpt_max = channel.syncpt_min;
                                }
                                continue;
                            }
                            let Some(entry) = retired.get_mut(&syncpt_id) else {
                                log::warn!(
                                    "[syncpt-orphan] rejected increment id={} count={} without retired state",
                                    syncpt_id,
                                    count
                                );
                                continue;
                            };
                            log::warn!(
                                "[syncpt-orphan] applied increment id={} count={} to retired channel",
                                syncpt_id,
                                count
                            );
                            entry.0 = entry.0.wrapping_add(count);
                            if !syncpoint_reached(entry.1, entry.0) {
                                entry.1 = entry.0;
                            }
                        }
                        gpu::PendingSyncpointEvent::Completion {
                            fd,
                            syncpt_id,
                            threshold,
                        } => {
                            if syncpt_id == 0 {
                                log::warn!(
                                    "[gpu-sync] rejected completion fd={} for syncpt id=0 threshold={}",
                                    fd,
                                    threshold
                                );
                                continue;
                            }
                            if let Some(channel) = channels.get_mut(&fd) {
                                if channel.syncpt_id != syncpt_id {
                                    log::warn!(
                                        "[gpu-sync] ignored mismatched channel completion fd={} expected_syncpt={} got_syncpt={}",
                                        fd,
                                        channel.syncpt_id,
                                        syncpt_id
                                    );
                                    continue;
                                }
                                if !syncpoint_reached(channel.syncpt_min, threshold) {
                                    channel.syncpt_min = threshold;
                                }
                                continue;
                            }
                            let Some(entry) = retired.get_mut(&syncpt_id) else {
                                log::warn!(
                                    "[gpu-sync] rejected completion fd={} syncpt={} threshold={} without retired state",
                                    fd,
                                    syncpt_id,
                                    threshold
                                );
                                continue;
                            };
                            if !syncpoint_reached(entry.0, threshold) {
                                entry.0 = threshold;
                            }
                            if !syncpoint_reached(entry.1, threshold) {
                                entry.1 = threshold;
                            }
                        }
                    }
                }
            }
        }

        self.cleanup_reached_ctrl_event_wait_failures();
    }

    fn cleanup_reached_ctrl_event_wait_failures(&self) {
        if self.ctrl_event_wait_failures.lock().is_empty() {
            return;
        }
        let channels = self.gpu.channels.lock();
        let retired = self.retired_syncpts.lock();
        self.ctrl_event_wait_failures
            .lock()
            .retain(|(id, threshold), _| {
                let current = channels
                    .values()
                    .find(|channel| channel.syncpt_id == *id)
                    .map(|channel| channel.syncpt_min)
                    .or_else(|| retired.get(id).map(|(min, _)| *min));
                !current.is_some_and(|current| syncpoint_reached(current, *threshold))
            });
    }

    pub fn pace_swap(&self, swap_interval: i32) {
        if !nexium_common::speed_limit::enabled() || swap_interval <= 0 {
            *self.last_swap_return.lock() = None;
            return;
        }
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
        if device == NvDevice::NvhostAsGpu {
            self.as_gpu_states.insert(fd, AsGpuState::default());
        }
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
        if self.files.get(&fd).map(|file| file.device) == Some(NvDevice::NvhostAsGpu)
            && self.gpu_async.is_some()
            && !self.wait_gpu_idle_checked()
        {
            log::error!("nvdrv:Close fd={} rejected after GPU drain failure", fd);
            return;
        }
        let device = self.files.remove(&fd).map(|file| file.device);
        if device == Some(NvDevice::NvhostAsGpu) {
            let allocation_bases = self
                .as_gpu_states
                .get(&fd)
                .map(|state| state.allocations.keys().copied().collect::<Vec<_>>())
                .unwrap_or_default();
            let mut released = true;
            for base in allocation_bases {
                released &= self.release_as_gpu_allocation(fd, base, None);
            }
            if released {
                released = self.release_as_gpu_fd_mappings(fd);
            }
            if released {
                self.as_gpu_states.remove(&fd);
            } else {
                log::error!(
                    "nvdrv:Close fd={} retained unreleased GPU VA allocations",
                    fd
                );
            }
        }
        if self.video_channels.remove(&fd).is_some() {
            self.video_decoder
                .submit(video_decode_thread::DecodeWork::Release { fd });
        }
        let removed_waits: Vec<_> = self
            .ctrl_event_waits
            .extract_if(|(wait_fd, _), _| *wait_fd == fd)
            .map(|(_, wait)| wait)
            .collect();
        for wait in removed_waits {
            self.clear_ctrl_event_wait_failure_if_unused(wait);
        }

        let mut channels = self.gpu.channels.lock();
        if let Some(channel) = channels.remove(&fd) {
            if channel.syncpt_id != 0 {
                self.retired_syncpts
                    .lock()
                    .insert(channel.syncpt_id, (channel.syncpt_min, channel.syncpt_max));
                self.ctrl_event_wait_failures
                    .lock()
                    .retain(|(id, _), _| *id != channel.syncpt_id);
            }
        }
        drop(channels);
        log::debug!("nvdrv:Close fd={}", fd);
    }

    fn clear_ctrl_event_wait_failure_if_unused(&self, wait: CtrlEventWait) {
        if self
            .ctrl_event_waits
            .values()
            .any(|candidate| *candidate == wait)
        {
            return;
        }
        self.ctrl_event_wait_failures
            .lock()
            .remove(&(wait.syncpt_id, wait.threshold));
    }

    fn remove_ctrl_event_wait_slot(&mut self, fd: u32, slot: u32) {
        if let Some(wait) = self.ctrl_event_waits.remove(&(fd, slot & 0xFF)) {
            self.clear_ctrl_event_wait_failure_if_unused(wait);
        }
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
        let min = channel.syncpt_min;
        drop(channels);
        self.ordered_submit_max.insert(syncpt_id, threshold);
        if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
            log::info!(
                "[syncpt] reserve fd={} flags={:#x} incr_value={} incr={} id={} min={} max={}",
                fd,
                flags,
                increment_value,
                increment,
                syncpt_id,
                min,
                threshold
            );
        }
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

    fn ctrl_event_wait_escape(&mut self, syncpt_id: u32, threshold: u32) -> Option<u32> {
        let key = (syncpt_id, threshold);
        let min = self.syncpoint_value(syncpt_id);
        if syncpoint_reached(min, threshold) {
            self.ctrl_event_wait_failures.lock().remove(&key);
            return Some(min);
        }
        let max = self.syncpoint_max(syncpt_id);
        let submitted = syncpoint_reached(max, threshold);
        let action = {
            let mut failures = self.ctrl_event_wait_failures.lock();
            ctrl_event_wait_failure_action(
                failures.entry(key).or_default(),
                submitted,
                syncpoint_escape_drain_enabled(),
            )
        };
        if action.log {
            log::warn!(
                "[syncpt-stall] id={} threshold={} min={} max={} missing_increments={} submitted={}",
                syncpt_id,
                threshold,
                min,
                max,
                max.wrapping_sub(min),
                submitted
            );
        }
        if !action.drain {
            return None;
        }
        self.wait_gpu_idle();
        let settled = self.syncpoint_value(syncpt_id);
        if syncpoint_reached(settled, threshold) {
            self.ctrl_event_wait_failures.lock().remove(&key);
            log::warn!(
                "[syncpt-stall] id={} threshold={} released at {} after gpu drain",
                syncpt_id,
                threshold,
                settled
            );
            Some(settled)
        } else {
            None
        }
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

    fn submit_emits_increments_value(flags: u32, eager: bool) -> bool {
        eager || flags & (1 << 1) != 0
    }

    fn submit_emits_increments(flags: u32) -> bool {
        static EAGER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        Self::submit_emits_increments_value(
            flags,
            *EAGER.get_or_init(|| std::env::var_os("NEXIUM_EAGER_SUBMIT_FENCE").is_some()),
        )
    }

    fn channel_submit_completion(
        &self,
        fd: u32,
        syncpt_id: u32,
        threshold: u32,
    ) -> Box<dyn FnOnce() + Send> {
        let gpu = Arc::clone(&self.gpu);
        Box::new(move || {
            gpu.record_syncpoint_completion(fd, syncpt_id, threshold);
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
        let min = self.syncpoint_value(id);
        if syncpoint_reached(min, threshold) {
            return true;
        }
        let max = self.syncpoint_max(id);
        if syncpoint_expired(min, max, threshold) {
            static UNREACHABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = UNREACHABLE.fetch_add(1, Ordering::Relaxed);
            if n < 16 || n % 4096 == 0 {
                log::warn!(
                    "[syncpt] threshold beyond reserved max treated as expired id={} threshold={} min={} max={} (n={})",
                    id,
                    threshold,
                    min,
                    max,
                    n + 1
                );
            }
            return true;
        }
        false
    }

    pub fn queue_buffer_fence_disposition(&self, id: u32, threshold: u32) -> FenceWaitDisposition {
        let current = self.syncpoint_value(id);
        let ordered = self.ordered_submit_max.get(&id).copied();
        let disposition = classify_submit_fence_wait(current, ordered, threshold);
        if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
            log::info!(
                "[syncpt] queue-fence id={} threshold={} current={} ordered_max={:?} max={} -> {:?}",
                id,
                threshold,
                current,
                ordered,
                self.syncpoint_max(id),
                disposition
            );
        }
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

        if ioctl_profile_enabled() {
            let started = std::time::Instant::now();
            let outcome =
                self.dispatch_ioctl_for_device(device, cmd, &req, mem_read, mem_write, mem_copy);
            ioctl_profile_record(
                device,
                cmd,
                outcome.result,
                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            );
            return outcome;
        }
        self.dispatch_ioctl_for_device(device, cmd, &req, mem_read, mem_write, mem_copy)
    }

    fn dispatch_ioctl_for_device(
        &mut self,
        device: NvDevice,
        cmd: u16,
        req: &IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> IoctlOutcome {
        let reuses_video_mappings = matches!(device, NvDevice::NvhostNvdec | NvDevice::NvhostVic)
            && cmd == 0x0009
            && self.channel_map_reuses_addresses(req);
        if self.gpu_async.is_some()
            && ioctl_requires_async_gpu_drain(device, cmd)
            && !reuses_video_mappings
            && !self.wait_gpu_idle_checked()
        {
            log::error!(
                "nvdrv:Ioctl device={:?} cmd={:#06x} rejected after GPU drain failure",
                device,
                cmd
            );
            return IoctlOutcome::error(0xA);
        }
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

    fn channel_map_reuses_addresses(&self, req: &IoctlRequest) -> bool {
        let requested = read_u32(&req.in_data, 0).unwrap_or(0) as usize;
        let parsed = req.in_data.len().saturating_sub(0x0c) / 8;
        (0..requested.min(parsed)).all(|index| {
            let handle_id = read_u32(&req.in_data, 0x0c + index * 8).unwrap_or(0);
            self.nvmap_handles
                .get(&handle_id)
                .is_some_and(|handle| handle.channel_map_address != 0)
        })
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
        if size == 0 || cpu_address == 0 || cpu_address.checked_add(u64::from(size)).is_none() {
            return 0;
        }

        let allocation_size = u64::from(size).max(0x1000);
        let map_address = self.gpu.alloc_gpu_va(allocation_size);
        let Ok(map_address_u32) = u32::try_from(map_address) else {
            if map_address != 0 {
                let _ = self.gpu.free_va(map_address, allocation_size);
            }
            return 0;
        };
        if map_address_u32 == 0 {
            return 0;
        }

        self.gpu.mappings.write().add_with_va_ownership(
            map_address,
            u64::from(size),
            cpu_address,
            handle_id,
            (map_address, allocation_size),
        );
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

        match registers[CODEC_METHOD] {
            3 => {}
            9 => {
                self.process_nvdec_vp9_execute(fd, registers, runtime, mem_read);
                return;
            }
            other => {
                log::warn!("[video-decode] fd={} unsupported NVDEC codec {}", fd, other);
                return;
            }
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

        let ffmpeg_dims =
            if video_ffmpeg::enabled() && !runtime.ffmpeg_failed.load(Ordering::Relaxed) {
                match context
                    .frame_width()
                    .and_then(|width| context.frame_height().map(|height| (width, height)))
                {
                    Ok(dims) => Some(dims),
                    Err(error) => {
                        log::warn!(
                            "[video-decode] fd={} ffmpeg dims unavailable ({}), using OpenH264",
                            fd,
                            error
                        );
                        runtime.ffmpeg_failed.store(true, Ordering::Relaxed);
                        None
                    }
                }
            } else {
                None
            };

        let is_idr = packet
            .windows(4)
            .any(|word| word[..3] == [0, 0, 1] && word[3] & 31 == 5);
        if let Some((width, height)) = ffmpeg_dims {
            let config = (video_ffmpeg::FfmpegCodec::H264, width, height);
            if runtime.ffmpeg_config != Some(config) {
                runtime.ffmpeg_config = Some(config);
                self.video_decoder
                    .submit(video_decode_thread::DecodeWork::Configure {
                        fd,
                        codec: video_ffmpeg::FfmpegCodec::H264,
                        width,
                        height,
                        failed: runtime.ffmpeg_failed.clone(),
                    });
            }
            self.video_decoder
                .submit(video_decode_thread::DecodeWork::Packet {
                    fd,
                    packet,
                    target_luma_iova: luma_iova,
                    detail: video_decode_thread::PacketDetail::H264 {
                        picture_index: picture_index as u32,
                        guest_frame: context.parameter_set.frame_number,
                        picture_order: if context.parameter_set.field_picture {
                            context.field_order_count
                                [usize::from(context.parameter_set.bottom_field)]
                        } else {
                            context.field_order_count[0].min(context.field_order_count[1])
                        },
                        is_idr,
                    },
                });
            return;
        }

        let frame = {
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

        video_decode_thread::lock_frame_cache(self.video_decoder.cache()).insert(luma_iova, frame);

        static DECODED_FRAMES: AtomicU64 = AtomicU64::new(0);
        let frame_index = DECODED_FRAMES.fetch_add(1, Ordering::Relaxed);
        if frame_index < 32 || frame_index % 300 == 0 || video_trace_enabled() {
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

    fn process_nvdec_vp9_execute(
        &mut self,
        fd: u32,
        registers: &[u32; video_host1x::ENGINE_REGISTER_COUNT],
        runtime: &mut VideoChannelRuntime,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) {
        const PICTURE_INFO_METHOD: usize = 0x101;
        const BITSTREAM_METHOD: usize = 0x102;
        const SURFACE_LUMA_BASE_METHOD: usize = 0x10c;
        const VP9_PROB_TAB_METHOD: usize = 0x170;
        const MAX_BITSTREAM_SIZE: usize = 32 * 1024 * 1024;

        if !video_ffmpeg::enabled() {
            if !runtime.vp9_unavailable_logged {
                runtime.vp9_unavailable_logged = true;
                log::warn!(
                    "[video-decode] fd={} VP9 stream requires ffmpeg; place ffmpeg.exe next to the emulator, on PATH, or set NEXIUM_FFMPEG",
                    fd
                );
            }
            return;
        }

        let picture_iova = u64::from(registers[PICTURE_INFO_METHOD]) << 8;
        let prob_iova = u64::from(registers[VP9_PROB_TAB_METHOD]) << 8;
        let bitstream_iova = u64::from(registers[BITSTREAM_METHOD]) << 8;
        let Some(picture_cpu) = self.video_cpu_address(picture_iova) else {
            log::warn!(
                "[video-decode] fd={} unmapped vp9 picture info iova={:#x}",
                fd,
                picture_iova
            );
            return;
        };
        let Some(prob_cpu) = self.video_cpu_address(prob_iova) else {
            log::warn!(
                "[video-decode] fd={} unmapped vp9 prob tab iova={:#x}",
                fd,
                prob_iova
            );
            return;
        };
        let Some(bitstream_cpu) = self.video_cpu_address(bitstream_iova) else {
            log::warn!(
                "[video-decode] fd={} unmapped vp9 bitstream iova={:#x}",
                fd,
                bitstream_iova
            );
            return;
        };

        let mut picture_bytes = vec![0u8; video_vp9::VP9_PICTURE_INFO_SIZE];
        if !mem_read(picture_cpu, &mut picture_bytes) {
            log::warn!(
                "[video-decode] fd={} failed vp9 picture info read cpu={:#x}",
                fd,
                picture_cpu
            );
            return;
        }
        let mut info = match video_vp9::parse_picture_info(&picture_bytes) {
            Ok(info) => info,
            Err(error) => {
                log::warn!(
                    "[video-decode] fd={} invalid vp9 picture info: {}",
                    fd,
                    error
                );
                return;
            }
        };
        let mut prob_bytes = vec![0u8; video_vp9::VP9_ENTROPY_PROBS_SIZE];
        if !mem_read(prob_cpu, &mut prob_bytes) {
            log::warn!(
                "[video-decode] fd={} failed vp9 prob tab read cpu={:#x}",
                fd,
                prob_cpu
            );
            return;
        }
        let (entropy, seg_probs) = match video_vp9::parse_entropy_probs(&prob_bytes) {
            Ok(parsed) => parsed,
            Err(error) => {
                log::warn!("[video-decode] fd={} invalid vp9 prob tab: {}", fd, error);
                return;
            }
        };
        info.entropy = entropy;
        for index in 0..4 {
            info.frame_offsets[index] = u64::from(registers[SURFACE_LUMA_BASE_METHOD + index]) << 8;
        }

        let bitstream_len = info.bitstream_size as usize;
        if bitstream_len == 0 || bitstream_len > MAX_BITSTREAM_SIZE {
            log::warn!(
                "[video-decode] fd={} invalid vp9 bitstream size {}",
                fd,
                bitstream_len
            );
            return;
        }
        let mut bitstream = vec![0u8; bitstream_len];
        if !mem_read(bitstream_cpu, &mut bitstream) {
            log::warn!(
                "[video-decode] fd={} failed vp9 bitstream read cpu={:#x} bytes={}",
                fd,
                bitstream_cpu,
                bitstream_len
            );
            return;
        }

        let width = info.frame_width as u32;
        let height = info.frame_height as u32;
        if info.y_dc_delta_q != 0 || info.uv_dc_delta_q != 0 || info.uv_ac_delta_q != 0 {
            static DELTA_Q_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !DELTA_Q_WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "[video-decode] fd={} vp9 stream uses nonzero delta_q ({},{},{}); composed header encoding for this case is unverified",
                    fd,
                    info.y_dc_delta_q,
                    info.uv_dc_delta_q,
                    info.uv_ac_delta_q
                );
            }
        }
        let config = (video_ffmpeg::FfmpegCodec::Vp9, width, height);
        if runtime.ffmpeg_config != Some(config) {
            runtime.ffmpeg_config = Some(config);
            runtime.vp9_packet_target = None;
            runtime
                .ffmpeg_failed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.video_decoder
                .submit(video_decode_thread::DecodeWork::Configure {
                    fd,
                    codec: video_ffmpeg::FfmpegCodec::Vp9,
                    width,
                    height,
                    failed: runtime.ffmpeg_failed.clone(),
                });
        }
        if runtime
            .ffmpeg_failed
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }

        let current_luma_iova = u64::from(registers[SURFACE_LUMA_BASE_METHOD + 3]) << 8;
        let packet_luma_iova =
            next_vp9_packet_target(&mut runtime.vp9_packet_target, current_luma_iova);
        let (packet, show_frame) = runtime.vp9_composer.compose(info, bitstream, &seg_probs);

        self.video_decoder
            .submit(video_decode_thread::DecodeWork::Packet {
                fd,
                packet,
                target_luma_iova: packet_luma_iova,
                detail: video_decode_thread::PacketDetail::Vp9 { show_frame },
            });
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

        let selected = self
            .video_decoder
            .wait_for_frame(input_luma_iova, std::time::Duration::from_millis(30))
            .map(|frame| (true, frame))
            .or_else(|| {
                let mut cache = video_decode_thread::lock_frame_cache(self.video_decoder.cache());
                if let Some(frame) = cache.take(input_luma_iova) {
                    return Some((true, frame));
                }
                cache.expire_decode(input_luma_iova);
                cache.latest_cloned().map(|(_, frame)| (false, frame))
            });
        let Some((exact_frame, frame)) = selected else {
            log::warn!(
                "[video-vic] fd={} no decoded frame for input luma={:#x}",
                fd,
                input_luma_iova
            );
            return;
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
        let mut successful_write_count = 0;
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
                break;
            }
            successful_write_count += 1;
        }
        let successful_writes = &mapped_writes[..successful_write_count];
        for (_, _, aliases) in successful_writes {
            for &(alias, size) in aliases {
                nexium_gpu::tex_invalidate::bump_region(alias, size);
            }
        }
        if let Some(renderer) = self.renderer.get().and_then(|renderer| renderer.as_ref()) {
            for (_, _, aliases) in successful_writes {
                for &(alias, _) in aliases {
                    renderer.invalidate_texture_address(alias);
                }
            }
            let invalidations = successful_writes
                .iter()
                .map(|(write, cpu_addr, aliases)| {
                    (*cpu_addr, write.bytes.len() as u64, aliases.clone())
                })
                .collect::<Vec<_>>();
            let renderer = Arc::clone(renderer);
            let job = move || {
                for (cpu_addr, size, aliases) in invalidations {
                    renderer.invalidate_render_target_range(cpu_addr, size, &aliases);
                }
            };
            if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
                render_thread.submit_named("vic-guest-write-invalidate", Box::new(job));
            } else {
                job();
            }
        }
        if successful_write_count != mapped_writes.len() {
            return;
        }

        if output_is_nv12 {
            video_decode_thread::lock_frame_cache(self.video_decoder.cache())
                .insert(output_luma_iova, frame);
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
        if frame_index < 32 || frame_index % 300 == 0 || video_trace_enabled() {
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
                        user_refcount: 1,
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
                    if let Some(handle) = self.nvmap_handles.get_mut(&id) {
                        handle.user_refcount = handle.user_refcount.saturating_add(1);
                    }
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
                    let final_reference = self
                        .nvmap_handles
                        .get(&handle)
                        .is_some_and(|entry| entry.user_refcount <= 1);
                    let (address, size, flags) = if final_reference {
                        let removed = self.nvmap_handles.remove(&handle).unwrap();
                        (removed.address, removed.size, 0u32)
                    } else if let Some(entry) = self.nvmap_handles.get_mut(&handle) {
                        entry.user_refcount -= 1;
                        (0, entry.size, 1u32)
                    } else {
                        (0, 0, 0u32)
                    };
                    out[8..16].copy_from_slice(&address.to_le_bytes());
                    out[16..20].copy_from_slice(&size.to_le_bytes());
                    out[20..24].copy_from_slice(&flags.to_le_bytes());
                    log::debug!(
                        "nvmap:Free handle={} address={:#x} size={} flags={}",
                        handle,
                        address,
                        size,
                        flags
                    );
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
            0x4712 => {
                if out.len() < 4 {
                    out.resize(4, 0);
                }
                out[0..4].copy_from_slice(&2u32.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:NumVsms → 2");
            }
            0x4713 => {
                if out.len() < 8 + 2 * MAX_VSMS {
                    out.resize(8 + 2 * MAX_VSMS, 0);
                }
                out[0] = 0;
                out[1] = 0;
                out[2] = 0;
                out[3] = 1;
                for index in 0..MAX_VSMS {
                    out[8 + 2 * index] = 0;
                    out[8 + 2 * index + 1] = index as u8;
                }
                log::debug!("nvhost-ctrl-gpu:VsmsMapping → sm0=(0,0) sm1=(0,1)");
            }
            0x4714 => {
                if out.len() >= 8 {
                    out[0..4].copy_from_slice(&0x07u32.to_le_bytes());
                    out[4..8].copy_from_slice(&0x01u32.to_le_bytes());
                }
                let legacy = !matches!(
                    std::env::var("NEXIUM_LEGACY_GFX").ok().as_deref(),
                    Some("0" | "off" | "false")
                );
                if legacy {
                    self.legacy_gfx.store(true, Ordering::Relaxed);
                }
                log::debug!(
                    "nvhost-ctrl-gpu:GetActiveSlotMask → slot=7 mask=1 (legacy_gfx={})",
                    legacy
                );
            }
            0x471c => {
                if out.len() < 16 {
                    out.resize(16, 0);
                }
                let ns = gpu::clock::nanoseconds();
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
                let total_size = u64::from(pages) * u64::from(page_size);
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
                if total_size == 0 {
                    return IoctlOutcome::error(0xB);
                }
                let big_page_size = self.as_gpu_states.entry(req.fd).or_default().big_page_size;
                if page_size != AS_GPU_SMALL_PAGE_SIZE && page_size != big_page_size {
                    return IoctlOutcome::error(0xB);
                }
                let sparse = (flags & 0x2) != 0;
                let big_pages = page_size == big_page_size;
                if sparse && !big_pages {
                    return IoctlOutcome::error(NVRESULT_NOT_IMPLEMENTED);
                }
                let alloc = if (flags & 0x1) != 0 {
                    if offset_in == 0 || !self.gpu.alloc_va_fixed_exclusive(offset_in, total_size) {
                        return IoctlOutcome::error(0xB);
                    }
                    offset_in
                } else {
                    self.gpu.alloc_va(total_size, big_pages)
                };
                if alloc == 0 {
                    return IoctlOutcome::error(0xB);
                }
                let allocation = AsGpuAllocation {
                    base: alloc,
                    size: total_size,
                    page_size,
                    sparse,
                    big_pages,
                };
                let state = self.as_gpu_states.get_mut(&req.fd).unwrap();
                if state.allocations.insert(alloc, allocation).is_some() {
                    let _ = self.gpu.free_va(alloc, total_size);
                    return IoctlOutcome::error(0xB);
                }
                state.initialized = true;
                if sparse {
                    let mapping_update = self
                        .gpu
                        .mappings
                        .write()
                        .add_sparse_as_gpu_with_metadata(req.fd, alloc, total_size);
                    self.invalidate_texture_mapping_update(&mapping_update);
                }
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
                    let removed = self
                        .gpu
                        .mappings
                        .write()
                        .unmap_as_gpu_with_metadata(req.fd, gpu_va);
                    if let Some(removed) = removed {
                        if let Some((owned_gpu_va, owned_size)) = removed.owned_va_range {
                            let _ = self.gpu.free_va(owned_gpu_va, owned_size);
                        }
                        let mapping_update = gpu::GpuMappingUpdate {
                            change: gpu::GpuMappingChange::Replaced,
                            changed_gpu_ranges: removed.changed_gpu_ranges,
                            epoch_transitions: removed.epoch_transitions,
                        };
                        self.invalidate_texture_mapping_update(&mapping_update);
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
                    let buffer_offset = i64::from_le_bytes([
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
                        let source = self
                            .gpu
                            .mappings
                            .read()
                            .remap_source_starting_at(req.fd, requested_offset);
                        let Some((source_cpu, source_size, source_nvmap, source_record_id)) =
                            source
                        else {
                            log::warn!(
                                "nvhost-as-gpu:MapBufferEx remap rejected base={:#x} buffer_offset={:#x} size={:#x}",
                                requested_offset,
                                buffer_offset,
                                mapping_size_in,
                            );
                            return IoctlOutcome::error(0xB);
                        };
                        if mapping_size_in == 0 || source_size < mapping_size_in {
                            return IoctlOutcome::error(0xB);
                        }
                        let Some(remap_va) = requested_offset.checked_add_signed(buffer_offset)
                        else {
                            return IoctlOutcome::error(0xB);
                        };
                        let Some(cpu_addr) = source_cpu.checked_add_signed(buffer_offset) else {
                            return IoctlOutcome::error(0xB);
                        };
                        if remap_va.checked_add(mapping_size_in).is_none()
                            || cpu_addr.checked_add(mapping_size_in).is_none()
                        {
                            return IoctlOutcome::error(0xB);
                        }
                        let mapping_update = self.gpu.mappings.write().add_owned(
                            req.fd,
                            remap_va,
                            mapping_size_in,
                            cpu_addr,
                            source_nvmap,
                            source_record_id,
                        );
                        self.invalidate_texture_mapping_update(&mapping_update);
                        self.as_gpu_states.entry(req.fd).or_default().initialized = true;
                        if out.len() >= 40 {
                            out[32..40].copy_from_slice(&requested_offset.to_le_bytes());
                        }
                        log::debug!(
                            "nvhost-as-gpu:MapBufferEx remap base={:#x} buffer_offset={:#x} gpu_va={:#x} cpu={:#x} size={:#x}",
                            requested_offset,
                            buffer_offset,
                            remap_va,
                            cpu_addr,
                            mapping_size_in,
                        );
                        return IoctlOutcome::ok(out);
                    }

                    let Some(handle) = self.nvmap_handles.get(&nvmap_id) else {
                        return IoctlOutcome::error(0xB);
                    };
                    if handle.address == 0 {
                        return IoctlOutcome::error(0xB);
                    }
                    let mapping_size = if mapping_size_in == 0 {
                        u64::from(handle.size)
                    } else {
                        mapping_size_in
                    };
                    let Some(handle_cpu) = handle.address.checked_add_signed(buffer_offset) else {
                        return IoctlOutcome::error(0xB);
                    };
                    if mapping_size == 0
                        || handle_cpu.checked_add(mapping_size).is_none()
                        || ((flags & 0x1) != 0
                            && requested_offset != 0
                            && requested_offset
                                .checked_add(mapping_size.max(0x1000))
                                .is_none())
                    {
                        return IoctlOutcome::error(0xB);
                    }
                    let fixed = (flags & 0x1) != 0;
                    let big_page_size = u64::from(
                        self.as_gpu_states
                            .get(&req.fd)
                            .map(|state| state.big_page_size)
                            .unwrap_or(AS_GPU_DEFAULT_BIG_PAGE_SIZE),
                    );
                    let handle_align = u64::from(handle.align);
                    let mapping_page_size = if handle_align % big_page_size == 0 {
                        big_page_size
                    } else if handle_align % u64::from(AS_GPU_SMALL_PAGE_SIZE) == 0 {
                        u64::from(AS_GPU_SMALL_PAGE_SIZE)
                    } else {
                        return IoctlOutcome::error(0xB);
                    };
                    let Some(owned_size) = mapping_size
                        .checked_add(mapping_page_size - 1)
                        .map(|value| value & !(mapping_page_size - 1))
                    else {
                        return IoctlOutcome::error(0xB);
                    };
                    let (gpu_va, allocation_base) = if fixed {
                        let Some(allocation) = self.as_gpu_allocation_containing(
                            req.fd,
                            requested_offset,
                            mapping_size,
                        ) else {
                            return IoctlOutcome::error(0xB);
                        };
                        if requested_offset == 0 {
                            return IoctlOutcome::error(0xB);
                        }
                        (requested_offset, Some(allocation.base))
                    } else {
                        (
                            self.gpu
                                .alloc_va_with_page_size(owned_size, mapping_page_size),
                            None,
                        )
                    };
                    let cpu_addr = handle_cpu;
                    log::debug!(
                        "nvhost-as-gpu:MapBufferEx flags={:#x} nvmap_id={} req_off={:#x} cpu_addr={:#x} size={:#x} → gpu_va={:#x}",
                        flags,
                        nvmap_id,
                        requested_offset,
                        cpu_addr,
                        mapping_size,
                        gpu_va
                    );

                    if gpu_va == 0 || gpu_va.checked_add(mapping_size).is_none() {
                        if !fixed && gpu_va != 0 {
                            let _ = self.gpu.free_va(gpu_va, owned_size);
                        }
                        return IoctlOutcome::error(0xB);
                    }
                    let mapping_update = if fixed {
                        self.gpu.mappings.write().add_as_gpu_mapping(
                            req.fd,
                            gpu_va,
                            mapping_size,
                            cpu_addr,
                            nvmap_id,
                            None,
                            allocation_base,
                            true,
                        )
                    } else {
                        self.gpu.mappings.write().add_as_gpu_mapping(
                            req.fd,
                            gpu_va,
                            mapping_size,
                            cpu_addr,
                            nvmap_id,
                            Some((gpu_va, owned_size)),
                            None,
                            true,
                        )
                    };
                    self.invalidate_texture_mapping_update(&mapping_update);
                    self.as_gpu_states.entry(req.fd).or_default().initialized = true;

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
                let big_page = self
                    .as_gpu_states
                    .get(&req.fd)
                    .map(|state| state.big_page_size)
                    .unwrap_or(AS_GPU_DEFAULT_BIG_PAGE_SIZE);
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
                if req.in_data.len() < 40 {
                    return IoctlOutcome::error(0xB);
                }
                let big_page_size = u32::from_le_bytes(req.in_data[8..12].try_into().unwrap());
                let va_start = u64::from_le_bytes(req.in_data[16..24].try_into().unwrap());
                let va_end = u64::from_le_bytes(req.in_data[24..32].try_into().unwrap());
                let va_split = u64::from_le_bytes(req.in_data[32..40].try_into().unwrap());
                let state = self.as_gpu_states.entry(req.fd).or_default();
                if state.initialized {
                    return IoctlOutcome::error(0x8);
                }
                let supported_range = (va_start == 0 && va_end == 0 && va_split == 0)
                    || (va_start == 0x0400_0000
                        && va_end == 1u64 << 37
                        && va_split == 1u64 << 34);
                if !state.allocations.is_empty()
                    || !supported_range
                    || (big_page_size != 0
                        && (!big_page_size.is_power_of_two() || (big_page_size & 0x30000) == 0))
                {
                    log::warn!(
                        "nvhost-as-gpu:AllocAsEx rejected big_page_size={:#x} va_start={:#x} va_end={:#x} va_split={:#x} allocations={}",
                        big_page_size,
                        va_start,
                        va_end,
                        va_split,
                        state.allocations.len(),
                    );
                    return IoctlOutcome::error(0xB);
                }
                if big_page_size != 0 {
                    state.big_page_size = big_page_size;
                }
                state.initialized = true;
                log::debug!(
                    "nvhost-as-gpu:AllocAsEx big_page_size={:#x} va_start={:#x} va_end={:#x} va_split={:#x} in_len={}",
                    big_page_size,
                    va_start,
                    va_end,
                    va_split,
                    req.in_data.len()
                );
            }
            0x4103 => {
                if req.in_data.len() < 16 {
                    return IoctlOutcome::error(0xB);
                }
                let gpu_va = u64::from_le_bytes(req.in_data[0..8].try_into().unwrap());
                let pages = u32::from_le_bytes(req.in_data[8..12].try_into().unwrap());
                let page_size = u32::from_le_bytes(req.in_data[12..16].try_into().unwrap());
                let size = u64::from(pages) * u64::from(page_size);
                if size == 0
                    || !self.release_as_gpu_allocation(req.fd, gpu_va, Some((size, page_size)))
                {
                    return IoctlOutcome::error(0xB);
                }
                log::debug!(
                    "nvhost-as-gpu:FreeSpace gpu_va={:#x} size={:#x}",
                    gpu_va,
                    size
                );
            }
            0x4114 => {
                if req.in_data.len() % 20 != 0 {
                    return IoctlOutcome::error(0xB);
                }
                let big_page_size = u64::from(
                    self.as_gpu_states
                        .get(&req.fd)
                        .map(|state| state.big_page_size)
                        .unwrap_or(AS_GPU_DEFAULT_BIG_PAGE_SIZE),
                );
                let num_entries = req.in_data.len() / 20;
                let mut actions = Vec::with_capacity(num_entries);
                for i in 0..num_entries {
                    let off = i * 20;
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
                    let Some(gpu_va) = u64::from(as_offset_big_pages).checked_mul(big_page_size)
                    else {
                        return IoctlOutcome::error(0xB);
                    };
                    let Some(size) = u64::from(big_pages).checked_mul(big_page_size) else {
                        return IoctlOutcome::error(0xB);
                    };
                    let Some(handle_off) =
                        u64::from(handle_offset_big_pages).checked_mul(big_page_size)
                    else {
                        return IoctlOutcome::error(0xB);
                    };
                    let Some(allocation) = self.as_gpu_allocation_containing(req.fd, gpu_va, size)
                    else {
                        return IoctlOutcome::error(0xB);
                    };
                    if size == 0
                        || !allocation.sparse
                        || !allocation.big_pages
                        || u64::from(allocation.page_size) != big_page_size
                    {
                        return IoctlOutcome::error(0xB);
                    }
                    if nvmap_handle == 0 {
                        actions.push((gpu_va, size, None));
                        continue;
                    }
                    let Some(handle) = self.nvmap_handles.get(&nvmap_handle) else {
                        return IoctlOutcome::error(0xB);
                    };
                    if handle.address == 0 {
                        return IoctlOutcome::error(0xB);
                    }
                    if handle_off
                        .checked_add(size)
                        .is_none_or(|end| end > u64::from(handle.size))
                    {
                        return IoctlOutcome::error(0xB);
                    }
                    let Some(cpu_addr) = handle.address.checked_add(handle_off) else {
                        return IoctlOutcome::error(0xB);
                    };
                    if cpu_addr.checked_add(size).is_none() {
                        return IoctlOutcome::error(0xB);
                    }
                    log::debug!(
                        "nvhost-as-gpu:Remap[{}/{}] nvmap_id={} cpu={:#x} → gpu_va={:#x} size={:#x}",
                        i,
                        num_entries,
                        nvmap_handle,
                        cpu_addr,
                        gpu_va,
                        size
                    );
                    actions.push((gpu_va, size, Some((cpu_addr, nvmap_handle))));
                }
                let mutated = !actions.is_empty();
                for (gpu_va, size, mapped) in actions {
                    let mapping_update = if let Some((cpu_addr, nvmap_handle)) = mapped {
                        self.gpu.mappings.write().add_as_gpu_mapping(
                            req.fd,
                            gpu_va,
                            size,
                            cpu_addr,
                            nvmap_handle,
                            None,
                            None,
                            false,
                        )
                    } else {
                        self.gpu
                            .mappings
                            .write()
                            .add_sparse_as_gpu_with_metadata(req.fd, gpu_va, size)
                    };
                    self.invalidate_texture_mapping_update(&mapping_update);
                }
                if mutated {
                    self.as_gpu_states.entry(req.fd).or_default().initialized = true;
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
                        if kick_timeline_enabled() {
                            log::warn!(
                                "[ktl] us={} submit fd={} syncpt={} target={} entries={}",
                                timeline_us(),
                                req.fd,
                                syncpt_id,
                                syncpt_value,
                                entries.len()
                            );
                        }
                        let async_enabled = self.gpu_async.is_some();
                        let queued = self.gpu_async.as_ref().is_some_and(|queue| {
                            queue.submit(AsyncGpuSubmission::Inline {
                                entries: entries.clone(),
                                completion: Self::submit_emits_increments(submit_flags).then_some(
                                    AsyncGpuCompletion {
                                        fd: req.fd,
                                        syncpt_id,
                                        threshold: syncpt_value,
                                    },
                                ),
                            })
                        });
                        if async_enabled && !queued {
                            log::error!("nvhost-gpu: async inline submission failed");
                            return IoctlOutcome::error(0xA);
                        }
                        if !queued {
                            let on_complete =
                                Self::submit_emits_increments(submit_flags).then(|| {
                                    self.channel_submit_completion(req.fd, syncpt_id, syncpt_value)
                                });
                            let _ = self.gpu.process_inline_gpfifo(
                                &entries,
                                mem_read,
                                mem_write,
                                mem_copy,
                                on_complete,
                            );
                            if !self.finish_sync_prep_submit("inline submission") {
                                return IoctlOutcome::error(0xA);
                            }
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
                        if std::env::var_os("NEXIUM_ENTRY_MIRROR").is_some() && address != 0 {
                            let mut mirror = vec![0u8; num_entries as usize * 8];
                            let read_ok = mem_read(address, &mut mirror);
                            let guest: Vec<gpu::CommandListHeader> = mirror
                                .chunks_exact(8)
                                .map(|c| gpu::CommandListHeader {
                                    address_lo: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                                    address_hi_and_count: u32::from_le_bytes([
                                        c[4], c[5], c[6], c[7],
                                    ]),
                                })
                                .collect();
                            let mismatches: Vec<String> = entries
                                .iter()
                                .zip(guest.iter())
                                .enumerate()
                                .filter(|(_, (a, b))| {
                                    a.address_lo != b.address_lo
                                        || a.address_hi_and_count != b.address_hi_and_count
                                })
                                .map(|(i, (a, b))| {
                                    format!(
                                        "{}:ioctl({:#x},n={})!=guest({:#x},n={})",
                                        i,
                                        a.address(),
                                        a.entry_count(),
                                        b.address(),
                                        b.entry_count()
                                    )
                                })
                                .collect();
                            if !mismatches.is_empty() || !read_ok {
                                log::error!(
                                    "[entry-mirror] params_addr={:#x} read_ok={} n={} mismatches={} {}",
                                    address,
                                    read_ok,
                                    num_entries,
                                    mismatches.len(),
                                    mismatches.join(" ")
                                );
                            }
                        }
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .gpfifo_entries
                            .fetch_add(entries.len() as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) =
                            self.reserve_channel_submit(req.fd, submit_flags, submit_fence_value);
                        if kick_timeline_enabled() {
                            log::warn!(
                                "[ktl] us={} submit fd={} syncpt={} target={} entries={}",
                                timeline_us(),
                                req.fd,
                                syncpt_id,
                                syncpt_value,
                                entries.len()
                            );
                        }
                        let async_enabled = self.gpu_async.is_some();
                        let queued = self.gpu_async.as_ref().is_some_and(|queue| {
                            queue.submit(AsyncGpuSubmission::Inline {
                                entries: entries.clone(),
                                completion: Self::submit_emits_increments(submit_flags).then_some(
                                    AsyncGpuCompletion {
                                        fd: req.fd,
                                        syncpt_id,
                                        threshold: syncpt_value,
                                    },
                                ),
                            })
                        });
                        if async_enabled && !queued {
                            log::error!("nvhost-gpu: async embedded submission failed");
                            return IoctlOutcome::error(0xA);
                        }
                        if !queued {
                            let on_complete =
                                Self::submit_emits_increments(submit_flags).then(|| {
                                    self.channel_submit_completion(req.fd, syncpt_id, syncpt_value)
                                });
                            let _ = self.gpu.process_inline_gpfifo(
                                &entries,
                                mem_read,
                                mem_write,
                                mem_copy,
                                on_complete,
                            );
                            if !self.finish_sync_prep_submit("embedded submission") {
                                return IoctlOutcome::error(0xA);
                            }
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
                        let async_entries = self.gpu_async.as_ref().and_then(|_| {
                            self.gpu
                                .snapshot_gpfifo_entries(address, num_entries, mem_read)
                        });
                        let queued = self
                            .gpu_async
                            .as_ref()
                            .zip(async_entries.as_ref())
                            .is_some_and(|(queue, entries)| {
                                queue.submit(AsyncGpuSubmission::Inline {
                                    entries: entries.clone(),
                                    completion: Self::submit_emits_increments(submit_flags)
                                        .then_some(AsyncGpuCompletion {
                                            fd: req.fd,
                                            syncpt_id,
                                            threshold: syncpt_value,
                                        }),
                                })
                            });
                        if async_entries.is_some() && !queued {
                            log::error!("nvhost-gpu: async kickoff submission failed");
                            return IoctlOutcome::error(0xA);
                        }
                        if !queued {
                            let on_complete =
                                Self::submit_emits_increments(submit_flags).then(|| {
                                    self.channel_submit_completion(req.fd, syncpt_id, syncpt_value)
                                });
                            if let Some(entries) = async_entries.as_ref() {
                                let _ = self.gpu.process_inline_gpfifo(
                                    entries,
                                    mem_read,
                                    mem_write,
                                    mem_copy,
                                    on_complete,
                                );
                            } else {
                                let _ = self.gpu.submit_gpfifo(
                                    address,
                                    num_entries,
                                    mem_read,
                                    mem_write,
                                    mem_copy,
                                    on_complete,
                                );
                            }
                            if !self.finish_sync_prep_submit("kickoff submission") {
                                return IoctlOutcome::error(0xA);
                            }
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
                    self.remove_ctrl_event_wait_slot(req.fd, event_id);
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
                        self.ctrl_event_wait_failures
                            .lock()
                            .remove(&(syncpt_id, threshold));
                        out[12..16].copy_from_slice(&current_val.to_le_bytes());
                        log::debug!(
                            "nvhost-ctrl:EventWait syncpt={} threshold={:#x} current={} → Success (already reached)",
                            syncpt_id,
                            threshold,
                            current_val
                        );
                    } else if let Some(settled) = self.ctrl_event_wait_escape(syncpt_id, threshold)
                    {
                        Self::fence_wait_stat(false);
                        out[12..16].copy_from_slice(&settled.to_le_bytes());
                    } else {
                        Self::fence_wait_stat(true);
                        let slot = self.next_ctrl_event_slot & 63;
                        self.next_ctrl_event_slot = self.next_ctrl_event_slot.wrapping_add(1);
                        let event_val: u32 = slot | ((syncpt_id & 0xFFF) << 16) | (1 << 28);
                        let replaced = self.ctrl_event_waits.insert(
                            (req.fd, slot),
                            CtrlEventWait {
                                syncpt_id,
                                threshold,
                            },
                        );
                        if let Some(replaced) = replaced {
                            self.clear_ctrl_event_wait_failure_if_unused(replaced);
                        }
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
                    let escaped = if syncpoint_reached(current_val, threshold) {
                        Some(current_val)
                    } else {
                        self.ctrl_event_wait_escape(syncpt_id, threshold)
                    };
                    if let Some(settled) = escaped {
                        Self::fence_wait_stat(false);
                        self.remove_ctrl_event_wait_slot(req.fd, event_id);
                        self.ctrl_event_wait_failures
                            .lock()
                            .remove(&(syncpt_id, threshold));
                        if out.len() >= 16 {
                            out[12..16].copy_from_slice(&settled.to_le_bytes());
                        }
                        log::debug!(
                            "nvhost-ctrl:EventWaitAsync syncpt={} threshold={:#x} current={} event_id={} → Success",
                            syncpt_id,
                            threshold,
                            settled,
                            event_id
                        );
                    } else {
                        Self::fence_wait_stat(true);
                        if out.len() >= 16 {
                            out[12..16].copy_from_slice(&event_id.to_le_bytes());
                        }
                        let replaced = self.ctrl_event_waits.insert(
                            (req.fd, event_id & 0xFF),
                            CtrlEventWait {
                                syncpt_id,
                                threshold,
                            },
                        );
                        if let Some(replaced) = replaced {
                            self.clear_ctrl_event_wait_failure_if_unused(replaced);
                        }
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
                    self.remove_ctrl_event_wait_slot(req.fd, event_id);
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
        let frames = self.frame_queue.drain();
        self.stats
            .frames_drained
            .fetch_add(frames.len() as u64, Ordering::Relaxed);
        frames
    }

    pub fn drain_next_frame(&self) -> Option<QueuedFrame> {
        self.drain_next_frame_due(std::time::Instant::now())
    }

    pub fn drain_next_frame_due(&self, now: std::time::Instant) -> Option<QueuedFrame> {
        let frame = self.frame_queue.pop_front_due(now);
        if frame.is_some() {
            self.stats.frames_drained.fetch_add(1, Ordering::Relaxed);
        }
        frame
    }

    pub fn submit_frame(&self, frame: QueuedFrame) {
        self.queue_buffer_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        enqueue_bounded_frame(&self.frame_queue, &self.stats, frame);
    }

    pub fn submit_frame_nonblocking(&self, frame: QueuedFrame) -> bool {
        self.queue_buffer_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let frame_queue = Arc::clone(&self.frame_queue);
        let stats = Arc::clone(&self.stats);
        crate::render_thread::present_thread().submit_named(
            "cpu-present-frame",
            Box::new(move || enqueue_bounded_frame(&frame_queue, &stats, frame)),
        )
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
                present_at: None,
                depth: None,
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
            present_at: None,
            depth: None,
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

impl Drop for Nvdrv {
    fn drop(&mut self) {
        if self.sync_prep_thread {
            self.sync_prep_thread = false;
            if !self.gpu.shutdown_prep_thread(false) {
                log::error!("nexium-nvdrv: synchronous GPU prep thread shutdown failed");
            }
        }
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

    struct CloseTestFrameQueue<'a>(&'a FrameQueueState);

    impl Drop for CloseTestFrameQueue<'_> {
        fn drop(&mut self) {
            self.0.close();
        }
    }

    fn install_test_sync_prep_thread(nvdrv: &mut Nvdrv) {
        let read: AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: AsyncMemoryCopy = Arc::new(|_, _, _| true);
        assert!(nvdrv.gpu.install_prep_thread(
            gpu::prep::PrepThreadResources {
                maxwell_dma: Arc::clone(&nvdrv.gpu.maxwell_dma),
                fermi_2d: Arc::clone(&nvdrv.gpu.fermi_2d),
                kepler_compute: Arc::clone(&nvdrv.gpu.kepler_compute),
                kepler_memory: Arc::clone(&nvdrv.gpu.kepler_memory),
                mappings: Arc::clone(&nvdrv.gpu.mappings),
                stats: Arc::clone(&nvdrv.gpu.stats),
                mem_read: read,
                mem_write: write,
                mem_copy: copy,
            },
            gpu::prep::PrepThreadBehavior::DrainEachKick,
        ));
        nvdrv.sync_prep_thread = true;
    }

    fn test_frame(value: u8) -> QueuedFrame {
        QueuedFrame {
            width: 1,
            height: 1,
            pixels: vec![value, 0, 0, 255],
            present_at: None,
            depth: None,
        }
    }

    #[test]
    fn sync_prep_mode_is_opt_in_and_mutually_exclusive() {
        assert_eq!(
            gpu_thread_mode_request(None, None, None),
            GpuThreadModeRequest::AsyncGpu
        );
        for disabled in ["", "0", "false", "off", "no", "unexpected"] {
            assert_eq!(
                gpu_thread_mode_request(None, Some(disabled), None),
                GpuThreadModeRequest::Inline
            );
        }
        assert_eq!(
            gpu_thread_mode_request(Some("1"), None, None),
            GpuThreadModeRequest::SyncPrepThread
        );
        assert_eq!(
            gpu_thread_mode_request(None, Some("true"), Some("on")),
            GpuThreadModeRequest::AsyncGpu
        );
        assert_eq!(
            gpu_thread_mode_request(Some("yes"), Some("1"), None),
            GpuThreadModeRequest::Conflict
        );
        assert_eq!(
            gpu_thread_mode_request(Some("on"), None, Some("true")),
            GpuThreadModeRequest::Conflict
        );
        for disabled in ["", "0", "false", "off", "no", "unexpected"] {
            assert_eq!(
                gpu_thread_mode_request(Some(disabled), None, None),
                GpuThreadModeRequest::AsyncGpu
            );
            assert_eq!(
                gpu_thread_mode_request(Some(disabled), Some("0"), None),
                GpuThreadModeRequest::Inline
            );
        }
    }

    #[test]
    fn sync_prep_idle_wait_drains_kick_without_async_queue() {
        let mut nvdrv = Nvdrv::new();
        install_test_sync_prep_thread(&mut nvdrv);
        let completed = Arc::new(AtomicBool::new(false));
        let completed_callback = Arc::clone(&completed);

        nvdrv.gpu.process_inline_gpfifo(
            &[],
            |_, bytes| {
                bytes.fill(0);
                true
            },
            |_, _| true,
            |_, _, _| true,
            Some(Box::new(move || {
                completed_callback.store(true, Ordering::Release)
            })),
        );

        assert!(nvdrv.gpu_async.is_none());
        assert!(nvdrv.wait_gpu_idle_checked());
        assert!(completed.load(Ordering::Acquire));
    }

    #[test]
    fn sync_prep_drop_joins_and_restores_inline_lane() {
        let mut nvdrv = Nvdrv::new();
        let gpu = Arc::clone(&nvdrv.gpu);
        install_test_sync_prep_thread(&mut nvdrv);

        assert!(nvdrv.gpu_async.is_none());
        assert!(gpu.pusher.lock().prep.is_threaded());
        drop(nvdrv);
        assert!(!gpu.pusher.lock().prep.is_threaded());
    }

    #[test]
    fn sync_prep_submit_propagates_latched_worker_failure() {
        let mut nvdrv = Nvdrv::new();
        install_test_sync_prep_thread(&mut nvdrv);
        {
            let pusher = nvdrv.gpu.pusher.lock();
            let gpu::prep::PrepLane::Threaded(handle) = &pusher.prep else {
                panic!("missing prep thread")
            };
            handle.failure_latch().store(true, Ordering::Release);
        }

        assert!(!nvdrv.finish_sync_prep_submit("test submission"));
        assert!(!nvdrv.gpu.shutdown_prep_thread(false));
        nvdrv.sync_prep_thread = false;
        assert!(!nvdrv.gpu.pusher.lock().prep.is_threaded());
    }

    #[test]
    fn async_pending_guard_releases_when_completion_is_discarded() {
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (guard, inflight) =
            AsyncGpuPendingGuard::reserve(Arc::clone(&pending), Arc::clone(&failed));

        assert_eq!(inflight, 1);

        drop(guard);

        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert!(failed.load(Ordering::Acquire));
    }

    #[test]
    fn async_pending_guard_records_success_without_faulting_queue() {
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (guard, inflight) =
            AsyncGpuPendingGuard::reserve(Arc::clone(&pending), Arc::clone(&failed));

        assert_eq!(inflight, 1);

        guard.complete();

        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert!(!failed.load(Ordering::Acquire));
    }

    #[test]
    fn queued_pending_guard_releases_after_post_empty_channel_drop() {
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = crossbeam::channel::bounded(1);
        assert!(rx.is_empty());
        let (guard, _) = AsyncGpuPendingGuard::reserve(Arc::clone(&pending), Arc::clone(&failed));
        let queued = QueuedAsyncGpuSubmission::tracked(
            AsyncGpuSubmission::Inline {
                entries: Vec::new(),
                completion: None,
            },
            guard,
        );

        assert!(tx.send(queued).is_ok());
        assert_eq!(pending.load(Ordering::Acquire), 1);
        drop(rx);
        drop(tx);

        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert!(failed.load(Ordering::Acquire));
    }

    #[test]
    fn async_gpu_barrier_waits_for_gated_worker_across_poll_expirations() {
        let failed = AtomicBool::new(false);
        let stopping = AtomicBool::new(false);
        let frames = FrameQueueState::new();
        let wait = AsyncGpuWait {
            failed: &failed,
            stopping: &stopping,
            closed: &frames.closed,
            poll_interval: std::time::Duration::from_millis(2),
            report_interval: std::time::Duration::from_millis(5),
        };
        let (work_tx, work_rx) = crossbeam::channel::bounded(1);
        let (entered_tx, entered_rx) = crossbeam::channel::bounded(1);
        let (release_tx, release_rx) = crossbeam::channel::bounded(1);
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        let (result_tx, result_rx) = crossbeam::channel::bounded(1);
        assert!(work_tx.send(AsyncGpuSubmission::Barrier(done_tx)).is_ok());

        std::thread::scope(|scope| {
            let _close = CloseTestFrameQueue(&frames);
            scope.spawn(move || {
                let AsyncGpuSubmission::Barrier(done) = work_rx.recv().unwrap() else {
                    panic!("expected drain barrier")
                };
                entered_tx.send(()).unwrap();
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(1))
                    .unwrap();
                done.send(true).unwrap();
            });
            entered_rx.recv().unwrap();
            scope.spawn(move || result_tx.send(wait.receive_barrier(&done_rx)).unwrap());

            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_millis(30)),
                Err(crossbeam::channel::RecvTimeoutError::Timeout)
            );
            assert!(!failed.load(Ordering::Acquire));
            release_tx.send(()).unwrap();
            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_secs(1)),
                Ok(true)
            );
        });
        assert!(!failed.load(Ordering::Acquire));
    }

    #[test]
    fn async_gpu_enqueue_preserves_owned_work_across_poll_expirations() {
        let failed = AtomicBool::new(false);
        let stopping = AtomicBool::new(false);
        let frames = FrameQueueState::new();
        let wait = AsyncGpuWait {
            failed: &failed,
            stopping: &stopping,
            closed: &frames.closed,
            poll_interval: std::time::Duration::from_millis(2),
            report_interval: std::time::Duration::from_millis(5),
        };
        let (tx, rx) = crossbeam::channel::bounded(1);
        let (result_tx, result_rx) = crossbeam::channel::bounded(1);
        tx.send(1).unwrap();

        std::thread::scope(|scope| {
            let _close = CloseTestFrameQueue(&frames);
            scope.spawn(|| result_tx.send(wait.send(&tx, 2, "test enqueue")).unwrap());
            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_millis(30)),
                Err(crossbeam::channel::RecvTimeoutError::Timeout)
            );
            assert!(!failed.load(Ordering::Acquire));
            assert_eq!(rx.recv().unwrap(), 1);
            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_secs(1)),
                Ok(Ok(()))
            );
            assert_eq!(rx.recv().unwrap(), 2);
            assert_eq!(rx.try_recv(), Err(crossbeam::channel::TryRecvError::Empty));
        });
    }

    #[test]
    fn async_gpu_wait_cancels_on_frame_queue_close_without_failure() {
        for enqueue in [false, true] {
            let failed = AtomicBool::new(false);
            let stopping = AtomicBool::new(false);
            let frames = FrameQueueState::new();
            let wait = AsyncGpuWait {
                failed: &failed,
                stopping: &stopping,
                closed: &frames.closed,
                poll_interval: std::time::Duration::from_millis(2),
                report_interval: std::time::Duration::from_millis(5),
            };
            let (tx, rx) = crossbeam::channel::bounded(1);
            let (result_tx, result_rx) = crossbeam::channel::bounded(1);
            if enqueue {
                tx.send(true).unwrap();
            }

            std::thread::scope(|scope| {
                let _close = CloseTestFrameQueue(&frames);
                scope.spawn(|| {
                    let completed = if enqueue {
                        let result = wait.send(&tx, false, "test enqueue");
                        assert_eq!(result, Err(false));
                        result.is_ok()
                    } else {
                        wait.receive_barrier(&rx)
                    };
                    result_tx.send(completed).unwrap();
                });
                assert_eq!(
                    result_rx.recv_timeout(std::time::Duration::from_millis(10)),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout)
                );
                frames.close();
                assert_eq!(
                    result_rx.recv_timeout(std::time::Duration::from_secs(1)),
                    Ok(false)
                );
            });
            if enqueue {
                assert_eq!(rx.try_recv(), Ok(true));
                assert_eq!(rx.try_recv(), Err(crossbeam::channel::TryRecvError::Empty));
            }
            assert!(!failed.load(Ordering::Acquire));
        }
    }

    #[test]
    fn async_gpu_barrier_preserves_failure_and_shutdown_results() {
        for case in 0..4 {
            let failed = AtomicBool::new(case == 2);
            let stopping = AtomicBool::new(case == 3);
            let closed = AtomicBool::new(false);
            let wait = AsyncGpuWait {
                failed: &failed,
                stopping: &stopping,
                closed: &closed,
                poll_interval: std::time::Duration::from_millis(2),
                report_interval: std::time::Duration::from_millis(5),
            };
            let (tx, rx) = crossbeam::channel::bounded(1);
            if case == 0 {
                tx.send(false).unwrap();
            }
            if case == 1 {
                drop(tx);
            }
            assert!(!wait.receive_barrier(&rx));
            assert_eq!(failed.load(Ordering::Acquire), case != 3);
        }
    }

    #[test]
    fn async_gpu_enqueue_disconnect_retains_work_and_fails() {
        let failed = AtomicBool::new(false);
        let stopping = AtomicBool::new(false);
        let closed = AtomicBool::new(false);
        let wait = AsyncGpuWait {
            failed: &failed,
            stopping: &stopping,
            closed: &closed,
            poll_interval: std::time::Duration::from_millis(2),
            report_interval: std::time::Duration::from_millis(5),
        };
        let (tx, rx) = crossbeam::channel::bounded(1);
        drop(rx);
        assert_eq!(wait.send(&tx, 7, "test enqueue"), Err(7));
        assert!(failed.load(Ordering::Acquire));
    }

    #[test]
    fn async_gpu_full_queue_cancellation_preserves_accepted_completions() {
        let gpu = Arc::new(GpuContext::new());
        gpu.mappings.write().add(0x5000, 4, 0x9000, 1);
        let frames = Arc::new(FrameQueueState::new());
        let (entered_tx, entered_rx) = crossbeam::channel::bounded(1);
        let (release_tx, release_rx) = crossbeam::channel::bounded(1);
        let read: AsyncMemoryRead = Arc::new(move |_, bytes| {
            entered_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            bytes.fill(0);
            true
        });
        let write: AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: AsyncMemoryCopy = Arc::new(|_, _, _| true);
        let queue = AsyncGpuQueue::new_with_mode(
            Arc::clone(&gpu),
            Arc::clone(&frames),
            read,
            write,
            copy,
            true,
        );
        assert!(queue.submit(AsyncGpuSubmission::Inline {
            entries: vec![gpu::CommandListHeader {
                address_lo: 0x5000,
                address_hi_and_count: 1 << 10,
            }],
            completion: Some(AsyncGpuCompletion {
                fd: 99,
                syncpt_id: 7,
                threshold: 1,
            }),
        }));
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        for _ in 0..queue.capacity {
            assert!(queue.submit(AsyncGpuSubmission::Inline {
                entries: Vec::new(),
                completion: None,
            }));
        }
        let (result_tx, result_rx) = crossbeam::channel::bounded(1);
        std::thread::scope(|scope| {
            let _close = CloseTestFrameQueue(&frames);
            scope.spawn(|| {
                result_tx
                    .send(queue.submit(AsyncGpuSubmission::Inline {
                        entries: Vec::new(),
                        completion: Some(AsyncGpuCompletion {
                            fd: 99,
                            syncpt_id: 7,
                            threshold: 2,
                        }),
                    }))
                    .unwrap();
            });
            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_millis(30)),
                Err(crossbeam::channel::RecvTimeoutError::Timeout)
            );
            assert_eq!(queue.pending.load(Ordering::Acquire), queue.capacity + 2);
            frames.close();
            assert_eq!(
                result_rx.recv_timeout(std::time::Duration::from_secs(1)),
                Ok(false)
            );
            assert_eq!(queue.pending.load(Ordering::Acquire), queue.capacity + 1);
            assert!(!queue.failed.load(Ordering::Acquire));
            assert!(gpu.syncpoint_events().is_empty());
        });

        let pending = Arc::clone(&queue.pending);
        let failed = Arc::clone(&queue.failed);
        release_tx.send(()).unwrap();
        drop(queue);
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert!(!failed.load(Ordering::Acquire));
        assert_eq!(
            gpu.syncpoint_events().iter().copied().collect::<Vec<_>>(),
            vec![gpu::PendingSyncpointEvent::Completion {
                fd: 99,
                syncpt_id: 7,
                threshold: 1,
            }]
        );
    }

    #[test]
    fn async_gpu_present_reservation_cancels_on_frame_queue_close() {
        let gpu = Arc::new(GpuContext::new());
        let frames = Arc::new(FrameQueueState::new());
        let read: AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: AsyncMemoryCopy = Arc::new(|_, _, _| true);
        let queue = AsyncGpuQueue::new_with_mode(gpu, Arc::clone(&frames), read, write, copy, true);
        let present_pending = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let presented = Arc::new(AtomicBool::new(false));
        let job_presented = Arc::clone(&presented);
        assert!(queue.submit(AsyncGpuSubmission::Present {
            job: Box::new(move || job_presented.store(true, Ordering::Release)),
            pending: Arc::clone(&present_pending),
            limit: 1,
            on_prepared: None,
        }));

        frames.close();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while queue.pending.load(Ordering::Acquire) != 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(queue.pending.load(Ordering::Acquire), 0);
        assert_eq!(present_pending.load(Ordering::Acquire), 1);
        assert!(!presented.load(Ordering::Acquire));
        assert!(!queue.failed.load(Ordering::Acquire));
        assert!(!queue.drain().completed);
        drop(queue);
    }

    #[test]
    fn async_gpu_queue_drains_and_joins_cleanly() {
        let gpu = Arc::new(GpuContext::new());
        let read: AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: AsyncMemoryCopy = Arc::new(|_, _, _| true);
        let queue = AsyncGpuQueue::new(gpu, Arc::new(FrameQueueState::new()), read, write, copy);

        assert!(queue.submit(AsyncGpuSubmission::Inline {
            entries: Vec::new(),
            completion: None,
        }));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while queue.pending.load(Ordering::Acquire) != 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(queue.pending.load(Ordering::Acquire), 0);
        assert!(!queue.failed.load(Ordering::Acquire));
        assert!(queue.drain().completed);

        let started = std::time::Instant::now();
        drop(queue);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn async_gpu_hard_kick_mode_is_default_and_soft_is_quarantined() {
        assert!(async_gpu_hard_kicks_value(Some("hard")));
        assert!(async_gpu_hard_kicks_value(Some(" HARD ")));
        assert!(async_gpu_hard_kicks_value(Some("1")));
        assert!(async_gpu_hard_kicks_value(None));
        assert!(!async_gpu_hard_kicks_value(Some("soft")));
        assert!(async_gpu_soft_kicks_value(Some(" Soft ")));
        for value in [None, Some("1"), Some("hard"), Some("soft")] {
            assert!(matches!(
                gpu_thread_mode_request(None, value, None),
                GpuThreadModeRequest::AsyncGpu
            ));
        }
        assert!(matches!(
            gpu_thread_mode_request(Some("1"), None, None),
            GpuThreadModeRequest::SyncPrepThread
        ));
        assert!(matches!(
            gpu_thread_mode_request(Some("1"), Some("hard"), None),
            GpuThreadModeRequest::Conflict
        ));
        assert!(!async_gpu_requires_quarantine(None, None));
        assert!(!async_gpu_requires_quarantine(Some("1"), None));
        assert!(!async_gpu_requires_quarantine(Some("hard"), Some("0")));
        assert!(async_gpu_requires_quarantine(Some("soft"), None));
        assert!(async_gpu_requires_quarantine(None, Some("1")));
    }

    #[test]
    fn async_gpu_hard_kick_queue_processes_and_drains() {
        let gpu = Arc::new(GpuContext::new());
        let read: AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: AsyncMemoryCopy = Arc::new(|_, _, _| true);
        let queue = AsyncGpuQueue::new_with_mode(
            gpu,
            Arc::new(FrameQueueState::new()),
            read,
            write,
            copy,
            true,
        );
        assert!(queue.hard_kicks);
        assert!(!queue.defer_small_rts);

        assert!(queue.submit(AsyncGpuSubmission::Inline {
            entries: Vec::new(),
            completion: None,
        }));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while queue.pending.load(Ordering::Acquire) != 0 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(queue.pending.load(Ordering::Acquire), 0);
        assert!(!queue.failed.load(Ordering::Acquire));
        assert!(queue.drain().completed);
        drop(queue);
    }

    #[test]
    fn async_completion_matches_submit_increment_semantics() {
        assert!(!Nvdrv::submit_emits_increments_value(0, false));
        assert!(!Nvdrv::submit_emits_increments_value(1 << 8, false));
        assert!(Nvdrv::submit_emits_increments_value(1 << 1, false));
        assert!(Nvdrv::submit_emits_increments_value(
            (1 << 1) | (1 << 8),
            false
        ));
        assert!(Nvdrv::submit_emits_increments_value(0, true));
    }

    #[test]
    fn async_gpu_drain_classifier_covers_mapping_mutations() {
        for cmd in [0x4102, 0x4103, 0x4105, 0x4106, 0x4114] {
            assert!(ioctl_requires_async_gpu_drain(NvDevice::NvhostAsGpu, cmd));
        }
        assert!(ioctl_requires_async_gpu_drain(
            NvDevice::NvhostNvdec,
            0x0009
        ));
        assert!(ioctl_requires_async_gpu_drain(NvDevice::NvhostVic, 0x0001));
        assert!(ioctl_requires_async_gpu_drain(NvDevice::NvhostVic, 0x0009));
        assert!(ioctl_requires_async_gpu_drain(NvDevice::Nvmap, 0x0105));
        assert!(!ioctl_requires_async_gpu_drain(NvDevice::NvhostGpu, 0x4808));
        assert!(!ioctl_requires_async_gpu_drain(NvDevice::NvhostGpu, 0x481B));
        assert!(!ioctl_requires_async_gpu_drain(
            NvDevice::NvhostAsGpu,
            0x4109
        ));
    }

    #[test]
    fn frame_handoff_backpressures_and_preserves_every_frame_in_fifo_order() {
        let queue = Arc::new(FrameQueueState::new());
        let stats = Arc::new(PipelineStats::default());
        for value in 0..FRAME_QUEUE_CAPACITY as u8 {
            enqueue_bounded_frame(&queue, &stats, test_frame(value));
        }

        let worker_queue = Arc::clone(&queue);
        let worker_stats = Arc::clone(&stats);
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = std::sync::mpsc::sync_channel(1);
        let producer = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            enqueue_bounded_frame(
                &worker_queue,
                &worker_stats,
                test_frame(FRAME_QUEUE_CAPACITY as u8),
            );
            finished_tx.send(()).unwrap();
        });

        entered_rx.recv().unwrap();
        assert!(finished_rx
            .recv_timeout(std::time::Duration::from_millis(20))
            .is_err());
        assert_eq!(queue.len(), FRAME_QUEUE_CAPACITY);
        assert_eq!(
            queue
                .pop_front_due(std::time::Instant::now())
                .unwrap()
                .pixels[0],
            0
        );
        finished_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        producer.join().unwrap();

        let delivered: Vec<u8> = (0..FRAME_QUEUE_CAPACITY)
            .map(|_| {
                queue
                    .pop_front_due(std::time::Instant::now())
                    .unwrap()
                    .pixels[0]
            })
            .collect();
        assert_eq!(delivered, vec![1, 2, 3, 4]);
        assert_eq!(
            stats.snapshot().frames_submitted,
            FRAME_QUEUE_CAPACITY as u64 + 1
        );
        assert!(queue.pop_front_due(std::time::Instant::now()).is_none());
    }

    #[test]
    fn frame_handoff_replaces_oldest_frame_when_the_presenter_stalls() {
        let queue = Arc::new(FrameQueueState::new());
        let stats = Arc::new(PipelineStats::default());
        for value in 0..FRAME_QUEUE_CAPACITY as u8 {
            enqueue_bounded_frame(&queue, &stats, test_frame(value));
        }
        let started = std::time::Instant::now();
        enqueue_bounded_frame(&queue, &stats, test_frame(FRAME_QUEUE_CAPACITY as u8));
        assert!(started.elapsed() >= FRAME_QUEUE_STALL_GRACE);
        assert!(started.elapsed() < FRAME_QUEUE_STALL_GRACE * 4);
        assert_eq!(queue.len(), FRAME_QUEUE_CAPACITY);
        let delivered: Vec<u8> = (0..FRAME_QUEUE_CAPACITY)
            .map(|_| {
                queue
                    .pop_front_due(std::time::Instant::now())
                    .unwrap()
                    .pixels[0]
            })
            .collect();
        assert_eq!(delivered, vec![1, 2, 3, 4]);
        assert!(queue.pop_front_due(std::time::Instant::now()).is_none());
    }

    #[test]
    fn frame_handoff_stays_in_mailbox_mode_until_the_presenter_pops_again() {
        let queue = Arc::new(FrameQueueState::new());
        let stats = Arc::new(PipelineStats::default());
        for value in 0..=FRAME_QUEUE_CAPACITY as u8 {
            enqueue_bounded_frame(&queue, &stats, test_frame(value));
        }
        assert!(queue.presenter_stalled.load(Ordering::Acquire));
        let started = std::time::Instant::now();
        enqueue_bounded_frame(&queue, &stats, test_frame(9));
        assert!(started.elapsed() < FRAME_QUEUE_STALL_GRACE);
        assert_eq!(queue.len(), FRAME_QUEUE_CAPACITY);
        assert_eq!(
            queue
                .pop_front_due(std::time::Instant::now())
                .unwrap()
                .pixels[0],
            2
        );
        assert!(!queue.presenter_stalled.load(Ordering::Acquire));
        enqueue_bounded_frame(&queue, &stats, test_frame(10));
        let delivered: Vec<u8> = (0..FRAME_QUEUE_CAPACITY)
            .map(|_| {
                queue
                    .pop_front_due(std::time::Instant::now())
                    .unwrap()
                    .pixels[0]
            })
            .collect();
        assert_eq!(delivered, vec![3, 4, 9, 10]);
    }

    #[test]
    fn frame_handoff_recovers_pacing_after_stalled_future_frames() {
        let queue = FrameQueueState::new();
        let period = std::time::Duration::from_nanos(16_666_667);
        let source_start = std::time::Instant::now() + std::time::Duration::from_secs(60);
        for index in 0..1000u32 {
            let mut frame = test_frame(index as u8);
            frame.present_at = Some(source_start + period * index);
            assert!(queue.enqueue(frame));
        }
        assert!(queue.presenter_stalled.load(Ordering::Acquire));
        assert_eq!(queue.len(), FRAME_QUEUE_CAPACITY);

        let now = std::time::Instant::now();
        let mut last_deadline = None;
        for index in 996..1000u32 {
            let frame = queue
                .pop_front_due(now)
                .expect("stalled frame must become due");
            assert_eq!(frame.pixels[0], index as u8);
            assert!(frame.present_at.is_some_and(|deadline| deadline <= now));
            last_deadline = frame.present_at;
        }
        assert!(!queue.presenter_stalled.load(Ordering::Acquire));

        let resumed_deadline = last_deadline.unwrap() + period;
        let mut resumed = test_frame(42);
        resumed.present_at = Some(source_start + period * 1000);
        assert!(queue.enqueue(resumed));
        assert!(queue
            .pop_front_due(resumed_deadline - std::time::Duration::from_nanos(1))
            .is_none());
        let frame = queue.pop_front_due(resumed_deadline).unwrap();
        assert_eq!(frame.pixels[0], 42);
        assert_eq!(frame.present_at, Some(resumed_deadline));
    }

    #[test]
    fn frame_handoff_resets_deadline_correction_when_source_schedule_resets() {
        let now = std::time::Instant::now();
        let period = std::time::Duration::from_nanos(16_666_667);
        for unpaced_reset in [false, true] {
            let queue = FrameQueueState::new();
            queue.presenter_stalled.store(true, Ordering::Release);
            let mut stalled = test_frame(1);
            stalled.present_at = Some(now + std::time::Duration::from_secs(60));
            assert!(queue.enqueue(stalled));
            assert!(queue.pop_front_due(std::time::Instant::now()).is_some());

            if unpaced_reset {
                assert!(queue.enqueue(test_frame(2)));
                assert!(queue.pop_front_due(std::time::Instant::now()).is_some());
            }
            let deadline = std::time::Instant::now() + period;
            let mut reset = test_frame(3);
            reset.present_at = Some(deadline);
            assert!(queue.enqueue(reset));
            assert!(queue
                .pop_front_due(deadline - std::time::Duration::from_nanos(1))
                .is_none());
            assert_eq!(
                queue.pop_front_due(deadline).unwrap().present_at,
                Some(deadline)
            );
        }
    }

    #[test]
    fn frame_handoff_source_reset_unblocks_retained_future_frames() {
        let period = std::time::Duration::from_nanos(16_666_667);
        for reset_mode in 0..3 {
            let queue = FrameQueueState::new();
            let now = std::time::Instant::now();
            for index in 0..FRAME_QUEUE_CAPACITY as u32 {
                let mut frame = test_frame(index as u8);
                frame.present_at = Some(now + std::time::Duration::from_secs(60) + period * index);
                assert!(queue.enqueue(frame));
            }
            queue.presenter_stalled.store(true, Ordering::Release);
            let mut reset = test_frame(9);
            reset.present_at = match reset_mode {
                0 => None,
                1 => Some(now - std::time::Duration::from_secs(1)),
                _ => Some(now + period),
            };
            assert!(queue.enqueue(reset));

            let after_enqueue = std::time::Instant::now();
            for expected in [1, 2, 3, 9] {
                let frame = queue
                    .pop_front_due(after_enqueue)
                    .expect("source reset must unblock retained future frames");
                assert_eq!(frame.pixels[0], expected);
            }
            assert!(!queue.presenter_stalled.load(Ordering::Acquire));
        }
    }

    #[test]
    fn frame_handoff_drain_restores_fifo_after_consumer_progress() {
        let queue = FrameQueueState::new();
        assert!(queue.enqueue(test_frame(1)));
        queue.presenter_stalled.store(true, Ordering::Release);
        let drained = queue.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].pixels[0], 1);
        assert!(!queue.presenter_stalled.load(Ordering::Acquire));
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn frame_handoff_future_deadline_wait_honors_close_and_stop() {
        for close_queue in [false, true] {
            let queue = Arc::new(FrameQueueState::new());
            let stopping = Arc::new(AtomicBool::new(false));
            let mut frame = test_frame(1);
            frame.present_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
            assert!(queue.enqueue(frame));

            let worker_queue = Arc::clone(&queue);
            let worker_stopping = Arc::clone(&stopping);
            let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
            let (finished_tx, finished_rx) = std::sync::mpsc::sync_channel(1);
            let worker = std::thread::spawn(move || {
                entered_tx.send(()).unwrap();
                let frame =
                    worker_queue.wait_pop_front_due(|| worker_stopping.load(Ordering::Acquire));
                finished_tx.send(frame.is_none()).unwrap();
            });
            entered_rx.recv().unwrap();
            if close_queue {
                queue.close();
                assert!(!queue.enqueue(test_frame(2)));
            } else {
                stopping.store(true, Ordering::Release);
            }
            assert!(finished_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap());
            worker.join().unwrap();
        }
    }

    #[test]
    fn cpu_present_submit_returns_without_waiting_for_host_fifo_capacity() {
        let nvdrv = Nvdrv::new();
        for value in 0..FRAME_QUEUE_CAPACITY as u8 {
            nvdrv.submit_frame(test_frame(value));
        }

        let started = std::time::Instant::now();
        assert!(nvdrv.submit_frame_nonblocking(test_frame(FRAME_QUEUE_CAPACITY as u8)));
        assert!(
            started.elapsed() < std::time::Duration::from_millis(50),
            "CPU QueueBuffer fallback waited for a full host FIFO"
        );

        assert_eq!(
            nvdrv
                .drain_next_frame_due(std::time::Instant::now())
                .unwrap()
                .pixels[0],
            0
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while nvdrv.frame_queue_depth() < FRAME_QUEUE_CAPACITY
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        assert_eq!(nvdrv.frame_queue_depth(), FRAME_QUEUE_CAPACITY);

        let delivered: Vec<u8> = (0..FRAME_QUEUE_CAPACITY)
            .map(|_| {
                nvdrv
                    .drain_next_frame_due(std::time::Instant::now())
                    .unwrap()
                    .pixels[0]
            })
            .collect();
        assert_eq!(delivered, vec![1, 2, 3, 4]);
    }

    #[test]
    fn queuebuffer_deadline_holds_the_fifo_head_without_leapfrogging() {
        let nvdrv = Nvdrv::new();
        let now = std::time::Instant::now();
        let mut interval_two_frame = test_frame(1);
        interval_two_frame.present_at = Some(now + std::time::Duration::from_millis(33));
        nvdrv.submit_frame(interval_two_frame);
        nvdrv.submit_frame(test_frame(2));

        assert!(nvdrv.drain_next_frame_due(now).is_none());
        assert_eq!(
            nvdrv
                .drain_next_frame_due(now + std::time::Duration::from_millis(33))
                .unwrap()
                .pixels[0],
            1
        );
        assert_eq!(
            nvdrv
                .drain_next_frame_due(now + std::time::Duration::from_millis(33))
                .unwrap()
                .pixels[0],
            2
        );
    }

    #[test]
    fn ordered_present_slot_waits_for_capacity_to_open() {
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let release_pending = Arc::clone(&pending);
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            release_pending.fetch_sub(1, Ordering::Release);
        });

        reserve_ordered_present_slot(&pending, 1);
        assert_eq!(pending.load(Ordering::Acquire), 1);
        release.join().unwrap();
    }

    #[test]
    fn ordered_present_slot_waits_past_the_old_drop_timeout() {
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let release_pending = Arc::clone(&pending);
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(275));
            release_pending.fetch_sub(1, Ordering::Release);
        });

        let waited_from = std::time::Instant::now();
        reserve_ordered_present_slot(&pending, 1);
        assert!(waited_from.elapsed() >= std::time::Duration::from_millis(250));
        assert_eq!(pending.load(Ordering::Acquire), 1);
        release.join().unwrap();
    }

    #[test]
    fn vp9_output_targets_follow_the_composed_packet_not_the_current_submit() {
        let mut pending_target = None;
        assert_eq!(next_vp9_packet_target(&mut pending_target, 0x1000), 0x1000);
        assert_eq!(next_vp9_packet_target(&mut pending_target, 0x2000), 0x1000);
        assert_eq!(next_vp9_packet_target(&mut pending_target, 0x3000), 0x2000);

        pending_target = None;
        assert_eq!(next_vp9_packet_target(&mut pending_target, 0x4000), 0x4000);
        assert_eq!(next_vp9_packet_target(&mut pending_target, 0x5000), 0x4000);
    }

    #[test]
    fn a_reconfigured_vp9_channel_clears_its_pending_packet_target() {
        let mut runtime = VideoChannelRuntime::new(NvDevice::NvhostNvdec);
        assert_eq!(
            next_vp9_packet_target(&mut runtime.vp9_packet_target, 0x1000),
            0x1000
        );
        assert_eq!(runtime.vp9_packet_target, Some(0x1000));

        let config = (video_ffmpeg::FfmpegCodec::Vp9, 1280, 720);
        assert_ne!(runtime.ffmpeg_config, Some(config));
        runtime.ffmpeg_config = Some(config);
        runtime.vp9_packet_target = None;

        assert_eq!(
            next_vp9_packet_target(&mut runtime.vp9_packet_target, 0x9000),
            0x9000
        );
    }

    #[test]
    fn syncpoint_expiry_follows_the_reserved_window() {
        assert!(syncpoint_expired(8, 8, 8));
        assert!(syncpoint_expired(9, 12, 8));
        assert!(!syncpoint_expired(8, 12, 9));
        assert!(syncpoint_expired(8, 8, 9));
        assert!(syncpoint_expired(5, 8, 9));
        assert!(!syncpoint_expired(u32::MAX - 1, 2, 1));
        assert!(syncpoint_expired(u32::MAX - 1, 2, u32::MAX - 1));
    }

    #[test]
    fn syncpoint_escape_requires_an_explicit_true_value() {
        use std::ffi::OsStr;

        assert!(!syncpoint_escape_drain_value_enabled(None));
        for disabled in ["", "0", "false", "OFF", " no ", "unexpected"] {
            assert!(!syncpoint_escape_drain_value_enabled(Some(OsStr::new(
                disabled
            ))));
        }
        for enabled in ["1", "true", "TRUE", "on", " yes "] {
            assert!(syncpoint_escape_drain_value_enabled(Some(OsStr::new(
                enabled
            ))));
        }
    }

    #[test]
    fn wait_stall_observation_logs_once_and_drains_once_when_later_submitted() {
        let mut state = CtrlEventWaitFailure::default();
        for _ in 0..CTRL_EVENT_WAIT_FAIL_LIMIT {
            assert_eq!(
                ctrl_event_wait_failure_action(&mut state, false, true),
                CtrlEventWaitFailureAction::default()
            );
        }
        assert_eq!(
            ctrl_event_wait_failure_action(&mut state, false, true),
            CtrlEventWaitFailureAction {
                log: true,
                drain: false
            }
        );
        assert_eq!(
            ctrl_event_wait_failure_action(&mut state, true, false),
            CtrlEventWaitFailureAction::default()
        );
        assert_eq!(
            ctrl_event_wait_failure_action(&mut state, true, true),
            CtrlEventWaitFailureAction {
                log: false,
                drain: true
            }
        );
        assert_eq!(
            ctrl_event_wait_failure_action(&mut state, true, true),
            CtrlEventWaitFailureAction::default()
        );
    }

    #[test]
    fn orphan_increment_before_and_after_close_are_equivalent() {
        fn configured_channel() -> (Nvdrv, u32, u32) {
            let mut nvdrv = Nvdrv::new();
            let fd = nvdrv.open("/dev/nvhost-gpu").unwrap();
            let syncpt = nvdrv.ensure_channel_syncpoint(fd).0;
            let mut channels = nvdrv.gpu.channels.lock();
            let channel = channels.get_mut(&fd).unwrap();
            channel.syncpt_min = 27_062;
            channel.syncpt_max = 27_066;
            drop(channels);
            (nvdrv, fd, syncpt)
        }

        let (mut before, before_fd, before_id) = configured_channel();
        before
            .gpu
            .record_embedded_syncpt_incrs(vec![(before_id, 4)]);
        before.poll_gpu_completions();
        before.close(before_fd);

        let (mut after, after_fd, after_id) = configured_channel();
        after.close(after_fd);
        after.gpu.record_embedded_syncpt_incrs(vec![(after_id, 4)]);
        after.poll_gpu_completions();

        assert_eq!(
            before.retired_syncpts.lock().get(&before_id),
            Some(&(27_066, 27_066))
        );
        assert_eq!(
            after.retired_syncpts.lock().get(&after_id),
            Some(&(27_066, 27_066))
        );
    }

    #[test]
    fn orphan_increments_handle_partial_multi_id_wrap_and_reject_unknown_ids() {
        let nvdrv = Nvdrv::new();
        nvdrv.retired_syncpts.lock().extend([
            (7, (100, 108)),
            (8, (u32::MAX - 1, 2)),
            (9, (u32::MAX - 1, 0)),
        ]);
        nvdrv
            .gpu
            .record_embedded_syncpt_incrs(vec![(7, 3), (8, 3), (9, 4), (0, 1), (99, 1)]);
        nvdrv.poll_gpu_completions();

        let retired = nvdrv.retired_syncpts.lock();
        assert_eq!(retired.get(&7), Some(&(103, 108)));
        assert_eq!(retired.get(&8), Some(&(1, 2)));
        assert_eq!(retired.get(&9), Some(&(2, 2)));
        assert!(!retired.contains_key(&0));
        assert!(!retired.contains_key(&99));
    }

    #[test]
    fn syncpoint_events_preserve_increment_then_floor_order() {
        let nvdrv = Nvdrv::new();
        nvdrv.retired_syncpts.lock().insert(7, (10, 14));
        nvdrv.gpu.record_embedded_syncpt_incrs(vec![(7, 2)]);
        nvdrv.gpu.record_syncpoint_completion(99, 7, 14);

        nvdrv.poll_gpu_completions();

        assert_eq!(nvdrv.retired_syncpts.lock().get(&7), Some(&(14, 14)));
    }

    #[test]
    fn syncpoint_events_preserve_floor_then_increment_order() {
        let nvdrv = Nvdrv::new();
        nvdrv.retired_syncpts.lock().insert(7, (10, 16));
        nvdrv.gpu.record_syncpoint_completion(99, 7, 14);
        nvdrv.gpu.record_embedded_syncpt_incrs(vec![(7, 2)]);

        nvdrv.poll_gpu_completions();

        assert_eq!(nvdrv.retired_syncpts.lock().get(&7), Some(&(16, 16)));
    }

    #[test]
    fn poll_between_gated_completion_and_increment_transfer_preserves_order() {
        let nvdrv = Nvdrv::new();
        nvdrv.retired_syncpts.lock().insert(7, (10, 12));
        let completion_gpu = Arc::clone(&nvdrv.gpu);
        let (completion, gate) = gpu::gate_syncpoint_completion(Some(Box::new(move || {
            completion_gpu.record_syncpoint_completion(99, 7, 12);
        })));
        let completion = completion.unwrap();
        let gate = gate.unwrap();
        let start = Arc::new(std::sync::Barrier::new(2));
        let fired = Arc::new(std::sync::Barrier::new(2));

        std::thread::scope(|scope| {
            let worker_start = Arc::clone(&start);
            let worker_fired = Arc::clone(&fired);
            scope.spawn(move || {
                worker_start.wait();
                completion();
                worker_fired.wait();
            });
            start.wait();
            fired.wait();

            nvdrv.poll_gpu_completions();
            assert_eq!(nvdrv.retired_syncpts.lock().get(&7), Some(&(10, 12)));

            nvdrv.gpu.record_embedded_syncpt_incrs(vec![(7, 2)]);
            gate.release();
        });

        nvdrv.poll_gpu_completions();
        assert_eq!(nvdrv.retired_syncpts.lock().get(&7), Some(&(12, 12)));
    }

    #[test]
    fn orphan_increments_do_not_cross_nvdrv_instances() {
        let first = Nvdrv::new();
        let second = Nvdrv::new();
        first.retired_syncpts.lock().insert(1, (20, 24));
        second.retired_syncpts.lock().insert(1, (40, 44));

        first.gpu.record_embedded_syncpt_incrs(vec![(1, 2)]);
        first.poll_gpu_completions();
        second.poll_gpu_completions();

        assert_eq!(first.retired_syncpts.lock().get(&1), Some(&(22, 24)));
        assert_eq!(second.retired_syncpts.lock().get(&1), Some(&(40, 44)));
    }

    #[test]
    fn event_wait_failure_state_cleans_up_on_completion_cancel_unregister_and_close() {
        let mut nvdrv = Nvdrv::new();
        let ctrl_fd = nvdrv.open("/dev/nvhost-ctrl").unwrap();
        let gpu_fd = nvdrv.open("/dev/nvhost-gpu").unwrap();
        let syncpt = nvdrv.ensure_channel_syncpoint(gpu_fd).0;
        let wait = CtrlEventWait {
            syncpt_id: syncpt,
            threshold: 4,
        };

        nvdrv
            .ctrl_event_wait_failures
            .lock()
            .insert((syncpt, 4), CtrlEventWaitFailure::default());
        nvdrv.gpu.record_syncpoint_completion(gpu_fd, syncpt, 4);
        nvdrv.poll_gpu_completions();
        assert!(nvdrv.ctrl_event_wait_failures.lock().is_empty());

        let request = |fd: u32, event_id: u32, cmd: u16| IoctlRequest {
            fd,
            ioctl_id: cmd as u32,
            in_data: event_id.to_le_bytes().to_vec(),
            inline_in_data: Vec::new(),
            out_size: 0,
        };
        for cmd in [0x001c, 0x0020] {
            nvdrv.ctrl_event_waits.insert((ctrl_fd, 3), wait);
            nvdrv
                .ctrl_event_wait_failures
                .lock()
                .insert((syncpt, 4), CtrlEventWaitFailure::default());
            let _ = nvdrv.nvhost_ctrl_ioctl(cmd, &request(ctrl_fd, 3u32, cmd));
            assert!(nvdrv.ctrl_event_waits.is_empty());
            assert!(nvdrv.ctrl_event_wait_failures.lock().is_empty());
        }

        nvdrv.ctrl_event_waits.insert((ctrl_fd, 5), wait);
        nvdrv
            .ctrl_event_wait_failures
            .lock()
            .insert((syncpt, 4), CtrlEventWaitFailure::default());
        nvdrv.close(ctrl_fd);
        assert!(nvdrv.ctrl_event_waits.is_empty());
        assert!(nvdrv.ctrl_event_wait_failures.lock().is_empty());
    }

    #[test]
    fn engine_increments_never_lower_the_reserved_max() {
        let mut nvdrv = Nvdrv::new();
        let gpu_fd = nvdrv.open("/dev/nvhost-gpu").unwrap();
        let (syncpt, threshold) = nvdrv.reserve_channel_submit(gpu_fd, 1 << 1, 0);
        assert_eq!(threshold, 2);
        nvdrv.gpu.record_embedded_syncpt_incrs(vec![(syncpt, 1)]);
        assert!(syncpoint_reached(nvdrv.syncpoint_max(syncpt), threshold));
        assert!(syncpoint_reached(
            nvdrv.syncpoint_max(syncpt),
            nvdrv.syncpoint_value(syncpt)
        ));
        nvdrv.gpu.record_embedded_syncpt_incrs(vec![(syncpt, 8)]);
        let min = nvdrv.syncpoint_value(syncpt);
        assert!(syncpoint_reached(min, threshold));
        assert!(syncpoint_reached(nvdrv.syncpoint_max(syncpt), min));
    }

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

    #[test]
    fn gpu_time_ioctl_uses_the_report_clock_epoch() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-ctrl-gpu").unwrap();
        let before = gpu::clock::nanoseconds();
        let result = nvdrv.dispatch_ioctl(request(fd, 0x8008_471c, Vec::new(), 8));
        let after = gpu::clock::nanoseconds();
        let reported = u64::from_le_bytes(result.data[..8].try_into().unwrap());
        assert!(reported >= before);
        assert!(reported <= after);
    }

    fn test_nvmap_handle(id: u32, size: u32, address: u64) -> NvmapHandle {
        NvmapHandle {
            id,
            user_refcount: 1,
            size,
            address,
            kind: 0,
            align: 0x1000,
            channel_map_address: 0,
            channel_pin_count: 0,
        }
    }

    fn alloc_as_gpu_space(
        nvdrv: &mut Nvdrv,
        fd: u32,
        base: u64,
        pages: u32,
        page_size: u32,
        flags: u32,
    ) -> IoctlOutcome {
        let mut input = vec![0u8; 24];
        input[0..4].copy_from_slice(&pages.to_le_bytes());
        input[4..8].copy_from_slice(&page_size.to_le_bytes());
        input[8..12].copy_from_slice(&flags.to_le_bytes());
        input[16..24].copy_from_slice(&base.to_le_bytes());
        nvdrv.dispatch_ioctl(request(fd, 0xc018_4102, input, 24))
    }

    fn free_as_gpu_space(
        nvdrv: &mut Nvdrv,
        fd: u32,
        base: u64,
        pages: u32,
        page_size: u32,
    ) -> IoctlOutcome {
        let mut input = vec![0u8; 16];
        input[0..8].copy_from_slice(&base.to_le_bytes());
        input[8..12].copy_from_slice(&pages.to_le_bytes());
        input[12..16].copy_from_slice(&page_size.to_le_bytes());
        nvdrv.dispatch_ioctl(request(fd, 0x4010_4103, input, 0))
    }

    fn as_gpu_remap_entry(base: u64, pages: u32, handle: u32, handle_pages: u32) -> Vec<u8> {
        let mut input = vec![0u8; 20];
        input[4..8].copy_from_slice(&handle.to_le_bytes());
        input[8..12].copy_from_slice(&handle_pages.to_le_bytes());
        input[12..16].copy_from_slice(&((base / 0x10000) as u32).to_le_bytes());
        input[16..20].copy_from_slice(&pages.to_le_bytes());
        input
    }

    #[test]
    fn nvmap_free_returns_backing_address_and_size() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvmap").unwrap();
        let handle = 7;
        let address = 0x10_44b6_6000;
        let size = 0x15e000;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, size, address));
        let mut input = vec![0u8; 24];
        input[..4].copy_from_slice(&handle.to_le_bytes());

        let freed = nvdrv.dispatch_ioctl(request(fd, 0xc018_0105, input, 24));

        assert_eq!(freed.result, 0);
        assert_eq!(
            u64::from_le_bytes(freed.data[8..16].try_into().unwrap()),
            address
        );
        assert_eq!(read_u32(&freed.data, 16), Some(size));
        assert_eq!(read_u32(&freed.data, 20), Some(0));
        assert!(!nvdrv.nvmap_handles.contains_key(&handle));
    }

    #[test]
    fn nvmap_free_accepts_null_handle() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvmap").unwrap();
        let freed = nvdrv.dispatch_ioctl(request(fd, 0xc018_0105, vec![0u8; 24], 24));

        assert_eq!(freed.result, 0);
        assert_eq!(freed.data, vec![0u8; 24]);
    }

    #[test]
    fn nvmap_free_retains_from_id_reference_until_final_free() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvmap").unwrap();
        let handle = 9;
        let address = 0x10_6000_0000;
        let size = 0x4000;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, size, address));
        let mut from_id = vec![0u8; 8];
        from_id[..4].copy_from_slice(&handle.to_le_bytes());
        let duplicated = nvdrv.dispatch_ioctl(request(fd, 0xc008_0103, from_id, 8));
        assert_eq!(read_u32(&duplicated.data, 4), Some(handle));

        let mut input = vec![0u8; 24];
        input[..4].copy_from_slice(&handle.to_le_bytes());
        let retained = nvdrv.dispatch_ioctl(request(fd, 0xc018_0105, input.clone(), 24));
        assert_eq!(
            u64::from_le_bytes(retained.data[8..16].try_into().unwrap()),
            0
        );
        assert_eq!(read_u32(&retained.data, 16), Some(size));
        assert_eq!(read_u32(&retained.data, 20), Some(1));
        assert_eq!(nvdrv.nvmap_handles[&handle].user_refcount, 1);

        let freed = nvdrv.dispatch_ioctl(request(fd, 0xc018_0105, input, 24));
        assert_eq!(
            u64::from_le_bytes(freed.data[8..16].try_into().unwrap()),
            address
        );
        assert_eq!(read_u32(&freed.data, 16), Some(size));
        assert_eq!(read_u32(&freed.data, 20), Some(0));
        assert!(!nvdrv.nvmap_handles.contains_key(&handle));
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
    fn map_buffer_ex_remap_updates_only_the_exact_offsetted_subrange() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x5_04d3_0000u64;
        let cpu = 0x4a_0200_0000u64;
        nvdrv
            .gpu
            .mappings
            .write()
            .add_as_gpu_mapping(fd, base, 0x400000, cpu, 77, None, None, true);

        let mut input = vec![0u8; 40];
        input[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        input[16..24].copy_from_slice(&0x2f0000u64.to_le_bytes());
        input[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        input[32..40].copy_from_slice(&base.to_le_bytes());
        let before = nvdrv.gpu.mappings.read().iter().count();
        let generations = [
            nexium_gpu::tex_invalidate::region_gen(base),
            nexium_gpu::tex_invalidate::region_gen(base + 0x2f0000),
            nexium_gpu::tex_invalidate::region_gen(base + 0x300000),
        ];

        let first = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input.clone(), 40));
        let second = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input, 40));

        assert_eq!(first.result, 0);
        assert_eq!(second.result, 0);
        assert_eq!(
            u64::from_le_bytes(first.data[32..40].try_into().unwrap()),
            base
        );
        let mappings = nvdrv.gpu.mappings.read();
        assert_eq!(mappings.iter().count(), before + 2);
        assert_eq!(
            mappings.cpu_address_for(base + 0x2f0000),
            Some(cpu + 0x2f0000)
        );
        assert!(mappings.iter().all(|mapping| mapping.nvmap_id != 0));
        assert_eq!(nexium_gpu::tex_invalidate::region_gen(base), generations[0]);
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen(base + 0x2f0000),
            generations[1]
        );
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(base + 0x300000),
            generations[2]
        );
    }

    #[test]
    fn map_buffer_ex_remap_honors_signed_negative_offsets_and_survives_source_unmap() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x7040_0000u64;
        let target = base - 0x10000;
        let source_cpu = 0x5040_0000u64;
        let restored_cpu = 0x6040_0000u64;
        nvdrv
            .gpu
            .mappings
            .write()
            .add(target, 0x10000, restored_cpu, 76);
        nvdrv
            .gpu
            .mappings
            .write()
            .add_as_gpu_mapping(fd, base, 0x20000, source_cpu, 77, None, None, true);

        let mut input = vec![0u8; 40];
        input[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        input[16..24].copy_from_slice(&(-0x10000i64).to_le_bytes());
        input[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        input[32..40].copy_from_slice(&base.to_le_bytes());

        let remapped = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input, 40));
        assert_eq!(remapped.result, 0);
        assert_eq!(
            u64::from_le_bytes(remapped.data[32..40].try_into().unwrap()),
            base
        );
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(target + 0x800),
            Some(source_cpu - 0x10000 + 0x800)
        );

        let unmapped =
            nvdrv.dispatch_ioctl(request(fd, 0xc008_4105, base.to_le_bytes().to_vec(), 8));
        assert_eq!(unmapped.result, 0);
        let mappings = nvdrv.gpu.mappings.read();
        assert_eq!(mappings.cpu_address_for(base), None);
        assert_eq!(
            mappings.cpu_address_for(target + 0x800),
            Some(source_cpu - 0x10000 + 0x800)
        );
    }

    #[test]
    fn map_buffer_ex_normal_mapping_honors_signed_negative_buffer_offsets() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let handle = 78;
        let base = 0x7140_0000u64;
        let handle_cpu = 0x5141_0000u64;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, handle_cpu));
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, base, 1, 0x10000, 1).result,
            0
        );
        let mut input = vec![0u8; 40];
        input[0..4].copy_from_slice(&1u32.to_le_bytes());
        input[8..12].copy_from_slice(&handle.to_le_bytes());
        input[16..24].copy_from_slice(&(-0x10000i64).to_le_bytes());
        input[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        input[32..40].copy_from_slice(&base.to_le_bytes());

        let mapped = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, input, 40));

        assert_eq!(mapped.result, 0);
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(base + 0x800),
            Some(handle_cpu - 0x10000 + 0x800)
        );
    }

    #[test]
    fn repeated_fixed_map_unmaps_to_a_hole_without_extra_va_ownership() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x0400_0000u64;
        let target = base + 0x10000;
        let handle = 79;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, 0x5240_0000));

        let mut allocation = vec![0u8; 24];
        allocation[0..4].copy_from_slice(&3u32.to_le_bytes());
        allocation[4..8].copy_from_slice(&0x10000u32.to_le_bytes());
        allocation[8..12].copy_from_slice(&1u32.to_le_bytes());
        allocation[16..24].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc018_4102, allocation, 24))
                .result,
            0
        );

        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        map[32..40].copy_from_slice(&target.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, map.clone(), 40))
                .result,
            0
        );
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, map, 40))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().iter().count(), 2);

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc008_4105, target.to_le_bytes().to_vec(), 8,))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(target), None);
        let mut free = vec![0u8; 16];
        free[0..8].copy_from_slice(&base.to_le_bytes());
        free[8..12].copy_from_slice(&3u32.to_le_bytes());
        free[12..16].copy_from_slice(&0x10000u32.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4010_4103, free, 0))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.alloc_va(0x30000, false), base);
    }

    #[test]
    fn remap_holes_do_not_block_unmap_and_unmap_blocks_older_remap_sources() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x7300_0000;
        let handle = 80;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, 0x5300_0000));
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, base, 1, 0x10000, 3).result,
            0
        );
        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        map[32..40].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, map, 40))
                .result,
            0
        );
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(
                    fd,
                    0x4014_4114,
                    as_gpu_remap_entry(base, 1, 0, 0),
                    0,
                ))
                .result,
            0
        );
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc008_4105, base.to_le_bytes().to_vec(), 8,))
                .result,
            0
        );
        let mut remap = vec![0u8; 40];
        remap[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        remap[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        remap[32..40].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, remap, 40))
                .result,
            0xB
        );
    }

    #[test]
    fn map_buffer_ex_remap_rejects_missing_or_oversized_base() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x5_04d3_0000u64;
        nvdrv.gpu.mappings.write().add_as_gpu_mapping(
            fd,
            base,
            0x10000,
            0x4a_0200_0000,
            77,
            None,
            None,
            true,
        );

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
    fn as_gpu_map_unmap_and_remap_bump_every_covered_texture_page() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let handle = 701;
        let base = 0x6e40_0000u64;
        let remap_base = 0x6e80_0000u64;
        let size = 0x20000u64;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, size as u32, 0x4b00_0000));
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, base, 2, 0x10000, 1).result,
            0
        );
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, remap_base, 2, 0x10000, 3).result,
            0
        );

        let generations_before = [
            nexium_gpu::tex_invalidate::region_gen(base),
            nexium_gpu::tex_invalidate::region_gen(base + 0x10000),
            nexium_gpu::tex_invalidate::region_gen(base + size),
        ];
        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&size.to_le_bytes());
        map[32..40].copy_from_slice(&base.to_le_bytes());

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, map, 40))
                .result,
            0
        );
        for (index, before) in generations_before[..2].iter().enumerate() {
            assert_ne!(
                nexium_gpu::tex_invalidate::region_gen(base + index as u64 * 0x10000),
                *before
            );
        }
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(base + size),
            generations_before[2]
        );

        let generations_before_unmap = [
            nexium_gpu::tex_invalidate::region_gen(base),
            nexium_gpu::tex_invalidate::region_gen(base + 0x10000),
            nexium_gpu::tex_invalidate::region_gen(base + size),
        ];
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc008_4105, base.to_le_bytes().to_vec(), 8))
                .result,
            0
        );
        for (index, before) in generations_before_unmap[..2].iter().enumerate() {
            assert_ne!(
                nexium_gpu::tex_invalidate::region_gen(base + index as u64 * 0x10000),
                *before
            );
        }
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(base + size),
            generations_before_unmap[2]
        );

        let remap_generations_before = [
            nexium_gpu::tex_invalidate::region_gen(remap_base),
            nexium_gpu::tex_invalidate::region_gen(remap_base + 0x10000),
            nexium_gpu::tex_invalidate::region_gen(remap_base + size),
        ];
        let mut remap = vec![0u8; 20];
        remap[4..8].copy_from_slice(&handle.to_le_bytes());
        remap[12..16].copy_from_slice(&((remap_base / 0x10000) as u32).to_le_bytes());
        remap[16..20].copy_from_slice(&2u32.to_le_bytes());

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, remap, 0))
                .result,
            0
        );
        for (index, before) in remap_generations_before[..2].iter().enumerate() {
            assert_ne!(
                nexium_gpu::tex_invalidate::region_gen(remap_base + index as u64 * 0x10000),
                *before
            );
        }
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(remap_base + size),
            remap_generations_before[2]
        );
    }

    #[test]
    fn sparse_remap_holes_shadow_aliases_and_preserve_parent_va_reservation() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x0400_0000u64;
        let target = base + 0x10000;
        let parent_size = 0x30000u64;
        let first_handle = 711;
        let second_handle = 712;
        let first_cpu = 0x4c00_0000u64;
        let second_cpu = 0x4d00_0000u64;
        nvdrv.nvmap_handles.insert(
            first_handle,
            test_nvmap_handle(first_handle, 0x10000, first_cpu),
        );
        nvdrv.nvmap_handles.insert(
            second_handle,
            test_nvmap_handle(second_handle, 0x10000, second_cpu),
        );

        let mut allocation = vec![0u8; 24];
        allocation[0..4].copy_from_slice(&3u32.to_le_bytes());
        allocation[4..8].copy_from_slice(&0x10000u32.to_le_bytes());
        allocation[8..12].copy_from_slice(&3u32.to_le_bytes());
        allocation[16..24].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc018_4102, allocation, 24))
                .result,
            0
        );

        let remap = |handle: u32| {
            let mut entry = vec![0u8; 20];
            entry[4..8].copy_from_slice(&handle.to_le_bytes());
            entry[12..16].copy_from_slice(&((target / 0x10000) as u32).to_le_bytes());
            entry[16..20].copy_from_slice(&1u32.to_le_bytes());
            entry
        };
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, remap(first_handle), 0))
                .result,
            0
        );
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(target + 0x800),
            Some(first_cpu + 0x800)
        );
        nexium_gpu::pitch_oracle::record_pitch_dst(target, 0x10000);
        let generations = [
            nexium_gpu::tex_invalidate::region_gen(target - 0x10000),
            nexium_gpu::tex_invalidate::region_gen(target),
            nexium_gpu::tex_invalidate::region_gen(target + 0x10000),
        ];

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, remap(0), 0))
                .result,
            0
        );
        {
            let mappings = nvdrv.gpu.mappings.read();
            assert_eq!(mappings.cpu_address_for(target + 0x800), None);
            assert_eq!(mappings.cpu_address_for_any32(target + 0x800), None);
            assert_eq!(mappings.nvmap_id_for(target + 0x800), None);
            assert_eq!(mappings.mapping_epoch_for(target + 0x800), None);
            assert!(mappings
                .gpu_regions_for_cpu_range(first_cpu, 0x10000)
                .is_empty());
        }
        assert!(!nexium_gpu::pitch_oracle::is_pitch_dst(target + 0x800));
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(target - 0x10000),
            generations[0]
        );
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen(target),
            generations[1]
        );
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(target + 0x10000),
            generations[2]
        );
        assert_eq!(nvdrv.gpu.alloc_va(parent_size, false), base + parent_size);

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, remap(second_handle), 0))
                .result,
            0
        );
        let mappings = nvdrv.gpu.mappings.read();
        assert_eq!(
            mappings.cpu_address_for(target + 0x800),
            Some(second_cpu + 0x800)
        );
        assert_eq!(
            mappings.gpu_regions_for_cpu_range(second_cpu, 0x10000),
            vec![(target, 0x10000)]
        );
    }

    #[test]
    fn free_space_requires_the_same_fd_and_exact_allocation_tuple() {
        let mut nvdrv = Nvdrv::new();
        let first_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let second_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x7400_0000;
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, first_fd, base, 2, 0x10000, 1).result,
            0
        );

        assert_eq!(
            free_as_gpu_space(&mut nvdrv, second_fd, base, 2, 0x10000).result,
            0xB
        );
        nvdrv.close(second_fd);
        assert!(nvdrv.as_gpu_states[&first_fd]
            .allocations
            .contains_key(&base));
        assert_eq!(
            free_as_gpu_space(&mut nvdrv, first_fd, base, 1, 0x10000).result,
            0xB
        );
        assert_eq!(
            free_as_gpu_space(&mut nvdrv, first_fd, base, 32, 0x1000).result,
            0xB
        );
        assert_eq!(
            free_as_gpu_space(&mut nvdrv, first_fd, base, 2, 0x10000).result,
            0
        );
        assert_eq!(
            free_as_gpu_space(&mut nvdrv, first_fd, base, 2, 0x10000).result,
            0xB
        );
    }

    #[test]
    fn as_gpu_fds_cannot_remap_or_unmap_each_others_roots() {
        let mut nvdrv = Nvdrv::new();
        let first_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let second_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x7480_0000;
        let handle = 720;
        let cpu = 0x5480_0000;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, cpu));
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, first_fd, base, 1, 0x10000, 1).result,
            0
        );
        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        map[32..40].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(first_fd, 0xc028_4106, map, 40))
                .result,
            0
        );

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(
                    second_fd,
                    0xc008_4105,
                    base.to_le_bytes().to_vec(),
                    8,
                ))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(base), Some(cpu));
        let mut remap = vec![0u8; 40];
        remap[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        remap[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        remap[32..40].copy_from_slice(&base.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(second_fd, 0xc028_4106, remap, 40))
                .result,
            0xB
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(base), Some(cpu));
    }

    #[test]
    fn remap_entries_require_sparse_bounds_valid_handles_and_atomic_preflight() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let non_sparse = 0x7500_0000;
        let sparse = 0x7510_0000;
        let handle = 721;
        let zero_handle = 722;
        let cpu = 0x5510_0000;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, cpu));
        nvdrv
            .nvmap_handles
            .insert(zero_handle, test_nvmap_handle(zero_handle, 0x10000, 0));
        let mut zero_map = vec![0u8; 40];
        zero_map[8..12].copy_from_slice(&zero_handle.to_le_bytes());
        zero_map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, zero_map, 40))
                .result,
            0xB
        );
        let missing = as_gpu_remap_entry(sparse, 1, handle, 0);
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, missing, 0))
                .result,
            0xB
        );
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, non_sparse, 1, 0x10000, 1).result,
            0
        );
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(
                    fd,
                    0x4014_4114,
                    as_gpu_remap_entry(non_sparse, 1, handle, 0),
                    0,
                ))
                .result,
            0xB
        );
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, sparse, 2, 0x10000, 3).result,
            0
        );
        for invalid in [
            as_gpu_remap_entry(sparse + 0x20000, 1, handle, 0),
            as_gpu_remap_entry(sparse, 2, handle, 0),
            as_gpu_remap_entry(sparse, 1, zero_handle, 0),
            as_gpu_remap_entry(sparse, 0, handle, 0),
        ] {
            assert_eq!(
                nvdrv
                    .dispatch_ioctl(request(fd, 0x4014_4114, invalid, 0))
                    .result,
                0xB
            );
        }
        let valid = as_gpu_remap_entry(sparse, 1, handle, 0);
        let invalid = as_gpu_remap_entry(sparse + 0x10000, 1, handle, 1);
        let mut batch = valid.clone();
        batch.extend_from_slice(&invalid);
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4028_4114, batch, 0))
                .result,
            0xB
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(sparse), None);
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4014_4114, valid, 0))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(sparse), Some(cpu));
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc008_4105, sparse.to_le_bytes().to_vec(), 8,))
                .result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(sparse), Some(cpu));
    }

    #[test]
    fn free_space_clears_its_range_but_preserves_negative_remap_aliases() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let base = 0x7600_0000;
        let source = base + 0x10000;
        let alias = base - 0x10000;
        let handle = 723;
        let cpu = 0x5610_0000;
        nvdrv
            .nvmap_handles
            .insert(handle, test_nvmap_handle(handle, 0x10000, cpu));
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, base, 3, 0x10000, 3).result,
            0
        );
        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        map[32..40].copy_from_slice(&source.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, map, 40))
                .result,
            0
        );
        let mut remap = vec![0u8; 40];
        remap[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        remap[16..24].copy_from_slice(&(-0x20000i64).to_le_bytes());
        remap[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        remap[32..40].copy_from_slice(&source.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, remap, 40))
                .result,
            0
        );
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(alias),
            Some(cpu - 0x20000)
        );

        assert_eq!(
            free_as_gpu_space(&mut nvdrv, fd, base, 3, 0x10000).result,
            0
        );
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(base), None);
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(source), None);
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(alias),
            Some(cpu - 0x20000)
        );
        assert!(nvdrv.gpu.alloc_va_fixed_exclusive(base, 0x30000));
        assert!(nvdrv.gpu.free_va(base, 0x30000));
    }

    #[test]
    fn non_sparse_free_space_preserves_a_preexisting_remap_alias() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let source = 0x7701_0000;
        let target = source - 0x10000;
        let cpu = 0x5711_0000;
        nvdrv
            .gpu
            .mappings
            .write()
            .add_as_gpu_mapping(fd, source, 0x10000, cpu, 727, None, None, true);
        let mut remap = vec![0u8; 40];
        remap[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        remap[16..24].copy_from_slice(&(-0x10000i64).to_le_bytes());
        remap[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        remap[32..40].copy_from_slice(&source.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4106, remap, 40))
                .result,
            0
        );
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(target + 0x800),
            Some(cpu - 0x10000 + 0x800)
        );
        assert_eq!(
            alloc_as_gpu_space(&mut nvdrv, fd, target, 1, 0x10000, 1).result,
            0
        );

        assert_eq!(
            free_as_gpu_space(&mut nvdrv, fd, target, 1, 0x10000).result,
            0
        );
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(target + 0x800),
            Some(cpu - 0x10000 + 0x800)
        );
        assert!(nvdrv.gpu.alloc_va_fixed_exclusive(target, 0x10000));
        assert!(nvdrv.gpu.free_va(target, 0x10000));
    }

    #[test]
    fn alloc_as_ex_accepts_explicit_default_range() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let mut input = vec![0u8; 40];
        input[8..12].copy_from_slice(&0x10000u32.to_le_bytes());
        input[16..24].copy_from_slice(&0x0400_0000u64.to_le_bytes());
        input[24..32].copy_from_slice(&(1u64 << 37).to_le_bytes());
        input[32..40].copy_from_slice(&(1u64 << 34).to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0x4028_4109, input, 0))
                .result,
            0
        );
        assert!(nvdrv.as_gpu_states[&fd].initialized);
        let regions = nvdrv.dispatch_ioctl(request(fd, 0xc040_4108, vec![0; 64], 64));
        assert_eq!(regions.result, 0);
        let region_u64 = |offset: usize| {
            u64::from_le_bytes(regions.data[offset..offset + 8].try_into().unwrap())
        };
        assert_eq!(region_u64(16), 0x0400_0000);
        assert_eq!(region_u64(40), 1u64 << 34);
        assert_eq!(region_u64(40) + region_u64(56) * 0x10000, 1u64 << 37);
    }

    #[test]
    fn alloc_as_ex_rejects_custom_ranges_without_state_mutation() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        for (offset, value) in [(16, 0x0400_0000u64), (24, 1u64 << 37), (32, 1u64 << 34)] {
            let mut input = vec![0u8; 40];
            input[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
            input[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert_eq!(
                nvdrv
                    .dispatch_ioctl(request(fd, 0xc028_4109, input, 40))
                    .result,
                0xB
            );
            let state = &nvdrv.as_gpu_states[&fd];
            assert!(!state.initialized);
            assert_eq!(state.big_page_size, AS_GPU_DEFAULT_BIG_PAGE_SIZE);
            assert!(state.allocations.is_empty());
        }

        let mut valid = vec![0u8; 40];
        valid[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4109, valid, 40))
                .result,
            0
        );
        assert!(nvdrv.as_gpu_states[&fd].initialized);
        assert_eq!(nvdrv.as_gpu_states[&fd].big_page_size, 0x20000);
    }

    #[test]
    fn alloc_as_ex_rejects_reinitialization_with_only_dynamic_maps() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let mut alloc_as = vec![0u8; 40];
        alloc_as[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4109, alloc_as.clone(), 40))
                .result,
            0
        );
        let handle = 730;
        let mut nvmap = test_nvmap_handle(handle, 0x10000, 0x5c00_0000);
        nvmap.align = 0x20000;
        nvdrv.nvmap_handles.insert(handle, nvmap);
        let mut map = vec![0u8; 40];
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        let mapped = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, map, 40));
        assert_eq!(mapped.result, 0);
        let gpu_va = u64::from_le_bytes(mapped.data[32..40].try_into().unwrap());
        assert!(nvdrv.as_gpu_states[&fd].allocations.is_empty());
        let mapping_count = nvdrv.gpu.mappings.read().iter().count();

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4109, alloc_as, 40))
                .result,
            0x8
        );
        assert!(nvdrv.as_gpu_states[&fd].initialized);
        assert_eq!(nvdrv.as_gpu_states[&fd].big_page_size, 0x20000);
        assert!(nvdrv.as_gpu_states[&fd].allocations.is_empty());
        assert_eq!(nvdrv.gpu.mappings.read().iter().count(), mapping_count);
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(gpu_va),
            Some(0x5c00_0000)
        );
    }

    #[test]
    fn dynamic_map_implicitly_initializes_defaults_before_alloc_as_ex() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let handle = 731;
        let mut nvmap = test_nvmap_handle(handle, 0x10000, 0x5d00_0000);
        nvmap.align = 0x10000;
        nvdrv.nvmap_handles.insert(handle, nvmap);
        let mut map = vec![0u8; 40];
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
        let mapped = nvdrv.dispatch_ioctl(request(fd, 0xc028_4106, map, 40));
        assert_eq!(mapped.result, 0);
        let gpu_va = u64::from_le_bytes(mapped.data[32..40].try_into().unwrap());
        assert!(nvdrv.as_gpu_states[&fd].initialized);
        assert_eq!(
            nvdrv.as_gpu_states[&fd].big_page_size,
            AS_GPU_DEFAULT_BIG_PAGE_SIZE
        );
        let mapping_count = nvdrv.gpu.mappings.read().iter().count();
        let mut alloc_as = vec![0u8; 40];
        alloc_as[8..12].copy_from_slice(&0x20000u32.to_le_bytes());

        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd, 0xc028_4109, alloc_as, 40))
                .result,
            0x8
        );
        assert!(nvdrv.as_gpu_states[&fd].initialized);
        assert_eq!(
            nvdrv.as_gpu_states[&fd].big_page_size,
            AS_GPU_DEFAULT_BIG_PAGE_SIZE
        );
        assert!(nvdrv.as_gpu_states[&fd].allocations.is_empty());
        assert_eq!(nvdrv.gpu.mappings.read().iter().count(), mapping_count);
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(gpu_va),
            Some(0x5d00_0000)
        );
    }

    #[test]
    fn mixed_fd_big_page_sizes_align_shared_dynamic_va_allocations() {
        let mut nvdrv = Nvdrv::new();
        let fd_64k = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let fd_128k = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let mut alloc_as = vec![0u8; 40];
        alloc_as[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(fd_128k, 0xc028_4109, alloc_as, 40))
                .result,
            0
        );
        let handle_64k = 728;
        let handle_128k = 729;
        let mut map_64k = test_nvmap_handle(handle_64k, 0x10000, 0x5a00_0000);
        map_64k.align = 0x10000;
        let mut map_128k = test_nvmap_handle(handle_128k, 0x10000, 0x5b00_0000);
        map_128k.align = 0x20000;
        nvdrv.nvmap_handles.insert(handle_64k, map_64k);
        nvdrv.nvmap_handles.insert(handle_128k, map_128k);
        let map = |handle: u32| {
            let mut input = vec![0u8; 40];
            input[8..12].copy_from_slice(&handle.to_le_bytes());
            input[24..32].copy_from_slice(&0x10000u64.to_le_bytes());
            input
        };
        let first = nvdrv.dispatch_ioctl(request(fd_64k, 0xc028_4106, map(handle_64k), 40));
        let second = nvdrv.dispatch_ioctl(request(fd_128k, 0xc028_4106, map(handle_128k), 40));
        assert_eq!((first.result, second.result), (0, 0));
        let first_va = u64::from_le_bytes(first.data[32..40].try_into().unwrap());
        let second_va = u64::from_le_bytes(second.data[32..40].try_into().unwrap());
        assert_eq!(first_va & 0xffff, 0);
        assert_eq!(second_va & 0x1ffff, 0);
        assert_eq!(second_va, first_va + 0x20000);
    }

    #[test]
    fn configured_big_pages_control_dynamic_mapping_alignment_and_close_cleanup() {
        let mut nvdrv = Nvdrv::new();
        let first_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let second_fd = nvdrv.open("/dev/nvhost-as-gpu").unwrap();
        let mut alloc_as = vec![0u8; 40];
        alloc_as[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
        assert_eq!(
            nvdrv
                .dispatch_ioctl(request(first_fd, 0xc028_4109, alloc_as, 40))
                .result,
            0
        );
        let small_handle = 724;
        let big_handle = 725;
        let other_handle = 726;
        let mut small = test_nvmap_handle(small_handle, 0x10000, 0x5700_0000);
        small.align = 0x10000;
        let mut big = test_nvmap_handle(big_handle, 0x28000, 0x5800_0000);
        big.align = 0x20000;
        let other = test_nvmap_handle(other_handle, 0x10000, 0x5900_0000);
        nvdrv.nvmap_handles.insert(small_handle, small);
        nvdrv.nvmap_handles.insert(big_handle, big);
        nvdrv.nvmap_handles.insert(other_handle, other);
        let map = |handle: u32, size: u64| {
            let mut input = vec![0u8; 40];
            input[8..12].copy_from_slice(&handle.to_le_bytes());
            input[24..32].copy_from_slice(&size.to_le_bytes());
            input
        };
        let small = nvdrv.dispatch_ioctl(request(
            first_fd,
            0xc028_4106,
            map(small_handle, 0x10000),
            40,
        ));
        let big =
            nvdrv.dispatch_ioctl(request(first_fd, 0xc028_4106, map(big_handle, 0x28000), 40));
        let other = nvdrv.dispatch_ioctl(request(
            second_fd,
            0xc028_4106,
            map(other_handle, 0x10000),
            40,
        ));
        assert_eq!((small.result, big.result, other.result), (0, 0, 0));
        let small_va = u64::from_le_bytes(small.data[32..40].try_into().unwrap());
        let big_va = u64::from_le_bytes(big.data[32..40].try_into().unwrap());
        let other_va = u64::from_le_bytes(other.data[32..40].try_into().unwrap());
        assert!(small_va < 0x4_0000_0000);
        assert_eq!(big_va & 0x1ffff, 0);
        assert!(big_va >= 0x4_0000_0000);

        nvdrv.close(first_fd);
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(small_va), None);
        assert_eq!(nvdrv.gpu.mappings.read().cpu_address_for(big_va), None);
        assert_eq!(
            nvdrv.gpu.mappings.read().cpu_address_for(other_va),
            Some(0x5900_0000)
        );
        assert!(nvdrv.gpu.alloc_va_fixed_exclusive(big_va, 0x40000));
        assert!(nvdrv.gpu.free_va(big_va, 0x40000));
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
    fn repeated_video_mapping_preserves_addresses_and_mapping_generation() {
        let mut nvdrv = Nvdrv::new();
        let fd = nvdrv.open("/dev/nvhost-nvdec").unwrap();
        nvdrv
            .nvmap_handles
            .insert(497, test_nvmap_handle(497, 0x8000, 0x4a07_f000_00));
        nvdrv
            .nvmap_handles
            .insert(498, test_nvmap_handle(498, 0x8000, 0x4a07_f100_00));
        let mut input = vec![0u8; 0x0c + 2 * 8];
        write_u32(&mut input, 0, 1);
        write_u32(&mut input, 0x0c, 497);
        write_u32(&mut input, 0x14, 498);
        let map_request = || request(fd, 0xc01c_0009, input.clone(), input.len());
        assert!(!nvdrv.channel_map_reuses_addresses(&map_request()));
        let first = nvdrv.dispatch_ioctl(map_request());
        let generation = nvdrv.gpu.mappings.read().generation();
        assert!(nvdrv.channel_map_reuses_addresses(&map_request()));
        let second = nvdrv.dispatch_ioctl(map_request());
        assert_eq!(first.data, second.data);
        assert_eq!(nvdrv.gpu.mappings.read().generation(), generation);
        assert_eq!(nvdrv.nvmap_handles[&497].channel_pin_count, 2);
        nvdrv.unpin_channel_buffer(497);
        nvdrv.unpin_channel_buffer(497);
        assert!(nvdrv.channel_map_reuses_addresses(&map_request()));
        write_u32(&mut input, 0, 2);
        assert!(!nvdrv.channel_map_reuses_addresses(&request(
            fd,
            0xc01c_0009,
            input.clone(),
            input.len()
        )));
        nvdrv.dispatch_ioctl(request(fd, 0xc01c_0009, input.clone(), input.len()));
        assert!(nvdrv.channel_map_reuses_addresses(&request(
            fd,
            0xc01c_0009,
            input.clone(),
            input.len()
        )));
        write_u32(&mut input, 0x14, 999);
        assert!(!nvdrv.channel_map_reuses_addresses(&request(
            fd,
            0xc01c_0009,
            input.clone(),
            input.len()
        )));
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
