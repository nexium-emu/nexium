use super::{Kernel, MUTEX_HAS_LISTENERS};
use crate::kernel::cpu_local::{cpu_mut, cpu_ref};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use crate::services::audio_renderer::behavior as audren_behavior;
use crate::kernel::{
    present_delivery_lane, AudioAdpcmContext, AudioAdpcmDecodeState, AudioAdpcmStreamKey,
    AudioRendererState, PresentDeliveryLanes, PresentMetadata, PresentMetadataQueue,
};
use nexium_common::result::{
    KERNEL_CANCELLED, KERNEL_INVALID_ADDRESS, KERNEL_INVALID_ENUM_VALUE, KERNEL_INVALID_HANDLE,
    KERNEL_INVALID_PRIORITY, KERNEL_INVALID_THREAD_STATE, KERNEL_NOT_IMPLEMENTED, KERNEL_TIMEOUT,
    SUCCESS,
};
use nexium_ipc as ipc;

const KERNEL_EVENT_INVALID_STATE: u32 = 1 | (125 << 9);

const AUDIO_PCM_INT16: u8 = 2;
const AUDIO_PCM_FLOAT: u8 = 5;
const AUDIO_PCM_ADPCM: u8 = 6;

fn audio_debug_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_AUDIO_DEBUG").is_some())
}

fn diagnostics_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_DIAG").is_some())
}

fn async_present_pipeline_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_ASYNC_PRESENT_PIPELINE")
                .ok()
                .as_deref(),
            Some("0") | Some("false") | Some("off") | Some("no")
        )
    })
}

fn try_select_ordered_present_source<R>(
    maxwell_dma: &parking_lot::Mutex<nexium_nvdrv::gpu::engines::MaxwellDma>,
    select: impl FnOnce(&nexium_nvdrv::gpu::engines::MaxwellDma) -> Option<R>,
) -> Option<R> {
    let maxwell_dma = maxwell_dma.try_lock()?;
    select(&maxwell_dma)
}

struct AcquiredBufferSlotGuard {
    queues: std::sync::Arc<
        parking_lot::Mutex<std::collections::HashMap<u32, nexium_nvdrv::BufferQueue>>,
    >,
    binder_id: u32,
    slot: u32,
}

impl AcquiredBufferSlotGuard {
    fn new(
        queues: std::sync::Arc<
            parking_lot::Mutex<std::collections::HashMap<u32, nexium_nvdrv::BufferQueue>>,
        >,
        binder_id: u32,
        slot: u32,
    ) -> Self {
        Self {
            queues,
            binder_id,
            slot,
        }
    }
}

impl Drop for AcquiredBufferSlotGuard {
    fn drop(&mut self) {
        let mut queues = self.queues.lock();
        let Some(queue) = queues.get_mut(&self.binder_id) else {
            return;
        };
        if !queue.release(self.slot) {
            log::warn!(
                "IGBP failed to release acquired slot binder={} slot={}",
                self.binder_id,
                self.slot
            );
        }
    }
}

fn release_rejected_present_slot_after_fences(
    queues: &std::sync::Arc<
        parking_lot::Mutex<std::collections::HashMap<u32, nexium_nvdrv::BufferQueue>>,
    >,
    binder_id: u32,
    slot: u32,
    fences: &[(u32, u32)],
    timeout: std::time::Duration,
    mut fence_reached: impl FnMut(u32, u32) -> bool,
) -> bool {
    let started = std::time::Instant::now();
    for &(syncpt_id, threshold) in fences {
        while !fence_reached(syncpt_id, threshold) {
            if started.elapsed() >= timeout {
                log::error!(
                    "IGBP rejected present fence timeout binder={} slot={} syncpt={} threshold={}; retaining Acquired slot",
                    binder_id,
                    slot,
                    syncpt_id,
                    threshold
                );
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }

    let mut queues = queues.lock();
    let Some(queue) = queues.get_mut(&binder_id) else {
        log::warn!(
            "IGBP rejected present lost BufferQueue binder={} slot={}; slot not released",
            binder_id,
            slot
        );
        return false;
    };
    if !queue.release(slot) {
        log::warn!(
            "IGBP rejected present failed to release acquired slot binder={} slot={}",
            binder_id,
            slot
        );
        return false;
    }
    true
}

#[cfg(test)]
mod bufferqueue_present_tests {
    use super::{
        release_rejected_present_slot_after_fences, submit_ordered_gpu_present,
        try_select_ordered_present_source, PresentDeliveryLanes, PresentMetadata,
        PresentMetadataQueue,
    };
    use nexium_nvdrv::bufferqueue::SlotState;
    use nexium_nvdrv::gpu::engines::MaxwellDma;
    use nexium_nvdrv::{
        BufferQueue, GraphicBuffer, Nvdrv, PipelinedPresentCompletion, PipelinedPresentFrame,
        PipelinedPresentReadback, PipelinedPresentSubmission,
    };
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn acquired_slot() -> Arc<Mutex<HashMap<u32, BufferQueue>>> {
        let mut queue = BufferQueue::new(7);
        queue.set_preallocated(0, GraphicBuffer::default());
        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(queue.queue_and_acquire(0));
        Arc::new(Mutex::new(HashMap::from([(7, queue)])))
    }

    #[test]
    fn rejected_present_releases_only_after_elided_fence_retires() {
        let queues = acquired_slot();
        assert!(release_rejected_present_slot_after_fences(
            &queues,
            7,
            0,
            &[(3, 9)],
            Duration::ZERO,
            |syncpt_id, threshold| syncpt_id == 3 && threshold == 9,
        ));
        let mut queues = queues.lock();
        let queue = queues.get_mut(&7).unwrap();
        assert_eq!(queue.slot_state(0), Some(SlotState::Free));
        assert_eq!(queue.try_dequeue(), Some(0));
    }

    #[test]
    fn rejected_present_timeout_retains_acquired_slot() {
        let queues = acquired_slot();
        assert!(!release_rejected_present_slot_after_fences(
            &queues,
            7,
            0,
            &[(3, 9)],
            Duration::ZERO,
            |_, _| false,
        ));
        let mut queues = queues.lock();
        let queue = queues.get_mut(&7).unwrap();
        assert_eq!(queue.slot_state(0), Some(SlotState::Acquired));
        assert_eq!(queue.try_dequeue(), None);
    }

    #[test]
    fn ordered_present_source_contention_returns_for_fallback() {
        let maxwell_dma = Arc::new(Mutex::new(MaxwellDma::new()));
        let held = maxwell_dma.lock();
        let worker_dma = Arc::clone(&maxwell_dma);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let selected = try_select_ordered_present_source(&worker_dma, |_| Some("exact"));
            tx.send(selected.or(Some("fallback"))).unwrap();
        });

        let selected = rx.recv_timeout(Duration::from_secs(1));
        drop(held);
        worker.join().unwrap();
        assert_eq!(selected.unwrap(), Some("fallback"));
    }

    #[test]
    fn completed_readback_uses_its_original_swap_deadline() {
        let nvdrv = Nvdrv::new();
        let now = std::time::Instant::now();
        let original_deadline = now + Duration::from_millis(33);
        let metadata: PresentMetadataQueue = Arc::new(Mutex::new(HashMap::from([(
            41,
            PresentMetadata {
                read_rect: None,
                transform: 0,
                queue_crop: None,
                present_at: Some(original_deadline),
            },
        )])));
        let lanes: PresentDeliveryLanes = Arc::new(Mutex::new(HashMap::new()));

        let emitted = submit_ordered_gpu_present(
            |_| PipelinedPresentReadback {
                completion: Some(PipelinedPresentCompletion::Ready(PipelinedPresentFrame {
                    present_id: 41,
                    width: 1,
                    height: 1,
                    pixels: vec![1, 2, 3, 255],
                    flip_y: Some(false),
                })),
                submission: PipelinedPresentSubmission::SourceUnavailable,
            },
            Arc::clone(&metadata),
            lanes,
            7,
            42,
            Arc::clone(&nvdrv.frame_queue),
            Arc::clone(&nvdrv.stats),
            1,
            1,
            0,
            None,
            Some(now),
        );

        assert!(emitted);
        assert!(nvdrv.drain_next_frame_due(now).is_none());
        assert_eq!(
            nvdrv
                .drain_next_frame_due(original_deadline)
                .unwrap()
                .pixels,
            vec![1, 2, 3, 255]
        );
        assert!(metadata.lock().is_empty());
    }
}

#[derive(Clone, Copy)]
enum OrderedPresentProfileOutcome {
    IdentityRejected,
    TargetRejected,
    Enqueued,
    Coalesced,
    Unavailable,
}

fn note_ordered_present_profile(outcome: OrderedPresentProfileOutcome) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static IDENTITY_REJECTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static TARGET_REJECTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static ENQUEUED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static COALESCED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static UNAVAILABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_ASYNC_GPU_PROFILE").is_some()) {
        return;
    }
    use std::sync::atomic::Ordering;
    match outcome {
        OrderedPresentProfileOutcome::IdentityRejected => {
            IDENTITY_REJECTED.fetch_add(1, Ordering::Relaxed);
        }
        OrderedPresentProfileOutcome::TargetRejected => {
            TARGET_REJECTED.fetch_add(1, Ordering::Relaxed);
        }
        OrderedPresentProfileOutcome::Enqueued => {
            ENQUEUED.fetch_add(1, Ordering::Relaxed);
        }
        OrderedPresentProfileOutcome::Coalesced => {
            COALESCED.fetch_add(1, Ordering::Relaxed);
        }
        OrderedPresentProfileOutcome::Unavailable => {
            UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
        }
    }
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if calls % 60 == 0 {
        log::warn!(
            "[async-present-prof] calls={} identity_rejected={} target_rejected={} enqueued={} coalesced={} unavailable={}",
            calls,
            IDENTITY_REJECTED.load(Ordering::Relaxed),
            TARGET_REJECTED.load(Ordering::Relaxed),
            ENQUEUED.load(Ordering::Relaxed),
            COALESCED.load(Ordering::Relaxed),
            UNAVAILABLE.load(Ordering::Relaxed),
        );
    }
}

fn pcm_bytes_per_sample(sample_format: u8) -> Option<usize> {
    match sample_format {
        AUDIO_PCM_INT16 => Some(2),
        AUDIO_PCM_FLOAT => Some(4),
        _ => None,
    }
}

fn decode_pcm_stereo(
    sample_format: u8,
    channels: usize,
    bytes: &[u8],
    left: &mut [f32],
    right: &mut [f32],
) -> usize {
    let Some(bytes_per_sample) = pcm_bytes_per_sample(sample_format) else {
        return 0;
    };
    if channels != 1 && channels != 2 {
        return 0;
    }
    let frame_stride = bytes_per_sample * channels;
    let frame_count = left.len().min(right.len()).min(bytes.len() / frame_stride);
    let decode_sample = |offset: usize| -> f32 {
        match sample_format {
            AUDIO_PCM_INT16 => {
                i16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as f32 / 32768.0
            }
            AUDIO_PCM_FLOAT => {
                let sample = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                if sample.is_finite() {
                    sample
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    };
    for frame in 0..frame_count {
        let offset = frame * frame_stride;
        left[frame] = decode_sample(offset);
        right[frame] = if channels == 2 {
            decode_sample(offset + bytes_per_sample)
        } else {
            left[frame]
        };
    }
    frame_count
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AudioWaveBufferSpan {
    frames: u32,
    looping: bool,
    loop_count: i32,
}

impl Default for AudioWaveBufferSpan {
    fn default() -> Self {
        Self {
            frames: 0,
            looping: false,
            loop_count: AUDIO_WAVE_BUFFER_LOOP_INFINITE,
        }
    }
}

const AUDIO_WAVE_BUFFER_LOOP_INFINITE: i32 = -1;

#[derive(Clone, Copy, Debug, Default)]
struct AudioVoiceMixSnapshot {
    wb_index: u16,
    is_new: bool,
    did_mix: bool,
    source_frames: u32,
    wb_count: u32,
    buffers: [AudioWaveBufferSpan; 4],
}

const AUDIO_RENDER_BLOCK_FRAMES: usize = 240;
const AUDIO_RING_HIGH_WATER_FRAMES: usize = 5_760;
const AUDIO_MAX_BLOCKS_PER_UPDATE: usize = 8;

fn audio_blocks_to_produce(queued_frames: usize) -> usize {
    if queued_frames >= AUDIO_RING_HIGH_WATER_FRAMES {
        return 0;
    }
    ((AUDIO_RING_HIGH_WATER_FRAMES - queued_frames) / AUDIO_RENDER_BLOCK_FRAMES)
        .clamp(1, AUDIO_MAX_BLOCKS_PER_UPDATE)
}

fn advance_audio_wave_buffers(
    previous_progress: u64,
    source_frames: u64,
    buffers: &[AudioWaveBufferSpan; 4],
) -> (u64, u32, bool) {
    let mut progress = previous_progress.saturating_add(source_frames);
    let mut completed = 0u32;
    for buffer in buffers {
        if buffer.frames == 0 {
            break;
        }
        let frames = buffer.frames as u64;
        if progress < frames {
            return (progress, completed, false);
        }
        if buffer.looping {
            if buffer.loop_count < 0 {
                return (progress % frames, completed, false);
            }
            let plays = (buffer.loop_count as u64).saturating_add(1);
            let total = frames.saturating_mul(plays);
            if progress < total {
                return (progress % frames, completed, false);
            }
            progress -= total;
            completed += 1;
            continue;
        }
        progress -= frames;
        completed += 1;
    }
    (progress, completed, progress != 0)
}

fn audio_source_advance(mut fraction_q15: i32, step_q15: i32, frames: usize) -> (usize, i32) {
    let mut source_frames = 0usize;
    for _ in 0..frames {
        let next = fraction_q15 + step_q15;
        source_frames += (next >> 15) as usize;
        fraction_q15 = next & 0x7fff;
    }
    (source_frames, fraction_q15)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GcAdpcmDecodeResult {
    decoded_samples: usize,
    bytes_read: usize,
    checkpoint: Option<AudioAdpcmContext>,
}

fn gc_adpcm_byte_range(
    start_sample: usize,
    sample_count: usize,
    buffer_size: usize,
) -> Option<(usize, usize)> {
    if sample_count == 0 {
        return Some((0, 0));
    }
    let start_frame = start_sample / 14;
    let start_in_frame = start_sample % 14;
    let start_byte = start_frame
        .checked_mul(8)?
        .checked_add(if start_in_frame == 0 {
            0
        } else {
            1 + start_in_frame / 2
        })?;
    let last_sample = start_sample.checked_add(sample_count)?.checked_sub(1)?;
    let end_byte = (last_sample / 14)
        .checked_mul(8)?
        .checked_add(2 + (last_sample % 14) / 2)?;
    if start_byte >= buffer_size {
        return Some((buffer_size, 0));
    }
    Some((start_byte, end_byte.min(buffer_size) - start_byte))
}

fn can_stream_gc_adpcm(
    state: AudioAdpcmDecodeState,
    key: AudioAdpcmStreamKey,
    requested_sample: usize,
    is_new: bool,
) -> bool {
    !is_new && state.valid && state.key == key && state.next_sample == requested_sample as u64
}

fn decode_gc_adpcm_range(
    data: &[u8],
    coeffs: &[i16; 16],
    initial_context: AudioAdpcmContext,
    start_sample: usize,
    sample_count: usize,
    output_skip: usize,
    output: &mut [i16],
    checkpoint_after: usize,
) -> GcAdpcmDecodeResult {
    output.fill(0);
    let mut context = initial_context;
    let mut sample_in_frame = start_sample % 14;
    let mut read_index = 0usize;
    let mut bytes_read = 0usize;
    let mut decoded_samples = 0usize;
    let mut checkpoint = (checkpoint_after == 0).then_some(context);
    let (mut c0, mut c1) = if sample_in_frame == 0 {
        (0i64, 0i64)
    } else {
        let coefficient_index = ((context.header >> 4) & 0xF) as usize;
        let Some(coefficients) = coeffs.get(coefficient_index * 2..coefficient_index * 2 + 2)
        else {
            return GcAdpcmDecodeResult {
                decoded_samples: 0,
                bytes_read: 0,
                checkpoint: None,
            };
        };
        (coefficients[0] as i64, coefficients[1] as i64)
    };

    while decoded_samples < sample_count {
        if sample_in_frame == 0 {
            let Some(&header) = data.get(read_index) else {
                break;
            };
            context.header = header;
            bytes_read = bytes_read.max(read_index + 1);
            read_index += 1;
            let coefficient_index = ((header >> 4) & 0xF) as usize;
            let Some(coefficients) = coeffs.get(coefficient_index * 2..coefficient_index * 2 + 2)
            else {
                break;
            };
            c0 = coefficients[0] as i64;
            c1 = coefficients[1] as i64;
        }

        let Some(&byte) = data.get(read_index) else {
            break;
        };
        bytes_read = bytes_read.max(read_index + 1);
        let nibble = if sample_in_frame & 1 == 0 {
            byte >> 4
        } else {
            read_index += 1;
            byte & 0xF
        };
        let code = if nibble >= 8 {
            nibble as i64 - 16
        } else {
            nibble as i64
        };
        let scale = (context.header & 0xF) as u32;
        let xn = code * (1i64 << scale);
        let prediction = c0 * context.yn0 as i64 + c1 * context.yn1 as i64;
        let sample = (((xn << 11) + 0x400 + prediction) >> 11).clamp(-0x8000, 0x7FFF);
        context.yn1 = context.yn0;
        context.yn0 = sample as i16;

        if decoded_samples >= output_skip {
            let output_index = decoded_samples - output_skip;
            if let Some(slot) = output.get_mut(output_index) {
                *slot = sample as i16;
            }
        }

        decoded_samples += 1;
        sample_in_frame += 1;
        if sample_in_frame == 14 {
            sample_in_frame = 0;
        }
        if decoded_samples == checkpoint_after {
            checkpoint = Some(context);
        }
    }

    GcAdpcmDecodeResult {
        decoded_samples,
        bytes_read,
        checkpoint,
    }
}

fn audio_renderer_output_slots(
    cmd_id: u32,
    recv_buffers: &[ipc::IpcBuffer],
    recv_statics: &[ipc::IpcBuffer],
) -> (Option<ipc::IpcBuffer>, Option<ipc::IpcBuffer>) {
    fn select(
        first: &[ipc::IpcBuffer],
        second: &[ipc::IpcBuffer],
    ) -> (Option<ipc::IpcBuffer>, Option<ipc::IpcBuffer>) {
        let mut buffers = first
            .iter()
            .chain(second.iter())
            .filter(|buffer| buffer.size > 0 && buffer.addr != 0)
            .copied();
        let output = buffers.next();
        let performance = buffers.find(|buffer| {
            output
                .map(|output| output.addr != buffer.addr)
                .unwrap_or(true)
        });
        (output, performance)
    }

    if cmd_id == 10 {
        select(recv_statics, recv_buffers)
    } else {
        select(recv_buffers, recv_statics)
    }
}

#[cfg(test)]
mod audio_pcm_tests {
    use super::{
        advance_audio_wave_buffers, audio_renderer_output_slots, can_stream_gc_adpcm,
        decode_gc_adpcm_range, decode_pcm_stereo, gc_adpcm_byte_range, AudioAdpcmContext,
        AudioAdpcmDecodeState, AudioAdpcmStreamKey, AudioWaveBufferSpan, AUDIO_PCM_FLOAT,
        AUDIO_PCM_INT16,
    };

    fn adpcm_fixture(frame_count: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(frame_count * 8);
        for frame in 0..frame_count {
            data.push((((frame % 4) as u8) << 4) | ((frame % 3) as u8));
            for byte in 0..7 {
                let high = ((frame * 3 + byte + 1) & 0xF) as u8;
                let low = ((frame * 5 + byte * 2 + 9) & 0xF) as u8;
                data.push((high << 4) | low);
            }
        }
        data
    }

    fn adpcm_coefficients() -> [i16; 16] {
        [
            0x0400, 0, 0x0600, -0x0200, 0x0800, -0x0400, 0x0a00, -0x0600, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
    }

    fn adpcm_stream_key() -> AudioAdpcmStreamKey {
        AudioAdpcmStreamKey {
            wb_index: 2,
            buffer_address: 0x1200_0000,
            buffer_size: 0x4000,
            start_offset: 5,
            end_offset: 40_000,
            context_address: 0x1300_0000,
            coefficient_address: 0x1400_0000,
            sample_rate: 48_000,
            looping: true,
            initial_header: 0x21,
            initial_yn0: 123,
            initial_yn1: -45,
            coefficients: adpcm_coefficients(),
        }
    }

    #[test]
    fn decodes_interleaved_float_stereo() {
        let samples = [0.25f32, -0.5, 1.25, -1.5];
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];

        assert_eq!(
            decode_pcm_stereo(AUDIO_PCM_FLOAT, 2, &bytes, &mut left, &mut right),
            2
        );
        assert_eq!(left, [0.25, 1.25]);
        assert_eq!(right, [-0.5, -1.5]);
    }

    #[test]
    fn decodes_int16_mono_to_both_channels() {
        let bytes = [0x00, 0x40, 0x00, 0xc0];
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];

        assert_eq!(
            decode_pcm_stereo(AUDIO_PCM_INT16, 1, &bytes, &mut left, &mut right),
            2
        );
        assert_eq!(left, [0.5, -0.5]);
        assert_eq!(right, left);
    }

    #[test]
    fn maps_mixed_autoselect_output_slots() {
        let output = nexium_ipc::IpcBuffer {
            addr: 0x1000,
            size: 0x260,
            mode: 0,
        };
        let performance = nexium_ipc::IpcBuffer {
            addr: 0x2000,
            size: 0x1000,
            mode: 0,
        };

        let slots = audio_renderer_output_slots(10, &[performance], &[output]);
        assert_eq!(slots.0.map(|buffer| buffer.addr), Some(output.addr));
        assert_eq!(slots.1.map(|buffer| buffer.addr), Some(performance.addr));
    }

    #[test]
    fn a_full_ring_stops_producing_and_a_shallow_one_keeps_the_old_single_block() {
        assert_eq!(
            super::audio_blocks_to_produce(super::AUDIO_RING_HIGH_WATER_FRAMES),
            0
        );
        assert_eq!(
            super::audio_blocks_to_produce(super::AUDIO_RING_HIGH_WATER_FRAMES + 4_800),
            0
        );
        assert_eq!(
            super::audio_blocks_to_produce(super::AUDIO_RING_HIGH_WATER_FRAMES - 1),
            1
        );
        assert_eq!(
            super::audio_blocks_to_produce(
                super::AUDIO_RING_HIGH_WATER_FRAMES - super::AUDIO_RENDER_BLOCK_FRAMES
            ),
            1
        );
    }

    #[test]
    fn a_drained_ring_refills_in_a_bounded_burst() {
        assert_eq!(
            super::audio_blocks_to_produce(0),
            super::AUDIO_MAX_BLOCKS_PER_UPDATE
        );
        assert_eq!(super::audio_blocks_to_produce(4_800), 4);

        let mut queued = 0usize;
        let mut updates = 0u32;
        while queued < super::AUDIO_RING_HIGH_WATER_FRAMES {
            queued += super::audio_blocks_to_produce(queued) * super::AUDIO_RENDER_BLOCK_FRAMES;
            updates += 1;
            assert!(updates < 24);
        }
        assert_eq!(queued, super::AUDIO_RING_HIGH_WATER_FRAMES);
        assert!(updates <= 6);
    }

    #[test]
    fn a_finite_loop_count_completes_the_buffer_but_infinite_never_does() {
        let finite = [
            AudioWaveBufferSpan {
                frames: 100,
                looping: true,
                loop_count: 2,
            },
            AudioWaveBufferSpan {
                frames: 50,
                looping: false,
                ..Default::default()
            },
            AudioWaveBufferSpan::default(),
            AudioWaveBufferSpan::default(),
        ];
        assert_eq!(advance_audio_wave_buffers(0, 250, &finite), (50, 0, false));
        assert_eq!(advance_audio_wave_buffers(0, 300, &finite), (0, 1, false));
        assert_eq!(advance_audio_wave_buffers(0, 330, &finite), (30, 1, false));

        let infinite = [
            AudioWaveBufferSpan {
                frames: 100,
                looping: true,
                loop_count: super::AUDIO_WAVE_BUFFER_LOOP_INFINITE,
            },
            AudioWaveBufferSpan::default(),
            AudioWaveBufferSpan::default(),
            AudioWaveBufferSpan::default(),
        ];
        assert_eq!(
            advance_audio_wave_buffers(0, 100_000, &infinite),
            (0, 0, false)
        );
    }

    #[test]
    fn advances_across_unequal_wave_buffers() {
        let buffers = [
            AudioWaveBufferSpan {
                frames: 100,
                looping: false,
                ..Default::default()
            },
            AudioWaveBufferSpan {
                frames: 300,
                looping: false,
                ..Default::default()
            },
            AudioWaveBufferSpan::default(),
            AudioWaveBufferSpan::default(),
        ];

        assert_eq!(
            advance_audio_wave_buffers(90, 250, &buffers),
            (240, 1, false)
        );
    }

    #[test]
    fn preserves_completion_before_a_looping_buffer() {
        let buffers = [
            AudioWaveBufferSpan {
                frames: 100,
                looping: false,
                ..Default::default()
            },
            AudioWaveBufferSpan {
                frames: 80,
                looping: true,
                ..Default::default()
            },
            AudioWaveBufferSpan::default(),
            AudioWaveBufferSpan::default(),
        ];

        assert_eq!(advance_audio_wave_buffers(90, 250, &buffers), (0, 1, false));
    }

    #[test]
    fn adpcm_chunked_decode_matches_one_shot_across_frame_boundaries() {
        let data = adpcm_fixture(6);
        let coefficients = adpcm_coefficients();
        let initial = AudioAdpcmContext {
            header: 0,
            yn0: 321,
            yn1: -123,
        };
        let mut expected = vec![0i16; 84];
        let expected_result =
            decode_gc_adpcm_range(&data, &coefficients, initial, 0, 84, 0, &mut expected, 84);
        assert_eq!(expected_result.decoded_samples, 84);

        let mut actual = Vec::with_capacity(84);
        let mut context = initial;
        let mut position = 0usize;
        for count in [1usize, 12, 2, 13, 14, 7, 35] {
            let (byte_offset, byte_count) =
                gc_adpcm_byte_range(position, count, data.len()).unwrap();
            let mut chunk = vec![0i16; count];
            let result = decode_gc_adpcm_range(
                &data[byte_offset..byte_offset + byte_count],
                &coefficients,
                context,
                position,
                count,
                0,
                &mut chunk,
                count,
            );
            assert_eq!(result.decoded_samples, count);
            context = result.checkpoint.unwrap();
            actual.extend_from_slice(&chunk);
            position += count;
        }

        assert_eq!(actual, expected);
        assert_eq!(context, expected_result.checkpoint.unwrap());
    }

    #[test]
    fn adpcm_streaming_windows_match_prefix_redecode_with_lookahead() {
        let data = adpcm_fixture(32);
        let coefficients = adpcm_coefficients();
        let initial = AudioAdpcmContext {
            header: 0,
            yn0: 777,
            yn1: -333,
        };
        let mut context = initial;
        let mut base = 5usize;
        let window = 23usize;
        let advance = 17usize;

        for update in 0..8 {
            let mut expected = vec![0i16; window];
            decode_gc_adpcm_range(
                &data,
                &coefficients,
                initial,
                0,
                base + window,
                base,
                &mut expected,
                base + advance,
            );

            let decode_start = if update == 0 { 0 } else { base };
            let decode_count = if update == 0 { base + window } else { window };
            let output_skip = if update == 0 { base } else { 0 };
            let checkpoint_after = if update == 0 { base + advance } else { advance };
            let decode_context = if update == 0 { initial } else { context };
            let (byte_offset, byte_count) =
                gc_adpcm_byte_range(decode_start, decode_count, data.len()).unwrap();
            let mut actual = vec![0i16; window];
            let result = decode_gc_adpcm_range(
                &data[byte_offset..byte_offset + byte_count],
                &coefficients,
                decode_context,
                decode_start,
                decode_count,
                output_skip,
                &mut actual,
                checkpoint_after,
            );

            assert_eq!(actual, expected);
            context = result.checkpoint.unwrap();
            base += advance;
        }
    }

    #[test]
    fn adpcm_stream_state_rejects_rewind_reset_and_parameter_changes() {
        let key = adpcm_stream_key();
        let state = AudioAdpcmDecodeState {
            valid: true,
            key,
            next_sample: 512,
            context: AudioAdpcmContext {
                header: 0x21,
                yn0: 10,
                yn1: -20,
            },
        };

        assert!(can_stream_gc_adpcm(state, key, 512, false));
        assert!(!can_stream_gc_adpcm(state, key, 0, false));
        assert!(!can_stream_gc_adpcm(state, key, 512, true));

        let mut changed = key;
        changed.buffer_address += 0x1000;
        assert!(!can_stream_gc_adpcm(state, changed, 512, false));
        changed = key;
        changed.initial_yn0 += 1;
        assert!(!can_stream_gc_adpcm(state, changed, 512, false));
        changed = key;
        changed.coefficients[3] += 1;
        assert!(!can_stream_gc_adpcm(state, changed, 512, false));
        changed = key;
        changed.looping = false;
        assert!(!can_stream_gc_adpcm(state, changed, 512, false));
    }

    #[test]
    fn adpcm_streaming_work_is_bounded_by_window_size() {
        let count = 243usize;
        let near = 13usize;
        let far = near + 14 * 100_000;
        let far_buffer_size = (far / 14 + 64) * 8;
        let (_, near_bytes) = gc_adpcm_byte_range(near, count, far_buffer_size).unwrap();
        let (_, far_bytes) = gc_adpcm_byte_range(far, count, far_buffer_size).unwrap();
        assert_eq!(far_bytes, near_bytes);
        assert!(far_bytes <= 152);

        let data = vec![0u8; far_bytes];
        let mut output = vec![0i16; count];
        let result = decode_gc_adpcm_range(
            &data,
            &[0; 16],
            AudioAdpcmContext::default(),
            far,
            count,
            0,
            &mut output,
            count,
        );
        assert_eq!(result.decoded_samples, count);
        assert!(result.bytes_read <= 152);
    }

    #[test]
    fn adpcm_invalid_predictors_fail_without_a_checkpoint() {
        let mut output = [1i16; 2];
        let frame_result = decode_gc_adpcm_range(
            &[0x80, 0, 0],
            &[0; 16],
            AudioAdpcmContext::default(),
            0,
            2,
            0,
            &mut output,
            1,
        );
        assert_eq!(frame_result.decoded_samples, 0);
        assert_eq!(frame_result.checkpoint, None);
        assert_eq!(output, [0, 0]);

        let mid_frame_result = decode_gc_adpcm_range(
            &[0],
            &[0; 16],
            AudioAdpcmContext {
                header: 0x80,
                yn0: 1,
                yn1: -1,
            },
            1,
            1,
            0,
            &mut output[..1],
            1,
        );
        assert_eq!(mid_frame_result.decoded_samples, 0);
        assert_eq!(mid_frame_result.checkpoint, None);
        assert_eq!(output[0], 0);
    }
}

fn svc_trace_capture(kernel: &Kernel) -> Option<(u64, u64, u64, u64, u64, u64, u64, u32)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_SVC_TRACE")
            .ok()
            .map(|v| v.parse::<u64>().unwrap_or(1000))
            .unwrap_or(0)
    });
    if limit == 0 {
        return None;
    }
    static SKIP: OnceLock<u64> = OnceLock::new();
    let skip = *SKIP.get_or_init(|| {
        std::env::var("NEXIUM_SVC_TRACE_SKIP")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    });
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < skip || n >= skip.saturating_add(limit) {
        return None;
    }
    let cpu = cpu_ref()?;
    Some((
        n,
        cpu.get_register(0),
        cpu.get_register(1),
        cpu.get_register(2),
        cpu.get_register(3),
        cpu.get_pc(),
        cpu.get_register(30),
        kernel.threads.current_handle().unwrap_or(0),
    ))
}

pub fn dispatch(kernel: &mut Kernel, imm: u16) -> u32 {
    log::trace!("SVC {:#04x}", imm);
    struct ProfileGuard(std::time::Instant, u16);
    impl Drop for ProfileGuard {
        fn drop(&mut self) {
            crate::kernel::profile::record_svc(self.1, self.0);
        }
    }
    let _guard = if crate::kernel::profile::enabled() {
        Some(ProfileGuard(std::time::Instant::now(), imm))
    } else {
        None
    };
    let __svc_trace_args = svc_trace_capture(kernel);
    let __svc_res = match imm {
        0x01 => svc_set_heap_size(kernel),
        0x02 => svc_set_memory_permission(kernel),
        0x03 => svc_set_memory_attribute(kernel),
        0x04 => svc_map_memory(kernel),
        0x05 => svc_unmap_memory(kernel),
        0x06 => svc_query_memory(kernel),
        0x07 => svc_exit_process(kernel),
        0x08 => svc_create_thread(kernel),
        0x09 => svc_start_thread(kernel),
        0x0a => svc_exit_thread(kernel),
        0x0b => svc_sleep_thread(kernel),
        0x0c => svc_get_thread_priority(kernel),
        0x0d => svc_set_thread_priority(kernel),
        0x0e => svc_get_thread_core_mask(kernel),
        0x0f => svc_set_thread_core_mask(kernel),
        0x10 => svc_get_current_processor_number(kernel),
        0x11 => svc_signal_event(kernel),
        0x12 => svc_clear_event(kernel),
        0x13 => svc_map_shared_memory(kernel),
        0x14 => svc_unmap_shared_memory(kernel),
        0x15 => svc_create_transfer_memory(kernel),
        0x16 => svc_close_handle(kernel),
        0x17 => svc_reset_signal(kernel),
        0x18 => svc_wait_synchronization(kernel),
        0x19 => svc_cancel_synchronization(kernel),
        0x1a => svc_arbitrate_lock(kernel),
        0x1b => svc_arbitrate_unlock(kernel),
        0x1c => svc_wait_process_wide_key_atomic(kernel),
        0x1d => svc_signal_process_wide_key(kernel),
        0x1e => svc_get_system_tick(kernel),
        0x1f => svc_connect_to_named_port(kernel),
        0x20 => svc_send_sync_request_light(kernel),
        0x21 => svc_send_sync_request(kernel),
        0x22 => svc_send_sync_request_with_user_buffer(kernel),
        0x23 => svc_send_async_request_with_user_buffer(kernel),
        0x24 => svc_get_process_id(kernel),
        0x25 => svc_get_thread_id(kernel),
        0x26 => svc_break(kernel),
        0x27 => svc_output_debug_string(kernel),
        0x28 => svc_return_from_exception(kernel),
        0x29 => svc_get_info(kernel),
        0x2a => svc_flush_entire_data_cache(kernel),
        0x2b => svc_flush_data_cache(kernel),
        0x2c => svc_map_physical_memory(kernel),
        0x2d => svc_unmap_physical_memory(kernel),
        0x2e => svc_get_debug_future_thread_info(kernel),
        0x2f => svc_get_last_thread_info(kernel),
        0x30 => svc_get_resource_limit_limit_value(kernel),
        0x31 => svc_get_resource_limit_current_value(kernel),
        0x32 => svc_set_thread_activity(kernel),
        0x33 => svc_get_thread_context3(kernel),
        0x34 => svc_wait_for_address(kernel),
        0x35 => svc_signal_to_address(kernel),
        0x36 => svc_synchronize_preemption_state(kernel),
        0x37 => svc_get_resource_limit_peak_value(kernel),
        0x40 => svc_create_session(kernel),
        0x41 => svc_accept_session(kernel),
        0x42 => svc_reply_and_receive_light(kernel),
        0x43 => svc_reply_and_receive(kernel),
        0x44 => svc_reply_and_receive_with_user_buffer(kernel),
        0x45 => svc_create_event(kernel),
        0x50 => svc_create_shared_memory(kernel),
        0x51 => svc_map_transfer_memory(kernel),
        0x52 => svc_unmap_transfer_memory(kernel),
        0x53 => svc_create_interrupt_event(kernel),
        0x54 => svc_query_io_mapping(kernel),
        0x5f => svc_debug_active_process(kernel),
        0x60 => svc_break_debug_process(kernel),
        0x61 => svc_terminate_debug_process(kernel),
        0x62 => svc_get_debug_event(kernel),
        0x63 => svc_continue_debug_event(kernel),
        0x64 => svc_get_process_list(kernel),
        0x65 => svc_get_thread_list(kernel),
        0x6f => svc_create_port(kernel),
        0x70 => svc_manage_named_port(kernel),
        0x71 => svc_connect_to_port(kernel),
        0x7c => svc_create_resource_limit(kernel),
        0x7d => svc_set_resource_limit_limit_value(kernel),
        0x7e => svc_call_secure_monitor(kernel),
        0x7f => svc_guest_probe(kernel),
        _ => {
            log::warn!("unknown SVC: {:#04x}", imm);
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, KERNEL_NOT_IMPLEMENTED as u64);
            }
            KERNEL_NOT_IMPLEMENTED
        }
    };
    if let Some((n, x0, x1, x2, x3, pc, lr, th)) = __svc_trace_args {
        log::info!(
            "[svc-trace #{}] th={:#x} svc={:#04x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} -> {:#x} pc={:#x} lr={:#x}",
            n,
            th,
            imm,
            x0,
            x1,
            x2,
            x3,
            __svc_res,
            pc,
            lr
        );
    }
    __svc_res
}

const GUEST_PROBE_SVC_INSN: u32 = 0xD400_0FE1;
const METROID_DREAD_TITLE_ID: u64 = 0x0100_9380_1237_C000;
const DREAD_RESOURCE_ALLOC_RETURN_PC: u64 = 0x081B_688C;
const DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE: [u8; 16] = [
    0xF6, 0x03, 0x00, 0xAA, 0x80, 0x2E, 0x40, 0xF9, 0x96, 0x42, 0x00, 0xF9, 0x08, 0x00, 0x40, 0xF9,
];
const DREAD_COMPAT_ALLOCATION_BASE: u64 = 0x70_0000_0000;
const DREAD_COMPAT_ALLOCATION_LIMIT: u64 = DREAD_COMPAT_ALLOCATION_BASE + 512 * 1024 * 1024;
const MAX_DREAD_COMPAT_ALLOCATION: u64 = 64 * 1024 * 1024;
const COMPAT_PAGE_SIZE: u64 = 0x1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompatibilityGuestProbePatch {
    pc: u64,
    signature: &'static [u8],
    name: &'static str,
}

fn compatibility_guest_probe_patch(title_id: u64) -> Option<CompatibilityGuestProbePatch> {
    (title_id == METROID_DREAD_TITLE_ID).then_some(CompatibilityGuestProbePatch {
        pc: DREAD_RESOURCE_ALLOC_RETURN_PC,
        signature: &DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE,
        name: "Metroid Dread 1.0.0 resource allocation fallback",
    })
}

pub(crate) fn install_compatibility_guest_probes(
    address_space: &nexium_memory::AddressSpace,
    title_id: u64,
) -> bool {
    let Some(patch) = compatibility_guest_probe_patch(title_id) else {
        return false;
    };
    let mut actual = vec![0u8; patch.signature.len()];
    if let Err(error) = address_space.read(patch.pc, &mut actual) {
        log::warn!(
            "[compat] skipped {}: could not read pc={:#x}: {error:?}",
            patch.name,
            patch.pc
        );
        return false;
    }
    if actual != patch.signature {
        log::warn!(
            "[compat] skipped {}: signature mismatch at pc={:#x}",
            patch.name,
            patch.pc
        );
        return false;
    }
    match address_space.write(patch.pc, &GUEST_PROBE_SVC_INSN.to_le_bytes()) {
        Ok(()) => {
            log::info!("[compat] enabled {}", patch.name);
            true
        }
        Err(error) => {
            log::warn!(
                "[compat] skipped {}: could not patch pc={:#x}: {error:?}",
                patch.name,
                patch.pc
            );
            false
        }
    }
}

#[derive(Clone, Copy)]
struct ResolvedGuestProbeAction {
    kind: &'static str,
    arg: u64,
    log_hits: bool,
}

fn resolve_guest_probe_action(
    title_id: u64,
    compatibility_enabled: bool,
    pc: u64,
) -> Option<ResolvedGuestProbeAction> {
    if compatibility_enabled
        && title_id == METROID_DREAD_TITLE_ID
        && pc == DREAD_RESOURCE_ALLOC_RETURN_PC
    {
        return Some(ResolvedGuestProbeAction {
            kind: "dread_allocret",
            arg: DREAD_COMPAT_ALLOCATION_BASE,
            log_hits: false,
        });
    }
    if let Some((kind, arg)) = guest_probe_actions().get(&pc) {
        return Some(ResolvedGuestProbeAction {
            kind: kind.as_str(),
            arg: *arg,
            log_hits: true,
        });
    }
    None
}

fn resolve_guest_probe_at(
    title_id: u64,
    compatibility_enabled: bool,
    pc_now: u64,
) -> Option<(u64, ResolvedGuestProbeAction)> {
    let previous = pc_now.wrapping_sub(4);
    if let Some(action) = resolve_guest_probe_action(title_id, compatibility_enabled, previous) {
        Some((previous, action))
    } else {
        resolve_guest_probe_action(title_id, compatibility_enabled, pc_now)
            .map(|action| (pc_now, action))
    }
}

fn plan_dread_compat_allocation(
    address_space: &nexium_memory::AddressSpace,
    cursor: u64,
    requested: u64,
) -> Option<(u64, u64, u64)> {
    if requested == 0 || requested > MAX_DREAD_COMPAT_ALLOCATION {
        return None;
    }
    let map_len = requested.checked_add(COMPAT_PAGE_SIZE - 1)? & !(COMPAT_PAGE_SIZE - 1);
    let current = if cursor == 0 {
        DREAD_COMPAT_ALLOCATION_BASE
    } else {
        cursor.max(DREAD_COMPAT_ALLOCATION_BASE)
    };
    let search_start = current.checked_add(COMPAT_PAGE_SIZE - 1)? & !(COMPAT_PAGE_SIZE - 1);
    let search_len = DREAD_COMPAT_ALLOCATION_LIMIT.checked_sub(search_start)?;
    if search_len == 0 {
        return None;
    }
    address_space
        .unmapped_gaps(search_start, search_len)
        .ok()?
        .into_iter()
        .find_map(|(gap_start, gap_end)| {
            let allocation = gap_start.checked_add(COMPAT_PAGE_SIZE - 1)? & !(COMPAT_PAGE_SIZE - 1);
            let end = allocation.checked_add(map_len)?;
            (end <= gap_end && end <= DREAD_COMPAT_ALLOCATION_LIMIT)
                .then_some((allocation, map_len, end))
        })
}

#[cfg(test)]
mod metroid_dread_compatibility_tests {
    use super::{
        compatibility_guest_probe_patch, install_compatibility_guest_probes,
        plan_dread_compat_allocation, resolve_guest_probe_at, COMPAT_PAGE_SIZE,
        DREAD_COMPAT_ALLOCATION_BASE, DREAD_COMPAT_ALLOCATION_LIMIT,
        DREAD_RESOURCE_ALLOC_RETURN_PC, DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE,
        GUEST_PROBE_SVC_INSN, MAX_DREAD_COMPAT_ALLOCATION, METROID_DREAD_TITLE_ID,
    };
    use nexium_memory::{AddressSpace, Perm};

    fn mapped_signature() -> AddressSpace {
        let address_space = AddressSpace::new();
        let page = DREAD_RESOURCE_ALLOC_RETURN_PC & !(COMPAT_PAGE_SIZE - 1);
        address_space
            .map(page, COMPAT_PAGE_SIZE, Perm::RX, "dread_compat_test")
            .unwrap();
        address_space
            .write(
                DREAD_RESOURCE_ALLOC_RETURN_PC,
                &DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE,
            )
            .unwrap();
        address_space
    }

    #[test]
    fn patch_is_exact_title_and_signature_gated() {
        assert!(compatibility_guest_probe_patch(METROID_DREAD_TITLE_ID).is_some());
        assert!(compatibility_guest_probe_patch(METROID_DREAD_TITLE_ID + 1).is_none());

        let address_space = mapped_signature();
        assert!(!install_compatibility_guest_probes(
            &address_space,
            METROID_DREAD_TITLE_ID + 1
        ));
        let mut actual = [0u8; 16];
        address_space
            .read(DREAD_RESOURCE_ALLOC_RETURN_PC, &mut actual)
            .unwrap();
        assert_eq!(actual, DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE);

        address_space
            .write(DREAD_RESOURCE_ALLOC_RETURN_PC + 4, &[0xFF])
            .unwrap();
        assert!(!install_compatibility_guest_probes(
            &address_space,
            METROID_DREAD_TITLE_ID
        ));
        address_space
            .read(DREAD_RESOURCE_ALLOC_RETURN_PC, &mut actual)
            .unwrap();
        assert_eq!(&actual[..4], &DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE[..4]);

        address_space
            .write(
                DREAD_RESOURCE_ALLOC_RETURN_PC,
                &DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE,
            )
            .unwrap();
        assert!(install_compatibility_guest_probes(
            &address_space,
            METROID_DREAD_TITLE_ID
        ));
        address_space
            .read(DREAD_RESOURCE_ALLOC_RETURN_PC, &mut actual)
            .unwrap();
        assert_eq!(&actual[..4], &GUEST_PROBE_SVC_INSN.to_le_bytes());
        assert_eq!(&actual[4..], &DREAD_RESOURCE_ALLOC_RETURN_SIGNATURE[4..]);
    }

    #[test]
    fn fallback_reservations_are_aligned_bounded_and_skip_collisions() {
        assert!(DREAD_COMPAT_ALLOCATION_LIMIT <= (1u64 << 39));
        let address_space = AddressSpace::new();
        assert_eq!(
            plan_dread_compat_allocation(&address_space, 0, 0x4002CD),
            Some((
                DREAD_COMPAT_ALLOCATION_BASE,
                0x401000,
                DREAD_COMPAT_ALLOCATION_BASE + 0x401000
            ))
        );
        address_space
            .map(
                DREAD_COMPAT_ALLOCATION_BASE,
                0x2000,
                Perm::RW,
                "occupied_compat_test",
            )
            .unwrap();
        assert_eq!(
            plan_dread_compat_allocation(&address_space, 0, 1),
            Some((
                DREAD_COMPAT_ALLOCATION_BASE + 0x2000,
                0x1000,
                DREAD_COMPAT_ALLOCATION_BASE + 0x3000
            ))
        );

        assert_eq!(plan_dread_compat_allocation(&address_space, 0, 0), None);
        assert!(plan_dread_compat_allocation(
            &address_space,
            DREAD_COMPAT_ALLOCATION_BASE + 0x2000,
            MAX_DREAD_COMPAT_ALLOCATION,
        )
        .is_some());
        assert_eq!(
            plan_dread_compat_allocation(&address_space, 0, MAX_DREAD_COMPAT_ALLOCATION + 1,),
            None
        );
        assert_eq!(
            plan_dread_compat_allocation(
                &address_space,
                DREAD_COMPAT_ALLOCATION_LIMIT - COMPAT_PAGE_SIZE,
                COMPAT_PAGE_SIZE * 2,
            ),
            None
        );
    }

    #[test]
    fn built_in_probe_resolves_both_backend_pc_forms_only_when_enabled() {
        assert!(resolve_guest_probe_at(
            METROID_DREAD_TITLE_ID,
            false,
            DREAD_RESOURCE_ALLOC_RETURN_PC + 4,
        )
        .is_none());
        for pc in [
            DREAD_RESOURCE_ALLOC_RETURN_PC,
            DREAD_RESOURCE_ALLOC_RETURN_PC + 4,
        ] {
            let (probe_pc, action) =
                resolve_guest_probe_at(METROID_DREAD_TITLE_ID, true, pc).unwrap();
            assert_eq!(probe_pc, DREAD_RESOURCE_ALLOC_RETURN_PC);
            assert_eq!(action.kind, "dread_allocret");
        }
    }
}

pub fn guest_probe_actions() -> &'static std::collections::HashMap<u64, (String, u64)> {
    use std::sync::OnceLock;
    static MAP: OnceLock<std::collections::HashMap<u64, (String, u64)>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        if let Ok(spec) = std::env::var("NEXIUM_GUEST_PROBE_SVC") {
            for item in spec.split(',') {
                let parts: Vec<&str> = item.trim().split(':').collect();
                if parts.len() != 3 {
                    continue;
                }
                let pc = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16);
                let arg = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16);
                if let (Ok(pc), Ok(arg)) = (pc, arg) {
                    map.insert(pc, (parts[1].to_string(), arg));
                }
            }
        }
        map
    })
}

fn svc_guest_probe(kernel: &mut Kernel) -> u32 {
    let Some(cpu) = cpu_mut() else {
        return 0;
    };
    let pc_now = cpu.get_pc();
    let Some((probe_pc, action)) = resolve_guest_probe_at(
        kernel.title_id,
        kernel.compatibility_guest_probe_enabled,
        pc_now,
    ) else {
        log::warn!("[guest-probe] svc 0x7f at pc={:#x} with no action", pc_now);
        return 0;
    };
    if pc_now == probe_pc {
        cpu.set_pc(probe_pc.wrapping_add(4));
    }
    let kind = action.kind;
    let arg = action.arg;
    let lr = cpu.get_register(30);
    let sp = cpu.get_sp();
    let x0 = cpu.get_register(0);
    let x1 = cpu.get_register(1);
    let x2 = cpu.get_register(2);
    let x3 = cpu.get_register(3);
    if action.log_hits {
        log::warn!(
            "[guest-probe] hit pc={:#x} kind={} arg={:#x} lr={:#x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} sp={:#x} thread={:?}",
            probe_pc,
            kind,
            arg,
            lr,
            x0,
            x1,
            x2,
            x3,
            sp,
            kernel.threads.current_handle()
        );
    }
    if action.log_hits && std::env::var_os("NEXIUM_GUEST_PROBE_DUMP").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static DUMPS: AtomicU64 = AtomicU64::new(0);
        let sequence = DUMPS.fetch_add(1, Ordering::Relaxed);
        if sequence < 8 {
            for (name, address) in [("x1", x1), ("x3", x3)] {
                let mut bytes = [0u8; 64];
                if address != 0 && kernel.address_space.read(address, &mut bytes).is_ok() {
                    log::warn!(
                        "[guest-probe-dump] #{} {}={:#x} bytes={:02x?}",
                        sequence,
                        name,
                        address,
                        bytes
                    );
                }
            }
        }
    }
    match kind {
        "subsp" => {
            cpu.set_sp(sp.wrapping_sub(arg));
        }
        "stpfp" => {
            let nsp = sp.wrapping_sub(arg);
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&cpu.get_register(29).to_le_bytes());
            buf[8..].copy_from_slice(&lr.to_le_bytes());
            let _ = kernel.address_space.write(nsp, &buf);
            cpu.set_sp(nsp);
        }
        "dread_allocret" => {
            if x0 != 0 {
                cpu.set_register(22, x0);
                return 0;
            }

            let owner = cpu.get_register(20);
            let mut size_bytes = [0u8; 8];
            let Some(size_address) = owner.checked_add(136).filter(|_| owner != 0) else {
                log::warn!(
                    "[compat] Metroid Dread allocation fallback rejected owner={:#x}",
                    owner
                );
                cpu.set_register(22, 0);
                return 0;
            };
            if kernel
                .address_space
                .read(size_address, &mut size_bytes)
                .is_err()
            {
                log::warn!(
                    "[compat] Metroid Dread allocation fallback could not read size from owner={:#x}",
                    owner
                );
                cpu.set_register(22, 0);
                return 0;
            }
            let requested = u64::from_le_bytes(size_bytes);
            let Some((allocation, map_len, next)) = plan_dread_compat_allocation(
                &kernel.address_space,
                kernel.compatibility_allocation_next,
                requested,
            ) else {
                log::warn!(
                    "[compat] Metroid Dread allocation fallback rejected owner={:#x} requested={:#x}",
                    owner,
                    requested
                );
                cpu.set_register(22, 0);
                return 0;
            };

            if let Err(error) = kernel.address_space.map(
                allocation,
                map_len,
                nexium_memory::Perm::RW,
                "metroid_dread_resource_fallback",
            ) {
                log::warn!(
                    "[compat] Metroid Dread allocation mapping failed at {:#x}: {error:?}",
                    allocation
                );
                cpu.set_register(22, 0);
                return 0;
            }
            let Some(region) = kernel.address_space.host_region_at(allocation) else {
                log::warn!("[compat] Metroid Dread allocation could not lease mapped region");
                cpu.set_register(22, 0);
                return 0;
            };
            if let Err(error) = unsafe {
                cpu.map_host(
                    region.base,
                    region.size,
                    region.perm,
                    region.host_ptr as *mut u8,
                )
            } {
                log::warn!("[compat] Metroid Dread allocation CPU mapping failed: {error}");
                cpu.set_register(22, 0);
                return 0;
            }

            kernel.compatibility_allocation_next = next;
            log::warn!(
                "[compat] Metroid Dread resource pool exhausted; supplied requested={:#x} at {:#x} (mapped={:#x})",
                requested,
                allocation,
                map_len
            );
            cpu.set_register(0, allocation);
            cpu.set_register(22, allocation);
        }
        other => {
            log::warn!("[guest-probe] unknown action kind {}", other);
        }
    }
    0
}

fn svc_set_heap_size(kernel: &mut Kernel) -> u32 {
    let size = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1)
    } else {
        return 1;
    };
    if size > kernel.heap_size {
        log::warn!(
            "svcSetHeapSize requested {:#x} > mapped heap {:#x}; clamping (guest may fault on overflow)",
            size,
            kernel.heap_size
        );
    }
    let committed = size.min(kernel.heap_size);
    if let Err(error) = kernel
        .address_space
        .resize_committed(kernel.heap_base, committed)
    {
        const KERNEL_OUT_OF_MEMORY: u32 = 1 | (104 << 9);
        log::error!(
            "svcSetHeapSize failed to resize heap @ {:#x} to {:#x}: {}",
            kernel.heap_base,
            committed,
            error
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_OUT_OF_MEMORY as u64);
        }
        return KERNEL_OUT_OF_MEMORY;
    }
    kernel.heap_committed = committed;
    log::debug!(
        "svcSetHeapSize size={:#x} -> heap_base={:#x} (heap mapped {:#x})",
        size,
        kernel.heap_base,
        kernel.heap_size
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, kernel.heap_base);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_permission(_kernel: &mut Kernel) -> u32 {
    let (addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
        )
    } else {
        (0, 0, 0)
    };
    log::debug!(
        "svcSetMemoryPermission addr={:#x} size={:#x} perm={:#x} (no-op)",
        addr,
        size,
        perm
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_attribute(_kernel: &mut Kernel) -> u32 {
    let (addr, size, mask, value) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        (0, 0, 0, 0)
    };
    log::debug!(
        "svcSetMemoryAttribute addr={:#x} size={:#x} mask={:#x} value={:#x} (no-op)",
        addr,
        size,
        mask,
        value
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_map_memory(kernel: &mut Kernel) -> u32 {
    let (dst, src, size) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
        )
    } else {
        return 1;
    };

    if size == 0 || (dst & 0xFFF) != 0 || (size & 0xFFF) != 0 {
        log::warn!(
            "svcMapMemory: bad args dst={:#x} src={:#x} size={:#x}",
            dst,
            src,
            size
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_ADDRESS as u64);
        }
        return KERNEL_INVALID_ADDRESS;
    }

    let map_rc = kernel
        .address_space
        .map(dst, size, nexium_memory::Perm::RW, "stack_mirror");
    let was_new = map_rc.is_ok();

    let mut buf = vec![0u8; size as usize];
    if kernel.address_space.read(src, &mut buf).is_ok() {
        let _ = kernel.address_space.write(dst, &buf);
    }

    if was_new {
        if let Some(region) = kernel.address_space.host_region_at(dst) {
            if let Some(cpu) = cpu_mut() {
                let plumb = unsafe {
                    cpu.map_host(
                        region.base,
                        region.size,
                        region.perm,
                        region.host_ptr as *mut u8,
                    )
                };
                match plumb {
                    Ok(_) => log::debug!(
                        "svcMapMemory dst={:#x} src={:#x} size={:#x} â†’ mapped + copied + plumbed to dynarmic",
                        dst,
                        src,
                        size
                    ),
                    Err(e) => log::warn!(
                        "svcMapMemory dst={:#x} size={:#x} mapped in AS but dynarmic map_host failed: {}",
                        dst,
                        size,
                        e
                    ),
                }
            }
        } else {
            log::warn!(
                "svcMapMemory dst={:#x}: AS region lookup failed after map()",
                dst
            );
        }
    } else if let Err(e) = map_rc {
        log::debug!(
            "svcMapMemory dst={:#x} src={:#x} size={:#x} â†’ already mapped ({:?}), refreshed contents only",
            dst,
            src,
            size,
            e
        );
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapMemory (no-op)");
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_memory(kernel: &mut Kernel) -> u32 {
    let (out_ptr, address) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(2))
    } else {
        return 1;
    };

    log::debug!(
        "svcQueryMemory out_ptr={:#x} address={:#x}",
        out_ptr,
        address
    );

    let info = synthesize_memory_info(kernel, address);
    let mut buf = [0u8; 0x28];
    buf[0..8].copy_from_slice(&info.addr.to_le_bytes());
    buf[8..16].copy_from_slice(&info.size.to_le_bytes());
    buf[16..20].copy_from_slice(&info.mem_type.to_le_bytes());
    buf[20..24].copy_from_slice(&info.attr.to_le_bytes());
    buf[24..28].copy_from_slice(&info.perm.to_le_bytes());
    buf[28..32].copy_from_slice(&0u32.to_le_bytes());
    buf[32..36].copy_from_slice(&0u32.to_le_bytes());
    buf[36..40].copy_from_slice(&0u32.to_le_bytes());

    if out_ptr != 0 {
        let _ = kernel.address_space.write(out_ptr, &buf);
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, 0);
    }
    SUCCESS
}

struct SynthMemInfo {
    addr: u64,
    size: u64,
    mem_type: u32,
    attr: u32,
    perm: u32,
}

fn synthesize_memory_info(kernel: &Kernel, address: u64) -> SynthMemInfo {
    let regions = kernel.address_space.regions();
    for r in &regions {
        if address >= r.base && address < r.base + r.size {
            let mem_type = if r.name.starts_with("codestatic") {
                0x03
            } else if r.name.starts_with("codemutable") {
                0x04
            } else if r.name.contains("text") || r.name.contains("rodata") {
                0x10
            } else if r.name.contains("data") || r.name.contains("bss") {
                0x11
            } else if r.name.starts_with("heap") || r.name.starts_with("physmem") {
                0x05
            } else if r.name.starts_with("stack") {
                0x07
            } else if r.name.starts_with("shared") {
                0x12
            } else {
                0x03
            };
            return SynthMemInfo {
                addr: r.base,
                size: r.size,
                mem_type,
                attr: 0,
                perm: r.perm.bits() as u32,
            };
        }
    }

    let next_base = regions
        .iter()
        .map(|r| r.base)
        .filter(|&b| b > address)
        .min()
        .unwrap_or(u64::MAX);
    let page_addr = address & !0xFFF;
    let gap_size = next_base.saturating_sub(page_addr);

    SynthMemInfo {
        addr: page_addr,
        size: if gap_size == 0 {
            0x10000_0000
        } else {
            gap_size
        },
        mem_type: 0,
        attr: 0,
        perm: 0,
    }
}

fn svc_exit_process(kernel: &mut Kernel) -> u32 {
    let (x0, lr) = cpu_ref()
        .map(|cpu| (cpu.get_register(0), cpu.get_register(30)))
        .unwrap_or((0, 0));
    log::info!(
        "svcExitProcess - terminating process (x0={:#x} lr={:#x})",
        x0,
        lr
    );
    if exit_trace_enabled() {
        trace_process_exit(kernel);
    }
    kernel.process_exited = true;
    SUCCESS
}

fn exit_trace_enabled() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_EXIT_TRACE")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

fn trace_process_exit(kernel: &Kernel) {
    let Some(cpu) = cpu_ref() else {
        return;
    };

    let pc = cpu.get_pc();
    let lr = cpu.get_register(30);
    let sp = cpu.get_sp();
    let fp = cpu.get_register(29);
    log::warn!(
        "[exit-trace] pc={:#x} lr={:#x} sp={:#x} fp={:#x} code={:#x}..{:#x}",
        pc,
        lr,
        sp,
        fp,
        kernel.code_base,
        kernel.code_base.saturating_add(kernel.code_size)
    );

    let mut cur_fp = fp;
    for depth in 0..24 {
        if cur_fp == 0 || cur_fp & 7 != 0 {
            break;
        }
        let mut frame = [0u8; 16];
        if kernel.address_space.read(cur_fp, &mut frame).is_err() {
            break;
        }
        let next_fp = u64::from_le_bytes(frame[0..8].try_into().unwrap());
        let saved_lr = u64::from_le_bytes(frame[8..16].try_into().unwrap());
        log::warn!(
            "[exit-trace] frame={} fp={:#x} next_fp={:#x} lr={:#x} code_off={:#x}",
            depth,
            cur_fp,
            next_fp,
            saved_lr,
            saved_lr.wrapping_sub(kernel.code_base)
        );
        if next_fp <= cur_fp || next_fp.saturating_sub(cur_fp) > 0x10_0000 {
            break;
        }
        cur_fp = next_fp;
    }

    let mut stack = [0u8; 0x800];
    if kernel.address_space.read(sp, &mut stack).is_err() {
        return;
    }
    let code_end = kernel.code_base.saturating_add(kernel.code_size);
    for (slot, bytes) in stack.chunks_exact(8).enumerate() {
        let value = u64::from_le_bytes(bytes.try_into().unwrap());
        if value >= kernel.code_base && value < code_end {
            log::warn!(
                "[exit-trace] stack+{:#x}={:#x} code_off={:#x}",
                slot * 8,
                value,
                value - kernel.code_base
            );
        }
    }
}

fn svc_map_shared_memory(kernel: &mut Kernel) -> u32 {
    let (handle, addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0) as u32,
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3) as u32,
        )
    } else {
        return 1;
    };
    log::debug!(
        "svcMapSharedMemory handle={:#x} addr={:#x} size={:#x} perm={:#x}",
        handle,
        addr,
        size,
        perm
    );
    if handle == 0 {
        log::error!(
            "svcMapSharedMemory called with handle=0 (addr={:#x} size={:#x}). \
             Upstream service returned no shared-memory handle. \
             Allocating zero-filled placeholder; expect downstream code to read zeros from this region.",
            addr,
            size
        );
    }

    if size as usize == crate::hid_state::HID_SHMEM_SIZE {
        if kernel.address_space.host_region_at(addr).is_none() {
            if let Err(e) =
                kernel
                    .address_space
                    .map(addr, size, nexium_memory::perm::Perm::RW, "hid_shmem")
            {
                log::error!("failed to add HID shmem to address space: {}", e);
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, 1);
                }
                return 1;
            }
        }
        let Some(region) = kernel.address_space.host_region_at(addr) else {
            log::error!("HID shmem address-space mapping has no host region");
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, 1);
            }
            return 1;
        };
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        hid.shmem_va = Some(addr);
        kernel.hid_mapped_host_ptr = Some(region.host_ptr as usize);
        unsafe {
            hid.bind_mapped_host(region.host_ptr);
        }
        log::debug!(
            "  â†’ recognized as HID shared memory, publishing synchronized guest VA {:#x}",
            addr
        );
        if let Some(cpu) = cpu_mut() {
            unsafe {
                if let Err(e) = cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                {
                    log::warn!("failed to map HID shmem in CPU: {}", e);
                }
            }
        }
    } else if kernel.time_shmem_handle == Some(handle) {
        log::debug!(
            "  â†’ recognized as time shared memory, mapping {} bytes at {:#x}",
            size,
            addr
        );
        let backing: Vec<u8> = kernel
            .time_shmem
            .as_deref()
            .map(|d| {
                let mut v = vec![0u8; size as usize];
                let copy_len = d.len().min(size as usize);
                v[..copy_len].copy_from_slice(&d[..copy_len]);
                v
            })
            .unwrap_or_else(|| vec![0u8; size as usize]);
        let needed_map = kernel.address_space.write(addr, &backing).is_err();
        if needed_map {
            let _ =
                kernel
                    .address_space
                    .map(addr, size, nexium_memory::perm::Perm::R, "time_shmem");
            let _ = kernel.address_space.write(addr, &backing);
        }
        if let Some(region) = kernel.address_space.host_region_at(addr) {
            if let Some(cpu) = cpu_mut() {
                unsafe {
                    if let Err(e) =
                        cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    {
                        log::warn!("failed to map time shmem in CPU: {}", e);
                    } else {
                        log::debug!("  â†’ registered time shmem at {:#x} with CPU", region.base);
                    }
                }
            }
        }
    } else if kernel.font_shmem_handle == Some(handle) {
        log::debug!(
            "  â†’ recognized as font shared memory, mapping {} bytes of font data at {:#x}",
            size,
            addr
        );
        let font_data: Vec<u8> = kernel
            .font_shmem
            .as_deref()
            .map(|d| {
                let mut v = vec![0u8; size as usize];
                let copy_len = d.len().min(size as usize);
                v[..copy_len].copy_from_slice(&d[..copy_len]);
                v
            })
            .unwrap_or_else(|| vec![0u8; size as usize]);
        let needed_map = kernel.address_space.write(addr, &font_data).is_err();
        if needed_map {
            let _ =
                kernel
                    .address_space
                    .map(addr, size, nexium_memory::perm::Perm::R, "font_shmem");
            let _ = kernel.address_space.write(addr, &font_data);
        }
        if let Some(region) = kernel.address_space.host_region_at(addr) {
            if let Some(cpu) = cpu_mut() {
                unsafe {
                    if let Err(e) =
                        cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    {
                        log::warn!("failed to map font shmem in CPU: {}", e);
                    } else {
                        log::debug!("  â†’ registered font shmem at {:#x} with CPU", region.base);
                    }
                }
            }
        }
    } else {
        let backing = vec![0u8; size as usize];
        let needed_map = kernel.address_space.write(addr, &backing).is_err();
        if needed_map {
            let _ = kernel
                .address_space
                .map(addr, size, nexium_memory::perm::Perm::RW, "shared");
            let _ = kernel.address_space.write(addr, &backing);
            if let Some(region) = kernel.address_space.host_region_at(addr) {
                if let Some(cpu) = cpu_mut() {
                    unsafe {
                        if let Err(e) =
                            cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                        {
                            log::warn!("failed to map shared mem in CPU: {}", e);
                        } else {
                            log::debug!(
                                "  â†’ registered shared mem at {:#x} with CPU",
                                region.base
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_shared_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapSharedMemory");
    SUCCESS
}

fn svc_signal_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSignalEvent (X0=event_handle)");

    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    };

    log::debug!("  signaling event handle {:#x}", handle);

    if let Some(_event) = kernel.handles.get_handle(handle) {
        kernel.event_signals.insert(handle, true);
        kernel.threads.signal_handle(handle);
        log::debug!("  event {:#x} signaled (waiters woken)", handle);
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
        }
        return SUCCESS;
    } else {
        log::warn!("  invalid event handle {:#x}", handle);
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }
}

fn completed_thread_wait_index(kernel: &Kernel, handles: &[u32]) -> Option<usize> {
    handles.iter().position(|h| {
        if !matches!(
            kernel
                .handles
                .get_handle(*h)
                .map(|handle| handle.handle_type),
            Some(HandleType::Thread)
        ) {
            return false;
        }
        if kernel.exited_thread_handles.contains_key(h) {
            return true;
        }
        kernel
            .threads
            .threads
            .get(h)
            .map(|thread| matches!(thread.state, crate::kernel::threads::ThreadState::Exited))
            .unwrap_or(false)
    })
}

fn ready_event_wait_index(
    handles: &[u32],
    applet_message_event: Option<u32>,
    applet_messages_pending: bool,
    event_signals: &std::collections::HashMap<u32, bool>,
) -> Option<usize> {
    handles.iter().position(|handle| {
        (Some(*handle) == applet_message_event && applet_messages_pending)
            || event_signals.get(handle).copied().unwrap_or(false)
    })
}

fn reset_event_signal(
    event_signals: &mut std::collections::HashMap<u32, bool>,
    is_event: bool,
    handle: u32,
) -> u32 {
    if !is_event {
        return KERNEL_INVALID_HANDLE;
    }
    let Some(signaled) = event_signals.get_mut(&handle) else {
        return KERNEL_EVENT_INVALID_STATE;
    };
    if !*signaled {
        return KERNEL_EVENT_INVALID_STATE;
    }
    *signaled = false;
    SUCCESS
}

#[cfg(test)]
mod event_wait_tests {
    use super::ready_event_wait_index;
    use super::{reset_event_signal, KERNEL_EVENT_INVALID_STATE, KERNEL_INVALID_HANDLE, SUCCESS};
    use std::collections::HashMap;

    #[test]
    fn readable_event_wait_does_not_consume_signal() {
        let signals = HashMap::from([(0x123, true)]);
        assert_eq!(
            ready_event_wait_index(&[0x122, 0x123], None, false, &signals),
            Some(1)
        );
        assert_eq!(signals.get(&0x123), Some(&true));
        assert_eq!(
            ready_event_wait_index(&[0x123], None, false, &signals),
            Some(0)
        );
    }

    #[test]
    fn applet_message_event_keeps_handle_ordering() {
        let signals = HashMap::from([(0x125, true)]);
        assert_eq!(
            ready_event_wait_index(&[0x124, 0x125], Some(0x124), true, &signals),
            Some(0)
        );
    }

    #[test]
    fn reset_signal_requires_a_pending_event() {
        let mut signals = HashMap::from([(0x123, true), (0x124, false)]);
        assert_eq!(reset_event_signal(&mut signals, true, 0x123), SUCCESS);
        assert_eq!(signals.get(&0x123), Some(&false));
        assert_eq!(
            reset_event_signal(&mut signals, true, 0x123),
            KERNEL_EVENT_INVALID_STATE
        );
        assert_eq!(
            reset_event_signal(&mut signals, true, 0x124),
            KERNEL_EVENT_INVALID_STATE
        );
        assert_eq!(
            reset_event_signal(&mut signals, false, 0x123),
            KERNEL_INVALID_HANDLE
        );
    }
}

fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 {
    let (handles_addr, count, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1),
            (cpu.get_register(2) as u32).min(0x40),
            cpu.get_register(3),
        )
    } else {
        return 1;
    };

    let mut handles: Vec<u32> = Vec::with_capacity(count as usize);
    if count > 0 && handles_addr != 0 {
        let mut buf = vec![0u8; count as usize * 4];
        if kernel.address_space.read(handles_addr, &mut buf).is_ok() {
            for i in 0..count as usize {
                let h = u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
                handles.push(h);
            }
        }
    }

    kernel.refresh_bufferqueue_events();

    for h in &handles {
        crate::kernel::profile::record_wait_handle(*h);
    }

    if let Some(i) = completed_thread_wait_index(kernel, &handles) {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
            cpu.set_register(1, i as u64);
        }
        return SUCCESS;
    }

    if let Some(i) = ready_event_wait_index(
        &handles,
        kernel.applet_message_event,
        !kernel.applet_messages.is_empty(),
        &kernel.event_signals,
    ) {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
            cpu.set_register(1, i as u64);
        }
        return SUCCESS;
    }

    if timeout_ns == 0 {
        log::trace!(
            "svcWaitSync(timeout=0) handles={:?} applet_msg_event={:?} applet_msgs_pending={}",
            handles,
            kernel.applet_message_event,
            kernel.applet_messages.len()
        );

        {
            let state = crate::hid_state::get_hid_state();
            let mut hid = state.lock();
            if hid.shmem_va.is_some() {
                let cur = hid.input.clone();
                hid.tick(cur);
            }
        }
        {
            const AUDIO_PERIOD: std::time::Duration = std::time::Duration::from_millis(20);
            use parking_lot::Mutex;
            use std::sync::OnceLock;
            static LAST_TICK: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
            let cell = LAST_TICK.get_or_init(|| Mutex::new(std::time::Instant::now()));
            let mut last = cell.lock();
            if last.elapsed() >= AUDIO_PERIOD {
                *last = std::time::Instant::now();
                crate::services::audio_out::handlers::drain_audio_spill(kernel);
                let now = std::time::Instant::now();
                signal_due_audio_sessions(kernel, now);
            }
        }

        const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
        if !kernel.threads.ready.is_empty() {
            kernel.yield_after_svc = true;
        }
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, TIMEOUT_ERROR as u64);
            cpu.set_register(1, 0);
        }
        return TIMEOUT_ERROR;
    }

    if let Some(handle) = kernel.threads.current_handle() {
        if kernel.threads.take_wait_cancelled(handle) {
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, KERNEL_CANCELLED as u64);
                cpu.set_register(1, 0);
            }
            return KERNEL_CANCELLED;
        }
    }

    {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        if hid.shmem_va.is_some() {
            let cur = hid.input.clone();
            hid.tick(cur);
        }
    }

    const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
    if let Some(cpu) = cpu_ref() {
        let wake_at = if timeout_ns == u64::MAX {
            None
        } else {
            Some(std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns))
        };
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingHandle {
                handles: handles.clone(),
                wake_at,
            },
        );
        kernel.yield_after_svc = true;
    }
    SUCCESS
}

fn svc_cancel_synchronization(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        return KERNEL_INVALID_HANDLE;
    };
    let is_thread = matches!(
        kernel.handles.get_handle(handle),
        Some(entry) if entry.handle_type == HandleType::Thread
    );
    let action = if is_thread {
        kernel.threads.cancel_synchronization(handle)
    } else {
        None
    };
    let result = if is_thread {
        SUCCESS
    } else {
        KERNEL_INVALID_HANDLE
    };
    log::debug!(
        "svcCancelSynchronization handle={:#x} action={} result={:#x}",
        handle,
        match action {
            Some(true) => "woke",
            Some(false) => "pending",
            None => "missing",
        },
        result
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, result as u64);
    }
    result
}

fn svc_arbitrate_lock(kernel: &mut Kernel) -> u32 {
    let (owner_handle, mutex_addr, self_handle) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0) as u32,
            cpu.get_register(1),
            cpu.get_register(2) as u32,
        )
    } else {
        return 1;
    };

    let mut cur = [0u8; 4];
    let cur_word = if kernel.address_space.read(mutex_addr, &mut cur).is_ok() {
        u32::from_le_bytes(cur)
    } else {
        0
    };
    let holder = cur_word & !MUTEX_HAS_LISTENERS;
    let lr = cpu_ref().map(|c| c.get_register(30)).unwrap_or(0);

    if cur_word != (owner_handle | MUTEX_HAS_LISTENERS) {
        log::debug!(
            "svcArbitrateLock mutex={:#x} owner={:#x} self={:#x} cur={:#x} holder={:#x} -> retry lr={:#x}",
            mutex_addr,
            owner_handle,
            self_handle,
            cur_word,
            holder,
            lr
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
        }
        return SUCCESS;
    }

    if !kernel.threads.threads.contains_key(&owner_handle) {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
        }
        return KERNEL_INVALID_HANDLE;
    }

    let cur = kernel.threads.current_handle().unwrap_or(0);
    if owner_handle == self_handle || cur != self_handle {
        log::warn!(
            "svcArbitrateLock SELF/MISMATCH park: mutex={:#x} owner={:#x} self={:#x} current={:#x}",
            mutex_addr,
            owner_handle,
            self_handle,
            cur
        );
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    if let Some(cpu) = cpu_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingMutex {
                mutex_addr,
                owner_handle,
                tag: self_handle,
            },
        );
        kernel.yield_after_svc = true;
    }
    log::debug!(
        "svcArbitrateLock mutex={:#x} contended (owner={:#x} self={:#x}) -> parked",
        mutex_addr,
        owner_handle,
        self_handle
    );
    SUCCESS
}

fn release_mutex_word(kernel: &mut Kernel, mutex_addr: u64, caller: u32) -> (u32, u32, u32) {
    loop {
        let cur = match kernel.address_space.atomic_load_u32(mutex_addr) {
            Ok(w) => w,
            Err(_) => return (0, 0, 0),
        };
        let holder = cur & !MUTEX_HAS_LISTENERS;
        if holder != 0 && holder != caller {
            if kernel.threads.has_mutex_waiters(mutex_addr) && cur & MUTEX_HAS_LISTENERS == 0 {
                match kernel.address_space.atomic_cas_u32(
                    mutex_addr,
                    cur,
                    cur | MUTEX_HAS_LISTENERS,
                ) {
                    Ok(true) => return (cur, cur | MUTEX_HAS_LISTENERS, 0),
                    Ok(false) => continue,
                    Err(_) => return (cur, cur, 0),
                }
            }
            return (cur, cur, 0);
        }
        let peek = kernel.threads.peek_one_mutex_waiter(mutex_addr);
        let new_word = match peek {
            Some((_h, tag, more)) => {
                if more {
                    tag | MUTEX_HAS_LISTENERS
                } else {
                    tag
                }
            }
            None => 0,
        };
        match kernel
            .address_space
            .atomic_cas_u32(mutex_addr, cur, new_word)
        {
            Ok(true) => {
                if let Some((h, _, _)) = peek {
                    kernel.threads.commit_wake_mutex_waiter(mutex_addr, h);
                }
                return (cur, new_word, peek.map(|(h, _, _)| h).unwrap_or(0));
            }
            Ok(false) => continue,
            Err(_) => return (cur, cur, 0),
        }
    }
}

pub(crate) fn nudge_preempt_for_wake(kernel: &mut Kernel, woken: u32) {
    if woken == 0 {
        return;
    }
    let core = crate::kernel::cpu_local::current_core();
    let Some(current) = kernel.threads.current[core] else {
        return;
    };
    let Some(t) = kernel.threads.threads.get(&woken) else {
        return;
    };
    if t.core >= 0 && t.core != core as i32 {
        return;
    }
    if kernel.threads.effective_priority(woken) < kernel.threads.effective_priority(current) {
        kernel.yield_after_svc = true;
        kernel.preempt_after_svc = true;
    }
}

fn svc_arbitrate_unlock(kernel: &mut Kernel) -> u32 {
    let mutex_addr = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        return 1;
    };
    let owner_handle = kernel.threads.current_handle().unwrap_or(0);
    let (prev_word, new_word, handed) = release_mutex_word(kernel, mutex_addr, owner_handle);
    nudge_preempt_for_wake(kernel, handed);
    log::debug!(
        "svcArbitrateUnlock mutex={:#x} self={:#x} word {:#x}->{:#x} handed={:#x}",
        mutex_addr,
        owner_handle,
        prev_word,
        new_word,
        handed
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn condvar_trace_enabled(condvar_addr: u64) -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_CONDVAR_TRACE")
            .ok()
            .map(|value| value.parse::<u64>().unwrap_or(20_000))
            .unwrap_or(0)
    });
    if limit == 0 {
        return false;
    }

    static FILTERS: OnceLock<Vec<u64>> = OnceLock::new();
    let filters = FILTERS.get_or_init(|| {
        std::env::var("NEXIUM_CONDVAR_TRACE_FILTER")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| {
                        u64::from_str_radix(part.trim().trim_start_matches("0x"), 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    if !filters.is_empty() && !filters.contains(&condvar_addr) {
        return false;
    }

    static COUNT: AtomicU64 = AtomicU64::new(0);
    COUNT.fetch_add(1, Ordering::Relaxed) < limit
}

fn condwait_backtrace_range() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static RANGE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let value = std::env::var("NEXIUM_CONDWAIT_BACKTRACE").ok()?;
        let mut parts = value
            .split(',')
            .filter_map(|part| u64::from_str_radix(part.trim().trim_start_matches("0x"), 16).ok());
        let min = parts.next().unwrap_or(500_000_000);
        let max = parts.next().unwrap_or(1_500_000_000);
        Some((min, max))
    })
}

fn log_condwait_backtrace(kernel: &Kernel, self_handle: u32, condvar_addr: u64) {
    let Some(cpu) = cpu_ref() else {
        return;
    };
    let mut frames = Vec::with_capacity(10);
    frames.push(cpu.get_register(30));
    let mut fp = cpu.get_register(29);
    for _ in 0..8 {
        if fp == 0 || fp & 0x7 != 0 {
            break;
        }
        let Some(next_fp) = kernel.debug_read_u64(fp) else {
            break;
        };
        let Some(lr) = kernel.debug_read_u64(fp + 8) else {
            break;
        };
        if lr == 0 {
            break;
        }
        frames.push(lr);
        if next_fp <= fp {
            break;
        }
        fp = next_fp;
    }
    let chain = frames
        .iter()
        .map(|frame| format!("{frame:#x}"))
        .collect::<Vec<_>>()
        .join(" ");
    log::warn!(
        "[condwait-backtrace] self={:#x} cond={:#x} frames=[{}]",
        self_handle,
        condvar_addr,
        chain
    );
}

fn svc_wait_process_wide_key_atomic(kernel: &mut Kernel) -> u32 {
    let (mutex_addr, condvar_addr, self_handle, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2) as u32,
            cpu.get_register(3),
        )
    } else {
        return 1;
    };
    if let Some((min, max)) = condwait_backtrace_range() {
        if timeout_ns >= min && timeout_ns <= max {
            log_condwait_backtrace(kernel, self_handle, condvar_addr);
        } else if timeout_ns == u64::MAX {
            use std::collections::HashSet;
            use std::sync::Mutex as StdMutex;
            use std::sync::OnceLock;
            static SEEN: OnceLock<StdMutex<HashSet<(u32, u64)>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| StdMutex::new(HashSet::new()));
            if seen.lock().unwrap().insert((self_handle, condvar_addr)) {
                log_condwait_backtrace(kernel, self_handle, condvar_addr);
            }
        }
    }
    let lr = cpu_ref().map(|c| c.get_register(30)).unwrap_or(0);
    log::trace!(
        "svcWaitProcessWideKeyAtomic mutex={:#x} condvar={:#x} self_handle={:#x} timeout_ns={} lr={:#x}",
        mutex_addr,
        condvar_addr,
        self_handle,
        timeout_ns,
        lr
    );

    let owner_handle = kernel.threads.current_handle().unwrap_or(0);
    let _ = kernel
        .address_space
        .write(condvar_addr, &1u32.to_le_bytes());
    let (prev_word, new_word, handed) = release_mutex_word(kernel, mutex_addr, owner_handle);
    if condvar_trace_enabled(condvar_addr) {
        log::warn!(
            "[condvar-trace] wait current={:#x} self={:#x} mutex={:#x} cond={:#x} word={:#x}->{:#x} handed={:#x} timeout={} lr={:#x}",
            owner_handle,
            self_handle,
            mutex_addr,
            condvar_addr,
            prev_word,
            new_word,
            handed,
            timeout_ns,
            lr
        );
    }
    log::debug!(
        "cond_wait: self={:#x} mutex={:#x} cond={:#x} word {:#x}->{:#x} handed={:#x} timeout={}",
        self_handle,
        mutex_addr,
        condvar_addr,
        prev_word,
        new_word,
        handed,
        timeout_ns
    );

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }

    if timeout_ns == 0 {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_TIMEOUT as u64);
        }
        return KERNEL_TIMEOUT;
    }

    let wake_at = if timeout_ns == u64::MAX {
        None
    } else {
        Some(std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns))
    };

    let _ = kernel
        .address_space
        .write(condvar_addr, &1u32.to_le_bytes());

    if let Some(cpu) = cpu_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingCondvar {
                mutex_addr,
                condvar_addr,
                wake_at,
                spurious_wake: false,
            },
        );
        kernel.yield_after_svc = true;
    }

    SUCCESS
}

fn svc_signal_process_wide_key(kernel: &mut Kernel) -> u32 {
    let (condvar_addr, count) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(1) as i32)
    } else {
        return 1;
    };

    let max = if count <= 0 { i32::MAX } else { count };
    let mut woken = 0;
    'outer: for _ in 0..max {
        let Some((handle, mutex_addr)) = kernel.threads.peek_one_condvar_waiter(condvar_addr)
        else {
            break;
        };

        loop {
            let cur_word = match kernel.address_space.atomic_load_u32(mutex_addr) {
                Ok(w) => w,
                Err(_) => break 'outer,
            };
            let holder = cur_word & !MUTEX_HAS_LISTENERS;

            if holder == 0 {
                let new_word = if kernel.threads.has_mutex_waiters(mutex_addr) {
                    handle | MUTEX_HAS_LISTENERS
                } else {
                    handle
                };
                match kernel
                    .address_space
                    .atomic_cas_u32(mutex_addr, cur_word, new_word)
                {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(_) => break 'outer,
                }
                kernel.threads.wake_condvar_to_ready(handle);
                nudge_preempt_for_wake(kernel, handle);
                log::debug!(
                    "cond_signal handoff: cond={:#x} handle={:#x} mutex={:#x} word {:#x}->{:#x}",
                    condvar_addr,
                    handle,
                    mutex_addr,
                    cur_word,
                    new_word
                );
            } else {
                let new_word = cur_word | MUTEX_HAS_LISTENERS;
                if new_word != cur_word {
                    match kernel
                        .address_space
                        .atomic_cas_u32(mutex_addr, cur_word, new_word)
                    {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(_) => break 'outer,
                    }
                }
                kernel
                    .threads
                    .wake_condvar_into_mutex_waiter(handle, mutex_addr, holder, handle);
                log::debug!(
                    "cond_signal requeue: cond={:#x} handle={:#x} mutex={:#x} word {:#x}->{:#x} holder={:#x}",
                    condvar_addr,
                    handle,
                    mutex_addr,
                    cur_word,
                    new_word,
                    holder
                );
            }
            break;
        }
        woken += 1;
    }
    if !kernel.threads.has_condvar_waiters(condvar_addr) {
        let _ = kernel
            .address_space
            .write(condvar_addr, &0u32.to_le_bytes());
    }
    if condvar_trace_enabled(condvar_addr) {
        let current = kernel.threads.current_handle().unwrap_or(0);
        let lr = cpu_ref().map(|cpu| cpu.get_register(30)).unwrap_or(0);
        log::warn!(
            "[condvar-trace] signal current={:#x} cond={:#x} count={} woken={} remaining={} lr={:#x}",
            current,
            condvar_addr,
            count,
            woken,
            kernel.threads.has_condvar_waiters(condvar_addr),
            lr
        );
    }
    log::trace!(
        "svcSignalProcessWideKey cond={:#x} count={} woken={}",
        condvar_addr,
        count,
        woken
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_system_tick(_kernel: &mut Kernel) -> u32 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let elapsed = EPOCH.get_or_init(std::time::Instant::now).elapsed();
    let ticks = (elapsed.as_nanos() as u64).wrapping_mul(19_200_000) / 1_000_000_000;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, ticks);
    }
    SUCCESS
}

fn domain_group(kernel: &Kernel, session_handle: u32) -> u32 {
    kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.domain_group)
        .unwrap_or(session_handle)
}

fn domain_group_handles(kernel: &Kernel, session_handle: u32) -> Vec<u32> {
    let group = domain_group(kernel, session_handle);
    kernel
        .sessions
        .iter()
        .filter_map(|(&handle, session)| {
            if session.is_domain && session.domain_group == group {
                Some(handle)
            } else {
                None
            }
        })
        .collect()
}

fn next_domain_object_id(kernel: &Kernel, session_handle: u32) -> u32 {
    let group = domain_group(kernel, session_handle);
    kernel
        .sessions
        .values()
        .filter(|session| session.is_domain && session.domain_group == group)
        .map(|session| session.next_domain_object_id)
        .max()
        .unwrap_or(0)
}

fn alloc_domain_object(kernel: &mut Kernel, session_handle: u32, service_name: &str) -> u32 {
    let object_id = next_domain_object_id(kernel, session_handle);
    let group = domain_group(kernel, session_handle);
    for session in kernel.sessions.values_mut() {
        if session.is_domain && session.domain_group == group {
            session
                .domain_objects
                .insert(object_id, service_name.to_string());
            session.next_domain_object_id = object_id.saturating_add(1);
        }
    }
    object_id
}

fn close_domain_object(kernel: &mut Kernel, session_handle: u32, object_id: u32) -> Vec<u32> {
    let handles = domain_group_handles(kernel, session_handle);
    for handle in &handles {
        if let Some(session) = kernel.sessions.get_mut(handle) {
            session.close_object(object_id);
        }
    }
    handles
}

fn service_for_domain_object(
    kernel: &Kernel,
    session_handle: u32,
    object_id: u32,
) -> Option<String> {
    if let Some(name) = kernel
        .sessions
        .get(&session_handle)
        .and_then(|session| session.service_for_object(object_id))
    {
        return Some(name.to_string());
    }

    let group = domain_group(kernel, session_handle);
    kernel.sessions.values().find_map(|session| {
        if session.is_domain && session.domain_group == group {
            session.service_for_object(object_id).map(str::to_string)
        } else {
            None
        }
    })
}

fn domain_object_keys(kernel: &Kernel, session_handle: u32, object_id: u32) -> Vec<(u32, u32)> {
    let mut keys = vec![(session_handle, object_id)];
    for handle in domain_group_handles(kernel, session_handle) {
        if handle != session_handle {
            keys.push((handle, object_id));
        }
    }
    keys
}

fn release_hwopus_session_state(kernel: &mut Kernel, session_handle: u32) {
    kernel.services.hwopus.close(session_handle);
    let Some(session) = kernel.sessions.get(&session_handle) else {
        return;
    };
    if !session.is_domain {
        kernel
            .hwopus_decoders
            .retain(|(handle, _), _| *handle != session_handle);
        return;
    }

    let group = session.domain_group;
    let has_sibling = kernel.sessions.iter().any(|(&handle, candidate)| {
        handle != session_handle && candidate.is_domain && candidate.domain_group == group
    });
    if !has_sibling {
        kernel
            .hwopus_decoders
            .retain(|(handle, _), _| *handle != group);
    }
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    let (tls_addr, session_handle) = if let Some(cpu) = cpu_ref() {
        let x0 = cpu.get_register(0) as u32;
        log::trace!("SendSyncRequest: X0={:#x}", x0);
        (cpu.get_tpidrro_el0(), x0)
    } else {
        return 1;
    };

    let mut tls_buf = vec![0u8; 0x100];
    if kernel.address_space.read(tls_addr, &mut tls_buf).is_err() {
        log::warn!("SendSyncRequest: failed to read TLS at {:#x}", tls_addr);
        return 1;
    }

    let port_name = match kernel.sessions.get(&session_handle) {
        Some(s) => s.port_name.clone(),
        None => {
            log::warn!(
                "SendSyncRequest: invalid session handle {:#x}",
                session_handle
            );
            return 1;
        }
    };

    let hipc_header_raw = u64::from_le_bytes([
        tls_buf[0], tls_buf[1], tls_buf[2], tls_buf[3], tls_buf[4], tls_buf[5], tls_buf[6],
        tls_buf[7],
    ]);
    let cmd_type = (hipc_header_raw & 0xFFFF) as u16;

    match cmd_type {
        2 => {
            log::debug!(
                "session Close session={:#x} service={}",
                session_handle,
                port_name
            );
            release_hwopus_session_state(kernel, session_handle);
            if port_name == "IAudioOut" {
                crate::services::audio_out::handlers::close_audio_out_session(
                    kernel,
                    session_handle,
                );
            }
            if port_name == "IAudioRenderer" {
                kernel.close_audio_renderer_session(session_handle);
            }
            kernel.sessions.remove(&session_handle);
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        5 | 7 => {
            log::debug!(
                "Control cmd_type={} session={:#x} service={}",
                cmd_type,
                session_handle,
                port_name
            );
            let response = handle_control_request(kernel, session_handle, &port_name, &tls_buf);
            if !response.is_empty() {
                let mut response_buf = tls_buf.clone();
                let copy_len = response.len().min(response_buf.len());
                response_buf[..copy_len].copy_from_slice(&response[..copy_len]);
                let _ = kernel.address_space.write(tls_addr, &response_buf);
            }
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        _ => {}
    }

    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    let ipc_parse_result = ipc::IpcCtx::parse(tls_buf.clone(), is_domain);
    let mut ctx = match ipc_parse_result {
        Ok(c) => c,
        Err(e) => {
            log::warn!(
                "Failed to parse IPC message: {:?} (is_domain={})",
                e,
                is_domain
            );
            return 1;
        }
    };

    let cmd_id = ctx.cmif_in.cmd_id;

    let dispatch_target = if let Some(d) = ctx.domain {
        if d.kind == 2 {
            let group = domain_group(kernel, session_handle);
            let domain_handles = close_domain_object(kernel, session_handle, d.object_id);
            kernel.hwopus_decoders.remove(&(group, d.object_id));
            kernel.close_audio_renderer_object(group, d.object_id);
            for handle in domain_handles {
                kernel.open_files.remove(&(handle, d.object_id));
                kernel.file_system_roots.remove(&(handle, d.object_id));
                kernel.open_host_files.remove(&(handle, d.object_id));
                kernel.open_romfs_files.remove(&(handle, d.object_id));
                kernel.open_romfs_file_paths.remove(&(handle, d.object_id));
                kernel.open_file_handles.remove(&(handle, d.object_id));
                kernel.open_dir_lists.remove(&(handle, d.object_id));
                kernel.hwopus_decoders.remove(&(handle, d.object_id));
                kernel.close_audio_renderer_object(handle, d.object_id);
            }
            log::debug!(
                "domain Close-object session={:#x} object_id={}",
                session_handle,
                d.object_id
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        match service_for_domain_object(kernel, session_handle, d.object_id) {
            Some(name) => name,
            None => {
                log::warn!(
                    "domain object_id={} not found on session={:#x} (port={}) â†’ InvalidObject 0xCE01",
                    d.object_id,
                    session_handle,
                    port_name
                );
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, SUCCESS as u64);
                }
                return 0xCE01;
            }
        }
    } else {
        port_name.clone()
    };

    log::trace!(
        "IPC request service=\"{}\" cmd={} in_data={} is_domain={}",
        dispatch_target,
        cmd_id,
        ctx.cmif_in_data_len,
        is_domain
    );
    ipc_trace_request(
        kernel,
        session_handle,
        &port_name,
        &dispatch_target,
        cmd_id,
        is_domain,
        &ctx,
    );
    maybe_thread_snapshot(kernel, &dispatch_target, cmd_id);

    if dispatch_target == "fatal:u" && cmd_id == 1 {
        if ctx.cmif_in_data_len >= 4 {
            let result = u32::from_le_bytes([
                ctx.buf[ctx.cmif_in_data_off],
                ctx.buf[ctx.cmif_in_data_off + 1],
                ctx.buf[ctx.cmif_in_data_off + 2],
                ctx.buf[ctx.cmif_in_data_off + 3],
            ]);
            let module = result & 0x1FF;
            let desc = (result >> 9) & 0x1FFF;
            log::error!(
                "**** fatal:u ThrowFatal result={:#010x} module={} description={} ****",
                result,
                module,
                desc
            );
        }
    }

    let ipc_diagnostics = diagnostics_enabled();
    let in_data_preview: Vec<u8> = if ipc_diagnostics {
        let start = ctx.cmif_in_data_off;
        let end = (start + 32).min(ctx.buf.len());
        if start < ctx.buf.len() {
            ctx.buf[start..end].to_vec()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let response = if dispatch_target == "sm:" {
        dispatch_sm_command_v2(kernel, &mut ctx)
    } else {
        let mut pending_frames = std::mem::take(&mut kernel.pending_frames);
        let response = dispatch_service_v2(
            kernel,
            &dispatch_target,
            &mut ctx,
            session_handle,
            &mut pending_frames,
        );
        kernel.pending_frames = pending_frames;
        response
    };

    if ipc_diagnostics {
        use parking_lot::Mutex;
        use std::collections::HashSet;
        use std::sync::OnceLock;
        static SEEN: OnceLock<Mutex<HashSet<(String, u32)>>> = OnceLock::new();
        let seen_cell = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let key = (dispatch_target.clone(), cmd_id);
        let is_first = seen_cell.lock().insert(key);
        if is_first {
            let resp_preview: Vec<String> = response
                .iter()
                .take(64)
                .map(|b| format!("{:02x}", b))
                .collect();
            let in_preview: Vec<String> = in_data_preview
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect();
            let reply_rc = response
                .windows(4)
                .position(|w| w == b"SFCO")
                .and_then(|off| response.get(off + 8..off + 12))
                .map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
                .unwrap_or(0);
            log::debug!(
                "IPC FIRST-OCCURRENCE response (compare with RustSwitch) service={} cmd={} rc={:#010x} response_len={} response_first64={} in_data_first32={}",
                dispatch_target,
                cmd_id,
                reply_rc,
                response.len(),
                resp_preview.join(","),
                in_preview.join(",")
            );
        }
    }

    let mut response_buf = vec![0u8; 0x100];
    let copy_len = response.len().min(response_buf.len());
    response_buf[..copy_len].copy_from_slice(&response[..copy_len]);

    if kernel.address_space.write(tls_addr, &response_buf).is_err() {
        log::warn!(
            "SendSyncRequest: failed to write TLS response at {:#x}",
            tls_addr
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn handle_control_request(
    kernel: &mut Kernel,
    session_handle: u32,
    port_name: &str,
    tls_buf: &[u8],
) -> Vec<u8> {
    let parse_result = ipc::IpcCtx::parse(tls_buf.to_vec(), false);
    let mut ctx = match parse_result {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    match ctx.cmif_in.cmd_id {
        0 => {
            log::debug!(
                "Control: ConvertCurrentObjectToDomain service={}",
                port_name
            );
            if let Some(session) = kernel.sessions.get_mut(&session_handle) {
                session.convert_to_domain();
            }
            build_ipc_response(&ctx, 0, &1u32.to_le_bytes(), &[])
        }
        1 => {
            log::debug!("Control: CopyFromCurrentDomain service={}", port_name);
            build_ipc_response(&ctx, 0, &[], &[])
        }
        2 | 4 => {
            let dup_handle = kernel.handles.create_handle(HandleType::Session);
            let mut session = Session::new(dup_handle, port_name.to_string());
            if let Some(orig) = kernel.sessions.get(&session_handle) {
                session.is_domain = orig.is_domain;
                session.domain_group = orig.domain_group;
                session.domain_objects = orig.domain_objects.clone();
                session.next_domain_object_id = orig.next_domain_object_id;
            }
            kernel.sessions.insert(dup_handle, session);
            log::debug!(
                "Control: CloneCurrentObject service={} dup={:#x}",
                port_name,
                dup_handle
            );
            build_ipc_response(&mut ctx, 0, &[], &[dup_handle])
        }
        3 => {
            log::debug!(
                "Control: QueryPointerBufferSize â†’ 0x500 service={}",
                port_name
            );
            build_ipc_response(&mut ctx, 0, &0x500u16.to_le_bytes(), &[])
        }
        other => {
            log::debug!("Control: unknown cmd={} service={}", other, port_name);
            build_ipc_response(&mut ctx, 0, &[], &[])
        }
    }
}

pub(crate) fn build_ipc_response(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    move_handles: &[u32],
) -> Vec<u8> {
    build_ipc_response_full(ctx, result, out_data, move_handles, &[], &[])
}

pub(crate) fn build_ipc_response_copy(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    copy_handles: &[u32],
) -> Vec<u8> {
    build_ipc_response_full(ctx, result, out_data, &[], copy_handles, &[])
}

fn build_ipc_response_full(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    move_handles: &[u32],
    copy_handles: &[u32],
    out_objects: &[u32],
) -> Vec<u8> {
    if ctx.tipc.is_some() {
        return build_tipc_response(result, out_data, move_handles, copy_handles);
    }

    let is_domain = ctx.domain.is_some();

    let mut raw_size = 0usize;
    if is_domain {
        raw_size += 16;
    }
    raw_size += 16;
    raw_size += out_data.len();
    if is_domain {
        raw_size += out_objects.len() * 4;
    }
    let raw_padded = (raw_size + 3) & !3;

    let mut special_bytes: Vec<u8> = Vec::new();
    let has_special_header = !move_handles.is_empty() || !copy_handles.is_empty();
    if has_special_header {
        let mut sh: u32 = 0;
        sh |= (copy_handles.len() as u32 & 0xF) << 1;
        sh |= (move_handles.len() as u32 & 0xF) << 5;
        special_bytes.extend_from_slice(&sh.to_le_bytes());
        for h in copy_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
        for h in move_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
    }

    let mut hipc: u64 = 0;
    hipc |= ((raw_padded / 4) as u64 & 0x3FF) << 32;
    if has_special_header {
        hipc |= 1u64 << 63;
    }

    let raw_data_off = (8 + special_bytes.len() + 15) & !15;
    let total = raw_data_off + raw_padded;
    let mut out = vec![0u8; total];

    out[0..8].copy_from_slice(&hipc.to_le_bytes());
    if !special_bytes.is_empty() {
        out[8..8 + special_bytes.len()].copy_from_slice(&special_bytes);
    }

    let mut p = raw_data_off;
    if is_domain {
        out[p..p + 4].copy_from_slice(&(out_objects.len() as u32).to_le_bytes());
        p += 16;
    }

    out[p..p + 4].copy_from_slice(b"SFCO");
    out[p + 4..p + 8].copy_from_slice(&1u32.to_le_bytes());
    out[p + 8..p + 12].copy_from_slice(&result.to_le_bytes());
    out[p + 12..p + 16].copy_from_slice(&ctx.cmif_in.token.to_le_bytes());
    p += 16;

    if !out_data.is_empty() && p + out_data.len() <= out.len() {
        out[p..p + out_data.len()].copy_from_slice(out_data);
        p += out_data.len();
    }

    if is_domain {
        for obj in out_objects {
            if p + 4 <= out.len() {
                out[p..p + 4].copy_from_slice(&obj.to_le_bytes());
                p += 4;
            }
        }
    }

    out
}

fn build_tipc_response(
    result: u32,
    out_data: &[u8],
    move_handles: &[u32],
    copy_handles: &[u32],
) -> Vec<u8> {
    let mut special_bytes: Vec<u8> = Vec::new();
    let has_special_header = !move_handles.is_empty() || !copy_handles.is_empty();
    if has_special_header {
        let mut sh: u32 = 0;
        sh |= (copy_handles.len() as u32 & 0xF) << 1;
        sh |= (move_handles.len() as u32 & 0xF) << 5;
        special_bytes.extend_from_slice(&sh.to_le_bytes());
        for h in copy_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
        for h in move_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
    }

    let raw_size = 4 + out_data.len();
    let raw_padded = (raw_size + 3) & !3;
    let mut hipc: u64 = ((raw_padded / 4) as u64 & 0x3FF) << 32;
    if has_special_header {
        hipc |= 1u64 << 63;
    }

    let mut out = vec![0u8; 8 + special_bytes.len() + raw_padded];
    out[0..8].copy_from_slice(&hipc.to_le_bytes());
    if !special_bytes.is_empty() {
        out[8..8 + special_bytes.len()].copy_from_slice(&special_bytes);
    }

    let p = 8 + special_bytes.len();
    out[p..p + 4].copy_from_slice(&result.to_le_bytes());
    if !out_data.is_empty() {
        out[p + 4..p + 4 + out_data.len()].copy_from_slice(out_data);
    }
    out
}

fn dispatch_sm_command_v2(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx) -> Vec<u8> {
    match ctx.cmif_in.cmd_id {
        0 => {
            let pid = ctx.send_pid;
            log::debug!("sm:RegisterClient pid={:?}", pid);
            build_ipc_response(ctx, 0, &[], &[])
        }
        1 => {
            let name_bytes = if ctx.cmif_in_data_off + 8 <= ctx.buf.len() {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&ctx.buf[ctx.cmif_in_data_off..ctx.cmif_in_data_off + 8]);
                arr
            } else {
                [0u8; 8]
            };
            let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(&name_bytes);
            let name = String::from_utf8_lossy(trimmed).into_owned();
            log::debug!("sm:GetServiceHandle name={} raw={:02x?}", name, name_bytes);

            let handle = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(handle, name.clone());
            kernel.sessions.insert(handle, session);

            build_ipc_response(ctx, 0, &[], &[handle])
        }
        2 => {
            log::debug!("sm:RegisterService");
            let handle = kernel.handles.create_handle(HandleType::Session);
            build_ipc_response(ctx, 0, &[], &[handle])
        }
        3 => {
            log::debug!("sm:UnregisterService");
            build_ipc_response(ctx, 0, &[], &[])
        }
        4 => {
            log::debug!("sm:DetachClient");
            build_ipc_response(ctx, 0, &[], &[])
        }
        other => {
            log::warn!("sm: unknown cmd={}", other);
            build_ipc_response(ctx, 1, &[], &[])
        }
    }
}

fn dump_throw_context(kernel: &Kernel) {
    let cpu = match cpu_ref() {
        Some(c) => c,
        None => return,
    };
    let base = kernel.code_base;
    log::warn!(
        "[throw] ===== uncaught-exception context (code_base={:#x}) =====",
        base
    );
    for row in 0..4 {
        let r = row * 8;
        log::warn!(
            "[throw] x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x}",
            r,
            cpu.get_register(r),
            r + 1,
            cpu.get_register(r + 1),
            r + 2,
            cpu.get_register(r + 2),
            r + 3,
            cpu.get_register(r + 3),
            r + 4,
            cpu.get_register(r + 4),
            r + 5,
            cpu.get_register(r + 5),
            r + 6,
            cpu.get_register(r + 6),
            r + 7,
            cpu.get_register(r + 7)
        );
    }
    log::warn!(
        "[throw] x28={:#018x} x29(fp)={:#018x} x30(lr)={:#018x} sp={:#018x}",
        cpu.get_register(28),
        cpu.get_register(29),
        cpu.get_register(30),
        cpu.get_sp()
    );
    let read_u64 = |va: u64| -> Option<u64> {
        let mut b = [0u8; 8];
        if kernel.address_space.read(va, &mut b).is_ok() {
            Some(u64::from_le_bytes(b))
        } else {
            None
        }
    };
    let read_u32 = |va: u64| -> Option<u32> {
        let mut b = [0u8; 4];
        if kernel.address_space.read(va, &mut b).is_ok() {
            Some(u32::from_le_bytes(b))
        } else {
            None
        }
    };
    let mut fp = cpu.get_register(29);
    for depth in 0..28u32 {
        if fp == 0 || (fp & 7) != 0 {
            break;
        }
        let next_fp = match read_u64(fp) {
            Some(v) => v,
            None => break,
        };
        let ret = match read_u64(fp.wrapping_add(8)) {
            Some(v) => v,
            None => break,
        };
        let off = ret.wrapping_sub(base);
        let mut words = String::new();
        for i in 0..6u64 {
            if let Some(w) = read_u32(ret.wrapping_sub(20).wrapping_add(i * 4)) {
                words.push_str(&format!("{:08x} ", w));
            }
        }
        log::warn!(
            "[throw] #{:02} ret=+{:#x} (raw={:#x}) fp={:#x} | callsite[ret-20..ret+4]= {}",
            depth,
            off,
            ret,
            fp,
            words
        );
        if next_fp <= fp {
            break;
        }
        fp = next_fp;
    }
    log::warn!("[throw] ===== end context =====");
}

fn dispatch_service_v2(
    kernel: &mut Kernel,
    port_name: &str,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    pending_frames: &mut Vec<crate::services::FrameOut>,
) -> Vec<u8> {
    struct IpcProfileGuard(std::time::Instant, String);
    impl Drop for IpcProfileGuard {
        fn drop(&mut self) {
            crate::kernel::profile::record_ipc(&self.1, self.0);
        }
    }
    let cmd_id = ctx.cmif_in.cmd_id;
    let _guard = if crate::kernel::profile::enabled() {
        let key = if std::env::var_os("NEXIUM_PROFILE_IPC_CMD").is_some() {
            format!("{port_name}.cmd{cmd_id}")
        } else {
            port_name.to_string()
        };
        Some(IpcProfileGuard(std::time::Instant::now(), key))
    } else {
        None
    };

    if port_name == "ILogService" && cmd_id == 0 {
        let sb = ctx
            .send_buffers
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .copied()
            .or_else(|| {
                ctx.send_statics
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .copied()
            });
        if let Some(b) = sb {
            let mut buf = vec![0u8; (b.size as usize).min(0x400)];
            if kernel.address_space.read(b.addr, &mut buf).is_ok() {
                let txt: String = buf
                    .iter()
                    .map(|&c| {
                        if (0x20..0x7f).contains(&c) {
                            c as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                log::warn!("[lm.Log] {}", txt);
                if txt.contains("bad_alloc") || txt.contains("uncaught") || txt.contains("abort") {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static DUMPED: AtomicBool = AtomicBool::new(false);
                    if !DUMPED.swap(true, Ordering::Relaxed) {
                        dump_throw_context(kernel);
                    }
                }
            }
        }
    }

    if port_name == "nvdrv"
        || port_name == "nvdrv:a"
        || port_name == "nvdrv:s"
        || port_name == "nvdrv:t"
    {
        return dispatch_nvdrv_command(kernel, ctx, port_name);
    }

    if port_name == "IHOSBinderDriver" && (cmd_id == 0 || cmd_id == 3) {
        return handle_binder_transact(kernel, ctx, session_handle);
    }

    if port_name == "csrng" && cmd_id == 0 {
        let target = ctx
            .recv_buffers
            .iter()
            .chain(ctx.recv_statics.iter())
            .find(|b| b.size > 0 && b.addr != 0)
            .copied();
        if let Some(buf) = target {
            let size = match usize::try_from(buf.size) {
                Ok(size) => size,
                Err(_) => {
                    log::error!("csrng: receive buffer too large: {:#x}", buf.size);
                    return build_ipc_response(ctx, 0xD401, &[], &[]);
                }
            };
            let mut bytes = vec![0u8; size];
            if let Err(err) = getrandom::fill(&mut bytes) {
                log::error!("csrng: OS random source failed: {}", err);
                return build_ipc_response(ctx, 0xD401, &[], &[]);
            }
            if let Err(err) = kernel.address_space.write(buf.addr, &bytes) {
                log::error!(
                    "csrng: failed to write {} bytes at {:#x}: {:?}",
                    size,
                    buf.addr,
                    err
                );
                return build_ipc_response(ctx, 0xD401, &[], &[]);
            }
            log::debug!(
                "csrng.GenerateRandomBytes size={} addr={:#x}",
                size,
                buf.addr
            );
        } else {
            log::warn!("csrng.GenerateRandomBytes: no receive buffer");
        }
        return build_ipc_response(ctx, 0, &[], &[]);
    }

    if port_name == "IStorageAccessor"
        && cmd_id == 10
        && crate::services::am::pending_applet_id() == crate::services::am::APPLET_ID_CONTROLLER
    {
        let sb = ctx
            .send_buffers
            .iter()
            .chain(ctx.send_statics.iter())
            .find(|b| b.size > 0 && b.addr != 0)
            .copied();
        if let Some(b) = sb {
            if b.size == 0x14 {
                let mut tmp = [0u8; 0x14];
                if kernel.address_space.read(b.addr, &mut tmp).is_ok()
                    && u32::from_le_bytes([tmp[0], tmp[1], tmp[2], tmp[3]]) == 0x14
                {
                    let style_set = u32::from_le_bytes([tmp[0xc], tmp[0xd], tmp[0xe], tmp[0xf]]);
                    let sel = crate::hid_state::apply_controller_applet_style(style_set);
                    crate::services::am::set_controller_selected_id(sel);
                    let events: Vec<u32> = kernel.services.hid.style_change_events.clone();
                    for h in events {
                        kernel.event_signals.insert(h, true);
                        kernel.threads.signal_handle(h);
                    }
                    log::debug!(
                        "am: controller applet configured style_set={:#x} selected_id={:#x}",
                        style_set,
                        sel
                    );
                }
            }
        }
    }

    if port_name == "IStorageAccessor"
        && cmd_id == 10
        && crate::services::am::pending_applet_id() == crate::services::am::APPLET_ID_SWKBD
    {
        let sb = ctx
            .send_buffers
            .iter()
            .chain(ctx.send_statics.iter())
            .find(|b| b.size > 0 && b.addr != 0)
            .copied();
        if let Some(b) = sb {
            let size = (b.size as usize).min(0x2000);
            let mut bytes = vec![0u8; size];
            if kernel.address_space.read(b.addr, &mut bytes).is_ok() {
                crate::swkbd_state::capture_storage_write(&bytes);
            }
        }
    }

    if port_name == "ILibraryAppletCreator"
        && cmd_id == 11
        && crate::services::am::pending_applet_id() == crate::services::am::APPLET_ID_SWKBD
    {
        let tmem_handle = ctx
            .copy_handles
            .first()
            .or_else(|| ctx.move_handles.first())
            .copied();
        if let Some(h) = tmem_handle {
            if let Some(&(addr, size)) = kernel.transfer_memories.get(&h) {
                kernel.swkbd_workbuf = Some((addr, size));
                log::debug!(
                    "swkbd: workbuf tmem handle={:#x} addr={:#x} size={:#x}",
                    h,
                    addr,
                    size
                );
            } else {
                log::warn!(
                    "swkbd: CreateTransferMemoryStorage with unknown tmem handle {:#x}",
                    h
                );
            }
        }
    }

    if let Some(buffer_data) = applet_buffer_response(port_name, cmd_id) {
        let read_offset = if port_name == "IStorageAccessorOut" && cmd_id == 11 {
            let off = ctx.cmif_in_data_off;
            if ctx.cmif_in_data_len >= 8 && off + 8 <= ctx.buf.len() {
                u64::from_le_bytes([
                    ctx.buf[off],
                    ctx.buf[off + 1],
                    ctx.buf[off + 2],
                    ctx.buf[off + 3],
                    ctx.buf[off + 4],
                    ctx.buf[off + 5],
                    ctx.buf[off + 6],
                    ctx.buf[off + 7],
                ]) as usize
            } else {
                0
            }
        } else {
            0
        };
        let buffer_data: &[u8] = if read_offset > 0 {
            buffer_data.get(read_offset..).unwrap_or(&[])
        } else {
            &buffer_data[..]
        };
        let target_buf = ctx
            .recv_buffers
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = target_buf {
            let write_len = buffer_data.len().min(buf.size as usize);
            let _ = kernel
                .address_space
                .write(buf.addr, &buffer_data[..write_len]);
            log::debug!(
                "  wrote {} bytes to recv buf at {:#x} (avail {}, offset {})",
                write_len,
                buf.addr,
                buf.size,
                read_offset
            );
        } else {
            log::debug!(
                "  no recv buffer/static available for {} cmd={}",
                port_name,
                cmd_id
            );
        }
    }

    if port_name == "ILibraryAppletCreator" && cmd_id == 0 {
        let off = ctx.cmif_in_data_off;
        if off + 4 <= ctx.buf.len() {
            let applet_id = u32::from_le_bytes([
                ctx.buf[off],
                ctx.buf[off + 1],
                ctx.buf[off + 2],
                ctx.buf[off + 3],
            ]);
            let applet_mode = if off + 8 <= ctx.buf.len() {
                u32::from_le_bytes([
                    ctx.buf[off + 4],
                    ctx.buf[off + 5],
                    ctx.buf[off + 6],
                    ctx.buf[off + 7],
                ])
            } else {
                0
            };
            log::debug!(
                "am: CreateLibraryApplet applet_id={:#x} mode={}",
                applet_id,
                applet_mode
            );
            crate::services::am::set_pending_applet_id(applet_id);
            if applet_id == crate::services::am::APPLET_ID_SWKBD {
                let generation = crate::swkbd_state::begin_applet();
                kernel.swkbd_workbuf = None;
                kernel.swkbd_state_changed_events.clear();
                log::info!("swkbd: applet created (gen={})", generation);
            }
        }
    }

    if port_name == "IApplicationFunctions" && cmd_id == 1 {
        let kind = ipc_input_u32(ctx, 0).unwrap_or(2);
        if kind == 1 {
            log::debug!("am: PopLaunchParameter(UserChannel) -> no data");
            return build_ipc_response(ctx, 0x480, &[], &[]);
        }
    }

    if let Some(sub_service) = crate::services::am::proxy_subsession(port_name, cmd_id) {
        log::debug!(
            "{} cmd={} â†’ returning {} sub-session",
            port_name,
            cmd_id,
            sub_service
        );
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if port_name == "fsp-srv" && matches!(cmd_id, 51 | 52 | 53) {
        if let Some(root) = fs_save_data_root(kernel, ctx) {
            log::debug!(
                "fsp-srv.OpenSaveDataFileSystem cmd={} title_id={:#018x} root={}",
                cmd_id,
                kernel.title_id,
                root.display()
            );
            return return_file_system_with_root(kernel, ctx, session_handle, root);
        }
        return build_ipc_response(ctx, 0x202, &[], &[]);
    }

    if port_name == "fsp-srv" && cmd_id == 202 && ctx.cmif_in_data_len >= 16 {
        let title_id_off = ctx.cmif_in_data_off + 8;
        let title_id = u64::from_le_bytes([
            ctx.buf[title_id_off],
            ctx.buf[title_id_off + 1],
            ctx.buf[title_id_off + 2],
            ctx.buf[title_id_off + 3],
            ctx.buf[title_id_off + 4],
            ctx.buf[title_id_off + 5],
            ctx.buf[title_id_off + 6],
            ctx.buf[title_id_off + 7],
        ]);
        let available = kernel.system_romfs(title_id).is_some()
            || matches!(title_id, 0x0100_0000_0000_0802 | 0x0100_0000_0000_0823);
        if available {
            let service = format!("IFsStorageSystemData:{title_id:016x}");
            return return_subsession(kernel, ctx, session_handle, &service);
        }
        log::warn!(
            "fsp-srv.OpenDataStorageByDataId: system archive {:#018x} unavailable",
            title_id
        );
        return build_ipc_response(ctx, 0x202, &[], &[]);
    }

    if let Some(response) = dispatch_aoc_bcat(kernel, port_name, ctx, cmd_id) {
        return response;
    }

    if let Some(sub_service) = subsession_service(port_name, cmd_id) {
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if matches!(port_name, "bsd:u" | "bsd:s") {
        let address_space = std::sync::Arc::clone(&kernel.address_space);
        let (rc, data, wait) = kernel.services.bsd.dispatch_ipc(&address_space, ctx);
        if wait {
            kernel.present_pace_until =
                Some(std::time::Instant::now() + std::time::Duration::from_millis(1));
        }
        return build_ipc_response(ctx, rc, &data, &[]);
    }

    if let Some((rc, data, handles)) =
        crate::services::am::dispatch_command(kernel, port_name, cmd_id)
    {
        log::trace!(
            "am.{}.cmd_{} rc={:#x} â†’ {} bytes, {} handle(s) [copy]",
            port_name,
            cmd_id,
            rc,
            data.len(),
            handles.len()
        );
        return build_ipc_response_copy(ctx, rc, &data, &handles);
    }

    if port_name == "IFriendService" {
        match cmd_id {
            0 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            10101 | 10400 | 20100 | 20101 | 20200 | 22010 => {
                return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]);
            }
            10120 | 10420 => return build_ipc_response(ctx, 0, &[1], &[]),
            10601 | 10610 | 10700 => return build_ipc_response(ctx, 0, &[], &[]),
            other => {
                log::debug!("IFriendService.cmd_{} stubbed empty success", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "IDatabaseService" {
        match cmd_id {
            0 => return build_ipc_response(ctx, 0, &[0], &[]),
            1 => return build_ipc_response(ctx, 0, &[0], &[]),
            2 => {
                let source = ipc_input_u32(ctx, 0).unwrap_or(0);
                let count = if source & 2 != 0 { 6u32 } else { 0 };
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            3 | 4 => {
                let source = ipc_input_u32(ctx, 0).unwrap_or(0);
                let requested = if source & 2 != 0 { 6usize } else { 0 };
                let stride = if cmd_id == 3 { 0x5c } else { 0x58 };
                let buffer = ctx
                    .recv_buffers
                    .iter()
                    .chain(ctx.recv_statics.iter())
                    .find(|buffer| buffer.addr != 0);
                let capacity = buffer.map_or(0, |buffer| buffer.size as usize / stride);
                let count = requested.min(capacity);
                if let Some(buffer) = buffer.filter(|_| count != 0) {
                    let mut elements = vec![0u8; count * stride];
                    for index in 0..count as u32 {
                        let offset = index as usize * stride;
                        let info = kernel.services.mii.build_default(index);
                        elements[offset..offset + 0x58].copy_from_slice(&info);
                        if cmd_id == 3 {
                            elements[offset + 0x58..offset + 0x5c]
                                .copy_from_slice(&1u32.to_le_bytes());
                        }
                    }
                    if let Err(error) = kernel.address_space.write_checked(buffer.addr, &elements) {
                        log::error!("mii.Get: failed to write output elements: {}", error);
                        return build_ipc_response(ctx, 0x47e, &(count as u32).to_le_bytes(), &[]);
                    }
                }
                let result = if count < requested { 0x47e } else { 0 };
                return build_ipc_response(ctx, result, &(count as u32).to_le_bytes(), &[]);
            }
            6 => {
                let gender = ipc_input_u32(ctx, 4).unwrap_or(2);
                let info = kernel.services.mii.build_random(gender);
                return build_ipc_response(ctx, 0, &info, &[]);
            }
            7 => {
                let index = ipc_input_u32(ctx, 0).unwrap_or(0);
                if index >= 6 {
                    return build_ipc_response(ctx, 0x27e, &[], &[]);
                }
                let info = kernel.services.mii.build_default(index);
                return build_ipc_response(ctx, 0, &info, &[]);
            }
            22 => {
                let version = ipc_input_u32(ctx, 0).unwrap_or(0);
                kernel.services.mii.set_interface_version(version);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            other => {
                log::warn!("IDatabaseService.cmd_{} stubbed empty success", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "INfpUser" {
        match cmd_id {
            0 | 1 => return build_ipc_response(ctx, 0, &[], &[]),
            2 => return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]),
            17 | 18 | 23 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                log::debug!("nfp IUser.cmd_{} â†’ event {:#x}", cmd_id, h);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            19 => return build_ipc_response(ctx, 0, &1u32.to_le_bytes(), &[]),
            20 | 21 => return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]),
            other => {
                log::debug!("nfp IUser.cmd_{} stubbed empty success", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "IFileSystem" {
        let fs_obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let path_str = fs_read_path(ctx, &kernel.address_space);
        let basename = std::path::Path::new(&path_str)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        match cmd_id {
            0 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    log::warn!(
                        "IFileSystem.CreateFile path={:?} â†’ 0x202 PathNotFound",
                        path_str
                    );
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                if let Some(parent) = host.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&host)
                {
                    Ok(_) => {
                        log::debug!("IFileSystem.CreateFile path={:?} â†’ SUCCESS", path_str);
                        return build_ipc_response(ctx, 0, &[], &[]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        return build_ipc_response(ctx, 0x402, &[], &[]);
                    }
                    Err(_) => return build_ipc_response(ctx, 0x402, &[], &[]),
                }
            }
            1 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_file(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            2 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::create_dir_all(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x402, &[], &[]),
                }
            }
            3 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_dir(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            4 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_dir_all(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            5 | 6 => {
                let all: Vec<_> = ctx
                    .send_statics
                    .iter()
                    .chain(ctx.send_buffers.iter())
                    .filter(|b| b.size > 0 && b.addr != 0)
                    .copied()
                    .collect();
                let mut new_path = String::new();
                if let Some(b) = all.get(1) {
                    let n = (b.size as usize).min(0x301);
                    let mut bytes = vec![0u8; n];
                    if kernel.address_space.read(b.addr, &mut bytes).is_ok() {
                        let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
                        new_path = String::from_utf8_lossy(&bytes[..end]).into_owned();
                    }
                }
                let old_host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let new_host = fs_host_path(kernel, session_handle, fs_obj_id, &new_path);
                match (old_host, new_host) {
                    (Some(o), Some(n)) => {
                        if let Some(parent) = n.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        match std::fs::rename(&o, &n) {
                            Ok(()) => {
                                log::debug!(
                                    "IFileSystem.Rename{} {:?} -> {:?} â†’ SUCCESS",
                                    if cmd_id == 5 { "File" } else { "Directory" },
                                    path_str,
                                    new_path
                                );
                                return build_ipc_response(ctx, 0, &[], &[]);
                            }
                            Err(e) => {
                                log::warn!(
                                    "IFileSystem.Rename {:?} -> {:?} â†’ err {}",
                                    path_str,
                                    new_path,
                                    e
                                );
                                return build_ipc_response(ctx, 0x202, &[], &[]);
                            }
                        }
                    }
                    _ => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            7 => {
                let entry_type: u32 = {
                    let in_homebrew = !basename.is_empty()
                        && kernel
                            .homebrew_dir
                            .as_ref()
                            .map(|d| d.join(&basename).is_file())
                            .unwrap_or(false);
                    if in_homebrew {
                        1
                    } else if let Some(host) =
                        fs_host_path(kernel, session_handle, fs_obj_id, &path_str)
                    {
                        match std::fs::metadata(&host) {
                            Ok(m) if m.is_dir() => 0,
                            Ok(_) => 1,
                            Err(_) => {
                                if let Some(ty) = romfs_entry_type(kernel.nro_romfs(), &path_str) {
                                    ty
                                } else {
                                    fs_trace_path("GetEntryType", &path_str, "not_found");
                                    log::debug!(
                                        "IFileSystem.GetEntryType path={:?} â†’ 0x202 NotFound",
                                        path_str
                                    );
                                    return build_ipc_response(ctx, 0x202, &[], &[]);
                                }
                            }
                        }
                    } else if let Some(ty) = romfs_entry_type(kernel.nro_romfs(), &path_str) {
                        ty
                    } else {
                        fs_trace_path("GetEntryType", &path_str, "not_found");
                        log::debug!(
                            "IFileSystem.GetEntryType path={:?} â†’ 0x202 NotFound",
                            path_str
                        );
                        return build_ipc_response(ctx, 0x202, &[], &[]);
                    }
                };
                fs_trace_path("GetEntryType", &path_str, &format!("type={}", entry_type));
                log::debug!(
                    "IFileSystem.GetEntryType path={:?} â†’ {}",
                    path_str,
                    entry_type
                );
                return build_ipc_response(ctx, 0, &entry_type.to_le_bytes(), &[]);
            }
            8 => {
                let mmap_arc: Option<std::sync::Arc<memmap2::Mmap>> = if !basename.is_empty() {
                    kernel.homebrew_dir.as_ref().and_then(|dir| {
                        let candidate = dir.join(&basename);
                        std::fs::File::open(&candidate)
                            .ok()
                            .and_then(|f| unsafe { memmap2::Mmap::map(&f) }.ok())
                            .map(std::sync::Arc::new)
                    })
                } else {
                    None
                };

                let is_domain = kernel
                    .sessions
                    .get(&session_handle)
                    .map(|s| s.is_domain)
                    .unwrap_or(false);
                let new_obj_id = if is_domain {
                    next_domain_object_id(kernel, session_handle)
                } else {
                    0
                };

                if let Some(m) = mmap_arc {
                    let mmap_len = m.len();
                    kernel.open_files.insert((session_handle, new_obj_id), m);
                    log::debug!(
                        "IFileSystem.OpenFile path={:?} â†’ IFile (NRO mmap {} bytes)",
                        path_str,
                        mmap_len
                    );
                } else {
                    let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                    if let Some(host) = host {
                        if host.is_file() {
                            kernel
                                .open_host_files
                                .insert((session_handle, new_obj_id), host.clone());
                            fs_trace_path(
                                "OpenFile",
                                &path_str,
                                &format!("host {}", host.display()),
                            );
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} â†’ IFile (host {})",
                                path_str,
                                host.display()
                            );
                        } else if let Some(rf) = romfs_open_file(kernel.nro_romfs(), &path_str) {
                            kernel
                                .open_romfs_files
                                .insert((session_handle, new_obj_id), rf);
                            kernel
                                .open_romfs_file_paths
                                .insert((session_handle, new_obj_id), path_str.clone());
                            fs_trace_path(
                                "OpenFile",
                                &path_str,
                                &format!("romfs off={:#x} size={}", rf.0, rf.1),
                            );
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} â†’ IFile (romfs off={:#x} size={})",
                                path_str,
                                rf.0,
                                rf.1
                            );
                        } else {
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} â†’ 0x202 NotFound (host miss)",
                                path_str
                            );
                            fs_trace_path("OpenFile", &path_str, "not_found_host_miss");
                            return build_ipc_response(ctx, 0x202, &[], &[]);
                        }
                    } else if let Some(rf) = romfs_open_file(kernel.nro_romfs(), &path_str) {
                        kernel
                            .open_romfs_files
                            .insert((session_handle, new_obj_id), rf);
                        kernel
                            .open_romfs_file_paths
                            .insert((session_handle, new_obj_id), path_str.clone());
                        fs_trace_path(
                            "OpenFile",
                            &path_str,
                            &format!("romfs off={:#x} size={}", rf.0, rf.1),
                        );
                        log::debug!(
                            "IFileSystem.OpenFile path={:?} â†’ IFile (romfs off={:#x} size={})",
                            path_str,
                            rf.0,
                            rf.1
                        );
                    } else {
                        fs_trace_path("OpenFile", &path_str, "not_found");
                        log::debug!(
                            "IFileSystem.OpenFile path={:?} â†’ 0x202 NotFound",
                            path_str
                        );
                        return build_ipc_response(ctx, 0x202, &[], &[]);
                    }
                }
                return return_subsession(kernel, ctx, session_handle, "IFile");
            }
            9 => {
                let in_off = ctx.cmif_in_data_off;
                let filter = if ctx.cmif_in_data_len >= 4 {
                    u32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ])
                } else {
                    0
                };

                let is_domain = kernel
                    .sessions
                    .get(&session_handle)
                    .map(|s| s.is_domain)
                    .unwrap_or(false);
                let new_obj_id = if is_domain {
                    next_domain_object_id(kernel, session_handle)
                } else {
                    0
                };

                let mut entries: Vec<(String, bool, u64)> = Vec::new();
                if let Some(host) = fs_host_path(kernel, session_handle, fs_obj_id, &path_str) {
                    let _ = std::fs::create_dir_all(&host);
                    if let Ok(rd) = std::fs::read_dir(&host) {
                        for e in rd.filter_map(|e| e.ok()) {
                            let Ok(md) = e.metadata() else { continue };
                            let name = e.file_name().to_string_lossy().into_owned();
                            let is_dir = md.is_dir();
                            if is_dir && filter & 1 == 0 {
                                continue;
                            }
                            if !is_dir && filter & 2 == 0 {
                                continue;
                            }
                            entries.push((name, is_dir, if is_dir { 0 } else { md.len() }));
                        }
                    }
                }
                let is_switch_path = path_str == "/switch" || path_str == "/switch/";
                if is_switch_path {
                    if let Some(dir) = &kernel.homebrew_dir {
                        let mut seen: std::collections::HashSet<String> =
                            entries.iter().map(|(n, _, _)| n.clone()).collect();
                        if let Ok(rd) = std::fs::read_dir(dir) {
                            for e in rd.filter_map(|e| e.ok()) {
                                let Ok(md) = e.metadata() else { continue };
                                let name = e.file_name().to_string_lossy().into_owned();
                                if seen.contains(&name) {
                                    continue;
                                }
                                let is_dir = md.is_dir();
                                if is_dir && filter & 1 == 0 {
                                    continue;
                                }
                                if !is_dir && filter & 2 == 0 {
                                    continue;
                                }
                                entries.push((
                                    name.clone(),
                                    is_dir,
                                    if is_dir { 0 } else { md.len() },
                                ));
                                seen.insert(name);
                            }
                        }
                    }
                }
                log::debug!(
                    "IFileSystem.OpenDirectory path={:?} filter={:#x} â†’ {} entries",
                    path_str,
                    filter,
                    entries.len()
                );
                fs_trace_path(
                    "OpenDirectory",
                    &path_str,
                    &format!("filter={:#x} entries={}", filter, entries.len()),
                );
                kernel
                    .open_dir_lists
                    .insert((session_handle, new_obj_id), (entries, 0));
                kernel.dir_cursor.insert(session_handle, 0);
                return return_subsession(kernel, ctx, session_handle, "IDirectory");
            }
            10 => return build_ipc_response(ctx, 0, &[], &[]),
            11 | 12 => {
                let huge: u64 = 64u64 * 1024 * 1024 * 1024;
                log::debug!(
                    "IFileSystem.Get{}SpaceSize â†’ {}",
                    if cmd_id == 11 { "Free" } else { "Total" },
                    huge
                );
                return build_ipc_response(ctx, 0, &huge.to_le_bytes(), &[]);
            }
            14 => {
                log::debug!("IFileSystem.GetFileTimeStampRaw â†’ zeros");
                return build_ipc_response(ctx, 0, &[0u8; 0x20], &[]);
            }
            _ => {
                log::warn!("IFileSystem.cmd_{} UNHANDLED â†’ empty SUCCESS", cmd_id);
            }
        }
    }

    if port_name == "IFile" {
        let obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let object_keys = domain_object_keys(kernel, session_handle, obj_id);
        let per_session = object_keys
            .iter()
            .find_map(|key| kernel.open_files.get(key).cloned());
        match cmd_id {
            0 => {
                let in_off = ctx.cmif_in_data_off;
                let avail = ctx.buf.len().saturating_sub(in_off);
                if avail < 24 {
                    log::warn!("IFile.Read: short input ({} bytes)", avail);
                    return build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[]);
                }
                let offset = i64::from_le_bytes([
                    ctx.buf[in_off + 8],
                    ctx.buf[in_off + 9],
                    ctx.buf[in_off + 10],
                    ctx.buf[in_off + 11],
                    ctx.buf[in_off + 12],
                    ctx.buf[in_off + 13],
                    ctx.buf[in_off + 14],
                    ctx.buf[in_off + 15],
                ]);
                let read_size = u64::from_le_bytes([
                    ctx.buf[in_off + 16],
                    ctx.buf[in_off + 17],
                    ctx.buf[in_off + 18],
                    ctx.buf[in_off + 19],
                    ctx.buf[in_off + 20],
                    ctx.buf[in_off + 21],
                    ctx.buf[in_off + 22],
                    ctx.buf[in_off + 23],
                ]);
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                let romfs_file = object_keys
                    .iter()
                    .find_map(|key| kernel.open_romfs_files.get(key).copied());
                let romfs_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_romfs_file_paths.get(key).cloned());
                let target = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let mut bytes_read: u64 = 0;
                if let Some(buf) = target {
                    if let Some(host) = host_path {
                        let want = (read_size as usize).min(buf.size as usize);
                        let mmap = if let Some(m) = kernel.host_file_cache.get(&host) {
                            Some(m.clone())
                        } else {
                            match std::fs::File::open(&host)
                                .and_then(|f| unsafe { memmap2::Mmap::map(&f) })
                            {
                                Ok(m) => {
                                    let a = std::sync::Arc::new(m);
                                    kernel.host_file_cache.insert(host.clone(), a.clone());
                                    Some(a)
                                }
                                Err(_) => None,
                            }
                        };
                        if let Some(m) = mmap {
                            let start = (offset.max(0) as usize).min(m.len());
                            let end = start.saturating_add(want).min(m.len());
                            let slice = &m[start..end];
                            if let Err(err) = kernel.address_space.write(buf.addr, slice) {
                                log::error!(
                                    "IFile.Read: guest write addr={:#x} len={:#x} failed: {}",
                                    buf.addr,
                                    slice.len(),
                                    err
                                );
                            }
                            bytes_read = slice.len() as u64;
                        }
                        log::debug!(
                            "IFile.Read (host {}) off={:#x} size={:#x} â†’ {} bytes",
                            host.display(),
                            offset,
                            read_size,
                            bytes_read
                        );
                    } else if let Some((base, size)) = romfs_file {
                        let romfs = kernel.nro_romfs();
                        let off = offset.max(0) as usize;
                        let start = base.saturating_add(off).min(romfs.len());
                        let remaining = size.saturating_sub(off);
                        let want = (read_size as usize).min(buf.size as usize).min(remaining);
                        let end = start.saturating_add(want).min(romfs.len());
                        let slice = &romfs[start..end];
                        if let Err(err) = kernel.address_space.write(buf.addr, slice) {
                            log::error!(
                                "IFile.Read: guest write addr={:#x} len={:#x} failed: {}",
                                buf.addr,
                                slice.len(),
                                err
                            );
                        }
                        bytes_read = slice.len() as u64;
                        log::debug!(
                            "IFile.Read (romfs off={:#x}) read_off={:#x} size={:#x} â†’ {} bytes",
                            base,
                            offset,
                            read_size,
                            bytes_read
                        );
                        fs_trace_read(
                            "IFile.Read",
                            romfs_path.as_deref().unwrap_or("<romfs-file>"),
                            base.saturating_add(off),
                            offset,
                            read_size,
                            bytes_read,
                        );
                    } else {
                        let file_bytes: &[u8] = match per_session.as_ref() {
                            Some(m) => &m[..],
                            None => kernel.nro_mmap.as_ref().map(|m| &m[..]).unwrap_or(&[]),
                        };
                        let start = (offset.max(0) as usize).min(file_bytes.len());
                        let want = (read_size as usize).min(buf.size as usize);
                        let end = start.saturating_add(want).min(file_bytes.len());
                        let slice = &file_bytes[start..end];
                        if let Err(err) = kernel.address_space.write(buf.addr, slice) {
                            log::error!(
                                "IFile.Read: guest write addr={:#x} len={:#x} failed: {}",
                                buf.addr,
                                slice.len(),
                                err
                            );
                        }
                        bytes_read = slice.len() as u64;
                        log::debug!(
                            "IFile.Read (sess={:#x} obj={}) off={:#x} size={:#x} â†’ {} bytes",
                            session_handle,
                            obj_id,
                            offset,
                            read_size,
                            slice.len(),
                        );
                    }
                } else {
                    log::warn!(
                        "IFile.Read: no recv buffer (off={:#x} size={:#x})",
                        offset,
                        read_size
                    );
                }
                return build_ipc_response(ctx, 0, &bytes_read.to_le_bytes(), &[]);
            }
            1 => {
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                if let Some(host) = host_path {
                    kernel.host_file_cache.remove(&host);
                    use std::io::{Seek, SeekFrom, Write};
                    let in_off = ctx.cmif_in_data_off;
                    if ctx.cmif_in_data_len >= 24 {
                        let offset = i64::from_le_bytes([
                            ctx.buf[in_off + 8],
                            ctx.buf[in_off + 9],
                            ctx.buf[in_off + 10],
                            ctx.buf[in_off + 11],
                            ctx.buf[in_off + 12],
                            ctx.buf[in_off + 13],
                            ctx.buf[in_off + 14],
                            ctx.buf[in_off + 15],
                        ]);
                        let size = u64::from_le_bytes([
                            ctx.buf[in_off + 16],
                            ctx.buf[in_off + 17],
                            ctx.buf[in_off + 18],
                            ctx.buf[in_off + 19],
                            ctx.buf[in_off + 20],
                            ctx.buf[in_off + 21],
                            ctx.buf[in_off + 22],
                            ctx.buf[in_off + 23],
                        ]);
                        if let Some(send_buf) = ctx
                            .send_buffers
                            .iter()
                            .find(|b| b.size > 0 && b.addr != 0)
                            .copied()
                        {
                            let n = (send_buf.size.min(size)) as usize;
                            let mut data = vec![0u8; n];
                            if kernel.address_space.read(send_buf.addr, &mut data).is_ok() {
                                let res = std::fs::OpenOptions::new()
                                    .write(true)
                                    .create(true)
                                    .open(&host)
                                    .and_then(|mut f| {
                                        f.seek(SeekFrom::Start(offset.max(0) as u64))?;
                                        f.write_all(&data)
                                    });
                                if res.is_ok() {
                                    for key in &object_keys {
                                        kernel.open_file_handles.remove(key);
                                    }
                                    kernel.host_file_cache.remove(&host);
                                    log::debug!(
                                        "IFile.Write (host {}) off={:#x} size={} â†’ SUCCESS",
                                        host.display(),
                                        offset,
                                        n
                                    );
                                    return build_ipc_response(ctx, 0, &[], &[]);
                                }
                            }
                        }
                    }
                    return build_ipc_response(ctx, 0x2EE602, &[], &[]);
                }
                log::debug!("IFile.Write (read-only mmap) â†’ SUCCESS discarded");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            2 => return build_ipc_response(ctx, 0, &[], &[]),
            3 => {
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                if let Some(host) = host_path {
                    kernel.host_file_cache.remove(&host);
                    let in_off = ctx.cmif_in_data_off;
                    if ctx.cmif_in_data_len >= 8 {
                        let new_size = u64::from_le_bytes([
                            ctx.buf[in_off],
                            ctx.buf[in_off + 1],
                            ctx.buf[in_off + 2],
                            ctx.buf[in_off + 3],
                            ctx.buf[in_off + 4],
                            ctx.buf[in_off + 5],
                            ctx.buf[in_off + 6],
                            ctx.buf[in_off + 7],
                        ]);
                        let res = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&host)
                            .and_then(|f| f.set_len(new_size));
                        if res.is_ok() {
                            kernel.host_file_cache.remove(&host);
                            return build_ipc_response(ctx, 0, &[], &[]);
                        }
                    }
                    return build_ipc_response(ctx, 0x2EE602, &[], &[]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let size: i64 = if let Some(host) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key))
                {
                    std::fs::metadata(host).map(|m| m.len() as i64).unwrap_or(0)
                } else if let Some((_, size)) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_romfs_files.get(key).copied())
                {
                    size as i64
                } else {
                    match per_session.as_ref() {
                        Some(m) => m.len() as i64,
                        None => kernel
                            .nro_mmap
                            .as_ref()
                            .map(|m| m.len() as i64)
                            .unwrap_or(0),
                    }
                };
                log::debug!(
                    "IFile.GetSize (sess={:#x} obj={}) â†’ {}",
                    session_handle,
                    obj_id,
                    size
                );
                return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
            }
            _ => {
                log::warn!("IFile.cmd_{} UNHANDLED â†’ empty SUCCESS", cmd_id);
            }
        }
    }

    if port_name == "IDirectory" {
        let obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let object_keys = domain_object_keys(kernel, session_handle, obj_id);
        match cmd_id {
            0 => {
                let target = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let buf = match target {
                    Some(b) => b,
                    None => {
                        log::warn!("IDirectory.Read: no recv buffer");
                        return build_ipc_response(ctx, 0, &0i64.to_le_bytes(), &[]);
                    }
                };
                let max_entries = (buf.size as usize) / 0x310;

                let dir_key = object_keys
                    .iter()
                    .copied()
                    .find(|key| kernel.open_dir_lists.contains_key(key));
                if let Some((entries, cursor)) =
                    dir_key.and_then(|key| kernel.open_dir_lists.get_mut(&key))
                {
                    let remaining = entries.len().saturating_sub(*cursor);
                    let to_emit = remaining.min(max_entries);
                    let mut payload = vec![0u8; to_emit * 0x310];
                    for (i, (name, is_dir, size)) in
                        entries.iter().skip(*cursor).take(to_emit).enumerate()
                    {
                        let base = i * 0x310;
                        let name_bytes = name.as_bytes();
                        let name_len = name_bytes.len().min(0x300);
                        payload[base..base + name_len].copy_from_slice(&name_bytes[..name_len]);
                        payload[base + 0x304] = if *is_dir { 0 } else { 1 };
                        payload[base + 0x308..base + 0x310].copy_from_slice(&size.to_le_bytes());
                    }
                    *cursor += to_emit;
                    if !payload.is_empty() {
                        let _ = kernel.address_space.write(buf.addr, &payload);
                    }
                    log::debug!(
                        "IDirectory.Read (host) â†’ {} of {} entries",
                        to_emit,
                        entries.len()
                    );
                    return build_ipc_response(ctx, 0, &(to_emit as i64).to_le_bytes(), &[]);
                }

                let cursor = *kernel.dir_cursor.get(&session_handle).unwrap_or(&0);
                let entries = enumerate_homebrew_nros(&kernel.homebrew_dir);
                let remaining = entries.len().saturating_sub(cursor);
                let to_emit = remaining.min(max_entries);
                let mut payload = vec![0u8; to_emit * 0x310];
                for (i, e) in entries.iter().skip(cursor).take(to_emit).enumerate() {
                    let base = i * 0x310;
                    let name_bytes = e.name.as_bytes();
                    let name_len = name_bytes.len().min(0x300);
                    payload[base..base + name_len].copy_from_slice(&name_bytes[..name_len]);
                    payload[base + 0x301 + 3] = 1;
                    payload[base + 0x308..base + 0x310].copy_from_slice(&e.size.to_le_bytes());
                }
                if !payload.is_empty() {
                    let _ = kernel.address_space.write(buf.addr, &payload);
                }
                kernel.dir_cursor.insert(session_handle, cursor + to_emit);
                log::debug!(
                    "IDirectory.Read (homebrew_dir fallback) cursor={} â†’ {} of {}",
                    cursor,
                    to_emit,
                    entries.len()
                );
                return build_ipc_response(ctx, 0, &(to_emit as i64).to_le_bytes(), &[]);
            }
            1 => {
                let count: i64 = if let Some((entries, _)) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_dir_lists.get(key))
                {
                    entries.len() as i64
                } else {
                    enumerate_homebrew_nros(&kernel.homebrew_dir).len() as i64
                };
                log::debug!("IDirectory.GetEntryCount â†’ {}", count);
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            _ => {
                log::warn!("IDirectory.cmd_{} UNHANDLED â†’ empty SUCCESS", cmd_id);
            }
        }
    }

    let system_data_title_id = port_name
        .strip_prefix("IFsStorageSystemData:")
        .and_then(|value| u64::from_str_radix(value, 16).ok());
    if port_name == "IFsStorage" || system_data_title_id.is_some() {
        let storage = if let Some(title_id) = system_data_title_id {
            kernel
                .system_romfs(title_id)
                .unwrap_or_else(|| match title_id {
                    0x0100_0000_0000_0802 => mii_model_romfs(),
                    0x0100_0000_0000_0823 => ng_word2_romfs(),
                    _ => &[],
                })
        } else {
            kernel.nro_romfs()
        };
        match cmd_id {
            0 => {
                let off_lo = ctx.cmif_in_data_off;
                let read_in = &ctx.buf[off_lo..off_lo + 16];
                let offset = i64::from_le_bytes([
                    read_in[0], read_in[1], read_in[2], read_in[3], read_in[4], read_in[5],
                    read_in[6], read_in[7],
                ]);
                let read_size = u64::from_le_bytes([
                    read_in[8],
                    read_in[9],
                    read_in[10],
                    read_in[11],
                    read_in[12],
                    read_in[13],
                    read_in[14],
                    read_in[15],
                ]);
                let target = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(buf) = target {
                    let virtual_len = if system_data_title_id.is_none() {
                        kernel
                            .application_romfs
                            .as_ref()
                            .map(|r| r.len() as usize)
                            .unwrap_or(storage.len())
                    } else {
                        storage.len()
                    };
                    let start = (offset.max(0) as usize).min(virtual_len);
                    let want = (read_size as usize).min(buf.size as usize);
                    let end = start.saturating_add(want).min(virtual_len);
                    let owned;
                    let slice = if let Some(romfs) = kernel
                        .application_romfs
                        .as_ref()
                        .filter(|_| system_data_title_id.is_none())
                    {
                        match romfs.read(start as u64, end - start) {
                            Ok(bytes) => {
                                owned = bytes;
                                &owned[..]
                            }
                            Err(err) => {
                                log::error!("IFsStorage.Read compressed storage failed: {}", err);
                                return build_ipc_response(ctx, 0xD401, &[], &[]);
                            }
                        }
                    } else {
                        &storage[start..end]
                    };
                    if let Err(err) = kernel.address_space.write_checked(buf.addr, slice) {
                        log::error!(
                            "IFsStorage.Read: guest write addr={:#x} len={:#x} failed: {}",
                            buf.addr,
                            slice.len(),
                            err
                        );
                        return build_ipc_response(ctx, 0xD401, &[], &[]);
                    }
                    log::debug!(
                        "IFsStorage.Read off={:#x} size={:#x} bytes={} total={}",
                        offset,
                        read_size,
                        slice.len(),
                        virtual_len
                    );
                    let path = if fs_trace_enabled() {
                        romfs_path_for_data_offset(storage, start)
                            .map(|(path, file_off, _)| {
                                let rel = start.saturating_sub(file_off);
                                format!("{}+{:#x}", path, rel)
                            })
                            .unwrap_or_else(|| "<romfs-meta>".to_string())
                    } else {
                        String::new()
                    };
                    fs_trace_read(
                        "IFsStorage.Read",
                        &path,
                        start,
                        offset,
                        read_size,
                        slice.len() as u64,
                    );
                } else {
                    log::warn!(
                        "IFsStorage.Read: no recv buffer (off={:#x} size={:#x})",
                        offset,
                        read_size
                    );
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let size = if system_data_title_id.is_none() {
                    kernel
                        .application_romfs
                        .as_ref()
                        .map(|r| r.len() as i64)
                        .unwrap_or(storage.len() as i64)
                } else {
                    storage.len() as i64
                };
                log::debug!("IFsStorage.GetSize â†’ {}", size);
                return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
            }
            _ => {}
        }
    }

    if (port_name == "audren:u" || port_name == "audren:a") && cmd_id == 0 {
        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let read_u32 = |o: usize| -> u32 {
            if in_avail >= o + 4 {
                u32::from_le_bytes([
                    ctx.buf[in_off + o],
                    ctx.buf[in_off + o + 1],
                    ctx.buf[in_off + o + 2],
                    ctx.buf[in_off + o + 3],
                ])
            } else {
                0
            }
        };
        let sample_rate = {
            let v = read_u32(0);
            if v == 0 {
                48000
            } else {
                v
            }
        };
        let sample_count = {
            let v = read_u32(4);
            if v == 0 {
                240
            } else {
                v
            }
        };
        let mix_buffer_count = read_u32(8);
        let voice_count = read_u32(0x10);
        let sink_count = read_u32(0x14);
        let effect_count = read_u32(0x18);
        let revision = read_u32(0x30);
        let revision_num = if audren_behavior::check_valid_revision(revision) {
            audren_behavior::get_revision_num(revision)
        } else {
            log::warn!(
                "audren:u OpenAudioRenderer unsupported revision {:#x} (decodes to {}); clamping to REV{}",
                revision,
                audren_behavior::get_revision_num(revision),
                audren_behavior::CURRENT_REVISION
            );
            audren_behavior::CURRENT_REVISION
        };

        let state = AudioRendererState {
            sample_rate,
            sample_count,
            mix_buffer_count,
            voice_count,
            sink_count,
            effect_count,
            revision,
            revision_num,
            state: 1,
            rendering_time_limit: 100,
            voice_drop_param: 1.0,
            voice_played_samples: Vec::new(),
            voice_wbufs_consumed: Vec::new(),
            voice_last_wb_index: Vec::new(),
            voice_wb_progress_frames: Vec::new(),
            voice_frac_q15: Vec::new(),
            voice_prev_gain: Vec::new(),
            voice_hist: Vec::new(),
            voice_adpcm_states: Vec::new(),
        };

        let is_domain = kernel
            .sessions
            .get(&session_handle)
            .map(|s| s.is_domain)
            .unwrap_or(false);
        log::debug!(
            "audren:u OpenAudioRenderer sr={} samples={} voices={} sinks={} effects={} rev={:#x} REV{} â†’ IAudioRenderer (domain={})",
            sample_rate,
            sample_count,
            voice_count,
            sink_count,
            effect_count,
            revision,
            revision_num,
            is_domain
        );

        if is_domain {
            let object_id = alloc_domain_object(kernel, session_handle, "IAudioRenderer");
            kernel
                .audio_renderers
                .insert((session_handle, object_id), state);
            return build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id]);
        } else {
            let h = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(h, "IAudioRenderer".to_string());
            kernel.sessions.insert(h, session);
            kernel.audio_renderers.insert((h, 0), state);
            return build_ipc_response(ctx, 0, &[], &[h]);
        }
    }
    if (port_name == "audren:u" || port_name == "audren:a") && cmd_id == 1 {
        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let rd = |o: usize| -> u64 {
            if in_avail >= o + 4 {
                u32::from_le_bytes([
                    ctx.buf[in_off + o],
                    ctx.buf[in_off + o + 1],
                    ctx.buf[in_off + o + 2],
                    ctx.buf[in_off + o + 3],
                ]) as u64
            } else {
                0
            }
        };
        let align_up = |v: u64, a: u64| (v + a - 1) & !(a - 1);
        let sample_count = {
            let v = rd(4);
            if v == 0 {
                240
            } else {
                v
            }
        };
        let mixes = rd(8);
        let sub_mixes = rd(0xC);
        let voices = rd(0x10);
        let sinks = rd(0x14);
        let effects = rd(0x18);
        const TARGET: u64 = 240;
        const MAXCH: u64 = 6;
        let mut size: u64 = 0x4000;
        size += (sub_mixes + 1) * 0xC00;
        size += voices * 0x1400;
        size += effects * 0x400;
        size += align_up(
            ((sinks + sub_mixes) * TARGET + sample_count) * 4 * (mixes + MAXCH),
            0x40,
        );
        size += (sinks + sub_mixes) * 0xC00;
        size += 0x40000;
        let computed = align_up(size, 0x1000);
        let work_buffer_size = std::env::var("NEXIUM_AUDIO_WORKBUF")
            .ok()
            .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| computed.clamp(0x20_0000, 0x80_0000));
        log::debug!(
            "audren GetWorkBufferSize voices={} effects={} mixes={} â†’ {:#x} (computed {:#x})",
            voices,
            effects,
            mixes,
            work_buffer_size,
            computed
        );
        return build_ipc_response(ctx, 0, &work_buffer_size.to_le_bytes(), &[]);
    }
    if (port_name == "audren:u" || port_name == "audren:a") && (cmd_id == 2 || cmd_id == 4) {
        return return_subsession(kernel, ctx, session_handle, "IAudioDevice");
    }

    if port_name == "IAudioRenderer" {
        let obj_id = ctx.domain.as_ref().map(|d| d.object_id).unwrap_or(0);
        let key = (session_handle, obj_id);
        let st = kernel
            .audio_renderers
            .entry(key)
            .or_insert(AudioRendererState {
                sample_rate: 48000,
                sample_count: 240,
                mix_buffer_count: 0,
                voice_count: 0,
                sink_count: 0,
                effect_count: 0,
                revision: audren_behavior::encode_revision(audren_behavior::CURRENT_REVISION),
                revision_num: audren_behavior::CURRENT_REVISION,
                state: 1,
                rendering_time_limit: 100,
                voice_drop_param: 1.0,
                voice_played_samples: Vec::new(),
                voice_wbufs_consumed: Vec::new(),
                voice_last_wb_index: Vec::new(),
                voice_wb_progress_frames: Vec::new(),
                voice_frac_q15: Vec::new(),
                voice_prev_gain: Vec::new(),
                voice_hist: Vec::new(),
                voice_adpcm_states: Vec::new(),
            });
        match cmd_id {
            0 => {
                let v = st.sample_rate;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            1 => {
                let v = st.sample_count;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            2 => {
                let v = st.mix_buffer_count;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            3 => {
                let v = st.state;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            4 | 10 => {
                let usable = |buffer: Option<&ipc::IpcBuffer>| {
                    buffer
                        .filter(|buffer| buffer.size > 0 && buffer.addr != 0)
                        .copied()
                };
                let in_buf =
                    usable(ctx.send_buffers.first()).or_else(|| usable(ctx.send_statics.first()));
                let (out_buf, perf_buf) =
                    audio_renderer_output_slots(cmd_id, &ctx.recv_buffers, &ctx.recv_statics);
                let revision = st.revision;
                let revision_num = st.revision_num;
                let wave_buffer_ver2 = audren_behavior::check_feature_supported(
                    audren_behavior::SupportTags::WaveBufferVer2,
                    revision_num,
                );
                let voice_drop_param = st.voice_drop_param;
                let frame = kernel.audio_renderer_frame_counter;

                let mut in_behavior_sz: u64 = 0;
                let mut in_mempools_sz: u64 = 0;
                let mut in_voices_sz: u64 = 0;
                let mut in_channels_sz: u64 = 0;
                let mut in_effects_sz: u64 = 0;
                let mut in_mixes_sz: u64 = 0;
                let mut in_sinks_sz: u64 = 0;
                let mut in_perf_sz: u64 = 0;
                let mut in_behavior_param: Option<audren_behavior::InParameter> = None;
                let mut mempool_in_states: Vec<u32> = Vec::new();
                if let Some(ib) = in_buf {
                    let mut hdr = [0u8; 0x40];
                    if (ib.size as usize) >= 0x40
                        && kernel.address_space.read(ib.addr, &mut hdr).is_ok()
                    {
                        let rd = |off: usize| {
                            u32::from_le_bytes([hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]])
                                as u64
                        };
                        in_behavior_sz = rd(0x04);
                        in_mempools_sz = rd(0x08);
                        in_voices_sz = rd(0x0C);
                        in_channels_sz = rd(0x10);
                        in_effects_sz = rd(0x14);
                        in_mixes_sz = rd(0x18);
                        in_sinks_sz = rd(0x1C);
                        in_perf_sz = rd(0x20);
                    }
                    if in_behavior_sz as usize >= audren_behavior::IN_PARAMETER_SIZE {
                        let mut block = [0u8; audren_behavior::IN_PARAMETER_SIZE];
                        if kernel
                            .address_space
                            .read(ib.addr.wrapping_add(0x40), &mut block)
                            .is_ok()
                        {
                            in_behavior_param = audren_behavior::InParameter::parse(&block);
                        }
                    }
                    if in_mempools_sz > 0 {
                        let mempool_count = (in_mempools_sz / 0x20) as usize;
                        let mempools_off = 0x40u64 + in_behavior_sz;
                        mempool_in_states.reserve(mempool_count);
                        for i in 0..mempool_count {
                            let off = mempools_off + (i as u64) * 0x20 + 0x10;
                            let mut sb = [0u8; 4];
                            if kernel
                                .address_space
                                .read(ib.addr.wrapping_add(off), &mut sb)
                                .is_ok()
                            {
                                mempool_in_states.push(u32::from_le_bytes(sb));
                            } else {
                                mempool_in_states.push(0);
                            }
                        }
                    }
                }
                let mempool_count = mempool_in_states.len();
                let voice_count_seen = (in_voices_sz / 0x170) as usize;
                let effect_count_seen = (in_effects_sz / 0xC0) as usize;
                if audio_debug_enabled() {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static LOGGED_LAYOUT: AtomicBool = AtomicBool::new(false);
                    if !LOGGED_LAYOUT.swap(true, Ordering::Relaxed) {
                        log::info!(
                            "[audio-debug] input={:#x} behavior={:#x} mempools={:#x} voices={:#x} channels={:#x} effects={:#x} sinks={:#x} parsed_voices={}",
                            in_buf.map(|b| b.size).unwrap_or(0),
                            in_behavior_sz,
                            in_mempools_sz,
                            in_voices_sz,
                            in_channels_sz,
                            in_effects_sz,
                            in_sinks_sz,
                            voice_count_seen
                        );
                    }
                }
                let mut effect_out_states: Vec<u8> = vec![4; effect_count_seen];
                if let Some(ib) = in_buf {
                    let effects_in_off =
                        0x40u64 + in_behavior_sz + in_mempools_sz + in_channels_sz + in_voices_sz;
                    for i in 0..effect_count_seen {
                        let off = effects_in_off + (i as u64) * 0xC0;
                        let mut eb = [0u8; 3];
                        if kernel
                            .address_space
                            .read(ib.addr.wrapping_add(off), &mut eb)
                            .is_ok()
                        {
                            let ty = eb[0];
                            let is_new = eb[1] != 0;
                            let enabled = eb[2] != 0;
                            effect_out_states[i] =
                                if ty != 0 && (st.state == 0 || is_new || enabled) {
                                    3
                                } else {
                                    4
                                };
                        }
                    }
                }
                if st.voice_played_samples.len() < voice_count_seen {
                    st.voice_played_samples.resize(voice_count_seen, 0);
                    st.voice_wbufs_consumed.resize(voice_count_seen, 0);
                    st.voice_last_wb_index.resize(voice_count_seen, 0);
                    st.voice_wb_progress_frames.resize(voice_count_seen, 0);
                    st.voice_frac_q15.resize(voice_count_seen, 0);
                    st.voice_prev_gain.resize(voice_count_seen, 0.0);
                    st.voice_hist.resize(voice_count_seen, [0.0f32; 6]);
                    st.voice_adpcm_states
                        .resize(voice_count_seen, AudioAdpcmDecodeState::default());
                }

                const TARGET_FRAMES: usize = AUDIO_RENDER_BLOCK_FRAMES;
                const TARGET_SR: f32 = 48_000.0;

                let queued_now = crate::audio_sink::host_audio_sink()
                    .map(|s| s.queued_frames())
                    .unwrap_or(0);

                let blocks_to_produce: usize = audio_blocks_to_produce(queued_now);
                let mut is_new_latched: Vec<bool> = vec![false; voice_count_seen];
                let mut big_out: Vec<f32> =
                    Vec::with_capacity(TARGET_FRAMES * 2 * blocks_to_produce);

                if blocks_to_produce == 0 {
                    if let Some(ib) = in_buf {
                        let voices_off: u64 =
                            0x40 + in_behavior_sz + in_mempools_sz + in_channels_sz;
                        let voice_info_stride: u64 = 0x170;
                        for vid in 0..voice_count_seen {
                            let vinfo_off = voices_off + (vid as u64) * voice_info_stride;
                            if vinfo_off + 0x42 > ib.size as u64 {
                                st.voice_adpcm_states[vid..].fill(AudioAdpcmDecodeState::default());
                                break;
                            }
                            let mut metadata = [0u8; 0x42];
                            if kernel
                                .address_space
                                .read(ib.addr.wrapping_add(vinfo_off), &mut metadata)
                                .is_err()
                            {
                                st.voice_adpcm_states[vid] = AudioAdpcmDecodeState::default();
                                continue;
                            }
                            let is_new = metadata[0x008] != 0;
                            let is_in_use = metadata[0x009] != 0;
                            let play_state = metadata[0x00A];
                            let sample_format = metadata[0x00B];
                            let sample_rate = u32::from_le_bytes([
                                metadata[0x00C],
                                metadata[0x00D],
                                metadata[0x00E],
                                metadata[0x00F],
                            ]);
                            let channel_count = u32::from_le_bytes([
                                metadata[0x018],
                                metadata[0x019],
                                metadata[0x01A],
                                metadata[0x01B],
                            ]);
                            let wb_count = u32::from_le_bytes([
                                metadata[0x03C],
                                metadata[0x03D],
                                metadata[0x03E],
                                metadata[0x03F],
                            ]);
                            let wb_index =
                                u16::from_le_bytes([metadata[0x040], metadata[0x041]]) as usize;
                            if is_new
                                || !is_in_use
                                || play_state != 0
                                || sample_format != AUDIO_PCM_ADPCM
                                || !(channel_count == 1 || channel_count == 2)
                                || sample_rate == 0
                                || wb_count == 0
                                || wb_index >= 4
                            {
                                st.voice_adpcm_states[vid] = AudioAdpcmDecodeState::default();
                            }
                        }
                    }
                }

                for _block in 0..blocks_to_produce {
                    let mut out_stereo = vec![0.0f32; TARGET_FRAMES * 2];
                    let mut block_consumed_wb = false;
                    let mut voice_snapshot =
                        vec![AudioVoiceMixSnapshot::default(); voice_count_seen];

                    'mix: {
                        let Some(ib) = in_buf else {
                            break 'mix;
                        };
                        if (ib.size as usize) < 0x40 || voice_count_seen == 0 {
                            break 'mix;
                        }
                        let voices_off: u64 =
                            0x40 + in_behavior_sz + in_mempools_sz + in_channels_sz;
                        let voice_info_stride: u64 = 0x170;

                        for vid in 0..voice_count_seen {
                            let vinfo_off = voices_off + (vid as u64) * voice_info_stride;
                            if vinfo_off + voice_info_stride > ib.size as u64 {
                                break;
                            }
                            let v0_addr = ib.addr.wrapping_add(vinfo_off);
                            let mut v = [0u8; 0x170];
                            if kernel.address_space.read(v0_addr, &mut v).is_err() {
                                continue;
                            }

                            let is_new = v[0x008] != 0;
                            let is_in_use = v[0x009] != 0;
                            let play_state = v[0x00A];
                            let sample_format = v[0x00B];
                            let sample_rate =
                                u32::from_le_bytes([v[0x00C], v[0x00D], v[0x00E], v[0x00F]]);
                            let channel_count =
                                u32::from_le_bytes([v[0x018], v[0x019], v[0x01A], v[0x01B]]);
                            let volume =
                                f32::from_le_bytes([v[0x020], v[0x021], v[0x022], v[0x023]]);
                            let wb_count =
                                u32::from_le_bytes([v[0x03C], v[0x03D], v[0x03E], v[0x03F]]);
                            let wb_index = u16::from_le_bytes([v[0x040], v[0x041]]) as usize;
                            voice_snapshot[vid].wb_index = wb_index as u16;
                            voice_snapshot[vid].is_new = is_new;

                            if is_new
                                || !is_in_use
                                || play_state != 0
                                || sample_format != AUDIO_PCM_ADPCM
                                || !(channel_count == 1 || channel_count == 2)
                                || sample_rate == 0
                                || wb_count == 0
                                || wb_index >= 4
                            {
                                st.voice_adpcm_states[vid] = AudioAdpcmDecodeState::default();
                            }

                            if audio_debug_enabled() && is_in_use {
                                use std::sync::atomic::{AtomicU64, Ordering};
                                static LOGGED_VOICES: AtomicU64 = AtomicU64::new(0);
                                let bit = 1u64 << ((vid as u64) & 63);
                                if LOGGED_VOICES.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                                    let first_wb = &v[0x060..0x098];
                                    let wb_addr = u64::from_le_bytes(
                                        first_wb[0x00..0x08].try_into().unwrap(),
                                    );
                                    let wb_size = u64::from_le_bytes(
                                        first_wb[0x08..0x10].try_into().unwrap(),
                                    );
                                    let wb_start = i32::from_le_bytes(
                                        first_wb[0x10..0x14].try_into().unwrap(),
                                    );
                                    let wb_end = i32::from_le_bytes(
                                        first_wb[0x14..0x18].try_into().unwrap(),
                                    );
                                    log::info!(
                                        "[audio-debug] voice[{vid}] new={is_new} used={is_in_use} state={play_state} fmt={sample_format} sr={sample_rate} ch={channel_count} vol={volume:.3} pitch={:.3} wb_count={wb_count} wb_index={wb_index} wb0=({wb_addr:#x},{wb_size:#x},{wb_start}..{wb_end},loop={},sent={})",
                                        f32::from_le_bytes(v[0x01C..0x020].try_into().unwrap()),
                                        first_wb[0x18] != 0,
                                        first_wb[0x1A] != 0,
                                    );
                                }
                            }

                            if is_in_use {
                                use std::sync::atomic::{AtomicU64, Ordering as O};
                                static SEEN_MASK: AtomicU64 = AtomicU64::new(0);
                                let bit = 1u64 << ((vid as u64) & 63);
                                let prev = SEEN_MASK.fetch_or(bit, O::Relaxed);
                                if prev & bit == 0 {
                                    let fmt_name = match sample_format {
                                        0 => "Invalid",
                                        1 => "PcmInt8",
                                        2 => "PcmInt16",
                                        3 => "PcmInt24",
                                        4 => "PcmInt32",
                                        5 => "PcmFloat",
                                        6 => "Adpcm",
                                        _ => "?",
                                    };
                                    log::trace!(
                                        "voice[{}] FIRST SEEN: fmt={}({}) ch={} sr={} vol={:.2} state={} wb_count={} wb_index={}",
                                        vid,
                                        fmt_name,
                                        sample_format,
                                        channel_count,
                                        sample_rate,
                                        volume,
                                        play_state,
                                        wb_count,
                                        wb_index
                                    );
                                }
                            }

                            if is_new {
                                if let Some(g) = st.voice_prev_gain.get_mut(vid) {
                                    *g = 0.0;
                                }
                            }

                            if !is_in_use
                                || play_state != 0
                                || (sample_format != AUDIO_PCM_INT16
                                    && sample_format != AUDIO_PCM_FLOAT
                                    && sample_format != AUDIO_PCM_ADPCM)
                                || !(channel_count == 1 || channel_count == 2)
                                || sample_rate == 0
                                || wb_count == 0
                                || wb_index >= 4
                            {
                                if let Some(g) = st.voice_prev_gain.get_mut(vid) {
                                    *g = 0.0;
                                }
                                continue;
                            }

                            let wb_base = 0x060 + wb_index * 0x38;
                            let wb = &v[wb_base..wb_base + 0x38];
                            let buffer_address = u64::from_le_bytes([
                                wb[0x00], wb[0x01], wb[0x02], wb[0x03], wb[0x04], wb[0x05],
                                wb[0x06], wb[0x07],
                            ]);
                            let buffer_size = u64::from_le_bytes([
                                wb[0x08], wb[0x09], wb[0x0A], wb[0x0B], wb[0x0C], wb[0x0D],
                                wb[0x0E], wb[0x0F],
                            ]);
                            let start_offset =
                                i32::from_le_bytes([wb[0x10], wb[0x11], wb[0x12], wb[0x13]]);
                            let end_offset =
                                i32::from_le_bytes([wb[0x14], wb[0x15], wb[0x16], wb[0x17]]);
                            if buffer_address == 0 || start_offset < 0 || end_offset <= start_offset
                            {
                                continue;
                            }

                            let mut wave_buffers = [AudioWaveBufferSpan::default(); 4];
                            for (queued_index, span) in wave_buffers
                                .iter_mut()
                                .enumerate()
                                .take((wb_count as usize).min(4))
                            {
                                let slot = (wb_index + queued_index) % 4;
                                let queued_base = 0x060 + slot * 0x38;
                                let queued = &v[queued_base..queued_base + 0x38];
                                let queued_address =
                                    u64::from_le_bytes(queued[0x00..0x08].try_into().unwrap());
                                let queued_start =
                                    i32::from_le_bytes(queued[0x10..0x14].try_into().unwrap());
                                let queued_end =
                                    i32::from_le_bytes(queued[0x14..0x18].try_into().unwrap());
                                if queued_address == 0
                                    || queued_start < 0
                                    || queued_end <= queued_start
                                {
                                    break;
                                }
                                span.frames = (queued_end - queued_start) as u32;
                                span.looping = queued[0x18] != 0;
                                span.loop_count = if wave_buffer_ver2 {
                                    i32::from_le_bytes(queued[0x1C..0x20].try_into().unwrap())
                                } else {
                                    AUDIO_WAVE_BUFFER_LOOP_INFINITE
                                };
                            }

                            let ch = channel_count as usize;
                            let ratio = sample_rate as f32 / TARGET_SR;
                            let in_frames_needed =
                                ((TARGET_FRAMES as f32) * ratio).ceil() as usize + 3;
                            let wb_total_frames = (end_offset - start_offset) as usize;
                            let cursor =
                                (st.voice_wb_progress_frames.get(vid).copied().unwrap_or(0)
                                    as usize)
                                    .min(wb_total_frames.saturating_sub(1));
                            let in_frames = in_frames_needed;
                            let initial_frac_q15 = st.voice_frac_q15.get(vid).copied().unwrap_or(0);
                            let step: i32 = ((sample_rate as f32 / TARGET_SR) * 32768.0) as i32;
                            let (checkpoint_source_frames, _) =
                                audio_source_advance(initial_frac_q15, step, TARGET_FRAMES);
                            let mut pcm_l = vec![0.0f32; in_frames];
                            let mut pcm_r = vec![0.0f32; in_frames];

                            if sample_format == AUDIO_PCM_ADPCM {
                                let previous_state = st.voice_adpcm_states[vid];
                                st.voice_adpcm_states[vid] = AudioAdpcmDecodeState::default();
                                let coeff_addr = u64::from_le_bytes([
                                    v[0x048], v[0x049], v[0x04A], v[0x04B], v[0x04C], v[0x04D],
                                    v[0x04E], v[0x04F],
                                ]);
                                let ctx_addr = u64::from_le_bytes([
                                    wb[0x20], wb[0x21], wb[0x22], wb[0x23], wb[0x24], wb[0x25],
                                    wb[0x26], wb[0x27],
                                ]);
                                let mut coeff_bytes = [0u8; 32];
                                if coeff_addr == 0
                                    || kernel
                                        .address_space
                                        .read(coeff_addr, &mut coeff_bytes)
                                        .is_err()
                                {
                                    continue;
                                }
                                let mut coeffs = [0i16; 16];
                                for i in 0..16 {
                                    coeffs[i] = i16::from_le_bytes([
                                        coeff_bytes[i * 2],
                                        coeff_bytes[i * 2 + 1],
                                    ]);
                                }
                                let mut initial_header = 0u16;
                                let (mut yn0_seed, mut yn1_seed) = (0i16, 0i16);
                                if ctx_addr != 0 {
                                    let mut ctx = [0u8; 6];
                                    if kernel.address_space.read(ctx_addr, &mut ctx).is_ok() {
                                        initial_header = u16::from_le_bytes([ctx[0], ctx[1]]);
                                        yn0_seed = i16::from_le_bytes([ctx[2], ctx[3]]);
                                        yn1_seed = i16::from_le_bytes([ctx[4], ctx[5]]);
                                    }
                                }
                                if buffer_address == 0 || buffer_size < 8 {
                                    continue;
                                }

                                let base = start_offset as usize + cursor;
                                let stream_key = AudioAdpcmStreamKey {
                                    wb_index: wb_index as u16,
                                    buffer_address,
                                    buffer_size,
                                    start_offset,
                                    end_offset,
                                    context_address: ctx_addr,
                                    coefficient_address: coeff_addr,
                                    sample_rate,
                                    looping: wb[0x18] != 0,
                                    initial_header,
                                    initial_yn0: yn0_seed,
                                    initial_yn1: yn1_seed,
                                    coefficients: coeffs,
                                };
                                let streaming =
                                    can_stream_gc_adpcm(previous_state, stream_key, base, is_new);
                                let (decode_start, decode_count, output_skip, decode_context) =
                                    if streaming {
                                        (base, in_frames, 0, previous_state.context)
                                    } else {
                                        (
                                            0,
                                            base.saturating_add(in_frames),
                                            base,
                                            AudioAdpcmContext {
                                                header: initial_header as u8,
                                                yn0: yn0_seed,
                                                yn1: yn1_seed,
                                            },
                                        )
                                    };
                                let checkpoint_after = if streaming {
                                    checkpoint_source_frames
                                } else {
                                    base.saturating_add(checkpoint_source_frames)
                                };
                                let buffer_size_usize =
                                    usize::try_from(buffer_size).unwrap_or(usize::MAX);
                                let Some((byte_offset, byte_count)) = gc_adpcm_byte_range(
                                    decode_start,
                                    decode_count,
                                    buffer_size_usize,
                                ) else {
                                    continue;
                                };
                                let mut adpcm = vec![0u8; byte_count];
                                let Some(read_address) =
                                    buffer_address.checked_add(byte_offset as u64)
                                else {
                                    continue;
                                };
                                if byte_count != 0
                                    && kernel.address_space.read(read_address, &mut adpcm).is_err()
                                {
                                    continue;
                                }
                                let mut decoded = vec![0i16; in_frames];
                                let decode_result = decode_gc_adpcm_range(
                                    &adpcm,
                                    &coeffs,
                                    decode_context,
                                    decode_start,
                                    decode_count,
                                    output_skip,
                                    &mut decoded,
                                    checkpoint_after,
                                );
                                debug_assert!(decode_result.bytes_read <= adpcm.len());
                                st.voice_adpcm_states[vid] = if let Some(context) =
                                    decode_result.checkpoint
                                {
                                    AudioAdpcmDecodeState {
                                        valid: true,
                                        key: stream_key,
                                        next_sample: base.saturating_add(checkpoint_source_frames)
                                            as u64,
                                        context,
                                    }
                                } else {
                                    AudioAdpcmDecodeState::default()
                                };
                                for f in 0..in_frames {
                                    pcm_l[f] = (decoded[f] as f32) / 32768.0;
                                    pcm_r[f] = pcm_l[f];
                                }
                                {
                                    use std::sync::atomic::{AtomicU64, Ordering as O};
                                    static DUMPED: AtomicU64 = AtomicU64::new(0);
                                    let bit = 1u64 << ((vid as u64) & 63);
                                    if DUMPED.fetch_or(bit, O::Relaxed) & bit == 0 {
                                        let nz = decoded.iter().filter(|s| **s != 0).count();
                                        log::trace!(
                                            "voice[{}] ADPCM: decoded={} nonzero={} coeff_addr={:#x} ctx_addr={:#x} start_off={} in_frames={} sr={} streaming={}",
                                            vid,
                                            decode_result.decoded_samples,
                                            nz,
                                            coeff_addr,
                                            ctx_addr,
                                            start_offset,
                                            in_frames,
                                            sample_rate,
                                            streaming
                                        );
                                    }
                                }
                            } else {
                                let Some(bytes_per_sample) = pcm_bytes_per_sample(sample_format)
                                else {
                                    continue;
                                };
                                let stride = bytes_per_sample * ch;
                                let queued = (wb_count as usize).min(4);
                                let mut got = 0usize;
                                let mut k = 0usize;
                                let mut slot_cursor = cursor;
                                while got < in_frames && k < queued {
                                    let slot = (wb_index + k) % 4;
                                    let sb = 0x060 + slot * 0x38;
                                    let swb = &v[sb..sb + 0x38];
                                    let s_addr = u64::from_le_bytes([
                                        swb[0x00], swb[0x01], swb[0x02], swb[0x03], swb[0x04],
                                        swb[0x05], swb[0x06], swb[0x07],
                                    ]);
                                    let s_size = u64::from_le_bytes([
                                        swb[0x08], swb[0x09], swb[0x0A], swb[0x0B], swb[0x0C],
                                        swb[0x0D], swb[0x0E], swb[0x0F],
                                    ]);
                                    let s_start = i32::from_le_bytes([
                                        swb[0x10], swb[0x11], swb[0x12], swb[0x13],
                                    ]);
                                    let s_end = i32::from_le_bytes([
                                        swb[0x14], swb[0x15], swb[0x16], swb[0x17],
                                    ]);
                                    let s_loop = swb[0x18] != 0;
                                    if s_addr == 0 || s_start < 0 || s_end <= s_start {
                                        break;
                                    }
                                    let s_total = (s_end - s_start) as usize;
                                    if slot_cursor >= s_total {
                                        if s_loop {
                                            slot_cursor = 0;
                                        } else {
                                            k += 1;
                                            slot_cursor = 0;
                                            continue;
                                        }
                                    }
                                    let avail = s_total - slot_cursor;
                                    let want = (in_frames - got).min(avail);
                                    let boff = ((s_start as u64) + slot_cursor as u64)
                                        .wrapping_mul(stride as u64);
                                    if boff.saturating_add((want * stride) as u64) > s_size {
                                        break;
                                    }
                                    let mut buf = vec![0u8; want * stride];
                                    if kernel
                                        .address_space
                                        .read(s_addr.wrapping_add(boff), &mut buf)
                                        .is_err()
                                    {
                                        break;
                                    }
                                    let decoded = decode_pcm_stereo(
                                        sample_format,
                                        ch,
                                        &buf,
                                        &mut pcm_l[got..got + want],
                                        &mut pcm_r[got..got + want],
                                    );
                                    if audio_debug_enabled() && decoded != 0 {
                                        use std::sync::atomic::{AtomicU64, Ordering};
                                        static LOGGED_PCM: AtomicU64 = AtomicU64::new(0);
                                        static LOGGED_NONZERO_PCM: AtomicU64 = AtomicU64::new(0);
                                        let bit = 1u64 << ((vid as u64) & 63);
                                        let peak = pcm_l[got..got + decoded]
                                            .iter()
                                            .chain(&pcm_r[got..got + decoded])
                                            .fold(0.0f32, |acc, sample| acc.max(sample.abs()));
                                        let first =
                                            LOGGED_PCM.fetch_or(bit, Ordering::Relaxed) & bit == 0;
                                        let first_nonzero = peak > 0.001
                                            && LOGGED_NONZERO_PCM.fetch_or(bit, Ordering::Relaxed)
                                                & bit
                                                == 0;
                                        if first || first_nonzero {
                                            log::info!(
                                                "[audio-debug] voice[{vid}] decoded={decoded} fmt={sample_format} bytes={} peak={peak:.6} addr={:#x} offset={:#x}{}",
                                                buf.len(),
                                                s_addr,
                                                boff,
                                                if first_nonzero { " FIRST-NONZERO-SOURCE" } else { "" }
                                            );
                                        }
                                    }
                                    got += decoded;
                                    if decoded != want {
                                        break;
                                    }
                                    if s_loop {
                                        slot_cursor += want;
                                        if slot_cursor >= s_total {
                                            slot_cursor = 0;
                                        }
                                    } else {
                                        k += 1;
                                        slot_cursor = 0;
                                    }
                                }
                                if vid == 0 {
                                    use std::sync::atomic::{AtomicU64, Ordering as O};
                                    static DC: AtomicU64 = AtomicU64::new(0);
                                    let n = DC.fetch_add(1, O::Relaxed);
                                    if n % 256 == 0 {
                                        log::trace!(
                                            "voice[0] chain wb_count={} wb_index={} cursor={} got={} in_frames={}",
                                            wb_count,
                                            wb_index,
                                            cursor,
                                            got,
                                            in_frames
                                        );
                                    }
                                }
                                if got == 0 {
                                    continue;
                                }
                            }

                            let phist = st.voice_hist.get(vid).copied().unwrap_or([0.0f32; 6]);
                            let mut frac_q15 = initial_frac_q15;
                            let master = voice_drop_param.clamp(0.0, 4.0);
                            let gain = volume * master * 0.5;
                            let smp_l = |i: isize| -> f32 {
                                if i < 0 {
                                    phist[0]
                                } else {
                                    pcm_l[(i as usize).min(in_frames - 1)]
                                }
                            };
                            let smp_r = |i: isize| -> f32 {
                                if i < 0 {
                                    phist[3]
                                } else {
                                    pcm_r[(i as usize).min(in_frames - 1)]
                                }
                            };
                            let prev_gain = st.voice_prev_gain.get(vid).copied().unwrap_or(0.0);
                            let gain_ramp = (gain - prev_gain) / TARGET_FRAMES as f32;
                            let mut ramped_gain = prev_gain;
                            let mut read_idx: usize = 0;
                            for i in 0..TARGET_FRAMES {
                                let bi = read_idx as isize;
                                let fraction = frac_q15 as f32 * (1.0 / 32768.0);
                                let left = smp_l(bi);
                                let right = smp_r(bi);
                                let ol = left + (smp_l(bi + 1) - left) * fraction;
                                let orr = right + (smp_r(bi + 1) - right) * fraction;
                                out_stereo[i * 2] += ol * ramped_gain;
                                out_stereo[i * 2 + 1] += orr * ramped_gain;
                                ramped_gain += gain_ramp;
                                let no = frac_q15 + step;
                                read_idx += (no >> 15) as usize;
                                frac_q15 = no & 0x7fff;
                            }
                            if let Some(g) = st.voice_prev_gain.get_mut(vid) {
                                *g = gain;
                            }
                            debug_assert_eq!(read_idx, checkpoint_source_frames);
                            let consumed =
                                read_idx.min(in_frames).saturating_sub(1).min(in_frames - 1);
                            let mut nh = [0.0f32; 6];
                            nh[0] = pcm_l[consumed];
                            nh[3] = pcm_r[consumed];
                            if let Some(h) = st.voice_hist.get_mut(vid) {
                                *h = nh;
                            }
                            if let Some(f) = st.voice_frac_q15.get_mut(vid) {
                                *f = frac_q15;
                            }

                            let src_frames_this_pass = read_idx as u32;
                            voice_snapshot[vid].did_mix = true;
                            voice_snapshot[vid].source_frames = src_frames_this_pass;
                            voice_snapshot[vid].wb_count = wb_count;
                            voice_snapshot[vid].buffers = wave_buffers;
                        }
                    }

                    {
                        use std::sync::atomic::{AtomicBool, Ordering as O};
                        static LOGGED: AtomicBool = AtomicBool::new(false);
                        static CONSUMED_LOGGED: AtomicBool = AtomicBool::new(false);
                        let mut any_mix = false;
                        let mut any_consumed = false;
                        for vid in 0..voice_count_seen {
                            let snapshot = voice_snapshot[vid];
                            let wb_now = snapshot.wb_index;
                            let wb_total = snapshot.buffers[0].frames;
                            if snapshot.did_mix {
                                any_mix = true;
                            }
                            if snapshot.is_new && !is_new_latched[vid] {
                                st.voice_played_samples[vid] = 0;
                                st.voice_wbufs_consumed[vid] = 0;
                                st.voice_last_wb_index[vid] = wb_now;
                                is_new_latched[vid] = true;
                                if let Some(p) = st.voice_wb_progress_frames.get_mut(vid) {
                                    *p = 0;
                                }
                            } else if snapshot.did_mix && wb_total > 0 {
                                st.voice_played_samples[vid] = st.voice_played_samples[vid]
                                    .wrapping_add(snapshot.source_frames as u64);

                                let prev_progress =
                                    st.voice_wb_progress_frames.get(vid).copied().unwrap_or(0);
                                let (new_progress, completed, exhausted) =
                                    advance_audio_wave_buffers(
                                        prev_progress,
                                        snapshot.source_frames as u64,
                                        &snapshot.buffers,
                                    );
                                if exhausted {
                                    use std::sync::atomic::{AtomicBool, Ordering as O2};
                                    static WARN_ONCE: AtomicBool = AtomicBool::new(false);
                                    if !WARN_ONCE.swap(true, O2::Relaxed) {
                                        log::warn!(
                                            "audio voice[{}] consume cap hit: residue={} wb_total={} completed={} cap={} (wb_count={})",
                                            vid,
                                            new_progress,
                                            wb_total,
                                            completed,
                                            snapshot.buffers.iter().filter(|buffer| buffer.frames != 0).count(),
                                            snapshot.wb_count
                                        );
                                    }
                                }

                                if completed > 0 {
                                    st.voice_wbufs_consumed[vid] =
                                        st.voice_wbufs_consumed[vid].wrapping_add(completed);
                                    any_consumed = true;
                                    block_consumed_wb = true;
                                    use std::sync::atomic::{AtomicU64, Ordering as O3};
                                    static PER_VOICE_LOGGED: AtomicU64 = AtomicU64::new(0);
                                    let bit = 1u64 << ((vid as u64) & 63);
                                    let prev_mask = PER_VOICE_LOGGED.fetch_or(bit, O3::Relaxed);
                                    if prev_mask & bit == 0 {
                                        log::trace!(
                                            "voice[{}] FIRST CONSUMED BUMP: wb_index={}, samples_played={} (consumed_now={}, wb_total={}, completed={}, frame {})",
                                            vid,
                                            wb_now,
                                            st.voice_played_samples[vid],
                                            st.voice_wbufs_consumed[vid],
                                            wb_total,
                                            completed,
                                            frame
                                        );
                                    }
                                }
                                if let Some(p) = st.voice_wb_progress_frames.get_mut(vid) {
                                    *p = new_progress;
                                }
                                st.voice_last_wb_index[vid] = wb_now;
                            } else if snapshot.did_mix {
                                st.voice_played_samples[vid] =
                                    st.voice_played_samples[vid].wrapping_add(TARGET_FRAMES as u64);
                                st.voice_last_wb_index[vid] = wb_now;
                            }
                        }
                        if any_mix && !LOGGED.swap(true, O::Relaxed) {
                            let mixed_ids: Vec<usize> = (0..voice_count_seen)
                                .filter(|&i| voice_snapshot[i].did_mix)
                                .collect();
                            log::trace!(
                                "audio multi-voice MIXED first time: voice_count_seen={} mixed_voices={:?} (frame {})",
                                voice_count_seen,
                                mixed_ids,
                                frame
                            );
                        }
                        if any_consumed && !CONSUMED_LOGGED.swap(true, O::Relaxed) {
                            let states: Vec<(usize, u32, u64)> = (0..voice_count_seen)
                                .filter(|&i| voice_snapshot[i].did_mix)
                                .map(|i| {
                                    (i, st.voice_wbufs_consumed[i], st.voice_played_samples[i])
                                })
                                .collect();
                            if audio_debug_enabled() {
                                log::info!(
                                    "[audio-debug] wavebuf FIRST CONSUMED: voices={:?} (frame {})",
                                    states,
                                    frame
                                );
                            } else {
                                log::trace!(
                                    "audio wavebuf FIRST CONSUMED: voices={:?} (frame {})",
                                    states,
                                    frame
                                );
                            }
                        }
                        if audio_debug_enabled() && voice_count_seen != 0 {
                            use std::sync::atomic::{AtomicU64, Ordering};
                            static STATUS_TICK: AtomicU64 = AtomicU64::new(0);
                            let tick = STATUS_TICK.fetch_add(1, Ordering::Relaxed);
                            if tick % 128 == 0 {
                                let snapshot = voice_snapshot[0];
                                log::info!(
                                    "[audio-debug] status tick={tick} frame={frame} voice0 index={} new={} mixed={} src_frames={} wb_frames={} wb_count={} looping={} progress={} played={} consumed={} host_queued={queued_now}",
                                    snapshot.wb_index,
                                    snapshot.is_new,
                                    snapshot.did_mix,
                                    snapshot.source_frames,
                                    snapshot.buffers[0].frames,
                                    snapshot.wb_count,
                                    snapshot.buffers[0].looping,
                                    st.voice_wb_progress_frames[0],
                                    st.voice_played_samples[0],
                                    st.voice_wbufs_consumed[0]
                                );
                            }
                        }
                    }
                    big_out.extend_from_slice(&out_stereo);
                    if block_consumed_wb {
                        break;
                    }
                }

                if let Some(ob) = out_buf {
                    let mut behavior =
                        audren_behavior::BehaviorInfo::from_user_revision(revision_num);
                    behavior.clear_error();
                    if let Some(param) = in_behavior_param {
                        if audren_behavior::check_valid_revision(param.revision) {
                            behavior.set_user_lib_revision(param.revision);
                        }
                        behavior.update_flags(param.flags);
                    }
                    let mempool_out_count = mempool_count;
                    let voice_out_count = voice_count_seen;
                    let effect_out_count = effect_count_seen;
                    let sink_out_count = (in_sinks_sz / 0x140) as usize;

                    let mempools_sz: u32 = (mempool_out_count as u32) * 0x10;
                    let voices_sz: u32 = (voice_out_count as u32) * 0x10;
                    let effect_status_size: u32 = if behavior.is_effect_info_version2_supported() {
                        0x90
                    } else {
                        0x10
                    };
                    let effects_sz: u32 = (effect_out_count as u32) * effect_status_size;
                    let sinks_sz: u32 = (sink_out_count as u32) * 0x20;
                    let perf_sz: u32 = if in_perf_sz == 0 { 0 } else { 0x10 };
                    let behaviour_sz: u32 = audren_behavior::OUT_STATUS_SIZE as u32;
                    let render_info_sz: u32 = if behavior.is_elapsed_frame_count_supported() {
                        0x10
                    } else {
                        0
                    };

                    let mempools_off = 0x40usize;
                    let voices_off = mempools_off + mempools_sz as usize;
                    let effects_off = voices_off + voices_sz as usize;
                    let sinks_off = effects_off + effects_sz as usize;
                    let perf_off = sinks_off + sinks_sz as usize;
                    let behaviour_off = perf_off + perf_sz as usize;
                    let render_info_off = behaviour_off + behaviour_sz as usize;
                    let total_size: u32 = render_info_off as u32 + render_info_sz;
                    let mut out = vec![0u8; total_size as usize];
                    out[0x00..0x04].copy_from_slice(&revision.to_le_bytes());
                    out[0x04..0x08].copy_from_slice(&behaviour_sz.to_le_bytes());
                    out[0x08..0x0C].copy_from_slice(&mempools_sz.to_le_bytes());
                    out[0x0C..0x10].copy_from_slice(&voices_sz.to_le_bytes());
                    out[0x10..0x14].copy_from_slice(&(in_channels_sz as u32).to_le_bytes());
                    out[0x14..0x18].copy_from_slice(&effects_sz.to_le_bytes());
                    out[0x18..0x1C].copy_from_slice(&(in_mixes_sz as u32).to_le_bytes());
                    out[0x1C..0x20].copy_from_slice(&sinks_sz.to_le_bytes());
                    out[0x20..0x24].copy_from_slice(&perf_sz.to_le_bytes());
                    out[0x28..0x2C].copy_from_slice(&render_info_sz.to_le_bytes());
                    out[0x3C..0x40].copy_from_slice(&total_size.to_le_bytes());

                    for (i, &in_state) in mempool_in_states.iter().enumerate() {
                        let new_state: u32 = match in_state {
                            4 => 5,
                            2 => 3,
                            s => s,
                        };
                        let off = mempools_off + i * 0x10;
                        out[off..off + 4].copy_from_slice(&new_state.to_le_bytes());
                    }

                    for vid in 0..voice_count_seen {
                        let off = voices_off + vid * 0x10;
                        let played = st.voice_played_samples.get(vid).copied().unwrap_or(0);
                        let consumed = st.voice_wbufs_consumed.get(vid).copied().unwrap_or(0);
                        out[off..off + 8].copy_from_slice(&played.to_le_bytes());
                        out[off + 8..off + 12].copy_from_slice(&consumed.to_le_bytes());
                    }
                    for (i, &state) in effect_out_states.iter().enumerate() {
                        let off = effects_off + i * effect_status_size as usize;
                        out[off] = state;
                    }
                    behavior
                        .out_status()
                        .write_to(&mut out[behaviour_off..behaviour_off + behaviour_sz as usize]);
                    if render_info_sz != 0 {
                        out[render_info_off..render_info_off + 8]
                            .copy_from_slice(&frame.to_le_bytes());
                    }

                    if ob.size < out.len() as u64 {
                        log::error!(
                            "IAudioRenderer.RequestUpdate{} output buffer is too small: have={:#x} need={:#x}",
                            if cmd_id == 10 { "Auto" } else { "" },
                            ob.size,
                            out.len()
                        );
                        return build_ipc_response(
                            ctx,
                            nexium_common::result::KERNEL_INVALID_SIZE,
                            &[],
                            &[],
                        );
                    }
                    let n = out.len();
                    let write_result = kernel.address_space.write(ob.addr, &out);
                    if let Err(error) = &write_result {
                        log::error!(
                            "IAudioRenderer.RequestUpdate{} failed to write output at {:#x}: {:?}",
                            if cmd_id == 10 { "Auto" } else { "" },
                            ob.addr,
                            error
                        );
                        return build_ipc_response(ctx, KERNEL_INVALID_ADDRESS, &[], &[]);
                    }
                    if audio_debug_enabled() {
                        use std::sync::atomic::{AtomicBool, Ordering};
                        static LOGGED_OUTPUT: AtomicBool = AtomicBool::new(false);
                        if !LOGGED_OUTPUT.swap(true, Ordering::Relaxed) {
                            let mut readback = [0u8; 0x10];
                            let read_result = kernel.address_space.read(ob.addr, &mut readback);
                            log::info!(
                                "[audio-debug] output addr={:#x} cap={:#x} wrote={:#x}/{:#x} write={:?} read={:?} hdr={:02x?} recv_alias={:?} recv_static={:?}",
                                ob.addr,
                                ob.size,
                                n,
                                out.len(),
                                write_result,
                                read_result,
                                readback,
                                ctx.recv_buffers,
                                ctx.recv_statics
                            );
                        }
                    }
                }
                if let Some(pb) = perf_buf {
                    let zero = vec![0u8; (pb.size as usize).min(0x100)];
                    let _ = kernel.address_space.write(pb.addr, &zero);
                }

                if std::env::var("NEXIUM_AUDIO_TEST_TONE").ok().as_deref() == Some("1") {
                    let base_phase = (frame as f32) * (TARGET_FRAMES as f32);
                    let phase_inc = std::f32::consts::TAU * 440.0 / TARGET_SR;
                    let total_frames = big_out.len() / 2;
                    for i in 0..total_frames {
                        let s = (((base_phase + i as f32) * phase_inc).sin()) * 0.25;
                        big_out[i * 2] = s;
                        big_out[i * 2 + 1] = s;
                    }
                }

                if let Some(sink) = crate::audio_sink::host_audio_sink() {
                    let pushed = sink.push_stereo_f32(&big_out);
                    if diagnostics_enabled() || audio_debug_enabled() {
                        let mix_peak = big_out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
                        use std::sync::atomic::{AtomicBool, Ordering as O};
                        static FIRST_PUSH: AtomicBool = AtomicBool::new(false);
                        static FIRST_NONZERO: AtomicBool = AtomicBool::new(false);
                        if !FIRST_PUSH.swap(true, O::Relaxed) {
                            log::info!(
                                "audio: first push to sink â€” pushed {} frames of {} mix_peak={:.4} (frame {})",
                                pushed,
                                TARGET_FRAMES,
                                mix_peak,
                                frame
                            );
                        }
                        if mix_peak > 0.001 && !FIRST_NONZERO.swap(true, O::Relaxed) {
                            log::info!(
                                "audio: FIRST NON-ZERO MIX â€” peak={:.4} pushed={}/{} (frame {})",
                                mix_peak,
                                pushed,
                                TARGET_FRAMES,
                                frame
                            );
                        }
                    }
                }

                log::trace!(
                    "IAudioRenderer.RequestUpdate{} in={:?} out={:?} perf={:?} mempools={} voices={} frame={}",
                    if cmd_id == 10 { "Auto" } else { "" },
                    in_buf.map(|b| b.size),
                    out_buf.map(|b| b.size),
                    perf_buf.map(|b| b.size),
                    mempool_count,
                    voice_count_seen,
                    frame
                );
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            5 => {
                st.state = 0;
                log::debug!("IAudioRenderer.Start");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            6 => {
                st.state = 1;
                log::debug!("IAudioRenderer.Stop");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            7 => {
                let event_handle = if let Some(&h) = kernel.audio_renderer_events.get(&key) {
                    h
                } else {
                    let h = kernel.handles.create_handle(HandleType::Event);
                    kernel.event_signals.insert(h, false);
                    kernel.audio_renderer_events.insert(key, h);
                    log::debug!(
                        "IAudioRenderer.QuerySystemEvent â†’ new event handle={:#x}",
                        h
                    );
                    h
                };
                return build_ipc_response_copy(ctx, 0, &[], &[event_handle]);
            }
            8 => {
                let in_off = ctx.cmif_in_data_off;
                if ctx.cmif_in_data_len >= 4 {
                    st.rendering_time_limit = u32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            9 => {
                let v = st.rendering_time_limit;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            12 => {
                let in_off = ctx.cmif_in_data_off;
                if ctx.cmif_in_data_len >= 4 {
                    st.voice_drop_param = f32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            13 => {
                let v = st.voice_drop_param;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            other => {
                log::warn!("IAudioRenderer.cmd_{} UNHANDLED â†’ empty SUCCESS", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "IAudioDevice" {
        match cmd_id {
            0 | 6 | 14 => {
                let buf = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(b) = buf {
                    let mut name = vec![0u8; (b.size as usize).min(0x100)];
                    let bytes = b"AudioTvOutput";
                    let n = bytes.len().min(name.len());
                    name[..n].copy_from_slice(&bytes[..n]);
                    let _ = kernel.address_space.write(b.addr, &name);
                }
                let count: u32 = 1;
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            1 | 7 => {
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            2 | 8 => {
                let vol: f32 = 1.0;
                return build_ipc_response(ctx, 0, &vol.to_le_bytes(), &[]);
            }
            3 | 10 | 13 => {
                let buf = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(b) = buf {
                    let mut name = vec![0u8; (b.size as usize).min(0x100)];
                    let bytes = b"AudioTvOutput";
                    let n = bytes.len().min(name.len());
                    name[..n].copy_from_slice(&bytes[..n]);
                    let _ = kernel.address_space.write(b.addr, &name);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 | 11 | 12 => {
                let h = if let Some(&h) = kernel.audio_buffer_events.get(&session_handle) {
                    h
                } else {
                    let h = kernel.handles.create_handle(HandleType::Event);
                    kernel.event_signals.insert(h, true);
                    kernel.audio_buffer_events.insert(session_handle, h);
                    h
                };
                kernel.event_signals.insert(h, true);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            5 => {
                let ch: u32 = 2;
                return build_ipc_response(ctx, 0, &ch.to_le_bytes(), &[]);
            }
            other => {
                log::debug!("IAudioDevice.cmd_{} â†’ empty SUCCESS", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "audout:u" && (cmd_id == 0 || cmd_id == 2) {
        let name_buf = ctx
            .recv_statics
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = name_buf {
            let cap = (buf.size as usize).min(0x100);
            let mut name = vec![0u8; cap];
            let bytes = b"DeviceOut";
            let n = bytes.len().min(cap);
            name[..n].copy_from_slice(&bytes[..n]);
            let _ = kernel.address_space.write(buf.addr, &name);
        }
        let count: u32 = 1;
        log::debug!(
            "audout:u ListAudioOuts cmd_{} â†’ count=1 (DeviceOut)",
            cmd_id
        );
        return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
    }

    if port_name == "audout:u" && (cmd_id == 1 || cmd_id == 3) {
        let name_buf = ctx
            .recv_statics
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = name_buf {
            let cap = (buf.size as usize).min(0x100);
            let mut name = vec![0u8; cap];
            let bytes = b"DeviceOut";
            let n = bytes.len().min(cap);
            name[..n].copy_from_slice(&bytes[..n]);
            let _ = kernel.address_space.write(buf.addr, &name);
        }

        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let sample_rate = if in_avail >= 4 {
            u32::from_le_bytes([
                ctx.buf[in_off],
                ctx.buf[in_off + 1],
                ctx.buf[in_off + 2],
                ctx.buf[in_off + 3],
            ])
        } else {
            0
        };
        let channel_count = if in_avail >= 6 {
            u16::from_le_bytes([ctx.buf[in_off + 4], ctx.buf[in_off + 5]])
        } else {
            0
        };
        let effective_rate = if sample_rate == 0 { 48000 } else { sample_rate };
        let effective_channels: u32 = if channel_count == 0 {
            2
        } else {
            channel_count as u32
        };

        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&effective_rate.to_le_bytes());
        out.extend_from_slice(&effective_channels.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());

        let is_domain = kernel
            .sessions
            .get(&session_handle)
            .map(|s| s.is_domain)
            .unwrap_or(false);
        log::debug!(
            "audout:u OpenAudioOut sample_rate={} channels={} â†’ IAudioOut (domain={})",
            effective_rate,
            effective_channels,
            is_domain
        );
        if is_domain {
            let object_id = alloc_domain_object(kernel, session_handle, "IAudioOut");
            return build_ipc_response_full(ctx, 0, &out, &[], &[], &[object_id]);
        } else {
            let h = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(h, "IAudioOut".to_string());
            kernel.sessions.insert(h, session);
            crate::services::audio_out::handlers::open_audio_out_session(kernel, h);
            return build_ipc_response(ctx, 0, &out, &[h]);
        }
    }

    if port_name == "IAudioOut" {
        use crate::services::audio_out::handlers as aout;
        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let in_u32 = if in_avail >= 4 {
            u32::from_le_bytes([
                ctx.buf[in_off],
                ctx.buf[in_off + 1],
                ctx.buf[in_off + 2],
                ctx.buf[in_off + 3],
            ])
        } else {
            0
        };
        let in_u64 = if in_avail >= 8 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&ctx.buf[in_off..in_off + 8]);
            u64::from_le_bytes(b)
        } else {
            0
        };
        match cmd_id {
            0 => {
                let v = aout::get_audio_out_state(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            1 => {
                let result = aout::start_audio_out_result(kernel, ctx, session_handle);
                return build_ipc_response(ctx, result, &[], &[]);
            }
            2 => {
                aout::stop_audio_out(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            3 | 7 => {
                const RESULT_BUFFER_COUNT_REACHED: u32 = 153 | (8 << 9);
                let full = aout::total_buffer_count(kernel, session_handle) >= 32;
                if full {
                    return build_ipc_response(ctx, RESULT_BUFFER_COUNT_REACHED, &[], &[]);
                }
                aout::append_audio_out_buffer(kernel, ctx, session_handle, in_u64);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let h = aout::register_buffer_event(kernel, ctx, session_handle);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            5 | 8 => {
                let n = aout::get_released_audio_out_buffer(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &n.to_le_bytes(), &[]);
            }
            6 => {
                let v = aout::contains_audio_out_buffer(kernel, ctx, session_handle, in_u64) as u8;
                return build_ipc_response(ctx, 0, &[v, 0, 0, 0], &[]);
            }
            9 => {
                let v = aout::get_audio_out_buffer_count(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            10 => {
                let v = aout::get_audio_out_played_sample_count(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            11 => {
                let v = aout::flush_audio_out_buffers(kernel, ctx, session_handle) as u8;
                return build_ipc_response(ctx, 0, &[v, 0, 0, 0], &[]);
            }
            12 => {
                aout::set_audio_out_volume(kernel, ctx, session_handle, in_u32);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            13 => {
                let v = aout::get_audio_out_volume(kernel, ctx, session_handle);
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            other => {
                log::debug!("IAudioOut.cmd_{} â†’ empty SUCCESS", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "set" || port_name == "set:sys" {
        if let Some(outcome) = cmif_dispatch_set(kernel, ctx) {
            log::debug!(
                "set.cmd_{} â†’ {} bytes (rc={:#x}) via #[service]",
                cmd_id,
                outcome.inline_out.len(),
                outcome.result
            );
            return build_ipc_response(ctx, outcome.result, &outcome.inline_out, &[]);
        }
    }

    if let Some(resp) =
        crate::services::generated::dispatch_generated(kernel, port_name, ctx, session_handle)
    {
        return resp;
    }

    if port_name == "fsp-srv" && cmd_id == 203 {
        log::debug!(
            "fsp-srv.OpenPatchDataStorageByCurrentProcess â†’ ResultTargetNotFound (no patch)"
        );
        return build_ipc_response(ctx, 0x7D402, &[], &[]);
    }

    if port_name == "fsp-srv" && cmd_id == 1005 {
        let mode: u32 = 0;
        return build_ipc_response(ctx, 0, &mode.to_le_bytes(), &[]);
    }

    if let Some((data, handle_opt)) = applet_command_response(kernel, port_name, cmd_id) {
        log::debug!(
            "{}.cmd_{} â†’ returning data ({} bytes, handle={:?})",
            port_name,
            cmd_id,
            data.len(),
            handle_opt
        );
        let handles: Vec<u32> = handle_opt.into_iter().collect();
        return build_ipc_response(ctx, 0, &data, &handles);
    }

    if matches!(port_name, "time:u" | "time:s" | "time:a" | "time:r") && cmd_id == 20 {
        let h = kernel.ensure_time_shmem_handle();
        log::debug!("time:u GetSharedMemoryNativeHandle â†’ handle={:#x}", h);
        return build_ipc_response_copy(ctx, 0, &[], &[h]);
    }

    if matches!(port_name, "time:u" | "time:s" | "time:a" | "time:r") && cmd_id == 200 {
        return build_ipc_response(ctx, 0, &[0], &[]);
    }

    if port_name == "prepo:u" && cmd_id == 10104 {
        return build_ipc_response(ctx, 0, &[], &[]);
    }

    if port_name == "hwopus" && matches!(cmd_id, 1 | 3 | 5 | 7 | 8 | 9) {
        let in_off = ctx.cmif_in_data_off;
        let channels = if ctx.cmif_in_data_len as usize >= 8 {
            u32::from_le_bytes([
                ctx.buf[in_off + 4],
                ctx.buf[in_off + 5],
                ctx.buf[in_off + 6],
                ctx.buf[in_off + 7],
            ])
        } else {
            2
        };
        let size = crate::services::hwopus::HwOpusService::work_buffer_size(channels);
        log::debug!(
            "hwopus GetWorkBufferSize channels={} â†’ {:#x}",
            channels,
            size
        );
        return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
    }

    if port_name == "hwopus" && (cmd_id == 0 || cmd_id == 2 || cmd_id == 4 || cmd_id == 6) {
        let in_off = ctx.cmif_in_data_off;
        let sample_rate = u32::from_le_bytes([
            ctx.buf[in_off],
            ctx.buf[in_off + 1],
            ctx.buf[in_off + 2],
            ctx.buf[in_off + 3],
        ]);
        let channels = u32::from_le_bytes([
            ctx.buf[in_off + 4],
            ctx.buf[in_off + 5],
            ctx.buf[in_off + 6],
            ctx.buf[in_off + 7],
        ]);
        kernel
            .services
            .hwopus
            .open(session_handle, sample_rate, channels);
        log::debug!(
            "hwopus OpenHardwareOpusDecoder rate={} ch={} â†’ IHardwareOpusDecoder",
            sample_rate,
            channels
        );
        let is_domain = kernel
            .sessions
            .get(&session_handle)
            .map(|s| s.is_domain)
            .unwrap_or(false);
        if is_domain {
            let object_id = alloc_domain_object(kernel, session_handle, "IHardwareOpusDecoder");
            return build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id]);
        } else {
            let h = kernel.handles.create_handle(HandleType::Session);
            kernel
                .sessions
                .insert(h, Session::new(h, "IHardwareOpusDecoder".to_string()));
            return build_ipc_response(ctx, 0, &[], &[h]);
        }
    }

    if port_name == "IHardwareOpusDecoder" {
        if cmd_id == 1 || cmd_id == 3 {
            return build_ipc_response(ctx, 0, &[], &[]);
        }
        if matches!(cmd_id, 0 | 2 | 4 | 5 | 6 | 7 | 8 | 9) {
            let in_off = ctx.cmif_in_data_off;
            let reset = if cmd_id == 6 || cmd_id == 7 {
                true
            } else {
                ctx.cmif_in_data_len as usize >= 1 && ctx.buf[in_off] != 0
            };
            let sb = ctx
                .send_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied();
            let input: Vec<u8> = if let Some(b) = sb {
                let mut d = vec![0u8; b.size as usize];
                if kernel.address_space.read(b.addr, &mut d).is_ok() {
                    d
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            let (data_size, sample_count, pcm) =
                kernel.services.hwopus.decode(session_handle, &input, reset);
            if let Some(rb) = ctx
                .recv_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
            {
                let n = (rb.size as usize).min(pcm.len());
                let _ = kernel.address_space.write(rb.addr, &pcm[..n]);
            }
            let mut out = Vec::new();
            out.extend_from_slice(&data_size.to_le_bytes());
            out.extend_from_slice(&sample_count.to_le_bytes());
            if matches!(cmd_id, 4 | 5 | 6 | 7 | 8 | 9) {
                out.extend_from_slice(&0u64.to_le_bytes());
            }
            return build_ipc_response(ctx, 0, &out, &[]);
        }
    }

    if port_name == "IAppletResource" && cmd_id == 0 {
        let h = kernel.handles.create_handle(HandleType::SharedMemory);
        log::debug!(
            "IAppletResource.GetSharedMemoryHandle â†’ hid_shmem_handle={:#x}",
            h
        );
        return build_ipc_response_copy(ctx, 0, &[], &[h]);
    }

    if port_name == "mm:u" {
        let data_start = ctx.cmif_in_data_off.min(ctx.buf.len());
        let data_end = data_start
            .saturating_add(ctx.cmif_in_data_len)
            .min(ctx.buf.len());
        let (result, out_data) = kernel
            .services
            .mm
            .dispatch(cmd_id, &ctx.buf[data_start..data_end]);
        return build_ipc_response(ctx, result, &out_data, &[]);
    }

    log::warn!(
        "dispatch_service_v2: {} cmd_{} FELL THROUGH to legacy dispatch_service (probably needs a real handler)",
        port_name,
        cmd_id
    );
    let tls_snapshot = ctx.buf.clone();
    let mut svc_ctx = crate::services::IpcCtx {
        tls_buf: &tls_snapshot,
        pending_frames,
    };
    let (result, out_data) = kernel
        .services
        .dispatch_service(port_name, cmd_id, &mut svc_ctx);
    build_ipc_response(ctx, result, &out_data, &[])
}

fn ipc_input_u32(ctx: &ipc::IpcCtx, relative_offset: usize) -> Option<u32> {
    let start = ctx.cmif_in_data_off.checked_add(relative_offset)?;
    let bytes = ctx.buf.get(start..start.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

const IGBP_REQUEST_BUFFER: u32 = 1;
const IGBP_SET_BUFFER_COUNT: u32 = 2;
const IGBP_DEQUEUE_BUFFER: u32 = 3;
const IGBP_DETACH_BUFFER: u32 = 4;
const IGBP_DETACH_NEXT_BUFFER: u32 = 5;
const IGBP_ATTACH_BUFFER: u32 = 6;
const IGBP_QUEUE_BUFFER: u32 = 7;
const IGBP_CANCEL_BUFFER: u32 = 8;
const IGBP_QUERY: u32 = 9;
const IGBP_CONNECT: u32 = 10;
const IGBP_DISCONNECT: u32 = 11;
const IGBP_ALLOCATE_BUFFERS: u32 = 13;
const IGBP_SET_PREALLOCATED_BUFFER: u32 = 14;

fn handle_binder_transact(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    _session_handle: u32,
) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    let (binder_id, code) = if ctx.cmif_in_data_len >= 8 {
        let off = ctx.cmif_in_data_off;
        let bid = i32::from_le_bytes([
            ctx.buf[off],
            ctx.buf[off + 1],
            ctx.buf[off + 2],
            ctx.buf[off + 3],
        ]);
        let c = u32::from_le_bytes([
            ctx.buf[off + 4],
            ctx.buf[off + 5],
            ctx.buf[off + 6],
            ctx.buf[off + 7],
        ]);
        (bid as u32, c)
    } else {
        (0u32, 0u32)
    };

    let mut in_parcel: Vec<u8> = Vec::new();
    let in_src = ctx
        .send_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .copied()
        .or_else(|| {
            ctx.send_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
        });
    if let Some(sb) = in_src {
        in_parcel.resize(sb.size as usize, 0);
        let _ = kernel.address_space.read(sb.addr, &mut in_parcel);
    }

    let reply = igbp_handle_transact(kernel, binder_id, code, &in_parcel);

    log::trace!(
        "IHOSBinderDriver.TransactParcel{} binder={} code={} in_size={} reply_size={}",
        if cmd_id == 3 { "Auto" } else { "" },
        binder_id,
        code,
        in_parcel.len(),
        reply.len()
    );

    if code == IGBP_REQUEST_BUFFER || code == IGBP_DEQUEUE_BUFFER {
        let preview = &in_parcel[..in_parcel.len().min(64)];
        log::trace!(
            "IGBP in code={} (len={}): {:02x?}",
            code,
            in_parcel.len(),
            preview
        );
        let preview = &reply[..reply.len().min(96)];
        log::trace!(
            "IGBP reply code={} (len={}): {:02x?}",
            code,
            reply.len(),
            preview
        );
    }

    let out_dst = ctx
        .recv_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .copied()
        .or_else(|| {
            ctx.recv_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
        });
    if let Some(rb) = out_dst {
        let n = reply.len().min(rb.size as usize);
        match kernel.address_space.write(rb.addr, &reply[..n]) {
            Ok(()) => {}
            Err(e) => log::error!(
                "binder reply write FAILED to {:#x} ({} bytes): {:?}",
                rb.addr,
                n,
                e
            ),
        }
    } else {
        log::warn!(
            "binder transact code={} produced {}-byte reply but no recv buffer descriptor",
            code,
            reply.len()
        );
    }

    build_ipc_response(ctx, 0, &[], &[])
}

fn igbp_handle_transact(
    kernel: &mut Kernel,
    binder_id: u32,
    code: u32,
    in_parcel: &[u8],
) -> Vec<u8> {
    let mut reader = ParcelReader::new(in_parcel);
    let _ = reader.skip_interface_token();

    match code {
        IGBP_CONNECT => {
            let _listener = reader.read_i32();
            let api = reader.read_i32().unwrap_or(0);
            let _producer_controlled = reader.read_i32();
            let (w, h) = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                bq.connected_api = api;
                (bq.width, bq.height)
            });
            log::debug!("IGBP::Connect binder={} api={} {}x{}", binder_id, api, w, h);
            let mut p = ParcelBuilder::new();
            p.write_bq_buffer_output(w, h);
            p.write_u32(0);
            p.finish()
        }
        IGBP_DISCONNECT => {
            log::debug!("IGBP::Disconnect binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_SET_PREALLOCATED_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let has = reader.read_i32().unwrap_or(0);
            if has == 0 {
                log::warn!(
                    "IGBP::SetPreallocatedBuffer slot={} has=0 â€” no buffer",
                    slot
                );
                return ParcelBuilder::new().finish();
            }
            let mut gb = parse_flattened_graphic_buffer(&mut reader);
            if let Some(ref mut g) = gb {
                if g.nvmap_id == 0 && g.kind == 254 {
                    let tiled_size = compute_tiled_size(g.stride, g.height, g.block_height_log2);
                    let needed = (g.buffer_offset as usize).saturating_add(tiled_size);
                    let pick = kernel
                        .nvdrv
                        .nvmap_handles
                        .iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) >= needed)
                        .min_by_key(|(_, h)| h.size as usize)
                        .map(|(id, _)| *id);
                    if let Some(id) = pick {
                        g.nvmap_id = id;
                        log::debug!(
                            "SetPreallocatedBuffer fixup: nvmap_id=0 â†’ {} (off={:#x} tiled_size={:#x} needed={:#x})",
                            id,
                            g.buffer_offset,
                            tiled_size,
                            needed
                        );
                    }
                }
            }
            let parsed = gb.is_some();
            let (nvmap_id, w, h, off) = gb
                .as_ref()
                .map(|g| (g.nvmap_id, g.width, g.height, g.buffer_offset))
                .unwrap_or((0, 0, 0, 0));
            kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                if let Some(gb) = gb {
                    bq.set_preallocated(slot, gb);
                }
            });
            log::debug!(
                "IGBP::SetPreallocatedBuffer binder={} slot={} parsed={} nvmap_id={} {}x{} off={:#x}",
                binder_id,
                slot,
                parsed,
                nvmap_id,
                w,
                h,
                off
            );
            if crate::services::am::mode_trace_enabled() {
                log::warn!(
                    "[mode-trace] SetPreallocatedBuffer binder={} slot={} {}x{} docked={}",
                    binder_id,
                    slot,
                    w,
                    h,
                    crate::hid_state::is_docked()
                );
            }
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_REQUEST_BUFFER => {
            kernel
                .nvdrv
                .stats
                .request_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let gb = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.request_buffer(slot).cloned());
            let mut p = ParcelBuilder::new();
            if let Some(gb) = gb {
                p.write_u32(1);
                p.write_flattened_graphic_buffer(&gb);
            } else {
                p.write_u32(0);
            }
            p.write_u32(0);
            log::debug!("IGBP::RequestBuffer binder={} slot={}", binder_id, slot);
            p.finish()
        }
        IGBP_DEQUEUE_BUFFER => {
            kernel
                .nvdrv
                .stats
                .dequeue_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let async_dequeue = reader.read_i32().unwrap_or(0) != 0;
            let _w = reader.read_u32();
            let _h = reader.read_u32();
            let _fmt = reader.read_i32();
            let _usage = reader.read_u32();
            let started = std::time::Instant::now();
            let (slot, free, deq, queued) = loop {
                let state = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                    (
                        bq.try_dequeue(),
                        bq.free.len(),
                        bq.dequeued.len(),
                        bq.queued.len(),
                    )
                });
                if let (Some(slot), free, deq, queued) = state {
                    break (Some(slot), free, deq, queued);
                }
                if async_dequeue {
                    let (_, free, deq, queued) = state;
                    break (None, free, deq, queued);
                }
                if started.elapsed() >= std::time::Duration::from_secs(3) {
                    let (_, free, deq, queued) = state;
                    log::error!(
                        "IGBP::DequeueBuffer timed out waiting for a released slot binder={} (free={} deq={} queued={})",
                        binder_id,
                        free,
                        deq,
                        queued
                    );
                    break (None, free, deq, queued);
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            };
            log::trace!(
                "IGBP::DequeueBuffer binder={} â†’ slot={} (free={} deq={} queued={})",
                binder_id,
                slot.unwrap_or(u32::MAX),
                free,
                deq,
                queued
            );
            let mut p = ParcelBuilder::new();
            p.write_u32(slot.unwrap_or(u32::MAX));
            p.write_u32(1);
            p.write_flattened_zero_fence();
            p.write_u32(if slot.is_some() { 0 } else { (-11i32) as u32 });
            p.finish()
        }
        IGBP_QUEUE_BUFFER => {
            kernel
                .nvdrv
                .stats
                .queue_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let flattened_size = reader.read_u32().unwrap_or(0);
            let flattened_size_hi = reader.read_u32().unwrap_or(0);
            let _timestamp = reader.read_u64();
            let _is_auto = reader.read_i32();
            let crop_l = reader.read_i32().unwrap_or(0);
            let crop_t = reader.read_i32().unwrap_or(0);
            let crop_r = reader.read_i32().unwrap_or(0);
            let crop_b = reader.read_i32().unwrap_or(0);
            let scaling = reader.read_i32().unwrap_or(0);
            log::trace!(
                "IGBP::QueueBuffer crop=({},{},{},{}) scaling={}",
                crop_l,
                crop_t,
                crop_r,
                crop_b,
                scaling
            );
            let transform = reader.read_i32().unwrap_or(0) as u32;
            let _sticky = reader.read_u32();
            let _async = reader.read_i32();
            let swap_interval = reader.read_i32().unwrap_or(1);
            let fence_count_raw = reader.read_i32().unwrap_or(-1);
            let fence_count = fence_count_raw.clamp(0, 4) as u32;
            let mut acquire_fences = Vec::with_capacity(fence_count as usize);
            for index in 0..4 {
                let syncpt_id = reader.read_u32().unwrap_or(u32::MAX);
                let threshold = reader.read_u32().unwrap_or(0);
                if index < fence_count && syncpt_id != u32::MAX {
                    acquire_fences.push((syncpt_id, threshold));
                }
            }
            if flattened_size != 0x54
                || flattened_size_hi != 0
                || !(0..=4).contains(&fence_count_raw)
            {
                log::warn!(
                    "IGBP::QueueBuffer malformed input binder={} slot={} size={:#x}:{:08x} fence_count={}",
                    binder_id,
                    slot,
                    flattened_size_hi,
                    flattened_size,
                    fence_count_raw,
                );
                let (qw, qh) = kernel
                    .nvdrv
                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                let mut p = ParcelBuilder::new();
                p.write_bq_buffer_output(qw, qh);
                p.write_u32((-22i32) as u32);
                return p.finish();
            }
            if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() && !acquire_fences.is_empty() {
                log::info!(
                    "IGBP::QueueBuffer acquire fences binder={} slot={} fences={:?}",
                    binder_id,
                    slot,
                    acquire_fences
                );
            }

            let mut strict_acquire_fences = Vec::new();
            let mut elided_ordered_fences = Vec::new();
            for &(syncpt_id, threshold) in &acquire_fences {
                match kernel
                    .nvdrv
                    .queue_buffer_fence_disposition(syncpt_id, threshold)
                {
                    nexium_nvdrv::FenceWaitDisposition::Reached => {}
                    nexium_nvdrv::FenceWaitDisposition::OrderedPredecessor => {
                        elided_ordered_fences.push((syncpt_id, threshold));
                        if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                            log::trace!(
                                "IGBP::QueueBuffer elided ordered acquire fence binder={} slot={} syncpt={} threshold={}",
                                binder_id,
                                slot,
                                syncpt_id,
                                threshold
                            );
                        }
                    }
                    nexium_nvdrv::FenceWaitDisposition::Strict => {
                        strict_acquire_fences.push((syncpt_id, threshold));
                    }
                }
            }
            let mut acquire_fences_ready = true;
            for &(syncpt_id, threshold) in &strict_acquire_fences {
                let wait_started = std::time::Instant::now();
                while !kernel.nvdrv.is_syncpoint_reached(syncpt_id, threshold) {
                    if wait_started.elapsed() >= std::time::Duration::from_secs(3) {
                        acquire_fences_ready = false;
                        log::error!(
                            "IGBP::QueueBuffer acquire fence timeout binder={} slot={} syncpt={} threshold={}",
                            binder_id,
                            slot,
                            syncpt_id,
                            threshold
                        );
                        break;
                    }
                    nexium_common::host_wake::micro_pause();
                }
            }
            if !acquire_fences_ready {
                let (qw, qh) = kernel
                    .nvdrv
                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                let mut p = ParcelBuilder::new();
                p.write_bq_buffer_output(qw, qh);
                p.write_u32((-110i32) as u32);
                return p.finish();
            }

            let prev_interval = kernel
                .bufferqueue_swap_intervals
                .insert(binder_id, swap_interval);
            if prev_interval != Some(swap_interval) {
                log::info!(
                    "IGBP::QueueBuffer binder={} swap_interval={} (was {:?})",
                    binder_id,
                    swap_interval,
                    prev_interval
                );
            }
            let gb_opt = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                let slot_acquired = bq.queue_and_acquire(slot);
                if let Some(entry) = bq.slots.get_mut(slot as usize) {
                    entry.last_swap_interval = swap_interval.max(0) as u32;
                }
                let pace_until = slot_acquired
                    .then(|| bq.schedule_swap(std::time::Instant::now(), swap_interval))
                    .flatten();
                let r = slot_acquired
                    .then(|| bq.request_buffer(slot).cloned())
                    .flatten();
                let slot_count = bq.slots.len();
                let has_buf = bq
                    .slots
                    .get(slot as usize)
                    .and_then(|s| s.buffer.as_ref())
                    .is_some();
                (r, slot_count, has_buf, slot_acquired, pace_until)
            });
            let (gb_opt, slot_count, has_buf, slot_acquired, pace_until) = gb_opt;
            if !slot_acquired {
                log::warn!(
                    "IGBP::QueueBuffer rejected invalid slot transition binder={} slot={}",
                    binder_id,
                    slot
                );
                let (qw, qh) = kernel
                    .nvdrv
                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                let mut p = ParcelBuilder::new();
                p.write_bq_buffer_output(qw, qh);
                p.write_u32((-22i32) as u32);
                return p.finish();
            }
            if crate::services::am::mode_trace_enabled() {
                use std::sync::atomic::{AtomicU64, Ordering};
                static QUEUE_TRACES: AtomicU64 = AtomicU64::new(0);
                if QUEUE_TRACES.fetch_add(1, Ordering::Relaxed) < 8 {
                    log::warn!(
                        "[mode-trace] QueueBuffer slot={} crop=({},{},{},{}) scaling={} buffer={}",
                        slot,
                        crop_l,
                        crop_t,
                        crop_r,
                        crop_b,
                        scaling,
                        gb_opt
                            .as_ref()
                            .map(|gb| format!("{}x{}", gb.width, gb.height))
                            .unwrap_or_else(|| "none".to_string())
                    );
                }
            }
            log::trace!(
                "IGBP::QueueBuffer binder={} slot={} swap_interval={} transform={:#x} slot_count={} has_buf={} gb_some={}",
                binder_id,
                slot,
                swap_interval,
                transform,
                slot_count,
                has_buf,
                gb_opt.is_some()
            );
            if transform != 0 && std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                log::warn!(
                    "IGBP::QueueBuffer transform binder={} slot={} transform={:#x}",
                    binder_id,
                    slot,
                    transform
                );
            }

            if let Some(gb) = gb_opt {
                if async_present_pipeline_enabled()
                    && std::env::var_os("NEXIUM_PRESENT_CPU_ONLY").is_none()
                {
                    let ordered_present_identity = (|| {
                        let linear_size = (gb.stride as usize)
                            .checked_mul(gb.height as usize)?
                            .checked_mul(4)?;
                        let tiled_size =
                            compute_tiled_size(gb.stride, gb.height, gb.block_height_log2);
                        let nvmap = kernel.nvdrv.nvmap_handles.get(&gb.nvmap_id)?;
                        let tiled_present = nvmap.size as usize >= tiled_size
                            && gb.kind == 254
                            && gb.block_height_log2 != 0;
                        let surface_size = if tiled_present {
                            tiled_size as u64
                        } else {
                            linear_size as u64
                        };
                        let buffer_offset = gb.buffer_offset as u64;
                        if nvmap.address == 0
                            || buffer_offset.checked_add(surface_size)? > nvmap.size as u64
                        {
                            return None;
                        }
                        let present_cpu_addr = nvmap.address.checked_add(buffer_offset)?;
                        let mut present_gpu_vas = {
                            let mappings = kernel.nvdrv.gpu.mappings.read();
                            mappings
                                .gpu_regions_for_cpu_range(present_cpu_addr, 4)
                                .into_iter()
                                .map(|(gpu_va, _)| gpu_va)
                                .filter(|&gpu_va| {
                                    mappings.nvmap_id_for(gpu_va) == Some(gb.nvmap_id)
                                        && mappings.cpu_address_for(gpu_va)
                                            == Some(present_cpu_addr)
                                        && mappings
                                            .cpu_range_for(gpu_va)
                                            .is_some_and(|(_, available)| available >= surface_size)
                                })
                                .collect::<Vec<_>>()
                        };
                        present_gpu_vas.sort_unstable();
                        present_gpu_vas.dedup();
                        if present_gpu_vas.is_empty() {
                            return None;
                        }
                        Some((
                            present_cpu_addr,
                            present_gpu_vas,
                            surface_size,
                            tiled_present,
                        ))
                    })();
                    let renderer_for_ordered_present = ordered_present_identity
                        .as_ref()
                        .and_then(|_| kernel.nvdrv.renderer().cloned());
                    let ordered_identity_valid = ordered_present_identity.is_some();
                    if let (
                        Some(renderer),
                        Some((
                            present_cpu_addr,
                            direct_gpu_vas,
                            present_surface_size,
                            present_tiled,
                        )),
                    ) = (renderer_for_ordered_present, ordered_present_identity)
                    {
                        let queue_crop: Option<(u32, u32, u32, u32)> = {
                            let cw = crop_r.saturating_sub(crop_l).max(0) as u32;
                            let ch = crop_b.saturating_sub(crop_t).max(0) as u32;
                            if crop_l >= 0
                                && crop_t >= 0
                                && cw > 0
                                && ch > 0
                                && crop_r as u32 <= gb.width
                                && crop_b as u32 <= gb.height
                                && (cw < gb.width || ch < gb.height)
                            {
                                Some((crop_l as u32, crop_t as u32, cw, ch))
                            } else {
                                None
                            }
                        };
                        let frame_queue = kernel.nvdrv.frame_queue.clone();
                        let bufferqueues = kernel.nvdrv.bufferqueues.clone();
                        let present_bufferqueues = std::sync::Arc::clone(&bufferqueues);
                        let stats = kernel.nvdrv.stats.clone();
                        let (present_nvmap_id, present_width, present_height) =
                            (gb.nvmap_id, gb.width, gb.height);
                        let present_source_vas =
                            renderer.present_alias_vas(gb.nvmap_id, gb.width, gb.height);
                        if !present_source_vas.is_empty() {
                            kernel
                                .nvdrv
                                .gpu
                                .maxwell_dma
                                .lock()
                                .register_present_surface(
                                    gb.nvmap_id,
                                    &direct_gpu_vas,
                                    gb.width,
                                    gb.height,
                                    &present_source_vas,
                                );
                        }
                        let maxwell_dma_for_ordered =
                            std::sync::Arc::clone(&kernel.nvdrv.gpu.maxwell_dma);
                        let address_space_for_ordered = kernel.address_space.clone();
                        let (present_stride, present_bh_log2) = (gb.stride, gb.block_height_log2);
                        let present_id = kernel.allocate_present_id();
                        let present_metadata = std::sync::Arc::clone(&kernel.present_metadata);
                        let present_delivery_lanes =
                            std::sync::Arc::clone(&kernel.present_delivery_lanes);
                        let queued = kernel.nvdrv.try_queue_ordered_present(move || {
                            let _slot_guard = AcquiredBufferSlotGuard::new(
                                present_bufferqueues,
                                binder_id,
                                slot,
                            );
                            let exact_present_source = try_select_ordered_present_source(
                                &maxwell_dma_for_ordered,
                                |maxwell_dma| {
                                    direct_gpu_vas
                                        .iter()
                                        .filter_map(|&present_va| {
                                            maxwell_dma
                                                .exact_present_source_token(
                                                    present_va,
                                                    present_width,
                                                    present_height,
                                                )
                                                .map(|token| (present_va, token))
                                        })
                                        .filter(|(_, token)| token.destination_is_current())
                                        .filter(|(_, token)| {
                                            renderer.render_target_stamp(token.source).is_some_and(
                                                |current| {
                                                    if token.source_may_advance {
                                                        current >= token.source_stamp
                                                    } else {
                                                        current == token.source_stamp
                                                    }
                                                },
                                            )
                                        })
                                        .max_by_key(|(_, token)| token.source_stamp)
                                },
                            );
                            let direct_present = renderer.newest_exact_present_target_at_vas(
                                present_nvmap_id,
                                present_width,
                                present_height,
                                &direct_gpu_vas,
                            );
                            if exact_present_source.is_none() && direct_present.is_none() {
                                use std::sync::atomic::{AtomicU64, Ordering};
                                static GUEST_PRESENTS: AtomicU64 = AtomicU64::new(0);
                                let sequence = GUEST_PRESENTS.fetch_add(1, Ordering::Relaxed);
                                if sequence < 3 || sequence % 600 == 0 {
                                    log::info!(
                                        "ordered present falling back to guest bytes #{} binder={} slot={} nvmap={} tiled={} searched_vas={}",
                                        sequence,
                                        binder_id,
                                        slot,
                                        present_nvmap_id,
                                        present_tiled,
                                        direct_gpu_vas.len()
                                    );
                                }
                                submit_ordered_cpu_present(
                                    move |_read_rect| {
                                        let mut raw =
                                            vec![0u8; usize::try_from(present_surface_size).ok()?];
                                        address_space_for_ordered
                                            .read(present_cpu_addr, &mut raw)
                                            .ok()?;
                                        let linear = (present_stride as usize)
                                            .checked_mul(present_height as usize)?
                                            .checked_mul(4)?;
                                        let mut pixels = if present_tiled {
                                            unswizzle_block_linear(
                                                &raw,
                                                present_stride,
                                                present_height,
                                                4,
                                                present_bh_log2,
                                            )
                                        } else {
                                            raw
                                        };
                                        if pixels.len() < linear {
                                            pixels.resize(linear, 0);
                                        }
                                        for px in pixels.chunks_exact_mut(4) {
                                            px[3] = 0xFF;
                                        }
                                        Some((present_stride, present_height, pixels, Some(false)))
                                    },
                                    frame_queue,
                                    stats,
                                    present_width,
                                    present_height,
                                    transform,
                                    queue_crop,
                                    pace_until,
                                );
                                return;
                            }
                            if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                                use std::sync::atomic::{AtomicU64, Ordering};
                                static ORDERED_KEYS: AtomicU64 = AtomicU64::new(0);
                                let sequence = ORDERED_KEYS.fetch_add(1, Ordering::Relaxed);
                                if sequence < 3 || sequence % 20 == 0 {
                                    log::info!(
                                        "[ordered-present-key #{}] slot={} nvmap={} cpu={:#x} mode={}",
                                        sequence,
                                        slot,
                                        present_nvmap_id,
                                        present_cpu_addr,
                                        if exact_present_source.is_some() {
                                            "exact-dma"
                                        } else {
                                            "newest-exact"
                                        }
                                    );
                                }
                            }
                            submit_ordered_gpu_present(
                                move |read_rect| {
                                    if let Some((_, token)) = exact_present_source {
                                        if token.source_may_advance {
                                            renderer.readback_live_provenance_pipelined(
                                                binder_id,
                                                present_id,
                                                token.source,
                                                token.source_stamp,
                                                read_rect,
                                            )
                                        } else {
                                            renderer.readback_exact_provenance_pipelined(
                                                binder_id,
                                                present_id,
                                                token.source,
                                                token.source_stamp,
                                                read_rect,
                                            )
                                        }
                                    } else if let Some((present_key, _)) = direct_present {
                                        renderer.readback_target_pipelined_pinned_at_va(
                                            binder_id,
                                            present_id,
                                            present_nvmap_id,
                                            present_width,
                                            present_height,
                                            present_key.gpu_va,
                                            present_cpu_addr,
                                            read_rect,
                                        )
                                    } else {
                                        unreachable!("ordered GPU present entered without a source")
                                    }
                                },
                                present_metadata,
                                present_delivery_lanes,
                                binder_id,
                                present_id,
                                frame_queue,
                                stats,
                                present_width,
                                present_height,
                                transform,
                                queue_crop,
                                pace_until,
                            );
                        });
                        match queued {
                            nexium_nvdrv::AsyncPresentSubmit::Enqueued => {
                                note_ordered_present_profile(
                                    OrderedPresentProfileOutcome::Enqueued,
                                );
                                kernel
                                    .nvdrv
                                    .queue_buffer_active
                                    .store(true, std::sync::atomic::Ordering::Relaxed);
                                let (qw, qh) = kernel
                                    .nvdrv
                                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                                let mut p = ParcelBuilder::new();
                                p.write_bq_buffer_output(qw, qh);
                                p.write_u32(0);
                                return p.finish();
                            }
                            nexium_nvdrv::AsyncPresentSubmit::Coalesced => {
                                note_ordered_present_profile(
                                    OrderedPresentProfileOutcome::Coalesced,
                                );
                                let _ = release_rejected_present_slot_after_fences(
                                    &bufferqueues,
                                    binder_id,
                                    slot,
                                    &elided_ordered_fences,
                                    std::time::Duration::from_secs(3),
                                    |syncpt_id, threshold| {
                                        kernel.nvdrv.is_syncpoint_reached(syncpt_id, threshold)
                                    },
                                );
                                kernel
                                    .nvdrv
                                    .queue_buffer_active
                                    .store(true, std::sync::atomic::Ordering::Relaxed);
                                let (qw, qh) = kernel
                                    .nvdrv
                                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                                let mut p = ParcelBuilder::new();
                                p.write_bq_buffer_output(qw, qh);
                                p.write_u32(0);
                                return p.finish();
                            }
                            nexium_nvdrv::AsyncPresentSubmit::Unavailable => {
                                note_ordered_present_profile(
                                    OrderedPresentProfileOutcome::Unavailable,
                                );
                                let _ = release_rejected_present_slot_after_fences(
                                    &bufferqueues,
                                    binder_id,
                                    slot,
                                    &elided_ordered_fences,
                                    std::time::Duration::from_secs(3),
                                    |syncpt_id, threshold| {
                                        kernel.nvdrv.is_syncpoint_reached(syncpt_id, threshold)
                                    },
                                );
                                kernel
                                    .nvdrv
                                    .queue_buffer_active
                                    .store(true, std::sync::atomic::Ordering::Relaxed);
                                let (qw, qh) = kernel
                                    .nvdrv
                                    .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
                                let mut p = ParcelBuilder::new();
                                p.write_bq_buffer_output(qw, qh);
                                p.write_u32(0);
                                return p.finish();
                            }
                        }
                    } else if !ordered_identity_valid {
                        note_ordered_present_profile(
                            OrderedPresentProfileOutcome::IdentityRejected,
                        );
                    } else {
                        note_ordered_present_profile(OrderedPresentProfileOutcome::TargetRejected);
                    }
                }
                let bpp: usize = 4;
                let linear_size = (gb.stride as usize) * (gb.height as usize) * bpp;
                let tiled_size = compute_tiled_size(gb.stride, gb.height, gb.block_height_log2);
                let resolved: Option<(u64, bool)> = if let Some(nvmap) =
                    kernel.nvdrv.nvmap_handles.get(&gb.nvmap_id)
                {
                    let actual_size = nvmap.size as usize;
                    let is_tiled =
                        actual_size >= tiled_size && gb.kind == 254 && gb.block_height_log2 != 0;
                    log::trace!(
                        "QueueBuffer fast-path: nvmap_id={} addr={:#x} off={:#x} size={:#x} kind={} bh_log2={} tiled={} (slot={})",
                        gb.nvmap_id,
                        nvmap.address,
                        gb.buffer_offset,
                        actual_size,
                        gb.kind,
                        gb.block_height_log2,
                        is_tiled,
                        slot
                    );
                    Some((nvmap.address.wrapping_add(gb.buffer_offset), is_tiled))
                } else {
                    let mut candidates: Vec<(u32, u64, u32)> = kernel
                        .nvdrv
                        .nvmap_handles
                        .iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) == linear_size)
                        .map(|(id, h)| (*id, h.address, h.size))
                        .collect();
                    candidates.sort_by_key(|(id, _, _)| *id);
                    if let Some(&(id, addr, size)) = candidates.last() {
                        log::debug!(
                            "QueueBuffer fallback pick newest: nvmap_id={} addr={:#x} size={:#x} (slot={} candidates={})",
                            id,
                            addr,
                            size,
                            slot,
                            candidates.len()
                        );
                        Some((addr, false))
                    } else {
                        log::warn!(
                            "QueueBuffer fallback: no exact-size candidate (linear_size={:#x} slot={} total_handles={})",
                            linear_size,
                            slot,
                            kernel.nvdrv.nvmap_handles.len()
                        );
                        None
                    }
                };
                kernel.nvdrv.wait_gpu_idle();
                let renderer_for_present = kernel.nvdrv.renderer().cloned();
                if let (Some(probe), Some(renderer)) = (
                    std::env::var("NEXIUM_PRESENT_PROBE_NVMAP")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok()),
                    renderer_for_present.as_ref(),
                ) {
                    use std::sync::atomic::{AtomicBool, Ordering as ProbeOrdering};
                    static PROBE_DONE: AtomicBool = AtomicBool::new(false);
                    if !PROBE_DONE.swap(true, ProbeOrdering::Relaxed) {
                        let probe_w = std::env::var("NEXIUM_PRESENT_PROBE_WIDTH")
                            .ok()
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(gb.width);
                        let probe_h = std::env::var("NEXIUM_PRESENT_PROBE_HEIGHT")
                            .ok()
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(gb.height);
                        if let Some(nvmap) = kernel.nvdrv.nvmap_handles.get(&probe) {
                            let cpu_len = ((probe_w as usize)
                                .saturating_mul(probe_h as usize)
                                .saturating_mul(4))
                            .min(nvmap.size as usize)
                            .min(16 * 1024 * 1024);
                            let mut cpu_bytes = vec![0u8; cpu_len];
                            let cpu_ok = nvmap.address != 0
                                && kernel
                                    .address_space
                                    .read(nvmap.address, &mut cpu_bytes)
                                    .is_ok();
                            let cpu_nonzero = cpu_bytes.iter().filter(|&&b| b != 0).count();
                            log::warn!(
                                "[present-probe] nvmap={} cpu_addr={:#x} cpu_size={:#x} cpu_read={} cpu_nonzero={}/{}",
                                probe,
                                nvmap.address,
                                nvmap.size,
                                cpu_ok,
                                cpu_nonzero,
                                cpu_bytes.len()
                            );
                        } else {
                            log::warn!("[present-probe] nvmap={} has no nvmap handle", probe);
                        }
                        match renderer.readback_target(probe, probe_w, probe_h) {
                            Some(bytes) => {
                                let rgb_nonzero = bytes
                                    .chunks_exact(4)
                                    .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                                    .count();
                                let max_rgb = bytes
                                    .chunks_exact(4)
                                    .map(|p| p[0].max(p[1]).max(p[2]))
                                    .max()
                                    .unwrap_or(0);
                                let avg_rgb = if bytes.is_empty() {
                                    0.0
                                } else {
                                    bytes
                                        .chunks_exact(4)
                                        .map(|p| (p[0] as u64 + p[1] as u64 + p[2] as u64) / 3)
                                        .sum::<u64>() as f64
                                        / (bytes.len() / 4) as f64
                                };
                                log::warn!(
                                    "[present-probe] nvmap={} {}x{} bytes={} rgb_nonzero={} max_rgb={} avg_rgb={:.3}",
                                    probe,
                                    probe_w,
                                    probe_h,
                                    bytes.len(),
                                    rgb_nonzero,
                                    max_rgb,
                                    avg_rgb
                                );
                                if let Some(home) = std::env::var_os("APPDATA") {
                                    let path = std::path::PathBuf::from(home)
                                        .join("NeXium")
                                        .join("logs")
                                        .join(format!("probe-{}.bmp", probe));
                                    let _ = save_rgba_bmp(&path, probe_w, probe_h, &bytes);
                                }
                            }
                            None => log::warn!(
                                "[present-probe] nvmap={} {}x{} had no readable render target",
                                probe,
                                probe_w,
                                probe_h
                            ),
                        }
                    }
                }
                let gpu_stats = kernel.nvdrv.stats.snapshot();
                let has_gpu_activity = gpu_stats.gpfifo_submits != 0
                    || gpu_stats.maxwell3d_draws != 0
                    || gpu_stats.maxwell3d_clears != 0
                    || gpu_stats.fermi_2d_blits != 0
                    || gpu_stats.maxwell_dma_blits != 0;
                let has_gpu_present_target = renderer_for_present.is_some() && has_gpu_activity;
                let cpu_present_only = std::env::var_os("NEXIUM_PRESENT_CPU_ONLY").is_some();
                let present_cpu_addr = resolved.map(|(addr, _)| addr).unwrap_or(0);
                let present_is_tiled = resolved.map(|(_, is_tiled)| is_tiled).unwrap_or(false);
                let present_surface_size = if present_is_tiled {
                    tiled_size
                } else {
                    linear_size
                } as u64;
                let mut mapped_present_vas = if present_cpu_addr != 0 {
                    let mappings = kernel.nvdrv.gpu.mappings.read();
                    mappings
                        .gpu_regions_for_cpu_range(present_cpu_addr, 4)
                        .into_iter()
                        .map(|(va, _)| va)
                        .filter(|&va| {
                            mappings.nvmap_id_for(va) == Some(gb.nvmap_id)
                                && mappings.cpu_address_for(va) == Some(present_cpu_addr)
                                && mappings
                                    .cpu_range_for(va)
                                    .is_some_and(|(_, available)| available >= present_surface_size)
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let resolve_aliases = if present_is_tiled {
                    let mappings = kernel.nvdrv.gpu.mappings.read();
                    mapped_present_vas
                        .iter()
                        .filter_map(|va| va.checked_add(tiled_size as u64))
                        .filter(|&va| {
                            let Some((mapping_start, mapping_size, mapping_cpu)) =
                                mappings.mapping_at(va)
                            else {
                                return false;
                            };
                            let Some(resolve_nvmap_id) = mappings.nvmap_id_for(va) else {
                                return false;
                            };
                            let Some(resolve_handle) =
                                kernel.nvdrv.nvmap_handles.get(&resolve_nvmap_id)
                            else {
                                return false;
                            };
                            mapping_start == va
                                && mapping_size >= linear_size as u64
                                && resolve_nvmap_id.checked_add(1) == Some(gb.nvmap_id)
                                && resolve_handle.address != 0
                                && mapping_cpu == resolve_handle.address
                                && resolve_handle.size as u64 >= linear_size as u64
                                && mappings.cpu_address_for(va) == Some(resolve_handle.address)
                                && mappings
                                    .cpu_range_for(va)
                                    .is_some_and(|(_, available)| available >= linear_size as u64)
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                mapped_present_vas.extend(resolve_aliases);
                mapped_present_vas.sort_unstable();
                mapped_present_vas.dedup();
                let present_source_vas = renderer_for_present
                    .as_ref()
                    .map(|renderer| renderer.present_alias_vas(gb.nvmap_id, gb.width, gb.height))
                    .unwrap_or_default();
                if !mapped_present_vas.is_empty() && !present_source_vas.is_empty() {
                    kernel
                        .nvdrv
                        .gpu
                        .maxwell_dma
                        .lock()
                        .register_present_surface(
                            gb.nvmap_id,
                            &mapped_present_vas,
                            gb.width,
                            gb.height,
                            &present_source_vas,
                        );
                }
                if let Some(r_async) =
                    renderer_for_present.filter(|_| has_gpu_present_target && !cpu_present_only)
                {
                    let rt_worker = nexium_nvdrv::render_thread::present_thread();
                    let fq = kernel.nvdrv.frame_queue.clone();
                    let bufferqueues = kernel.nvdrv.bufferqueues.clone();
                    let qba = kernel.nvdrv.queue_buffer_active.clone();
                    let stats = kernel.nvdrv.stats.clone();
                    let (pw, ph, pnv) = (gb.width, gb.height, gb.nvmap_id);
                    let maxwell_dma_for_present =
                        std::sync::Arc::clone(&kernel.nvdrv.gpu.maxwell_dma);
                    qba.store(true, std::sync::atomic::Ordering::Relaxed);
                    let present_profile = std::env::var_os("NEXIUM_NVDRV_PROFILE").is_some();
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static PRESENT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_ENQUEUED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_DROPPED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_EXECUTED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_READY: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_EMPTY: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_NS: AtomicU64 = AtomicU64::new(0);
                    let attempt = if present_profile {
                        PRESENT_ATTEMPTS.fetch_add(1, Ordering::Relaxed) + 1
                    } else {
                        0
                    };
                    let queue_crop: Option<(u32, u32, u32, u32)> = {
                        let cw = crop_r.saturating_sub(crop_l).max(0) as u32;
                        let ch = crop_b.saturating_sub(crop_t).max(0) as u32;
                        if crop_l >= 0
                            && crop_t >= 0
                            && cw > 0
                            && ch > 0
                            && crop_r as u32 <= gb.width
                            && crop_b as u32 <= gb.height
                            && (cw < gb.width || ch < gb.height)
                        {
                            Some((crop_l as u32, crop_t as u32, cw, ch))
                        } else {
                            None
                        }
                    };
                    let slot_guard = AcquiredBufferSlotGuard::new(bufferqueues, binder_id, slot);
                    let present_id = kernel.allocate_present_id();
                    let present_metadata = std::sync::Arc::clone(&kernel.present_metadata);
                    let present_delivery_lanes =
                        std::sync::Arc::clone(&kernel.present_delivery_lanes);
                    rt_worker.submit_named("fallback-present-readback", Box::new(move || {
                        let _slot_guard = slot_guard;
                        let t0 = std::time::Instant::now();
                        let exact_present_source = try_select_ordered_present_source(
                            &maxwell_dma_for_present,
                            |maxwell_dma| {
                                mapped_present_vas
                                    .iter()
                                    .filter_map(|&present_va| {
                                        maxwell_dma
                                            .exact_present_source_token(present_va, pw, ph)
                                            .map(|token| (present_va, token))
                                    })
                                    .filter(|(_, token)| token.destination_is_current())
                                    .filter(|(_, token)| {
                                        r_async.render_target_stamp(token.source).is_some_and(
                                            |current| {
                                                if token.source_may_advance {
                                                    current >= token.source_stamp
                                                } else {
                                                    current == token.source_stamp
                                                }
                                            },
                                        )
                                    })
                                    .max_by_key(|(_, token)| token.source_stamp)
                            },
                        );
                        let direct_present = r_async.newest_exact_present_target_at_vas(
                            pnv,
                            pw,
                            ph,
                            &mapped_present_vas,
                        );
                        if diagnostics_enabled()
                            || std::env::var_os("NEXIUM_PRESENT_KEYS").is_some()
                        {
                            use std::sync::atomic::{AtomicU64, Ordering as O2};
                            static QP: AtomicU64 = AtomicU64::new(0);
                            let n = QP.fetch_add(1, O2::Relaxed);
                            if n < 3
                                || n % 300 == 0
                                || (std::env::var_os("NEXIUM_PRESENT_KEYS").is_some()
                                    && n % 20 == 0)
                            {
                                let (present_gpu_va, alias_mode, stamp) = exact_present_source
                                    .as_ref()
                                    .map(|(va, token)| (*va, "exact-dma", token.source_stamp))
                                    .or_else(|| {
                                        direct_present
                                            .as_ref()
                                            .map(|(key, stamp)| (key.gpu_va, "newest-exact", *stamp))
                                    })
                                    .unwrap_or((0, "none", 0));
                                log::info!(
                                    "[queue-present #{}] slot={} nvmap={} buf_off={:#x} present_cpu_addr={:#x} present_gpu_va={:#x} alias_mode={} stamp={}",
                                    n,
                                    slot,
                                    pnv,
                                    gb.buffer_offset,
                                    present_cpu_addr,
                                    present_gpu_va,
                                    alias_mode,
                                    stamp
                                );
                            }
                        }
                        let emitted = submit_ordered_gpu_present(
                            move |read_rect| {
                                if let Some((_, token)) = exact_present_source {
                                    if token.source_may_advance {
                                        r_async.readback_live_provenance_pipelined(
                                            binder_id,
                                            present_id,
                                            token.source,
                                            token.source_stamp,
                                            read_rect,
                                        )
                                    } else {
                                        r_async.readback_exact_provenance_pipelined(
                                            binder_id,
                                            present_id,
                                            token.source,
                                            token.source_stamp,
                                            read_rect,
                                        )
                                    }
                                } else if let Some((present_key, _)) = direct_present {
                                    r_async.readback_target_pipelined_pinned_at_va(
                                        binder_id,
                                        present_id,
                                        pnv,
                                        pw,
                                        ph,
                                        present_key.gpu_va,
                                        present_cpu_addr,
                                        read_rect,
                                    )
                                } else {
                                    nexium_nvdrv::PipelinedPresentReadback {
                                        completion: None,
                                        submission: nexium_nvdrv::PipelinedPresentSubmission::SourceUnavailable,
                                    }
                                }
                            },
                            present_metadata,
                            present_delivery_lanes,
                            binder_id,
                            present_id,
                            fq,
                            stats,
                            pw,
                            ph,
                            transform,
                            queue_crop,
                            pace_until,
                        );
                        if present_profile {
                            if emitted {
                                PRESENT_READY.fetch_add(1, Ordering::Relaxed);
                            } else {
                                PRESENT_EMPTY.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        if present_profile {
                            let elapsed = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                            PRESENT_NS.fetch_add(elapsed, Ordering::Relaxed);
                            let exec = PRESENT_EXECUTED.fetch_add(1, Ordering::Relaxed) + 1;
                            if exec % 60 == 0 {
                                let total_ns = PRESENT_NS.load(Ordering::Relaxed);
                                log::warn!(
                                    "[nvprof] present_exec executed={} ready={} empty={} avg_ms={:.3}",
                                    exec,
                                    PRESENT_READY.load(Ordering::Relaxed),
                                    PRESENT_EMPTY.load(Ordering::Relaxed),
                                    total_ns as f64 / exec as f64 / 1_000_000.0
                                );
                            }
                        }
                    }));
                    let submitted = true;
                    if present_profile {
                        if submitted {
                            PRESENT_ENQUEUED.fetch_add(1, Ordering::Relaxed);
                        } else {
                            PRESENT_DROPPED.fetch_add(1, Ordering::Relaxed);
                        }
                        if attempt % 60 == 0 {
                            log::warn!(
                                "[nvprof] present_enqueue attempts={} enqueued={} dropped={}",
                                attempt,
                                PRESENT_ENQUEUED.load(Ordering::Relaxed),
                                PRESENT_DROPPED.load(Ordering::Relaxed)
                            );
                        }
                    }
                } else if let Some((addr, is_tiled)) = resolved {
                    let read_size = if is_tiled { tiled_size } else { linear_size };
                    let mut raw = vec![0u8; read_size];
                    let mut effective_tiled = is_tiled;
                    let mut effective_addr = addr;
                    let mut effective_bh_log2 = gb.block_height_log2;
                    if kernel.address_space.read(addr, &mut raw).is_ok() {
                        let nonzero = raw.iter().filter(|&&b| b != 0).count();
                        if nonzero > 0 && slot < 2 {
                            log::debug!(
                                "QueueBuffer slot={} addr={:#x} nonzero_bytes={}/{} first16={:02x?}",
                                slot,
                                addr,
                                nonzero,
                                read_size,
                                &raw[..16.min(raw.len())]
                            );
                        }
                        if is_tiled && raw.iter().all(|&b| b == 0) {
                            let (tiled_rt_cpu, dma_bh_log2, dma_stride, dma_height) = {
                                let dma = kernel.nvdrv.gpu.maxwell_dma.lock();
                                (
                                    dma.last_tiled_dst_cpu,
                                    dma.last_tiled_dst_bh_log2,
                                    dma.last_tiled_dst_stride,
                                    dma.last_tiled_dst_height,
                                )
                            };
                            let mut found = false;
                            if tiled_rt_cpu != 0 {
                                let dma_tiled_size = compute_tiled_size(
                                    dma_stride.max(gb.stride),
                                    dma_height.max(gb.height),
                                    dma_bh_log2,
                                );
                                let mut tiled_raw = vec![0u8; dma_tiled_size.max(tiled_size)];
                                if kernel
                                    .address_space
                                    .read(tiled_rt_cpu, &mut tiled_raw)
                                    .is_ok()
                                    && tiled_raw.iter().any(|&b| b != 0)
                                {
                                    log::debug!(
                                        "QueueBuffer tiled-rt-redirect: slot={} slot_tiled={:#x} â†’ rt_cpu={:#x} (gralloc_bh={} dma_bh={} dma_stride={} dma_h={})",
                                        slot,
                                        addr,
                                        tiled_rt_cpu,
                                        gb.block_height_log2,
                                        dma_bh_log2,
                                        dma_stride,
                                        dma_height
                                    );
                                    raw = tiled_raw;
                                    effective_tiled = true;
                                    effective_addr = tiled_rt_cpu;
                                    effective_bh_log2 = dma_bh_log2;
                                    found = true;
                                }
                            }
                            if !found {
                                let mut best: Option<(u32, u64)> = None;
                                for (id, h) in &kernel.nvdrv.nvmap_handles {
                                    if h.address == 0 || (h.size as usize) != linear_size {
                                        continue;
                                    }
                                    match best {
                                        None => {
                                            best = Some((*id, h.address));
                                        }
                                        Some((best_id, _)) if *id > best_id => {
                                            best = Some((*id, h.address));
                                        }
                                        _ => {}
                                    }
                                }
                                if let Some((id, lin_addr)) = best {
                                    let mut lin_raw = vec![0u8; linear_size];
                                    if kernel.address_space.read(lin_addr, &mut lin_raw).is_ok()
                                        && lin_raw.iter().any(|&b| b != 0)
                                    {
                                        log::debug!(
                                            "QueueBuffer tiled-empty fallback nvmap_id={} addr={:#x} (slot={})",
                                            id,
                                            lin_addr,
                                            slot
                                        );
                                        raw = lin_raw;
                                        effective_tiled = false;
                                        effective_addr = lin_addr;
                                    }
                                }
                            }
                        }
                        let fermi_frame = kernel.nvdrv.drain_fermi2d_frame();
                        if let Some(qf) = fermi_frame.as_ref() {
                            log::debug!(
                                "QueueBuffer Fermi2D-captured frame: {}x{} ({} bytes)",
                                qf.width,
                                qf.height,
                                qf.pixels.len()
                            );
                        }
                        let vk_readback = kernel
                            .nvdrv
                            .renderer()
                            .and_then(|r| r.readback_target(gb.nvmap_id, gb.width, gb.height))
                            .map(|mut bytes| {
                                let row = (gb.width as usize) * 4;
                                let h = gb.height as usize;
                                if bytes.len() >= row * h {
                                    for y in 0..h / 2 {
                                        let top = y * row;
                                        let bot = (h - 1 - y) * row;
                                        let (a, b) = bytes.split_at_mut(bot);
                                        a[top..top + row].swap_with_slice(&mut b[..row]);
                                    }
                                }
                                let readback_len = bytes.len();
                                let (present_w, present_h, mut bytes) =
                                    maybe_crop_present_subwindow(bytes, gb.width, gb.height);
                                log::debug!(
                                    "QueueBuffer vk_readback: gb={}x{} stride={} readback_bytes={} â†’ present {}x{} (cropped={})",
                                    gb.width,
                                    gb.height,
                                    gb.stride,
                                    readback_len,
                                    present_w,
                                    present_h,
                                    present_w != gb.width || present_h != gb.height
                                );
                                make_present_opaque(&mut bytes);
                                dump_present_frame(&bytes, present_w, present_h);
                                (present_w, present_h, bytes)
                            });
                        let have_gpu_frame = fermi_frame.is_some() || vk_readback.is_some();
                        let legacy_gfx = kernel
                            .nvdrv
                            .legacy_gfx
                            .load(std::sync::atomic::Ordering::Relaxed);
                        let (pixels, rgb_nz) = if have_gpu_frame {
                            (Vec::new(), 0usize)
                        } else {
                            let mut pixels = if effective_tiled {
                                unswizzle_block_linear(
                                    &raw,
                                    gb.stride,
                                    gb.height,
                                    bpp,
                                    effective_bh_log2,
                                )
                            } else {
                                raw
                            };
                            if pixels.len() < linear_size {
                                pixels.resize(linear_size, 0);
                            }
                            if legacy_gfx {
                                let row_bytes = (gb.width * (bpp as u32)) as usize;
                                let h = gb.height as usize;
                                for y in 0..h / 2 {
                                    let top = y * row_bytes;
                                    let bot = (h - 1 - y) * row_bytes;
                                    if bot + row_bytes <= pixels.len() {
                                        let (a, b) = pixels.split_at_mut(bot);
                                        a[top..top + row_bytes]
                                            .swap_with_slice(&mut b[..row_bytes]);
                                    }
                                }
                            }
                            for px in pixels.chunks_exact_mut(4) {
                                px[3] = 0xFF;
                            }
                            let rgb_nz = pixels
                                .chunks_exact(4)
                                .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                                .count();
                            (pixels, rgb_nz)
                        };
                        let (frame_w, frame_h, mut frame_pixels) = if let Some(qf) = fermi_frame {
                            (qf.width, qf.height, qf.pixels)
                        } else if let Some((w, h, bytes)) = vk_readback {
                            nexium_common::frame_present::set_last_presented(w, h, bytes.clone());
                            (w, h, bytes)
                        } else if rgb_nz >= 16 {
                            if legacy_gfx {
                                if let Some((x0, y0, w, h)) =
                                    active_bbox(&pixels, gb.width, gb.height)
                                {
                                    let area_ratio = (w as f32 * h as f32)
                                        / (gb.width as f32 * gb.height as f32);
                                    if area_ratio < 0.65 && w >= 64 && h >= 64 {
                                        let upscaled = crop_and_upscale(
                                            &pixels, gb.width, x0, y0, w, h, gb.width, gb.height,
                                        );
                                        log::debug!(
                                            "QueueBuffer legacy_gfx sub-window: src=({},{}) {}x{} â†’ upscale to {}x{}",
                                            x0,
                                            y0,
                                            w,
                                            h,
                                            gb.width,
                                            gb.height
                                        );
                                        (gb.width, gb.height, upscaled)
                                    } else {
                                        (gb.width, gb.height, pixels)
                                    }
                                } else {
                                    (gb.width, gb.height, pixels)
                                }
                            } else {
                                (gb.width, gb.height, pixels)
                            }
                        } else if let Some((w, h, sdl_pixels)) =
                            try_compose_from_sdl_surface(kernel, gb.width, gb.height)
                        {
                            log::debug!(
                                "QueueBuffer SDL_Surface fallback: {}x{} (back buffer had only {} nonzero RGB pixels)",
                                w,
                                h,
                                rgb_nz
                            );
                            (w, h, sdl_pixels)
                        } else if legacy_gfx {
                            if let Some(renderer) = kernel.nvdrv.renderer() {
                                let r = renderer.clone();
                                let mut color = kernel.nvdrv.last_clear_color();
                                if color[3] < 0.5 {
                                    color[3] = 1.0;
                                }
                                let clears = kernel.nvdrv.last_clear_count();
                                if r.clear_target(gb.nvmap_id, gb.width, gb.height, 0, color)
                                    .is_ok()
                                {
                                    if let Some(bytes) =
                                        r.readback_target(gb.nvmap_id, gb.width, gb.height)
                                    {
                                        log::debug!(
                                            "QueueBuffer legacy_gfx Vulkan clear-only fallback: {}x{} color=[{:.2},{:.2},{:.2},{:.2}] clears={} â†’ {} bytes",
                                            gb.width,
                                            gb.height,
                                            color[0],
                                            color[1],
                                            color[2],
                                            color[3],
                                            clears,
                                            bytes.len()
                                        );
                                        (gb.width, gb.height, bytes)
                                    } else {
                                        (gb.width, gb.height, pixels)
                                    }
                                } else {
                                    (gb.width, gb.height, pixels)
                                }
                            } else {
                                (gb.width, gb.height, pixels)
                            }
                        } else {
                            (gb.width, gb.height, pixels)
                        };
                        make_present_opaque(&mut frame_pixels);
                        let nz = frame_pixels.iter().filter(|b| **b != 0).count();
                        let rgb_nz = frame_pixels
                            .chunks_exact(4)
                            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                            .count();
                        let checksum: u32 = frame_pixels
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .fold(0u32, |a, b| a.wrapping_add(b));
                        log::trace!(
                            "QueueBuffer submit slot={} parsed_nvmap_id={} addr={:#x} {}x{} tiled={} nz={} rgb_nz={} cksum={:#x}",
                            slot,
                            gb.nvmap_id,
                            effective_addr,
                            frame_w,
                            frame_h,
                            effective_tiled,
                            nz,
                            rgb_nz,
                            checksum
                        );
                        {
                            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                            static FRAME_SEQ: AtomicU64 = AtomicU64::new(0);
                            static FIRST_NONBLACK: AtomicBool = AtomicBool::new(false);
                            static LAST_RGB_NZ: AtomicU64 = AtomicU64::new(0);
                            let seq = FRAME_SEQ.fetch_add(1, Ordering::Relaxed);
                            let is_first_nonblack =
                                rgb_nz > 0 && !FIRST_NONBLACK.swap(true, Ordering::Relaxed);
                            let should_dump = nexium_common::dumps::enabled()
                                && ((seq > 0 && seq % 300 == 60) || is_first_nonblack);
                            if should_dump {
                                if let Some(home) = std::env::var_os("APPDATA") {
                                    let path = std::path::PathBuf::from(home)
                                        .join("NeXium")
                                        .join("logs")
                                        .join(format!("compose-{}.bmp", seq));
                                    let _ = save_rgba_bmp(&path, frame_w, frame_h, &frame_pixels);
                                    log::warn!(
                                        "FRAME DUMP seq={} rgb_nz={} â†’ {}",
                                        seq,
                                        rgb_nz,
                                        path.display()
                                    );
                                }
                            }
                            if is_first_nonblack {
                                log::warn!("FIRST NON-BLACK FRAME seq={} rgb_nz={}", seq, rgb_nz);
                            }
                            if seq % 60 == 0 {
                                let prev = LAST_RGB_NZ.swap(rgb_nz as u64, Ordering::Relaxed);
                                if (prev == 0) != (rgb_nz == 0) {
                                    log::warn!(
                                        "frame heartbeat seq={} rgb_nz={} (was {})",
                                        seq,
                                        rgb_nz,
                                        prev
                                    );
                                }
                            }
                        }
                        let (frame_w, frame_h, frame_pixels) = {
                            let cw = crop_r.saturating_sub(crop_l).max(0) as u32;
                            let ch = crop_b.saturating_sub(crop_t).max(0) as u32;
                            let valid = crop_l >= 0
                                && crop_t >= 0
                                && cw > 0
                                && ch > 0
                                && crop_r as u32 <= frame_w
                                && crop_b as u32 <= frame_h;
                            if valid && (cw < frame_w || ch < frame_h) {
                                log::debug!(
                                    "QueueBuffer honoring crop rect ({},{},{},{}) â†’ present {}x{} (was {}x{})",
                                    crop_l, crop_t, crop_r, crop_b, cw, ch, frame_w, frame_h
                                );
                                let cropped = crop_and_upscale(
                                    &frame_pixels,
                                    frame_w,
                                    crop_l as u32,
                                    crop_t as u32,
                                    cw,
                                    ch,
                                    cw,
                                    ch,
                                );
                                (cw, ch, cropped)
                            } else {
                                (frame_w, frame_h, frame_pixels)
                            }
                        };
                        dump_present_frame(&frame_pixels, frame_w, frame_h);
                        let _ = kernel
                            .nvdrv
                            .submit_frame_nonblocking(nexium_nvdrv::QueuedFrame {
                                width: frame_w,
                                height: frame_h,
                                pixels: frame_pixels,
                                present_at: pace_until,
                            });
                    } else {
                        log::warn!(
                            "QueueBuffer: failed to read slot {} addr={:#x} read_size={:#x}",
                            slot,
                            addr,
                            read_size
                        );
                    }
                    kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                        let _ = bq.release(slot);
                    });
                } else {
                    log::warn!(
                        "QueueBuffer: no nvmap candidate for size {} (slot {})",
                        linear_size,
                        slot
                    );
                    kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                        let _ = bq.release(slot);
                    });
                }
            } else {
                log::warn!("QueueBuffer: slot {} has no GraphicBuffer", slot);
                if slot_acquired {
                    kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                        let _ = bq.release(slot);
                    });
                }
            }

            let _ = swap_interval;

            let (qw, qh) = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
            let mut p = ParcelBuilder::new();
            p.write_bq_buffer_output(qw, qh);
            p.write_u32(0);
            p.finish()
        }
        IGBP_CANCEL_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let flattened_size = reader.read_u64().unwrap_or(0);
            let fence_count_raw = reader.read_i32().unwrap_or(-1);
            let fence_count = fence_count_raw.clamp(0, 4) as u32;
            let mut cancel_fences = Vec::with_capacity(fence_count as usize);
            for index in 0..4 {
                let syncpt_id = reader.read_u32().unwrap_or(u32::MAX);
                let threshold = reader.read_u32().unwrap_or(0);
                if index < fence_count && syncpt_id != u32::MAX {
                    cancel_fences.push((syncpt_id, threshold));
                }
            }
            if (flattened_size != 0 && flattened_size != 0x24)
                || !(0..=4).contains(&fence_count_raw)
            {
                log::warn!(
                    "IGBP::CancelBuffer malformed fence binder={} slot={} size={:#x} count={}",
                    binder_id,
                    slot,
                    flattened_size,
                    fence_count_raw
                );
                let mut p = ParcelBuilder::new();
                p.write_u32((-22i32) as u32);
                return p.finish();
            }
            if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() && !cancel_fences.is_empty() {
                log::info!(
                    "IGBP::CancelBuffer binder={} slot={} fences={:?}",
                    binder_id,
                    slot,
                    cancel_fences
                );
            }
            kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.cancel(slot));
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUERY => {
            let what = reader.read_i32().unwrap_or(0);
            let (w, h) = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
            let value: i32 = match what {
                0 => w as i32,
                1 => h as i32,
                2 => 1,
                3 => 2,
                _ => 0,
            };
            log::debug!("IGBP::Query what={} â†’ {}", what, value);
            let mut p = ParcelBuilder::new();
            p.write_u32(value as u32);
            p.write_u32(0);
            p.finish()
        }
        IGBP_SET_BUFFER_COUNT => {
            let count = reader.read_i32().unwrap_or(0);
            log::debug!("IGBP::SetBufferCount binder={} count={}", binder_id, count);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_DETACH_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.cancel(slot));
            log::debug!("IGBP::DetachBuffer binder={} slot={}", binder_id, slot);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_DETACH_NEXT_BUFFER => {
            log::debug!("IGBP::DetachNextBuffer binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.write_u32(0);
            p.write_u32(0);
            p.finish()
        }
        IGBP_ATTACH_BUFFER => {
            log::debug!("IGBP::AttachBuffer binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.write_u32(0);
            p.finish()
        }
        IGBP_ALLOCATE_BUFFERS => {
            let async_ = reader.read_i32().unwrap_or(0);
            log::debug!(
                "IGBP::AllocateBuffers binder={} async={}",
                binder_id,
                async_
            );
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        other => {
            log::debug!("IGBP::Unknown code={} binder={}", other, binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
    }
}

struct ParcelReader<'a> {
    data: &'a [u8],
    payload_off: usize,
    cursor: usize,
    objects_off: usize,
    objects_size: usize,
}

impl<'a> ParcelReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let (payload_off, objects_off, objects_size) = if data.len() >= 16 {
            let po = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            let os = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            let oo = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
            (po, oo, os)
        } else {
            (0, 0, 0)
        };
        Self {
            data,
            payload_off,
            cursor: payload_off,
            objects_off,
            objects_size,
        }
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.cursor + 4 > self.data.len() {
            return None;
        }
        let v = u32::from_le_bytes([
            self.data[self.cursor],
            self.data[self.cursor + 1],
            self.data[self.cursor + 2],
            self.data[self.cursor + 3],
        ]);
        self.cursor += 4;
        Some(v)
    }

    fn read_i32(&mut self) -> Option<i32> {
        self.read_u32().map(|v| v as i32)
    }

    fn read_u64(&mut self) -> Option<u64> {
        let lo = self.read_u32()? as u64;
        let hi = self.read_u32()? as u64;
        Some(lo | (hi << 32))
    }

    fn skip_interface_token(&mut self) -> Option<()> {
        let _strict_policy = self.read_u32()?;
        let len = self.read_i32()?;
        if len <= 0 {
            return Some(());
        }
        let byte_len = ((len as usize) + 1) * 2;
        let padded = (byte_len + 3) & !3;
        self.cursor += padded;
        Some(())
    }

    fn first_binder_handle(&self) -> Option<u32> {
        if self.objects_size < FLAT_BINDER_OBJECT_SIZE || self.objects_off == 0 {
            return None;
        }
        let end = self.objects_off.checked_add(FLAT_BINDER_OBJECT_SIZE)?;
        if end > self.data.len() {
            return None;
        }
        let obj = &self.data[self.objects_off..end];
        let handle = u32::from_le_bytes([obj[8], obj[9], obj[10], obj[11]]);
        Some(handle)
    }
}

const FLAT_BINDER_OBJECT_SIZE: usize = 24;

struct ParcelBuilder {
    payload: Vec<u8>,
}

impl ParcelBuilder {
    fn new() -> Self {
        Self {
            payload: Vec::new(),
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.payload.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bq_buffer_output(&mut self, w: u32, h: u32) {
        self.write_u32(w);
        self.write_u32(h);
        self.write_u32(0);
        self.write_u32(0);
    }

    fn write_flattened_zero_fence(&mut self) {
        self.write_u32(36);
        self.write_u32(0);
        self.write_u32(0);
        for _ in 0..4 {
            self.write_u32(0);
            self.write_u32(0);
        }
    }

    fn write_flattened_graphic_buffer(&mut self, gb: &nexium_nvdrv::GraphicBuffer) {
        const NUM_INTS: u32 = 81;
        const HEADER_U32S: u32 = 10;
        let body_size = (HEADER_U32S + NUM_INTS) * 4;
        self.write_u32(body_size);
        self.write_u32(0);
        self.write_u32(0x47424652);
        self.write_u32(gb.width);
        self.write_u32(gb.height);
        self.write_u32(gb.stride);
        self.write_u32(gb.format);
        self.write_u32(gb.usage);
        self.write_u32(42);
        self.write_u32(1);
        self.write_u32(0);
        self.write_u32(NUM_INTS);
        let mut ints = [0u32; NUM_INTS as usize];
        ints[0] = 0xFFFF_FFFF;
        ints[1] = gb.nvmap_id;
        ints[2] = 0;
        ints[3] = 0xDAFF_CAFF;
        ints[4] = 42;
        ints[5] = 0;
        ints[6] = gb.usage;
        ints[7] = gb.format;
        ints[8] = gb.format;
        ints[9] = gb.stride;
        ints[10] = gb.width.saturating_mul(gb.height).saturating_mul(4);
        ints[11] = 1;
        ints[12] = 0;
        ints[13] = gb.width;
        ints[14] = gb.height;
        ints[18] = gb.stride.saturating_mul(4);
        ints[19] = gb.nvmap_id;
        ints[20] = gb.buffer_offset as u32;
        ints[21] = gb.kind;
        ints[22] = gb.block_height_log2;
        for v in ints {
            self.write_u32(v);
        }
    }

    fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.payload.len());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&((16 + self.payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
}

fn active_bbox(pixels: &[u8], width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let w = width as usize;
    let h = height as usize;
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut min_y = h;
    let mut max_y = 0usize;
    for y in 0..h {
        let row_off = y * w * 4;
        for x in 0..w {
            let p = &pixels[row_off + x * 4..row_off + x * 4 + 3];
            if p[0] != 0 || p[1] != 0 || p[2] != 0 {
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if max_x < min_x || max_y < min_y {
        return None;
    }
    Some((
        min_x as u32,
        min_y as u32,
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
    ))
}

fn crop_and_upscale(
    src: &[u8],
    src_stride_px: u32,
    sx: u32,
    sy: u32,
    sw: u32,
    sh: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    for dy in 0..dst_h {
        let yy = sy + dy * sh / dst_h;
        for dx in 0..dst_w {
            let xx = sx + dx * sw / dst_w;
            let s = ((yy * src_stride_px + xx) * 4) as usize;
            let d = ((dy * dst_w + dx) * 4) as usize;
            out[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    out
}

fn make_present_opaque(pixels: &mut [u8]) {
    for px in pixels.chunks_exact_mut(4) {
        px[3] = 0xFF;
    }
}

fn legacy_present_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_LEGACY_PRESENT").is_some())
}

fn present_read_rect(width: u32, height: u32) -> Option<[u32; 4]> {
    cached_present_crop(width, height).map(|(x0, y0, w, h)| {
        if should_flip_vulkan_present(width, height) {
            [x0, height.saturating_sub(y0).saturating_sub(h), w, h]
        } else {
            [x0, y0, w, h]
        }
    })
}

fn submit_present_frame(
    read_w: u32,
    read_h: u32,
    bytes: Vec<u8>,
    flip_y: Option<bool>,
    metadata: PresentMetadata,
    frame_queue: &nexium_nvdrv::FrameQueue,
    stats: &std::sync::Arc<nexium_nvdrv::PipelineStats>,
) {
    let (present_w, present_h, bytes) = if legacy_present_enabled() {
        let (w, h, mut b) =
            prepare_vulkan_present_frame(bytes, read_w, read_h, metadata.transform, flip_y);
        make_present_opaque(&mut b);
        (w, h, b)
    } else {
        let (w, h, mut b) = if metadata
            .read_rect
            .map_or(false, |r| r[2] == read_w && r[3] == read_h)
        {
            (read_w, read_h, bytes)
        } else {
            maybe_crop_present_subwindow(bytes, read_w, read_h)
        };
        if flip_y.unwrap_or_else(|| should_flip_vulkan_present(w, h)) {
            flip_present_v(&mut b, w, h);
        }
        apply_present_transform(&mut b, w, h, metadata.transform);
        make_present_opaque(&mut b);
        (w, h, b)
    };
    let (present_w, present_h, bytes) = match metadata.queue_crop {
        Some((cx, cy, cw, ch))
            if cx + cw <= present_w
                && cy + ch <= present_h
                && (cw < present_w || ch < present_h) =>
        {
            let cropped = crop_and_upscale(&bytes, present_w, cx, cy, cw, ch, cw, ch);
            (cw, ch, cropped)
        }
        _ => (present_w, present_h, bytes),
    };
    dump_present_frame(&bytes, present_w, present_h);
    if std::env::var_os("NEXIUM_FRAME_PRESENT_CACHE").is_some() {
        nexium_common::frame_present::set_last_presented(present_w, present_h, bytes.clone());
    }
    nexium_nvdrv::enqueue_bounded_frame(
        frame_queue,
        stats,
        nexium_nvdrv::QueuedFrame {
            width: present_w,
            height: present_h,
            pixels: bytes,
            present_at: metadata.present_at,
        },
    );
}

fn submit_ordered_cpu_present<F>(
    readback_fn: F,
    frame_queue: nexium_nvdrv::FrameQueue,
    stats: std::sync::Arc<nexium_nvdrv::PipelineStats>,
    width: u32,
    height: u32,
    transform: u32,
    queue_crop: Option<(u32, u32, u32, u32)>,
    present_at: Option<std::time::Instant>,
) where
    F: FnOnce(Option<[u32; 4]>) -> Option<(u32, u32, Vec<u8>, Option<bool>)>,
{
    let metadata = PresentMetadata {
        read_rect: present_read_rect(width, height),
        transform,
        queue_crop,
        present_at,
    };
    let readback = readback_fn(metadata.read_rect);
    let Some((read_w, read_h, bytes, flip_y)) = readback else {
        return;
    };
    submit_present_frame(
        read_w,
        read_h,
        bytes,
        flip_y,
        metadata,
        &frame_queue,
        &stats,
    );
}

fn submit_ordered_gpu_present<F>(
    mut readback_fn: F,
    present_metadata: PresentMetadataQueue,
    present_delivery_lanes: PresentDeliveryLanes,
    binder_id: u32,
    present_id: u64,
    frame_queue: nexium_nvdrv::FrameQueue,
    stats: std::sync::Arc<nexium_nvdrv::PipelineStats>,
    width: u32,
    height: u32,
    transform: u32,
    queue_crop: Option<(u32, u32, u32, u32)>,
    present_at: Option<std::time::Instant>,
) -> bool
where
    F: FnMut(Option<[u32; 4]>) -> nexium_nvdrv::PipelinedPresentReadback,
{
    const BACKPRESSURE_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(1);

    let delivery_lane = present_delivery_lane(&present_delivery_lanes, binder_id);
    let _delivery_guard = delivery_lane.lock();
    let metadata = PresentMetadata {
        read_rect: present_read_rect(width, height),
        transform,
        queue_crop,
        present_at,
    };
    present_metadata.lock().insert(present_id, metadata);
    let retry_started = std::time::Instant::now();
    let mut next_backpressure_report = std::time::Duration::from_millis(100);
    let mut emitted = false;
    loop {
        let readback = readback_fn(metadata.read_rect);
        if let Some(completion) = readback.completion {
            match completion {
                nexium_nvdrv::PipelinedPresentCompletion::Ready(frame) => {
                    if let Some(metadata) = present_metadata.lock().remove(&frame.present_id) {
                        submit_present_frame(
                            frame.width,
                            frame.height,
                            frame.pixels,
                            frame.flip_y,
                            metadata,
                            &frame_queue,
                            &stats,
                        );
                        emitted = true;
                    } else {
                        log::warn!(
                            "ordered GPU present lost metadata binder={} present_id={}",
                            binder_id,
                            frame.present_id
                        );
                    }
                }
                nexium_nvdrv::PipelinedPresentCompletion::Dropped { present_id } => {
                    present_metadata.lock().remove(&present_id);
                }
            }
        }

        match readback.submission {
            nexium_nvdrv::PipelinedPresentSubmission::Submitted => return emitted,
            nexium_nvdrv::PipelinedPresentSubmission::Backpressured => {
                let waited = retry_started.elapsed();
                if waited >= next_backpressure_report {
                    log::warn!(
                        "ordered GPU present backpressured binder={} present_id={} waited_ms={:.1}",
                        binder_id,
                        present_id,
                        waited.as_secs_f64() * 1000.0,
                    );
                    next_backpressure_report = next_backpressure_report
                        .saturating_add(std::time::Duration::from_millis(100));
                }
                std::thread::sleep(BACKPRESSURE_RETRY_DELAY);
            }
            nexium_nvdrv::PipelinedPresentSubmission::SourceUnavailable
            | nexium_nvdrv::PipelinedPresentSubmission::Failed => {
                present_metadata.lock().remove(&present_id);
                return emitted;
            }
        }
    }
}

fn outside_crop_has_visible(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> bool {
    let row = width as usize * 4;
    let x1 = x0.saturating_add(w);
    let y1 = y0.saturating_add(h);
    let mut visible = 0u32;
    let mut min_rgb = [255u8; 3];
    let mut max_rgb = [0u8; 3];
    for y in (0..height).step_by(4) {
        let row_off = y as usize * row;
        for x in (0..width).step_by(4) {
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                continue;
            }
            let p = row_off + x as usize * 4;
            if p + 2 >= pixels.len() {
                continue;
            }
            let rgb = [pixels[p], pixels[p + 1], pixels[p + 2]];
            if rgb[0].max(rgb[1]).max(rgb[2]) > 4 {
                visible += 1;
                for i in 0..3 {
                    min_rgb[i] = min_rgb[i].min(rgb[i]);
                    max_rgb[i] = max_rgb[i].max(rgb[i]);
                }
                if visible >= 64 {
                    let range = (max_rgb[0] - min_rgb[0])
                        .max(max_rgb[1] - min_rgb[1])
                        .max(max_rgb[2] - min_rgb[2]);
                    let hi = max_rgb[0].max(max_rgb[1]).max(max_rgb[2]);
                    let lo = min_rgb[0].min(min_rgb[1]).min(min_rgb[2]);
                    if range > 24 {
                        return true;
                    }
                    if hi.saturating_sub(lo) > 48 {
                        return true;
                    }
                }
            }
        }
    }
    if visible >= 64 {
        let range = (max_rgb[0] - min_rgb[0])
            .max(max_rgb[1] - min_rgb[1])
            .max(max_rgb[2] - min_rgb[2]);
        let hi = max_rgb[0].max(max_rgb[1]).max(max_rgb[2]);
        let lo = min_rgb[0].min(min_rgb[1]).min(min_rgb[2]);
        if range > 24 {
            return true;
        }
        if hi.saturating_sub(lo) > 48 {
            return true;
        }
    }
    false
}

fn present_crop_slot() -> &'static std::sync::Mutex<Option<(u32, u32, u32, u32, u32, u32)>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<(u32, u32, u32, u32, u32, u32)>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

fn cached_present_crop(width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    present_crop_slot().lock().ok().and_then(|slot| {
        let (dst_w, dst_h, x0, y0, w, h) = (*slot)?;
        (dst_w == width && dst_h == height).then_some((x0, y0, w, h))
    })
}

fn crop_flipped_opaque(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let src_row = src_w as usize * 4;
    let dst_row = w as usize * 4;
    let mut out = vec![0u8; h as usize * dst_row];
    if src.len() < src_h as usize * src_row {
        return out;
    }
    for dy in 0..h as usize {
        let Some(src_y) = (src_h as usize).checked_sub(1 + y0 as usize + dy) else {
            continue;
        };
        let src_off = src_y
            .saturating_mul(src_row)
            .saturating_add(x0 as usize * 4);
        let dst_off = dy * dst_row;
        if src_off + dst_row > src.len() || dst_off + dst_row > out.len() {
            continue;
        }
        out[dst_off..dst_off + dst_row].copy_from_slice(&src[src_off..src_off + dst_row]);
        for px in out[dst_off..dst_off + dst_row].chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
    }
    out
}

fn prepare_vulkan_present_frame(
    mut bytes: Vec<u8>,
    width: u32,
    height: u32,
    transform: u32,
    flip_y: Option<bool>,
) -> (u32, u32, Vec<u8>) {
    if let Some((x0, y0, w, h)) = cached_present_crop(width, height) {
        let mut cropped = crop_flipped_opaque(&bytes, width, height, x0, y0, w, h);
        apply_present_transform(&mut cropped, w, h, transform);
        return (w, h, cropped);
    }
    let do_flip = match flip_y {
        Some(f) => f,
        None => should_flip_vulkan_present(width, height),
    };
    if do_flip {
        flip_present_v(&mut bytes, width, height);
    }
    let (present_w, present_h, mut bytes) = maybe_crop_present_subwindow(bytes, width, height);
    apply_present_transform(&mut bytes, present_w, present_h, transform);
    make_present_opaque(&mut bytes);
    (present_w, present_h, bytes)
}

fn should_flip_vulkan_present(width: u32, height: u32) -> bool {
    !(width == 1600 && height == 900)
}

fn apply_present_transform(bytes: &mut [u8], width: u32, height: u32, transform: u32) {
    if transform & 0x1 != 0 {
        flip_present_h(bytes, width, height);
    }
    if transform & 0x2 != 0 {
        flip_present_v(bytes, width, height);
    }
}

fn flip_present_h(bytes: &mut [u8], width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let row = w * 4;
    if w == 0 || h == 0 || bytes.len() < row * h {
        return;
    }
    for y in 0..h {
        let base = y * row;
        for x in 0..w / 2 {
            let a = base + x * 4;
            let b = base + (w - 1 - x) * 4;
            for c in 0..4 {
                bytes.swap(a + c, b + c);
            }
        }
    }
}

fn flip_present_v(bytes: &mut [u8], width: u32, height: u32) {
    let row = width as usize * 4;
    let h = height as usize;
    if row == 0 || h == 0 || bytes.len() < row * h {
        return;
    }
    for y in 0..h / 2 {
        let top = y * row;
        let bot = (h - 1 - y) * row;
        let (a, b) = bytes.split_at_mut(bot);
        a[top..top + row].swap_with_slice(&mut b[..row]);
    }
}

fn present_subwindow_has_content(
    pixels: &[u8],
    width: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> bool {
    let mut sampled = 0usize;
    let mut visible = 0usize;
    let mut cell_sampled = [0usize; 9];
    let mut cell_visible = [0usize; 9];
    let row = width as usize * 4;
    for y in (y0..y0.saturating_add(h)).step_by(4) {
        let cell_y = ((y - y0) as usize * 3 / h as usize).min(2);
        let row_off = y as usize * row;
        for x in (x0..x0.saturating_add(w)).step_by(4) {
            let cell_x = ((x - x0) as usize * 3 / w as usize).min(2);
            let cell = cell_y * 3 + cell_x;
            sampled += 1;
            cell_sampled[cell] += 1;
            let p = row_off + x as usize * 4;
            if p + 2 < pixels.len() && pixels[p].max(pixels[p + 1]).max(pixels[p + 2]) > 4 {
                visible += 1;
                cell_visible[cell] += 1;
            }
        }
    }
    if sampled == 0 || visible * 100 < sampled * 15 {
        return false;
    }
    cell_visible
        .iter()
        .zip(cell_sampled.iter())
        .filter(|(visible, sampled)| **sampled != 0 && **visible * 100 >= **sampled)
        .count()
        >= 6
}

fn maybe_crop_present_subwindow(bytes: Vec<u8>, width: u32, height: u32) -> (u32, u32, Vec<u8>) {
    if std::env::var_os("NEXIUM_PRESENT_SUBWINDOW_CROP").is_none() {
        if let Ok(mut slot) = present_crop_slot().lock() {
            *slot = None;
        }
        return (width, height, bytes);
    }
    crop_present_subwindow(bytes, width, height)
}

fn crop_present_subwindow(bytes: Vec<u8>, width: u32, height: u32) -> (u32, u32, Vec<u8>) {
    if width < 1600 || height < 900 || bytes.len() < (width as usize) * (height as usize) * 4 {
        return (width, height, bytes);
    }
    if let Ok(mut slot) = present_crop_slot().lock() {
        if let Some((dst_w, dst_h, x0, y0, w, h)) = *slot {
            if dst_w == width && dst_h == height {
                if !outside_crop_has_visible(&bytes, width, height, x0, y0, w, h) {
                    return (w, h, crop_and_upscale(&bytes, width, x0, y0, w, h, w, h));
                }
                log::debug!(
                    "QueueBuffer Vulkan sub-window invalidated: cached=({},{}) {}x{} target={}x{}",
                    x0,
                    y0,
                    w,
                    h,
                    width,
                    height
                );
                *slot = None;
            }
        }
    }
    let Some((x0, y0, w, h)) = active_bbox(&bytes, width, height) else {
        return (width, height, bytes);
    };
    let area_ratio = (w as f32 * h as f32) / (width as f32 * height as f32);
    let src_aspect = w as f32 / h as f32;
    let dst_aspect = width as f32 / height as f32;
    let aspect_delta = ((src_aspect / dst_aspect) - 1.0).abs();
    let inset = x0 > 4 || y0 > 4 || x0 + w + 4 < width || y0 + h + 4 < height;
    let anchored = x0 <= 4 || y0 <= 4 || x0 + w + 4 >= width || y0 + h + 4 >= height;
    if inset
        && anchored
        && w >= 640
        && h >= 360
        && area_ratio >= 0.30
        && area_ratio <= 0.80
        && aspect_delta <= 0.05
        && present_subwindow_has_content(&bytes, width, x0, y0, w, h)
    {
        if let Ok(mut slot) = present_crop_slot().lock() {
            *slot = Some((width, height, x0, y0, w, h));
        }
        log::debug!(
            "QueueBuffer Vulkan sub-window: src=({},{}) {}x{} -> crop",
            x0,
            y0,
            w,
            h
        );
        (w, h, crop_and_upscale(&bytes, width, x0, y0, w, h, w, h))
    } else {
        (width, height, bytes)
    }
}

#[cfg(test)]
mod present_crop_tests {
    use super::{active_bbox, crop_present_subwindow, present_crop_slot};

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        GUARD
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap()
    }

    fn reset_crop() {
        *present_crop_slot().lock().unwrap() = None;
    }

    fn fill_rect(pixels: &mut [u8], width: u32, x0: u32, y0: u32, w: u32, h: u32, color: [u8; 4]) {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let p = ((y * width + x) * 4) as usize;
                pixels[p..p + 4].copy_from_slice(&color);
            }
        }
    }

    #[test]
    fn hud_only_frame_does_not_latch_present_crop() {
        let _guard = test_guard();
        reset_crop();
        let (width, height) = (1920, 1080);
        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        fill_rect(&mut pixels, width, 64, 900, 1266, 178, [32, 96, 224, 255]);
        fill_rect(&mut pixels, width, 950, 371, 32, 32, [255, 255, 255, 255]);
        assert_eq!(
            active_bbox(&pixels, width, height),
            Some((64, 371, 1266, 707))
        );

        let (present_w, present_h, returned) = crop_present_subwindow(pixels, width, height);
        assert_eq!((present_w, present_h), (width, height));
        assert_eq!(returned.len(), width as usize * height as usize * 4);
        reset_crop();
    }

    #[test]
    fn populated_subwindow_is_still_cropped() {
        let _guard = test_guard();
        reset_crop();
        let (width, height) = (1920, 1080);
        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        fill_rect(&mut pixels, width, 0, 180, 1280, 720, [24, 48, 96, 255]);

        let (present_w, present_h, returned) = crop_present_subwindow(pixels, width, height);
        assert_eq!((present_w, present_h), (1280, 720));
        assert_eq!(returned.len(), 1280 * 720 * 4);
        reset_crop();
    }
}

fn dump_present_frame(bytes: &[u8], width: u32, height: u32) {
    if std::env::var("NEXIUM_PRESENT_DUMP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let every = std::env::var("NEXIUM_PRESENT_DUMP_EVERY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v != 0)
            .unwrap_or(60);
        if seq % every == 0 {
            if let Some(home) = std::env::var_os("APPDATA") {
                let path = std::path::PathBuf::from(home)
                    .join("NeXium")
                    .join("logs")
                    .join(format!("present-{}.bmp", seq));
                if save_rgba_bmp(&path, width, height, bytes).is_ok() {
                    log::warn!("PRESENT DUMP seq={} -> {}", seq, path.display());
                }
            }
        }
    }
}

fn save_rgba_bmp(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let row_bytes = (width as usize) * 3;
    let row_padded = (row_bytes + 3) & !3;
    let pixel_bytes = row_padded * height as usize;
    let file_size = 54 + pixel_bytes;
    let mut f = std::fs::File::create(path)?;
    let mut h = Vec::with_capacity(54);
    h.extend_from_slice(b"BM");
    h.extend_from_slice(&(file_size as u32).to_le_bytes());
    h.extend_from_slice(&[0u8; 4]);
    h.extend_from_slice(&54u32.to_le_bytes());
    h.extend_from_slice(&40u32.to_le_bytes());
    h.extend_from_slice(&width.to_le_bytes());
    h.extend_from_slice(&height.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&24u16.to_le_bytes());
    h.extend_from_slice(&[0u8; 24]);
    f.write_all(&h)?;
    let mut row = vec![0u8; row_padded];
    for y in (0..height as usize).rev() {
        let src_off = y * (width as usize) * 4;
        for x in 0..width as usize {
            let s = src_off + x * 4;
            row[x * 3] = rgba.get(s + 2).copied().unwrap_or(0);
            row[x * 3 + 1] = rgba.get(s + 1).copied().unwrap_or(0);
            row[x * 3 + 2] = rgba.get(s).copied().unwrap_or(0);
        }
        f.write_all(&row)?;
    }
    Ok(())
}

fn try_compose_from_sdl_surface(
    kernel: &Kernel,
    fb_width: u32,
    fb_height: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    const CANDIDATES: &[(u32, u32)] = &[
        (1280, 720),
        (640, 360),
        (854, 480),
        (1920, 1080),
        (1280, 768),
    ];
    for h in kernel.nvdrv.nvmap_handles.values() {
        if h.address == 0 {
            continue;
        }
        let Some(&(width, height)) = CANDIDATES
            .iter()
            .find(|(w, hh)| (*w as u64) * (*hh as u64) * 4 == h.size as u64)
        else {
            continue;
        };
        let mut linear = vec![0u8; h.size as usize];
        if kernel.address_space.read(h.address, &mut linear).is_err() {
            continue;
        }
        let nz = linear.iter().filter(|b| **b != 0).count();
        if nz < 256 {
            continue;
        }
        for px in linear.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
        log::debug!(
            "compose: SDL_Surface candidate nvmap_id={} cpu={:#x} {}x{} nz={}",
            h.id,
            h.address,
            width,
            height,
            nz
        );
        if width == fb_width && height == fb_height {
            return Some((width, height, linear));
        }
        let dst_w = fb_width;
        let dst_h = fb_height;
        let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
        for dy in 0..dst_h {
            let sy = (dy as u64 * height as u64 / dst_h as u64) as u32;
            for dx in 0..dst_w {
                let sx = (dx as u64 * width as u64 / dst_w as u64) as u32;
                let s = ((sy * width + sx) * 4) as usize;
                let d = ((dy * dst_w + dx) * 4) as usize;
                out[d..d + 4].copy_from_slice(&linear[s..s + 4]);
            }
        }
        return Some((dst_w, dst_h, out));
    }
    None
}

fn parse_flattened_graphic_buffer(
    reader: &mut ParcelReader,
) -> Option<nexium_nvdrv::GraphicBuffer> {
    let _length = reader.read_u32()?;
    let _fd_count = reader.read_u32()?;
    let _magic = reader.read_u32();
    let width = reader.read_u32().unwrap_or(0);
    let height = reader.read_u32().unwrap_or(0);
    let stride = reader.read_u32().unwrap_or(0);
    let format = reader.read_u32().unwrap_or(0);
    let usage = reader.read_u32().unwrap_or(0);

    let _pid = reader.read_u32();
    let _refcount = reader.read_u32();
    let _num_fds = reader.read_u32();
    let num_ints = reader.read_u32().unwrap_or(0) as usize;

    let mut ints = Vec::with_capacity(num_ints);
    for _ in 0..num_ints {
        ints.push(reader.read_u32().unwrap_or(0));
    }
    let inline_nvmap_id = ints.get(1).copied().unwrap_or(0);
    let binder_handle = reader.first_binder_handle().unwrap_or(0);
    let nvmap_id = if binder_handle != 0 {
        binder_handle
    } else if inline_nvmap_id != 0 {
        inline_nvmap_id
    } else {
        ints.get(19).copied().unwrap_or(0)
    };
    let buffer_offset = ints.get(20).copied().unwrap_or(0);
    let kind = ints.get(21).copied().unwrap_or(0);
    let block_height_log2 = ints.get(22).copied().unwrap_or(4);

    log::debug!(
        "parse_flattened_graphic_buffer: nvmap_id={} (binder_handle={} inline={}) off={:#x} kind={} bh_log2={}",
        nvmap_id,
        binder_handle,
        inline_nvmap_id,
        buffer_offset,
        kind,
        block_height_log2
    );

    Some(nexium_nvdrv::GraphicBuffer {
        width,
        height,
        stride,
        format,
        usage,
        kind,
        nvmap_id,
        buffer_offset: buffer_offset as u64,
        size: stride * height * 4,
        block_height_log2,
    })
}

fn deferred_ctrl_wait_event_id(
    ioctl_cmd: u16,
    ioctl_result: u32,
    input_event_id: Option<u32>,
    output: &[u8],
) -> Option<u32> {
    if ioctl_result != 5 {
        return None;
    }

    match ioctl_cmd {
        0x001d => output
            .get(12..16)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        0x001e => input_event_id,
        _ => None,
    }
}

#[cfg(test)]
mod ctrl_wait_event_tests {
    use super::deferred_ctrl_wait_event_id;

    #[test]
    fn allocation_wait_uses_returned_event_id() {
        let event_id = 0x102a_0017u32;
        let mut output = [0u8; 16];
        output[12..16].copy_from_slice(&event_id.to_le_bytes());

        assert_eq!(
            deferred_ctrl_wait_event_id(0x001d, 5, Some(0xdead_beef), &output),
            Some(event_id)
        );
    }

    #[test]
    fn async_wait_uses_input_event_id() {
        assert_eq!(
            deferred_ctrl_wait_event_id(0x001e, 5, Some(23), &[]),
            Some(23)
        );
    }

    #[test]
    fn successful_or_malformed_wait_does_not_arm_an_event() {
        assert_eq!(
            deferred_ctrl_wait_event_id(0x001d, 0, Some(7), &[0u8; 16]),
            None
        );
        assert_eq!(
            deferred_ctrl_wait_event_id(0x001d, 5, Some(7), &[0u8; 15]),
            None
        );
        assert_eq!(
            deferred_ctrl_wait_event_id(0x001e, 5, None, &[0u8; 16]),
            None
        );
    }
}

fn dispatch_nvdrv_command(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, port_name: &str) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    log::trace!("nvdrv:{}.cmd_{}", port_name, cmd_id);

    match cmd_id {
        0 => {
            let buf_src = ctx
                .send_statics
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
                .or_else(|| {
                    ctx.send_buffers
                        .iter()
                        .find(|b| b.size > 0 && b.addr != 0)
                        .copied()
                });
            let path = if let Some(sb) = buf_src {
                let mut buf = vec![0u8; sb.size as usize];
                let _ = kernel.address_space.read(sb.addr, &mut buf);
                let trimmed = buf.split(|&b| b == 0).next().unwrap_or(&buf);
                String::from_utf8_lossy(trimmed).into_owned()
            } else {
                String::new()
            };
            log::debug!(
                "nvdrv:Open path='{}' (sb={:?})",
                path,
                buf_src.map(|b| (b.addr, b.size))
            );
            let fd = kernel.nvdrv.open(&path).unwrap_or(0);
            let mut out = Vec::new();
            out.extend_from_slice(&fd.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            build_ipc_response(ctx, 0, &out, &[])
        }
        1 | 11 | 12 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else {
                0
            };
            let ioctl_id = if ctx.cmif_in_data_len >= 8 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off + 4],
                    ctx.buf[ctx.cmif_in_data_off + 5],
                    ctx.buf[ctx.cmif_in_data_off + 6],
                    ctx.buf[ctx.cmif_in_data_off + 7],
                ])
            } else {
                0
            };
            let ioctl_cmd = (ioctl_id & 0xFFFF) as u16;

            let in_srcs: Vec<_> = ctx
                .send_buffers
                .iter()
                .chain(ctx.send_statics.iter())
                .filter(|b| b.size > 0 && b.addr != 0)
                .copied()
                .collect();
            let read_input = |buf: Option<ipc::IpcBuffer>| -> Vec<u8> {
                let Some(sb) = buf else {
                    return Vec::new();
                };
                let mut data = vec![0u8; sb.size as usize];
                let _ = kernel.address_space.read(sb.addr, &mut data);
                data
            };
            let in_data = read_input(in_srcs.first().copied());
            let inline_in_data = if cmd_id == 11 {
                read_input(in_srcs.get(1).copied())
            } else {
                Vec::new()
            };

            let out_dsts: Vec<_> = ctx
                .recv_buffers
                .iter()
                .chain(ctx.recv_statics.iter())
                .filter(|b| b.size > 0 && b.addr != 0)
                .copied()
                .collect();
            let out_dst = out_dsts.first().copied();
            let out_size = out_dst.map(|b| b.size as usize).unwrap_or(0);

            if cmd_id == 1 {
                log::trace!(
                    "nvdrv:Ioctl fd={} ioctl_id={:#x} send_buf={:?} send_static={:?} recv_buf={:?}",
                    fd,
                    ioctl_id,
                    ctx.send_buffers
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>(),
                    ctx.send_statics
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>(),
                    ctx.recv_buffers
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>()
                );
            }
            if ioctl_cmd == 0x4808 || ioctl_cmd == 0x481b {
                log::trace!(
                    "nvdrv:SubmitGPFIFO ioctl cmd_id={} fd={} ioctl={:#x} in={} inline={} recv={:?}",
                    cmd_id,
                    fd,
                    ioctl_id,
                    in_data.len(),
                    inline_in_data.len(),
                    out_dst.map(|b| (b.addr, b.size))
                );
            }

            let ctrl_event_id_in = in_data
                .get(12..16)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
            let ctrl_cancel_id_in = in_data
                .get(0..4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
            let req = nexium_nvdrv::IoctlRequest {
                fd,
                ioctl_id,
                in_data,
                inline_in_data,
                out_size,
            };
            let addr_space = kernel.address_space.clone();
            let addr_space_w = kernel.address_space.clone();
            let addr_space_c = kernel.address_space.clone();
            let ioctl_profile = if crate::kernel::profile::enabled()
                && std::env::var_os("NEXIUM_PROFILE_IOCTL").is_some()
            {
                let device = kernel
                    .nvdrv
                    .device_for_fd(fd)
                    .map(|d| format!("{:?}", d))
                    .unwrap_or_else(|| "invalid".to_string());
                Some((
                    std::time::Instant::now(),
                    format!("nvdrv.{}.{:#06x}", device, (ioctl_id & 0xFFFF) as u16),
                ))
            } else {
                None
            };
            let outcome = kernel.nvdrv.dispatch_ioctl_with_mem_and_copy(
                req,
                &|addr, buf| addr_space.read(addr, buf).is_ok(),
                &|addr, buf| addr_space_w.write(addr, buf).is_ok(),
                &|src, dst, len| addr_space_c.copy(src, dst, len).is_ok(),
            );
            if let Some((start, key)) = ioctl_profile {
                crate::kernel::profile::record_ipc(&key, start);
            }

            if !outcome.data.is_empty() {
                if let Some(buf) = out_dst {
                    let n = outcome.data.len().min(buf.size as usize);
                    let _ = kernel.address_space.write(buf.addr, &outcome.data[..n]);
                }
                if cmd_id == 12 {
                    if let Some(buf) = out_dsts.get(1).copied() {
                        let inline = match ioctl_cmd {
                            0x4705 if outcome.data.len() > 16 => &outcome.data[16..],
                            0x4706 if outcome.data.len() >= 20 => &outcome.data[16..20],
                            _ => &[],
                        };
                        if !inline.is_empty() {
                            let n = inline.len().min(buf.size as usize);
                            let _ = kernel.address_space.write(buf.addr, &inline[..n]);
                            log::debug!(
                                "nvdrv:Ioctl3 inline out ioctl={:#x} wrote {} bytes to {:#x}",
                                ioctl_id,
                                n,
                                buf.addr
                            );
                        }
                    }
                }
            }

            if kernel.nvdrv.device_for_fd(fd) == Some(nexium_nvdrv::NvDevice::NvhostCtrl) {
                if let Some(event_id) = deferred_ctrl_wait_event_id(
                    ioctl_cmd,
                    outcome.result,
                    ctrl_event_id_in,
                    &outcome.data,
                ) {
                    if crate::kernel::Kernel::fence_profile_enabled() {
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static TRACKED: AtomicU64 = AtomicU64::new(0);
                        static UNTRACKED: AtomicU64 = AtomicU64::new(0);
                        let tracked = kernel.gpu_event_tokens.contains_key(&(fd, event_id & 0xFF))
                            && kernel.nvdrv.ctrl_event_wait(fd, event_id).is_some();
                        let total = if tracked {
                            TRACKED.fetch_add(1, Ordering::Relaxed)
                                + 1
                                + UNTRACKED.load(Ordering::Relaxed)
                        } else {
                            UNTRACKED.fetch_add(1, Ordering::Relaxed)
                                + 1
                                + TRACKED.load(Ordering::Relaxed)
                        };
                        if total % 256 == 0 {
                            log::warn!(
                                "[fence-arm] tracked={} untracked={}",
                                TRACKED.load(Ordering::Relaxed),
                                UNTRACKED.load(Ordering::Relaxed)
                            );
                        }
                    }
                    if let (Some(&handle), Some(wait)) = (
                        crate::kernel::Kernel::fence_signal_fix_enabled()
                            .then(|| kernel.gpu_event_tokens.get(&(fd, event_id & 0xFF)))
                            .flatten(),
                        kernel.nvdrv.ctrl_event_wait(fd, event_id),
                    ) {
                        if kernel
                            .nvdrv
                            .is_syncpoint_reached(wait.syncpt_id, wait.threshold)
                        {
                            kernel.event_signals.insert(handle, true);
                            kernel.threads.signal_handle(handle);
                        } else {
                            kernel.event_signals.insert(handle, false);
                            kernel
                                .gpu_fence_events
                                .insert(handle, (wait.syncpt_id, wait.threshold));
                            kernel.record_fence_armed(handle);
                        }
                    }
                }
            }

            if kernel.nvdrv.device_for_fd(fd) == Some(nexium_nvdrv::NvDevice::NvhostCtrl)
                && ioctl_cmd == 0x001c
            {
                if let Some(event_id) = ctrl_cancel_id_in {
                    if let Some(&handle) = kernel.gpu_event_tokens.get(&(fd, event_id & 0xFF)) {
                        kernel.gpu_fence_events.remove(&handle);
                        kernel.event_signals.insert(handle, false);
                    }
                }
            }

            {
                let fence_handles: Vec<u32> = kernel
                    .gpu_fence_events
                    .iter()
                    .filter_map(|(&handle, &(syncpt_id, threshold))| {
                        kernel
                            .nvdrv
                            .is_syncpoint_reached(syncpt_id, threshold)
                            .then_some(handle)
                    })
                    .collect();
                for fh in fence_handles {
                    kernel.gpu_fence_events.remove(&fh);
                    kernel.record_fence_signal(fh);
                    kernel.event_signals.insert(fh, true);
                    kernel.threads.signal_handle(fh);
                    log::debug!(
                        "nvdrv:SubmitGPFIFO â†’ signaling gpu_fence_event handle={:#x}",
                        fh
                    );
                }
            }

            build_ipc_response(ctx, 0, &outcome.result.to_le_bytes(), &[])
        }
        2 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else {
                0
            };
            kernel.nvdrv.close(fd);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        3 => {
            log::debug!("nvdrv:Initialize");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        4 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else {
                0
            };
            let event_id = if ctx.cmif_in_data_len >= 8 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off + 4],
                    ctx.buf[ctx.cmif_in_data_off + 5],
                    ctx.buf[ctx.cmif_in_data_off + 6],
                    ctx.buf[ctx.cmif_in_data_off + 7],
                ])
            } else {
                0
            };
            let is_nvhost_ctrl_fd = kernel
                .nvdrv
                .files
                .get(&fd)
                .map(|f| f.device == nexium_nvdrv::NvDevice::NvhostCtrl)
                .unwrap_or(false);
            let h = kernel.handles.create_handle(HandleType::Event);
            if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
                log::info!(
                    "[syncpt] query-event fd={} event_id={:#x} ctrl={}",
                    fd,
                    event_id,
                    is_nvhost_ctrl_fd
                );
            }
            if is_nvhost_ctrl_fd {
                if crate::kernel::Kernel::fence_signal_fix_enabled() {
                    kernel.gpu_event_tokens.insert((fd, event_id & 0xFF), h);
                }
                let wait = kernel.nvdrv.ctrl_event_wait(fd, event_id);
                let signaled = wait
                    .map(|wait| {
                        kernel
                            .nvdrv
                            .is_syncpoint_reached(wait.syncpt_id, wait.threshold)
                    })
                    .unwrap_or(false);
                kernel.event_signals.insert(h, signaled);
                if let Some(wait) = wait.filter(|_| !signaled) {
                    kernel
                        .gpu_fence_events
                        .insert(h, (wait.syncpt_id, wait.threshold));
                    kernel.record_fence_armed(h);
                }
                log::debug!(
                    "nvdrv:QueryEvent fd={} event_id={:#x} (nvhost-ctrl) â†’ fence event handle={:#x} signaled={}",
                    fd,
                    event_id,
                    h,
                    signaled
                );
            } else {
                kernel.event_signals.insert(h, false);
                log::debug!(
                    "nvdrv:QueryEvent fd={} event_id={:#x} â†’ event handle={:#x} (unsignaled)",
                    fd,
                    event_id,
                    h
                );
            }
            build_ipc_response_copy(ctx, 0, &0u32.to_le_bytes(), &[h])
        }
        8 => {
            log::debug!("nvdrv:SetClientPID");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        13 => {
            log::debug!("nvdrv:GetStatus");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        other => {
            log::debug!("nvdrv: unknown cmd={}", other);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
    }
}

fn applet_buffer_response(port_name: &str, cmd_id: u32) -> Option<Vec<u8>> {
    match (port_name, cmd_id) {
        ("IApplicationDisplayService", 2020)
        | ("IApplicationDisplayService", 2030)
        | ("IManagerDisplayService", 2012) => Some(build_native_window_parcel(0x100)),
        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => Some(build_igbp_success_parcel()),
        ("ILaunchParamStorageAccessor", 11) => Some(build_launch_parameter()),
        ("IStorageAccessorOut", 11) => Some(crate::services::am::applet_out_data()),
        ("acc:u0" | "acc:u1" | "acc:aa", 2) | ("acc:u0" | "acc:u1" | "acc:aa", 3) => {
            Some(build_user_id_list())
        }
        ("IProfile", 0) => Some(vec![0u8; 0x80]),
        _ => None,
    }
}

fn build_launch_parameter() -> Vec<u8> {
    let mut out = vec![0u8; 0x88];
    out[0..4].copy_from_slice(&0xC794_97CAu32.to_le_bytes());
    out[4..8].copy_from_slice(&1u32.to_le_bytes());
    out[8..24].copy_from_slice(&crate::services::am::ACCOUNT_UID);
    out
}

fn build_user_id_list() -> Vec<u8> {
    let mut out = vec![0u8; 0x80];
    out[0..16].copy_from_slice(&crate::services::am::ACCOUNT_UID);
    out
}

fn build_igbp_success_parcel() -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1280u32.to_le_bytes());
    payload.extend_from_slice(&720u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&2u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::with_capacity(16 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn build_native_window_parcel(binder_handle: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0x2u32.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&binder_handle.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(b"dispdrv\0");
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::with_capacity(16 + payload.len() + 4);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

pub(crate) fn return_subsession(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    sub_service: &str,
) -> Vec<u8> {
    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    if is_domain {
        let object_id = alloc_domain_object(kernel, session_handle, sub_service);
        log::debug!("â†’ {} sub-object id={}", sub_service, object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, sub_service.to_string());
        kernel.sessions.insert(h, session);
        log::debug!("â†’ {} sub-session handle={:#x}", sub_service, h);
        build_ipc_response(ctx, 0, &[], &[h])
    }
}

fn return_file_system_with_root(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    root: std::path::PathBuf,
) -> Vec<u8> {
    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    if is_domain {
        let object_id = alloc_domain_object(kernel, session_handle, "IFileSystem");
        for handle in domain_group_handles(kernel, session_handle) {
            kernel
                .file_system_roots
                .insert((handle, object_id), root.clone());
        }
        log::debug!("-> IFileSystem sub-object id={}", object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, "IFileSystem".to_string());
        kernel.sessions.insert(h, session);
        kernel.file_system_roots.insert((h, 0), root);
        log::debug!("-> IFileSystem sub-session handle={:#x}", h);
        build_ipc_response(ctx, 0, &[], &[h])
    }
}

const BCAT_NO_OPEN_ENTRY: u32 = 122 | (7 << 9);

fn aoc_base_title_id(title_id: u64) -> u64 {
    (title_id & !0xfff) + 0x1000
}

fn persistent_event(kernel: &mut Kernel, slot: &mut Option<u32>, signalled: bool) -> u32 {
    if let Some(h) = *slot {
        return h;
    }
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, signalled);
    *slot = Some(h);
    h
}

fn write_first_recv_target(kernel: &mut Kernel, ctx: &ipc::IpcCtx, bytes: &[u8]) -> bool {
    let target = ctx
        .recv_statics
        .iter()
        .chain(ctx.recv_buffers.iter())
        .find(|b| b.size > 0 && b.addr != 0)
        .copied();
    let Some(buf) = target else {
        return false;
    };
    let n = (buf.size as usize).min(bytes.len());
    kernel.address_space.write(buf.addr, &bytes[..n]).is_ok()
}

fn delivery_cache_progress_blob() -> Vec<u8> {
    let mut blob = vec![0u8; 0x200];
    blob[0..4].copy_from_slice(&9u32.to_le_bytes());
    blob
}

fn dispatch_aoc_bcat(
    kernel: &mut Kernel,
    port_name: &str,
    ctx: &mut ipc::IpcCtx,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    match port_name {
        "aoc:u" => match cmd_id {
            0 | 1 | 2 | 3 => Some(build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])),
            4 | 5 => {
                let base = aoc_base_title_id(kernel.title_id);
                Some(build_ipc_response(ctx, 0, &base.to_le_bytes(), &[]))
            }
            6 | 7 | 11 | 12 | 50 | 200 | 300 | 302 => Some(build_ipc_response(ctx, 0, &[], &[])),
            8 | 10 => {
                let mut slot = kernel.aoc_change_event;
                let h = persistent_event(kernel, &mut slot, false);
                kernel.aoc_change_event = slot;
                Some(build_ipc_response_copy(ctx, 0, &[], &[h]))
            }
            9 => Some(build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[])),
            13 => Some(build_ipc_response(ctx, 0, &[0u8], &[])),
            301 => Some(build_ipc_response(ctx, 0, &[], &[])),
            _ => None,
        },
        "IPurchaseEventManager" => match cmd_id {
            0 | 1 | 2 => Some(build_ipc_response(ctx, 0, &[], &[])),
            3 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                kernel.event_signals.insert(h, false);
                Some(build_ipc_response_copy(ctx, 0, &[], &[h]))
            }
            4 => Some(build_ipc_response(ctx, 0, &[], &[])),
            _ => None,
        },
        "bcat:u" | "bcat:a" | "bcat:m" | "bcat:s" => None,
        "IBcatService" => match cmd_id {
            10200 | 20301 | 20400 | 20401 | 20410 | 30100 | 30200 | 30201 | 30202 | 30203
            | 30210 | 30300 | 90202 | 90301 => Some(build_ipc_response(ctx, 0, &[], &[])),
            90201 => Some(build_ipc_response(ctx, 0, &[], &[])),
            90100 | 90200 => Some(build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])),
            90300 => Some(build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])),
            _ => None,
        },
        "IDeliveryCacheProgressService" => match cmd_id {
            0 => {
                let mut slot = kernel.bcat_progress_event;
                let h = persistent_event(kernel, &mut slot, true);
                kernel.bcat_progress_event = slot;
                Some(build_ipc_response_copy(ctx, 0, &[], &[h]))
            }
            1 => {
                let blob = delivery_cache_progress_blob();
                if !write_first_recv_target(kernel, ctx, &blob) {
                    log::warn!("IDeliveryCacheProgressService.Get: no output buffer");
                }
                Some(build_ipc_response(ctx, 0, &[], &[]))
            }
            _ => None,
        },
        "IDeliveryCacheStorageService" => match cmd_id {
            10 => Some(build_ipc_response(ctx, 0, &[0u8; 16], &[])),
            _ => None,
        },
        "IDeliveryCacheDirectoryService" => match cmd_id {
            0 => Some(build_ipc_response(ctx, BCAT_NO_OPEN_ENTRY, &[], &[])),
            1 | 2 => Some(build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])),
            _ => None,
        },
        "IDeliveryCacheFileService" => match cmd_id {
            0 => Some(build_ipc_response(ctx, BCAT_NO_OPEN_ENTRY, &[], &[])),
            1 => Some(build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[])),
            2 => Some(build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[])),
            3 => Some(build_ipc_response(ctx, 0, &[0u8; 16], &[])),
            _ => None,
        },
        "IDeliveryCacheStorageUpdateNotifier" => match cmd_id {
            0 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                kernel.event_signals.insert(h, false);
                Some(build_ipc_response_copy(ctx, 0, &[], &[h]))
            }
            _ => None,
        },
        _ => None,
    }
}

fn subsession_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("hid", 0) => Some("IAppletResource"),
        ("IApplicationCreator", 0) => Some("IApplicationAccessor"),
        ("ILibraryAppletCreator", 0) => Some("ILibraryAppletAccessor"),
        ("time:s" | "time:u" | "time:a" | "time:r", 0) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 1) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 2) => Some("ISteadyClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 3) => Some("ITimeZoneService"),
        ("time:s" | "time:u" | "time:a" | "time:r", 4) => Some("ISystemClock"),
        ("friend:u" | "friend:a" | "friend:s" | "friend:v" | "friend:m", 0) => {
            Some("IFriendService")
        }
        ("mii:u" | "mii:e", 0) => Some("IDatabaseService"),
        ("nfp:user", 0) => Some("INfpUser"),
        ("bcat:u" | "bcat:a" | "bcat:m" | "bcat:s", 0) => Some("IBcatService"),
        ("bcat:u" | "bcat:a" | "bcat:m" | "bcat:s", 1 | 2) => Some("IDeliveryCacheStorageService"),
        ("bcat:u" | "bcat:a" | "bcat:m" | "bcat:s", 3 | 4) => Some("IDeliveryCacheProgressService"),
        ("IBcatService", 10100 | 10101 | 20100 | 20101) => Some("IDeliveryCacheProgressService"),
        ("IBcatService", 20300) => Some("IDeliveryCacheStorageUpdateNotifier"),
        ("IDeliveryCacheStorageService", 0) => Some("IDeliveryCacheFileService"),
        ("IDeliveryCacheStorageService", 1) => Some("IDeliveryCacheDirectoryService"),
        ("aoc:u", 100 | 101) => Some("IPurchaseEventManager"),
        ("fsp-srv", 18) => Some("IFileSystem"),
        ("fsp-srv", 200) => Some("IFsStorage"),
        ("vi:m" | "vi:s" | "vi:u", 0) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 1) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 2) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 3) => Some("IApplicationDisplayService"),
        ("IApplicationDisplayService", 100) => Some("IHOSBinderDriver"),
        ("IApplicationDisplayService", 101) => Some("ISystemDisplayService"),
        ("IApplicationDisplayService", 102) => Some("IManagerDisplayService"),
        ("IApplicationDisplayService", 103) => Some("IHOSBinderDriver"),
        ("appletAE" | "appletOE", 0) => Some("IApplicationProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        ("apm" | "apm:p", 0) => Some("IApmManager"),
        ("IApmManager", 0) => Some("IApmSession"),
        ("pctl:a" | "pctl:r" | "pctl:s" | "pctl", 0) => Some("IParentalControlService"),
        ("pctl:a" | "pctl:r" | "pctl:s" | "pctl", 1) => Some("IParentalControlService"),
        _ => None,
    }
}

fn applet_command_response(
    _kernel: &mut Kernel,
    port_name: &str,
    cmd_id: u32,
) -> Option<(Vec<u8>, Option<u32>)> {
    match (port_name, cmd_id) {
        ("IDebugFunctions", _) => Some((Vec::new(), None)),

        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => Some((Vec::new(), None)),

        ("IFileSystem", _) => Some((Vec::new(), None)),
        ("fsp-srv", _) => Some((Vec::new(), None)),

        ("psm", _) => Some((Vec::new(), None)),
        ("set", _) | ("set:sys", _) => Some((Vec::new(), None)),
        ("nvdrv:a", _) | ("nvdrv", _) | ("nvdrv:s", _) | ("nvdrv:t", _) => {
            Some((0u32.to_le_bytes().to_vec(), None))
        }

        ("IParentalControlService", cmd) => {
            let out: Vec<u8> = match cmd {
                1031 | 1061 | 1403 | 1453 | 1455 | 1458 => vec![0u8],
                1018 | 1065 => vec![1u8],
                1032 | 1039 | 1206 => 0u32.to_le_bytes().to_vec(),
                _ => Vec::new(),
            };
            Some((out, None))
        }

        _ => None,
    }
}

fn applet_proxy_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("appletAE" | "appletOE", 100) => Some("ISystemAppletProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        ("appletAE" | "appletOE", 300) => Some("IOverlayAppletProxy"),
        ("appletAE" | "appletOE", 350) => Some("IApplicationProxy"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            0,
        ) => Some("ICommonStateGetter"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            1,
        ) => Some("ISelfController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            2,
        ) => Some("IWindowController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            3,
        ) => Some("IAudioController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            4,
        ) => Some("IDisplayController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            10,
        ) => Some("IProcessWindingController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            11,
        ) => Some("ILibraryAppletCreator"),
        ("ISystemAppletProxy", 20) => Some("IApplicationFunctions"),
        ("ISystemAppletProxy", 21) => Some("IHomeMenuFunctions"),
        ("ISystemAppletProxy", 22) => Some("IGlobalStateController"),
        ("ISystemAppletProxy", 23) => Some("IApplicationCreator"),
        ("IApplicationProxy", 20) => Some("IApplicationFunctions"),
        ("IApplicationProxy", 1000) => Some("IDebugFunctions"),
        ("ISystemAppletProxy", 1000) => Some("IDebugFunctions"),
        ("ILibraryAppletProxy", 1000) => Some("IDebugFunctions"),
        ("IOverlayAppletProxy", 1000) => Some("IDebugFunctions"),
        _ => None,
    }
}

fn dispatch_sm_command(
    kernel: &mut Kernel,
    cmd_id: u32,
    tls_buf: &[u8],
    cmif_data_off: usize,
    cmif_data_len: usize,
    parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    match cmd_id {
        0 => dispatch_sm_register_client(kernel, tls_buf, cmif_data_off, cmif_data_len, parsed_ctx),
        1 => dispatch_sm_get_service_handle(
            kernel,
            tls_buf,
            cmif_data_off,
            cmif_data_len,
            parsed_ctx,
        ),
        2 => dispatch_sm_register_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        3 => dispatch_sm_unregister_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        _ => {
            log::warn!("unknown SM command: {}", cmd_id);
            (1, Vec::new())
        }
    }
}

fn dispatch_sm_register_client(
    _kernel: &mut Kernel,
    _tls_buf: &[u8],
    _cmif_data_off: usize,
    _cmif_data_len: usize,
    parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    let pid = parsed_ctx.and_then(|ctx| ctx.send_pid);
    log::debug!("SM::RegisterClient pid={:?}", pid);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_register_service(
    _kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };
    log::debug!("SM::RegisterService '{}'", service_name);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_unregister_service(
    _kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };
    log::debug!("SM::UnregisterService '{}'", service_name);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_get_service_handle(
    kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
    _parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };

    log::debug!(
        "SM::GetServiceHandle '{}' data_off={:#x}",
        service_name,
        cmif_data_off
    );

    let handle = kernel.handles.create_handle(HandleType::Session);
    let final_name = if !service_name.is_empty() {
        service_name
    } else {
        "unknown".to_string()
    };
    let session = Session::new(handle, final_name.clone());
    kernel.sessions.insert(handle, session);

    log::debug!(
        "SM: returning handle {:#x} for service '{}'",
        handle,
        final_name
    );

    let mut response = Vec::new();
    response.extend_from_slice(&handle.to_le_bytes());
    (SUCCESS, response)
}

fn write_ipc_response(buf: &mut [u8], data_offset: usize, result: u32, token: u32) {
    write_ipc_response_with_data(buf, data_offset, result, token, &[]);
}

fn write_ipc_response_with_data(
    buf: &mut [u8],
    data_offset: usize,
    result: u32,
    token: u32,
    out_data: &[u8],
) {
    let hipc_resp: u64 = 0x0000_0004_0000_0000;
    buf[0..8].copy_from_slice(&hipc_resp.to_le_bytes());

    let off = (data_offset + 3) & !3;
    if buf.len() >= off + 16 {
        buf[off..off + 4].copy_from_slice(b"SFCO");
        buf[off + 4..off + 8].copy_from_slice(&0u32.to_le_bytes());
        buf[off + 8..off + 12].copy_from_slice(&result.to_le_bytes());
        buf[off + 12..off + 16].copy_from_slice(&token.to_le_bytes());

        if !out_data.is_empty() && off + 16 + out_data.len() <= buf.len() {
            buf[off + 16..off + 16 + out_data.len()].copy_from_slice(out_data);
        }
    }
}

fn svc_get_thread_id(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1) as u32
    } else {
        0
    };
    let target = if handle == 0 || handle == 0xFFFF8000 {
        kernel
            .threads
            .current_handle()
            .unwrap_or(kernel.main_thread_handle)
    } else {
        handle
    };
    let tid = kernel
        .threads
        .threads
        .get(&target)
        .map(|t| t.tid)
        .unwrap_or(1);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, tid);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_id(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0x4F4F4F4F_4F4F4F4F);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_clear_event(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    kernel.event_signals.insert(handle, false);
    kernel.refresh_bufferqueue_event(handle);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reset_signal(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    let is_event = matches!(
        kernel.handles.get_handle(handle),
        Some(entry) if entry.handle_type == HandleType::Event
    );
    let result = reset_event_signal(&mut kernel.event_signals, is_event, handle);
    if result == SUCCESS {
        kernel.refresh_bufferqueue_event(handle);
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, result as u64);
    }
    result
}

fn svc_wait_for_address(kernel: &mut Kernel) -> u32 {
    let (addr, arb_type, value, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1) as u32,
            cpu.get_register(2) as u32,
            cpu.get_register(3),
        )
    } else {
        return 1;
    };

    let (read_ok, current, should_wait) = loop {
        let Ok(current) = kernel.address_space.atomic_load_u32(addr) else {
            break (false, 0, false);
        };
        let should_wait = match arb_type {
            0 | 1 => (current as i32) < (value as i32),
            2 => current == value,
            _ => false,
        };
        if arb_type == 1 && should_wait {
            match kernel
                .address_space
                .atomic_cas_u32(addr, current, current.wrapping_sub(1))
            {
                Ok(true) => break (true, current, true),
                Ok(false) => continue,
                Err(_) => break (false, current, false),
            }
        }
        break (true, current, should_wait);
    };
    let trace_arbiter = std::env::var_os("NEXIUM_ARBITER_TRACE").is_some();
    if trace_arbiter {
        log::warn!(
            "[arb-wait] addr={:#x} type={} value={} timeout_ns={} read_ok={} current={} should_wait={}",
            addr,
            arb_type,
            value,
            timeout_ns,
            read_ok,
            current,
            should_wait
        );
    }

    if !read_ok || !should_wait {
        const KERNEL_INVALID_STATE: u32 = 1 | (125 << 9);
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_STATE as u64);
        }
        return KERNEL_INVALID_STATE;
    }

    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if timeout_ns == 0 {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_TIMEOUT as u64);
        }
        return KERNEL_TIMEOUT;
    }

    let wake_at = if timeout_ns == u64::MAX {
        None
    } else {
        Some(std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns))
    };
    if let Some(cpu) = cpu_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingArbiter {
                addr,
                value,
                wake_at,
            },
        );
    }
    kernel.yield_after_svc = true;

    SUCCESS
}

fn svc_signal_to_address(kernel: &mut Kernel) -> u32 {
    let (addr, signal_type, value, count) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1) as u32,
            cpu.get_register(2) as u32,
            cpu.get_register(3) as i32,
        )
    } else {
        return 1;
    };

    let mut buf = [0u8; 4];
    let _ = kernel.address_space.read(addr, &mut buf);
    let current = u32::from_le_bytes(buf);
    let trace_arbiter = std::env::var_os("NEXIUM_ARBITER_TRACE").is_some();
    if trace_arbiter {
        let waiters = kernel
            .threads
            .threads
            .values()
            .filter(|thread| {
                matches!(
                    &thread.state,
                    crate::kernel::threads::ThreadState::WaitingArbiter { addr: a, .. }
                        if *a == addr
                )
            })
            .count();
        log::warn!(
            "[arb-signal] addr={:#x} type={} value={} count={} current={} waiters={}",
            addr,
            signal_type,
            value,
            count,
            current,
            waiters
        );
    }

    const KERNEL_INVALID_STATE: u32 = 1 | (125 << 9);

    match signal_type {
        0 => {}
        1 => {
            let ok = matches!(
                kernel
                    .address_space
                    .atomic_cas_u32(addr, value, value.wrapping_add(1)),
                Ok(true)
            );
            if !ok {
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, KERNEL_INVALID_STATE as u64);
                }
                return KERNEL_INVALID_STATE;
            }
        }
        2 => {
            let waiters = kernel
                .threads
                .threads
                .values()
                .filter(|thread| {
                    matches!(
                        &thread.state,
                        crate::kernel::threads::ThreadState::WaitingArbiter { addr: a, .. }
                            if *a == addr
                    )
                })
                .count() as i32;
            let new_value = if waiters == 0 {
                value.wrapping_add(1)
            } else if count <= 0 {
                value.wrapping_sub(2)
            } else if waiters <= count {
                value.wrapping_sub(1)
            } else {
                value
            };
            let ok = matches!(
                kernel.address_space.atomic_cas_u32(addr, value, new_value),
                Ok(true)
            );
            if !ok {
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, KERNEL_INVALID_STATE as u64);
                }
                return KERNEL_INVALID_STATE;
            }
        }
        _ => {}
    }

    if count <= 0 {
        if kernel.threads.wake_all_on_arbiter(addr) > 0 {
            kernel.yield_after_svc = true;
        }
    } else {
        for _ in 0..count {
            match kernel.threads.wake_one_on_arbiter(addr) {
                Some(woken) => nudge_preempt_for_wake(kernel, woken),
                None => break,
            }
        }
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_break(kernel: &mut Kernel) -> u32 {
    let reason = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        0
    };
    let info_va = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1)
    } else {
        0
    };
    let info_size = if let Some(cpu) = cpu_ref() {
        cpu.get_register(2) as usize
    } else {
        0
    };

    log::warn!(
        "svcBreak: reason={:#x}, info_va={:#x}, info_size={:#x}",
        reason,
        info_va,
        info_size
    );

    if let Some(cpu) = cpu_ref() {
        let pc = cpu.get_pc();
        let lr = cpu.get_register(30);
        let sp = cpu.get_register(31);
        let fp = cpu.get_register(29);

        log::warn!("  PC={:#x}, LR={:#x}, SP={:#x}, FP={:#x}", pc, lr, sp, fp);

        let mut callers = Vec::new();
        let mut cur_fp = fp;
        for i in 0..8 {
            if cur_fp == 0 || cur_fp & 7 != 0 {
                break;
            }
            let mut frame = [0u8; 16];
            if kernel.address_space.read(cur_fp, &mut frame).is_err() {
                break;
            }
            let next_fp = u64::from_le_bytes([
                frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7],
            ]);
            let saved_lr = u64::from_le_bytes([
                frame[8], frame[9], frame[10], frame[11], frame[12], frame[13], frame[14],
                frame[15],
            ]);
            callers.push((i, saved_lr));
            if next_fp <= cur_fp || next_fp.saturating_sub(cur_fp) > 0x10_0000 {
                break;
            }
            cur_fp = next_fp;
        }

        for (i, addr) in &callers {
            log::warn!(
                "  Stack[{}]: {:#x} (offset {:#x})",
                i,
                addr,
                addr.wrapping_sub(kernel.code_base)
            );
        }

        if info_size > 0 && info_size <= 0x1000 {
            let mut info_buf = vec![0u8; info_size.min(0x80)];
            if kernel.address_space.read(info_va, &mut info_buf).is_ok() {
                log::warn!("  Info buffer: {:02x?}", &info_buf);
            }
        }
    }

    kernel.process_exited = true;
    SUCCESS
}

fn svc_output_debug_string(kernel: &mut Kernel) -> u32 {
    if !diagnostics_enabled() {
        return SUCCESS;
    }

    log::debug!("svcOutputDebugString (X0=str_ptr, X1=str_len)");

    if let Some(cpu) = cpu_ref() {
        let str_ptr = cpu.get_register(0);
        let str_len = cpu.get_register(1);

        if str_ptr > 0 && str_len > 0 && str_len < 262_144 {
            let mut buf = vec![0u8; str_len as usize];
            match kernel.address_space.read(str_ptr, &mut buf) {
                Ok(()) => {
                    let output = std::str::from_utf8(&buf).unwrap_or("[invalid utf8]");
                    log::debug!("OutputDebugString: {}", output);
                }
                Err(e) => {
                    log::warn!("Failed to read debug string from {:#x}: {:?}", str_ptr, e);
                }
            }
        }
    }

    SUCCESS
}

fn svc_connect_to_named_port(kernel: &mut Kernel) -> u32 {
    log::debug!("svcConnectToNamedPort (X1=port_name_ptr)");

    let port_name_ptr = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1)
    } else {
        return 1;
    };

    let port_name = if port_name_ptr > 0 {
        let mut buf = [0u8; 32];
        match kernel.address_space.read(port_name_ptr, &mut buf) {
            Ok(()) => {
                let mut len = 0;
                for (i, &byte) in buf.iter().enumerate() {
                    if byte == 0 {
                        len = i;
                        break;
                    }
                    if i == buf.len() - 1 {
                        len = buf.len();
                    }
                }
                let name_str = std::str::from_utf8(&buf[..len])
                    .unwrap_or("invalid")
                    .to_string();
                log::debug!(
                    "  port_name: '{}' (len={}) PC={:#x}",
                    name_str,
                    len,
                    cpu_ref().map(|c| c.get_pc()).unwrap_or(0)
                );
                name_str
            }
            Err(_) => {
                log::warn!("failed to read port name from {:#x}", port_name_ptr);
                return 1;
            }
        }
    } else {
        return 1;
    };

    let handle = kernel.handles.create_handle(HandleType::Session);
    let session = Session::new(handle, port_name.clone());
    kernel.sessions.insert(handle, session);

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, handle as u64);
    } else {
        log::error!("kernel.cpu is None!");
    }

    log::debug!(
        "created session handle {:#x} to port '{}'",
        handle,
        port_name
    );
    SUCCESS
}

fn svc_get_info(kernel: &mut Kernel) -> u32 {
    let (info_type, handle, sub) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1) as u32,
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        return 1;
    };

    let val: u64 = match info_type {
        0 => {
            if sub != 0 {
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(1, 0);
                    cpu.set_register(0, KERNEL_INVALID_ENUM_VALUE as u64);
                }
                return KERNEL_INVALID_ENUM_VALUE;
            }
            if handle as u32 != 0xFFFF8001 && handle as u32 != kernel.process_handle {
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(1, 0);
                    cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
                }
                return KERNEL_INVALID_HANDLE;
            }
            0xF
        }
        1 => 0x0001_0000_0000,
        2 => kernel.alias_base,
        3 => kernel.alias_size,
        4 => kernel.heap_base,
        5 => kernel.heap_size,
        6 => {
            if kernel.is_application {
                kernel.total_memory
            } else {
                0x80_000_000
            }
        }
        7 => {
            if kernel.is_application {
                kernel.code_size + kernel.stack_size + kernel.heap_committed + 0x100_0000
            } else {
                0x40_000_000
            }
        }
        8 => 0,
        9 => kernel.stack_base,
        10 => kernel.stack_size,
        11 => {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let c = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut z = c
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        12 => kernel.aslr_base,
        13 => kernel.aslr_size,
        14 => kernel.stack_base,
        15 => 0x8000_0000,
        16 => kernel.system_resource_size,
        17 => 0,
        18 => 0,
        19 => 0,
        20 => kernel.tls_base + 0x200,
        21 => kernel.total_memory,
        22 => (kernel.code_size + kernel.stack_size + kernel.heap_committed + 0x100_0000)
            .min(kernel.total_memory),

        23 => u64::from(kernel.is_application),
        24 | 25 | 26 | 27 => 0,

        28 => 0x1000,

        29 => kernel.cycle_count,

        30 => 1,

        31 => 0,

        41 => 0,
        _ => {
            log::warn!(
                "svcGetInfo: unsupported type {} â€” returning InvalidEnumValue (0xF001)",
                info_type
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, 0xF001);
                cpu.set_register(1, 0);
            }
            return 0xF001;
        }
    };

    log::debug!("svcGetInfo type={} -> {:#x}", info_type, val);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, val);
    }
    SUCCESS
}

fn svc_map_physical_memory(kernel: &mut Kernel) -> u32 {
    const KERNEL_OUT_OF_MEMORY: u32 = 1 | (104 << 9);
    let (addr, size) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(1))
    } else {
        return 1;
    };

    if size == 0 || (addr & 0xFFF) != 0 || (size & 0xFFF) != 0 {
        log::warn!(
            "svcMapPhysicalMemory: bad args addr={:#x} size={:#x}",
            addr,
            size
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_ADDRESS as u64);
        }
        return KERNEL_INVALID_ADDRESS;
    }

    let gaps = match kernel.address_space.unmapped_gaps(addr, size) {
        Ok(gaps) => gaps,
        Err(e) => {
            log::warn!(
                "svcMapPhysicalMemory: invalid range addr={:#x} size={:#x}: {:?}",
                addr,
                size,
                e
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, KERNEL_INVALID_ADDRESS as u64);
            }
            return KERNEL_INVALID_ADDRESS;
        }
    };

    for (gs, ge) in &gaps {
        if let Err(e) = kernel
            .address_space
            .map(*gs, ge - gs, nexium_memory::Perm::RW, "physmem")
        {
            log::error!(
                "svcMapPhysicalMemory: map {:#x}..{:#x} failed: {:?}",
                gs,
                ge,
                e
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, KERNEL_OUT_OF_MEMORY as u64);
            }
            return KERNEL_OUT_OF_MEMORY;
        }
        if let Some(region) = kernel.address_space.host_region_at(*gs) {
            if let Some(cpu) = cpu_mut() {
                let plumb = unsafe {
                    cpu.map_host(
                        region.base,
                        region.size,
                        region.perm,
                        region.host_ptr as *mut u8,
                    )
                };
                if let Err(e) = plumb {
                    log::warn!(
                        "svcMapPhysicalMemory: JIT map_host {:#x} len={:#x} failed (error: {}), attempting unmap-and-remap",
                        region.base,
                        region.size,
                        e
                    );
                    let _ = unsafe { cpu.unmap_host(region.base, region.size) };
                    let retry = unsafe {
                        cpu.map_host(
                            region.base,
                            region.size,
                            region.perm,
                            region.host_ptr as *mut u8,
                        )
                    };
                    if let Err(re) = retry {
                        log::error!(
                            "svcMapPhysicalMemory: JIT map_host retry {:#x} len={:#x} failed: {}",
                            region.base,
                            region.size,
                            re
                        );
                    }
                }
            }
        }
    }

    log::trace!(
        "svcMapPhysicalMemory addr={:#x} size={:#x} â†’ {} new region(s)",
        addr,
        size,
        gaps.len()
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_physical_memory(_kernel: &mut Kernel) -> u32 {
    let (addr, size) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(1))
    } else {
        return 1;
    };
    log::debug!("svcUnmapPhysicalMemory addr={:#x} size={:#x}", addr, size);
    if let Some(cpu) = cpu_mut() {
        let _ = unsafe { cpu.unmap_host(addr, size) };
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateEvent");
    let writable = kernel.handles.create_handle(HandleType::Event);
    let readable = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(writable, false);
    kernel.event_signals.insert(readable, false);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, writable as u64);
        cpu.set_register(2, readable as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    log::debug!(
        "  created event writable={:#x} readable={:#x}",
        writable,
        readable
    );
    SUCCESS
}

fn svc_map_transfer_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapTransferMemory");
    SUCCESS
}

fn dump_regs(_kernel: &Kernel, tag: &str) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Some(cpu) = cpu_ref() {
        log::trace!(
            "  [{}] X0={:#x} X1={:#x} X2={:#x} X3={:#x} X4={:#x} X8={:#x} X19={:#x} X30={:#x}",
            tag,
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
            cpu.get_register(4),
            cpu.get_register(8),
            cpu.get_register(19),
            cpu.get_register(30),
        );
    }
}

fn svc_create_transfer_memory(kernel: &mut Kernel) -> u32 {
    dump_regs(kernel, "CreateTmem ENTRY");
    let (addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        return 1;
    };
    let handle = kernel.handles.create_handle(HandleType::TransferMemory);
    kernel.transfer_memories.insert(handle, (addr, size));
    log::debug!(
        "svcCreateTransferMemory addr={:#x} size={:#x} perm={:#x} â†’ handle={:#x}",
        addr,
        size,
        perm,
        handle
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
    }
    dump_regs(kernel, "CreateTmem EXIT");
    SUCCESS
}

fn svc_close_handle(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    let kind = kernel
        .handles
        .get_handle(handle)
        .map(|h| format!("{:?}", h.handle_type))
        .unwrap_or_else(|| "unknown".into());
    log::debug!("svcCloseHandle handle={:#x} ({})", handle, kind);
    dump_regs(kernel, "CloseHandle ENTRY");
    release_hwopus_session_state(kernel, handle);
    let is_audio_out = kernel
        .sessions
        .get(&handle)
        .is_some_and(|session| session.port_name == "IAudioOut");
    if is_audio_out {
        crate::services::audio_out::handlers::close_audio_out_session(kernel, handle);
        kernel.sessions.remove(&handle);
    }
    if kernel.bufferqueue_events.remove(&handle).is_some() {
        kernel.event_signals.remove(&handle);
    }
    if let Some(closed) = kernel.handles.close_handle(handle) {
        if closed.handle_type == HandleType::Thread {
            kernel.exited_thread_handles.remove(&handle);
        }
        if closed.handle_type == HandleType::TransferMemory {
            kernel.transfer_memories.remove(&handle);
        }
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn active_core_count() -> i32 {
    if std::env::var("NEXIUM_SINGLECORE").is_ok() {
        1
    } else {
        std::env::var("NEXIUM_CPU_CORES")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(crate::kernel::threads::NUM_CORES as i32)
            .clamp(1, crate::kernel::threads::NUM_CORES as i32)
    }
}

fn resolve_thread_pseudo_handle(kernel: &Kernel, handle: u32) -> u32 {
    if handle == 0xFFFF8000 {
        kernel
            .threads
            .current_handle()
            .unwrap_or(kernel.main_thread_handle)
    } else {
        handle
    }
}

fn svc_create_thread(kernel: &mut Kernel) -> u32 {
    let (entry, arg, sp, priority, core) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
            cpu.get_register(4) as i32,
            cpu.get_register(5) as i32,
        )
    } else {
        return 1;
    };

    let ideal_core =
        match crate::kernel::threads::resolve_create_thread_core(core, kernel.process_ideal_core) {
            Ok(resolved) => resolved,
            Err(code) => {
                log::warn!(
                    "svcCreateThread entry={:#x} rejected invalid core {}",
                    entry,
                    core
                );
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(1, 0);
                    cpu.set_register(0, code as u64);
                }
                return code;
            }
        };

    if !(0..=63).contains(&priority) {
        log::warn!(
            "svcCreateThread entry={:#x} rejected invalid priority {}",
            entry,
            priority
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(1, 0);
            cpu.set_register(0, KERNEL_INVALID_PRIORITY as u64);
        }
        return KERNEL_INVALID_PRIORITY;
    }

    let handle = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Thread);
    let tls_va = kernel.threads.alloc_tls();
    let _ = kernel.address_space.write(tls_va, &[0u8; 0x1000]);

    let mut ctx = crate::kernel::threads::ThreadCtx::zero();
    ctx.x[0] = arg;
    ctx.sp = sp;
    ctx.pc = entry;
    ctx.tpidrro_el0 = tls_va;

    kernel.threads.add_thread(handle, ctx, tls_va, sp, arg);
    if let Some(t) = kernel.threads.threads.get_mut(&handle) {
        t.priority = priority;
        t.ideal_core = ideal_core;
        t.affinity_mask = 1u64 << ideal_core;
        t.core = ideal_core.min(active_core_count() - 1);
    }

    log::debug!(
        "svcCreateThread entry={:#x} arg={:#x} sp={:#x} prio={} core={} ideal={} -> handle={:#x} tls={:#x}",
        entry,
        arg,
        sp,
        priority,
        core,
        ideal_core,
        handle,
        tls_va
    );

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_start_thread(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        return 1;
    };
    let code = match kernel.threads.threads.get(&handle).map(|t| &t.state) {
        Some(crate::kernel::threads::ThreadState::Created) => {
            kernel
                .threads
                .transition_state(handle, crate::kernel::threads::ThreadState::Ready);
            SUCCESS
        }
        Some(_) => KERNEL_INVALID_THREAD_STATE,
        None if kernel.exited_thread_handles.contains_key(&handle) => KERNEL_INVALID_THREAD_STATE,
        None => KERNEL_INVALID_HANDLE,
    };
    log::debug!("svcStartThread handle={:#x} -> {:#x}", handle, code);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, code as u64);
    }
    code
}

fn svc_exit_thread(kernel: &mut Kernel) -> u32 {
    let current = kernel.threads.current_handle();
    log::debug!("svcExitThread current={:?}", current);
    if let Some(handle) = current {
        let state = kernel
            .threads
            .threads
            .get(&handle)
            .map(|t| crate::kernel::ExitedThreadState {
                ideal_core: t.ideal_core,
                affinity_mask: t.affinity_mask,
                priority: t.priority,
            })
            .unwrap_or(crate::kernel::ExitedThreadState {
                ideal_core: kernel.process_ideal_core,
                affinity_mask: 1u64 << kernel.process_ideal_core,
                priority: 0x2C,
            });
        kernel.exited_thread_handles.insert(handle, state);
        kernel.threads.signal_handle(handle);
    }
    if let Some(cpu) = cpu_ref() {
        kernel
            .threads
            .yield_with_state(cpu, crate::kernel::threads::ThreadState::Exited);
    }
    SUCCESS
}

fn svc_sleep_thread(kernel: &mut Kernel) -> u32 {
    let ns = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        0
    };
    let signed = ns as i64;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    if signed > 0 {
        let dur = std::time::Duration::from_nanos(ns);
        let wake_at = std::time::Instant::now() + dur;
        if let Some(cpu) = cpu_ref() {
            kernel.threads.yield_with_state(
                cpu,
                crate::kernel::threads::ThreadState::Sleeping { wake_at },
            );
        }
    } else if matches!(signed, 0 | -1 | -2) {
        if let Some(cpu) = cpu_ref() {
            kernel
                .threads
                .yield_with_state(cpu, crate::kernel::threads::ThreadState::Ready);
        }
    }
    SUCCESS
}

fn svc_flush_data_cache(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_priority(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1) as u32
    } else {
        return 1;
    };
    let target = resolve_thread_pseudo_handle(kernel, handle);
    let prio = if kernel.threads.threads.contains_key(&target) {
        Some(kernel.threads.effective_priority(target))
    } else {
        kernel
            .exited_thread_handles
            .get(&target)
            .map(|t| t.priority)
    };
    let Some(prio) = prio else {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(1, 0);
            cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
        }
        return KERNEL_INVALID_HANDLE;
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, prio as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_priority(kernel: &mut Kernel) -> u32 {
    let (handle, priority) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0) as u32, cpu.get_register(1) as i32)
    } else {
        return 1;
    };
    let code = if !(0..=63).contains(&priority) {
        KERNEL_INVALID_PRIORITY
    } else {
        let target = resolve_thread_pseudo_handle(kernel, handle);
        if let Some(t) = kernel.threads.threads.get_mut(&target) {
            t.priority = priority;
            SUCCESS
        } else if kernel.exited_thread_handles.contains_key(&target) {
            SUCCESS
        } else {
            KERNEL_INVALID_HANDLE
        }
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, code as u64);
    }
    code
}

fn svc_get_thread_core_mask(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(2) as u32
    } else {
        return 1;
    };
    let target = resolve_thread_pseudo_handle(kernel, handle);
    let Some((ideal_core, affinity_mask)) = kernel.threads.thread_core_mask(target).or_else(|| {
        kernel
            .exited_thread_handles
            .get(&target)
            .map(|t| (t.ideal_core, t.affinity_mask))
    }) else {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(1, 0);
            cpu.set_register(2, 0);
            cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
        }
        return KERNEL_INVALID_HANDLE;
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, ideal_core as u32 as u64);
        cpu.set_register(2, affinity_mask);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_core_mask(kernel: &mut Kernel) -> u32 {
    let (handle, core_id, affinity_mask) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0) as u32,
            cpu.get_register(1) as u32 as i32,
            cpu.get_register(2),
        )
    } else {
        return 1;
    };
    let target = resolve_thread_pseudo_handle(kernel, handle);
    let mut result = kernel.threads.set_thread_core_mask(
        target,
        core_id,
        affinity_mask,
        kernel.process_ideal_core,
        active_core_count(),
    );
    if result == Err(KERNEL_INVALID_HANDLE) {
        if let Some(t) = kernel.exited_thread_handles.get(&target) {
            result = if core_id == crate::kernel::threads::IDEAL_CORE_NO_UPDATE
                && t.ideal_core >= 0
                && affinity_mask & (1u64 << t.ideal_core) == 0
            {
                Err(nexium_common::result::KERNEL_INVALID_COMBINATION)
            } else {
                Ok(())
            };
        }
    }
    log::info!(
        "svcSetThreadCoreMask handle={:#x} core={} mask={:#x} -> {:?}",
        handle,
        core_id,
        affinity_mask,
        result
    );
    let code = match result {
        Ok(()) => SUCCESS,
        Err(code) => code,
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, code as u64);
    }
    code
}

fn svc_get_current_processor_number(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, crate::kernel::cpu_local::current_core() as u64);
    }
    0
}

fn svc_send_sync_request_light(kernel: &mut Kernel) -> u32 {
    svc_send_sync_request(kernel)
}

fn svc_send_sync_request_with_user_buffer(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequestWithUserBuffer (treating as svcSendSyncRequest)");
    svc_send_sync_request(kernel)
}

fn svc_send_async_request_with_user_buffer(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendAsyncRequestWithUserBuffer");
    let handle = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(handle, true);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_return_from_exception(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReturnFromException");
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_flush_entire_data_cache(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_future_thread_info(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_last_thread_info(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_limit_value(_kernel: &mut Kernel) -> u32 {
    let limitable = if let Some(cpu) = cpu_ref() {
        cpu.get_register(2) as u32
    } else {
        0
    };
    let value: u64 = match limitable {
        0 => 0x40_000_000,
        1 => 1024,
        2 => 1024,
        3 => 8,
        4 => 0x80_000,
        5 => 64,
        _ => 0,
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, value);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_current_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_peak_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_activity(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_context3(kernel: &mut Kernel) -> u32 {
    let (out_ptr, handle) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(1) as u32)
    } else {
        return 1;
    };
    if out_ptr != 0 {
        if let Some(t) = kernel.threads.threads.get(&handle) {
            let mut buf = [0u8; 0x320];
            for i in 0..29 {
                let off = i * 8;
                buf[off..off + 8].copy_from_slice(&t.ctx.x[i].to_le_bytes());
            }
            buf[0xe8..0xf0].copy_from_slice(&t.ctx.sp.to_le_bytes());
            buf[0xf8..0x100].copy_from_slice(&t.ctx.pc.to_le_bytes());
            let _ = kernel.address_space.write(out_ptr, &buf);
        }
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_synchronize_preemption_state(kernel: &mut Kernel) -> u32 {
    kernel.synchronize_user_preemption_state();
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_session(kernel: &mut Kernel) -> u32 {
    let server = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    let client = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_accept_session(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reply_and_receive_light(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_reply_and_receive(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReplyAndReceive (stub â†’ TIMEOUT)");
    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_TIMEOUT as u64);
    }
    KERNEL_TIMEOUT
}

fn svc_reply_and_receive_with_user_buffer(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::SharedMemory);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_transfer_memory(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_interrupt_event(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(h, false);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_io_mapping(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_debug_active_process(_kernel: &mut Kernel) -> u32 {
    const KERNEL_INVALID_HANDLE: u32 = 1 | (114 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
    }
    KERNEL_INVALID_HANDLE
}

fn svc_break_debug_process(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_terminate_debug_process(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_event(_kernel: &mut Kernel) -> u32 {
    const KERNEL_NO_DEBUG_EVENT: u32 = 1 | (140 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_NO_DEBUG_EVENT as u64);
    }
    KERNEL_NO_DEBUG_EVENT
}

fn svc_continue_debug_event(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_list(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 1);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_list(kernel: &mut Kernel) -> u32 {
    let count = kernel.threads.threads.len() as u64;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, count);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_port(kernel: &mut Kernel) -> u32 {
    let server = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    let client = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_manage_named_port(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_connect_to_port(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_resource_limit(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Process);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_resource_limit_limit_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_call_secure_monitor(_kernel: &mut Kernel) -> u32 {
    let smc_id = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    log::debug!(
        "svcCallSecureMonitor smc_id={:#x} (HLE: returning success)",
        smc_id
    );
    if let Some(cpu) = cpu_mut() {
        for r in 0..=7 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn fs_base_root() -> Option<std::path::PathBuf> {
    Some(nexium_common::paths::root())
}

fn fs_sd_root(kernel: &mut Kernel) -> Option<std::path::PathBuf> {
    if kernel.sd_root.is_none() {
        let root = fs_base_root()?.join("sdmc");
        if let Err(e) = std::fs::create_dir_all(&root) {
            log::warn!("fs: failed to create SD root {}: {}", root.display(), e);
            return None;
        }
        kernel.sd_root = Some(root);
    }
    kernel.sd_root.clone()
}

fn fs_save_data_root(kernel: &Kernel, ctx: &ipc::IpcCtx) -> Option<std::path::PathBuf> {
    let data_start = ctx.cmif_in_data_off.min(ctx.buf.len());
    let data_end = data_start
        .saturating_add(ctx.cmif_in_data_len)
        .min(ctx.buf.len());
    let data = &ctx.buf[data_start..data_end];
    let space_id = data.first().copied().unwrap_or(1);
    let attr_off = 8usize;
    let program_id = fs_read_le_u64(data, attr_off).unwrap_or(0);
    let system_save_data_id = fs_read_le_u64(data, attr_off + 24).unwrap_or(0);
    let save_type = data.get(attr_off + 32).copied().unwrap_or(1);
    let user_id = fs_user_id_hex(data, attr_off + 8);
    let title_id = if program_id != 0 {
        program_id
    } else {
        kernel.title_id
    };
    let base = fs_base_root()?;
    let title = format!("{:016x}", title_id);
    let root = match space_id {
        0 => base
            .join("nand")
            .join("system")
            .join("save")
            .join(format!("{:016x}", system_save_data_id))
            .join(&user_id),
        1 => match save_type {
            4 => base
                .join("nand")
                .join("temp")
                .join("0000000000000000")
                .join(&user_id)
                .join(&title),
            5 => base
                .join("nand")
                .join("user")
                .join("save")
                .join("cache")
                .join(&title),
            _ => base
                .join("nand")
                .join("user")
                .join("save")
                .join("0000000000000000")
                .join(&user_id)
                .join(&title),
        },
        2 | 4 => base
            .join("sdmc")
            .join("save")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
        3 => base
            .join("nand")
            .join("temp")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
        _ => base
            .join("nand")
            .join("user")
            .join("save")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
    };
    if let Err(e) = std::fs::create_dir_all(&root) {
        log::warn!("fs: failed to create save root {}: {}", root.display(), e);
        return None;
    }
    Some(root)
}

fn fs_read_le_u64(data: &[u8], off: usize) -> Option<u64> {
    let bytes = data.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn fs_user_id_hex(data: &[u8], off: usize) -> String {
    let mut bytes = [0u8; 16];
    if let Some(src) = data.get(off..off.saturating_add(16)) {
        if src.len() == 16 {
            bytes.copy_from_slice(src);
        }
    }
    let mut low_bytes = [0u8; 8];
    let mut high_bytes = [0u8; 8];
    low_bytes.copy_from_slice(&bytes[0..8]);
    high_bytes.copy_from_slice(&bytes[8..16]);
    let low = u64::from_le_bytes(low_bytes);
    let high = u64::from_le_bytes(high_bytes);
    format!("{:016x}{:016x}", high, low)
}

fn fs_object_root(
    kernel: &mut Kernel,
    session_handle: u32,
    object_id: u32,
) -> Option<std::path::PathBuf> {
    let root = domain_object_keys(kernel, session_handle, object_id)
        .into_iter()
        .find_map(|key| kernel.file_system_roots.get(&key).cloned());
    root.or_else(|| fs_sd_root(kernel))
}

fn fs_host_path(
    kernel: &mut Kernel,
    session_handle: u32,
    object_id: u32,
    hos: &str,
) -> Option<std::path::PathBuf> {
    let root = fs_object_root(kernel, session_handle, object_id)?;
    fs_translate(&root, hos)
}

fn fs_translate(root: &std::path::Path, hos: &str) -> Option<std::path::PathBuf> {
    let trimmed = hos.trim_start_matches(|c: char| c.is_ascii_alphabetic() || c == ':');
    let trimmed = trimmed.trim_start_matches(|c| c == '/' || c == '\\');
    let rel = std::path::Path::new(trimmed);
    for c in rel.components() {
        if matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::Prefix(_)
                | std::path::Component::RootDir
        ) {
            return None;
        }
    }
    Some(root.join(rel))
}

fn fs_read_path(ctx: &ipc::IpcCtx, addr_space: &nexium_memory::AddressSpace) -> String {
    let buf = ctx
        .send_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .or_else(|| ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
        .copied();
    let Some(b) = buf else { return String::new() };
    let n = (b.size as usize).min(0x301);
    let mut bytes = vec![0u8; n];
    if addr_space.read(b.addr, &mut bytes).is_err() {
        return String::new();
    }
    let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn ng_word2_romfs() -> &'static [u8] {
    use std::sync::OnceLock;

    static ROMFS: OnceLock<Vec<u8>> = OnceLock::new();
    ROMFS.get_or_init(build_ng_word2_romfs).as_slice()
}

fn mii_model_romfs() -> &'static [u8] {
    use std::sync::OnceLock;

    static ROMFS: OnceLock<Vec<u8>> = OnceLock::new();
    ROMFS.get_or_init(build_mii_model_romfs).as_slice()
}

fn build_mii_model_romfs() -> Vec<u8> {
    const NFTR_HEADER: &[u8] = b"NFTR\x01\0\0\0\0\0\0\0\0\0\0\0";
    const NFSR_HEADER: &[u8] = b"NFSR\x01\0\0\0\0\0\0\0\0\0\0\0";

    let files = [
        ("NXTextureLowLinear.dat", NFTR_HEADER),
        ("NXTextureLowSRGB.dat", NFTR_HEADER),
        ("NXTextureMidLinear.dat", NFTR_HEADER),
        ("NXTextureMidSRGB.dat", NFTR_HEADER),
        ("ShapeHigh.dat", NFSR_HEADER),
        ("ShapeMid.dat", NFSR_HEADER),
    ]
    .into_iter()
    .map(|(name, data)| (name.to_string(), data.to_vec()))
    .collect();
    build_flat_romfs(files)
}

fn build_ng_word2_romfs() -> Vec<u8> {
    const AC_NX_DATA: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x08, 0xd5, 0x2c, 0x09, 0x5c, 0x04, 0x00, 0x61, 0x63, 0x72, 0x61, 0x77,
        0x00, 0xed, 0xc1, 0x01, 0x0d, 0x00, 0x00, 0x00, 0xc2, 0x20, 0xfb, 0xa7, 0xb6, 0xc7, 0x07,
        0x0c, 0x00, 0x00, 0x00, 0xc8, 0x3b, 0x11, 0x00, 0x1c, 0xc7, 0x00, 0x10, 0x00, 0x00,
    ];

    let mut files = Vec::with_capacity(52);
    for index in 0..16 {
        for suffix in ["b1_nx", "b2_nx", "not_b_nx"] {
            files.push((format!("ac_{}_{}", index, suffix), AC_NX_DATA.to_vec()));
        }
    }
    files.extend([
        ("ac_common_b1_nx".to_string(), AC_NX_DATA.to_vec()),
        ("ac_common_b2_nx".to_string(), AC_NX_DATA.to_vec()),
        ("ac_common_not_b_nx".to_string(), AC_NX_DATA.to_vec()),
        ("version.dat".to_string(), vec![0, 0, 0, 0x1a]),
    ]);
    build_flat_romfs(files)
}

fn build_flat_romfs(mut files: Vec<(String, Vec<u8>)>) -> Vec<u8> {
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let align4 = |value: usize| (value + 3) & !3;
    let align16 = |value: usize| (value + 15) & !15;
    let dir_hash_count = 3usize;
    let file_hash_count = romfs_hash_table_entry_count(files.len());
    let dir_hash_size = dir_hash_count * 4;
    let file_hash_size = file_hash_count * 4;
    let dir_table_size = 0x18usize;
    let file_table_size = files
        .iter()
        .map(|(name, _)| 0x20 + align4(name.len()))
        .sum::<usize>();

    let mut file_offsets = Vec::with_capacity(files.len());
    let mut file_partition_size = 0usize;
    for (_, data) in &files {
        file_partition_size = align16(file_partition_size);
        file_offsets.push(file_partition_size);
        file_partition_size += data.len();
    }

    let file_partition_ofs = 0x200usize;
    let dir_hash_ofs = align4(file_partition_ofs + file_partition_size);
    let dir_table_ofs = dir_hash_ofs + dir_hash_size;
    let file_hash_ofs = dir_table_ofs + dir_table_size;
    let file_table_ofs = file_hash_ofs + file_hash_size;
    let mut romfs =
        vec![0u8; (file_table_ofs + file_table_size).max(file_partition_ofs + file_partition_size)];

    romfs[0..8].copy_from_slice(&0x50u64.to_le_bytes());
    romfs[8..16].copy_from_slice(&(dir_hash_ofs as u64).to_le_bytes());
    romfs[16..24].copy_from_slice(&(dir_hash_size as u64).to_le_bytes());
    romfs[24..32].copy_from_slice(&(dir_table_ofs as u64).to_le_bytes());
    romfs[32..40].copy_from_slice(&(dir_table_size as u64).to_le_bytes());
    romfs[40..48].copy_from_slice(&(file_hash_ofs as u64).to_le_bytes());
    romfs[48..56].copy_from_slice(&(file_hash_size as u64).to_le_bytes());
    romfs[56..64].copy_from_slice(&(file_table_ofs as u64).to_le_bytes());
    romfs[64..72].copy_from_slice(&(file_table_size as u64).to_le_bytes());
    romfs[72..80].copy_from_slice(&(file_partition_ofs as u64).to_le_bytes());

    let mut dir_hash = vec![u32::MAX; dir_hash_count];
    let mut file_hash = vec![u32::MAX; file_hash_count];
    let root_hash = romfs_path_hash(0, &[]);
    dir_hash[(root_hash as usize) % dir_hash_count] = 0;
    for (index, word) in dir_hash.iter().enumerate() {
        let offset = dir_hash_ofs + index * 4;
        romfs[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }

    romfs[dir_table_ofs..dir_table_ofs + 4].copy_from_slice(&0u32.to_le_bytes());
    romfs[dir_table_ofs + 4..dir_table_ofs + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    romfs[dir_table_ofs + 8..dir_table_ofs + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    romfs[dir_table_ofs + 12..dir_table_ofs + 16].copy_from_slice(&0u32.to_le_bytes());
    romfs[dir_table_ofs + 16..dir_table_ofs + 20].copy_from_slice(&u32::MAX.to_le_bytes());
    romfs[dir_table_ofs + 20..dir_table_ofs + 24].copy_from_slice(&0u32.to_le_bytes());

    let mut entry_offset = 0usize;
    for (index, ((name, data), &data_offset)) in files.iter().zip(&file_offsets).enumerate() {
        let entry_size = 0x20 + align4(name.len());
        let sibling = if index + 1 < files.len() {
            (entry_offset + entry_size) as u32
        } else {
            u32::MAX
        };
        let hash = romfs_path_hash(0, name.as_bytes());
        let bucket = (hash as usize) % file_hash_count;
        let table_entry = file_hash[bucket];
        file_hash[bucket] = entry_offset as u32;
        let base = file_table_ofs + entry_offset;
        romfs[base..base + 4].copy_from_slice(&0u32.to_le_bytes());
        romfs[base + 4..base + 8].copy_from_slice(&sibling.to_le_bytes());
        romfs[base + 8..base + 16].copy_from_slice(&(data_offset as u64).to_le_bytes());
        romfs[base + 16..base + 24].copy_from_slice(&(data.len() as u64).to_le_bytes());
        romfs[base + 24..base + 28].copy_from_slice(&table_entry.to_le_bytes());
        romfs[base + 28..base + 32].copy_from_slice(&(name.len() as u32).to_le_bytes());
        romfs[base + 32..base + 32 + name.len()].copy_from_slice(name.as_bytes());
        let data_base = file_partition_ofs + data_offset;
        romfs[data_base..data_base + data.len()].copy_from_slice(data);
        entry_offset += entry_size;
    }

    for (index, word) in file_hash.iter().enumerate() {
        let offset = file_hash_ofs + index * 4;
        romfs[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
    romfs
}

fn romfs_path_hash(parent: u32, name: &[u8]) -> u32 {
    let mut hash = parent ^ 123_456_789;
    for byte in name {
        hash = hash.rotate_right(5) ^ u32::from(*byte);
    }
    hash
}

fn romfs_hash_table_entry_count(entry_count: usize) -> usize {
    if entry_count < 3 {
        return 3;
    }
    if entry_count < 19 {
        return entry_count | 1;
    }

    let mut count = entry_count;
    while [2, 3, 5, 7, 11, 13, 17]
        .into_iter()
        .any(|divisor| count % divisor == 0)
    {
        count += 1;
    }
    count
}

#[derive(Clone, Copy)]
struct RomfsHeader {
    dir_meta_off: usize,
    dir_meta_size: usize,
    file_meta_off: usize,
    file_meta_size: usize,
    file_data_off: usize,
}

enum RomfsEntry {
    Dir,
    File { offset: usize, size: usize },
}

fn romfs_u32(data: &[u8], off: usize) -> Option<u32> {
    let bytes = data.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn romfs_u64(data: &[u8], off: usize) -> Option<u64> {
    let bytes = data.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn romfs_usize(data: &[u8], off: usize) -> Option<usize> {
    usize::try_from(romfs_u64(data, off)?).ok()
}

fn romfs_header(romfs: &[u8]) -> Option<RomfsHeader> {
    if romfs_u64(romfs, 0)? != 0x50 {
        return None;
    }
    Some(RomfsHeader {
        dir_meta_off: romfs_usize(romfs, 0x18)?,
        dir_meta_size: romfs_usize(romfs, 0x20)?,
        file_meta_off: romfs_usize(romfs, 0x38)?,
        file_meta_size: romfs_usize(romfs, 0x40)?,
        file_data_off: romfs_usize(romfs, 0x48)?,
    })
}

fn romfs_components(path: &str) -> Option<Vec<&str>> {
    let path = path.trim_matches(char::from(0)).trim();
    let path = path.strip_prefix("rom:").unwrap_or(path);
    let mut out = Vec::new();
    for part in path.split(|c| c == '/' || c == '\\') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return None;
        }
        out.push(part);
    }
    Some(out)
}

fn romfs_name(
    romfs: &[u8],
    entry_abs: usize,
    name_off: usize,
    name_len_off: usize,
) -> Option<&str> {
    let name_len = romfs_u32(romfs, entry_abs.checked_add(name_len_off)?)? as usize;
    let start = entry_abs.checked_add(name_off)?;
    let end = start.checked_add(name_len)?;
    std::str::from_utf8(romfs.get(start..end)?).ok()
}

fn romfs_child_dir(romfs: &[u8], hdr: RomfsHeader, dir_off: u32, name: &str) -> Option<u32> {
    let dir_rel = usize::try_from(dir_off).ok()?;
    if dir_rel >= hdr.dir_meta_size {
        return None;
    }
    let mut child = romfs_u32(
        romfs,
        hdr.dir_meta_off.checked_add(dir_rel)?.checked_add(0x08)?,
    )?;
    while child != u32::MAX {
        let rel = usize::try_from(child).ok()?;
        if rel >= hdr.dir_meta_size {
            return None;
        }
        let abs = hdr.dir_meta_off.checked_add(rel)?;
        if romfs_name(romfs, abs, 0x18, 0x14)? == name {
            return Some(child);
        }
        child = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }
    None
}

fn romfs_child_file(
    romfs: &[u8],
    hdr: RomfsHeader,
    dir_off: u32,
    name: &str,
) -> Option<(usize, usize)> {
    let dir_rel = usize::try_from(dir_off).ok()?;
    if dir_rel >= hdr.dir_meta_size {
        return None;
    }
    let mut child = romfs_u32(
        romfs,
        hdr.dir_meta_off.checked_add(dir_rel)?.checked_add(0x0c)?,
    )?;
    while child != u32::MAX {
        let rel = usize::try_from(child).ok()?;
        if rel >= hdr.file_meta_size {
            return None;
        }
        let abs = hdr.file_meta_off.checked_add(rel)?;
        if romfs_name(romfs, abs, 0x20, 0x1c)? == name {
            let rel_off = usize::try_from(romfs_u64(romfs, abs.checked_add(0x08)?)?).ok()?;
            let size = usize::try_from(romfs_u64(romfs, abs.checked_add(0x10)?)?).ok()?;
            let offset = hdr.file_data_off.checked_add(rel_off)?;
            return Some((offset, size));
        }
        child = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }
    None
}

fn romfs_find_entry(romfs: &[u8], path: &str) -> Option<RomfsEntry> {
    let hdr = romfs_header(romfs)?;
    let comps = romfs_components(path)?;
    if comps.is_empty() {
        return Some(RomfsEntry::Dir);
    }
    let mut dir = 0u32;
    for (i, name) in comps.iter().enumerate() {
        let last = i + 1 == comps.len();
        if last {
            if let Some(child_dir) = romfs_child_dir(romfs, hdr, dir, name) {
                let rel = usize::try_from(child_dir).ok()?;
                if rel < hdr.dir_meta_size {
                    return Some(RomfsEntry::Dir);
                }
            }
            if let Some((offset, size)) = romfs_child_file(romfs, hdr, dir, name) {
                return Some(RomfsEntry::File { offset, size });
            }
            return None;
        }
        dir = romfs_child_dir(romfs, hdr, dir, name)?;
    }
    None
}

fn romfs_open_file(romfs: &[u8], path: &str) -> Option<(usize, usize)> {
    match romfs_find_entry(romfs, path)? {
        RomfsEntry::File { offset, size } => Some((offset, size)),
        RomfsEntry::Dir => None,
    }
}

fn romfs_entry_type(romfs: &[u8], path: &str) -> Option<u32> {
    match romfs_find_entry(romfs, path)? {
        RomfsEntry::Dir => Some(0),
        RomfsEntry::File { .. } => Some(1),
    }
}

#[cfg(test)]
mod synthetic_system_archive_tests {
    use super::{build_mii_model_romfs, build_ng_word2_romfs, romfs_open_file};

    fn file<'a>(romfs: &'a [u8], path: &str) -> &'a [u8] {
        let (offset, size) = romfs_open_file(romfs, path).expect("synthetic archive file");
        &romfs[offset..offset + size]
    }

    #[test]
    fn mii_model_archive_has_expected_header_only_resources() {
        let romfs = build_mii_model_romfs();
        for name in [
            "NXTextureLowLinear.dat",
            "NXTextureLowSRGB.dat",
            "NXTextureMidLinear.dat",
            "NXTextureMidSRGB.dat",
        ] {
            assert_eq!(file(&romfs, name), b"NFTR\x01\0\0\0\0\0\0\0\0\0\0\0");
        }
        for name in ["ShapeHigh.dat", "ShapeMid.dat"] {
            assert_eq!(file(&romfs, name), b"NFSR\x01\0\0\0\0\0\0\0\0\0\0\0");
        }
        assert!(romfs_open_file(&romfs, "missing.dat").is_none());
    }

    #[test]
    fn shared_flat_romfs_builder_preserves_ng_word2_files() {
        let romfs = build_ng_word2_romfs();
        assert_eq!(file(&romfs, "version.dat"), [0, 0, 0, 0x1a]);
        assert_eq!(&file(&romfs, "ac_0_b1_nx")[..4], [0x1f, 0x8b, 0x08, 0x08]);
    }
}

fn fs_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_FS_TRACE")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

fn fs_trace_matches(path: &str) -> bool {
    use std::sync::OnceLock;
    static FILTERS: OnceLock<Vec<String>> = OnceLock::new();
    let filters = FILTERS.get_or_init(|| {
        std::env::var("NEXIUM_FS_TRACE_FILTER")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    if filters.is_empty() {
        return true;
    }
    let hay = path.to_ascii_lowercase();
    filters.iter().any(|needle| hay.contains(needle))
}

fn signal_due_audio_sessions(kernel: &mut Kernel, now: std::time::Instant) {
    use crate::services::audio_out::handlers as aout;
    let sessions: Vec<(u32, u32)> = kernel
        .audio_buffer_events
        .iter()
        .map(|(s, e)| (*s, *e))
        .collect();
    for (sess, ev) in sessions {
        let due = aout::check_due_signal(kernel, sess, now);
        if matches!(due, aout::DueSignal::None) {
            continue;
        }
        let slot = kernel.event_signals.entry(ev).or_insert(false);
        if *slot {
            continue;
        }
        *slot = true;
        kernel.threads.signal_handle(ev);
        if matches!(due, aout::DueSignal::BufferDue) {
            aout::mark_due_signaled(kernel, sess);
        }
    }
}

fn fs_trace_path(kind: &str, path: &str, detail: &str) {
    if !fs_trace_enabled() || !fs_trace_matches(path) {
        return;
    }
    log::warn!("[fs-trace] {} path={} {}", kind, path, detail);
}

fn fs_trace_read(
    kind: &str,
    path: &str,
    abs_offset: usize,
    read_offset: i64,
    read_size: u64,
    bytes_read: u64,
) {
    if !fs_trace_enabled() || !fs_trace_matches(path) {
        return;
    }

    log::warn!(
        "[fs-trace] {} path={} abs={:#x} read_off={:#x} size={:#x} bytes={}",
        kind,
        path,
        abs_offset,
        read_offset,
        read_size,
        bytes_read
    );
}

fn ipc_trace_request(
    kernel: &Kernel,
    session_handle: u32,
    port_name: &str,
    dispatch_target: &str,
    cmd_id: u32,
    is_domain: bool,
    ctx: &ipc::IpcCtx,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_IPC_TRACE")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    });
    if !enabled {
        return;
    }

    static FILTERS: OnceLock<Vec<String>> = OnceLock::new();
    let filters = FILTERS.get_or_init(|| {
        std::env::var("NEXIUM_IPC_TRACE_FILTER")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    if !filters.is_empty() {
        let hay = format!(
            "{} {} cmd={} session={:#x}",
            port_name, dispatch_target, cmd_id, session_handle
        )
        .to_ascii_lowercase();
        if !filters.iter().any(|needle| hay.contains(needle)) {
            return;
        }
    }

    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_IPC_TRACE_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(20_000)
    });
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if limit != 0 && n >= limit {
        return;
    }

    let (pc, lr, x20, x21) = cpu_ref()
        .map(|cpu| {
            (
                cpu.get_pc(),
                cpu.get_register(30),
                cpu.get_register(20),
                cpu.get_register(21),
            )
        })
        .unwrap_or((0, 0, 0, 0));
    let domain = ctx
        .domain
        .map(|d| {
            format!(
                "kind={} obj={} in_objs={} data={}",
                d.kind, d.object_id, d.num_in_objects, d.data_size
            )
        })
        .unwrap_or_else(|| "-".to_string());
    let in_preview: Vec<String> = if ctx.cmif_in_data_off < ctx.buf.len() {
        let end = (ctx.cmif_in_data_off + ctx.cmif_in_data_len.min(24)).min(ctx.buf.len());
        ctx.buf[ctx.cmif_in_data_off..end]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    } else {
        Vec::new()
    };
    let ptr_preview = |addr: u64| -> String {
        if addr == 0 {
            return "-".to_string();
        }
        let mut bytes = [0u8; 32];
        if kernel.address_space.read(addr, &mut bytes).is_err() {
            return "unreadable".to_string();
        }
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(",")
    };

    log::warn!(
        "[ipc-trace] n={} thread={:?} sess={:#x} port={} target={} cmd={} domain={} is_domain={} in_len={} pc={:#x} lr={:#x} x20={:#x} x21={:#x} x20mem={} x21mem={} send={} recv={} sstat={} rstat={} in={}",
        n,
        kernel.threads.current,
        session_handle,
        port_name,
        dispatch_target,
        cmd_id,
        domain,
        is_domain,
        ctx.cmif_in_data_len,
        pc,
        lr,
        x20,
        x21,
        ptr_preview(x20),
        ptr_preview(x21),
        ipc_trace_buffers(&ctx.send_buffers),
        ipc_trace_buffers(&ctx.recv_buffers),
        ipc_trace_buffers(&ctx.send_statics),
        ipc_trace_buffers(&ctx.recv_statics),
        in_preview.join(",")
    );
}

fn ipc_trace_buffers(buffers: &[ipc::IpcBuffer]) -> String {
    if buffers.is_empty() {
        return "-".to_string();
    }
    let mut parts: Vec<String> = buffers
        .iter()
        .take(3)
        .map(|b| format!("{:#x}:{:#x}:{}", b.addr, b.size, b.mode))
        .collect();
    if buffers.len() > parts.len() {
        parts.push(format!("+{}", buffers.len() - parts.len()));
    }
    parts.join("|")
}

fn maybe_thread_snapshot(kernel: &Kernel, dispatch_target: &str, cmd_id: u32) {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    static PERIOD_MS: OnceLock<Option<u64>> = OnceLock::new();
    let Some(period_ms) = *PERIOD_MS.get_or_init(|| {
        std::env::var("NEXIUM_THREAD_SNAPSHOT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
    }) else {
        return;
    };

    static LAST: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
    let now = Instant::now();
    let last_cell = LAST.get_or_init(|| Mutex::new(None));
    let Ok(mut last) = last_cell.lock() else {
        return;
    };
    if last
        .as_ref()
        .map(|prev| prev.elapsed().as_millis() < period_ms as u128)
        .unwrap_or(false)
    {
        return;
    }
    *last = Some(now);
    kernel.log_thread_snapshot(&format!("ipc-{}-cmd{}", dispatch_target, cmd_id));
}

fn romfs_path_for_data_offset(romfs: &[u8], data_off: usize) -> Option<(String, usize, usize)> {
    let hdr = romfs_header(romfs)?;
    romfs_path_for_data_offset_in_dir(romfs, hdr, 0, "", data_off, 0)
}

fn romfs_path_for_data_offset_in_dir(
    romfs: &[u8],
    hdr: RomfsHeader,
    dir_off: u32,
    prefix: &str,
    data_off: usize,
    depth: usize,
) -> Option<(String, usize, usize)> {
    if depth > 64 {
        return None;
    }
    let dir_rel = usize::try_from(dir_off).ok()?;
    if dir_rel >= hdr.dir_meta_size {
        return None;
    }
    let dir_abs = hdr.dir_meta_off.checked_add(dir_rel)?;

    let mut file = romfs_u32(romfs, dir_abs.checked_add(0x0c)?)?;
    while file != u32::MAX {
        let rel = usize::try_from(file).ok()?;
        if rel >= hdr.file_meta_size {
            return None;
        }
        let abs = hdr.file_meta_off.checked_add(rel)?;
        let name = romfs_name(romfs, abs, 0x20, 0x1c)?;
        let rel_off = usize::try_from(romfs_u64(romfs, abs.checked_add(0x08)?)?).ok()?;
        let size = usize::try_from(romfs_u64(romfs, abs.checked_add(0x10)?)?).ok()?;
        let file_off = hdr.file_data_off.checked_add(rel_off)?;
        if data_off >= file_off && data_off.saturating_sub(file_off) < size {
            let path = if prefix.is_empty() {
                format!("/{}", name)
            } else {
                format!("{}/{}", prefix, name)
            };
            return Some((path, file_off, size));
        }
        file = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }

    let mut child = romfs_u32(romfs, dir_abs.checked_add(0x08)?)?;
    while child != u32::MAX {
        let rel = usize::try_from(child).ok()?;
        if rel >= hdr.dir_meta_size {
            return None;
        }
        let abs = hdr.dir_meta_off.checked_add(rel)?;
        let name = romfs_name(romfs, abs, 0x18, 0x14)?;
        let child_prefix = if prefix.is_empty() {
            format!("/{}", name)
        } else {
            format!("{}/{}", prefix, name)
        };
        if let Some(hit) =
            romfs_path_for_data_offset_in_dir(romfs, hdr, child, &child_prefix, data_off, depth + 1)
        {
            return Some(hit);
        }
        child = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }

    None
}

fn compute_tiled_size(stride: u32, height: u32, block_height_log2: u32) -> usize {
    const GOB_W: usize = 64;
    const GOB_H: usize = 8;
    const GOB_SIZE: usize = 512;
    let bpp: usize = 4;
    let width_bytes = stride as usize * bpp;
    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_rows = (height as usize + rows_per_block - 1) / rows_per_block;
    gobs_per_row * block_rows * block_height * GOB_SIZE
}

fn unswizzle_block_linear(
    src: &[u8],
    stride: u32,
    height: u32,
    bpp: usize,
    block_height_log2: u32,
) -> Vec<u8> {
    const GOB_W: usize = 64;
    const GOB_H: usize = 8;
    const GOB_SIZE: usize = 512;
    let stride_px = stride as usize;
    let height = height as usize;
    let dst_stride = stride_px * bpp;
    let mut dst = vec![0u8; dst_stride * height];
    let width_bytes = stride_px * bpp;
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
        for x in 0..stride_px {
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

struct AddressSpaceMemory<'a> {
    addr_space: &'a nexium_memory::AddressSpace,
}

impl nexium_cmif::Memory for AddressSpaceMemory<'_> {
    fn read(&self, addr: u64, dst: &mut [u8]) -> bool {
        self.addr_space.read(addr, dst).is_ok()
    }

    fn write(&self, addr: u64, src: &[u8]) -> bool {
        self.addr_space.write(addr, src).is_ok()
    }
}

fn make_cmif_ctx<'a>(
    ctx: &'a ipc::IpcCtx,
    mem: &'a AddressSpaceMemory<'a>,
    recv_buffers: &'a [nexium_cmif::CmifBuffer],
    recv_statics: &'a [nexium_cmif::CmifBuffer],
    send_buffers: &'a [nexium_cmif::CmifBuffer],
    send_statics: &'a [nexium_cmif::CmifBuffer],
) -> nexium_cmif::DispatchCtx<'a> {
    let in_off = ctx.cmif_in_data_off;
    let in_len = ctx.cmif_in_data_len;
    let end = (in_off + in_len).min(ctx.buf.len());
    nexium_cmif::DispatchCtx {
        input_data: &ctx.buf[in_off..end],
        recv_buffers,
        recv_statics,
        send_buffers,
        send_statics,
        mem,
    }
}

fn convert_buffers(src: &[ipc::IpcBuffer]) -> Vec<nexium_cmif::CmifBuffer> {
    src.iter()
        .map(|b| nexium_cmif::CmifBuffer {
            addr: b.addr,
            size: b.size,
        })
        .collect()
}

struct HomebrewEntry {
    name: String,
    size: i64,
}

fn enumerate_homebrew_nros(dir: &Option<std::path::PathBuf>) -> Vec<HomebrewEntry> {
    let Some(dir) = dir else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<HomebrewEntry> = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let is_nro = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("nro"))
            .unwrap_or(false);
        if !is_nro {
            continue;
        }
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let size = entry.metadata().map(|m| m.len() as i64).unwrap_or(0);
        out.push(HomebrewEntry { name, size });
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn cmif_dispatch_set(
    kernel: &mut Kernel,
    ctx: &ipc::IpcCtx,
) -> Option<nexium_cmif::DispatchOutcome> {
    let recv_buffers = convert_buffers(&ctx.recv_buffers);
    let recv_statics = convert_buffers(&ctx.recv_statics);
    let send_buffers = convert_buffers(&ctx.send_buffers);
    let send_statics = convert_buffers(&ctx.send_statics);
    let mem = AddressSpaceMemory {
        addr_space: &*kernel.address_space,
    };
    let mut cmif_ctx = make_cmif_ctx(
        ctx,
        &mem,
        &recv_buffers,
        &recv_statics,
        &send_buffers,
        &send_statics,
    );
    kernel
        .services
        .set
        .dispatch_cmif(ctx.cmif_in.cmd_id, &mut cmif_ctx)
}
