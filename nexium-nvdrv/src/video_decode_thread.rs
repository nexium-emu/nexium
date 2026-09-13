use crate::video_decode::OwnedI420Frame;
use crate::video_ffmpeg::{i420_frame, FfmpegCodec, FfmpegDecoder};
use crossbeam::channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

pub const FRAME_CACHE_CAPACITY: usize = 32;
const PENDING_TARGET_WARNING_THRESHOLD: usize = 64;
const OUTPUT_POLL_INTERVAL: Duration = Duration::from_millis(1);
const H264_OUTPUT_WAIT_BUDGET: Duration = Duration::from_millis(4);

#[derive(Default)]
pub struct VideoFrameCacheState {
    frames: HashMap<u64, OwnedI420Frame>,
    order: VecDeque<u64>,
    pending: HashMap<u64, (u64, PacketDetail)>,
    next_generation: u64,
}

impl VideoFrameCacheState {
    pub fn insert(&mut self, luma_iova: u64, frame: OwnedI420Frame) {
        self.frames.insert(luma_iova, frame);
        self.order.retain(|key| *key != luma_iova);
        self.order.push_back(luma_iova);
        while self.order.len() > FRAME_CACHE_CAPACITY {
            if let Some(old_key) = self.order.pop_front() {
                self.frames.remove(&old_key);
            }
        }
    }

    pub fn remove(&mut self, luma_iova: u64) {
        self.frames.remove(&luma_iova);
        self.order.retain(|key| *key != luma_iova);
    }

    pub fn take(&mut self, luma_iova: u64) -> Option<OwnedI420Frame> {
        self.order.retain(|key| *key != luma_iova);
        self.frames.remove(&luma_iova)
    }

    fn begin_decode(&mut self, luma_iova: u64, detail: PacketDetail) -> u64 {
        self.remove(luma_iova);
        self.next_generation = self.next_generation.wrapping_add(1);
        self.pending
            .insert(luma_iova, (self.next_generation, detail));
        self.next_generation
    }

    fn publish_decoded(&mut self, luma_iova: u64, generation: u64, frame: OwnedI420Frame) {
        if self.pending.get(&luma_iova).map(|pending| pending.0) == Some(generation) {
            self.pending.remove(&luma_iova);
            self.insert(luma_iova, frame);
        }
    }

    pub fn expire_decode(&mut self, luma_iova: u64) {
        self.pending.remove(&luma_iova);
    }

    pub fn get_cloned(&self, luma_iova: u64) -> Option<OwnedI420Frame> {
        self.frames.get(&luma_iova).cloned()
    }

    pub fn latest_cloned(&self) -> Option<(u64, OwnedI420Frame)> {
        self.order
            .iter()
            .rev()
            .find_map(|key| self.frames.get(key).map(|frame| (*key, frame.clone())))
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

pub type VideoFrameCache = Arc<Mutex<VideoFrameCacheState>>;

pub fn new_frame_cache() -> VideoFrameCache {
    Arc::new(Mutex::new(VideoFrameCacheState::default()))
}

pub fn lock_frame_cache(cache: &VideoFrameCache) -> MutexGuard<'_, VideoFrameCacheState> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PacketDetail {
    Vp9 {
        show_frame: bool,
    },
    H264 {
        picture_index: u32,
        guest_frame: u32,
        picture_order: i32,
        is_idr: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingFrameTarget {
    generation: u64,
    luma_iova: u64,
    packet_len: usize,
    detail: PacketDetail,
}

fn queue_frame_target(
    targets: &mut VecDeque<PendingFrameTarget>,
    luma_iova: u64,
    packet_len: usize,
    detail: PacketDetail,
) {
    if matches!(detail, PacketDetail::Vp9 { show_frame: false }) {
        return;
    }
    targets.push_back(PendingFrameTarget {
        generation: 0,
        luma_iova,
        packet_len,
        detail,
    });
}

fn pop_frame_target(targets: &mut VecDeque<PendingFrameTarget>) -> Option<PendingFrameTarget> {
    if !matches!(targets.front()?.detail, PacketDetail::H264 { .. }) {
        return targets.pop_front();
    }
    let boundary = targets
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(index, target)| {
            matches!(target.detail, PacketDetail::H264 { is_idr: true, .. }).then_some(index)
        })
        .unwrap_or(targets.len());
    let index = (0..boundary).min_by_key(|index| match targets[*index].detail {
        PacketDetail::H264 { picture_order, .. } => picture_order,
        _ => i32::MAX,
    })?;
    targets.remove(index)
}

pub enum DecodeWork {
    Configure {
        fd: u32,
        codec: FfmpegCodec,
        width: u32,
        height: u32,
        failed: Arc<AtomicBool>,
    },
    Packet {
        fd: u32,
        packet: Vec<u8>,
        target_luma_iova: u64,
        detail: PacketDetail,
    },
    Release {
        fd: u32,
    },
}

enum Message {
    Work(DecodeWork, Option<u64>),
    Shutdown,
}

struct ChannelDecoder {
    decoder: FfmpegDecoder,
    failed: Arc<AtomicBool>,
    targets: VecDeque<PendingFrameTarget>,
}

struct DecodeWorkerState {
    cache: VideoFrameCache,
    channels: HashMap<u32, ChannelDecoder>,
}

impl DecodeWorkerState {
    fn new(cache: VideoFrameCache) -> Self {
        Self {
            cache,
            channels: HashMap::new(),
        }
    }

    fn handle(&mut self, work: DecodeWork, generation: Option<u64>) {
        match work {
            DecodeWork::Configure {
                fd,
                codec,
                width,
                height,
                failed,
            } => self.configure(fd, codec, width, height, failed),
            DecodeWork::Packet {
                fd,
                packet,
                target_luma_iova,
                detail,
            } => self.decode_packet(fd, &packet, target_luma_iova, detail, generation),
            DecodeWork::Release { fd } => {
                self.channels.remove(&fd);
            }
        }
    }

    fn configure(
        &mut self,
        fd: u32,
        codec: FfmpegCodec,
        width: u32,
        height: u32,
        failed: Arc<AtomicBool>,
    ) {
        self.channels.remove(&fd);
        match FfmpegDecoder::new(width, height, codec) {
            Ok(decoder) => {
                match codec {
                    FfmpegCodec::Vp9 => log::info!(
                        "[video-decode] fd={} ffmpeg vp9 software decoder {}x{}",
                        fd,
                        width,
                        height
                    ),
                    FfmpegCodec::H264 => log::info!(
                        "[video-decode] fd={} ffmpeg software decoder {}x{}",
                        fd,
                        width,
                        height
                    ),
                }
                self.channels.insert(
                    fd,
                    ChannelDecoder {
                        decoder,
                        failed,
                        targets: VecDeque::new(),
                    },
                );
            }
            Err(error) => {
                log::warn!("[video-decode] fd={} ffmpeg init failed: {}", fd, error);
                failed.store(true, Ordering::Relaxed);
            }
        }
    }

    fn decode_packet(
        &mut self,
        fd: u32,
        packet: &[u8],
        target_luma_iova: u64,
        detail: PacketDetail,
        generation: Option<u64>,
    ) {
        let healthy = match self.channels.get_mut(&fd) {
            Some(channel) => decode_into_cache(
                &self.cache,
                fd,
                channel,
                packet,
                target_luma_iova,
                detail,
                generation,
            ),
            None => return,
        };
        if !healthy {
            if let Some(channel) = self.channels.remove(&fd) {
                channel.failed.store(true, Ordering::Relaxed);
            }
        }
    }

    fn has_pending_frames(&self) -> bool {
        self.channels
            .values()
            .any(|channel| !channel.targets.is_empty())
    }

    fn drain_outputs(&mut self) {
        let cache = &self.cache;
        self.channels.retain(|fd, channel| {
            let healthy = drain_channel_outputs(cache, *fd, channel);
            if !healthy {
                channel.failed.store(true, Ordering::Relaxed);
            }
            healthy
        });
    }
}

fn decode_into_cache(
    cache: &VideoFrameCache,
    fd: u32,
    channel: &mut ChannelDecoder,
    packet: &[u8],
    target_luma_iova: u64,
    detail: PacketDetail,
    generation: Option<u64>,
) -> bool {
    let previous_count = channel.targets.len();
    queue_frame_target(&mut channel.targets, target_luma_iova, packet.len(), detail);
    if channel.targets.len() > previous_count {
        let generation = generation
            .unwrap_or_else(|| lock_frame_cache(cache).begin_decode(target_luma_iova, detail));
        channel.targets.back_mut().unwrap().generation = generation;
    }
    if previous_count == PENDING_TARGET_WARNING_THRESHOLD && channel.targets.len() > previous_count
    {
        log::warn!(
            "[video-decode] fd={} decoder is more than {} visible frames behind",
            fd,
            PENDING_TARGET_WARNING_THRESHOLD
        );
    }
    if let Err(error) = channel.decoder.submit(packet) {
        log::warn!("[video-decode] fd={} ffmpeg submit failed: {}", fd, error);
        return false;
    }
    drain_channel_outputs(cache, fd, channel)
}

fn drain_channel_outputs(cache: &VideoFrameCache, fd: u32, channel: &mut ChannelDecoder) -> bool {
    let width = channel.decoder.width();
    let height = channel.decoder.height();
    loop {
        let raw = match channel.decoder.receive() {
            Ok(Some(raw)) => raw,
            Ok(None) => return true,
            Err(error) => {
                log::warn!("[video-decode] fd={} ffmpeg decode failed: {}", fd, error);
                return false;
            }
        };
        let Some(target) = pop_frame_target(&mut channel.targets) else {
            log::warn!(
                "[video-decode] fd={} ffmpeg produced an unassociated frame",
                fd
            );
            continue;
        };
        let frame = match i420_frame(width, height, &raw) {
            Ok(frame) => frame,
            Err(error) => {
                log::warn!("[video-decode] fd={} frame invalid: {}", fd, error);
                return false;
            }
        };
        lock_frame_cache(cache).publish_decoded(target.luma_iova, target.generation, frame);
        log_decoded_frame(
            fd,
            width,
            height,
            target.packet_len,
            target.luma_iova,
            target.luma_iova,
            target.detail,
        );
    }
}

fn log_decoded_frame(
    fd: u32,
    width: u32,
    height: u32,
    packet_len: usize,
    target_luma_iova: u64,
    luma_iova: u64,
    detail: PacketDetail,
) {
    static DECODED_FRAMES: AtomicU64 = AtomicU64::new(0);
    let frame_index = DECODED_FRAMES.fetch_add(1, Ordering::Relaxed);
    if frame_index >= 32 && frame_index % 300 != 0 && !crate::video_trace_enabled() {
        return;
    }
    match detail {
        PacketDetail::Vp9 { show_frame } => log::info!(
            "[video-decode] vp9 frame={} fd={} {}x{} bytes={} packet_luma_iova={:#x} luma_iova={:#x} show={}",
            frame_index,
            fd,
            width,
            height,
            packet_len,
            target_luma_iova,
            luma_iova,
            show_frame
        ),
        PacketDetail::H264 {
            picture_index,
            guest_frame,
            picture_order,
            ..
        } => log::info!(
            "[video-decode] frame={} fd={} {}x{} bytes={} luma_iova={:#x} picture={} guest_frame={} poc={}",
            frame_index,
            fd,
            width,
            height,
            packet_len,
            luma_iova,
            picture_index,
            guest_frame,
            picture_order
        ),
    }
}

fn worker_loop(rx: Receiver<Message>, cache: VideoFrameCache) {
    let mut state = DecodeWorkerState::new(cache);
    loop {
        let message = if state.has_pending_frames() {
            match rx.recv_timeout(OUTPUT_POLL_INTERVAL) {
                Ok(message) => Some(message),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match rx.recv() {
                Ok(message) => Some(message),
                Err(_) => return,
            }
        };
        match message {
            Some(Message::Work(work, generation)) => state.handle(work, generation),
            Some(Message::Shutdown) => return,
            None => {}
        }
        state.drain_outputs();
    }
}

enum Dispatch {
    Worker {
        tx: Sender<Message>,
        handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    },
    Inline(Mutex<DecodeWorkerState>),
}

pub struct VideoDecoder {
    cache: VideoFrameCache,
    dispatch: OnceLock<Dispatch>,
}

impl VideoDecoder {
    pub fn new() -> Self {
        Self {
            cache: new_frame_cache(),
            dispatch: OnceLock::new(),
        }
    }

    pub fn cache(&self) -> &VideoFrameCache {
        if let Some(Dispatch::Inline(state)) = self.dispatch.get() {
            state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain_outputs();
        }
        &self.cache
    }

    pub fn wait_for_frame(&self, luma_iova: u64, timeout: Duration) -> Option<OwnedI420Frame> {
        let started = std::time::Instant::now();
        let requested_deadline = started + timeout;
        let h264_deadline = started + timeout.min(H264_OUTPUT_WAIT_BUDGET);
        loop {
            let deadline = {
                let mut cache = lock_frame_cache(self.cache());
                if let Some(frame) = cache.take(luma_iova) {
                    return Some(frame);
                }
                match cache.pending.get(&luma_iova) {
                    Some((_, PacketDetail::Vp9 { .. })) => requested_deadline,
                    Some((_, PacketDetail::H264 { .. })) => h264_deadline,
                    None => return None,
                }
            };
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            std::thread::sleep(remaining.min(OUTPUT_POLL_INTERVAL));
        }
    }

    pub fn submit(&self, work: DecodeWork) {
        let generation = match &work {
            DecodeWork::Packet {
                target_luma_iova,
                detail,
                ..
            } if !matches!(detail, PacketDetail::Vp9 { show_frame: false }) => {
                Some(lock_frame_cache(&self.cache).begin_decode(*target_luma_iova, *detail))
            }
            _ => None,
        };
        match self.dispatch() {
            Dispatch::Worker { tx, .. } => {
                let _ = tx.send(Message::Work(work, generation));
            }
            Dispatch::Inline(state) => {
                let mut state = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.handle(work, generation);
                state.drain_outputs();
            }
        }
    }

    fn dispatch(&self) -> &Dispatch {
        self.dispatch.get_or_init(|| {
            if !asynchronous_decode_enabled() {
                return Dispatch::Inline(Mutex::new(DecodeWorkerState::new(self.cache.clone())));
            }
            let (tx, rx) = unbounded();
            let cache = self.cache.clone();
            match std::thread::Builder::new()
                .name("nexium-video-decode".into())
                .spawn(move || worker_loop(rx, cache))
            {
                Ok(handle) => Dispatch::Worker {
                    tx,
                    handle: Mutex::new(Some(handle)),
                },
                Err(error) => {
                    log::warn!(
                        "[video-decode] worker thread unavailable ({}), decoding inline",
                        error
                    );
                    Dispatch::Inline(Mutex::new(DecodeWorkerState::new(self.cache.clone())))
                }
            }
        })
    }
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        let Some(Dispatch::Worker { tx, handle }) = self.dispatch.get() else {
            return;
        };
        let _ = tx.send(Message::Shutdown);
        if let Some(handle) = handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }
}

fn asynchronous_decode_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("NEXIUM_ASYNC_NVDEC").ok().as_deref() != Some("0"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_decode::OwnedI420Frame;

    fn test_frame(width: usize, height: usize) -> OwnedI420Frame {
        let y = vec![0u8; width * height];
        let c = vec![0u8; width * height / 4];
        OwnedI420Frame::from_strided_planes(
            width,
            height,
            (width, width / 2, width / 2),
            &y,
            &c,
            &c,
        )
        .expect("frame")
    }

    #[test]
    #[ignore = "requires FFmpeg; set NEXIUM_FFMPEG and run with --ignored"]
    fn h264_b_frames_keep_their_guest_output_surfaces() {
        const STREAM: &[u8] = include_bytes!("../tests/fixtures/h264_b_frames.h264");
        let offsets: Vec<_> = STREAM
            .windows(5)
            .enumerate()
            .filter_map(|(i, word)| (word == [0, 0, 0, 1, 9]).then_some(i))
            .chain(std::iter::once(STREAM.len()))
            .collect();
        let order = [
            0, 8, 4, 2, 6, 16, 12, 10, 14, 24, 20, 18, 22, 32, 28, 26, 30, 40, 36, 34, 38, 46, 42,
            44,
        ];
        assert_eq!(offsets.len(), order.len() + 1);
        let cache = new_frame_cache();
        let mut channel = ChannelDecoder {
            decoder: FfmpegDecoder::new(16, 16, FfmpegCodec::H264).unwrap(),
            failed: Arc::new(AtomicBool::new(false)),
            targets: VecDeque::new(),
        };
        for (i, packet) in offsets.windows(2).enumerate() {
            assert!(decode_into_cache(
                &cache,
                1,
                &mut channel,
                &STREAM[packet[0]..packet[1]],
                0x1000 + i as u64 * 0x100,
                PacketDetail::H264 {
                    picture_index: i as u32,
                    guest_frame: i as u32,
                    picture_order: order[i],
                    is_idr: i == 0,
                },
                None
            ));
        }
        channel.decoder.close_input();
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !channel.targets.is_empty() && std::time::Instant::now() < deadline {
            let healthy = drain_channel_outputs(&cache, 1, &mut channel);
            if !healthy {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(channel.targets.is_empty());
        for (i, poc) in order.into_iter().enumerate() {
            let frame = lock_frame_cache(&cache)
                .get_cloned(0x1000 + i as u64 * 0x100)
                .unwrap();
            assert_eq!(
                frame.y()[0],
                (32 + 4 * poc) as u8,
                "decoded packet {i} poc={poc}"
            );
        }
    }

    #[test]
    fn h264_picture_order_reset_keeps_prior_sequence_first() {
        let mut targets = VecDeque::new();
        for (index, order) in [0, 8, 4, 2, 6, 0, 6, 2, 4].into_iter().enumerate() {
            queue_frame_target(
                &mut targets,
                index as u64,
                1,
                PacketDetail::H264 {
                    picture_index: index as u32,
                    guest_frame: index as u32,
                    picture_order: order,
                    is_idr: index == 0 || index == 5,
                },
            );
        }
        let received: Vec<_> = std::iter::from_fn(|| pop_frame_target(&mut targets))
            .map(|target| target.luma_iova)
            .collect();
        assert_eq!(received, [0, 3, 2, 4, 1, 5, 7, 8, 6]);
    }

    #[test]
    fn vic_waits_for_its_requested_surface_instead_of_a_newer_frame() {
        let decoder = VideoDecoder::new();
        lock_frame_cache(decoder.cache()).insert(0x2000, test_frame(4, 4));
        let cache = decoder.cache().clone();
        let generation =
            lock_frame_cache(&cache).begin_decode(0x1000, PacketDetail::Vp9 { show_frame: true });
        let ready = std::sync::Arc::new(std::sync::Barrier::new(2));
        let writer_ready = ready.clone();
        let worker = std::thread::spawn(move || {
            writer_ready.wait();
            std::thread::sleep(Duration::from_millis(5));
            lock_frame_cache(&cache).publish_decoded(0x1000, generation, test_frame(2, 2));
        });
        ready.wait();
        let requested = decoder
            .wait_for_frame(0x1000, Duration::from_secs(1))
            .unwrap();
        worker.join().unwrap();
        assert_eq!(requested.width(), 2);
        assert!(decoder.wait_for_frame(0x3000, Duration::ZERO).is_none());
    }

    #[test]
    fn h264_wait_does_not_block_the_packets_needed_for_reordering() {
        let decoder = VideoDecoder::new();
        lock_frame_cache(decoder.cache()).begin_decode(
            0x1000,
            PacketDetail::H264 {
                picture_index: 0,
                guest_frame: 0,
                picture_order: 0,
                is_idr: true,
            },
        );
        let started = std::time::Instant::now();
        assert!(decoder
            .wait_for_frame(0x1000, Duration::from_secs(1))
            .is_none());
        assert!(started.elapsed() < Duration::from_millis(250));
    }

    #[test]
    fn reused_video_surface_rejects_late_prior_decode() {
        let mut cache = VideoFrameCacheState::default();
        let first = cache.begin_decode(0x1000, PacketDetail::Vp9 { show_frame: true });
        let second = cache.begin_decode(0x1000, PacketDetail::Vp9 { show_frame: true });
        cache.publish_decoded(0x1000, first, test_frame(2, 2));
        assert!(cache.get_cloned(0x1000).is_none());
        cache.publish_decoded(0x1000, second, test_frame(4, 4));
        assert_eq!(cache.take(0x1000).unwrap().width(), 4);
        assert!(cache.get_cloned(0x1000).is_none());
        let expired = cache.begin_decode(0x1000, PacketDetail::Vp9 { show_frame: true });
        cache.expire_decode(0x1000);
        cache.publish_decoded(0x1000, expired, test_frame(2, 2));
        assert!(cache.get_cloned(0x1000).is_none());
        let current = cache.begin_decode(0x1000, PacketDetail::Vp9 { show_frame: true });
        cache.publish_decoded(0x1000, current, test_frame(6, 6));
        assert_eq!(cache.take(0x1000).unwrap().width(), 6);
    }

    #[test]
    fn invisible_vp9_packets_do_not_shift_visible_output_surfaces() {
        let mut targets = VecDeque::new();
        for (luma_iova, show_frame) in [
            (0x1000, true),
            (0x2000, false),
            (0x3000, false),
            (0x4000, true),
            (0x5000, false),
            (0x6000, true),
        ] {
            queue_frame_target(
                &mut targets,
                luma_iova,
                luma_iova as usize / 256,
                PacketDetail::Vp9 { show_frame },
            );
        }
        for (expected_iova, expected_bytes) in [(0x1000, 16), (0x4000, 64), (0x6000, 96)] {
            let target = targets.pop_front().expect("visible output target");
            assert_eq!(target.luma_iova, expected_iova);
            assert_eq!(target.packet_len, expected_bytes);
            assert_eq!(target.detail, PacketDetail::Vp9 { show_frame: true });
        }
        assert!(targets.is_empty());
    }

    #[test]
    fn delayed_output_keeps_all_surface_and_packet_associations() {
        let mut targets = VecDeque::new();
        let packet_count = PENDING_TARGET_WARNING_THRESHOLD * 2;
        for index in 0..packet_count {
            queue_frame_target(
                &mut targets,
                0x1000 + index as u64 * 0x100,
                index + 1,
                PacketDetail::H264 {
                    picture_index: index as u32,
                    guest_frame: index as u32 + 100,
                    picture_order: index as i32,
                    is_idr: index == 0,
                },
            );
        }
        assert_eq!(targets.len(), packet_count);
        for index in 0..packet_count {
            let target = targets.pop_front().expect("pending output target");
            assert_eq!(target.luma_iova, 0x1000 + index as u64 * 0x100);
            assert_eq!(target.packet_len, index + 1);
            assert_eq!(
                target.detail,
                PacketDetail::H264 {
                    picture_index: index as u32,
                    guest_frame: index as u32 + 100,
                    picture_order: index as i32,
                    is_idr: index == 0,
                }
            );
        }
        assert!(targets.is_empty());
    }

    #[test]
    #[ignore = "requires FFmpeg; set NEXIUM_FFMPEG and run with --ignored"]
    fn worker_publishes_visible_frames_without_later_packets() {
        let fixture = include_bytes!("../tests/fixtures/vp9_hidden_frames.ivf");
        let decoder = VideoDecoder::new();
        let cache = decoder.cache().clone();
        let worker_cache = cache.clone();
        let (tx, rx) = unbounded();
        let handle = std::thread::spawn(move || worker_loop(rx, worker_cache));
        assert!(decoder
            .dispatch
            .set(Dispatch::Worker {
                tx,
                handle: Mutex::new(Some(handle)),
            })
            .is_ok());
        let failed = Arc::new(AtomicBool::new(false));
        decoder.submit(DecodeWork::Configure {
            fd: 1,
            codec: FfmpegCodec::Vp9,
            width: 16,
            height: 16,
            failed: failed.clone(),
        });
        let mut offset = 32;
        for index in 0..24 {
            let size = u32::from_le_bytes(fixture[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 12;
            let packet = fixture[offset..offset + size].to_vec();
            offset += size;
            let show_frame = index % 3 != 1;
            let target_luma_iova = 0x1000 + index as u64 * 0x100;
            decoder.submit(DecodeWork::Packet {
                fd: 1,
                packet,
                target_luma_iova,
                detail: PacketDetail::Vp9 { show_frame },
            });
            if !show_frame {
                continue;
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            let frame = loop {
                if let Some(frame) = lock_frame_cache(&cache).get_cloned(target_luma_iova) {
                    break frame;
                }
                assert!(!failed.load(Ordering::Relaxed), "FFmpeg decoding failed");
                assert!(
                    std::time::Instant::now() < deadline,
                    "visible packet {index} was not published without later input"
                );
                std::thread::sleep(Duration::from_millis(2));
            };
            let expected_luma = 16 + 8 * index as u8;
            assert!(frame.y().iter().all(|&luma| luma == expected_luma));
        }
        assert_eq!(offset, fixture.len());
        let cache = lock_frame_cache(&cache);
        assert_eq!(cache.len(), 16);
        for index in (0..24).filter(|index| index % 3 == 1) {
            assert!(cache.get_cloned(0x1000 + index * 0x100).is_none());
        }
    }

    #[test]
    fn frame_cache_evicts_the_oldest_entry_past_capacity() {
        let mut cache = VideoFrameCacheState::default();
        for index in 0..FRAME_CACHE_CAPACITY as u64 + 4 {
            cache.insert(index, test_frame(2, 2));
        }
        assert_eq!(cache.len(), FRAME_CACHE_CAPACITY);
        assert!(cache.get_cloned(0).is_none());
        assert!(cache.get_cloned(3).is_none());
        assert!(cache.get_cloned(4).is_some());
    }

    #[test]
    fn reinserting_a_key_refreshes_its_position_without_growing_the_cache() {
        let mut cache = VideoFrameCacheState::default();
        for index in 0..FRAME_CACHE_CAPACITY as u64 {
            cache.insert(index, test_frame(2, 2));
        }
        cache.insert(0, test_frame(2, 2));
        cache.insert(u64::MAX, test_frame(2, 2));
        assert_eq!(cache.len(), FRAME_CACHE_CAPACITY);
        assert!(cache.get_cloned(0).is_some());
        assert!(cache.get_cloned(1).is_none());
    }

    #[test]
    fn latest_cloned_returns_the_most_recently_inserted_frame() {
        let mut cache = VideoFrameCacheState::default();
        assert!(cache.latest_cloned().is_none());
        cache.insert(0x1000, test_frame(2, 2));
        cache.insert(0x2000, test_frame(2, 2));
        cache.insert(0x1000, test_frame(2, 2));
        assert_eq!(cache.latest_cloned().map(|(key, _)| key), Some(0x1000));
    }

    #[test]
    fn submitting_work_returns_without_waiting_for_the_decoder() {
        let decoder = VideoDecoder::new();
        let failed = Arc::new(AtomicBool::new(false));
        let started = std::time::Instant::now();
        for index in 0..64u64 {
            decoder.submit(DecodeWork::Packet {
                fd: 1,
                packet: vec![0u8; 4096],
                target_luma_iova: index,
                detail: PacketDetail::Vp9 { show_frame: true },
            });
        }
        decoder.submit(DecodeWork::Release { fd: 1 });
        let _ = failed;
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
    }
}
