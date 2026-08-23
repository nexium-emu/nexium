use crate::video_decode::OwnedI420Frame;
use crate::video_ffmpeg::{i420_frame, FfmpegCodec, FfmpegDecoder};
use crossbeam::channel::{unbounded, Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

pub const FRAME_CACHE_CAPACITY: usize = 32;
const MAX_PENDING_TARGETS: usize = 64;

#[derive(Default)]
pub struct VideoFrameCacheState {
    frames: HashMap<u64, OwnedI420Frame>,
    order: VecDeque<u64>,
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
    },
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
    Work(DecodeWork),
    Shutdown,
}

struct ChannelDecoder {
    decoder: FfmpegDecoder,
    failed: Arc<AtomicBool>,
    targets: VecDeque<u64>,
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

    fn handle(&mut self, work: DecodeWork) {
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
            } => self.decode_packet(fd, &packet, target_luma_iova, detail),
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
    ) {
        let healthy = match self.channels.get_mut(&fd) {
            Some(channel) => {
                decode_into_cache(&self.cache, fd, channel, packet, target_luma_iova, detail)
            }
            None => return,
        };
        if !healthy {
            if let Some(channel) = self.channels.remove(&fd) {
                channel.failed.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn decode_into_cache(
    cache: &VideoFrameCache,
    fd: u32,
    channel: &mut ChannelDecoder,
    packet: &[u8],
    target_luma_iova: u64,
    detail: PacketDetail,
) -> bool {
    channel.targets.push_back(target_luma_iova);
    while channel.targets.len() > MAX_PENDING_TARGETS {
        channel.targets.pop_front();
        log::warn!(
            "[video-decode] fd={} decoder is more than {} packets behind; output surfaces will shift",
            fd,
            MAX_PENDING_TARGETS
        );
    }
    if let Err(error) = channel.decoder.submit(packet) {
        log::warn!("[video-decode] fd={} ffmpeg submit failed: {}", fd, error);
        return false;
    }
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
        let Some(luma_iova) = channel.targets.pop_front() else {
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
        lock_frame_cache(cache).insert(luma_iova, frame);
        log_decoded_frame(fd, width, height, packet.len(), target_luma_iova, luma_iova, detail);
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
    if frame_index >= 32 && frame_index % 300 != 0 {
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
        } => log::info!(
            "[video-decode] frame={} fd={} {}x{} bytes={} luma_iova={:#x} picture={} guest_frame={}",
            frame_index,
            fd,
            width,
            height,
            packet_len,
            luma_iova,
            picture_index,
            guest_frame
        ),
    }
}

fn worker_loop(rx: Receiver<Message>, cache: VideoFrameCache) {
    let mut state = DecodeWorkerState::new(cache);
    while let Ok(message) = rx.recv() {
        match message {
            Message::Work(work) => state.handle(work),
            Message::Shutdown => return,
        }
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
        &self.cache
    }

    pub fn submit(&self, work: DecodeWork) {
        match self.dispatch() {
            Dispatch::Worker { tx, .. } => {
                let _ = tx.send(Message::Work(work));
            }
            Dispatch::Inline(state) => {
                state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .handle(work);
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
