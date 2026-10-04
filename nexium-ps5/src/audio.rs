use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

unsafe extern "C" {
    fn sceAudioOutInit() -> i32;
    fn sceAudioOutOpen(user: i32, port_type: i32, index: i32, grain: u32, rate: u32, format: u32) -> i32;
    fn sceAudioOutOutput(port: i32, samples: *const c_void) -> i32;
    fn sceAudioOutClose(port: i32) -> i32;
}

pub const RATE: u32 = 48_000;
pub const GRAIN: usize = 256;
const SYSTEM_USER: i32 = 0xff;
const FORMAT_S16_STEREO: u32 = 1;

pub struct Ring {
    data: Vec<i16>,
    read: usize,
    len: usize,
}

impl Ring {
    fn new(frames: usize) -> Self {
        Self { data: vec![0; frames * 2], read: 0, len: 0 }
    }

    pub fn free_frames(&self) -> usize {
        (self.data.len() - self.len) / 2
    }

    pub fn push(&mut self, interleaved: &[i16]) -> usize {
        let n = interleaved.len().min(self.data.len() - self.len) & !1;
        let cap = self.data.len();
        for (i, &s) in interleaved[..n].iter().enumerate() {
            self.data[(self.read + self.len + i) % cap] = s;
        }
        self.len += n;
        n / 2
    }

    fn pop_into(&mut self, out: &mut [i16]) -> usize {
        let n = out.len().min(self.len);
        let cap = self.data.len();
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = self.data[(self.read + i) % cap];
        }
        out[n..].fill(0);
        self.read = (self.read + n) % cap;
        self.len -= n;
        n / 2
    }
}

#[derive(Default)]
pub struct AudioStats {
    pub grains: AtomicU64,
    pub audible_grains: AtomicU64,
    pub peak: AtomicU64,
    pub underrun_frames: AtomicU64,
    pub max_block_us: AtomicU64,
}

pub struct AudioOut {
    pub ring: Arc<Mutex<Ring>>,
    pub stats: Arc<AudioStats>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    port: i32,
}

impl AudioOut {
    pub fn open(buffer_frames: usize) -> Result<Self, String> {
        unsafe {
            let init = sceAudioOutInit();
            if init != 0 && init as u32 != 0x8026_000e {
                return Err(format!("sceAudioOutInit {init:#x}"));
            }
            let port = sceAudioOutOpen(SYSTEM_USER, 0, 0, GRAIN as u32, RATE, FORMAT_S16_STEREO);
            if port < 0 {
                return Err(format!("sceAudioOutOpen {port:#x}"));
            }
            let ring = Arc::new(Mutex::new(Ring::new(buffer_frames)));
            let stats = Arc::new(AudioStats::default());
            let stop = Arc::new(AtomicBool::new(false));
            let worker = {
                let (ring, stats, stop) = (ring.clone(), stats.clone(), stop.clone());
                std::thread::Builder::new()
                    .name("nexium-audio-out".into())
                    .spawn(move || {
                        let mut grain = vec![0i16; GRAIN * 2];
                        while !stop.load(Ordering::Relaxed) {
                            let got = ring.lock().unwrap().pop_into(&mut grain);
                            if got < GRAIN {
                                stats.underrun_frames.fetch_add((GRAIN - got) as u64, Ordering::Relaxed);
                            }
                            let started = Instant::now();
                            sceAudioOutOutput(port, grain.as_ptr().cast());
                            let us = started.elapsed().as_micros() as u64;
                            stats.max_block_us.fetch_max(us, Ordering::Relaxed);
                            stats.grains.fetch_add(1, Ordering::Relaxed);
                        }
                        sceAudioOutOutput(port, std::ptr::null());
                    })
                    .map_err(|e| format!("audio worker: {e}"))?
            };
            Ok(Self { ring, stats, stop, worker: Some(worker), port })
        }
    }
}

pub struct PullAudioOut {
    pub stats: Arc<AudioStats>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    port: i32,
}

impl PullAudioOut {
    pub fn open(mut fill: Box<dyn FnMut(&mut [i16]) + Send>) -> Result<Self, String> {
        unsafe {
            let init = sceAudioOutInit();
            if init != 0 && init as u32 != 0x8026_000e {
                return Err(format!("sceAudioOutInit {init:#x}"));
            }
            let port = sceAudioOutOpen(SYSTEM_USER, 0, 0, GRAIN as u32, RATE, FORMAT_S16_STEREO);
            if port < 0 {
                return Err(format!("sceAudioOutOpen {port:#x}"));
            }
            let stats = Arc::new(AudioStats::default());
            let stop = Arc::new(AtomicBool::new(false));
            let worker = {
                let (stats, stop) = (stats.clone(), stop.clone());
                std::thread::Builder::new()
                    .name("nexium-audio-out".into())
                    .spawn(move || {
                        let mut grain = vec![0i16; GRAIN * 2];
                        while !stop.load(Ordering::Relaxed) {
                            fill(&mut grain);
                            let peak = grain.iter().map(|s| s.unsigned_abs() as u64).max().unwrap_or(0);
                            if peak > 64 {
                                stats.audible_grains.fetch_add(1, Ordering::Relaxed);
                            }
                            stats.peak.fetch_max(peak, Ordering::Relaxed);
                            let started = Instant::now();
                            sceAudioOutOutput(port, grain.as_ptr().cast());
                            stats.max_block_us.fetch_max(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                            stats.grains.fetch_add(1, Ordering::Relaxed);
                        }
                        sceAudioOutOutput(port, std::ptr::null());
                    })
                    .map_err(|e| format!("audio worker: {e}"))?
            };
            Ok(Self { stats, stop, worker: Some(worker), port })
        }
    }
}

impl Drop for PullAudioOut {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        unsafe {
            sceAudioOutClose(self.port);
        }
    }
}

impl Drop for AudioOut {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        unsafe {
            sceAudioOutClose(self.port);
        }
    }
}
