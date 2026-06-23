pub mod audio_lut;
pub mod cpu_context;
pub mod cpu_local;
pub mod handles;
pub mod hid;
pub mod profile;
pub mod session;
pub mod svc;
pub mod svc_defs;
pub mod threads;

use crate::services::FrameOut;
use crate::services::Services;
use nexium_cpu::Cpu;
use nexium_memory::AddressSpace;
use nexium_nvdrv::Nvdrv;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub(crate) const MUTEX_HAS_LISTENERS: u32 = 0x4000_0000;

pub struct Kernel {
    pub address_space: Arc<AddressSpace>,
    pub handles: handles::HandleTable,
    pub threads: threads::Threads,
    pub services: Services,
    pub nvdrv: Nvdrv,
    pub hid: Arc<Mutex<hid::HidShared>>,
    pub sessions: HashMap<u32, session::Session>,
    pub event_signals: HashMap<u32, bool>,
    pub exited_thread_handles: HashSet<u32>,
    pub pending_condvar_signals: HashMap<u64, u32>,
    pub audio_render_condvar: Option<u64>,
    pub tls_buffer: [u8; 0x100],
    pub pending_frames: Vec<FrameOut>,

    pub code_base: u64,
    pub code_size: u64,
    pub heap_base: u64,
    pub heap_size: u64,
    pub stack_base: u64,
    pub stack_size: u64,
    pub tls_base: u64,
    pub tls_pool_base: u64,

    pub aslr_base: u64,
    pub aslr_size: u64,
    pub alias_base: u64,
    pub alias_size: u64,
    pub is_application: bool,
    pub title_id: u64,
    pub total_memory: u64,
    pub heap_committed: u64,

    pub cycle_count: u64,
    pub next_vsync_cycle: u64,
    pub last_sdl_capture: std::time::Instant,
    pub display_ready: bool,
    pub process_exited: bool,
    pub vsync_poll_count: u64,

    pub process_handle: u32,
    pub main_thread_handle: u32,

    pub applet_messages: VecDeque<u32>,
    pub applet_message_event: Option<u32>,
    pub vsync_handles: HashSet<u32>,
    pub nvdrv_sync_events: HashSet<u32>,
    pub gpu_fence_events: HashSet<u32>,
    pub last_vsync: std::time::Instant,
    /// Last time we refreshed the HID shmem from `hid_input`. Refreshed on
    /// both vsync and applet-event polls (the latter being how SDL2-using
    /// games drive their input loop, since SDL_PollEvent pumps applet
    /// events without ever waiting on a vsync handle). The timestamp
    /// throttles those refreshes to ~60 Hz so we don't spam the shmem on
    /// the kHz-rate timeout=0 polls. (RustSwitch commit 49ca133)
    pub last_hid_tick: std::time::Instant,
    /// Per-(svc_imm, calling_lr) streak detector across all SVCs so a guest
    /// spinning on any particular call site shows up. Fires the first time
    /// we cross each power-of-two threshold so logs grow O(log N).
    pub last_generic_svc_imm: u16,
    pub last_generic_svc_lr: u64,
    pub generic_svc_streak: u64,
    pub next_generic_svc_streak_log: u64,
    pub applet_focus_state: u8,
    pub applet_operation_mode: u8,
    pub applet_performance_mode: u32,
    pub display_resolution_change_event: Option<u32>,
    pub library_applet_launchable_event: Option<u32>,
    pub accumulated_suspended_tick_event: Option<u32>,
    pub gpu_error_detected_event: Option<u32>,
    pub friend_invitation_event: Option<u32>,
    pub notification_event: Option<u32>,
    pub acquired_sleep_lock_event: Option<u32>,

    pub nro_mmap: Option<Arc<memmap2::Mmap>>,
    pub nro_romfs_range: Option<std::ops::Range<usize>>,

    pub homebrew_dir: Option<std::path::PathBuf>,
    pub dir_cursor: HashMap<u32, usize>,
    pub open_files: HashMap<(u32, u32), Arc<memmap2::Mmap>>,
    pub sd_root: Option<std::path::PathBuf>,
    pub file_system_roots: HashMap<(u32, u32), std::path::PathBuf>,
    pub open_host_files: HashMap<(u32, u32), std::path::PathBuf>,
    pub open_romfs_files: HashMap<(u32, u32), (usize, usize)>,
    pub open_file_handles: HashMap<(u32, u32), std::fs::File>,
    pub host_file_cache: HashMap<std::path::PathBuf, Arc<memmap2::Mmap>>,
    pub open_dir_lists: HashMap<(u32, u32), (Vec<(String, bool, u64)>, usize)>,

    pub yield_after_svc: bool,
    pub present_pace_until: Option<std::time::Instant>,

    pub font_shmem: Option<Vec<u8>>,
    pub font_shmem_handle: Option<u32>,
    pub font_offsets: [(u32, u32); 6],

    pub time_shmem: Option<Vec<u8>>,
    pub time_shmem_handle: Option<u32>,

    pub audio_out_buffers: HashMap<u32, VecDeque<u64>>,
    pub audio_buffer_events: HashMap<u32, u32>,
    pub audio_out_volumes: HashMap<u32, u32>,
    pub audio_out_state: HashMap<u32, u8>,

    pub audio_renderers: HashMap<(u32, u32), AudioRendererState>,
    pub audio_renderer_events: HashMap<(u32, u32), u32>,
    pub audio_renderer_frame_counter: u64,
    pub audio_renderer_last_tick: std::time::Instant,
    pub audio_renderer_last_consumed: u64,
}

#[derive(Clone, Debug)]
pub struct AudioRendererState {
    pub sample_rate: u32,
    pub sample_count: u32,
    pub mix_buffer_count: u32,
    pub voice_count: u32,
    pub sink_count: u32,
    pub effect_count: u32,
    pub revision: u32,
    pub state: u32,
    pub rendering_time_limit: u32,
    pub voice_drop_param: f32,
    pub voice_played_samples: Vec<u64>,
    pub voice_wbufs_consumed: Vec<u32>,
    pub voice_last_wb_index: Vec<u16>,
    pub voice_is_new_seen: Vec<bool>,
    pub voice_wb_progress_frames: Vec<u64>,
    pub voice_frac_q15: Vec<i32>,
    pub voice_hist: Vec<[f32; 6]>,
}

impl Kernel {
    pub fn new(
        address_space: Arc<AddressSpace>,
        code_base: u64,
        code_size: u64,
        heap_base: u64,
        heap_size: u64,
        stack_base: u64,
        stack_size: u64,
        tls_base: u64,
        tls_pool_base: u64,
    ) -> Self {
        let mut handles = handles::HandleTable::new();
        let process_handle = handles.create_handle(handles::HandleType::Process);
        let main_thread_handle = handles.create_handle(handles::HandleType::Thread);

        let main_stack_top = stack_base + stack_size - 0x20;
        let threads =
            threads::Threads::new(main_thread_handle, tls_base, main_stack_top, tls_pool_base);

        Self {
            address_space,
            handles,
            threads,
            services: Services::new(),
            nvdrv: Nvdrv::new(),
            hid: Arc::new(Mutex::new(hid::HidShared::new())),
            sessions: HashMap::new(),
            event_signals: HashMap::new(),
            exited_thread_handles: HashSet::new(),
            pending_condvar_signals: HashMap::new(),
            audio_render_condvar: None,
            tls_buffer: [0u8; 0x100],
            pending_frames: Vec::new(),
            code_base,
            code_size,
            heap_base,
            heap_size,
            stack_base,
            stack_size,
            tls_base,
            tls_pool_base,
            aslr_base: code_base,
            aslr_size: 0x40_0000_0000,
            alias_base: code_base,
            alias_size: 0x4_0000_0000,
            is_application: false,
            title_id: 0,
            total_memory: 0x8000_0000,
            heap_committed: 0,
            cycle_count: 0,
            next_vsync_cycle: 16_666_667,
            last_sdl_capture: std::time::Instant::now(),
            display_ready: false,
            process_exited: false,
            vsync_poll_count: 0,
            process_handle,
            main_thread_handle,
            applet_messages: VecDeque::new(),
            applet_message_event: None,
            vsync_handles: HashSet::new(),
            nvdrv_sync_events: HashSet::new(),
            gpu_fence_events: HashSet::new(),
            last_vsync: std::time::Instant::now(),
            last_hid_tick: std::time::Instant::now(),
            last_generic_svc_imm: 0xFFFF,
            last_generic_svc_lr: 0,
            generic_svc_streak: 0,
            next_generic_svc_streak_log: 64,
            applet_focus_state: 1,
            applet_operation_mode: 0,
            applet_performance_mode: 0,
            display_resolution_change_event: None,
            library_applet_launchable_event: None,
            accumulated_suspended_tick_event: None,
            gpu_error_detected_event: None,
            friend_invitation_event: None,
            notification_event: None,
            acquired_sleep_lock_event: None,
            nro_mmap: None,
            nro_romfs_range: None,
            homebrew_dir: None,
            dir_cursor: HashMap::new(),
            open_files: HashMap::new(),
            sd_root: None,
            file_system_roots: HashMap::new(),
            open_host_files: HashMap::new(),
            open_romfs_files: HashMap::new(),
            open_file_handles: HashMap::new(),
            host_file_cache: HashMap::new(),
            open_dir_lists: HashMap::new(),
            yield_after_svc: false,
            present_pace_until: None,
            font_shmem: None,
            font_shmem_handle: None,
            font_offsets: [(0, 0); 6],
            time_shmem: None,
            time_shmem_handle: None,
            audio_out_buffers: HashMap::new(),
            audio_buffer_events: HashMap::new(),
            audio_out_volumes: HashMap::new(),
            audio_out_state: HashMap::new(),
            audio_renderers: HashMap::new(),
            audio_renderer_events: HashMap::new(),
            audio_renderer_frame_counter: 0,
            audio_renderer_last_tick: std::time::Instant::now(),
            audio_renderer_last_consumed: 0,
        }
    }

    fn debug_read_u32(&self, addr: u64) -> Option<u32> {
        let mut bytes = [0u8; 4];
        self.address_space
            .read(addr, &mut bytes)
            .ok()
            .map(|_| u32::from_le_bytes(bytes))
    }

    fn debug_wait_deadline(wake_at: Option<std::time::Instant>) -> String {
        match wake_at {
            Some(deadline) => {
                let now = std::time::Instant::now();
                if deadline > now {
                    format!(
                        "timeout_in_ms={:.1}",
                        (deadline - now).as_secs_f64() * 1000.0
                    )
                } else {
                    "timeout_due".to_string()
                }
            }
            None => "timeout=infinite".to_string(),
        }
    }

    fn debug_handle_tags(&self, handle: u32) -> String {
        let mut tags: Vec<&'static str> = Vec::new();
        if Some(handle) == self.applet_message_event {
            tags.push("applet_msg");
        }
        if self.vsync_handles.contains(&handle) {
            tags.push("vsync");
        }
        if self.nvdrv_sync_events.contains(&handle) {
            tags.push("nvdrv_sync");
        }
        if self.gpu_fence_events.contains(&handle) {
            tags.push("gpu_fence");
        }
        if self.audio_renderer_events.values().any(|&h| h == handle) {
            tags.push("audren");
        }
        if self.audio_buffer_events.values().any(|&h| h == handle) {
            tags.push("audout");
        }
        if self.event_signals.get(&handle).copied().unwrap_or(false) {
            tags.push("signaled");
        }

        let ty = self
            .handles
            .get_handle(handle)
            .map(|h| format!("{:?}", h.handle_type))
            .unwrap_or_else(|| "Unknown".to_string());

        if tags.is_empty() {
            format!("{:#x}:{}", handle, ty)
        } else {
            format!("{:#x}:{}:{}", handle, ty, tags.join("|"))
        }
    }

    pub fn log_thread_snapshot(&self, label: &str) {
        log::warn!(
            "[thread-snapshot:{}] current={:?} ready={:?} pending_condvars={:?} events={} audio_render_events={}",
            label,
            self.threads.current,
            self.threads.ready,
            self.pending_condvar_signals,
            self.event_signals.len(),
            self.audio_renderer_events.len()
        );

        let mut handles: Vec<u32> = self.threads.threads.keys().copied().collect();
        handles.sort_unstable();

        for handle in handles {
            let Some(t) = self.threads.threads.get(&handle) else {
                continue;
            };
            let state = match &t.state {
                threads::ThreadState::Created => "Created".to_string(),
                threads::ThreadState::Ready => "Ready".to_string(),
                threads::ThreadState::Running => "Running".to_string(),
                threads::ThreadState::Sleeping { wake_at } => {
                    format!("Sleeping {}", Self::debug_wait_deadline(Some(*wake_at)))
                }
                threads::ThreadState::WaitingHandle { handles, wake_at } => {
                    let waited = handles
                        .iter()
                        .map(|h| self.debug_handle_tags(*h))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!(
                        "WaitingHandle [{}] {}",
                        waited,
                        Self::debug_wait_deadline(*wake_at)
                    )
                }
                threads::ThreadState::WaitingMutex { mutex_addr } => {
                    let word = self.debug_read_u32(*mutex_addr).unwrap_or(0);
                    let holder = word & !MUTEX_HAS_LISTENERS;
                    format!(
                        "WaitingMutex addr={:#x} word={:#x} holder={:#x} listeners={}",
                        mutex_addr,
                        word,
                        holder,
                        (word & MUTEX_HAS_LISTENERS) != 0
                    )
                }
                threads::ThreadState::WaitingCondvar {
                    mutex_addr,
                    condvar_addr,
                    wake_at,
                    spurious_wake,
                } => {
                    let mutex_word = self.debug_read_u32(*mutex_addr).unwrap_or(0);
                    let cond_word = self.debug_read_u32(*condvar_addr).unwrap_or(0);
                    let pending = self
                        .pending_condvar_signals
                        .get(condvar_addr)
                        .copied()
                        .unwrap_or(0);
                    format!(
                        "WaitingCondvar mutex={:#x} mutex_word={:#x} cond={:#x} cond_word={:#x} pending={} spurious={} {}",
                        mutex_addr,
                        mutex_word,
                        condvar_addr,
                        cond_word,
                        pending,
                        spurious_wake,
                        Self::debug_wait_deadline(*wake_at)
                    )
                }
                threads::ThreadState::WaitingArbiter {
                    addr,
                    value,
                    wake_at,
                } => {
                    let word = self.debug_read_u32(*addr).unwrap_or(0);
                    format!(
                        "WaitingArbiter addr={:#x} expected={:#x} word={:#x} {}",
                        addr,
                        value,
                        word,
                        Self::debug_wait_deadline(*wake_at)
                    )
                }
                threads::ThreadState::Exited => "Exited".to_string(),
            };

            log::warn!(
                "[thread-snapshot:{}] h={:#x} tid={} core={} prio={} state={} pc={:#x} lr={:#x} sp={:#x} tls={:#x}",
                label,
                handle,
                t.tid,
                t.core,
                t.priority,
                state,
                t.ctx.pc,
                t.ctx.x[30],
                t.ctx.sp,
                t.tls_va
            );
        }
    }

    pub fn signal_vsync(&mut self) {
        let vsyncs: Vec<u32> = self.vsync_handles.iter().copied().collect();
        for h in vsyncs {
            self.event_signals.insert(h, true);
            self.threads.signal_handle(h);
            self.nvdrv
                .stats
                .vsync_signals
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn tick_audio_renderers(&mut self) {
        const FRAMES_PER_AUDIO_FRAME: u64 = 240;
        const MAX_BACKLOG_BLOCKS: u64 = 400;
        const TARGET_QUEUE_BLOCKS: u64 = 16;

        let to_signal: Vec<u32> = self
            .audio_renderers
            .iter()
            .filter(|(_, st)| st.state == 0)
            .filter_map(|(key, _)| self.audio_renderer_events.get(key).copied())
            .collect();
        let event_already_pending = to_signal
            .iter()
            .any(|ev| self.event_signals.get(ev).copied().unwrap_or(false));

        let blocks = if let Some(sink) = crate::audio_sink::host_audio_sink() {
            let mut n = sink.drain_pending_events();
            if !to_signal.is_empty() && !event_already_pending {
                let queued_blocks = (sink.queued_frames() as u64) / FRAMES_PER_AUDIO_FRAME;
                if queued_blocks < TARGET_QUEUE_BLOCKS {
                    n = n.max(TARGET_QUEUE_BLOCKS - queued_blocks);
                }
            }
            if n == 0 {
                if self.audio_renderer_last_consumed == 0 {
                    self.audio_renderer_last_consumed = sink.samples_consumed();
                }
                return;
            }
            self.audio_renderer_last_consumed = self
                .audio_renderer_last_consumed
                .wrapping_add(n * FRAMES_PER_AUDIO_FRAME);
            if n > MAX_BACKLOG_BLOCKS {
                n = MAX_BACKLOG_BLOCKS;
            }
            n
        } else {
            let now = std::time::Instant::now();
            if now.duration_since(self.audio_renderer_last_tick)
                < std::time::Duration::from_millis(5)
            {
                return;
            }
            self.audio_renderer_last_tick = now;
            1
        };
        if blocks == 0 {
            return;
        }

        if to_signal.is_empty() {
            return;
        }
        if event_already_pending {
            if let Some(sink) = crate::audio_sink::host_audio_sink() {
                sink.repost_pending_events(blocks);
            }
            return;
        }

        self.audio_renderer_frame_counter = self.audio_renderer_frame_counter.wrapping_add(1);
        for ev in &to_signal {
            self.event_signals.insert(*ev, true);
            self.threads.signal_handle(*ev);
        }
        if blocks > 1 {
            if let Some(sink) = crate::audio_sink::host_audio_sink() {
                sink.repost_pending_events(blocks - 1);
            }
        }
    }

    pub fn nro_romfs(&self) -> &[u8] {
        match (&self.nro_mmap, &self.nro_romfs_range) {
            (Some(mmap), Some(range)) => &mmap[range.clone()],
            _ => &[],
        }
    }

    pub fn wake_due_sleepers(&mut self, now: std::time::Instant) {
        let timed_out: Vec<(u32, u64, u64, bool)> = self
            .threads
            .threads
            .iter()
            .filter_map(|(h, t)| match &t.state {
                threads::ThreadState::WaitingCondvar {
                    mutex_addr,
                    condvar_addr,
                    wake_at: Some(d),
                    spurious_wake,
                    ..
                } if *d <= now => Some((*h, *mutex_addr, *condvar_addr, *spurious_wake)),
                _ => None,
            })
            .collect();

        for (h, mutex_addr, condvar_addr, spurious_wake) in timed_out {
            let had_pending = if let Some(n) = self.pending_condvar_signals.get_mut(&condvar_addr) {
                *n = n.saturating_sub(1);
                let remove = *n == 0;
                if remove {
                    self.pending_condvar_signals.remove(&condvar_addr);
                }
                true
            } else {
                false
            };
            if !had_pending && !spurious_wake {
                if let Some(t) = self.threads.threads.get_mut(&h) {
                    t.ctx.x[0] = nexium_common::result::KERNEL_TIMEOUT as u64;
                }
            }
            if self.reacquire_condvar_mutex(h, mutex_addr) {
                self.threads
                    .transition_state(h, threads::ThreadState::Ready);
            } else {
                self.threads
                    .transition_state(h, threads::ThreadState::WaitingMutex { mutex_addr });
            }
            if !self.threads.has_condvar_waiters(condvar_addr) {
                let _ = self.address_space.write(condvar_addr, &0u32.to_le_bytes());
            }
        }

        self.threads.wake_due_sleepers(now);
    }

    pub fn reacquire_condvar_mutex(&mut self, handle: u32, mutex_addr: u64) -> bool {
        let mut cur = [0u8; 4];
        let cur_word = if self.address_space.read(mutex_addr, &mut cur).is_ok() {
            u32::from_le_bytes(cur)
        } else {
            0
        };
        let holder = cur_word & !MUTEX_HAS_LISTENERS;
        if holder == 0 || holder == handle {
            let more = self.threads.has_mutex_waiters(mutex_addr);
            let new_word = if more {
                handle | MUTEX_HAS_LISTENERS
            } else {
                handle | (cur_word & MUTEX_HAS_LISTENERS)
            };
            let _ = self
                .address_space
                .write(mutex_addr, &new_word.to_le_bytes());
            true
        } else {
            let _ = self
                .address_space
                .write(mutex_addr, &(cur_word | MUTEX_HAS_LISTENERS).to_le_bytes());
            false
        }
    }

    pub fn ensure_thread_loaded(&mut self) -> Option<u32> {
        self.wake_due_sleepers(std::time::Instant::now());
        let cpu = cpu_local::cpu_mut()?;
        self.threads.ensure_thread_loaded(cpu)
    }

    pub fn yield_to_scheduler(&mut self) {
        if let Some(cpu) = cpu_local::cpu_ref() {
            self.threads.yield_current(cpu);
        }
    }

    pub fn init_cpu(&self, backend: nexium_cpu::CpuBackendKind) -> Result<Cpu, String> {
        let mut cpu = Cpu::new(backend)?;
        for region in self.address_space.host_regions() {
            unsafe {
                cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    .map_err(|e| format!("CPU map_host failed for {:#x}: {}", region.base, e))?;
            }
        }
        log::info!(
            "Kernel CPU initialized with {} mapped regions",
            self.address_space.host_regions().len()
        );
        Ok(cpu)
    }

    pub fn drain_frames(&mut self) -> Vec<FrameOut> {
        for qf in self.nvdrv.drain_frames() {
            self.pending_frames.push(FrameOut {
                width: qf.width,
                height: qf.height,
                pixels: qf.pixels,
            });
        }

        let qb_active = self
            .nvdrv
            .queue_buffer_active
            .load(std::sync::atomic::Ordering::Relaxed);
        if !qb_active
            && self.pending_frames.is_empty()
            && self.last_sdl_capture.elapsed() >= std::time::Duration::from_millis(16)
        {
            self.last_sdl_capture = std::time::Instant::now();
            let addr_space = self.address_space.clone();
            if let Some(qf) = self
                .nvdrv
                .try_capture_sdl_surface(|addr, buf| addr_space.read(addr, buf).is_ok())
            {
                self.pending_frames.push(FrameOut {
                    width: qf.width,
                    height: qf.height,
                    pixels: qf.pixels,
                });
            }
        }

        std::mem::take(&mut self.pending_frames)
    }

    fn synthesize_test_frame(&self) -> FrameOut {
        let w = 1280u32;
        let h = 720u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let phase = (self.cycle_count / 100_000) as f32 * 0.02;

        for y in 0..h {
            let ty = y as f32 / h as f32;
            let bg_r = (0x10 as f32 + ty * 8.0) as u8;
            let bg_g = (0x10 as f32 + ty * 6.0) as u8;
            let bg_b = (0x18 as f32 + ty * 12.0) as u8;
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = bg_r;
                pixels[i + 1] = bg_g;
                pixels[i + 2] = bg_b;
                pixels[i + 3] = 0xFF;
            }
        }

        let band_y = h / 2;
        let band_h = 4u32;
        for y in band_y..(band_y + band_h).min(h) {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let alpha = ((x as f32 / w as f32 + phase).sin() * 0.5 + 0.5) * 255.0;
                pixels[i] = 0xE0;
                pixels[i + 1] = (alpha * 0.16) as u8 + 0x2A;
                pixels[i + 2] = (alpha * 0.16) as u8 + 0x2A;
                pixels[i + 3] = 0xFF;
            }
        }

        let pulse = ((phase * 2.0).sin() * 0.5 + 0.5) * 80.0;
        let dot_r = 60u32 + pulse as u32;
        let cx = w / 2;
        let cy = h / 2 - 80;
        for y in cy.saturating_sub(dot_r)..(cy + dot_r).min(h) {
            for x in cx.saturating_sub(dot_r)..(cx + dot_r).min(w) {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let r2 = (dx * dx + dy * dy) as u32;
                if r2 < dot_r * dot_r {
                    let i = ((y * w + x) * 4) as usize;
                    let fade = 1.0 - (r2 as f32).sqrt() / dot_r as f32;
                    pixels[i] = (0xE0 as f32 * fade + 0x10 as f32 * (1.0 - fade)) as u8;
                    pixels[i + 1] = (0x2A as f32 * fade + 0x10 as f32 * (1.0 - fade)) as u8;
                    pixels[i + 2] = (0x2A as f32 * fade + 0x18 as f32 * (1.0 - fade)) as u8;
                }
            }
        }

        FrameOut {
            width: w,
            height: h,
            pixels,
        }
    }

    pub fn ensure_font_shmem_handle(&mut self) -> u32 {
        if let Some(h) = self.font_shmem_handle {
            return h;
        }
        const SHMEM_SIZE: usize = 0x1100000;
        const BFTTF_NAMES: [&str; 6] = [
            "nintendo_udsg-r_std_003.bfttf",
            "nintendo_udsg-r_org_zh-cn_003.bfttf",
            "nintendo_udsg-r_ext_zh-cn_003.bfttf",
            "nintendo_udjxh-db_zh-tw_003.bfttf",
            "nintendo_udsg-r_ko_003.bfttf",
            "nintendo_ext_003.bfttf",
        ];
        const BFTTF_KEY: [u8; 4] = [0x49, 0x62, 0x18, 0x06];

        let fonts_dir = directories::BaseDirs::new()
            .map(|b| b.config_dir().join("NeXium").join("system").join("fonts"));

        let mut buf = vec![0u8; SHMEM_SIZE];
        let mut offsets = [(0u32, 0u32); 6];
        let mut off = 0u32;
        let mut loaded = 0usize;

        if let Some(dir) = fonts_dir {
            for (i, name) in BFTTF_NAMES.iter().enumerate() {
                let path = dir.join(name);
                let raw = match std::fs::read(&path) {
                    Ok(r) => r,
                    Err(_) => {
                        log::warn!("pl:u font not found: {}", path.display());
                        continue;
                    }
                };
                if raw.len() < 8 {
                    log::warn!("pl:u font too small: {}", path.display());
                    continue;
                }
                let decoded_len = raw.len() - 8;
                let end = off as usize + decoded_len;
                if end > SHMEM_SIZE {
                    log::warn!("pl:u font shmem overflow at type {}", i);
                    break;
                }
                for (j, &b) in raw[8..].iter().enumerate() {
                    buf[off as usize + j] = b ^ BFTTF_KEY[j & 3];
                }
                offsets[i] = (off, decoded_len as u32);
                off = (off + decoded_len as u32 + 3) & !3;
                loaded += 1;
                log::info!(
                    "pl:u loaded font type {}: {} ({} bytes decoded)",
                    i,
                    name,
                    decoded_len
                );
            }
        }

        if loaded == 0 {
            log::warn!("pl:u no system fonts found in {{config}}/NeXium/system/fonts/ — using built-in fallback (NotoMono)");
            const FALLBACK: &[u8] = include_bytes!("../data/fallback_font.ttf");
            let mut off = 0u32;
            for i in 0..6usize {
                let end = off as usize + FALLBACK.len();
                if end <= SHMEM_SIZE {
                    buf[off as usize..end].copy_from_slice(FALLBACK);
                    offsets[i] = (off, FALLBACK.len() as u32);
                    off += FALLBACK.len() as u32;
                }
            }
        }

        self.font_shmem = Some(buf);
        self.font_offsets = offsets;
        let h = self
            .handles
            .create_handle(handles::HandleType::SharedMemory);
        self.font_shmem_handle = Some(h);
        h
    }

    pub fn ensure_time_shmem_handle(&mut self) -> u32 {
        if let Some(h) = self.time_shmem_handle {
            return h;
        }
        const TIME_SHMEM_SIZE: usize = 0x1000;
        self.time_shmem = Some(vec![0u8; TIME_SHMEM_SIZE]);
        let h = self
            .handles
            .create_handle(handles::HandleType::SharedMemory);
        self.time_shmem_handle = Some(h);
        log::info!(
            "time:u allocated KSharedMemory handle={:#x} size={:#x}",
            h,
            TIME_SHMEM_SIZE
        );
        h
    }

    pub fn dispatch_svc(&mut self, imm: u16) -> u32 {
        svc::dispatch(self, imm)
    }
}
