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
const TLS_USER_DISABLE_COUNT_OFFSET: u64 = 0x100;
const TLS_USER_INTERRUPT_FLAG_OFFSET: u64 = 0x102;

pub struct Kernel {
    pub address_space: Arc<AddressSpace>,
    pub handles: handles::HandleTable,
    pub threads: threads::Threads,
    pub services: Services,
    pub nvdrv: Nvdrv,
    pub hid: Arc<Mutex<hid::HidShared>>,
    hid_mapped_host_ptr: Option<usize>,
    pub sessions: HashMap<u32, session::Session>,
    pub event_signals: HashMap<u32, bool>,
    pub exited_thread_handles: HashMap<u32, ExitedThreadState>,
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
    pub system_resource_size: u64,

    pub cycle_count: u64,
    pub next_vsync_cycle: u64,
    pub last_sdl_capture: std::time::Instant,
    pub display_ready: bool,
    pub process_exited: bool,
    pub vsync_poll_count: u64,

    pub process_handle: u32,
    pub main_thread_handle: u32,
    pub process_ideal_core: i32,

    pub applet_messages: VecDeque<u32>,
    pub applet_message_event: Option<u32>,
    pub vsync_handles: HashSet<u32>,
    pub bufferqueue_swap_intervals: HashMap<u32, i32>,
    pub bufferqueue_events: HashMap<u32, u32>,
    bufferqueue_event_generation: u64,
    pub nvdrv_sync_events: HashSet<u32>,
    pub gpu_fence_events: HashMap<u32, (u32, u32)>,
    pub gpu_fence_armed: HashMap<u32, std::time::Instant>,
    pub gpu_event_tokens: HashMap<(u32, u32), u32>,
    pub last_hid_tick: std::time::Instant,
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
    pub aoc_change_event: Option<u32>,
    pub bcat_progress_event: Option<u32>,

    pub nro_mmap: Option<Arc<memmap2::Mmap>>,
    pub nro_romfs_range: Option<std::ops::Range<usize>>,
    pub application_romfs: Option<nexium_loader::LazyRomfs>,
    pub system_romfs_mmap: Option<Arc<memmap2::Mmap>>,
    pub system_romfs_ranges: HashMap<u64, std::ops::Range<usize>>,

    pub homebrew_dir: Option<std::path::PathBuf>,
    pub dir_cursor: HashMap<u32, usize>,
    pub open_files: HashMap<(u32, u32), Arc<memmap2::Mmap>>,
    pub sd_root: Option<std::path::PathBuf>,
    pub file_system_roots: HashMap<(u32, u32), std::path::PathBuf>,
    pub open_host_files: HashMap<(u32, u32), std::path::PathBuf>,
    pub open_romfs_files: HashMap<(u32, u32), (usize, usize)>,
    pub open_romfs_file_paths: HashMap<(u32, u32), String>,
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

    pub audio_out_sessions: HashMap<u32, crate::services::audio_out::handlers::AudioOutSession>,
    pub audio_buffer_events: HashMap<u32, u32>,
    pub audio_out_last_tick: std::time::Instant,

    pub audio_renderers: HashMap<(u32, u32), AudioRendererState>,
    pub audio_renderer_events: HashMap<(u32, u32), u32>,
    pub audio_renderer_frame_counter: u64,
    pub audio_renderer_last_tick: std::time::Instant,
    pub audio_renderer_last_consumed: u64,
    pub hwopus_decoders: HashMap<(u32, u32), crate::services::hwopus::DecoderState>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExitedThreadState {
    pub ideal_core: i32,
    pub affinity_mask: u64,
    pub priority: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadWaitClass {
    Running,
    Ready,
    Created,
    Sleeping,
    Waiting,
    Exited,
}

#[derive(Clone, Debug)]
pub struct ThreadWaitEntry {
    pub handle: u32,
    pub tid: u64,
    pub state_class: ThreadWaitClass,
    pub status: String,
    pub detail: String,
    pub core: i32,
    pub ideal_core: i32,
    pub affinity_mask: u64,
    pub priority: i32,
    pub effective_priority: i32,
    pub pc: u64,
    pub lr: u64,
    pub waiters: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioAdpcmContext {
    pub header: u8,
    pub yn0: i16,
    pub yn1: i16,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioAdpcmStreamKey {
    pub wb_index: u16,
    pub buffer_address: u64,
    pub buffer_size: u64,
    pub start_offset: i32,
    pub end_offset: i32,
    pub context_address: u64,
    pub coefficient_address: u64,
    pub sample_rate: u32,
    pub looping: bool,
    pub initial_header: u16,
    pub initial_yn0: i16,
    pub initial_yn1: i16,
    pub coefficients: [i16; 16],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioAdpcmDecodeState {
    pub valid: bool,
    pub key: AudioAdpcmStreamKey,
    pub next_sample: u64,
    pub context: AudioAdpcmContext,
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
    pub voice_wb_progress_frames: Vec<u64>,
    pub voice_frac_q15: Vec<i32>,
    pub voice_hist: Vec<[f32; 6]>,
    pub voice_adpcm_states: Vec<AudioAdpcmDecodeState>,
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
        let mut nvdrv = Nvdrv::new();
        let gpu_address_space = Arc::clone(&address_space);
        nvdrv.set_guest_memory_writer(move |addr, bytes| {
            gpu_address_space.write(addr, bytes).is_ok()
        });
        let gpu_read_space = Arc::clone(&address_space);
        let gpu_write_space = Arc::clone(&address_space);
        let gpu_copy_space = Arc::clone(&address_space);
        nvdrv.set_gpu_async_memory(
            Arc::new(move |addr, buf| gpu_read_space.read(addr, buf).is_ok()),
            Arc::new(move |addr, buf| gpu_write_space.write(addr, buf).is_ok()),
            Arc::new(move |src, dst, len| gpu_copy_space.copy(src, dst, len).is_ok()),
        );

        let docked = crate::hid_state::is_docked();
        let _ = crate::hid_state::take_console_mode_dirty();

        Self {
            address_space,
            handles,
            threads,
            services: Services::new(),
            nvdrv,
            hid: Arc::new(Mutex::new(hid::HidShared::new())),
            hid_mapped_host_ptr: None,
            sessions: HashMap::new(),
            event_signals: HashMap::new(),
            exited_thread_handles: HashMap::new(),
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
            system_resource_size: 0,
            cycle_count: 0,
            next_vsync_cycle: 16_666_667,
            last_sdl_capture: std::time::Instant::now(),
            display_ready: false,
            process_exited: false,
            vsync_poll_count: 0,
            process_handle,
            main_thread_handle,
            process_ideal_core: 0,
            applet_messages: VecDeque::new(),
            applet_message_event: None,
            vsync_handles: HashSet::new(),
            bufferqueue_swap_intervals: HashMap::new(),
            bufferqueue_events: HashMap::new(),
            bufferqueue_event_generation: 0,
            nvdrv_sync_events: HashSet::new(),
            gpu_fence_events: HashMap::new(),
            gpu_fence_armed: HashMap::new(),
            gpu_event_tokens: HashMap::new(),
            last_hid_tick: std::time::Instant::now(),
            last_generic_svc_imm: 0xFFFF,
            last_generic_svc_lr: 0,
            generic_svc_streak: 0,
            next_generic_svc_streak_log: 64,
            applet_focus_state: 1,
            applet_operation_mode: if docked { 1 } else { 0 },
            applet_performance_mode: if docked { 1 } else { 0 },
            display_resolution_change_event: None,
            library_applet_launchable_event: None,
            accumulated_suspended_tick_event: None,
            gpu_error_detected_event: None,
            friend_invitation_event: None,
            notification_event: None,
            acquired_sleep_lock_event: None,
            aoc_change_event: None,
            bcat_progress_event: None,
            nro_mmap: None,
            nro_romfs_range: None,
            application_romfs: None,
            system_romfs_mmap: None,
            system_romfs_ranges: HashMap::new(),
            homebrew_dir: None,
            dir_cursor: HashMap::new(),
            open_files: HashMap::new(),
            sd_root: None,
            file_system_roots: HashMap::new(),
            open_host_files: HashMap::new(),
            open_romfs_files: HashMap::new(),
            open_romfs_file_paths: HashMap::new(),
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
            audio_out_sessions: HashMap::new(),
            audio_buffer_events: HashMap::new(),
            audio_out_last_tick: std::time::Instant::now(),
            audio_renderers: HashMap::new(),
            audio_renderer_events: HashMap::new(),
            audio_renderer_frame_counter: 0,
            audio_renderer_last_tick: std::time::Instant::now(),
            audio_renderer_last_consumed: 0,
            hwopus_decoders: HashMap::new(),
        }
    }

    fn debug_read_u32(&self, addr: u64) -> Option<u32> {
        let mut bytes = [0u8; 4];
        self.address_space
            .read(addr, &mut bytes)
            .ok()
            .map(|_| u32::from_le_bytes(bytes))
    }

    fn debug_read_u64(&self, addr: u64) -> Option<u64> {
        let mut bytes = [0u8; 8];
        self.address_space
            .read(addr, &mut bytes)
            .ok()
            .map(|_| u64::from_le_bytes(bytes))
    }

    fn debug_is_code_ptr(addr: u64) -> bool {
        (0x0800_0000..0x0b80_0000).contains(&addr)
    }

    fn debug_stack_code_hits(&self, sp: u64, len: usize) -> Vec<String> {
        let mut hits = Vec::new();
        let mut seen = HashSet::new();
        for off in (0..len).step_by(8) {
            let Some(value) = self.debug_read_u64(sp.wrapping_add(off as u64)) else {
                continue;
            };
            if Self::debug_is_code_ptr(value) && seen.insert(value) {
                hits.push(format!("+{:#x}:{:#x}", off, value));
                if hits.len() >= 32 {
                    break;
                }
            }
        }
        hits
    }

    fn debug_ascii_at(&self, addr: u64) -> Option<String> {
        if addr < 0x1000 {
            return None;
        }
        let mut bytes = [0u8; 96];
        self.address_space.read(addr, &mut bytes).ok()?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        if end < 4 {
            return None;
        }
        let s = &bytes[..end];
        if !s.iter().all(|&b| b == b' ' || (0x21..=0x7e).contains(&b)) {
            return None;
        }
        if !s.iter().any(|&b| b.is_ascii_alphabetic()) {
            return None;
        }
        Some(String::from_utf8_lossy(s).into_owned())
    }

    fn debug_inline_ascii_runs(bytes: &[u8]) -> Vec<String> {
        let mut runs = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            while i < bytes.len() && !(bytes[i] == b' ' || (0x21..=0x7e).contains(&bytes[i])) {
                i += 1;
            }
            let start = i;
            while i < bytes.len() && (bytes[i] == b' ' || (0x21..=0x7e).contains(&bytes[i])) {
                i += 1;
            }
            if i - start >= 4 && bytes[start..i].iter().any(|&b| b.is_ascii_alphabetic()) {
                let s = String::from_utf8_lossy(&bytes[start..i.min(start + 64)]).into_owned();
                runs.push(format!("+{:#x}:\"{}\"", start, s));
                if runs.len() >= 8 {
                    break;
                }
            }
        }
        runs
    }

    fn debug_thread_arg_hits(&self, arg: u64) -> String {
        let mut bytes = [0u8; 0x180];
        if self.address_space.read(arg, &mut bytes).is_err() {
            return "unreadable".to_string();
        }

        let mut parts = Self::debug_inline_ascii_runs(&bytes);
        let mut seen = HashSet::new();
        for off in (0..bytes.len()).step_by(8) {
            let value = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            if Self::debug_is_code_ptr(value) && seen.insert(value) {
                parts.push(format!("+{:#x}:code={:#x}", off, value));
            } else if seen.insert(value) {
                if let Some(s) = self.debug_ascii_at(value) {
                    parts.push(format!("+{:#x}->{:#x}:\"{}\"", off, value, s));
                }
            }
            if parts.len() >= 24 {
                break;
            }
        }

        if parts.is_empty() {
            "hits=[]".to_string()
        } else {
            format!("hits=[{}]", parts.join(","))
        }
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
        if self.bufferqueue_events.contains_key(&handle) {
            tags.push("bufferqueue");
        }
        if self.nvdrv_sync_events.contains(&handle) {
            tags.push("nvdrv_sync");
        }
        if self.gpu_fence_events.contains_key(&handle) {
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

    pub fn thread_wait_tree(&self) -> Vec<ThreadWaitEntry> {
        let mut handles: Vec<u32> = self.threads.threads.keys().copied().collect();
        handles.sort_unstable();
        let mut entries = Vec::with_capacity(handles.len());
        for handle in handles {
            let Some(t) = self.threads.threads.get(&handle) else {
                continue;
            };
            let (state_class, status, detail) = match &t.state {
                threads::ThreadState::Created => {
                    (ThreadWaitClass::Created, "initialized", String::new())
                }
                threads::ThreadState::Ready => (ThreadWaitClass::Ready, "ready", String::new()),
                threads::ThreadState::Running => {
                    (ThreadWaitClass::Running, "running", String::new())
                }
                threads::ThreadState::Sleeping { wake_at } => (
                    ThreadWaitClass::Sleeping,
                    "sleeping",
                    Self::debug_wait_deadline(Some(*wake_at)),
                ),
                threads::ThreadState::WaitingHandle { handles, wake_at } => {
                    let waited = handles
                        .iter()
                        .map(|h| self.debug_handle_tags(*h))
                        .collect::<Vec<_>>()
                        .join(", ");
                    (
                        ThreadWaitClass::Waiting,
                        "waiting for objects",
                        format!("[{}] {}", waited, Self::debug_wait_deadline(*wake_at)),
                    )
                }
                threads::ThreadState::WaitingMutex {
                    mutex_addr,
                    owner_handle,
                    tag,
                } => (
                    ThreadWaitClass::Waiting,
                    "waiting for mutex",
                    format!(
                        "addr={:#x} owner={:#x} tag={:#x} word={:#x}",
                        mutex_addr,
                        owner_handle,
                        tag,
                        self.debug_read_u32(*mutex_addr).unwrap_or(0)
                    ),
                ),
                threads::ThreadState::WaitingCondvar {
                    mutex_addr,
                    condvar_addr,
                    wake_at,
                    ..
                } => (
                    ThreadWaitClass::Waiting,
                    "waiting for condition variable",
                    format!(
                        "cond={:#x} mutex={:#x} pending={} {}",
                        condvar_addr,
                        mutex_addr,
                        self.pending_condvar_signals
                            .get(condvar_addr)
                            .copied()
                            .unwrap_or(0),
                        Self::debug_wait_deadline(*wake_at)
                    ),
                ),
                threads::ThreadState::WaitingArbiter {
                    addr,
                    value,
                    wake_at,
                } => (
                    ThreadWaitClass::Waiting,
                    "waiting for address arbiter",
                    format!(
                        "addr={:#x} expected={:#x} word={:#x} {}",
                        addr,
                        value,
                        self.debug_read_u32(*addr).unwrap_or(0),
                        Self::debug_wait_deadline(*wake_at)
                    ),
                ),
                threads::ThreadState::Exited => {
                    (ThreadWaitClass::Exited, "terminated", String::new())
                }
            };
            let mut waiters: Vec<u32> = self
                .threads
                .threads
                .iter()
                .filter(|(_, other)| match &other.state {
                    threads::ThreadState::WaitingMutex { owner_handle, .. } => {
                        *owner_handle == handle
                    }
                    threads::ThreadState::WaitingHandle { handles, .. } => {
                        handles.contains(&handle)
                    }
                    _ => false,
                })
                .map(|(h, _)| *h)
                .collect();
            waiters.sort_unstable();
            entries.push(ThreadWaitEntry {
                handle,
                tid: t.tid,
                state_class,
                status: status.to_string(),
                detail,
                core: t.core,
                ideal_core: t.ideal_core,
                affinity_mask: t.affinity_mask,
                priority: t.priority,
                effective_priority: self.threads.effective_priority(handle),
                pc: t.ctx.pc,
                lr: t.ctx.x[30],
                waiters,
            });
        }
        entries
    }

    pub fn log_thread_snapshot(&self, label: &str) {
        let scan_stacks = std::env::var_os("NEXIUM_THREAD_STACK_SCAN").is_some();
        let scan_args = std::env::var_os("NEXIUM_THREAD_ARG_SCAN").is_some();

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
                threads::ThreadState::WaitingMutex {
                    mutex_addr,
                    owner_handle,
                    tag,
                } => {
                    let word = self.debug_read_u32(*mutex_addr).unwrap_or(0);
                    let holder = word & !MUTEX_HAS_LISTENERS;
                    format!(
                        "WaitingMutex addr={:#x} owner={:#x} tag={:#x} word={:#x} holder={:#x} listeners={}",
                        mutex_addr,
                        owner_handle,
                        tag,
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
                "[thread-snapshot:{}] h={:#x} tid={} core={} prio={} state={} pc={:#x} lr={:#x} sp={:#x} tls={:#x} arg={:#x}",
                label,
                handle,
                t.tid,
                t.core,
                t.priority,
                state,
                t.ctx.pc,
                t.ctx.x[30],
                t.ctx.sp,
                t.tls_va,
                t.entry_arg
            );

            if scan_stacks {
                let hits = self.debug_stack_code_hits(t.ctx.sp, 0x800);
                log::warn!(
                    "[thread-stack-scan:{}] h={:#x} sp={:#x} hits=[{}]",
                    label,
                    handle,
                    t.ctx.sp,
                    hits.join(",")
                );
            }
            if scan_args && t.entry_arg != 0 {
                log::warn!(
                    "[thread-arg-scan:{}] h={:#x} arg={:#x} {}",
                    label,
                    handle,
                    t.entry_arg,
                    self.debug_thread_arg_hits(t.entry_arg)
                );
            }
        }
    }

    pub fn signal_vsync(&mut self) {
        if crate::hid_state::take_console_mode_dirty() {
            let docked = crate::hid_state::is_docked();
            self.applet_operation_mode = if docked { 1 } else { 0 };
            self.applet_performance_mode = if docked { 1 } else { 0 };
            crate::services::am::queue_message(self, 30);
            crate::services::am::queue_message(self, 31);
            if let Some(handle) = self.display_resolution_change_event {
                self.event_signals.insert(handle, true);
                self.threads.signal_handle(handle);
            }
            log::info!(
                "console mode -> {}",
                if docked { "Docked" } else { "Handheld" }
            );
        }
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

    fn set_bufferqueue_event_level(&mut self, handle: u32, signaled: bool) {
        let was_signaled = self.event_signals.insert(handle, signaled).unwrap_or(false);
        if signaled && !was_signaled {
            self.threads.signal_handle(handle);
        }
    }

    pub fn register_bufferqueue_event(&mut self, handle: u32, binder_id: u32) {
        self.bufferqueue_events.insert(handle, binder_id);
        self.event_signals.insert(handle, false);
        self.refresh_bufferqueue_event(handle);
    }

    pub fn refresh_bufferqueue_event(&mut self, handle: u32) {
        let Some(&binder_id) = self.bufferqueue_events.get(&handle) else {
            return;
        };
        let signaled = self
            .nvdrv
            .with_bufferqueue(binder_id, |queue| queue.has_free_slot());
        self.set_bufferqueue_event_level(handle, signaled);
    }

    pub fn refresh_bufferqueue_events(&mut self) {
        let generation = self.nvdrv.bufferqueue_state_generation();
        if generation == self.bufferqueue_event_generation {
            return;
        }

        let events: Vec<(u32, u32)> = self
            .bufferqueue_events
            .iter()
            .map(|(&handle, &binder_id)| (handle, binder_id))
            .collect();
        for (handle, binder_id) in events {
            let signaled = self
                .nvdrv
                .with_bufferqueue(binder_id, |queue| queue.has_free_slot());
            self.set_bufferqueue_event_level(handle, signaled);
        }

        self.bufferqueue_event_generation = generation;
    }

    pub fn close_audio_renderer_object(&mut self, session: u32, object_id: u32) {
        self.audio_renderers.remove(&(session, object_id));
        if let Some(ev) = self.audio_renderer_events.remove(&(session, object_id)) {
            self.event_signals.remove(&ev);
        }
    }

    pub fn close_audio_renderer_session(&mut self, session: u32) {
        self.audio_renderers.retain(|&(s, _), _| s != session);
        let mut removed: Vec<u32> = Vec::new();
        self.audio_renderer_events.retain(|&(s, _), ev| {
            if s == session {
                removed.push(*ev);
                false
            } else {
                true
            }
        });
        for ev in removed {
            self.event_signals.remove(&ev);
        }
    }

    pub fn tick_audio_renderers(&mut self) {
        const FRAMES_PER_AUDIO_FRAME: u64 = 240;
        const MAX_BACKLOG_BLOCKS: u64 = 400;
        const TARGET_QUEUE_BLOCKS: u64 = 16;

        let now = std::time::Instant::now();
        if now.saturating_duration_since(self.audio_out_last_tick)
            >= std::time::Duration::from_millis(5)
        {
            self.audio_out_last_tick = now;
            crate::services::audio_out::handlers::poll_audio_outs(self);
        }

        let to_signal: Vec<u32> = self
            .audio_renderers
            .iter()
            .filter(|(_, st)| st.state == 0)
            .filter_map(|(key, _)| self.audio_renderer_events.get(key).copied())
            .collect();
        let event_already_pending = to_signal
            .iter()
            .any(|ev| self.event_signals.get(ev).copied().unwrap_or(false));
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CALLS: AtomicU64 = AtomicU64::new(0);
            let calls = CALLS.fetch_add(1, Ordering::Relaxed);
            if calls % 2048 == 0 {
                log::debug!(
                    "[audren-tick] calls={} renderers={} to_signal={} pending={} sink={}",
                    calls,
                    self.audio_renderers.len(),
                    to_signal.len(),
                    event_already_pending,
                    crate::audio_sink::host_audio_sink().is_some()
                );
            }
        }

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

    pub fn system_romfs(&self, title_id: u64) -> Option<&[u8]> {
        let mmap = self.system_romfs_mmap.as_ref()?;
        let range = self.system_romfs_ranges.get(&title_id)?;
        mmap.get(range.clone())
    }

    pub fn drain_gpu_fence_events(&mut self) {
        if self.gpu_fence_events.is_empty() {
            return;
        }
        let reached: Vec<u32> = self
            .gpu_fence_events
            .iter()
            .filter_map(|(&handle, &(syncpt_id, threshold))| {
                self.nvdrv
                    .is_syncpoint_reached(syncpt_id, threshold)
                    .then_some(handle)
            })
            .collect();
        for handle in reached {
            let pending = self.gpu_fence_events.remove(&handle);
            self.record_fence_signal(handle);
            self.event_signals.insert(handle, true);
            self.threads.signal_handle(handle);
            if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
                if let Some((syncpt_id, threshold)) = pending {
                    log::info!(
                        "[syncpt] scheduler drained gpu_fence_event handle={:#x} syncpt={} threshold={}",
                        handle,
                        syncpt_id,
                        threshold
                    );
                }
            }
        }
    }

    pub fn fence_profile_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_FENCE_PROFILE").is_some())
    }

    pub fn fence_signal_fix_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("NEXIUM_FENCE_SIGNAL_FIX").map_or(true, |value| value != "0")
        })
    }

    pub fn record_fence_armed(&mut self, handle: u32) {
        if Self::fence_profile_enabled() {
            self.gpu_fence_armed
                .insert(handle, std::time::Instant::now());
        }
    }

    pub fn record_fence_signal(&mut self, handle: u32) {
        use std::sync::atomic::{AtomicU64, Ordering};
        if Self::fence_profile_enabled() {
            static SIGNALS: AtomicU64 = AtomicU64::new(0);
            let signals = SIGNALS.fetch_add(1, Ordering::Relaxed) + 1;
            if signals % 64 == 0 {
                log::warn!("[fence-signal] total={}", signals);
            }
        }
        let Some(armed) = self.gpu_fence_armed.remove(&handle) else {
            return;
        };
        static NS: AtomicU64 = AtomicU64::new(0);
        static MAX_NS: AtomicU64 = AtomicU64::new(0);
        static N: AtomicU64 = AtomicU64::new(0);
        let ns = armed.elapsed().as_nanos() as u64;
        NS.fetch_add(ns, Ordering::Relaxed);
        MAX_NS.fetch_max(ns, Ordering::Relaxed);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 256 == 0 {
            let total = NS.swap(0, Ordering::Relaxed);
            let max = MAX_NS.swap(0, Ordering::Relaxed);
            log::warn!(
                "[fence-prof] signals={} window_avg_ms={:.2} window_max_ms={:.2} pending={}",
                n,
                total as f64 / 256.0 / 1_000_000.0,
                max as f64 / 1_000_000.0,
                self.gpu_fence_events.len()
            );
        }
    }

    pub fn wake_due_sleepers(&mut self, now: std::time::Instant) {
        self.refresh_bufferqueue_events();
        self.drain_gpu_fence_events();
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
            log::debug!(
                "cond_timeout: handle={:#x} mutex={:#x} cond={:#x} had_pending={} spurious={}",
                h,
                mutex_addr,
                condvar_addr,
                had_pending,
                spurious_wake
            );
            if !had_pending && !spurious_wake {
                if let Some(t) = self.threads.threads.get_mut(&h) {
                    t.ctx.x[0] = nexium_common::result::KERNEL_TIMEOUT as u64;
                }
                self.threads
                    .transition_state(h, threads::ThreadState::Ready);
            } else if self.reacquire_condvar_mutex(h, mutex_addr, h) {
                self.threads
                    .transition_state(h, threads::ThreadState::Ready);
            } else {
                let mut cur = [0u8; 4];
                let cur_word = if self.address_space.read(mutex_addr, &mut cur).is_ok() {
                    u32::from_le_bytes(cur)
                } else {
                    0
                };
                self.threads.transition_state(
                    h,
                    threads::ThreadState::WaitingMutex {
                        mutex_addr,
                        owner_handle: cur_word & !MUTEX_HAS_LISTENERS,
                        tag: h,
                    },
                );
            }
        }

        self.threads.wake_due_sleepers(now);
    }

    pub fn reacquire_condvar_mutex(&mut self, handle: u32, mutex_addr: u64, tag: u32) -> bool {
        loop {
            let cur_word = match self.address_space.atomic_load_u32(mutex_addr) {
                Ok(w) => w,
                Err(_) => return false,
            };
            let holder = cur_word & !MUTEX_HAS_LISTENERS;
            if holder == 0 || holder == handle {
                let more = self.threads.has_mutex_waiters(mutex_addr);
                let new_word = if more {
                    tag | MUTEX_HAS_LISTENERS
                } else {
                    tag | (cur_word & MUTEX_HAS_LISTENERS)
                };
                match self
                    .address_space
                    .atomic_cas_u32(mutex_addr, cur_word, new_word)
                {
                    Ok(true) => {
                        log::debug!(
                            "cond_reacquire: handle={:#x} mutex={:#x} word {:#x}->{:#x}",
                            handle,
                            mutex_addr,
                            cur_word,
                            new_word
                        );
                        return true;
                    }
                    Ok(false) => continue,
                    Err(_) => return false,
                }
            } else {
                if cur_word & MUTEX_HAS_LISTENERS == 0 {
                    match self.address_space.atomic_cas_u32(
                        mutex_addr,
                        cur_word,
                        cur_word | MUTEX_HAS_LISTENERS,
                    ) {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(_) => return false,
                    }
                }
                return false;
            }
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

    fn current_thread_tls_va(&self) -> Option<u64> {
        let handle = self.threads.current_handle()?;
        self.threads
            .threads
            .get(&handle)
            .map(|thread| thread.tls_va)
    }

    fn current_user_disable_count(&self) -> Option<u16> {
        let tls_va = self.current_thread_tls_va()?;
        let mut bytes = [0u8; 2];
        if let Err(error) = self
            .address_space
            .read(tls_va + TLS_USER_DISABLE_COUNT_OFFSET, &mut bytes)
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "failed to read user preemption-disable count at {:#x}: {:?}; allowing host preemption",
                    tls_va + TLS_USER_DISABLE_COUNT_OFFSET,
                    error
                );
            }
            return None;
        }
        Some(u16::from_le_bytes(bytes))
    }

    fn write_user_interrupt_flag(&self, tls_va: u64, value: u16) -> bool {
        if let Err(error) = self.address_space.write(
            tls_va + TLS_USER_INTERRUPT_FLAG_OFFSET,
            &value.to_le_bytes(),
        ) {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "failed to write user preemption interrupt flag at {:#x}: {:?}",
                    tls_va + TLS_USER_INTERRUPT_FLAG_OFFSET,
                    error
                );
            }
            return false;
        }
        true
    }

    fn defer_user_preemption_if_disabled(&mut self) -> bool {
        if self.threads.current_user_preemption_pending() {
            return true;
        }
        if self.current_user_disable_count().unwrap_or(0) == 0 {
            return false;
        }
        if !self.threads.mark_current_user_preemption_pending() {
            return false;
        }
        let flag_written = self
            .current_thread_tls_va()
            .is_some_and(|tls_va| self.write_user_interrupt_flag(tls_va, 1));
        if !flag_written {
            self.threads.take_current_user_preemption_pending();
        }
        true
    }

    pub fn try_yield_current_ready(&mut self, cpu: &Cpu) -> bool {
        let core = cpu_local::current_core() as i32;
        if !self.threads.has_ready_for_core(core) || self.defer_user_preemption_if_disabled() {
            return false;
        }
        self.threads
            .yield_with_state(cpu, threads::ThreadState::Ready)
            .is_some()
    }

    pub(crate) fn synchronize_user_preemption_state(&mut self) {
        let tls_va = self.current_thread_tls_va();
        let was_pending = self.threads.take_current_user_preemption_pending();
        if let Some(tls_va) = tls_va {
            self.write_user_interrupt_flag(tls_va, 0);
        }
        if was_pending
            && self
                .threads
                .has_ready_for_core(cpu_local::current_core() as i32)
        {
            self.yield_after_svc = true;
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

        let mut load_bfttf = |font_type: usize, raw: &[u8], source: &str| {
            if offsets[font_type].1 != 0 {
                return;
            }
            if raw.len() < 8 {
                log::warn!("pl:u font too small: {}", source);
                return;
            }
            let decoded_len = raw.len() - 8;
            let end = off as usize + decoded_len;
            if end > SHMEM_SIZE {
                log::warn!("pl:u font shmem overflow at type {}", font_type);
                return;
            }
            for (j, &byte) in raw[8..].iter().enumerate() {
                buf[off as usize + j] = byte ^ BFTTF_KEY[j & 3];
            }
            offsets[font_type] = (off, decoded_len as u32);
            off = (off + decoded_len as u32 + 3) & !3;
            loaded += 1;
            log::info!(
                "pl:u loaded font type {} from {} ({} bytes decoded)",
                font_type,
                source,
                decoded_len
            );
        };

        if let Some(dir) = fonts_dir {
            for (i, name) in BFTTF_NAMES.iter().enumerate() {
                let path = dir.join(name);
                let raw = match std::fs::read(&path) {
                    Ok(r) => r,
                    Err(_) => {
                        log::debug!("pl:u external font not found: {}", path.display());
                        continue;
                    }
                };
                load_bfttf(i, &raw, &path.display().to_string());
            }
        }

        for title_id in 0x0100_0000_0000_0810..=0x0100_0000_0000_0814 {
            let Some(romfs) = self.system_romfs(title_id) else {
                continue;
            };
            let Some(header) = nexium_loader::romfs::romfs_header(romfs) else {
                log::warn!("pl:u bundled font archive {:#018x} has no RomFS", title_id);
                continue;
            };
            for (name, start, size) in nexium_loader::romfs::romfs_dir_files(romfs, header, 0) {
                let Some(font_type) = BFTTF_NAMES
                    .iter()
                    .position(|expected| expected.eq_ignore_ascii_case(&name))
                else {
                    continue;
                };
                let Some(raw) = romfs.get(start..start.saturating_add(size)) else {
                    log::warn!(
                        "pl:u bundled font {} has invalid range {:#x}+{:#x}",
                        name,
                        start,
                        size
                    );
                    continue;
                };
                load_bfttf(
                    font_type,
                    raw,
                    &format!("bundled {:#018x}/{}", title_id, name),
                );
            }
        }
        drop(load_bfttf);

        if loaded < BFTTF_NAMES.len() {
            log::warn!("pl:u no system fonts found in {{config}}/NeXium/system/fonts/ — using built-in fallback (NotoMono)");
            const FALLBACK: &[u8] = include_bytes!("../data/fallback_font.ttf");
            for i in 0..6usize {
                if offsets[i].1 != 0 {
                    continue;
                }
                let end = off as usize + FALLBACK.len();
                if end <= SHMEM_SIZE {
                    buf[off as usize..end].copy_from_slice(FALLBACK);
                    offsets[i] = (off, FALLBACK.len() as u32);
                    off = (off + FALLBACK.len() as u32 + 3) & !3;
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
        self.time_shmem = Some(build_time_shmem());
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
        self.defer_user_preemption_if_disabled();
        svc::dispatch(self, imm)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Some(ptr) = self.hid_mapped_host_ptr.take() {
            let state = crate::hid_state::get_hid_state();
            state.lock().unbind_mapped_host(ptr);
        }
    }
}

#[cfg(test)]
mod user_preemption_tests {
    use super::*;

    const TEST_TLS: u64 = 0x10_0000;

    fn test_kernel(map_tls: bool) -> Kernel {
        test_kernel_at(map_tls, TEST_TLS)
    }

    fn test_kernel_at(map_tls: bool, tls_base: u64) -> Kernel {
        let address_space = Arc::new(AddressSpace::new());
        if map_tls {
            address_space
                .map(tls_base, 0x1000, nexium_memory::Perm::RW, "test_tls")
                .unwrap();
        }
        Kernel::new(
            address_space,
            0x80_0000,
            0x1000,
            0x90_0000,
            0x1000,
            0xa0_0000,
            0x1000,
            tls_base,
            tls_base + 0x1000,
        )
    }

    fn add_ready_core_zero_thread(kernel: &mut Kernel) {
        const READY_HANDLE: u32 = 0xfeed;
        let ready_tls = kernel.tls_base + 0x1000;
        kernel
            .threads
            .add_thread(READY_HANDLE, threads::ThreadCtx::zero(), ready_tls, 0, 0);
        kernel.threads.threads.get_mut(&READY_HANDLE).unwrap().core = 0;
        kernel
            .threads
            .transition_state(READY_HANDLE, threads::ThreadState::Ready);
    }

    fn read_tls_u16(kernel: &Kernel, offset: u64) -> u16 {
        let mut bytes = [0u8; 2];
        kernel
            .address_space
            .read(kernel.tls_base + offset, &mut bytes)
            .unwrap();
        u16::from_le_bytes(bytes)
    }

    #[test]
    fn disabled_boundary_marks_pending_and_sets_interrupt_flag() {
        let mut kernel = test_kernel(true);
        add_ready_core_zero_thread(&mut kernel);
        kernel
            .address_space
            .write(
                kernel.tls_base + TLS_USER_DISABLE_COUNT_OFFSET,
                &1u16.to_le_bytes(),
            )
            .unwrap();
        let current = kernel.threads.current_handle();
        let ready = kernel.threads.ready.clone();

        assert!(kernel.defer_user_preemption_if_disabled());
        assert!(kernel.threads.current_user_preemption_pending());
        assert_eq!(read_tls_u16(&kernel, TLS_USER_INTERRUPT_FLAG_OFFSET), 1);
        assert_eq!(kernel.threads.current_handle(), current);
        assert_eq!(kernel.threads.ready, ready);
    }

    #[test]
    fn pending_survives_zero_count_until_svc36_clears_and_reschedules() {
        let mut kernel = test_kernel_at(true, TEST_TLS + 0x2_0000);
        add_ready_core_zero_thread(&mut kernel);
        kernel
            .address_space
            .write(
                kernel.tls_base + TLS_USER_DISABLE_COUNT_OFFSET,
                &1u16.to_le_bytes(),
            )
            .unwrap();
        assert!(kernel.defer_user_preemption_if_disabled());

        kernel
            .address_space
            .write(
                kernel.tls_base + TLS_USER_DISABLE_COUNT_OFFSET,
                &0u16.to_le_bytes(),
            )
            .unwrap();
        assert!(kernel.defer_user_preemption_if_disabled());
        assert_eq!(kernel.dispatch_svc(0x36), nexium_common::result::SUCCESS);

        assert!(!kernel.threads.current_user_preemption_pending());
        assert_eq!(read_tls_u16(&kernel, TLS_USER_INTERRUPT_FLAG_OFFSET), 0);
        assert!(kernel.yield_after_svc);
    }

    #[test]
    fn zero_count_and_unmapped_tls_fail_open() {
        let mut mapped = test_kernel_at(true, TEST_TLS + 0x4_0000);
        assert!(!mapped.defer_user_preemption_if_disabled());
        assert!(!mapped.threads.current_user_preemption_pending());

        let mut unmapped = test_kernel(false);
        assert!(!unmapped.defer_user_preemption_if_disabled());
        assert!(!unmapped.threads.current_user_preemption_pending());
    }

    #[test]
    fn ready_transition_wakes_parked_core() {
        let kernel = Arc::new(Mutex::new(test_kernel(false)));
        const HANDLE: u32 = 0xb100;
        let wakers = {
            let mut k = kernel.lock();
            k.threads
                .add_thread(HANDLE, threads::ThreadCtx::zero(), 0, 0, 0);
            k.threads.threads.get_mut(&HANDLE).unwrap().core = 1;
            k.threads.wakers.clone()
        };
        let parker = {
            let kernel = Arc::clone(&kernel);
            let wakers = wakers.clone();
            std::thread::spawn(move || {
                let mut guard = kernel.lock();
                let start = std::time::Instant::now();
                while !guard.threads.ready.contains(&HANDLE) {
                    wakers.park_core(&mut guard, 1, std::time::Duration::from_secs(4));
                    if start.elapsed() > std::time::Duration::from_secs(12) {
                        break;
                    }
                }
                start.elapsed()
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(100));
        kernel
            .lock()
            .threads
            .transition_state(HANDLE, threads::ThreadState::Ready);
        let elapsed = parker.join().unwrap();
        assert!(elapsed < std::time::Duration::from_secs(3));
    }

    #[test]
    fn wake_nudge_yields_only_for_better_priority_same_core() {
        let mut kernel = test_kernel(false);
        const WOKEN: u32 = 0xb200;
        kernel
            .threads
            .add_thread(WOKEN, threads::ThreadCtx::zero(), 0, 0, 0);
        {
            let t = kernel.threads.threads.get_mut(&WOKEN).unwrap();
            t.core = 0;
            t.priority = 5;
        }
        kernel.yield_after_svc = false;
        svc::nudge_preempt_for_wake(&mut kernel, WOKEN);
        assert!(kernel.yield_after_svc);

        kernel.yield_after_svc = false;
        kernel.threads.threads.get_mut(&WOKEN).unwrap().priority = 60;
        svc::nudge_preempt_for_wake(&mut kernel, WOKEN);
        assert!(!kernel.yield_after_svc);

        kernel.yield_after_svc = false;
        {
            let t = kernel.threads.threads.get_mut(&WOKEN).unwrap();
            t.priority = 5;
            t.core = 2;
        }
        svc::nudge_preempt_for_wake(&mut kernel, WOKEN);
        assert!(!kernel.yield_after_svc);
    }

    #[test]
    fn thread_wait_tree_reports_states_and_waiters() {
        let mut kernel = test_kernel(false);
        const OWNER: u32 = 0xa100;
        const BLOCKED: u32 = 0xa200;
        kernel
            .threads
            .add_thread(OWNER, threads::ThreadCtx::zero(), 0, 0, 0);
        kernel
            .threads
            .transition_state(OWNER, threads::ThreadState::Ready);
        kernel
            .threads
            .add_thread(BLOCKED, threads::ThreadCtx::zero(), 0, 0, 0);
        kernel.threads.transition_state(
            BLOCKED,
            threads::ThreadState::WaitingMutex {
                mutex_addr: 0x1000,
                owner_handle: OWNER,
                tag: OWNER,
            },
        );

        let tree = kernel.thread_wait_tree();
        let owner = tree.iter().find(|e| e.handle == OWNER).unwrap();
        let blocked = tree.iter().find(|e| e.handle == BLOCKED).unwrap();

        assert_eq!(owner.state_class, ThreadWaitClass::Ready);
        assert_eq!(owner.waiters, vec![BLOCKED]);
        assert_eq!(blocked.state_class, ThreadWaitClass::Waiting);
        assert_eq!(blocked.status, "waiting for mutex");
        assert!(blocked.detail.contains("owner=0xa100"));
        assert!(blocked.waiters.is_empty());
    }

    #[test]
    fn bufferqueue_event_level_tracks_producer_availability() {
        let mut kernel = test_kernel(false);
        let handle = kernel.handles.create_handle(handles::HandleType::Event);
        kernel.register_bufferqueue_event(handle, 7);
        assert_eq!(kernel.event_signals.get(&handle), Some(&false));

        kernel.nvdrv.with_bufferqueue(7, |queue| {
            queue.set_preallocated(0, nexium_nvdrv::GraphicBuffer::default());
        });
        kernel.refresh_bufferqueue_events();
        assert_eq!(kernel.event_signals.get(&handle), Some(&true));

        assert_eq!(
            kernel
                .nvdrv
                .with_bufferqueue(7, |queue| queue.try_dequeue()),
            Some(0)
        );
        kernel.refresh_bufferqueue_events();
        assert_eq!(kernel.event_signals.get(&handle), Some(&false));

        assert!(kernel.nvdrv.with_bufferqueue(7, |queue| queue.cancel(0)));
        kernel.refresh_bufferqueue_events();
        assert_eq!(kernel.event_signals.get(&handle), Some(&true));
    }
}

pub(crate) const TIME_SHMEM_SIZE: usize = 0x1000;

fn build_time_shmem() -> Vec<u8> {
    let mut buf = vec![0u8; TIME_SHMEM_SIZE];
    let now_unix_s: i64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let source_id: [u8; 16] = *b"NeXiumSteady\0\0\0\0";

    let put_i64 = |buf: &mut [u8], off: usize, v: i64| {
        buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    };
    let put_u32 = |buf: &mut [u8], off: usize, v: u32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };

    put_u32(&mut buf, 0x00, 1);
    put_i64(&mut buf, 0x20, now_unix_s);
    buf[0x28..0x38].copy_from_slice(&source_id);

    put_u32(&mut buf, 0x38, 1);
    put_i64(&mut buf, 0x60, now_unix_s);
    put_i64(&mut buf, 0x68, 0);
    buf[0x70..0x80].copy_from_slice(&source_id);

    put_u32(&mut buf, 0x80, 1);
    put_i64(&mut buf, 0xA8, now_unix_s);
    put_i64(&mut buf, 0xB0, 0);
    buf[0xB8..0xC8].copy_from_slice(&source_id);

    put_u32(&mut buf, 0xC8, 1);
    buf[0xCD] = 0;

    put_u32(&mut buf, 0xD0, 1);
    put_i64(&mut buf, 0x110, 0);
    put_i64(&mut buf, 0x118, 1 << 14);
    put_i64(&mut buf, 0x120, 14);
    put_i64(&mut buf, 0x128, 0);
    put_i64(&mut buf, 0x130, i64::MAX);
    buf[0x138..0x148].copy_from_slice(&source_id);

    buf
}
