pub mod svc;
pub mod svc_defs;
pub mod threads;
pub mod handles;
pub mod hid;
pub mod session;
pub mod cpu_context;

use nexium_memory::AddressSpace;
use nexium_nvdrv::Nvdrv;
use crate::services::Services;
use crate::services::FrameOut;
use nexium_cpu::Cpu;
use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use parking_lot::Mutex;

pub struct Kernel {
    pub address_space: Arc<AddressSpace>,
    pub handles: handles::HandleTable,
    pub threads: threads::Threads,
    pub services: Services,
    pub nvdrv: Nvdrv,
    pub hid: Arc<Mutex<hid::HidShared>>,
    pub sessions: HashMap<u32, session::Session>,
    pub event_signals: HashMap<u32, bool>,
    pub tls_buffer: [u8; 0x100],
    pub cpu: Option<Cpu>,
    pub pending_frames: Vec<FrameOut>,

    pub code_base: u64,
    pub code_size: u64,
    pub heap_base: u64,
    pub heap_size: u64,
    pub stack_base: u64,
    pub stack_size: u64,
    pub tls_base: u64,
    pub tls_pool_base: u64,

    pub cycle_count: u64,
    pub next_vsync_cycle: u64,
    pub display_ready: bool,
    pub process_exited: bool,
    pub vsync_poll_count: u64,

    pub process_handle: u32,
    pub main_thread_handle: u32,

    pub applet_messages: VecDeque<u32>,
    pub applet_message_event: Option<u32>,
    pub vsync_handles: HashSet<u32>,
    pub last_vsync: std::time::Instant,
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

    pub nro_romfs: Vec<u8>,
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
        let threads = threads::Threads::new(main_thread_handle, tls_base, main_stack_top, tls_pool_base);

        Self {
            address_space,
            handles,
            threads,
            services: Services::new(),
            nvdrv: Nvdrv::new(),
            hid: Arc::new(Mutex::new(hid::HidShared::new())),
            sessions: HashMap::new(),
            event_signals: HashMap::new(),
            tls_buffer: [0u8; 0x100],
            cpu: None,
            pending_frames: Vec::new(),
            code_base,
            code_size,
            heap_base,
            heap_size,
            stack_base,
            stack_size,
            tls_base,
            tls_pool_base,
            cycle_count: 0,
            next_vsync_cycle: 16_666_667,
            display_ready: false,
            process_exited: false,
            vsync_poll_count: 0,
            process_handle,
            main_thread_handle,
            applet_messages: VecDeque::from(vec![crate::services::am::msg::FOCUS_STATE_CHANGED]),
            applet_message_event: None,
            vsync_handles: HashSet::new(),
            last_vsync: std::time::Instant::now(),
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
            nro_romfs: Vec::new(),
        }
    }

    pub fn ensure_thread_loaded(&mut self) -> Option<u32> {
        let cpu = self.cpu.as_mut()?;
        self.threads.ensure_thread_loaded(cpu)
    }

    pub fn yield_to_scheduler(&mut self) {
        if let Some(cpu) = self.cpu.as_ref() {
            self.threads.yield_current(cpu);
        }
    }

    pub fn init_cpu(&mut self) -> Result<(), String> {
        let mut cpu = Cpu::new_dynarmic()?;
        for region in self.address_space.host_regions() {
            unsafe {
                cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    .map_err(|e| format!("CPU map_host failed for {:#x}: {}", region.base, e))?;
            }
        }
        log::info!("Kernel CPU initialized with {} mapped regions", self.address_space.host_regions().len());
        self.cpu = Some(cpu);
        Ok(())
    }

    pub fn drain_frames(&mut self) -> Vec<FrameOut> {
        for qf in self.nvdrv.drain_frames() {
            self.pending_frames.push(FrameOut {
                width: qf.width,
                height: qf.height,
                pixels: qf.pixels,
            });
        }

        let qb_active = self.nvdrv.queue_buffer_active.load(std::sync::atomic::Ordering::Relaxed);
        if !qb_active && self.pending_frames.is_empty() && self.cycle_count >= self.next_vsync_cycle {
            self.next_vsync_cycle = self.cycle_count + 1_000_000;
            let addr_space = self.address_space.clone();
            if let Some(qf) = self.nvdrv.try_capture_sdl_surface(|addr, buf| addr_space.read(addr, buf).is_ok()) {
                self.pending_frames.push(FrameOut {
                    width: qf.width, height: qf.height, pixels: qf.pixels,
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

        FrameOut { width: w, height: h, pixels }
    }

    pub fn dispatch_svc(&mut self, imm: u16) -> u32 {
        svc::dispatch(self, imm)
    }
}
