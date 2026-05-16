pub mod svc;
pub mod svc_defs;
pub mod threads;
pub mod handles;
pub mod hid;
pub mod session;
pub mod cpu_context;

use crate::memory::AddressSpace;
use crate::nvdrv::Nvdrv;
use crate::services::Services;
use crate::services::FrameOut;
use crate::cpu::Cpu;
use std::sync::Arc;
use std::collections::HashMap;
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

    pub cycle_count: u64,
    pub next_vsync_cycle: u64,
    pub display_ready: bool,
    pub process_exited: bool,
    pub vsync_poll_count: u64,

    pub process_handle: u32,
    pub main_thread_handle: u32,
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
    ) -> Self {
        let mut handles = handles::HandleTable::new();
        let process_handle = handles.create_handle(handles::HandleType::Process);
        let main_thread_handle = handles.create_handle(handles::HandleType::Thread);

        Self {
            address_space,
            handles,
            threads: threads::Threads::new(),
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
            cycle_count: 0,
            next_vsync_cycle: 16_666_667,
            display_ready: false,
            process_exited: false,
            vsync_poll_count: 0,
            process_handle,
            main_thread_handle,
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

        if self.pending_frames.is_empty() && self.cycle_count >= self.next_vsync_cycle {
            self.next_vsync_cycle = self.cycle_count + 1_000_000;
            let frame = self.synthesize_test_frame();
            log::debug!("vsync test frame: cycle={} {}x{}", self.cycle_count, frame.width, frame.height);
            self.pending_frames.push(frame);
        }

        std::mem::take(&mut self.pending_frames)
    }

    fn synthesize_test_frame(&self) -> FrameOut {
        let w = 1280u32;
        let h = 720u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let phase = (self.cycle_count / 100_000) as f32 * 0.05;
        let cx_f = w as f32 / 2.0;
        let cy_f = h as f32 / 2.0;
        let max_r = (cx_f.powi(2) + cy_f.powi(2)).sqrt();

        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let dx = x as f32 - cx_f;
                let dy = y as f32 - cy_f;
                let r = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);
                let hue = (angle / std::f32::consts::PI * 180.0 + 180.0 + phase * 30.0) % 360.0;
                let val = 1.0 - (r / max_r).min(1.0) * 0.4;

                let c = val;
                let h_sect = hue / 60.0;
                let frac = h_sect - h_sect.floor();
                let q = c * (1.0 - frac);
                let t = c * frac;

                let (r_c, g_c, b_c) = match h_sect as u32 % 6 {
                    0 => (c, t, 0.0),
                    1 => (q, c, 0.0),
                    2 => (0.0, c, t),
                    3 => (0.0, q, c),
                    4 => (t, 0.0, c),
                    _ => (c, 0.0, q),
                };

                pixels[i] = (r_c * 255.0) as u8;
                pixels[i + 1] = (g_c * 255.0) as u8;
                pixels[i + 2] = (b_c * 255.0) as u8;
                pixels[i + 3] = 0xFF;
            }
        }
        FrameOut { width: w, height: h, pixels }
    }

    pub fn dispatch_svc(&mut self, imm: u16) -> u32 {
        svc::dispatch(self, imm)
    }
}
