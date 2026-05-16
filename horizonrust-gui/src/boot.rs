use horizonrust_core::boot::{BootConfig, BootContext};
use horizonrust_core::services::FrameOut;
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl From<FrameOut> for Frame {
    fn from(f: FrameOut) -> Self {
        Self { width: f.width, height: f.height, pixels: f.pixels }
    }
}

pub struct EmulationHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub frame_rx: Receiver<Frame>,
    pub thread_handle: Option<thread::JoinHandle<Result<(), String>>>,
}

impl EmulationHandle {
    pub fn new(nro_path: &str) -> Result<Self, String> {
        let nro_path = nro_path.to_string();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = Arc::clone(&stop_flag);
        let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(2);

        let thread_handle = thread::spawn(move || {
            log::info!("Booting NRO: {}", nro_path);

            if !Path::new(&nro_path).exists() {
                return Err(format!("NRO file not found: {}", nro_path));
            }

            let config = BootConfig::new(&nro_path);
            let mut boot_ctx = BootContext::new(config)?;

            log::info!("Starting emulation loop");
            let max_cycles = 1_000_000_000u64;
            let mut cycle_count = 0u64;
            let mut svc_count = 0u32;

            loop {
                if stop_flag_clone.load(Ordering::Relaxed) {
                    log::info!("Stopping emulation");
                    break;
                }

                if let Some(cpu) = &mut boot_ctx.kernel.cpu {
                    let event = cpu.run(100_000);
                    cycle_count += 100_000;

                    match event {
                        horizonrust_core::cpu::CpuEvent::Running => {
                            if cycle_count % 10_000_000 == 0 {
                                log::debug!("CPU running... {} cycles", cycle_count);
                            }
                        }
                        horizonrust_core::cpu::CpuEvent::Svc(imm) => {
                            svc_count += 1;
                            log::trace!("SVC {:#04x} (count: {})", imm, svc_count);

                            let result = boot_ctx.kernel.dispatch_svc(imm);

                            if let Some(cpu) = &mut boot_ctx.kernel.cpu {
                                cpu.set_register(0, result as u64);
                            }

                            for f in boot_ctx.kernel.drain_frames() {
                                let _ = frame_tx.try_send(f.into());
                            }
                        }
                        horizonrust_core::cpu::CpuEvent::Stalled => {
                            log::info!("CPU stalled at {:#x}", cpu.get_pc());
                            break;
                        }
                        horizonrust_core::cpu::CpuEvent::Interrupted => {
                            log::info!("CPU interrupted");
                            break;
                        }
                        horizonrust_core::cpu::CpuEvent::Exception(code) => {
                            log::error!("CPU exception {:#x}", code);
                            break;
                        }
                    }

                    if cycle_count > max_cycles {
                        log::warn!("Max cycles exceeded");
                        break;
                    }
                } else {
                    return Err("CPU not initialized".to_string());
                }
            }

            log::info!("Emulation complete: {} cycles, {} SVCs", cycle_count, svc_count);
            Ok(())
        });

        Ok(Self {
            stop_flag,
            frame_rx,
            thread_handle: Some(thread_handle),
        })
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }
}

impl Drop for EmulationHandle {
    fn drop(&mut self) {
        self.stop();
    }
}
