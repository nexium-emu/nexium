use crate::memory::{AddressSpace, Perm};
use crate::loader::{Loader, Nro};
use crate::kernel::Kernel;
use crate::cpu::CpuEvent;
use std::sync::Arc;

pub struct BootConfig {
    pub nro_path: String,
    pub code_size: u64,
    pub heap_size: u64,
    pub stack_size: u64,
}

impl BootConfig {
    pub fn new(nro_path: &str) -> Self {
        Self {
            nro_path: nro_path.to_string(),
            code_size: 256 * 1024 * 1024,
            heap_size: 256 * 1024 * 1024,
            stack_size: 1 * 1024 * 1024,
        }
    }
}

pub struct BootContext {
    pub nro: Nro,
    pub address_space: Arc<AddressSpace>,
    pub kernel: Kernel,
}

impl BootContext {
    pub fn new(config: BootConfig) -> Result<Self, String> {
        log::info!("Loading NRO from: {}", config.nro_path);
        let nro = Loader::load_nro(&config.nro_path)?;

        log::info!("Creating address space");
        let address_space = Arc::new(AddressSpace::new());

        let code_base: u64 = 0x8000_0000_0000;
        let heap_base: u64 = 0x9000_0000_0000;
        let stack_base: u64 = 0xA000_0000_0000;
        let tls_base: u64 = 0xB000_0000_0000;

        log::info!("Mapping memory regions");
        address_space.map(code_base, config.code_size, Perm::RX, "code")
            .map_err(|e| format!("Failed to map code: {:?}", e))?;
        address_space.map(heap_base, config.heap_size, Perm::RW, "heap")
            .map_err(|e| format!("Failed to map heap: {:?}", e))?;
        address_space.map(stack_base, config.stack_size, Perm::RW, "stack")
            .map_err(|e| format!("Failed to map stack: {:?}", e))?;
        address_space.map(tls_base, 0x1000, Perm::RW, "tls")
            .map_err(|e| format!("Failed to map tls: {:?}", e))?;

        log::info!("Loading NRO segments into memory");
        let text_va = code_base;
        let ro_va = text_va + nro.text.size as u64;
        let data_va = ro_va + nro.ro.size as u64;

        log::info!("  text @ {:#x} ({} bytes)", text_va, nro.text.size);
        log::info!("  ro   @ {:#x} ({} bytes)", ro_va, nro.ro.size);
        log::info!("  data @ {:#x} ({} bytes)", data_va, nro.data.size);

        address_space.write(text_va, &nro.text.data)
            .map_err(|e| format!("Failed to write text segment: {:?}", e))?;
        address_space.write(ro_va, &nro.ro.data)
            .map_err(|e| format!("Failed to write ro segment: {:?}", e))?;
        address_space.write(data_va, &nro.data.data)
            .map_err(|e| format!("Failed to write data segment: {:?}", e))?;

        let mut kernel = Kernel::new(
            address_space.clone(),
            code_base,
            config.code_size,
            heap_base,
            config.heap_size,
            stack_base,
            config.stack_size,
        );

        log::info!("Initializing CPU");
        kernel.init_cpu()
            .map_err(|e| format!("Failed to init CPU: {}", e))?;

        if let Some(cpu) = &mut kernel.cpu {
            log::info!("Setting up CPU registers");
            let entry_point = text_va;
            let sp = stack_base + config.stack_size;
            let heap_end = heap_base;

            cpu.set_pc(entry_point);
            cpu.set_sp(sp);
            cpu.set_tpidrro_el0(tls_base);
            cpu.set_register(0, 1);
            cpu.set_register(1, 0);
            cpu.set_register(2, heap_end);

            log::info!("  PC: {:#x}", cpu.get_pc());
            log::info!("  SP: {:#x}", cpu.get_sp());
            log::info!("  TPIDRRO_EL0: {:#x}", cpu.get_tpidrro_el0());
            log::info!("  X0 (main_thread_handle): {}", cpu.get_register(0));
            log::info!("  X1 (entrypoint_arg): {}", cpu.get_register(1));
            log::info!("  X2 (heap_base): {:#x}", cpu.get_register(2));
        }

        log::info!("Boot context ready");
        Ok(BootContext {
            nro,
            address_space,
            kernel,
        })
    }

    pub fn run(&mut self) -> Result<u32, String> {
        log::info!("Starting CPU execution");

        let max_cycles = 1_000_000_000u64;
        let mut cycle_count = 0u64;
        let mut svc_count = 0u32;

        loop {
            if let Some(cpu) = &mut self.kernel.cpu {
                let event = cpu.run(100_000);

                cycle_count += 100_000;

                match event {
                    crate::cpu::CpuEvent::Running => {
                        if cycle_count % 10_000_000 == 0 {
                            log::info!("CPU running... {} cycles executed (no SVCs yet)", cycle_count);
                        }
                    }
                    crate::cpu::CpuEvent::Svc(imm) => {
                        svc_count += 1;
                        log::info!("SVC {:#04x} (count: {})", imm, svc_count);

                        let result = self.kernel.dispatch_svc(imm);

                        if let Some(cpu) = &mut self.kernel.cpu {
                            cpu.set_register(0, result as u64);
                        }

                        if result == 0 || result == 1 {
                            continue;
                        }
                    }
                    crate::cpu::CpuEvent::Stalled => {
                        log::info!("CPU stalled at {:#x}", cpu.get_pc());
                        break;
                    }
                    crate::cpu::CpuEvent::Interrupted => {
                        log::info!("CPU interrupted");
                        break;
                    }
                    crate::cpu::CpuEvent::Exception(code) => {
                        log::error!("CPU exception {:#x}", code);
                        break;
                    }
                }

                if cycle_count > max_cycles {
                    log::warn!("Max cycles exceeded, stopping execution");
                    break;
                }
            } else {
                return Err("CPU not initialized".to_string());
            }
        }

        log::info!("Execution complete: {} cycles, {} SVCs", cycle_count, svc_count);
        Ok(0)
    }
}
