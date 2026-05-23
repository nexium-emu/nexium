use nexium_memory::{AddressSpace, Perm};
use nexium_loader::{Loader, Nro};
use crate::kernel::Kernel;
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

        let code_base: u64 = 0x80_0000_0000;
        let heap_base: u64 = 0x90_0000_0000;
        let stack_base: u64 = 0xA0_0000_0000;
        let env_base: u64 = 0xB0_0000_0000;
        let tls_base: u64 = 0xB0_0000_1000;
        let exit_stub_va: u64 = 0xB0_0000_2000;

        log::info!("Mapping memory regions");

        log::info!("  Mapping code @ {:#x} (size {:#x})", code_base, config.code_size);
        address_space.map(code_base, config.code_size, Perm::RX, "code")
            .map_err(|e| format!("Failed to map code: {:?}", e))?;

        log::info!("  Mapping heap @ {:#x} (size {:#x})", heap_base, config.heap_size);
        address_space.map(heap_base, config.heap_size, Perm::RW, "heap")
            .map_err(|e| format!("Failed to map heap: {:?}", e))?;

        log::info!("  Mapping stack @ {:#x} (size {:#x})", stack_base, config.stack_size);
        address_space.map(stack_base, config.stack_size, Perm::RW, "stack")
            .map_err(|e| format!("Failed to map stack: {:?}", e))?;

        log::info!("  Mapping extras (env+tls+exit_stub+tls_pool) @ {:#x} (size 0x110000)", env_base);
        address_space.map(env_base, 0x110000, Perm::RW, "extras")
            .map_err(|e| format!("Failed to map extras: {:?}", e))?;

        log::info!("  Writing exit stub SVC instruction @ {:#x}", exit_stub_va);
        let svc_exit_insn: u32 = 0xD400_00E1;
        address_space.write(exit_stub_va, &svc_exit_insn.to_le_bytes())
            .map_err(|e| format!("Failed to write exit stub: {:?}", e))?;

        log::info!("  Writing NRO ({} bytes) at {:#x} from mmap", nro.bytes().len(), code_base);
        address_space.write(code_base, nro.bytes())
            .map_err(|e| format!("Failed to write NRO file: {:?}", e))?;

        let tls_pool_base: u64 = env_base + 0x10000;
        let mut kernel = Kernel::new(
            address_space.clone(),
            code_base,
            config.code_size,
            heap_base,
            config.heap_size,
            stack_base,
            config.stack_size,
            tls_base,
            tls_pool_base,
        );
        kernel.nro_mmap = Some(nro.mmap_arc());
        kernel.nro_romfs_range = nro.romfs_range();
        kernel.homebrew_dir = resolve_homebrew_dir(&config.nro_path);
        log::info!("homebrew_dir = {:?}", kernel.homebrew_dir);

        log::info!("  Initializing environment block @ {:#x}", env_base);
        let nro_filename = std::path::Path::new(&config.nro_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("hbmenu.nro");
        let argv_path = format!("sdmc:/{}", nro_filename);
        let env_builder = nexium_loader::EnvBlockBuilder::new()
            .with_handles(kernel.main_thread_handle, kernel.process_handle)
            .with_heap(heap_base, config.heap_size)
            .with_argv(&argv_path)
            .with_next_load_path(&argv_path);
        env_builder.build_into(&address_space, env_base)?;

        log::info!("Initializing CPU");
        kernel.init_cpu()
            .map_err(|e| format!("Failed to init CPU: {}", e))?;

        if let Some(cpu) = &mut kernel.cpu {
            log::info!("Setting up CPU registers");
            let entry_point = code_base;
            let sp = stack_base + config.stack_size - 0x20;

            cpu.set_pc(entry_point);
            cpu.set_sp(sp);
            cpu.set_tpidrro_el0(tls_base);
            cpu.set_register(0, env_base);
            cpu.set_register(1, u64::MAX);
            cpu.set_register(30, exit_stub_va);

            log::info!("  PC: {:#x}", cpu.get_pc());
            log::info!("  SP: {:#x}", cpu.get_sp());
            log::info!("  X0 (env_block, NRO mode): {:#x}", cpu.get_register(0));
            log::info!("  X1 (sentinel): {:#x}", cpu.get_register(1));
            log::info!("  X30 (exit_stub): {:#x}", cpu.get_register(30));
            log::info!("  TPIDRRO_EL0: {:#x}", cpu.get_tpidrro_el0());
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
                    nexium_cpu::CpuEvent::Running => {
                        if cycle_count % 10_000_000 == 0 {
                            log::info!("CPU running... {} cycles executed (no SVCs yet)", cycle_count);
                        }
                    }
                    nexium_cpu::CpuEvent::Svc(imm) => {
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
                    nexium_cpu::CpuEvent::Stalled => {
                        log::info!("CPU stalled at {:#x}", cpu.get_pc());
                        break;
                    }
                    nexium_cpu::CpuEvent::Interrupted => {
                        log::info!("CPU interrupted");
                        break;
                    }
                    nexium_cpu::CpuEvent::Exception(code) => {
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

fn resolve_homebrew_dir(loaded_nro_path: &str) -> Option<std::path::PathBuf> {
    let appdata_nro = directories::BaseDirs::new()
        .map(|d| d.config_dir().join("NeXium").join("NRO"));

    if let Some(dir) = &appdata_nro {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("Failed to create homebrew dir {:?}: {}", dir, e);
        }
    }

    if let Some(loaded_dir) = std::path::Path::new(loaded_nro_path).parent() {
        if let Some(dst) = &appdata_nro {
            if loaded_dir != dst {
                if let Ok(rd) = std::fs::read_dir(loaded_dir) {
                    for entry in rd.flatten() {
                        let p = entry.path();
                        let is_nro = p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("nro")).unwrap_or(false);
                        if !is_nro { continue; }
                        let Some(name) = p.file_name() else { continue };
                        let dst_path = dst.join(name);
                        if !dst_path.exists() {
                            match std::fs::copy(&p, &dst_path) {
                                Ok(n) => log::info!("Migrated {:?} → {:?} ({} bytes)", p, dst_path, n),
                                Err(e) => log::warn!("Failed to migrate {:?}: {}", p, e),
                            }
                        }
                    }
                }
            }
        }
    }

    appdata_nro
}
