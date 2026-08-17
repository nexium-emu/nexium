use crate::kernel::Kernel;
use nexium_cpu::Cpu;
use nexium_loader::{Application, LoadedProgram, Loader, Nro};
use nexium_memory::{AddressSpace, Perm};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct BootConfig {
    pub nro_path: String,
    pub code_size: u64,
    pub heap_size: u64,
    pub stack_size: u64,
    pub loader_path: Option<String>,
    pub argv_override: Option<String>,
    pub cpu_backend: nexium_cpu::CpuBackendKind,
}

impl BootConfig {
    pub fn new(nro_path: &str) -> Self {
        Self {
            nro_path: nro_path.to_string(),
            code_size: 256 * 1024 * 1024,
            heap_size: 1536 * 1024 * 1024,
            stack_size: 16 * 1024 * 1024,
            loader_path: None,
            argv_override: None,
            cpu_backend: nexium_cpu::CpuBackendKind::default(),
        }
    }
}

pub struct BootContext {
    pub program: LoadedProgram,
    pub address_space: Arc<AddressSpace>,
    pub kernel: Arc<Mutex<Kernel>>,
    pub cpu: Option<Cpu>,
}

impl BootContext {
    pub fn new(config: BootConfig) -> Result<Self, String> {
        match Loader::load_any(&config.nro_path)? {
            LoadedProgram::Nro(nro) => Self::new_nro(config, nro),
            LoadedProgram::Application(app) => Self::new_application(config, app),
        }
    }

    fn new_nro(config: BootConfig, nro: Nro) -> Result<Self, String> {
        log::info!("Loading NRO from: {}", config.nro_path);

        log::info!("Creating address space");
        let address_space = Arc::new(AddressSpace::new());

        let code_base: u64 = 0x80_0000_0000;
        let heap_base: u64 = 0x90_0000_0000;
        let stack_base: u64 = 0xA0_0000_0000;
        let env_base: u64 = 0xB0_0000_0000;
        let tls_base: u64 = 0xB0_0000_1000;
        let exit_stub_va: u64 = 0xB0_0000_2000;

        log::info!("Mapping memory regions");

        const PAGE_SIZE: u64 = 0x1000;
        let image_size = nro.total_memory_size();
        let code_size = ((image_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)).max(config.code_size);
        log::info!(
            "NRO image size {:#x}, code region size {:#x}",
            image_size,
            code_size
        );
        let data_mmap_start = nro.data.mmap_range.start as u64;
        let split = data_mmap_start & !(PAGE_SIZE - 1);
        let split = split.min(code_size);
        if split == 0 || split >= code_size {
            log::info!(
                "  Mapping code @ {:#x} (size {:#x}) RX (no split — data section out of range)",
                code_base,
                code_size
            );
            address_space
                .map(code_base, code_size, Perm::RX, "code")
                .map_err(|e| format!("Failed to map code: {:?}", e))?;
        } else {
            log::info!(
                "  Mapping code text+ro @ {:#x} (size {:#x}) RX",
                code_base,
                split
            );
            address_space
                .map(code_base, split, Perm::RX, "code_rx")
                .map_err(|e| format!("Failed to map code_rx: {:?}", e))?;
            let rw_size = code_size - split;
            log::info!(
                "  Mapping code data+bss @ {:#x} (size {:#x}) RW",
                code_base + split,
                rw_size
            );
            address_space
                .map(code_base + split, rw_size, Perm::RW, "code_rw")
                .map_err(|e| format!("Failed to map code_rw: {:?}", e))?;
        }

        log::info!(
            "  Mapping heap @ {:#x} (size {:#x})",
            heap_base,
            config.heap_size
        );
        address_space
            .map(heap_base, config.heap_size, Perm::RW, "heap")
            .map_err(|e| format!("Failed to map heap: {:?}", e))?;

        log::info!(
            "  Mapping stack @ {:#x} (size {:#x})",
            stack_base,
            config.stack_size
        );
        address_space
            .map(stack_base, config.stack_size, Perm::RW, "stack")
            .map_err(|e| format!("Failed to map stack: {:?}", e))?;

        log::info!(
            "  Mapping extras (env+tls+exit_stub+tls_pool) @ {:#x} (size 0x110000)",
            env_base
        );
        address_space
            .map(env_base, 0x110000, Perm::RW, "extras")
            .map_err(|e| format!("Failed to map extras: {:?}", e))?;

        log::info!("  Writing exit stub SVC instruction @ {:#x}", exit_stub_va);
        let svc_exit_insn: u32 = 0xD400_00E1;
        address_space
            .write(exit_stub_va, &svc_exit_insn.to_le_bytes())
            .map_err(|e| format!("Failed to write exit stub: {:?}", e))?;

        log::info!(
            "  Writing NRO ({} bytes) at {:#x} from mmap (split at {:#x})",
            nro.bytes().len(),
            code_base,
            split
        );
        let bytes = nro.bytes();
        let split_idx = (split as usize).min(bytes.len());
        if split == 0 || split >= code_size {
            address_space
                .write(code_base, bytes)
                .map_err(|e| format!("Failed to write NRO file: {:?}", e))?;
        } else {
            if split_idx > 0 {
                address_space
                    .write(code_base, &bytes[..split_idx])
                    .map_err(|e| format!("Failed to write NRO text+ro: {:?}", e))?;
            }
            if bytes.len() > split_idx {
                address_space
                    .write(code_base + split, &bytes[split_idx..])
                    .map_err(|e| format!("Failed to write NRO data: {:?}", e))?;
            }
        }

        let tls_pool_base: u64 = env_base + 0x10000;
        let mut kernel = Kernel::new(
            address_space.clone(),
            code_base,
            code_size,
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
        nexium_common::paths::init();
        let nro_filename = std::path::Path::new(&config.nro_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("hbmenu.nro");
        let app_name = nexium_common::paths::app_name_from_nro(&config.nro_path);
        let _ = nexium_common::paths::sdmc_app_dir(&app_name);
        let argv_path = format!("sdmc:/switch/{}/{}", app_name, nro_filename);
        let argv_string = config
            .argv_override
            .clone()
            .unwrap_or_else(|| argv_path.clone());
        let next_load_path = config
            .loader_path
            .clone()
            .unwrap_or_else(|| argv_path.clone());
        let env_builder = nexium_loader::EnvBlockBuilder::new()
            .with_handles(kernel.main_thread_handle, kernel.process_handle)
            .with_heap(heap_base, config.heap_size)
            .with_argv(&argv_string)
            .with_next_load_path(&next_load_path);
        env_builder.build_into(&address_space, env_base)?;

        log::info!("Initializing CPU (backend: {})", config.cpu_backend.label());
        let mut cpu = kernel
            .init_cpu(config.cpu_backend)
            .map_err(|e| format!("Failed to init CPU: {}", e))?;

        {
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
            program: LoadedProgram::Nro(nro),
            address_space,
            kernel: Arc::new(Mutex::new(kernel)),
            cpu: Some(cpu),
        })
    }

    fn new_application(config: BootConfig, app: Application) -> Result<Self, String> {
        log::info!(
            "Loading application: {} (title_id={:#018x})",
            config.nro_path,
            app.title_id
        );

        let address_space = Arc::new(AddressSpace::new());

        let bits: u32 = match app.npdm.address_space {
            nexium_loader::npdm::AddressSpaceType::Is39Bit => 39,
            nexium_loader::npdm::AddressSpaceType::Is36Bit => 36,
            other => {
                log::warn!("address space {:?} unsupported; using 36-bit layout", other);
                36
            }
        };
        let space: u64 = 1u64 << bits;

        let code_base: u64 = 0x800_0000;
        let heap_base: u64 = space / 8;
        let alias_base: u64 = space / 2;
        let stack_base: u64 = space * 3 / 4;
        let env_base: u64 = stack_base + 0x4000_0000;
        let tls_base: u64 = env_base + 0x1000;
        let exit_stub_va: u64 = env_base + 0x2000;
        let aslr_base: u64 = code_base;
        let aslr_size: u64 = space - code_base;
        let alias_size: u64 = if bits == 39 {
            0x10_0000_0000
        } else {
            0x1_8000_0000
        };
        let heap_size: u64 = 0xCC00_0000;

        const PAGE_SIZE: u64 = 0x1000;
        let code_size = app.total_code_size.max(PAGE_SIZE);
        log::info!(
            "address space {}-bit: code@{:#x} heap@{:#x} stack@{:#x} alias@{:#x} aslr_size={:#x}",
            bits,
            code_base,
            heap_base,
            stack_base,
            alias_base,
            aslr_size
        );

        log::info!(
            "Mapping {} NSO module(s), code region {:#x} (size {:#x})",
            app.modules.len(),
            code_base,
            code_size
        );
        for m in &app.modules {
            let base = code_base + m.load_offset;
            let image = m.nso.image_size as u64;
            let data_off = (m.nso.data.mem_offset as u64) & !(PAGE_SIZE - 1);
            let static_size = data_off;
            let mutable_size = image - data_off;

            if static_size > 0 {
                address_space
                    .map(
                        base,
                        static_size,
                        Perm::RX,
                        format!("codestatic_{}", m.name),
                    )
                    .map_err(|e| format!("Failed to map {} text/ro: {:?}", m.name, e))?;
                address_space
                    .write(base, &m.nso.module_image[..static_size as usize])
                    .map_err(|e| format!("Failed to write {} text/ro: {:?}", m.name, e))?;
            }
            if mutable_size > 0 {
                address_space
                    .map(
                        base + data_off,
                        mutable_size,
                        Perm::RW,
                        format!("codemutable_{}", m.name),
                    )
                    .map_err(|e| format!("Failed to map {} data/bss: {:?}", m.name, e))?;
                address_space
                    .write(base + data_off, &m.nso.module_image[data_off as usize..])
                    .map_err(|e| format!("Failed to write {} data/bss: {:?}", m.name, e))?;
            }
            log::info!(
                "  module {} @ {:#x} static={:#x} mutable={:#x}",
                m.name,
                base,
                static_size,
                mutable_size
            );
        }

        for (&pc, (kind, arg)) in crate::kernel::svc::guest_probe_actions() {
            let insn: u32 = 0xD400_0FE1;
            match address_space.write(pc, &insn.to_le_bytes()) {
                Ok(()) => log::warn!(
                    "[guest-probe] patched svc 0x7f at pc={:#x} kind={} arg={:#x}",
                    pc,
                    kind,
                    arg
                ),
                Err(e) => log::warn!("[guest-probe] patch failed at pc={:#x}: {:?}", pc, e),
            }
        }

        log::info!("  Mapping heap @ {:#x} (size {:#x})", heap_base, heap_size);
        address_space
            .map_reserved(heap_base, heap_size, Perm::RW, "heap")
            .map_err(|e| format!("Failed to map heap: {:?}", e))?;

        log::info!(
            "  Mapping stack @ {:#x} (size {:#x})",
            stack_base,
            config.stack_size
        );
        address_space
            .map(stack_base, config.stack_size, Perm::RW, "stack")
            .map_err(|e| format!("Failed to map stack: {:?}", e))?;

        log::info!(
            "  Mapping extras (env+tls+exit_stub+tls_pool) @ {:#x} (size 0x110000)",
            env_base
        );
        address_space
            .map(env_base, 0x110000, Perm::RW, "extras")
            .map_err(|e| format!("Failed to map extras: {:?}", e))?;

        let svc_exit_insn: u32 = 0xD400_00E1;
        address_space
            .write(exit_stub_va, &svc_exit_insn.to_le_bytes())
            .map_err(|e| format!("Failed to write exit stub: {:?}", e))?;

        let tls_pool_base: u64 = env_base + 0x10000;
        let mut kernel = Kernel::new(
            address_space.clone(),
            code_base,
            code_size,
            heap_base,
            heap_size,
            stack_base,
            config.stack_size,
            tls_base,
            tls_pool_base,
        );

        let (romfs_mmap, romfs_range) = match &app.romfs {
            Some(r) => (Some(r.mmap.clone()), Some(r.range.clone())),
            None => (None, None),
        };
        kernel.nro_mmap = romfs_mmap;
        kernel.nro_romfs_range = romfs_range;
        kernel.application_romfs = app.romfs.clone();
        kernel.system_romfs_mmap = if app.system_romfs.is_empty() {
            None
        } else {
            Some(app.mmap.clone())
        };
        kernel.system_romfs_ranges = app
            .system_romfs
            .iter()
            .map(|(&title_id, romfs)| (title_id, romfs.range.clone()))
            .collect();
        kernel.aslr_base = aslr_base;
        kernel.aslr_size = aslr_size;
        kernel.alias_base = alias_base;
        kernel.alias_size = alias_size;
        kernel.is_application = true;
        kernel.title_id = app.title_id;
        kernel.total_memory = 0xCD50_0000;
        kernel.system_resource_size = app.npdm.system_resource_size as u64;
        log::info!(
            "npdm system_resource_size={:#x} (VAMM {})",
            kernel.system_resource_size,
            if kernel.system_resource_size != 0 {
                "enabled"
            } else {
                "disabled"
            }
        );
        kernel.process_ideal_core = (app.npdm.main_thread_core as i32)
            .clamp(0, crate::kernel::threads::NUM_CORES as i32 - 1);
        if let Some(main) = kernel.threads.threads.get_mut(&kernel.main_thread_handle) {
            main.ideal_core = kernel.process_ideal_core;
            main.affinity_mask = 1u64 << kernel.process_ideal_core;
            main.priority = app.npdm.main_thread_priority as i32;
        }
        log::info!(
            "npdm main_thread_core={} main_thread_priority={} (process ideal core)",
            kernel.process_ideal_core,
            app.npdm.main_thread_priority
        );

        nexium_common::paths::init();

        log::info!("Initializing CPU (backend: {})", config.cpu_backend.label());
        let mut cpu = kernel
            .init_cpu(config.cpu_backend)
            .map_err(|e| format!("Failed to init CPU: {}", e))?;

        let main_thread_handle = kernel.main_thread_handle;
        {
            let entry_point = code_base;
            let sp = stack_base + config.stack_size - 0x20;
            cpu.set_pc(entry_point);
            cpu.set_sp(sp);
            cpu.set_tpidrro_el0(tls_base);
            cpu.set_register(0, 0);
            cpu.set_register(1, main_thread_handle as u64);
            cpu.set_register(30, exit_stub_va);
            log::info!(
                "  PC: {:#x}  SP: {:#x}  X0: 0  X1 (main_thread): {:#x}  X30 (exit_stub): {:#x}",
                cpu.get_pc(),
                cpu.get_sp(),
                main_thread_handle,
                exit_stub_va
            );
            log::info!("  TPIDRRO_EL0: {:#x}", cpu.get_tpidrro_el0());
        }

        log::info!("Application boot context ready");
        Ok(BootContext {
            program: LoadedProgram::Application(app),
            address_space,
            kernel: Arc::new(Mutex::new(kernel)),
            cpu: Some(cpu),
        })
    }

    pub fn chained_load_path(&self) -> Option<String> {
        const ENV_BASE: u64 = 0xB0_0000_0000;
        const NEXTLOAD_PATH_VA: u64 = ENV_BASE + 0xA00;
        let mut buf = vec![0u8; 0x301];
        self.address_space.read(NEXTLOAD_PATH_VA, &mut buf).ok()?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        if end == 0 {
            return None;
        }
        let raw = std::str::from_utf8(&buf[..end]).ok()?.trim();
        if raw.is_empty() {
            return None;
        }
        let trimmed = raw
            .strip_prefix("sdmc:/")
            .or_else(|| raw.strip_prefix("sdmc:"))
            .unwrap_or(raw);
        let filename = std::path::Path::new(trimmed)
            .file_name()?
            .to_str()?
            .to_string();
        let kguard = self.kernel.lock();
        let homebrew = kguard.homebrew_dir.as_ref()?;
        let host_path = homebrew.join(&filename);
        if !host_path.exists() {
            log::warn!(
                "chained_load_path: requested {:?} not found in {:?}",
                raw,
                homebrew
            );
            return None;
        }
        Some(host_path.to_string_lossy().into_owned())
    }

    pub fn chained_load_argv(&self) -> Option<String> {
        const ENV_BASE: u64 = 0xB0_0000_0000;
        const NEXTLOAD_ARGV_VA: u64 = ENV_BASE + 0xC00;
        let mut buf = vec![0u8; 0x400];
        self.address_space.read(NEXTLOAD_ARGV_VA, &mut buf).ok()?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        if end == 0 {
            return None;
        }
        let raw = std::str::from_utf8(&buf[..end]).ok()?.trim();
        if raw.is_empty() {
            return None;
        }
        Some(raw.to_string())
    }

    pub fn run(&mut self) -> Result<u32, String> {
        log::info!("Starting CPU execution");

        let max_cycles = 1_000_000_000u64;
        let mut cycle_count = 0u64;
        let mut svc_count = 0u32;

        let mut cpu = self
            .cpu
            .take()
            .ok_or_else(|| "CPU not initialized".to_string())?;
        let _cpu_guard = crate::kernel::cpu_local::set_current_cpu(&mut cpu, 0);
        use crate::kernel::cpu_local::{cpu_mut, cpu_ref};

        loop {
            let run = cpu_mut().unwrap().run_with_count(100_000);
            let event = run.event;

            cycle_count += run.retired;

            match event {
                nexium_cpu::CpuEvent::Running => {
                    if cycle_count % 10_000_000 == 0 {
                        log::trace!(
                            "CPU running... {} cycles executed (no SVCs yet)",
                            cycle_count
                        );
                    }
                }
                nexium_cpu::CpuEvent::Svc(imm) => {
                    svc_count += 1;
                    log::trace!("SVC {:#04x} (count: {})", imm, svc_count);

                    let result = self.kernel.lock().dispatch_svc(imm);

                    if imm != 0x7f {
                        cpu_mut().unwrap().set_register(0, result as u64);
                    }

                    if result == 0 || result == 1 {
                        continue;
                    }
                }
                nexium_cpu::CpuEvent::Stalled => {
                    log::info!("CPU stalled at {:#x}", cpu_ref().unwrap().get_pc());
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
        }

        log::info!(
            "Execution complete: {} cycles, {} SVCs",
            cycle_count,
            svc_count
        );
        Ok(0)
    }
}

fn resolve_homebrew_dir(loaded_nro_path: &str) -> Option<std::path::PathBuf> {
    let appdata_nro =
        directories::BaseDirs::new().map(|d| d.config_dir().join("NeXium").join("NRO"));

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
                        let is_nro = p
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|e| e.eq_ignore_ascii_case("nro"))
                            .unwrap_or(false);
                        if !is_nro {
                            continue;
                        }
                        let Some(name) = p.file_name() else { continue };
                        let dst_path = dst.join(name);
                        if !dst_path.exists() {
                            match std::fs::copy(&p, &dst_path) {
                                Ok(n) => {
                                    log::info!("Migrated {:?} → {:?} ({} bytes)", p, dst_path, n)
                                }
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
