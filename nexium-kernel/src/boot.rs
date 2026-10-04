use crate::kernel::Kernel;
use nexium_cpu::Cpu;
use nexium_loader::{Application, ContainerKind, LoadedProgram, Nro};
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

fn map_extras_and_exit_stub(
    address_space: &AddressSpace,
    env_base: u64,
    exit_stub_va: u64,
    nce: bool,
) -> Result<(), String> {
    const EXTRAS_SIZE: u64 = 0x110000;
    if !nce {
        address_space
            .map(env_base, EXTRAS_SIZE, Perm::RW, "extras")
            .map_err(|e| format!("Failed to map extras: {:?}", e))?;
        let svc_exit_insn: u32 = 0xD400_00E1;
        return address_space
            .write(exit_stub_va, &svc_exit_insn.to_le_bytes())
            .map_err(|e| format!("Failed to write exit stub: {:?}", e));
    }
    let stub_page = exit_stub_va & !0xFFF;
    address_space
        .map(env_base, stub_page - env_base, Perm::RW, "extras_env")
        .map_err(|e| format!("Failed to map extras_env: {:?}", e))?;
    address_space
        .map(stub_page, 0x1000, Perm::RX, "exit_stub")
        .map_err(|e| format!("Failed to map exit_stub: {:?}", e))?;
    address_space
        .map(
            stub_page + 0x1000,
            env_base + EXTRAS_SIZE - (stub_page + 0x1000),
            Perm::RW,
            "extras_rest",
        )
        .map_err(|e| format!("Failed to map extras_rest: {:?}", e))?;
    let brk_exit_insn =
        nexium_cpu::nce_patch::a64::brk(nexium_cpu::nce_layout::EXIT_STUB_BRK_IMM as u32);
    address_space
        .write(exit_stub_va, &brk_exit_insn.to_le_bytes())
        .map_err(|e| format!("Failed to write exit stub: {:?}", e))
}

pub(crate) fn map_application_module(
    address_space: &AddressSpace,
    name: &str,
    nso: &nexium_loader::nso::Nso,
    base: u64,
) -> Result<(), String> {
    let image_end = u64::from(nso.image_size);
    let data_start = u64::from(nso.data.mem_offset);
    let text_end = u64::from(nso.text.mem_offset) + u64::from(nso.text.decompressed_size);
    if data_start & 0xfff != 0 || data_start > image_end || text_end > data_start {
        return Err(format!("Invalid NSO segment layout for {name}: data must start on a page after text"));
    }
    let ro_start = if nso.ro.decompressed_size == 0 {
        data_start
    } else {
        let start = u64::from(nso.ro.mem_offset);
        let end = start + u64::from(nso.ro.decompressed_size);
        if start & 0xfff != 0 || start < text_end || end > data_start {
            return Err(format!("Invalid NSO segment layout for {name}: read-only data must start on a page between text and data"));
        }
        start
    };
    if image_end as usize != nso.module_image.len() || image_end & 0xfff != 0
        || base.checked_add(image_end).is_none()
    {
        return Err(format!("Invalid NSO image range for {name}"));
    }
    for (start, end, perm, prefix) in [
        (0, ro_start, Perm::RX, "codestatic_text"),
        (ro_start, data_start, Perm::RO, "codestatic_rodata"),
        (data_start, image_end, Perm::RW, "codemutable"),
    ] {
        if start == end { continue; }
        address_space.map(base + start, end - start, perm, format!("{prefix}_{name}"))
            .map_err(|error| format!("Failed to map {name} {prefix}: {error}"))?;
        address_space.write(base + start, &nso.module_image[start as usize..end as usize])
            .map_err(|error| format!("Failed to write {name} {prefix}: {error}"))?;
    }
    Ok(())
}

fn apply_nce_patch(
    address_space: &AddressSpace,
    name: &str,
    text: &[u8],
    text_va: u64,
    patch_base: u64,
) -> Result<u64, String> {
    let words = nexium_cpu::nce_patch::words_from_bytes(text);
    let out = nexium_cpu::nce_patch::patch_module(
        &words,
        text_va,
        patch_base,
        nexium_cpu::nce_host_counter_hz(),
    )
    .map_err(|e| format!("NCE patch of {} failed: {}", name, e))?;
    let section_len = nexium_cpu::nce_patch::page_align(out.section.len()) as u64;
    address_space
        .write(text_va, &nexium_cpu::nce_patch::bytes_from_words(&out.text))
        .map_err(|e| format!("Failed to write patched {} text: {:?}", name, e))?;
    address_space
        .map(patch_base, section_len, Perm::RX, format!("codepatch_{}", name))
        .map_err(|e| format!("Failed to map {} patch section: {:?}", name, e))?;
    address_space
        .write(patch_base, &out.section)
        .map_err(|e| format!("Failed to write {} patch section: {:?}", name, e))?;
    nexium_cpu::nce_register_post_handlers(&out.post_handlers);
    log::info!(
        "  nce patch {}: text@{:#x} section@{:#x} size={:#x} svc={} mrs={} msr={} counter={} exclusive={}",
        name,
        text_va,
        patch_base,
        section_len,
        out.svc_count,
        out.mrs_count,
        out.msr_count,
        out.counter_count,
        out.exclusive_count
    );
    Ok(section_len)
}

impl BootContext {
    pub fn new(config: BootConfig) -> Result<Self, String> {
        let file = std::fs::File::open(&config.nro_path)
            .map_err(|error| format!("open {}: {error}", config.nro_path))?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", config.nro_path))?);
        match nexium_loader::detect(&config.nro_path, &mmap) {
            ContainerKind::Nro => Self::new_nro(config, Nro::parse_mmap(mmap)?),
            ContainerKind::Unknown => Err(format!("unrecognized file format: {}", config.nro_path)),
            _ => {
                drop(mmap);
                let app = Application::load_with_content(
                    &config.nro_path, &nexium_common::paths::content_dir(),
                )?;
                Self::new_application(config, app)
            }
        }
    }

    fn new_nro(config: BootConfig, nro: Nro) -> Result<Self, String> {
        log::info!("Loading NRO from: {}", config.nro_path);

        log::info!("Creating address space");
        let address_space = Arc::new(AddressSpace::new());

        let nce = matches!(config.cpu_backend, nexium_cpu::CpuBackendKind::Nce);
        let direct_base = nexium_memory::fastmem::direct_va_base().unwrap_or(0);
        if nce && direct_base == 0 {
            return Err("NCE backend requires the direct-mapped fastmem arena".to_string());
        }
        if direct_base != 0 {
            log::info!("guest address space is direct-mapped at {:#x}", direct_base);
        }
        let code_base: u64 = direct_base + 0x1_0000_0000;
        let heap_base: u64 = direct_base + 0x4_0000_0000;
        let stack_base: u64 = direct_base + 0x8_0000_0000;
        let env_base: u64 = direct_base + 0x10_0000_0000;
        let tls_base: u64 = env_base + 0x1000;
        let exit_stub_va: u64 = env_base + 0x2000;

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
        map_extras_and_exit_stub(&address_space, env_base, exit_stub_va, nce)?;

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

        if nce {
            let text_range = nro.text.mmap_range.clone();
            let text_va = code_base + text_range.start as u64;
            let text_len = text_range.len() & !3;
            let text = bytes[text_range.start..text_range.start + text_len].to_vec();
            let patch_base = code_base + code_size;
            apply_nce_patch(&address_space, "nro", &text, text_va, patch_base)?;
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
        kernel.address_space_end = direct_base + (1u64 << 39).min(nexium_memory::fastmem::arena_size());
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

    fn new_application(config: BootConfig, mut app: Application) -> Result<Self, String> {
        log::info!(
            "Loading application: {} (title_id={:#018x})",
            config.nro_path,
            app.title_id
        );

        let mods = nexium_loader::mods::discover_mods(&nexium_common::paths::mod_roots(), app.title_id)?;
        for entry in &mods {
            log::info!("[mods] {} enabled={} romfs={} exefs={} path={}", entry.name,
                entry.enabled, entry.has_romfs, entry.has_exefs, entry.path.display());
        }
        nexium_loader::mods::apply_exefs_mods(&mut app, &mods)?;
        let application_romfs = nexium_loader::mods::build_mod_romfs(&app, &mods)?;
        let address_space = Arc::new(AddressSpace::new());

        let a32 = !app.npdm.is_64bit;
        let bits: u32 = match app.npdm.address_space {
            nexium_loader::npdm::AddressSpaceType::Is39Bit => 39,
            nexium_loader::npdm::AddressSpaceType::Is36Bit => 36,
            nexium_loader::npdm::AddressSpaceType::Is32Bit
            | nexium_loader::npdm::AddressSpaceType::Is32BitNoMap => 32,
            #[allow(unreachable_patterns)]
            other => {
                log::warn!("address space {:?} unsupported; using 36-bit layout", other);
                36
            }
        };
        if a32 {
            log::info!(
                "application is AArch32 ({:?}); using the 32-bit process layout and the Dynarmic A32 backend",
                app.npdm.address_space
            );
        }
        let space: u64 = 1u64 << bits;

        let nce = matches!(config.cpu_backend, nexium_cpu::CpuBackendKind::Nce) && !a32;
        let direct_base = nexium_memory::fastmem::direct_va_base().unwrap_or(0);
        if nce && direct_base == 0 {
            return Err("NCE backend requires the direct-mapped fastmem arena".to_string());
        }
        if a32 && direct_base != 0 {
            return Err("AArch32 applications need the standard fastmem arena; restart without NCE".to_string());
        }
        if direct_base != 0 {
            log::info!("guest address space is direct-mapped at {:#x}", direct_base);
        }
        let arena_limit: u64 = nexium_memory::fastmem::arena_size();
        let window: u64 = direct_base + space.min(arena_limit);
        let (code_base, alias_base, alias_size, heap_base, heap_size, stack_base, env_base, aslr_base, aslr_size) =
            if bits == 32 {
                (
                    0x20_0000u64,
                    0u64,
                    0u64,
                    0x4000_0000u64,
                    0x8000_0000u64,
                    0x3E00_0000u64,
                    0x3F00_0000u64,
                    0x20_0000u64,
                    0xFFE0_0000u64,
                )
            } else {
                let alias_size: u64 = if bits == 39 {
                    0x10_0000_0000
                } else {
                    0x1_8000_0000
                };
                let heap_size: u64 = 0xCC00_0000;
                let stack_region_size: u64 = 0x8000_0000;
                let code_base: u64 = direct_base + 0x800_0000;
                let code_span: u64 = 0x1_0000_0000;
                let mut alias_size = alias_size;
                let mut cursor = code_base + code_span;
                let tail = heap_size + stack_region_size + 0x4000_0000;
                if cursor + alias_size + tail > window {
                    let room = window.saturating_sub(cursor + tail);
                    let shrunk = room.min(alias_size).max(0x8000_0000);
                    log::warn!(
                        "alias region {:#x} does not fit the {:#x} window; using {:#x}",
                        alias_size,
                        window,
                        shrunk
                    );
                    alias_size = shrunk;
                }
                let alias_base: u64 = cursor;
                cursor += alias_size;
                let heap_base: u64 = cursor;
                cursor += heap_size;
                let stack_base: u64 = cursor;
                cursor += stack_region_size;
                let env_base: u64 = cursor;
                (
                    code_base,
                    alias_base,
                    alias_size,
                    heap_base,
                    heap_size,
                    stack_base,
                    env_base,
                    code_base,
                    window - code_base,
                )
            };
        let tls_base: u64 = env_base + 0x1000;
        let exit_stub_va: u64 = env_base + 0x2000;

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
        let mut patch_cursor = code_base + code_size;
        for m in &app.modules {
            let base = code_base + m.load_offset;
            let image = m.nso.image_size as u64;
            let data_off = (m.nso.data.mem_offset as u64) & !(PAGE_SIZE - 1);
            let static_size = data_off;
            let mutable_size = image - data_off;

            map_application_module(&address_space, &m.name, &m.nso, base)?;
            log::info!(
                "  module {} @ {:#x} static={:#x} mutable={:#x}",
                m.name,
                base,
                static_size,
                mutable_size
            );
            if nce {
                let text_off = m.nso.text.mem_offset as usize;
                let text_len = (m.nso.text.decompressed_size as usize) & !3;
                let text = &m.nso.module_image[text_off..text_off + text_len];
                let section_len =
                    apply_nce_patch(&address_space, &m.name, text, base + text_off as u64, patch_cursor)?;
                patch_cursor += section_len;
            }
        }
        let code_size = if nce {
            patch_cursor - code_base
        } else {
            code_size
        };

        let compatibility_guest_probe_enabled =
            crate::kernel::svc::install_compatibility_guest_probes(&address_space, app.title_id);

        for (&pc, (kind, arg)) in crate::kernel::svc::guest_probe_actions() {
            if nce {
                log::warn!("[guest-probe] pc={:#x} kind={} ignored: SVC probes are not supported under NCE", pc, kind);
                continue;
            }
            if a32 {
                log::warn!("[guest-probe] pc={:#x} kind={} ignored: SVC probes are not supported for AArch32", pc, kind);
                continue;
            }
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
        map_extras_and_exit_stub(&address_space, env_base, exit_stub_va, nce)?;
        if a32 {
            address_space
                .write(exit_stub_va, &0xEF00_0007u32.to_le_bytes())
                .map_err(|e| format!("Failed to write A32 exit stub: {:?}", e))?;
        }

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
        kernel.compatibility_guest_probe_enabled = compatibility_guest_probe_enabled;

        let (romfs_mmap, romfs_range) = match &app.romfs {
            Some(r) if r.as_slice().len() as u64 == r.len() => {
                (Some(r.mmap.clone()), Some(r.range.clone()))
            }
            _ => (None, None),
        };
        kernel.nro_mmap = romfs_mmap;
        kernel.nro_romfs_range = romfs_range;
        kernel.application_romfs = application_romfs;
        if !app.display_version.is_empty() {
            kernel.application_display_version.fill(0);
            let bytes = app.display_version.as_bytes();
            let size = bytes.len().min(kernel.application_display_version.len());
            kernel.application_display_version[..size].copy_from_slice(&bytes[..size]);
        }
        kernel.system_romfs = app.system_romfs.clone();
        kernel.add_on_content = app.add_on_content.clone();
        kernel.patch_romfs = app.patch_romfs.clone();
        kernel.address_space_end = window;
        kernel.aslr_base = aslr_base;
        kernel.aslr_size = aslr_size;
        kernel.alias_base = alias_base;
        kernel.alias_size = alias_size;
        kernel.is_application = true;
        kernel.title_id = app.title_id;
        kernel.save_data_sizes.set_control(app.application_control.as_deref());
        kernel.guest_isa = if a32 {
            nexium_cpu::GuestIsa::AArch32
        } else {
            nexium_cpu::GuestIsa::AArch64
        };
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
