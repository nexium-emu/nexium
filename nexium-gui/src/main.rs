#![allow(dead_code)]

#[cfg(all(windows, target_arch = "x86_64"))]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod app;
mod app_settings;
mod audio;
mod boot;
mod carousel;
mod controller_art;
mod controller_config;
mod content_manager;
mod debugger;
mod depth_emit;
mod frame_scaler;
mod hd_rumble;
mod homebrew;
mod input;
mod motion_input;
mod library;
mod native_game;
mod mods;
mod performance;
mod playtime;
mod profile;
mod rumble_output;
mod shop;
mod splash;
mod steamgrid;
mod ui_audio;
mod updater;
mod vkeyboard;

use app::HorizonApp;
use app_settings::AppSettings;
use nexium_common::FileLogger;

fn host_vsync_enabled(preference: bool, environment_override: Option<&str>) -> bool {
    match environment_override {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => preference,
    }
}

fn host_surface_config(vsync: bool) -> eframe::SurfaceConfig {
    eframe::SurfaceConfig {
        present_mode: if vsync {
            eframe::wgpu::PresentMode::Fifo
        } else {
            eframe::wgpu::PresentMode::AutoNoVsync
        },
        desired_maximum_frame_latency: Some(2),
    }
}

fn host_wgpu_setup(preferred: Option<&str>) -> eframe::egui_wgpu::WgpuSetupCreateNew {
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends =
        eframe::wgpu::Backends::from_env().unwrap_or(eframe::wgpu::Backends::VULKAN);
    if let Some(selected) = preferred
        .and_then(|id| {
            nexium_gpu::adapter::available_devices()
                .iter()
                .find(|device| device.id == id)
        })
        .cloned()
    {
        setup.native_adapter_selector = Some(std::sync::Arc::new(move |adapters, surface| {
            let compatible = |adapter: &&eframe::wgpu::Adapter| {
                surface.is_none_or(|surface| adapter.is_surface_supported(surface))
            };
            let selected_adapter = adapters.iter().filter(compatible).find(|adapter| {
                let info = adapter.get_info();
                info.vendor == selected.vendor_id && info.device == selected.device_id
            });
            let adapter = selected_adapter
                .or_else(|| {
                    log::warn!("Selected GPU unavailable for the window; using an available GPU");
                    adapters
                        .iter()
                        .filter(compatible)
                        .find(|adapter| {
                            adapter.get_info().device_type == eframe::wgpu::DeviceType::DiscreteGpu
                        })
                        .or_else(|| adapters.iter().find(compatible))
                })
                .ok_or_else(|| "No GPU can present to the application window".to_string())?;
            log::info!("Window GPU: {}", adapter.get_info().name);
            Ok(adapter.clone())
        }));
    }
    setup
}

fn main() -> Result<(), eframe::Error> {
    std::env::set_var("DISABLE_MANGOHUD", "1");

    if std::env::var("NEXIUM_DIAG").is_ok_and(|v| v != "lite") {
        let defaults: [(&str, &str); 14] = [
            ("NEXIUM_RT_STATS", "1"),
            ("NEXIUM_RT_STATS_PERIOD", "120"),
            ("NEXIUM_TEXDUMP", "1"),
            ("NEXIUM_PROBE_SHADE", "1"),
            ("NEXIUM_TEX_BIND_LOG", "1"),
            ("NEXIUM_CBUF_WATCH", "0:0:0x40,16:0:0x40"),
            ("NEXIUM_BIND_TRACE_FS", "all"),
            ("NEXIUM_DUMP_SHADERS", "all"),
            ("NEXIUM_RT_ALIAS_DBG", "1"),
            ("NEXIUM_RT_ALIAS_SYNC_DBG", "1"),
            ("NEXIUM_VTX_DBG", "all"),
            ("NEXIUM_ZETA_DBG", "1"),
            ("NEXIUM_SPIRV_PHI_DBG", "1"),
            ("NEXIUM_PRESENT_KEYS", "1"),
        ];
        for (k, v) in defaults {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
    }

    let settings = AppSettings::load();

    let (logger, log_buffer) = FileLogger::new(500).unwrap_or_else(|e| {
        eprintln!("Failed to initialize logger: {}", e);
        panic!("Logger initialization failed");
    });

    let log_filter = std::env::var("NEXIUM_LOG_LEVEL")
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "error" => Some(log::LevelFilter::Error),
            "warn" => Some(log::LevelFilter::Warn),
            "info" => Some(log::LevelFilter::Info),
            "debug" => Some(log::LevelFilter::Debug),
            "trace" => Some(log::LevelFilter::Trace),
            _ => None,
        })
        .unwrap_or_else(|| settings.log_level.to_filter());
    if let Err(e) = logger.init(log_filter) {
        eprintln!("Failed to set logger: {}", e);
    }

    log::info!("=== NeXium - Nintendo Switch Emulator ===");
    nexium_gpu::adapter::set_preferred_device(settings.gpu_device.clone());

    let docked = std::env::var("NEXIUM_DOCKED").map_or(settings.docked, |value| value != "0");
    nexium_core::hid_state::set_docked(docked);
    log::info!(
        "console mode initialized as {}",
        if docked { "Docked" } else { "Handheld" }
    );

    #[cfg(windows)]
    fault_logger::install();

    nexium_common::async_compile::set_enabled(settings.async_shaders);
    nexium_common::fast_gpu_time::set_enabled(settings.fast_gpu_time);
    nexium_common::force_max_clocks::set_enabled(settings.force_max_clocks);
    nexium_common::depth_share::set_enabled(
        std::env::var("NEXIUM_DEPTH_SHARE").map_or(settings.depth_share, |v| v != "0"),
    );
    log::info!(
        "depth share for ReShade add-ons: {}",
        if nexium_common::depth_share::enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    #[cfg(windows)]
    {
        #[link(name = "winmm")]
        extern "system" {
            fn timeBeginPeriod(uperiod: u32) -> u32;
        }
        unsafe {
            timeBeginPeriod(1);
        }
        log::info!("timer resolution pinned to 1ms");
    }

    audio::init_host_audio(
        settings.audio_output_device.as_deref(),
        settings.audio_volume,
    );

    let nro_arg = std::env::args().nth(1);

    let icon = eframe::icon_data::from_png_bytes(
        include_bytes!("../../branding/png/logo-256.png").as_ref(),
    )
    .expect("logo PNG decode");

    let host_vsync = host_vsync_enabled(
        settings.vsync,
        std::env::var("NEXIUM_HOST_VSYNC").ok().as_deref(),
    );
    let surface_config = host_surface_config(host_vsync);
    log::info!(
        "host surface: vsync={} present_mode={:?} max_frame_latency={:?}",
        host_vsync,
        surface_config.present_mode,
        surface_config.desired_maximum_frame_latency,
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NeXium")
            .with_app_id("NeXium")
            .with_icon(icon)
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(host_wgpu_setup(
                settings.gpu_device.as_deref(),
            )),
            ..Default::default()
        }
        .with_surface_config(surface_config),
        ..Default::default()
    };

    let log_buf_for_app = log_buffer.clone();
    let result = eframe::run_native(
        "NeXium",
        options,
        Box::new(move |cc| {
            Ok(Box::new(HorizonApp::new(
                cc,
                log_buf_for_app.clone(),
                settings.clone(),
                nro_arg.clone(),
            )))
        }),
    );
    if let Err(e) = &result {
        log::error!("eframe exited with error: {e}");
    }
    log::logger().flush();
    #[cfg(windows)]
    if std::env::var("NEXIUM_FAST_EXIT").map_or(true, |v| v != "0") {
        fast_exit::terminate(u32::from(result.is_err()));
    }
    result
}

#[cfg(windows)]
mod fast_exit {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn TerminateProcess(process: *mut std::ffi::c_void, exit_code: u32) -> i32;
    }

    pub fn terminate(code: u32) {
        unsafe {
            TerminateProcess(GetCurrentProcess(), code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{host_surface_config, host_vsync_enabled, host_wgpu_setup};

    #[test]
    fn host_wgpu_setup_defaults_to_vulkan_only() {
        if std::env::var_os("WGPU_BACKEND").is_none() {
            assert_eq!(
                host_wgpu_setup(None).instance_descriptor.backends,
                eframe::wgpu::Backends::VULKAN
            );
        }
    }

    #[test]
    fn host_vsync_override_only_accepts_known_values() {
        assert!(host_vsync_enabled(false, Some("1")));
        assert!(!host_vsync_enabled(true, Some("off")));
        assert!(host_vsync_enabled(true, Some("unexpected")));
    }

    #[test]
    fn host_surface_config_matches_vsync_preference() {
        assert_eq!(
            host_surface_config(true).present_mode,
            eframe::wgpu::PresentMode::Fifo
        );
        assert_eq!(
            host_surface_config(false).present_mode,
            eframe::wgpu::PresentMode::AutoNoVsync
        );
        assert_eq!(
            host_surface_config(true).desired_maximum_frame_latency,
            Some(2)
        );
    }
}

#[cfg(windows)]
mod fault_logger {
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    static ARENA_BASE: AtomicUsize = AtomicUsize::new(0);
    const ARENA_SIZE: u64 = 1u64 << 40;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut std::ffi::c_void,
        num_params: u32,
        information: [usize; 15],
    }
    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut std::ffi::c_void,
    }

    const GP_NAMES: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];

    unsafe extern "system" fn handler(info: *mut ExceptionPointers) -> i32 {
        const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
        const ACCESS_VIOLATION: u32 = 0xC000_0005;
        if !info.is_null() {
            let rec = (*info).exception_record;
            if !rec.is_null() && (*rec).code == ACCESS_VIOLATION {
                let access = (*rec).information[0];
                let fault = (*rec).information[1] as u64;
                let base = ARENA_BASE.load(Ordering::Relaxed) as u64;
                let in_arena = base != 0 && fault >= base && fault < base + ARENA_SIZE;
                static N: AtomicU32 = AtomicU32::new(0);
                if !in_arena && N.fetch_add(1, Ordering::Relaxed) < 8 {
                    let _ = nexium_common::FileLogger::flush_on_fault();
                    let kind = match access {
                        0 => "read",
                        1 => "write",
                        8 => "exec",
                        _ => "?",
                    };
                    let region = if in_arena { "in-arena" } else { "out-of-arena" };
                    eprintln!(
                        "[host-AV] ACCESS VIOLATION fault={:#x} guest_addr={:#x} {} {} insn={:p} base={:#x}",
                        fault,
                        fault.wrapping_sub(base),
                        kind,
                        region,
                        (*rec).address,
                        base
                    );
                    let ctx = (*info).context_record;
                    if !ctx.is_null() {
                        let gp = (ctx as *const u8).add(0x78) as *const u64;
                        for i in 0..16 {
                            let v = core::ptr::read_unaligned(gp.add(i));
                            let tag = if base != 0 && v >= base && v < base + ARENA_SIZE {
                                format!(" (arena+{:#x})", v.wrapping_sub(base))
                            } else if let Some((name, _, rva)) = module_of(v) {
                                format!(" ({}+{:#x})", name, rva)
                            } else {
                                String::new()
                            };
                            eprintln!("[host-AV]   {}={:#018x}{}", GP_NAMES[i], v, tag);
                        }
                        let rip =
                            core::ptr::read_unaligned((ctx as *const u8).add(0xf8) as *const u64);
                        eprintln!("[host-AV]   rip={:#018x}", rip);
                        let exe_base = GetModuleHandleW(core::ptr::null()) as u64;
                        if let Some((name, mod_base, rva)) = module_of(rip) {
                            eprintln!(
                                "[host-AV]   rip module={} base={:#x} rva={:#x} in_exe={}",
                                name,
                                mod_base,
                                rva,
                                mod_base == exe_base
                            );
                        }
                        let rsp =
                            core::ptr::read_unaligned((ctx as *const u8).add(0x98) as *const u64);
                        let mut region: MemoryBasicInformation = core::mem::zeroed();
                        let mut scan_end = rsp;
                        if VirtualQuery(
                            rsp as *const std::ffi::c_void,
                            &mut region,
                            core::mem::size_of::<MemoryBasicInformation>(),
                        ) != 0
                            && region.state == 0x1000
                            && region.protect & 0x101 == 0
                        {
                            scan_end = (region.base_address as u64)
                                .saturating_add(region.region_size as u64)
                                .min(rsp.saturating_add(0x4000));
                        }
                        eprintln!("[host-AV]   stack scan {:#x}..{:#x}", rsp, scan_end);
                        let mut found = 0u32;
                        let mut off = 0u64;
                        while rsp + off + 8 <= scan_end && found < 32 {
                            let v = core::ptr::read_unaligned((rsp + off) as *const u64);
                            if let Some((name, _, rva)) = module_of(v) {
                                eprintln!("[host-AV]   stack[{:#x}] {}+{:#x}", off, name, rva);
                                found += 1;
                            }
                            off += 8;
                        }
                    }
                }
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    #[repr(C)]
    struct MemoryBasicInformation {
        base_address: *mut std::ffi::c_void,
        allocation_base: *mut std::ffi::c_void,
        allocation_protect: u32,
        partition_id: u16,
        region_size: usize,
        state: u32,
        protect: u32,
        kind: u32,
    }

    unsafe fn module_of(addr: u64) -> Option<(String, u64, u64)> {
        if addr < 0x10000 {
            return None;
        }
        let mut hmod: *mut std::ffi::c_void = core::ptr::null_mut();
        if GetModuleHandleExW(0x6, addr as *const u16, &mut hmod) == 0 || hmod.is_null() {
            return None;
        }
        let mut name = [0u16; 260];
        let n = GetModuleFileNameW(hmod, name.as_mut_ptr(), 260) as usize;
        let path = &name[..n.min(260)];
        let start = path
            .iter()
            .rposition(|&c| c == u16::from(b'\\'))
            .map_or(0, |p| p + 1);
        Some((
            String::from_utf16_lossy(&path[start..]),
            hmod as u64,
            addr.wrapping_sub(hmod as u64),
        ))
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
        ) -> *mut std::ffi::c_void;
        fn GetModuleHandleW(name: *const u16) -> *mut std::ffi::c_void;
        fn GetModuleHandleExW(
            flags: u32,
            addr: *const u16,
            module: *mut *mut std::ffi::c_void,
        ) -> i32;
        fn GetModuleFileNameW(module: *mut std::ffi::c_void, buf: *mut u16, size: u32) -> u32;
        fn VirtualQuery(
            address: *const std::ffi::c_void,
            buffer: *mut MemoryBasicInformation,
            length: usize,
        ) -> usize;
    }

    pub fn install() {
        if let Some(base) = nexium_memory::fastmem::base() {
            ARENA_BASE.store(base as usize, Ordering::Relaxed);
        }
        unsafe {
            AddVectoredExceptionHandler(1, handler);
        }
        log::info!(
            "[fault-logger] vectored AV handler installed (arena base={:#x})",
            ARENA_BASE.load(Ordering::Relaxed)
        );
    }
}
