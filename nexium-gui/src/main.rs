#![allow(dead_code)]

mod app;
mod app_settings;
mod audio;
mod boot;
mod carousel;
mod controller_art;
mod controller_config;
mod debugger;
mod homebrew;
mod input;
mod library;
mod performance;
mod playtime;
mod profile;
mod shop;
mod splash;
mod steamgrid;
mod ui_audio;
mod updater;
mod vkeyboard;

use app::HorizonApp;
use app_settings::AppSettings;
use nexium_common::FileLogger;

fn main() -> Result<(), eframe::Error> {
    // Re-enabled for the guest by the game window process.
    std::env::set_var("DISABLE_MANGOHUD", "1");

    let settings = AppSettings::load();

    let (logger, log_buffer) = FileLogger::new(500).unwrap_or_else(|e| {
        eprintln!("Failed to initialize logger: {}", e);
        panic!("Logger initialization failed");
    });

    if let Err(e) = logger.init(settings.log_level.to_filter()) {
        eprintln!("Failed to set logger: {}", e);
    }

    log::info!("=== NeXium - Nintendo Switch Emulator ===");

    if let Ok(v) = std::env::var("NEXIUM_DOCKED") {
        let docked = v != "0";
        nexium_core::hid_state::set_docked(docked);
        log::info!(
            "console mode initialized from NEXIUM_DOCKED={}",
            if docked { "1" } else { "0" }
        );
    }

    #[cfg(windows)]
    fault_logger::install();

    nexium_common::async_compile::set_enabled(settings.async_shaders);

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

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NeXium")
            .with_app_id("NeXium")
            .with_icon(icon)
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        vsync: settings.vsync,
        ..Default::default()
    };

    let log_buf_for_app = log_buffer.clone();
    eframe::run_native(
        "NeXium",
        options,
        Box::new(move |cc| {
            Ok(Box::new(HorizonApp::new(
                cc,
                log_buf_for_app.clone(),
                nro_arg.clone(),
            )))
        }),
    )
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
                            } else {
                                String::new()
                            };
                            eprintln!("[host-AV]   {}={:#018x}{}", GP_NAMES[i], v, tag);
                        }
                        let rip =
                            core::ptr::read_unaligned((ctx as *const u8).add(0xf8) as *const u64);
                        eprintln!("[host-AV]   rip={:#018x}", rip);
                        let exe_base = GetModuleHandleW(core::ptr::null()) as u64;
                        let mut hmod: *mut std::ffi::c_void = core::ptr::null_mut();
                        let ok = GetModuleHandleExW(0x6, rip as *const u16, &mut hmod);
                        let mod_base = hmod as u64;
                        if ok != 0 && mod_base != 0 {
                            let mut name = [0u16; 260];
                            let n = GetModuleFileNameW(hmod, name.as_mut_ptr(), 260);
                            let fname = String::from_utf16_lossy(&name[..n as usize]);
                            eprintln!(
                                "[host-AV]   rip module={} base={:#x} rva={:#x} in_exe={}",
                                fname,
                                mod_base,
                                rip.wrapping_sub(mod_base),
                                mod_base == exe_base
                            );
                        }
                        let rsp =
                            core::ptr::read_unaligned((ctx as *const u8).add(0x98) as *const u64);
                        let mut found = 0u32;
                        let mut off = 0u64;
                        while off < 0x4000 && found < 24 {
                            let v = core::ptr::read_unaligned((rsp + off) as *const u64);
                            if v > 0x10000 {
                                let mut hm: *mut std::ffi::c_void = core::ptr::null_mut();
                                if GetModuleHandleExW(0x6, v as *const u16, &mut hm) != 0
                                    && hm as u64 == exe_base
                                {
                                    eprintln!(
                                        "[host-AV]   stack[{:#x}] nexium.exe+{:#x}",
                                        off,
                                        v.wrapping_sub(exe_base)
                                    );
                                    found += 1;
                                }
                            }
                            off += 8;
                        }
                    }
                }
            }
        }
        EXCEPTION_CONTINUE_SEARCH
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
