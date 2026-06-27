#![allow(dead_code)]

mod app;
mod app_settings;
mod audio;
mod boot;
mod controller_art;
mod controller_config;
mod debugger;
mod input;
mod performance;

use app::HorizonApp;
use app_settings::AppSettings;
use nexium_common::FileLogger;

fn main() -> Result<(), eframe::Error> {
    let settings = AppSettings::load();

    let (logger, log_buffer) = FileLogger::new(500).unwrap_or_else(|e| {
        eprintln!("Failed to initialize logger: {}", e);
        panic!("Logger initialization failed");
    });

    if let Err(e) = logger.init(settings.log_level.to_filter()) {
        eprintln!("Failed to set logger: {}", e);
    }

    log::info!("=== NeXium - Nintendo Switch Emulator ===");

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

// Diagnostic: a vectored exception handler that logs guest-triggered host access
// violations (fastmem dereference of a wild/out-of-arena guest pointer) before
// the process dies, so we can recover the bad guest address. Read-only observer:
// it always returns EXCEPTION_CONTINUE_SEARCH and never alters control flow.
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
                if !in_arena {
                    static N: AtomicU32 = AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        let kind = match access {
                            0 => "read",
                            1 => "write",
                            8 => "exec",
                            _ => "?",
                        };
                        eprintln!(
                            "[host-AV] ACCESS VIOLATION fault={:#x} guest_addr={:#x} {} insn={:p} (out-of-arena base={:#x})",
                            fault,
                            fault.wrapping_sub(base),
                            kind,
                            (*rec).address,
                            base
                        );
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
