#![allow(dead_code)]

mod app;
mod boot;
mod input;
mod audio;
mod debugger;
mod performance;
mod controller_config;
mod app_settings;

use app::HorizonApp;
use app_settings::AppSettings;
use nexium_common::FileLogger;

fn main() -> Result<(), eframe::Error> {
    let settings = AppSettings::load();

    let (logger, log_buffer) = FileLogger::new(500)
        .unwrap_or_else(|e| {
            eprintln!("Failed to initialize logger: {}", e);
            panic!("Logger initialization failed");
        });

    if let Err(e) = logger.init(settings.log_level.to_filter()) {
        eprintln!("Failed to set logger: {}", e);
    }

    log::info!("=== NeXium - Nintendo Switch Emulator ===");

    audio::init_host_audio(
        settings.audio_output_device.as_deref(),
        settings.audio_volume,
    );

    let nro_arg = std::env::args().nth(1);

    let icon = eframe::icon_data::from_png_bytes(
        include_bytes!("../../branding/png/logo-256.png").as_ref(),
    ).expect("logo PNG decode");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NeXium")
            .with_icon(icon)
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    let log_buf_for_app = log_buffer.clone();
    eframe::run_native(
        "NeXium",
        options,
        Box::new(move |cc| {
            Ok(Box::new(HorizonApp::new(cc, log_buf_for_app.clone(), nro_arg.clone())))
        }),
    )
}
