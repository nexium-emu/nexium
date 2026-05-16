mod app;
mod boot;
mod input;
mod audio;
mod debugger;
mod performance;

use app::HorizonApp;
use horizonrust_common::FileLogger;
use log::LevelFilter;

fn main() -> Result<(), eframe::Error> {
    let (logger, log_buffer) = FileLogger::new(500)
        .unwrap_or_else(|e| {
            eprintln!("Failed to initialize logger: {}", e);
            panic!("Logger initialization failed");
        });

    if let Err(e) = logger.init(LevelFilter::Info) {
        eprintln!("Failed to set logger: {}", e);
    }

    log::info!("=== HorizonRust - Nintendo Switch Emulator ===");

    let nro_arg = std::env::args().nth(1);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("HorizonRust")
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    let log_buf_for_app = log_buffer.clone();
    eframe::run_native(
        "HorizonRust",
        options,
        Box::new(move |cc| {
            Ok(Box::new(HorizonApp::new(cc, log_buf_for_app.clone(), nro_arg.clone())))
        }),
    )
}
