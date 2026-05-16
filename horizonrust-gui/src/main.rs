mod app;
mod boot;
mod input;
mod audio;
mod debugger;
mod performance;

use app::HorizonApp;
use horizonrust_common::BufferedLogger;
use log::LevelFilter;
use std::sync::Mutex;
use std::collections::VecDeque;

fn main() -> Result<(), eframe::Error> {
    let (logger, log_buffer) = BufferedLogger::new(500);
    let _ = logger.init(LevelFilter::Info);

    log::info!("=== HorizonRust - Nintendo Switch Emulator ===");

    let log_buffer = std::sync::Arc::new(Mutex::new(log_buffer));

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
            Ok(Box::new(HorizonApp::new(cc, log_buf_for_app.clone())))
        }),
    )
}
