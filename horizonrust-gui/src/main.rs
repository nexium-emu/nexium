mod app;
mod boot;
mod input;

use app::HorizonApp;

fn main() -> Result<(), eframe::Error> {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("=== HorizonRust - Nintendo Switch Emulator ===");
    log::info!("Phase 4 - GUI Integration");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("HorizonRust")
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "HorizonRust",
        options,
        Box::new(|cc| Ok(Box::new(HorizonApp::new(cc)))),
    )
}
