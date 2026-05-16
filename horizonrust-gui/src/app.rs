use eframe::egui;
use gilrs::Gilrs;
use crate::boot::EmulationHandle;
use crate::input::InputSnapshot;

pub struct HorizonApp {
    nro_path: String,
    emulation_handle: Option<EmulationHandle>,
    show_file_picker: bool,
    show_settings: bool,
    gilrs: Option<Gilrs>,
    last_input: InputSnapshot,
}

impl HorizonApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                log::warn!("Failed to initialize input (Gilrs): {}", e);
                None
            }
        };

        Self {
            nro_path: String::new(),
            emulation_handle: None,
            show_file_picker: false,
            show_settings: false,
            gilrs,
            last_input: InputSnapshot::default(),
        }
    }

    fn update_input(&mut self) {
        if let Some(ref mut gilrs) = self.gilrs {
            self.last_input = InputSnapshot::update_from_gamepad(gilrs);
        }
    }

    fn boot_nro(&mut self) {
        if self.nro_path.is_empty() {
            log::warn!("NRO path is empty");
            return;
        }

        if let Ok(handle) = EmulationHandle::new(&self.nro_path) {
            self.emulation_handle = Some(handle);
            log::info!("Booted NRO: {}", self.nro_path);
        } else {
            log::error!("Failed to boot NRO: {}", self.nro_path);
        }
    }

    fn stop_emulation(&mut self) {
        if let Some(mut handle) = self.emulation_handle.take() {
            handle.stop();
            log::info!("Emulation stopped");
        }
    }
}

impl eframe::App for HorizonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_input();

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open NRO...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Nintendo Homebrew", &["nro"])
                        .pick_file()
                    {
                        self.nro_path = path.to_string_lossy().to_string();
                    }
                    ui.close_menu();
                }

                if ui.button("Exit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("Emulation", |ui| {
                if ui.button("Boot NRO").clicked() {
                    self.boot_nro();
                    ui.close_menu();
                }

                if ui.button("Stop").clicked() {
                    self.stop_emulation();
                    ui.close_menu();
                }
            });

            ui.menu_button("Settings", |ui| {
                if ui.button("Settings Panel").clicked() {
                    self.show_settings = !self.show_settings;
                    ui.close_menu();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("HorizonRust - Nintendo Switch Emulator");
            ui.label("Phase 4 - GUI Integration");

            ui.separator();

            ui.label(format!("NRO Path: {}", self.nro_path));

            if ui.button("Select NRO...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Nintendo Homebrew", &["nro"])
                    .pick_file()
                {
                    self.nro_path = path.to_string_lossy().to_string();
                }
            }

            if ui.button("Boot").clicked() {
                self.boot_nro();
            }

            if let Some(handle) = &self.emulation_handle {
                if handle.is_running() {
                    if ui.button("Stop").clicked() {
                        self.stop_emulation();
                    }
                    ui.label("Emulation running...");
                } else {
                    ui.label("Emulation complete");
                    self.emulation_handle = None;
                }
            }

            ui.separator();
            ui.label("Input Status:");
            ui.label(format!("A: {}", self.last_input.a_pressed));
            ui.label(format!("B: {}", self.last_input.b_pressed));
            ui.label(format!("X: {}", self.last_input.x_pressed));
            ui.label(format!("Y: {}", self.last_input.y_pressed));
        });

        if self.show_settings {
            egui::Window::new("Settings").open(&mut self.show_settings).show(ctx, |ui| {
                ui.label("Settings Panel");
                ui.label("Audio Volume: 100%");
                ui.label("GPU Backend: Vulkan");
                ui.label("CPU Backend: Dynarmic");
            });
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}
