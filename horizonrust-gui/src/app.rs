use eframe::egui;
use gilrs::Gilrs;
use crate::boot::EmulationHandle;
use crate::input::InputSnapshot;
use crate::debugger::DebuggerState;
use crate::performance::PerformanceMonitor;

pub struct HorizonApp {
    nro_path: String,
    emulation_handle: Option<EmulationHandle>,
    game_texture: Option<egui::TextureHandle>,
    show_settings: bool,
    gilrs: Option<Gilrs>,
    last_input: InputSnapshot,
    debugger: DebuggerState,
    performance: PerformanceMonitor,
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
            game_texture: None,
            show_settings: false,
            gilrs,
            last_input: InputSnapshot::default(),
            debugger: DebuggerState::new(),
            performance: PerformanceMonitor::new(),
        }
    }

    fn update_input(&mut self) {
        if let Some(ref mut gilrs) = self.gilrs {
            self.last_input = InputSnapshot::update_from_gamepad(gilrs);
        }
    }

    fn poll_frames(&mut self, ctx: &egui::Context) {
        let Some(handle) = &self.emulation_handle else { return };
        while let Ok(frame) = handle.frame_rx.try_recv() {
            if frame.width == 0 || frame.height == 0 || frame.pixels.is_empty() {
                continue;
            }
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize],
                &frame.pixels,
            );
            match &mut self.game_texture {
                Some(tex) => tex.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.game_texture = Some(ctx.load_texture(
                        "game_frame",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
        }
    }

    fn boot_nro(&mut self) {
        if self.nro_path.is_empty() {
            log::warn!("NRO path is empty");
            return;
        }

        match EmulationHandle::new(&self.nro_path) {
            Ok(handle) => {
                self.emulation_handle = Some(handle);
                self.game_texture = None;
                log::info!("Booted NRO: {}", self.nro_path);
            }
            Err(e) => log::error!("Failed to boot NRO: {}", e),
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
        self.poll_frames(ctx);

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

            ui.menu_button("Debug", |ui| {
                if ui.button("Memory Viewer").clicked() {
                    self.debugger.toggle_memory();
                    ui.close_menu();
                }
                if ui.button("CPU Registers").clicked() {
                    self.debugger.toggle_registers();
                    ui.close_menu();
                }
                if ui.button("Disassembler").clicked() {
                    self.debugger.toggle_disasm();
                    ui.close_menu();
                }
                if ui.button("Log Viewer").clicked() {
                    self.debugger.toggle_logs();
                    ui.close_menu();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.game_texture {
                let available = ui.available_size();
                let tex_size = tex.size_vec2();
                let scale = (available.x / tex_size.x).min(available.y / tex_size.y).min(1.0);
                let display_size = egui::vec2(tex_size.x * scale, tex_size.y * scale);
                ui.centered_and_justified(|ui| {
                    ui.image((tex.id(), display_size));
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.heading("HorizonRust");
                    ui.separator();
                    ui.label(format!("NRO: {}", if self.nro_path.is_empty() { "(none)" } else { &self.nro_path }));

                    if ui.button("Select NRO...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Nintendo Homebrew", &["nro"])
                            .pick_file()
                        {
                            self.nro_path = path.to_string_lossy().to_string();
                        }
                    }

                    if !self.nro_path.is_empty() {
                        if ui.button("Boot").clicked() {
                            self.boot_nro();
                        }
                    }

                    if let Some(handle) = &self.emulation_handle {
                        if handle.is_running() {
                            ui.separator();
                            ui.label("Emulation running...");
                            if ui.button("Stop").clicked() {
                                self.stop_emulation();
                            }
                        } else {
                            ui.label("Emulation complete");
                            self.emulation_handle = None;
                        }
                    }

                    ui.separator();
                    ui.label(format!("FPS: {:.1}  Frame: {:.2}ms  SVCs: {}  Cycles: {}",
                        self.performance.get_fps(),
                        self.performance.get_frame_time(),
                        self.performance.get_svc_count(),
                        self.performance.get_cycle_count(),
                    ));

                    ui.separator();
                    ui.label(format!("A:{} B:{} X:{} Y:{}",
                        self.last_input.a_pressed as u8,
                        self.last_input.b_pressed as u8,
                        self.last_input.x_pressed as u8,
                        self.last_input.y_pressed as u8,
                    ));
                });
            }
        });

        self.performance.record_frame();

        if self.show_settings {
            egui::Window::new("Settings").open(&mut self.show_settings).show(ctx, |ui| {
                ui.label("Audio Volume: 100%");
                ui.label("GPU Backend: Vulkan");
                ui.label("CPU Backend: Dynarmic");
            });
        }

        if self.debugger.show_memory {
            egui::Window::new("Memory Viewer")
                .open(&mut self.debugger.show_memory)
                .show(ctx, |ui| {
                    ui.label("Memory Address:");
                    ui.text_edit_singleline(&mut format!("{:#x}", self.debugger.memory_address));
                    ui.label(format!("Size: {} bytes", self.debugger.memory_size));
                    ui.label("[Memory content would be displayed here]");
                });
        }

        if self.debugger.show_registers {
            egui::Window::new("CPU Registers")
                .open(&mut self.debugger.show_registers)
                .show(ctx, |ui| {
                    ui.label("X0: 0x00000000");
                    ui.label("X1: 0x00000000");
                    ui.label("PC: 0x00000000");
                    ui.label("SP: 0x00000000");
                });
        }

        if self.debugger.show_disasm {
            egui::Window::new("Disassembler")
                .open(&mut self.debugger.show_disasm)
                .show(ctx, |ui| {
                    ui.label("[Disassembly would be displayed here]");
                });
        }

        if self.debugger.show_logs {
            egui::Window::new("Log Viewer")
                .open(&mut self.debugger.show_logs)
                .show(ctx, |ui| {
                    if ui.button("Clear Logs").clicked() {
                        self.debugger.log_history.lock().clear();
                    }
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            for entry in self.debugger.log_history.lock().iter() {
                                ui.label(entry.clone());
                            }
                        });
                });
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}
