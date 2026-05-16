use eframe::egui;
use eframe::egui::{Color32, FontId, RichText, Stroke, Rounding, Vec2, Sense};
use gilrs::Gilrs;
use crate::boot::EmulationHandle;
use crate::input::InputSnapshot;
use crate::debugger::DebuggerState;
use crate::performance::PerformanceMonitor;

const ACCENT: Color32 = Color32::from_rgb(0xE8, 0x30, 0x30);
const BG_PANEL: Color32 = Color32::from_rgb(0x14, 0x14, 0x16);
const BG_CARD: Color32 = Color32::from_rgb(0x1C, 0x1C, 0x1F);
const BG_HOVER: Color32 = Color32::from_rgb(0x26, 0x26, 0x2A);
const TEXT_DIM: Color32 = Color32::from_rgb(0x88, 0x88, 0x96);
const TEXT_BRIGHT: Color32 = Color32::from_rgb(0xF0, 0xF0, 0xF4);
const BORDER: Color32 = Color32::from_rgb(0x2C, 0x2C, 0x32);

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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);

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
                Some(tex) => tex.set(image, egui::TextureOptions::NEAREST),
                None => {
                    self.game_texture = Some(ctx.load_texture(
                        "game_frame",
                        image,
                        egui::TextureOptions::NEAREST,
                    ));
                }
            }
        }
    }

    fn boot_nro(&mut self) {
        if self.nro_path.is_empty() {
            return;
        }
        match EmulationHandle::new(&self.nro_path) {
            Ok(handle) => {
                self.emulation_handle = Some(handle);
                self.game_texture = None;
                log::info!("Booted: {}", self.nro_path);
            }
            Err(e) => log::error!("Boot failed: {}", e),
        }
    }

    fn stop_emulation(&mut self) {
        if let Some(mut handle) = self.emulation_handle.take() {
            handle.stop();
        }
    }

    fn is_running(&self) -> bool {
        self.emulation_handle.as_ref().map_or(false, |h| h.is_running())
    }
}

fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.visuals.dark_mode = true;
    style.visuals.panel_fill = BG_PANEL;
    style.visuals.window_fill = BG_CARD;
    style.visuals.faint_bg_color = BG_CARD;
    style.visuals.extreme_bg_color = Color32::from_rgb(0x0C, 0x0C, 0x0E);
    style.visuals.code_bg_color = Color32::from_rgb(0x0C, 0x0C, 0x0E);
    style.visuals.override_text_color = Some(TEXT_BRIGHT);

    style.visuals.window_stroke = Stroke::new(1.0, BORDER);
    style.visuals.window_rounding = Rounding::same(6.0);
    style.visuals.menu_rounding = Rounding::same(6.0);

    style.visuals.widgets.noninteractive.bg_fill = BG_CARD;
    style.visuals.widgets.noninteractive.weak_bg_fill = BG_CARD;
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);
    style.visuals.widgets.noninteractive.rounding = Rounding::same(4.0);

    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(0x22, 0x22, 0x26);
    style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(0x22, 0x22, 0x26);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.inactive.rounding = Rounding::same(4.0);

    style.visuals.widgets.hovered.bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.weak_bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x44, 0x44, 0x4C));
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.hovered.rounding = Rounding::same(4.0);

    style.visuals.widgets.active.bg_fill = ACCENT;
    style.visuals.widgets.active.weak_bg_fill = ACCENT;
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    style.visuals.widgets.active.rounding = Rounding::same(4.0);

    style.visuals.widgets.open.bg_fill = BG_HOVER;
    style.visuals.widgets.open.weak_bg_fill = BG_HOVER;
    style.visuals.widgets.open.bg_stroke = Stroke::new(1.0, ACCENT);
    style.visuals.widgets.open.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.open.rounding = Rounding::same(4.0);

    style.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(0xE8, 0x30, 0x30, 0x40);
    style.visuals.selection.stroke = Stroke::new(1.0, ACCENT);

    style.visuals.popup_shadow = egui::epaint::Shadow {
        offset: Vec2::new(0.0, 4.0),
        blur: 12.0,
        spread: 0.0,
        color: Color32::from_black_alpha(120),
    };

    style.spacing.item_spacing = Vec2::new(8.0, 4.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.menu_margin = egui::Margin::same(4.0);
    style.spacing.window_margin = egui::Margin::same(12.0);

    ctx.set_style(style);
}

fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let desired_size = Vec2::new(
        ui.fonts(|f| f.glyph_width(&FontId::proportional(14.0), 'x')) * label.len() as f32 + 32.0,
        28.0,
    );
    let (rect, resp) = ui.allocate_exact_size(desired_size, Sense::click());
    let bg = if resp.is_pointer_button_down_on() {
        Color32::from_rgb(0xC0, 0x20, 0x20)
    } else if resp.hovered() {
        Color32::from_rgb(0xF0, 0x40, 0x40)
    } else {
        ACCENT
    };
    ui.painter().rect_filled(rect, Rounding::same(4.0), bg);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(13.0),
        Color32::WHITE,
    );
    resp
}

fn ghost_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let desired_size = Vec2::new(
        ui.fonts(|f| f.glyph_width(&FontId::proportional(14.0), 'x')) * label.len() as f32 + 32.0,
        28.0,
    );
    let (rect, resp) = ui.allocate_exact_size(desired_size, Sense::click());
    let (bg, stroke_color) = if resp.is_pointer_button_down_on() {
        (BG_HOVER, ACCENT)
    } else if resp.hovered() {
        (BG_HOVER, Color32::from_rgb(0x66, 0x66, 0x72))
    } else {
        (Color32::TRANSPARENT, BORDER)
    };
    ui.painter().rect(rect, Rounding::same(4.0), bg, Stroke::new(1.0, stroke_color));
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(13.0),
        TEXT_BRIGHT,
    );
    resp
}

impl eframe::App for HorizonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_input();
        self.poll_frames(ctx);

        if self.emulation_handle.as_ref().map_or(false, |h| !h.is_running()) {
            self.emulation_handle = None;
        }

        egui::TopBottomPanel::top("menu_bar")
            .frame(egui::Frame::none()
                .fill(Color32::from_rgb(0x10, 0x10, 0x12))
                .inner_margin(egui::Margin { left: 8.0, right: 8.0, top: 2.0, bottom: 2.0 })
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.label(RichText::new("HorizonRust").color(TEXT_BRIGHT).size(13.0).strong());
                    ui.add_space(8.0);

                    ui.separator();
                    ui.add_space(4.0);

                    ui.menu_button(RichText::new("File").color(TEXT_BRIGHT).size(13.0), |ui| {
                        if ui.button("Open NRO…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Nintendo Homebrew", &["nro"])
                                .pick_file()
                            {
                                self.nro_path = path.to_string_lossy().to_string();
                            }
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Exit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    ui.menu_button(RichText::new("Emulation").color(TEXT_BRIGHT).size(13.0), |ui| {
                        if ui.button("Boot NRO").clicked() {
                            self.boot_nro();
                            ui.close_menu();
                        }
                        if ui.button("Stop").clicked() {
                            self.stop_emulation();
                            ui.close_menu();
                        }
                    });

                    ui.menu_button(RichText::new("Debug").color(TEXT_BRIGHT).size(13.0), |ui| {
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

                    ui.menu_button(RichText::new("Settings").color(TEXT_BRIGHT).size(13.0), |ui| {
                        if ui.button("Preferences").clicked() {
                            self.show_settings = !self.show_settings;
                            ui.close_menu();
                        }
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let fps = self.performance.get_fps();
                        let color = if fps >= 55.0 { Color32::from_rgb(0x4C, 0xD9, 0x64) }
                            else if fps >= 30.0 { Color32::from_rgb(0xF5, 0xA6, 0x23) }
                            else { Color32::from_rgb(0xE8, 0x30, 0x30) };
                        ui.label(RichText::new(format!("{:.0} fps", fps)).color(color).size(12.0).monospace());
                        ui.add_space(4.0);
                        ui.label(RichText::new("·").color(TEXT_DIM).size(12.0));
                        ui.add_space(4.0);
                        let status = if self.is_running() {
                            RichText::new("● Running").color(Color32::from_rgb(0x4C, 0xD9, 0x64)).size(12.0)
                        } else {
                            RichText::new("○ Idle").color(TEXT_DIM).size(12.0)
                        };
                        ui.label(status);
                    });
                });
            });

        egui::TopBottomPanel::bottom("status_bar")
            .frame(egui::Frame::none()
                .fill(Color32::from_rgb(0x10, 0x10, 0x12))
                .inner_margin(egui::Margin { left: 12.0, right: 12.0, top: 4.0, bottom: 4.0 })
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let nro_name = if self.nro_path.is_empty() {
                        "No file loaded".to_string()
                    } else {
                        std::path::Path::new(&self.nro_path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| self.nro_path.clone())
                    };
                    ui.label(RichText::new(&nro_name).color(TEXT_DIM).size(11.0));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(
                            format!("SVCs: {}  Cycles: {}  Frame: {:.1}ms",
                                self.performance.get_svc_count(),
                                self.performance.get_cycle_count(),
                                self.performance.get_frame_time(),
                            )
                        ).color(TEXT_DIM).size(11.0).monospace());
                    });
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG_PANEL))
            .show(ctx, |ui| {
                if let Some(tex) = &self.game_texture {
                    let avail = ui.available_size();
                    let tex_size = tex.size_vec2();
                    let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y);
                    let display_size = tex_size * scale;
                    ui.centered_and_justified(|ui| {
                        ui.image((tex.id(), display_size));
                    });
                } else {
                    let nro_path = self.nro_path.clone();
                    let running = self.is_running();
                    let action = draw_idle_screen(ui, &nro_path, running);
                    match action {
                        IdleAction::SelectFile => {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Nintendo Homebrew", &["nro"])
                                .pick_file()
                            {
                                self.nro_path = path.to_string_lossy().to_string();
                            }
                        }
                        IdleAction::Boot => self.boot_nro(),
                        IdleAction::Stop => self.stop_emulation(),
                        IdleAction::None => {}
                    }
                }
            });

        self.performance.record_frame();

        if self.show_settings {
            egui::Window::new("Settings")
                .open(&mut self.show_settings)
                .resizable(false)
                .default_width(320.0)
                .show(ctx, |ui| {
                    settings_panel(ui);
                });
        }

        if self.debugger.show_memory {
            egui::Window::new("Memory Viewer")
                .open(&mut self.debugger.show_memory)
                .default_size([480.0, 320.0])
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Address:").color(TEXT_DIM).size(12.0));
                        ui.label(RichText::new(format!("{:#018x}", self.debugger.memory_address))
                            .color(TEXT_BRIGHT).size(12.0).monospace());
                    });
                    ui.label(RichText::new("[Connect CPU state to read live memory]").color(TEXT_DIM).size(11.0));
                });
        }

        if self.debugger.show_registers {
            egui::Window::new("CPU Registers")
                .open(&mut self.debugger.show_registers)
                .default_size([300.0, 360.0])
                .show(ctx, |ui| {
                    register_row(ui, "PC", 0x0000_0000_0000_0000);
                    register_row(ui, "SP", 0x0000_0000_0000_0000);
                    ui.separator();
                    for i in 0..=8u32 {
                        register_row(ui, &format!("X{}", i), 0);
                    }
                });
        }

        if self.debugger.show_disasm {
            egui::Window::new("Disassembler")
                .open(&mut self.debugger.show_disasm)
                .default_size([440.0, 280.0])
                .show(ctx, |ui| {
                    ui.label(RichText::new("[Connect CPU state to disassemble at PC]").color(TEXT_DIM).size(11.0));
                });
        }

        if self.debugger.show_logs {
            egui::Window::new("Log Viewer")
                .open(&mut self.debugger.show_logs)
                .default_size([600.0, 320.0])
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ghost_button(ui, "Clear").clicked() {
                            self.debugger.log_history.lock().clear();
                        }
                    });
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for entry in self.debugger.log_history.lock().iter() {
                                ui.label(RichText::new(entry).size(11.0).monospace().color(TEXT_DIM));
                            }
                        });
                });
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

enum IdleAction {
    None,
    SelectFile,
    Boot,
    Stop,
}

fn draw_idle_screen(ui: &mut egui::Ui, nro_path: &str, running: bool) -> IdleAction {
    let avail = ui.available_size();
    let mut action = IdleAction::None;

    ui.vertical_centered(|ui| {
        ui.add_space(avail.y * 0.22);

        ui.label(RichText::new("HorizonRust").size(32.0).strong().color(TEXT_BRIGHT));
        ui.add_space(4.0);
        ui.label(RichText::new("Nintendo Switch Emulator").size(13.0).color(TEXT_DIM));

        ui.add_space(40.0);

        egui::Frame::none()
            .fill(BG_CARD)
            .stroke(Stroke::new(1.0, BORDER))
            .rounding(Rounding::same(8.0))
            .inner_margin(egui::Margin::same(20.0))
            .show(ui, |ui| {
                ui.set_width(360.0);

                if nro_path.is_empty() {
                    ui.label(RichText::new("No file selected").color(TEXT_DIM).size(12.0));
                } else {
                    let name = std::path::Path::new(nro_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| nro_path.to_string());
                    ui.label(RichText::new(&name).color(TEXT_BRIGHT).size(13.0).strong());
                    ui.label(RichText::new(nro_path).color(TEXT_DIM).size(10.0));
                }

                ui.add_space(12.0);

                ui.horizontal(|ui| {
                    if ghost_button(ui, "Select NRO…").clicked() {
                        action = IdleAction::SelectFile;
                    }
                    ui.add_space(8.0);
                    if !nro_path.is_empty() && !running {
                        if accent_button(ui, "Boot").clicked() {
                            action = IdleAction::Boot;
                        }
                    }
                    if running {
                        if ghost_button(ui, "Stop").clicked() {
                            action = IdleAction::Stop;
                        }
                    }
                });

                if running {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("●").color(Color32::from_rgb(0x4C, 0xD9, 0x64)).size(10.0));
                        ui.label(RichText::new("Emulation running — waiting for first frame").color(TEXT_DIM).size(12.0));
                    });
                }
            });
    });

    action
}

fn settings_panel(ui: &mut egui::Ui) {
    egui::Grid::new("settings_grid").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
        ui.label(RichText::new("CPU Backend").color(TEXT_DIM).size(12.0));
        ui.label(RichText::new("Dynarmic").color(TEXT_BRIGHT).size(12.0));
        ui.end_row();

        ui.label(RichText::new("GPU Backend").color(TEXT_DIM).size(12.0));
        ui.label(RichText::new("Vulkan").color(TEXT_BRIGHT).size(12.0));
        ui.end_row();

        ui.label(RichText::new("Audio").color(TEXT_DIM).size(12.0));
        ui.label(RichText::new("100%").color(TEXT_BRIGHT).size(12.0));
        ui.end_row();
    });
}

fn register_row(ui: &mut egui::Ui, name: &str, value: u64) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{:<4}", name)).color(TEXT_DIM).size(12.0).monospace());
        ui.label(RichText::new(format!("{:#018x}", value)).color(TEXT_BRIGHT).size(12.0).monospace());
    });
}
