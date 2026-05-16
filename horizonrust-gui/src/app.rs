use eframe::egui;
use eframe::egui::{
    Color32, FontId, Rounding, Sense, Stroke, Vec2,
};
use gilrs::Gilrs;
use crate::boot::EmulationHandle;
use crate::input::InputSnapshot;
use crate::debugger::DebuggerState;
use crate::performance::PerformanceMonitor;

const BG:        Color32 = Color32::from_rgb(0x0F, 0x0F, 0x11);
const BG_RAISED: Color32 = Color32::from_rgb(0x18, 0x18, 0x1C);
const BG_INPUT:  Color32 = Color32::from_rgb(0x20, 0x20, 0x26);
const BORDER:    Color32 = Color32::from_rgb(0x2A, 0x2A, 0x32);
const ACCENT:    Color32 = Color32::from_rgb(0xE0, 0x2A, 0x2A);
const ACCENT_HV: Color32 = Color32::from_rgb(0xF0, 0x3C, 0x3C);
const TEXT:      Color32 = Color32::from_rgb(0xEC, 0xEC, 0xF0);
const MUTED:     Color32 = Color32::from_rgb(0x70, 0x70, 0x80);
const GREEN:     Color32 = Color32::from_rgb(0x3C, 0xD4, 0x5C);
const AMBER:     Color32 = Color32::from_rgb(0xF5, 0xA6, 0x23);

pub struct HorizonApp {
    nro_path: String,
    emulation_handle: Option<EmulationHandle>,
    game_texture: Option<egui::TextureHandle>,
    show_settings: bool,
    gilrs: Option<Gilrs>,
    last_input: InputSnapshot,
    debugger: DebuggerState,
    performance: PerformanceMonitor,
    log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
}

impl HorizonApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
        nro_arg: Option<String>,
    ) -> Self {
        Self::apply_theme(&cc.egui_ctx);
        let gilrs = Gilrs::new().ok().or_else(|| { log::warn!("Gilrs init failed"); None });
        let nro_path = nro_arg.unwrap_or_default();
        let mut app = Self {
            nro_path: nro_path.clone(),
            emulation_handle: None,
            game_texture: None,
            show_settings: false,
            gilrs,
            last_input: InputSnapshot::default(),
            debugger: DebuggerState::new(),
            performance: PerformanceMonitor::new(),
            log_buffer,
        };
        if !nro_path.is_empty() {
            if let Ok(handle) = EmulationHandle::new(&nro_path) {
                app.emulation_handle = Some(handle);
                log::info!("Auto-loaded NRO: {}", nro_path);
            } else {
                log::error!("Failed to load NRO: {}", nro_path);
            }
        }
        app
    }

    fn apply_theme(ctx: &egui::Context) {
        let mut s = (*ctx.style()).clone();
        s.visuals.dark_mode = true;
        s.visuals.panel_fill = BG;
        s.visuals.window_fill = BG_RAISED;
        s.visuals.faint_bg_color = BG_RAISED;
        s.visuals.extreme_bg_color = BG;
        s.visuals.override_text_color = Some(TEXT);
        s.visuals.window_stroke = Stroke::new(1.0, BORDER);
        s.visuals.window_rounding = Rounding::same(8.0);
        s.visuals.menu_rounding = Rounding::same(6.0);
        s.visuals.popup_shadow = egui::epaint::Shadow {
            offset: Vec2::new(0.0, 6.0), blur: 16.0, spread: 0.0,
            color: Color32::from_black_alpha(100),
        };
        for w in [
            &mut s.visuals.widgets.noninteractive,
            &mut s.visuals.widgets.inactive,
            &mut s.visuals.widgets.hovered,
            &mut s.visuals.widgets.active,
            &mut s.visuals.widgets.open,
        ] {
            w.rounding = Rounding::same(4.0);
        }
        s.visuals.widgets.noninteractive.bg_fill = BG_RAISED;
        s.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        s.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, MUTED);
        s.visuals.widgets.inactive.bg_fill = BG_INPUT;
        s.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
        s.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        s.visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x28, 0x28, 0x30);
        s.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x44, 0x44, 0x52));
        s.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
        s.visuals.widgets.active.bg_fill = ACCENT;
        s.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
        s.visuals.widgets.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
        s.visuals.widgets.open.bg_fill = BG_INPUT;
        s.visuals.widgets.open.bg_stroke = Stroke::new(1.0, ACCENT);
        s.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(0xE0, 0x2A, 0x2A, 0x50);
        s.spacing.item_spacing = Vec2::new(6.0, 4.0);
        s.spacing.button_padding = Vec2::new(10.0, 5.0);
        s.spacing.menu_margin = egui::Margin::same(6.0);
        s.spacing.window_margin = egui::Margin::same(12.0);
        ctx.set_style(s);
    }

    fn poll_frames(&mut self, ctx: &egui::Context) {
        let Some(handle) = &self.emulation_handle else { return };
        while let Ok(frame) = handle.frame_rx.try_recv() {
            if frame.width == 0 || frame.height == 0 || frame.pixels.is_empty() { continue; }
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize], &frame.pixels,
            );
            match &mut self.game_texture {
                Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                None => {
                    self.game_texture = Some(
                        ctx.load_texture("game_frame", img, egui::TextureOptions::NEAREST)
                    );
                }
            }
        }
    }

    fn boot_nro(&mut self) {
        if self.nro_path.is_empty() { return; }
        match EmulationHandle::new(&self.nro_path) {
            Ok(h) => { self.emulation_handle = Some(h); self.game_texture = None; }
            Err(e) => log::error!("Boot: {}", e),
        }
    }

    fn stop_emulation(&mut self) {
        if let Some(mut h) = self.emulation_handle.take() { h.stop(); }
    }

    fn is_running(&self) -> bool {
        self.emulation_handle.as_ref().map_or(false, |h| h.is_running())
    }
}

fn pill_button(ui: &mut egui::Ui, label: &str, filled: bool) -> egui::Response {
    let font = FontId::proportional(12.5);
    let text_w = ui.fonts(|f| f.layout_no_wrap(label.to_string(), font.clone(), TEXT).size().x);
    let size = Vec2::new(text_w + 24.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    let (bg, text_col, stroke) = if filled {
        let c = if resp.is_pointer_button_down_on() { Color32::from_rgb(0xBC, 0x20, 0x20) }
                else if resp.hovered() { ACCENT_HV } else { ACCENT };
        (c, Color32::WHITE, Stroke::NONE)
    } else {
        let c = if resp.hovered() { Color32::from_rgb(0x28, 0x28, 0x30) } else { Color32::TRANSPARENT };
        let bc = if resp.hovered() { Color32::from_rgb(0x50, 0x50, 0x60) } else { BORDER };
        (c, TEXT, Stroke::new(1.0, bc))
    };

    ui.painter().rect(rect, Rounding::same(5.0), bg, stroke);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, label, font, text_col);
    resp
}

impl eframe::App for HorizonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref mut g) = self.gilrs {
            self.last_input = InputSnapshot::update_from_gamepad(g);
        }
        self.poll_frames(ctx);

        if self.emulation_handle.as_ref().map_or(false, |h| !h.is_running()) {
            self.emulation_handle = None;
        }

        let fps = self.performance.get_fps();
        let running = self.is_running();

        egui::TopBottomPanel::top("topbar")
            .exact_height(36.0)
            .frame(egui::Frame::none()
                .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(12.0);

                    ui.label(egui::RichText::new("HorizonRust")
                        .size(13.5).strong().color(TEXT));

                    ui.add_space(8.0);
                    ui.painter().vline(
                        ui.cursor().left(), ui.max_rect().y_range(),
                        Stroke::new(1.0, BORDER),
                    );
                    ui.add_space(8.0);

                    ui.menu_button(egui::RichText::new("File").size(13.0).color(TEXT), |ui| {
                        if ui.button("Open NRO…").clicked() {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("Nintendo Homebrew", &["nro"]).pick_file() {
                                self.nro_path = p.to_string_lossy().to_string();
                            }
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Exit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.menu_button(egui::RichText::new("Emulation").size(13.0).color(TEXT), |ui| {
                        if ui.button("Boot").clicked() { self.boot_nro(); ui.close_menu(); }
                        if ui.button("Stop").clicked() { self.stop_emulation(); ui.close_menu(); }
                    });
                    ui.menu_button(egui::RichText::new("Debug").size(13.0).color(TEXT), |ui| {
                        if ui.button("Memory").clicked()    { self.debugger.toggle_memory(); ui.close_menu(); }
                        if ui.button("Registers").clicked() { self.debugger.toggle_registers(); ui.close_menu(); }
                        if ui.button("Disassembler").clicked() { self.debugger.toggle_disasm(); ui.close_menu(); }
                        if ui.button("Logs").clicked()      { self.debugger.toggle_logs(); ui.close_menu(); }
                    });
                    ui.menu_button(egui::RichText::new("Settings").size(13.0).color(TEXT), |ui| {
                        if ui.button("Preferences").clicked() { self.show_settings = !self.show_settings; ui.close_menu(); }
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(12.0);
                        let (fps_col, fps_str) = if fps >= 55.0 { (GREEN, format!("{:.0} fps", fps)) }
                            else if fps >= 28.0 { (AMBER, format!("{:.0} fps", fps)) }
                            else { (Color32::from_rgb(0xE0, 0x40, 0x40), format!("{:.0} fps", fps)) };
                        ui.label(egui::RichText::new(fps_str).size(12.0).color(fps_col).monospace());
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                        ui.add_space(6.0);
                        let (dot, status_col) = if running { ("●", GREEN) } else { ("○", MUTED) };
                        let status = if running { "Running" } else { "Idle" };
                        ui.label(egui::RichText::new(format!("{} {}", dot, status))
                            .size(12.0).color(status_col));
                    });
                });
            });

        egui::TopBottomPanel::bottom("statusbar")
            .exact_height(22.0)
            .frame(egui::Frame::none()
                .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(12.0);
                    let name = std::path::Path::new(&self.nro_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "No file".into());
                    ui.label(egui::RichText::new(name).size(11.0).color(MUTED));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(12.0);
                        ui.label(egui::RichText::new(format!(
                            "Frame {:.1}ms  ·  SVCs {}  ·  Cycles {}",
                            self.performance.get_frame_time(),
                            self.performance.get_svc_count(),
                            self.performance.get_cycle_count(),
                        )).size(11.0).color(MUTED).monospace());
                    });
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG))
            .show(ctx, |ui| {
                if let Some(tex) = &self.game_texture {
                    let avail = ui.available_size();
                    let tsz = tex.size_vec2();
                    let scale = (avail.x / tsz.x).min(avail.y / tsz.y);
                    ui.centered_and_justified(|ui| {
                        ui.image((tex.id(), tsz * scale));
                    });
                } else {
                    let nro_path = self.nro_path.clone();
                    let running = self.is_running();
                    let action = idle_screen(ui, &nro_path, running);
                    match action {
                        0 => { if let Some(p) = rfd::FileDialog::new()
                                   .add_filter("Nintendo Homebrew", &["nro"]).pick_file() {
                                   self.nro_path = p.to_string_lossy().to_string();
                               } }
                        1 => self.boot_nro(),
                        2 => self.stop_emulation(),
                        _ => {}
                    }
                }
            });

        self.performance.record_frame();

        if self.show_settings {
            egui::Window::new("Preferences")
                .open(&mut self.show_settings)
                .resizable(false).default_width(280.0)
                .show(ctx, |ui| { settings_content(ui); });
        }

        debug_windows(ctx, &mut self.debugger, &self.log_buffer);

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn idle_screen(ui: &mut egui::Ui, nro_path: &str, running: bool) -> u8 {
    let mut action = 255u8;
    let avail = ui.available_size();

    ui.vertical_centered(|ui| {
        ui.add_space((avail.y * 0.24).max(40.0));

        ui.label(egui::RichText::new("HorizonRust")
            .size(36.0).strong().color(TEXT));
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Nintendo Switch Emulator")
            .size(13.0).color(MUTED));

        ui.add_space(36.0);

        let painter = ui.painter();
        let card_w = 380.0;
        let card_rect = egui::Rect::from_center_size(
            egui::pos2(ui.max_rect().center().x, ui.cursor().top() + 80.0),
            egui::vec2(card_w, 160.0),
        );
        painter.rect(card_rect, Rounding::same(10.0), BG_RAISED, Stroke::new(1.0, BORDER));

        ui.allocate_ui_with_layout(
            egui::vec2(card_w, 160.0),
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                ui.add_space(20.0);

                if nro_path.is_empty() {
                    ui.label(egui::RichText::new("No file selected")
                        .size(13.0).color(MUTED));
                } else {
                    let fname = std::path::Path::new(nro_path)
                        .file_name().map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| nro_path.to_string());
                    ui.label(egui::RichText::new(&fname).size(14.0).strong().color(TEXT));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(nro_path).size(10.5).color(MUTED));
                }

                ui.add_space(16.0);

                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    if pill_button(ui, "Select NRO…", false).clicked() { action = 0; }
                    ui.add_space(8.0);
                    if !nro_path.is_empty() && !running {
                        if pill_button(ui, "Boot", true).clicked() { action = 1; }
                    }
                    if running {
                        if pill_button(ui, "Stop", false).clicked() { action = 2; }
                    }
                });

                if running {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 8.0), Sense::hover());
                        ui.painter().circle_filled(rect.center(), 3.5, GREEN);
                        ui.label(egui::RichText::new("Running — awaiting first frame")
                            .size(11.5).color(MUTED));
                    });
                }
            },
        );
    });

    action
}

fn settings_content(ui: &mut egui::Ui) {
    egui::Grid::new("prefs").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
        row(ui, "CPU Backend", "Dynarmic (JIT)");
        row(ui, "GPU Backend", "Vulkan (ash)");
        row(ui, "Audio", "Enabled · 100%");
        row(ui, "Resolution", "1280 × 720");
    });
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).size(12.0).color(MUTED));
    ui.label(egui::RichText::new(value).size(12.0).color(TEXT));
    ui.end_row();
}

fn debug_windows(
    ctx: &egui::Context,
    dbg: &mut DebuggerState,
    log_buffer: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
) {
    if dbg.show_memory {
        egui::Window::new("Memory").open(&mut dbg.show_memory)
            .default_size([480.0, 300.0]).show(ctx, |ui| {
            ui.label(egui::RichText::new(format!("Address  {:#018x}", dbg.memory_address))
                .size(12.0).monospace().color(MUTED));
            ui.label(egui::RichText::new("Connect CPU state to read live memory")
                .size(11.0).color(MUTED));
        });
    }
    if dbg.show_registers {
        egui::Window::new("Registers").open(&mut dbg.show_registers)
            .default_size([280.0, 340.0]).show(ctx, |ui| {
            for (n, v) in [("PC","0x0"),("SP","0x0"),("X0","0x0"),("X1","0x0"),("X2","0x0")] {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{:<4}", n)).size(12.0).monospace().color(MUTED));
                    ui.label(egui::RichText::new(v).size(12.0).monospace().color(TEXT));
                });
            }
        });
    }
    if dbg.show_disasm {
        egui::Window::new("Disassembler").open(&mut dbg.show_disasm)
            .default_size([440.0, 260.0]).show(ctx, |ui| {
            ui.label(egui::RichText::new("Connect CPU state to disassemble at PC")
                .size(11.0).color(MUTED));
        });
    }
    if dbg.show_logs {
        egui::Window::new("Logs").open(&mut dbg.show_logs)
            .default_size([580.0, 300.0]).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if pill_button(ui, "Clear", false).clicked() {
                    if let Ok(mut buf) = log_buffer.lock() {
                        buf.clear();
                    }
                }
            });
            ui.add_space(4.0);
            egui::ScrollArea::vertical().auto_shrink([false;2]).stick_to_bottom(true)
                .show(ui, |ui| {
                    if let Ok(buf) = log_buffer.lock() {
                        for entry in buf.iter() {
                            ui.label(egui::RichText::new(entry).size(11.0).monospace().color(MUTED));
                        }
                    }
                });
        });
    }
}
