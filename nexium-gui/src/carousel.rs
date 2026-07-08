use eframe::egui;
use eframe::egui::{Color32, FontId, Rounding, Sense, Stroke, Vec2};
use crate::library::{GameEntry, Library};

#[derive(Clone, Debug, PartialEq)]
pub enum BootStage {
    None,
    Transitioning {
        game_index: usize,
        start_time: f32,
        launch_path: String,
    },
    AwaitingFrame {
        game_index: usize,
        start_time: f32,
    },
}

pub struct CarouselState {
    pub selected: usize,
    pub scroll_offset: f32,
    pub hover_scale: f32,
    pub ambient_color: Color32,
    pub active_dock: bool,
    pub dock_selected: usize,
    pub is_dragging: bool,
    pub drag_start_x: f32,
    pub drag_start_offset: f32,
    pub drag_moved: bool,
    pub boot_stage: BootStage,
    pub palette_open: bool,
    pub palette_selected: usize,
}

impl CarouselState {
    pub fn new() -> Self {
        Self {
            selected: 0,
            scroll_offset: 0.0,
            hover_scale: 1.0,
            ambient_color: Color32::from_rgb(0x2F, 0xB4, 0xEF),
            active_dock: false,
            dock_selected: 0,
            is_dragging: false,
            drag_start_x: 0.0,
            drag_start_offset: 0.0,
            drag_moved: false,
            boot_stage: BootStage::None,
            palette_open: false,
            palette_selected: 0,
        }
    }
}

pub enum CarouselAction {
    None,
    Launch(String),
    Resume,
    OpenSettings,
    OpenController,
    OpenDebug,
    StopEmulation,
    SwitchToGrid,
    Quit,
    AddFolder,
    Rescan,
    SetTheme(crate::app_settings::CarouselTheme),
}

const DOCK_COUNT: usize = 9;

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

pub fn shadowed_text(
    painter: &egui::Painter,
    pos: egui::Pos2,
    align: egui::Align2,
    text: &str,
    font: FontId,
    color: Color32,
    bold: bool,
) {
    let alpha_pct = color.a() as f32 / 255.0;
    let outline_col = Color32::from_rgba_unmultiplied(0, 0, 0, (240.0 * alpha_pct) as u8);
    let offsets = [-1.5f32, 0.0f32, 1.5f32];
    for &dx in &offsets {
        for &dy in &offsets {
            if dx != 0.0 || dy != 0.0 {
                painter.text(pos + Vec2::new(dx, dy), align, text, font.clone(), outline_col);
                if bold {
                    painter.text(pos + Vec2::new(dx + 0.8, dy), align, text, font.clone(), outline_col);
                }
            }
        }
    }
    painter.text(pos, align, text, font.clone(), color);
    if bold {
        painter.text(pos + Vec2::new(0.8, 0.0), align, text, font, color);
    }
}

pub fn draw_rounded_image(
    painter: &egui::Painter,
    tex_id: egui::TextureId,
    rect: egui::Rect,
    r: f32,
    tint: Color32,
) {
    let mut mesh = egui::epaint::Mesh::with_texture(tex_id);
    
    // Center vertex
    let center = rect.center();
    mesh.vertices.push(egui::epaint::Vertex {
        pos: center,
        uv: egui::pos2(0.5, 0.5),
        color: tint,
    });
    
    let x0 = rect.min.x;
    let y0 = rect.min.y;
    let x1 = rect.max.x;
    let y1 = rect.max.y;
    
    let mut pts = Vec::new();
    
    let corners = [
        (egui::pos2(x1 - r, y0 + r), -std::f32::consts::FRAC_PI_2, 0.0f32),
        (egui::pos2(x1 - r, y1 - r), 0.0f32, std::f32::consts::FRAC_PI_2),
        (egui::pos2(x0 + r, y1 - r), std::f32::consts::FRAC_PI_2, std::f32::consts::PI),
        (egui::pos2(x0 + r, y0 + r), std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2),
    ];
    
    let steps_per_corner = 8;
    for &(c_center, start_angle, end_angle) in &corners {
        for s in 0..=steps_per_corner {
            let angle = start_angle + (s as f32 / steps_per_corner as f32) * (end_angle - start_angle);
            pts.push(c_center + Vec2::new(angle.cos(), angle.sin()) * r);
        }
    }
    
    let w = rect.width();
    let h = rect.height();
    
    for pt in pts {
        let uv_x = if w > 0.0 { (pt.x - x0) / w } else { 0.0 };
        let uv_y = if h > 0.0 { (pt.y - y0) / h } else { 0.0 };
        
        mesh.vertices.push(egui::epaint::Vertex {
            pos: pt,
            uv: egui::pos2(uv_x, uv_y),
            color: tint,
        });
    }
    
    let n_vertices = mesh.vertices.len() as u32;
    for i in 1..(n_vertices - 1) {
        mesh.indices.push(0);
        mesh.indices.push(i);
        mesh.indices.push(i + 1);
    }
    mesh.indices.push(0);
    mesh.indices.push(n_vertices - 1);
    mesh.indices.push(1);
    
    painter.add(egui::Shape::mesh(mesh));
}

pub fn draw_gradient_rounded_rect(
    painter: &egui::Painter,
    center: egui::Pos2,
    rect: egui::Rect,
    r: f32,
    t: f32,
    alpha: u8,
) {
    let mut mesh = egui::epaint::Mesh::default();
    
    mesh.vertices.push(egui::epaint::Vertex {
        pos: center,
        uv: egui::pos2(0.0, 0.0),
        color: Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
    });
    
    let x0 = rect.min.x;
    let y0 = rect.min.y;
    let x1 = rect.max.x;
    let y1 = rect.max.y;
    
    let mut pts = Vec::new();
    
    let corners = [
        (egui::pos2(x1 - r, y0 + r), -std::f32::consts::FRAC_PI_2, 0.0f32),
        (egui::pos2(x1 - r, y1 - r), 0.0f32, std::f32::consts::FRAC_PI_2),
        (egui::pos2(x0 + r, y1 - r), std::f32::consts::FRAC_PI_2, std::f32::consts::PI),
        (egui::pos2(x0 + r, y0 + r), std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2),
    ];
    
    let steps_per_corner = 8;
    for &(c_center, start_angle, end_angle) in &corners {
        for s in 0..=steps_per_corner {
            let angle = start_angle + (s as f32 / steps_per_corner as f32) * (end_angle - start_angle);
            pts.push(c_center + Vec2::new(angle.cos(), angle.sin()) * r);
        }
    }
    
    let min_y = rect.min.y;
    let max_y = rect.max.y;
    let min_x = rect.min.x;
    let max_x = rect.max.x;
    let h = max_y - min_y;
    let w = max_x - min_x;
    
    let center_dx = if w > 0.0 { (center.x - min_x) / w - 0.5 } else { 0.0 };
    let center_dy = if h > 0.0 { (center.y - min_y) / h - 0.5 } else { 0.0 };
    let center_angle = center_dy.atan2(center_dx);
    let center_phase = center_angle + t * 1.5;
    let cr = ((center_phase.sin() * 0.45 + 0.55) * 255.0) as u8;
    let cg = (((center_phase + 2.09).sin() * 0.35 + 0.65) * 255.0) as u8;
    mesh.vertices[0].color = Color32::from_rgba_unmultiplied(cr, cg, 255, alpha);

    for pt in pts {
        let tx = if w > 0.0 { (pt.x - min_x) / w } else { 0.0 };
        let ty = if h > 0.0 { (pt.y - min_y) / h } else { 0.0 };
        
        let dx = tx - 0.5;
        let dy = ty - 0.5;
        let angle = dy.atan2(dx);
        let phase = angle + t * 1.5;
        
        let r_val = ((phase.sin() * 0.45 + 0.55) * 255.0).clamp(0.0, 255.0) as u8;
        let g_val = (((phase + 2.09).sin() * 0.35 + 0.65) * 255.0).clamp(0.0, 255.0) as u8;
        
        mesh.vertices.push(egui::epaint::Vertex {
            pos: pt,
            uv: egui::pos2(0.0, 0.0),
            color: Color32::from_rgba_unmultiplied(r_val, g_val, 255, alpha),
        });
    }
    
    let n_vertices = mesh.vertices.len() as u32;
    for i in 1..(n_vertices - 1) {
        mesh.indices.push(0);
        mesh.indices.push(i);
        mesh.indices.push(i + 1);
    }
    mesh.indices.push(0);
    mesh.indices.push(n_vertices - 1);
    mesh.indices.push(1);
    
    painter.add(egui::Shape::mesh(mesh));
}


fn draw_gradient_circle(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    border_width: f32,
    t: f32,
    alpha: u8,
) {
    let mut mesh = egui::epaint::Mesh::default();
    mesh.vertices.push(egui::epaint::Vertex {
        pos: center,
        uv: egui::pos2(0.0, 0.0),
        color: Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
    });
    let n_points = 32;
    let outer_r = radius + border_width;
    for i in 0..=n_points {
        let angle = (i as f32 / n_points as f32) * std::f32::consts::TAU;
        let pos = center + Vec2::new(angle.cos(), angle.sin()) * outer_r;
        
        let phase = angle + t * 1.5;
        let r = ((phase.sin() * 0.45 + 0.55) * 255.0).clamp(0.0, 255.0) as u8;
        let g = (((phase + 2.09).sin() * 0.35 + 0.65) * 255.0).clamp(0.0, 255.0) as u8;
        
        mesh.vertices.push(egui::epaint::Vertex {
            pos,
            uv: egui::pos2(0.0, 0.0),
            color: Color32::from_rgba_unmultiplied(r, g, 255, alpha),
        });
    }
    for i in 1..=n_points {
        mesh.indices.push(0);
        mesh.indices.push(i as u32);
        mesh.indices.push((i + 1) as u32);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn draw_wave_background(
    painter: &egui::Painter,
    rect: egui::Rect,
    color: Color32,
    t: f32,
    _glow_cx: f32,
    _glow_cy: f32,
    opacity: f32,
) {
    if opacity <= 0.001 { return; }
    let bg_alpha = (opacity * 255.0) as u8;
    painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(0x08, 0x08, 0x0C, bg_alpha));

    let w = rect.width();
    let h = rect.height();
    let steps = 128usize;

    let bands: &[(f32, f32, f32, f32, u8)] = &[
        (0.55, 0.012, 0.0,  1.5,  18),
        (0.42, 0.018, 1.1,  1.2,  14),
        (0.70, 0.009, 2.3,  0.9,  10),
        (0.30, 0.022, 0.6,  1.7,   8),
        (0.85, 0.007, 3.5,  0.7,   6),
    ];

    for &(base_frac, amp_frac, phase_off, speed, alpha) in bands {
        let base_y = rect.min.y + h * base_frac;
        let amp    = h * amp_frac;
        let phase  = t * speed + phase_off;

        let mut pts: Vec<egui::Pos2> = Vec::with_capacity(steps + 4);

        pts.push(egui::pos2(rect.min.x, rect.max.y));
        pts.push(egui::pos2(rect.max.x, rect.max.y));

        for s in (0..=steps).rev() {
            let x  = rect.min.x + (s as f32 / steps as f32) * w;
            let nx = s as f32 / steps as f32;
            let y  = base_y
                + (nx * std::f32::consts::TAU + phase).sin() * amp
                + (nx * std::f32::consts::TAU * 1.7 + phase * 0.8).cos() * amp * 0.4;
            pts.push(egui::pos2(x, y));
        }

        let band_alpha = ((alpha as f32) * opacity) as u8;
        if band_alpha == 0 { continue; }

        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), band_alpha);
        painter.add(egui::Shape::convex_polygon(pts.clone(), fill, Stroke::NONE));

        if pts.len() >= 2 {
            let crest_pts: Vec<egui::Pos2> = pts[2..].to_vec();
            let outer_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (band_alpha / 2).max(1));
            painter.add(egui::Shape::line(crest_pts.clone(), Stroke::new(2.0, outer_col)));
            let inner_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), band_alpha);
            painter.add(egui::Shape::line(crest_pts, Stroke::new(1.0, inner_col)));
        }
    }
}

pub fn carousel_view(
    state: &mut CarouselState,
    lib: &mut Library,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    last_input: &crate::input::InputSnapshot,
    is_running: bool,
    playing: Option<usize>,
    playing_alpha: f32,
    theme: crate::app_settings::CarouselTheme,
    scale_factor: f32,
    alpha_factor: f32,
) -> CarouselAction {
    let mut action = CarouselAction::None;
    let n_games = lib.games.len();

    let bg_rect = ui.max_rect();
    let t = ui.input(|i| i.time) as f32;
    let dt = ui.input(|i| i.stable_dt).min(0.1);

    let screen_height = bg_rect.height();
    let hero_size = (screen_height * 0.35).clamp(260.0, 480.0);
    let stride = hero_size + (bg_rect.width() * 0.022).clamp(14.0, 36.0);

    let hero_cx   = bg_rect.min.x + bg_rect.width() * 0.22;
    let hero_cy   = bg_rect.min.y + bg_rect.height() * 0.44;

    if n_games == 0 {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("carousel_bg")));
        draw_wave_background(&painter, bg_rect, state.ambient_color, t, hero_cx, hero_cy, 1.0);
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("No games found").size(22.0).color(Color32::WHITE));
                ui.add_space(14.0);
                if ui.button("Add folder").clicked() {
                    action = CarouselAction::AddFolder;
                }
            });
        });
        ctx.request_repaint();
        return action;
    }

    let ui_opacity = if let BootStage::Transitioning { start_time, .. } = state.boot_stage {
        ((1.0 - (t - start_time) / 0.35).clamp(0.0, 1.0)) * alpha_factor
    } else {
        alpha_factor
    };

    let pointer_pos = ui.input(|i| i.pointer.hover_pos());
    let pointer_down = ui.input(|i| i.pointer.any_down());
    let pointer_pressed = ui.input(|i| i.pointer.any_pressed());
    let pointer_released = ui.input(|i| i.pointer.any_released());
    let dock_y = bg_rect.max.y - 112.0;

    if state.boot_stage == BootStage::None && !state.palette_open {
        if pointer_pressed {
            if let Some(pos) = pointer_pos {
                if pos.y < dock_y {
                    state.is_dragging = true;
                    state.drag_start_x = pos.x;
                    state.drag_start_offset = state.scroll_offset;
                    state.drag_moved = false;
                }
            }
        }

        if state.is_dragging && pointer_down {
            if let Some(pos) = pointer_pos {
                let delta_x = pos.x - state.drag_start_x;
                if delta_x.abs() > 8.0 {
                    state.drag_moved = true;
                }
                state.scroll_offset = (state.drag_start_offset - delta_x / stride).clamp(0.0, (n_games.saturating_sub(1)) as f32);
                
                let nearest = state.scroll_offset.round().clamp(0.0, (n_games.saturating_sub(1)) as f32) as usize;
                if nearest != state.selected {
                    state.selected = nearest;
                    state.active_dock = false;
                }
            }
        }

        if pointer_released || !pointer_down {
            if state.is_dragging {
                state.is_dragging = false;
                state.selected = state.scroll_offset.round().clamp(0.0, (n_games.saturating_sub(1)) as f32) as usize;
            }
        }
    }

    if state.selected >= n_games {
        state.selected = n_games - 1;
    }

    if state.is_dragging {
        // Controlled directly by drag
    } else if state.boot_stage == BootStage::None {
        state.scroll_offset += (state.selected as f32 - state.scroll_offset) * (dt * 11.0).min(1.0);
    }

    let target_color = match theme.color() {
        Some((r, g, b)) => Color32::from_rgb(r, g, b),
        None => lib.games[state.selected].dominant_color,
    };
    state.ambient_color = lerp_color(state.ambient_color, target_color, (dt * 10.0).min(1.0));

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("carousel_bg")));
    draw_wave_background(&painter, bg_rect, state.ambient_color, t, hero_cx, hero_cy, ui_opacity);

    let screen_center = bg_rect.center();

    for i in 0..n_games {
        let diff     = i as f32 - state.scroll_offset;
        let abs_diff = diff.abs();
        if abs_diff > 4.5 && state.boot_stage == BootStage::None { continue; }

        let scale   = (1.0 - abs_diff * 0.18).max(0.38);
        let mut alpha_f = (1.0 - abs_diff * 0.38).max(0.12).min(1.0);
        let sz      = hero_size * scale;
        let cx      = hero_cx + diff * stride;
        let cy      = hero_cy + abs_diff * 18.0;

        let mut draw_cx = cx;
        let mut draw_cy = cy;
        let mut draw_sz = sz;
        let mut scale_x = 1.0f32;
        let mut scale_y = 1.0f32;
        let mut glow_expansion = 6.0f32;
        let mut glow_alpha = 100u8;
        let mut border_alpha = 255u8;
        
        let is_hero = i == state.selected && !state.active_dock;

        if let BootStage::Transitioning { game_index, start_time, .. } = &state.boot_stage {
            let elapsed = t - *start_time;
            if i == *game_index {
                let land_end = 0.42f32;
                let zoom_end = 0.90f32;
                if elapsed < land_end {
                    let squish_end = 0.10f32;
                    if elapsed < squish_end {
                        let p = elapsed / squish_end;
                        let s = p * p * 0.18;
                        scale_y = 1.0 - s;
                        scale_x = 1.0 + s * 0.65;
                        glow_expansion = 6.0 + p * 12.0;
                        glow_alpha = (100.0 + p * 70.0) as u8;
                    } else {
                        let hop = (elapsed - squish_end) / (land_end - squish_end);
                        let arc = (hop * std::f32::consts::PI).sin();
                        draw_cy -= arc * hero_size * 0.17;
                        let stretch = arc * 0.13;
                        scale_y = 1.0 + stretch;
                        scale_x = 1.0 - stretch * 0.5;
                        if hop > 0.80 {
                            let land = (hop - 0.80) / 0.20;
                            let squash = (land * std::f32::consts::PI).sin() * 0.17;
                            scale_y = 1.0 - squash;
                            scale_x = 1.0 + squash * 0.7;
                        }
                        glow_expansion = 6.0 + arc * 36.0;
                        glow_alpha = (110.0 + arc * 145.0) as u8;
                    }
                } else {
                    let p = ((elapsed - land_end) / (zoom_end - land_end)).clamp(0.0, 1.0);
                    let ease = p * p * p;
                    let target_center = bg_rect.center();
                    draw_cx += (target_center.x - draw_cx) * ease;
                    draw_cy += (target_center.y - draw_cy) * ease;
                    draw_sz *= 1.0 + ease * 6.0;
                    glow_expansion = 6.0 + (1.0 - p) * 24.0;
                    glow_alpha = ((1.0 - p) * 150.0) as u8;
                    border_alpha = ((1.0 - p) * 255.0) as u8;
                }
            } else {
                let progress = (elapsed / 0.24).min(1.0);
                alpha_f *= 1.0 - progress;
            }
        }

        alpha_f *= alpha_factor;

        if alpha_f <= 0.001 { continue; }

        let card_scale_hover = if is_hero && state.boot_stage == BootStage::None {
            state.hover_scale
        } else {
            1.0
        };
        let final_sz = draw_sz * card_scale_hover * scale_factor;
        
        let card_center = egui::pos2(draw_cx, draw_cy);
        let scaled_center = screen_center + (card_center - screen_center) * scale_factor;

        let draw_rect = egui::Rect::from_center_size(
            scaled_center,
            Vec2::new(final_sz * scale_x, final_sz * scale_y)
        );

        painter.rect_filled(
            draw_rect.translate(Vec2::new(0.0, 6.0)),
            Rounding::same(14.0 * scale_factor),
            Color32::from_rgba_premultiplied(0, 0, 0, (120.0 * alpha_f) as u8),
        );

        if is_hero && border_alpha > 0 {
            draw_gradient_rounded_rect(&painter, draw_rect.center(), draw_rect.expand(glow_expansion * scale_factor), 16.0 * scale_factor, t, glow_alpha);
            draw_gradient_rounded_rect(&painter, draw_rect.center(), draw_rect.expand(2.5 * scale_factor), 14.0 * scale_factor, t, border_alpha);
        }

        let resp = ui.allocate_rect(draw_rect, Sense::click());
        if resp.clicked() && !state.drag_moved && state.boot_stage == BootStage::None {
            if state.selected == i {
                if state.active_dock {
                    state.active_dock = false;
                } else if playing == Some(i) {
                    action = CarouselAction::Resume;
                } else {
                    state.boot_stage = BootStage::Transitioning {
                        game_index: i,
                        start_time: t,
                        launch_path: lib.games[i].path.to_string_lossy().to_string(),
                    };
                }
            } else {
                state.selected = i;
                state.active_dock = false;
            }
        }

        let tint = Color32::from_white_alpha((alpha_f * 255.0) as u8);
        painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(0x14, 0x14, 0x1A, (alpha_f * 255.0) as u8));
        if let Some(tex) = lib.texture(ctx, i) {
            draw_rounded_image(&painter, tex.id(), draw_rect, 14.0 * scale_factor, tint);
        } else {
            painter.text(draw_rect.center(), egui::Align2::CENTER_CENTER, &lib.games[i].format, FontId::proportional(final_sz * 0.13), Color32::from_rgba_unmultiplied(0x70, 0x70, 0x80, (alpha_f * 255.0) as u8));
        }

        if playing == Some(i) && playing_alpha > 0.01 && state.boot_stage == BootStage::None {
            let pill_font = FontId::proportional((draw_rect.width() * 0.072).clamp(10.0, 20.0));
            let fh = pill_font.size;
            let label = "Playing";
            let text_w = ui.fonts(|f| {
                f.layout_no_wrap(label.to_string(), pill_font.clone(), Color32::WHITE)
                    .size()
                    .x
            });
            let dot_r = fh * 0.26;
            let pad_x = fh * 0.7;
            let gap = fh * 0.4;
            let pill_h = fh * 1.7;
            let pill_w = pad_x * 2.0 + dot_r * 2.0 + gap + text_w;
            let margin = draw_rect.width() * 0.055;
            let pill_rect = egui::Rect::from_min_size(
                egui::pos2(
                    draw_rect.max.x - margin - pill_w,
                    draw_rect.max.y - margin - pill_h,
                ),
                Vec2::new(pill_w, pill_h),
            );
            let pulse = 0.55 + 0.45 * (t * 3.0).sin();
            let base = Color32::from_rgb(0x35, 0xD0, 0x6A);
            let a = |x: f32| (x * alpha_f * playing_alpha).clamp(0.0, 255.0) as u8;
            painter.rect_filled(
                pill_rect.expand(2.5),
                Rounding::same(pill_h * 0.5 + 2.5),
                Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a(70.0 * pulse)),
            );
            painter.rect_filled(
                pill_rect,
                Rounding::same(pill_h * 0.5),
                Color32::from_rgba_unmultiplied(0x0C, 0x14, 0x0E, a(215.0)),
            );
            painter.rect_stroke(
                pill_rect,
                Rounding::same(pill_h * 0.5),
                Stroke::new(1.4, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a(235.0))),
            );
            let dot_c = egui::pos2(pill_rect.min.x + pad_x + dot_r, pill_rect.center().y);
            painter.circle_filled(dot_c, dot_r * (0.85 + 0.15 * pulse), Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a(255.0)));
            painter.text(
                egui::pos2(dot_c.x + dot_r + gap, pill_rect.center().y),
                egui::Align2::LEFT_CENTER,
                label,
                pill_font,
                Color32::from_rgba_unmultiplied(0xEA, 0xFF, 0xEE, a(255.0)),
            );
        }
    }

    if let BootStage::Transitioning { start_time, .. } = state.boot_stage {
        let elapsed = t - start_time;
        if elapsed >= 0.42 {
            let p = ((elapsed - 0.42) / (0.90 - 0.42)).min(1.0);
            let overlay_alpha = (p * p * 255.0) as u8;
            painter.rect_filled(bg_rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(0, 0, 0, overlay_alpha));
        }
    }

    let meta_x = hero_cx;
    let meta_y = hero_cy + hero_size * 0.52 + 36.0;
    let selected_game = &lib.games[state.selected];

    let title_col = if state.active_dock { Color32::from_rgb(0xC0, 0xC0, 0xCC) } else { Color32::WHITE };
    let title_alpha = (ui_opacity * title_col.a() as f32) as u8;
    if title_alpha > 0 {
        let meta_pos = egui::pos2(meta_x, meta_y);
        let scaled_meta_pos = screen_center + (meta_pos - screen_center) * scale_factor;
        shadowed_text(&painter, scaled_meta_pos, egui::Align2::CENTER_CENTER, &selected_game.title, FontId::proportional(28.0 * scale_factor), Color32::from_rgba_unmultiplied(title_col.r(), title_col.g(), title_col.b(), title_alpha), true);

        let sub = format!(
            "{}  ·  {}  ·  {:.1} MB",
            if selected_game.author.is_empty() { "Unknown" } else { &selected_game.author },
            selected_game.format,
            selected_game.size as f32 / (1024.0 * 1024.0),
        );
        let sub_alpha = (ui_opacity * 222.0) as u8;
        shadowed_text(&painter, scaled_meta_pos + Vec2::new(0.0, 32.0 * scale_factor), egui::Align2::CENTER_CENTER, &sub, FontId::proportional(14.0 * scale_factor), Color32::from_rgba_unmultiplied(0xD2, 0xD2, 0xDE, sub_alpha), false);
    }


    let dock_items: [(&str, &str); DOCK_COUNT] = [
        ("⊞", "Grid View"),
        ("↺", "Rescan"),
        ("▶", "Boot"),
        ("■", "Stop"),
        ("⌨", "Controller"),
        ("⚙", "Settings"),
        ("🎨", "Theme"),
        ("◈", "Debug"),
        ("⏻", "Quit"),
    ];

    let item_size  = 52.0f32;
    let dock_gap   = 14.0f32;
    let dock_total = dock_items.len() as f32 * item_size + (dock_items.len() - 1) as f32 * dock_gap;
    let dock_cx    = bg_rect.center().x;
    let dock_y     = bg_rect.max.y - 112.0;

    let dock_center = egui::pos2(dock_cx, dock_y + item_size * 0.5);
    let scaled_dock_center = screen_center + (dock_center - screen_center) * scale_factor;
    let scaled_item_size = item_size * scale_factor;
    let scaled_dock_gap = dock_gap * scale_factor;
    let scaled_dock_total = dock_total * scale_factor;

    let dock_bg = egui::Rect::from_center_size(
        scaled_dock_center,
        Vec2::new(scaled_dock_total + 44.0 * scale_factor, scaled_item_size + 22.0 * scale_factor),
    );
    let dock_bg_alpha = (ui_opacity * 190.0) as u8;
    if dock_bg_alpha > 0 {
        painter.rect_filled(dock_bg, Rounding::same(32.0 * scale_factor), Color32::from_rgba_premultiplied(0x10, 0x10, 0x14, dock_bg_alpha));

        let scaled_dock_sx = scaled_dock_center.x - scaled_dock_total * 0.5;

        for (idx, (icon, label)) in dock_items.iter().enumerate() {
            let x = scaled_dock_sx + idx as f32 * (scaled_item_size + scaled_dock_gap) + scaled_item_size * 0.5;
            let y = scaled_dock_center.y;
            let base = egui::Rect::from_center_size(egui::pos2(x, y), Vec2::splat(scaled_item_size));
            let is_sel = state.active_dock && state.dock_selected == idx;
            let draw = if is_sel { base.expand(4.0 * scale_factor) } else { base };

            let resp = ui.allocate_rect(draw, Sense::click());
            if resp.clicked() && state.boot_stage == BootStage::None {
                if state.active_dock && state.dock_selected == idx {
                    if idx == 6 {
                        state.palette_open = !state.palette_open;
                        if state.palette_open {
                            state.palette_selected = crate::app_settings::CarouselTheme::all()
                                .iter()
                                .position(|x| *x == theme)
                                .unwrap_or(0);
                        }
                    } else {
                        action = match idx {
                            0 => CarouselAction::SwitchToGrid,
                            1 => CarouselAction::Rescan,
                            2 => CarouselAction::Launch(selected_game.path.to_string_lossy().to_string()),
                            3 => CarouselAction::StopEmulation,
                            4 => CarouselAction::OpenController,
                            5 => CarouselAction::OpenSettings,
                            7 => CarouselAction::OpenDebug,
                            8 => CarouselAction::Quit,
                            _ => CarouselAction::None,
                        };
                    }
                } else {
                    state.active_dock = true;
                    state.dock_selected = idx;
                }
            }

            let fill = Color32::from_rgba_premultiplied(0x18, 0x18, 0x20, (ui_opacity * 200.0) as u8);
            if is_sel {
                let c_glow = (ui_opacity * 110.0) as u8;
                let c_border = (ui_opacity * 255.0) as u8;
                draw_gradient_circle(&painter, draw.center(), draw.width() * 0.5, 4.0 * scale_factor, t, c_glow);
                draw_gradient_circle(&painter, draw.center(), draw.width() * 0.5, 1.5 * scale_factor, t, c_border);
                painter.circle_filled(draw.center(), draw.width() * 0.5, fill);
            } else {
                painter.circle(draw.center(), draw.width() * 0.5, fill, Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x2C, 0x2C, 0x36, (ui_opacity * 255.0) as u8)));
            }

            let icon_color = if is_sel { Color32::WHITE } else { Color32::from_rgb(0xD0, 0xD0, 0xDC) };
            let final_icon_color = Color32::from_rgba_unmultiplied(icon_color.r(), icon_color.g(), icon_color.b(), (ui_opacity * icon_color.a() as f32) as u8);
            painter.text(draw.center(), egui::Align2::CENTER_CENTER, icon, FontId::proportional(draw.width() * 0.42), final_icon_color);

            if is_sel {
                shadowed_text(
                    &painter,
                    egui::pos2(scaled_dock_center.x, scaled_dock_center.y + scaled_item_size + 24.0 * scale_factor),
                    egui::Align2::CENTER_CENTER,
                    label,
                    FontId::proportional(14.0 * scale_factor),
                    Color32::from_rgba_unmultiplied(255, 255, 255, (ui_opacity * 255.0) as u8),
                    false,
                );
            }
        }
    }

    if state.palette_open {
        let themes = crate::app_settings::CarouselTheme::all();
        if state.palette_selected >= themes.len() {
            state.palette_selected = 0;
        }
        let sw = 46.0f32;
        let sw_gap = 12.0f32;
        let total = themes.len() as f32 * sw + (themes.len() - 1) as f32 * sw_gap;
        let center = egui::pos2(bg_rect.center().x, dock_y - 104.0);
        let scaled_center = screen_center + (center - screen_center) * scale_factor;
        let s_sw = sw * scale_factor;
        let s_gap = sw_gap * scale_factor;
        let s_total = total * scale_factor;

        let panel = egui::Rect::from_center_size(
            scaled_center,
            Vec2::new(s_total + 52.0 * scale_factor, s_sw + 92.0 * scale_factor),
        );
        painter.rect_filled(panel, Rounding::same(20.0 * scale_factor), Color32::from_rgba_premultiplied(0x10, 0x10, 0x16, 240));
        painter.rect_stroke(panel, Rounding::same(20.0 * scale_factor), Stroke::new(1.0, Color32::from_rgb(0x2C, 0x2C, 0x38)));

        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.min.y + 20.0 * scale_factor), egui::Align2::CENTER_CENTER, "Background Theme", FontId::proportional(15.0 * scale_factor), Color32::WHITE, true);

        let sx = scaled_center.x - s_total * 0.5;
        let row_y = scaled_center.y + 6.0 * scale_factor;
        for (i, th) in themes.iter().enumerate() {
            let x = sx + i as f32 * (s_sw + s_gap) + s_sw * 0.5;
            let r = egui::Rect::from_center_size(egui::pos2(x, row_y), Vec2::splat(s_sw));
            let resp = ui.allocate_rect(r, Sense::click());
            if resp.hovered() {
                state.palette_selected = i;
            }
            if resp.clicked() {
                action = CarouselAction::SetTheme(*th);
            }
            let is_sel = state.palette_selected == i;
            let rounding = Rounding::same(10.0 * scale_factor);
            if is_sel {
                draw_gradient_rounded_rect(&painter, r.center(), r.expand(4.5 * scale_factor), 13.0 * scale_factor, t, 210);
            }
            match th.color() {
                Some((cr, cg, cb)) => {
                    painter.rect_filled(r, rounding, Color32::from_rgb(cr, cg, cb));
                }
                None => draw_gradient_rounded_rect(&painter, r.center(), r, 10.0 * scale_factor, t, 255),
            }
            painter.rect_stroke(r, rounding, Stroke::new(1.3, Color32::from_rgba_unmultiplied(255, 255, 255, if is_sel { 235 } else { 55 })));
        }

        let cur = themes.get(state.palette_selected).copied().unwrap_or_default();
        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.max.y - 18.0 * scale_factor), egui::Align2::CENTER_CENTER, cur.label(), FontId::proportional(14.0 * scale_factor), Color32::from_rgb(0xD2, 0xD2, 0xDE), false);
    }

    let hint_connected = last_input.connected;
    let hint = match (hint_connected, is_running) {
        (true, true) => "🎮  [Home/A] Resume  ·  [X] Stop  ·  [Left/Right] Browse  ·  [Down] Dock",
        (true, false) => "🎮  [A] Launch  ·  [B] Back  ·  [Left/Right] Browse  ·  [Down] Dock  ·  [Up] Games",
        (false, true) => "⌨  [Enter] Resume  ·  [X] Stop  ·  [Left/Right] Browse  ·  [Down] Dock",
        (false, false) => "⌨  [Enter] Launch  ·  [Esc] Back  ·  [Left/Right] Browse  ·  [Down] Dock  ·  [Up] Games",
    };
    let hint_alpha = (ui_opacity * 222.0) as u8;
    if hint_alpha > 0 {
        let hint_pos = egui::pos2(bg_rect.min.x + 24.0, bg_rect.max.y - 24.0);
        let scaled_hint_pos = screen_center + (hint_pos - screen_center) * scale_factor;
        shadowed_text(
            &painter,
            scaled_hint_pos,
            egui::Align2::LEFT_BOTTOM,
            hint,
            FontId::proportional(12.0 * scale_factor),
            Color32::from_rgba_unmultiplied(0xD2, 0xD2, 0xDE, hint_alpha),
            false,
        );
    }

    if state.boot_stage == BootStage::None {
        handle_input(state, last_input, ui, &mut action, n_games, is_running, playing, theme, selected_game);
        if let CarouselAction::Launch(path) = action {
            state.boot_stage = BootStage::Transitioning {
                game_index: state.selected,
                start_time: t,
                launch_path: path,
            };
            action = CarouselAction::None;
        }
    }

    if let BootStage::Transitioning { game_index, start_time, launch_path } = &state.boot_stage {
        let elapsed = t - *start_time;
        if elapsed >= 0.90 {
            action = CarouselAction::Launch(launch_path.clone());
            state.boot_stage = BootStage::AwaitingFrame {
                game_index: *game_index,
                start_time: t,
            };
        }
    }

    ctx.request_repaint();
    action
}

#[allow(clippy::too_many_arguments)]
fn handle_input(
    state: &mut CarouselState,
    last_input: &crate::input::InputSnapshot,
    ui: &mut egui::Ui,
    action: &mut CarouselAction,
    n_games: usize,
    is_running: bool,
    playing: Option<usize>,
    theme: crate::app_settings::CarouselTheme,
    selected_game: &GameEntry,
) {
    let mut left   = ui.input(|i| i.key_pressed(egui::Key::ArrowLeft));
    let mut right  = ui.input(|i| i.key_pressed(egui::Key::ArrowRight));
    let mut up     = ui.input(|i| i.key_pressed(egui::Key::ArrowUp));
    let mut down   = ui.input(|i| i.key_pressed(egui::Key::ArrowDown));
    let mut select = ui.input(|i| i.key_pressed(egui::Key::Enter));
    let mut back   = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let mut stop   = ui.input(|i| i.key_pressed(egui::Key::X));

    if last_input.connected {
        use crate::controller_config::SwitchButton;
        if last_input.is(SwitchButton::DLeft)  { left   = true; }
        if last_input.is(SwitchButton::DRight) { right  = true; }
        if last_input.is(SwitchButton::DUp)    { up     = true; }
        if last_input.is(SwitchButton::DDown)  { down   = true; }
        if last_input.is(SwitchButton::A)      { select = true; }
        if last_input.is(SwitchButton::B)      { back   = true; }
        if last_input.is(SwitchButton::X)      { stop   = true; }

        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_NAV: AtomicU64 = AtomicU64::new(0);
        let now  = ui.input(|i| i.time);
        let last = f64::from_bits(LAST_NAV.load(Ordering::Relaxed));
        if now - last > 0.18 {
            let lx = last_input.lx();
            let ly = last_input.ly();
            let mut nav = false;
            if lx < -0.5 { left  = true; nav = true; }
            if lx >  0.5 { right = true; nav = true; }
            if ly >  0.5 { up    = true; nav = true; }
            if ly < -0.5 { down  = true; nav = true; }
            if nav { LAST_NAV.store(now.to_bits(), Ordering::Relaxed); }
        }
    }

    if state.palette_open {
        let themes = crate::app_settings::CarouselTheme::all();
        if left && state.palette_selected > 0 {
            state.palette_selected -= 1;
        }
        if right && state.palette_selected + 1 < themes.len() {
            state.palette_selected += 1;
        }
        if select {
            if let Some(th) = themes.get(state.palette_selected) {
                *action = CarouselAction::SetTheme(*th);
            }
        }
        if back {
            state.palette_open = false;
        }
        return;
    }

    if left {
        if state.active_dock {
            if state.dock_selected > 0 { state.dock_selected -= 1; }
        } else if state.selected > 0 {
            state.selected -= 1;
        }
    }
    if right {
        if state.active_dock {
            if state.dock_selected < DOCK_COUNT - 1 { state.dock_selected += 1; }
        } else if state.selected < n_games - 1 {
            state.selected += 1;
        }
    }
    if up   && state.active_dock  { state.active_dock = false; }
    if down && !state.active_dock { state.active_dock = true; }

    if select {
        if state.active_dock {
            if state.dock_selected == 6 {
                state.palette_open = true;
                state.palette_selected = crate::app_settings::CarouselTheme::all()
                    .iter()
                    .position(|x| *x == theme)
                    .unwrap_or(0);
            } else {
                *action = match state.dock_selected {
                    0 => CarouselAction::SwitchToGrid,
                    1 => CarouselAction::Rescan,
                    2 => CarouselAction::Launch(selected_game.path.to_string_lossy().to_string()),
                    3 => CarouselAction::StopEmulation,
                    4 => CarouselAction::OpenController,
                    5 => CarouselAction::OpenSettings,
                    7 => CarouselAction::OpenDebug,
                    8 => CarouselAction::Quit,
                    _ => CarouselAction::None,
                };
            }
        } else if playing == Some(state.selected) {
            *action = CarouselAction::Resume;
        } else {
            *action = CarouselAction::Launch(selected_game.path.to_string_lossy().to_string());
        }
    }
    if back {
        if state.active_dock {
            state.active_dock = false;
        }
    }
    if stop && is_running && !state.active_dock {
        *action = CarouselAction::StopEmulation;
    }
}
