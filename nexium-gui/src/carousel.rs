use eframe::egui;
use eframe::egui::{Color32, FontId, Rounding, Sense, Stroke, Vec2};
use crate::library::Library;

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
    pub dock_anim: f32,
    pub dock_focus: f32,
    pub theme_t: f32,
    pub x_held: bool,
    pub is_dragging: bool,
    pub drag_start_x: f32,
    pub drag_start_offset: f32,
    pub drag_moved: bool,
    pub boot_stage: BootStage,
    pub palette_open: bool,
    pub palette_selected: usize,
    pub palette_t: f32,
    pub profile_focused: bool,
    pub search_buf: String,
    pub search_focused: bool,
    pub profile_click_time: Option<f32>,
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
            dock_anim: 0.0,
            dock_focus: 0.0,
            theme_t: 0.0,
            x_held: false,
            is_dragging: false,
            drag_start_x: 0.0,
            drag_start_offset: 0.0,
            drag_moved: false,
            boot_stage: BootStage::None,
            palette_open: false,
            palette_selected: 0,
            palette_t: 0.0,
            profile_focused: false,
            search_buf: String::new(),
            search_focused: false,
            profile_click_time: None,
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
    OpenProfile,
    ToggleFavorite(String),
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
    let lum = 0.299 * color.r() as f32 + 0.587 * color.g() as f32 + 0.114 * color.b() as f32;
    let (oc, oa) = if lum < 130.0 { (255u8, 130.0f32) } else { (0u8, 240.0f32) };
    let outline_col = Color32::from_rgba_unmultiplied(oc, oc, oc, (oa * alpha_pct) as u8);
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

pub fn draw_rainbow_rounded_rect(
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
        color: Color32::from_rgba_unmultiplied(200, 200, 200, alpha),
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
    
    for pt in pts {
        let tx = if w > 0.0 { (pt.x - min_x) / w } else { 0.0 };
        let ty = if h > 0.0 { (pt.y - min_y) / h } else { 0.0 };
        
        let dx = tx - 0.5;
        let dy = ty - 0.5;
        let angle = dy.atan2(dx);
        let hue = ((angle / std::f32::consts::TAU) + t * 0.08) % 1.0;
        let hue = if hue < 0.0 { hue + 1.0 } else { hue };
        let (cr, cg, cb) = hsv_to_rgb(hue, 0.85, 0.85);
        
        mesh.vertices.push(egui::epaint::Vertex {
            pos: pt,
            uv: egui::pos2(0.0, 0.0),
            color: Color32::from_rgba_unmultiplied(cr, cg, cb, alpha),
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

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h_i = (h * 6.0) as i32;
    let f = h * 6.0 - h_i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    
    let (r, g, b) = match h_i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
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

pub fn draw_backdrop(
    painter: &egui::Painter,
    rect: egui::Rect,
    color: Color32,
    t: f32,
    theme: crate::app_settings::BackdropTheme,
    opacity: f32,
    light_t: f32,
) {
    use crate::app_settings::BackdropTheme;
    let (dark_base, light_base) = match theme {
        BackdropTheme::None => (Color32::from_rgb(0x18, 0x18, 0x1F), Color32::from_rgb(0xCE, 0xCE, 0xD8)),
        _ => (Color32::from_rgb(0x08, 0x08, 0x0C), Color32::from_rgb(0xD6, 0xD6, 0xE0)),
    };
    let base = lerp_color(dark_base, light_base, light_t);
    match theme {
        BackdropTheme::Waves => {
            let cx = rect.min.x + rect.width() * 0.22;
            let cy = rect.min.y + rect.height() * 0.44;
            draw_wave_background(painter, rect, color, t, cx, cy, opacity, base);
        }
        BackdropTheme::Gradient => draw_gradient_backdrop(painter, rect, color, opacity, base),
        BackdropTheme::None => {
            if opacity <= 0.001 { return; }
            let a = (opacity * 255.0) as u8;
            painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a));
        }
    }
}

fn draw_gradient_backdrop(
    painter: &egui::Painter,
    rect: egui::Rect,
    color: Color32,
    opacity: f32,
    base: Color32,
) {
    if opacity <= 0.001 { return; }
    let base_alpha = (opacity * 255.0) as u8;
    painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), base_alpha));

    let top_y = rect.min.y + rect.height() * 0.42;
    let bot_y = rect.max.y;
    let cols = 96usize;
    let peak = 225.0f32;

    let mut mesh = egui::epaint::Mesh::default();
    for c in 0..=cols {
        let nx = c as f32 / cols as f32;
        let x = rect.min.x + nx * rect.width();
        let horiz = {
            let d = (nx - 0.5).abs() * 2.0;
            (1.0 - d * d).max(0.0)
        };
        let bottom_alpha = ((peak * horiz * opacity).clamp(0.0, 255.0)) as u8;
        let top_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 0);
        let bot_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), bottom_alpha);
        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x, top_y), uv: egui::pos2(0.0, 0.0), color: top_col });
        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x, bot_y), uv: egui::pos2(0.0, 0.0), color: bot_col });
    }
    for c in 0..cols {
        let tl = 2 * c as u32;
        let bl = tl + 1;
        let tr = tl + 2;
        let br = tl + 3;
        mesh.indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub fn draw_wave_background(
    painter: &egui::Painter,
    rect: egui::Rect,
    color: Color32,
    t: f32,
    _glow_cx: f32,
    _glow_cy: f32,
    opacity: f32,
    base: Color32,
) {
    if opacity <= 0.001 { return; }
    let bg_alpha = (opacity * 255.0) as u8;
    painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), bg_alpha));

    let w = rect.width();
    let h = rect.height();
    let steps = 128usize;

    let bands: &[(f32, f32, f32, f32, u8)] = &[
        (0.55, 0.012, 0.0,  1.5,  70),
        (0.42, 0.018, 1.1,  1.2,  58),
        (0.70, 0.009, 2.3,  0.9,  44),
        (0.30, 0.022, 0.6,  1.7,  34),
        (0.85, 0.007, 3.5,  0.7,  24),
    ];

    for &(base_frac, amp_frac, phase_off, speed, alpha) in bands {
        let base_y = rect.min.y + h * base_frac;
        let amp    = h * amp_frac;
        let phase  = t * speed + phase_off;

        let band_alpha = ((alpha as f32) * opacity) as u8;
        if band_alpha == 0 { continue; }
        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), band_alpha);

        let mut mesh = egui::epaint::Mesh::default();
        for s in 0..=steps {
            let x  = rect.min.x + (s as f32 / steps as f32) * w;
            let nx = s as f32 / steps as f32;
            let y  = base_y
                + (nx * std::f32::consts::TAU + phase).sin() * amp
                + (nx * std::f32::consts::TAU * 1.7 + phase * 0.8).cos() * amp * 0.4;
            
            mesh.vertices.push(egui::epaint::Vertex {
                pos: egui::pos2(x, y),
                uv: egui::pos2(0.0, 0.0),
                color: fill,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: egui::pos2(x, rect.max.y),
                uv: egui::pos2(0.0, 0.0),
                color: fill,
            });
        }
        
        for i in 0..steps {
            let tl = 2 * i as u32;
            let bl = tl + 1;
            let tr = tl + 2;
            let br = tl + 3;
            mesh.indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
        let crest_pts: Vec<egui::Pos2> = (0..=steps).map(|s| mesh.vertices[2 * s].pos).collect();
        painter.add(egui::Shape::mesh(mesh));
        let outer_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (band_alpha / 2).max(1));
        painter.add(egui::Shape::line(crest_pts.clone(), Stroke::new(2.0, outer_col)));
        let inner_col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), band_alpha);
        painter.add(egui::Shape::line(crest_pts, Stroke::new(1.0, inner_col)));
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
    profile_tex: Option<egui::TextureId>,
    profile_name: &str,
    interactive: bool,
    scale_factor: f32,
    alpha_factor: f32,
    backdrop_theme: crate::app_settings::BackdropTheme,
    light_mode: bool,
    favorites: &[std::path::PathBuf],
) -> CarouselAction {
    let mut action = CarouselAction::None;

    // 1. Filter games based on search_buf
    let mut filtered_indices: Vec<usize> = if state.search_buf.is_empty() {
        (0..lib.games.len()).collect()
    } else {
        let q = state.search_buf.to_lowercase();
        (0..lib.games.len())
            .filter(|&idx| lib.games[idx].title.to_lowercase().contains(&q))
            .collect()
    };
    filtered_indices.sort_by(|&a, &b| {
        let fa = favorites.iter().any(|p| *p == lib.games[a].path);
        let fb = favorites.iter().any(|p| *p == lib.games[b].path);
        fb.cmp(&fa)
    });

    // The items in the carousel will be the filtered games, plus the "Add Dir" card!
    let n_items = filtered_indices.len() + 1;

    let bg_rect = ui.max_rect();
    let t = ui.input(|i| i.time) as f32;
    let dt = ui.input(|i| i.stable_dt).min(0.1);

    let theme_target = if light_mode { 1.0 } else { 0.0 };
    state.theme_t += (theme_target - state.theme_t) * (dt * 5.0).min(1.0);
    if (state.theme_t - theme_target).abs() < 0.002 {
        state.theme_t = theme_target;
    }
    let th = state.theme_t;
    let tl = |dark: Color32, light: Color32| lerp_color(dark, light, th);
    let col_text = tl(Color32::WHITE, Color32::from_rgb(0x1E, 0x1E, 0x28));
    let col_muted = tl(Color32::from_rgb(0xD0, 0xD0, 0xDC), Color32::from_rgb(0x5A, 0x5A, 0x66));
    let col_surface = tl(Color32::from_rgb(0x18, 0x18, 0x20), Color32::from_rgb(0xFF, 0xFF, 0xFF));
    let col_bar = tl(Color32::from_rgb(0x10, 0x10, 0x14), Color32::from_rgb(0xF2, 0xF2, 0xF6));
    let col_border = tl(Color32::from_rgb(0x2C, 0x2C, 0x36), Color32::from_rgb(0xC6, 0xC6, 0xD0));
    let col_clock = tl(Color32::from_rgb(0xF0, 0xF0, 0xF6), Color32::from_rgb(0x20, 0x20, 0x2A));

    let screen_height = bg_rect.height();
    let hero_size = (screen_height * 0.35).clamp(260.0, 480.0);
    let stride = hero_size + (bg_rect.width() * 0.012).clamp(10.0, 22.0);

    let hero_cx   = bg_rect.min.x + bg_rect.width() * 0.22;
    let hero_cy   = bg_rect.min.y + bg_rect.height() * 0.44;

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

    // Draw and handle Search Bar (centered top)
    let s = (screen_height / 820.0).clamp(1.0, 2.4);
    let search_h = 36.0 * s * scale_factor;
    let search_w = 320.0 * s * scale_factor;
    let search_center = egui::pos2(bg_rect.center().x, bg_rect.min.y + 42.0 * s);
    let search_rect = egui::Rect::from_center_size(search_center, Vec2::new(search_w, search_h));

    if interactive && state.boot_stage == BootStage::None && !state.palette_open && !state.profile_focused {
        let primary_clicked = ui.input(|i| i.pointer.primary_clicked());
        if primary_clicked {
            if let Some(pos) = pointer_pos {
                let clicked_search = search_rect.contains(pos);
                if clicked_search != state.search_focused {
                    state.search_focused = clicked_search;
                }
            }
        }

        if state.search_focused {
            let events = ui.input(|i| i.events.clone());
            let mut changed = false;
            for ev in events {
                match ev {
                    egui::Event::Text(txt) => {
                        for ch in txt.chars() {
                            if !ch.is_control() && state.search_buf.chars().count() < 30 {
                                state.search_buf.push(ch);
                                changed = true;
                            }
                        }
                    }
                    egui::Event::Key { key: egui::Key::Backspace, pressed: true, .. } => {
                        state.search_buf.pop();
                        changed = true;
                    }
                    egui::Event::Key { key: egui::Key::Escape, pressed: true, .. } => {
                        state.search_focused = false;
                    }
                    egui::Event::Key { key: egui::Key::Enter, pressed: true, .. } => {
                        state.search_focused = false;
                    }
                    _ => {}
                }
            }
            if changed {
                state.selected = 0;
            }
        }
    }

    if interactive && state.boot_stage == BootStage::None && !state.search_focused && !state.palette_open {
        if pointer_pressed {
            if let Some(pos) = pointer_pos {
                if pos.y < dock_y && pos.y > bg_rect.min.y + 70.0 * s {
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
                state.scroll_offset = (state.drag_start_offset - delta_x / stride).clamp(0.0, (n_items.saturating_sub(1)) as f32);
                
                let nearest = state.scroll_offset.round().clamp(0.0, (n_items.saturating_sub(1)) as f32) as usize;
                if nearest != state.selected {
                    state.selected = nearest;
                    state.active_dock = false;
                }
            }
        }

        if pointer_released || !pointer_down {
            if state.is_dragging {
                state.is_dragging = false;
                state.selected = state.scroll_offset.round().clamp(0.0, (n_items.saturating_sub(1)) as f32) as usize;
            }
        }
    }

    if state.selected >= n_items {
        state.selected = n_items - 1;
    }

    if state.is_dragging {
        // Controlled directly by drag
    } else if state.boot_stage == BootStage::None {
        state.scroll_offset += (state.selected as f32 - state.scroll_offset) * (dt * 11.0).min(1.0);
    }

    let target_color = if state.selected == filtered_indices.len() {
        Color32::from_rgb(0x35, 0x38, 0x42)
    } else if theme == crate::app_settings::CarouselTheme::Rgb {
        let speed = 0.025;
        let hue = (t * speed) % 1.0;
        let (cr, cg, cb) = hsv_to_rgb(hue, 0.85, 0.85);
        Color32::from_rgb(cr, cg, cb)
    } else {
        match theme.color() {
            Some((r, g, b)) => Color32::from_rgb(r, g, b),
            None => {
                if state.selected < filtered_indices.len() {
                    lib.games[filtered_indices[state.selected]].dominant_color
                } else {
                    Color32::from_rgb(0x35, 0x38, 0x42)
                }
            }
        }
    };
    state.ambient_color = lerp_color(state.ambient_color, target_color, (dt * 10.0).min(1.0));

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("carousel_bg")));
    draw_backdrop(&painter, bg_rect, state.ambient_color, t, backdrop_theme, ui_opacity, th);

    let screen_center = bg_rect.center();
    let scale_pos = |p: egui::Pos2| -> egui::Pos2 {
        screen_center + (p - screen_center) * scale_factor
    };

    // Draw Search Bar UI
    let sb_alpha = (ui_opacity * 255.0) as u8;
    if sb_alpha > 0 {
        let accent = {
            let c = state.ambient_color;
            let f = |x: u8| (x as f32 + (255.0 - x as f32) * 0.4) as u8;
            Color32::from_rgb(f(c.r()), f(c.g()), f(c.b()))
        };
        let sb_bg = Color32::from_rgba_unmultiplied(col_bar.r(), col_bar.g(), col_bar.b(), (ui_opacity * 220.0) as u8);
        let sb_border = if state.search_focused {
            accent
        } else {
            Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), sb_alpha)
        };
        painter.rect_filled(search_rect, Rounding::same(18.0 * s * scale_factor), sb_bg);
        painter.rect_stroke(search_rect, Rounding::same(18.0 * s * scale_factor), Stroke::new(1.5 * scale_factor, sb_border));

        let icon_pos = scale_pos(search_rect.min + Vec2::new(16.0 * s, search_rect.height() * 0.5));
        painter.text(icon_pos, egui::Align2::LEFT_CENTER, "🔍", FontId::proportional(14.0 * s * scale_factor), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), sb_alpha));

        let text_pos = scale_pos(search_rect.min + Vec2::new(38.0 * s, search_rect.height() * 0.5));
        if state.search_buf.is_empty() {
            painter.text(text_pos, egui::Align2::LEFT_CENTER, "Search games...", FontId::proportional(14.0 * s * scale_factor), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), sb_alpha));
        } else {
            painter.text(text_pos, egui::Align2::LEFT_CENTER, &state.search_buf, FontId::proportional(14.0 * s * scale_factor), Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), sb_alpha));
        }

        if state.search_focused && (t * 2.0) as usize % 2 == 0 {
            let text_w = ui.fonts(|f| f.layout_no_wrap(state.search_buf.clone(), FontId::proportional(14.0 * s * scale_factor), Color32::WHITE).size().x);
            let caret_pos = text_pos + Vec2::new(text_w + 2.0 * scale_factor, 0.0);
            painter.line_segment([caret_pos - Vec2::new(0.0, 8.0 * s * scale_factor), caret_pos + Vec2::new(0.0, 8.0 * s * scale_factor)], Stroke::new(1.8 * scale_factor, accent));
        }
    }

    let mut want_fav: Option<String> = None;
    for i in 0..n_items {
        let diff     = i as f32 - state.scroll_offset;
        let abs_diff = diff.abs();
        if abs_diff > 4.6 && state.boot_stage == BootStage::None { continue; }

        let scale   = (1.0 - abs_diff * 0.05).max(0.82);
        let mut alpha_f = (1.0 - abs_diff * 0.14).clamp(0.0, 1.0);
        if abs_diff > 3.6 {
            alpha_f *= (4.6 - abs_diff).clamp(0.0, 1.0);
        }
        let sz      = hero_size * scale;
        let cx      = hero_cx + diff * stride;
        let cy      = hero_cy + abs_diff * 6.0;

        let mut draw_cx = cx;
        let mut draw_cy = cy;
        let mut draw_sz = sz;
        let mut scale_x = 1.0f32;
        let mut scale_y = 1.0f32;
        let mut glow_expansion = 6.0f32;
        let mut glow_alpha = 100u8;
        let mut border_alpha = 255u8;
        
        let is_hero = i == state.selected && !state.active_dock && !state.profile_focused;

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

        if state.boot_stage == BootStage::None {
            let phase = i as f32 * 0.9;
            draw_cy += (t * 1.25 + phase).sin() * 2.6 * scale;
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
            let final_glow_alpha = ((glow_alpha as f32) * alpha_f).clamp(0.0, 255.0) as u8;
            let final_border_alpha = ((border_alpha as f32) * alpha_f).clamp(0.0, 255.0) as u8;
            if final_border_alpha > 0 {
                draw_gradient_rounded_rect(&painter, draw_rect.center(), draw_rect.expand(glow_expansion * scale_factor), 16.0 * scale_factor, t, final_glow_alpha);
                draw_gradient_rounded_rect(&painter, draw_rect.center(), draw_rect.expand(2.5 * scale_factor), 14.0 * scale_factor, t, final_border_alpha);
            }
        }

        let is_add_dir = i == filtered_indices.len();
        let resp = ui.allocate_rect(draw_rect, Sense::click());
        if interactive && resp.clicked() && !state.drag_moved && state.boot_stage == BootStage::None {
            if state.selected == i {
                if state.active_dock {
                    state.active_dock = false;
                } else if is_add_dir {
                    action = CarouselAction::AddFolder;
                } else if playing == Some(filtered_indices[i]) {
                    action = CarouselAction::Resume;
                } else {
                    state.boot_stage = BootStage::Transitioning {
                        game_index: i,
                        start_time: t,
                        launch_path: lib.games[filtered_indices[i]].path.to_string_lossy().to_string(),
                    };
                }
            } else {
                state.selected = i;
                state.active_dock = false;
            }
        }

        if interactive && !is_add_dir && state.boot_stage == BootStage::None {
            let path_string = lib.games[filtered_indices[i]].path.to_string_lossy().to_string();
            let favd = favorites.iter().any(|p| *p == lib.games[filtered_indices[i]].path);
            resp.context_menu(|ui| {
                let label = if favd { "★  Unfavorite Game" } else { "☆  Favorite Game" };
                if ui.button(label).clicked() {
                    want_fav = Some(path_string.clone());
                    ui.close_menu();
                }
            });
        }

        let tint = Color32::from_white_alpha((alpha_f * 255.0) as u8);
        if is_add_dir {
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(col_surface.r(), col_surface.g(), col_surface.b(), (alpha_f * 255.0) as u8));
            painter.rect_stroke(draw_rect, Rounding::same(14.0 * scale_factor), Stroke::new(1.0 * scale_factor, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), (alpha_f * 255.0) as u8)));

            let center = draw_rect.center() - Vec2::new(0.0, 20.0 * scale_factor);
            let cross_len = 24.0 * scale_factor;
            let stroke_w = 3.0 * scale_factor;
            let stroke_color = Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), (alpha_f * 255.0) as u8);
            
            painter.line_segment(
                [center - Vec2::new(cross_len * 0.5, 0.0), center + Vec2::new(cross_len * 0.5, 0.0)],
                Stroke::new(stroke_w, stroke_color),
            );
            painter.line_segment(
                [center - Vec2::new(0.0, cross_len * 0.5), center + Vec2::new(0.0, cross_len * 0.5)],
                Stroke::new(stroke_w, stroke_color),
            );

            painter.text(
                draw_rect.center() + Vec2::new(0.0, 36.0 * scale_factor),
                egui::Align2::CENTER_CENTER,
                "Add Dir",
                FontId::proportional(final_sz * 0.085),
                Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), (alpha_f * 255.0) as u8),
            );
        } else {
            let real_idx = filtered_indices[i];
            let card_fill = tl(Color32::from_rgb(0x14, 0x14, 0x1A), Color32::from_rgb(0xE6, 0xE6, 0xEC));
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(card_fill.r(), card_fill.g(), card_fill.b(), (alpha_f * 255.0) as u8));
            if let Some(tex) = lib.texture(ctx, real_idx) {
                draw_rounded_image(&painter, tex.id(), draw_rect, 14.0 * scale_factor, tint);
            } else {
                painter.text(draw_rect.center(), egui::Align2::CENTER_CENTER, &lib.games[real_idx].format, FontId::proportional(final_sz * 0.13), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), (alpha_f * 255.0) as u8));
            }

            if !is_hero {
                painter.rect_stroke(
                    draw_rect,
                    Rounding::same(14.0 * scale_factor),
                    Stroke::new(1.5 * scale_factor, Color32::from_rgba_unmultiplied(0x50, 0x50, 0x60, (alpha_f * 170.0) as u8)),
                );
                painter.rect_stroke(
                    draw_rect.expand(1.0 * scale_factor),
                    Rounding::same(15.0 * scale_factor),
                    Stroke::new(1.0 * scale_factor, Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, (alpha_f * 120.0) as u8)),
                );
            }

            if favorites.iter().any(|p| *p == lib.games[real_idx].path) {
                let badge_r = draw_rect.width() * 0.10;
                let badge_c = egui::pos2(draw_rect.min.x + badge_r + 8.0 * scale_factor, draw_rect.min.y + badge_r + 8.0 * scale_factor);
                painter.circle_filled(badge_c, badge_r, Color32::from_rgba_unmultiplied(0x10, 0x10, 0x16, (alpha_f * 200.0) as u8));
                painter.text(
                    badge_c,
                    egui::Align2::CENTER_CENTER,
                    "★",
                    FontId::proportional(badge_r * 1.3),
                    Color32::from_rgba_unmultiplied(0xF5, 0xC1, 0x42, (alpha_f * 255.0) as u8),
                );
            }

            if playing == Some(real_idx) && playing_alpha > 0.01 && state.boot_stage == BootStage::None {
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
    }

    if let Some(p) = want_fav {
        action = CarouselAction::ToggleFavorite(p);
    }

    if let BootStage::Transitioning { start_time, .. } = state.boot_stage {
        let elapsed = t - start_time;
        if elapsed >= 0.42 {
            let p = ((elapsed - 0.42) / (0.90 - 0.42)).min(1.0);
            let overlay_alpha = (p * p * 255.0) as u8;
            painter.rect_filled(bg_rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(0, 0, 0, overlay_alpha));
        }
    }

    let hero_bob = if state.boot_stage == BootStage::None {
        (t * 1.25 + state.selected as f32 * 0.9).sin() * 2.6
    } else {
        0.0
    };
    let meta_x = hero_cx;
    let meta_y = hero_cy + hero_size * 0.52 + 36.0 + hero_bob;

    let launch_path = if state.selected < filtered_indices.len() {
        Some(lib.games[filtered_indices[state.selected]].path.to_string_lossy().to_string())
    } else {
        None
    };

    let (sel_title, sel_sub) = if state.selected == filtered_indices.len() {
        ("Add Folder".to_string(), "Add folder destinations for your decrypted ROMs".to_string())
    } else {
        let selected_game = &lib.games[filtered_indices[state.selected]];
        let title = selected_game.title.clone();
        let sub = format!(
            "{}  ·  {}  ·  {:.1} MB",
            if selected_game.author.is_empty() { "Unknown" } else { &selected_game.author },
            selected_game.format,
            selected_game.size as f32 / (1024.0 * 1024.0),
        );
        (title, sub)
    };

    let title_col = if state.active_dock { col_muted } else { col_text };
    let title_alpha = (ui_opacity * 255.0) as u8;
    if title_alpha > 0 {
        let meta_pos = egui::pos2(meta_x, meta_y);
        let scaled_meta_pos = screen_center + (meta_pos - screen_center) * scale_factor;
        shadowed_text(&painter, scaled_meta_pos, egui::Align2::CENTER_CENTER, &sel_title, FontId::proportional(28.0 * scale_factor), Color32::from_rgba_unmultiplied(title_col.r(), title_col.g(), title_col.b(), title_alpha), true);
        let sub_alpha = (ui_opacity * 222.0) as u8;
        shadowed_text(&painter, scaled_meta_pos + Vec2::new(0.0, 32.0 * scale_factor), egui::Align2::CENTER_CENTER, &sel_sub, FontId::proportional(14.0 * scale_factor), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), sub_alpha), false);
    }


    let dock_items: [(&str, &str); DOCK_COUNT] = [
        ("⊞", "Grid View"),
        ("↺", "Rescan"),
        ("▶", "Boot"),
        ("■", "Stop"),
        ("⌨", "Controller"),
        ("⚙", "Settings"),
        ("🎨", "Theme"),
        ("🪳", "Debug"),
        ("✕", "Quit"),
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
        painter.rect_filled(dock_bg, Rounding::same(32.0 * scale_factor), Color32::from_rgba_premultiplied(col_bar.r(), col_bar.g(), col_bar.b(), dock_bg_alpha));

        let scaled_dock_sx = scaled_dock_center.x - scaled_dock_total * 0.5;

        let step_px = scaled_item_size + scaled_dock_gap;

        let focus_target = if state.active_dock { 1.0 } else { 0.0 };
        state.dock_focus += (focus_target - state.dock_focus) * (dt * 16.0).min(1.0);
        state.dock_anim += (state.dock_selected as f32 - state.dock_anim) * (dt * 16.0).min(1.0);
        if (state.dock_anim - state.dock_selected as f32).abs() < 0.001 {
            state.dock_anim = state.dock_selected as f32;
        }

        for idx in 0..dock_items.len() {
            let x = scaled_dock_sx + idx as f32 * step_px + scaled_item_size * 0.5;
            let base = egui::Rect::from_center_size(egui::pos2(x, scaled_dock_center.y), Vec2::splat(scaled_item_size));

            let resp = ui.allocate_rect(base, Sense::click());
            if interactive && resp.clicked() && state.boot_stage == BootStage::None {
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
                            2 => {
                                if let Some(path) = &launch_path {
                                    CarouselAction::Launch(path.clone())
                                } else {
                                    CarouselAction::AddFolder
                                }
                            }
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

            let fill = Color32::from_rgba_premultiplied(col_surface.r(), col_surface.g(), col_surface.b(), (ui_opacity * 200.0) as u8);
            painter.circle(base.center(), base.width() * 0.5, fill, Stroke::new(1.0, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), (ui_opacity * 255.0) as u8)));
        }

        if state.dock_focus > 0.004 {
            let hl_x = scaled_dock_sx + state.dock_anim * step_px + scaled_item_size * 0.5;
            let hl_center = egui::pos2(hl_x, scaled_dock_center.y);
            let hl_r = scaled_item_size * 0.5 + 4.0 * scale_factor;
            let fo = ui_opacity * state.dock_focus;
            draw_gradient_circle(&painter, hl_center, hl_r, 4.0 * scale_factor, t, (fo * 110.0) as u8);
            draw_gradient_circle(&painter, hl_center, hl_r, 1.5 * scale_factor, t, (fo * 255.0) as u8);
            painter.circle_filled(hl_center, hl_r, Color32::from_rgba_premultiplied(col_surface.r(), col_surface.g(), col_surface.b(), (fo * 200.0) as u8));
        }

        for (idx, (icon, label)) in dock_items.iter().enumerate() {
            let x = scaled_dock_sx + idx as f32 * step_px + scaled_item_size * 0.5;
            let draw = egui::Rect::from_center_size(egui::pos2(x, scaled_dock_center.y), Vec2::splat(scaled_item_size));
            let prox = (1.0 - (idx as f32 - state.dock_anim).abs()).clamp(0.0, 1.0) * state.dock_focus;
            let icon_color = lerp_color(col_muted, col_text, prox);
            let final_icon_color = Color32::from_rgba_unmultiplied(icon_color.r(), icon_color.g(), icon_color.b(), (ui_opacity * 255.0) as u8);

            if *label == "Debug" {
                let center = draw.center() + Vec2::new(0.0, 2.0 * scale_factor);
                let head_r = 2.5 * scale_factor;
                let head_center = center - Vec2::new(0.0, 7.5 * scale_factor);
                
                // Antennae (curved lines)
                let ant_stroke = Stroke::new(1.0 * scale_factor, final_icon_color);
                painter.line_segment([head_center, head_center + Vec2::new(-4.5, -6.5) * scale_factor], ant_stroke);
                painter.line_segment([head_center, head_center + Vec2::new(4.5, -6.5) * scale_factor], ant_stroke);
                
                // Legs
                let leg_stroke = Stroke::new(1.2 * scale_factor, final_icon_color);
                // Left legs
                painter.line_segment([center - Vec2::new(2.5, 3.5) * scale_factor, center - Vec2::new(7.5, 6.0) * scale_factor], leg_stroke);
                painter.line_segment([center - Vec2::new(3.0, 0.0) * scale_factor, center - Vec2::new(8.5, 0.0) * scale_factor], leg_stroke);
                painter.line_segment([center - Vec2::new(2.5, -3.5) * scale_factor, center - Vec2::new(7.5, -6.0) * scale_factor], leg_stroke);
                // Right legs
                painter.line_segment([center + Vec2::new(2.5, -3.5) * scale_factor, center + Vec2::new(7.5, -6.0) * scale_factor], leg_stroke);
                painter.line_segment([center + Vec2::new(3.0, 0.0) * scale_factor, center + Vec2::new(8.5, 0.0) * scale_factor], leg_stroke);
                painter.line_segment([center + Vec2::new(2.5, 3.5) * scale_factor, center + Vec2::new(7.5, 6.0) * scale_factor], leg_stroke);
                
                // Head
                painter.circle_filled(head_center, head_r, final_icon_color);
                
                // Body (ellipse/rounded rect)
                let body_rect = egui::Rect::from_center_size(center + Vec2::new(0.0, 0.5 * scale_factor), Vec2::new(8.0 * scale_factor, 13.0 * scale_factor));
                painter.rect_filled(body_rect, Rounding::same(4.0 * scale_factor), final_icon_color);
            } else if *label == "Quit" {
                let center = draw.center();
                let size = 6.0 * scale_factor;
                let stroke_w = 2.2 * scale_factor;
                painter.line_segment(
                    [center - Vec2::new(size, size), center + Vec2::new(size, size)],
                    Stroke::new(stroke_w, final_icon_color),
                );
                painter.line_segment(
                    [center + Vec2::new(-size, size), center + Vec2::new(size, -size)],
                    Stroke::new(stroke_w, final_icon_color),
                );
            } else {
                painter.text(draw.center(), egui::Align2::CENTER_CENTER, icon, FontId::proportional(draw.width() * 0.42), final_icon_color);
            }
        }

        if state.dock_focus > 0.004 {
            let sel = state.dock_selected.min(dock_items.len() - 1);
            let label = dock_items[sel].1;
            let alpha = ((1.0 - state.palette_t) * ui_opacity * state.dock_focus * 255.0) as u8;
            if alpha > 0 {
                let text_y = scaled_dock_center.y - (scaled_item_size + 22.0 * scale_factor) * 0.5 - 14.0 * scale_factor;
                shadowed_text(
                    &painter,
                    egui::pos2(scaled_dock_center.x, text_y),
                    egui::Align2::CENTER_BOTTOM,
                    label,
                    FontId::proportional(17.5 * scale_factor),
                    Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), alpha),
                    true,
                );
            }
        }
    }

    let palette_target = if state.palette_open { 1.0 } else { 0.0 };
    let palette_step = (dt * 8.0).clamp(0.0, 0.14);
    if state.palette_t < palette_target {
        state.palette_t = (state.palette_t + palette_step).min(palette_target);
    } else if state.palette_t > palette_target {
        state.palette_t = (state.palette_t - palette_step).max(palette_target);
    }
    if !state.palette_open && state.palette_t < 0.004 {
        state.palette_t = 0.0;
    }
    if state.palette_t > 0.004 {
        let themes = crate::app_settings::CarouselTheme::all();
        if state.palette_selected >= themes.len() {
            state.palette_selected = 0;
        }
        let ease = state.palette_t * state.palette_t * (3.0 - 2.0 * state.palette_t);
        let pa = (ease * ui_opacity).clamp(0.0, 1.0);
        let pop = 0.92 + 0.08 * ease;
        let a = |x: f32| (x * pa).clamp(0.0, 255.0) as u8;

        let sw = 46.0f32;
        let sw_gap = 12.0f32;
        let total = themes.len() as f32 * sw + (themes.len() - 1) as f32 * sw_gap;
        let rise = (1.0 - ease) * 22.0;
        let center = egui::pos2(bg_rect.center().x, dock_y - 104.0 + rise);
        let scaled_center = screen_center + (center - screen_center) * scale_factor;
        let s_sw = sw * scale_factor * pop;
        let s_gap = sw_gap * scale_factor * pop;
        let s_total = total * scale_factor * pop;

        let panel = egui::Rect::from_center_size(
            scaled_center,
            Vec2::new(s_total + 52.0 * scale_factor * pop, s_sw + 92.0 * scale_factor * pop),
        );
        let interactive = state.palette_open && state.palette_t > 0.6;
        painter.rect_filled(panel, Rounding::same(20.0 * scale_factor), Color32::from_rgba_unmultiplied(0x10, 0x10, 0x16, a(240.0)));
        painter.rect_stroke(panel, Rounding::same(20.0 * scale_factor), Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x2C, 0x2C, 0x38, a(255.0))));

        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.min.y + 20.0 * scale_factor), egui::Align2::CENTER_CENTER, "Background Theme", FontId::proportional(15.0 * scale_factor), Color32::from_rgba_unmultiplied(255, 255, 255, a(255.0)), true);

        let sx = scaled_center.x - s_total * 0.5;
        let row_y = scaled_center.y + 6.0 * scale_factor;
        for (i, th) in themes.iter().enumerate() {
            let x = sx + i as f32 * (s_sw + s_gap) + s_sw * 0.5;
            let r = egui::Rect::from_center_size(egui::pos2(x, row_y), Vec2::splat(s_sw));
            if interactive {
                let resp = ui.allocate_rect(r, Sense::click());
                if resp.hovered() {
                    state.palette_selected = i;
                }
                if resp.clicked() {
                    action = CarouselAction::SetTheme(*th);
                }
            }
            let is_sel = state.palette_selected == i;
            let rounding = Rounding::same(10.0 * scale_factor);
            if is_sel {
                if *th == crate::app_settings::CarouselTheme::Rgb {
                    draw_rainbow_rounded_rect(&painter, r.center(), r.expand(4.5 * scale_factor), 13.0 * scale_factor, t, a(210.0));
                } else {
                    draw_gradient_rounded_rect(&painter, r.center(), r.expand(4.5 * scale_factor), 13.0 * scale_factor, t, a(210.0));
                }
            }
            if *th == crate::app_settings::CarouselTheme::Rgb {
                draw_rainbow_rounded_rect(&painter, r.center(), r, 10.0 * scale_factor, t, a(255.0));
            } else {
                match th.color() {
                    Some((cr, cg, cb)) => {
                        painter.rect_filled(r, rounding, Color32::from_rgba_unmultiplied(cr, cg, cb, a(255.0)));
                    }
                    None => draw_gradient_rounded_rect(&painter, r.center(), r, 10.0 * scale_factor, t, a(255.0)),
                }
            }
            painter.rect_stroke(r, rounding, Stroke::new(1.3, Color32::from_rgba_unmultiplied(255, 255, 255, a(if is_sel { 235.0 } else { 55.0 }))));
        }

        let cur = themes.get(state.palette_selected).copied().unwrap_or_default();
        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.max.y - 18.0 * scale_factor), egui::Align2::CENTER_CENTER, cur.label(), FontId::proportional(14.0 * scale_factor), Color32::from_rgba_unmultiplied(0xD2, 0xD2, 0xDE, a(255.0)), false);
    }

    if ui_opacity > 0.01 {
        let top_alpha = (ui_opacity * 255.0) as u8;
        let top_s = (bg_rect.height() / 820.0).clamp(1.0, 2.4);
        let av_r = 26.0f32 * top_s;
        let av_center = egui::pos2(bg_rect.min.x + 34.0 * top_s + av_r, bg_rect.min.y + 26.0 * top_s + av_r);
        let scaled_av = screen_center + (av_center - screen_center) * scale_factor;
        let scaled_av_r = av_r * scale_factor;
        let av_rect = egui::Rect::from_center_size(scaled_av, Vec2::splat(scaled_av_r * 2.0));
        let av_resp = ui.allocate_rect(av_rect.expand(3.0 * scale_factor), Sense::click());
        if interactive && av_resp.clicked() && state.boot_stage == BootStage::None && !state.palette_open && state.profile_click_time.is_none() {
            state.profile_click_time = Some(t);
        }
        let accent = {
            let c = state.ambient_color;
            let f = |x: u8| (x as f32 + (255.0 - x as f32) * 0.4) as u8;
            Color32::from_rgb(f(c.r()), f(c.g()), f(c.b()))
        };
        let ring_c = if state.profile_focused || av_resp.hovered() {
            accent
        } else {
            tl(Color32::from_rgb(0x3A, 0x3A, 0x46), Color32::from_rgb(0xC6, 0xC6, 0xD0))
        };

        let mut dance_scale_x = 1.0f32;
        let mut dance_scale_y = 1.0f32;
        let mut dance_offset = Vec2::ZERO;
        let mut dance_rotation = 0.0f32;

        if let Some(start_t) = state.profile_click_time {
            let elapsed = t - start_t;
            let duration = 0.35f32;
            if elapsed < duration {
                let p = elapsed / duration;
                let squish = (elapsed * 35.0).sin() * 0.18 * (1.0 - p);
                dance_scale_x = 1.0 + squish;
                dance_scale_y = 1.0 - squish;
                dance_offset.y = -(elapsed * 25.0).sin().abs() * 8.0 * scale_factor * (1.0 - p);
                dance_offset.x = (elapsed * 20.0).cos() * 5.0 * scale_factor * (1.0 - p);
                dance_rotation = (elapsed * 30.0).sin() * 0.15 * (1.0 - p);
                ui.ctx().request_repaint();
            } else {
                state.profile_click_time = None;
                action = CarouselAction::OpenProfile;
            }
        }

        let cos_r = dance_rotation.cos();
        let sin_r = dance_rotation.sin();
        let transform_pt = |pt: egui::Pos2| -> egui::Pos2 {
            let scaled = scaled_av + (pt - scaled_av) * Vec2::new(dance_scale_x, dance_scale_y);
            let rotated = scaled_av + Vec2::new(
                (scaled.x - scaled_av.x) * cos_r - (scaled.y - scaled_av.y) * sin_r,
                (scaled.x - scaled_av.x) * sin_r + (scaled.y - scaled_av.y) * cos_r,
            );
            rotated + dance_offset
        };

        let transformed_av = transform_pt(scaled_av);

        let show_glow = state.profile_focused || av_resp.hovered();
        if show_glow {
            let final_glow_alpha = (100.0 * ui_opacity).clamp(0.0, 255.0) as u8;
            let final_border_alpha = (255.0 * ui_opacity).clamp(0.0, 255.0) as u8;
            draw_gradient_circle(&painter, transformed_av, scaled_av_r, 6.0 * scale_factor, t, final_glow_alpha);
            draw_gradient_circle(&painter, transformed_av, scaled_av_r - 1.5 * scale_factor, 1.5 * scale_factor, t, final_border_alpha);
        }

        let av_bg = tl(Color32::from_rgb(0x0C, 0x0C, 0x12), Color32::from_rgb(0xFF, 0xFF, 0xFF));
        painter.circle_filled(transformed_av, (scaled_av_r + 2.0 * scale_factor) * dance_scale_x.max(dance_scale_y), Color32::from_rgba_unmultiplied(av_bg.r(), av_bg.g(), av_bg.b(), top_alpha));
        match profile_tex {
            Some(tid) => {
                let mut mesh = egui::epaint::Mesh::with_texture(tid);
                let segs = 40;
                let tint = Color32::from_white_alpha(top_alpha);
                mesh.vertices.push(egui::epaint::Vertex { pos: transformed_av, uv: egui::pos2(0.5, 0.5), color: tint });
                for i in 0..=segs {
                    let ang = (i as f32 / segs as f32) * std::f32::consts::TAU;
                    let (s, c) = ang.sin_cos();
                    let orig_pos = scaled_av + Vec2::new(c, s) * scaled_av_r;
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos: transform_pt(orig_pos),
                        uv: egui::pos2(0.5 + c * 0.5, 0.5 + s * 0.5),
                        color: tint,
                    });
                }
                let n = mesh.vertices.len() as u32;
                for i in 1..n - 1 {
                    mesh.indices.extend_from_slice(&[0, i, i + 1]);
                }
                mesh.indices.extend_from_slice(&[0, n - 1, 1]);
                painter.add(egui::Shape::mesh(mesh));
            }
            None => {
                painter.circle_filled(transformed_av, scaled_av_r * dance_scale_x.max(dance_scale_y), Color32::from_rgba_unmultiplied(col_surface.r(), col_surface.g(), col_surface.b(), top_alpha));
                painter.text(transformed_av, egui::Align2::CENTER_CENTER, "＋", FontId::proportional(scaled_av_r * 0.9 * dance_scale_x.max(dance_scale_y)), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), top_alpha));
            }
        }
        if !show_glow {
            painter.circle_stroke(transformed_av, scaled_av_r, Stroke::new(2.0 * scale_factor, Color32::from_rgba_unmultiplied(ring_c.r(), ring_c.g(), ring_c.b(), top_alpha)));
        }

        if state.profile_focused || av_resp.hovered() {
            let name_pos = egui::pos2(scaled_av.x + scaled_av_r + 12.0 * scale_factor, scaled_av.y) + dance_offset;
            shadowed_text(&painter, name_pos, egui::Align2::LEFT_CENTER, profile_name, FontId::proportional(16.0 * top_s * scale_factor), Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), top_alpha), true);
        }

        let now = chrono::Local::now();
        let clock = now.format("%-I:%M %p").to_string();
        let clock_pos = egui::pos2(bg_rect.max.x - 34.0 * top_s, bg_rect.min.y + 34.0 * top_s);
        let scaled_clock = screen_center + (clock_pos - screen_center) * scale_factor;
        shadowed_text(&painter, scaled_clock, egui::Align2::RIGHT_CENTER, &clock, FontId::proportional(20.0 * top_s * scale_factor), Color32::from_rgba_unmultiplied(col_clock.r(), col_clock.g(), col_clock.b(), top_alpha), true);
    }

    if interactive && state.boot_stage == BootStage::None {
        let launch_path = if state.selected < filtered_indices.len() {
            Some(lib.games[filtered_indices[state.selected]].path.to_string_lossy().to_string())
        } else {
            None
        };
        let is_playing = if state.selected < filtered_indices.len() {
            playing == Some(filtered_indices[state.selected])
        } else {
            false
        };
        handle_input(
            state,
            last_input,
            ui,
            &mut action,
            n_items,
            is_running,
            is_playing,
            theme,
            launch_path,
        );
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
    n_items: usize,
    is_running: bool,
    is_playing: bool,
    theme: crate::app_settings::CarouselTheme,
    launch_path: Option<String>,
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

    let x_down_now = ui.input(|i| i.key_down(egui::Key::X))
        || (last_input.connected && last_input.is(crate::controller_config::SwitchButton::X));
    let x_edge = x_down_now && !state.x_held;
    state.x_held = x_down_now;

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

    if state.profile_focused {
        if select && state.profile_click_time.is_none() {
            state.profile_click_time = Some(ui.input(|i| i.time) as f32);
        }
        if back || down || left || right {
            state.profile_focused = false;
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
        } else if state.selected < n_items - 1 {
            state.selected += 1;
        }
    }
    if up {
        if state.active_dock {
            state.active_dock = false;
        } else {
            state.profile_focused = true;
        }
    }
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
                    2 => {
                        if let Some(path) = &launch_path {
                            CarouselAction::Launch(path.clone())
                        } else {
                            CarouselAction::AddFolder
                        }
                    }
                    3 => CarouselAction::StopEmulation,
                    4 => CarouselAction::OpenController,
                    5 => CarouselAction::OpenSettings,
                    7 => CarouselAction::OpenDebug,
                    8 => CarouselAction::Quit,
                    _ => CarouselAction::None,
                };
            }
        } else if is_playing {
            *action = CarouselAction::Resume;
        } else if let Some(path) = &launch_path {
            *action = CarouselAction::Launch(path.clone());
        } else {
            *action = CarouselAction::AddFolder;
        }
    }
    if back {
        if state.active_dock {
            state.active_dock = false;
        }
    }
    if !state.active_dock {
        if is_running {
            if stop {
                *action = CarouselAction::StopEmulation;
            }
        } else if x_edge {
            if let Some(p) = &launch_path {
                *action = CarouselAction::ToggleFavorite(p.clone());
            }
        }
    }
}
