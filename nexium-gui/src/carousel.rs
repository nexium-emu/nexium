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
    pub a_held: bool,
    pub a_edge: bool,
    pub x_edge: bool,
    pub b_held: bool,
    pub b_edge: bool,
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
    pub profile_push_at: Option<f32>,
    pub pending_center: Option<usize>,
    pub sel_held: bool,
    pub sel_edge: bool,
    pub game_menu_open: bool,
    pub game_menu_sel: usize,
    pub game_menu_anim: f32,
    pub dl_dialog: bool,
    pub active_list: Option<usize>,
    pub list_anim: f32,
}

impl CarouselState {
    pub fn new() -> Self {
        Self {
            selected: CS_FRONT,
            scroll_offset: CS_FRONT as f32,
            hover_scale: 1.0,
            ambient_color: Color32::from_rgb(0x2F, 0xB4, 0xEF),
            active_dock: false,
            dock_selected: 0,
            dock_anim: 0.0,
            dock_focus: 0.0,
            theme_t: 0.0,
            x_held: false,
            a_held: false,
            a_edge: false,
            x_edge: false,
            b_held: false,
            b_edge: false,
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
            profile_push_at: None,
            pending_center: None,
            sel_held: false,
            sel_edge: false,
            game_menu_open: false,
            game_menu_sel: 0,
            game_menu_anim: 0.0,
            dl_dialog: false,
            active_list: None,
            list_anim: 0.0,
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
    DownloadIcon(String),
    OpenShop,
    OpenCarouselSettings,
}

const DOCK_COUNT: usize = 10;

/// Number of special cards before the games (the Carousel Settings cog).
/// Set to 0 to hide it (WIP), 1 to show it.
pub const CS_FRONT: usize = 1;

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}


fn network_kind() -> u8 {
    use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last = LAST.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < 4 {
        return CACHE.load(Ordering::Relaxed);
    }
    LAST.store(now, Ordering::Relaxed);
    let kind = detect_network_kind();
    CACHE.store(kind, Ordering::Relaxed);
    kind
}

#[cfg(target_os = "linux")]
fn detect_network_kind() -> u8 {
    let Ok(entries) = std::fs::read_dir("/sys/class/net") else {
        return 0;
    };
    let mut result = 0u8;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name == "lo" || name.starts_with("docker") || name.starts_with("veth") || name.starts_with("br-") {
            continue;
        }
        let p = e.path();
        let oper = std::fs::read_to_string(p.join("operstate")).unwrap_or_default();
        let carrier = std::fs::read_to_string(p.join("carrier")).unwrap_or_default();
        if oper.trim() != "up" && carrier.trim() != "1" {
            continue;
        }
        if p.join("wireless").exists() || name.starts_with("wl") {
            return 2;
        }
        result = 1;
    }
    result
}

#[cfg(not(target_os = "linux"))]
fn detect_network_kind() -> u8 {
    0
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
        BackdropTheme::Space => draw_space_backdrop(painter, rect, color, t, opacity),
        BackdropTheme::None => {
            if opacity <= 0.001 { return; }
            let a = (opacity * 255.0) as u8;
            painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a));
        }
    }
}

fn draw_space_backdrop(painter: &egui::Painter, rect: egui::Rect, color: Color32, t: f32, opacity: f32) {
    if opacity <= 0.001 {
        return;
    }
    let a = |x: f32| (x * opacity).clamp(0.0, 255.0) as u8;
    let sh = |c: Color32, f: f32| -> Color32 {
        let adj = |v: u8| if f >= 0.0 { (v as f32 + (255.0 - v as f32) * f) as u8 } else { (v as f32 * (1.0 + f)) as u8 };
        Color32::from_rgb(adj(c.r()), adj(c.g()), adj(c.b()))
    };
    let hash = |i: u32| -> f32 {
        let x = ((i.wrapping_mul(2654435761)) ^ 0x9E3779B9) as f32;
        (x.sin() * 43758.547).fract().abs()
    };
    let w = rect.width();
    let h = rect.height();

    // deep-space background, faintly tinted by the theme colour
    let deep = sh(color, -0.86);
    painter.rect_filled(rect, Rounding::ZERO, Color32::from_rgba_unmultiplied(deep.r().max(4), deep.g().max(5), deep.b().max(8), a(255.0)));

    // starfield (twinkling), some tinted toward the theme colour
    for i in 0..160u32 {
        let sx = rect.min.x + hash(i * 2) * w;
        let sy = rect.min.y + hash(i * 2 + 1) * h * 0.80;
        let tw = 0.3 + 0.7 * (0.5 + 0.5 * (t * 1.7 + hash(i * 3) * 30.0).sin());
        let r = 0.6 + hash(i * 5) * 1.8;
        let tinted = hash(i * 7) > 0.55;
        let col = if tinted { sh(color, 0.55) } else { Color32::from_rgb(0xFF, 0xFF, 0xFF) };
        painter.circle_filled(egui::pos2(sx, sy), r, Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), a(210.0 * tw)));
    }

    // shooting stars across the top — one every ~3.4s, tinted by theme
    let streak = sh(color, 0.6);
    let period = 3.4;
    let idx = (t / period).floor();
    let ph = (t / period).fract();
    if ph < 0.16 {
        let prog = ph / 0.16;
        let sy0 = rect.min.y + h * (0.05 + hash(idx as u32 * 13 + 1) * 0.32);
        let dir = if hash(idx as u32 * 17) > 0.5 { 1.0 } else { -1.0 };
        let sx0 = if dir > 0.0 { rect.min.x - 60.0 + prog * (w + 120.0) } else { rect.max.x + 60.0 - prog * (w + 120.0) };
        let head = egui::pos2(sx0, sy0 + prog * h * 0.10);
        let fade = (prog * std::f32::consts::PI).sin();
        for s in 0..9 {
            let ft = s as f32 / 9.0;
            let px = head - Vec2::new(dir * 15.0 * s as f32, 4.5 * s as f32);
            painter.circle_filled(px, 1.9 * (1.0 - ft), Color32::from_rgba_unmultiplied(streak.r(), streak.g(), streak.b(), a(235.0 * fade * (1.0 - ft))));
        }
        painter.circle_filled(head, 2.6, Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, a(255.0 * fade)));
    }

    // rotating planet along the bottom, coloured from the theme
    let pr = w * 0.95;
    let pc = egui::pos2(rect.center().x, rect.max.y + pr * 0.58);
    let ocean = sh(color, -0.42);
    let land = sh(color, 0.28);
    let lit = sh(color, 0.14);
    // atmosphere glow
    for g in 0..6 {
        let e = (6 - g) as f32 * 5.0;
        let glow = sh(color, 0.5);
        painter.circle_stroke(pc, pr + e, Stroke::new(4.5, Color32::from_rgba_unmultiplied(glow.r(), glow.g(), glow.b(), a(18.0 * (1.0 - g as f32 / 6.0)))));
    }
    painter.circle_filled(pc, pr, Color32::from_rgba_unmultiplied(ocean.r(), ocean.g(), ocean.b(), a(255.0)));
    // soft lit hemisphere (sun upper-left) — clipped inside the planet
    let planet_clip = painter.with_clip_rect(egui::Rect::from_center_size(pc, Vec2::splat(pr * 2.0)));
    planet_clip.circle_filled(pc - Vec2::new(pr * 0.28, pr * 0.22), pr * 0.86, Color32::from_rgba_unmultiplied(lit.r(), lit.g(), lit.b(), a(90.0)));

    // continents (lat/long projected onto the sphere; longitude scrolls = rotation)
    let conts: [(f32, f32, f32); 7] = [
        (0.0, 0.35, 1.0),
        (1.0, -0.05, 0.8),
        (2.1, 0.15, 1.1),
        (3.0, 0.45, 0.7),
        (3.9, -0.2, 0.9),
        (4.8, 0.05, 1.0),
        (5.6, 0.28, 0.75),
    ];
    for (bl, lat, sz) in conts {
        let lam = bl + t * 0.09;
        let coslam = lam.cos();
        if coslam <= 0.12 {
            continue;
        }
        let sx = pc.x + pr * lat.cos() * lam.sin();
        let sy = pc.y - pr * lat.sin();
        if sy > rect.max.y + pr * 0.06 {
            continue;
        }
        let cr = pr * 0.11 * sz * (0.4 + 0.6 * coslam);
        // darken toward the trailing (right) edge for a subtle terminator
        let shade = (0.55 + 0.45 * ((sx - pc.x) / pr * -1.0 + 0.5)).clamp(0.35, 1.0);
        planet_clip.circle_filled(egui::pos2(sx, sy), cr, Color32::from_rgba_unmultiplied((land.r() as f32 * shade) as u8, (land.g() as f32 * shade) as u8, (land.b() as f32 * shade) as u8, a(255.0)));
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
    let steps = 96usize;

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
    ib: &mut Option<crate::input::InputBackend>,
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
    eu_dates: bool,
    dockbar_theme: crate::app_settings::DockbarTheme,
    carousel_order: &[crate::app_settings::CarouselRef],
    lists: &[crate::app_settings::GameList],
    icon_reveal: Option<(usize, f32, Option<egui::TextureHandle>)>,
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
    let order_rank: std::collections::HashMap<&std::path::PathBuf, usize> = carousel_order
        .iter()
        .filter_map(|e| match e { crate::app_settings::CarouselRef::Game(p) => Some(p), _ => None })
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    filtered_indices.sort_by_key(|&idx| order_rank.get(&lib.games[idx].path).copied().unwrap_or(usize::MAX));
    filtered_indices.sort_by(|&a, &b| {
        let fa = favorites.iter().any(|p| *p == lib.games[a].path);
        let fb = favorites.iter().any(|p| *p == lib.games[b].path);
        fb.cmp(&fa)
    });

    let in_list = state.active_list.filter(|&li| li < lists.len());
    let list_games: Vec<usize> = if let Some(li) = in_list {
        lists[li].games.iter().filter_map(|pth| lib.games.iter().position(|g| &g.path == pth)).collect()
    } else {
        Vec::new()
    };

    let list_order: Vec<usize> = if in_list.is_none() && state.search_buf.is_empty() {
        let mut ord: Vec<usize> = Vec::new();
        for e in carousel_order {
            if let crate::app_settings::CarouselRef::List(nm) = e {
                if let Some(li) = lists.iter().position(|l| &l.name == nm) {
                    if !ord.contains(&li) { ord.push(li); }
                }
            }
        }
        for li in 0..lists.len() {
            if !ord.contains(&li) { ord.push(li); }
        }
        ord
    } else {
        Vec::new()
    };
    let n_lists = list_order.len();
    let front = CS_FRONT;

    if in_list.is_none() {
        if let Some(lib_idx) = state.pending_center.take() {
            if let Some(pos) = filtered_indices.iter().position(|&x| x == lib_idx) {
                state.selected = pos + front + n_lists;
                state.scroll_offset = (pos + front + n_lists) as f32;
            }
        }
    }

    let games_base = if in_list.is_some() { front } else { front + n_lists };
    let n_games = if in_list.is_some() { list_games.len() } else { filtered_indices.len() };
    let n_items = if in_list.is_some() { front + list_games.len() } else { games_base + n_games + 1 };
    let game_src = if in_list.is_some() { &list_games } else { &filtered_indices };
    let game_of = |i: usize| -> Option<usize> {
        if i >= games_base && i < games_base + n_games {
            game_src.get(i - games_base).copied()
        } else {
            None
        }
    };
    let list_of = |i: usize| -> Option<usize> {
        if in_list.is_none() && i >= front && i < front + n_lists {
            list_order.get(i - front).copied()
        } else {
            None
        }
    };

    let bg_rect = ui.max_rect();
    let t = ui.input(|i| i.time) as f32;
    let dt = ui.input(|i| i.stable_dt).min(0.1);
    state.list_anim += ((if in_list.is_some() { 1.0 } else { 0.0 }) - state.list_anim) * (dt * 8.0).min(1.0);

    // Space is always a dark sky, so keep the UI bright on it (no black-on-black).
    let space_bg = backdrop_theme == crate::app_settings::BackdropTheme::Space;
    let theme_target = if light_mode && !space_bg { 1.0 } else { 0.0 };
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

    let target_color = if game_of(state.selected).is_none() {
        Color32::from_rgb(0x35, 0x38, 0x42)
    } else if theme == crate::app_settings::CarouselTheme::Rgb {
        let speed = 0.025;
        let hue = (t * speed) % 1.0;
        let (cr, cg, cb) = hsv_to_rgb(hue, 0.85, 0.85);
        Color32::from_rgb(cr, cg, cb)
    } else {
        match theme.color() {
            Some((r, g, b)) => Color32::from_rgb(r, g, b),
            None => match game_of(state.selected) {
                Some(gi) => lib.games[gi].dominant_color,
                None => Color32::from_rgb(0x35, 0x38, 0x42),
            },
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
    let mut want_download: Option<String> = None;
    let mut want_close = false;
    let mut fav_first: Option<egui::Rect> = None;
    let mut lib_first: Option<egui::Rect> = None;
    // Show as many cards as fit the window so wide/stretched displays fill out.
    let cull_max = (((bg_rect.max.x - hero_cx) / stride) + 1.6).max(4.6);
    let fade_start = cull_max - 1.6;
    for i in 0..n_items {
        let diff     = i as f32 - state.scroll_offset;
        let abs_diff = diff.abs();
        if abs_diff > cull_max && state.boot_stage == BootStage::None { continue; }

        let scale   = (1.0 - abs_diff * 0.05).max(0.82);
        let mut alpha_f = (1.0 - abs_diff * 0.03).clamp(0.0, 1.0);
        if abs_diff > fade_start {
            alpha_f *= ((cull_max - abs_diff) / (cull_max - fade_start)).clamp(0.0, 1.0);
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

        if in_list.is_some() && state.list_anim < 0.999 {
            let pop = (state.list_anim * (n_items as f32 + 1.5) - i as f32).clamp(0.0, 1.0);
            let e = pop * pop * (3.0 - 2.0 * pop);
            alpha_f *= e;
            draw_sz *= 0.5 + 0.5 * e;
            draw_cx += (1.0 - e) * hero_size * 0.22;
            draw_cy += (1.0 - e) * hero_size * 0.12;
        }

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

        let is_exit_list = in_list.is_some() && i == 0;
        let is_settings = in_list.is_none() && front > 0 && i == 0;
        let is_add_dir = in_list.is_none() && i == n_items - 1;
        let card_game = game_of(i);
        let resp = ui.interact(draw_rect, egui::Id::new(("carousel_card", i)), Sense::click());
        if interactive && resp.clicked() && !state.drag_moved && state.boot_stage == BootStage::None {
            if state.selected == i {
                if state.active_dock {
                    state.active_dock = false;
                } else if is_exit_list {
                    state.active_list = None;
                    state.selected = CS_FRONT;
                    state.scroll_offset = CS_FRONT as f32;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                } else if is_settings {
                    action = CarouselAction::OpenCarouselSettings;
                } else if let Some(li) = list_of(i) {
                    state.active_list = Some(li);
                    state.selected = 1;
                    state.scroll_offset = 1.0;
                    state.list_anim = 0.0;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                } else if is_add_dir {
                    action = CarouselAction::AddFolder;
                } else if let Some(gi) = card_game {
                    if playing == Some(gi) {
                        action = CarouselAction::Resume;
                    } else if is_running {
                        action = CarouselAction::Launch(lib.games[gi].path.to_string_lossy().to_string());
                    } else {
                        state.boot_stage = BootStage::Transitioning {
                            game_index: i,
                            start_time: t,
                            launch_path: lib.games[gi].path.to_string_lossy().to_string(),
                        };
                    }
                }
            } else {
                state.selected = i;
                state.active_dock = false;
            }
        }

        if interactive && card_game.is_some() && state.boot_stage == BootStage::None {
            let gi = card_game.unwrap();
            let path_string = lib.games[gi].path.to_string_lossy().to_string();
            let favd = favorites.iter().any(|p| *p == lib.games[gi].path);
            let is_playing_card = playing == Some(gi);
            resp.context_menu(|ui| {
                if is_playing_card && ui.button("⏹  Close Game").clicked() {
                    want_close = true;
                    ui.close_menu();
                }
                let label = if favd { "★  Unfavorite Game" } else { "☆  Favorite Game" };
                if ui.button(label).clicked() {
                    want_fav = Some(path_string.clone());
                    ui.close_menu();
                }
                if ui.button("🖼  Download Icon…").clicked() {
                    want_download = Some(path_string.clone());
                    ui.close_menu();
                }
            });
        }

        let tint = Color32::from_white_alpha((alpha_f * 255.0) as u8);
        if is_exit_list {
            let a8 = (alpha_f * 255.0) as u8;
            let bg = Color32::from_rgb(0xC0, 0x39, 0x3B);
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), a8));
            let top = egui::Rect::from_min_max(draw_rect.min, egui::pos2(draw_rect.max.x, draw_rect.center().y));
            painter.rect_filled(top, Rounding { nw: 14.0 * scale_factor, ne: 14.0 * scale_factor, sw: 0.0, se: 0.0 }, Color32::from_white_alpha((alpha_f * 22.0) as u8));
            let ctr = draw_rect.center();
            let r = draw_rect.width() * 0.22;
            let white = Color32::from_white_alpha(a8);
            let w = draw_rect.width() * 0.052;
            painter.line_segment([ctr + Vec2::new(-r, -r), ctr + Vec2::new(r, r)], Stroke::new(w, white));
            painter.line_segment([ctr + Vec2::new(r, -r), ctr + Vec2::new(-r, r)], Stroke::new(w, white));
        } else if is_settings {
            let a8 = (alpha_f * 255.0) as u8;
            // app-icon backdrop (indigo, with a soft top gloss)
            let bg = Color32::from_rgb(0x53, 0x5E, 0xC8);
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), a8));
            let top = egui::Rect::from_min_max(draw_rect.min, egui::pos2(draw_rect.max.x, draw_rect.center().y));
            painter.rect_filled(top, Rounding { nw: 14.0 * scale_factor, ne: 14.0 * scale_factor, sw: 0.0, se: 0.0 }, Color32::from_white_alpha((alpha_f * 22.0) as u8));

            let ctr = draw_rect.center();
            let cw = draw_rect.width();
            let white = Color32::from_white_alpha(a8);
            let body = cw * 0.20;
            let tip = cw * 0.30;
            let teeth = 8;
            for k in 0..teeth {
                let ang = k as f32 / teeth as f32 * std::f32::consts::TAU;
                let radial = Vec2::new(ang.cos(), ang.sin());
                let tangent = Vec2::new(-ang.sin(), ang.cos());
                let hl = (tip - body) * 0.5 + cw * 0.04;
                let rm = tip - hl;
                let hw = cw * 0.055;
                let c = ctr + radial * rm;
                let quad = vec![
                    c + radial * hl + tangent * hw,
                    c + radial * hl - tangent * hw,
                    c - radial * hl - tangent * hw,
                    c - radial * hl + tangent * hw,
                ];
                painter.add(egui::Shape::convex_polygon(quad, white, Stroke::NONE));
            }
            painter.circle_filled(ctr, body, white);
            painter.circle_filled(ctr, body * 0.42, Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), a8));
        } else if let Some(li) = list_of(i) {
            let a8 = (alpha_f * 255.0) as u8;
            let base = tl(Color32::from_rgb(0x14, 0x14, 0x1A), Color32::from_rgb(0xE6, 0xE6, 0xEC));
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a8));
            let acc = state.ambient_color;
            painter.rect_filled(draw_rect.shrink(3.0 * scale_factor), Rounding::same(12.0 * scale_factor), Color32::from_rgba_unmultiplied(acc.r(), acc.g(), acc.b(), (alpha_f * 235.0) as u8));
            let list = &lists[li];
            painter.text(draw_rect.center() - Vec2::new(0.0, draw_rect.height() * 0.06), egui::Align2::CENTER_CENTER, &list.name, FontId::proportional(final_sz * 0.11), Color32::from_white_alpha(a8));
            painter.text(draw_rect.center() + Vec2::new(0.0, draw_rect.height() * 0.10), egui::Align2::CENTER_CENTER, &format!("{} games", list.games.len()), FontId::proportional(final_sz * 0.075), Color32::from_white_alpha((alpha_f * 210.0) as u8));
            if !is_hero {
                painter.rect_stroke(draw_rect, Rounding::same(14.0 * scale_factor), Stroke::new(1.5 * scale_factor, Color32::from_rgba_unmultiplied(0x50, 0x50, 0x60, (alpha_f * 170.0) as u8)));
            }
        } else if is_add_dir {
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
            let real_idx = game_of(i).unwrap();
            let card_fill = tl(Color32::from_rgb(0x14, 0x14, 0x1A), Color32::from_rgb(0xE6, 0xE6, 0xEC));
            painter.rect_filled(draw_rect, Rounding::same(14.0 * scale_factor), Color32::from_rgba_unmultiplied(card_fill.r(), card_fill.g(), card_fill.b(), (alpha_f * 255.0) as u8));
            if let Some(tex) = lib.texture(ctx, real_idx) {
                draw_rounded_image(&painter, tex.id(), draw_rect, 14.0 * scale_factor, tint);
            } else {
                painter.text(draw_rect.center(), egui::Align2::CENTER_CENTER, &lib.games[real_idx].format, FontId::proportional(final_sz * 0.13), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), (alpha_f * 255.0) as u8));
            }

            if let Some((rev_idx, rev_t, old_tex)) = &icon_reveal {
                if *rev_idx == real_idx {
                    const REVEAL_DUR: f32 = 0.95;
                    if *rev_t < REVEAL_DUR {
                        let prog = (rev_t / REVEAL_DUR).clamp(0.0, 1.0);
                        let e = 1.0 - (1.0 - prog).powi(3);
                        let w = draw_rect.width();
                        let h = draw_rect.height();
                        let skew = h * 0.28;
                        let span = w + skew;
                        let head = draw_rect.min.x + span * e;
                        let accent = lib.games[real_idx].dominant_color;

                        if let Some(old) = old_tex {
                            let ahead = egui::Rect::from_min_max(
                                egui::pos2(head, draw_rect.min.y),
                                egui::pos2(draw_rect.max.x, draw_rect.max.y),
                            );
                            if ahead.width() > 1.0 {
                                let clip = painter.with_clip_rect(ahead);
                                draw_rounded_image(&clip, old.id(), draw_rect, 14.0 * scale_factor, tint);
                            }
                        }

                        let pnt = painter.with_clip_rect(draw_rect);
                        let band = (w * 0.10).max(10.0);
                        let top_x = head;
                        let bot_x = head - skew;
                        let paint_quad = vec![
                            egui::pos2(top_x - band, draw_rect.min.y),
                            egui::pos2(top_x, draw_rect.min.y),
                            egui::pos2(bot_x, draw_rect.max.y),
                            egui::pos2(bot_x - band, draw_rect.max.y),
                        ];
                        let a = ((1.0 - (prog - 0.85).max(0.0) / 0.15) * alpha_f).clamp(0.0, 1.0);
                        pnt.add(egui::Shape::convex_polygon(
                            paint_quad,
                            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (230.0 * a) as u8),
                            Stroke::NONE,
                        ));
                        let flick = 0.7 + 0.3 * (t * 40.0).sin();
                        let gloss = vec![
                            egui::pos2(top_x - band * 0.35, draw_rect.min.y),
                            egui::pos2(top_x, draw_rect.min.y),
                            egui::pos2(bot_x, draw_rect.max.y),
                            egui::pos2(bot_x - band * 0.35, draw_rect.max.y),
                        ];
                        pnt.add(egui::Shape::convex_polygon(
                            gloss,
                            Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, (200.0 * a * flick) as u8),
                            Stroke::NONE,
                        ));
                        let sp = (w * 0.02).max(1.0);
                        for k in 0..8 {
                            let ky = (k as f32 / 7.0) * h;
                            let jig = (t * 26.0 + k as f32 * 2.1).sin() * band * 0.4;
                            let px = (top_x - (bot_x - top_x) * (ky / h)) + jig;
                            let rad = sp * (0.5 + 0.5 * ((t * 18.0 + k as f32).sin() * 0.5 + 0.5));
                            pnt.circle_filled(
                                egui::pos2(px, draw_rect.min.y + ky),
                                rad,
                                Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, (180.0 * a) as u8),
                            );
                        }
                        ui.ctx().request_repaint();
                    }
                }
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

            let favd_here = favorites.iter().any(|p| *p == lib.games[real_idx].path);
            if favd_here {
                if fav_first.is_none() || draw_rect.min.x < fav_first.unwrap().min.x {
                    fav_first = Some(draw_rect);
                }
            } else if lib_first.is_none() || draw_rect.min.x < lib_first.unwrap().min.x {
                lib_first = Some(draw_rect);
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

                // player avatar at the top-left, with a little idle dance every ~3s
                let av_sz = draw_rect.width() * 0.17;
                let mut avc = egui::pos2(draw_rect.min.x + av_sz * 0.5 + 10.0 * scale_factor, draw_rect.min.y + av_sz * 0.5 + 10.0 * scale_factor);
                let (mut sx, mut sy) = (1.0f32, 1.0f32);
                let cyc = t.rem_euclid(3.0);
                if cyc < 0.55 {
                    let e = cyc / 0.55;
                    let wob = (e * std::f32::consts::TAU).sin() * 0.12 * (1.0 - e);
                    sx = 1.0 + wob;
                    sy = 1.0 - wob;
                    avc.y -= (e * std::f32::consts::PI).sin() * av_sz * 0.10;
                }
                let arect = egui::Rect::from_center_size(avc, Vec2::new(av_sz * sx, av_sz * sy));
                painter.circle_filled(avc, av_sz * 0.5 * sx.max(sy) + 2.5 * scale_factor, Color32::from_rgba_unmultiplied(0x0C, 0x14, 0x0E, a(230.0)));
                painter.circle_stroke(avc, av_sz * 0.5 * sx.max(sy) + 2.5 * scale_factor, Stroke::new(1.6 * scale_factor, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a(235.0))));
                if let Some(tex) = profile_tex {
                    draw_rounded_image(&painter, tex, arect, av_sz * 0.5, Color32::from_white_alpha(a(255.0)));
                }
            }
        }
    }

    // Section headers above the rows (e.g. "Favorites"). Kept clean — text only.
    let draw_section = |painter: &egui::Painter, rect: egui::Rect, star: bool, label: &str| {
        let hy = rect.min.y - 30.0 * scale_factor;
        let hx = rect.min.x + 4.0 * scale_factor;
        let col = Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), (ui_opacity * 235.0) as u8);
        let fs = 22.0 * scale_factor;
        let mut x = hx;
        if star {
            painter.text(egui::pos2(x, hy), egui::Align2::LEFT_CENTER, "★", FontId::proportional(fs * 0.92), Color32::from_rgba_unmultiplied(0xF5, 0xC1, 0x42, (ui_opacity * 255.0) as u8));
            x += fs * 1.1;
        }
        shadowed_text(painter, egui::pos2(x, hy), egui::Align2::LEFT_CENTER, label, FontId::proportional(fs), col, true);
    };
    if state.boot_stage == BootStage::None && in_list.is_none() {
        if let Some(r) = fav_first {
            draw_section(&painter, r, true, "Favorites");
        }
        if fav_first.is_some() {
            if let Some(r) = lib_first {
                draw_section(&painter, r, false, "Library");
            }
        }
    }

    // Console-OS version tag, bottom-left — plain grey translucent text.
    {
        let os_tag = concat!("NexOS-", env!("NEXIUM_GIT_HASH"), "-v", env!("CARGO_PKG_VERSION"), "  ·  IN-DEV");
        let pos = egui::pos2(bg_rect.min.x + 22.0 * scale_factor, bg_rect.max.y - 18.0 * scale_factor);
        painter.text(
            pos,
            egui::Align2::LEFT_BOTTOM,
            os_tag,
            FontId::proportional(15.5 * scale_factor),
            Color32::from_rgba_unmultiplied(0x80, 0x80, 0x88, (ui_opacity * 150.0) as u8),
        );
    }

    if want_close {
        action = CarouselAction::StopEmulation;
    }
    if let Some(p) = want_fav {
        action = CarouselAction::ToggleFavorite(p);
    }
    if let Some(p) = want_download {
        action = CarouselAction::DownloadIcon(p);
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

    let launch_path = game_of(state.selected).map(|gi| lib.games[gi].path.to_string_lossy().to_string());

    let (sel_title, sel_sub) = if in_list.is_some() && state.selected == 0 {
        let li = in_list.unwrap();
        ("Exit List".to_string(), format!("Back to the main carousel  ·  {}", lists[li].name))
    } else if CS_FRONT > 0 && state.selected == 0 {
        ("Carousel Settings".to_string(), "Create Lists, & build your Carousel how you see fit.".to_string())
    } else if let Some(li) = list_of(state.selected) {
        (lists[li].name.clone(), format!("List  ·  {} games  ·  Press A to open", lists[li].games.len()))
    } else if let Some(gi) = game_of(state.selected) {
        let selected_game = &lib.games[gi];
        let title = selected_game.title.clone();
        let sub = format!(
            "{}  ·  {}  ·  {:.1} MB",
            if selected_game.author.is_empty() { "Unknown" } else { &selected_game.author },
            selected_game.format,
            selected_game.size as f32 / (1024.0 * 1024.0),
        );
        (title, sub)
    } else {
        ("Add Folder".to_string(), "Add folder destinations for your decrypted ROMs".to_string())
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
        ("🛍", "Shop"),
        ("🎨", "Color"),
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
    let dockbar_simple = dockbar_theme == crate::app_settings::DockbarTheme::Simple;
    let dock_bg_alpha = (ui_opacity * if th > 0.5 { 236.0 } else { 200.0 }) as u8;
    if dock_bg_alpha > 0 {
        let round = 32.0 * scale_factor;
        if !dockbar_simple {
            for k in 0..9 {
                let e = (k as f32 + 1.0) * 1.7 * scale_factor;
                let fade = 1.0 - k as f32 / 9.0;
                let a = (ui_opacity * if th > 0.5 { 15.0 } else { 11.0 } * fade) as u8;
                if a == 0 {
                    continue;
                }
                painter.rect_filled(dock_bg.expand(e).translate(Vec2::new(0.0, e * 0.9)), Rounding::same(round + e), Color32::from_black_alpha(a));
            }
        }
        let bar_alpha = if dockbar_simple { (ui_opacity * 190.0) as u8 } else { dock_bg_alpha };
        painter.rect_filled(dock_bg, Rounding::same(round), Color32::from_rgba_premultiplied(col_bar.r(), col_bar.g(), col_bar.b(), bar_alpha));
        if !dockbar_simple {
            let h = dock_bg.height();
            let edge = |dy: f32| -> f32 {
                if dy >= round { 0.0 } else { round - (round * round - (round - dy) * (round - dy)).max(0.0).sqrt() }
            };
            let uv = egui::epaint::WHITE_UV;
            let band_mesh = |top: bool, band: f32, rows: usize, peak: f32, peak_at: f32, black: bool| {
                let mut mesh = egui::epaint::Mesh::default();
                for k in 0..=rows {
                    let f = k as f32 / rows as f32;
                    let dy = 1.0 * scale_factor + f * band;
                    let ins = edge(dy);
                    let y = if top { dock_bg.min.y + dy } else { dock_bg.max.y - dy };
                    let g = if peak_at <= 0.001 {
                        1.0 - f
                    } else if f <= peak_at {
                        f / peak_at
                    } else {
                        (1.0 - f) / (1.0 - peak_at)
                    };
                    let g = g.clamp(0.0, 1.0);
                    let a = (peak * g * g) as u8;
                    let col = if black { Color32::from_black_alpha(a) } else { Color32::from_white_alpha(a) };
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(dock_bg.min.x + ins, y), uv, color: col });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(dock_bg.max.x - ins, y), uv, color: col });
                }
                for k in 0..rows as u32 {
                    let i = k * 2;
                    mesh.indices.extend_from_slice(&[i, i + 1, i + 2, i + 1, i + 3, i + 2]);
                }
                painter.add(egui::Shape::mesh(mesh));
            };
            band_mesh(true, h * 0.60, 26, ui_opacity * if th > 0.5 { 46.0 } else { 60.0 }, 0.26, false);
            band_mesh(false, h * 0.40, 20, ui_opacity * if th > 0.5 { 34.0 } else { 26.0 }, 0.0, true);
        }
        painter.rect_stroke(dock_bg, Rounding::same(round), Stroke::new(1.1 * scale_factor, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), (ui_opacity * 255.0) as u8)));

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
                    if idx == 7 {
                        if state.palette_open {
                            state.palette_open = false;
                            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                        } else {
                            state.palette_open = true;
                            state.palette_selected = crate::app_settings::CarouselTheme::all()
                                .iter()
                                .position(|x| *x == theme)
                                .unwrap_or(0);
                            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
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
                            6 => CarouselAction::OpenShop,
                            8 => CarouselAction::OpenDebug,
                            9 => CarouselAction::Quit,
                            _ => CarouselAction::None,
                        };
                    }
                } else {
                    state.active_dock = true;
                    state.dock_selected = idx;
                }
            }

            let fill = Color32::from_rgba_premultiplied(col_surface.r(), col_surface.g(), col_surface.b(), (ui_opacity * 200.0) as u8);
            let br = base.width() * 0.5;
            painter.circle(base.center(), br, fill, Stroke::new(1.0, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), (ui_opacity * 255.0) as u8)));
            if !dockbar_simple {
                let arc: Vec<egui::Pos2> = (0..=14)
                    .map(|k| {
                        let a = std::f32::consts::PI * (1.08 + 0.84 * (k as f32 / 14.0));
                        base.center() + Vec2::new(a.cos(), a.sin()) * (br - 1.3 * scale_factor)
                    })
                    .collect();
                painter.add(egui::Shape::line(arc, Stroke::new(1.1 * scale_factor, Color32::from_white_alpha((ui_opacity * if th > 0.5 { 90.0 } else { 55.0 }) as u8))));
            }
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
            } else if *label == "Shop" {
                let center = draw.center();
                let u = draw.width() * 0.20;
                let body = egui::Rect::from_min_max(
                    center + Vec2::new(-u, -u * 0.55),
                    center + Vec2::new(u, u * 1.25),
                );
                painter.rect_filled(body, Rounding::same(2.5 * scale_factor), final_icon_color);
                let handle_c = egui::pos2(center.x, body.min.y);
                let arc: Vec<egui::Pos2> = (0..=12)
                    .map(|i| {
                        let a = std::f32::consts::PI * (1.0 + i as f32 / 12.0);
                        handle_c + Vec2::new(a.cos() * u * 0.62, a.sin() * u * 0.62)
                    })
                    .collect();
                painter.add(egui::Shape::line(arc, Stroke::new(1.7 * scale_factor, final_icon_color)));
                let hole = if final_icon_color.r() as u16 + final_icon_color.g() as u16 + final_icon_color.b() as u16 > 384 {
                    Color32::from_rgb(0x18, 0x18, 0x20)
                } else {
                    Color32::from_rgb(0xF2, 0xF2, 0xF6)
                };
                painter.text(body.center() + Vec2::new(0.0, u * 0.1), egui::Align2::CENTER_CENTER, "H", FontId::proportional(u * 1.4), hole);
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

    let gm_target = if state.game_menu_open { 1.0 } else { 0.0 };
    state.game_menu_anim += (gm_target - state.game_menu_anim) * (dt * 18.0).min(1.0);
    if state.game_menu_anim > 0.004 && game_of(state.selected).is_some() {
        let gi = game_of(state.selected).unwrap();
        let favd = favorites.iter().any(|p| *p == lib.games[gi].path);
        let e = { let a = state.game_menu_anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };
        let sc = |p: egui::Pos2| screen_center + (p - screen_center) * scale_factor;
        let accent = state.ambient_color;
        painter.rect_filled(bg_rect, Rounding::ZERO, Color32::from_black_alpha((e * 90.0) as u8));

        let pw = 236.0;
        let rowh = 44.0;
        let ph = rowh * 2.0 + 22.0;
        let slide = (1.0 - e) * 18.0;
        let ax = hero_cx + hero_size * 0.5 + 30.0 + slide;
        let ay = hero_cy - ph * 0.5;
        let panel = egui::Rect::from_min_max(sc(egui::pos2(ax, ay)), sc(egui::pos2(ax + pw, ay + ph)));
        painter.rect_filled(panel.translate(Vec2::new(0.0, 8.0 * scale_factor)), Rounding::same(16.0 * scale_factor), Color32::from_black_alpha((e * 120.0) as u8));
        let pfill = tl(Color32::from_rgb(0x1B, 0x1B, 0x24), Color32::from_rgb(0xFB, 0xFB, 0xFE));
        painter.rect_filled(panel, Rounding::same(16.0 * scale_factor), Color32::from_rgba_unmultiplied(pfill.r(), pfill.g(), pfill.b(), (e * 255.0) as u8));
        painter.rect_stroke(panel, Rounding::same(16.0 * scale_factor), Stroke::new(1.2 * scale_factor, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), (e * 255.0) as u8)));

        let labels = [if favd { "Unfavorite Game" } else { "Favorite Game" }, "Download Icon"];
        for (i, label) in labels.iter().enumerate() {
            let ry0 = ay + 11.0 + i as f32 * rowh;
            let row = egui::Rect::from_min_max(sc(egui::pos2(ax + 8.0, ry0)), sc(egui::pos2(ax + pw - 8.0, ry0 + rowh - 4.0)));
            if i == state.game_menu_sel {
                painter.rect_filled(row, Rounding::same(10.0 * scale_factor), Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (e * 60.0) as u8));
                painter.rect_stroke(row, Rounding::same(10.0 * scale_factor), Stroke::new(1.4 * scale_factor, Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (e * 220.0) as u8)));
            }
            let tc = if i == 0 && favd {
                Color32::from_rgb(0xF5, 0xC1, 0x42)
            } else {
                col_text
            };
            shadowed_text(&painter, sc(egui::pos2(ax + 22.0, ry0 + (rowh - 4.0) * 0.5)), egui::Align2::LEFT_CENTER, label, FontId::proportional(15.0 * scale_factor), Color32::from_rgba_unmultiplied(tc.r(), tc.g(), tc.b(), (e * 255.0) as u8), false);
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
        if interactive && pointer_pressed {
            if let Some(pos) = pointer_pos {
                if !panel.contains(pos) {
                    state.palette_open = false;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        }

        painter.rect_filled(panel, Rounding::same(20.0 * scale_factor), Color32::from_rgba_unmultiplied(col_bar.r(), col_bar.g(), col_bar.b(), a(240.0)));
        painter.rect_stroke(panel, Rounding::same(20.0 * scale_factor), Stroke::new(1.0, Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), a(255.0))));

        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.min.y + 20.0 * scale_factor), egui::Align2::CENTER_CENTER, "Background Color", FontId::proportional(15.0 * scale_factor), Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), a(255.0)), true);

        let sx = scaled_center.x - s_total * 0.5;
        let row_y = scaled_center.y + 6.0 * scale_factor;
        for (i, th) in themes.iter().enumerate() {
            let x = sx + i as f32 * (s_sw + s_gap) + s_sw * 0.5;
            let r = egui::Rect::from_center_size(egui::pos2(x, row_y), Vec2::splat(s_sw));
            if interactive {
                let resp = ui.allocate_rect(r, Sense::click());
                if resp.hovered() {
                    if state.palette_selected != i {
                        state.palette_selected = i;
                        crate::ui_audio::play_move();
                    }
                }
                if resp.clicked() {
                    action = CarouselAction::SetTheme(*th);
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
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
            let sw_stroke = if is_sel {
                Color32::from_rgba_unmultiplied(col_text.r(), col_text.g(), col_text.b(), a(235.0))
            } else {
                Color32::from_rgba_unmultiplied(col_border.r(), col_border.g(), col_border.b(), a(160.0))
            };
            painter.rect_stroke(r, rounding, Stroke::new(1.3, sw_stroke));
        }

        let cur = themes.get(state.palette_selected).copied().unwrap_or_default();
        shadowed_text(&painter, egui::pos2(scaled_center.x, panel.max.y - 18.0 * scale_factor), egui::Align2::CENTER_CENTER, cur.label(), FontId::proportional(14.0 * scale_factor), Color32::from_rgba_unmultiplied(col_muted.r(), col_muted.g(), col_muted.b(), a(255.0)), false);
    }

    if ui_opacity > 0.01 {
        let top_alpha = (ui_opacity * 255.0) as u8;
        let top_s = (bg_rect.height() / 820.0).clamp(1.0, 2.4);
        let av_r = 26.0f32 * top_s;
        let av_center = egui::pos2(bg_rect.min.x + 34.0 * top_s + av_r, bg_rect.min.y + 26.0 * top_s + av_r);
        let scaled_av = screen_center + (av_center - screen_center) * scale_factor;
        let scaled_av_r = av_r * scale_factor;
        let av_rect = egui::Rect::from_center_size(scaled_av, Vec2::splat(scaled_av_r * 2.0));
        let av_resp = ui.interact(av_rect.expand(3.0 * scale_factor), egui::Id::new("carousel_avatar"), Sense::click());
        if interactive && av_resp.clicked() && state.boot_stage == BootStage::None && !state.palette_open && state.profile_click_time.is_none() {
            state.profile_click_time = Some(t);
            crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
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

        // mini "pushed-up" squeeze when the profile is focused from below
        if let Some(pt) = state.profile_push_at {
            let e = t - pt;
            let dur = 0.34f32;
            if e < dur {
                let p = e / dur;
                let s = (p * std::f32::consts::PI).sin() * (1.0 - p * 0.4);
                dance_scale_y += 0.16 * s;
                dance_scale_x -= 0.09 * s;
                dance_offset.y -= 10.0 * scale_factor * s;
                ui.ctx().request_repaint();
            } else {
                state.profile_push_at = None;
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
        let (clock, date) = if eu_dates {
            (now.format("%H:%M").to_string(), now.format("%a  %-d %b").to_string())
        } else {
            (now.format("%-I:%M %p").to_string(), now.format("%a  %b %-d").to_string())
        };
        let cc = Color32::from_rgba_unmultiplied(col_clock.r(), col_clock.g(), col_clock.b(), top_alpha);
        let clock_font = FontId::proportional(20.0 * top_s * scale_factor);
        let date_font = FontId::proportional(13.0 * top_s * scale_factor);

        let net = network_kind();
        let net_x = bg_rect.max.x - 30.0 * top_s;
        let clock_right = if net > 0 { net_x - 34.0 * top_s } else { bg_rect.max.x - 30.0 * top_s };
        let clock_pos = egui::pos2(clock_right, bg_rect.min.y + 34.0 * top_s);
        let scaled_clock = screen_center + (clock_pos - screen_center) * scale_factor;
        shadowed_text(&painter, scaled_clock, egui::Align2::RIGHT_CENTER, &clock, clock_font.clone(), cc, true);

        let clock_w = ui.fonts(|f| f.layout_no_wrap(clock.clone(), clock_font.clone(), cc).size().x) / scale_factor;
        let date_pos = egui::pos2(clock_right - clock_w - 14.0 * top_s, bg_rect.min.y + 34.0 * top_s);
        let scaled_date = screen_center + (date_pos - screen_center) * scale_factor;
        shadowed_text(&painter, scaled_date, egui::Align2::RIGHT_CENTER, &date, date_font, Color32::from_rgba_unmultiplied(col_clock.r(), col_clock.g(), col_clock.b(), (top_alpha as f32 * 0.82) as u8), false);

        if net > 0 {
            let nc = egui::pos2(net_x, bg_rect.min.y + 34.0 * top_s);
            let s = |p: egui::Pos2| screen_center + (p - screen_center) * scale_factor;
            let u = 8.0 * top_s;
            if net == 2 {
                let base = s(nc + Vec2::new(0.0, u * 0.7));
                painter.circle_filled(base, 1.7 * scale_factor, cc);
                for (k, rr) in [0.45f32, 0.8, 1.15].iter().enumerate() {
                    let rad = u * rr * scale_factor;
                    let arc: Vec<egui::Pos2> = (0..=12)
                        .map(|i| {
                            let a = std::f32::consts::PI * (1.25 + 0.5 * (i as f32 / 12.0));
                            base + Vec2::new(a.cos(), a.sin()) * rad
                        })
                        .collect();
                    let al = (top_alpha as f32 * (1.0 - k as f32 * 0.18)) as u8;
                    painter.add(egui::Shape::line(arc, Stroke::new(1.6 * scale_factor, Color32::from_rgba_unmultiplied(cc.r(), cc.g(), cc.b(), al))));
                }
            } else {
                let sx = |x: f32, y: f32| s(nc + Vec2::new(x * u, y * u));
                let st = Stroke::new(2.4 * scale_factor, cc);
                let cable = [
                    (0.0f32, -0.95f32),
                    (0.5, -0.62),
                    (0.5, -0.12),
                    (-0.5, 0.12),
                    (-0.5, 0.62),
                    (0.0, 0.95),
                ];
                let pts: Vec<egui::Pos2> = cable.iter().map(|(x, y)| sx(*x, *y)).collect();
                painter.add(egui::Shape::line(pts, st));
                for (py, dir) in [(-1.28f32, -1.0f32), (1.28f32, 1.0f32)] {
                    let body = egui::Rect::from_center_size(sx(0.0, py), Vec2::new(u * 1.05 * scale_factor, u * 0.74 * scale_factor));
                    painter.rect_filled(body, Rounding::same(1.6 * scale_factor), cc);
                    for k in -1..=1 {
                        let px = k as f32 * 0.3;
                        painter.line_segment([sx(px, py + dir * 0.34), sx(px, py + dir * 0.62)], Stroke::new(1.4 * scale_factor, cc));
                    }
                }
            }
        }

        if last_input.connected {
            let s = top_s * scale_factor;
            let col = Color32::from_rgba_unmultiplied(col_clock.r(), col_clock.g(), col_clock.b(), top_alpha);
            let y = scaled_av.y + scaled_av_r + 24.0 * scale_factor + dance_offset.y;
            let gx = scaled_av.x + dance_offset.x;

            let gw = 22.0 * s;
            let gh = 14.0 * s;
            let glyph_half = 6.0 * s;
            let gap = 9.0 * s;

            let bc = egui::pos2(gx, y);
            if last_input.wired {
                let z = 6.0 * s;
                let bodyr = egui::Rect::from_center_size(bc, Vec2::new(z * 0.95, z * 0.85));
                painter.rect_filled(bodyr, Rounding::same(1.5 * scale_factor), col);
                let prong = Stroke::new(1.6 * scale_factor, col);
                painter.line_segment([egui::pos2(bc.x - 0.30 * z, bc.y - 0.42 * z), egui::pos2(bc.x - 0.30 * z, bc.y - 1.05 * z)], prong);
                painter.line_segment([egui::pos2(bc.x + 0.30 * z, bc.y - 0.42 * z), egui::pos2(bc.x + 0.30 * z, bc.y - 1.05 * z)], prong);
                painter.line_segment([egui::pos2(bc.x, bc.y + 0.42 * z), egui::pos2(bc.x, bc.y + 1.10 * z)], prong);
            } else {
                let bz = 6.5 * s;
                let bt_norm = [
                    (-0.55, 0.5),
                    (0.55, -0.5),
                    (0.0, -1.0),
                    (0.0, 1.0),
                    (0.55, 0.5),
                    (-0.55, -0.5),
                ];
                let bt: Vec<egui::Pos2> = bt_norm
                    .iter()
                    .map(|(px, py)| bc + Vec2::new(px * bz * 0.62, py * bz))
                    .collect();
                painter.add(egui::Shape::line(bt, Stroke::new(1.5 * scale_factor, col)));
            }

            let gbody = egui::Rect::from_min_size(
                egui::pos2(gx - glyph_half - gap - gw, y - gh * 0.5),
                Vec2::new(gw, gh),
            );
            painter.rect_stroke(gbody, Rounding::same(gh * 0.48), Stroke::new(1.6 * scale_factor, col));
            let lc = egui::pos2(gbody.min.x + gw * 0.27, gbody.center().y);
            let arm = 2.3 * s;
            painter.line_segment([lc - Vec2::new(arm, 0.0), lc + Vec2::new(arm, 0.0)], Stroke::new(1.5 * scale_factor, col));
            painter.line_segment([lc - Vec2::new(0.0, arm), lc + Vec2::new(0.0, arm)], Stroke::new(1.5 * scale_factor, col));
            let rc = egui::pos2(gbody.max.x - gw * 0.27, gbody.center().y);
            painter.circle_filled(rc + Vec2::new(1.9 * s, 0.0), 1.3 * s, col);
            painter.circle_filled(rc - Vec2::new(1.9 * s, 0.0), 1.3 * s, col);

            let bw = 22.0 * s;
            let bh = 12.0 * s;
            let nub = 2.5 * s;
            let x = gx + glyph_half + gap;
            let bbody = egui::Rect::from_min_max(
                egui::pos2(x, y - bh * 0.5),
                egui::pos2(x + bw, y + bh * 0.5),
            );
            painter.rect_stroke(bbody, Rounding::same(2.5 * scale_factor), Stroke::new(1.6 * scale_factor, col));
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x + bw, y - bh * 0.22),
                    egui::pos2(x + bw + nub, y + bh * 0.22),
                ),
                Rounding::same(1.0 * scale_factor),
                col,
            );
            let fc = if last_input.charging {
                Color32::from_rgb(0x35, 0xD0, 0x6A)
            } else {
                Color32::from_rgb(0x9A, 0x9A, 0xA6)
            };
            let frac = last_input
                .battery
                .map(|p| (p as f32 / 100.0).clamp(0.06, 1.0))
                .unwrap_or(1.0);
            let inner = bbody.shrink(2.2 * scale_factor);
            let mut f = inner;
            f.set_width(inner.width() * frac);
            painter.rect_filled(f, Rounding::same(1.5 * scale_factor), Color32::from_rgba_unmultiplied(fc.r(), fc.g(), fc.b(), top_alpha));
            if last_input.charging {
                let c = bbody.center();
                let z = 3.4 * s;
                let bolt = vec![
                    c + Vec2::new(0.35 * z, -1.3 * z),
                    c + Vec2::new(-0.45 * z, 0.15 * z),
                    c + Vec2::new(0.1 * z, 0.15 * z),
                    c + Vec2::new(-0.35 * z, 1.3 * z),
                ];
                painter.add(egui::Shape::line(bolt, Stroke::new(1.8 * scale_factor, Color32::from_rgba_unmultiplied(0x10, 0x28, 0x18, top_alpha))));
            }
            let label = match last_input.battery {
                Some(p) => format!("{}%", p),
                None => "--%".to_string(),
            };
            shadowed_text(
                &painter,
                egui::pos2(bbody.max.x + nub + 7.0 * s, y),
                egui::Align2::LEFT_CENTER,
                &label,
                FontId::proportional(13.0 * s),
                col,
                false,
            );
        }
    }

    if interactive && state.boot_stage == BootStage::None {
        let launch_path = game_of(state.selected).map(|gi| lib.games[gi].path.to_string_lossy().to_string());
        let is_playing = game_of(state.selected).map_or(false, |gi| playing == Some(gi));
        let list_selected = list_of(state.selected);
        handle_input(
            state,
            last_input,
            ib,
            ui,
            &mut action,
            n_items,
            is_running,
            is_playing,
            theme,
            launch_path,
            list_selected,
        );
        if let CarouselAction::Launch(path) = &action {
            if !is_running {
                let path = path.clone();
                state.boot_stage = BootStage::Transitioning {
                    game_index: state.selected,
                    start_time: t,
                    launch_path: path,
                };
                action = CarouselAction::None;
            }
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
    ib: &mut Option<crate::input::InputBackend>,
    ui: &mut egui::Ui,
    action: &mut CarouselAction,
    n_items: usize,
    is_running: bool,
    is_playing: bool,
    theme: crate::app_settings::CarouselTheme,
    launch_path: Option<String>,
    list_selected: Option<usize>,
) {
    let mut left   = ui.input(|i| i.key_pressed(egui::Key::ArrowLeft));
    let mut right  = ui.input(|i| i.key_pressed(egui::Key::ArrowRight));
    let mut up     = ui.input(|i| i.key_pressed(egui::Key::ArrowUp));
    let mut down   = ui.input(|i| i.key_pressed(egui::Key::ArrowDown));
    let mut back   = ui.input(|i| i.key_pressed(egui::Key::Escape));

    if last_input.connected {
        use crate::controller_config::SwitchButton;
        if last_input.is(SwitchButton::DLeft)  { left   = true; }
        if last_input.is(SwitchButton::DRight) { right  = true; }
        if last_input.is(SwitchButton::DUp)    { up     = true; }
        if last_input.is(SwitchButton::DDown)  { down   = true; }
        if state.b_edge                        { back   = true; }

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

    let mut from_wheel = false;
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_WHEEL: AtomicU64 = AtomicU64::new(0);
        let now = ui.input(|i| i.time);
        let (wx, wy) = ui.input(|i| (i.smooth_scroll_delta.x, i.smooth_scroll_delta.y));
        let w = if wx.abs() > wy.abs() { wx } else { -wy };
        if w.abs() > 1.5 {
            let last = f64::from_bits(LAST_WHEEL.load(Ordering::Relaxed));
            if now - last > 0.10 {
                if w > 0.0 { right = true; } else { left = true; }
                from_wheel = true;
                LAST_WHEEL.store(now.to_bits(), Ordering::Relaxed);
            }
        }
    }

    let x_edge = state.x_edge;
    let select = state.a_edge;

    if state.game_menu_open {
        if up && state.game_menu_sel > 0 {
            state.game_menu_sel -= 1;
            crate::ui_audio::play_move();
        }
        if down && state.game_menu_sel < 1 {
            state.game_menu_sel += 1;
            crate::ui_audio::play_move();
        }
        if select {
            if let Some(p) = &launch_path {
                *action = if state.game_menu_sel == 0 {
                    CarouselAction::ToggleFavorite(p.clone())
                } else {
                    CarouselAction::DownloadIcon(p.clone())
                };
                crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
            }
            state.game_menu_open = false;
        }
        if back {
            state.game_menu_open = false;
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
        }
        return;
    }

    if state.sel_edge
        && !state.active_dock
        && !state.profile_focused
        && !state.palette_open
        && launch_path.is_some()
    {
        state.game_menu_open = true;
        state.game_menu_sel = 0;
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        return;
    }

    if state.palette_open {
        let themes = crate::app_settings::CarouselTheme::all();
        if left && state.palette_selected > 0 {
            state.palette_selected -= 1;
            crate::ui_audio::play_move();
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(15000, 15000, 35);
                }
            }
        }
        if right && state.palette_selected + 1 < themes.len() {
            state.palette_selected += 1;
            crate::ui_audio::play_move();
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(15000, 15000, 35);
                }
            }
        }
        if select {
            if let Some(th) = themes.get(state.palette_selected) {
                *action = CarouselAction::SetTheme(*th);
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                if last_input.connected {
                    if let Some(ref mut backend) = ib {
                        let _ = backend.rumble(28000, 28000, 60);
                    }
                }
            }
        }
        if back {
            state.palette_open = false;
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(20000, 20000, 45);
                }
            }
        }
        return;
    }

    if state.profile_focused {
        if select && state.profile_click_time.is_none() {
            state.profile_click_time = Some(ui.input(|i| i.time) as f32);
            crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(28000, 28000, 60);
                }
            }
        }
        if back || down || left || right {
            state.profile_focused = false;
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(15000, 15000, 35);
                }
            }
        }
        return;
    }

    let pre_sel = state.selected;
    let pre_dock = state.active_dock;
    let pre_dsel = state.dock_selected;
    let pre_prof = state.profile_focused;

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
            if !state.profile_focused {
                state.profile_push_at = Some(ui.input(|i| i.time) as f32);
            }
            state.profile_focused = true;
        }
    }
    if down && !state.active_dock { state.active_dock = true; }

    if state.selected != pre_sel
        || state.active_dock != pre_dock
        || state.dock_selected != pre_dsel
        || state.profile_focused != pre_prof
    {
        crate::ui_audio::play_move();
        if last_input.connected && !from_wheel {
            if let Some(ref mut backend) = ib {
                let _ = backend.rumble(15000, 15000, 35);
            }
        }
    }

    if select {
        let is_game_launch = !state.active_dock && !is_playing && launch_path.is_some();
        crate::ui_audio::play(if is_game_launch {
            crate::ui_audio::Sfx::WhistleSquish
        } else {
            crate::ui_audio::Sfx::Select
        });
        if last_input.connected {
            if let Some(ref mut backend) = ib {
                let _ = backend.rumble(28000, 28000, 60);
            }
        }
        if state.active_dock {
            if state.dock_selected == 7 {
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
                    6 => CarouselAction::OpenShop,
                    8 => CarouselAction::OpenDebug,
                    9 => CarouselAction::Quit,
                    _ => CarouselAction::None,
                };
            }
        } else if state.active_list.is_some() && state.selected == 0 {
            state.active_list = None;
            state.selected = CS_FRONT;
            state.scroll_offset = CS_FRONT as f32;
        } else if CS_FRONT > 0 && state.selected == 0 {
            *action = CarouselAction::OpenCarouselSettings;
        } else if let Some(li) = list_selected {
            state.active_list = Some(li);
            state.selected = 1;
            state.scroll_offset = 1.0;
            state.list_anim = 0.0;
        } else if is_playing {
            *action = CarouselAction::Resume;
        } else if let Some(path) = &launch_path {
            *action = CarouselAction::Launch(path.clone());
        } else {
            *action = CarouselAction::AddFolder;
        }
    }
    if back {
        crate::ui_audio::play(crate::ui_audio::Sfx::Back);
        if last_input.connected {
            if let Some(ref mut backend) = ib {
                let _ = backend.rumble(20000, 20000, 45);
            }
        }
        if state.active_dock {
            state.active_dock = false;
        } else if state.active_list.is_some() {
            state.active_list = None;
            state.selected = CS_FRONT;
            state.scroll_offset = CS_FRONT as f32;
        }
    }
    if !state.active_dock && x_edge {
        if is_running && is_playing {
            *action = CarouselAction::StopEmulation;
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(20000, 20000, 45);
                }
            }
        } else if let Some(p) = &launch_path {
            *action = CarouselAction::ToggleFavorite(p.clone());
            if last_input.connected {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(24000, 24000, 50);
                }
            }
        }
    }
}
