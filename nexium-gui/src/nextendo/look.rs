use eframe::egui;
use eframe::egui::{Color32, CornerRadius, FontId, Pos2, Rect, Stroke, Vec2};
use std::collections::HashMap;
use std::sync::Arc;

pub const ONLINE: Color32 = Color32::from_rgb(0x35, 0xD0, 0x6A);
pub const PLAYING: Color32 = Color32::from_rgb(0x2F, 0xB4, 0xEF);
pub const WARNING: Color32 = Color32::from_rgb(0xF5, 0xA6, 0x23);
pub const DANGER: Color32 = Color32::from_rgb(0xE8, 0x40, 0x48);
pub const BADGE: Color32 = Color32::from_rgb(0xE6, 0x2A, 0x3C);

#[derive(Clone, Copy)]
pub struct Palette {
    pub text: Color32,
    pub muted: Color32,
    pub faint: Color32,
    pub panel: Color32,
    pub panel_hover: Color32,
    pub input: Color32,
    pub border: Color32,
    pub selected: Color32,
    pub light: bool,
}

pub fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
        mix(a.a(), b.a()),
    )
}

pub fn alpha(color: Color32, opacity: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        color.r(),
        color.g(),
        color.b(),
        (color.a() as f32 * opacity.clamp(0.0, 1.0)) as u8,
    )
}

pub fn brighten(color: Color32, amount: f32) -> Color32 {
    let lift = |x: u8| (x as f32 + (255.0 - x as f32) * amount) as u8;
    Color32::from_rgb(lift(color.r()), lift(color.g()), lift(color.b()))
}

pub fn palette(light: f32) -> Palette {
    Palette {
        text: lerp(
            Color32::from_rgb(0xEC, 0xEC, 0xF0),
            Color32::from_rgb(0x1E, 0x1E, 0x28),
            light,
        ),
        muted: lerp(
            Color32::from_rgb(0x8A, 0x8A, 0x98),
            Color32::from_rgb(0x60, 0x60, 0x6A),
            light,
        ),
        faint: lerp(
            Color32::from_rgb(0x5C, 0x5C, 0x6A),
            Color32::from_rgb(0x9A, 0x9A, 0xA6),
            light,
        ),
        panel: lerp(
            Color32::from_rgb(0x16, 0x16, 0x1E),
            Color32::from_rgb(0xFF, 0xFF, 0xFF),
            light,
        ),
        panel_hover: lerp(
            Color32::from_rgb(0x1C, 0x1C, 0x26),
            Color32::from_rgb(0xF3, 0xF3, 0xF8),
            light,
        ),
        input: lerp(
            Color32::from_rgb(0x20, 0x20, 0x2A),
            Color32::from_rgb(0xE4, 0xE4, 0xEC),
            light,
        ),
        border: lerp(
            Color32::from_rgb(0x30, 0x30, 0x3C),
            Color32::from_rgb(0xC6, 0xC6, 0xD0),
            light,
        ),
        selected: lerp(
            Color32::from_rgb(0x1E, 0x1E, 0x28),
            Color32::from_rgb(0xDD, 0xDD, 0xE6),
            light,
        ),
        light: light > 0.5,
    }
}

pub fn readable_on(fill: Color32) -> Color32 {
    let luma = 0.299 * fill.r() as f32 + 0.587 * fill.g() as f32 + 0.114 * fill.b() as f32;
    if luma > 160.0 {
        Color32::from_rgb(0x12, 0x14, 0x1C)
    } else {
        Color32::WHITE
    }
}

pub fn card(painter: &egui::Painter, rect: Rect, radius: f32, fill: Color32, border: Color32, width: f32) {
    let rounding = CornerRadius::from(radius);
    painter.rect_filled(rect, rounding, fill);
    painter.rect_stroke(rect, rounding, Stroke::new(width, border), egui::StrokeKind::Outside);
}

pub fn soft_shadow(painter: &egui::Painter, rect: Rect, radius: f32, strength: f32) {
    for step in 0..8 {
        let spread = (step as f32 + 1.0) * 1.6;
        let fade = 1.0 - step as f32 / 8.0;
        painter.rect_filled(
            rect.expand(spread).translate(Vec2::new(0.0, spread * 0.8)),
            CornerRadius::from(radius + spread),
            Color32::from_black_alpha((strength * 14.0 * fade) as u8),
        );
    }
}

pub fn status_dot(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32, ring: Color32) {
    painter.circle_filled(center, radius + radius * 0.45, ring);
    painter.circle_filled(center, radius, color);
}

pub fn network_glyph(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    let stroke = Stroke::new((radius * 0.13).max(1.2), color);
    painter.circle_stroke(center, radius, stroke);
    let meridian: Vec<Pos2> = (0..40)
        .map(|step| {
            let angle = step as f32 / 40.0 * std::f32::consts::TAU;
            center + Vec2::new(angle.cos() * radius * 0.42, angle.sin() * radius)
        })
        .collect();
    painter.add(egui::Shape::closed_line(meridian, stroke));
    painter.line_segment(
        [center - Vec2::new(radius, 0.0), center + Vec2::new(radius, 0.0)],
        stroke,
    );
    for offset in [-0.5f32, 0.5] {
        let y = center.y + radius * offset;
        let half = radius * (1.0 - offset * offset).sqrt();
        painter.line_segment(
            [Pos2::new(center.x - half, y), Pos2::new(center.x + half, y)],
            Stroke::new(stroke.width * 0.8, color),
        );
    }
    painter.circle_filled(
        center + Vec2::new(radius * 0.72, -radius * 0.72),
        radius * 0.24,
        color,
    );
}

pub fn person_glyph(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    painter.circle_filled(center - Vec2::new(0.0, radius * 0.32), radius * 0.36, color);
    let body: Vec<Pos2> = (0..=20)
        .map(|step| {
            let angle = std::f32::consts::PI + step as f32 / 20.0 * std::f32::consts::PI;
            center + Vec2::new(angle.cos() * radius * 0.62, radius * 0.62 + angle.sin() * radius * 0.5)
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(body, color, Stroke::NONE));
}

pub fn spinner(painter: &egui::Painter, center: Pos2, radius: f32, time: f64, color: Color32) {
    let start = (time * 4.2) as f32;
    let sweep = 1.6 + 0.8 * ((time * 2.1) as f32).sin();
    let points: Vec<Pos2> = (0..=28)
        .map(|step| {
            let angle = start + sweep * step as f32 / 28.0;
            center + Vec2::new(angle.cos(), angle.sin()) * radius
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        Stroke::new((radius * 0.22).max(1.6), color),
    ));
}

pub fn check_glyph(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.16).max(1.4), color);
    painter.line_segment(
        [
            center + Vec2::new(-size * 0.42, 0.0),
            center + Vec2::new(-size * 0.12, size * 0.3),
        ],
        stroke,
    );
    painter.line_segment(
        [
            center + Vec2::new(-size * 0.12, size * 0.3),
            center + Vec2::new(size * 0.45, -size * 0.32),
        ],
        stroke,
    );
}

pub fn cross_glyph(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.15).max(1.4), color);
    let half = size * 0.34;
    painter.line_segment(
        [center + Vec2::new(-half, -half), center + Vec2::new(half, half)],
        stroke,
    );
    painter.line_segment(
        [center + Vec2::new(-half, half), center + Vec2::new(half, -half)],
        stroke,
    );
}

pub fn plus_glyph(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.15).max(1.4), color);
    let half = size * 0.38;
    painter.line_segment([center - Vec2::new(half, 0.0), center + Vec2::new(half, 0.0)], stroke);
    painter.line_segment([center - Vec2::new(0.0, half), center + Vec2::new(0.0, half)], stroke);
}

pub fn copy_glyph(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.11).max(1.2), color);
    let back = Rect::from_center_size(center + Vec2::new(-size * 0.12, -size * 0.12), Vec2::splat(size * 0.56));
    let front = Rect::from_center_size(center + Vec2::new(size * 0.12, size * 0.12), Vec2::splat(size * 0.56));
    painter.rect_stroke(back, CornerRadius::from(size * 0.1), stroke, egui::StrokeKind::Middle);
    painter.rect_stroke(front, CornerRadius::from(size * 0.1), stroke, egui::StrokeKind::Middle);
}

pub fn refresh_glyph(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.13).max(1.3), color);
    let radius = size * 0.36;
    let arc: Vec<Pos2> = (0..=24)
        .map(|step| {
            let angle = -0.6 + step as f32 / 24.0 * 5.0;
            center + Vec2::new(angle.cos(), angle.sin()) * radius
        })
        .collect();
    let tip = *arc.last().unwrap_or(&center);
    painter.add(egui::Shape::line(arc, stroke));
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip + Vec2::new(size * 0.2, -size * 0.02),
            tip + Vec2::new(-size * 0.04, -size * 0.22),
            tip + Vec2::new(-size * 0.06, size * 0.14),
        ],
        color,
        Stroke::NONE,
    ));
}

pub fn counter_badge(painter: &egui::Painter, anchor: Pos2, height: f32, count: usize) {
    let text = if count > 99 { "99+".to_string() } else { count.to_string() };
    let width = (height * (0.55 + 0.32 * text.len() as f32)).max(height);
    let rect = Rect::from_center_size(anchor, Vec2::new(width, height));
    painter.rect_filled(rect, CornerRadius::from(height * 0.5), BADGE);
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        FontId::proportional(height * 0.66),
        Color32::WHITE,
    );
}

pub fn round_image(painter: &egui::Painter, center: Pos2, radius: f32, texture: egui::TextureId, tint: Color32) {
    let mut mesh = egui::epaint::Mesh::with_texture(texture);
    mesh.vertices.push(egui::epaint::Vertex {
        pos: center,
        uv: egui::pos2(0.5, 0.5),
        color: tint,
    });
    let segments = 48;
    for step in 0..=segments {
        let angle = step as f32 / segments as f32 * std::f32::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        mesh.vertices.push(egui::epaint::Vertex {
            pos: center + Vec2::new(cos, sin) * radius,
            uv: egui::pos2(0.5 + cos * 0.5, 0.5 + sin * 0.5),
            color: tint,
        });
    }
    let count = mesh.vertices.len() as u32;
    for index in 1..count - 1 {
        mesh.indices.extend_from_slice(&[0, index, index + 1]);
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub fn initials(name: &str) -> String {
    let mut letters = name
        .split(|ch: char| ch.is_whitespace() || ch == '_' || ch == '-')
        .filter_map(|word| word.chars().next())
        .filter(|ch| ch.is_alphanumeric())
        .take(2)
        .collect::<String>()
        .to_uppercase();
    if letters.is_empty() {
        letters = "?".into();
    }
    letters
}

pub fn identity_color(seed: u64) -> Color32 {
    const COLORS: [Color32; 8] = [
        Color32::from_rgb(0x4C, 0x8D, 0xF6),
        Color32::from_rgb(0xE8, 0x5D, 0x75),
        Color32::from_rgb(0x2B, 0xB6, 0x8C),
        Color32::from_rgb(0xF2, 0x9C, 0x38),
        Color32::from_rgb(0x9B, 0x6B, 0xF2),
        Color32::from_rgb(0x2F, 0xB4, 0xEF),
        Color32::from_rgb(0xD9, 0x5B, 0xC8),
        Color32::from_rgb(0x6F, 0xB4, 0x3C),
    ];
    let mixed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 59;
    COLORS[mixed as usize % COLORS.len()]
}

pub fn avatar(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    texture: Option<egui::TextureId>,
    name: &str,
    seed: u64,
    opacity: f32,
) {
    match texture {
        Some(texture) => round_image(
            painter,
            center,
            radius,
            texture,
            Color32::from_white_alpha((opacity * 255.0) as u8),
        ),
        None => {
            let base = identity_color(seed);
            painter.circle_filled(center, radius, alpha(base, opacity));
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                initials(name),
                FontId::proportional(radius * 0.82),
                alpha(Color32::WHITE, opacity),
            );
        }
    }
}

pub fn ellipsize(ui: &egui::Ui, text: &str, font: &FontId, max_width: f32) -> String {
    let width = |value: &str| {
        ui.fonts_mut(|fonts| {
            fonts
                .layout_no_wrap(value.to_string(), font.clone(), Color32::WHITE)
                .size()
                .x
        })
    };
    if width(text) <= max_width {
        return text.to_string();
    }
    let mut chars: Vec<char> = text.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate: String = chars.iter().collect::<String>() + "…";
        if width(&candidate) <= max_width {
            return candidate;
        }
    }
    "…".into()
}

pub fn text_width(ui: &egui::Ui, text: &str, font: &FontId) -> f32 {
    ui.fonts_mut(|fonts| {
        fonts
            .layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE)
            .size()
            .x
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    Primary,
    Secondary,
    Danger,
    Quiet,
}

#[allow(clippy::too_many_arguments)]
pub fn button(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    rect: Rect,
    label: &str,
    kind: ButtonKind,
    accent: Color32,
    pal: Palette,
    focused: bool,
    enabled: bool,
    font_size: f32,
) -> egui::Response {
    let response = ui.allocate_rect(rect, if enabled { egui::Sense::click() } else { egui::Sense::hover() });
    let hovered = enabled && response.hovered();
    paint_button(painter, rect, label, kind, accent, pal, hovered, focused, enabled, font_size);
    if hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

#[allow(clippy::too_many_arguments)]
pub fn paint_button(
    painter: &egui::Painter,
    rect: Rect,
    label: &str,
    kind: ButtonKind,
    accent: Color32,
    pal: Palette,
    hovered: bool,
    focused: bool,
    enabled: bool,
    font_size: f32,
) {
    let radius = rect.height() * 0.28;
    let (fill, border, text) = match kind {
        ButtonKind::Primary => {
            let base = if hovered || focused { brighten(accent, 0.12) } else { accent };
            (base, brighten(base, 0.25), readable_on(base))
        }
        ButtonKind::Secondary => (
            if hovered { pal.panel_hover } else { pal.input },
            if focused { accent } else { pal.border },
            pal.text,
        ),
        ButtonKind::Danger => (
            if hovered { alpha(DANGER, 0.22) } else { alpha(DANGER, 0.12) },
            if focused || hovered { DANGER } else { alpha(DANGER, 0.55) },
            if pal.light { Color32::from_rgb(0xB8, 0x1E, 0x2A) } else { Color32::from_rgb(0xFF, 0x8A, 0x90) },
        ),
        ButtonKind::Quiet => (
            if hovered { pal.panel_hover } else { Color32::TRANSPARENT },
            if focused { accent } else { Color32::TRANSPARENT },
            if hovered || focused { pal.text } else { pal.muted },
        ),
    };
    let (fill, border, text) = if enabled {
        (fill, border, text)
    } else if kind == ButtonKind::Primary {
        (alpha(accent, 0.16), if focused { border } else { alpha(accent, 0.3) }, alpha(pal.text, 0.42))
    } else {
        (alpha(fill, 0.45), if focused { border } else { alpha(border, 0.45) }, alpha(text, 0.45))
    };
    painter.rect_filled(rect, CornerRadius::from(radius), fill);
    if border != Color32::TRANSPARENT {
        painter.rect_stroke(
            rect,
            CornerRadius::from(radius),
            Stroke::new(if focused { 2.2 } else { 1.2 }, border),
            egui::StrokeKind::Outside,
        );
    }
    if focused && kind == ButtonKind::Primary {
        painter.rect_stroke(
            rect.expand(3.0),
            CornerRadius::from(radius + 3.0),
            Stroke::new(1.6, alpha(accent, 0.7)),
            egui::StrokeKind::Outside,
        );
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(font_size),
        text,
    );
}

pub fn switch(painter: &egui::Painter, rect: Rect, on: f32, accent: Color32, pal: Palette) {
    let track = lerp(pal.input, accent, on);
    let radius = rect.height() * 0.5;
    painter.rect_filled(rect, CornerRadius::from(radius), track);
    painter.rect_stroke(
        rect,
        CornerRadius::from(radius),
        Stroke::new(1.0, lerp(pal.border, brighten(accent, 0.2), on)),
        egui::StrokeKind::Inside,
    );
    let knob_x = rect.min.x + radius + (rect.width() - radius * 2.0) * on;
    painter.circle_filled(
        Pos2::new(knob_x, rect.center().y + 1.0),
        radius - 3.0,
        Color32::from_black_alpha(40),
    );
    painter.circle_filled(Pos2::new(knob_x, rect.center().y), radius - 3.0, Color32::WHITE);
}

pub struct AvatarCache {
    textures: HashMap<u64, (Arc<Vec<u8>>, egui::TextureHandle)>,
    failed: HashMap<u64, Arc<Vec<u8>>>,
}

impl AvatarCache {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            failed: HashMap::new(),
        }
    }

    pub fn texture(
        &mut self,
        ctx: &egui::Context,
        avatars: &HashMap<u64, Arc<Vec<u8>>>,
        pid: u64,
    ) -> Option<egui::TextureId> {
        let bytes = avatars.get(&pid)?;
        if let Some((cached, texture)) = self.textures.get(&pid) {
            if Arc::ptr_eq(cached, bytes) {
                return Some(texture.id());
            }
        }
        if self.failed.get(&pid).is_some_and(|failed| Arc::ptr_eq(failed, bytes)) {
            return None;
        }
        let Some(image) = decode(bytes) else {
            self.failed.insert(pid, bytes.clone());
            return None;
        };
        let texture = ctx.load_texture(
            format!("nextendo-avatar-{pid}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        let id = texture.id();
        self.textures.insert(pid, (bytes.clone(), texture));
        Some(id)
    }

    pub fn clear(&mut self) {
        self.textures.clear();
        self.failed.clear();
    }
}

fn decode(bytes: &[u8]) -> Option<egui::ColorImage> {
    let image = image::load_from_memory(bytes).ok()?;
    let image = if image.width() > 192 || image.height() > 192 {
        image.resize_to_fill(192, 192, image::imageops::FilterType::Triangle)
    } else {
        image
    };
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        rgba.as_raw(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initials_pick_two_words() {
        assert_eq!(initials("Super Mario"), "SM");
        assert_eq!(initials("luigi"), "L");
        assert_eq!(initials("big_boo_king"), "BB");
        assert_eq!(initials(""), "?");
    }

    #[test]
    fn readable_text_contrasts_with_its_fill() {
        assert_eq!(readable_on(Color32::WHITE), Color32::from_rgb(0x12, 0x14, 0x1C));
        assert_eq!(readable_on(Color32::from_rgb(0x10, 0x10, 0x40)), Color32::WHITE);
    }

    #[test]
    fn colors_lerp_and_fade() {
        assert_eq!(lerp(Color32::BLACK, Color32::WHITE, 0.5), Color32::from_rgb(128, 128, 128));
        assert_eq!(alpha(Color32::WHITE, 0.0).a(), 0);
    }
}
