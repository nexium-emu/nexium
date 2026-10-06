use super::look;
use eframe::egui;
use eframe::egui::{Color32, CornerRadius, FontId, Pos2, Rect, Stroke, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChipTone {
    SignedOut,
    Connecting,
    Online,
    Offline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeaderChip {
    pub label: String,
    pub detail: String,
    pub tone: ChipTone,
    pub avatar: Option<egui::TextureId>,
    pub seed: u64,
    pub alerts: usize,
}

#[derive(Clone, Copy)]
pub struct ChipColors {
    pub bar: Color32,
    pub border: Color32,
    pub text: Color32,
    pub muted: Color32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnlineBadge {
    pub players: Option<u32>,
    pub required: &'static str,
    pub installed: String,
    pub version_ok: bool,
    pub enabled: bool,
}

impl OnlineBadge {
    pub fn headline(&self) -> String {
        if !self.version_ok {
            let installed = self.installed.trim().trim_start_matches(['v', 'V']);
            return if installed.is_empty() {
                format!("Online play needs version {}", self.required)
            } else {
                format!("Online play needs version {}  ·  you have {}", self.required, installed)
            };
        }
        match self.players {
            Some(0) | None => "Online with Nextendo".to_string(),
            Some(1) => "Online with Nextendo  ·  1 player online".to_string(),
            Some(count) => format!("Online with Nextendo  ·  {} players online", group(count)),
        }
    }
}

pub fn group(value: u32) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn tone_color(tone: ChipTone) -> Color32 {
    match tone {
        ChipTone::SignedOut => Color32::from_rgb(0x8A, 0x8A, 0x98),
        ChipTone::Connecting => look::WARNING,
        ChipTone::Online => look::ONLINE,
        ChipTone::Offline => look::WARNING,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn header_chip(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    right: f32,
    center_y: f32,
    scale: f32,
    chip: &HeaderChip,
    colors: ChipColors,
    opacity: f32,
    time: f64,
    focused: bool,
) -> egui::Response {
    let height = 30.0 * scale;
    let label_font = FontId::proportional(13.5 * scale);
    let detail_font = FontId::proportional(12.0 * scale);
    let label_w = look::text_width(ui, &chip.label, &label_font);
    let detail_w = if chip.detail.is_empty() {
        0.0
    } else {
        look::text_width(ui, &chip.detail, &detail_font) + 14.0 * scale
    };
    let icon = 22.0 * scale;
    let width = 8.0 * scale + icon + 9.0 * scale + label_w + detail_w + 13.0 * scale;
    let rect = Rect::from_min_max(
        Pos2::new(right - width, center_y - height * 0.5),
        Pos2::new(right, center_y + height * 0.5),
    );
    let response = ui.interact(rect, egui::Id::new("nextendo_header_chip"), egui::Sense::click());
    let hovered = response.hovered();
    let rounding = CornerRadius::from(height * 0.5);
    painter.rect_filled(
        rect.translate(Vec2::new(0.0, 2.0 * scale)),
        rounding,
        Color32::from_black_alpha((opacity * 70.0) as u8),
    );
    painter.rect_filled(rect, rounding, look::alpha(colors.bar, opacity));
    let accent = tone_color(chip.tone);
    let border = if hovered || focused {
        look::alpha(accent, opacity)
    } else {
        look::alpha(colors.border, opacity)
    };
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(if hovered || focused { 1.6 } else { 1.0 } * scale, border),
        egui::StrokeKind::Outside,
    );
    let icon_center = Pos2::new(rect.min.x + 8.0 * scale + icon * 0.5, center_y);
    match chip.tone {
        ChipTone::SignedOut => {
            look::network_glyph(painter, icon_center, icon * 0.4, look::alpha(colors.text, opacity));
        }
        ChipTone::Connecting if chip.avatar.is_none() => {
            look::spinner(painter, icon_center, icon * 0.38, time, look::alpha(accent, opacity));
        }
        _ => {
            look::avatar(
                painter,
                icon_center,
                icon * 0.5,
                chip.avatar,
                &chip.label,
                chip.seed,
                opacity,
            );
            look::status_dot(
                painter,
                icon_center + Vec2::new(icon * 0.36, icon * 0.36),
                icon * 0.15,
                look::alpha(accent, opacity),
                look::alpha(colors.bar, opacity),
            );
        }
    }
    let text_x = icon_center.x + icon * 0.5 + 9.0 * scale;
    painter.text(
        Pos2::new(text_x, center_y),
        egui::Align2::LEFT_CENTER,
        &chip.label,
        label_font,
        look::alpha(colors.text, opacity),
    );
    if !chip.detail.is_empty() {
        let sep_x = text_x + label_w + 7.0 * scale;
        painter.circle_filled(
            Pos2::new(sep_x, center_y),
            1.6 * scale,
            look::alpha(colors.muted, opacity),
        );
        painter.text(
            Pos2::new(sep_x + 7.0 * scale, center_y),
            egui::Align2::LEFT_CENTER,
            &chip.detail,
            detail_font,
            look::alpha(colors.muted, opacity),
        );
    }
    if chip.alerts > 0 {
        look::counter_badge(
            painter,
            Pos2::new(rect.max.x - 4.0 * scale, rect.min.y + 2.0 * scale),
            15.0 * scale,
            chip.alerts,
        );
    }
    if hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

pub fn card_badge(ui: &egui::Ui, painter: &egui::Painter, card: Rect, badge: &OnlineBadge, opacity: f32) {
    if opacity <= 0.01 || !badge.enabled {
        return;
    }
    let font = FontId::proportional((card.width() * 0.05).clamp(10.0, 15.0));
    let height = font.size * 1.75;
    let text = if !badge.version_ok {
        "Update".to_string()
    } else {
        match badge.players {
            Some(count) if count > 0 => group(count),
            _ => "Online".to_string(),
        }
    };
    let text_w = look::text_width(ui, &text, &font);
    let glyph = height * 0.62;
    let width = height * 0.42 + glyph + height * 0.28 + text_w + height * 0.5;
    let margin = card.width() * 0.05;
    let rect = Rect::from_min_size(
        Pos2::new(card.max.x - margin - width, card.min.y + margin),
        Vec2::new(width, height),
    );
    let tint = if badge.version_ok { look::ONLINE } else { look::WARNING };
    let rounding = CornerRadius::from(height * 0.5);
    painter.rect_filled(
        rect,
        rounding,
        Color32::from_rgba_unmultiplied(0x0A, 0x0E, 0x14, (opacity * 210.0) as u8),
    );
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(1.3, look::alpha(tint, opacity * 0.9)),
        egui::StrokeKind::Outside,
    );
    let glyph_center = Pos2::new(rect.min.x + height * 0.42 + glyph * 0.5, rect.center().y);
    look::network_glyph(painter, glyph_center, glyph * 0.42, look::alpha(tint, opacity));
    painter.text(
        Pos2::new(glyph_center.x + glyph * 0.5 + height * 0.28, rect.center().y),
        egui::Align2::LEFT_CENTER,
        text,
        font,
        look::alpha(Color32::from_rgb(0xEE, 0xF4, 0xF8), opacity),
    );
}

pub fn status_line(
    ui: &egui::Ui,
    painter: &egui::Painter,
    center: Pos2,
    scale: f32,
    badge: &OnlineBadge,
    colors: ChipColors,
    opacity: f32,
) {
    if opacity <= 0.01 || !badge.enabled {
        return;
    }
    let font = FontId::proportional(13.0 * scale);
    let text = badge.headline();
    let text_w = look::text_width(ui, &text, &font);
    let height = 26.0 * scale;
    let glyph = 14.0 * scale;
    let width = 12.0 * scale + glyph + 8.0 * scale + text_w + 14.0 * scale;
    let rect = Rect::from_center_size(center, Vec2::new(width, height));
    let tint = if badge.version_ok { look::ONLINE } else { look::WARNING };
    let rounding = CornerRadius::from(height * 0.5);
    painter.rect_filled(rect, rounding, look::alpha(colors.bar, opacity * 0.92));
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(1.1 * scale, look::alpha(tint, opacity * 0.75)),
        egui::StrokeKind::Outside,
    );
    let glyph_center = Pos2::new(rect.min.x + 12.0 * scale + glyph * 0.5, center.y);
    look::network_glyph(painter, glyph_center, glyph * 0.45, look::alpha(tint, opacity));
    painter.text(
        Pos2::new(glyph_center.x + glyph * 0.5 + 8.0 * scale, center.y),
        egui::Align2::LEFT_CENTER,
        text,
        font,
        look::alpha(colors.text, opacity),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_group_by_thousands() {
        assert_eq!(group(0), "0");
        assert_eq!(group(999), "999");
        assert_eq!(group(1000), "1,000");
        assert_eq!(group(1234567), "1,234,567");
    }

    #[test]
    fn badge_headlines_explain_the_state() {
        let mut badge = OnlineBadge {
            players: Some(16),
            required: "4.0.0",
            installed: "4.0.0".into(),
            version_ok: true,
            enabled: true,
        };
        assert_eq!(badge.headline(), "Online with Nextendo  ·  16 players online");
        badge.players = Some(1);
        assert_eq!(badge.headline(), "Online with Nextendo  ·  1 player online");
        badge.version_ok = false;
        badge.installed = "3.0.3".into();
        assert_eq!(badge.headline(), "Online play needs version 4.0.0  ·  you have 3.0.3");
        badge.installed.clear();
        assert_eq!(badge.headline(), "Online play needs version 4.0.0");
    }
}
