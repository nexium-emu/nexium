use super::look;
use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::HashMap;
use std::sync::Arc;

const ENTER: f64 = 0.28;
const LEAVE: f64 = 0.24;
const VISIBLE: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Friend,
    Request,
    Success,
    Info,
    Warning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastTarget {
    Account,
    Friends,
    Requests,
}

struct Toast {
    id: u64,
    kind: ToastKind,
    title: String,
    body: String,
    pid: Option<u64>,
    target: ToastTarget,
    born: Option<f64>,
    lifetime: f64,
    dismissed: Option<f64>,
}

pub struct Toasts {
    queue: Vec<Toast>,
    next_id: u64,
    shown: Vec<Rect>,
}

impl Toasts {
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
            next_id: 0,
            shown: Vec::new(),
        }
    }

    pub fn shown(&self) -> &[Rect] {
        &self.shown
    }

    pub fn push(&mut self, kind: ToastKind, title: impl Into<String>, body: impl Into<String>, pid: Option<u64>, target: ToastTarget) {
        self.next_id += 1;
        let title = title.into();
        if self.queue.iter().any(|toast| toast.title == title && toast.dismissed.is_none()) {
            return;
        }
        self.queue.push(Toast {
            id: self.next_id,
            kind,
            title,
            body: body.into(),
            pid,
            target,
            born: None,
            lifetime: if matches!(kind, ToastKind::Warning) { 8.0 } else { 5.5 },
            dismissed: None,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        area: Rect,
        top_offset: f32,
        avatars: &mut look::AvatarCache,
        avatar_bytes: &HashMap<u64, Arc<Vec<u8>>>,
        accent: Color32,
        light: bool,
    ) -> Option<ToastTarget> {
        self.shown.clear();
        if self.queue.is_empty() {
            return None;
        }
        let now = ctx.input(|input| input.time);
        let pal = look::palette(if light { 1.0 } else { 0.0 });
        let scale = (area.height() / 820.0).clamp(1.0, 1.8);
        let width = 372.0 * scale;
        let height = 78.0 * scale;
        let gap = 12.0 * scale;
        let margin = 22.0 * scale;
        let mut clicked = None;
        for (slot, toast) in self.queue.iter_mut().take(VISIBLE).enumerate() {
            let born = *toast.born.get_or_insert(now);
            let age = now - born;
            if toast.dismissed.is_none() && age > toast.lifetime {
                toast.dismissed = Some(now);
            }
            let enter = ((age / ENTER).min(1.0)) as f32;
            let leave = toast
                .dismissed
                .map(|at| (((now - at) / LEAVE).min(1.0)) as f32)
                .unwrap_or(0.0);
            let appear = 1.0 - (1.0 - enter).powi(3);
            let vanish = leave * leave;
            let opacity = (appear * (1.0 - vanish)).clamp(0.0, 1.0);
            let slide = (1.0 - appear + vanish) * (width + margin);
            let pos = Pos2::new(
                area.max.x - margin - width + slide,
                area.min.y + top_offset + margin + slot as f32 * (height + gap),
            );
            let rect = Rect::from_min_size(pos, Vec2::new(width, height));
            if opacity > 0.02 {
                self.shown.push(rect.expand(8.0 * scale));
            }
            let response = egui::Area::new(egui::Id::new(("nextendo_toast", toast.id)))
                .order(egui::Order::Tooltip)
                .fixed_pos(pos)
                .interactable(true)
                .show(ctx, |ui| {
                    let (rect, response) = ui.allocate_exact_size(rect.size(), Sense::click());
                    let painter = ui.painter();
                    let hovered = response.hovered();
                    if opacity > 0.02 {
                        look::soft_shadow(painter, rect, 14.0 * scale, opacity * 1.4);
                        let fill = if light {
                            Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, (opacity * 250.0) as u8)
                        } else {
                            Color32::from_rgba_unmultiplied(0x17, 0x17, 0x1F, (opacity * 246.0) as u8)
                        };
                        let tint = match toast.kind {
                            ToastKind::Friend => look::ONLINE,
                            ToastKind::Request => accent,
                            ToastKind::Success => look::ONLINE,
                            ToastKind::Info => accent,
                            ToastKind::Warning => look::WARNING,
                        };
                        painter.rect_filled(rect, CornerRadius::from(14.0 * scale), fill);
                        painter.rect_stroke(
                            rect,
                            CornerRadius::from(14.0 * scale),
                            Stroke::new(
                                1.0,
                                look::alpha(if hovered { tint } else { pal.border }, opacity),
                            ),
                            egui::StrokeKind::Outside,
                        );
                        let strip = Rect::from_min_size(
                            rect.min + Vec2::new(0.0, 14.0 * scale),
                            Vec2::new(3.5 * scale, rect.height() - 28.0 * scale),
                        );
                        painter.rect_filled(strip, CornerRadius::from(2.0 * scale), look::alpha(tint, opacity));
                        let icon_center = Pos2::new(rect.min.x + 40.0 * scale, rect.center().y);
                        let radius = 21.0 * scale;
                        match toast.pid {
                            Some(pid) => {
                                let texture = avatars.texture(ctx, avatar_bytes, pid);
                                look::avatar(painter, icon_center, radius, texture, &toast.title, pid, opacity);
                                if toast.kind == ToastKind::Friend {
                                    look::status_dot(
                                        painter,
                                        icon_center + Vec2::new(radius * 0.72, radius * 0.72),
                                        5.0 * scale,
                                        look::alpha(look::ONLINE, opacity),
                                        fill,
                                    );
                                }
                            }
                            None => {
                                painter.circle_filled(icon_center, radius, look::alpha(tint, opacity * 0.16));
                                match toast.kind {
                                    ToastKind::Success => look::check_glyph(painter, icon_center, radius * 0.95, look::alpha(tint, opacity)),
                                    ToastKind::Warning => {
                                        painter.text(
                                            icon_center,
                                            egui::Align2::CENTER_CENTER,
                                            "!",
                                            FontId::proportional(radius * 1.1),
                                            look::alpha(tint, opacity),
                                        );
                                    }
                                    _ => look::network_glyph(painter, icon_center, radius * 0.55, look::alpha(tint, opacity)),
                                }
                            }
                        }
                        let text_x = icon_center.x + radius + 14.0 * scale;
                        let text_w = rect.max.x - text_x - 16.0 * scale;
                        let title_font = FontId::proportional(15.5 * scale);
                        let body_font = FontId::proportional(13.0 * scale);
                        let title = look::ellipsize(ui, &toast.title, &title_font, text_w);
                        let body = look::ellipsize(ui, &toast.body, &body_font, text_w);
                        painter.text(
                            Pos2::new(text_x, rect.center().y - 10.0 * scale),
                            egui::Align2::LEFT_CENTER,
                            title,
                            title_font,
                            look::alpha(pal.text, opacity),
                        );
                        painter.text(
                            Pos2::new(text_x, rect.center().y + 12.0 * scale),
                            egui::Align2::LEFT_CENTER,
                            body,
                            body_font,
                            look::alpha(pal.muted, opacity),
                        );
                        let remaining = if toast.dismissed.is_none() {
                            (1.0 - (age / toast.lifetime)).clamp(0.0, 1.0) as f32
                        } else {
                            0.0
                        };
                        let bar = Rect::from_min_size(
                            Pos2::new(rect.min.x + 18.0 * scale, rect.max.y - 5.0 * scale),
                            Vec2::new((rect.width() - 36.0 * scale) * remaining, 2.0 * scale),
                        );
                        painter.rect_filled(bar, CornerRadius::from(1.0), look::alpha(tint, opacity * 0.5));
                    }
                    if hovered {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    response
                })
                .inner;
            if response.hovered() && toast.dismissed.is_none() {
                toast.born = Some(born.max(now - toast.lifetime * 0.5));
            }
            if response.clicked() && toast.dismissed.is_none() {
                toast.dismissed = Some(now);
                clicked = Some(toast.target);
            }
        }
        self.queue.retain(|toast| {
            toast
                .dismissed
                .map_or(true, |at| now - at < LEAVE + 0.05)
        });
        if !self.queue.is_empty() {
            ctx.request_repaint();
        }
        clicked
    }
}
