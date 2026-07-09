use eframe::egui::{self, Color32, FontId, Pos2, Shape, Stroke, Vec2};
use std::f32::consts::{PI, TAU};
use std::time::Instant;

const DURATION: f32 = 3.9;
const FADE: f32 = 0.7;
const ORBIT_END: f32 = 1.8;
const CONV_END: f32 = 2.35;
const TEXT_START: f32 = 2.7;

const RING: Color32 = Color32::from_rgb(0x5C, 0xF2, 0xFF);
const RING_DIM: Color32 = Color32::from_rgb(0x3D, 0xBE, 0xFF);

pub struct Splash {
    start: Option<Instant>,
    finished: bool,
}

pub enum Step {
    Intro,
    Fade,
    Done,
}

impl Splash {
    pub fn new() -> Self {
        Self {
            start: None,
            finished: false,
        }
    }

    pub fn finished() -> Self {
        Self {
            start: None,
            finished: true,
        }
    }

    pub fn active(&self) -> bool {
        !self.finished
    }

    pub fn step(&mut self, ctx: &egui::Context) -> Step {
        if self.start.is_none() {
            crate::ui_audio::play(crate::ui_audio::Sfx::Boot);
        }
        let start = *self.start.get_or_insert_with(Instant::now);
        let mut elapsed = start.elapsed().as_secs_f32();

        let skip = ctx.input(|i| {
            i.pointer.any_pressed()
                || i.events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))
        });
        if skip && elapsed < DURATION {
            elapsed = DURATION;
            self.start = Some(Instant::now() - std::time::Duration::from_secs_f32(elapsed));
        }

        if elapsed < DURATION {
            let painter = ctx.layer_painter(egui::LayerId::background());
            let rect = ctx.screen_rect();
            painter.rect_filled(rect, 0.0, Color32::BLACK);
            paint_scene(&painter, rect, elapsed, 1.0);
            ctx.request_repaint();
            Step::Intro
        } else if elapsed < DURATION + FADE {
            let prog = (elapsed - DURATION) / FADE;
            let black_a = 1.0 - ((prog - 0.55) / 0.45).clamp(0.0, 1.0);
            let scene_a = (1.0 - prog).clamp(0.0, 1.0);
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("splash_fade"),
            ));
            let rect = ctx.screen_rect();
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha((black_a * 255.0) as u8));
            paint_scene(&painter, rect, elapsed, scene_a);
            ctx.request_repaint();
            Step::Fade
        } else {
            self.finished = true;
            Step::Done
        }
    }
}

fn paint_scene(p: &egui::Painter, rect: egui::Rect, e: f32, ga: f32) {
    let center = rect.center();
    let r = rect.width().min(rect.height()) * 0.17;
    let dt = e - CONV_END;

    let settle = 1.0 + 0.12 * (1.0 - smoothstep(0.0, 1.0, e.min(1.0)));
    let windup = if dt < 0.0 {
        -0.14 * smoothstep(CONV_END - 0.22, CONV_END, e)
    } else {
        0.28 * ((dt / 0.05).clamp(0.0, 1.0)) * (1.0 - smoothstep(0.05, 0.55, dt))
    };
    let fade_zoom = if e > DURATION {
        (e - DURATION) / FADE * 0.18
    } else {
        0.0
    };
    let scale = settle + windup + fade_zoom;
    let (rx, ry) = (r * scale, r * scale * 0.42);

    if dt >= 0.0 {
        let wash = (1.0 - (dt / 0.45).clamp(0.0, 1.0)).powi(2);
        if wash > 0.0 {
            p.rect_filled(rect, 0.0, alpha(RING, (wash * 70.0 * ga) as u8));
        }
        let diag = rect.size().length() * 0.62;
        for off in [0.0_f32, 0.12, 0.26] {
            let s = (dt / 0.7 - off).max(0.0);
            if s > 0.0 && s < 1.0 {
                let rad = ease_out(s) * diag;
                let a = ((1.0 - s).powi(2) * 230.0 * ga) as u8;
                p.circle_stroke(center, rad, Stroke::new(4.0 * (1.0 - s) + 1.0, alpha(RING, a)));
            }
        }
    }

    let flash = (1.0 - (dt / 0.3).clamp(0.0, 1.0)).max(0.0) * (dt >= 0.0) as i32 as f32;
    let ring_a = (smoothstep(0.15, 1.0, e) * 0.85 + flash * 0.6).min(1.0) * ga;
    for &(rot, delay) in &[(0.80_f32, 0.0_f32), (-0.80_f32, 0.22_f32)] {
        let draw_dur = 0.85;
        let dp = ((e - 0.18 - delay) / draw_dur).clamp(0.0, 1.0);
        let draw_t = ease_out(dp);
        ring(p, center, rx, ry, rot, ring_a, draw_t);
        let sf = e - (0.18 + delay + draw_dur);
        if sf >= 0.0 && sf < 0.5 {
            spark(p, ellipse_pt(center, rx, ry, rot, 0.0), r, sf, ga);
        }
    }

    let rf = 1.0 - smoothstep(ORBIT_END, CONV_END, e);
    let orb_a = (1.0 - smoothstep(CONV_END - 0.05, CONV_END + 0.12, e)) * ga;
    if orb_a > 0.01 {
        for (rot, phase) in [(0.80_f32, 0.0_f32), (-0.80_f32, PI)] {
            let ang = e * TAU * 0.95 + phase;
            for k in 0..6 {
                let pa = ang - k as f32 * 0.13;
                let pos = center + (ellipse_pt(center, rx, ry, rot, pa) - center) * rf;
                let ta = orb_a * (1.0 - k as f32 / 6.0) * if k == 0 { 1.0 } else { 0.4 };
                orb(p, pos, r * 0.11 * if k == 0 { 1.0 } else { 0.7 }, ta);
            }
        }
    }

    if dt >= 0.0 {
        let core = smoothstep(CONV_END, CONV_END + 0.4, e) * ga;
        let pulse = 1.0 + 0.05 * (e * 7.0).sin();
        orb(p, center, r * (0.13 + flash * 0.14) * pulse, (core + flash * 0.7).min(1.0));
    }

    let text_a = smoothstep(TEXT_START, TEXT_START + 0.6, e) * ga;
    if text_a > 0.0 {
        let size = (rect.width().min(rect.height()) * 0.055).clamp(22.0, 52.0);
        let y = rect.max.y - rect.height() * 0.095 - size * 1.4;
        title(p, center.x, y, size, text_a);
    }
}

fn ease_out(x: f32) -> f32 {
    1.0 - (1.0 - x).powi(3)
}

fn spark(p: &egui::Painter, c: Pos2, r: f32, sf: f32, ga: f32) {
    let k = (sf / 0.45).clamp(0.0, 1.0);
    let fade = (1.0 - k).powi(2);

    let flash = (1.0 - (sf / 0.12).clamp(0.0, 1.0)).powi(2);
    if flash > 0.0 {
        p.circle_filled(c, r * (0.09 + flash * 0.20), alpha(Color32::WHITE, (flash * 255.0 * ga) as u8));
        p.circle_filled(c, r * (0.26 + flash * 0.34), alpha(RING, (flash * 130.0 * ga) as u8));
    }

    let rr = ease_out(k) * r * 0.95;
    p.circle_stroke(c, rr, Stroke::new((1.0 - k) * 3.2 + 0.5, alpha(RING, (fade * 210.0 * ga) as u8)));

    let n = 8;
    for i in 0..n {
        let ang = i as f32 / n as f32 * TAU + (i % 3) as f32 * 0.4;
        let dir = Vec2::new(ang.cos(), ang.sin());
        let flick = (0.55 + 0.45 * (sf * 45.0 + i as f32 * 1.7).sin()).max(0.0);
        let spread = 0.5 + 0.5 * ((i * 7 % 5) as f32 / 5.0);
        let d0 = ease_out(k) * r * (0.35 + 0.75 * spread);
        let len = r * 0.22 * (1.0 - k);
        let a0 = c + dir * d0;
        let a1 = c + dir * (d0 + len);
        p.add(Shape::line(
            vec![a0, a1],
            Stroke::new((1.0 - k) * 2.2 + 0.4, alpha(RING, (fade * flick * 235.0 * ga) as u8)),
        ));
    }
}

fn ring(p: &egui::Painter, c: Pos2, rx: f32, ry: f32, rot: f32, a: f32, draw_t: f32) {
    if a <= 0.01 || draw_t <= 0.0 {
        return;
    }
    let steps = 96usize;
    let end = ((draw_t.clamp(0.0, 1.0) * steps as f32).ceil() as usize).max(1);
    let pts: Vec<Pos2> = (0..=end.min(steps))
        .map(|i| ellipse_pt(c, rx, ry, rot, (i as f32 / steps as f32) * TAU))
        .collect();
    if pts.len() < 2 {
        return;
    }
    let w = rx * 0.03;
    p.add(Shape::line(pts.clone(), Stroke::new(w * 2.6, alpha(RING_DIM, (55.0 * a) as u8))));
    p.add(Shape::line(pts, Stroke::new(w, alpha(RING, (255.0 * a) as u8))));
    if draw_t < 1.0 {
        let tip = ellipse_pt(c, rx, ry, rot, draw_t.clamp(0.0, 1.0) * TAU);
        p.circle_filled(tip, w * 2.4, alpha(Color32::WHITE, (a * 235.0) as u8));
        p.circle_filled(tip, w * 4.2, alpha(RING, (a * 90.0) as u8));
    }
}

fn orb(p: &egui::Painter, c: Pos2, r: f32, a: f32) {
    if a <= 0.0 {
        return;
    }
    for i in 0..6 {
        let rr = r * (0.6 + i as f32 * 0.7);
        let al = (a * 130.0 / (i as f32 * 1.6 + 1.0)) as u8;
        p.circle_filled(c, rr, alpha(RING, al));
    }
    p.circle_filled(c, r * 0.5, alpha(Color32::WHITE, (a * 255.0) as u8));
}

fn title(p: &egui::Painter, cx: f32, y: f32, size: f32, a: f32) {
    let glow = alpha(RING, (a * 60.0) as u8);
    let soft = alpha(RING_DIM, (a * 32.0) as u8);
    let main = Color32::from_rgba_unmultiplied(0xEA, 0xF8, 0xFF, (a * 255.0) as u8);
    for off in [(-4.0, 0.0), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0)] {
        tracked(p, cx + off.0, y + off.1, size, soft);
    }
    for off in [(-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (-2.0, -2.0), (2.0, 2.0)] {
        tracked(p, cx + off.0, y + off.1, size, glow);
    }
    tracked(p, cx, y, size, main);
}

fn tracked(p: &egui::Painter, cx: f32, y: f32, size: f32, col: Color32) {
    let font = FontId::proportional(size);
    let track = size * 0.22;
    let widths: Vec<f32> = "NeXium"
        .chars()
        .map(|c| p.layout_no_wrap(c.to_string(), font.clone(), col).size().x)
        .collect();
    let total: f32 = widths.iter().sum::<f32>() + track * (widths.len() as f32 - 1.0);
    let mut x = cx - total / 2.0;
    for (c, w) in "NeXium".chars().zip(widths) {
        p.text(
            egui::pos2(x + w / 2.0, y),
            egui::Align2::CENTER_CENTER,
            c,
            font.clone(),
            col,
        );
        x += w + track;
    }
}

fn ellipse_pt(c: Pos2, rx: f32, ry: f32, rot: f32, ang: f32) -> Pos2 {
    let (sa, ca) = ang.sin_cos();
    let (x, y) = (rx * ca, ry * sa);
    let (sr, cr) = rot.sin_cos();
    c + Vec2::new(x * cr - y * sr, x * sr + y * cr)
}

fn alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
