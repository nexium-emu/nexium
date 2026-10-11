use super::{format_file_size, reveal_in_file_manager, HorizonApp};
use crate::dump::{DumpJob, Outcome};
use crate::nextendo::look::{self, ButtonKind};
use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Vec2};
use nexium_loader::dump::DumpSummary;
use std::path::PathBuf;
use std::time::Duration;

const FINISHED_VISIBLE: f32 = 12.0;
const FRAME_INTERVAL: Duration = Duration::from_millis(33);

#[derive(Clone, Copy, PartialEq, Eq)]
enum CardAction {
    None,
    Cancel,
    OpenFolder,
    Dismiss,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Glyph {
    Spinner,
    Check,
    Cross,
    Warning,
}

struct Card {
    title: String,
    detail: String,
    tint: Color32,
    glyph: Glyph,
    fraction: Option<f32>,
    running: bool,
    remaining: Option<f32>,
    wrap_detail: bool,
    button: &'static str,
    button_kind: ButtonKind,
    button_enabled: bool,
    button_action: CardAction,
}

fn file_count(files: u64) -> String {
    if files == 1 {
        "1 file".to_string()
    } else {
        format!("{files} files")
    }
}

fn summary_text(summary: &DumpSummary) -> String {
    let files =
        summary.exefs.map_or(0, |stats| stats.files) + summary.romfs.map_or(0, |stats| stats.files);
    let bytes =
        summary.exefs.map_or(0, |stats| stats.bytes) + summary.romfs.map_or(0, |stats| stats.bytes);
    let mut text = format!("{}  ·  {}", file_count(files), format_file_size(bytes));
    match (summary.romfs, summary.exefs) {
        (None, Some(_)) => text.push_str("  ·  no RomFS"),
        (Some(_), None) => text.push_str("  ·  no ExeFS"),
        _ => {}
    }
    text
}

fn describe(job: &DumpJob, accent: Color32) -> Card {
    let remaining =
        |elapsed: Duration| Some((1.0 - elapsed.as_secs_f32() / FINISHED_VISIBLE).clamp(0.0, 1.0));
    match job.outcome() {
        None => {
            let progress = job.progress();
            let cancelling = job.is_cancelling();
            let fraction = (progress.bytes_total > 0).then(|| {
                (progress.bytes_done as f64 / progress.bytes_total as f64).clamp(0.0, 1.0) as f32
            });
            let detail = match fraction {
                _ if cancelling => "Stopping the dump…".to_string(),
                Some(fraction) => format!(
                    "{}%  ·  {} of {}",
                    (fraction * 100.0) as u32,
                    format_file_size(progress.bytes_done),
                    format_file_size(progress.bytes_total)
                ),
                None => "Reading the game files…".to_string(),
            };
            Card {
                title: if cancelling {
                    "Cancelling…".to_string()
                } else {
                    format!("Dumping {}", job.title())
                },
                detail,
                tint: accent,
                glyph: Glyph::Spinner,
                fraction,
                running: true,
                remaining: None,
                wrap_detail: false,
                button: "Cancel",
                button_kind: ButtonKind::Quiet,
                button_enabled: !cancelling,
                button_action: CardAction::Cancel,
            }
        }
        Some((Outcome::Done(summary), elapsed)) => Card {
            title: format!("Dumped {}", job.title()),
            detail: summary_text(summary),
            tint: look::ONLINE,
            glyph: Glyph::Check,
            fraction: None,
            running: false,
            remaining: remaining(elapsed),
            wrap_detail: false,
            button: "Open Folder",
            button_kind: ButtonKind::Secondary,
            button_enabled: job.destination().is_some(),
            button_action: CardAction::OpenFolder,
        },
        Some((Outcome::Cancelled, elapsed)) => Card {
            title: "Dump cancelled".to_string(),
            detail: "Files written so far were kept.".to_string(),
            tint: accent,
            glyph: Glyph::Cross,
            fraction: None,
            running: false,
            remaining: remaining(elapsed),
            wrap_detail: false,
            button: "Dismiss",
            button_kind: ButtonKind::Quiet,
            button_enabled: true,
            button_action: CardAction::Dismiss,
        },
        Some((Outcome::Failed(message), _)) => Card {
            title: format!("Couldn't dump {}", job.title()),
            detail: message.clone(),
            tint: look::WARNING,
            glyph: Glyph::Warning,
            fraction: None,
            running: false,
            remaining: None,
            wrap_detail: true,
            button: "Dismiss",
            button_kind: ButtonKind::Quiet,
            button_enabled: true,
            button_action: CardAction::Dismiss,
        },
    }
}

fn paint_glyph(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    glyph: Glyph,
    tint: Color32,
    time: f64,
) {
    painter.circle_filled(center, radius, look::alpha(tint, 0.16));
    match glyph {
        Glyph::Spinner => look::spinner(painter, center, radius * 0.6, time, tint),
        Glyph::Check => look::check_glyph(painter, center, radius * 0.95, tint),
        Glyph::Cross => look::cross_glyph(painter, center, radius * 0.8, tint),
        Glyph::Warning => {
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "!",
                FontId::proportional(radius * 1.1),
                tint,
            );
        }
    }
}

fn card_scale(ctx: &egui::Context) -> f32 {
    (ctx.viewport_rect().height() / 820.0).clamp(1.0, 1.8)
}

fn draw_card(
    ctx: &egui::Context,
    job: &DumpJob,
    top_offset: f32,
    accent: Color32,
    light: bool,
) -> (Rect, CardAction) {
    let area = ctx.viewport_rect();
    let scale = card_scale(ctx);
    let card = describe(job, accent);
    let width = 372.0 * scale;
    let height = if card.wrap_detail { 120.0 } else { 88.0 } * scale;
    let margin = 22.0 * scale;
    let pos = Pos2::new(
        area.max.x - margin - width,
        area.min.y + top_offset + margin,
    );
    let size = Vec2::new(width, height);
    let pal = look::palette(if light { 1.0 } else { 0.0 });
    let time = ctx.input(|input| input.time);
    let action = egui::Area::new(egui::Id::new("dump_card"))
        .order(egui::Order::Tooltip)
        .fixed_pos(pos)
        .interactable(true)
        .show(ctx, |ui| {
            let (rect, body) = ui.allocate_exact_size(size, Sense::click());
            let painter = ui.painter().clone();
            let fill = if light {
                Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 250)
            } else {
                Color32::from_rgba_unmultiplied(0x17, 0x17, 0x1F, 246)
            };
            look::soft_shadow(&painter, rect, 14.0 * scale, 1.4);
            look::card(
                &painter,
                rect,
                14.0 * scale,
                fill,
                if body.hovered() && !card.running {
                    card.tint
                } else {
                    pal.border
                },
                1.0,
            );
            let strip = Rect::from_min_size(
                rect.min + Vec2::new(0.0, 14.0 * scale),
                Vec2::new(3.5 * scale, rect.height() - 28.0 * scale),
            );
            painter.rect_filled(strip, CornerRadius::from(2.0 * scale), card.tint);

            let radius = 21.0 * scale;
            let icon = Pos2::new(rect.min.x + 40.0 * scale, rect.min.y + 37.0 * scale);
            paint_glyph(&painter, icon, radius, card.glyph, card.tint, time);

            let text_x = icon.x + radius + 14.0 * scale;
            let button_size = Vec2::new(88.0 * scale, 24.0 * scale);
            let button_y = if card.wrap_detail {
                rect.max.y - 14.0 * scale - button_size.y
            } else {
                rect.min.y + 49.0 * scale - button_size.y * 0.5
            };
            let button_rect = Rect::from_min_size(
                Pos2::new(rect.max.x - 16.0 * scale - button_size.x, button_y),
                button_size,
            );
            let title_font = FontId::proportional(15.5 * scale);
            let detail_font = FontId::proportional(13.0 * scale);
            let title_width = rect.max.x - text_x - 16.0 * scale;
            let detail_width = if card.wrap_detail {
                title_width
            } else {
                button_rect.min.x - 10.0 * scale - text_x
            };
            let title = look::ellipsize(ui, &card.title, &title_font, title_width);
            painter.text(
                Pos2::new(text_x, rect.min.y + 25.0 * scale),
                egui::Align2::LEFT_CENTER,
                title,
                title_font,
                pal.text,
            );
            if card.wrap_detail {
                let mut layout = egui::text::LayoutJob::simple(
                    card.detail.clone(),
                    detail_font,
                    pal.muted,
                    detail_width,
                );
                layout.wrap.max_rows = 2;
                layout.wrap.break_anywhere = true;
                layout.wrap.overflow_character = Some('…');
                let galley = painter.layout_job(layout);
                painter.galley(
                    Pos2::new(text_x, rect.min.y + 42.0 * scale),
                    galley,
                    pal.muted,
                );
            } else {
                let detail = look::ellipsize(ui, &card.detail, &detail_font, detail_width);
                painter.text(
                    Pos2::new(text_x, rect.min.y + 49.0 * scale),
                    egui::Align2::LEFT_CENTER,
                    detail,
                    detail_font,
                    pal.muted,
                );
            }

            if card.running {
                let track = Rect::from_min_size(
                    Pos2::new(rect.min.x + 18.0 * scale, rect.min.y + 68.0 * scale),
                    Vec2::new(rect.width() - 36.0 * scale, 5.0 * scale),
                );
                let rounding = CornerRadius::from(2.5 * scale);
                painter.rect_filled(track, rounding, pal.input);
                let fill_rect = match card.fraction {
                    Some(fraction) => Rect::from_min_size(
                        track.min,
                        Vec2::new(track.width() * fraction, track.height()),
                    ),
                    None => {
                        let segment = track.width() * 0.3;
                        let travel = (time * 0.8).fract() as f32;
                        Rect::from_min_size(
                            Pos2::new(
                                track.min.x + (track.width() - segment) * travel,
                                track.min.y,
                            ),
                            Vec2::new(segment, track.height()),
                        )
                    }
                };
                painter.rect_filled(fill_rect, rounding, card.tint);
            }
            if let Some(remaining) = card.remaining {
                let bar = Rect::from_min_size(
                    Pos2::new(rect.min.x + 18.0 * scale, rect.max.y - 5.0 * scale),
                    Vec2::new((rect.width() - 36.0 * scale) * remaining, 2.0 * scale),
                );
                painter.rect_filled(bar, CornerRadius::from(1.0), look::alpha(card.tint, 0.5));
            }

            let button = look::button(
                ui,
                &painter,
                button_rect,
                card.button,
                card.button_kind,
                accent,
                pal,
                false,
                card.button_enabled,
                13.0 * scale,
            );
            if button.clicked() {
                card.button_action
            } else if !card.running && body.clicked() {
                CardAction::Dismiss
            } else {
                CardAction::None
            }
        })
        .inner;
    (Rect::from_min_size(pos, size), action)
}

impl HorizonApp {
    pub(super) fn start_dump(&mut self, path: PathBuf, ctx: &egui::Context) {
        if self.dump_job.as_ref().is_some_and(DumpJob::is_running) {
            crate::ui_audio::play(crate::ui_audio::Sfx::Error);
            return;
        }
        let title = self
            .library
            .index_of_path(&path)
            .map(|index| self.library.games[index].title.clone())
            .unwrap_or_else(|| {
                path.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });
        let Some(parent) = rfd::FileDialog::new()
            .set_title(format!("Dump RomFS & ExeFS of {title}"))
            .pick_folder()
        else {
            return;
        };
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        self.dump_job = Some(DumpJob::start(path, title, parent, ctx));
    }

    pub(super) fn drive_dump(&mut self, ctx: &egui::Context, top_offset: f32) -> f32 {
        if self.dump_job.is_none() {
            return 0.0;
        }
        let accent = self.theme_accent();
        let light = self.app_settings.light_mode;
        let Some(job) = self.dump_job.as_mut() else {
            return 0.0;
        };
        if let Some(outcome) = job.poll() {
            crate::ui_audio::play(match outcome {
                Outcome::Done(_) => crate::ui_audio::Sfx::WhistleOk,
                Outcome::Cancelled => crate::ui_audio::Sfx::Back,
                Outcome::Failed(_) => crate::ui_audio::Sfx::Error,
            });
        }
        let expired = matches!(
            job.outcome(),
            Some((Outcome::Done(_) | Outcome::Cancelled, elapsed)) if elapsed.as_secs_f32() > FINISHED_VISIBLE
        );
        let (rect, action) = if expired {
            (None, CardAction::Dismiss)
        } else {
            let (rect, action) = draw_card(ctx, job, top_offset, accent, light);
            (Some(rect), action)
        };
        job.card = rect;
        let mut dismiss = action == CardAction::Dismiss;
        match action {
            CardAction::Cancel => job.cancel(),
            CardAction::OpenFolder => {
                if let Some(destination) = job.destination() {
                    reveal_in_file_manager(destination);
                }
                dismiss = true;
            }
            CardAction::Dismiss | CardAction::None => {}
        }
        if dismiss {
            self.dump_job = None;
            return 0.0;
        }
        ctx.request_repaint_after(FRAME_INTERVAL);
        rect.map_or(0.0, |rect| rect.height() + 12.0 * card_scale(ctx))
    }
}

pub(super) fn card_rect(job: Option<&DumpJob>) -> Option<Rect> {
    job.and_then(|job| job.card).map(|rect| rect.expand(8.0))
}
