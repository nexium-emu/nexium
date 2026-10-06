use crate::app_settings::{
    AppSettings, AspectMode, CpuBackend, FilterMode, LogLevel, MotionRecenterButton,
};
use crate::audio::{
    current_stream_info, list_output_devices, push_test_tone, set_master_volume, AudioStreamInfo,
};
use crate::boot::EmulationHandle;
use crate::controller_art::{PRO_BODY, PRO_LEFT_HANDLE};
use crate::controller_config::{ControllerConfig, SwitchButton};
use crate::debugger::DebuggerState;
use crate::input::{InputBackend, InputSnapshot};
use crate::performance::PerformanceMonitor;
use eframe::egui;
use eframe::egui::{Color32, CornerRadius, FontId, Sense, Stroke, Vec2};
use std::sync::Arc;

mod nextendo_glue;

const BG: Color32 = Color32::from_rgb(0x0F, 0x0F, 0x11);
const BG_RAISED: Color32 = Color32::from_rgb(0x18, 0x18, 0x1C);
const BG_INPUT: Color32 = Color32::from_rgb(0x20, 0x20, 0x26);
const BORDER: Color32 = Color32::from_rgb(0x2A, 0x2A, 0x32);
const ACCENT: Color32 = Color32::from_rgb(0x2F, 0xB4, 0xEF);
const ACCENT_HV: Color32 = Color32::from_rgb(0x5C, 0xF2, 0xFF);
const ACCENT_DK: Color32 = Color32::from_rgb(0x0D, 0x6F, 0xBB);
const DANGER: Color32 = Color32::from_rgb(0xE0, 0x2A, 0x2A);
const DANGER_HV: Color32 = Color32::from_rgb(0xF0, 0x3C, 0x3C);
const TEXT: Color32 = Color32::from_rgb(0xEC, 0xEC, 0xF0);
const MUTED: Color32 = Color32::from_rgb(0x70, 0x70, 0x80);
const GREEN: Color32 = Color32::from_rgb(0x3C, 0xD4, 0x5C);
const AMBER: Color32 = Color32::from_rgb(0xF5, 0xA6, 0x23);

struct NativeGameTexture {
    texture: eframe::wgpu::Texture,
    id: egui::TextureId,
    width: u32,
    height: u32,
    filter: FilterMode,
    scaler: Option<crate::frame_scaler::FrameScaler>,
    dirty: bool,
}

fn legacy_gui_upload() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_LEGACY_GUI_UPLOAD").is_some())
}

enum StatusIcon {
    Play,
    Pause,
    Stop,
    Dock,
    Handheld,
}

fn status_icon(ui: &mut egui::Ui, icon: StatusIcon, color: Color32) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(13.0, 13.0), Sense::click());
    let p = ui.painter();
    let r = rect.shrink(1.5);
    match icon {
        StatusIcon::Play => {
            p.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(r.left() + 1.5, r.top() + 0.5),
                    egui::pos2(r.right() - 0.5, r.center().y),
                    egui::pos2(r.left() + 1.5, r.bottom() - 0.5),
                ],
                color,
                Stroke::NONE,
            ));
        }
        StatusIcon::Pause => {
            let w = r.width() * 0.3;
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(r.left() + 0.5, r.top()),
                    Vec2::new(w, r.height()),
                ),
                1.0,
                color,
            );
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(r.right() - w - 0.5, r.top()),
                    Vec2::new(w, r.height()),
                ),
                1.0,
                color,
            );
        }
        StatusIcon::Stop => {
            p.rect_filled(r.shrink(1.0), 1.5, color);
        }
        StatusIcon::Dock => {
            let screen = egui::Rect::from_min_max(
                egui::pos2(r.left() + r.width() * 0.22, r.top()),
                egui::pos2(r.right() - r.width() * 0.22, r.top() + r.height() * 0.58),
            );
            p.rect_stroke(
                screen,
                1.0,
                Stroke::new(1.2_f32, color),
                egui::StrokeKind::Outside,
            );
            let dock = egui::Rect::from_min_max(
                egui::pos2(r.left(), r.top() + r.height() * 0.48),
                r.right_bottom(),
            );
            p.rect_filled(dock, 1.5, color);
        }
        StatusIcon::Handheld => {
            let body =
                egui::Rect::from_center_size(r.center(), Vec2::new(r.width(), r.height() * 0.64));
            let jc_w = body.width() * 0.26;
            p.rect_filled(
                egui::Rect::from_min_size(body.left_top(), Vec2::new(jc_w, body.height())),
                2.0,
                color,
            );
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(body.right() - jc_w, body.top()),
                    Vec2::new(jc_w, body.height()),
                ),
                2.0,
                color,
            );
            p.rect_stroke(
                body,
                2.0,
                Stroke::new(1.1_f32, color),
                egui::StrokeKind::Outside,
            );
        }
    }
    resp
}

const PERF_GRAPH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
const PERF_READOUT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

fn gui_rate_stats(kind: usize, count: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_GUI_PROFILE").is_some()) {
        return;
    }
    static COUNTS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static LAST: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
        std::sync::OnceLock::new();
    COUNTS[kind.min(2)].fetch_add(count, Ordering::Relaxed);
    let last = LAST.get_or_init(|| std::sync::Mutex::new(std::time::Instant::now()));
    let Ok(mut guard) = last.try_lock() else {
        return;
    };
    let elapsed = guard.elapsed().as_secs_f64();
    if elapsed >= 2.0 {
        *guard = std::time::Instant::now();
        let u = COUNTS[0].swap(0, Ordering::Relaxed);
        let r = COUNTS[1].swap(0, Ordering::Relaxed);
        let d = COUNTS[2].swap(0, Ordering::Relaxed);
        log::info!(
            "[gui] updates/s={:.0} frames_new/s={:.0} frames_extra_dropped/s={:.0}",
            u as f64 / elapsed,
            r as f64 / elapsed,
            d as f64 / elapsed
        );
    }
}

fn take_speed_limit_shortcut(input: &mut egui::InputState) -> bool {
    take_ctrl_shortcut(input, egui::Key::U)
}

fn take_ctrl_shortcut(input: &mut egui::InputState, shortcut: egui::Key) -> bool {
    let mut toggle = false;
    input.events.retain(|event| {
        if let egui::Event::Key { key, pressed: true, repeat, modifiers, .. } = event {
            if *key == shortcut && modifiers.ctrl && !modifiers.alt && !modifiers.shift {
                toggle |= !repeat;
                return false;
            }
        }
        true
    });
    toggle
}

fn take_plain_shortcut(input: &mut egui::InputState, shortcut: egui::Key) -> bool {
    let mut hit = false;
    input.events.retain(|event| {
        if let egui::Event::Key { key, pressed: true, repeat, modifiers, .. } = event {
            if *key == shortcut && !modifiers.ctrl && !modifiers.alt {
                hit |= !repeat;
                return false;
            }
        }
        true
    });
    hit
}

fn motion_recenter_key(name: &str) -> Option<egui::Key> {
    egui::Key::ALL
        .iter()
        .copied()
        .find(|key| format!("{:?}", key).eq_ignore_ascii_case(name))
}

fn rising_edge(held: &mut bool, down: bool) -> bool {
    let edge = down && !*held;
    *held = down;
    edge
}

fn request_game_frame_repaint(ctx: &egui::Context) {
    ctx.request_repaint_after(std::time::Duration::from_nanos(1));
}

fn request_idle_repaint(ctx: &egui::Context, idle_for: std::time::Duration) {
    let predicted_dt = ctx.input(|input| {
        std::time::Duration::try_from_secs_f32(input.predicted_dt).unwrap_or_default()
    });
    ctx.request_repaint_after(idle_for.saturating_add(predicted_dt));
}

fn take_next_game_frame(
    frame_rx: &std::sync::mpsc::Receiver<crate::boot::Frame>,
) -> Option<crate::boot::Frame> {
    loop {
        match frame_rx.try_recv() {
            Ok(frame) if frame.width > 0 && frame.height > 0 && !frame.pixels.is_empty() => {
                return Some(frame);
            }
            Ok(_) => {}
            Err(std::sync::mpsc::TryRecvError::Empty)
            | Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
        }
    }
}

fn game_display_rect(
    panel: egui::Rect,
    texture_size: Vec2,
    aspect: AspectMode,
    output_scale: u8,
    pixels_per_point: f32,
) -> egui::Rect {
    if !panel.is_positive() || texture_size.x <= 0.0 || texture_size.y <= 0.0 {
        return egui::Rect::from_center_size(panel.center(), Vec2::ZERO);
    }
    if aspect == AspectMode::Integer {
        let pixel_min = (panel.min.to_vec2() * pixels_per_point).ceil();
        let pixel_max = (panel.max.to_vec2() * pixels_per_point).floor();
        let available = pixel_max - pixel_min;
        if available.x <= 0.0 || available.y <= 0.0 {
            return egui::Rect::from_center_size(panel.center(), Vec2::ZERO);
        }
        let fit = (available.x / texture_size.x).min(available.y / texture_size.y);
        let scale = if fit >= 1.0 {
            fit.floor().min(output_scale.max(1) as f32)
        } else {
            fit
        };
        let size = (texture_size * scale).floor();
        let origin = pixel_min + ((available - size) * 0.5).floor();
        return egui::Rect::from_min_max(
            (origin / pixels_per_point).to_pos2(),
            ((origin + size) / pixels_per_point).to_pos2(),
        );
    }
    let available = panel.size();
    let size = if aspect == AspectMode::Stretch {
        available
    } else {
        texture_size * (available.x / texture_size.x).min(available.y / texture_size.y)
    };
    egui::Rect::from_center_size(panel.center(), size)
}

#[cfg(test)]
mod game_display_tests {
    use super::game_display_rect;
    use crate::app_settings::AspectMode;
    use egui::{pos2, vec2, Rect};

    #[test]
    fn integer_scale_uses_physical_pixels_at_fractional_dpi() {
        let panel = Rect::from_min_max(pos2(0.0, 30.4), pos2(1280.0, 696.8));
        let rect = game_display_rect(panel, vec2(1280.0, 720.0), AspectMode::Integer, 1, 1.25);
        assert!(panel.contains_rect(rect));
        assert_eq!(rect.size() * 1.25, vec2(1280.0, 720.0));
        assert_eq!(rect.min.to_vec2() * 1.25, vec2(160.0, 94.0));
        assert!(rect.max.y * 1.25 <= 871.0);
    }

    #[test]
    fn integer_scale_shrinks_to_fit_below_one_times() {
        let panel = Rect::from_min_max(pos2(0.0, 30.0), pos2(960.0, 510.0));
        let rect = game_display_rect(panel, vec2(1280.0, 720.0), AspectMode::Integer, 3, 1.0);
        assert!(panel.contains_rect(rect));
        assert_eq!(rect.size(), vec2(853.0, 480.0));
        assert_eq!(rect.max.y, 510.0);
    }

    #[test]
    fn integer_scale_respects_requested_scale_and_panel_limit() {
        let panel = Rect::from_min_size(pos2(0.0, 20.0), vec2(2048.0, 1200.0));
        let two = game_display_rect(panel, vec2(1280.0, 720.0), AspectMode::Integer, 3, 1.5);
        let one = game_display_rect(panel, vec2(1280.0, 720.0), AspectMode::Integer, 1, 1.5);
        assert!(panel.contains_rect(two));
        assert!(panel.contains_rect(one));
        assert_eq!((two.size() * 1.5).round(), vec2(2560.0, 1440.0));
        assert_eq!((one.size() * 1.5).round(), vec2(1280.0, 720.0));
    }

    #[test]
    fn integer_scale_rounds_panel_edges_inward() {
        let panel = Rect::from_min_max(pos2(0.25, 30.25), pos2(1024.5, 606.1));
        let rect = game_display_rect(panel, vec2(1280.0, 720.0), AspectMode::Integer, 1, 1.25);
        assert!(panel.contains_rect(rect));
        assert!((rect.min.x * 1.25).round() >= (panel.min.x * 1.25).ceil());
        assert!((rect.min.y * 1.25).round() >= (panel.min.y * 1.25).ceil());
        assert!((rect.max.x * 1.25).round() <= (panel.max.x * 1.25).floor());
        assert!((rect.max.y * 1.25).round() <= (panel.max.y * 1.25).floor());
    }
}

#[cfg(test)]
mod frame_receive_tests {
    use super::{request_game_frame_repaint, take_next_game_frame};
    use crate::boot::Frame;

    #[test]
    fn speed_shortcut_consumes_ctrl_u_without_repeating_or_consuming_plain_u() {
        let event = |ctrl, repeat| egui::Event::Key {
            key: egui::Key::U,
            physical_key: None,
            pressed: true,
            repeat,
            modifiers: egui::Modifiers { ctrl, ..Default::default() },
        };
        let mut input = egui::InputState::default();
        input.events = vec![event(false, false)];
        assert!(!super::take_speed_limit_shortcut(&mut input));
        assert_eq!(input.events.len(), 1);
        input.events = vec![event(true, false)];
        assert!(super::take_speed_limit_shortcut(&mut input));
        assert!(input.events.is_empty());
        input.events = vec![event(true, true)];
        assert!(!super::take_speed_limit_shortcut(&mut input));
        assert!(input.events.is_empty());
    }

    #[test]
    fn game_frame_notification_does_not_schedule_a_second_repaint() {
        let ctx = egui::Context::default();
        for _ in 0..4 {
            ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        }
        request_game_frame_repaint(&ctx);
        let mut output = ctx.run_ui(Default::default(), |_| {});
        output.textures_delta.clear();
        assert_eq!(
            output.viewport_output[&egui::ViewportId::ROOT].repaint_delay,
            std::time::Duration::MAX
        );
    }

    fn frame(marker: u8) -> Frame {
        Frame {
            width: 1,
            height: 1,
            pixels: vec![marker, 0, 0, 0],
            depth: None,
        }
    }

    #[test]
    fn receives_one_fifo_frame_per_paint() {
        let (tx, rx) = std::sync::mpsc::sync_channel(3);
        tx.send(frame(1)).unwrap();
        tx.send(frame(2)).unwrap();
        tx.send(frame(3)).unwrap();

        assert_eq!(take_next_game_frame(&rx).unwrap().pixels[0], 1);
        assert_eq!(take_next_game_frame(&rx).unwrap().pixels[0], 2);
        assert_eq!(take_next_game_frame(&rx).unwrap().pixels[0], 3);
        assert!(take_next_game_frame(&rx).is_none());
    }

    #[test]
    fn skips_only_malformed_frames_before_the_next_valid_frame() {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        tx.send(Frame {
            width: 0,
            height: 1,
            pixels: vec![0; 4],
            depth: None,
        })
        .unwrap();
        tx.send(frame(7)).unwrap();

        assert_eq!(take_next_game_frame(&rx).unwrap().pixels[0], 7);
        assert!(take_next_game_frame(&rx).is_none());
    }
}

#[cfg(test)]
mod motion_settings_tests {
    use super::{
        motion_calibration_text, motion_note, motion_recenter_key, motion_recenter_text,
        motion_row_values, motion_status_text, motion_summary, recenter_key_conflict,
        recenter_key_label, recenter_key_press, rising_edge, take_plain_shortcut, InputDevice,
        MotionKeyCapture, MOTION_CALIBRATING, MOTION_MISSING, MOTION_NO_DATA, MOTION_NO_DATA_NOTE,
        MOTION_UNCALIBRATED,
    };
    use crate::app_settings::{AppSettings, MotionRecenterButton};
    use crate::controller_config::{ControllerConfig, SwitchButton};
    use egui::{Event, InputState, Key, Modifiers};
    use nexium_core::hid_motion::CalibrationState;

    fn key_event(key: Key, ctrl: bool, repeat: bool) -> Event {
        Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat,
            modifiers: Modifiers { ctrl, ..Default::default() },
        }
    }

    #[test]
    fn recenter_hotkey_fires_once_per_press_and_ignores_ctrl_chords() {
        let mut input = InputState::default();
        input.events = vec![key_event(Key::F8, false, false)];
        assert!(take_plain_shortcut(&mut input, Key::F8));
        assert!(input.events.is_empty());
        input.events = vec![key_event(Key::F8, false, true)];
        assert!(!take_plain_shortcut(&mut input, Key::F8));
        assert!(input.events.is_empty());
        input.events = vec![key_event(Key::F8, true, false), key_event(Key::F9, false, false)];
        assert!(!take_plain_shortcut(&mut input, Key::F8));
        assert_eq!(input.events.len(), 2);
    }

    #[test]
    fn recenter_button_fires_on_the_rising_edge_only() {
        let mut held = false;
        let edges: Vec<bool> = [false, true, true, false, true]
            .into_iter()
            .map(|down| rising_edge(&mut held, down))
            .collect();
        assert_eq!(edges, [false, true, false, false, true]);
    }

    #[test]
    fn recenter_key_names_resolve_like_bindings() {
        assert_eq!(motion_recenter_key("F8"), Some(Key::F8));
        assert_eq!(motion_recenter_key("f8"), Some(Key::F8));
        assert_eq!(motion_recenter_key("ArrowUp"), Some(Key::ArrowUp));
        assert_eq!(motion_recenter_key(""), None);
        assert_eq!(motion_recenter_key("NotAKey"), None);
    }

    #[test]
    fn motion_rows_show_status_capture_and_feedback() {
        let defaults = AppSettings::default();
        assert_eq!(
            motion_row_values(
                &defaults,
                Some("Nintendo Switch Joy-Con (L/R)"),
                false,
                false,
                "Calibrated"
            ),
            ["On", "Joy-Con (L/R)", "F8", "SL / SR", "", "Calibrated"]
        );
        assert_eq!(
            motion_row_values(&defaults, None, true, true, "")[1..],
            ["No motion controller", "Press a key\u{2026}", "SL / SR", "Done", ""]
        );
        let off = AppSettings {
            motion_enabled: false,
            motion_recenter_key: String::new(),
            motion_recenter_button: MotionRecenterButton::Unbound,
            ..AppSettings::default()
        };
        assert_eq!(
            motion_row_values(&off, Some("Pro Controller"), false, false, ""),
            ["Off", "Off", "\u{2014}", "None", "", ""]
        );
        assert_eq!(
            motion_status_text(true, Some("DualSense Wireless Controller")),
            "DualSense Wireless Controller"
        );
    }

    #[test]
    fn recenter_key_capture_refuses_menu_home_and_bound_keys() {
        let mut config = ControllerConfig::default();
        let capture = |config: &ControllerConfig, keys: &[(Key, bool)]| {
            let events: Vec<Event> =
                keys.iter().map(|(key, repeat)| key_event(*key, false, *repeat)).collect();
            recenter_key_press(&events, config)
        };
        assert!(matches!(
            capture(&config, &[(Key::F8, false)]),
            Some(MotionKeyCapture::Set(key)) if key == "F8"
        ));
        for key in [Key::Backspace, Key::Delete] {
            assert!(matches!(
                capture(&config, &[(key, false)]),
                Some(MotionKeyCapture::Set(key)) if key.is_empty()
            ));
        }
        assert!(matches!(
            capture(&config, &[(Key::Escape, false)]),
            Some(MotionKeyCapture::Cancel)
        ));
        assert!(capture(&config, &[(Key::F8, true)]).is_none());
        assert!(capture(&config, &[]).is_none());
        for (key, notice) in [
            (Key::Enter, "Enter is used by the menus."),
            (Key::Space, "Space is used by the menus."),
            (Key::Tab, "Tab is used by the menus."),
            (Key::ArrowDown, "ArrowDown is used by the menus."),
            (Key::Home, "Home opens the Home menu."),
            (Key::Backtick, "Backtick opens the Home menu."),
            (Key::R, "R is already bound to ZL."),
            (Key::C, "C is already bound to A."),
        ] {
            assert!(matches!(
                capture(&config, &[(key, false)]),
                Some(MotionKeyCapture::Refuse(text)) if text == notice
            ));
        }
        assert!(matches!(
            capture(&config, &[(Key::ArrowDown, false), (Key::P, false)]),
            Some(MotionKeyCapture::Set(key)) if key == "P"
        ));
        assert_eq!(recenter_key_conflict(Key::F8, &config), None);
        config.set_binding(SwitchButton::X, "f8".to_string());
        assert!(matches!(
            capture(&config, &[(Key::F8, false)]),
            Some(MotionKeyCapture::Refuse(text)) if text == "F8 is already bound to X."
        ));
        assert_eq!(
            recenter_key_conflict(Key::F8, &config).as_deref(),
            Some("F8 is already bound to X.")
        );
    }

    #[test]
    fn motion_hints_share_the_recenter_text_and_missing_button_note() {
        let defaults = AppSettings::default();
        let mut config = ControllerConfig::default();
        assert_eq!(motion_recenter_text(&defaults, &config, false), "F8 or SL / SR");
        assert_eq!(
            motion_recenter_text(&defaults, &config, true),
            "F8 or SL / SR (not on this controller)"
        );
        let cleared = AppSettings {
            motion_recenter_key: String::new(),
            motion_recenter_button: MotionRecenterButton::Unbound,
            ..AppSettings::default()
        };
        assert_eq!(motion_recenter_text(&cleared, &config, false), "\u{2014}");
        assert_eq!(motion_note(&defaults, &config, None, false), None);
        assert_eq!(motion_note(&cleared, &config, None, true), None);
        assert_eq!(
            motion_note(&defaults, &config, None, true).as_deref(),
            Some("SL / SR is not on this controller.")
        );
        assert_eq!(
            motion_note(&defaults, &config, Some("Enter is used by the menus."), true).as_deref(),
            Some("Enter is used by the menus.")
        );
        let stale = AppSettings {
            motion_recenter_key: "ArrowDown".to_string(),
            ..AppSettings::default()
        };
        assert_eq!(
            motion_note(&stale, &config, None, true).as_deref(),
            Some("ArrowDown is used by the menus.")
        );
        assert_eq!(
            motion_summary(&defaults, &config),
            "Motion controls: On \u{b7} Recenter key: F8 \u{b7} change under Input Device \u{203a} Motion"
        );
        assert_eq!(
            motion_summary(&cleared, &config),
            "Motion controls: On \u{b7} Recenter key: \u{2014} \u{b7} change under Input Device \u{203a} Motion"
        );
        assert_eq!(motion_status_text(true, None), MOTION_MISSING);
        assert_eq!(recenter_key_label(&defaults, &config), "F8");
        config.set_binding(SwitchButton::X, "F8".to_string());
        assert_eq!(recenter_key_label(&defaults, &config), "F8 (off)");
        assert_eq!(recenter_key_label(&cleared, &config), "\u{2014}");
        assert_eq!(motion_recenter_text(&defaults, &config, false), "F8 (off) or SL / SR");
        assert_eq!(
            motion_summary(&defaults, &config),
            "Motion controls: On \u{b7} Recenter key: F8 (off) \u{b7} change under Input Device \u{203a} Motion"
        );
        assert_eq!(
            motion_note(&defaults, &config, None, false).as_deref(),
            Some("F8 is already bound to X.")
        );
    }

    #[test]
    fn calibration_text_reports_the_least_calibrated_source() {
        let hint = "SDL HIDAPI disabled by environment; Joy-Con motion unavailable";
        let uncalibrated = Some(CalibrationState::Uncalibrated);
        let capturing = Some(CalibrationState::Capturing);
        let calibrated = Some(CalibrationState::Calibrated);
        let fresh = [false; 3];
        assert_eq!(
            motion_calibration_text(false, [uncalibrated; 3], [true; 3], Some(hint)),
            ("", None)
        );
        assert_eq!(motion_calibration_text(true, [None; 3], fresh, None), ("", None));
        assert_eq!(motion_calibration_text(true, [None; 3], fresh, Some(hint)), ("", Some(hint)));
        assert_eq!(
            motion_calibration_text(true, [None, calibrated, capturing], fresh, None),
            ("Calibrating\u{2026}", Some(MOTION_CALIBRATING))
        );
        assert_eq!(
            motion_calibration_text(true, [None, uncalibrated, capturing], fresh, Some(hint)),
            ("Not calibrated", Some(MOTION_UNCALIBRATED))
        );
        assert_eq!(
            motion_calibration_text(true, [calibrated, None, None], fresh, None),
            ("Calibrated", None)
        );
        let right_silent = [false, true, false];
        let primary_silent = [true, false, false];
        assert_eq!(
            motion_calibration_text(true, [None, uncalibrated, calibrated], right_silent, None),
            (MOTION_NO_DATA, Some(MOTION_NO_DATA_NOTE))
        );
        assert_eq!(
            motion_calibration_text(true, [None, calibrated, calibrated], primary_silent, None),
            ("Calibrated", None)
        );
    }

    #[test]
    fn input_device_selector_cycles_through_motion() {
        assert!(InputDevice::Keyboard.cycle(1) == InputDevice::Gamepad);
        assert!(InputDevice::Gamepad.cycle(1) == InputDevice::Motion);
        assert!(InputDevice::Motion.cycle(1) == InputDevice::Vibration);
        assert!(InputDevice::Vibration.cycle(1) == InputDevice::Keyboard);
        assert!(InputDevice::Keyboard.cycle(-1) == InputDevice::Vibration);
    }
}

#[cfg(test)]
mod vibration_settings_tests {
    use super::{bar_toggle_label, shader_build_label, step_vibration_strength, vibration_row_values};
    use crate::app_settings::AppSettings;

    #[test]
    fn vibration_rows_and_strength_steps() {
        assert_eq!(
            vibration_row_values(&AppSettings::default(), "HD \u{b7} Joy-Con (L/R)"),
            ["On", "HD \u{b7} Joy-Con (L/R)", "100%", ""]
        );
        let off = AppSettings {
            vibration_enabled: false,
            vibration_strength: 40,
            ..AppSettings::default()
        };
        assert_eq!(vibration_row_values(&off, "Off"), ["Off", "Off", "40%", ""]);
        assert_eq!(step_vibration_strength(100, 1), 100);
        assert_eq!(step_vibration_strength(10, -1), 10);
        assert_eq!(step_vibration_strength(100, -1), 90);
        assert_eq!(step_vibration_strength(40, 1), 50);
        assert_eq!(step_vibration_strength(37, -1), 30);
        assert_eq!(step_vibration_strength(0, 1), 20);
    }

    #[test]
    fn compact_bar_labels_drop_the_state() {
        assert_eq!(bar_toggle_label("Motion", true, false), "Motion: On");
        assert_eq!(bar_toggle_label("Motion", false, false), "Motion: Off");
        assert_eq!(bar_toggle_label("Vibration", true, false), "Vibration: On");
        assert_eq!(bar_toggle_label("Vibration", false, true), "Vibration");
        assert_eq!(shader_build_label(12, false), "Building shaders: 12");
        assert_eq!(shader_build_label(12, true), "Shaders: 12");
    }
}

#[cfg(test)]
mod controller_choice_tests {
    use super::{pad_choice_row, pad_choice_text};

    #[test]
    fn controller_pickers_name_the_automatic_choice() {
        let pads = [
            (1, "Nintendo Switch Joy-Con (L/R)".to_string()),
            (2, "Nintendo N64 Controller".to_string()),
        ];
        let pair = Some("Nintendo Switch Joy-Con (L/R)");
        let n64 = Some("Nintendo N64 Controller");
        let pro = Some("Nintendo Switch Pro Controller");
        assert_eq!(
            pad_choice_text(None, &pads, Some(1)),
            "Automatic \u{b7} Joy-Con (L/R)"
        );
        assert_eq!(
            pad_choice_text(None, &pads, Some(2)),
            "Automatic \u{b7} N64 Controller"
        );
        assert_eq!(
            pad_choice_text(None, &pads, Some(3)),
            "Automatic \u{b7} Gamepad"
        );
        assert_eq!(
            pad_choice_text(None, &pads, None),
            "Automatic \u{b7} No Gamepad Connected"
        );
        assert_eq!(
            pad_choice_text(pair, &pads, Some(1)),
            "Nintendo Switch Joy-Con (L/R)"
        );
        assert_eq!(
            pad_choice_text(pro, &pads, Some(1)),
            "Pro Controller \u{b7} not connected"
        );
        assert_eq!(
            pad_choice_text(pro, &pads[..0], None),
            "Pro Controller \u{b7} not connected"
        );
        assert_eq!(
            pad_choice_text(n64, &pads, Some(1)),
            "N64 Controller \u{b7} not responding"
        );
        assert_eq!(
            pad_choice_text(n64, &pads[..0], None),
            "N64 Controller \u{b7} not connected"
        );
        assert_eq!(
            pad_choice_text(pair, &pads, None),
            "Joy-Con (L/R) \u{b7} not responding"
        );
    }

    #[test]
    fn controller_pickers_mark_the_saved_choice() {
        let pads = [
            (1, "Nintendo Switch Pro Controller".to_string()),
            (2, "Nintendo N64 Controller".to_string()),
            (3, "Nintendo Switch Pro Controller".to_string()),
        ];
        let pro = Some("Nintendo Switch Pro Controller");
        let n64 = Some("Nintendo N64 Controller");
        let pair = Some("Nintendo Switch Joy-Con (L/R)");
        assert_eq!(pad_choice_row(None, &pads, Some(2)), 0);
        assert_eq!(pad_choice_row(None, &pads, None), 0);
        assert_eq!(pad_choice_row(pro, &pads, Some(3)), 3);
        assert_eq!(pad_choice_row(pro, &pads, Some(1)), 1);
        assert_eq!(pad_choice_row(pro, &pads, Some(2)), 1);
        assert_eq!(pad_choice_row(n64, &pads, None), 2);
        assert_eq!(pad_choice_row(pair, &pads, Some(1)), 4);
        assert_eq!(pad_choice_row(pro, &pads[..0], None), 1);
    }
}

fn diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_DIAG").is_some())
}

pub struct HorizonApp {
    nro_path: String,
    emulation_handle: Option<EmulationHandle>,
    game_texture: Option<egui::TextureHandle>,
    egui_ctx: egui::Context,
    wgpu_state: Option<eframe::egui_wgpu::RenderState>,
    game_texture_native: Option<NativeGameTexture>,
    native_game: Option<crate::native_game::NativeGameWindow>,
    native_overlay_rect: Option<egui::Rect>,
    game_bar_tab_rect: Option<egui::Rect>,
    game_depth: Option<std::sync::Arc<nexium_gpu::PresentDepth>>,
    game_depth_seq: u64,
    last_game_rect: Option<egui::Rect>,
    mouse_wheel_accum: egui::Vec2,
    show_settings: bool,
    settings_tab: SettingsTab,
    prefs_anim: f32,
    prefs_focus: bool,
    prefs_row: usize,
    prefs_col: usize,
    prefs_col1_row: usize,
    prefs_dropdown_open: bool,
    prefs_dropdown_sel: usize,
    prefs_lr_held: bool,
    prefs_nav_cd: f64,
    prefs_nav_held_dir: u8,
    prefs_nav_held_since: f64,
    prefs_ab_held: bool,
    prefs_key_editing: bool,
    prefs_key_menu: bool,
    prefs_ctrl_scroll: usize,
    prefs_key_caret: usize,
    prefs_key_anchor: usize,
    vkeyboard: crate::vkeyboard::VirtualKeyboard,
    vk_target: VkTarget,
    swkbd_text: String,
    swkbd_gen: u64,
    swkbd_max: usize,
    swkbd_min: usize,
    swkbd_title: String,
    swkbd_password: bool,
    updater: crate::updater::Updater,
    update_restart_shown: bool,
    input: Option<InputBackend>,
    last_input: InputSnapshot,
    debugger: DebuggerState,
    performance: PerformanceMonitor,
    log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    controller_config: ControllerConfig,
    rebinding: Option<SwitchButton>,
    rebinding_pad: Option<SwitchButton>,
    rebinding_pad_wait_release: bool,
    prefs_rebind_cool: bool,
    prefs_enter_held: bool,
    input_device: InputDevice,
    motion_key_capture: bool,
    motion_key_notice: Option<String>,
    startup_gpu_device: Option<String>,
    startup_motion_enabled: bool,
    app_settings: AppSettings,
    last_buttons_logged: u64,
    last_sticks_logged: [i32; 4],
    audio_device_cache: Option<Vec<String>>,
    splash: crate::splash::Splash,
    library: crate::library::Library,
    stop_fade: Option<std::time::Instant>,
    pause_anim: Option<std::time::Instant>,
    resume_anim: Option<std::time::Instant>,
    stop_anim: Option<std::time::Instant>,
    pill_fade: Option<std::time::Instant>,
    playing_path: Option<std::path::PathBuf>,
    last_home: bool,
    motion_recenter_held: bool,
    profile_texture: Option<egui::TextureHandle>,
    profile_reload: bool,
    show_profile: bool,
    profile_anim: f32,
    profile: crate::profile::ProfileState,
    play_times: crate::playtime::PlayTimes,
    last_playtime_save: std::time::Instant,
    pub carousel: crate::carousel::CarouselState,
    icon_picker: Option<IconPicker>,
    shop: crate::shop::ShopState,
    active_downloads: Vec<(
        String,
        std::sync::Arc<std::sync::Mutex<crate::library::DownloadInfo>>,
    )>,
    download_toast: Option<(String, std::time::Instant)>,
    motion_toast: Option<(&'static str, std::time::Instant)>,
    game_bar_spare: f32,
    game_bar_compact: bool,
    game_bar_terse: bool,
    game_bar_terse_spare: f32,
    top_bar_spare: f32,
    top_bar_compact: bool,
    top_bar_terse: bool,
    top_bar_terse_spare: f32,
    carousel_settings_open: bool,
    cs_anim: f32,
    cs_tab: usize,
    cs_scroll: f32,
    cs_selected: usize,
    cs_focus_grid: bool,
    cs_nav_cd: f64,
    cs_nav_held_dir: u8,
    cs_nav_held_since: f64,
    cs_ab_held: bool,
    cs_list_picks: Vec<std::path::PathBuf>,
    cs_pick_anim: std::collections::HashMap<std::path::PathBuf, (f32, f32)>,
    cs_creating: bool,
    cs_new_name: String,
    cs_new_align: crate::app_settings::ListAlignment,
    cs_dialog_row: usize,
    cs_name_editing: bool,
    cs_name_caret: usize,
    cs_name_anchor: usize,
    cs_create_btn: f32,
    cs_plus_held: bool,
    cs_rename_idx: Option<usize>,
    cs_list_menu: Option<usize>,
    cs_list_menu_sel: usize,
    cs_x_held: bool,
    cs_build_grab: Option<usize>,
    cs_build_insert: usize,
    cs_row_scroll: f32,
    cs_grab_lift: f32,
    perf_pos: Option<egui::Pos2>,
    perf_drag: bool,
    perf_grab_off: egui::Vec2,
    perf_hist: Vec<f32>,
    perf_scale: f32,
    perf_sampled: Option<std::time::Instant>,
    perf_readout_at: Option<std::time::Instant>,
    perf_readout: (f32, f32, u64, u64),
    last_frame_res: (u32, u32),
    music_was_on: bool,
    music_fade_start: Option<std::time::Instant>,
    icon_reveal: Option<(usize, std::time::Instant, Option<egui::TextureHandle>)>,
    key_test: Option<std::sync::Arc<std::sync::Mutex<KeyTest>>>,
    key_test_result: Option<(bool, std::time::Instant)>,
    confirm: Option<ConfirmDialog>,
    teardown_at: Option<std::time::Instant>,
    pending_boot: Option<String>,
    modal_cd: f64,
    modal_hold: bool,
    modal_anim: f32,
    modal_snap: Option<ModalSnap>,
    pending_quick: Option<String>,
    modal_active_frame_start: bool,
    game_info_path: Option<std::path::PathBuf>,
    mod_manager: Option<crate::mods::ModManager>,
    content_manager: Option<crate::content_manager::ContentManager>,
    firmware_prompt: Option<crate::firmware_prompt::FirmwarePrompt>,
    firmware_status: Option<Result<Option<nexium_loader::firmware::InstalledFirmware>, String>>,
    nextendo: crate::nextendo::Nextendo,
    nextendo_state: crate::nextendo::State,
    nextendo_hub: crate::nextendo::hub::HubState,
    nextendo_catalog: crate::nextendo::Catalog,
    nextendo_toasts: crate::nextendo::toasts::Toasts,
    nextendo_reported_title: Option<u64>,
    nextendo_session_path: Option<std::path::PathBuf>,
    nextendo_decorated: Option<std::time::Instant>,
    show_nextendo: bool,
    nextendo_anim: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    General,
    Controller,
    Graphics,
    Audio,
    Emulation,
    Logging,
}

enum ConfirmKind {
    CloseGame,
    LaunchGame(String),
    QuickLaunch(String),
    DeleteList(usize),
    UpdateInstall,
    UpdateRestart,
}

enum GameInfoAction {
    Launch(std::path::PathBuf),
    ManageMods(std::path::PathBuf),
    ManageContent(std::path::PathBuf),
    ToggleFavorite(std::path::PathBuf),
    DownloadIcon(std::path::PathBuf),
}

struct ConfirmDialog {
    title: String,
    body: String,
    confirm_label: String,
    selected: usize,
    kind: ConfirmKind,
}

struct ModalSnap {
    teardown: bool,
    title: String,
    body: String,
    label: String,
}

#[derive(Default)]
struct IconFetch {
    done: bool,
    error: Option<String>,
    items: Vec<(String, Option<Vec<u8>>)>,
}

#[derive(Default)]
struct IconApply {
    done: bool,
    bytes: Option<Vec<u8>>,
}

#[derive(Default)]
struct KeyTest {
    done: bool,
    ok: bool,
}

struct IconPicker {
    game_idx: usize,
    game_path: std::path::PathBuf,
    search: String,
    editing: bool,
    anim: f32,
    selected: usize,
    scroll: f32,
    follow_sel: bool,
    squish_at: Option<f64>,
    nav_cd: f64,
    hold: bool,
    built: bool,
    full_urls: Vec<String>,
    thumbs: Vec<Option<egui::TextureHandle>>,
    fetch: std::sync::Arc<std::sync::Mutex<IconFetch>>,
    apply: Option<std::sync::Arc<std::sync::Mutex<IconApply>>>,
}

fn start_icon_fetch(key: String, query: String) -> std::sync::Arc<std::sync::Mutex<IconFetch>> {
    let shared = std::sync::Arc::new(std::sync::Mutex::new(IconFetch::default()));
    let s2 = shared.clone();
    std::thread::spawn(move || {
        let mut result = IconFetch {
            done: true,
            error: None,
            items: Vec::new(),
        };
        if key.trim().is_empty() {
            result.error = Some("No SteamGridDB API key — set it in Preferences > General".into());
        } else {
            let icons = crate::steamgrid::fetch_icons(&key, &query, 15);
            if icons.is_empty() {
                result.error = Some("No icons found for that name".into());
            } else {
                for ic in icons {
                    let bytes = crate::steamgrid::curl_bytes(&ic.thumb_url);
                    result.items.push((ic.full_url, bytes));
                }
            }
        }
        if let Ok(mut g) = s2.lock() {
            *g = result;
        }
    });
    shared
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputDevice {
    Keyboard,
    Gamepad,
    Motion,
    Vibration,
}

impl InputDevice {
    fn cycle(self, dir: i32) -> Self {
        const ORDER: [InputDevice; 4] = [
            InputDevice::Keyboard,
            InputDevice::Gamepad,
            InputDevice::Motion,
            InputDevice::Vibration,
        ];
        let index = ORDER.iter().position(|device| *device == self).unwrap_or(0) as i32;
        ORDER[(index + dir).rem_euclid(ORDER.len() as i32) as usize]
    }
}

const PAD_AUTOMATIC: &str = "Automatic";

fn short_pad_name(name: &str) -> &str {
    name.strip_prefix("Nintendo Switch ")
        .or_else(|| name.strip_prefix("Nintendo "))
        .unwrap_or(name)
}

fn pad_choice_text<T: PartialEq>(
    saved: Option<&str>,
    pads: &[(T, String)],
    active: Option<T>,
) -> String {
    let name = active.map(|id| {
        pads.iter()
            .find(|(pad, _)| *pad == id)
            .map_or("Gamepad", |(_, name)| name.as_str())
    });
    match (saved, name) {
        (None, Some(name)) => format!("{} \u{b7} {}", PAD_AUTOMATIC, short_pad_name(name)),
        (None, None) => format!("{} \u{b7} No Gamepad Connected", PAD_AUTOMATIC),
        (Some(saved), Some(name)) if name == saved => name.to_string(),
        (Some(saved), _) => format!(
            "{} \u{b7} {}",
            short_pad_name(saved),
            if pads.iter().any(|(_, name)| name == saved) {
                "not responding"
            } else {
                "not connected"
            }
        ),
    }
}

fn pad_choice_row<T: PartialEq>(
    saved: Option<&str>,
    pads: &[(T, String)],
    active: Option<T>,
) -> usize {
    let Some(saved) = saved else {
        return 0;
    };
    pads.iter()
        .position(|(pad, name)| Some(pad) == active.as_ref() && name == saved)
        .or_else(|| pads.iter().position(|(_, name)| name == saved))
        .map_or(pads.len() + 1, |index| index + 1)
}

const MOTION_ROW_LABELS: [&str; 6] = [
    "Motion Controls",
    "Motion Source",
    "Recenter Key",
    "Recenter Button",
    "Recenter Now",
    "Calibrate Gyro",
];

const MOTION_HELP: [&str; 6] = [
    "Hold the controller pointing at the screen, then recenter.",
    "Joy-Con pair: one-Joy-Con games read the right Joy-Con.",
    "Motion uses the Controller choice when it has a gyro.",
    "The key and button recenter while a game is running.",
    "Setting the key: Backspace clears it, Esc cancels.",
    "Off: games see the controller lying still.",
];

const MOTION_RECENTERED: &str = "Motion recentered";

const MOTION_OFF: &str = "Motion controls are off";

const MOTION_MISSING: &str = "No motion controller";

const MOTION_CALIBRATING: &str = "Calibrating: keep still";

const MOTION_PUT_DOWN: &str = "Put the controller down to calibrate";

const MOTION_RESTART: &str = "Single Joy-Con layout changes after restarting NeXium.";

const MOTION_RECENTER_ACTION: &str = "Recenter";

const MOTION_CALIBRATE_ACTION: &str = "Calibrate";

const MOTION_UNCALIBRATED: &str = "Not calibrated: put the controller on a flat surface for 2 s";

const MOTION_NO_DATA: &str = "No motion data";

const MOTION_NO_DATA_NOTE: &str = "No motion data: reconnect it or pick another controller";

const MOTION_BUTTON_MISSING: &str = "not on this controller";

fn motion_key_label(name: &str) -> &str {
    if name.is_empty() {
        "\u{2014}"
    } else {
        name
    }
}

fn motion_status_text(enabled: bool, source: Option<&str>) -> String {
    if !enabled {
        return "Off".to_string();
    }
    source.map_or_else(
        || MOTION_MISSING.to_string(),
        |name| name.trim_start_matches("Nintendo Switch ").to_string(),
    )
}

fn motion_calibration_text(
    enabled: bool,
    states: [Option<nexium_core::hid_motion::CalibrationState>; 3],
    stale: [bool; 3],
    hint_note: Option<&'static str>,
) -> (&'static str, Option<&'static str>) {
    if !enabled {
        return ("", None);
    }
    if states.iter().all(Option::is_none) {
        return ("", hint_note);
    }
    if states.iter().zip(stale).any(|(state, stale)| state.is_some() && stale) {
        return (MOTION_NO_DATA, Some(MOTION_NO_DATA_NOTE));
    }
    if states.contains(&Some(nexium_core::hid_motion::CalibrationState::Uncalibrated)) {
        ("Not calibrated", Some(MOTION_UNCALIBRATED))
    } else if states.contains(&Some(nexium_core::hid_motion::CalibrationState::Capturing)) {
        ("Calibrating\u{2026}", Some(MOTION_CALIBRATING))
    } else {
        ("Calibrated", None)
    }
}

fn motion_row_values(
    settings: &AppSettings,
    source: Option<&str>,
    capturing: bool,
    recentered: bool,
    calibration: &str,
) -> [String; 6] {
    [
        String::from(if settings.motion_enabled { "On" } else { "Off" }),
        motion_status_text(settings.motion_enabled, source),
        if capturing {
            "Press a key\u{2026}".to_string()
        } else {
            motion_key_label(&settings.motion_recenter_key).to_string()
        },
        settings.motion_recenter_button.label().to_string(),
        String::from(if recentered { "Done" } else { "" }),
        calibration.to_string(),
    ]
}

enum MotionKeyCapture {
    Cancel,
    Set(String),
    Refuse(String),
}

fn recenter_key_conflict(key: egui::Key, config: &ControllerConfig) -> Option<String> {
    let name = format!("{:?}", key);
    match key {
        egui::Key::Home | egui::Key::Backtick => Some(format!("{} opens the Home menu.", name)),
        egui::Key::Enter
        | egui::Key::Space
        | egui::Key::Tab
        | egui::Key::ArrowUp
        | egui::Key::ArrowDown
        | egui::Key::ArrowLeft
        | egui::Key::ArrowRight => Some(format!("{} is used by the menus.", name)),
        _ => SwitchButton::all()
            .iter()
            .find(|button| {
                config
                    .binding_for(**button)
                    .is_some_and(|binding| binding.eq_ignore_ascii_case(&name))
            })
            .map(|button| format!("{} is already bound to {}.", name, button.display_name())),
    }
}

fn recenter_key_press(
    events: &[egui::Event],
    config: &ControllerConfig,
) -> Option<MotionKeyCapture> {
    let mut refused = None;
    for event in events {
        let egui::Event::Key {
            key,
            pressed: true,
            repeat: false,
            ..
        } = event
        else {
            continue;
        };
        match key {
            egui::Key::Escape => return Some(MotionKeyCapture::Cancel),
            egui::Key::Backspace | egui::Key::Delete => {
                return Some(MotionKeyCapture::Set(String::new()));
            }
            _ => match recenter_key_conflict(*key, config) {
                Some(notice) => refused = Some(notice),
                None => return Some(MotionKeyCapture::Set(format!("{:?}", key))),
            },
        }
    }
    refused.map(MotionKeyCapture::Refuse)
}

fn recenter_key_label(settings: &AppSettings, config: &ControllerConfig) -> String {
    let label = motion_key_label(&settings.motion_recenter_key);
    if motion_recenter_key(&settings.motion_recenter_key)
        .and_then(|key| recenter_key_conflict(key, config))
        .is_some()
    {
        format!("{} (off)", label)
    } else {
        label.to_string()
    }
}

fn motion_recenter_text(
    settings: &AppSettings,
    config: &ControllerConfig,
    button_missing: bool,
) -> String {
    let button = settings.motion_recenter_button;
    let mut ways = Vec::new();
    if !settings.motion_recenter_key.is_empty() {
        ways.push(recenter_key_label(settings, config));
    }
    if button != MotionRecenterButton::Unbound {
        ways.push(if button_missing {
            format!("{} ({})", button.label(), MOTION_BUTTON_MISSING)
        } else {
            button.label().to_string()
        });
    }
    if ways.is_empty() {
        motion_key_label("").to_string()
    } else {
        ways.join(" or ")
    }
}

fn motion_note(
    settings: &AppSettings,
    config: &ControllerConfig,
    key_notice: Option<&str>,
    button_missing: bool,
) -> Option<String> {
    let button = settings.motion_recenter_button;
    key_notice
        .map(str::to_string)
        .or_else(|| {
            motion_recenter_key(&settings.motion_recenter_key)
                .and_then(|key| recenter_key_conflict(key, config))
        })
        .or_else(|| {
            (button_missing && button != MotionRecenterButton::Unbound)
                .then(|| format!("{} is {}.", button.label(), MOTION_BUTTON_MISSING))
        })
}

fn motion_summary(settings: &AppSettings, config: &ControllerConfig) -> String {
    format!(
        "Motion controls: {} \u{b7} Recenter key: {} \u{b7} change under Input Device \u{203a} Motion",
        if settings.motion_enabled { "On" } else { "Off" },
        recenter_key_label(settings, config),
    )
}

const VIBRATION_ROW_LABELS: [&str; 4] = [
    "Vibration",
    "Vibration Output",
    "Vibration Strength",
    "Test Vibration",
];

const VIBRATION_HELP: [&str; 3] = [
    "HD Rumble plays game vibration on Joy-Con and Pro Controllers.",
    "Other controllers use their standard rumble motors.",
    "Off: games and menus never vibrate the controller.",
];

const VIBRATION_TEST_ACTION: &str = "Test";

const VIBRATION_NO_CONTROLLER: &str = "No controller";

fn vibration_row_values(settings: &AppSettings, status: &str) -> [String; 4] {
    [
        String::from(if settings.vibration_enabled { "On" } else { "Off" }),
        status.to_string(),
        format!("{}%", settings.vibration_strength),
        String::new(),
    ]
}

fn step_vibration_strength(current: u8, dir: i32) -> u8 {
    let current = (current.clamp(10, 100) + 5) / 10 * 10;
    (i32::from(current) + dir.signum() * 10).clamp(10, 100) as u8
}

fn bar_toggle_label(name: &str, on: bool, compact: bool) -> String {
    if compact {
        name.to_string()
    } else {
        format!("{}: {}", name, if on { "On" } else { "Off" })
    }
}

fn shader_build_label(building: i64, compact: bool) -> String {
    if compact {
        format!("Shaders: {building}")
    } else {
        format!("Building shaders: {building}")
    }
}

fn motion_toast_rect(screen: egui::Rect, below_download: bool) -> egui::Rect {
    let top = screen.min.y + if below_download { 98.0 } else { 24.0 };
    egui::Rect::from_min_size(
        egui::pos2(screen.center().x - 130.0, top),
        egui::vec2(260.0, 44.0),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VkTarget {
    None,
    ApiKey,
    ListName,
    Swkbd,
}

impl HorizonApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
        app_settings: AppSettings,
        nro_arg: Option<String>,
    ) -> Self {
        Self::apply_theme(&cc.egui_ctx);
        crate::ui_audio::init(app_settings.audio_output_device.as_deref());
        let gamepad = app_settings.gamepad.clone();
        let input = InputBackend::new(app_settings.motion_enabled, gamepad).or_else(|| {
            log::warn!("SDL3 gamepad init failed");
            None
        });
        let nro_path = nro_arg.unwrap_or_default();
        let nextendo = crate::nextendo::Nextendo::start(
            cc.egui_ctx.clone(),
            nextendo_glue::preferences(&app_settings.nextendo),
        );
        let mut app = Self {
            nro_path: nro_path.clone(),
            emulation_handle: None,
            game_texture: None,
            egui_ctx: cc.egui_ctx.clone(),
            wgpu_state: cc.wgpu_render_state.clone(),
            game_texture_native: None,
            native_game: crate::native_game::NativeGameWindow::new(cc),
            native_overlay_rect: None,
            game_bar_tab_rect: None,
            game_depth: None,
            game_depth_seq: 0,
            last_game_rect: None,
            mouse_wheel_accum: egui::Vec2::ZERO,
            show_settings: false,
            settings_tab: SettingsTab::General,
            prefs_anim: 0.0,
            prefs_focus: false,
            prefs_row: 0,
            prefs_col: 0,
            prefs_col1_row: 0,
            prefs_dropdown_open: false,
            prefs_dropdown_sel: 0,
            prefs_lr_held: false,
            prefs_nav_cd: 0.0,
            prefs_nav_held_dir: 0,
            prefs_nav_held_since: 0.0,
            prefs_ab_held: false,
            prefs_key_editing: false,
            prefs_key_menu: false,
            prefs_ctrl_scroll: 0,
            prefs_key_caret: 0,
            prefs_key_anchor: 0,
            vkeyboard: crate::vkeyboard::VirtualKeyboard::new(),
            vk_target: VkTarget::None,
            swkbd_text: String::new(),
            swkbd_gen: 0,
            swkbd_max: 500,
            swkbd_min: 0,
            swkbd_title: String::new(),
            swkbd_password: false,
            updater: crate::updater::Updater::new(),
            update_restart_shown: false,
            input,
            last_input: InputSnapshot::default(),
            debugger: DebuggerState::new(),
            performance: PerformanceMonitor::new(),
            log_buffer,
            controller_config: ControllerConfig::load(),
            rebinding: None,
            rebinding_pad: None,
            rebinding_pad_wait_release: false,
            prefs_rebind_cool: false,
            prefs_enter_held: false,
            input_device: InputDevice::Keyboard,
            motion_key_capture: false,
            motion_key_notice: None,
            startup_gpu_device: app_settings.gpu_device.clone(),
            startup_motion_enabled: app_settings.motion_enabled,
            app_settings,
            last_buttons_logged: 0,
            last_sticks_logged: [0; 4],
            audio_device_cache: None,
            splash: crate::splash::Splash::new(),
            library: crate::library::Library::new(),
            stop_fade: None,
            pause_anim: None,
            resume_anim: None,
            stop_anim: None,
            pill_fade: None,
            playing_path: None,
            last_home: false,
            motion_recenter_held: false,
            profile_texture: None,
            profile_reload: false,
            show_profile: false,
            profile_anim: 0.0,
            profile: crate::profile::ProfileState::new(),
            play_times: crate::playtime::PlayTimes::load(),
            last_playtime_save: std::time::Instant::now(),
            carousel: crate::carousel::CarouselState::new(),
            icon_picker: None,
            shop: crate::shop::ShopState::new(),
            active_downloads: Vec::new(),
            download_toast: None,
            motion_toast: None,
            game_bar_spare: f32::MAX,
            game_bar_compact: false,
            game_bar_terse: false,
            game_bar_terse_spare: f32::MAX,
            top_bar_spare: f32::MAX,
            top_bar_compact: false,
            top_bar_terse: false,
            top_bar_terse_spare: f32::MAX,
            carousel_settings_open: false,
            cs_anim: 0.0,
            cs_tab: 0,
            cs_scroll: 0.0,
            cs_selected: 0,
            cs_focus_grid: false,
            cs_nav_cd: 0.0,
            cs_nav_held_dir: 0,
            cs_nav_held_since: 0.0,
            cs_ab_held: false,
            cs_list_picks: Vec::new(),
            cs_pick_anim: std::collections::HashMap::new(),
            cs_creating: false,
            cs_new_name: String::new(),
            cs_new_align: crate::app_settings::ListAlignment::Manual,
            cs_dialog_row: 0,
            cs_name_editing: false,
            cs_name_caret: 0,
            cs_name_anchor: 0,
            cs_create_btn: 0.0,
            cs_plus_held: false,
            cs_rename_idx: None,
            cs_list_menu: None,
            cs_list_menu_sel: 0,
            cs_x_held: false,
            cs_build_grab: None,
            cs_build_insert: 0,
            cs_row_scroll: 0.0,
            cs_grab_lift: 0.0,
            perf_pos: None,
            perf_drag: false,
            perf_grab_off: egui::Vec2::ZERO,
            perf_hist: Vec::new(),
            perf_scale: 1.0,
            perf_sampled: None,
            perf_readout_at: None,
            perf_readout: (0.0, 0.0, 0, 0),
            last_frame_res: (0, 0),
            music_was_on: false,
            music_fade_start: None,
            icon_reveal: None,
            key_test: None,
            key_test_result: None,
            confirm: None,
            teardown_at: None,
            pending_boot: None,
            modal_cd: 0.0,
            modal_hold: false,
            modal_anim: 0.0,
            modal_snap: None,
            pending_quick: None,
            modal_active_frame_start: false,
            game_info_path: None,
            mod_manager: None,
            content_manager: None,
            firmware_prompt: None,
            firmware_status: None,
            nextendo,
            nextendo_state: crate::nextendo::State::default(),
            nextendo_hub: crate::nextendo::hub::HubState::new(),
            nextendo_catalog: crate::nextendo::Catalog::new(cc.egui_ctx.clone()),
            nextendo_toasts: crate::nextendo::toasts::Toasts::new(),
            nextendo_reported_title: None,
            nextendo_session_path: None,
            nextendo_decorated: None,
            show_nextendo: false,
            nextendo_anim: 0.0,
        };
        app.reload_profile_texture(&cc.egui_ctx);
        crate::ui_audio::set_sfx_volume(app.app_settings.sfx_volume);
        crate::ui_audio::set_carousel_track(app.app_settings.menu_music_track);
        app.library
            .rescan(&cc.egui_ctx, &app.app_settings.library_folders);
        if !nro_path.is_empty() {
            let backend = app.app_settings.effective_cpu_backend().to_cpu_kind();
            let repaint_ctx = cc.egui_ctx.clone();
            let target = app.native_game.as_mut().map(|window| window.begin_game(&cc.egui_ctx));
            if let Ok(handle) = EmulationHandle::new_with_presentation(
                &nro_path,
                backend,
                Some(std::sync::Arc::new(move || {
                    request_game_frame_repaint(&repaint_ctx)
                })),
                target,
            ) {
                app.emulation_handle = Some(handle);
                let launched = std::path::PathBuf::from(&nro_path);
                let game_index = app.library.index_of_path(&launched).unwrap_or(0);
                app.carousel.boot_stage = crate::carousel::BootStage::AwaitingFrame {
                    game_index,
                    start_time: 0.0,
                };
                cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Title(
                    app.game_window_title(&launched),
                ));
                app.playing_path = Some(launched);
                log::info!("Auto-loaded NRO: {} (CPU: {})", nro_path, backend.label());
            } else {
                log::error!("Failed to load NRO: {}", nro_path);
            }
        }
        app
    }

    fn apply_theme(ctx: &egui::Context) {
        let mut s = (*ctx.global_style()).clone();
        s.visuals.dark_mode = true;
        s.visuals.panel_fill = BG;
        s.visuals.window_fill = BG_RAISED;
        s.visuals.faint_bg_color = BG_RAISED;
        s.visuals.extreme_bg_color = BG;
        s.visuals.override_text_color = Some(TEXT);
        s.visuals.window_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.window_corner_radius = CornerRadius::same(8);
        s.visuals.menu_corner_radius = CornerRadius::same(6);
        s.visuals.popup_shadow = egui::epaint::Shadow {
            offset: [0, 6],
            blur: 16,
            spread: 0,
            color: Color32::from_black_alpha(100),
        };
        for w in [
            &mut s.visuals.widgets.noninteractive,
            &mut s.visuals.widgets.inactive,
            &mut s.visuals.widgets.hovered,
            &mut s.visuals.widgets.active,
            &mut s.visuals.widgets.open,
        ] {
            w.corner_radius = CornerRadius::same(4);
        }
        s.visuals.widgets.noninteractive.bg_fill = BG_RAISED;
        s.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, MUTED);
        s.visuals.widgets.inactive.bg_fill = BG_INPUT;
        s.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
        s.visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x28, 0x28, 0x30);
        s.visuals.widgets.hovered.bg_stroke =
            Stroke::new(1.0_f32, Color32::from_rgb(0x44, 0x44, 0x52));
        s.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
        s.visuals.widgets.active.bg_fill = ACCENT;
        s.visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
        s.visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
        s.visuals.widgets.open.bg_fill = BG_INPUT;
        s.visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, ACCENT);
        s.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(0x2F, 0xB4, 0xEF, 0x50);
        s.spacing.item_spacing = Vec2::new(6.0, 4.0);
        s.spacing.button_padding = Vec2::new(10.0, 5.0);
        s.spacing.menu_margin = egui::Margin::same(6);
        s.spacing.window_margin = egui::Margin::same(12);
        ctx.set_global_style(s);
    }

    fn toggle_favorite_path(&mut self, path: std::path::PathBuf) {
        crate::ui_audio::play(crate::ui_audio::Sfx::Favorite);
        if let Some(pos) = self
            .app_settings
            .favorites
            .iter()
            .position(|item| *item == path)
        {
            self.app_settings.favorites.remove(pos);
        } else {
            self.app_settings.favorites.push(path);
        }
        let _ = self.app_settings.save();
    }

    fn open_content_manager(&mut self, path: std::path::PathBuf, ctx: &egui::Context) {
        let (title, cover, accent) = if let Some(index) = self.library.index_of_path(&path) {
            let title = self.library.games[index].title.clone();
            let accent = self.library.games[index].dominant_color;
            (title, self.library.texture(ctx, index), accent)
        } else {
            (path.file_stem().unwrap_or_default().to_string_lossy().into_owned(), None, ACCENT)
        };
        self.game_info_path = None;
        self.mod_manager = None;
        self.content_manager = Some(crate::content_manager::ContentManager::new(path, title, cover, accent, ctx));
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
    }

    fn open_game_file(&mut self, path: std::path::PathBuf, then: crate::firmware_prompt::Then, ctx: &egui::Context) {
        if crate::firmware_prompt::may_contain_firmware(&path) {
            self.game_info_path = None;
            let accent = self.theme_accent();
            self.firmware_prompt = Some(crate::firmware_prompt::FirmwarePrompt::new(path, then, accent, ctx));
            return;
        }
        match then {
            crate::firmware_prompt::Then::Boot => {
                self.nro_path = path.to_string_lossy().to_string();
                self.boot_nro(ctx);
            }
            crate::firmware_prompt::Then::Select => self.nro_path = path.to_string_lossy().to_string(),
            crate::firmware_prompt::Then::Nothing => {}
        }
    }

    fn firmware_status(&mut self) -> Result<Option<nexium_loader::firmware::InstalledFirmware>, String> {
        self.firmware_status
            .get_or_insert_with(|| nexium_loader::firmware::installed_firmware(&nexium_common::paths::firmware_dir()))
            .clone()
    }

    fn install_firmware_from_settings(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Install firmware")
            .add_filter("Firmware (.dnsp or .dxci)", &["dnsp", "dxci"])
            .pick_file()
        else {
            return;
        };
        self.show_settings = false;
        self.prefs_anim = 0.0;
        self.open_game_file(path, crate::firmware_prompt::Then::Nothing, ctx);
    }

    fn remove_firmware_from_settings(&mut self) {
        let Ok(Some(installed)) = self.firmware_status() else { return };
        self.show_settings = false;
        self.prefs_anim = 0.0;
        let accent = self.theme_accent();
        self.firmware_prompt = Some(crate::firmware_prompt::FirmwarePrompt::remove(installed, accent));
    }

    fn open_mod_manager(&mut self, path: std::path::PathBuf, ctx: &egui::Context) {
        self.game_info_path = None;
        self.mod_manager = Some(crate::mods::ModManager::new(path, ctx));
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
    }

    fn open_icon_picker(&mut self, path: std::path::PathBuf) {
        let Some(idx) = self.library.index_of_path(&path) else {
            return;
        };
        let title = self.library.games[idx].title.clone();
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        self.icon_picker = Some(IconPicker {
            game_idx: idx,
            game_path: path,
            search: title.clone(),
            editing: false,
            anim: 0.0,
            selected: 0,
            scroll: 0.0,
            follow_sel: true,
            squish_at: None,
            nav_cd: 0.0,
            hold: true,
            built: false,
            full_urls: Vec::new(),
            thumbs: Vec::new(),
            fetch: start_icon_fetch(self.app_settings.steamgriddb_key.clone(), title),
            apply: None,
        });
    }

    fn draw_game_info(&mut self, ctx: &egui::Context) -> Option<GameInfoAction> {
        let Some(path) = self.game_info_path.clone() else {
            return None;
        };
        let Some(game_idx) = self.library.index_of_path(&path) else {
            self.game_info_path = None;
            return None;
        };

        let game = &self.library.games[game_idx];
        let title = game.title.clone();
        let author = game.author.clone();
        let format = game.format.to_string();
        let size = game.size;
        let dominant = game.dominant_color;
        let favorite = self.app_settings.favorites.iter().any(|item| *item == path);
        let icon = self
            .library
            .texture(ctx, game_idx)
            .map(|texture| texture.id());
        let metadata = std::fs::metadata(&path).ok();
        let exists = metadata.is_some();
        let file_size = metadata.as_ref().map(|item| item.len()).unwrap_or(size);
        let modified = metadata.and_then(|item| item.modified().ok());
        let path_text = path.to_string_lossy().to_string();
        let modified_text = modified
            .map(format_system_time)
            .unwrap_or_else(|| "Unavailable".to_string());
        let status_text = if exists { "Available" } else { "Missing" };
        let status_color = if exists { GREEN } else { DANGER_HV };
        let mut open = true;
        let mut close_requested = false;
        let mut action = None;

        egui::Window::new("Game Information")
            .id(egui::Id::new("game_information_window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([650.0, 490.0])
            .min_width(560.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let cover_size = Vec2::splat(164.0);
                    let (cover_rect, _) = ui.allocate_exact_size(cover_size, Sense::hover());
                    if let Some(texture) = icon {
                        egui::Image::new((texture, cover_size))
                            .corner_radius(CornerRadius::same(8))
                            .paint_at(ui, cover_rect);
                    } else {
                        ui.painter().rect_filled(
                            cover_rect,
                            CornerRadius::same(8),
                            dominant.gamma_multiply(0.45),
                        );
                        ui.painter().rect_stroke(
                            cover_rect,
                            CornerRadius::same(8),
                            Stroke::new(1.0_f32, BORDER),
                            egui::StrokeKind::Outside,
                        );
                        ui.painter().text(
                            cover_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            &format,
                            FontId::proportional(24.0),
                            TEXT,
                        );
                    }
                    ui.add_space(16.0);
                    ui.vertical(|ui| {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&title).size(24.0).strong().color(TEXT));
                        if author.trim().is_empty() {
                            ui.label(egui::RichText::new("Unknown developer").color(MUTED));
                        } else {
                            ui.label(egui::RichText::new(&author).size(14.0).color(MUTED));
                        }
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&format).strong().color(ACCENT_HV));
                            ui.label(egui::RichText::new("  |  ").color(BORDER));
                            ui.label(egui::RichText::new(format_file_size(file_size)).color(MUTED));
                        });
                        ui.add_space(10.0);
                        ui.label(
                            egui::RichText::new(if favorite {
                                "Favorite"
                            } else {
                                "Not a favorite"
                            })
                            .color(if favorite {
                                AMBER
                            } else {
                                MUTED
                            }),
                        );
                    });
                });

                ui.add_space(14.0);
                ui.separator();
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("File Details")
                        .size(15.0)
                        .strong()
                        .color(TEXT),
                );
                ui.add_space(4.0);
                egui::Grid::new("game_information_details")
                    .num_columns(2)
                    .spacing(Vec2::new(18.0, 8.0))
                    .show(ui, |ui| {
                        let detail = |ui: &mut egui::Ui, label: &str, value: &str| {
                            ui.label(egui::RichText::new(label).color(MUTED));
                            ui.label(egui::RichText::new(value).color(TEXT));
                            ui.end_row();
                        };
                        detail(ui, "Format", &format);
                        detail(ui, "Size", &format_file_size(file_size));
                        detail(ui, "Status", status_text);
                        detail(ui, "Last modified", &modified_text);
                        ui.label(egui::RichText::new("Location").color(MUTED));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&path_text).monospace().color(TEXT));
                            if ui.small_button("Copy").clicked() {
                                let _ = arboard::Clipboard::new().and_then(|mut clipboard| {
                                    clipboard.set_text(path_text.clone())
                                });
                            }
                        });
                        ui.end_row();
                    });

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(exists, egui::Button::new("Launch"))
                        .clicked()
                    {
                        action = Some(GameInfoAction::Launch(path.clone()));
                    }
                    if ui
                        .button(if favorite { "Unfavorite" } else { "Favorite" })
                        .clicked()
                    {
                        action = Some(GameInfoAction::ToggleFavorite(path.clone()));
                    }
                    if ui.button("Change Icon").clicked() {
                        action = Some(GameInfoAction::DownloadIcon(path.clone()));
                    }
                    if ui.add_enabled(exists, egui::Button::new("Manage Mods")).clicked() {
                        action = Some(GameInfoAction::ManageMods(path.clone()));
                    }
                    if ui.add_enabled(exists, egui::Button::new("Updates & DLC")).clicked() {
                        action = Some(GameInfoAction::ManageContent(path.clone()));
                    }
                    if ui.button("Open Containing Folder").clicked() {
                        reveal_in_file_manager(&path);
                    }
                    if ui.button("Close").clicked() {
                        close_requested = true;
                    }
                });
                ui.colored_label(status_color, status_text);
            });

        if close_requested || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            open = false;
        }
        if !open {
            self.game_info_path = None;
        } else if matches!(
            &action,
            Some(GameInfoAction::Launch(_) | GameInfoAction::DownloadIcon(_) | GameInfoAction::ManageMods(_) | GameInfoAction::ManageContent(_))
        ) {
            self.game_info_path = None;
        }
        action
    }

    fn library_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        const MARGIN: f32 = 26.0;
        ui.add_space(16.0);
        ui.horizontal(|ui| {
            ui.add_space(MARGIN);
            ui.label(
                egui::RichText::new("Library")
                    .size(22.0)
                    .strong()
                    .color(TEXT),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(MARGIN);
                if pill_button(ui, "Open file…", true).clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("Switch games", &["nro", "dxci", "dnsp"])
                        .pick_file()
                    {
                        self.open_game_file(p, crate::firmware_prompt::Then::Boot, ctx);
                    }
                }
                ui.add_space(8.0);
                if pill_button(ui, "Refresh", false).clicked() {
                    self.library.rescan(ctx, &self.app_settings.library_folders);
                }
            });
        });
        ui.add_space(12.0);

        if !self.library.loaded {
            ui.centered_and_justified(|ui| ui.add(egui::Spinner::new().size(28.0)));
            return;
        }
        {
            let w = &mut ui.style_mut().visuals.widgets;
            let dim = Color32::from_rgb(0x1E, 0x5D, 0x7A);
            w.inactive.bg_fill = dim;
            w.inactive.weak_bg_fill = dim;
            w.hovered.bg_fill = ACCENT;
            w.hovered.weak_bg_fill = ACCENT;
            w.active.bg_fill = ACCENT_HV;
            w.active.weak_bg_fill = ACCENT_HV;
        }

        let mut launch: Option<String> = None;
        let mut info_request: Option<std::path::PathBuf> = None;
        let mut mods_request: Option<std::path::PathBuf> = None;
        let mut content_request: Option<std::path::PathBuf> = None;
        let mut favorite_request: Option<std::path::PathBuf> = None;
        let mut download_request: Option<std::path::PathBuf> = None;
        let mut reveal_request: Option<std::path::PathBuf> = None;
        let mut add_folder = false;
        let info_open = self.game_info_path.is_some();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(8.0);
                let tw = 154.0;
                let gap = 20.0;
                let n_games = self.library.games.len();
                let total_tiles = n_games + 1;
                let avail = ui.clip_rect().width();
                let cols = (((avail - 2.0 * MARGIN + gap) / (tw + gap)).floor() as usize)
                    .clamp(1, total_tiles);
                let total = cols as f32 * tw + (cols.saturating_sub(1)) as f32 * gap;
                let left = ((avail - total) * 0.5).max(MARGIN);
                let mut idx = 0;
                while idx < total_tiles {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.add_space(left);
                        for c in 0..cols {
                            if idx >= total_tiles {
                                break;
                            }
                            if idx < n_games {
                                let tex = self.library.texture(ctx, idx);
                                let selected = self.library.selected == Some(idx);
                                let game_path = self.library.games[idx].path.clone();
                                let favorite = self
                                    .app_settings
                                    .favorites
                                    .iter()
                                    .any(|path| *path == game_path);
                                let resp =
                                    game_tile(ui, &self.library.games[idx], tex.as_ref(), selected);
                                if resp.clicked() {
                                    self.library.selected = Some(idx);
                                }
                                if resp.double_clicked() {
                                    launch = Some(game_path.to_string_lossy().to_string());
                                }
                                if !info_open {
                                    resp.context_menu(|menu| {
                                        if menu.button("Launch").clicked() {
                                            launch = Some(game_path.to_string_lossy().to_string());
                                            menu.close();
                                        }
                                        if menu.button("View Game Information").clicked() {
                                            info_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        if menu.button("Manage Mods").clicked() {
                                            mods_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        if menu.button("Updates & DLC").clicked() {
                                            content_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        menu.separator();
                                        let favorite_label = if favorite {
                                            "Unfavorite Game"
                                        } else {
                                            "Favorite Game"
                                        };
                                        if menu.button(favorite_label).clicked() {
                                            favorite_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        if menu.button("Download Icon").clicked() {
                                            download_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        menu.separator();
                                        if menu.button("Open Containing Folder").clicked() {
                                            reveal_request = Some(game_path.clone());
                                            menu.close();
                                        }
                                        if menu.button("Copy Path").clicked() {
                                            let _ = arboard::Clipboard::new().and_then(
                                                |mut clipboard| {
                                                    clipboard.set_text(
                                                        game_path.to_string_lossy().to_string(),
                                                    )
                                                },
                                            );
                                            menu.close();
                                        }
                                    });
                                }
                            } else if add_folder_tile(ui).clicked() {
                                add_folder = true;
                            }
                            if c + 1 < cols {
                                ui.add_space(gap);
                            }
                            idx += 1;
                        }
                    });
                    ui.add_space(20.0);
                }
                ui.add_space(4.0);
            });
        if let Some(path) = launch {
            self.nro_path = path;
            self.boot_nro(ctx);
        } else if let Some(path) = content_request {
            self.open_content_manager(path, ctx);
        } else if let Some(path) = mods_request {
            self.open_mod_manager(path, ctx);
        } else if let Some(path) = info_request {
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
            self.game_info_path = Some(path);
        } else if let Some(path) = favorite_request {
            self.toggle_favorite_path(path);
        } else if let Some(path) = download_request {
            self.open_icon_picker(path);
        } else if let Some(path) = reveal_request {
            reveal_in_file_manager(&path);
        }
        if add_folder {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                if !self.app_settings.library_folders.contains(&dir) {
                    self.app_settings.library_folders.push(dir);
                    let _ = self.app_settings.save();
                }
                self.library.rescan(ctx, &self.app_settings.library_folders);
            }
        }
    }

    fn poll_frames(&mut self, ctx: &egui::Context) {
        let mut observed_native_frame = false;
        if let Some(window) = &mut self.native_game {
            let new_frames = window.poll();
            if new_frames != 0 {
                self.game_depth = None;
                gui_rate_stats(1, new_frames);
                self.performance.record_frames(new_frames);
                observed_native_frame = true;
                self.last_frame_res = (window.dimensions[0], window.dimensions[1]);
                self.carousel.boot_stage = crate::carousel::BootStage::None;
                if self.game_texture.is_none() {
                    self.game_texture = Some(ctx.load_texture("native_game_preview",
                        egui::ColorImage::new([1, 1], vec![Color32::BLACK]), egui::TextureOptions::LINEAR));
                }
            }
            if let Some(snapshot) = window.target.as_ref().and_then(|target| target.take_snapshot()) {
                let image = egui::ColorImage::from_rgba_unmultiplied(
                    [snapshot.width as usize, snapshot.height as usize], &snapshot.pixels);
                if let Some(texture) = &mut self.game_texture {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.game_texture = Some(ctx.load_texture("native_game_preview", image, egui::TextureOptions::LINEAR));
                }
            }
        }
        let Some(handle) = &self.emulation_handle else {
            if !observed_native_frame {
                self.performance.tick();
            }
            return;
        };
        let tex_opts = match self.app_settings.filter {
            FilterMode::Nearest => egui::TextureOptions::NEAREST,
            _ => egui::TextureOptions::LINEAR,
        };
        if let Some(mut frame) = take_next_game_frame(&handle.frame_rx) {
            if let Some(window) = &mut self.native_game { window.active = false; }
            gui_rate_stats(1, 1);
            self.performance.record_frame();
            self.last_frame_res = (frame.width, frame.height);
            log::trace!(
                "frame in: {}x{} ({} bytes)",
                frame.width,
                frame.height,
                frame.pixels.len()
            );
            if self.wgpu_state.is_some()
                && (!legacy_gui_upload()
                    || !matches!(self.app_settings.filter, FilterMode::Nearest | FilterMode::Linear))
            {
                self.upload_frame_native(&mut frame);
                return;
            }
            self.free_native_texture();
            let image_size = [frame.width as usize, frame.height as usize];
            let img = egui::ColorImage::from_rgba_unmultiplied(image_size, &frame.pixels);
            match &mut self.game_texture {
                Some(t) if t.size() == image_size => {
                    t.set_partial([0, 0], img, tex_opts);
                }
                Some(t) => t.set(img, tex_opts),
                None => {
                    self.game_texture = Some(ctx.load_texture("game_frame", img, tex_opts));
                    self.carousel.boot_stage = crate::carousel::BootStage::None;
                }
            }
        } else if !observed_native_frame {
            self.performance.tick();
        }
    }

    fn upload_frame_native(&mut self, frame: &mut crate::boot::Frame) {
        let Some(rs) = self.wgpu_state.clone() else {
            return;
        };
        let need = frame.width as usize * frame.height as usize * 4;
        if frame.pixels.len() < need {
            return;
        }
        let filter = self.app_settings.filter;
        let recreate = self.game_texture_native.as_ref().map_or(true, |t| {
            t.width != frame.width || t.height != frame.height
        });
        if recreate {
            self.free_native_texture();
            let texture = rs.device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("nexium_game_frame"),
                size: eframe::wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING
                    | eframe::wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let wgpu_filter = match filter {
                FilterMode::Nearest => eframe::wgpu::FilterMode::Nearest,
                _ => eframe::wgpu::FilterMode::Linear,
            };
            let id = rs
                .renderer
                .write()
                .register_native_texture(&rs.device, &view, wgpu_filter);
            self.game_texture_native = Some(NativeGameTexture {
                texture,
                id,
                width: frame.width,
                height: frame.height,
                filter,
                scaler: None,
                dirty: true,
            });
        }
        if self.carousel.boot_stage != crate::carousel::BootStage::None {
            self.carousel.boot_stage = crate::carousel::BootStage::None;
        }
        let Some(t) = self.game_texture_native.as_mut() else {
            return;
        };
        t.dirty = true;
        let _ = rs.device.poll(eframe::wgpu::PollType::Poll);
        rs.queue.write_texture(
            eframe::wgpu::TexelCopyTextureInfo {
                texture: &t.texture,
                mip_level: 0,
                origin: eframe::wgpu::Origin3d::ZERO,
                aspect: eframe::wgpu::TextureAspect::All,
            },
            &frame.pixels[..need],
            eframe::wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * 4),
                rows_per_image: Some(frame.height),
            },
            eframe::wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        if let Some(depth) = frame.depth.take() {
            self.game_depth = Some(std::sync::Arc::new(depth));
            self.game_depth_seq = self.game_depth_seq.wrapping_add(1);
        }
    }

    fn filtered_game_texture(&mut self, rect: egui::Rect, pixels_per_point: f32) -> Option<egui::TextureId> {
        if self.native_game.as_ref().is_some_and(|window| window.active) {
            return None;
        }
        let rs = self.wgpu_state.as_ref()?;
        let texture = self.game_texture_native.as_mut()?;
        let filter = self.app_settings.filter;
        if texture.filter != filter {
            let sampler = if filter == FilterMode::Nearest {
                eframe::wgpu::FilterMode::Nearest
            } else {
                eframe::wgpu::FilterMode::Linear
            };
            rs.renderer.write().update_egui_texture_from_wgpu_texture(
                &rs.device,
                &texture.texture.create_view(&Default::default()),
                sampler,
                texture.id,
            );
            texture.filter = filter;
            texture.dirty = true;
        }
        if matches!(filter, FilterMode::Nearest | FilterMode::Linear) {
            if let Some(id) = texture.scaler.take().and_then(|scaler| scaler.id) {
                rs.renderer.write().free_texture(&id);
            }
            return Some(texture.id);
        }
        let size = [
            (rect.width() * pixels_per_point).round().max(1.0) as u32,
            (rect.height() * pixels_per_point).round().max(1.0) as u32,
        ];
        let scaler = texture.scaler.get_or_insert_with(|| crate::frame_scaler::FrameScaler::new(&rs.device));
        let id = scaler.render(
            rs,
            &texture.texture,
            size,
            filter.scaling_filter(),
            self.app_settings.fsr_sharpness,
            texture.dirty,
        );
        texture.dirty = false;
        Some(id)
    }

    fn free_native_texture(&mut self) {
        self.game_depth = None;
        if let Some(t) = self.game_texture_native.take() {
            if let Some(rs) = &self.wgpu_state {
                let mut renderer = rs.renderer.write();
                renderer.free_texture(&t.id);
                if let Some(id) = t.scaler.and_then(|scaler| scaler.id) {
                    renderer.free_texture(&id);
                }
            }
        }
    }

    fn modal_active(&self) -> bool {
        self.confirm.is_some()
            || self.teardown_at.is_some()
            || self.icon_picker.is_some()
            || self.shop.open
            || self.carousel_settings_open
            || self.game_info_path.is_some()
            || self.mod_manager.is_some()
            || self.content_manager.is_some()
            || self.firmware_prompt.is_some()
            || self.vkeyboard.open
    }

    fn rebind_in_progress(&self) -> bool {
        self.rebinding.is_some() || self.rebinding_pad.is_some() || self.motion_key_capture
    }

    fn game_input_active(&self) -> bool {
        self.emulation_handle
            .as_ref()
            .is_some_and(|handle| handle.is_running() && !handle.is_paused())
            && self.game_display().is_some()
            && !self.modal_active()
            && !self.show_settings
            && !self.show_profile
            && self.profile_anim == 0.0
            && !self.nextendo_overlay_active()
            && self.pause_anim.is_none()
            && self.resume_anim.is_none()
            && !(self.vk_target == VkTarget::Swkbd && self.vkeyboard.active())
    }

    fn motion_present(&self) -> bool {
        self.input.as_ref().is_some_and(|ib| ib.motion_attached())
            || nexium_core::hid_motion::any_source_connected()
    }

    fn motion_button_missing(&self) -> bool {
        let button = self.app_settings.motion_recenter_button;
        self.app_settings.motion_enabled
            && button != MotionRecenterButton::Unbound
            && self
                .input
                .as_ref()
                .is_some_and(|ib| ib.motion_attached() && !ib.recenter_button_available(button))
    }

    fn update_icon_picker(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        const ICON_GRID_COLS: usize = 5;
        if self.icon_picker.is_none() {
            return;
        }
        let now = ctx.input(|i| i.time);
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let light = self.app_settings.light_mode;
        let accent = self.theme_accent();
        let panel = if light {
            Color32::from_rgb(0xF5, 0xF5, 0xF9)
        } else {
            Color32::from_rgb(0x16, 0x16, 0x20)
        };
        let text = if light {
            Color32::from_rgb(0x1E, 0x1E, 0x28)
        } else {
            Color32::from_rgb(0xEC, 0xEC, 0xF0)
        };
        let muted = if light {
            Color32::from_rgb(0x60, 0x60, 0x6A)
        } else {
            Color32::from_rgb(0x9A, 0x9A, 0xA6)
        };
        let border = if light {
            Color32::from_rgb(0xC6, 0xC6, 0xD0)
        } else {
            Color32::from_rgb(0x32, 0x32, 0x3E)
        };
        let field_bg = if light {
            Color32::from_rgb(0xE6, 0xE6, 0xEC)
        } else {
            Color32::from_rgb(0x24, 0x24, 0x2E)
        };

        {
            let p = self.icon_picker.as_mut().unwrap();
            if !p.built {
                let mut ready: Option<Vec<(String, Option<Vec<u8>>)>> = None;
                if let Ok(g) = p.fetch.lock() {
                    if g.done {
                        ready = Some(g.items.clone());
                    }
                }
                if let Some(items) = ready {
                    for (i, (url, bytes)) in items.into_iter().enumerate() {
                        p.full_urls.push(url);
                        let tex = bytes
                            .as_deref()
                            .and_then(|b| image::load_from_memory(b).ok())
                            .map(|im| im.to_rgba8())
                            .map(|rgba| {
                                let (w, h) = rgba.dimensions();
                                let ci = egui::ColorImage::from_rgba_unmultiplied(
                                    [w as usize, h as usize],
                                    rgba.as_raw(),
                                );
                                ctx.load_texture(
                                    format!("sgdb_thumb_{i}"),
                                    ci,
                                    egui::TextureOptions::LINEAR,
                                )
                            });
                        p.thumbs.push(tex);
                    }
                    p.built = true;
                }
            }
        }

        let mut apply_now: Option<Vec<u8>> = None;
        let mut apply_failed = false;
        if let Some(p) = self.icon_picker.as_ref() {
            if let Some(a) = &p.apply {
                if let Ok(g) = a.lock() {
                    if g.done {
                        match &g.bytes {
                            Some(b) => apply_now = Some(b.clone()),
                            None => apply_failed = true,
                        }
                    }
                }
            }
        }
        if let Some(bytes) = apply_now {
            if let Some(img) = image::load_from_memory(&bytes).ok().map(|im| {
                let rgba = im.to_rgba8();
                let (w, h) = rgba.dimensions();
                let (rgba, w, h) = if w.max(h) > 256 {
                    let r = image::imageops::resize(
                        &rgba,
                        256,
                        256,
                        image::imageops::FilterType::Lanczos3,
                    );
                    (r.into_raw(), 256u32, 256u32)
                } else {
                    (rgba.into_raw(), w, h)
                };
                egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba)
            }) {
                let (idx, path) = {
                    let p = self.icon_picker.as_ref().unwrap();
                    (p.game_idx, p.game_path.clone())
                };
                if let Some(cache) = crate::library::custom_icon_path(&path) {
                    if let Some(parent) = cache.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&cache, &bytes);
                }
                let old_tex = self.library.texture(ctx, idx);
                self.library.set_icon(idx, img);
                self.icon_reveal = Some((idx, std::time::Instant::now(), old_tex));
                self.carousel.pending_center = Some(idx);
                crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
            }
            self.icon_picker = None;
            return;
        }
        if apply_failed {
            crate::ui_audio::play(crate::ui_audio::Sfx::Error);
            self.icon_picker = None;
            return;
        }

        let (kb_left, kb_right, kb_enter, kb_esc, kb_up, kb_down) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
            )
        });
        let text_events: Vec<egui::Event> = ctx.input(|i| i.events.clone());
        let li = self.last_input;
        use crate::controller_config::SwitchButton;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let a_edge = kb_enter || (gp_a && !self.hold_pick());
        let b_edge = kb_esc || (gp_b && !self.hold_pick());

        let ready_nav = now - self.icon_picker.as_ref().unwrap().nav_cd > 0.16;
        let sel_before = self.icon_picker.as_ref().unwrap().selected;
        let n = self.icon_picker.as_ref().unwrap().thumbs.len();
        let dancing = self.icon_picker.as_ref().unwrap().squish_at.is_some();
        let editing = self.icon_picker.as_ref().unwrap().editing;

        let mut close = false;
        let mut research = false;
        if !dancing {
            let p = self.icon_picker.as_mut().unwrap();
            if editing {
                for ev in &text_events {
                    match ev {
                        egui::Event::Text(txt) => {
                            for c in txt.chars() {
                                if !c.is_control() && p.search.chars().count() < 48 {
                                    p.search.push(c);
                                }
                            }
                        }
                        egui::Event::Key {
                            key: egui::Key::Backspace,
                            pressed: true,
                            ..
                        } => {
                            p.search.pop();
                        }
                        _ => {}
                    }
                }
                if kb_enter {
                    p.editing = false;
                    p.built = false;
                    p.selected = 0;
                    p.scroll = 0.0;
                    p.full_urls.clear();
                    p.thumbs.clear();
                    research = true;
                } else if kb_esc || b_edge {
                    p.editing = false;
                }
            } else {
                let mut mv_left = kb_left;
                let mut mv_right = kb_right;
                let mut mv_up = kb_up;
                let mut mv_down = kb_down;
                if ready_nav && li.connected {
                    if li.is(SwitchButton::DLeft) || li.lx() < -0.5 {
                        mv_left = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DRight) || li.lx() > 0.5 {
                        mv_right = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DUp) || li.ly() > 0.5 {
                        mv_up = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DDown) || li.ly() < -0.5 {
                        mv_down = true;
                        p.nav_cd = now;
                    }
                }
                if mv_left && p.selected > 0 {
                    p.selected -= 1;
                    crate::ui_audio::play_move();
                }
                if mv_right && n > 0 && p.selected + 1 < n {
                    p.selected += 1;
                    crate::ui_audio::play_move();
                }
                if mv_down && p.selected + ICON_GRID_COLS < n {
                    p.selected += ICON_GRID_COLS;
                    crate::ui_audio::play_move();
                }
                if mv_up && p.selected >= ICON_GRID_COLS {
                    p.selected -= ICON_GRID_COLS;
                    crate::ui_audio::play_move();
                } else if kb_up && p.selected < ICON_GRID_COLS {
                    p.editing = true;
                }
                if p.selected != sel_before {
                    p.follow_sel = true;
                }
                if a_edge && n > 0 && p.thumbs.get(p.selected).map_or(false, |t| t.is_some()) {
                    p.squish_at = Some(now);
                    crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
                }
                if b_edge {
                    close = true;
                }
            }
        }
        self.set_hold_pick(gp_a || gp_b);
        if close {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            self.icon_picker = None;
            return;
        }
        if research {
            let q = self
                .icon_picker
                .as_ref()
                .map(|p| p.search.clone())
                .unwrap_or_default();
            let key = self.app_settings.steamgriddb_key.clone();
            if let Some(p) = self.icon_picker.as_mut() {
                p.fetch = start_icon_fetch(key, q);
            }
        }

        {
            let p = self.icon_picker.as_mut().unwrap();
            p.anim += (1.0 - p.anim) * (dt * 12.0).min(1.0);
            if let Some(t0) = p.squish_at {
                if (now - t0) as f32 > 0.34 && p.apply.is_none() {
                    let url = p.full_urls.get(p.selected).cloned().unwrap_or_default();
                    let shared = std::sync::Arc::new(std::sync::Mutex::new(IconApply::default()));
                    let s2 = shared.clone();
                    std::thread::spawn(move || {
                        let bytes = crate::steamgrid::curl_bytes(&url);
                        if let Ok(mut g) = s2.lock() {
                            g.done = true;
                            g.bytes = bytes;
                        }
                    });
                    p.apply = Some(shared);
                }
            }
        }

        ctx.request_repaint();
        let (anim, search, selected, scroll, squish_at, editing, built, thumbs) = {
            let p = self.icon_picker.as_ref().unwrap();
            (
                p.anim,
                p.search.clone(),
                p.selected,
                p.scroll,
                p.squish_at,
                p.editing,
                p.built,
                p.thumbs.clone(),
            )
        };
        let err = self
            .icon_picker
            .as_ref()
            .and_then(|p| p.fetch.lock().ok().and_then(|g| g.error.clone()));
        let ease = {
            let a = anim.clamp(0.0, 1.0);
            a * a * (3.0 - 2.0 * a)
        };
        let t = ctx.input(|i| i.time) as f32;
        let screen = ctx.viewport_rect();
        let mut paint = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("icon_picker"),
        ));
        paint.set_opacity(ease);
        paint.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(205));

        let pop = 0.92 + 0.08 * ease;
        let w = (screen.width() * 0.72).clamp(560.0, 1040.0) * pop;
        let h = (screen.height() * 0.6).clamp(360.0, 560.0) * pop;
        let box_rect = egui::Rect::from_center_size(screen.center(), Vec2::new(w, h));
        paint.rect_filled(
            box_rect.translate(Vec2::new(0.0, 12.0)),
            CornerRadius::same(20),
            Color32::from_black_alpha(90),
        );
        paint.rect_filled(box_rect, CornerRadius::same(20), panel);
        paint.rect_stroke(
            box_rect,
            CornerRadius::same(20),
            Stroke::new(1.5_f32, border),
            egui::StrokeKind::Outside,
        );
        paint.text(
            egui::pos2(box_rect.center().x, box_rect.min.y + 34.0),
            egui::Align2::CENTER_CENTER,
            "Choose an Icon",
            FontId::proportional(22.0),
            text,
        );

        let field = egui::Rect::from_min_size(
            egui::pos2(box_rect.min.x + 40.0, box_rect.min.y + 58.0),
            Vec2::new(w - 80.0, 40.0),
        );
        let field_resp = ui.allocate_rect(field, egui::Sense::click());
        paint.rect_filled(field, CornerRadius::same(10), field_bg);
        paint.rect_stroke(
            field,
            CornerRadius::same(10),
            Stroke::new(
                if editing { 2.0_f32 } else { 1.0_f32 },
                if editing { accent } else { border },
            ),
            egui::StrokeKind::Outside,
        );
        let query_disp = if search.is_empty() {
            "Type a game name…".to_string()
        } else {
            search.clone()
        };
        let qcol = if search.is_empty() { muted } else { text };
        paint.text(
            egui::pos2(field.min.x + 14.0, field.center().y),
            egui::Align2::LEFT_CENTER,
            &query_disp,
            FontId::proportional(17.0),
            qcol,
        );
        if editing && (now * 1.6).fract() < 0.5 {
            let tw = ui.fonts_mut(|f| {
                f.layout_no_wrap(search.clone(), FontId::proportional(17.0), text)
                    .size()
                    .x
            });
            let cx = (field.min.x + 14.0 + tw).min(field.max.x - 10.0);
            paint.line_segment(
                [
                    egui::pos2(cx, field.center().y - 11.0),
                    egui::pos2(cx, field.center().y + 11.0),
                ],
                Stroke::new(2.0_f32, accent),
            );
        }

        let content = egui::Rect::from_min_max(
            egui::pos2(box_rect.min.x + 24.0, field.max.y + 20.0),
            egui::pos2(box_rect.max.x - 24.0, box_rect.max.y - 54.0),
        );
        let mut new_selected: Option<usize> = None;
        let mut dbl_selected: Option<usize> = None;
        let mut scroll_target = scroll;
        let mut cell_px = 1.0f32;
        let mut max_scroll_rows = 0.0f32;
        let wheel_dy = ctx.input(|i| i.smooth_scroll_delta.y);
        if !built {
            paint.text(
                content.center(),
                egui::Align2::CENTER_CENTER,
                "Searching SteamGridDB…",
                FontId::proportional(18.0),
                muted,
            );
        } else if let Some(e) = err {
            paint.text(
                content.center() - Vec2::new(0.0, 10.0),
                egui::Align2::CENTER_CENTER,
                &e,
                FontId::proportional(17.0),
                muted,
            );
            paint.text(
                content.center() + Vec2::new(0.0, 22.0),
                egui::Align2::CENTER_CENTER,
                "[Up] search a different name  ·  [B] Cancel",
                FontId::proportional(13.0),
                muted,
            );
        } else if thumbs.is_empty() {
            paint.text(
                content.center() - Vec2::new(0.0, 10.0),
                egui::Align2::CENTER_CENTER,
                "No icons found for this name.",
                FontId::proportional(17.0),
                muted,
            );
            paint.text(
                content.center() + Vec2::new(0.0, 22.0),
                egui::Align2::CENTER_CENTER,
                "[Up] try a different name  ·  [B] Cancel",
                FontId::proportional(13.0),
                muted,
            );
        } else {
            let cols = ICON_GRID_COLS;
            let gap = 18.0;
            let pad_top = 10.0;
            let tile = (((content.width() - gap * (cols as f32 - 1.0)) / cols as f32).min(118.0))
                .max(48.0);
            let cell = tile + gap;
            let n_items = thumbs.len();
            let rows = (n_items + cols - 1) / cols;
            let sel_row = (selected / cols) as f32;
            let rows_vis = ((content.height() - pad_top) / cell).max(1.0);
            let max_scroll = (rows as f32 - rows_vis).max(0.0);
            let mut kv = scroll;
            if sel_row < scroll {
                kv = sel_row;
            } else if sel_row + 1.0 > scroll + rows_vis {
                kv = sel_row + 1.0 - rows_vis;
            }
            scroll_target = kv.clamp(0.0, max_scroll);
            cell_px = cell;
            max_scroll_rows = max_scroll;

            let grid_w = tile * cols as f32 + gap * (cols as f32 - 1.0);
            let x0 = content.center().x - grid_w * 0.5;
            let y0 = content.min.y + pad_top - scroll * cell;
            let clip = paint.with_clip_rect(content);
            for (i, thumb) in thumbs.iter().enumerate() {
                let row = i / cols;
                let col = i % cols;
                let cx = x0 + col as f32 * cell + tile * 0.5;
                let cy = y0 + row as f32 * cell + tile * 0.5;
                if cy + tile < content.min.y || cy - tile > content.max.y {
                    continue;
                }
                let sel = i == selected;
                let (mut sx, mut sy) = (1.0f32, 1.0f32);
                let mut sz = tile;
                if sel {
                    if let Some(t0) = squish_at {
                        let e = (now - t0) as f32;
                        if e < 0.34 {
                            let q = (e * 34.0).sin() * 0.16 * (1.0 - e / 0.34);
                            sx = 1.0 + q;
                            sy = 1.0 - q;
                        }
                    }
                    sz *= 1.06;
                }
                let r =
                    egui::Rect::from_center_size(egui::pos2(cx, cy), Vec2::new(sz * sx, sz * sy));
                clip.rect_filled(
                    r.translate(Vec2::new(0.0, 4.0)),
                    CornerRadius::same(14),
                    Color32::from_black_alpha(80),
                );
                if sel {
                    crate::carousel::draw_gradient_rounded_rect(
                        &clip,
                        r.center(),
                        r.expand(6.0),
                        18.0,
                        t,
                        180,
                    );
                    crate::carousel::draw_gradient_rounded_rect(
                        &clip,
                        r.center(),
                        r.expand(3.0),
                        16.0,
                        t,
                        255,
                    );
                }
                clip.rect_filled(r, CornerRadius::same(14), field_bg);
                if let Some(tex) = thumb {
                    crate::carousel::draw_rounded_image(&clip, tex.id(), r, 14.0, Color32::WHITE);
                } else {
                    clip.text(
                        r.center(),
                        egui::Align2::CENTER_CENTER,
                        "×",
                        FontId::proportional(sz * 0.3),
                        muted,
                    );
                }
                if content.contains(egui::pos2(cx, cy)) {
                    let resp = ui.allocate_rect(r, egui::Sense::click());
                    if resp.double_clicked() {
                        dbl_selected = Some(i);
                    } else if resp.clicked() {
                        new_selected = Some(i);
                    }
                }
            }
        }

        let hint = if editing {
            "Type a name  ·  [Enter] Search  ·  [Esc] Done"
        } else {
            "[D-Pad] Browse  ·  [A] Choose  ·  Click search to rename  ·  [B] Cancel"
        };
        paint.text(
            egui::pos2(box_rect.center().x, box_rect.max.y - 26.0),
            egui::Align2::CENTER_CENTER,
            hint,
            FontId::proportional(13.0),
            muted,
        );

        let mut dbl_whistle = false;
        if let Some(pp) = self.icon_picker.as_mut() {
            if let Some(s) = new_selected {
                pp.selected = s;
            }
            if let Some(s) = dbl_selected {
                pp.selected = s;
                if pp.squish_at.is_none() && pp.thumbs.get(s).map_or(false, |t| t.is_some()) {
                    pp.squish_at = Some(now);
                    dbl_whistle = true;
                }
            }
            if wheel_dy.abs() > 0.1 && max_scroll_rows > 0.0 {
                pp.scroll = (pp.scroll - wheel_dy / cell_px).clamp(0.0, max_scroll_rows);
                pp.follow_sel = false;
            } else if pp.follow_sel {
                pp.scroll += (scroll_target - pp.scroll) * (dt * 16.0).min(1.0);
            }
            if field_resp.clicked() {
                pp.editing = true;
            }
        }
        if dbl_whistle {
            crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
        }
    }

    fn hold_pick(&self) -> bool {
        self.icon_picker.as_ref().map_or(false, |p| p.hold)
    }
    fn set_hold_pick(&mut self, v: bool) {
        if let Some(p) = self.icon_picker.as_mut() {
            p.hold = v;
        }
    }

    fn cs_working_order(&self) -> Vec<crate::app_settings::CarouselRef> {
        use crate::app_settings::CarouselRef;
        let list_names: Vec<String> = self
            .app_settings
            .carousel_lists
            .iter()
            .map(|l| l.name.clone())
            .collect();
        let game_paths: Vec<std::path::PathBuf> = self
            .library
            .games
            .iter()
            .filter(|g| g.download.is_none())
            .map(|g| g.path.clone())
            .collect();
        let mut out: Vec<CarouselRef> = Vec::new();
        for r in &self.app_settings.carousel_order {
            let keep = match r {
                CarouselRef::List(n) => list_names.iter().any(|x| x == n),
                CarouselRef::Game(p) => game_paths.iter().any(|x| x == p),
            };
            if keep && !out.contains(r) {
                out.push(r.clone());
            }
        }
        for n in &list_names {
            let r = CarouselRef::List(n.clone());
            if !out.contains(&r) {
                out.push(r);
            }
        }
        for p in &game_paths {
            let r = CarouselRef::Game(p.clone());
            if !out.contains(&r) {
                out.push(r);
            }
        }
        out
    }

    fn update_carousel_settings(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let target = if self.carousel_settings_open {
            1.0
        } else {
            0.0
        };
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.cs_anim += (target - self.cs_anim) * (dt * 12.0).min(1.0);
        if self.cs_anim < 0.004 {
            return;
        }
        let ease = {
            let a = self.cs_anim.clamp(0.0, 1.0);
            a * a * (3.0 - 2.0 * a)
        };
        let t = ctx.input(|i| i.time) as f32;
        let full = ctx.viewport_rect();
        let s = (full.height() / 820.0).clamp(1.0, 2.4);
        let backdrop_theme = self.app_settings.backdrop_theme;
        let lightish = self.app_settings.light_mode;
        let pick = |dark: egui::Color32, lite: egui::Color32| if lightish { lite } else { dark };
        let text = pick(
            egui::Color32::from_rgb(0xEC, 0xEC, 0xF0),
            egui::Color32::from_rgb(0x1E, 0x1E, 0x28),
        );
        let muted = pick(
            egui::Color32::from_rgb(0x8A, 0x8A, 0x98),
            egui::Color32::from_rgb(0x60, 0x60, 0x6A),
        );
        let panel = pick(
            egui::Color32::from_rgb(0x16, 0x16, 0x1E),
            egui::Color32::from_rgb(0xFF, 0xFF, 0xFF),
        );
        let border = pick(
            egui::Color32::from_rgb(0x30, 0x30, 0x3C),
            egui::Color32::from_rgb(0xC6, 0xC6, 0xD0),
        );
        let sel = pick(
            egui::Color32::from_rgb(0x1E, 0x1E, 0x28),
            egui::Color32::from_rgb(0xDD, 0xDD, 0xE6),
        );
        let hover = pick(
            egui::Color32::from_rgb(0x18, 0x18, 0x22),
            egui::Color32::from_rgb(0xEA, 0xEA, 0xF0),
        );
        let accent = self.theme_accent();

        let center = full.center();
        let sf = 0.965 + 0.035 * ease;
        let sp = |p: egui::Pos2| center + (p - center) * sf;
        let sr = |r: egui::Rect| {
            egui::Rect::from_center_size(center + (r.center() - center) * sf, r.size() * sf)
        };

        let mut p = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("carousel_settings"),
        ));
        p.set_opacity(ease);
        crate::carousel::draw_backdrop(
            &p,
            full,
            accent,
            t,
            backdrop_theme,
            1.0,
            if lightish { 1.0 } else { 0.0 },
            None,
        );
        let scrim = if lightish {
            egui::Color32::from_rgba_unmultiplied(0xFA, 0xFA, 0xFC, 235)
        } else {
            egui::Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 140)
        };
        p.rect_filled(full, egui::CornerRadius::ZERO, scrim);

        let mx = full.width() * 0.055;
        let header_font = 34.0 * s;
        p.text(
            sp(egui::pos2(full.min.x + mx, full.min.y + 40.0 * s)),
            egui::Align2::LEFT_TOP,
            "Carousel Settings",
            egui::FontId::proportional(header_font * sf),
            text,
        );
        let header_y = full.min.y + 40.0 * s + header_font + 20.0 * s;
        p.line_segment(
            [
                sp(egui::pos2(full.min.x + mx, header_y)),
                sp(egui::pos2(full.max.x - mx, header_y)),
            ],
            egui::Stroke::new(1.0_f32, border),
        );

        use crate::controller_config::SwitchButton;
        let now = ctx.input(|i| i.time);
        let li = self.last_input;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let cs_gp_a_edge = gp_a && !self.cs_ab_held;
        let a_edge = ctx.input(|i| i.key_pressed(egui::Key::Enter)) || (gp_a && !self.cs_ab_held);
        let b_edge = ctx.input(|i| i.key_pressed(egui::Key::Escape)) || (gp_b && !self.cs_ab_held);
        self.cs_ab_held = gp_a || gp_b;
        let (mut nu, mut nd, mut nl, mut nr) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        let gp_plus = li.connected && li.is(SwitchButton::Plus);
        let plus_edge =
            (gp_plus && !self.cs_plus_held) || ctx.input(|i| i.key_pressed(egui::Key::N));
        self.cs_plus_held = gp_plus;
        let gp_x = li.connected && li.is(SwitchButton::X);
        let x_edge = (gp_x && !self.cs_x_held) || ctx.input(|i| i.key_pressed(egui::Key::E));
        self.cs_x_held = gp_x;

        const INITIAL_DELAY: f64 = 0.70;
        const REPEAT_RATE: f64 = 0.14;
        if li.connected {
            let cur_dir: u8 = if li.is(SwitchButton::DUp) || li.ly() > 0.5 {
                1
            } else if li.is(SwitchButton::DDown) || li.ly() < -0.5 {
                2
            } else if li.is(SwitchButton::DLeft) || li.lx() < -0.5 {
                3
            } else if li.is(SwitchButton::DRight) || li.lx() > 0.5 {
                4
            } else {
                0
            };
            if cur_dir == 0 {
                self.cs_nav_held_dir = 0;
                self.cs_nav_held_since = 0.0;
            } else if cur_dir != self.cs_nav_held_dir {
                self.cs_nav_held_dir = cur_dir;
                self.cs_nav_held_since = now;
                self.cs_nav_cd = now;
                match cur_dir {
                    1 => nu = true,
                    2 => nd = true,
                    3 => nl = true,
                    _ => nr = true,
                }
            } else {
                let held_for = now - self.cs_nav_held_since;
                if held_for >= INITIAL_DELAY && now - self.cs_nav_cd >= REPEAT_RATE {
                    self.cs_nav_cd = now;
                    match cur_dir {
                        1 => nu = true,
                        2 => nd = true,
                        3 => nl = true,
                        _ => nr = true,
                    }
                }
            }
        } else {
            self.cs_nav_held_dir = 0;
            self.cs_nav_held_since = 0.0;
        }

        let menu_open = self.cs_list_menu.is_some();
        let (raw_nu, raw_nd, raw_a, raw_b) = (nu, nd, a_edge, b_edge);
        let block = self.confirm.is_some() || menu_open || self.vkeyboard.open;
        let (a_edge, b_edge, plus_edge, x_edge) = if block {
            (false, false, false, false)
        } else {
            (a_edge, b_edge, plus_edge, x_edge)
        };
        let (nu, nd, nl, nr) = if block {
            (false, false, false, false)
        } else {
            (nu, nd, nl, nr)
        };

        let side_w = (full.width() * 0.22).max(300.0);
        let side_x = full.min.x + mx;
        let side_top = header_y + 34.0 * s;
        let item_h = 64.0 * s;
        let tabs = ["Create A List", "Manage Carousel"];
        for (i, label) in tabs.iter().enumerate() {
            let base = egui::Rect::from_min_size(
                egui::pos2(side_x, side_top + i as f32 * (item_h + 10.0 * s)),
                egui::Vec2::new(side_w, item_h),
            );
            let r = sr(base);
            let selected = self.cs_tab == i;
            let ring = if !self.cs_focus_grid { accent } else { border };
            let rounding = egui::CornerRadius::from(12.0 * s);
            if selected {
                p.rect_filled(r, rounding, sel);
                p.rect_stroke(
                    r,
                    rounding,
                    egui::Stroke::new(1.8_f32, ring),
                    egui::StrokeKind::Outside,
                );
                let bar = sr(egui::Rect::from_min_size(
                    base.min + egui::Vec2::new(6.0 * s, 12.0 * s),
                    egui::Vec2::new(4.0 * s, base.height() - 24.0 * s),
                ));
                p.rect_filled(bar, egui::CornerRadius::from(2.0 * s), ring);
            } else if ui.rect_contains_pointer(r) {
                p.rect_filled(r, rounding, hover);
            }
            p.text(
                sp(egui::pos2(base.min.x + 26.0 * s, base.center().y)),
                egui::Align2::LEFT_CENTER,
                *label,
                egui::FontId::proportional(19.0 * s * sf),
                if selected { text } else { muted },
            );
            if !block && ui.allocate_rect(r, egui::Sense::click()).clicked() {
                self.cs_tab = i;
                self.cs_focus_grid = false;
                self.cs_build_grab = None;
                self.cs_selected = 0;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        }
        if !self.cs_focus_grid {
            if nu && self.cs_tab > 0 {
                self.cs_tab -= 1;
                self.cs_build_grab = None;
                self.cs_selected = 0;
                crate::ui_audio::play_move();
            }
            if nd && self.cs_tab + 1 < tabs.len() {
                self.cs_tab += 1;
                self.cs_build_grab = None;
                self.cs_selected = 0;
                crate::ui_audio::play_move();
            }
        }

        let footer_y = full.max.y - 60.0 * s;
        p.line_segment(
            [
                sp(egui::pos2(full.min.x + mx, footer_y)),
                sp(egui::pos2(full.max.x - mx, footer_y)),
            ],
            egui::Stroke::new(1.0_f32, border),
        );
        let pad = self.last_input.connected;
        let hint = if self.cs_creating {
            if pad {
                "Change    ·    [A] Edit    ·    [B] Cancel"
            } else {
                "[Arrows] Change    ·    [Enter] Edit    ·    [Esc] Cancel"
            }
        } else if self.cs_tab == 0 && self.cs_focus_grid {
            if pad {
                "Move    ·    [A] Pick    ·    [+] Create List    ·    [B] Back"
            } else {
                "[Arrows] Move    ·    [Enter] Pick    ·    [N] Create List    ·    [Esc] Back"
            }
        } else if self.cs_tab == 1 && self.cs_focus_grid {
            if pad {
                "Move    ·    [A] Grab / Drop    ·    [X] Edit list    ·    [B] Back"
            } else {
                "[Arrows] Move    ·    [Enter] Grab / Drop    ·    [E] Edit list    ·    [Esc] Back"
            }
        } else {
            if pad {
                "Move    ·    [B] Back"
            } else {
                "[Arrows] Move    ·    [Esc] Back"
            }
        };
        let hint_font = egui::FontId::proportional(14.0 * s * sf);
        let hy = full.max.y - 30.0 * s;
        let hint_w = ui.fonts_mut(|f| {
            f.layout_no_wrap(hint.to_string(), hint_font.clone(), muted)
                .size()
                .x
        });
        p.text(
            sp(egui::pos2(full.max.x - mx, hy)),
            egui::Align2::RIGHT_CENTER,
            hint,
            hint_font,
            muted,
        );
        if pad {
            let dp_r = 8.0 * s * sf;
            draw_dpad(
                &p,
                sp(egui::pos2(full.max.x - mx - hint_w - dp_r - 8.0 * s, hy)),
                dp_r,
                muted,
            );
        }

        let content = egui::Rect::from_min_max(
            egui::pos2(side_x + side_w + 48.0 * s, side_top),
            egui::pos2(full.max.x - mx, footer_y - 20.0 * s),
        );

        let dlg = self.cs_creating;
        if self.cs_tab == 0 {
            let games: Vec<usize> = (0..self.library.games.len())
                .filter(|&i| self.library.games[i].download.is_none())
                .collect();
            let n = games.len();
            if n == 0 {
                p.text(
                    sp(content.center()),
                    egui::Align2::CENTER_CENTER,
                    "No games in your library yet.",
                    egui::FontId::proportional(18.0 * s * sf),
                    muted,
                );
            } else {
                let cols = 5usize;
                let pad = 16.0 * s;
                let grid = egui::Rect::from_min_max(
                    content.min + egui::Vec2::splat(pad),
                    egui::pos2(content.max.x - pad, content.max.y - 66.0 * s),
                );
                let gap = 22.0 * s;
                let tile =
                    ((grid.width() - gap * (cols as f32 - 1.0)) / cols as f32).min(186.0 * s);
                let cell_w = tile + gap;
                let cell_h = tile + 30.0 * s;
                let rows = (n + cols - 1) / cols;
                let max_scroll = (rows as f32 * cell_h - grid.height()).max(0.0);
                if !dlg {
                    if self.cs_focus_grid {
                        if nr && self.cs_selected + 1 < n {
                            self.cs_selected += 1;
                            crate::ui_audio::play_move();
                        }
                        if nl {
                            if self.cs_selected % cols == 0 {
                                self.cs_focus_grid = false;
                            } else {
                                self.cs_selected -= 1;
                                crate::ui_audio::play_move();
                            }
                        }
                        if nd && self.cs_selected + cols < n {
                            self.cs_selected += cols;
                            crate::ui_audio::play_move();
                        }
                        if nu && self.cs_selected >= cols {
                            self.cs_selected -= cols;
                            crate::ui_audio::play_move();
                        }
                    } else if nr {
                        self.cs_focus_grid = true;
                        crate::ui_audio::play_move();
                    }
                }
                self.cs_selected = self.cs_selected.min(n.saturating_sub(1));
                let wheel = ui.input(|i| i.smooth_scroll_delta.y);
                if !dlg && wheel.abs() > 0.1 {
                    self.cs_scroll = (self.cs_scroll - wheel).clamp(0.0, max_scroll);
                }
                if self.cs_focus_grid {
                    let srow = (self.cs_selected / cols) as f32 * cell_h;
                    if srow < self.cs_scroll {
                        self.cs_scroll = srow;
                    } else if srow + tile > self.cs_scroll + grid.height() {
                        self.cs_scroll = srow + tile - grid.height();
                    }
                }
                self.cs_scroll = self.cs_scroll.clamp(0.0, max_scroll);

                let mut toggle: Option<usize> = None;
                if !dlg && self.cs_focus_grid && a_edge {
                    toggle = Some(self.cs_selected);
                }

                let clip = p.with_clip_rect(sr(content));
                for (vi, &gi) in games.iter().enumerate() {
                    let rr = vi / cols;
                    let cc = vi % cols;
                    let tx = grid.min.x + cc as f32 * cell_w;
                    let ty = grid.min.y + rr as f32 * cell_h - self.cs_scroll;
                    if ty + tile < content.min.y || ty > grid.max.y {
                        continue;
                    }
                    let path = self.library.games[gi].path.clone();
                    let picked = self.cs_list_picks.iter().any(|p2| *p2 == path);
                    let (fade, pop) = {
                        let e = self
                            .cs_pick_anim
                            .entry(path.clone())
                            .or_insert((if picked { 1.0 } else { 0.0 }, 0.0));
                        let target = if picked { 1.0 } else { 0.0 };
                        e.0 += (target - e.0) * (dt * 9.0).min(1.0);
                        e.1 *= 1.0 - (dt * 6.0).min(1.0);
                        (e.0, e.1)
                    };
                    let bounce = (pop.clamp(0.0, 1.0) * std::f32::consts::PI).sin() * 11.0 * s;
                    let base = egui::Rect::from_min_size(
                        egui::pos2(tx, ty - bounce),
                        egui::Vec2::splat(tile),
                    );
                    let rect = sr(base);
                    let selg = self.cs_focus_grid && vi == self.cs_selected;
                    if selg {
                        crate::carousel::draw_gradient_rounded_rect(
                            &clip,
                            rect.center(),
                            rect.expand(5.0),
                            13.0,
                            t,
                            110,
                        );
                        crate::carousel::draw_gradient_rounded_rect(
                            &clip,
                            rect.center(),
                            rect.expand(2.5),
                            11.0,
                            t,
                            255,
                        );
                    }
                    clip.rect_filled(rect, egui::CornerRadius::same(9), panel);
                    if let Some(tex) = self.library.texture(ctx, gi) {
                        let base_g = if lightish { 155.0 } else { 66.0 };
                        let g = (base_g + (255.0 - base_g) * fade) as u8;
                        crate::carousel::draw_rounded_image(
                            &clip,
                            tex.id(),
                            rect,
                            9.0,
                            egui::Color32::from_rgb(g, g, g),
                        );
                    } else {
                        let g = (110.0 + 90.0 * fade) as u8;
                        clip.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            &self.library.games[gi].title,
                            egui::FontId::proportional(12.0 * s),
                            egui::Color32::from_rgb(g, g, g),
                        );
                    }
                    if fade > 0.15 {
                        let br = 15.0 * s;
                        let bc = rect.right_top() + egui::Vec2::new(-br - 5.0 * s, br + 5.0 * s);
                        let a = (fade * 255.0) as u8;
                        clip.circle_filled(
                            bc,
                            br,
                            egui::Color32::from_rgba_unmultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                a,
                            ),
                        );
                        clip.circle_stroke(
                            bc,
                            br,
                            egui::Stroke::new(
                                1.5_f32,
                                egui::Color32::from_rgba_unmultiplied(
                                    255,
                                    255,
                                    255,
                                    (a as f32 * 0.5) as u8,
                                ),
                            ),
                        );
                        draw_check(
                            &clip,
                            bc,
                            br * 1.15,
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, a),
                        );
                    }
                    if !dlg
                        && !block
                        && ui.rect_contains_pointer(rect)
                        && ui.allocate_rect(rect, egui::Sense::click()).clicked()
                    {
                        toggle = Some(vi);
                        self.cs_focus_grid = true;
                        self.cs_selected = vi;
                    }
                }

                if let Some(vi) = toggle {
                    let gi = games[vi];
                    let path = self.library.games[gi].path.clone();
                    if let Some(pos) = self.cs_list_picks.iter().position(|p2| *p2 == path) {
                        self.cs_list_picks.remove(pos);
                    } else {
                        self.cs_list_picks.push(path.clone());
                    }
                    self.cs_pick_anim.entry(path).or_insert((0.0, 0.0)).1 = 1.0;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
            }

            let want_btn = !self.cs_list_picks.is_empty() && !dlg;
            self.cs_create_btn +=
                ((if want_btn { 1.0 } else { 0.0 }) - self.cs_create_btn) * (dt * 10.0).min(1.0);
            if self.cs_create_btn > 0.01 {
                let bw = 224.0 * s;
                let bh = 50.0 * s;
                let base = egui::Rect::from_min_size(
                    egui::pos2(content.max.x - bw, footer_y - 16.0 * s - bh),
                    egui::Vec2::new(bw, bh),
                );
                let r = sr(base);
                let a = (self.cs_create_btn * 255.0) as u8;
                p.rect_filled(
                    r,
                    egui::CornerRadius::from(12.0 * s),
                    egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a),
                );
                p.text(
                    r.center(),
                    egui::Align2::CENTER_CENTER,
                    &format!("Create List  ({})", self.cs_list_picks.len()),
                    egui::FontId::proportional(18.0 * s * sf),
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, a),
                );
                let clicked = !block && ui.allocate_rect(r, egui::Sense::click()).clicked();
                if want_btn && (clicked || plus_edge) {
                    self.cs_creating = true;
                    self.cs_dialog_row = 0;
                    self.cs_name_editing = false;
                    if self.cs_new_name.trim().is_empty() {
                        self.cs_new_name =
                            format!("List {}", self.app_settings.carousel_lists.len() + 1);
                    }
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
            }
        } else {
            use crate::app_settings::CarouselRef;
            let order = self.cs_working_order();
            let mut slots: Vec<(bool, usize, usize)> = Vec::new();
            for r in &order {
                match r {
                    CarouselRef::Game(pth) => {
                        if let Some(gi) = self.library.games.iter().position(|g| &g.path == pth) {
                            slots.push((false, gi, 0));
                        }
                    }
                    CarouselRef::List(nm) => {
                        if let Some(li) = self
                            .app_settings
                            .carousel_lists
                            .iter()
                            .position(|l| &l.name == nm)
                        {
                            let c = self.app_settings.carousel_lists[li].games.len();
                            slots.push((true, li, c));
                        }
                    }
                }
            }
            let n = slots.len();
            if n == 0 {
                p.text(
                    sp(content.center()),
                    egui::Align2::CENTER_CENTER,
                    "No games in your library yet.",
                    egui::FontId::proportional(18.0 * s * sf),
                    muted,
                );
            } else {
                let grabbing = self.cs_build_grab.is_some();
                self.cs_grab_lift +=
                    ((if grabbing { 1.0 } else { 0.0 }) - self.cs_grab_lift) * (dt * 12.0).min(1.0);
                self.cs_selected = self.cs_selected.min(n - 1);

                let grab_g = self.cs_build_grab;
                let vis: Vec<usize> = (0..n).filter(|&i| Some(i) != grab_g).collect();
                let m = vis.len();

                if self.cs_focus_grid {
                    if grabbing {
                        if nl {
                            self.cs_build_insert = self.cs_build_insert.saturating_sub(1);
                            crate::ui_audio::play_move();
                        }
                        if nr && self.cs_build_insert < m {
                            self.cs_build_insert += 1;
                            crate::ui_audio::play_move();
                        }
                    } else {
                        if nr && self.cs_selected + 1 < n {
                            self.cs_selected += 1;
                            crate::ui_audio::play_move();
                        }
                        if nl {
                            if self.cs_selected == 0 {
                                self.cs_focus_grid = false;
                            } else {
                                self.cs_selected -= 1;
                                crate::ui_audio::play_move();
                            }
                        }
                    }
                } else if nr {
                    self.cs_focus_grid = true;
                    crate::ui_audio::play_move();
                }

                let mut commit = false;
                if self.cs_focus_grid && a_edge {
                    if grabbing {
                        commit = true;
                    } else {
                        self.cs_build_grab = Some(self.cs_selected);
                        self.cs_build_insert = self.cs_selected;
                        self.cs_grab_lift = 0.0;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                }

                let tile = (content.height() * 0.44).clamp(96.0 * s, 152.0 * s);
                let gap = 34.0 * s;
                let cell = tile + gap;
                let left_margin = 46.0 * s;
                let view_w = content.width() - left_margin - 30.0 * s;
                let cnt = if grabbing { m } else { n };
                let total = (cnt as f32 * cell - gap).max(tile);
                let max_scroll = (total - view_w).max(0.0);
                let focus_idx = if grabbing {
                    self.cs_build_insert.min(cnt.saturating_sub(1))
                } else {
                    self.cs_selected
                };
                let wheel = ui.input(|i| {
                    let d = i.smooth_scroll_delta;
                    if d.x.abs() > d.y.abs() {
                        d.x
                    } else {
                        d.y
                    }
                });
                if wheel.abs() > 0.1 {
                    self.cs_row_scroll = (self.cs_row_scroll - wheel).clamp(0.0, max_scroll);
                }
                if nl || nr || a_edge {
                    let il = focus_idx as f32 * cell;
                    let ir = il + tile;
                    if il < self.cs_row_scroll {
                        self.cs_row_scroll = il;
                    } else if ir > self.cs_row_scroll + view_w {
                        self.cs_row_scroll = ir - view_w;
                    }
                }
                self.cs_row_scroll = self.cs_row_scroll.clamp(0.0, max_scroll);
                let row_x0 = content.min.x + left_margin - self.cs_row_scroll;
                let row_cy = content.center().y + 14.0 * s;
                let slot_x = |i: usize| row_x0 + i as f32 * cell;
                let clip_rect = egui::Rect::from_min_max(
                    egui::pos2(content.min.x, content.min.y - 4.0 * s),
                    content.max,
                );
                let clip = p.with_clip_rect(sr(clip_rect));

                let draw_tile = |clip: &egui::Painter,
                                 lib: &mut crate::library::Library,
                                 lists: &[crate::app_settings::GameList],
                                 rect: egui::Rect,
                                 is_list: bool,
                                 idx: usize,
                                 count: usize,
                                 ss: f32| {
                    clip.rect_filled(rect, egui::CornerRadius::same(10), panel);
                    if is_list {
                        clip.rect_filled(
                            rect.shrink(2.0),
                            egui::CornerRadius::same(9),
                            egui::Color32::from_rgba_unmultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                220,
                            ),
                        );
                        clip.text(
                            rect.center() - egui::Vec2::new(0.0, 9.0 * ss),
                            egui::Align2::CENTER_CENTER,
                            &lists[idx].name,
                            egui::FontId::proportional(15.0 * ss),
                            egui::Color32::WHITE,
                        );
                        clip.text(
                            rect.center() + egui::Vec2::new(0.0, 16.0 * ss),
                            egui::Align2::CENTER_CENTER,
                            &format!("{} games", count),
                            egui::FontId::proportional(12.0 * ss),
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210),
                        );
                    } else if let Some(tex) = lib.texture(ctx, idx) {
                        crate::carousel::draw_rounded_image(
                            clip,
                            tex.id(),
                            rect,
                            10.0,
                            egui::Color32::WHITE,
                        );
                    }
                };

                for (pos, &si) in vis.iter().enumerate() {
                    let (is_list, idx, count) = slots[si];
                    let tx = slot_x(pos);
                    if tx + tile < content.min.x || tx > content.max.x {
                        continue;
                    }
                    let base = egui::Rect::from_min_size(
                        egui::pos2(tx, row_cy - tile / 2.0),
                        egui::Vec2::splat(tile),
                    );
                    let rect = sr(base);
                    let focused = !grabbing && self.cs_focus_grid && si == self.cs_selected;
                    if focused {
                        crate::carousel::draw_gradient_rounded_rect(
                            &clip,
                            rect.center(),
                            rect.expand(6.0),
                            14.0,
                            t,
                            110,
                        );
                        crate::carousel::draw_gradient_rounded_rect(
                            &clip,
                            rect.center(),
                            rect.expand(3.0),
                            12.0,
                            t,
                            255,
                        );
                    }
                    draw_tile(
                        &clip,
                        &mut self.library,
                        &self.app_settings.carousel_lists,
                        rect,
                        is_list,
                        idx,
                        count,
                        s,
                    );
                }

                if grabbing {
                    for k in 0..=m {
                        let lx = slot_x(k) - gap / 2.0;
                        let active = k == self.cs_build_insert;
                        let col = if active {
                            accent
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                70,
                            )
                        };
                        let w = if active { 5.0_f32 } else { 2.5_f32 };
                        let top = row_cy - tile / 2.0 - 10.0 * s;
                        let bot = row_cy + tile / 2.0 + 10.0 * s;
                        clip.line_segment(
                            [sp(egui::pos2(lx, top)), sp(egui::pos2(lx, bot))],
                            egui::Stroke::new(w, col),
                        );
                        if active {
                            clip.circle_filled(sp(egui::pos2(lx, top)), 5.0 * s, accent);
                            clip.circle_filled(sp(egui::pos2(lx, bot)), 5.0 * s, accent);
                        }
                    }
                    if let Some(g) = grab_g {
                        let (is_list, idx, count) = slots[g];
                        let gtile = tile * 0.74;
                        let gx = (slot_x(self.cs_build_insert.min(m)) - gap / 2.0).clamp(
                            content.min.x + gtile / 2.0 + 8.0 * s,
                            content.max.x - gtile / 2.0 - 8.0 * s,
                        );
                        let gy = row_cy
                            - tile / 2.0
                            - 18.0 * s
                            - self.cs_grab_lift * 24.0 * s
                            - gtile / 2.0;
                        let grect = sr(egui::Rect::from_center_size(
                            egui::pos2(gx, gy),
                            egui::Vec2::splat(gtile),
                        ));
                        clip.rect_filled(
                            grect.expand(4.0),
                            egui::CornerRadius::same(11),
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 130),
                        );
                        draw_tile(
                            &clip,
                            &mut self.library,
                            &self.app_settings.carousel_lists,
                            grect,
                            is_list,
                            idx,
                            count,
                            s * 0.74,
                        );
                    }
                }

                let mut grab_req: Option<usize> = None;
                let mut menu_req: Option<usize> = None;
                if !grabbing && !block {
                    for (pos, &si) in vis.iter().enumerate() {
                        let (is_list, list_idx, _) = slots[si];
                        let rect = sr(egui::Rect::from_min_size(
                            egui::pos2(slot_x(pos), row_cy - tile / 2.0),
                            egui::Vec2::splat(tile),
                        ));
                        let resp = ui.allocate_rect(rect, egui::Sense::click());
                        if resp.clicked() && ui.rect_contains_pointer(rect) {
                            grab_req = Some(si);
                        }
                        if is_list && resp.secondary_clicked() {
                            menu_req = Some(list_idx);
                            self.cs_selected = si;
                        }
                    }
                    if let Some(si) = grab_req {
                        self.cs_focus_grid = true;
                        self.cs_selected = si;
                        self.cs_build_grab = Some(si);
                        self.cs_build_insert = si;
                        self.cs_grab_lift = 0.0;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                    if self.cs_focus_grid && x_edge {
                        if let Some(&si) = vis.get(self.cs_selected.min(m.saturating_sub(1))) {
                            if slots[si].0 {
                                menu_req = Some(slots[si].1);
                            }
                        }
                    }
                    if let Some(li) = menu_req {
                        self.cs_list_menu = Some(li);
                        self.cs_list_menu_sel = 0;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                    }
                } else if grabbing && !block {
                    for pos in 0..m {
                        let rect = sr(egui::Rect::from_min_size(
                            egui::pos2(slot_x(pos), row_cy - tile / 2.0),
                            egui::Vec2::splat(tile),
                        ));
                        if ui.rect_contains_pointer(rect)
                            && ui.allocate_rect(rect, egui::Sense::click()).clicked()
                        {
                            self.cs_build_insert = pos;
                            commit = true;
                        }
                    }
                }

                if commit {
                    if let Some(g) = self.cs_build_grab.take() {
                        let ins = self.cs_build_insert.min(m);
                        let mut ord = order.clone();
                        let item = ord.remove(g);
                        ord.insert(ins.min(ord.len()), item);
                        self.app_settings.carousel_order = ord;
                        let _ = self.app_settings.save();
                        self.cs_selected = ins.min(n - 1);
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                }

                p.text(
                    sp(egui::pos2(content.min.x, content.max.y - 6.0 * s)),
                    egui::Align2::LEFT_BOTTOM,
                    if grabbing {
                        "Placing \u{2014} move to a slot, [A] to drop, [B] to cancel"
                    } else {
                        "[A] Grab \u{00B7} [X] Edit a list"
                    },
                    egui::FontId::proportional(14.0 * s * sf),
                    muted,
                );
            }
        }

        if let Some(li) = self.cs_list_menu {
            if li >= self.app_settings.carousel_lists.len() {
                self.cs_list_menu = None;
            } else {
                p.rect_filled(
                    full,
                    egui::CornerRadius::ZERO,
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
                );
                let mw = 320.0 * s;
                let rowh = 54.0 * s;
                let mh = rowh * 2.0 + 66.0 * s;
                let mc = full.center();
                let mr = sr(egui::Rect::from_center_size(mc, egui::vec2(mw, mh)));
                p.rect_filled(mr, egui::CornerRadius::from(14.0 * s), panel);
                p.rect_stroke(
                    mr,
                    egui::CornerRadius::from(14.0 * s),
                    egui::Stroke::new(1.5_f32, border),
                    egui::StrokeKind::Outside,
                );
                let name = self.app_settings.carousel_lists[li].name.clone();
                p.text(
                    sp(egui::pos2(mc.x, mc.y - mh / 2.0 + 26.0 * s)),
                    egui::Align2::CENTER_CENTER,
                    &name,
                    egui::FontId::proportional(17.0 * s * sf),
                    text,
                );
                let red = egui::Color32::from_rgb(0xE0, 0x5A, 0x5A);
                let items = ["Rename List", "Delete List"];
                let base_y = mc.y - mh / 2.0 + 52.0 * s;
                if raw_nu && self.cs_list_menu_sel > 0 {
                    self.cs_list_menu_sel -= 1;
                    crate::ui_audio::play_move();
                }
                if raw_nd && self.cs_list_menu_sel < 1 {
                    self.cs_list_menu_sel += 1;
                    crate::ui_audio::play_move();
                }
                let mut choose: Option<usize> = None;
                for (i, label) in items.iter().enumerate() {
                    let rb = egui::Rect::from_min_size(
                        egui::pos2(mc.x - mw / 2.0 + 14.0 * s, base_y + i as f32 * rowh),
                        egui::vec2(mw - 28.0 * s, rowh - 8.0 * s),
                    );
                    let r = sr(rb);
                    let selrow = self.cs_list_menu_sel == i;
                    let accent_row = if i == 1 { red } else { accent };
                    if selrow {
                        p.rect_filled(r, egui::CornerRadius::from(9.0 * s), sel);
                        p.rect_stroke(
                            r,
                            egui::CornerRadius::from(9.0 * s),
                            egui::Stroke::new(2.0_f32, accent_row),
                            egui::StrokeKind::Outside,
                        );
                    } else if ui.rect_contains_pointer(r) {
                        p.rect_filled(r, egui::CornerRadius::from(9.0 * s), hover);
                    }
                    p.text(
                        sp(rb.center()),
                        egui::Align2::CENTER_CENTER,
                        *label,
                        egui::FontId::proportional(17.0 * s * sf),
                        if i == 1 { red } else { text },
                    );
                    let resp = ui.allocate_rect(r, egui::Sense::click());
                    if ui.rect_contains_pointer(r) {
                        self.cs_list_menu_sel = i;
                    }
                    if resp.clicked() {
                        choose = Some(i);
                    }
                }
                if raw_a {
                    choose = Some(self.cs_list_menu_sel);
                }
                if let Some(c) = choose {
                    self.cs_list_menu = None;
                    if c == 0 {
                        self.cs_creating = true;
                        self.cs_rename_idx = Some(li);
                        self.cs_dialog_row = 0;
                        self.cs_name_editing = false;
                        self.cs_new_name = self.app_settings.carousel_lists[li].name.clone();
                        self.cs_new_align = self.app_settings.carousel_lists[li].align;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    } else {
                        let nm = self.app_settings.carousel_lists[li].name.clone();
                        self.confirm = Some(ConfirmDialog {
                            title: "Delete List".into(),
                            body: format!("Delete the list \u{201C}{}\u{201D}? The games themselves are not removed.", nm),
                            confirm_label: "Delete".into(),
                            selected: 0,
                            kind: ConfirmKind::DeleteList(li),
                        });
                        self.modal_hold = true;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                    }
                } else if raw_b {
                    self.cs_list_menu = None;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        }

        if self.cs_creating {
            p.rect_filled(
                full,
                egui::CornerRadius::ZERO,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
            );
            let dw = 580.0 * s;
            let dh = 384.0 * s;
            let cx = full.center().x;
            let top = full.center().y - dh / 2.0;
            let dr = sr(egui::Rect::from_center_size(
                full.center(),
                egui::vec2(dw, dh),
            ));
            p.rect_filled(dr, egui::CornerRadius::from(16.0 * s), panel);
            p.rect_stroke(
                dr,
                egui::CornerRadius::from(16.0 * s),
                egui::Stroke::new(1.5_f32, border),
                egui::StrokeKind::Outside,
            );
            p.text(
                sp(egui::pos2(cx, top + 34.0 * s)),
                egui::Align2::CENTER_CENTER,
                if self.cs_rename_idx.is_some() {
                    "Rename List"
                } else {
                    "Create List"
                },
                egui::FontId::proportional(24.0 * s * sf),
                text,
            );

            let field_w = dw - 88.0 * s;
            let field_h = 70.0 * s;
            let name_base = egui::Rect::from_min_size(
                egui::pos2(cx - field_w / 2.0, top + 78.0 * s),
                egui::vec2(field_w, field_h),
            );
            let align_base = egui::Rect::from_min_size(
                egui::pos2(cx - field_w / 2.0, top + 78.0 * s + field_h + 18.0 * s),
                egui::vec2(field_w, field_h),
            );
            let btn_w = (field_w - 16.0 * s) / 2.0;
            let create_base = egui::Rect::from_min_size(
                egui::pos2(cx - field_w / 2.0, top + dh - 74.0 * s),
                egui::vec2(btn_w, 50.0 * s),
            );
            let cancel_base = egui::Rect::from_min_size(
                egui::pos2(cx - field_w / 2.0 + btn_w + 16.0 * s, top + dh - 74.0 * s),
                egui::vec2(btn_w, 50.0 * s),
            );

            let mut do_create = false;
            let mut do_cancel = false;

            if self.cs_name_editing {
            } else {
                if nu && self.cs_dialog_row > 0 {
                    self.cs_dialog_row -= 1;
                    crate::ui_audio::play_move();
                }
                if nd && self.cs_dialog_row < 3 {
                    self.cs_dialog_row += 1;
                    crate::ui_audio::play_move();
                }
                if self.cs_dialog_row == 1 {
                    if nl {
                        self.cs_new_align = self.cs_new_align.prev();
                        crate::ui_audio::play_move();
                    }
                    if nr {
                        self.cs_new_align = self.cs_new_align.next();
                        crate::ui_audio::play_move();
                    }
                }
                if a_edge {
                    match self.cs_dialog_row {
                        0 => {
                            if cs_gp_a_edge {
                                self.open_vkeyboard(VkTarget::ListName);
                            } else {
                                self.cs_name_editing = true;
                            }
                        }
                        1 => {
                            self.cs_new_align = self.cs_new_align.next();
                            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                        }
                        2 => do_create = true,
                        _ => do_cancel = true,
                    }
                }
                if b_edge {
                    do_cancel = true;
                }
            }

            let value_font = egui::FontId::proportional(18.0 * s * sf);
            let name_val_w = ui.fonts_mut(|f| {
                f.layout_no_wrap(self.cs_new_name.clone(), value_font.clone(), text)
                    .size()
                    .x
            });
            let caret_on = self.cs_name_editing && (t * 1.6).fract() < 0.5;
            let draw_field = |p: &egui::Painter,
                              base: egui::Rect,
                              row: usize,
                              label: &str,
                              value: &str,
                              is_align: bool,
                              editing: bool| {
                let r = sr(base);
                let selected = self.cs_dialog_row == row;
                p.rect_filled(r, egui::CornerRadius::from(10.0 * s), sel);
                p.rect_stroke(
                    r,
                    egui::CornerRadius::from(10.0 * s),
                    egui::Stroke::new(
                        if selected { 2.0_f32 } else { 1.0_f32 },
                        if selected { accent } else { border },
                    ),
                    egui::StrokeKind::Outside,
                );
                p.text(
                    sp(egui::pos2(base.min.x + 18.0 * s, base.min.y + 14.0 * s)),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(12.0 * s * sf),
                    muted,
                );
                let val_y = base.min.y + 38.0 * s;
                p.text(
                    sp(egui::pos2(base.min.x + 18.0 * s, val_y)),
                    egui::Align2::LEFT_TOP,
                    value,
                    egui::FontId::proportional(18.0 * s * sf),
                    text,
                );
                if editing && caret_on {
                    let cx0 = base.min.x + 18.0 * s + name_val_w + 2.0 * s;
                    let cr = egui::Rect::from_min_max(
                        egui::pos2(cx0, val_y + 2.0 * s),
                        egui::pos2(cx0 + 2.0 * s, val_y + 20.0 * s),
                    );
                    p.rect_filled(sr(cr), egui::CornerRadius::ZERO, text);
                }
                if is_align {
                    p.text(
                        sp(egui::pos2(base.max.x - 40.0 * s, base.center().y)),
                        egui::Align2::CENTER_CENTER,
                        "\u{2039}",
                        egui::FontId::proportional(22.0 * s * sf),
                        muted,
                    );
                    p.text(
                        sp(egui::pos2(base.max.x - 16.0 * s, base.center().y)),
                        egui::Align2::CENTER_CENTER,
                        "\u{203A}",
                        egui::FontId::proportional(22.0 * s * sf),
                        muted,
                    );
                }
            };
            draw_field(&p, name_base, 0, "List Name", "", false, false);
            draw_field(
                &p,
                align_base,
                1,
                "List Alignment",
                self.cs_new_align.label(),
                true,
                false,
            );
            let nm_tp = sp(egui::pos2(
                name_base.min.x + 18.0 * s,
                name_base.min.y + 46.0 * s,
            ));
            let nm_sel =
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 90);
            let nm_events = ui.input(|i| i.events.clone());
            let nfr = text_field(
                &p,
                ui,
                sr(name_base),
                nm_tp.x,
                nm_tp.y,
                &mut self.cs_new_name,
                &mut self.cs_name_caret,
                &mut self.cs_name_anchor,
                value_font.clone(),
                text,
                muted,
                nm_sel,
                "",
                false,
                28,
                self.cs_name_editing,
                caret_on,
                if self.cs_name_editing {
                    &nm_events
                } else {
                    &[]
                },
            );
            if nfr.clicked {
                self.cs_dialog_row = 0;
                self.cs_name_editing = true;
            }
            if nfr.secondary_clicked {
                self.cs_dialog_row = 0;
                self.cs_name_editing = true;
                let c = clipboard_text();
                for ch in c.chars() {
                    if !ch.is_control() && self.cs_new_name.chars().count() < 28 {
                        self.cs_new_name.push(ch);
                    }
                }
            }
            if self.cs_name_editing && (nfr.commit || nfr.cancel) {
                self.cs_name_editing = false;
            }

            let create_label = if self.cs_rename_idx.is_some() {
                "Save"
            } else {
                "Create"
            };
            for (row, base, lbl, fill) in [
                (2usize, create_base, create_label, true),
                (3usize, cancel_base, "Cancel", false),
            ] {
                let r = sr(base);
                let selected = self.cs_dialog_row == row;
                if fill {
                    p.rect_filled(r, egui::CornerRadius::from(11.0 * s), accent);
                    p.text(
                        r.center(),
                        egui::Align2::CENTER_CENTER,
                        lbl,
                        egui::FontId::proportional(18.0 * s * sf),
                        egui::Color32::WHITE,
                    );
                } else {
                    p.rect_filled(r, egui::CornerRadius::from(11.0 * s), sel);
                    p.text(
                        r.center(),
                        egui::Align2::CENTER_CENTER,
                        lbl,
                        egui::FontId::proportional(18.0 * s * sf),
                        text,
                    );
                }
                if selected {
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(11.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                }
            }

            if ui
                .allocate_rect(sr(align_base), egui::Sense::click())
                .clicked()
            {
                self.cs_dialog_row = 1;
                self.cs_new_align = self.cs_new_align.next();
            }
            if ui
                .allocate_rect(sr(create_base), egui::Sense::click())
                .clicked()
            {
                do_create = true;
            }
            if ui
                .allocate_rect(sr(cancel_base), egui::Sense::click())
                .clicked()
            {
                do_cancel = true;
            }

            if do_create {
                if let Some(li) = self.cs_rename_idx {
                    if li < self.app_settings.carousel_lists.len() {
                        let old = self.app_settings.carousel_lists[li].name.clone();
                        let name = {
                            let nm = self.cs_new_name.trim().to_string();
                            if nm.is_empty() {
                                old.clone()
                            } else {
                                nm
                            }
                        };
                        for e in self.app_settings.carousel_order.iter_mut() {
                            if let crate::app_settings::CarouselRef::List(n) = e {
                                if *n == old {
                                    *n = name.clone();
                                }
                            }
                        }
                        self.app_settings.carousel_lists[li].name = name;
                        self.app_settings.carousel_lists[li].align = self.cs_new_align;
                        let _ = self.app_settings.save();
                    }
                    self.cs_rename_idx = None;
                    self.cs_new_name.clear();
                    self.cs_creating = false;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
                } else {
                    let name = {
                        let nm = self.cs_new_name.trim().to_string();
                        if nm.is_empty() {
                            format!("List {}", self.app_settings.carousel_lists.len() + 1)
                        } else {
                            nm
                        }
                    };
                    let mut games = self.cs_list_picks.clone();
                    match self.cs_new_align {
                        crate::app_settings::ListAlignment::Manual => {}
                        crate::app_settings::ListAlignment::Alphabetical
                        | crate::app_settings::ListAlignment::ReverseAlphabetical => {
                            let title_of = |pp: &std::path::PathBuf| {
                                self.library
                                    .games
                                    .iter()
                                    .find(|g| &g.path == pp)
                                    .map(|g| g.title.to_lowercase())
                                    .unwrap_or_default()
                            };
                            games.sort_by(|a, b| title_of(a).cmp(&title_of(b)));
                            if self.cs_new_align
                                == crate::app_settings::ListAlignment::ReverseAlphabetical
                            {
                                games.reverse();
                            }
                        }
                    }
                    self.app_settings
                        .carousel_lists
                        .push(crate::app_settings::GameList {
                            name,
                            games,
                            align: self.cs_new_align,
                        });
                    let _ = self.app_settings.save();
                    self.cs_list_picks.clear();
                    self.cs_pick_anim.clear();
                    self.cs_new_name.clear();
                    self.cs_creating = false;
                    self.cs_create_btn = 0.0;
                    self.cs_tab = 1;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
                }
            } else if do_cancel {
                self.cs_creating = false;
                self.cs_rename_idx = None;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            }
        } else if b_edge {
            if self.cs_build_grab.is_some() {
                self.cs_build_grab = None;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else if self.cs_focus_grid {
                self.cs_focus_grid = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else {
                self.carousel_settings_open = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            }
        }
        ctx.request_repaint();
    }

    fn theme_accent(&self) -> Color32 {
        match self.app_settings.carousel_theme.color() {
            Some((r, g, b)) => Color32::from_rgb(r, g, b),
            None => {
                let c = self.carousel.theme_color;
                let mx = c.r().max(c.g()).max(c.b());
                let mn = c.r().min(c.g()).min(c.b());
                if (mx as i32 - mn as i32) < 20 && mx < 110 {
                    Color32::from_rgb(0x2F, 0xB4, 0xEF)
                } else {
                    c
                }
            }
        }
    }

    fn open_vkeyboard(&mut self, target: VkTarget) {
        let cur = match target {
            VkTarget::ApiKey => self.app_settings.steamgriddb_key.clone(),
            VkTarget::ListName => self.cs_new_name.clone(),
            VkTarget::Swkbd => self.swkbd_text.clone(),
            VkTarget::None => return,
        };
        let max = match target {
            VkTarget::ApiKey => 80,
            VkTarget::Swkbd => self.swkbd_max,
            _ => 28,
        };
        self.vk_target = target;
        self.vkeyboard.show(&cur, max);
    }

    fn poll_swkbd_request(&mut self) {
        if nexium_core::swkbd_state::take_gui_cancel() {
            if self.vk_target == VkTarget::Swkbd {
                self.vkeyboard.open = false;
            }
            return;
        }
        if self.vkeyboard.active() && self.vk_target != VkTarget::Swkbd {
            return;
        }
        let Some(req) = nexium_core::swkbd_state::take_request() else {
            return;
        };
        self.swkbd_gen = req.generation;
        self.swkbd_text = req.initial_text.clone();
        self.swkbd_max = if req.max_len == 0 {
            500
        } else {
            req.max_len as usize
        };
        self.swkbd_min = req.min_len as usize;
        self.swkbd_title = if !req.header_text.is_empty() {
            req.header_text.clone()
        } else if !req.sub_text.is_empty() {
            req.sub_text.clone()
        } else {
            req.guide_text.clone()
        };
        self.swkbd_password = req.password;
        self.vk_target = VkTarget::Swkbd;
        let text = self.swkbd_text.clone();
        let title = self.swkbd_title.clone();
        self.vkeyboard.show_titled(&text, self.swkbd_max, &title);
        self.vkeyboard.mask = self.swkbd_password;
    }

    fn drive_vkeyboard(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll_swkbd_request();
        if self.vkeyboard.open {
            let stale = match self.vk_target {
                VkTarget::ApiKey => !self.show_settings,
                VkTarget::ListName => !self.cs_creating,
                VkTarget::Swkbd => self.emulation_handle.is_none(),
                VkTarget::None => true,
            };
            if stale {
                if self.vk_target == VkTarget::Swkbd {
                    nexium_core::swkbd_state::post_response(
                        nexium_core::swkbd_state::SwkbdResponse {
                            generation: self.swkbd_gen,
                            accepted: false,
                            text: String::new(),
                        },
                    );
                }
                self.vkeyboard.open = false;
            }
        }
        if self.vk_target == VkTarget::None && !self.vkeyboard.active() {
            return;
        }
        let accent = self.theme_accent();
        let light = self.app_settings.light_mode;
        let res = match self.vk_target {
            VkTarget::ApiKey => {
                let before = self.app_settings.steamgriddb_key.clone();
                let r = self.vkeyboard.update(
                    ctx,
                    ui,
                    &mut self.app_settings.steamgriddb_key,
                    &self.last_input,
                    accent,
                    light,
                );
                if self.app_settings.steamgriddb_key != before {
                    let _ = self.app_settings.save();
                }
                r
            }
            VkTarget::ListName => self.vkeyboard.update(
                ctx,
                ui,
                &mut self.cs_new_name,
                &self.last_input,
                accent,
                light,
            ),
            VkTarget::Swkbd => {
                let r = self.vkeyboard.update(
                    ctx,
                    ui,
                    &mut self.swkbd_text,
                    &self.last_input,
                    accent,
                    light,
                );
                match r {
                    crate::vkeyboard::VkResult::Accept => {
                        if self.swkbd_text.chars().count() < self.swkbd_min {
                            let text = self.swkbd_text.clone();
                            let title = self.swkbd_title.clone();
                            self.vkeyboard.show_titled(&text, self.swkbd_max, &title);
                            self.vkeyboard.mask = self.swkbd_password;
                        } else {
                            nexium_core::swkbd_state::post_response(
                                nexium_core::swkbd_state::SwkbdResponse {
                                    generation: self.swkbd_gen,
                                    accepted: true,
                                    text: self.swkbd_text.clone(),
                                },
                            );
                        }
                    }
                    crate::vkeyboard::VkResult::Cancel => {
                        nexium_core::swkbd_state::post_response(
                            nexium_core::swkbd_state::SwkbdResponse {
                                generation: self.swkbd_gen,
                                accepted: false,
                                text: String::new(),
                            },
                        );
                    }
                    crate::vkeyboard::VkResult::None => {}
                }
                r
            }
            VkTarget::None => {
                let mut discard = String::new();
                self.vkeyboard
                    .update(ctx, ui, &mut discard, &self.last_input, accent, light)
            }
        };
        let _ = res;
        if !self.vkeyboard.active() {
            self.vk_target = VkTarget::None;
        }
    }

    fn update_status_display(&self, t: f32) -> (String, Color32, bool) {
        let light = self.app_settings.light_mode;
        let grey = if light {
            Color32::from_rgb(0x88, 0x88, 0x92)
        } else {
            Color32::from_rgb(0x70, 0x70, 0x7A)
        };
        let green = Color32::from_rgb(0x35, 0xD0, 0x6A);
        let red = if light {
            Color32::from_rgb(0xC0, 0x3A, 0x3A)
        } else {
            Color32::from_rgb(0xE0, 0x6A, 0x6A)
        };
        if !crate::updater::Updater::supported() {
            return ("Windows: use GitHub releases".to_string(), grey, false);
        }
        if self.updater.is_downloading() {
            let pct = (self.updater.progress() * 100.0) as i32;
            return (format!("Downloading  {}%", pct), green, false);
        }
        match self.updater.status() {
            crate::updater::Status::Idle | crate::updater::Status::Checking => {
                let dots = ((t * 2.0) as usize % 3) + 1;
                (format!("Checking{}", ".".repeat(dots)), grey, false)
            }
            crate::updater::Status::UpToDate => ("Fully up-to-date.".to_string(), grey, false),
            crate::updater::Status::Available(_) => ("Update Available!".to_string(), green, true),
            crate::updater::Status::Error(_) => ("Update check failed".to_string(), red, false),
        }
    }

    fn drive_updater(&mut self, ctx: &egui::Context) {
        self.updater.check();
        if self.updater.is_downloading() {
            ctx.request_repaint();
        }
        if let Some(ok) = self.updater.download_result() {
            if ok
                && !self.update_restart_shown
                && self.confirm.is_none()
                && self.teardown_at.is_none()
            {
                self.update_restart_shown = true;
                self.confirm = Some(ConfirmDialog {
                    title: "Software Update".into(),
                    body: "Download Complete!\nWould you like to restart & open the latest release now?".into(),
                    confirm_label: "Yes".into(),
                    selected: 1,
                    kind: ConfirmKind::UpdateRestart,
                });
            }
        }
    }

    fn open_update_confirm(&mut self) {
        if self.confirm.is_some() || self.updater.available_release().is_none() {
            return;
        }
        let exe_path = std::env::current_exe()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| "the executable location".to_string());
        let parent_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().to_string()))
            .unwrap_or_else(|| "app folder".to_string());

        let body = format!(
            "You are about to install the latest Linux build.\n\n\
             This will replace the active binary at:\n{}\n\n\
             A backup of the current build will be saved in:\n{}/backups/\n\n\
             Would you like to proceed?",
            exe_path, parent_dir
        );

        self.confirm = Some(ConfirmDialog {
            title: "Software Update".into(),
            body,
            confirm_label: "Yes".into(),
            selected: 1,
            kind: ConfirmKind::UpdateInstall,
        });
        self.modal_hold = true;
    }

    fn update_preferences(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let target = if self.show_settings { 1.0 } else { 0.0 };
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.prefs_anim += (target - self.prefs_anim) * (dt * 12.0).min(1.0);
        if self.prefs_anim < 0.004 {
            self.prefs_key_editing = false;
            self.prefs_key_menu = false;
            return;
        }
        let ease = {
            let a = self.prefs_anim.clamp(0.0, 1.0);
            a * a * (3.0 - 2.0 * a)
        };
        let t = ctx.input(|i| i.time) as f32;
        let full = ctx.viewport_rect();
        let s = (full.height() / 820.0).clamp(1.0, 2.4);
        let backdrop_theme = self.app_settings.backdrop_theme;
        let lightish = self.app_settings.light_mode;
        let pick = |dark: egui::Color32, lite: egui::Color32| if lightish { lite } else { dark };
        let text = pick(
            egui::Color32::from_rgb(0xEC, 0xEC, 0xF0),
            egui::Color32::from_rgb(0x1E, 0x1E, 0x28),
        );
        let muted = pick(
            egui::Color32::from_rgb(0x8A, 0x8A, 0x98),
            egui::Color32::from_rgb(0x60, 0x60, 0x6A),
        );
        let panel = pick(
            egui::Color32::from_rgb(0x16, 0x16, 0x1E),
            egui::Color32::from_rgb(0xFF, 0xFF, 0xFF),
        );
        let border = pick(
            egui::Color32::from_rgb(0x30, 0x30, 0x3C),
            egui::Color32::from_rgb(0xC6, 0xC6, 0xD0),
        );
        let sel = pick(
            egui::Color32::from_rgb(0x1E, 0x1E, 0x28),
            egui::Color32::from_rgb(0xDD, 0xDD, 0xE6),
        );
        let hover = pick(
            egui::Color32::from_rgb(0x18, 0x18, 0x22),
            egui::Color32::from_rgb(0xEA, 0xEA, 0xF0),
        );
        let field_bg = pick(
            egui::Color32::from_rgb(0x20, 0x20, 0x2A),
            egui::Color32::from_rgb(0xE4, 0xE4, 0xEC),
        );
        let accent = self.theme_accent();

        let center = full.center();
        let sf = 0.965 + 0.035 * ease;
        let sp = |p: egui::Pos2| center + (p - center) * sf;
        let sr = |r: egui::Rect| {
            egui::Rect::from_center_size(center + (r.center() - center) * sf, r.size() * sf)
        };

        let mut p = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("preferences"),
        ));
        p.set_opacity(ease);
        crate::carousel::draw_backdrop(
            &p,
            full,
            accent,
            t,
            backdrop_theme,
            1.0,
            if lightish { 1.0 } else { 0.0 },
            None,
        );
        let scrim = if lightish {
            egui::Color32::from_rgba_unmultiplied(0xFA, 0xFA, 0xFC, 235)
        } else {
            egui::Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 140)
        };
        p.rect_filled(full, egui::CornerRadius::ZERO, scrim);

        let mx = full.width() * 0.055;
        let header_font = 34.0 * s;
        p.text(
            sp(egui::pos2(full.min.x + mx, full.min.y + 40.0 * s)),
            egui::Align2::LEFT_TOP,
            "Preferences",
            egui::FontId::proportional(header_font * sf),
            text,
        );
        let header_y = full.min.y + 40.0 * s + header_font + 20.0 * s;
        p.line_segment(
            [
                sp(egui::pos2(full.min.x + mx, header_y)),
                sp(egui::pos2(full.max.x - mx, header_y)),
            ],
            egui::Stroke::new(1.0_f32, border),
        );

        use crate::controller_config::SwitchButton;
        let now = ctx.input(|i| i.time);
        let li = self.last_input;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let gp_a_edge = gp_a && !self.prefs_ab_held;
        let a_edge =
            ctx.input(|i| i.key_pressed(egui::Key::Enter)) || (gp_a && !self.prefs_ab_held);
        let b_edge =
            ctx.input(|i| i.key_pressed(egui::Key::Escape)) || (gp_b && !self.prefs_ab_held);
        self.prefs_ab_held = gp_a || gp_b;
        let (a_edge, b_edge) = if self.prefs_rebind_cool {
            self.prefs_rebind_cool = false;
            (false, false)
        } else {
            (a_edge, b_edge)
        };
        let (mut nu, mut nd, mut nl, mut nr) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        const INITIAL_DELAY: f64 = 0.70;
        const REPEAT_RATE: f64 = 0.14;
        let mut nav_stick = false;
        if li.connected {
            let (cur_dir, cur_is_dpad): (u8, bool) = if li.is(SwitchButton::DUp) {
                (1, true)
            } else if li.ly() > 0.5 {
                (1, false)
            } else if li.is(SwitchButton::DDown) {
                (2, true)
            } else if li.ly() < -0.5 {
                (2, false)
            } else if li.is(SwitchButton::DLeft) {
                (3, true)
            } else if li.lx() < -0.5 {
                (3, false)
            } else if li.is(SwitchButton::DRight) {
                (4, true)
            } else if li.lx() > 0.5 {
                (4, false)
            } else {
                (0, false)
            };
            if cur_dir == 0 {
                self.prefs_nav_held_dir = 0;
                self.prefs_nav_held_since = 0.0;
            } else if cur_dir != self.prefs_nav_held_dir {
                self.prefs_nav_held_dir = cur_dir;
                self.prefs_nav_held_since = now;
                self.prefs_nav_cd = now;
                match cur_dir {
                    1 => nu = true,
                    2 => nd = true,
                    3 => nl = true,
                    _ => nr = true,
                }
                nav_stick = !cur_is_dpad;
            } else {
                let held_for = now - self.prefs_nav_held_since;
                if held_for >= INITIAL_DELAY && now - self.prefs_nav_cd >= REPEAT_RATE {
                    self.prefs_nav_cd = now;
                    match cur_dir {
                        1 => nu = true,
                        2 => nd = true,
                        3 => nl = true,
                        _ => nr = true,
                    }
                    nav_stick = !cur_is_dpad;
                }
            }
        } else {
            self.prefs_nav_held_dir = 0;
            self.prefs_nav_held_since = 0.0;
        }
        let gp_l = li.connected && li.is(SwitchButton::L);
        let gp_r = li.connected && li.is(SwitchButton::R);
        let l_bumper_edge = gp_l && !self.prefs_lr_held;
        let r_bumper_edge = gp_r && !self.prefs_lr_held;
        self.prefs_lr_held = gp_l || gp_r;

        let rebinding_active = self.rebind_in_progress();
        let editing = self.prefs_key_editing;
        let block = self.prefs_key_menu || rebinding_active || self.vkeyboard.open;
        let cancel_edge = b_edge;
        let (a_edge, b_edge, nu, nd, nl, nr, l_bumper_edge, r_bumper_edge) = if editing || block {
            (false, false, false, false, false, false, false, false)
        } else {
            (a_edge, b_edge, nu, nd, nl, nr, l_bumper_edge, r_bumper_edge)
        };
        let mut b_edge = b_edge;

        let side_w = (full.width() * 0.22).max(300.0);
        let side_x = full.min.x + mx;
        let side_top = header_y + 34.0 * s;
        let item_h = 60.0 * s;
        let tabs = [
            "General",
            "Controller",
            "Graphics",
            "Audio",
            "Emulation",
            "Logging",
        ];
        let tab_of = [
            SettingsTab::General,
            SettingsTab::Controller,
            SettingsTab::Graphics,
            SettingsTab::Audio,
            SettingsTab::Emulation,
            SettingsTab::Logging,
        ];
        let cur = tab_of
            .iter()
            .position(|x| *x == self.settings_tab)
            .unwrap_or(0);
        for (i, label) in tabs.iter().enumerate() {
            let base = egui::Rect::from_min_size(
                egui::pos2(side_x, side_top + i as f32 * (item_h + 9.0 * s)),
                egui::Vec2::new(side_w, item_h),
            );
            let r = sr(base);
            let selected = cur == i;
            let ring = if !self.prefs_focus { accent } else { border };
            let rounding = egui::CornerRadius::from(12.0 * s);
            if selected {
                p.rect_filled(r, rounding, sel);
                p.rect_stroke(
                    r,
                    rounding,
                    egui::Stroke::new(1.8_f32, ring),
                    egui::StrokeKind::Outside,
                );
                let bar = sr(egui::Rect::from_min_size(
                    base.min + egui::Vec2::new(6.0 * s, 12.0 * s),
                    egui::Vec2::new(4.0 * s, base.height() - 24.0 * s),
                ));
                p.rect_filled(bar, egui::CornerRadius::from(2.0 * s), ring);
            } else if ui.rect_contains_pointer(r) {
                p.rect_filled(r, rounding, hover);
            }
            p.text(
                sp(egui::pos2(base.min.x + 26.0 * s, base.center().y)),
                egui::Align2::LEFT_CENTER,
                *label,
                egui::FontId::proportional(19.0 * s * sf),
                if selected { text } else { muted },
            );
            if !editing && ui.allocate_rect(r, egui::Sense::click()).clicked() {
                self.settings_tab = tab_of[i];
                self.prefs_focus = false;
                self.prefs_row = 0;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        }
        let right_held = li.connected && (li.is(SwitchButton::DRight) || li.lx() > 0.5);
        if !right_held {
            self.prefs_enter_held = false;
        }

        if !self.prefs_focus && !editing {
            if nu && cur > 0 {
                self.settings_tab = tab_of[cur - 1];
                self.prefs_row = 0;
                crate::ui_audio::play_move();
            }
            if nd && cur + 1 < tabs.len() {
                self.settings_tab = tab_of[cur + 1];
                self.prefs_row = 0;
                crate::ui_audio::play_move();
            }
            if nr {
                self.prefs_focus = true;
                self.prefs_row = 0;
                self.prefs_enter_held = true;
                crate::ui_audio::play_move();
            }
        }
        let nr = nr && !self.prefs_enter_held;

        let footer_y = full.max.y - 60.0 * s;
        p.line_segment(
            [
                sp(egui::pos2(full.min.x + mx, footer_y)),
                sp(egui::pos2(full.max.x - mx, footer_y)),
            ],
            egui::Stroke::new(1.0_f32, border),
        );
        let pad = li.connected;
        let hint = if editing {
            "Type your key    ·    [Enter] Save    ·    [Esc] Cancel"
        } else if !self.prefs_focus {
            if pad {
                "Move    ·    [B] Back"
            } else {
                "[Arrows] Move    ·    [Esc] Back"
            }
        } else if self.settings_tab == SettingsTab::Controller {
            if pad {
                "Move    ·    [L/R] Left/Right Option Columns    ·    [A] Select    ·    [B] Back"
            } else {
                "[Arrows] Move    ·    [Enter] Select    ·    [Esc] Back"
            }
        } else if pad {
            "Move    ·    [A] Select    ·    [B] Back"
        } else {
            "[Arrows] Move    ·    [Enter] Select    ·    [Esc] Back"
        };
        let hint_font = egui::FontId::proportional(14.0 * s * sf);
        let hy = full.max.y - 30.0 * s;
        let hint_w = ui.fonts_mut(|f| {
            f.layout_no_wrap(hint.to_string(), hint_font.clone(), muted)
                .size()
                .x
        });
        p.text(
            sp(egui::pos2(full.max.x - mx, hy)),
            egui::Align2::RIGHT_CENTER,
            hint,
            hint_font,
            muted,
        );
        if pad && !editing {
            let dp_r = 8.0 * s * sf;
            draw_dpad(
                &p,
                sp(egui::pos2(full.max.x - mx - hint_w - dp_r - 8.0 * s, hy)),
                dp_r,
                muted,
            );
        }

        let content = egui::Rect::from_min_max(
            egui::pos2(side_x + side_w + 48.0 * s, side_top),
            egui::pos2(full.max.x - mx, footer_y - 20.0 * s),
        );

        if self.settings_tab != SettingsTab::Controller || self.input_device != InputDevice::Motion {
            self.motion_key_capture = false;
        }
        if self.settings_tab == SettingsTab::General {
            let backends = crate::app_settings::CpuBackend::all();
            let rows: [(&str, String, bool); 3] = [
                (
                    "CPU Backend",
                    self.app_settings.cpu_backend.label().to_string(),
                    true,
                ),
                (
                    "GPU Backend",
                    self.app_settings.gpu_backend.label().to_string(),
                    true,
                ),
                (
                    "Resolution",
                    self.app_settings.resolution_preset.label().to_string(),
                    true,
                ),
            ];
            let n_rows = rows.len();
            let key_idx = n_rows;
            let n = n_rows + 1;
            if !self.prefs_focus {
                self.prefs_row = 0;
            }
            self.prefs_row = self.prefs_row.min(n - 1);
            if self.prefs_focus && !editing && !block {
                if nu && self.prefs_row > 0 {
                    self.prefs_row -= 1;
                    crate::ui_audio::play_move();
                }
                if nd && self.prefs_row + 1 < n {
                    self.prefs_row += 1;
                    crate::ui_audio::play_move();
                }
            }
            let row_h = 60.0 * s;
            let mut y = content.min.y + 4.0 * s;
            let mut act_row = usize::MAX;
            let mut dir = 0i32;
            for (i, (label, value, interactive)) in rows.iter().enumerate() {
                let base = egui::Rect::from_min_size(
                    egui::pos2(content.min.x, y),
                    egui::Vec2::new(content.width(), row_h - 12.0 * s),
                );
                let r = sr(base);
                let selrow = self.prefs_focus && self.prefs_row == i;
                if selrow {
                    p.rect_filled(r, egui::CornerRadius::from(12.0 * s), sel);
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                } else {
                    p.rect_filled(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Color32::from_rgba_unmultiplied(
                            panel.r(),
                            panel.g(),
                            panel.b(),
                            (ease * 90.0) as u8,
                        ),
                    );
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(1.0_f32, border),
                        egui::StrokeKind::Outside,
                    );
                }
                p.text(
                    sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
                    egui::Align2::LEFT_CENTER,
                    *label,
                    egui::FontId::proportional(18.0 * s * sf),
                    text,
                );
                let lx = base.max.x - 210.0 * s;
                let rx = base.max.x - 22.0 * s;
                let vcol = if *interactive && selrow {
                    accent
                } else {
                    muted
                };
                if *interactive {
                    p.text(
                        sp(egui::pos2((lx + rx) * 0.5, base.center().y)),
                        egui::Align2::CENTER_CENTER,
                        value,
                        egui::FontId::proportional(16.0 * s * sf),
                        vcol,
                    );
                    let la = egui::Rect::from_center_size(
                        egui::pos2(lx, base.center().y),
                        egui::Vec2::splat(34.0 * s),
                    );
                    let ra = egui::Rect::from_center_size(
                        egui::pos2(rx, base.center().y),
                        egui::Vec2::splat(34.0 * s),
                    );
                    p.text(
                        sp(la.center()),
                        egui::Align2::CENTER_CENTER,
                        "\u{2039}",
                        egui::FontId::proportional(22.0 * s * sf),
                        if selrow { accent } else { muted },
                    );
                    p.text(
                        sp(ra.center()),
                        egui::Align2::CENTER_CENTER,
                        "\u{203A}",
                        egui::FontId::proportional(22.0 * s * sf),
                        if selrow { accent } else { muted },
                    );
                    if !block && ui.rect_contains_pointer(r) {
                        let rowc = ui.allocate_rect(r, egui::Sense::click()).clicked();
                        let lc = ui.allocate_rect(sr(la), egui::Sense::click()).clicked();
                        let rc = ui.allocate_rect(sr(ra), egui::Sense::click()).clicked();
                        if lc || rc || rowc {
                            self.prefs_focus = true;
                            self.prefs_row = i;
                            act_row = i;
                            dir = if lc { -1 } else { 1 };
                        }
                    }
                } else {
                    p.text(
                        sp(egui::pos2(base.max.x - 22.0 * s, base.center().y)),
                        egui::Align2::RIGHT_CENTER,
                        value,
                        egui::FontId::proportional(16.0 * s * sf),
                        muted,
                    );
                }
                y += row_h;
            }
            if self.prefs_focus
                && self.prefs_row < n_rows
                && rows[self.prefs_row].2
                && act_row == usize::MAX
            {
                if nl {
                    act_row = self.prefs_row;
                    dir = -1;
                } else if nr || a_edge {
                    act_row = self.prefs_row;
                    dir = 1;
                }
            }
            if act_row != usize::MAX {
                match act_row {
                    0 => {
                        let idx = backends
                            .iter()
                            .position(|x| *x == self.app_settings.cpu_backend)
                            .unwrap_or(0);
                        self.app_settings.cpu_backend = backends
                            [((idx as i32 + dir).rem_euclid(backends.len() as i32)) as usize];
                    }
                    1 => {
                        self.app_settings.gpu_backend = if dir >= 0 {
                            self.app_settings.gpu_backend.next()
                        } else {
                            self.app_settings.gpu_backend.prev()
                        };
                    }
                    2 => {
                        self.app_settings.resolution_preset = if dir >= 0 {
                            self.app_settings.resolution_preset.next()
                        } else {
                            self.app_settings.resolution_preset.prev()
                        };
                    }
                    _ => {}
                }
                let _ = self.app_settings.save();
                crate::ui_audio::play_move();
            }
            y += 12.0 * s;
            p.text(
                sp(egui::pos2(content.min.x + 2.0 * s, y)),
                egui::Align2::LEFT_TOP,
                "STEAMGRIDDB",
                egui::FontId::proportional(13.0 * s * sf),
                muted,
            );
            y += 26.0 * s;
            let key_h = 66.0 * s;
            let key_base = egui::Rect::from_min_size(
                egui::pos2(content.min.x, y),
                egui::Vec2::new(content.width(), key_h),
            );
            let key_r = sr(key_base);
            let key_sel = self.prefs_focus && self.prefs_row == key_idx;
            p.rect_filled(key_r, egui::CornerRadius::from(12.0 * s), field_bg);
            p.rect_stroke(
                key_r,
                egui::CornerRadius::from(12.0 * s),
                egui::Stroke::new(
                    if key_sel || editing { 2.0_f32 } else { 1.0_f32 },
                    if key_sel || editing { accent } else { border },
                ),
                egui::StrokeKind::Outside,
            );
            p.text(
                sp(egui::pos2(
                    key_base.min.x + 18.0 * s,
                    key_base.min.y + 14.0 * s,
                )),
                egui::Align2::LEFT_TOP,
                "SteamGridDB API Key",
                egui::FontId::proportional(12.0 * s * sf),
                muted,
            );
            let ktx = key_base.min.x + 18.0 * s;
            let ktmy = key_base.min.y + 46.0 * s;
            let ktp = sp(egui::pos2(ktx, ktmy));
            let kfont = egui::FontId::proportional(18.0 * s * sf);
            let sel_col =
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 90);
            let blink = editing && (t * 1.6).fract() < 0.5;
            let key_events = ui.input(|i| i.events.clone());
            let fr = text_field(
                &p,
                ui,
                key_r,
                ktp.x,
                ktp.y,
                &mut self.app_settings.steamgriddb_key,
                &mut self.prefs_key_caret,
                &mut self.prefs_key_anchor,
                kfont,
                text,
                muted,
                sel_col,
                "Not set",
                true,
                80,
                editing && !block,
                blink,
                if editing && !block { &key_events } else { &[] },
            );
            p.text(
                sp(egui::pos2(content.min.x + 2.0 * s, y + key_h + 12.0 * s)),
                egui::Align2::LEFT_TOP,
                "Get a key at steamgriddb.com/profile/preferences/api",
                egui::FontId::proportional(13.0 * s * sf),
                muted,
            );
            if fr.changed {
                let _ = self.app_settings.save();
            }
            if !block {
                if fr.secondary_clicked {
                    self.prefs_focus = true;
                    self.prefs_row = key_idx;
                    self.prefs_key_editing = true;
                    self.prefs_key_menu = true;
                } else if fr.clicked {
                    self.prefs_focus = true;
                    self.prefs_row = key_idx;
                    self.prefs_key_editing = true;
                } else if self.prefs_focus && self.prefs_row == key_idx && a_edge {
                    if gp_a_edge {
                        self.open_vkeyboard(VkTarget::ApiKey);
                    } else {
                        self.prefs_key_editing = true;
                    }
                }
            }
            if editing && (fr.commit || fr.cancel) {
                self.prefs_key_editing = false;
                let _ = self.app_settings.save();
            }
            if self.prefs_key_menu {
                let items = ["Paste", "Copy", "Clear"];
                let mw = 180.0 * s;
                let ih = 42.0 * s;
                let mtop = key_base.max.y + 6.0 * s;
                let mrect = sr(egui::Rect::from_min_size(
                    egui::pos2(key_base.min.x + 12.0 * s, mtop),
                    egui::vec2(mw, ih * items.len() as f32 + 12.0 * s),
                ));
                p.rect_filled(mrect, egui::CornerRadius::from(10.0 * s), panel);
                p.rect_stroke(
                    mrect,
                    egui::CornerRadius::from(10.0 * s),
                    egui::Stroke::new(1.5_f32, border),
                    egui::StrokeKind::Outside,
                );
                let mut act: Option<usize> = None;
                for (i, label) in items.iter().enumerate() {
                    let ib = egui::Rect::from_min_size(
                        egui::pos2(key_base.min.x + 18.0 * s, mtop + 6.0 * s + i as f32 * ih),
                        egui::vec2(mw - 12.0 * s, ih - 4.0 * s),
                    );
                    let ir = sr(ib);
                    if ui.rect_contains_pointer(ir) {
                        p.rect_filled(ir, egui::CornerRadius::from(8.0 * s), hover);
                        if ui.allocate_rect(ir, egui::Sense::click()).clicked() {
                            act = Some(i);
                        }
                    }
                    p.text(
                        sp(egui::pos2(ib.min.x + 12.0 * s, ib.center().y)),
                        egui::Align2::LEFT_CENTER,
                        *label,
                        egui::FontId::proportional(16.0 * s * sf),
                        text,
                    );
                }
                let outside =
                    ui.input(|i| i.pointer.primary_clicked()) && !ui.rect_contains_pointer(mrect);
                if let Some(a) = act {
                    match a {
                        0 => {
                            let c = clipboard_text();
                            for ch in c.chars() {
                                if !ch.is_control()
                                    && self.app_settings.steamgriddb_key.chars().count() < 80
                                {
                                    self.app_settings.steamgriddb_key.push(ch);
                                }
                            }
                            let _ = self.app_settings.save();
                        }
                        1 => {
                            let _ = arboard::Clipboard::new().and_then(|mut c| {
                                c.set_text(self.app_settings.steamgriddb_key.clone())
                            });
                        }
                        _ => {
                            self.app_settings.steamgriddb_key.clear();
                            let _ = self.app_settings.save();
                        }
                    }
                    self.prefs_key_menu = false;
                } else if outside {
                    self.prefs_key_menu = false;
                }
            }
        } else if self.settings_tab == SettingsTab::Graphics {
            let scale = self.app_settings.output_scale.clamp(1, 3);
            let on = |b: bool| {
                if b {
                    "On".to_string()
                } else {
                    "Off".to_string()
                }
            };
            let rows: [(&str, String); 12] = [
                ("GPU", self.app_settings.gpu_device_label()),
                ("Aspect Mode", self.app_settings.aspect.label().to_string()),
                ("Output Scale", format!("{}x", scale)),
                (
                    "Texture Filter",
                    self.app_settings.filter.label().to_string(),
                ),
                ("FSR Sharpness", format!("{}%", self.app_settings.fsr_sharpness)),
                ("High-DPI Aware", on(self.app_settings.dpi_aware)),
                ("V-Sync", on(self.app_settings.vsync)),
                ("Async Shaders", on(self.app_settings.async_shaders)),
                ("Fast GPU Time", on(self.app_settings.fast_gpu_time)),
                ("Force Max Clocks", on(self.app_settings.force_max_clocks)),
                (
                    "GPU Accuracy",
                    if self.app_settings.normal_gpu_accuracy { "Normal" } else { "High" }.to_string(),
                ),
                ("Depth Share", on(self.app_settings.depth_share)),
            ];
            let n = rows.len();
            if !self.prefs_focus {
                self.prefs_row = 0;
            }
            self.prefs_row = self.prefs_row.min(n - 1);
            if self.prefs_focus {
                if nu && self.prefs_row > 0 {
                    self.prefs_row -= 1;
                    crate::ui_audio::play_move();
                }
                if nd && self.prefs_row + 1 < n {
                    self.prefs_row += 1;
                    crate::ui_audio::play_move();
                }
            }
            let row_h = ((content.height() - 52.0 * s) / n as f32).min(62.0 * s);
            let mut y = content.min.y + 4.0 * s;
            let mut dir = 0i32;
            for (i, (label, value)) in rows.iter().enumerate() {
                let base = egui::Rect::from_min_size(
                    egui::pos2(content.min.x, y),
                    egui::Vec2::new(content.width(), row_h - 12.0 * s),
                );
                let r = sr(base);
                let selrow = self.prefs_focus && i == self.prefs_row;
                if selrow {
                    p.rect_filled(r, egui::CornerRadius::from(12.0 * s), sel);
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                } else {
                    p.rect_filled(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Color32::from_rgba_unmultiplied(
                            panel.r(),
                            panel.g(),
                            panel.b(),
                            (ease * 90.0) as u8,
                        ),
                    );
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(1.0_f32, border),
                        egui::StrokeKind::Outside,
                    );
                }
                p.text(
                    sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
                    egui::Align2::LEFT_CENTER,
                    *label,
                    egui::FontId::proportional(18.0 * s * sf),
                    text,
                );
                let val_col = if selrow { accent } else { muted };
                let lx = if i == 0 {
                    base.min.x + 180.0 * s
                } else {
                    base.max.x - 150.0 * s
                };
                let rx = base.max.x - 22.0 * s;
                let value_clip = sr(egui::Rect::from_min_max(
                    egui::pos2(lx + 20.0 * s, base.min.y),
                    egui::pos2(rx - 20.0 * s, base.max.y),
                ));
                let value_painter = if i == 0 {
                    p.with_clip_rect(value_clip)
                } else {
                    p.clone()
                };
                value_painter.text(
                    sp(egui::pos2((lx + rx) * 0.5, base.center().y)),
                    egui::Align2::CENTER_CENTER,
                    value,
                    egui::FontId::proportional(16.0 * s * sf),
                    val_col,
                );
                let la = egui::Rect::from_center_size(
                    egui::pos2(lx, base.center().y),
                    egui::Vec2::splat(34.0 * s),
                );
                let ra = egui::Rect::from_center_size(
                    egui::pos2(rx, base.center().y),
                    egui::Vec2::splat(34.0 * s),
                );
                p.text(
                    sp(la.center()),
                    egui::Align2::CENTER_CENTER,
                    "\u{2039}",
                    egui::FontId::proportional(22.0 * s * sf),
                    if selrow { accent } else { muted },
                );
                p.text(
                    sp(ra.center()),
                    egui::Align2::CENTER_CENTER,
                    "\u{203A}",
                    egui::FontId::proportional(22.0 * s * sf),
                    if selrow { accent } else { muted },
                );
                if ui.rect_contains_pointer(r) {
                    let rowc = ui.allocate_rect(r, egui::Sense::click()).clicked();
                    let lc = ui.allocate_rect(sr(la), egui::Sense::click()).clicked();
                    let rc = ui.allocate_rect(sr(ra), egui::Sense::click()).clicked();
                    if lc || rc || rowc {
                        self.prefs_focus = true;
                        self.prefs_row = i;
                        if lc {
                            dir = -1;
                        } else if rc {
                            dir = 1;
                        } else {
                            dir = 1;
                        }
                    }
                }
                y += row_h;
            }
            if self.prefs_focus && dir == 0 {
                if nl {
                    dir = -1;
                } else if nr || a_edge {
                    dir = 1;
                }
            }
            if dir != 0 {
                match self.prefs_row {
                    0 => self.app_settings.cycle_gpu_device(dir),
                    1 => {
                        let all = crate::app_settings::AspectMode::all();
                        let idx = all
                            .iter()
                            .position(|x| *x == self.app_settings.aspect)
                            .unwrap_or(0);
                        self.app_settings.aspect =
                            all[((idx as i32 + dir).rem_euclid(all.len() as i32)) as usize];
                    }
                    2 => {
                        self.app_settings.output_scale =
                            (((scale as i32 - 1 + dir).rem_euclid(3)) + 1) as u8;
                    }
                    3 => {
                        let all = crate::app_settings::FilterMode::all();
                        let idx = all
                            .iter()
                            .position(|x| *x == self.app_settings.filter)
                            .unwrap_or(0);
                        self.app_settings.filter =
                            all[((idx as i32 + dir).rem_euclid(all.len() as i32)) as usize];
                    }
                    4 => {
                        self.app_settings.fsr_sharpness =
                            (i32::from(self.app_settings.fsr_sharpness) + dir * 5).clamp(0, 100) as u8;
                    }
                    5 => self.app_settings.dpi_aware = !self.app_settings.dpi_aware,
                    6 => self.app_settings.vsync = !self.app_settings.vsync,
                    7 => {
                        self.app_settings.async_shaders = !self.app_settings.async_shaders;
                        nexium_common::async_compile::set_enabled(self.app_settings.async_shaders);
                    }
                    8 => {
                        self.app_settings.fast_gpu_time = !self.app_settings.fast_gpu_time;
                        nexium_common::fast_gpu_time::set_enabled(self.app_settings.fast_gpu_time);
                    }
                    9 => {
                        self.app_settings.force_max_clocks = !self.app_settings.force_max_clocks;
                        nexium_common::force_max_clocks::set_enabled(
                            self.app_settings.force_max_clocks,
                        );
                    }
                    10 => {
                        self.app_settings.normal_gpu_accuracy = !self.app_settings.normal_gpu_accuracy;
                        nexium_common::gpu_accuracy::set_normal(self.app_settings.normal_gpu_accuracy);
                    }
                    11 => {
                        self.app_settings.depth_share = !self.app_settings.depth_share;
                        nexium_common::depth_share::set_enabled(self.app_settings.depth_share);
                    }
                    _ => {}
                }
                let _ = self.app_settings.save();
                crate::ui_audio::play_move();
            }
            let (gpu_status, restart_pending) = gpu_selection_status(
                self.app_settings.gpu_device.as_deref(),
                self.startup_gpu_device.as_deref(),
                &self.active_gpu_name(),
            );
            p.text(
                sp(egui::pos2(content.min.x + 2.0 * s, y + 2.0 * s)),
                egui::Align2::LEFT_TOP,
                gpu_status,
                egui::FontId::proportional(13.0 * s * sf),
                if restart_pending { AMBER } else { muted },
            );
        } else if self.settings_tab == SettingsTab::Controller {
            let (nu, nd, nl, nr) = if nav_stick {
                (false, false, false, false)
            } else {
                (nu, nd, nl, nr)
            };
            let events = ui.input(|i| i.events.clone());
            if let Some(btn) = self.rebinding {
                let mut done: Option<String> = None;
                let mut cancel = false;
                for ev in &events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = ev
                    {
                        if *key == egui::Key::Escape {
                            cancel = true;
                        } else {
                            done = Some(format!("{:?}", key));
                        }
                    }
                }
                if cancel {
                    self.rebinding = None;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                } else if let Some(k) = done {
                    self.controller_config.set_binding(btn, k);
                    self.rebinding = None;
                    let _ = self.controller_config.save();
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
            } else if self.rebinding_pad.is_some() {
                let esc = events.iter().any(|ev| {
                    matches!(
                        ev,
                        egui::Event::Key {
                            key: egui::Key::Escape,
                            pressed: true,
                            ..
                        }
                    )
                });
                if esc {
                    self.rebinding_pad = None;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            } else if self.motion_key_capture {
                let outcome = if cancel_edge {
                    Some(MotionKeyCapture::Cancel)
                } else {
                    recenter_key_press(&events, &self.controller_config)
                };
                match outcome {
                    Some(MotionKeyCapture::Cancel) => {
                        self.motion_key_capture = false;
                        self.prefs_rebind_cool = true;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                    }
                    Some(MotionKeyCapture::Set(key)) => {
                        self.app_settings.motion_recenter_key = key;
                        self.motion_key_capture = false;
                        self.prefs_rebind_cool = true;
                        let _ = self.app_settings.save();
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                    Some(MotionKeyCapture::Refuse(notice)) => {
                        self.motion_key_notice = Some(notice);
                        crate::ui_audio::play(crate::ui_audio::Sfx::Error);
                    }
                    None => {}
                }
            }

            let kb = self.input_device == InputDevice::Keyboard;
            let motion_page = self.input_device == InputDevice::Motion;
            let vibration_page = self.input_device == InputDevice::Vibration;
            let binds: Vec<SwitchButton> = if motion_page || vibration_page {
                Vec::new()
            } else if kb {
                SwitchButton::all().to_vec()
            } else {
                ControllerConfig::pad_list().to_vec()
            };
            let n = if motion_page {
                MOTION_ROW_LABELS.len() + 1
            } else if vibration_page {
                VIBRATION_ROW_LABELS.len() + 1
            } else {
                binds.len() + 1
            };
            if !self.prefs_focus {
                self.prefs_row = 0;
                self.prefs_col = 0;
            }

            let nav_gamepads = self
                .input
                .as_ref()
                .map(|ib| ib.list_gamepads())
                .unwrap_or_default();
            let nav_choice = pad_choice_row(
                self.app_settings.gamepad.as_deref(),
                &nav_gamepads,
                self.input.as_ref().and_then(|ib| ib.get_active_id()),
            );

            if self.prefs_focus && !rebinding_active && !self.prefs_dropdown_open {
                if r_bumper_edge {
                    if self.prefs_col == 0 {
                        self.prefs_col = 1;
                        self.prefs_col1_row = 0;
                        crate::ui_audio::play_move();
                    } else if self.prefs_col1_row == 1 {
                        self.prefs_col1_row = 2;
                        crate::ui_audio::play_move();
                    }
                } else if l_bumper_edge {
                    if self.prefs_col == 1 {
                        if self.prefs_col1_row == 2 {
                            self.prefs_col1_row = 1;
                        } else {
                            self.prefs_col = 0;
                        }
                        crate::ui_audio::play_move();
                    }
                } else if self.prefs_col == 0
                    && self.prefs_row >= 1
                    && nr
                    && !(motion_page && (self.prefs_row == 1 || self.prefs_row == 4))
                    && !(vibration_page && (self.prefs_row == 1 || self.prefs_row == 3))
                {
                    self.prefs_col = 1;
                    self.prefs_col1_row = 0;
                    crate::ui_audio::play_move();
                } else if self.prefs_col == 1 && self.prefs_col1_row == 0 && nl {
                    self.prefs_col = 0;
                    crate::ui_audio::play_move();
                }
            }

            self.prefs_row = self.prefs_row.min(n - 1);

            if self.prefs_focus && self.prefs_col == 0 && !rebinding_active {
                if nu && self.prefs_row > 0 {
                    self.prefs_row -= 1;
                    crate::ui_audio::play_move();
                }
                if nd && self.prefs_row + 1 < n {
                    self.prefs_row += 1;
                    crate::ui_audio::play_move();
                }
            }

            if self.prefs_focus && self.prefs_col == 1 && !rebinding_active {
                if self.prefs_dropdown_open {
                    let cnt = nav_gamepads.len().max(nav_choice) + 1;
                    if nu {
                        self.prefs_dropdown_sel = (self.prefs_dropdown_sel + cnt - 1) % cnt;
                        crate::ui_audio::play_move();
                    }
                    if nd {
                        self.prefs_dropdown_sel = (self.prefs_dropdown_sel + 1) % cnt;
                        crate::ui_audio::play_move();
                    }
                    if a_edge {
                        let pick = match self.prefs_dropdown_sel {
                            sel if sel == nav_choice => None,
                            0 => Some(None),
                            sel => nav_gamepads.get(sel - 1).map(|(gid, _)| Some(*gid)),
                        };
                        if let (Some(pick), Some(ib)) = (pick, self.input.as_mut()) {
                            self.app_settings.gamepad = ib.set_active_id(pick);
                            let _ = self.app_settings.save();
                        }
                        self.prefs_dropdown_open = false;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                    if b_edge {
                        self.prefs_dropdown_open = false;
                        b_edge = false;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                    }
                } else {
                    if nu && self.prefs_col1_row > 0 {
                        self.prefs_col1_row -= 1;
                        crate::ui_audio::play_move();
                    }
                    if nd && self.prefs_col1_row < 2 {
                        self.prefs_col1_row += 1;
                        crate::ui_audio::play_move();
                    }
                    match self.prefs_col1_row {
                        0 => {
                            if a_edge {
                                self.prefs_dropdown_sel = nav_choice;
                                self.prefs_dropdown_open = true;
                                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                            }
                        }
                        1 => {
                            if nl {
                                self.app_settings.left_deadzone =
                                    (self.app_settings.left_deadzone - 0.02).clamp(0.0, 0.5);
                                let _ = self.app_settings.save();
                                crate::ui_audio::play_move();
                            }
                            if nr {
                                self.app_settings.left_deadzone =
                                    (self.app_settings.left_deadzone + 0.02).clamp(0.0, 0.5);
                                let _ = self.app_settings.save();
                                crate::ui_audio::play_move();
                            }
                        }
                        _ => {
                            if nl {
                                self.app_settings.right_deadzone =
                                    (self.app_settings.right_deadzone - 0.02).clamp(0.0, 0.5);
                                let _ = self.app_settings.save();
                                crate::ui_audio::play_move();
                            }
                            if nr {
                                self.app_settings.right_deadzone =
                                    (self.app_settings.right_deadzone + 0.02).clamp(0.0, 0.5);
                                let _ = self.app_settings.save();
                                crate::ui_audio::play_move();
                            }
                        }
                    }
                }
            }
            if !self.prefs_focus || self.prefs_col != 1 {
                self.prefs_dropdown_open = false;
            }

            let head_h = 56.0 * s;
            let list_w = (content.width() * 0.46).max(300.0 * s);
            let head = egui::Rect::from_min_size(
                egui::pos2(content.min.x, content.min.y + 2.0 * s),
                egui::Vec2::new(list_w, head_h - 10.0 * s),
            );
            let hr = sr(head);
            let head_sel = self.prefs_focus && self.prefs_col == 0 && self.prefs_row == 0;
            if head_sel {
                p.rect_filled(hr, egui::CornerRadius::from(12.0 * s), sel);
                p.rect_stroke(
                    hr,
                    egui::CornerRadius::from(12.0 * s),
                    egui::Stroke::new(2.0_f32, accent),
                    egui::StrokeKind::Outside,
                );
            } else {
                p.rect_filled(
                    hr,
                    egui::CornerRadius::from(12.0 * s),
                    egui::Color32::from_rgba_unmultiplied(
                        panel.r(),
                        panel.g(),
                        panel.b(),
                        (ease * 90.0) as u8,
                    ),
                );
                p.rect_stroke(
                    hr,
                    egui::CornerRadius::from(12.0 * s),
                    egui::Stroke::new(1.0_f32, border),
                    egui::StrokeKind::Outside,
                );
            }
            p.text(
                sp(egui::pos2(head.min.x + 22.0 * s, head.center().y)),
                egui::Align2::LEFT_CENTER,
                "Input Device",
                egui::FontId::proportional(18.0 * s * sf),
                text,
            );
            let dev_val = match self.input_device {
                InputDevice::Keyboard => "Keyboard",
                InputDevice::Gamepad => "Gamepad",
                InputDevice::Motion => "Motion",
                InputDevice::Vibration => "Vibration",
            };
            let dvcol = if head_sel { accent } else { muted };
            let hlx = head.max.x - 150.0 * s;
            let hrx = head.max.x - 22.0 * s;
            p.text(
                sp(egui::pos2((hlx + hrx) * 0.5, head.center().y)),
                egui::Align2::CENTER_CENTER,
                dev_val,
                egui::FontId::proportional(16.0 * s * sf),
                dvcol,
            );
            let hla = egui::Rect::from_center_size(
                egui::pos2(hlx, head.center().y),
                egui::Vec2::splat(34.0 * s),
            );
            let hra = egui::Rect::from_center_size(
                egui::pos2(hrx, head.center().y),
                egui::Vec2::splat(34.0 * s),
            );
            p.text(
                sp(hla.center()),
                egui::Align2::CENTER_CENTER,
                "\u{2039}",
                egui::FontId::proportional(22.0 * s * sf),
                dvcol,
            );
            p.text(
                sp(hra.center()),
                egui::Align2::CENTER_CENTER,
                "\u{203A}",
                egui::FontId::proportional(22.0 * s * sf),
                dvcol,
            );
            let mut dev_step = 0i32;
            if !rebinding_active && ui.rect_contains_pointer(hr) {
                let hc = ui.allocate_rect(hr, egui::Sense::click()).clicked();
                let hlc = ui.allocate_rect(sr(hla), egui::Sense::click()).clicked();
                let hrc = ui.allocate_rect(sr(hra), egui::Sense::click()).clicked();
                if hc || hlc || hrc {
                    self.prefs_focus = true;
                    self.prefs_row = 0;
                    self.prefs_col = 0;
                    dev_step = if hlc { -1 } else { 1 };
                }
            }
            if dev_step == 0
                && self.prefs_focus
                && self.prefs_col == 0
                && self.prefs_row == 0
                && !rebinding_active
            {
                if nl {
                    dev_step = -1;
                } else if nr || a_edge {
                    dev_step = 1;
                }
            }
            if dev_step != 0 {
                self.input_device = self.input_device.cycle(dev_step);
                self.prefs_row = 0;
                crate::ui_audio::play_move();
            }

            let pad_label = |g: crate::controller_config::GpButton| -> &'static str {
                use crate::controller_config::GpButton::*;
                match g {
                    South => "Bottom Btn",
                    East => "Right Btn",
                    West => "Left Btn",
                    North => "Top Btn",
                    L => "L Bumper",
                    R => "R Bumper",
                    ZL => "L Trigger",
                    ZR => "R Trigger",
                    Plus => "Start",
                    Minus => "Back",
                    LStick => "L Stick",
                    RStick => "R Stick",
                    Up => "Pad Up",
                    Down => "Pad Down",
                    Left => "Pad Left",
                    Right => "Pad Right",
                }
            };
            let list_top = content.min.y + head_h + 6.0 * s;
            let list_right = content.min.x + list_w;
            let row_h = 48.0 * s;
            let avail = (content.max.y - list_top).max(row_h);
            let vis = (avail / row_h).floor().max(1.0) as usize;
            let selb = if self.prefs_row >= 1 {
                self.prefs_row - 1
            } else {
                0
            };
            let max_first = binds.len().saturating_sub(vis);
            let list_clip = sr(egui::Rect::from_min_max(
                egui::pos2(content.min.x, list_top - 2.0 * s),
                egui::pos2(list_right, content.max.y),
            ));
            let hover_in_list = ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .map_or(false, |p| list_clip.contains(p))
            });
            if hover_in_list {
                let wheel = ctx.input(|i| i.smooth_scroll_delta.y);
                if wheel > 2.0 {
                    self.prefs_ctrl_scroll = self.prefs_ctrl_scroll.saturating_sub(1);
                } else if wheel < -2.0 {
                    self.prefs_ctrl_scroll = (self.prefs_ctrl_scroll + 1).min(max_first);
                }
            }
            if self.prefs_focus && self.prefs_col == 0 {
                if selb < self.prefs_ctrl_scroll {
                    self.prefs_ctrl_scroll = selb;
                }
                if selb >= self.prefs_ctrl_scroll + vis {
                    self.prefs_ctrl_scroll = selb + 1 - vis;
                }
            }
            if vis < binds.len() {
                let track = sr(egui::Rect::from_min_max(
                    egui::pos2(list_right - 9.0 * s, list_top),
                    egui::pos2(list_right - 2.0 * s, content.max.y),
                ));
                let tresp = ui.allocate_rect(track, egui::Sense::click_and_drag());
                if (tresp.dragged() || tresp.clicked()) && !rebinding_active {
                    if let Some(pos) = tresp.interact_pointer_pos() {
                        let frac =
                            ((pos.y - track.min.y) / track.height().max(1.0)).clamp(0.0, 1.0);
                        self.prefs_ctrl_scroll = (frac * max_first as f32).round() as usize;
                    }
                }
            }
            self.prefs_ctrl_scroll = self.prefs_ctrl_scroll.min(max_first);
            let first = self.prefs_ctrl_scroll;
            let lp = p.with_clip_rect(list_clip);
            let mut clicked_bind: Option<usize> = None;
            for vi in 0..vis {
                let bi = first + vi;
                if bi >= binds.len() {
                    break;
                }
                let btn = binds[bi];
                let base = egui::Rect::from_min_size(
                    egui::pos2(content.min.x, list_top + vi as f32 * row_h),
                    egui::Vec2::new(list_w - 10.0 * s, row_h - 8.0 * s),
                );
                let r = sr(base);
                let selrow = self.prefs_focus && self.prefs_col == 0 && self.prefs_row == bi + 1;
                let this_rebind =
                    (kb && self.rebinding == Some(btn)) || (!kb && self.rebinding_pad == Some(btn));
                if this_rebind {
                    lp.rect_filled(
                        r,
                        egui::CornerRadius::from(10.0 * s),
                        egui::Color32::from_rgba_unmultiplied(
                            accent.r(),
                            accent.g(),
                            accent.b(),
                            40,
                        ),
                    );
                    lp.rect_stroke(
                        r,
                        egui::CornerRadius::from(10.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                } else if selrow {
                    lp.rect_filled(r, egui::CornerRadius::from(10.0 * s), sel);
                    lp.rect_stroke(
                        r,
                        egui::CornerRadius::from(10.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                } else {
                    lp.rect_filled(
                        r,
                        egui::CornerRadius::from(10.0 * s),
                        egui::Color32::from_rgba_unmultiplied(
                            panel.r(),
                            panel.g(),
                            panel.b(),
                            (ease * 70.0) as u8,
                        ),
                    );
                    lp.rect_stroke(
                        r,
                        egui::CornerRadius::from(10.0 * s),
                        egui::Stroke::new(1.0_f32, border),
                        egui::StrokeKind::Outside,
                    );
                }
                lp.text(
                    sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
                    egui::Align2::LEFT_CENTER,
                    btn.display_name(),
                    egui::FontId::proportional(16.0 * s * sf),
                    text,
                );
                let cur = if kb {
                    self.controller_config
                        .binding_for(btn)
                        .unwrap_or("\u{2014}")
                        .to_string()
                } else {
                    self.controller_config
                        .pad_for(btn)
                        .map(pad_label)
                        .unwrap_or("\u{2014}")
                        .to_string()
                };
                let vcol = if this_rebind {
                    accent
                } else if selrow {
                    accent
                } else {
                    muted
                };
                let vtext = if this_rebind {
                    "Press a key\u{2026}".to_string()
                } else {
                    cur
                };
                lp.text(
                    sp(egui::pos2(base.max.x - 22.0 * s, base.center().y)),
                    egui::Align2::RIGHT_CENTER,
                    &vtext,
                    egui::FontId::proportional(15.0 * s * sf),
                    vcol,
                );
                if !rebinding_active
                    && ui.rect_contains_pointer(r)
                    && ui.allocate_rect(r, egui::Sense::click()).clicked()
                {
                    clicked_bind = Some(bi);
                }
            }
            if vis < binds.len() {
                let sb_x = list_right - 8.0 * s;
                p.rect_filled(
                    sr(egui::Rect::from_min_size(
                        egui::pos2(sb_x, list_top),
                        egui::Vec2::new(5.0 * s, avail),
                    )),
                    egui::CornerRadius::from(3.0 * s),
                    egui::Color32::from_rgba_unmultiplied(muted.r(), muted.g(), muted.b(), 40),
                );
                let frac_h = (vis as f32 / binds.len() as f32) * avail;
                let frac_y = list_top + (first as f32 / binds.len().max(1) as f32) * avail;
                p.rect_filled(
                    sr(egui::Rect::from_min_size(
                        egui::pos2(sb_x, frac_y),
                        egui::Vec2::new(5.0 * s, frac_h),
                    )),
                    egui::CornerRadius::from(3.0 * s),
                    egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200),
                );
            }
            if !motion_page && !vibration_page {
                let start = sp(egui::pos2(full.min.x + mx, hy));
                let end_x = sp(egui::pos2(full.max.x - mx, hy)).x - hint_w - 40.0 * s;
                p.with_clip_rect(egui::Rect::from_min_max(
                    egui::pos2(start.x, start.y - 14.0 * s),
                    egui::pos2(end_x, start.y + 14.0 * s),
                ))
                .text(
                    start,
                    egui::Align2::LEFT_CENTER,
                    motion_summary(&self.app_settings, &self.controller_config),
                    egui::FontId::proportional(13.0 * s * sf),
                    muted,
                );
            }
            if motion_page || vibration_page {
                let labels: &[&str] = if motion_page {
                    &MOTION_ROW_LABELS
                } else {
                    &VIBRATION_ROW_LABELS
                };
                let help: &[&str] = if motion_page { &MOTION_HELP } else { &VIBRATION_HELP };
                let cycle_rows: [usize; 2] = if motion_page { [0, 3] } else { [0, 2] };
                let page_enabled = if motion_page {
                    self.app_settings.motion_enabled
                } else {
                    self.app_settings.vibration_enabled
                };
                let (values, calibration_note): (Vec<String>, Option<&str>) = if motion_page {
                    let source = self.input.as_ref().and_then(|ib| ib.motion_source());
                    let (calibration, calibration_note) = motion_calibration_text(
                        self.app_settings.motion_enabled,
                        nexium_core::hid_motion::calibration_states(),
                        nexium_core::hid_motion::stale_sources(),
                        self.input.as_ref().and_then(|ib| ib.motion_hint_note()),
                    );
                    let values = motion_row_values(
                        &self.app_settings,
                        source.as_deref(),
                        self.motion_key_capture,
                        self.motion_recently_recentered(),
                        calibration,
                    );
                    (Vec::from(values), calibration_note)
                } else {
                    let status = self.vibration_status();
                    (Vec::from(vibration_row_values(&self.app_settings, &status)), None)
                };
                let key_note = if motion_page {
                    motion_note(
                        &self.app_settings,
                        &self.controller_config,
                        self.motion_key_notice
                            .as_deref()
                            .filter(|_| self.motion_key_capture),
                        self.motion_button_missing(),
                    )
                } else {
                    None
                };
                let restart_note = (motion_page
                    && self.app_settings.motion_enabled != self.startup_motion_enabled)
                    .then_some(MOTION_RESTART);
                let notes = [key_note.as_deref(), restart_note, calibration_note];
                let text_lines = help.len() + notes.iter().flatten().count();
                let row_h = ((content.max.y - list_top - 6.0 * s - text_lines as f32 * 20.0 * s)
                    / labels.len() as f32)
                    .clamp(34.0 * s, row_h);
                let mut motion_act: Option<(usize, i32)> = None;
                for (i, value) in values.iter().enumerate() {
                    let base = egui::Rect::from_min_size(
                        egui::pos2(content.min.x, list_top + i as f32 * row_h),
                        egui::Vec2::new(list_w - 10.0 * s, row_h - 8.0 * s),
                    );
                    let r = sr(base);
                    let selrow = self.prefs_focus && self.prefs_col == 0 && self.prefs_row == i + 1;
                    let capturing = motion_page && i == 2 && self.motion_key_capture;
                    let dim = i >= 2 && !page_enabled;
                    let action = match (motion_page, i) {
                        (true, 4) => Some(MOTION_RECENTER_ACTION),
                        (true, 5) => Some(MOTION_CALIBRATE_ACTION),
                        (false, 3) => Some(VIBRATION_TEST_ACTION),
                        _ => None,
                    };
                    let value = match action {
                        Some(text) if value.is_empty() => text,
                        _ => value.as_str(),
                    };
                    if capturing {
                        p.rect_filled(
                            r,
                            egui::CornerRadius::from(10.0 * s),
                            egui::Color32::from_rgba_unmultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                40,
                            ),
                        );
                        p.rect_stroke(
                            r,
                            egui::CornerRadius::from(10.0 * s),
                            egui::Stroke::new(2.0_f32, accent),
                            egui::StrokeKind::Outside,
                        );
                    } else if selrow {
                        p.rect_filled(r, egui::CornerRadius::from(10.0 * s), sel);
                        p.rect_stroke(
                            r,
                            egui::CornerRadius::from(10.0 * s),
                            egui::Stroke::new(2.0_f32, accent),
                            egui::StrokeKind::Outside,
                        );
                    } else {
                        p.rect_filled(
                            r,
                            egui::CornerRadius::from(10.0 * s),
                            egui::Color32::from_rgba_unmultiplied(
                                panel.r(),
                                panel.g(),
                                panel.b(),
                                (ease * 70.0) as u8,
                            ),
                        );
                        p.rect_stroke(
                            r,
                            egui::CornerRadius::from(10.0 * s),
                            egui::Stroke::new(1.0_f32, border),
                            egui::StrokeKind::Outside,
                        );
                    }
                    let label_end = p
                        .text(
                            sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
                            egui::Align2::LEFT_CENTER,
                            labels[i],
                            egui::FontId::proportional(16.0 * s * sf),
                            if dim { muted } else { text },
                        )
                        .max
                        .x;
                    let vcol = if capturing || (selrow && i != 1 && !dim) {
                        accent
                    } else {
                        muted
                    };
                    if cycle_rows.contains(&i) {
                        let lx = base.max.x - 150.0 * s;
                        let rx = base.max.x - 22.0 * s;
                        p.text(
                            sp(egui::pos2((lx + rx) * 0.5, base.center().y)),
                            egui::Align2::CENTER_CENTER,
                            value,
                            egui::FontId::proportional(15.0 * s * sf),
                            vcol,
                        );
                        let la = egui::Rect::from_center_size(
                            egui::pos2(lx, base.center().y),
                            egui::Vec2::splat(34.0 * s),
                        );
                        let ra = egui::Rect::from_center_size(
                            egui::pos2(rx, base.center().y),
                            egui::Vec2::splat(34.0 * s),
                        );
                        p.text(
                            sp(la.center()),
                            egui::Align2::CENTER_CENTER,
                            "\u{2039}",
                            egui::FontId::proportional(22.0 * s * sf),
                            vcol,
                        );
                        p.text(
                            sp(ra.center()),
                            egui::Align2::CENTER_CENTER,
                            "\u{203A}",
                            egui::FontId::proportional(22.0 * s * sf),
                            vcol,
                        );
                        if !rebinding_active && ui.rect_contains_pointer(r) {
                            let rc = ui.allocate_rect(r, egui::Sense::click()).clicked();
                            let lc = ui.allocate_rect(sr(la), egui::Sense::click()).clicked();
                            let rtc = ui.allocate_rect(sr(ra), egui::Sense::click()).clicked();
                            if rc || lc || rtc {
                                motion_act = Some((i, if lc { -1 } else { 1 }));
                            }
                        }
                    } else {
                        let anchor = sp(egui::pos2(base.max.x - 22.0 * s, base.center().y));
                        let mut job = egui::text::LayoutJob::simple_singleline(
                            value.to_string(),
                            egui::FontId::proportional(15.0 * s * sf),
                            vcol,
                        );
                        job.wrap = egui::text::TextWrapping::truncate_at_width(
                            anchor.x - label_end - 16.0 * s * sf,
                        );
                        let galley = p.layout_job(job);
                        let at = egui::Align2::RIGHT_CENTER.anchor_size(anchor, galley.size());
                        p.galley(at.min, galley, vcol);
                        if (!rebinding_active || capturing)
                            && ui.rect_contains_pointer(r)
                            && ui.allocate_rect(r, egui::Sense::click()).clicked()
                        {
                            motion_act = Some((i, 1));
                        }
                    }
                }
                let mut help_y = list_top + labels.len() as f32 * row_h + 6.0 * s;
                for note in notes.into_iter().flatten() {
                    p.text(
                        sp(egui::pos2(content.min.x + 2.0 * s, help_y)),
                        egui::Align2::LEFT_TOP,
                        note,
                        egui::FontId::proportional(13.0 * s * sf),
                        AMBER,
                    );
                    help_y += 20.0 * s;
                }
                for (k, line) in help.iter().enumerate() {
                    let line_y = help_y + k as f32 * 20.0 * s;
                    if line_y + 16.0 * s > content.max.y {
                        break;
                    }
                    p.text(
                        sp(egui::pos2(content.min.x + 2.0 * s, line_y)),
                        egui::Align2::LEFT_TOP,
                        *line,
                        egui::FontId::proportional(13.0 * s * sf),
                        muted,
                    );
                }
                if let Some((i, _)) = motion_act {
                    self.prefs_focus = true;
                    self.prefs_col = 0;
                    self.prefs_row = i + 1;
                } else if self.prefs_focus
                    && self.prefs_col == 0
                    && self.prefs_row >= 1
                    && !rebinding_active
                {
                    let i = self.prefs_row - 1;
                    let cycles = cycle_rows.contains(&i);
                    if cycles && nl {
                        motion_act = Some((i, -1));
                    } else if (cycles && nr) || a_edge {
                        motion_act = Some((i, 1));
                    }
                }
                if motion_page {
                    match motion_act {
                        Some((0, _)) => {
                            self.app_settings.motion_enabled = !self.app_settings.motion_enabled;
                            let _ = self.app_settings.save();
                            crate::ui_audio::play_move();
                        }
                        Some((2, _)) => {
                            if self.motion_key_capture {
                                self.motion_key_capture = false;
                                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                            } else {
                                self.motion_key_capture = true;
                                self.motion_key_notice = None;
                                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                            }
                        }
                        Some((3, dir)) => {
                            let button = self.app_settings.motion_recenter_button;
                            self.app_settings.motion_recenter_button =
                                if dir < 0 { button.prev() } else { button.next() };
                            let _ = self.app_settings.save();
                            crate::ui_audio::play_move();
                        }
                        Some((4, _)) => {
                            self.recenter_motion();
                            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                        }
                        Some((5, _)) => {
                            self.calibrate_motion();
                            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                        }
                        _ => {}
                    }
                } else {
                    match motion_act {
                        Some((0, _)) => {
                            self.app_settings.vibration_enabled = !self.app_settings.vibration_enabled;
                            let _ = self.app_settings.save();
                            crate::ui_audio::play_move();
                            self.test_vibration();
                        }
                        Some((2, dir)) if self.app_settings.vibration_enabled => {
                            let strength =
                                step_vibration_strength(self.app_settings.vibration_strength, dir);
                            crate::ui_audio::play_move();
                            if strength != self.app_settings.vibration_strength {
                                self.app_settings.vibration_strength = strength;
                                let _ = self.app_settings.save();
                                self.test_vibration();
                            }
                        }
                        Some((3, _)) if self.app_settings.vibration_enabled => {
                            self.test_vibration();
                            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                        }
                        _ => {}
                    }
                }
            }

            let hl_btn: Option<SwitchButton> = if kb {
                self.rebinding.or(
                    if self.prefs_focus && self.prefs_col == 0 && self.prefs_row >= 1 {
                        binds.get(selb).copied()
                    } else {
                        None
                    },
                )
            } else {
                self.rebinding_pad.or(
                    if self.prefs_focus && self.prefs_col == 0 && self.prefs_row >= 1 {
                        binds.get(selb).copied()
                    } else {
                        None
                    },
                )
            };
            let cw_area = egui::Rect::from_min_max(
                egui::pos2(list_right + 30.0 * s, list_top),
                egui::pos2(content.max.x, content.max.y),
            );
            let controller_area = egui::Rect::from_min_max(
                egui::pos2(cw_area.min.x, cw_area.min.y),
                egui::pos2(cw_area.max.x, cw_area.max.y - 135.0 * s),
            );
            let cw = controller_area.width().min(controller_area.height() * 1.34);
            let ch = cw / 1.34;
            let cbase =
                egui::Rect::from_center_size(controller_area.center(), egui::Vec2::new(cw, ch));

            let mut gamepads = Vec::new();
            let mut active_id = None;
            if let Some(ref ib) = self.input {
                gamepads = ib.list_gamepads();
                active_id = ib.get_active_id();
            }

            let tab_rect = egui::Rect::from_min_max(
                egui::pos2(cbase.min.x, content.min.y + 2.0 * s),
                egui::pos2(cbase.max.x, content.min.y + 2.0 * s + head_h - 10.0 * s),
            );
            let tab_sr = sr(tab_rect);

            let tab_focused = self.prefs_focus && self.prefs_col == 1 && self.prefs_col1_row == 0;
            if tab_focused {
                p.rect_filled(tab_sr, egui::CornerRadius::from(8.0 * s), sel);
                p.rect_stroke(
                    tab_sr,
                    egui::CornerRadius::from(8.0 * s),
                    egui::Stroke::new(2.0_f32, accent),
                    egui::StrokeKind::Outside,
                );
            } else {
                p.rect_filled(
                    tab_sr,
                    egui::CornerRadius::from(8.0 * s),
                    egui::Color32::from_rgba_unmultiplied(
                        panel.r(),
                        panel.g(),
                        panel.b(),
                        (ease * 90.0) as u8,
                    ),
                );
                p.rect_stroke(
                    tab_sr,
                    egui::CornerRadius::from(8.0 * s),
                    egui::Stroke::new(1.0_f32, border),
                    egui::StrokeKind::Outside,
                );
            }

            let divider_x = tab_sr.min.x + 130.0 * s * sf;
            p.line_segment(
                [
                    egui::pos2(divider_x, tab_sr.min.y + 6.0 * s * sf),
                    egui::pos2(divider_x, tab_sr.max.y - 6.0 * s * sf),
                ],
                egui::Stroke::new(1.0_f32, border),
            );

            p.text(
                sp(egui::pos2(tab_rect.min.x + 15.0 * s, tab_rect.center().y)),
                egui::Align2::LEFT_CENTER,
                "Controller",
                egui::FontId::proportional(14.0 * s * sf),
                text,
            );

            let choice_row =
                pad_choice_row(self.app_settings.gamepad.as_deref(), &gamepads, active_id);
            let selected_name =
                pad_choice_text(self.app_settings.gamepad.as_deref(), &gamepads, active_id);
            let name_col = if gamepads.is_empty() { muted } else { text };
            let name_x = divider_x + 12.0 * s * sf;
            let name_clip = egui::Rect::from_min_max(
                egui::pos2(name_x - 2.0 * s, tab_sr.min.y),
                egui::pos2(tab_sr.max.x - 22.0 * s * sf, tab_sr.max.y),
            );
            let np = p.with_clip_rect(name_clip);
            np.text(
                sp(egui::pos2(name_x, tab_rect.center().y)),
                egui::Align2::LEFT_CENTER,
                &selected_name,
                egui::FontId::proportional(14.0 * s * sf),
                name_col,
            );

            let cvx = tab_rect.max.x - 16.0 * s;
            let cvy = tab_rect.center().y;
            let cv_col = if tab_focused { accent } else { muted };
            let cv_stroke = egui::Stroke::new(1.6 * sf, cv_col);
            if self.prefs_dropdown_open {
                p.line_segment(
                    [
                        sp(egui::pos2(cvx - 5.0 * s, cvy + 3.0 * s)),
                        sp(egui::pos2(cvx, cvy - 3.0 * s)),
                    ],
                    cv_stroke,
                );
                p.line_segment(
                    [
                        sp(egui::pos2(cvx, cvy - 3.0 * s)),
                        sp(egui::pos2(cvx + 5.0 * s, cvy + 3.0 * s)),
                    ],
                    cv_stroke,
                );
            } else {
                p.line_segment(
                    [
                        sp(egui::pos2(cvx - 5.0 * s, cvy - 3.0 * s)),
                        sp(egui::pos2(cvx, cvy + 3.0 * s)),
                    ],
                    cv_stroke,
                );
                p.line_segment(
                    [
                        sp(egui::pos2(cvx, cvy + 3.0 * s)),
                        sp(egui::pos2(cvx + 5.0 * s, cvy - 3.0 * s)),
                    ],
                    cv_stroke,
                );
            }

            if !rebinding_active
                && ui.rect_contains_pointer(tab_sr)
                && ui.allocate_rect(tab_sr, egui::Sense::click()).clicked()
            {
                self.prefs_focus = true;
                self.prefs_col = 1;
                self.prefs_col1_row = 0;
                if self.prefs_dropdown_open {
                    self.prefs_dropdown_open = false;
                } else {
                    self.prefs_dropdown_sel = choice_row;
                    self.prefs_dropdown_open = true;
                }
            }

            draw_controller(&p, sr(cbase), hl_btn, accent, lightish, &self.last_input);

            if self.prefs_dropdown_open {
                let missing = self
                    .app_settings
                    .gamepad
                    .as_ref()
                    .filter(|_| choice_row > gamepads.len())
                    .map(|name| format!("{} (not connected)", name));
                let items = gamepads.len().max(1) + 1 + usize::from(missing.is_some());
                let ih = 40.0 * s;
                let dd = egui::Rect::from_min_max(
                    egui::pos2(tab_rect.min.x, tab_rect.max.y + 5.0 * s),
                    egui::pos2(
                        tab_rect.max.x,
                        tab_rect.max.y + 5.0 * s + ih * items as f32 + 8.0 * s,
                    ),
                );
                let dd_sr = sr(dd);
                p.rect_filled(dd_sr, egui::CornerRadius::from(9.0 * s), panel);
                p.rect_stroke(
                    dd_sr,
                    egui::CornerRadius::from(9.0 * s),
                    egui::Stroke::new(1.5_f32, accent),
                    egui::StrokeKind::Outside,
                );
                if gamepads.is_empty() {
                    let y = dd.min.y + 4.0 * s + ih * (items as f32 - 0.5);
                    p.text(
                        sp(egui::pos2(dd.center().x, y)),
                        egui::Align2::CENTER_CENTER,
                        "No gamepads detected",
                        egui::FontId::proportional(13.0 * s * sf),
                        muted,
                    );
                }
                let rows = gamepads
                    .iter()
                    .map(|(gid, gname)| (Some(*gid), gname.as_str()));
                let choices = std::iter::once((None, PAD_AUTOMATIC))
                    .chain(rows)
                    .chain(missing.as_deref().map(|label| (None, label)));
                for (i, (gid, gname)) in choices.enumerate() {
                    let ib_rect = egui::Rect::from_min_size(
                        egui::pos2(dd.min.x + 6.0 * s, dd.min.y + 4.0 * s + i as f32 * ih),
                        egui::vec2(dd.width() - 12.0 * s, ih - 2.0 * s),
                    );
                    let ir = sr(ib_rect);
                    let sel_item = self.prefs_dropdown_sel == i;
                    let is_active = i == choice_row;
                    if sel_item {
                        p.rect_filled(
                            ir,
                            egui::CornerRadius::from(6.0 * s),
                            egui::Color32::from_rgba_unmultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                70,
                            ),
                        );
                    } else if ui.rect_contains_pointer(ir) {
                        p.rect_filled(ir, egui::CornerRadius::from(6.0 * s), hover);
                    }
                    let ip = p.with_clip_rect(ir);
                    ip.text(
                        sp(egui::pos2(ib_rect.min.x + 12.0 * s, ib_rect.center().y)),
                        egui::Align2::LEFT_CENTER,
                        gname,
                        egui::FontId::proportional(13.0 * s * sf),
                        if sel_item {
                            accent
                        } else if i > gamepads.len() {
                            muted
                        } else {
                            text
                        },
                    );
                    if is_active {
                        p.text(
                            sp(egui::pos2(ib_rect.max.x - 12.0 * s, ib_rect.center().y)),
                            egui::Align2::RIGHT_CENTER,
                            "\u{2022}",
                            egui::FontId::proportional(18.0 * s * sf),
                            accent,
                        );
                    }
                    if !rebinding_active
                        && ui.rect_contains_pointer(ir)
                        && ui.allocate_rect(ir, egui::Sense::click()).clicked()
                    {
                        if let Some(ib) = self.input.as_mut().filter(|_| i != choice_row) {
                            self.app_settings.gamepad = ib.set_active_id(gid);
                            let _ = self.app_settings.save();
                        }
                        self.prefs_dropdown_open = false;
                    }
                }
                let outside = ui.input(|i| i.pointer.primary_clicked())
                    && !ui.rect_contains_pointer(dd_sr)
                    && !ui.rect_contains_pointer(tab_sr);
                if outside {
                    self.prefs_dropdown_open = false;
                }
            }

            let vis_center_y = cbase.max.y + 40.0 * s;

            let ls_center = egui::pos2(cw_area.center().x - 90.0 * s, vis_center_y);
            let rs_center = egui::pos2(cw_area.center().x + 90.0 * s, vis_center_y);
            let circle_r = 38.0 * s;

            p.circle_filled(
                sp(ls_center),
                circle_r,
                egui::Color32::from_rgba_unmultiplied(panel.r(), panel.g(), panel.b(), 100),
            );
            p.circle_stroke(sp(ls_center), circle_r, egui::Stroke::new(1.0_f32, border));
            p.circle_filled(
                sp(rs_center),
                circle_r,
                egui::Color32::from_rgba_unmultiplied(panel.r(), panel.g(), panel.b(), 100),
            );
            p.circle_stroke(sp(rs_center), circle_r, egui::Stroke::new(1.0_f32, border));

            let ls_dz_r = circle_r * self.app_settings.left_deadzone;
            let rs_dz_r = circle_r * self.app_settings.right_deadzone;
            p.circle_filled(
                sp(ls_center),
                ls_dz_r,
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 60),
            );
            p.circle_stroke(sp(ls_center), ls_dz_r, egui::Stroke::new(1.2_f32, accent));
            p.circle_filled(
                sp(rs_center),
                rs_dz_r,
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 60),
            );
            p.circle_stroke(sp(rs_center), rs_dz_r, egui::Stroke::new(1.2_f32, accent));

            p.line_segment(
                [
                    sp(egui::pos2(ls_center.x - circle_r, ls_center.y)),
                    sp(egui::pos2(ls_center.x + circle_r, ls_center.y)),
                ],
                egui::Stroke::new(0.5_f32, border),
            );
            p.line_segment(
                [
                    sp(egui::pos2(ls_center.x, ls_center.y - circle_r)),
                    sp(egui::pos2(ls_center.x, ls_center.y + circle_r)),
                ],
                egui::Stroke::new(0.5_f32, border),
            );
            p.line_segment(
                [
                    sp(egui::pos2(rs_center.x - circle_r, rs_center.y)),
                    sp(egui::pos2(rs_center.x + circle_r, rs_center.y)),
                ],
                egui::Stroke::new(0.5_f32, border),
            );
            p.line_segment(
                [
                    sp(egui::pos2(rs_center.x, rs_center.y - circle_r)),
                    sp(egui::pos2(rs_center.x, rs_center.y + circle_r)),
                ],
                egui::Stroke::new(0.5_f32, border),
            );

            let ls_raw = [self.last_input.raw_sticks[0], self.last_input.raw_sticks[1]];
            let rs_raw = [self.last_input.raw_sticks[2], self.last_input.raw_sticks[3]];

            let ls_dot = egui::pos2(
                ls_center.x + ls_raw[0] * circle_r,
                ls_center.y - ls_raw[1] * circle_r,
            );
            let rs_dot = egui::pos2(
                rs_center.x + rs_raw[0] * circle_r,
                rs_center.y - rs_raw[1] * circle_r,
            );

            p.line_segment(
                [sp(ls_center), sp(ls_dot)],
                egui::Stroke::new(1.5_f32, accent),
            );
            p.line_segment(
                [sp(rs_center), sp(rs_dot)],
                egui::Stroke::new(1.5_f32, accent),
            );

            p.circle_filled(sp(ls_dot), 4.0 * s, accent);
            p.circle_stroke(sp(ls_dot), 4.0 * s, egui::Stroke::new(1.0_f32, text));
            p.circle_filled(sp(rs_dot), 4.0 * s, accent);
            p.circle_stroke(sp(rs_dot), 4.0 * s, egui::Stroke::new(1.0_f32, text));

            p.text(
                sp(egui::pos2(ls_center.x, ls_center.y - circle_r - 12.0 * s)),
                egui::Align2::CENTER_CENTER,
                "Left Stick",
                egui::FontId::proportional(11.0 * s * sf),
                text,
            );
            p.text(
                sp(egui::pos2(rs_center.x, rs_center.y - circle_r - 12.0 * s)),
                egui::Align2::CENTER_CENTER,
                "Right Stick",
                egui::FontId::proportional(11.0 * s * sf),
                text,
            );

            let dz_sel = self.prefs_focus && self.prefs_col == 1 && !self.prefs_dropdown_open;
            let mut dz_drag: Option<(bool, f32)> = None;
            let dz_slider = |p: &egui::Painter,
                             cx: f32,
                             label: &str,
                             value: f32,
                             selected: bool|
             -> egui::Rect {
                let track_w = 150.0 * s;
                let ty = cbase.max.y + 100.0 * s;
                let tl = cx - track_w * 0.5;
                let tr = cx + track_w * 0.5;
                p.text(
                    sp(egui::pos2(cx, ty - 20.0 * s)),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(11.0 * s * sf),
                    if selected { accent } else { text },
                );
                let track = egui::Rect::from_min_max(
                    egui::pos2(tl, ty - 3.0 * s),
                    egui::pos2(tr, ty + 3.0 * s),
                );
                p.rect_filled(
                    sr(track),
                    egui::CornerRadius::from(3.0 * s),
                    egui::Color32::from_rgba_unmultiplied(muted.r(), muted.g(), muted.b(), 70),
                );
                let frac = (value / 0.5).clamp(0.0, 1.0);
                let knobx = tl + frac * (tr - tl);
                let fill = egui::Rect::from_min_max(
                    egui::pos2(tl, ty - 3.0 * s),
                    egui::pos2(knobx, ty + 3.0 * s),
                );
                p.rect_filled(sr(fill), egui::CornerRadius::from(3.0 * s), accent);
                if selected {
                    p.rect_stroke(
                        sr(track.expand(4.0 * s)),
                        egui::CornerRadius::from(6.0 * s),
                        egui::Stroke::new(1.5_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                }
                p.circle(
                    sp(egui::pos2(knobx, ty)),
                    8.0 * s * sf,
                    if selected { accent } else { text },
                    egui::Stroke::new(2.0_f32, accent),
                );
                p.text(
                    sp(egui::pos2(cx, ty + 20.0 * s)),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}%", value * 200.0),
                    egui::FontId::proportional(11.0 * s * sf),
                    if selected { accent } else { muted },
                );
                sr(egui::Rect::from_min_max(
                    egui::pos2(tl, ty - 14.0 * s),
                    egui::pos2(tr, ty + 14.0 * s),
                ))
            };
            let lgrab = dz_slider(
                &p,
                cw_area.center().x - 90.0 * s,
                "L Deadzone",
                self.app_settings.left_deadzone,
                dz_sel && self.prefs_col1_row == 1,
            );
            let rgrab = dz_slider(
                &p,
                cw_area.center().x + 90.0 * s,
                "R Deadzone",
                self.app_settings.right_deadzone,
                dz_sel && self.prefs_col1_row == 2,
            );
            if !rebinding_active && !self.prefs_dropdown_open {
                for (is_left, grab) in [(true, lgrab), (false, rgrab)] {
                    if ui.rect_contains_pointer(grab) {
                        let resp = ui.allocate_rect(grab, egui::Sense::click_and_drag());
                        if resp.dragged() || resp.clicked() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                let frac =
                                    ((pos.x - grab.min.x) / grab.width().max(1.0)).clamp(0.0, 1.0);
                                dz_drag = Some((is_left, frac * 0.5));
                            }
                        }
                    }
                }
            }
            if let Some((is_left, v)) = dz_drag {
                self.prefs_focus = true;
                self.prefs_col = 1;
                if is_left {
                    self.app_settings.left_deadzone = v;
                    self.prefs_col1_row = 1;
                } else {
                    self.app_settings.right_deadzone = v;
                    self.prefs_col1_row = 2;
                }
                let _ = self.app_settings.save();
            }
            let start_rebind = |app: &mut Self, bi: usize| {
                let btn = binds[bi];
                app.prefs_focus = true;
                app.prefs_row = bi + 1;
                if kb {
                    app.rebinding = Some(btn);
                } else {
                    app.rebinding_pad = Some(btn);
                    app.rebinding_pad_wait_release = true;
                }
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            };
            if let Some(bi) = clicked_bind {
                start_rebind(self, bi);
            } else if self.prefs_focus
                && self.prefs_col == 0
                && self.prefs_row >= 1
                && !rebinding_active
                && !self.prefs_dropdown_open
                && a_edge
                && !motion_page
                && !vibration_page
            {
                start_rebind(self, self.prefs_row - 1);
            }
        } else if self.settings_tab == SettingsTab::Audio {
            if self.audio_device_cache.is_none() {
                self.audio_device_cache = Some(list_output_devices());
            }
            let devices: Vec<String> = self.audio_device_cache.clone().unwrap_or_default();
            let n = 7usize;
            if !self.prefs_focus {
                self.prefs_row = 0;
            }
            self.prefs_row = self.prefs_row.min(n - 1);
            if self.prefs_focus && !block {
                if nu && self.prefs_row > 0 {
                    self.prefs_row -= 1;
                    crate::ui_audio::play_move();
                }
                if nd && self.prefs_row + 1 < n {
                    self.prefs_row += 1;
                    crate::ui_audio::play_move();
                }
            }
            let labels = [
                "Output Device",
                "Master Volume",
                "Music Volume",
                "SFX Volume",
                "Mute Music",
                "Mute SFX",
                "Menu Music",
            ];
            let row_h = 58.0 * s;
            let mut y = content.min.y + 6.0 * s;
            let mut drag_set: Option<(usize, f32)> = None;
            for i in 0..n {
                let base = egui::Rect::from_min_size(
                    egui::pos2(content.min.x, y),
                    egui::Vec2::new(content.width(), row_h - 12.0 * s),
                );
                let r = sr(base);
                let selrow = self.prefs_focus && self.prefs_row == i;
                if selrow {
                    p.rect_filled(r, egui::CornerRadius::from(12.0 * s), sel);
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(2.0_f32, accent),
                        egui::StrokeKind::Outside,
                    );
                } else {
                    p.rect_filled(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Color32::from_rgba_unmultiplied(
                            panel.r(),
                            panel.g(),
                            panel.b(),
                            (ease * 90.0) as u8,
                        ),
                    );
                    p.rect_stroke(
                        r,
                        egui::CornerRadius::from(12.0 * s),
                        egui::Stroke::new(1.0_f32, border),
                        egui::StrokeKind::Outside,
                    );
                }
                let desc = match i {
                    0 => "Output device for all audio. Restart NeXium to apply changes.",
                    6 => {
                        "Either Select All, or specific tracks for your Main Carousel Music Themes."
                    }
                    _ => "",
                };
                if desc.is_empty() {
                    p.text(
                        sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
                        egui::Align2::LEFT_CENTER,
                        labels[i],
                        egui::FontId::proportional(18.0 * s * sf),
                        text,
                    );
                } else {
                    p.text(
                        sp(egui::pos2(
                            base.min.x + 22.0 * s,
                            base.center().y - 10.0 * s,
                        )),
                        egui::Align2::LEFT_CENTER,
                        labels[i],
                        egui::FontId::proportional(18.0 * s * sf),
                        text,
                    );
                    p.text(
                        sp(egui::pos2(
                            base.min.x + 22.0 * s,
                            base.center().y + 12.0 * s,
                        )),
                        egui::Align2::LEFT_CENTER,
                        desc,
                        egui::FontId::proportional(12.0 * s * sf),
                        muted,
                    );
                }
                if (1..=3).contains(&i) {
                    let val = match i {
                        1 => self.app_settings.audio_volume,
                        2 => self.app_settings.music_volume,
                        _ => self.app_settings.sfx_volume,
                    };
                    let tl = base.min.x + 230.0 * s;
                    let tr = base.max.x - 96.0 * s;
                    let ty = base.center().y;
                    let track = egui::Rect::from_min_max(
                        egui::pos2(tl, ty - 3.0 * s),
                        egui::pos2(tr, ty + 3.0 * s),
                    );
                    p.rect_filled(
                        sr(track),
                        egui::CornerRadius::from(3.0 * s),
                        egui::Color32::from_rgba_unmultiplied(muted.r(), muted.g(), muted.b(), 70),
                    );
                    let knobx = tl + val * (tr - tl);
                    let fill = egui::Rect::from_min_max(
                        egui::pos2(tl, ty - 3.0 * s),
                        egui::pos2(knobx, ty + 3.0 * s),
                    );
                    p.rect_filled(sr(fill), egui::CornerRadius::from(3.0 * s), accent);
                    p.circle(
                        sp(egui::pos2(knobx, ty)),
                        9.0 * s * sf,
                        if selrow { accent } else { text },
                        egui::Stroke::new(2.0_f32, accent),
                    );
                    p.text(
                        sp(egui::pos2(base.max.x - 22.0 * s, base.center().y)),
                        egui::Align2::RIGHT_CENTER,
                        format!("{}%", (val * 100.0).round() as i32),
                        egui::FontId::proportional(16.0 * s * sf),
                        if selrow { accent } else { muted },
                    );
                    let grab = sr(egui::Rect::from_min_max(
                        egui::pos2(tl - 12.0 * s, ty - 16.0 * s),
                        egui::pos2(tr + 12.0 * s, ty + 16.0 * s),
                    ));
                    if !block && ui.rect_contains_pointer(grab) {
                        let resp = ui.allocate_rect(grab, egui::Sense::click_and_drag());
                        if resp.dragged() || resp.clicked() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                let line = sr(egui::Rect::from_min_max(
                                    egui::pos2(tl, ty),
                                    egui::pos2(tr, ty),
                                ));
                                let frac =
                                    ((pos.x - line.min.x) / line.width().max(1.0)).clamp(0.0, 1.0);
                                drag_set = Some((i, frac));
                                self.prefs_focus = true;
                                self.prefs_row = i;
                            }
                        }
                    }
                } else {
                    let lx = base.max.x - 190.0 * s;
                    let rx = base.max.x - 22.0 * s;
                    let mv: std::borrow::Cow<str> = match i {
                        0 => shorten_device(&self.app_settings.audio_output_device).into(),
                        4 => {
                            if self.app_settings.music_muted {
                                "On".into()
                            } else {
                                "Off".into()
                            }
                        }
                        5 => {
                            if self.app_settings.sfx_muted {
                                "On".into()
                            } else {
                                "Off".into()
                            }
                        }
                        _ => crate::ui_audio::carousel_track_label(
                            self.app_settings.menu_music_track,
                        )
                        .into(),
                    };
                    let mvsize = if i == 0 { 13.5 } else { 16.0 };
                    p.text(
                        sp(egui::pos2((lx + rx) * 0.5, base.center().y)),
                        egui::Align2::CENTER_CENTER,
                        mv.as_ref(),
                        egui::FontId::proportional(mvsize * s * sf),
                        if selrow { accent } else { muted },
                    );
                    let la = egui::Rect::from_center_size(
                        egui::pos2(lx, base.center().y),
                        egui::Vec2::splat(34.0 * s),
                    );
                    let ra = egui::Rect::from_center_size(
                        egui::pos2(rx, base.center().y),
                        egui::Vec2::splat(34.0 * s),
                    );
                    p.text(
                        sp(la.center()),
                        egui::Align2::CENTER_CENTER,
                        "\u{2039}",
                        egui::FontId::proportional(22.0 * s * sf),
                        if selrow { accent } else { muted },
                    );
                    p.text(
                        sp(ra.center()),
                        egui::Align2::CENTER_CENTER,
                        "\u{203A}",
                        egui::FontId::proportional(22.0 * s * sf),
                        if selrow { accent } else { muted },
                    );
                    if !block && ui.rect_contains_pointer(r) {
                        let rowc = ui.allocate_rect(r, egui::Sense::click()).clicked();
                        let lc = ui.allocate_rect(sr(la), egui::Sense::click()).clicked();
                        let rc = ui.allocate_rect(sr(ra), egui::Sense::click()).clicked();
                        if lc || rc || rowc {
                            self.prefs_focus = true;
                            self.prefs_row = i;
                            let dir = if lc { -1 } else { 1 };
                            apply_audio_cycle(&mut self.app_settings, i, dir, &devices);
                            crate::ui_audio::play_move();
                        }
                    }
                }
                y += row_h;
            }
            let mut key_dir = 0i32;
            if self.prefs_focus && !block {
                if nl {
                    key_dir = -1;
                } else if nr || a_edge {
                    key_dir = 1;
                }
            }
            if let Some((ri, frac)) = drag_set {
                match ri {
                    1 => {
                        self.app_settings.audio_volume = frac;
                        set_master_volume(frac);
                    }
                    2 => self.app_settings.music_volume = frac,
                    3 => self.app_settings.sfx_volume = frac,
                    _ => {}
                }
                crate::ui_audio::set_sfx_volume(if self.app_settings.sfx_muted {
                    0.0
                } else {
                    self.app_settings.sfx_volume
                });
                let _ = self.app_settings.save();
            } else if key_dir != 0 {
                let step = 0.05 * key_dir as f32;
                match self.prefs_row {
                    1 => {
                        self.app_settings.audio_volume =
                            (self.app_settings.audio_volume + step).clamp(0.0, 1.0);
                        set_master_volume(self.app_settings.audio_volume);
                    }
                    2 => {
                        self.app_settings.music_volume =
                            (self.app_settings.music_volume + step).clamp(0.0, 1.0)
                    }
                    3 => {
                        self.app_settings.sfx_volume =
                            (self.app_settings.sfx_volume + step).clamp(0.0, 1.0)
                    }
                    r => apply_audio_cycle(&mut self.app_settings, r, key_dir, &devices),
                }
                crate::ui_audio::set_sfx_volume(if self.app_settings.sfx_muted {
                    0.0
                } else {
                    self.app_settings.sfx_volume
                });
                let _ = self.app_settings.save();
                crate::ui_audio::play_move();
            }
        } else if self.settings_tab == SettingsTab::Emulation {
            let firmware = self.firmware_status();
            let rows: Vec<(String, String)> = vec![
                (
                    "Multicore CPU".to_string(),
                    if self.app_settings.multicore {
                        "On"
                    } else {
                        "Off"
                    }
                    .to_string(),
                ),
                (
                    "Async Shader Compile".to_string(),
                    if self.app_settings.async_shaders {
                        "On"
                    } else {
                        "Off"
                    }
                    .to_string(),
                ),
                (
                    "Firmware".to_string(),
                    match &firmware {
                        Ok(Some(installed)) => installed.display_version.clone(),
                        Ok(None) => "Not installed".to_string(),
                        Err(_) => "Unreadable".to_string(),
                    },
                ),
                ("Install Firmware".to_string(), "Choose file".to_string()),
                (
                    "Remove Firmware".to_string(),
                    if matches!(firmware, Ok(Some(_))) { "Remove" } else { "None" }.to_string(),
                ),
            ];
            let mut row = self.prefs_row;
            let hit = pref_value_rows(
                &p,
                ui,
                &sp,
                &sr,
                content,
                s,
                sf,
                ease,
                accent,
                text,
                muted,
                border,
                panel,
                sel,
                self.prefs_focus,
                block,
                &mut row,
                &rows,
                nu,
                nd,
                nl,
                nr,
                a_edge,
            );
            self.prefs_row = row;
            if let Some((ri, _)) = hit {
                match ri {
                    0 => self.app_settings.multicore = !self.app_settings.multicore,
                    1 => {
                        self.app_settings.async_shaders = !self.app_settings.async_shaders;
                        nexium_common::async_compile::set_enabled(self.app_settings.async_shaders);
                    }
                    3 => self.install_firmware_from_settings(ctx),
                    4 => self.remove_firmware_from_settings(),
                    _ => {}
                }
                let _ = self.app_settings.save();
                crate::ui_audio::play_move();
            }
        } else if self.settings_tab == SettingsTab::Logging {
            let levels = crate::app_settings::LogLevel::all();
            let rows: Vec<(String, String)> = vec![(
                "Log Level".to_string(),
                self.app_settings.log_level.label().to_string(),
            )];
            let mut row = self.prefs_row;
            let hit = pref_value_rows(
                &p,
                ui,
                &sp,
                &sr,
                content,
                s,
                sf,
                ease,
                accent,
                text,
                muted,
                border,
                panel,
                sel,
                self.prefs_focus,
                block,
                &mut row,
                &rows,
                nu,
                nd,
                nl,
                nr,
                a_edge,
            );
            self.prefs_row = row;
            if let Some((_, dir)) = hit {
                let idx = levels
                    .iter()
                    .position(|x| *x == self.app_settings.log_level)
                    .unwrap_or(0);
                self.app_settings.log_level =
                    levels[((idx as i32 + dir).rem_euclid(levels.len() as i32)) as usize];
                log::set_max_level(self.app_settings.log_level.to_filter());
                let _ = self.app_settings.save();
                crate::ui_audio::play_move();
            }
        } else {
            p.text(
                sp(content.center() - egui::Vec2::new(0.0, 14.0)),
                egui::Align2::CENTER_CENTER,
                "Not in the console UI yet.",
                egui::FontId::proportional(20.0 * s * sf),
                text,
            );
            p.text(
                sp(content.center() + egui::Vec2::new(0.0, 20.0)),
                egui::Align2::CENTER_CENTER,
                "Switch to Grid View to change these for now.",
                egui::FontId::proportional(15.0 * s * sf),
                muted,
            );
        }

        if !editing && b_edge {
            if self.prefs_focus {
                self.prefs_focus = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else {
                self.show_settings = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            }
        }
        ctx.request_repaint();
    }

    fn draw_perf_overlay(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let now = std::time::Instant::now();
        let cap = 140usize;
        if self
            .perf_sampled
            .map_or(true, |at| now.duration_since(at) >= PERF_GRAPH_INTERVAL)
        {
            self.perf_sampled = Some(now);
            self.perf_hist.push(self.performance.get_fps());
            if self.perf_hist.len() > cap {
                let n = self.perf_hist.len() - cap;
                self.perf_hist.drain(0..n);
            }
        }
        if self
            .perf_readout_at
            .map_or(true, |at| now.duration_since(at) >= PERF_READOUT_INTERVAL)
        {
            self.perf_readout_at = Some(now);
            let (svc, cyc) = self
                .emulation_handle
                .as_ref()
                .and_then(|h| h.stats.try_lock().map(|s| (s.svc_count, s.cycle_count)))
                .unwrap_or((self.perf_readout.2, self.perf_readout.3));
            self.perf_readout = (
                self.performance.get_fps(),
                self.performance.get_frame_time(),
                svc,
                cyc,
            );
        }
        let (fps, ft, svc, cyc) = self.perf_readout;
        let (rw, rh) = self.last_frame_res;
        let (mut fmin, mut fmax) = (f32::MAX, 0.0f32);
        for &v in &self.perf_hist {
            if v > 0.0 {
                fmin = fmin.min(v);
                fmax = fmax.max(v);
            }
        }
        if fmin == f32::MAX {
            fmin = 0.0;
        }

        let fps_col = |v: f32| {
            if v >= 30.0 {
                GREEN
            } else if v >= 20.0 {
                AMBER
            } else {
                DANGER
            }
        };

        let full = ui.max_rect();
        let base_bw = 224.0;
        let base_bh = 138.0;
        let mut sc = self.perf_scale.clamp(0.7, 3.0);
        let margin = 14.0;
        let corner = self.app_settings.perf_overlay_corner;
        let default_pos = egui::pos2(
            if corner.is_left() {
                full.min.x + margin
            } else {
                full.max.x - base_bw * sc - margin
            },
            if corner.is_top() {
                full.min.y + margin
            } else {
                full.max.y - base_bh * sc - margin
            },
        );
        let mut pos = self.perf_pos.unwrap_or(default_pos);
        let bw = base_bw * sc;
        let bh = base_bh * sc;

        let hit = egui::Rect::from_min_size(pos, egui::vec2(bw, bh));
        let resp = ui.interact(
            hit,
            egui::Id::new("perf_overlay_drag"),
            egui::Sense::click_and_drag(),
        );
        let grip = (20.0 * sc).clamp(16.0, 40.0);
        let grip_rect =
            egui::Rect::from_min_size(egui::pos2(pos.x + bw - grip, pos.y), egui::vec2(grip, grip));
        let gresp = ui.interact(
            grip_rect,
            egui::Id::new("perf_resize"),
            egui::Sense::click_and_drag(),
        );
        if gresp.hovered() || gresp.dragged() {
            ctx.set_cursor_icon(egui::CursorIcon::ResizeNeSw);
        }
        let resizing = gresp.dragged() || gresp.drag_started();
        if resizing {
            if let Some(mp) = ctx.pointer_interact_pos() {
                let new_bw = (mp.x - pos.x).clamp(base_bw * 0.7, base_bw * 3.0);
                self.perf_scale = new_bw / base_bw;
                self.perf_drag = false;
            }
        }
        sc = self.perf_scale.clamp(0.7, 3.0);
        let bw = base_bw * sc;
        let bh = base_bh * sc;

        let over_grip = ctx
            .pointer_interact_pos()
            .map_or(false, |mp| grip_rect.contains(mp));
        if !resizing {
            if resp.drag_started() && !over_grip {
                self.perf_drag = true;
                if let Some(mp) = ctx.pointer_interact_pos() {
                    self.perf_grab_off = mp - pos;
                }
            }
            if self.perf_drag {
                if let Some(mp) = ctx.pointer_interact_pos() {
                    pos = mp - self.perf_grab_off;
                }
                if !ui.input(|i| i.pointer.any_down()) {
                    self.perf_drag = false;
                }
            }
        }
        pos.x = pos.x.clamp(
            full.min.x + 4.0,
            (full.max.x - bw - 4.0).max(full.min.x + 4.0),
        );
        pos.y = pos.y.clamp(
            full.min.y + 4.0,
            (full.max.y - bh - 4.0).max(full.min.y + 4.0),
        );
        if self.perf_drag || self.perf_pos.is_some() {
            self.perf_pos = Some(pos);
        }
        let rect = egui::Rect::from_min_size(pos, egui::vec2(bw, bh));
        self.native_overlay_rect = Some(rect.expand(3.0));

        let p = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("perf_overlay"),
        ));
        p.rect_filled(
            rect,
            egui::CornerRadius::from(6.0 * sc),
            egui::Color32::from_rgba_unmultiplied(0x0A, 0x0A, 0x0D, 236),
        );
        p.rect_stroke(
            rect,
            egui::CornerRadius::from(6.0 * sc),
            egui::Stroke::new(
                1.0_f32,
                if resp.hovered() || self.perf_drag || resizing {
                    egui::Color32::from_gray(110)
                } else {
                    egui::Color32::from_gray(54)
                },
            ),
            egui::StrokeKind::Outside,
        );
        let mono = |sz: f32| egui::FontId::monospace(sz * sc);
        let white = egui::Color32::from_rgb(0xE8, 0xE8, 0xEC);
        let lx = rect.min.x + 9.0 * sc;
        let rxr = rect.max.x - 9.0 * sc;
        let mut y = rect.min.y + 7.0 * sc;
        p.text(
            egui::pos2(lx, y),
            egui::Align2::LEFT_TOP,
            if nexium_common::speed_limit::enabled() { "NeXium" } else { "NeXium | Unlocked" },
            mono(10.5),
            white,
        );
        p.text(
            egui::pos2(rxr, y),
            egui::Align2::RIGHT_TOP,
            format!("[{}x{}]", rw, rh),
            mono(10.5),
            white,
        );
        y += 15.0 * sc;
        p.text(
            egui::pos2(lx, y),
            egui::Align2::LEFT_TOP,
            format!("FPS: {:>6.1}", fps),
            mono(10.5),
            fps_col(fps),
        );
        p.text(
            egui::pos2(rxr, y),
            egui::Align2::RIGHT_TOP,
            format!("[{:.0} {:.0}]", fmin, fmax),
            mono(10.0),
            MUTED,
        );
        y += 14.0 * sc;
        p.text(
            egui::pos2(lx, y),
            egui::Align2::LEFT_TOP,
            format!("Frame:{:>5.1}ms", ft),
            mono(10.5),
            white,
        );
        y += 14.0 * sc;
        p.text(
            egui::pos2(lx, y),
            egui::Align2::LEFT_TOP,
            format!("SVCs:{:>10}", svc),
            mono(10.5),
            egui::Color32::from_rgb(0x4C, 0xC2, 0xF0),
        );
        y += 14.0 * sc;
        let cyc_str = if cyc >= 1_000_000_000 {
            format!("{:.2}B", cyc as f64 / 1e9)
        } else if cyc >= 1_000_000 {
            format!("{:.1}M", cyc as f64 / 1e6)
        } else {
            format!("{}", cyc)
        };
        p.text(
            egui::pos2(lx, y),
            egui::Align2::LEFT_TOP,
            format!("Cyc: {:>10}", cyc_str),
            mono(10.5),
            egui::Color32::from_rgb(0x86, 0xD0, 0x5A),
        );
        y += 17.0 * sc;

        let graph =
            egui::Rect::from_min_max(egui::pos2(lx, y), egui::pos2(rxr, rect.max.y - 8.0 * sc));
        p.rect_filled(
            graph,
            egui::CornerRadius::from(2.0 * sc),
            egui::Color32::from_black_alpha(140),
        );
        for k in 1..4 {
            let gx = graph.min.x + graph.width() * k as f32 / 4.0;
            p.line_segment(
                [egui::pos2(gx, graph.min.y), egui::pos2(gx, graph.max.y)],
                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(30)),
            );
        }
        let n = self.perf_hist.len();
        if n > 1 {
            let scale = 70.0f32.max(fmax);
            let bwbar = graph.width() / cap as f32;
            for (i, &v) in self.perf_hist.iter().enumerate() {
                let hnorm = (v / scale).clamp(0.0, 1.0);
                let bx = graph.min.x + i as f32 * bwbar;
                let by = graph.max.y - hnorm * graph.height();
                p.line_segment(
                    [egui::pos2(bx, graph.max.y), egui::pos2(bx, by)],
                    egui::Stroke::new(bwbar.max(1.0), fps_col(v)),
                );
            }
        }
        let gc = egui::Color32::from_gray(if gresp.hovered() || resizing { 150 } else { 90 });
        for k in 0..2 {
            let off = 4.0 * sc + k as f32 * 4.0 * sc;
            p.line_segment(
                [
                    egui::pos2(rect.max.x - off, rect.min.y + 3.0 * sc),
                    egui::pos2(rect.max.x - 3.0 * sc, rect.min.y + off),
                ],
                egui::Stroke::new(1.5 * sc, gc),
            );
        }
    }

    fn resolve_confirm(&mut self, confirmed: bool) {
        let Some(dlg) = self.confirm.take() else {
            return;
        };
        if !confirmed {
            return;
        }
        match dlg.kind {
            ConfirmKind::CloseGame => {
                self.stop_emulation();
                self.teardown_at = Some(std::time::Instant::now());
                self.pending_boot = None;
            }
            ConfirmKind::LaunchGame(path) => {
                self.stop_emulation();
                self.teardown_at = Some(std::time::Instant::now());
                self.pending_boot = Some(path);
            }
            ConfirmKind::QuickLaunch(path) => {
                if let Some(idx) = self.library.index_of_path(&std::path::PathBuf::from(&path)) {
                    let n = self.library.move_to_front(idx);
                    self.carousel.selected = n + crate::carousel::CS_FRONT;
                    self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                }
                self.show_profile = false;
                if self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel {
                    self.app_settings.view_mode = crate::app_settings::ViewMode::Carousel;
                    let _ = self.app_settings.save();
                }
                self.pending_quick = Some(path);
            }
            ConfirmKind::DeleteList(li) => {
                if li < self.app_settings.carousel_lists.len() {
                    let name = self.app_settings.carousel_lists[li].name.clone();
                    self.app_settings.carousel_lists.remove(li);
                    self.app_settings.carousel_order.retain(
                        |e| !matches!(e, crate::app_settings::CarouselRef::List(n) if *n == name),
                    );
                    let _ = self.app_settings.save();
                    self.cs_selected = 0;
                    self.cs_build_grab = None;
                }
            }
            ConfirmKind::UpdateInstall => {
                if let Some(rel) = self.updater.available_release() {
                    self.update_restart_shown = false;
                    self.updater.start_download(&rel);
                }
            }
            ConfirmKind::UpdateRestart => {
                self.stop_emulation();
                match self.updater.install_and_relaunch() {
                    Ok(()) => {
                        std::process::exit(0);
                    }
                    Err(e) => {
                        log::error!("update install failed: {}", e);
                    }
                }
            }
        }
    }

    fn quick_launch_tick(&mut self, ctx: &egui::Context) {
        if self.pending_quick.is_none() {
            return;
        }
        if self.show_profile || self.profile_anim > 0.02 || self.nextendo_overlay_active() {
            ctx.request_repaint();
            return;
        }
        let path = self.pending_quick.take().unwrap();
        let running = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_running());
        if running || crate::boot::emu_alive() {
            self.stop_emulation();
            self.teardown_at = Some(std::time::Instant::now());
            self.pending_boot = Some(path);
        } else {
            let now = ctx.input(|i| i.time) as f32;
            self.carousel.boot_stage = crate::carousel::BootStage::Transitioning {
                game_index: self.carousel.selected,
                start_time: now,
                launch_path: path,
            };
        }
    }

    fn handle_modal_input(&mut self, ctx: &egui::Context) {
        if self.confirm.is_none() {
            self.modal_hold = false;
            return;
        }
        use crate::controller_config::SwitchButton;
        let now = ctx.input(|i| i.time);
        let (kb_left, kb_right, kb_enter, kb_esc) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        let li = self.last_input;
        let ready = now - self.modal_cd > 0.18;
        let mut mv = if kb_left {
            -1
        } else if kb_right {
            1
        } else {
            0
        };
        if ready && li.connected {
            let lx = li.lx();
            if li.is(SwitchButton::DLeft) || lx < -0.5 {
                mv = -1;
                self.modal_cd = now;
            } else if li.is(SwitchButton::DRight) || lx > 0.5 {
                mv = 1;
                self.modal_cd = now;
            }
        }
        if mv != 0 {
            if let Some(c) = self.confirm.as_mut() {
                let ns = if mv < 0 { 0 } else { 1 };
                if ns != c.selected {
                    c.selected = ns;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Move);
                }
            }
        }
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let a_edge = kb_enter || (gp_a && !self.modal_hold);
        let b_edge = kb_esc || (gp_b && !self.modal_hold);
        self.modal_hold = gp_a || gp_b;
        if b_edge {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            self.confirm = None;
            self.carousel.boot_stage = crate::carousel::BootStage::None;
        } else if a_edge {
            let confirmed = self.confirm.as_ref().map_or(false, |c| c.selected == 1);
            crate::ui_audio::play(if confirmed {
                crate::ui_audio::Sfx::WhistleOk
            } else {
                crate::ui_audio::Sfx::Back
            });
            self.resolve_confirm(confirmed);
        }
    }

    fn teardown_tick(&mut self, ctx: &egui::Context) {
        if let Some(start) = self.teardown_at {
            ctx.request_repaint();
            let elapsed = start.elapsed().as_secs_f32();
            let dead = !crate::boot::emu_alive();
            if elapsed >= 8.0 && !dead {
                self.teardown_at = None;
                self.pending_boot = None;
            } else if elapsed >= 0.5 && dead {
                self.teardown_at = None;
                if let Some(path) = self.pending_boot.take() {
                    let now = ctx.input(|i| i.time) as f32;
                    self.carousel.boot_stage = crate::carousel::BootStage::Transitioning {
                        game_index: self.carousel.selected,
                        start_time: now,
                        launch_path: path,
                    };
                }
            }
        }
    }

    fn draw_modal(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let active = self.confirm.is_some() || self.teardown_at.is_some();
        let target = if active { 1.0 } else { 0.0 };
        let speed = if target > self.modal_anim { 12.0 } else { 10.0 };
        self.modal_anim += (target - self.modal_anim) * (dt * speed).min(1.0);
        if (self.modal_anim - target).abs() < 0.003 {
            self.modal_anim = target;
        }

        if active {
            let snap = if self.teardown_at.is_some() {
                ModalSnap {
                    teardown: true,
                    title: String::new(),
                    body: String::new(),
                    label: String::new(),
                }
            } else if let Some(c) = &self.confirm {
                ModalSnap {
                    teardown: false,
                    title: c.title.clone(),
                    body: c.body.clone(),
                    label: c.confirm_label.clone(),
                }
            } else {
                return;
            };
            self.modal_snap = Some(snap);
        }
        if self.modal_anim <= 0.004 {
            if !active {
                self.modal_snap = None;
            }
            return;
        }
        let Some(snap) = self.modal_snap.as_ref() else {
            return;
        };
        let (is_teardown, title, body, label) = (
            snap.teardown,
            snap.title.clone(),
            snap.body.clone(),
            snap.label.clone(),
        );

        let a = self.modal_anim.clamp(0.0, 1.0);
        let ease = a * a * (3.0 - 2.0 * a);

        let light = self.app_settings.light_mode;
        let panel = if light {
            Color32::from_rgb(0xF5, 0xF5, 0xF9)
        } else {
            Color32::from_rgb(0x18, 0x18, 0x22)
        };
        let text = if light {
            Color32::from_rgb(0x1E, 0x1E, 0x28)
        } else {
            Color32::from_rgb(0xEC, 0xEC, 0xF0)
        };
        let muted = if light {
            Color32::from_rgb(0x60, 0x60, 0x6A)
        } else {
            Color32::from_rgb(0x9A, 0x9A, 0xA6)
        };
        let border = if light {
            Color32::from_rgb(0xC6, 0xC6, 0xD0)
        } else {
            Color32::from_rgb(0x32, 0x32, 0x3E)
        };
        let btn_fill = if light {
            Color32::from_rgb(0xEA, 0xEA, 0xF0)
        } else {
            Color32::from_rgb(0x24, 0x24, 0x2E)
        };
        let accent = self.theme_accent();

        let screen = ctx.viewport_rect();
        let mut p = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("modal_overlay"),
        ));
        p.set_opacity(ease);
        p.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(195));

        let pop = 0.90 + 0.10 * ease;
        let body_font = FontId::proportional(19.0);
        let lines: Vec<&str> = body.split('\n').collect();
        let body_w = lines
            .iter()
            .map(|ln| {
                ui.fonts_mut(|f| {
                    f.layout_no_wrap(ln.to_string(), body_font.clone(), text)
                        .size()
                        .x
                })
            })
            .fold(0.0_f32, f32::max);
        let base_w = (body_w + 72.0).clamp(380.0, (screen.width() - 60.0).max(400.0));
        let w = base_w * pop;
        let extra_lines = lines.len().saturating_sub(1) as f32;
        let h = (224.0 + extra_lines * 26.0) * pop;
        let box_rect = egui::Rect::from_center_size(screen.center(), Vec2::new(w, h));
        p.rect_filled(
            box_rect.translate(Vec2::new(0.0, 10.0)),
            CornerRadius::same(18),
            Color32::from_black_alpha(90),
        );
        p.rect_filled(box_rect, CornerRadius::same(18), panel);
        p.rect_stroke(
            box_rect,
            CornerRadius::same(18),
            Stroke::new(1.5_f32, border),
            egui::StrokeKind::Outside,
        );

        if is_teardown {
            p.text(
                box_rect.center() - Vec2::new(0.0, 14.0 * pop),
                egui::Align2::CENTER_CENTER,
                "Please Wait…",
                FontId::proportional(25.0 * pop),
                text,
            );
            p.text(
                box_rect.center() + Vec2::new(0.0, 24.0 * pop),
                egui::Align2::CENTER_CENTER,
                "Shutting down the current game",
                FontId::proportional(14.0 * pop),
                muted,
            );
            ctx.request_repaint();
            return;
        }

        p.text(
            egui::pos2(box_rect.center().x, box_rect.min.y + 40.0 * pop),
            egui::Align2::CENTER_CENTER,
            &title,
            FontId::proportional(15.0 * pop),
            muted,
        );
        let line_h = 26.0 * pop;
        let block_top = box_rect.center().y - 14.0 * pop - extra_lines * line_h * 0.5;
        for (i, ln) in lines.iter().enumerate() {
            p.text(
                egui::pos2(box_rect.center().x, block_top + i as f32 * line_h),
                egui::Align2::CENTER_CENTER,
                *ln,
                FontId::proportional(19.0 * pop),
                text,
            );
        }

        let btn_w = w * 0.42;
        let btn_h = 46.0 * pop;
        let by = box_rect.max.y - btn_h - 18.0 * pop;
        let gap = w * 0.05;
        let cancel_rect = egui::Rect::from_min_size(
            egui::pos2(box_rect.center().x - gap * 0.5 - btn_w, by),
            Vec2::new(btn_w, btn_h),
        );
        let ok_rect = egui::Rect::from_min_size(
            egui::pos2(box_rect.center().x + gap * 0.5, by),
            Vec2::new(btn_w, btn_h),
        );

        let selected = self.confirm.as_ref().map_or(0, |c| c.selected);
        let draw_btn = |rect: egui::Rect, label: &str, sel: bool| {
            let rounding = CornerRadius::same(10);
            p.rect_filled(rect, rounding, btn_fill);
            if sel {
                p.rect_filled(
                    rect,
                    rounding,
                    Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 42),
                );
                p.rect_stroke(
                    rect,
                    rounding,
                    Stroke::new(2.6_f32, accent),
                    egui::StrokeKind::Outside,
                );
            } else {
                p.rect_stroke(
                    rect,
                    rounding,
                    Stroke::new(1.2_f32, border),
                    egui::StrokeKind::Outside,
                );
            }
            let col = if sel {
                let f = |x: u8| (x as f32 + (255.0 - x as f32) * 0.2) as u8;
                Color32::from_rgb(f(accent.r()), f(accent.g()), f(accent.b()))
            } else {
                text
            };
            p.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional(18.0 * pop),
                col,
            );
        };
        draw_btn(cancel_rect, "Cancel", selected == 0);
        draw_btn(ok_rect, &label, selected == 1);

        if self.confirm.is_some() {
            let cancel_resp = ui.allocate_rect(cancel_rect, egui::Sense::click());
            let ok_resp = ui.allocate_rect(ok_rect, egui::Sense::click());
            if let Some(c) = self.confirm.as_mut() {
                if cancel_resp.hovered() {
                    c.selected = 0;
                }
                if ok_resp.hovered() {
                    c.selected = 1;
                }
            }
            if cancel_resp.clicked() {
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                self.confirm = None;
                self.carousel.boot_stage = crate::carousel::BootStage::None;
            } else if ok_resp.clicked() {
                crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
                self.resolve_confirm(true);
            }
        }
        ctx.request_repaint();
    }

    fn native_game_visible(&self, carousel_mode: bool) -> bool {
        !carousel_mode && self.emulation_handle.as_ref().is_some_and(|h| h.is_running() && !h.is_paused())
            && !self.modal_active() && !self.show_settings && !self.show_profile
            && self.profile_anim == 0.0 && !self.nextendo_overlay_active()
            && self.pause_anim.is_none() && self.resume_anim.is_none()
            && !self.debugger.show_memory && !self.debugger.show_registers && !self.debugger.show_disasm
            && !self.debugger.show_logs && !self.debugger.show_performance && !self.debugger.show_wait_tree
    }

    fn game_display(&self) -> Option<(egui::TextureId, Vec2)> {
        if let Some(window) = self.native_game.as_ref().filter(|window| window.active) {
            return self.game_texture.as_ref().map(|texture| (texture.id(),
                Vec2::new(window.dimensions[0] as f32, window.dimensions[1] as f32)));
        }
        if let Some(t) = &self.game_texture_native {
            Some((t.id, Vec2::new(t.width as f32, t.height as f32)))
        } else {
            self.game_texture.as_ref().map(|t| (t.id(), t.size_vec2()))
        }
    }

    fn request_launch(&mut self, path: String, ctx: &egui::Context, running: bool) {
        if running {
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
            self.confirm = Some(ConfirmDialog {
                title: "Launch Game".into(),
                body: "Close the current game and launch this one?".into(),
                confirm_label: "Launch".into(),
                selected: 0,
                kind: ConfirmKind::LaunchGame(path),
            });
            self.modal_hold = true;
        } else if crate::boot::emu_alive() {
            self.teardown_at = Some(std::time::Instant::now());
            self.pending_boot = Some(path);
        } else {
            self.nro_path = path;
            self.boot_nro(ctx);
        }
    }

    fn recenter_motion(&mut self) {
        let message = if !self.app_settings.motion_enabled {
            MOTION_OFF
        } else if !self.motion_present() {
            MOTION_MISSING
        } else {
            nexium_core::hid_state::recenter_motion();
            if let Some(ib) = self.input.as_mut() {
                ib.rumble_motion(20000, 20000, 45);
            }
            log::info!("motion: recentered");
            MOTION_RECENTERED
        };
        self.motion_toast = Some((message, std::time::Instant::now()));
        self.egui_ctx.request_repaint();
    }

    fn calibrate_motion(&mut self) {
        let message = if !self.app_settings.motion_enabled {
            MOTION_OFF
        } else if !self.motion_present() {
            MOTION_MISSING
        } else {
            nexium_core::hid_motion::request_calibration();
            log::info!("motion: calibration requested");
            if nexium_core::hid_motion::stale_sources().contains(&true) {
                MOTION_NO_DATA
            } else if nexium_core::hid_motion::MotionSource::ALL
                .into_iter()
                .all(nexium_core::hid_motion::source_at_rest)
            {
                MOTION_CALIBRATING
            } else {
                MOTION_PUT_DOWN
            }
        };
        self.motion_toast = Some((message, std::time::Instant::now()));
        self.egui_ctx.request_repaint();
    }

    fn motion_recently_recentered(&self) -> bool {
        self.motion_toast.is_some_and(|(message, at)| {
            message == MOTION_RECENTERED && at.elapsed() < std::time::Duration::from_millis(1200)
        })
    }

    fn draw_motion_toast(&mut self, ctx: &egui::Context) {
        let Some((message, since)) = self.motion_toast else {
            return;
        };
        let el = since.elapsed().as_secs_f32();
        if el > 1.6 {
            self.motion_toast = None;
            return;
        }
        let fade = (el.min(0.15) / 0.15).min((1.6 - el) / 0.3).clamp(0.0, 1.0);
        let rect = motion_toast_rect(ctx.viewport_rect(), self.download_toast.is_some());
        let mut tp = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("motion_toast"),
        ));
        tp.set_opacity(fade);
        tp.rect_filled(
            rect.translate(egui::Vec2::new(0.0, 4.0)),
            egui::CornerRadius::same(14),
            egui::Color32::from_black_alpha(90),
        );
        tp.rect_filled(rect, egui::CornerRadius::same(14), BG_RAISED);
        tp.rect_stroke(
            rect,
            egui::CornerRadius::same(14),
            egui::Stroke::new(1.5_f32, self.theme_accent()),
            egui::StrokeKind::Outside,
        );
        tp.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            message,
            egui::FontId::proportional(15.0),
            TEXT,
        );
        ctx.request_repaint();
    }

    fn draw_motion_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, compact: bool) {
        let enabled = self.app_settings.motion_enabled;
        let resp = ui.add(
            egui::Label::new(
                egui::RichText::new(bar_toggle_label("Motion", enabled, compact))
                    .size(12.0)
                    .color(if enabled { GREEN } else { AMBER }),
            )
            .sense(egui::Sense::click()),
        );
        if resp.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        let mut hover = format!(
            "Toggle motion controls. Recenter: {}",
            motion_recenter_text(
                &self.app_settings,
                &self.controller_config,
                self.motion_button_missing(),
            )
        );
        if compact {
            hover = format!("{}. {}", bar_toggle_label("Motion", enabled, false), hover);
        }
        if enabled != self.startup_motion_enabled {
            hover = format!("{}. {}", hover, MOTION_RESTART);
        }
        if resp.on_hover_text(hover).clicked() {
            self.app_settings.motion_enabled = !enabled;
            let _ = self.app_settings.save();
        }
    }

    fn vibration_status(&self) -> String {
        self.input.as_ref().map_or_else(
            || VIBRATION_NO_CONTROLLER.to_string(),
            |ib| ib.rumble_status_text(self.app_settings.vibration_enabled),
        )
    }

    fn test_vibration(&self) {
        if !self.app_settings.vibration_enabled {
            return;
        }
        if let Some(ib) = self.input.as_ref() {
            ib.preview_rumble(self.app_settings.vibration_strength);
        }
    }

    fn draw_vibration_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, compact: bool) {
        let enabled = self.app_settings.vibration_enabled;
        let resp = ui.add(
            egui::Label::new(
                egui::RichText::new(bar_toggle_label("Vibration", enabled, compact))
                    .size(12.0)
                    .color(if enabled { GREEN } else { AMBER }),
            )
            .sense(egui::Sense::click()),
        );
        if resp.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        let mut hover = format!(
            "Toggle controller vibration (strength {}%)",
            self.app_settings.vibration_strength
        );
        if compact {
            hover = format!("{}. {}", bar_toggle_label("Vibration", enabled, false), hover);
        }
        if resp.on_hover_text(hover).clicked() {
            self.app_settings.vibration_enabled = !enabled;
            let _ = self.app_settings.save();
            self.test_vibration();
        }
    }

    fn draw_docked_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, compact: bool) {
        let docked = nexium_core::hid_state::is_docked();
        let (mode_icon, mode_txt, mode_col) = if docked {
            (StatusIcon::Dock, "Docked", GREEN)
        } else {
            (StatusIcon::Handheld, "Handheld", AMBER)
        };
        let mode_resp = if compact {
            status_icon(ui, mode_icon, mode_col)
        } else {
            let label_resp = ui.add(
                egui::Label::new(egui::RichText::new(mode_txt).size(12.0).color(mode_col))
                    .sense(egui::Sense::click()),
            );
            ui.add_space(2.0);
            let icon_resp = status_icon(ui, mode_icon, mode_col);
            label_resp.union(icon_resp)
        };
        if mode_resp.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        let mut hover =
            "Toggle Docked / Handheld (Pro Controller vs Handheld; a Joy-Con pair stays Dual Joy-Con)"
                .to_string();
        if compact {
            hover = format!("{}. {}", mode_txt, hover);
        }
        if mode_resp.on_hover_text(hover).clicked() {
            let docked = !docked;
            nexium_core::hid_state::set_docked(docked);
            self.app_settings.docked = docked;
            let _ = self.app_settings.save();
        }
    }

    fn draw_game_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        fn bar_item(
            ctx: &egui::Context,
            ui: &mut egui::Ui,
            text: String,
            color: Color32,
            hover: &str,
        ) -> egui::Response {
            let resp = ui
                .add(
                    egui::Label::new(egui::RichText::new(text).size(12.0).color(color))
                        .sense(egui::Sense::click()),
                )
                .on_hover_text(hover);
            if resp.hovered() {
                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            resp
        }
        fn bar_divider(ui: &mut egui::Ui) {
            ui.add_space(8.0);
            ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
            ui.add_space(8.0);
        }
        egui::Panel::bottom("gamebar")
            .exact_size(26.0)
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                    .stroke(Stroke::new(1.0_f32, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(12.0);
                    let gpu_name = self.active_gpu_name();
                    let building = nexium_common::shader_progress::in_flight();
                    let motion_enabled = self.app_settings.motion_enabled;
                    let vibration_enabled = self.app_settings.vibration_enabled;
                    let (name_saving, state_saving) = ui.fonts_mut(|f| {
                        let mut width = |text: String| {
                            f.layout_no_wrap(text, egui::FontId::proportional(12.0), MUTED)
                                .size()
                                .x
                        };
                        let mut name_saving = width(format!("  {}", gpu_name));
                        if building > 0 {
                            name_saving += width(shader_build_label(building, false))
                                - width(shader_build_label(building, true));
                        }
                        let mut state_saving = width(bar_toggle_label("Motion", motion_enabled, false))
                            - width(bar_toggle_label("Motion", motion_enabled, true));
                        state_saving += width(bar_toggle_label("Vibration", vibration_enabled, false))
                            - width(bar_toggle_label("Vibration", vibration_enabled, true));
                        (name_saving, state_saving)
                    });
                    let compact = self.game_bar_spare < if self.game_bar_compact { 24.0 } else { 0.0 };
                    let terse = self.game_bar_terse_spare < if self.game_bar_terse { 24.0 } else { 0.0 };
                    let backend = self.app_settings.gpu_backend.label();
                    if compact {
                        ui.label(egui::RichText::new(backend).size(12.0).color(MUTED))
                            .on_hover_text(gpu_name);
                    } else {
                        ui.label(
                            egui::RichText::new(format!("{}  {}", backend, gpu_name))
                                .size(12.0)
                                .color(MUTED),
                        );
                    }
                    bar_divider(ui);
                    self.draw_docked_toggle(ctx, ui, false);
                    bar_divider(ui);
                    self.draw_motion_toggle(ctx, ui, terse);
                    bar_divider(ui);
                    self.draw_vibration_toggle(ctx, ui, terse);
                    bar_divider(ui);
                    let aspect = self.app_settings.aspect;
                    if bar_item(
                        ctx,
                        ui,
                        format!("Aspect: {}", aspect.label()),
                        TEXT,
                        "Cycle aspect mode",
                    )
                    .clicked()
                    {
                        let all = crate::app_settings::AspectMode::all();
                        let index = all.iter().position(|mode| *mode == aspect).unwrap_or(0);
                        self.app_settings.aspect = all[(index + 1) % all.len()];
                        let _ = self.app_settings.save();
                    }
                    bar_divider(ui);
                    let filter = self.app_settings.filter;
                    if bar_item(
                        ctx,
                        ui,
                        format!("Filter: {}", filter.label()),
                        TEXT,
                        "Cycle upscaling filter",
                    )
                    .clicked()
                    {
                        let all = FilterMode::all();
                        let index = all.iter().position(|mode| *mode == filter).unwrap_or(0);
                        self.app_settings.filter = all[(index + 1) % all.len()];
                        let _ = self.app_settings.save();
                    }
                    if self.app_settings.filter == FilterMode::Fsr {
                        ui.add_space(8.0);
                        ui.scope(|ui| {
                            ui.spacing_mut().slider_width = 64.0;
                            let slider = if ui.available_width() < 720.0 {
                                ui.add(egui::DragValue::new(&mut self.app_settings.fsr_sharpness)
                                    .range(0..=100).prefix("Sharp: ").suffix("%"))
                            } else {
                                ui.add(egui::Slider::new(&mut self.app_settings.fsr_sharpness, 0..=100)
                                    .suffix("%").text("Sharpness").trailing_fill(true))
                            }.on_hover_text("Drag to adjust FSR sharpness: 0% disables sharpening, 100% is strongest");
                            if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                                let _ = self.app_settings.save();
                            }
                        });
                    }
                    bar_divider(ui);
                    let vsync = self.app_settings.vsync;
                    if bar_item(
                        ctx,
                        ui,
                        format!("VSync: {}", if vsync { "On" } else { "Off" }),
                        if vsync { GREEN } else { AMBER },
                        "Toggle host VSync",
                    )
                    .clicked()
                    {
                        self.app_settings.vsync = !vsync;
                        let _ = self.app_settings.save();
                    }
                    bar_divider(ui);
                    let limited = nexium_common::speed_limit::enabled();
                    if bar_item(
                        ctx,
                        ui,
                        format!("Speed: {}", if limited { "Limited" } else { "Unlocked" }),
                        if limited { TEXT } else { AMBER },
                        "Toggle the emulation speed limit (Ctrl+U)",
                    )
                    .clicked()
                    {
                        nexium_common::speed_limit::toggle();
                        if let Some(path) = &self.playing_path {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Title(
                                self.game_window_title(path),
                            ));
                        }
                    }

                    bar_divider(ui);
                    let fps = self.performance.get_fps();
                    if bar_item(
                        ctx,
                        ui,
                        format!("{fps:.1} FPS"),
                        if fps >= 55.0 { GREEN } else if fps >= 28.0 { AMBER } else { DANGER },
                        "Toggle performance overlay (Ctrl+O)",
                    ).clicked() {
                        self.app_settings.perf_overlay_hidden = !self.app_settings.perf_overlay_hidden;
                        let _ = self.app_settings.save();
                    }
                    if building > 0 {
                        bar_divider(ui);
                        let label = ui.label(egui::RichText::new(shader_build_label(building, compact)).size(12.0).color(AMBER));
                        if compact {
                            label.on_hover_text(shader_build_label(building, false));
                        }
                    } else if nexium_common::shader_progress::recently_active() {
                        bar_divider(ui);
                        ui.label(egui::RichText::new(format!("Built {} shaders", nexium_common::shader_progress::burst_built())).size(12.0).color(MUTED));
                    }

                    let left_end = ui.min_rect().max.x;
                    let right_start = ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(12.0);
                        if bar_item(ctx, ui, "Hide".to_string(), MUTED, "Hide this bar (Ctrl+B shows it again)")
                            .clicked()
                        {
                            self.app_settings.game_bar = false;
                            let _ = self.app_settings.save();
                        }
                        bar_divider(ui);
                        let mut volume = self.app_settings.audio_volume;
                        ui.spacing_mut().slider_width = 110.0;
                        let slider = ui.add(
                            egui::Slider::new(&mut volume, 0.0..=1.0)
                                .show_value(false)
                                .trailing_fill(true),
                        );
                        if slider.changed() {
                            self.app_settings.audio_volume = volume;
                            set_master_volume(volume);
                        }
                        if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                            let _ = self.app_settings.save();
                        }
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!("Volume {:.0}%", volume * 100.0))
                                .size(12.0)
                                .color(TEXT),
                        );
                        ui.min_rect().min.x
                    }).inner;
                    let spare = right_start - left_end - 12.0;
                    self.game_bar_spare = spare
                        - if terse {
                            name_saving + state_saving
                        } else if compact {
                            name_saving
                        } else {
                            0.0
                        };
                    self.game_bar_terse_spare = self.game_bar_spare + name_saving;
                    self.game_bar_compact = compact;
                    self.game_bar_terse = terse;
                });
            });
    }

    fn draw_game_bar_reveal_tab(&mut self, ctx: &egui::Context) {
        let screen = ctx.viewport_rect();
        let near_bottom = ctx
            .input(|input| input.pointer.hover_pos())
            .is_some_and(|pos| pos.y >= screen.max.y - 10.0);
        let tab = egui::Rect::from_min_size(
            egui::pos2(screen.center().x - 26.0, screen.max.y - 14.0),
            egui::vec2(52.0, 14.0),
        );
        let hovering_tab = ctx
            .input(|input| input.pointer.hover_pos())
            .is_some_and(|pos| tab.contains(pos));
        if !near_bottom && !hovering_tab {
            return;
        }
        self.game_bar_tab_rect = Some(tab.expand(3.0));
        let response = egui::Area::new(egui::Id::new("gamebar_reveal"))
            .order(egui::Order::Foreground)
            .fixed_pos(tab.min)
            .show(ctx, |ui| {
                let (rect, response) = ui.allocate_exact_size(tab.size(), egui::Sense::click());
                ui.painter().rect(
                    rect,
                    CornerRadius { nw: 5, ne: 5, sw: 0, se: 0 },
                    Color32::from_rgb(0x0C, 0x0C, 0x0E),
                    Stroke::new(1.0_f32, BORDER),
                    egui::StrokeKind::Inside,
                );
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "▴ bar",
                    FontId::proportional(10.5),
                    if response.hovered() { TEXT } else { MUTED },
                );
                response
            })
            .inner;
        if response.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if response.on_hover_text("Show the game bar (Ctrl+B)").clicked() {
            self.app_settings.game_bar = true;
            let _ = self.app_settings.save();
        }
    }

    fn active_gpu_name(&self) -> String {
        self.wgpu_state
            .as_ref()
            .map_or_else(|| "Unknown".into(), |state| state.adapter.get_info().name)
    }

    fn game_window_title(&self, path: &std::path::Path) -> String {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let meta = if ext == "nro" {
            nexium_loader::read_nro_metadata(path)
        } else {
            nexium_loader::read_container_metadata(path)
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();
        let (game, mut version) = match meta {
            Some(m) => {
                let name = if m.title.is_empty() { stem } else { m.title };
                (name, m.version)
            }
            None => (stem, String::new()),
        };
        if ext != "nro" {
            if let Ok(Some(title_id)) = nexium_loader::read_application_title_id(path) {
                if let Ok(content) = nexium_loader::content::list_game_content(&nexium_common::paths::content_dir(), title_id) {
                    if let Some(update) = content.entries.iter().find(|entry| {
                        entry.enabled && entry.kind == nexium_loader::content::ContentKind::Update
                    }) {
                        if !update.display_version.trim().is_empty() {
                            version = update.display_version.trim().to_string();
                        }
                    }
                }
            }
        }
        let mut title = format!("NeXium {} | {}", env!("CARGO_PKG_VERSION"), game);
        if !version.is_empty() {
            title.push_str(&format!(" {}", version));
        }
        if let Some(rs) = self.wgpu_state.as_ref() {
            let info = rs.adapter.get_info();
            title.push_str(&format!(" | {:?} | {}", info.backend, info.name));
        }
        if !nexium_common::speed_limit::enabled() {
            title.push_str(" | Unlocked");
        }
        title
    }

    fn boot_nro(&mut self, ctx: &egui::Context) {
        if self.nro_path.is_empty() {
            return;
        }
        let backend = self.app_settings.effective_cpu_backend().to_cpu_kind();
        log::info!("Boot: NRO={} CPU={}", self.nro_path, backend.label());
        if let Some(mut old) = self.emulation_handle.take() {
            old.stop();
        }
        self.stop_fade = None;
        self.stop_anim = None;
        self.resume_anim = None;
        let repaint_ctx = ctx.clone();
        let target = self.native_game.as_mut().map(|window| window.begin_game(ctx));
        match EmulationHandle::new_with_presentation(
            &self.nro_path,
            backend,
            Some(std::sync::Arc::new(move || {
                request_game_frame_repaint(&repaint_ctx)
            })),
            target,
        ) {
            Ok(h) => {
                crate::ui_audio::play(crate::ui_audio::Sfx::GameBoot);
                self.emulation_handle = Some(h);
                self.game_texture = None;
                self.free_native_texture();
                self.pause_anim = None;
                self.pill_fade = None;
                let path = std::path::PathBuf::from(&self.nro_path);
                if let Some(idx) = self.library.index_of_path(&path) {
                    let n = self.library.move_to_front(idx);
                    self.carousel.selected = n + crate::carousel::CS_FRONT;
                    self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.game_window_title(&path)));
                self.playing_path = Some(path);
            }
            Err(e) => log::error!("Boot: {}", e),
        }
    }

    fn stop_emulation(&mut self) {
        self.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::Title("NeXium".to_string()));
        self.play_times.save_if_dirty();
        self.carousel.boot_stage = crate::carousel::BootStage::None;
        let was_paused = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_paused());
        let carousel = self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel;
        if let Some(mut h) = self.emulation_handle.take() {
            h.stop();
            self.pause_anim = None;
            self.resume_anim = None;
            let has_frame = self.game_texture.is_some() || self.game_texture_native.is_some();
            if carousel && self.playing_path.is_some() {
                if !was_paused && has_frame {
                    self.stop_anim = Some(std::time::Instant::now());
                } else {
                    self.pill_fade = Some(std::time::Instant::now());
                    self.game_texture = None;
                }
            } else if has_frame {
                self.stop_fade = Some(std::time::Instant::now());
                self.playing_path = None;
            } else {
                self.playing_path = None;
            }
        } else {
            self.playing_path = None;
        }
    }

    fn reload_profile_texture(&mut self, ctx: &egui::Context) {
        static DEFAULT_AVATAR: &[u8] = include_bytes!("../../branding/png/logo-256.png");
        self.profile_texture = None;

        let bytes: std::borrow::Cow<[u8]> = match self.app_settings.profile_avatar.clone() {
            Some(path) => match std::fs::read(&path) {
                Ok(b) => std::borrow::Cow::Owned(b),
                Err(_) => {
                    log::warn!("profile avatar unreadable: {}", path.display());
                    std::borrow::Cow::Borrowed(DEFAULT_AVATAR)
                }
            },
            None => std::borrow::Cow::Borrowed(DEFAULT_AVATAR),
        };

        let img = match image::load_from_memory(&bytes)
            .or_else(|_| image::load_from_memory(DEFAULT_AVATAR))
        {
            Ok(i) => i,
            Err(_) => return,
        };

        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let side = w.min(h);
        let ox = (w - side) / 2;
        let oy = (h - side) / 2;
        let cropped = image::imageops::crop_imm(&rgba, ox, oy, side, side).to_image();
        let cropped = if side > 256 {
            image::imageops::resize(&cropped, 256, 256, image::imageops::FilterType::Lanczos3)
        } else {
            cropped
        };
        let (fw, fh) = cropped.dimensions();
        let color =
            egui::ColorImage::from_rgba_unmultiplied([fw as usize, fh as usize], cropped.as_raw());
        self.profile_texture =
            Some(ctx.load_texture("profile_avatar", color, egui::TextureOptions::LINEAR));
    }

    fn pick_profile_avatar(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg"])
            .pick_file()
        {
            self.app_settings.profile_avatar = Some(p);
            let _ = self.app_settings.save();
            self.profile_reload = true;
        }
    }

    fn game_draw_rect(&self, panel: egui::Rect, tsz: Vec2, pixels_per_point: f32) -> egui::Rect {
        game_display_rect(
            panel,
            tsz,
            self.app_settings.aspect,
            self.app_settings.output_scale,
            pixels_per_point,
        )
    }

    fn is_running(&self) -> bool {
        self.emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_running())
    }
}

pub fn clipboard_text() -> String {
    arboard::Clipboard::new()
        .and_then(|mut c| c.get_text())
        .unwrap_or_default()
}

fn guest_pointer_position(position: egui::Pos2, rect: egui::Rect) -> Option<[i32; 2]> {
    if rect.width() <= 0.0 || rect.height() <= 0.0 || !rect.contains(position) {
        return None;
    }
    let normalized = (position - rect.min) / rect.size();
    Some([
        ((normalized.x * 1280.0).round() as i32).clamp(0, 1279),
        ((normalized.y * 720.0).round() as i32).clamp(0, 719),
    ])
}

fn guest_wheel_delta(input: &egui::InputState) -> egui::Vec2 {
    input.events.iter().filter_map(|event| {
        let egui::Event::MouseWheel { unit, delta, .. } = event else {
            return None;
        };
        Some(match unit {
            egui::MouseWheelUnit::Line => *delta,
            egui::MouseWheelUnit::Point => *delta / 40.0,
            egui::MouseWheelUnit::Page => *delta * (input.viewport_rect().height() / 40.0),
        })
    }).fold(egui::Vec2::ZERO, |total, delta| total + delta)
}

fn guest_keyboard_input(
    input: &egui::InputState,
    enabled: bool,
    blocked: bool,
    speed_shortcut: bool,
    overlay_shortcut: bool,
    recenter_key: Option<egui::Key>,
) -> nexium_core::hid_state::KeyboardInput {
    use nexium_core::hid_state::{KeyboardInput, KEYBOARD_MOD_CONTROL, KEYBOARD_MOD_SHIFT, KEYBOARD_MOD_LEFT_ALT};
    let mut keyboard = KeyboardInput { connected: enabled, ..KeyboardInput::default() };
    if !enabled || blocked || !input.focused {
        return keyboard;
    }
    for key in &input.keys_down {
        if (speed_shortcut && *key == egui::Key::U)
            || (overlay_shortcut && *key == egui::Key::O)
            || recenter_key == Some(*key)
        {
            continue;
        }
        if let Some(index) = hid_keyboard_usage(*key) {
            keyboard.keys[usize::from(index / 8)] |= 1 << (index % 8);
        }
    }
    if input.modifiers.ctrl { keyboard.modifiers |= KEYBOARD_MOD_CONTROL; }
    if input.modifiers.shift { keyboard.modifiers |= KEYBOARD_MOD_SHIFT; }
    if input.modifiers.alt { keyboard.modifiers |= KEYBOARD_MOD_LEFT_ALT; }
    keyboard
}

#[cfg(test)]
mod guest_input_tests {
    use super::{guest_keyboard_input, guest_pointer_position, guest_wheel_delta};
    use egui::{pos2, vec2, Event, InputState, Key, Modifiers, MouseWheelUnit, Rect};

    #[test]
    fn f12_reaches_direct_keyboard_without_controller_mapping() {
        let mut input = InputState::default();
        input.focused = true;
        input.keys_down.insert(Key::F12);
        let keyboard = guest_keyboard_input(&input, true, false, false, false, None);
        assert!(keyboard.connected);
        assert_eq!(keyboard.keys[8], 1 << 5);
        assert_eq!(keyboard.keys.iter().map(|byte| byte.count_ones()).sum::<u32>(), 1);
        for (enabled, blocked, focused) in [(false, false, true), (true, true, true), (true, false, false)] {
            input.focused = focused;
            let keyboard = guest_keyboard_input(&input, enabled, blocked, false, false, None);
            assert_eq!(keyboard.keys, [0; 32]);
            assert_eq!(keyboard.modifiers, 0);
        }
        input.focused = true;
        input.keys_down.clear();
        assert_eq!(guest_keyboard_input(&input, true, false, false, false, None).keys, [0; 32]);
    }

    #[test]
    fn emulator_shortcuts_do_not_press_guest_letters() {
        let mut input = InputState::default();
        input.focused = true;
        input.keys_down.extend([Key::U, Key::O, Key::F12]);
        let keyboard = guest_keyboard_input(&input, true, false, true, true, None);
        assert_eq!(keyboard.keys[8], 1 << 5);
        assert_eq!(keyboard.keys.iter().map(|byte| byte.count_ones()).sum::<u32>(), 1);
    }

    #[test]
    fn recenter_hotkey_does_not_press_a_guest_key() {
        let mut input = InputState::default();
        input.focused = true;
        input.keys_down.extend([Key::F8, Key::F12]);
        let keyboard = guest_keyboard_input(&input, true, false, false, false, Some(Key::F8));
        assert_eq!(keyboard.keys[8], 1 << 5);
        assert_eq!(keyboard.keys.iter().map(|byte| byte.count_ones()).sum::<u32>(), 1);
    }

    #[test]
    fn mouse_coordinates_exclude_letterboxing_and_status_bar() {
        let rect = Rect::from_min_size(pos2(100.0, 30.0), vec2(1600.0, 900.0));
        assert_eq!(guest_pointer_position(rect.center(), rect), Some([640, 360]));
        assert_eq!(guest_pointer_position(rect.min, rect), Some([0, 0]));
        assert_eq!(guest_pointer_position(rect.max, rect), Some([1279, 719]));
        assert_eq!(guest_pointer_position(pos2(90.0, 480.0), rect), None);
        assert_eq!(guest_pointer_position(pos2(900.0, 950.0), rect), None);
        let scaled = Rect::from_min_max(rect.min / 1.5, rect.max / 1.5);
        assert_eq!(guest_pointer_position(rect.center() / 1.5, scaled), Some([640, 360]));
    }

    #[test]
    fn wheel_preserves_axes_notches_and_fractional_trackpad_motion() {
        let mut input = InputState::default();
        let wheel = |unit, delta| Event::MouseWheel {
            unit, delta, modifiers: Modifiers { shift: true, ..Default::default() },
            phase: egui::TouchPhase::Move,
        };
        input.events.push(wheel(MouseWheelUnit::Line, vec2(2.0, -3.0)));
        assert_eq!(guest_wheel_delta(&input), vec2(2.0, -3.0));
        input.events = vec![wheel(MouseWheelUnit::Point, vec2(0.0, 10.0))];
        let mut total = egui::Vec2::ZERO;
        for _ in 0..4 { total += guest_wheel_delta(&input); }
        assert_eq!(total, vec2(0.0, 1.0));
        input.events.clear();
        assert_eq!(guest_wheel_delta(&input), egui::Vec2::ZERO);
    }
}

pub fn raw_wheel_delta(i: &egui::InputState) -> egui::Vec2 {
    let mut total = egui::Vec2::ZERO;
    for event in &i.events {
        if let egui::Event::MouseWheel {
            unit,
            delta,
            modifiers,
            ..
        } = event
        {
            let mut d = match unit {
                egui::MouseWheelUnit::Point => *delta,
                egui::MouseWheelUnit::Line => 40.0 * *delta,
                egui::MouseWheelUnit::Page => i.viewport_rect().height() * *delta,
            };
            if modifiers.shift {
                d = egui::vec2(d.x + d.y, 0.0);
            }
            total += d;
        }
    }
    total
}

pub fn feed_text_input(events: &[egui::Event], buf: &mut String, max: usize, paste_now: bool) {
    let mut ctrl_v = false;
    let mut saw_paste = false;
    let push = |buf: &mut String, s: &str| {
        for ch in s.chars() {
            if !ch.is_control() && buf.chars().count() < max {
                buf.push(ch);
            }
        }
    };
    for ev in events {
        match ev {
            egui::Event::Text(txt) => push(buf, txt),
            egui::Event::Paste(txt) => {
                saw_paste = true;
                push(buf, txt);
            }
            egui::Event::Key {
                key: egui::Key::Backspace,
                pressed: true,
                ..
            } => {
                buf.pop();
            }
            egui::Event::Key {
                key: egui::Key::V,
                pressed: true,
                modifiers,
                ..
            } if modifiers.ctrl || modifiers.command => {
                ctrl_v = true;
            }
            _ => {}
        }
    }
    if paste_now || (ctrl_v && !saw_paste) {
        let c = clipboard_text();
        push(buf, &c);
    }
}

#[derive(Default)]
pub struct FieldResp {
    pub clicked: bool,
    pub secondary_clicked: bool,
    pub commit: bool,
    pub cancel: bool,
    pub changed: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn text_field(
    p: &egui::Painter,
    ui: &mut egui::Ui,
    area: egui::Rect,
    text_x: f32,
    mid_y: f32,
    buf: &mut String,
    caret: &mut usize,
    anchor: &mut usize,
    font: egui::FontId,
    text_col: Color32,
    muted_col: Color32,
    sel_col: Color32,
    placeholder: &str,
    password: bool,
    max: usize,
    active: bool,
    blink_on: bool,
    events: &[egui::Event],
) -> FieldResp {
    let mut resp = FieldResp::default();
    let mut cs: Vec<char> = buf.chars().collect();
    let mut car = (*caret).min(cs.len());
    let mut anc = (*anchor).min(cs.len());

    let disp_of = |cs: &[char]| -> String {
        if password {
            "\u{2022}".repeat(cs.len())
        } else {
            cs.iter().collect()
        }
    };
    let width_to = |ui: &egui::Ui, cs: &[char], idx: usize| -> f32 {
        let d = disp_of(&cs[..idx.min(cs.len())]);
        ui.fonts_mut(|f| f.layout_no_wrap(d, font.clone(), text_col).size().x)
    };
    let hit = |ui: &egui::Ui, cs: &[char], px: f32| -> usize {
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for i in 0..=cs.len() {
            let x = text_x + width_to(ui, cs, i);
            let d = (x - px).abs();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best
    };

    let ir = ui.allocate_rect(area, egui::Sense::click_and_drag());
    if ir.secondary_clicked() {
        resp.secondary_clicked = true;
    }
    if ir.clicked() {
        resp.clicked = true;
    }
    if ir.double_clicked() {
        anc = 0;
        car = cs.len();
    } else if let Some(pos) = ir.interact_pointer_pos() {
        if ir.drag_started() || ir.clicked() {
            let h = hit(ui, &cs, pos.x);
            car = h;
            anc = h;
        } else if ir.dragged() {
            car = hit(ui, &cs, pos.x);
        }
    }

    if active {
        let del_sel = |cs: &mut Vec<char>, car: &mut usize, anc: &mut usize| -> bool {
            let lo = (*car).min(*anc);
            let hi = (*car).max(*anc);
            if lo != hi {
                cs.drain(lo..hi);
                *car = lo;
                *anc = lo;
                true
            } else {
                false
            }
        };
        let insert = |cs: &mut Vec<char>, car: &mut usize, anc: &mut usize, s: &str| {
            let lo = (*car).min(*anc);
            let hi = (*car).max(*anc);
            if lo != hi {
                cs.drain(lo..hi);
                *car = lo;
            }
            for ch in s.chars() {
                if !ch.is_control() && cs.len() < max {
                    cs.insert(*car, ch);
                    *car += 1;
                }
            }
            *anc = *car;
        };
        let copy_sel = |cs: &[char], car: usize, anc: usize| {
            let lo = car.min(anc);
            let hi = car.max(anc);
            if lo != hi {
                let sel: String = cs[lo..hi].iter().collect();
                let _ = arboard::Clipboard::new().and_then(|mut c| c.set_text(sel));
            }
        };
        let mut saw_paste = false;
        let mut ctrl_v = false;
        for ev in events {
            match ev {
                egui::Event::Text(s) => {
                    insert(&mut cs, &mut car, &mut anc, s);
                    resp.changed = true;
                }
                egui::Event::Paste(s) => {
                    saw_paste = true;
                    insert(&mut cs, &mut car, &mut anc, s);
                    resp.changed = true;
                }
                egui::Event::Copy => copy_sel(&cs, car, anc),
                egui::Event::Cut => {
                    copy_sel(&cs, car, anc);
                    if del_sel(&mut cs, &mut car, &mut anc) {
                        resp.changed = true;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::Backspace,
                    pressed: true,
                    ..
                } => {
                    if !del_sel(&mut cs, &mut car, &mut anc) && car > 0 {
                        cs.remove(car - 1);
                        car -= 1;
                    }
                    anc = car;
                    resp.changed = true;
                }
                egui::Event::Key {
                    key: egui::Key::Delete,
                    pressed: true,
                    ..
                } => {
                    if !del_sel(&mut cs, &mut car, &mut anc) && car < cs.len() {
                        cs.remove(car);
                    }
                    anc = car;
                    resp.changed = true;
                }
                egui::Event::Key {
                    key: egui::Key::ArrowLeft,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    if car > 0 {
                        car -= 1;
                    }
                    if !modifiers.shift {
                        anc = car;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::ArrowRight,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    if car < cs.len() {
                        car += 1;
                    }
                    if !modifiers.shift {
                        anc = car;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::Home,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    car = 0;
                    if !modifiers.shift {
                        anc = 0;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::End,
                    pressed: true,
                    modifiers,
                    ..
                } => {
                    car = cs.len();
                    if !modifiers.shift {
                        anc = car;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::A,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.ctrl || modifiers.command => {
                    anc = 0;
                    car = cs.len();
                }
                egui::Event::Key {
                    key: egui::Key::C,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.ctrl || modifiers.command => copy_sel(&cs, car, anc),
                egui::Event::Key {
                    key: egui::Key::X,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.ctrl || modifiers.command => {
                    copy_sel(&cs, car, anc);
                    if del_sel(&mut cs, &mut car, &mut anc) {
                        resp.changed = true;
                    }
                }
                egui::Event::Key {
                    key: egui::Key::V,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.ctrl || modifiers.command => {
                    ctrl_v = true;
                }
                egui::Event::Key {
                    key: egui::Key::Enter,
                    pressed: true,
                    ..
                } => {
                    resp.commit = true;
                }
                egui::Event::Key {
                    key: egui::Key::Escape,
                    pressed: true,
                    ..
                } => {
                    resp.cancel = true;
                }
                _ => {}
            }
        }
        if ctrl_v && !saw_paste {
            let c = clipboard_text();
            insert(&mut cs, &mut car, &mut anc, &c);
            resp.changed = true;
        }
    }

    car = car.min(cs.len());
    anc = anc.min(cs.len());

    let disp = disp_of(&cs);
    if cs.is_empty() && !active {
        p.text(
            egui::pos2(text_x, mid_y),
            egui::Align2::LEFT_CENTER,
            placeholder,
            font.clone(),
            muted_col,
        );
    } else {
        if active {
            let lo = car.min(anc);
            let hi = car.max(anc);
            if lo != hi {
                let x0 = text_x + width_to(ui, &cs, lo);
                let x1 = text_x + width_to(ui, &cs, hi);
                let fh = font.size;
                let selr = egui::Rect::from_min_max(
                    egui::pos2(x0, mid_y - fh * 0.62),
                    egui::pos2(x1, mid_y + fh * 0.62),
                );
                p.rect_filled(selr, egui::CornerRadius::same(2), sel_col);
            }
        }
        p.text(
            egui::pos2(text_x, mid_y),
            egui::Align2::LEFT_CENTER,
            &disp,
            font.clone(),
            text_col,
        );
        if active && blink_on {
            let cx = text_x + width_to(ui, &cs, car);
            let fh = font.size;
            p.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(cx, mid_y - fh * 0.6),
                    egui::pos2(cx + 1.5_f32.max(fh * 0.06), mid_y + fh * 0.6),
                ),
                egui::CornerRadius::ZERO,
                text_col,
            );
        }
    }

    *caret = car;
    *anchor = anc;
    *buf = cs.iter().collect();
    resp
}

fn draw_check(p: &egui::Painter, c: egui::Pos2, sz: f32, col: Color32) {
    let a = c + egui::Vec2::new(-sz * 0.42, sz * 0.02);
    let b = c + egui::Vec2::new(-sz * 0.10, sz * 0.32);
    let d = c + egui::Vec2::new(sz * 0.46, -sz * 0.34);
    let stroke = egui::Stroke::new((sz * 0.20).max(1.6), col);
    p.line_segment([a, b], stroke);
    p.line_segment([b, d], stroke);
}

fn draw_dpad(p: &egui::Painter, c: egui::Pos2, r: f32, col: Color32) {
    let thick = r * 0.66;
    let round = egui::CornerRadius::from(thick * 0.28);
    p.rect_filled(
        egui::Rect::from_center_size(c, egui::Vec2::new(r * 2.0, thick)),
        round,
        col,
    );
    p.rect_filled(
        egui::Rect::from_center_size(c, egui::Vec2::new(thick, r * 2.0)),
        round,
        col,
    );
}

fn shorten_device(dev: &Option<String>) -> String {
    match dev {
        None => "System default".to_string(),
        Some(name) => {
            let max = 22;
            if name.chars().count() > max {
                let s: String = name.chars().take(max - 1).collect();
                format!("{}…", s)
            } else {
                name.clone()
            }
        }
    }
}

fn cycle_audio_device(current: &Option<String>, devices: &[String], dir: i32) -> Option<String> {
    let cur = match current {
        None => 0usize,
        Some(name) => devices
            .iter()
            .position(|d| d == name)
            .map(|p| p + 1)
            .unwrap_or(0),
    };
    let len = (devices.len() + 1) as i32;
    let next = (cur as i32 + dir).rem_euclid(len) as usize;
    if next == 0 {
        None
    } else {
        devices.get(next - 1).cloned()
    }
}

fn apply_audio_cycle(cfg: &mut AppSettings, row: usize, dir: i32, devices: &[String]) {
    match row {
        0 => cfg.audio_output_device = cycle_audio_device(&cfg.audio_output_device, devices, dir),
        4 => cfg.music_muted = !cfg.music_muted,
        5 => {
            cfg.sfx_muted = !cfg.sfx_muted;
            crate::ui_audio::set_sfx_volume(if cfg.sfx_muted { 0.0 } else { cfg.sfx_volume });
        }
        6 => {
            cfg.menu_music_track = cycle_menu_music(cfg.menu_music_track, dir);
            crate::ui_audio::set_carousel_track(cfg.menu_music_track);
        }
        _ => {}
    }
    let _ = cfg.save();
}

fn cycle_menu_music(current: u8, dir: i32) -> u8 {
    let order: [u8; crate::ui_audio::CAROUSEL_TRACK_COUNT + 1] = {
        let mut o = [crate::ui_audio::CAROUSEL_ALL; crate::ui_audio::CAROUSEL_TRACK_COUNT + 1];
        for (i, slot) in o
            .iter_mut()
            .take(crate::ui_audio::CAROUSEL_TRACK_COUNT)
            .enumerate()
        {
            *slot = i as u8;
        }
        o
    };
    let len = order.len() as i32;
    let cur = order.iter().position(|&v| v == current).unwrap_or(0) as i32;
    let next = ((cur + dir.signum()) % len + len) % len;
    order[next as usize]
}

#[allow(clippy::too_many_arguments)]
fn pref_value_rows(
    p: &egui::Painter,
    ui: &mut egui::Ui,
    sp: &dyn Fn(egui::Pos2) -> egui::Pos2,
    sr: &dyn Fn(egui::Rect) -> egui::Rect,
    content: egui::Rect,
    s: f32,
    sf: f32,
    ease: f32,
    accent: Color32,
    text: Color32,
    muted: Color32,
    border: Color32,
    panel: Color32,
    sel: Color32,
    focus: bool,
    block: bool,
    row: &mut usize,
    rows: &[(String, String)],
    nu: bool,
    nd: bool,
    nl: bool,
    nr: bool,
    a_edge: bool,
) -> Option<(usize, i32)> {
    let n = rows.len();
    if !focus {
        *row = 0;
    }
    *row = (*row).min(n.saturating_sub(1));
    if focus && !block {
        if nu && *row > 0 {
            *row -= 1;
            crate::ui_audio::play_move();
        }
        if nd && *row + 1 < n {
            *row += 1;
            crate::ui_audio::play_move();
        }
    }
    let row_h = 62.0 * s;
    let mut y = content.min.y + 4.0 * s;
    let mut result = None;
    for (i, (label, value)) in rows.iter().enumerate() {
        let base = egui::Rect::from_min_size(
            egui::pos2(content.min.x, y),
            egui::Vec2::new(content.width(), row_h - 12.0 * s),
        );
        let r = sr(base);
        let selrow = focus && i == *row;
        if selrow {
            p.rect_filled(r, egui::CornerRadius::from(12.0 * s), sel);
            p.rect_stroke(
                r,
                egui::CornerRadius::from(12.0 * s),
                egui::Stroke::new(2.0_f32, accent),
                egui::StrokeKind::Outside,
            );
        } else {
            p.rect_filled(
                r,
                egui::CornerRadius::from(12.0 * s),
                Color32::from_rgba_unmultiplied(
                    panel.r(),
                    panel.g(),
                    panel.b(),
                    (ease * 90.0) as u8,
                ),
            );
            p.rect_stroke(
                r,
                egui::CornerRadius::from(12.0 * s),
                egui::Stroke::new(1.0_f32, border),
                egui::StrokeKind::Outside,
            );
        }
        p.text(
            sp(egui::pos2(base.min.x + 22.0 * s, base.center().y)),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(18.0 * s * sf),
            text,
        );
        let lx = base.max.x - 150.0 * s;
        let rx = base.max.x - 22.0 * s;
        p.text(
            sp(egui::pos2((lx + rx) * 0.5, base.center().y)),
            egui::Align2::CENTER_CENTER,
            value,
            egui::FontId::proportional(16.0 * s * sf),
            if selrow { accent } else { muted },
        );
        let la = egui::Rect::from_center_size(
            egui::pos2(lx, base.center().y),
            egui::Vec2::splat(34.0 * s),
        );
        let ra = egui::Rect::from_center_size(
            egui::pos2(rx, base.center().y),
            egui::Vec2::splat(34.0 * s),
        );
        p.text(
            sp(la.center()),
            egui::Align2::CENTER_CENTER,
            "\u{2039}",
            egui::FontId::proportional(22.0 * s * sf),
            if selrow { accent } else { muted },
        );
        p.text(
            sp(ra.center()),
            egui::Align2::CENTER_CENTER,
            "\u{203A}",
            egui::FontId::proportional(22.0 * s * sf),
            if selrow { accent } else { muted },
        );
        if !block && ui.rect_contains_pointer(r) {
            let rowc = ui.allocate_rect(r, egui::Sense::click()).clicked();
            let lc = ui.allocate_rect(sr(la), egui::Sense::click()).clicked();
            let rc = ui.allocate_rect(sr(ra), egui::Sense::click()).clicked();
            if lc || rc || rowc {
                *row = i;
                result = Some((i, if lc { -1 } else { 1 }));
            }
        }
        y += row_h;
    }
    if focus && !block && result.is_none() {
        if nl {
            result = Some((*row, -1));
        } else if nr || a_edge {
            result = Some((*row, 1));
        }
    }
    result
}

fn draw_controller(
    p: &egui::Painter,
    cr: egui::Rect,
    hl: Option<SwitchButton>,
    accent: Color32,
    light: bool,
    live: &crate::input::InputSnapshot,
) {
    use crate::controller_config::SwitchButton::*;
    let s = (cr.width() / 430.0).min(cr.height() / 320.0);
    let c = cr.center();
    let map = |x: f32, y: f32| c + Vec2::new(x * s, (y + 6.0) * s);

    let outline = if light {
        Color32::from_rgb(0x90, 0x90, 0x9F)
    } else {
        Color32::from_rgb(0x4C, 0x4C, 0x58)
    };
    let body_fill = if light {
        Color32::from_rgb(0xE6, 0xE6, 0xEB)
    } else {
        Color32::from_rgb(0x22, 0x22, 0x2C)
    };
    let btn_bg = if light {
        Color32::from_rgb(0xCD, 0xCD, 0xD6)
    } else {
        Color32::from_rgb(0x20, 0x20, 0x26)
    };
    let btn_border = if light {
        Color32::from_rgb(0xAC, 0xAC, 0xBA)
    } else {
        Color32::from_rgb(0x2A, 0x2A, 0x32)
    };
    let btn_muted = if light {
        Color32::from_rgb(0x3C, 0x3C, 0x4C)
    } else {
        Color32::from_rgb(0x70, 0x70, 0x80)
    };
    let knob_default = if light {
        Color32::from_rgb(0x8A, 0x8A, 0x9A)
    } else {
        Color32::from_rgb(0x62, 0x62, 0x72)
    };

    let body_stroke = egui::Stroke::new((2.0 * s).max(1.2), outline);
    let is = |b: SwitchButton| hl == Some(b) || (live.connected && live.is(b));
    let lstick_on =
        is(StickL) || is(StickLUp) || is(StickLDown) || is(StickLLeft) || is(StickLRight);
    let rstick_on =
        is(StickR) || is(StickRUp) || is(StickRDown) || is(StickRLeft) || is(StickRRight);

    let n = PRO_BODY.len() / 2;
    let mut body: Vec<egui::Pos2> = Vec::with_capacity(n * 2);
    for i in 0..n {
        body.push(map(PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    for i in (0..n).rev() {
        body.push(map(-PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    let hn = PRO_LEFT_HANDLE.len() / 2;
    let lh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    let rh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(-PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    p.add(egui::Shape::convex_polygon(
        lh.clone(),
        body_fill,
        egui::Stroke::NONE,
    ));
    p.add(egui::Shape::convex_polygon(
        rh.clone(),
        body_fill,
        egui::Stroke::NONE,
    ));
    p.add(egui::Shape::convex_polygon(
        body.clone(),
        body_fill,
        egui::Stroke::NONE,
    ));
    p.add(egui::Shape::closed_line(lh, body_stroke));
    p.add(egui::Shape::closed_line(rh, body_stroke));
    p.add(egui::Shape::closed_line(body, body_stroke));

    let face = |center: egui::Pos2, r: f32, label: &str, on: bool| {
        let fill = if on { accent } else { btn_bg };
        p.circle(center, r, fill, egui::Stroke::new(1.0_f32, btn_border));
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { btn_muted };
            p.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional((r * 0.95).max(7.0)),
                tc,
            );
        }
    };
    let pad = |center: egui::Pos2, sz: Vec2, label: &str, on: bool| {
        let r = egui::Rect::from_center_size(center, sz);
        let fill = if on { accent } else { btn_bg };
        p.rect(
            r,
            CornerRadius::same(3),
            fill,
            egui::Stroke::new(1.0_f32, btn_border),
            egui::StrokeKind::Outside,
        );
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { btn_muted };
            p.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional(9.0 * s.max(0.8)),
                tc,
            );
        }
    };
    let stick = |base: egui::Pos2, ax: f32, ay: f32, on: bool| {
        let ring = 22.0 * s;
        let knob = 13.0 * s;
        let max_off = ring - knob;
        let koff = Vec2::new(
            ax.clamp(-1.0, 1.0) * max_off,
            (-ay).clamp(-1.0, 1.0) * max_off,
        );
        let kc = base + koff;
        p.circle(
            base,
            ring,
            btn_bg,
            egui::Stroke::new(1.5_f32, if on { accent } else { outline }),
        );
        let kcol = if on { accent } else { knob_default };
        p.circle(kc, knob, kcol, egui::Stroke::new(1.0_f32, btn_border));
    };

    pad(
        map(-120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZL",
        is(ZL),
    );
    pad(
        map(120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZR",
        is(ZR),
    );
    pad(
        map(-120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "L",
        is(L),
    );
    pad(
        map(120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "R",
        is(R),
    );

    let lx = if live.connected { live.lx() } else { 0.0 };
    let ly = if live.connected { live.ly() } else { 0.0 };
    let rx = if live.connected { live.rx() } else { 0.0 };
    let ry = if live.connected { live.ry() } else { 0.0 };
    stick(map(-111.0, -55.0), lx, ly, lstick_on);

    let dd = 19.0 * s;
    let dsz = Vec2::splat(16.0 * s);
    let dp = map(-61.0, 0.0);
    pad(dp + Vec2::new(0.0, -dd), dsz, "", is(DUp));
    pad(dp + Vec2::new(0.0, dd), dsz, "", is(DDown));
    pad(dp + Vec2::new(-dd, 0.0), dsz, "", is(DLeft));
    pad(dp + Vec2::new(dd, 0.0), dsz, "", is(DRight));

    let fr = 15.0 * s;
    face(map(136.0, -56.0), fr, "A", is(A));
    face(map(105.0, -25.0), fr, "B", is(B));
    face(map(105.0, -87.0), fr, "X", is(X));
    face(map(74.0, -56.0), fr, "Y", is(Y));

    stick(map(51.0, 0.0), rx, ry, rstick_on);

    face(map(-50.0, -86.0), 9.0 * s, "-", is(Minus));
    face(map(50.0, -86.0), 9.0 * s, "+", is(Plus));
}

fn draw_dissolve(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let (cols, rows) = (44usize, 26usize);
    let cw = rect.width() / cols as f32;
    let ch = rect.height() / rows as f32;
    let mut mesh = egui::Mesh::with_texture(tex_id);
    for j in 0..rows {
        for i in 0..cols {
            let local = (t - cell_hash(i, j) * 0.55) / 0.35;
            if local >= 1.0 {
                continue;
            }
            let a = 1.0 - local.max(0.0);
            let drop = if local > 0.0 {
                local * local * ch * 12.0
            } else {
                0.0
            };
            let jitter = if local > 0.0 {
                (cell_hash(i + 7, j + 3) - 0.5) * local * cw * 1.6
            } else {
                0.0
            };
            let min = egui::pos2(
                rect.min.x + i as f32 * cw + jitter,
                rect.min.y + j as f32 * ch + drop,
            );
            let cell = egui::Rect::from_min_size(min, Vec2::new(cw + 0.5, ch + 0.5));
            let uv = egui::Rect::from_min_max(
                egui::pos2(i as f32 / cols as f32, j as f32 / rows as f32),
                egui::pos2((i + 1) as f32 / cols as f32, (j + 1) as f32 / rows as f32),
            );
            mesh.add_rect_with_uv(cell, uv, Color32::from_white_alpha((a * 255.0) as u8));
        }
    }
    p.add(egui::Shape::mesh(mesh));
}

fn draw_zoom_fade_out(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let t = t.clamp(0.0, 1.0);
    let scale = 1.0 - 0.08 * t;
    let alpha = 1.0 - t;
    let draw_rect = egui::Rect::from_center_size(rect.center(), rect.size() * scale);
    p.image(
        tex_id,
        draw_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        Color32::from_white_alpha((alpha * 255.0) as u8),
    );
}

fn draw_zoom_fade_in(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let t = t.clamp(0.0, 1.0);
    let scale = 1.06 - 0.06 * t;
    let alpha = t;
    let draw_rect = egui::Rect::from_center_size(rect.center(), rect.size() * scale);
    p.image(
        tex_id,
        draw_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        Color32::from_white_alpha((alpha * 255.0) as u8),
    );
}

fn cell_hash(i: usize, j: usize) -> f32 {
    let mut n = (i as u32)
        .wrapping_mul(374761393)
        .wrapping_add((j as u32).wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0xffff) as f32 / 65535.0
}

fn game_tile(
    ui: &mut egui::Ui,
    game: &crate::library::GameEntry,
    tex: Option<&egui::TextureHandle>,
    selected: bool,
) -> egui::Response {
    let tile = Vec2::new(154.0, 202.0);
    let (rect, resp) = ui.allocate_exact_size(tile, Sense::click());
    let hovered = resp.hovered();
    let t = ui.input(|i| i.time) as f32;

    let bg = if selected {
        BG_INPUT
    } else if hovered {
        Color32::from_rgb(0x1C, 0x1C, 0x22)
    } else {
        BG_RAISED
    };
    let rounding = CornerRadius::same(10);
    ui.painter().rect_filled(rect, rounding, bg);

    if selected {
        animated_border(ui.painter(), rect, 10.0, t);
        ui.ctx().request_repaint();
    } else if hovered {
        for i in 1..=3 {
            let e = i as f32;
            ui.painter().rect_stroke(
                rect.expand(e * 1.5),
                CornerRadius::from(10.0 + e * 1.5),
                Stroke::new(
                    1.5_f32,
                    Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, (34 / i) as u8),
                ),
                egui::StrokeKind::Outside,
            );
        }
        ui.painter().rect_stroke(
            rect,
            rounding,
            Stroke::new(1.4_f32, ACCENT_HV),
            egui::StrokeKind::Outside,
        );
    } else {
        ui.painter().rect_stroke(
            rect,
            rounding,
            Stroke::new(1.0_f32, BORDER),
            egui::StrokeKind::Outside,
        );
    }

    let pad = 11.0;
    let icon_sz = tile.x - pad * 2.0;
    let icon_rect = egui::Rect::from_min_size(rect.min + Vec2::splat(pad), Vec2::splat(icon_sz));
    if let Some(tex) = tex {
        egui::Image::new((tex.id(), icon_rect.size()))
            .corner_radius(CornerRadius::same(6))
            .paint_at(ui, icon_rect);
    } else {
        ui.painter().rect_filled(
            icon_rect,
            CornerRadius::same(6),
            Color32::from_rgb(0x12, 0x12, 0x16),
        );
        ui.painter().text(
            icon_rect.center(),
            egui::Align2::CENTER_CENTER,
            game.format,
            FontId::proportional(22.0),
            MUTED,
        );
    }

    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 14.0),
        egui::Align2::CENTER_CENTER,
        elide(&game.title, 20),
        FontId::proportional(12.5),
        if selected {
            TEXT
        } else {
            Color32::from_rgb(0xC8, 0xC8, 0xD2)
        },
    );

    let sub = if game.author.is_empty() {
        game.format.to_string()
    } else {
        elide(&game.author, 22)
    };
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 32.0),
        egui::Align2::CENTER_CENTER,
        sub,
        FontId::proportional(10.5),
        MUTED,
    );

    resp.on_hover_text(&game.title)
}

fn add_folder_tile(ui: &mut egui::Ui) -> egui::Response {
    let tile = Vec2::new(154.0, 202.0);
    let (rect, resp) = ui.allocate_exact_size(tile, Sense::click());
    let hovered = resp.hovered();

    let bg = if hovered {
        Color32::from_rgb(0x1C, 0x1C, 0x22)
    } else {
        BG_RAISED
    };
    let rounding = CornerRadius::same(10);
    ui.painter().rect_filled(rect, rounding, bg);
    if hovered {
        ui.painter().rect_stroke(
            rect,
            rounding,
            Stroke::new(1.4_f32, ACCENT_HV),
            egui::StrokeKind::Outside,
        );
    } else {
        ui.painter().rect_stroke(
            rect,
            rounding,
            Stroke::new(1.0_f32, BORDER),
            egui::StrokeKind::Outside,
        );
    }

    let pad = 11.0;
    let icon_sz = tile.x - pad * 2.0;
    let icon_rect = egui::Rect::from_min_size(rect.min + Vec2::splat(pad), Vec2::splat(icon_sz));
    ui.painter().rect_filled(
        icon_rect,
        CornerRadius::same(6),
        Color32::from_rgb(0x12, 0x12, 0x16),
    );
    ui.painter().text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "+",
        FontId::proportional(48.0),
        if hovered { ACCENT_HV } else { MUTED },
    );
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 14.0),
        egui::Align2::CENTER_CENTER,
        "Add Folder",
        FontId::proportional(12.5),
        if hovered {
            TEXT
        } else {
            Color32::from_rgb(0xC8, 0xC8, 0xD2)
        },
    );
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 32.0),
        egui::Align2::CENTER_CENTER,
        ".dnsp / .dxci",
        FontId::proportional(10.5),
        MUTED,
    );

    resp.on_hover_text("Add a folder of .dnsp / .dxci games")
}

fn animated_border(p: &egui::Painter, rect: egui::Rect, r: f32, t: f32) {
    let pulse = 0.5 + 0.5 * (t * 2.1).sin();
    for i in 1..=4 {
        let e = i as f32 * 2.4;
        let fade = 1.0 - (i as f32 - 1.0) / 4.0;
        let a = (52.0 * fade * (0.5 + 0.5 * pulse)) as u8;
        p.rect_stroke(
            rect.expand(e),
            CornerRadius::from(r + e),
            Stroke::new(
                2.2_f32,
                Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, a),
            ),
            egui::StrokeKind::Outside,
        );
    }
    p.rect_stroke(
        rect,
        CornerRadius::from(r),
        Stroke::new(1.8_f32, lerp_col(ACCENT, ACCENT_HV, pulse)),
        egui::StrokeKind::Outside,
    );
}

fn lerp_col(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

fn elide(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    } else {
        s.to_string()
    }
}

fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let size = bytes as f64;
    if size >= GB {
        format!("{:.2} GB", size / GB)
    } else if size >= MB {
        format!("{:.2} MB", size / MB)
    } else if size >= KB {
        format!("{:.1} KB", size / KB)
    } else {
        format!("{bytes} B")
    }
}

fn format_system_time(time: std::time::SystemTime) -> String {
    chrono::DateTime::<chrono::Local>::from(time)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

fn reveal_in_file_manager(path: &std::path::Path) {
    #[cfg(windows)]
    {
        if path.is_dir() {
            let _ = std::process::Command::new("explorer.exe").arg(path).spawn();
        } else {
            let _ = std::process::Command::new("explorer.exe")
                .arg(format!("/select,{}", path.display()))
                .spawn();
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let target = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().unwrap_or(path).to_path_buf()
        };
        let _ = std::process::Command::new("xdg-open").arg(target).spawn();
    }
}

fn pill_button(ui: &mut egui::Ui, label: &str, filled: bool) -> egui::Response {
    let font = FontId::proportional(12.5);
    let text_w = ui.fonts_mut(|f| {
        f.layout_no_wrap(label.to_string(), font.clone(), TEXT)
            .size()
            .x
    });
    let size = Vec2::new(text_w + 24.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    let (bg, text_col, stroke) = if filled {
        let c = if resp.is_pointer_button_down_on() {
            ACCENT_DK
        } else if resp.hovered() {
            ACCENT_HV
        } else {
            ACCENT
        };
        (c, Color32::WHITE, Stroke::NONE)
    } else {
        let c = if resp.hovered() {
            Color32::from_rgb(0x28, 0x28, 0x30)
        } else {
            Color32::TRANSPARENT
        };
        let bc = if resp.hovered() {
            Color32::from_rgb(0x50, 0x50, 0x60)
        } else {
            BORDER
        };
        (c, TEXT, Stroke::new(1.0_f32, bc))
    };

    ui.painter().rect(
        rect,
        CornerRadius::same(5),
        bg,
        stroke,
        egui::StrokeKind::Outside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        font,
        text_col,
    );
    resp
}

impl eframe::App for HorizonApp {
    fn on_exit(&mut self) {
        if let Some(ib) = self.input.as_mut() { ib.shutdown_rumble(); }
        if let Some(mut handle) = self.emulation_handle.take() { handle.stop_blocking(); }
        while crate::boot::emu_alive() { std::thread::sleep(std::time::Duration::from_millis(5)); }
        self.native_game.take();
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|input| input.viewport().visible() != Some(false)) {
            return;
        }
        if let Some(window) = &mut self.native_game { window.hide(); }
        let Some(handle) = &self.emulation_handle else {
            return;
        };
        let mut dropped = 0u64;
        while take_next_game_frame(&handle.frame_rx).is_some() {
            dropped += 1;
        }
        if dropped != 0 {
            gui_rate_stats(2, dropped);
            static NOTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NOTED.fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                log::info!(
                    "[gui] window not painting (minimized or hidden); dropping game frames until it is shown again"
                );
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        self.native_overlay_rect = None;
        self.game_bar_tab_rect = None;
        if ctx.input(|i| i.viewport().close_requested()) {
            self.play_times.save_if_dirty();
            if let Some(mut h) = self.emulation_handle.take() {
                h.stop_blocking();
            }
            return;
        }
        gui_rate_stats(0, 1);
        self.nextendo_tick(ctx);
        let frame_driven_game = self
            .emulation_handle
            .as_ref()
            .is_some_and(|h| h.is_running() && !h.is_paused())
            && self.game_display().is_some()
            && !self.modal_active()
            && !self.show_settings
            && !self.show_profile
            && self.profile_anim == 0.0
            && !self.nextendo_overlay_active()
            && self.pause_anim.is_none()
            && self.resume_anim.is_none();
        if self.splash.active() {
            if self.emulation_handle.is_some() {
                self.splash = crate::splash::Splash::finished();
            } else if let crate::splash::Step::Intro = self.splash.step(ctx) {
                return;
            }
        }

        self.library.poll();

        for pending in self.shop.new_installs.drain(..).collect::<Vec<_>>() {
            self.active_downloads.push((pending.title, pending.info));
        }
        let mut di = 0;
        while di < self.active_downloads.len() {
            let (done, ok) = self.active_downloads[di]
                .1
                .lock()
                .map(|g| (g.done, g.ok))
                .unwrap_or((false, false));
            if done {
                let (title, _) = self.active_downloads.remove(di);
                if ok {
                    self.download_toast = Some((title, std::time::Instant::now()));
                    self.library.rescan(ctx, &self.app_settings.library_folders);
                }
            } else {
                di += 1;
            }
        }

        if let Some((_, tm, _)) = &self.icon_reveal {
            if tm.elapsed().as_secs_f32() > 1.0 {
                self.icon_reveal = None;
            }
        }

        if self.profile_reload {
            self.profile_reload = false;
            self.reload_profile_texture(ctx);
        }

        if let Some(ref mut ib) = self.input {
            ib.set_motion_enabled(self.app_settings.motion_enabled);
            self.last_input = ib.poll(
                &self.controller_config,
                self.app_settings.left_deadzone,
                self.app_settings.right_deadzone,
            );
            if self.last_input.connected && !frame_driven_game {
                request_idle_repaint(ctx, std::time::Duration::from_millis(8));
            }
            if let Some(btn) = self.rebinding_pad {
                if self.rebinding_pad_wait_release {
                    if ib.first_pressed().is_none() {
                        self.rebinding_pad_wait_release = false;
                    }
                } else if let Some(gp) = ib.first_pressed() {
                    self.controller_config.set_pad(btn, gp);
                    let _ = self.controller_config.save();
                    self.rebinding_pad = None;
                    self.prefs_rebind_cool = true;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
                let _ = btn;
            }
        }

        {
            use crate::controller_config::SwitchButton;
            let a_down = ctx.input(|i| i.key_down(egui::Key::Enter))
                || (self.last_input.connected && self.last_input.is(SwitchButton::A));
            let x_down = ctx.input(|i| i.key_down(egui::Key::X))
                || (self.last_input.connected && self.last_input.is(SwitchButton::X));
            let b_down = self.last_input.connected && self.last_input.is(SwitchButton::B);
            let sel_down = ctx.input(|i| i.key_down(egui::Key::Tab))
                || (self.last_input.connected && self.last_input.is(SwitchButton::Minus));
            self.carousel.a_edge = a_down && !self.carousel.a_held;
            self.carousel.x_edge = x_down && !self.carousel.x_held;
            self.carousel.b_edge = b_down && !self.carousel.b_held;
            self.carousel.sel_edge = sel_down && !self.carousel.sel_held;
            self.carousel.a_held = a_down;
            self.carousel.x_held = x_down;
            self.carousel.b_held = b_down;
            self.carousel.sel_held = sel_down;
        }

        self.modal_active_frame_start = self.modal_active();
        self.handle_modal_input(ctx);
        self.teardown_tick(ctx);
        self.quick_launch_tick(ctx);

        {
            let (m_run, m_pause) = self
                .emulation_handle
                .as_ref()
                .map_or((false, false), |h| (h.is_running(), h.is_paused()));
            let carousel_view =
                self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel;
            let playing_fs = m_run && !m_pause;
            let base_on = carousel_view && !playing_fs && !self.splash.active();
            let shop_music = self.shop.visible();
            let music_mode = if shop_music {
                crate::ui_audio::MusicMode::Shop
            } else if self.carousel_settings_open {
                crate::ui_audio::MusicMode::Settings
            } else {
                crate::ui_audio::MusicMode::Carousel
            };
            if base_on && !self.music_was_on {
                self.music_fade_start = Some(std::time::Instant::now());
            }
            self.music_was_on = base_on;
            let ramp = if base_on {
                self.music_fade_start
                    .map(|s| (s.elapsed().as_secs_f32() / 5.0).clamp(0.0, 1.0))
                    .unwrap_or(1.0)
            } else {
                1.0
            };
            let mut target = if base_on && !self.app_settings.music_muted {
                self.app_settings.music_volume * ramp
            } else {
                0.0
            };
            let mut lowpass = 0.0f32;
            let profiling =
                self.show_profile || self.profile_anim > 0.01 || self.nextendo_overlay_active();
            let on_music_slider = profiling
                && self.profile.tab == crate::profile::ProfileTab::Settings
                && self.profile.focus_content
                && self.profile.row_selected == 3;
            if profiling && !on_music_slider {
                target *= 0.34;
                lowpass = 0.75;
            }
            if self.modal_active() && !shop_music && !self.carousel_settings_open {
                lowpass = lowpass.max(0.6);
            }
            crate::ui_audio::set_music(music_mode, target, lowpass);
            if base_on && ramp < 1.0 {
                ctx.request_repaint();
            }

            let booting_now = m_run
                && !m_pause
                && carousel_view
                && self.game_display().is_none()
                && !self.splash.active();
            if booting_now {
                crate::ui_audio::play_looped(crate::ui_audio::Sfx::AwaitFrame);
            } else {
                crate::ui_audio::stop_loop(crate::ui_audio::Sfx::AwaitFrame);
            }
            if self.teardown_at.is_some() {
                crate::ui_audio::play_looped(crate::ui_audio::Sfx::PleaseWait);
            } else {
                crate::ui_audio::stop_loop(crate::ui_audio::Sfx::PleaseWait);
            }
        }

        if !self.show_settings {
            self.motion_key_capture = false;
            self.rebinding = None;
            self.rebinding_pad = None;
        }
        {
            let kb_home =
                ctx.input(|i| i.key_pressed(egui::Key::Home) || i.key_pressed(egui::Key::Backtick));
            let gp_home = self.last_input.home;
            let home_edge = kb_home || (gp_home && !self.last_home);
            self.last_home = gp_home;

            let (running, paused) = self
                .emulation_handle
                .as_ref()
                .map_or((false, false), |h| (h.is_running(), h.is_paused()));

            if home_edge
                && running
                && !self.modal_active()
                && !self.rebind_in_progress()
            {
                if paused {
                    if let Some(h) = self.emulation_handle.as_ref() {
                        h.resume();
                    }
                    self.pause_anim = None;
                    self.resume_anim = Some(std::time::Instant::now());
                } else if self.game_display().is_some() {
                    if let Some(h) = self.emulation_handle.as_ref() {
                        h.pause();
                    }
                    if self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel {
                        self.app_settings.view_mode = crate::app_settings::ViewMode::Carousel;
                        let _ = self.app_settings.save();
                    }
                    if let Some(pp) = self.playing_path.clone() {
                        if let Some(idx) = self.library.index_of_path(&pp) {
                            let n = self.library.move_to_front(idx);
                            self.carousel.selected = n + crate::carousel::CS_FRONT;
                            self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                        }
                    }
                    self.carousel.active_dock = false;
                    self.pause_anim = Some(std::time::Instant::now());
                } else {
                    self.stop_emulation();
                    self.teardown_at = Some(std::time::Instant::now());
                    self.pending_boot = None;
                }
            }
        }

        let speed_shortcut_active = self.emulation_handle.as_ref().is_some_and(|h| h.is_running())
            && !self.rebind_in_progress()
            && !self.modal_active() && !ctx.text_edit_focused();
        if speed_shortcut_active && ctx.input_mut(take_speed_limit_shortcut) {
            let limited = nexium_common::speed_limit::toggle();
            log::info!("Emulation speed: {}", if limited { "limited" } else { "unlocked" });
            if let Some(path) = &self.playing_path {
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.game_window_title(path)));
            }
            ctx.request_repaint();
        }
        if speed_shortcut_active
            && ctx.input_mut(|input| take_ctrl_shortcut(input, egui::Key::O))
        {
            self.app_settings.perf_overlay_hidden = !self.app_settings.perf_overlay_hidden;
            let _ = self.app_settings.save();
            ctx.request_repaint();
        }
        let overlay_key_held = speed_shortcut_active
            && ctx.input(|i| i.modifiers.ctrl && i.key_down(egui::Key::O));
        let speed_key_held = speed_shortcut_active
            && ctx.input(|i| i.modifiers.ctrl && i.key_down(egui::Key::U));
        if speed_shortcut_active
            && ctx.input_mut(|input| take_ctrl_shortcut(input, egui::Key::B))
        {
            self.app_settings.game_bar = !self.app_settings.game_bar;
            let _ = self.app_settings.save();
            ctx.request_repaint();
        }
        let recenter_key = motion_recenter_key(&self.app_settings.motion_recenter_key)
            .filter(|key| recenter_key_conflict(*key, &self.controller_config).is_none())
            .filter(|_| {
                speed_shortcut_active
                    && self.game_input_active()
                    && !self.prefs_key_editing
                    && !self.profile.name_editing
            });
        let key_recenter = recenter_key
            .is_some_and(|key| ctx.input_mut(|input| take_plain_shortcut(input, key)));
        let button_down = self.input.as_ref().is_some_and(|ib| {
            ib.motion_attached() && ib.recenter_button_down(self.app_settings.motion_recenter_button)
        });
        let button_recenter = rising_edge(&mut self.motion_recenter_held, button_down);
        let button_active = self.emulation_handle.as_ref().is_some_and(|h| h.is_running())
            && !self.rebind_in_progress()
            && !self.modal_active();
        if key_recenter || (button_active && button_recenter) {
            self.recenter_motion();
        }

        if !self.app_settings.emulate_mouse {
            self.mouse_wheel_accum = egui::Vec2::ZERO;
        }
        if !self.rebind_in_progress() {
            let pressed: Vec<String> = ctx.input(|i| {
                let mut v = Vec::new();
                for ev in &i.events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = ev
                    {
                        if !(speed_key_held && *key == egui::Key::U)
                            && !(overlay_key_held && *key == egui::Key::O)
                            && recenter_key != Some(*key)
                        {
                            v.push(format!("{:?}", key));
                        }
                    }
                }
                for k in [
                    egui::Key::A,
                    egui::Key::B,
                    egui::Key::C,
                    egui::Key::D,
                    egui::Key::E,
                    egui::Key::F,
                    egui::Key::G,
                    egui::Key::H,
                    egui::Key::I,
                    egui::Key::J,
                    egui::Key::K,
                    egui::Key::L,
                    egui::Key::M,
                    egui::Key::N,
                    egui::Key::O,
                    egui::Key::P,
                    egui::Key::Q,
                    egui::Key::R,
                    egui::Key::S,
                    egui::Key::T,
                    egui::Key::U,
                    egui::Key::V,
                    egui::Key::W,
                    egui::Key::X,
                    egui::Key::Y,
                    egui::Key::Z,
                    egui::Key::Num0,
                    egui::Key::Num1,
                    egui::Key::Num2,
                    egui::Key::Num3,
                    egui::Key::Num4,
                    egui::Key::Num5,
                    egui::Key::Num6,
                    egui::Key::Num7,
                    egui::Key::Num8,
                    egui::Key::Num9,
                    egui::Key::ArrowUp,
                    egui::Key::ArrowDown,
                    egui::Key::ArrowLeft,
                    egui::Key::ArrowRight,
                    egui::Key::Enter,
                    egui::Key::Tab,
                    egui::Key::Space,
                    egui::Key::Escape,
                ] {
                    if i.key_down(k) && !(speed_key_held && k == egui::Key::U)
                        && !(overlay_key_held && k == egui::Key::O)
                        && recenter_key != Some(k)
                    {
                        let s = format!("{:?}", k);
                        if !v.contains(&s) {
                            v.push(s);
                        }
                    }
                }
                v
            });

            let (kb_buttons, kb_sticks) = self.controller_config.buttons_pressed(&pressed);
            let (gp_buttons, gp_sticks) = if self.last_input.connected {
                self.last_input.to_npad()
            } else {
                (0u64, [0i32; 4])
            };
            let swkbd_open = self.vk_target == VkTarget::Swkbd && self.vkeyboard.active();
            let mut buttons = kb_buttons | gp_buttons;
            let mut sticks = kb_sticks;
            for i in 0..4 {
                if gp_sticks[i].abs() > sticks[i].abs() {
                    sticks[i] = gp_sticks[i];
                }
            }
            if swkbd_open {
                buttons = 0;
                sticks = [0i32; 4];
            }
            if !swkbd_open
                && std::env::var_os("NEXIUM_TEST_AUTOPRESS_A").is_some()
                && self
                    .emulation_handle
                    .as_ref()
                    .is_some_and(|handle| handle.is_running())
            {
                let elapsed = ctx.input(|input| input.time) - 30.0;
                if elapsed >= 0.0 && elapsed % 8.0 < 0.25 {
                    buttons |= SwitchButton::A.npad_bit();
                }
            }
            if diagnostics_enabled()
                && (buttons != self.last_buttons_logged || sticks != self.last_sticks_logged)
            {
                self.last_buttons_logged = buttons;
                self.last_sticks_logged = sticks;
                log::info!(
                    "input change: pressed={:?} buttons={:#x} sticks={:?}",
                    pressed,
                    buttons,
                    sticks
                );
            }
            let state = nexium_core::hid_state::get_hid_state();
            let mut hid = state.lock();
            hid.update_input(nexium_core::hid_state::ControllerInput {
                buttons,
                stick_l_x: sticks[0],
                stick_l_y: sticks[1],
                stick_r_x: sticks[2],
                stick_r_y: sticks[3],
            });
            hid.publish_motion();

            let game_input_active = self.game_input_active();
            let keyboard_blocked = !game_input_active || ctx.text_edit_focused();
            let keyboard = ctx.input(|input| guest_keyboard_input(
                input,
                self.app_settings.emulate_keyboard,
                keyboard_blocked,
                speed_key_held,
                overlay_key_held,
                recenter_key,
            ));
            let pointer = ctx.input(|input| {
                if !game_input_active || !input.focused {
                    return None;
                }
                guest_pointer_position(input.pointer.latest_pos()?, self.last_game_rect?)
            });
            let pointer_blocked = ctx.egui_is_using_pointer()
                || ctx.input(|input| input.pointer.latest_pos())
                    .and_then(|position| ctx.layer_id_at(position))
                    .is_some_and(|layer| layer.order != egui::Order::Background);
            let mut mouse = nexium_core::hid_state::MouseInput::default();
            let mut touch = hid.touch;
            touch.pressed = false;
            if self.app_settings.emulate_mouse || self.app_settings.emulate_touch {
                let previous = hid.mouse;
                mouse = nexium_core::hid_state::MouseInput {
                    connected: self.app_settings.emulate_mouse,
                    buttons: 0,
                    ..previous
                };
                let wheel_delta = if ctx.current_pass_index() == 0 {
                    ctx.input(guest_wheel_delta)
                } else {
                    egui::Vec2::ZERO
                };
                let wheel_accum = &mut self.mouse_wheel_accum;
                ctx.input(|i| {
                    if let Some([x, y]) = pointer {
                        mouse.x = x;
                        mouse.y = y;
                    }
                    if pointer.is_some() && !pointer_blocked {
                        if self.app_settings.emulate_mouse {
                            *wheel_accum += wheel_delta;
                            mouse.wheel_x = wheel_accum.x as i32;
                            mouse.wheel_y = wheel_accum.y as i32;
                        }
                        if i.pointer.primary_down() {
                            mouse.buttons |= nexium_core::hid_state::MOUSE_BUTTON_LEFT;
                        }
                        if i.pointer.secondary_down() {
                            mouse.buttons |= nexium_core::hid_state::MOUSE_BUTTON_RIGHT;
                        }
                        if i.pointer.middle_down() {
                            mouse.buttons |= nexium_core::hid_state::MOUSE_BUTTON_MIDDLE;
                        }
                        if i.pointer.button_down(egui::PointerButton::Extra1) {
                            mouse.buttons |= nexium_core::hid_state::MOUSE_BUTTON_BACK;
                        }
                        if i.pointer.button_down(egui::PointerButton::Extra2) {
                            mouse.buttons |= nexium_core::hid_state::MOUSE_BUTTON_FORWARD;
                        }
                    }
                });
                if self.app_settings.emulate_touch {
                    touch = nexium_core::hid_state::TouchInput {
                        x: mouse.x.max(0) as u32,
                        y: mouse.y.max(0) as u32,
                        pressed: pointer.is_some() && !pointer_blocked
                            && ctx.input(|i| i.pointer.primary_down()),
                    };
                }
                if !self.app_settings.emulate_mouse {
                    mouse = nexium_core::hid_state::MouseInput::default();
                }
            }
            hid.update_devices(mouse, keyboard, touch);
        } else {
            let state = nexium_core::hid_state::get_hid_state();
            let mut hid = state.lock();
            let mouse = nexium_core::hid_state::MouseInput { buttons: 0, ..hid.mouse };
            let keyboard = nexium_core::hid_state::KeyboardInput {
                connected: self.app_settings.emulate_keyboard,
                ..Default::default()
            };
            let touch = nexium_core::hid_state::TouchInput { pressed: false, ..hid.touch };
            hid.update_input(Default::default());
            hid.publish_motion();
            hid.update_devices(mouse, keyboard, touch);
        }

        let rumble_live = self.game_input_active() && crate::boot::emu_alive();
        let rumble_strength = if self.app_settings.vibration_enabled {
            self.app_settings.vibration_strength
        } else {
            0
        };
        if let Some(ib) = self.input.as_mut() {
            ib.set_guest_rumble(rumble_live, rumble_strength);
        }

        self.poll_frames(ctx);

        if self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| !h.is_running())
        {
            self.emulation_handle = None;
            self.pause_anim = None;
            if self.stop_anim.is_none() && self.pill_fade.is_none() {
                self.playing_path = None;
            }
        }

        let fps = self.performance.get_fps();
        let running = self.is_running();
        let paused = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_paused());

        if running && !paused {
            let dt = ctx.input(|i| i.stable_dt).min(0.5) as f64;
            if let Some(p) = self.playing_path.clone() {
                self.play_times.add(&p, dt);
            }
        }
        if self.last_playtime_save.elapsed() >= std::time::Duration::from_secs(20) {
            self.play_times.save_if_dirty();
            self.last_playtime_save = std::time::Instant::now();
        }

        let playing_alpha = if let Some(start) = self.pill_fade {
            let f = start.elapsed().as_secs_f32() / 0.40;
            if f >= 1.0 {
                self.pill_fade = None;
                self.playing_path = None;
                0.0
            } else {
                1.0 - f * f
            }
        } else {
            1.0
        };
        let playing_index = self
            .playing_path
            .as_ref()
            .and_then(|p| self.library.index_of_path(p));

        {
            let ptarget = if self.show_profile { 1.0 } else { 0.0 };
            let pdt = ctx.input(|i| i.stable_dt).min(0.1);
            let speed = 3.6;
            if self.profile_anim < ptarget {
                self.profile_anim = (self.profile_anim + pdt * speed).min(ptarget);
            } else if self.profile_anim > ptarget {
                self.profile_anim = (self.profile_anim - pdt * speed).max(ptarget);
            }
        }
        let profile_showing = self.show_profile || self.profile_anim > 0.0;

        let carousel_mode = self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel
            && (!running
                || self.game_display().is_none()
                || paused
                || self.stop_anim.is_some()
                || self.pill_fade.is_some()
                || self.resume_anim.is_some());

        let show_chrome = self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel;

        if show_chrome {
            egui::Panel::top("topbar")
                .exact_size(36.0)
                .frame(
                    egui::Frame::NONE
                        .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                        .stroke(Stroke::new(1.0_f32, BORDER)),
                )
                .show(ui, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(12.0);

                        ui.label(
                            egui::RichText::new("NeXium")
                                .size(13.5)
                                .strong()
                                .color(TEXT),
                        );

                        ui.add_space(8.0);
                        ui.painter().vline(
                            ui.cursor().left(),
                            ui.max_rect().y_range(),
                            Stroke::new(1.0_f32, BORDER),
                        );
                        ui.add_space(8.0);

                        ui.menu_button(egui::RichText::new("File").size(13.0).color(TEXT), |ui| {
                            if ui.button("Open game…").clicked() {
                                if let Some(p) = rfd::FileDialog::new()
                                    .add_filter("Switch games", &["nro", "dxci", "dnsp"])
                                    .pick_file()
                                {
                                    self.open_game_file(p, crate::firmware_prompt::Then::Select, ctx);
                                }
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Exit").clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                        ui.menu_button(
                            egui::RichText::new("Emulation").size(13.0).color(TEXT),
                            |ui| {
                                ui.set_min_width(110.0);
                                if running {
                                    let label = if paused { "Resume" } else { "Pause" };
                                    if ui.button(label).clicked() {
                                        if let Some(h) = self.emulation_handle.as_ref() {
                                            if paused {
                                                h.resume();
                                            } else {
                                                h.pause();
                                            }
                                        }
                                        self.pause_anim = None;
                                        self.resume_anim = None;
                                        ui.close();
                                    }
                                    ui.separator();
                                }
                                if ui.button("Boot").clicked() {
                                    self.boot_nro(ctx);
                                    ui.close();
                                }
                                if ui.button("Stop").clicked() {
                                    self.stop_emulation();
                                    ui.close();
                                }
                            },
                        );
                        ui.menu_button(egui::RichText::new("Debug").size(13.0).color(TEXT), |ui| {
                            if ui.button("Memory").clicked() {
                                self.debugger.toggle_memory();
                                ui.close();
                            }
                            if ui.button("Registers").clicked() {
                                self.debugger.toggle_registers();
                                ui.close();
                            }
                            if ui.button("Disassembler").clicked() {
                                self.debugger.toggle_disasm();
                                ui.close();
                            }
                            if ui.button("Wait Tree").clicked() {
                                self.debugger.toggle_wait_tree();
                                ui.close();
                            }
                            if ui.button("Logs").clicked() {
                                self.debugger.toggle_logs();
                                ui.close();
                            }
                            if ui.button("Performance Tuning...").clicked() {
                                self.debugger.toggle_performance();
                                ui.close();
                            }
                            ui.separator();
                            let mut dumps_on = nexium_common::dumps::enabled();
                            if ui.checkbox(&mut dumps_on, "Frame dumps (.bmp)").changed() {
                                nexium_common::dumps::set_enabled(dumps_on);
                            }
                        });
                        ui.menu_button(
                            egui::RichText::new("Settings").size(13.0).color(TEXT),
                            |ui| {
                                if ui.button("Preferences").clicked() {
                                    self.show_settings = !self.show_settings;
                                    ui.close();
                                }
                            },
                        );
                        ui.add_space(2.0);
                        self.nextendo_menu_button(ui);

                        let left_end = ui.min_rect().max.x;
                        let paused = false;
                        let (run_icon, status, status_col) = if paused {
                            (StatusIcon::Pause, "Paused", AMBER)
                        } else if running {
                            (StatusIcon::Play, "Running", GREEN)
                        } else {
                            (StatusIcon::Stop, "Idle", MUTED)
                        };
                        let mode_txt = if nexium_core::hid_state::is_docked() {
                            "Docked"
                        } else {
                            "Handheld"
                        };
                        let motion_enabled = self.app_settings.motion_enabled;
                        let vibration_enabled = self.app_settings.vibration_enabled;
                        let (label_saving, state_saving) = ui.fonts_mut(|f| {
                            let mut width = |text: String, size: f32| {
                                f.layout_no_wrap(text, egui::FontId::proportional(size), MUTED)
                                    .size()
                                    .x
                            };
                            let mut label_saving =
                                width(status.to_string(), 12.0) + width(mode_txt.to_string(), 12.0) + 16.0;
                            label_saving += width("⊞ Carousel".to_string(), 12.5) - width("⊞".to_string(), 12.5);
                            let mut state_saving = width(bar_toggle_label("Motion", motion_enabled, false), 12.0)
                                - width(bar_toggle_label("Motion", motion_enabled, true), 12.0);
                            state_saving += width(bar_toggle_label("Vibration", vibration_enabled, false), 12.0)
                                - width(bar_toggle_label("Vibration", vibration_enabled, true), 12.0);
                            (label_saving, state_saving)
                        });
                        let compact = self.top_bar_spare < if self.top_bar_compact { 24.0 } else { 0.0 };
                        let terse = self.top_bar_terse_spare < if self.top_bar_terse { 24.0 } else { 0.0 };

                        let right_start = ui
                            .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.add_space(12.0);
                                let (fps_col, fps_str) = if fps >= 55.0 {
                                    (GREEN, format!("{:.0} fps", fps))
                                } else if fps >= 28.0 {
                                    (AMBER, format!("{:.0} fps", fps))
                                } else {
                                    (DANGER, format!("{:.0} fps", fps))
                                };
                                ui.label(
                                    egui::RichText::new(fps_str)
                                        .size(12.0)
                                        .color(fps_col)
                                        .monospace(),
                                );
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                                ui.add_space(6.0);
                                if compact {
                                    status_icon(ui, run_icon, status_col).on_hover_text(status);
                                } else {
                                    ui.label(egui::RichText::new(status).size(12.0).color(status_col));
                                    ui.add_space(2.0);
                                    status_icon(ui, run_icon, status_col);
                                }
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                                ui.add_space(6.0);
                                self.draw_vibration_toggle(ctx, ui, terse);
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                                ui.add_space(6.0);
                                self.draw_motion_toggle(ctx, ui, terse);
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                                ui.add_space(6.0);
                                self.draw_docked_toggle(ctx, ui, compact);
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                                ui.add_space(6.0);
                                let carousel = if compact {
                                    pill_button(ui, "⊞", false).on_hover_text("Carousel view")
                                } else {
                                    pill_button(ui, "⊞ Carousel", false)
                                };
                                if carousel.clicked() {
                                    self.app_settings.view_mode =
                                        crate::app_settings::ViewMode::Carousel;
                                    let _ = self.app_settings.save();
                                }
                                ui.min_rect().min.x
                            })
                            .inner;
                        let spare = right_start - left_end - 12.0;
                        self.top_bar_spare = spare
                            - if terse {
                                label_saving + state_saving
                            } else if compact {
                                label_saving
                            } else {
                                0.0
                            };
                        self.top_bar_terse_spare = self.top_bar_spare + label_saving;
                        self.top_bar_compact = compact;
                        self.top_bar_terse = terse;
                    });
                });
        }

        if show_chrome {
            egui::Panel::bottom("statusbar")
                .exact_size(22.0)
                .frame(
                    egui::Frame::NONE
                        .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                        .stroke(Stroke::new(1.0_f32, BORDER)),
                )
                .show(ui, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(12.0);
                        let name = std::path::Path::new(&self.nro_path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "No file".into());
                        ui.label(egui::RichText::new(name).size(11.0).color(MUTED));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(12.0);
                            let stats = self
                                .emulation_handle
                                .as_ref()
                                .map(|h| h.stats.lock().clone())
                                .unwrap_or_default();
                            let building = nexium_common::shader_progress::in_flight();
                            let building_prefix = if building > 0 {
                                format!("Building Shader(s): {}  ·  ", building)
                            } else if nexium_common::shader_progress::recently_active() {
                                format!(
                                    "Built {} Shader(s)  ·  ",
                                    nexium_common::shader_progress::burst_built()
                                )
                            } else {
                                String::new()
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}{:.1} FPS  ·  Frame {:.1}ms  ·  SVCs {}  ·  Cycles {}",
                                    building_prefix,
                                    self.performance.get_fps(),
                                    self.performance.get_frame_time(),
                                    stats.svc_count,
                                    stats.cycle_count,
                                ))
                                .size(11.0)
                                .color(MUTED)
                                .monospace(),
                            );
                        });
                    });
                });
        }

        let game_bar_context = running
            && !carousel_mode
            && !show_chrome
            && self.game_display().is_some();
        if game_bar_context && self.app_settings.game_bar {
            self.draw_game_bar(ctx, ui);
        } else if game_bar_context {
            self.draw_game_bar_reveal_tab(ctx);
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG))
            .show(ui, |ui| {
                let full_rect = ui.max_rect();
                if self.nextendo_overlay_active() {
                    if carousel_mode {
                        let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                        let _ = crate::carousel::carousel_view(
                            &mut self.carousel,
                            &mut self.library,
                            ctx,
                            ui,
                            &self.last_input,
                            &mut self.input,
                            running,
                            playing_index,
                            playing_alpha,
                            self.app_settings.carousel_theme,
                            profile_tex,
                            &self.app_settings.profile_name,
                            false,
                            1.0,
                            1.0,
                            self.app_settings.backdrop_theme,
                            self.app_settings.light_mode,
                            &self.app_settings.favorites,
                            self.app_settings.eu_dates,
                            self.app_settings.dockbar_theme,
                            &self.app_settings.carousel_order,
                            &self.app_settings.carousel_lists,
                            None,
                            false,
                        );
                    }
                    self.draw_nextendo_hub(ctx, ui, full_rect, running);
                    ctx.request_repaint();
                } else if profile_showing {
                    let avatar_tex = self.profile_texture.as_ref().map(|t| t.id());
                    let ambient = self.carousel.ambient_color;
                    let accent = self.theme_accent();

                    let smooth = |x: f32| {
                        let x = x.clamp(0.0, 1.0);
                        x * x * (3.0 - 2.0 * x)
                    };
                    let p = self.profile_anim.clamp(0.0, 1.0);
                    let backdrop_split = 0.4;
                    let backdrop_opacity = smooth(p / backdrop_split);
                    let content_opacity = smooth((p - backdrop_split) / (1.0 - backdrop_split));

                    let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                    let carousel_scale = 1.0;
                    let carousel_alpha = 1.0;
                    let _ = crate::carousel::carousel_view(
                        &mut self.carousel,
                        &mut self.library,
                        ctx,
                        ui,
                        &self.last_input,
                        &mut self.input,
                        running,
                        playing_index,
                        playing_alpha,
                        self.app_settings.carousel_theme,
                        profile_tex,
                        &self.app_settings.profile_name,
                        false,
                        carousel_scale,
                        carousel_alpha,
                        self.app_settings.backdrop_theme,
                        self.app_settings.light_mode,
                        &self.app_settings.favorites,
                        self.app_settings.eu_dates,
                        self.app_settings.dockbar_theme,
                        &self.app_settings.carousel_order,
                        &self.app_settings.carousel_lists,
                        self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                        matches!(self.updater.status(), crate::updater::Status::Available(_)) && !self.updater.is_downloading(),
                    );

                    let profile_scale = 0.92 + 0.08 * content_opacity;
                    let profile_active = !self.modal_active() && !self.modal_active_frame_start;
                    let (upd_text, upd_color, upd_click) = self.update_status_display(ctx.input(|i| i.time) as f32);
                    let action = crate::profile::profile_view(
                        &mut self.profile,
                        &mut self.library,
                        &self.play_times,
                        ctx,
                        ui,
                        full_rect,
                        &self.app_settings.profile_name,
                        avatar_tex,
                        ambient,
                        accent,
                        content_opacity,
                        backdrop_opacity,
                        profile_scale,
                        self.app_settings.backdrop_theme,
                        self.app_settings.dockbar_theme,
                        self.app_settings.perf_overlay_corner,
                        self.app_settings.light_mode,
                        self.app_settings.music_volume,
                        self.app_settings.sfx_volume,
                        self.app_settings.eu_dates,
                        self.app_settings.music_muted,
                        self.app_settings.sfx_muted,
                        &upd_text,
                        upd_color,
                        upd_click,
                        profile_active,
                        &self.last_input,
                        &mut self.input,
                    );
                    if self.show_profile {
                        match action {
                            crate::profile::ProfileAction::Close => {
                                self.show_profile = false;
                                self.carousel.profile_focused = false;
                            }
                            crate::profile::ProfileAction::QuickLaunch(path) => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                self.confirm = Some(ConfirmDialog {
                                    title: "Quick Launch".into(),
                                    body: "Quick Launch this game? Any open game will be shut down first.".into(),
                                    confirm_label: "Launch".into(),
                                    selected: 0,
                                    kind: ConfirmKind::QuickLaunch(path),
                                });
                                self.modal_hold = true;
                            }
                            crate::profile::ProfileAction::StartUpdate => {
                                self.open_update_confirm();
                            }
                            crate::profile::ProfileAction::PickIcon => {
                                self.pick_profile_avatar();
                            }
                            crate::profile::ProfileAction::SetName(name) => {
                                self.app_settings.profile_name = name;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetBackdropTheme(theme) => {
                                if self.app_settings.backdrop_theme != theme {
                                    self.app_settings.backdrop_theme = theme;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetDockbarTheme(theme) => {
                                if self.app_settings.dockbar_theme != theme {
                                    self.app_settings.dockbar_theme = theme;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetOverlayCorner(corner) => {
                                if self.app_settings.perf_overlay_corner != corner {
                                    self.app_settings.perf_overlay_corner = corner;
                                    self.perf_pos = None;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetLightMode(on) => {
                                if self.app_settings.light_mode != on {
                                    self.app_settings.light_mode = on;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetMusicVolume(v) => {
                                self.app_settings.music_volume = v.clamp(0.0, 1.0);
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetSfxVolume(v) => {
                                self.app_settings.sfx_volume = v.clamp(0.0, 1.0);
                                if !self.app_settings.sfx_muted {
                                    crate::ui_audio::set_sfx_volume(self.app_settings.sfx_volume);
                                }
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetEuDates(on) => {
                                self.app_settings.eu_dates = on;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetMuteMusic(on) => {
                                self.app_settings.music_muted = on;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetMuteSfx(on) => {
                                self.app_settings.sfx_muted = on;
                                crate::ui_audio::set_sfx_volume(if on { 0.0 } else { self.app_settings.sfx_volume });
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::None => {}
                        }
                    }
                    ctx.request_repaint();
                } else if carousel_mode {
                    let booting = running && !paused && self.game_display().is_none();
                    if booting {
                        let bg_rect = ui.max_rect();
                        let painter = ui.painter();
                        let t = ui.input(|i| i.time) as f32;

                        painter.rect_filled(bg_rect, CornerRadius::ZERO, Color32::from_rgb(0x06, 0x06, 0x08));

                        let launched = std::path::PathBuf::from(&self.nro_path);
                        let game_index = self.library.index_of_path(&launched);
                        let game_info = game_index.and_then(|gi| self.library.games.get(gi)).map(|g| {
                            (g.title.clone(), g.format.to_string(), g.dominant_color)
                        });
                        let dom_color = game_info.as_ref().map(|(_, _, col)| *col).unwrap_or(Color32::from_rgb(0x2F, 0xB4, 0xEF));

                        let elapsed_awaiting = if let crate::carousel::BootStage::AwaitingFrame { start_time, .. } = self.carousel.boot_stage {
                            (t - start_time).max(0.0)
                        } else {
                            0.0
                        };
                        let fade_alpha = (elapsed_awaiting / 0.55).min(1.0);

                        let glow_intensity = (((t * 2.2).sin() * 0.12 + 0.18).clamp(0.0, 1.0)) * fade_alpha;
                        let center = bg_rect.center();
                        for r_offset in (0..12).rev() {
                            let r = 140.0 + r_offset as f32 * 12.0;
                            let alpha = (4.0 * (1.0 - r_offset as f32 / 12.0) * glow_intensity * fade_alpha) as u8;
                            let col = Color32::from_rgba_unmultiplied(dom_color.r(), dom_color.g(), dom_color.b(), alpha);
                            painter.circle_filled(center - Vec2::new(0.0, 48.0), r, col);
                        }

                        let card_sz = 260.0f32;
                        let card_rect = egui::Rect::from_center_size(center - Vec2::new(0.0, 48.0), Vec2::splat(card_sz));

                        crate::carousel::draw_gradient_rounded_rect(&painter, card_rect.center(), card_rect.expand(6.0), 16.0, t, (120.0 * fade_alpha) as u8);
                        crate::carousel::draw_gradient_rounded_rect(&painter, card_rect.center(), card_rect.expand(2.5), 14.0, t, (255.0 * fade_alpha) as u8);

                        painter.rect_filled(card_rect, CornerRadius::same(14), Color32::from_rgba_unmultiplied(0x14, 0x14, 0x1A, (fade_alpha * 255.0) as u8));
                        let card_tint = Color32::from_white_alpha((fade_alpha * 255.0) as u8);
                        if let Some(tex) = game_index.and_then(|gi| self.library.texture(ctx, gi)) {
                            crate::carousel::draw_rounded_image(&painter, tex.id(), card_rect, 14.0, card_tint);
                        } else if let Some((_, format, _)) = &game_info {
                            painter.text(card_rect.center(), egui::Align2::CENTER_CENTER, format, FontId::proportional(card_sz * 0.13), Color32::from_rgba_unmultiplied(0x70, 0x70, 0x80, (fade_alpha * 255.0) as u8));
                        }

                        if let Some((title, _, _)) = &game_info {
                            let text_y = center.y + card_sz * 0.5 + 24.0;
                            let title_alpha = (fade_alpha * 255.0) as u8;
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y), egui::Align2::CENTER_CENTER, title, FontId::proportional(28.0), Color32::from_rgba_unmultiplied(255, 255, 255, title_alpha), true);

                            let dots = match (t as u64) % 4 {
                                0 => "",
                                1 => ".",
                                2 => "..",
                                3 => "...",
                                _ => "...",
                            };
                            let await_text = format!("Awaiting First Frame{}", dots);
                            let await_alpha = (fade_alpha * 255.0) as u8;
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y + 40.0), egui::Align2::CENTER_CENTER, &await_text, FontId::proportional(15.0), Color32::from_rgba_unmultiplied(0x00, 0xE5, 0xFF, await_alpha), false);
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y + 74.0), egui::Align2::CENTER_CENTER, "[Home / `] Cancel", FontId::proportional(13.0), Color32::from_rgba_unmultiplied(0xC0, 0xC0, 0xCC, await_alpha), false);
                        }

                        request_idle_repaint(ctx, std::time::Duration::from_millis(16));
                    } else {
                        let panel = ui.max_rect();
                        let scale_f = if let Some(start) = self.pause_anim.or(self.stop_anim) {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            let e = 1.0 - (1.0 - f) * (1.0 - f);
                            0.90 + 0.10 * e
                        } else {
                            1.0
                        };
                        let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                        let modal_active = self.modal_active();
                        let action = crate::carousel::carousel_view(
                            &mut self.carousel,
                            &mut self.library,
                            ctx,
                            ui,
                            &self.last_input,
                            &mut self.input,
                            running,
                            playing_index,
                            playing_alpha,
                            self.app_settings.carousel_theme,
                            profile_tex,
                            &self.app_settings.profile_name,
                            !self.show_settings && !modal_active,
                            scale_f,
                            1.0,
                            self.app_settings.backdrop_theme,
                            self.app_settings.light_mode,
                            &self.app_settings.favorites,
                            self.app_settings.eu_dates,
                            self.app_settings.dockbar_theme,
                            &self.app_settings.carousel_order,
                            &self.app_settings.carousel_lists,
                            self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                            matches!(self.updater.status(), crate::updater::Status::Available(_)) && !self.updater.is_downloading(),
                        );
                        match action {
                            crate::carousel::CarouselAction::Launch(path) => {
                                self.request_launch(path, ctx, running);
                            }
                            crate::carousel::CarouselAction::SetTheme(th) => {
                                if self.app_settings.carousel_theme != th {
                                    self.app_settings.carousel_theme = th;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::carousel::CarouselAction::Resume => {
                                if let Some(h) = self.emulation_handle.as_ref() {
                                    h.resume();
                                }
                                self.pause_anim = None;
                                self.resume_anim = Some(std::time::Instant::now());
                            }
                            crate::carousel::CarouselAction::OpenProfile => {
                                self.profile.tab = crate::profile::ProfileTab::Profile;
                                self.profile.name_buf = self.app_settings.profile_name.clone();
                                self.profile.focus_content = false;
                                self.profile.row_selected = 0;
                                self.show_profile = true;
                            }
                            crate::carousel::CarouselAction::SwitchToGrid => {
                                self.app_settings.view_mode = crate::app_settings::ViewMode::Grid;
                                let _ = self.app_settings.save();
                            }
                            crate::carousel::CarouselAction::StopEmulation => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                self.confirm = Some(ConfirmDialog {
                                    title: "Close Game".into(),
                                    body: "Close the current game?".into(),
                                    confirm_label: "Close".into(),
                                    selected: 0,
                                    kind: ConfirmKind::CloseGame,
                                });
                                self.modal_hold = true;
                            }
                            crate::carousel::CarouselAction::OpenSettings => {
                                self.settings_tab = SettingsTab::General;
                                self.show_settings = true;
                            }
                            crate::carousel::CarouselAction::OpenController => {
                                self.settings_tab = SettingsTab::Controller;
                                self.show_settings = true;
                            }
                            crate::carousel::CarouselAction::OpenDebug => {
                                self.debugger.toggle_logs();
                            }
                            crate::carousel::CarouselAction::Quit => {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                            crate::carousel::CarouselAction::Rescan => {
                                self.library.rescan(ctx, &self.app_settings.library_folders);
                            }
                            crate::carousel::CarouselAction::AddFolder => {
                                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    if !self.app_settings.library_folders.contains(&dir) {
                                        self.app_settings.library_folders.push(dir);
                                        let _ = self.app_settings.save();
                                    }
                                    self.library.rescan(ctx, &self.app_settings.library_folders);
                                }
                            }
                            crate::carousel::CarouselAction::ToggleFavorite(path) => {
                                self.toggle_favorite_path(std::path::PathBuf::from(path));
                            }
                            crate::carousel::CarouselAction::DownloadIcon(path) => {
                                self.open_icon_picker(std::path::PathBuf::from(path));
                            }
                            crate::carousel::CarouselAction::ManageMods(path) => {
                                self.open_mod_manager(std::path::PathBuf::from(path), ctx);
                            }
                            crate::carousel::CarouselAction::ManageContent(path) => {
                                self.open_content_manager(std::path::PathBuf::from(path), ctx);
                            }
                            crate::carousel::CarouselAction::ViewGameInfo(path) => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                self.game_info_path = Some(std::path::PathBuf::from(path));
                            }
                            crate::carousel::CarouselAction::OpenShop => {
                                self.shop.open();
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                            }
                            crate::carousel::CarouselAction::OpenCarouselSettings => {
                                self.carousel_settings_open = true;
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                            }
                            crate::carousel::CarouselAction::OpenUpdate => {
                                self.open_update_confirm();
                            }
                            crate::carousel::CarouselAction::OpenNextendo => {
                                self.open_nextendo(crate::nextendo::hub::Tab::Overview);
                            }
                            crate::carousel::CarouselAction::None => {}
                        }

                        if let Some(start) = self.pause_anim.or(self.stop_anim) {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            if f >= 1.0 {
                                self.pause_anim = None;
                                if self.stop_anim.take().is_some() {
                                    self.game_texture = None;
                                    if self.playing_path.is_some() {
                                        self.pill_fade = Some(std::time::Instant::now());
                                    }
                                }
                            } else if let Some((tid, tsz)) = self.game_display() {
                                let ef = 1.0 - (1.0 - f) * (1.0 - f);
                                let rect = self.game_draw_rect(panel, tsz, ctx.pixels_per_point());
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("home_zoom"),
                                ));
                                draw_zoom_fade_out(&p, rect, tid, ef);
                            }
                            ctx.request_repaint();
                        } else if let Some(start) = self.resume_anim {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            if let Some((tid, tsz)) = self.game_display() {
                                let ef = 1.0 - (1.0 - f) * (1.0 - f);
                                let rect = self.game_draw_rect(panel, tsz, ctx.pixels_per_point());
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("home_zoom"),
                                ));
                                let cover = (ef * 1.5).min(1.0);
                                p.rect_filled(
                                    panel,
                                    CornerRadius::ZERO,
                                    Color32::from_rgba_unmultiplied(BG.r(), BG.g(), BG.b(), (cover * 255.0) as u8),
                                );
                                draw_zoom_fade_in(&p, rect, tid, ef);
                            }
                            if f >= 1.0 {
                                self.resume_anim = None;
                            }
                            ctx.request_repaint();
                        }
                    }
                } else if let Some(start) = self.stop_fade {
                    let t = start.elapsed().as_secs_f32() / 0.9;
                    let panel = ui.max_rect();
                    if self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel {
                        let scale_f = 0.92 + 0.08 * t.min(1.0);
                        let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                        let _ = crate::carousel::carousel_view(
                            &mut self.carousel,
                            &mut self.library,
                            ctx,
                            ui,
                            &self.last_input,
                            &mut self.input,
                            running,
                            playing_index,
                            playing_alpha,
                            self.app_settings.carousel_theme,
                            profile_tex,
                            &self.app_settings.profile_name,
                            true,
                            scale_f,
                            t.min(1.0),
                            self.app_settings.backdrop_theme,
                            self.app_settings.light_mode,
                            &self.app_settings.favorites,
                            self.app_settings.eu_dates,
                            self.app_settings.dockbar_theme,
                            &self.app_settings.carousel_order,
                            &self.app_settings.carousel_lists,
                            self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                            matches!(self.updater.status(), crate::updater::Status::Available(_)) && !self.updater.is_downloading(),
                        );
                        if t < 1.05 {
                            if let Some((tid, tsz)) = self.game_display() {
                                let rect = self.game_draw_rect(panel, tsz, ctx.pixels_per_point());
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("stop_dissolve"),
                                ));
                                draw_zoom_fade_out(&p, rect, tid, t);
                            }
                        }
                    } else {
                        let display = self.game_display();
                        self.library_view(ui, ctx);
                        if let Some((tid, tsz)) = display {
                            let rect = self.game_draw_rect(panel, tsz, ctx.pixels_per_point());
                            let p = ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Foreground,
                                egui::Id::new("stop_dissolve"),
                            ));
                            draw_dissolve(&p, rect, tid, t);
                        }
                    }
                    ctx.request_repaint();
                    if t >= 1.05 {
                        self.stop_fade = None;
                        self.game_texture = None;
                        self.free_native_texture();
                    }
                } else if running {
                    if let Some((tid, tsz)) = self.game_display() {
                        let draw_rect = self.game_draw_rect(ui.max_rect(), tsz, ctx.pixels_per_point());
                        self.last_game_rect = Some(draw_rect);
                        ui.allocate_rect(draw_rect, egui::Sense::hover());
                        if self.native_game.as_ref().is_some_and(|window| window.active)
                            && self.native_game_visible(carousel_mode)
                        {
                            ui.painter().rect_filled(draw_rect, 0.0, Color32::BLACK);
                        } else {
                            let tid = self.filtered_game_texture(draw_rect, ctx.pixels_per_point()).unwrap_or(tid);
                            egui::Image::new((tid, draw_rect.size())).paint_at(ui, draw_rect);
                        }
                        if let Some(depth) = self.game_depth.clone() {
                            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                                draw_rect,
                                crate::depth_emit::DepthEmit {
                                    depth,
                                    seq: self.game_depth_seq,
                                    rect: draw_rect,
                                },
                            ));
                        }
                    } else {
                        self.library_view(ui, ctx);
                    }
                } else {
                    if self.game_texture.is_some() || self.game_texture_native.is_some() {
                        self.game_texture = None;
                        self.free_native_texture();
                    }
                    self.library_view(ui, ctx);
                }
                self.draw_modal(ctx, ui);
                let game_info_action = self.draw_game_info(ctx);
                if let Some(manager) = &mut self.mod_manager {
                    if !manager.show(ctx, &self.last_input) {
                        self.mod_manager = None;
                    }
                }
                let content_game_running = self.emulation_handle.as_ref().is_some_and(|handle| handle.is_running());
                if let Some(manager) = &mut self.content_manager {
                    if !manager.show(ctx, &self.last_input, content_game_running) {
                        self.content_manager = None;
                    }
                }
                if !self.modal_active() {
                    let dropped = ctx.input(|input| {
                        input.raw.dropped_files.iter()
                            .map(|file| file.path().to_path_buf())
                            .find(|path| crate::firmware_prompt::may_contain_firmware(path))
                    });
                    if let Some(path) = dropped {
                        self.open_game_file(path, crate::firmware_prompt::Then::Nothing, ctx);
                    }
                }
                if let Some(prompt) = &mut self.firmware_prompt {
                    if let crate::firmware_prompt::Outcome::Done(then, path) = prompt.show(ctx, &self.last_input, content_game_running) {
                        self.firmware_prompt = None;
                        self.firmware_status = None;
                        match then {
                            crate::firmware_prompt::Then::Boot => {
                                self.nro_path = path.to_string_lossy().to_string();
                                self.boot_nro(ctx);
                            }
                            crate::firmware_prompt::Then::Select => self.nro_path = path.to_string_lossy().to_string(),
                            crate::firmware_prompt::Then::Nothing => {}
                        }
                    }
                }
                self.update_icon_picker(ctx, ui);
                if let Some(action) = game_info_action {
                    match action {
                        GameInfoAction::Launch(path) => {
                            self.request_launch(path.to_string_lossy().to_string(), ctx, running);
                        }
                        GameInfoAction::ManageMods(path) => {
                            self.open_mod_manager(path, ctx);
                        }
                        GameInfoAction::ManageContent(path) => {
                            self.open_content_manager(path, ctx);
                        }
                        GameInfoAction::ToggleFavorite(path) => {
                            self.toggle_favorite_path(path);
                        }
                        GameInfoAction::DownloadIcon(path) => {
                            self.open_icon_picker(path);
                        }
                    }
                }
                if running
                    && !carousel_mode
                    && self.game_display().is_some()
                    && self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel
                {
                    if !self.app_settings.perf_overlay_hidden {
                        self.draw_perf_overlay(ctx, ui);
                    }
                }
                {
                    let accent = self.theme_accent();
                    self.shop.update(ctx, ui, self.app_settings.light_mode, accent, &self.last_input);
                }
                if let Some((title, since)) = &self.download_toast {
                    let el = since.elapsed().as_secs_f32();
                    if el > 4.5 {
                        self.download_toast = None;
                    } else {
                        let fade = (el.min(0.3) / 0.3).min((4.5 - el) / 0.5).clamp(0.0, 1.0);
                        let screen = ctx.viewport_rect();
                        let mut tp = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("dl_toast")));
                        tp.set_opacity(fade);
                        let w = 360.0;
                        let rect = egui::Rect::from_min_size(egui::pos2(screen.center().x - w * 0.5, screen.min.y + 24.0), egui::Vec2::new(w, 62.0));
                        tp.rect_filled(rect.translate(egui::Vec2::new(0.0, 4.0)), egui::CornerRadius::same(14), egui::Color32::from_black_alpha(90));
                        tp.rect_filled(rect, egui::CornerRadius::same(14), egui::Color32::from_rgb(0x1C, 0x24, 0x1E));
                        tp.rect_stroke(rect, egui::CornerRadius::same(14), egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(0x35, 0xD0, 0x6A)), egui::StrokeKind::Outside);
                        let cc = egui::pos2(rect.min.x + 30.0, rect.center().y);
                        tp.circle_filled(cc, 12.0, egui::Color32::from_rgb(0x35, 0xD0, 0x6A));
                        tp.add(egui::Shape::line(vec![cc + egui::Vec2::new(-5.0, 0.0), cc + egui::Vec2::new(-1.5, 4.0), cc + egui::Vec2::new(6.0, -5.0)], egui::Stroke::new(2.2_f32, egui::Color32::WHITE)));
                        tp.text(egui::pos2(rect.min.x + 54.0, rect.center().y - 9.0), egui::Align2::LEFT_CENTER, "Download Complete!", egui::FontId::proportional(16.0), egui::Color32::WHITE);
                        tp.text(egui::pos2(rect.min.x + 54.0, rect.center().y + 11.0), egui::Align2::LEFT_CENTER, title, egui::FontId::proportional(13.0), egui::Color32::from_gray(0xB0));
                        ctx.request_repaint();
                    }
                }
                self.draw_motion_toast(ctx);
                self.update_carousel_settings(ctx, ui);
                if self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel {
                    self.update_preferences(ctx, ui);
                }
                self.drive_vkeyboard(ctx, ui);
                self.drive_updater(ctx);
            });

        if self.show_settings
            && self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel
        {
            let mut open = self.show_settings;
            let mut tab = self.settings_tab;
            let mut cfg = self.controller_config.clone();
            let mut rebinding = self.rebinding;
            let mut save_needed = false;
            let mut app_cfg = self.app_settings.clone();
            let mut app_save_needed = false;
            let active_gpu = self.active_gpu_name();
            let mut test_key: Option<String> = None;
            let last_input = self.last_input;
            let gp_name = self.input.as_ref().and_then(|ib| ib.name());
            let mut rebinding_pad = self.rebinding_pad;
            let mut input_device = self.input_device;
            let mut motion_key_capture = self.motion_key_capture;
            let mut motion_key_notice = self.motion_key_notice.clone();
            let motion_button_missing = self.motion_button_missing();
            let startup_motion_enabled = self.startup_motion_enabled;
            let mut recenter_now = false;
            let mut calibrate_now = false;
            let motion_source = self.input.as_ref().and_then(|ib| ib.motion_source());
            let (calibration, calibration_note) = motion_calibration_text(
                self.app_settings.motion_enabled,
                nexium_core::hid_motion::calibration_states(),
                nexium_core::hid_motion::stale_sources(),
                self.input.as_ref().and_then(|ib| ib.motion_hint_note()),
            );
            let motion_values = motion_row_values(
                &self.app_settings,
                motion_source.as_deref(),
                self.motion_key_capture,
                self.motion_recently_recentered(),
                calibration,
            );
            let vibration_values = vibration_row_values(&self.app_settings, &self.vibration_status());
            let mut vibration_test = false;
            let pad_list = self.input.as_ref().map(|ib| ib.list_gamepads()).unwrap_or_default();
            let active_pad = self.input.as_ref().and_then(|ib| ib.get_active_id());
            let mut pad_pick = None;
            let firmware_status = self.firmware_status();
            let mut firmware_action = None;

            let screen = ctx.viewport_rect();
            let max_h = (screen.height() - 80.0).clamp(360.0, 760.0);
            let max_w = (screen.width() - 80.0).clamp(520.0, 980.0);
            let body_shown = egui::Window::new("Preferences")
                .open(&mut open)
                .resizable(true)
                .default_size([max_w.min(900.0), max_h.min(560.0)])
                .max_height(max_h)
                .max_width(max_w)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(tab == SettingsTab::General, "General")
                            .clicked()
                        {
                            tab = SettingsTab::General;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Controller, "Controller")
                            .clicked()
                        {
                            tab = SettingsTab::Controller;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Graphics, "Graphics")
                            .clicked()
                        {
                            tab = SettingsTab::Graphics;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Audio, "Audio")
                            .clicked()
                        {
                            tab = SettingsTab::Audio;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Emulation, "Emulation")
                            .clicked()
                        {
                            tab = SettingsTab::Emulation;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Logging, "Logging")
                            .clicked()
                        {
                            tab = SettingsTab::Logging;
                        }
                    });
                    ui.separator();
                    ui.add_space(6.0);

                    match tab {
                        SettingsTab::General => {
                            settings_content(ui);
                            ui.add_space(10.0);
                            ui.separator();
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("SteamGridDB").strong());
                            ui.label(
                                egui::RichText::new(
                                    "Paste your API key to download custom game icons.",
                                )
                                .weak()
                                .size(12.0),
                            );
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("API Key");
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut app_cfg.steamgriddb_key)
                                        .password(true)
                                        .hint_text("Paste key here")
                                        .desired_width(260.0),
                                );
                                if resp.changed() {
                                    app_save_needed = true;
                                }
                                if ui.button("Test").clicked() {
                                    test_key = Some(app_cfg.steamgriddb_key.clone());
                                }
                            });
                            ui.hyperlink_to(
                                "Get a key at steamgriddb.com/profile/preferences/api",
                                "https://www.steamgriddb.com/profile/preferences/api",
                            );
                        }
                        SettingsTab::Controller => {
                            controller_settings_content(
                                ui,
                                &mut cfg,
                                &mut rebinding,
                                &mut rebinding_pad,
                                &mut save_needed,
                                &last_input,
                                gp_name.as_deref(),
                                &mut input_device,
                                &mut app_cfg,
                                &mut app_save_needed,
                                &mut motion_key_capture,
                                &mut motion_key_notice,
                                &motion_values,
                                motion_button_missing,
                                &mut recenter_now,
                                calibration_note,
                                &mut calibrate_now,
                                &vibration_values,
                                &mut vibration_test,
                                startup_motion_enabled,
                                &pad_list,
                                active_pad,
                                &mut pad_pick,
                            );
                        }
                        SettingsTab::Graphics => {
                            graphics_settings_content(
                                ui,
                                &mut app_cfg,
                                &mut app_save_needed,
                                self.startup_gpu_device.as_deref(),
                                &active_gpu,
                            );
                        }
                        SettingsTab::Audio => {
                            audio_settings_content(
                                ui,
                                &mut app_cfg,
                                &mut app_save_needed,
                                &mut self.audio_device_cache,
                            );
                        }
                        SettingsTab::Emulation => {
                            emulation_settings_content(ui, &mut app_cfg, &mut app_save_needed);
                            firmware_action = firmware_settings_content(ui, &firmware_status);
                        }
                        SettingsTab::Logging => {
                            logging_settings_content(ui, &mut app_cfg, &mut app_save_needed);
                        }
                    }
                })
                .is_some_and(|response| response.inner.is_some());

            self.show_settings = open;
            self.settings_tab = tab;
            self.controller_config = cfg;
            self.rebinding = rebinding.filter(|_| body_shown);
            self.rebinding_pad = rebinding_pad.filter(|_| body_shown);
            self.input_device = input_device;
            self.app_settings = app_cfg;
            if let (Some(id), Some(ib)) = (pad_pick, self.input.as_mut()) {
                self.app_settings.gamepad = ib.set_active_id(id);
                app_save_needed = true;
            }
            self.motion_key_capture = motion_key_capture
                && body_shown
                && open
                && tab == SettingsTab::Controller
                && input_device == InputDevice::Motion;
            self.motion_key_notice = motion_key_notice;
            if save_needed {
                if let Err(e) = self.controller_config.save() {
                    log::warn!("Failed to save controller config: {}", e);
                }
            }
            if app_save_needed {
                if let Err(e) = self.app_settings.save() {
                    log::warn!("Failed to save app settings: {}", e);
                }
            }
            if recenter_now {
                self.recenter_motion();
            }
            if calibrate_now {
                self.calibrate_motion();
            }
            match firmware_action {
                Some(FirmwareSettingsAction::Install) => self.install_firmware_from_settings(ctx),
                Some(FirmwareSettingsAction::Remove) => self.remove_firmware_from_settings(),
                None => {}
            }
            if vibration_test {
                self.test_vibration();
            }
            if let Some(key) = test_key {
                let state = std::sync::Arc::new(std::sync::Mutex::new(KeyTest::default()));
                self.key_test = Some(state.clone());
                self.key_test_result = None;
                let ctx2 = ctx.clone();
                std::thread::spawn(move || {
                    let ok = crate::steamgrid::verify_key(&key);
                    if let Ok(mut g) = state.lock() {
                        g.done = true;
                        g.ok = ok;
                    }
                    ctx2.request_repaint();
                });
            }
        }

        if let Some(state) = &self.key_test {
            let mut finished: Option<bool> = None;
            if let Ok(g) = state.lock() {
                if g.done {
                    finished = Some(g.ok);
                }
            }
            if let Some(ok) = finished {
                self.key_test = None;
                self.key_test_result = Some((ok, std::time::Instant::now()));
                crate::ui_audio::play(if ok {
                    crate::ui_audio::Sfx::Whistle
                } else {
                    crate::ui_audio::Sfx::Error
                });
            }
        }
        if self.key_test.is_some() {
            egui::Window::new("SteamGridDB")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Testing connection…");
                    });
                    ui.add_space(4.0);
                });
            ctx.request_repaint();
        } else if let Some((ok, _)) = self.key_test_result {
            let mut still_open = true;
            egui::Window::new("SteamGridDB")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(6.0);
                    ui.vertical_centered(|ui| {
                        if ok {
                            ui.label(
                                egui::RichText::new("✔  Connected")
                                    .size(20.0)
                                    .strong()
                                    .color(Color32::from_rgb(0x35, 0xD0, 0x6A)),
                            );
                            ui.add_space(4.0);
                            ui.label("Your API key is valid. You can now download icons.");
                        } else {
                            ui.label(
                                egui::RichText::new("✖  Failed")
                                    .size(20.0)
                                    .strong()
                                    .color(Color32::from_rgb(0xE8, 0x33, 0x50)),
                            );
                            ui.add_space(4.0);
                            ui.label("Couldn't verify the key. Check the key and your connection.");
                        }
                        ui.add_space(10.0);
                        if ui.button("OK").clicked() {
                            still_open = false;
                        }
                    });
                    ui.add_space(6.0);
                });
            if !still_open {
                self.key_test_result = None;
            }
        }

        let snapshot = self.emulation_handle.as_ref().map(|h| h.snapshot());
        let mem_req_handle = self
            .emulation_handle
            .as_ref()
            .map(|h| Arc::clone(&h.mem_request));
        let mem_req: Option<Box<dyn Fn(u64)>> = mem_req_handle.map(|m| {
            Box::new(move |addr: u64| {
                *m.lock() = addr;
            }) as Box<dyn Fn(u64)>
        });
        debug_windows(
            ctx,
            &mut self.debugger,
            snapshot.as_ref(),
            mem_req.as_deref(),
            &self.log_buffer,
            &mut self.app_settings,
        );

        let toast_top = if self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel {
            0.0
        } else {
            36.0
        };
        self.draw_nextendo_toasts(ctx, toast_top);

        let native_visible = self.native_game_visible(carousel_mode);
        if let Some(window) = &mut self.native_game {
            let mut holes: Vec<_> = self.native_overlay_rect.into_iter().chain(self.game_bar_tab_rect).collect();
            holes.extend(self.nextendo_toasts.shown().iter().copied());
            if self.download_toast.is_some() {
                let screen = ctx.viewport_rect();
                holes.push(egui::Rect::from_min_size(egui::pos2(screen.center().x - 184.0, screen.min.y + 20.0), egui::vec2(368.0, 74.0)));
            }
            if self.motion_toast.is_some() {
                holes.push(motion_toast_rect(ctx.viewport_rect(), self.download_toast.is_some()).expand(6.0));
            }
            window.update(self.last_game_rect.filter(|_| native_visible), &holes, ctx.pixels_per_point(),
                nexium_common::speed_limit::enabled()
                    && crate::host_vsync_enabled(self.app_settings.vsync, std::env::var("NEXIUM_VSYNC").ok().as_deref()),
                self.app_settings.filter.scaling_filter(), self.app_settings.fsr_sharpness);
        }
        request_idle_repaint(
            ctx,
            std::time::Duration::from_millis(if frame_driven_game { 100 } else { 16 }),
        );
    }
}

fn idle_screen(ui: &mut egui::Ui, nro_path: &str, running: bool) -> u8 {
    let mut action = 255u8;
    let avail = ui.available_size();

    ui.vertical_centered(|ui| {
        ui.add_space((avail.y * 0.24).max(40.0));

        ui.label(
            egui::RichText::new("NeXium")
                .size(36.0)
                .strong()
                .color(TEXT),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Nintendo Switch Emulator")
                .size(13.0)
                .color(MUTED),
        );

        ui.add_space(36.0);

        let painter = ui.painter();
        let card_w = 380.0;
        let card_rect = egui::Rect::from_center_size(
            egui::pos2(ui.max_rect().center().x, ui.cursor().top() + 80.0),
            egui::vec2(card_w, 160.0),
        );
        painter.rect(
            card_rect,
            CornerRadius::same(10),
            BG_RAISED,
            Stroke::new(1.0_f32, BORDER),
            egui::StrokeKind::Outside,
        );

        ui.allocate_ui_with_layout(
            egui::vec2(card_w, 160.0),
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                ui.add_space(20.0);

                if nro_path.is_empty() {
                    ui.label(
                        egui::RichText::new("No file selected")
                            .size(13.0)
                            .color(MUTED),
                    );
                } else {
                    let fname = std::path::Path::new(nro_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| nro_path.to_string());
                    ui.label(egui::RichText::new(&fname).size(14.0).strong().color(TEXT));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(nro_path).size(10.5).color(MUTED));
                }

                ui.add_space(16.0);

                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    if pill_button(ui, "Select NRO…", false).clicked() {
                        action = 0;
                    }
                    ui.add_space(8.0);
                    if !nro_path.is_empty() && !running {
                        if pill_button(ui, "Boot", true).clicked() {
                            action = 1;
                        }
                    }
                    if running {
                        if pill_button(ui, "Stop", false).clicked() {
                            action = 2;
                        }
                    }
                });

                if running {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 8.0), Sense::hover());
                        ui.painter().circle_filled(rect.center(), 3.5, GREEN);
                        ui.label(
                            egui::RichText::new("Running — awaiting first frame")
                                .size(11.5)
                                .color(MUTED),
                        );
                    });
                }
            },
        );
    });

    action
}

fn settings_content(ui: &mut egui::Ui) {
    egui::Grid::new("prefs")
        .num_columns(2)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            row(ui, "CPU Backend", "Dynarmic (JIT)");
            row(ui, "GPU Backend", "Vulkan (ash)");
            row(ui, "Audio", "Enabled · 100%");
            row(ui, "Resolution", "1280 × 720");
        });
}

#[allow(clippy::too_many_arguments)]
fn controller_settings_content(
    ui: &mut egui::Ui,
    cfg: &mut ControllerConfig,
    rebinding: &mut Option<SwitchButton>,
    rebinding_pad: &mut Option<SwitchButton>,
    save_needed: &mut bool,
    input: &InputSnapshot,
    gp_name: Option<&str>,
    input_device: &mut InputDevice,
    app_cfg: &mut AppSettings,
    app_save_needed: &mut bool,
    motion_key_capture: &mut bool,
    motion_key_notice: &mut Option<String>,
    motion_values: &[String; 6],
    motion_button_missing: bool,
    recenter_now: &mut bool,
    calibration_note: Option<&str>,
    calibrate_now: &mut bool,
    vibration_values: &[String; 4],
    vibration_test: &mut bool,
    startup_motion_enabled: bool,
    pads: &[(sdl3::joystick::JoystickId, String)],
    active_pad: Option<sdl3::joystick::JoystickId>,
    pad_pick: &mut Option<Option<sdl3::joystick::JoystickId>>,
) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Player 1")
                .size(15.0)
                .strong()
                .color(TEXT),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if input.connected {
                let name = gp_name.unwrap_or("Gamepad");
                let pro = gp_name
                    .map(crate::input::is_pro_controller)
                    .unwrap_or(false);
                let tag = if pro {
                    "  ·  auto-mapped 1:1"
                } else {
                    "  ·  standard mapping"
                };
                ui.label(
                    egui::RichText::new(format!("{}{}", name, tag))
                        .size(12.0)
                        .color(GREEN),
                );
                ui.label(egui::RichText::new("●").size(13.0).color(GREEN));
            } else {
                ui.label(
                    egui::RichText::new("No gamepad detected")
                        .size(12.0)
                        .color(MUTED),
                );
                ui.label(egui::RichText::new("○").size(13.0).color(MUTED));
            }
        });
    });
    ui.label(egui::RichText::new("Switch Pro Controllers map 1:1 automatically. Other gamepads use a standard layout. Keyboard and gamepad both work at once.")
        .size(11.0).color(MUTED));
    ui.add_space(10.0);

    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.set_min_width(470.0);
            ui.set_max_width(470.0);
            draw_controller_diagram(ui, input);
        });
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Controller").size(12.0).color(MUTED));
                let saved = app_cfg.gamepad.as_deref();
                let choice = pad_choice_row(saved, pads, active_pad);
                let automatic = choice == 0;
                egui::ComboBox::from_id_salt("controller_sel")
                    .selected_text(pad_choice_text(saved, pads, active_pad))
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(automatic, PAD_AUTOMATIC).clicked() && !automatic {
                            *pad_pick = Some(None);
                        }
                        for (i, (id, name)) in pads.iter().enumerate() {
                            let current = choice == i + 1;
                            if ui.selectable_label(current, name.as_str()).clicked() && !current {
                                *pad_pick = Some(Some(*id));
                            }
                        }
                        if let Some(saved) = saved.filter(|_| choice > pads.len()) {
                            let label = format!("{} (not connected)", saved);
                            ui.add_enabled_ui(false, |ui| ui.selectable_label(true, label));
                        }
                    });
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Input device").size(12.0).color(MUTED));
                let sel = match *input_device {
                    InputDevice::Keyboard => "Keyboard".to_string(),
                    InputDevice::Gamepad => gp_name.unwrap_or("Gamepad").to_string(),
                    InputDevice::Motion => "Motion".to_string(),
                    InputDevice::Vibration => "Vibration".to_string(),
                };
                egui::ComboBox::from_id_salt("input_device_sel")
                    .selected_text(sel)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(input_device, InputDevice::Keyboard, "Keyboard");
                        ui.selectable_value(
                            input_device,
                            InputDevice::Gamepad,
                            gp_name.unwrap_or("Gamepad"),
                        );
                        ui.selectable_value(input_device, InputDevice::Motion, "Motion");
                        ui.selectable_value(input_device, InputDevice::Vibration, "Vibration");
                    });
            });
            ui.add_space(6.0);
            if *input_device == InputDevice::Motion {
                *rebinding = None;
                *rebinding_pad = None;
                motion_settings_content(
                    ui,
                    app_cfg,
                    app_save_needed,
                    cfg,
                    motion_key_capture,
                    motion_key_notice,
                    motion_values,
                    motion_button_missing,
                    recenter_now,
                    calibration_note,
                    calibrate_now,
                    startup_motion_enabled,
                );
                return;
            }
            if *input_device == InputDevice::Vibration {
                *rebinding = None;
                *rebinding_pad = None;
                vibration_settings_content(
                    ui,
                    app_cfg,
                    app_save_needed,
                    vibration_values,
                    vibration_test,
                );
                return;
            }

            if let Some(btn) = *rebinding {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Press a key for {}…", btn.display_name()))
                            .size(12.0)
                            .color(AMBER),
                    );
                    if ui.small_button("Cancel").clicked() {
                        *rebinding = None;
                    }
                });
                let new_key = ui.input(|i| {
                    for ev in &i.events {
                        if let egui::Event::Key {
                            key, pressed: true, ..
                        } = ev
                        {
                            return Some(format!("{:?}", key));
                        }
                    }
                    None
                });
                if let Some(k) = new_key {
                    cfg.set_binding(btn, k);
                    *rebinding = None;
                    *save_needed = true;
                }
            } else if let Some(btn) = *rebinding_pad {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Press a button for {}…", btn.display_name()))
                            .size(12.0)
                            .color(AMBER),
                    );
                    if ui.small_button("Cancel").clicked() {
                        *rebinding_pad = None;
                    }
                });
            } else {
                ui.label(
                    egui::RichText::new("Click Rebind, then press the key or button.")
                        .size(11.0)
                        .color(MUTED),
                );
            }
            ui.label(
                egui::RichText::new(motion_summary(app_cfg, cfg))
                    .size(11.0)
                    .color(MUTED),
            );
            ui.add_space(4.0);

            let list_h = (ui.available_height() - 46.0).max(160.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(list_h)
                .show(ui, |ui| {
                    egui::Grid::new("binds")
                        .num_columns(3)
                        .spacing([10.0, 6.0])
                        .striped(true)
                        .show(ui, |ui| match *input_device {
                            InputDevice::Keyboard => {
                                for btn in SwitchButton::all() {
                                    ui.label(
                                        egui::RichText::new(btn.display_name())
                                            .size(12.0)
                                            .color(TEXT),
                                    );
                                    let cur = cfg.binding_for(*btn).unwrap_or("—").to_string();
                                    ui.label(
                                        egui::RichText::new(cur)
                                            .size(12.0)
                                            .monospace()
                                            .color(MUTED),
                                    );
                                    if ui.small_button("Rebind").clicked() {
                                        *rebinding = Some(*btn);
                                        *rebinding_pad = None;
                                    }
                                    ui.end_row();
                                }
                            }
                            InputDevice::Gamepad => {
                                for btn in ControllerConfig::pad_list() {
                                    ui.label(
                                        egui::RichText::new(btn.display_name())
                                            .size(12.0)
                                            .color(TEXT),
                                    );
                                    let cur =
                                        cfg.pad_for(*btn).map(|g| g.display_name()).unwrap_or("—");
                                    ui.label(egui::RichText::new(cur).size(12.0).color(MUTED));
                                    if ui.small_button("Rebind").clicked() {
                                        *rebinding_pad = Some(*btn);
                                        *rebinding = None;
                                    }
                                    ui.end_row();
                                }
                            }
                            InputDevice::Motion | InputDevice::Vibration => {}
                        });
                });
        });
    });

    if matches!(*input_device, InputDevice::Motion | InputDevice::Vibration) {
        return;
    }
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if pill_button(ui, "Reset to Defaults", false).clicked() {
            *cfg = ControllerConfig::default();
            *save_needed = true;
        }
        if pill_button(ui, "Save", true).clicked() {
            *save_needed = true;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn motion_settings_content(
    ui: &mut egui::Ui,
    app_cfg: &mut AppSettings,
    app_save_needed: &mut bool,
    config: &ControllerConfig,
    key_capture: &mut bool,
    key_notice: &mut Option<String>,
    values: &[String; 6],
    button_missing: bool,
    recenter_now: &mut bool,
    calibration_note: Option<&str>,
    calibrate_now: &mut bool,
    startup_motion_enabled: bool,
) {
    let mut key_handled = false;
    if *key_capture {
        match ui.input(|i| recenter_key_press(&i.events, config)) {
            Some(MotionKeyCapture::Cancel) => {
                *key_capture = false;
                key_handled = true;
            }
            Some(MotionKeyCapture::Set(key)) => {
                app_cfg.motion_recenter_key = key;
                *key_capture = false;
                *app_save_needed = true;
                key_handled = true;
            }
            Some(MotionKeyCapture::Refuse(notice)) => {
                *key_notice = Some(notice);
                key_handled = true;
            }
            None => {}
        }
        if key_handled {
            ui.ctx().request_repaint();
        }
    }
    egui::Grid::new("motion")
        .num_columns(3)
        .spacing([10.0, 6.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[0]).size(12.0).color(TEXT));
            if ui.checkbox(&mut app_cfg.motion_enabled, "Enable motion controls").changed() {
                *app_save_needed = true;
            }
            ui.end_row();
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[1]).size(12.0).color(TEXT));
            ui.label(egui::RichText::new(values[1].as_str()).size(12.0).color(MUTED));
            ui.end_row();
            if !app_cfg.motion_enabled {
                ui.multiply_opacity(0.45);
            }
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[2]).size(12.0).color(TEXT));
            ui.label(
                egui::RichText::new(values[2].as_str())
                    .size(12.0)
                    .monospace()
                    .color(if *key_capture { AMBER } else { MUTED }),
            );
            if ui
                .small_button(if *key_capture { "Cancel" } else { "Rebind" })
                .clicked()
                && !key_handled
            {
                *key_capture = !*key_capture;
                *key_notice = None;
            }
            ui.end_row();
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[3]).size(12.0).color(TEXT));
            egui::ComboBox::from_id_salt("motion_recenter_button")
                .selected_text(values[3].as_str())
                .show_ui(ui, |ui| {
                    for button in MotionRecenterButton::all() {
                        if ui
                            .selectable_value(
                                &mut app_cfg.motion_recenter_button,
                                *button,
                                button.label(),
                            )
                            .changed()
                        {
                            *app_save_needed = true;
                        }
                    }
                });
            ui.end_row();
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[4]).size(12.0).color(TEXT));
            ui.label(egui::RichText::new(values[4].as_str()).size(12.0).color(GREEN));
            if ui.small_button(MOTION_RECENTER_ACTION).clicked() {
                *recenter_now = true;
            }
            ui.end_row();
            ui.label(egui::RichText::new(MOTION_ROW_LABELS[5]).size(12.0).color(TEXT));
            ui.label(egui::RichText::new(values[5].as_str()).size(12.0).color(
                match values[5].as_str() {
                    "" => MUTED,
                    "Calibrated" => GREEN,
                    _ => AMBER,
                },
            ));
            if ui.small_button(MOTION_CALIBRATE_ACTION).clicked() {
                *calibrate_now = true;
            }
            ui.end_row();
        });
    if let Some(note) = motion_note(
        app_cfg,
        config,
        key_notice.as_deref().filter(|_| *key_capture),
        button_missing,
    ) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(note).size(11.0).color(AMBER));
    }
    if app_cfg.motion_enabled != startup_motion_enabled {
        ui.label(egui::RichText::new(MOTION_RESTART).size(11.0).color(AMBER));
    }
    if let Some(note) = calibration_note {
        ui.label(egui::RichText::new(note).size(11.0).color(AMBER));
    }
    ui.add_space(6.0);
    for line in MOTION_HELP {
        ui.label(egui::RichText::new(line).size(11.0).color(MUTED));
    }
}

fn vibration_settings_content(
    ui: &mut egui::Ui,
    app_cfg: &mut AppSettings,
    app_save_needed: &mut bool,
    values: &[String; 4],
    test_now: &mut bool,
) {
    egui::Grid::new("vibration")
        .num_columns(3)
        .spacing([10.0, 6.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label(egui::RichText::new(VIBRATION_ROW_LABELS[0]).size(12.0).color(TEXT));
            if ui.checkbox(&mut app_cfg.vibration_enabled, "Enable vibration").changed() {
                *app_save_needed = true;
                *test_now |= app_cfg.vibration_enabled;
            }
            ui.end_row();
            ui.label(egui::RichText::new(VIBRATION_ROW_LABELS[1]).size(12.0).color(TEXT));
            ui.label(egui::RichText::new(values[1].as_str()).size(12.0).color(MUTED));
            ui.end_row();
            if !app_cfg.vibration_enabled {
                ui.multiply_opacity(0.45);
            }
            ui.label(egui::RichText::new(VIBRATION_ROW_LABELS[2]).size(12.0).color(TEXT));
            let strength = ui.add_enabled(
                app_cfg.vibration_enabled,
                egui::Slider::new(&mut app_cfg.vibration_strength, 10..=100)
                    .step_by(10.0)
                    .suffix("%"),
            );
            if strength.changed() {
                *test_now = true;
            }
            if strength.drag_stopped() || (strength.changed() && !strength.dragged()) {
                *app_save_needed = true;
            }
            ui.end_row();
            ui.label(egui::RichText::new(VIBRATION_ROW_LABELS[3]).size(12.0).color(TEXT));
            ui.label(egui::RichText::new(values[3].as_str()).size(12.0).color(GREEN));
            if ui
                .add_enabled(
                    app_cfg.vibration_enabled,
                    egui::Button::new(VIBRATION_TEST_ACTION).small(),
                )
                .clicked()
            {
                *test_now = true;
            }
            ui.end_row();
        });
    ui.add_space(6.0);
    for line in VIBRATION_HELP {
        ui.label(egui::RichText::new(line).size(11.0).color(MUTED));
    }
}

fn draw_controller_diagram(ui: &mut egui::Ui, input: &InputSnapshot) {
    let avail = ui.available_width();
    let s = (avail / 430.0).min(0.82);
    let h = 300.0 * s;
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(avail, h), Sense::hover());
    let painter = ui.painter_at(rect);
    let c = rect.center();
    let map = |x: f32, y: f32| c + Vec2::new(x * s, (y + 6.0) * s);

    painter.rect(
        rect,
        CornerRadius::same(10),
        BG,
        Stroke::NONE,
        egui::StrokeKind::Outside,
    );

    let outline = Color32::from_rgb(0x4C, 0x4C, 0x58);
    let body_stroke = Stroke::new((2.0 * s).max(1.2), outline);

    let n = PRO_BODY.len() / 2;
    let mut body: Vec<egui::Pos2> = Vec::with_capacity(n * 2);
    for i in 0..n {
        body.push(map(PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    for i in (0..n).rev() {
        body.push(map(-PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    let hn = PRO_LEFT_HANDLE.len() / 2;
    let lh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    let rh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(-PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    painter.add(egui::Shape::closed_line(lh, body_stroke));
    painter.add(egui::Shape::closed_line(rh, body_stroke));
    painter.add(egui::Shape::closed_line(body, body_stroke));

    let face = |center: egui::Pos2, r: f32, label: &str, on: bool| {
        let fill = if on { ACCENT } else { BG_INPUT };
        painter.circle(center, r, fill, Stroke::new(1.0_f32, BORDER));
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { MUTED };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional((r * 0.95).max(7.0)),
                tc,
            );
        }
    };
    let pad = |center: egui::Pos2, sz: Vec2, label: &str, on: bool| {
        let r = egui::Rect::from_center_size(center, sz);
        let fill = if on { ACCENT } else { BG_INPUT };
        painter.rect(
            r,
            CornerRadius::same(3),
            fill,
            Stroke::new(1.0_f32, BORDER),
            egui::StrokeKind::Outside,
        );
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { MUTED };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional(9.0 * s.max(0.8)),
                tc,
            );
        }
    };
    let stick = |base: egui::Pos2, sx: f32, sy: f32, clicked: bool| {
        let ring = 22.0 * s;
        let knob = 13.0 * s;
        painter.circle(base, ring, BG_INPUT, Stroke::new(1.5_f32, outline));
        let off = Vec2::new(sx, -sy) * (ring - knob - 1.0);
        let kc = if clicked {
            ACCENT
        } else {
            Color32::from_rgb(0x62, 0x62, 0x72)
        };
        painter.circle(base + off, knob, kc, Stroke::new(1.0_f32, BORDER));
    };

    pad(
        map(-120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZL",
        input.is(SwitchButton::ZL),
    );
    pad(
        map(120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZR",
        input.is(SwitchButton::ZR),
    );
    pad(
        map(-120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "L",
        input.is(SwitchButton::L),
    );
    pad(
        map(120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "R",
        input.is(SwitchButton::R),
    );

    stick(
        map(-111.0, -55.0),
        input.lx(),
        input.ly(),
        input.is(SwitchButton::StickL),
    );

    let dd = 19.0 * s;
    let dsz = Vec2::splat(16.0 * s);
    let dp = map(-61.0, 0.0);
    pad(
        dp + Vec2::new(0.0, -dd),
        dsz,
        "",
        input.is(SwitchButton::DUp),
    );
    pad(
        dp + Vec2::new(0.0, dd),
        dsz,
        "",
        input.is(SwitchButton::DDown),
    );
    pad(
        dp + Vec2::new(-dd, 0.0),
        dsz,
        "",
        input.is(SwitchButton::DLeft),
    );
    pad(
        dp + Vec2::new(dd, 0.0),
        dsz,
        "",
        input.is(SwitchButton::DRight),
    );

    let fr = 15.0 * s;
    face(map(136.0, -56.0), fr, "A", input.is(SwitchButton::A));
    face(map(105.0, -25.0), fr, "B", input.is(SwitchButton::B));
    face(map(105.0, -87.0), fr, "X", input.is(SwitchButton::X));
    face(map(74.0, -56.0), fr, "Y", input.is(SwitchButton::Y));

    stick(
        map(51.0, 0.0),
        input.rx(),
        input.ry(),
        input.is(SwitchButton::StickR),
    );

    face(
        map(-50.0, -86.0),
        9.0 * s,
        "-",
        input.is(SwitchButton::Minus),
    );
    face(map(50.0, -86.0), 9.0 * s, "+", input.is(SwitchButton::Plus));
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).size(12.0).color(MUTED));
    ui.label(egui::RichText::new(value).size(12.0).color(TEXT));
    ui.end_row();
}

fn gpu_selection_status(
    selected: Option<&str>,
    startup: Option<&str>,
    active: &str,
) -> (String, bool) {
    let restart_pending = selected != startup;
    let message = if restart_pending {
        "GPU change saved. Fully close and reopen NeXium to apply."
    } else {
        "Auto prefers a dGPU. GPU changes require restarting NeXium."
    };
    (format!("Active GPU: {active}\n{message}"), restart_pending)
}

fn graphics_settings_content(
    ui: &mut egui::Ui,
    cfg: &mut AppSettings,
    save_needed: &mut bool,
    startup_gpu: Option<&str>,
    active_gpu: &str,
) {
    ui.label(egui::RichText::new("GPU").size(13.0).strong().color(TEXT));
    egui::ComboBox::from_id_salt("graphics_gpu_device")
        .selected_text(cfg.gpu_device_label())
        .show_ui(ui, |ui| {
            *save_needed |= ui
                .selectable_value(&mut cfg.gpu_device, None, "Auto")
                .changed();
            for device in nexium_gpu::adapter::available_devices() {
                *save_needed |= ui
                    .selectable_value(&mut cfg.gpu_device, Some(device.id.clone()), device.label())
                    .changed();
            }
        });
    let (gpu_status, restart_pending) =
        gpu_selection_status(cfg.gpu_device.as_deref(), startup_gpu, active_gpu);
    ui.label(
        egui::RichText::new(gpu_status)
            .size(11.0)
            .color(if restart_pending { AMBER } else { MUTED }),
    );
    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Aspect Mode")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Letterbox keeps the game's aspect ratio. Integer uses pixel-perfect 1x/2x/3x and fits smaller windows. Stretch fills the window.").size(11.0).color(MUTED));
    ui.add_space(6.0);
    for m in AspectMode::all() {
        if ui.radio(cfg.aspect == *m, m.label()).clicked() {
            cfg.aspect = *m;
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Output Scale")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("Used only by Integer mode (1x / 2x / 3x).")
            .size(11.0)
            .color(MUTED),
    );
    ui.add_space(6.0);
    let mut s = cfg.output_scale.max(1).min(3);
    ui.horizontal(|ui| {
        for v in 1u8..=3u8 {
            if ui.radio(s == v, format!("{}x", v)).clicked() {
                s = v;
            }
        }
    });
    if s != cfg.output_scale {
        cfg.output_scale = s;
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Texture Filter")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Choose how the game image is scaled to fit the window.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);
    for f in FilterMode::all() {
        if ui.radio(cfg.filter == *f, f.label()).clicked() {
            cfg.filter = *f;
            *save_needed = true;
        }
    }
    if cfg.filter == FilterMode::Fsr {
        ui.add_space(6.0);
        if ui.add(
            egui::Slider::new(&mut cfg.fsr_sharpness, 0..=100)
                .suffix("%")
                .text("FSR sharpness"),
        ).on_hover_text("0% disables sharpening; 100% is strongest").changed() {
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Windowing")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    let mut dpi = cfg.dpi_aware;
    if ui
        .checkbox(&mut dpi, "High-DPI aware (Windows — requires restart)")
        .changed()
    {
        cfg.dpi_aware = dpi;
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Off lets Windows DPI virtualization upscale (blurry). On reports physical pixels — required for crisp output on 4K/2.5K displays.").size(10.5).color(MUTED));

    ui.add_space(8.0);
    let mut vsync = cfg.vsync;
    if ui
        .checkbox(&mut vsync, "V-Sync (requires restart)")
        .changed()
    {
        cfg.vsync = vsync;
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Shaders")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    let mut async_shaders = cfg.async_shaders;
    if ui
        .checkbox(&mut async_shaders, "Asynchronous shader compilation")
        .changed()
    {
        cfg.async_shaders = async_shaders;
        nexium_common::async_compile::set_enabled(async_shaders);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Builds new pipelines on a background thread so the game never stalls to compile. New effects pop in a frame or two the first time they appear. Off = compile on demand (brief hitch on first sight, no pop-in). Disk cache makes later launches stutter-free either way.").size(10.5).color(MUTED));
    ui.add_space(6.0);
    let mut depth_share = cfg.depth_share;
    if ui
        .checkbox(&mut depth_share, "Share game depth with ReShade add-ons")
        .changed()
    {
        cfg.depth_share = depth_share;
        nexium_common::depth_share::set_enabled(depth_share);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Copies the guest's main depth buffer with every frame and replays it into a depth attachment on the host device, so ReShade's Generic Depth and add-ons such as DLSS 5 Feed see real depth. Costs one extra readback per frame; leave off unless an overlay needs it.").size(10.5).color(MUTED));

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("GPU Timing")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    let mut fast_gpu_time = cfg.fast_gpu_time;
    if ui
        .checkbox(&mut fast_gpu_time, "Fast GPU time (applies on next boot)")
        .changed()
    {
        cfg.fast_gpu_time = fast_gpu_time;
        nexium_common::fast_gpu_time::set_enabled(fast_gpu_time);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Reports GPU timestamps 256x faster than real time, so games that scale their resolution from measured GPU time (Mario Kart 8 Deluxe, for one) stay at full resolution. Off = hardware-rate timestamps.").size(10.5).color(MUTED));
    ui.add_space(6.0);
    let mut force_max_clocks = cfg.force_max_clocks;
    if ui
        .checkbox(&mut force_max_clocks, "Force maximum clocks")
        .changed()
    {
        cfg.force_max_clocks = force_max_clocks;
        nexium_common::force_max_clocks::set_enabled(force_max_clocks);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Keeps a small compute job running on the host GPU while a game renders, so the driver doesn't drop to low clocks between the emulator's short bursts of work. Uses more power and runs the GPU warmer.").size(10.5).color(MUTED));
    ui.add_space(6.0);
    let mut normal_gpu_accuracy = cfg.normal_gpu_accuracy;
    if ui
        .checkbox(&mut normal_gpu_accuracy, "Normal GPU accuracy (applies on next boot)")
        .changed()
    {
        cfg.normal_gpu_accuracy = normal_gpu_accuracy;
        nexium_common::gpu_accuracy::set_normal(normal_gpu_accuracy);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Tells the game its GPU work is finished as soon as it is submitted, instead of waiting for the host GPU, like yuzu's Normal accuracy. Faster in games that wait on the GPU every frame; games that read back GPU results may glitch. Off = High accuracy.").size(10.5).color(MUTED));
}

fn audio_settings_content(
    ui: &mut egui::Ui,
    cfg: &mut AppSettings,
    save_needed: &mut bool,
    device_cache: &mut Option<Vec<String>>,
) {
    ui.label(
        egui::RichText::new("Output Device")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Pick the output for game audio, menu music, and sound effects. Restart NeXium \
         to apply device changes.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);

    if device_cache.is_none() {
        *device_cache = Some(list_output_devices());
    }
    let devices: Vec<String> = device_cache.clone().unwrap_or_default();

    let current_label: String = match &cfg.audio_output_device {
        None => "System default".to_string(),
        Some(n) => n.clone(),
    };
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("audio_output_device")
            .selected_text(current_label.clone())
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(cfg.audio_output_device.is_none(), "System default")
                    .clicked()
                {
                    if cfg.audio_output_device.is_some() {
                        cfg.audio_output_device = None;
                        *save_needed = true;
                    }
                }
                for name in &devices {
                    let selected = cfg.audio_output_device.as_deref() == Some(name.as_str());
                    if ui.selectable_label(selected, name).clicked() && !selected {
                        cfg.audio_output_device = Some(name.clone());
                        *save_needed = true;
                    }
                }
            });
        if ui.button("Refresh").clicked() {
            *device_cache = Some(list_output_devices());
        }
    });

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Active Stream")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    match current_stream_info() {
        Some(AudioStreamInfo {
            device_name,
            sample_rate,
            channels,
            sample_format,
        }) => {
            ui.label(
                egui::RichText::new(format!(
                    "{}  ·  {} Hz  ·  {} ch  ·  {}",
                    device_name, sample_rate, channels, sample_format,
                ))
                .size(11.0)
                .color(MUTED),
            );
        }
        None => {
            ui.label(
                egui::RichText::new("No audio stream open (init failed or audio disabled).")
                    .size(11.0)
                    .color(AMBER),
            );
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Master Volume")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Applied live in the audio callback. 1.0 = unity, 2.0 = +6 dB (may clip).",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);
    let mut vol = cfg.audio_volume;
    if ui
        .add(egui::Slider::new(&mut vol, 0.0..=2.0).text("vol"))
        .changed()
    {
        cfg.audio_volume = vol;
        set_master_volume(vol);
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Diagnostics")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    if ui.button("Play 1s 440 Hz test tone").clicked() {
        let pushed = push_test_tone(440.0, 1.0);
        log::info!("Audio test tone: pushed {} frames to sink", pushed);
    }
}

fn emulation_settings_content(ui: &mut egui::Ui, cfg: &mut AppSettings, save_needed: &mut bool) {
    ui.label(
        egui::RichText::new("CPU Backend")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(egui::RichText::new(
        "Selects which AArch64 JIT runs guest code. Applies on next boot — does not affect a running game.",
    ).size(11.0).color(MUTED));
    ui.add_space(8.0);

    for backend in CpuBackend::all() {
        let compiled = backend.is_compiled_in();
        let label = if compiled {
            backend.label().to_string()
        } else {
            format!("{} — not in build", backend.label())
        };
        ui.add_enabled_ui(compiled, |ui| {
            if ui.radio(cfg.cpu_backend == *backend, label).clicked() {
                cfg.cpu_backend = *backend;
                *save_needed = true;
            }
        });
    }

    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(
            "Dynarmic: MerryMage's mature C++ JIT — the baseline used by yuzu/Ryujinx. \
        Rustarmic: our own Rust AArch64→x86_64 JIT — newer, useful when debugging guest \
        behaviour that dynarmic's opaque codegen makes hard to inspect.",
        )
        .size(10.5)
        .color(MUTED),
    );

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Multi-Core")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Runs each guest core on its own host thread (3 user cores). Applies on next boot.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(8.0);
    let resp = ui.checkbox(&mut cfg.multicore, "Enable multi-core CPU emulation");
    if resp.changed() {
        *save_needed = true;
    }
    resp.on_hover_text("Keep this on — disables only for debugging");
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Emulated devices")
            .size(12.0)
            .color(MUTED),
    );
    let resp = ui.checkbox(&mut cfg.emulate_mouse, "Mouse (direct mouse input)");
    if resp.changed() {
        *save_needed = true;
    }
    resp.on_hover_text(
        "Report the host mouse to the game as a USB mouse; the cursor maps onto the game viewport",
    );
    let resp = ui.checkbox(&mut cfg.emulate_keyboard, "Keyboard");
    if resp.changed() {
        *save_needed = true;
    }
    resp.on_hover_text(
        "Report host keys to the game as a USB keyboard alongside the keyboard-to-controller bindings",
    );
    let resp = ui.checkbox(&mut cfg.emulate_touch, "Touchscreen (tap with mouse)");
    if resp.changed() {
        *save_needed = true;
    }
    resp.on_hover_text(
        "Left-clicking the game viewport sends a touchscreen tap at the cursor position — most point-and-click games use this, not the USB mouse",
    );
}

#[derive(Clone, Copy)]
enum FirmwareSettingsAction {
    Install,
    Remove,
}

fn firmware_settings_content(
    ui: &mut egui::Ui,
    status: &Result<Option<nexium_loader::firmware::InstalledFirmware>, String>,
) -> Option<FirmwareSettingsAction> {
    let mut action = None;
    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Firmware")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Install system firmware from your own decrypted dumps: a firmware .dnsp from NXDecrypt, or a .dxci whose update partition carries firmware. Games use its system archives, such as fonts, when they don't bundle their own.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(8.0);
    let installed = match status {
        Ok(Some(installed)) => {
            ui.label(egui::RichText::new(format!("Installed: firmware {}", installed.display_version)).color(TEXT));
            ui.label(
                egui::RichText::new(format!("{} files, installed from {}", installed.files.len(), installed.source))
                    .size(10.5)
                    .color(MUTED),
            );
            true
        }
        Ok(None) => {
            ui.label(egui::RichText::new("No firmware installed").color(MUTED));
            false
        }
        Err(error) => {
            ui.label(egui::RichText::new(format!("Installed firmware can't be read: {error}")).color(Color32::from_rgb(240, 119, 119)));
            false
        }
    };
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("Install firmware…").clicked() {
            action = Some(FirmwareSettingsAction::Install);
        }
        if ui.add_enabled(installed, egui::Button::new("Remove firmware")).clicked() {
            action = Some(FirmwareSettingsAction::Remove);
        }
    });
    action
}

fn logging_settings_content(ui: &mut egui::Ui, cfg: &mut AppSettings, save_needed: &mut bool) {
    ui.label(
        egui::RichText::new("Log Level")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("Takes effect on restart.")
            .size(11.0)
            .color(MUTED),
    );
    ui.add_space(8.0);

    for level in LogLevel::all() {
        if ui.radio(cfg.log_level == *level, level.label()).clicked() {
            cfg.log_level = *level;
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    let path = AppSettings::config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "unknown".into());
    ui.label(
        egui::RichText::new(format!("Config: {}", path))
            .size(10.5)
            .color(MUTED),
    );
}

fn hid_keyboard_usage(key: egui::Key) -> Option<u8> {
    use egui::Key;
    Some(match key {
        Key::A => 4,
        Key::B => 5,
        Key::C => 6,
        Key::D => 7,
        Key::E => 8,
        Key::F => 9,
        Key::G => 10,
        Key::H => 11,
        Key::I => 12,
        Key::J => 13,
        Key::K => 14,
        Key::L => 15,
        Key::M => 16,
        Key::N => 17,
        Key::O => 18,
        Key::P => 19,
        Key::Q => 20,
        Key::R => 21,
        Key::S => 22,
        Key::T => 23,
        Key::U => 24,
        Key::V => 25,
        Key::W => 26,
        Key::X => 27,
        Key::Y => 28,
        Key::Z => 29,
        Key::Num1 => 30,
        Key::Num2 => 31,
        Key::Num3 => 32,
        Key::Num4 => 33,
        Key::Num5 => 34,
        Key::Num6 => 35,
        Key::Num7 => 36,
        Key::Num8 => 37,
        Key::Num9 => 38,
        Key::Num0 => 39,
        Key::Enter => 40,
        Key::Escape => 41,
        Key::Backspace => 42,
        Key::Tab => 43,
        Key::Space => 44,
        Key::Minus => 45,
        Key::Equals => 46,
        Key::OpenBracket => 47,
        Key::CloseBracket => 48,
        Key::Backslash => 49,
        Key::Semicolon => 51,
        Key::Quote => 52,
        Key::Backtick => 53,
        Key::Comma => 54,
        Key::Period => 55,
        Key::Slash => 56,
        Key::F1 => 58,
        Key::F2 => 59,
        Key::F3 => 60,
        Key::F4 => 61,
        Key::F5 => 62,
        Key::F6 => 63,
        Key::F7 => 64,
        Key::F8 => 65,
        Key::F9 => 66,
        Key::F10 => 67,
        Key::F11 => 68,
        Key::F12 => 69,
        Key::Insert => 73,
        Key::Home => 74,
        Key::PageUp => 75,
        Key::Delete => 76,
        Key::End => 77,
        Key::PageDown => 78,
        Key::ArrowRight => 79,
        Key::ArrowLeft => 80,
        Key::ArrowDown => 81,
        Key::ArrowUp => 82,
        _ => return None,
    })
}

fn wait_tree_color(class: nexium_kernel::kernel::ThreadWaitClass) -> Color32 {
    use nexium_kernel::kernel::ThreadWaitClass as C;
    match class {
        C::Running => GREEN,
        C::Ready => Color32::from_rgb(0x2f, 0x9e, 0x4f),
        C::Sleeping => AMBER,
        C::Waiting => Color32::from_rgb(0xe5, 0x53, 0x53),
        C::Created => Color32::from_rgb(0x46, 0xc8, 0xd8),
        C::Exited => MUTED,
    }
}

fn debug_windows(
    ctx: &egui::Context,
    dbg: &mut DebuggerState,
    snapshot: Option<&crate::boot::CpuSnapshot>,
    mem_request: Option<&dyn Fn(u64)>,
    log_buffer: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    app_settings: &mut AppSettings,
) {
    if dbg.show_memory {
        egui::Window::new("Memory")
            .open(&mut dbg.show_memory)
            .default_size([520.0, 340.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Address").size(12.0).color(MUTED));
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut dbg.memory_address_input)
                            .desired_width(160.0)
                            .font(egui::TextStyle::Monospace),
                    );
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let s = dbg.memory_address_input.trim().trim_start_matches("0x");
                        if let Ok(addr) = u64::from_str_radix(s, 16) {
                            dbg.memory_address = addr;
                            if let Some(req) = mem_request {
                                req(addr);
                            }
                        }
                    }
                    if ui.small_button("Go").clicked() {
                        let s = dbg.memory_address_input.trim().trim_start_matches("0x");
                        if let Ok(addr) = u64::from_str_radix(s, 16) {
                            dbg.memory_address = addr;
                            if let Some(req) = mem_request {
                                req(addr);
                            }
                        }
                    }
                });
                ui.add_space(4.0);

                if let Some(snap) = snapshot {
                    if snap.mem_data.is_empty() || snap.mem_address != dbg.memory_address {
                        if let Some(req) = mem_request {
                            req(dbg.memory_address);
                        }
                        ui.label(
                            egui::RichText::new("Reading memory…")
                                .size(11.0)
                                .color(MUTED),
                        );
                    } else {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut row = 0;
                                while row * 16 < snap.mem_data.len() {
                                    let off = row * 16;
                                    let addr = snap.mem_address + off as u64;
                                    let slice =
                                        &snap.mem_data[off..(off + 16).min(snap.mem_data.len())];
                                    let hex: Vec<String> =
                                        slice.iter().map(|b| format!("{:02x}", b)).collect();
                                    let ascii: String = slice
                                        .iter()
                                        .map(|&b| {
                                            if (0x20..0x7F).contains(&b) {
                                                b as char
                                            } else {
                                                '.'
                                            }
                                        })
                                        .collect();
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:016x}", addr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(hex.join(" "))
                                                .size(11.5)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.label(
                                            egui::RichText::new(ascii)
                                                .size(11.5)
                                                .monospace()
                                                .color(AMBER),
                                        );
                                    });
                                    row += 1;
                                }
                            });
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_registers {
        egui::Window::new("Registers")
            .open(&mut dbg.show_registers)
            .default_size([320.0, 480.0])
            .show(ctx, |ui| {
                if let Some(snap) = snapshot {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            egui::Grid::new("regs")
                                .num_columns(2)
                                .spacing([12.0, 2.0])
                                .show(ui, |ui| {
                                    for (n, v) in [
                                        ("PC", snap.pc),
                                        ("SP", snap.sp),
                                        ("TPIDRRO", snap.tpidrro_el0),
                                    ] {
                                        ui.label(
                                            egui::RichText::new(format!("{:<8}", n))
                                                .size(12.0)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:#018x}", v))
                                                .size(12.0)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.end_row();
                                    }
                                    for i in 0..31 {
                                        ui.label(
                                            egui::RichText::new(format!("X{:<7}", i))
                                                .size(12.0)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:#018x}", snap.x[i]))
                                                .size(12.0)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_disasm {
        egui::Window::new("Disassembler")
            .open(&mut dbg.show_disasm)
            .default_size([520.0, 340.0])
            .show(ctx, |ui| {
                if let Some(snap) = snapshot {
                    if snap.instruction_bytes.is_empty() {
                        ui.label(
                            egui::RichText::new("No instruction data yet")
                                .size(11.0)
                                .color(MUTED),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!("PC {:#018x}", snap.pc))
                                .size(12.0)
                                .monospace()
                                .color(AMBER),
                        );
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut off = 0;
                                while off + 4 <= snap.instruction_bytes.len() {
                                    let instr = u32::from_le_bytes([
                                        snap.instruction_bytes[off],
                                        snap.instruction_bytes[off + 1],
                                        snap.instruction_bytes[off + 2],
                                        snap.instruction_bytes[off + 3],
                                    ]);
                                    let addr = snap.pc + off as u64;
                                    let mnemonic = disasm_arm64(instr);
                                    let color = if off == 0 { ACCENT } else { TEXT };
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:016x}", addr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:08x}", instr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(mnemonic)
                                                .size(11.5)
                                                .monospace()
                                                .color(color),
                                        );
                                    });
                                    off += 4;
                                }
                            });
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_wait_tree {
        egui::Window::new("Wait Tree")
            .open(&mut dbg.show_wait_tree)
            .default_size([580.0, 440.0])
            .show(ctx, |ui| {
                if let Some(snap) = snapshot {
                    if snap.threads.is_empty() {
                        ui.label(
                            egui::RichText::new("No thread data yet")
                                .size(11.0)
                                .color(MUTED),
                        );
                    } else {
                        use nexium_kernel::kernel::ThreadWaitClass;
                        let count = |class: ThreadWaitClass| {
                            snap.threads
                                .iter()
                                .filter(|t| t.state_class == class)
                                .count()
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "{} threads — {} running, {} ready, {} waiting, {} sleeping",
                                snap.threads.len(),
                                count(ThreadWaitClass::Running),
                                count(ThreadWaitClass::Ready),
                                count(ThreadWaitClass::Waiting),
                                count(ThreadWaitClass::Sleeping),
                            ))
                            .size(11.0)
                            .color(MUTED),
                        );
                        ui.add_space(4.0);
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                for t in &snap.threads {
                                    let color = wait_tree_color(t.state_class);
                                    let header = format!(
                                        "{:#06x} tid={:<3} core={} prio={:<2} {}",
                                        t.handle, t.tid, t.core, t.effective_priority, t.status
                                    );
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new(header)
                                            .size(12.0)
                                            .monospace()
                                            .color(color),
                                    )
                                    .show(ui, |ui| {
                                        let row = |ui: &mut egui::Ui, text: String| {
                                            ui.label(
                                                egui::RichText::new(text)
                                                    .size(11.5)
                                                    .monospace()
                                                    .color(TEXT),
                                            );
                                        };
                                        if !t.detail.is_empty() {
                                            row(ui, t.detail.clone());
                                        }
                                        row(
                                            ui,
                                            format!(
                                                "processor = core {} (ideal {}, affinity {:#x})",
                                                t.core, t.ideal_core, t.affinity_mask
                                            ),
                                        );
                                        row(
                                            ui,
                                            format!(
                                                "priority = {} (current) / {} (base)",
                                                t.effective_priority, t.priority
                                            ),
                                        );
                                        row(ui, format!("PC = {:#x} LR = {:#x}", t.pc, t.lr));
                                        if t.waiters.is_empty() {
                                            ui.label(
                                                egui::RichText::new("waited by no thread")
                                                    .size(11.5)
                                                    .monospace()
                                                    .color(MUTED),
                                            );
                                        } else {
                                            row(
                                                ui,
                                                format!(
                                                    "waited by: {}",
                                                    t.waiters
                                                        .iter()
                                                        .map(|h| format!("{:#x}", h))
                                                        .collect::<Vec<_>>()
                                                        .join(", ")
                                                ),
                                            );
                                        }
                                    });
                                }
                            });
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    let mut open_performance_from_logs = false;
    if dbg.show_logs {
        egui::Window::new("Logs")
            .open(&mut dbg.show_logs)
            .default_size([580.0, 300.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if pill_button(ui, "Clear", false).clicked() {
                        if let Ok(mut buf) = log_buffer.lock() {
                            buf.clear();
                        }
                    }
                    if pill_button(ui, "Performance Tuning", false).clicked() {
                        open_performance_from_logs = true;
                    }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if let Ok(buf) = log_buffer.lock() {
                            for entry in buf.iter() {
                                ui.label(
                                    egui::RichText::new(entry)
                                        .size(11.0)
                                        .monospace()
                                        .color(MUTED),
                                );
                            }
                        }
                    });
            });
    }

    if open_performance_from_logs {
        dbg.show_performance = true;
    }

    if dbg.show_performance {
        let mut changed = false;
        egui::Window::new("Performance Tuning")
            .open(&mut dbg.show_performance)
            .default_size([620.0, 620.0])
            .min_size([480.0, 420.0])
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("Advanced performance controls")
                        .size(14.0)
                        .strong()
                        .color(TEXT),
                );
                ui.add_space(3.0);
                ui.label(
                    egui::RichText::new(
                        "Changes apply after restarting NeXium. Values supplied by your launcher take precedence.",
                    )
                    .size(11.0)
                    .color(AMBER),
                );
                ui.label(
                    egui::RichText::new(
                        "Unchecked options use automatic defaults. Hover any option for behavior, dependencies, and compatibility notes.",
                    )
                    .size(10.5)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if pill_button(ui, "Enable safe preset", true).clicked() {
                        app_settings.performance_debug.enable_safe_preset();
                        changed = true;
                    }
                    if pill_button(ui, "Reset overrides", false).clicked() {
                        app_settings.performance_debug.clear();
                        changed = true;
                    }
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "{} of 14 enabled",
                            app_settings
                                .performance_debug
                                .enabled_overrides()
                                .len()
                        ))
                        .size(11.0)
                        .color(MUTED),
                    );
                });
                ui.add_space(8.0);
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        performance_debug_section(ui, "CPU recompiler");
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.cpu_backend_dynarmic,
                            "Dynarmic CPU Recompiler",
                            "Uses NeXium's high-performance ARM64 dynamic recompiler.",
                            "Forces the Dynarmic backend even when another backend is selected in the normal CPU settings. Dynarmic is already the default.",
                            "NEXIUM_CPU_BACKEND=dynarmic",
                        );
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.cpu_cores_4,
                            "Four-Core CPU Emulation",
                            "Runs the full four-core guest CPU configuration.",
                            "Overrides the guest CPU core count to four. This is already the normal default and may still be superseded by single-core compatibility mode.",
                            "NEXIUM_CPU_CORES=4",
                        );
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings
                                .performance_debug
                                .dynarmic_code_page_cache,
                            "Dynarmic Instruction Page Cache",
                            "Caches guest code-page lookups inside Dynarmic.",
                            "Reduces repeated page-table lookups while translating and executing guest code. This cache is already enabled by default.",
                            "DYNARMIC_CODE_PAGE_CACHE=1",
                        );
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.dynarmic_jit_size_64,
                            "64 MiB Dynarmic JIT Cache",
                            "Uses a 64 MiB generated-code cache for each guest core.",
                            "Pins Dynarmic's per-core JIT cache to the stable 64 MiB baseline. Larger caches can increase memory use and have caused long transition stalls.",
                            "DYNARMIC_JIT_SIZE=64",
                        );

                        performance_debug_section(ui, "GPU scheduling");
                        ui.label(
                            egui::RichText::new(
                                "Temporarily quarantined: the parallel submission path can stall guest fences and trigger a host GPU reset.",
                            )
                            .size(10.5)
                            .color(DANGER_HV),
                        );
                        ui.add_enabled_ui(false, |ui| {
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.async_gpu,
                                "Asynchronous GPU Submission",
                                "Moves guest GPU submissions to a dedicated worker (default on; NEXIUM_ASYNC_GPU=0 restores synchronous submission).",
                                "Enabled by default since 2026-09-06 with hard per-kick boundaries.",
                                "NEXIUM_ASYNC_GPU=1",
                            );
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings
                                    .performance_debug
                                    .async_gpu_defer_smallrt,
                                "Deferred Small Render-Target Writeback",
                                "Batches small render-target readbacks until synchronization is required.",
                                "Developer-only. Requires the quarantined soft-boundary asynchronous GPU path.",
                                "NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1\nNEXIUM_ASYNC_GPU=soft\nNEXIUM_ASYNC_GPU_DEFER_SMALLRT=1",
                            );
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.gpu_pipeline,
                                "Parallel GPU Command Preparation",
                                "Separates command decoding and preparation from submission.",
                                "Developer-only while host completion and device-loss recovery are being hardened.",
                                "NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1\nNEXIUM_ASYNC_GPU=1\nNEXIUM_GPU_PIPELINE=1",
                            );
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.fermi_lazy_drain,
                                "Reduced 2D Copy Synchronization",
                                "Avoids redundant render-thread drains for safe 2D copies.",
                                "Developer-only. Requires the quarantined parallel GPU path.",
                                "NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1\nNEXIUM_FERMI_LAZY_DRAIN=1",
                            );
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.fermi_async_blit,
                                "Asynchronous 2D Render-Target Copies",
                                "Offloads compatible 2D copies to the render thread.",
                                "Hard-disabled because its deferred layout publication is not Vulkan-safe. The synchronous exact-copy path remains active.",
                                "Hard-disabled in this build",
                            );
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.async_smallrt_wb,
                                "Asynchronous Small-Target Writeback",
                                "Completes deferred small-target readbacks on the render thread.",
                                "Developer-only while Vulkan resource lifetime validation is incomplete.",
                                "NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1\nNEXIUM_ASYNC_GPU=1\nNEXIUM_GPU_PIPELINE=1\nNEXIUM_ASYNC_SMALLRT_WB=1",
                            );
                        });

                        performance_debug_section(ui, "GPU data caching");
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.input_mirror,
                            "GPU Input Mirroring",
                            "Reuses unchanged GPU input data tracked through fast memory.",
                            "Caches guest GPU inputs through fastmem write-watch tracking. It remains inactive when the required fast-memory support is unavailable.",
                            "NEXIUM_INPUT_MIRROR=1",
                        );
                        ui.add_enabled_ui(false, |ui| {
                            changed |= performance_debug_toggle(
                                ui,
                                &mut app_settings.performance_debug.cbuf_writethrough,
                                "Constant-Buffer Write-Through",
                                "Updates mirrored constants in place instead of invalidating them.",
                                "Developer-only. Requires the quarantined parallel GPU path.",
                                "NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1\nNEXIUM_INPUT_MIRROR=1\nNEXIUM_CBUF_WRITETHROUGH=1",
                            );
                        });
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.resident_vb,
                            "Resident Vertex Buffer Cache",
                            "Keeps frequently reused vertex data resident on the GPU.",
                            "Maintains a resident Vulkan buffer cache for mirrored vertex chunks to reduce repeated uploads. Falls back safely when mirroring is unavailable.",
                            "NEXIUM_RESIDENT_VB=1",
                        );
                        changed |= performance_debug_toggle(
                            ui,
                            &mut app_settings.performance_debug.resident_cbuf,
                            "Resident Constant Buffer Cache",
                            "Reuses stable constant-buffer snapshots across draws.",
                            "Retains immutable mirrored constant-buffer page snapshots and reuses them across queued draws.",
                            "NEXIUM_RESIDENT_CBUF=1",
                        );
                    });
            });
        if changed {
            if let Err(error) = app_settings.save() {
                log::warn!("failed to save performance debug settings: {error}");
            }
        }
    }
}

fn performance_debug_section(ui: &mut egui::Ui, label: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(label).size(13.0).strong().color(TEXT));
    ui.add_space(3.0);
}

fn performance_debug_toggle(
    ui: &mut egui::Ui,
    enabled: &mut bool,
    title: &str,
    summary: &str,
    details: &str,
    technical_override: &str,
) -> bool {
    let checkbox = ui.checkbox(enabled, egui::RichText::new(title).size(12.0).color(TEXT));
    let changed = checkbox.changed();
    let summary_response = ui
        .indent(title, |ui| {
            ui.label(egui::RichText::new(summary).size(10.5).color(MUTED))
        })
        .inner;
    checkbox
        .on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_ui(|ui| performance_debug_tooltip(ui, title, details, technical_override));
    summary_response
        .on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_ui(|ui| performance_debug_tooltip(ui, title, details, technical_override));
    ui.add_space(4.0);
    changed
}

fn performance_debug_tooltip(
    ui: &mut egui::Ui,
    title: &str,
    details: &str,
    technical_override: &str,
) {
    ui.set_max_width(380.0);
    ui.label(egui::RichText::new(title).size(12.0).strong().color(TEXT));
    ui.add_space(3.0);
    ui.label(egui::RichText::new(details).size(11.0).color(TEXT));
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new("Requires a full application restart")
            .size(10.5)
            .color(AMBER),
    );
    ui.add_space(3.0);
    ui.label(
        egui::RichText::new("Technical control")
            .size(9.5)
            .color(MUTED),
    );
    ui.label(
        egui::RichText::new(technical_override)
            .size(10.0)
            .monospace()
            .color(MUTED),
    );
}

fn disasm_arm64(instr: u32) -> String {
    if instr == 0xD503201F {
        return "nop".into();
    }
    if instr == 0xD65F03C0 {
        return "ret".into();
    }
    if instr == 0xD4200000 {
        return "brk #0".into();
    }

    let op = instr >> 24;

    if (instr & 0xFFE0_0000) == 0xD400_0000 {
        let imm = (instr >> 5) & 0xFFFF;
        return format!("svc #{:#x}", imm);
    }
    if (instr & 0xFC00_0000) == 0x9400_0000 {
        let imm26 = instr & 0x03FF_FFFF;
        let off = if imm26 & 0x0200_0000 != 0 {
            ((imm26 | 0xFC00_0000) as i32) * 4
        } else {
            (imm26 as i32) * 4
        };
        return format!("bl pc{:+}", off);
    }
    if (instr & 0xFC00_0000) == 0x1400_0000 {
        let imm26 = instr & 0x03FF_FFFF;
        let off = if imm26 & 0x0200_0000 != 0 {
            ((imm26 | 0xFC00_0000) as i32) * 4
        } else {
            (imm26 as i32) * 4
        };
        return format!("b pc{:+}", off);
    }
    if (instr & 0xFFC0_0000) == 0x9100_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rd = instr & 0x1F;
        return format!("add x{}, x{}, #{:#x}", rd, rn, imm12);
    }
    if (instr & 0xFFC0_0000) == 0xD100_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rd = instr & 0x1F;
        return format!("sub x{}, x{}, #{:#x}", rd, rn, imm12);
    }
    if (instr & 0xFFC0_0000) == 0xF940_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rt = instr & 0x1F;
        return format!("ldr x{}, [x{}, #{:#x}]", rt, rn, imm12 * 8);
    }
    if (instr & 0xFFC0_0000) == 0xF900_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rt = instr & 0x1F;
        return format!("str x{}, [x{}, #{:#x}]", rt, rn, imm12 * 8);
    }
    if (instr & 0xFFE0_0000) == 0xD280_0000 {
        let imm16 = (instr >> 5) & 0xFFFF;
        let rd = instr & 0x1F;
        return format!("mov x{}, #{:#x}", rd, imm16);
    }

    format!("? .word {:#010x}  (op={:#x})", instr, op)
}
