use crate::input::InputSnapshot;
use crate::library::Library;
use crate::playtime::{format_playtime, PlayTimes};
use eframe::egui;
use eframe::egui::{Color32, FontId, Rounding, Sense, Stroke, Vec2};

#[derive(Clone, Copy)]
struct Pal {
    text: Color32,
    muted: Color32,
    panel: Color32,
    input_bg: Color32,
    border: Color32,
    sel: Color32,
    hover: Color32,
    soft_text: Color32,
}

fn lerp_col(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

fn palette(t: f32) -> Pal {
    Pal {
        text: lerp_col(Color32::from_rgb(0xEC, 0xEC, 0xF0), Color32::from_rgb(0x1E, 0x1E, 0x28), t),
        muted: lerp_col(Color32::from_rgb(0x8A, 0x8A, 0x98), Color32::from_rgb(0x60, 0x60, 0x6A), t),
        panel: lerp_col(Color32::from_rgb(0x16, 0x16, 0x1E), Color32::from_rgb(0xFF, 0xFF, 0xFF), t),
        input_bg: lerp_col(Color32::from_rgb(0x20, 0x20, 0x2A), Color32::from_rgb(0xE4, 0xE4, 0xEC), t),
        border: lerp_col(Color32::from_rgb(0x30, 0x30, 0x3C), Color32::from_rgb(0xC6, 0xC6, 0xD0), t),
        sel: lerp_col(Color32::from_rgb(0x1E, 0x1E, 0x28), Color32::from_rgb(0xDD, 0xDD, 0xE6), t),
        hover: lerp_col(Color32::from_rgb(0x18, 0x18, 0x22), Color32::from_rgb(0xEA, 0xEA, 0xF0), t),
        soft_text: lerp_col(Color32::from_rgb(0xEC, 0xEC, 0xF0), Color32::from_rgb(0x3E, 0x3E, 0x4A), t),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProfileTab {
    Profile,
    RecentlyPlayed,
    Settings,
    System,
}

pub struct ProfileState {
    pub tab: ProfileTab,
    pub name_buf: String,
    pub list_scroll: f32,
    pub nav_cooldown: f64,
    pub focus_content: bool,
    pub row_selected: usize,
    pub tab_anim: f32,
    pub shown_tab: ProfileTab,
    pub name_editing: bool,
    pub b_held: bool,
    pub x_held: bool,
    pub theme_t: f32,
    pub avatar_bounce: Option<f32>,
    // celebration particles: (spawn_time, angle, speed, kind) — kind 255 = star, else confetti color idx
    pub avatar_stars: Vec<(f32, f32, f32, u8)>,
}

impl ProfileState {
    pub fn new() -> Self {
        Self {
            tab: ProfileTab::Profile,
            name_buf: String::new(),
            list_scroll: 0.0,
            nav_cooldown: 0.0,
            focus_content: false,
            row_selected: 0,
            tab_anim: 1.0,
            shown_tab: ProfileTab::Profile,
            name_editing: false,
            b_held: false,
            x_held: false,
            theme_t: 0.0,
            avatar_bounce: None,
            avatar_stars: Vec::new(),
        }
    }
}

pub enum ProfileAction {
    None,
    Close,
    PickIcon,
    SetName(String),
    SetBackdropTheme(crate::app_settings::BackdropTheme),
    SetDockbarTheme(crate::app_settings::DockbarTheme),
    SetLightMode(bool),
    SetMusicVolume(f32),
    SetSfxVolume(f32),
    SetEuDates(bool),
    SetMuteMusic(bool),
    SetMuteSfx(bool),
    QuickLaunch(String),
}

fn brighten(c: Color32, amt: f32) -> Color32 {
    let f = |x: u8| (x as f32 + (255.0 - x as f32) * amt) as u8;
    Color32::from_rgb(f(c.r()), f(c.g()), f(c.b()))
}

fn draw_circle_avatar(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    tex: Option<egui::TextureId>,
    ring: Color32,
    scale_factor: f32,
    pal: Pal,
) {
    let alpha = 255;
    let bg_color = Color32::from_rgba_unmultiplied(pal.panel.r(), pal.panel.g(), pal.panel.b(), alpha);
    painter.circle_filled(center, radius + 4.0 * scale_factor, bg_color);
    match tex {
        Some(tid) => {
            let mut mesh = egui::epaint::Mesh::with_texture(tid);
            let segs = 56;
            let color = Color32::from_rgba_premultiplied(alpha, alpha, alpha, alpha);
            mesh.vertices.push(egui::epaint::Vertex {
                pos: center,
                uv: egui::pos2(0.5, 0.5),
                color,
            });
            for i in 0..=segs {
                let ang = (i as f32 / segs as f32) * std::f32::consts::TAU;
                let (s, c) = ang.sin_cos();
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: center + Vec2::new(c, s) * radius,
                    uv: egui::pos2(0.5 + c * 0.5, 0.5 + s * 0.5),
                    color,
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
            let default_bg = Color32::from_rgba_unmultiplied(pal.input_bg.r(), pal.input_bg.g(), pal.input_bg.b(), alpha);
            painter.circle_filled(center, radius, default_bg);
            let text_color = Color32::from_rgba_unmultiplied(pal.muted.r(), pal.muted.g(), pal.muted.b(), alpha);
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "👤",
                FontId::proportional(radius * 1.05),
                text_color,
            );
        }
    }
    let ring_color = ring;
    painter.circle_stroke(center, radius, Stroke::new(3.0 * scale_factor, ring_color));
}

#[allow(clippy::too_many_arguments)]
pub fn profile_view(
    state: &mut ProfileState,
    lib: &mut Library,
    play: &PlayTimes,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    full: egui::Rect,
    profile_name: &str,
    avatar_tex: Option<egui::TextureId>,
    ambient: Color32,
    accent: Color32,
    content_opacity: f32,
    backdrop_opacity: f32,
    scale_factor: f32,
    backdrop_theme: crate::app_settings::BackdropTheme,
    dockbar_theme: crate::app_settings::DockbarTheme,
    light_mode: bool,
    music_volume: f32,
    sfx_volume: f32,
    eu_dates: bool,
    music_muted: bool,
    sfx_muted: bool,
    active: bool,
    last_input: &InputSnapshot,
    ib: &mut Option<crate::input::InputBackend>,
) -> ProfileAction {
    let mut action = ProfileAction::None;
    let t = ui.input(|i| i.time) as f32;
    let now = ui.input(|i| i.time);
    let s = (full.height() / 820.0).clamp(1.0, 2.4);
    let accent = brighten(accent, 0.15);

    let dt = ui.input(|i| i.stable_dt).min(0.1);
    let theme_target = if light_mode { 1.0 } else { 0.0 };
    state.theme_t += (theme_target - state.theme_t) * (dt * 5.0).min(1.0);
    if (state.theme_t - theme_target).abs() < 0.002 {
        state.theme_t = theme_target;
    }
    let pal = palette(state.theme_t);

    let screen_center = full.center();
    let scale_pos = |p: egui::Pos2| -> egui::Pos2 {
        screen_center + (p - screen_center) * scale_factor
    };
    let scale_rect = |r: egui::Rect| -> egui::Rect {
        egui::Rect::from_center_size(
            screen_center + (r.center() - screen_center) * scale_factor,
            r.size() * scale_factor,
        )
    };

    let mut backdrop_painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("profile_backdrop"),
    ));
    backdrop_painter.set_opacity(backdrop_opacity);

    crate::carousel::draw_backdrop(&backdrop_painter, full, ambient, t, backdrop_theme, 1.0, state.theme_t);

    let scrim = if state.theme_t < 0.5 {
        Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, (140.0 * (1.0 - state.theme_t * 2.0)) as u8)
    } else {
        Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, (55.0 * (state.theme_t * 2.0 - 1.0)) as u8)
    };
    backdrop_painter.rect_filled(full, Rounding::ZERO, scrim);

    let mut painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("profile_bg"),
    ));
    painter.set_opacity(content_opacity);

    let mx = full.width() * 0.055;
    let header_font = 34.0 * s;
    painter.text(
        scale_pos(egui::pos2(full.min.x + mx, full.min.y + 40.0 * s)),
        egui::Align2::LEFT_TOP,
        "Profile",
        FontId::proportional(header_font * scale_factor),
        pal.text,
    );
    let header_y = full.min.y + 40.0 * s + header_font + 20.0 * s;
    
    let h_y = scale_pos(egui::pos2(0.0, header_y)).y;
    let x_start = scale_pos(egui::pos2(full.min.x + mx, 0.0)).x;
    let x_end = scale_pos(egui::pos2(full.max.x - mx, 0.0)).x;
    painter.hline(
        x_start..=x_end,
        h_y,
        Stroke::new(1.0 * scale_factor, pal.border),
    );
    
    let footer_y = full.max.y - 60.0 * s;
    let f_y = scale_pos(egui::pos2(0.0, footer_y)).y;
    painter.hline(
        x_start..=x_end,
        f_y,
        Stroke::new(1.0 * scale_factor, pal.border),
    );

    let editing = state.name_editing;

    let n_games = lib.games.len();
    let b_down = last_input.connected && last_input.is(crate::controller_config::SwitchButton::B);
    let b_edge = b_down && !state.b_held;
    state.b_held = b_down;
    if b_edge && !editing {
        if let Some(ref mut backend) = ib {
            let _ = backend.rumble(20000, 20000, 45);
        }
    }
    let mut back = (ui.input(|i| i.key_pressed(egui::Key::Escape)) || b_edge) && !editing;
    let mut up = false;
    let mut down = false;
    let mut enter = false;
    let mut leave = false;
    let mut tab_left = false;
    let mut tab_right = false;

    if last_input.connected && !editing {
        use crate::controller_config::SwitchButton;
        let ready = now - state.nav_cooldown > 0.16;
        let mut navved = false;
        let ly = last_input.ly();
        let mut gp_rumble = None;
        if ready {
            if last_input.is(SwitchButton::DUp) || ly > 0.5 {
                up = true;
                navved = true;
                gp_rumble = Some((15000, 15000, 35));
            }
            if last_input.is(SwitchButton::DDown) || ly < -0.5 {
                down = true;
                navved = true;
                gp_rumble = Some((15000, 15000, 35));
            }
            if last_input.is(SwitchButton::L) {
                tab_left = true;
                navved = true;
                gp_rumble = Some((15000, 15000, 35));
            }
            if last_input.is(SwitchButton::R) {
                tab_right = true;
                navved = true;
                gp_rumble = Some((15000, 15000, 35));
            }
            if last_input.is(SwitchButton::A) || last_input.is(SwitchButton::DRight) {
                enter = true;
                navved = true;
                gp_rumble = Some((28000, 28000, 60));
            }
            if last_input.is(SwitchButton::DLeft) {
                leave = true;
                navved = true;
                gp_rumble = Some((15000, 15000, 35));
            }
        }
        if navved {
            state.nav_cooldown = now;
            if let Some((low, high, ms)) = gp_rumble {
                if let Some(ref mut backend) = ib {
                    let _ = backend.rumble(low, high, ms);
                }
            }
        }
    }

    if !active {
        back = false;
        up = false;
        down = false;
        enter = false;
        leave = false;
        tab_left = false;
        tab_right = false;
    }

    // X (or M) toggles mute on the volume rows
    let x_down = last_input.connected && last_input.is(crate::controller_config::SwitchButton::X);
    let mut mute_toggle = (x_down && !state.x_held) || ui.input(|i| i.key_pressed(egui::Key::M));
    state.x_held = x_down;
    if !active || editing {
        mute_toggle = false;
    }

    const TAB_ORDER: [ProfileTab; 4] = [
        ProfileTab::Profile,
        ProfileTab::RecentlyPlayed,
        ProfileTab::Settings,
        ProfileTab::System,
    ];
    let tab_idx = TAB_ORDER.iter().position(|x| *x == state.tab).unwrap_or(0);
    const N_SETTINGS: usize = 6;
    let launch_enter = enter && state.focus_content;

    if tab_left {
        state.tab = TAB_ORDER[(tab_idx + TAB_ORDER.len() - 1) % TAB_ORDER.len()];
        state.focus_content = false;
        crate::ui_audio::play_move();
    }
    if tab_right {
        state.tab = TAB_ORDER[(tab_idx + 1) % TAB_ORDER.len()];
        state.focus_content = false;
        crate::ui_audio::play_move();
    }

    if !state.focus_content {
        if up && tab_idx > 0 {
            state.tab = TAB_ORDER[tab_idx - 1];
            crate::ui_audio::play_move();
        }
        if down && tab_idx < TAB_ORDER.len() - 1 {
            state.tab = TAB_ORDER[tab_idx + 1];
            crate::ui_audio::play_move();
        }
        if enter {
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            match state.tab {
                ProfileTab::RecentlyPlayed if n_games > 0 => {
                    state.focus_content = true;
                    state.row_selected = state.row_selected.min(n_games.saturating_sub(1));
                }
                ProfileTab::Settings => {
                    state.focus_content = true;
                    state.row_selected = 0;
                }
                _ => {}
            }
        }
        if back {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            action = ProfileAction::Close;
        }
    } else if state.tab == ProfileTab::Settings {
        if up {
            let prev = state.row_selected;
            state.row_selected = state.row_selected.saturating_sub(1);
            if state.row_selected != prev {
                crate::ui_audio::play_move();
            }
        }
        if down {
            let prev = state.row_selected;
            state.row_selected = (state.row_selected + 1).min(N_SETTINGS - 1);
            if state.row_selected != prev {
                crate::ui_audio::play_move();
            }
        }
        if back {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            state.focus_content = false;
        }
        if state.row_selected == 0 {
            if enter {
                action = ProfileAction::SetBackdropTheme(backdrop_theme.next());
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if leave {
                action = ProfileAction::SetBackdropTheme(backdrop_theme.prev());
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        } else if state.row_selected == 1 {
            if enter {
                action = ProfileAction::SetDockbarTheme(dockbar_theme.next());
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if leave {
                action = ProfileAction::SetDockbarTheme(dockbar_theme.prev());
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        } else if state.row_selected == 2 && (enter || leave) {
            action = ProfileAction::SetLightMode(!light_mode);
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        } else if state.row_selected == 3 {
            if mute_toggle {
                action = ProfileAction::SetMuteMusic(!music_muted);
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if enter {
                action = ProfileAction::SetMusicVolume((music_volume + 0.05).min(1.0));
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if leave {
                action = ProfileAction::SetMusicVolume((music_volume - 0.05).max(0.0));
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        } else if state.row_selected == 4 {
            if mute_toggle {
                action = ProfileAction::SetMuteSfx(!sfx_muted);
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if enter {
                action = ProfileAction::SetSfxVolume((sfx_volume + 0.05).min(1.0));
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            } else if leave {
                action = ProfileAction::SetSfxVolume((sfx_volume - 0.05).max(0.0));
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        } else if state.row_selected == 5 && (enter || leave) {
            action = ProfileAction::SetEuDates(!eu_dates);
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        }
    } else {
        if up {
            let prev = state.row_selected;
            state.row_selected = state.row_selected.saturating_sub(1);
            if state.row_selected != prev {
                crate::ui_audio::play_move();
            }
        }
        if down {
            let prev = state.row_selected;
            state.row_selected = (state.row_selected + 1).min(n_games.saturating_sub(1));
            if state.row_selected != prev {
                crate::ui_audio::play_move();
            }
        }
        if back || leave {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            state.focus_content = false;
        }
    }
    let sidebar_focused = !state.focus_content;

    let side_w = (full.width() * 0.22).max(300.0);
    let side_x = full.min.x + mx;
    let side_top = header_y + 34.0 * s;
    let tabs = [
        (ProfileTab::Profile, "Profile"),
        (ProfileTab::RecentlyPlayed, "Recently Played"),
        (ProfileTab::Settings, "Settings"),
        (ProfileTab::System, "System Information"),
    ];
    let item_h = 64.0 * s;
    for (idx, (tab, label)) in tabs.iter().enumerate() {
        let base_rect = egui::Rect::from_min_size(
            egui::pos2(side_x, side_top + idx as f32 * (item_h + 10.0 * s)),
            Vec2::new(side_w, item_h),
        );
        let r = scale_rect(base_rect);
        let resp = ui.allocate_rect(r, Sense::click());
        if resp.clicked() {
            state.tab = *tab;
            state.focus_content = false;
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        }
        let selected = state.tab == *tab;
        let ring = if sidebar_focused { accent } else { pal.border };
        let rounding = Rounding::same(12.0 * s * scale_factor);
        if selected {
            painter.rect_filled(r, rounding, pal.sel);
            painter.rect_stroke(r, rounding, Stroke::new(1.8 * scale_factor, ring));
            
            let bar_rect = scale_rect(egui::Rect::from_min_size(
                base_rect.min + Vec2::new(6.0 * s, 12.0 * s),
                Vec2::new(4.0 * s, base_rect.height() - 24.0 * s),
            ));
            painter.rect_filled(
                bar_rect,
                Rounding::same(2.0 * s * scale_factor),
                ring,
            );
        } else if resp.hovered() {
            painter.rect_filled(r, rounding, pal.hover);
        }
        painter.text(
            scale_pos(egui::pos2(base_rect.min.x + 26.0 * s, base_rect.center().y)),
            egui::Align2::LEFT_CENTER,
            label,
            FontId::proportional(19.0 * s * scale_factor),
            if selected { pal.text } else { pal.muted },
        );
    }

    let content_full = egui::Rect::from_min_max(
        egui::pos2(side_x + side_w + 48.0 * s, side_top),
        egui::pos2(full.max.x - mx, footer_y - 20.0 * s),
    );

    if state.tab != state.shown_tab {
        state.tab_anim = 0.0;
        state.shown_tab = state.tab;
        state.list_scroll = 0.0;
    }
    let dt = ui.input(|i| i.stable_dt).min(0.1);
    state.tab_anim = (state.tab_anim + dt * 9.0).min(1.0);
    let ease = state.tab_anim * state.tab_anim * (3.0 - 2.0 * state.tab_anim);
    let slide = (1.0 - ease) * 46.0 * s;
    let content = content_full.translate(Vec2::new(slide, 0.0));

    let mut content_painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("profile_content"),
    ));
    content_painter.set_opacity(ease * content_opacity);

    match state.tab {
        ProfileTab::Profile => {
            profile_page(
                ui,
                &content_painter,
                content,
                s,
                profile_name,
                avatar_tex,
                accent,
                state,
                &mut action,
                scale_factor,
                &scale_pos,
                &scale_rect,
                ease * content_opacity,
                pal,
            );
        }
        ProfileTab::RecentlyPlayed => {
            recently_played_page(
                ctx,
                ui,
                &content_painter,
                content,
                s,
                lib,
                play,
                accent,
                state,
                launch_enter,
                &mut action,
                scale_factor,
                &scale_pos,
                &scale_rect,
                ease * content_opacity,
                pal,
            );
        }
        ProfileTab::Settings => {
            settings_page(
                ui,
                &content_painter,
                content,
                s,
                accent,
                state,
                backdrop_theme,
                dockbar_theme,
                light_mode,
                music_volume,
                sfx_volume,
                eu_dates,
                music_muted,
                sfx_muted,
                &mut action,
                scale_factor,
                &scale_pos,
                &scale_rect,
                pal,
            );
        }
        ProfileTab::System => {
            system_page(&content_painter, content, s, scale_factor, &scale_pos, &scale_rect, pal);
        }
    }
    if ease < 1.0 {
        ctx.request_repaint();
    }

    let on_volume = state.tab == ProfileTab::Settings && state.focus_content && (state.row_selected == 3 || state.row_selected == 4);
    let hint = if last_input.connected {
        if on_volume {
            "🎮  [Up/Down] Select      [X] To Mute      [B] Back to Menu"
        } else if state.focus_content {
            "🎮  [Up/Down] Select      [B] Back to Menu"
        } else {
            "🎮  [Up/Down] Move      [A] Enter      [B] Back"
        }
    } else if on_volume {
        "⌨  [Esc] Back      [M] To Mute      Click a tab"
    } else {
        "⌨  [Esc] Back      Click a tab      Scroll wheel"
    };
    painter.text(
        scale_pos(egui::pos2(full.min.x + mx, full.max.y - 30.0 * s)),
        egui::Align2::LEFT_CENTER,
        hint,
        FontId::proportional(14.0 * s * scale_factor),
        pal.muted,
    );

    ctx.request_repaint();
    action
}

#[allow(clippy::too_many_arguments)]
fn system_page(
    painter: &egui::Painter,
    content: egui::Rect,
    s: f32,
    scale_factor: f32,
    scale_pos: &impl Fn(egui::Pos2) -> egui::Pos2,
    scale_rect: &impl Fn(egui::Rect) -> egui::Rect,
    pal: Pal,
) {
    let rows: [(&str, String); 4] = [
        ("NeXium Version", format!("v{}", env!("CARGO_PKG_VERSION"))),
        ("Commit", env!("NEXIUM_GIT_HASH").to_string()),
        ("Build", format!("{} · {}", std::env::consts::OS, std::env::consts::ARCH)),
        ("Update Status", "Auto-update check coming soon".to_string()),
    ];

    let row_h = 58.0 * s;
    let row_gap = 10.0 * s;
    for (idx, (label, value)) in rows.iter().enumerate() {
        let row = egui::Rect::from_min_size(
            content.min + Vec2::new(0.0, idx as f32 * (row_h + row_gap)),
            Vec2::new(content.width(), row_h),
        );
        let r = scale_rect(row);
        let rounding = Rounding::same(12.0 * s * scale_factor);
        painter.rect_filled(r, rounding, pal.panel);
        painter.rect_stroke(r, rounding, Stroke::new(1.0 * scale_factor, pal.border));
        painter.text(
            scale_pos(egui::pos2(row.min.x + 24.0 * s, row.center().y)),
            egui::Align2::LEFT_CENTER,
            *label,
            FontId::proportional(17.0 * s * scale_factor),
            pal.muted,
        );
        painter.text(
            scale_pos(egui::pos2(row.max.x - 26.0 * s, row.center().y)),
            egui::Align2::RIGHT_CENTER,
            value,
            FontId::proportional(18.0 * s * scale_factor),
            pal.text,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn settings_page(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    content: egui::Rect,
    s: f32,
    accent: Color32,
    state: &mut ProfileState,
    backdrop_theme: crate::app_settings::BackdropTheme,
    dockbar_theme: crate::app_settings::DockbarTheme,
    light_mode: bool,
    music_volume: f32,
    sfx_volume: f32,
    eu_dates: bool,
    music_muted: bool,
    sfx_muted: bool,
    action: &mut ProfileAction,
    scale_factor: f32,
    scale_pos: &impl Fn(egui::Pos2) -> egui::Pos2,
    scale_rect: &impl Fn(egui::Rect) -> egui::Rect,
    pal: Pal,
) {
    let row_h = 68.0 * s;
    let row_gap = 12.0 * s;
    
    let rows: [(&str, &str, String); 6] = [
        (
            "Backdrop Theme",
            "Background style behind the menus",
            backdrop_theme.label().to_string(),
        ),
        (
            "Dockbar Theme",
            "Look of the bottom dockbar",
            dockbar_theme.label().to_string(),
        ),
        (
            "Appearance",
            "Light or dark styling of the carousel",
            if light_mode { "Light".to_string() } else { "Dark".to_string() },
        ),
        ("Menu Music", "Background music volume in the carousel", "".to_string()),
        ("SFX Volume", "Sound effects volume for UI interactions", "".to_string()),
        (
            "Time Preference",
            "Either 12 hour clock or 24 hour Military Time",
            if eu_dates { "24h".to_string() } else { "12h".to_string() },
        ),
    ];

    for (idx, (title, subtitle, value)) in rows.iter().enumerate() {
        let row = egui::Rect::from_min_size(
            content.min + Vec2::new(0.0, idx as f32 * (row_h + row_gap)),
            Vec2::new(content.width(), row_h),
        );
        let r = scale_rect(row);
        let resp = ui.allocate_rect(r, Sense::click());
        
        let focused = state.focus_content && state.row_selected == idx;
        
        if resp.clicked() {
            state.focus_content = true;
            state.row_selected = idx;
            match idx {
                0 => {
                    *action = ProfileAction::SetBackdropTheme(backdrop_theme.next());
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
                1 => {
                    *action = ProfileAction::SetDockbarTheme(dockbar_theme.next());
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
                2 => {
                    *action = ProfileAction::SetLightMode(!light_mode);
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
                5 => {
                    *action = ProfileAction::SetEuDates(!eu_dates);
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
                _ => {}
            }
        }
        
        let rounding = Rounding::same(12.0 * s * scale_factor);
        painter.rect_filled(r, rounding, pal.panel);
        let ring = if focused { accent } else { pal.border };
        painter.rect_stroke(r, rounding, Stroke::new(1.8 * scale_factor, ring));

        painter.text(
            scale_pos(egui::pos2(row.min.x + 24.0 * s, row.center().y - 9.0 * s)),
            egui::Align2::LEFT_CENTER,
            *title,
            FontId::proportional(19.0 * s * scale_factor),
            pal.text,
        );
        painter.text(
            scale_pos(egui::pos2(row.min.x + 24.0 * s, row.center().y + 15.0 * s)),
            egui::Align2::LEFT_CENTER,
            *subtitle,
            FontId::proportional(13.0 * s * scale_factor),
            pal.muted,
        );

        let arrow_col = if focused { accent } else { pal.muted };

        if idx == 3 || idx == 4 {
            let val = if idx == 3 { music_volume } else { sfx_volume };
            // mute checkbox to the left of the slider arrows
            let is_muted = if idx == 3 { music_muted } else { sfx_muted };
            let cb = scale_rect(egui::Rect::from_center_size(egui::pos2(row.max.x - 340.0 * s, row.center().y), Vec2::splat(24.0 * s)));
            painter.rect_filled(cb, Rounding::same(5.0 * s * scale_factor), if is_muted { accent } else { pal.input_bg });
            painter.rect_stroke(cb, Rounding::same(5.0 * s * scale_factor), Stroke::new(1.5 * scale_factor, pal.border));
            if is_muted {
                let c = cb.center();
                let z = cb.width() * 0.3;
                painter.add(egui::Shape::line(vec![c + Vec2::new(-z, 0.0), c + Vec2::new(-z * 0.2, z * 0.7), c + Vec2::new(z, -z * 0.8)], Stroke::new(2.2 * scale_factor, Color32::WHITE)));
            }
            painter.text(scale_pos(egui::pos2(row.max.x - 340.0 * s, row.center().y - 20.0 * s)), egui::Align2::CENTER_CENTER, "Mute", FontId::proportional(11.0 * s * scale_factor), pal.muted);
            if ui.allocate_rect(cb, Sense::click()).clicked() {
                *action = if idx == 3 { ProfileAction::SetMuteMusic(!is_muted) } else { ProfileAction::SetMuteSfx(!is_muted) };
                state.focus_content = true;
                state.row_selected = idx;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
            
            // Slider layout dimensions
            let slider_w = 160.0 * s;
            let slider_h = 6.0 * s;
            let track_rect = egui::Rect::from_min_size(
                egui::pos2(row.max.x - 280.0 * s, row.center().y - slider_h * 0.5),
                Vec2::new(slider_w, slider_h),
            );
            let scaled_track = scale_rect(track_rect);
            
            // Slider interaction (drag & click)
            let slider_resp = ui.allocate_rect(scaled_track.expand(8.0 * scale_factor), Sense::click_and_drag());
            let mut new_val = val;
            if slider_resp.clicked() || slider_resp.dragged() {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let pct = ((pos.x - scaled_track.min.x) / scaled_track.width()).clamp(0.0, 1.0);
                    new_val = pct;
                }
            }
            if new_val != val {
                if idx == 3 {
                    *action = ProfileAction::SetMusicVolume(new_val);
                } else {
                    *action = ProfileAction::SetSfxVolume(new_val);
                }
            }

            // Arrow bounds
            let left_arrow_rect = scale_rect(egui::Rect::from_center_size(
                egui::pos2(row.max.x - 300.0 * s, row.center().y),
                Vec2::splat(30.0 * s),
            ));
            let right_arrow_rect = scale_rect(egui::Rect::from_center_size(
                egui::pos2(row.max.x - 30.0 * s, row.center().y),
                Vec2::splat(30.0 * s),
            ));

            let la_resp = ui.allocate_rect(left_arrow_rect, Sense::click());
            let ra_resp = ui.allocate_rect(right_arrow_rect, Sense::click());

            if la_resp.clicked() {
                let step = (val - 0.05).max(0.0);
                if idx == 3 {
                    *action = ProfileAction::SetMusicVolume(step);
                } else {
                    *action = ProfileAction::SetSfxVolume(step);
                }
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
            if ra_resp.clicked() {
                let step = (val + 0.05).min(1.0);
                if idx == 3 {
                    *action = ProfileAction::SetMusicVolume(step);
                } else {
                    *action = ProfileAction::SetSfxVolume(step);
                }
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }

            let arrow_col_l = if focused || la_resp.hovered() { accent } else { pal.muted };
            let arrow_col_r = if focused || ra_resp.hovered() { accent } else { pal.muted };

            // Draw track background
            let rr = Rounding::same(slider_h * 0.5 * scale_factor);
            painter.rect_filled(scaled_track, rr, pal.input_bg);
            
            // Draw filled track
            let mut fill = scaled_track;
            fill.max.x = scaled_track.min.x + scaled_track.width() * val.clamp(0.0, 1.0);
            painter.rect_filled(fill, rr, accent);

            // Draw slider handle circle
            let handle_x = scaled_track.min.x + scaled_track.width() * val.clamp(0.0, 1.0);
            let handle_center = egui::pos2(handle_x, scaled_track.center().y);
            let handle_r = 8.0 * s * scale_factor;
            painter.circle_filled(handle_center, handle_r, Color32::WHITE);
            painter.circle_stroke(handle_center, handle_r, Stroke::new(1.5 * scale_factor, accent));

            // Draw percentage label
            let val_label = if val <= 0.001 {
                "Off".to_string()
            } else {
                format!("{}%", (val * 100.0).round() as i32)
            };
            painter.text(
                scale_pos(egui::pos2(row.max.x - 70.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                val_label,
                FontId::proportional(18.0 * s * scale_factor),
                pal.text,
            );

            // Draw arrows
            painter.text(
                scale_pos(egui::pos2(row.max.x - 300.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                "‹",
                FontId::proportional(26.0 * s * scale_factor),
                arrow_col_l,
            );
            painter.text(
                scale_pos(egui::pos2(row.max.x - 30.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                "›",
                FontId::proportional(26.0 * s * scale_factor),
                arrow_col_r,
            );
        } else {
            // clickable left/right arrows (mouse users): left = previous, right = next
            let la = scale_rect(egui::Rect::from_center_size(egui::pos2(row.max.x - 168.0 * s, row.center().y), Vec2::splat(34.0 * s)));
            let ra = scale_rect(egui::Rect::from_center_size(egui::pos2(row.max.x - 30.0 * s, row.center().y), Vec2::splat(34.0 * s)));
            let la_c = ui.allocate_rect(la, Sense::click()).clicked();
            let ra_c = ui.allocate_rect(ra, Sense::click()).clicked();
            if la_c || ra_c {
                let fwd = ra_c;
                match idx {
                    0 => *action = ProfileAction::SetBackdropTheme(if fwd { backdrop_theme.next() } else { backdrop_theme.prev() }),
                    1 => *action = ProfileAction::SetDockbarTheme(if fwd { dockbar_theme.next() } else { dockbar_theme.prev() }),
                    2 => *action = ProfileAction::SetLightMode(!light_mode),
                    5 => *action = ProfileAction::SetEuDates(!eu_dates),
                    _ => {}
                }
                state.focus_content = true;
                state.row_selected = idx;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
            painter.text(
                scale_pos(egui::pos2(row.max.x - 168.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                "‹",
                FontId::proportional(26.0 * s * scale_factor),
                arrow_col,
            );
            painter.text(
                scale_pos(egui::pos2(row.max.x - 100.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                value.clone(),
                FontId::proportional(18.0 * s * scale_factor),
                pal.text,
            );
            painter.text(
                scale_pos(egui::pos2(row.max.x - 32.0 * s, row.center().y)),
                egui::Align2::CENTER_CENTER,
                "›",
                FontId::proportional(26.0 * s * scale_factor),
                arrow_col,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_page(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    content: egui::Rect,
    s: f32,
    profile_name: &str,
    avatar_tex: Option<egui::TextureId>,
    accent: Color32,
    state: &mut ProfileState,
    action: &mut ProfileAction,
    scale_factor: f32,
    scale_pos: &impl Fn(egui::Pos2) -> egui::Pos2,
    scale_rect: &impl Fn(egui::Rect) -> egui::Rect,
    _opacity: f32,
    pal: Pal,
) {
    let av_r = (content.height() * 0.18).clamp(90.0, 190.0);
    let av_center = egui::pos2(content.min.x + av_r + 20.0 * s, content.center().y - av_r * 0.2);
    let now = ui.input(|i| i.time) as f32;
    let sc_av = scale_pos(av_center);
    let sc_avr = av_r * scale_factor;

    let av_resp = ui.allocate_rect(egui::Rect::from_center_size(sc_av, Vec2::splat(sc_avr * 2.0)), Sense::click());
    if av_resp.clicked() {
        state.avatar_bounce = Some(now);
        // roll for a rare celebration (~1 in 4)
        let rng = |k: f32| { let h = ((now * 61.7 + k * 13.13).sin() * 43758.5453).fract(); h.abs() };
        if rng(1.0) < 0.24 {
            crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
            state.avatar_stars.push((now, -0.85, 0.0, 255)); // the star
            for k in 0..14 {
                let ang = -std::f32::consts::PI * (0.15 + rng(k as f32 * 2.0) * 0.7);
                let spd = 260.0 + rng(k as f32 * 3.1) * 360.0;
                state.avatar_stars.push((now, ang, spd, (k % 6) as u8));
            }
        } else {
            crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
        }
        if state.avatar_stars.len() > 80 {
            let drop = state.avatar_stars.len() - 80;
            state.avatar_stars.drain(0..drop);
        }
    }

    // springy bounce on the avatar
    let mut bounce = 1.0f32;
    if let Some(bt) = state.avatar_bounce {
        let e = now - bt;
        if e < 0.42 {
            let p = e / 0.42;
            bounce = 1.0 + (p * std::f32::consts::PI * 2.0).sin() * 0.14 * (1.0 - p);
            ui.ctx().request_repaint();
        } else {
            state.avatar_bounce = None;
        }
    }
    draw_circle_avatar(painter, sc_av, sc_avr * bounce, avatar_tex, accent, scale_factor, pal);

    // celebration particles (short-lived confetti + a star that burns up fast)
    state.avatar_stars.retain(|(t0, _, _, kind)| now - t0 < if *kind == 255 { 1.1 } else { 1.0 });
    let palette = [
        Color32::from_rgb(0xE8, 0x33, 0x50),
        Color32::from_rgb(0x2F, 0xB4, 0xEF),
        Color32::from_rgb(0x35, 0xD0, 0x6A),
        Color32::from_rgb(0xF5, 0xC1, 0x42),
        Color32::from_rgb(0xC9, 0x5C, 0xF6),
        Color32::from_rgb(0xFF, 0x8A, 0x3D),
    ];
    let head = sc_av + Vec2::new(sc_avr * 0.62, -sc_avr * 0.7);
    for &(t0, ang, spd, kind) in &state.avatar_stars {
        let age = now - t0;
        if kind == 255 {
            let p = (age / 1.1).min(1.0);
            let pop = if p < 0.18 { p / 0.18 } else { 1.0 };
            let fade = if p > 0.6 { ((1.0 - p) / 0.4).clamp(0.0, 1.0) } else { 1.0 };
            let sp = head + Vec2::new(0.0, -sc_avr * 0.10 * p) + Vec2::new((ang).cos(), (ang).sin()) * sc_avr * 0.12 * p;
            let sr = sc_avr * 0.16 * pop * (0.7 + 0.3 * fade);
            let col = Color32::from_rgba_unmultiplied(0xFF, 0xCB, 0x2E, (fade * 255.0) as u8);
            let outline = Color32::from_rgba_unmultiplied(0xC8, 0x8A, 0x00, (fade * 255.0) as u8);
            let pts: Vec<egui::Pos2> = (0..10)
                .map(|k| {
                    let a = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::PI / 5.0;
                    let rr = if k % 2 == 0 { sr } else { sr * 0.5 };
                    sp + Vec2::new(a.cos() * rr, a.sin() * rr)
                })
                .collect();
            // filled star via a triangle fan from the center (convex_polygon can't do concave)
            let mut mesh = egui::epaint::Mesh::default();
            let mk = |pos: egui::Pos2, c: Color32| egui::epaint::Vertex { pos, uv: egui::pos2(0.0, 0.0), color: c };
            mesh.vertices.push(mk(sp, col));
            for &pp in &pts {
                mesh.vertices.push(mk(pp, col));
            }
            for k in 0..10u32 {
                mesh.indices.extend_from_slice(&[0, 1 + k, 1 + ((k + 1) % 10)]);
            }
            painter.add(egui::Shape::mesh(mesh));
            painter.add(egui::Shape::closed_line(pts, Stroke::new(1.6 * scale_factor, outline)));
        } else {
            let p = (age / 1.0).min(1.0);
            let fade = (1.0 - p * p).clamp(0.0, 1.0);
            let vel = Vec2::new(ang.cos(), ang.sin()) * spd * scale_factor;
            let pos = sc_av + vel * age + Vec2::new(0.0, 620.0 * scale_factor * age * age);
            let col = palette[kind as usize % palette.len()];
            let c = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), (fade * 255.0) as u8);
            let sz = sc_avr * 0.05;
            let rot = age * 12.0 + kind as f32;
            let rx = Vec2::new(rot.cos(), rot.sin()) * sz;
            let ry = Vec2::new(-rot.sin(), rot.cos()) * sz * 0.5;
            painter.add(egui::Shape::convex_polygon(vec![pos + rx + ry, pos + rx - ry, pos - rx - ry, pos - rx + ry], c, Stroke::NONE));
        }
        ui.ctx().request_repaint();
    }

    let btn_w = av_r * 2.0;
    let btn_h = 42.0 * s;
    let btn_rect = egui::Rect::from_min_size(
        egui::pos2(av_center.x - av_r, av_center.y + av_r + 22.0 * s),
        Vec2::new(btn_w, btn_h),
    );
    let scaled_btn_rect = scale_rect(btn_rect);
    let btn_resp = ui.allocate_rect(scaled_btn_rect, Sense::click());
    let rounding = Rounding::same(9.0 * s * scale_factor);
    let fill_a = if btn_resp.hovered() { 150 } else { 115 };
    let fill = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), fill_a);
    let border = if btn_resp.hovered() { brighten(accent, 0.2) } else { accent };
    painter.rect_filled(scaled_btn_rect, rounding, fill);
    painter.rect_stroke(scaled_btn_rect, rounding, Stroke::new(2.0 * scale_factor, border));
    let text_pos = scaled_btn_rect.center();
    let font_id = FontId::proportional(15.5 * s * scale_factor);
    let af = fill_a as f32 / 255.0;
    let er = accent.r() as f32 * af + pal.panel.r() as f32 * (1.0 - af);
    let eg = accent.g() as f32 * af + pal.panel.g() as f32 * (1.0 - af);
    let eb = accent.b() as f32 * af + pal.panel.b() as f32 * (1.0 - af);
    let lum = 0.299 * er + 0.587 * eg + 0.114 * eb;
    let (txt_col, halo) = if lum > 150.0 {
        (Color32::from_rgb(0x1A, 0x1A, 0x22), Color32::from_rgba_unmultiplied(255, 255, 255, 130))
    } else {
        (Color32::WHITE, Color32::from_rgba_unmultiplied(0, 0, 0, 120))
    };
    for off in [Vec2::new(-1.2, -1.2), Vec2::new(1.2, -1.2), Vec2::new(-1.2, 1.2), Vec2::new(1.2, 1.2)] {
        painter.text(text_pos + off, egui::Align2::CENTER_CENTER, "Change Icon", font_id.clone(), halo);
    }
    for dx in [0.0f32, 0.7] {
        painter.text(text_pos + Vec2::new(dx, 0.0), egui::Align2::CENTER_CENTER, "Change Icon", font_id.clone(), txt_col);
    }
    if btn_resp.clicked() {
        *action = ProfileAction::PickIcon;
    }

    let right_x = av_center.x + av_r + 60.0 * s;
    let label_y = av_center.y - 46.0 * s;
    painter.text(
        scale_pos(egui::pos2(right_x, label_y)),
        egui::Align2::LEFT_CENTER,
        "DISPLAY NAME",
        FontId::proportional(14.0 * s * scale_factor),
        pal.muted,
    );

    let field_w = (content.max.x - right_x).min(520.0 * s).max(200.0);
    let field_rect = egui::Rect::from_min_size(
        egui::pos2(right_x, label_y + 22.0 * s),
        Vec2::new(field_w, 48.0 * s),
    );
    let scaled_field_rect = scale_rect(field_rect);

    let primary_clicked = ui.input(|i| i.pointer.primary_clicked());
    if primary_clicked {
        state.name_editing = ui.rect_contains_pointer(scaled_field_rect);
    }

    if state.name_editing {
        let events = ui.input(|i| i.events.clone());
        for ev in events {
            match ev {
                egui::Event::Text(txt) => {
                    for ch in txt.chars() {
                        if !ch.is_control() && state.name_buf.chars().count() < 24 {
                            state.name_buf.push(ch);
                        }
                    }
                }
                egui::Event::Key {
                    key: egui::Key::Backspace,
                    pressed: true,
                    ..
                } => {
                    state.name_buf.pop();
                }
                egui::Event::Key {
                    key: egui::Key::Enter,
                    pressed: true,
                    ..
                } => {
                    let name = state.name_buf.trim().to_string();
                    let name = if name.is_empty() { "Player".to_string() } else { name };
                    state.name_buf = name.clone();
                    state.name_editing = false;
                    if name != profile_name {
                        *action = ProfileAction::SetName(name);
                    }
                }
                egui::Event::Key {
                    key: egui::Key::Escape,
                    pressed: true,
                    ..
                } => {
                    state.name_buf = profile_name.to_string();
                    state.name_editing = false;
                }
                _ => {}
            }
        }
    }

    painter.rect_filled(scaled_field_rect, Rounding::same(9.0 * s * scale_factor), pal.input_bg);
    painter.rect_stroke(
        scaled_field_rect,
        Rounding::same(9.0 * s * scale_factor),
        Stroke::new(1.8 * scale_factor, if state.name_editing { accent } else { pal.border }),
    );

    let font = FontId::proportional(22.0 * s * scale_factor);
    let inner_pad = 14.0 * s * scale_factor;
    let avail = field_w * scale_factor - inner_pad * 2.0;
    let galley = painter.layout_no_wrap(state.name_buf.clone(), font.clone(), pal.soft_text);
    let tw = galley.size().x;
    let scroll_off = (tw - avail).max(0.0);
    let text_clip = scaled_field_rect.shrink2(Vec2::new(inner_pad * 0.5, 0.0));
    let tp = painter.with_clip_rect(text_clip);
    let text_x = scaled_field_rect.min.x + inner_pad - scroll_off;
    tp.text(
        egui::pos2(text_x, scaled_field_rect.center().y),
        egui::Align2::LEFT_CENTER,
        &state.name_buf,
        font.clone(),
        pal.soft_text,
    );
    if state.name_editing {
        let blink = (ui.input(|i| i.time) * 1.6).fract() < 0.5;
        if blink {
            let caret_x = (text_x + tw).min(scaled_field_rect.max.x - inner_pad * 0.4);
            tp.line_segment(
                [
                    egui::pos2(caret_x, scaled_field_rect.center().y - 13.0 * s * scale_factor),
                    egui::pos2(caret_x, scaled_field_rect.center().y + 13.0 * s * scale_factor),
                ],
                Stroke::new(2.0 * scale_factor, pal.soft_text),
            );
        }
    } else if state.name_buf.is_empty() {
        tp.text(
            egui::pos2(scaled_field_rect.min.x + inner_pad, scaled_field_rect.center().y),
            egui::Align2::LEFT_CENTER,
            "Enter a name…",
            font,
            pal.muted,
        );
    }

    painter.text(
        scale_pos(egui::pos2(right_x, field_rect.max.y + 20.0 * s)),
        egui::Align2::LEFT_TOP,
        if state.name_editing { "Press Enter to save  ·  Esc to cancel" } else { "Click to edit  ·  Enter to save" },
        FontId::proportional(12.5 * s * scale_factor),
        pal.muted,
    );
}

#[allow(clippy::too_many_arguments)]
fn recently_played_page(
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    base_painter: &egui::Painter,
    content: egui::Rect,
    s: f32,
    lib: &mut Library,
    play: &PlayTimes,
    accent: Color32,
    state: &mut ProfileState,
    enter: bool,
    action: &mut ProfileAction,
    scale_factor: f32,
    scale_pos: &impl Fn(egui::Pos2) -> egui::Pos2,
    scale_rect: &impl Fn(egui::Rect) -> egui::Rect,
    _opacity: f32,
    pal: Pal,
) {
    let mut order: Vec<usize> = (0..lib.games.len()).collect();
    order.sort_by(|&a, &b| {
        let ta = play.get(&lib.games[a].path);
        let tb = play.get(&lib.games[b].path);
        tb.cmp(&ta).then(
            lib.games[a]
                .title
                .to_lowercase()
                .cmp(&lib.games[b].title.to_lowercase()),
        )
    });

    let scaled_content = scale_rect(content);
    let painter = base_painter.with_clip_rect(scaled_content);

    if order.is_empty() {
        painter.text(
            scale_pos(content.min + Vec2::new(0.0, 20.0 * s)),
            egui::Align2::LEFT_TOP,
            "No games found",
            FontId::proportional(17.0 * s * scale_factor),
            pal.muted,
        );
        return;
    }
    state.row_selected = state.row_selected.min(order.len() - 1);

    if state.focus_content && enter {
        let path = lib.games[order[state.row_selected]].path.to_string_lossy().to_string();
        *action = ProfileAction::QuickLaunch(path);
    }

    let row_h = 52.0 * s;
    let gap = 8.0 * s;
    let top_pad = 18.0 * s;
    let left_pad = 14.0 * s;
    let right_pad = 24.0 * s;
    let row_w = content.width() - left_pad - right_pad;
    let step = row_h + gap;
    let total = order.len() as f32 * step - gap + top_pad * 2.0;
    let view_h = content.height();

    let sel_center = top_pad + state.row_selected as f32 * step + row_h * 0.5;
    let mut target = sel_center - view_h * 0.5;
    let max_scroll = (total - view_h).max(0.0);
    target = target.clamp(0.0, max_scroll);
    if state.focus_content {
        state.list_scroll += (target - state.list_scroll) * 0.35;
    }
    let wheel = ui.input(|i| i.raw_scroll_delta.y);
    if wheel != 0.0 {
        state.list_scroll = (state.list_scroll - wheel).clamp(0.0, max_scroll);
    }
    state.list_scroll = state.list_scroll.clamp(0.0, max_scroll);

    for (vis, &i) in order.iter().enumerate() {
        let y = content.min.y + top_pad + vis as f32 * step - state.list_scroll;
        if y + row_h < content.min.y || y > content.max.y {
            continue;
        }
        let rect = egui::Rect::from_min_size(
            egui::pos2(content.min.x + left_pad, y),
            Vec2::new(row_w, row_h),
        );
        let title = lib.games[i].title.clone();
        let secs = play.get(&lib.games[i].path);
        let tex = lib.texture(ctx, i);
        let scaled_rect = scale_rect(rect);
        let row_resp = ui.interact(scaled_rect, egui::Id::new(("rp_row", i)), Sense::click());
        if row_resp.clicked() {
            if state.focus_content && state.row_selected == vis {
                *action = ProfileAction::QuickLaunch(lib.games[i].path.to_string_lossy().to_string());
            } else {
                state.focus_content = true;
                state.row_selected = vis;
            }
        }
        let is_sel = state.focus_content && vis == state.row_selected;

        painter.rect_filled(
            scaled_rect,
            Rounding::same(9.0 * s * scale_factor),
            if is_sel { pal.sel } else { pal.panel },
        );
        painter.rect_stroke(
            scaled_rect,
            Rounding::same(9.0 * s * scale_factor),
            Stroke::new(if is_sel { 2.0 * scale_factor } else { 1.0 * scale_factor }, if is_sel { accent } else { pal.border }),
        );

        let pad = 7.0 * s;
        let icon = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + pad, rect.min.y + pad),
            Vec2::splat(row_h - pad * 2.0),
        );
        let scaled_icon = scale_rect(icon);
        if let Some(tex) = tex {
            let mut mesh = egui::epaint::Mesh::with_texture(tex.id());
            let color = Color32::from_rgba_premultiplied(255, 255, 255, 255);
            mesh.add_rect_with_uv(
                scaled_icon,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                color,
            );
            painter.add(egui::Shape::mesh(mesh));
        } else {
            painter.rect_filled(scaled_icon, Rounding::same(7.0 * s * scale_factor), Color32::from_rgb(0x22, 0x22, 0x2C));
        }

        painter.text(
            scale_pos(egui::pos2(icon.max.x + 14.0 * s, rect.center().y - 10.0 * s)),
            egui::Align2::LEFT_CENTER,
            &title,
            FontId::proportional(16.0 * s * scale_factor),
            pal.text,
        );
        let time_str = if secs == 0 {
            "Never played".to_string()
        } else {
            format_playtime(secs)
        };
        painter.text(
            scale_pos(egui::pos2(icon.max.x + 14.0 * s, rect.center().y + 11.0 * s)),
            egui::Align2::LEFT_CENTER,
            &time_str,
            FontId::proportional(12.5 * s * scale_factor),
            pal.muted,
        );
    }

    if max_scroll > 0.0 {
        let track_x = content.max.x - 5.0 * s;
        let frac_view = (view_h / total).min(1.0);
        let bar_h = (view_h * frac_view).max(30.0 * s);
        let bar_y = content.min.y + (state.list_scroll / max_scroll) * (view_h - bar_h);
        
        let track_rect = scale_rect(egui::Rect::from_min_size(egui::pos2(track_x, content.min.y), Vec2::new(4.0 * s, view_h)));
        let bar_rect = scale_rect(egui::Rect::from_min_size(egui::pos2(track_x, bar_y), Vec2::new(4.0 * s, bar_h)));

        painter.rect_filled(
            track_rect,
            Rounding::same(2.0 * s * scale_factor),
            Color32::from_rgb(0x20, 0x20, 0x2A),
        );
        painter.rect_filled(
            bar_rect,
            Rounding::same(2.0 * s * scale_factor),
            accent,
        );
    }
}
