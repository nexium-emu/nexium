use super::look::{self, ButtonKind, Palette};
use super::service::{normalize_friend_code, Busy, Friend, GameAccess, Phase, Request, State};
use super::widgets::group;
use crate::controller_config::SwitchButton;
use crate::input::InputSnapshot;
use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use std::path::PathBuf;

const NAV_REPEAT: f64 = 0.16;
const ACCOUNT_URL: &str = "https://nextendo.network";
const TAB_COUNT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Friends,
    Requests,
    Players,
    AddFriend,
    OnlinePlay,
}

const TABS: [Tab; TAB_COUNT] = [
    Tab::Overview,
    Tab::Friends,
    Tab::Requests,
    Tab::Players,
    Tab::AddFriend,
    Tab::OnlinePlay,
];

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::Overview => "Account",
            Tab::Friends => "Friends",
            Tab::Requests => "Friend Requests",
            Tab::Players => "Recent Players",
            Tab::AddFriend => "Add Friend",
            Tab::OnlinePlay => "Online Play",
        }
    }

    fn index(self) -> usize {
        TABS.iter().position(|tab| *tab == self).unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HubAction {
    None,
    Close,
    SignIn,
    CancelSignIn,
    ReopenBrowser,
    OpenUrl(String),
    SignOut,
    Refresh,
    AddFriend(String),
    Answer { pid: u64, accept: bool },
    Remove(u64),
    SetOnlinePlay(bool),
    SetSharePresence(bool),
    SetFriendAlerts(bool),
    SetServers { server: String, nat: String },
    RestoreServers,
    Launch(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubSettings {
    pub online_play: bool,
    pub share_presence: bool,
    pub friend_alerts: bool,
    pub server: String,
    pub nat: String,
}

#[derive(Clone, Debug)]
pub struct LibraryTitle {
    pub title_id: u64,
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub icon: Option<egui::TextureId>,
}

pub struct HubFrame<'a> {
    pub state: &'a State,
    pub settings: &'a HubSettings,
    pub titles: &'a [LibraryTitle],
    pub ambient: Color32,
    pub accent: Color32,
    pub backdrop: crate::app_settings::BackdropTheme,
    pub light_mode: bool,
    pub content_opacity: f32,
    pub backdrop_opacity: f32,
    pub zoom: f32,
    pub active: bool,
    pub input: &'a InputSnapshot,
    pub running_title: Option<u64>,
}

#[derive(Default)]
struct TextBuf {
    text: String,
    caret: usize,
    anchor: usize,
    editing: bool,
}

impl TextBuf {
    fn set(&mut self, text: &str) {
        self.text = text.to_string();
        self.caret = self.text.chars().count();
        self.anchor = self.caret;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    FriendCode,
    Server,
    Nat,
}

#[derive(Clone, Debug, PartialEq)]
enum Confirm {
    SignOut,
    Remove { pid: u64, name: String },
}

pub struct HubState {
    pub tab: Tab,
    shown_tab: Tab,
    tab_anim: f32,
    focus_content: bool,
    row: usize,
    col: usize,
    scroll: [f32; TAB_COUNT],
    nav_cooldown: f64,
    held_a: bool,
    held_b: bool,
    held_y: bool,
    held_shoulder: bool,
    code: TextBuf,
    server: TextBuf,
    nat: TextBuf,
    servers_loaded: bool,
    server_error: Option<String>,
    keyboard: crate::vkeyboard::VirtualKeyboard,
    keyboard_field: Option<Field>,
    confirm: Option<Confirm>,
    confirm_choice: usize,
    theme_t: f32,
    pub avatars: look::AvatarCache,
    copied_at: Option<f64>,
    awaiting_add: bool,
    requested: std::collections::HashSet<u64>,
}

impl HubState {
    pub fn new() -> Self {
        Self {
            tab: Tab::Overview,
            shown_tab: Tab::Overview,
            tab_anim: 1.0,
            focus_content: false,
            row: 0,
            col: 0,
            scroll: [0.0; TAB_COUNT],
            nav_cooldown: 0.0,
            held_a: false,
            held_b: false,
            held_y: false,
            held_shoulder: false,
            code: TextBuf::default(),
            server: TextBuf::default(),
            nat: TextBuf::default(),
            servers_loaded: false,
            server_error: None,
            keyboard: crate::vkeyboard::VirtualKeyboard::new(),
            keyboard_field: None,
            confirm: None,
            confirm_choice: 0,
            theme_t: 0.0,
            avatars: look::AvatarCache::new(),
            copied_at: None,
            awaiting_add: false,
            requested: std::collections::HashSet::new(),
        }
    }

    pub fn open(&mut self, tab: Tab) {
        self.tab = tab;
        self.focus_content = tab != Tab::Overview;
        self.row = 0;
        self.col = 0;
        self.confirm = None;
        self.servers_loaded = false;
        self.server_error = None;
    }

    fn editing(&self) -> bool {
        self.code.editing || self.server.editing || self.nat.editing || self.keyboard.open
    }

    fn stop_editing(&mut self) {
        self.code.editing = false;
        self.server.editing = false;
        self.nat.editing = false;
    }
}

#[derive(Default, Clone, Copy)]
struct Pad {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    accept: bool,
    back: bool,
    refresh: bool,
    prev_tab: bool,
    next_tab: bool,
}

fn read_pad(state: &mut HubState, ui: &egui::Ui, input: &InputSnapshot, now: f64, typing: bool) -> Pad {
    let mut pad = Pad::default();
    if !typing {
        ui.input(|keys| {
            pad.up |= keys.key_pressed(egui::Key::ArrowUp);
            pad.down |= keys.key_pressed(egui::Key::ArrowDown);
            pad.left |= keys.key_pressed(egui::Key::ArrowLeft);
            pad.right |= keys.key_pressed(egui::Key::ArrowRight);
            pad.accept |= keys.key_pressed(egui::Key::Enter);
            pad.back |= keys.key_pressed(egui::Key::Escape);
            pad.refresh |= keys.key_pressed(egui::Key::F5);
        });
    }
    if !input.connected {
        state.held_a = false;
        state.held_b = false;
        state.held_y = false;
        state.held_shoulder = false;
        return pad;
    }
    let a = input.is(SwitchButton::A);
    let b = input.is(SwitchButton::B);
    let y = input.is(SwitchButton::Y);
    let l = input.is(SwitchButton::L);
    let r = input.is(SwitchButton::R);
    if !state.keyboard.open {
        pad.accept |= a && !state.held_a;
        pad.back |= b && !state.held_b;
        pad.refresh |= y && !state.held_y;
        if !state.held_shoulder {
            pad.prev_tab |= l;
            pad.next_tab |= r;
        }
        if now - state.nav_cooldown > NAV_REPEAT {
            let ly = input.ly();
            let lx = input.lx();
            let up = input.is(SwitchButton::DUp) || ly > 0.5;
            let down = input.is(SwitchButton::DDown) || ly < -0.5;
            let left = input.is(SwitchButton::DLeft) || lx < -0.5;
            let right = input.is(SwitchButton::DRight) || lx > 0.5;
            if up || down || left || right {
                state.nav_cooldown = now;
            }
            pad.up |= up;
            pad.down |= down;
            pad.left |= left;
            pad.right |= right;
        }
    }
    state.held_a = a;
    state.held_b = b;
    state.held_y = y;
    state.held_shoulder = l || r;
    pad
}

struct Nav {
    rows: Vec<Vec<Rect>>,
    row: usize,
    col: usize,
    accept: bool,
    content: bool,
    clip: Rect,
}

impl Nav {
    fn slot(&mut self, row: usize, rect: Rect) -> (bool, bool) {
        while self.rows.len() <= row {
            self.rows.push(Vec::new());
        }
        let col = self.rows[row].len();
        self.rows[row].push(rect);
        let focused = self.content && self.row == row && self.col == col;
        (focused, focused && self.accept)
    }

    fn next_row(&self) -> usize {
        self.rows.len()
    }
}

struct Page<'a, 'b> {
    ui: &'a mut egui::Ui,
    painter: egui::Painter,
    frame: &'a HubFrame<'b>,
    pal: Palette,
    accent: Color32,
    s: f32,
    time: f64,
    nav: Nav,
    action: HubAction,
    enabled: bool,
}

impl Page<'_, '_> {
    fn interactive(&self, rect: Rect) -> Rect {
        if self.enabled {
            rect.intersect(self.nav.clip)
        } else {
            Rect::NOTHING
        }
    }

    fn button(&mut self, row: usize, rect: Rect, label: &str, kind: ButtonKind, enabled: bool) -> bool {
        let (focused, accepted) = self.nav.slot(row, rect);
        let hit = self.interactive(rect);
        let response = (enabled && hit.is_positive()).then(|| self.ui.allocate_rect(hit, Sense::click()));
        let hovered = response.as_ref().is_some_and(|response| response.hovered());
        look::paint_button(
            &self.painter,
            rect,
            label,
            kind,
            self.accent,
            self.pal,
            hovered,
            focused,
            enabled,
            15.0 * self.s,
        );
        if hovered {
            self.ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        enabled && (response.is_some_and(|response| response.clicked()) || accepted)
    }

    fn icon_button(&mut self, row: usize, rect: Rect, draw: impl Fn(&egui::Painter, Pos2, f32, Color32), tooltip: &str) -> bool {
        let (focused, accepted) = self.nav.slot(row, rect);
        let hit = self.interactive(rect);
        let response = if hit.is_positive() {
            Some(self.ui.allocate_rect(hit, Sense::click()))
        } else {
            None
        };
        let hovered = response.as_ref().is_some_and(|response| response.hovered());
        let radius = rect.height() * 0.3;
        if hovered || focused {
            self.painter.rect_filled(rect, CornerRadius::from(radius), self.pal.panel_hover);
        }
        if focused {
            self.painter.rect_stroke(
                rect,
                CornerRadius::from(radius),
                Stroke::new(1.8, self.accent),
                egui::StrokeKind::Outside,
            );
        }
        let color = if hovered || focused { self.pal.text } else { self.pal.muted };
        draw(&self.painter, rect.center(), rect.height() * 0.62, color);
        let clicked = response.is_some_and(|response| {
            if response.hovered() {
                self.ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            response.on_hover_text(tooltip).clicked()
        });
        clicked || accepted
    }

    fn text(&self, pos: Pos2, align: egui::Align2, text: &str, size: f32, color: Color32) -> Rect {
        self.painter
            .text(pos, align, text, FontId::proportional(size * self.s), color)
    }

    fn paragraph(&self, pos: Pos2, text: &str, size: f32, color: Color32, width: f32) -> f32 {
        let galley = self.painter.layout(
            text.to_string(),
            FontId::proportional(size * self.s),
            color,
            width,
        );
        let height = galley.size().y;
        self.painter.galley(pos, galley, color);
        height
    }

    fn label(&self, pos: Pos2, text: &str) {
        self.text(pos, egui::Align2::LEFT_CENTER, text, 12.5, self.pal.faint);
    }

    fn panel(&self, rect: Rect, focused: bool) {
        look::card(
            &self.painter,
            rect,
            12.0 * self.s,
            self.pal.panel,
            if focused { self.accent } else { self.pal.border },
            if focused { 1.8 } else { 1.0 },
        );
    }

    fn banner(&self, rect: Rect, color: Color32, message: &str) {
        look::card(
            &self.painter,
            rect,
            10.0 * self.s,
            look::alpha(color, 0.12),
            look::alpha(color, 0.55),
            1.0,
        );
        self.painter.circle_filled(
            Pos2::new(rect.min.x + 20.0 * self.s, rect.center().y),
            4.0 * self.s,
            color,
        );
        let available = rect.width() - 48.0 * self.s;
        let font = FontId::proportional(14.0 * self.s);
        let text = look::ellipsize(self.ui, message, &font, available);
        self.painter.text(
            Pos2::new(rect.min.x + 34.0 * self.s, rect.center().y),
            egui::Align2::LEFT_CENTER,
            text,
            font,
            self.pal.text,
        );
    }

    fn row_hover(&mut self, rect: Rect) -> bool {
        let hit = self.interactive(rect);
        hit.is_positive() && self.ui.rect_contains_pointer(hit)
    }

    fn avatar(&mut self, center: Pos2, radius: f32, pid: u64, name: &str, avatars: &mut look::AvatarCache) {
        let texture = avatars.texture(self.ui.ctx(), &self.frame.state.avatars, pid);
        look::avatar(&self.painter, center, radius, texture, name, pid, 1.0);
    }

    fn toggle_row(&mut self, row: usize, rect: Rect, title: &str, subtitle: &str, on: bool) -> bool {
        let (focused, accepted) = self.nav.slot(row, rect);
        let hit = self.interactive(rect);
        let response = hit.is_positive().then(|| self.ui.allocate_rect(hit, Sense::click()));
        let hovered = response.as_ref().is_some_and(|response| response.hovered());
        look::card(
            &self.painter,
            rect,
            12.0 * self.s,
            if hovered { self.pal.panel_hover } else { self.pal.panel },
            if focused { self.accent } else { self.pal.border },
            if focused { 1.8 } else { 1.0 },
        );
        self.text(
            Pos2::new(rect.min.x + 24.0 * self.s, rect.center().y - 10.0 * self.s),
            egui::Align2::LEFT_CENTER,
            title,
            18.0,
            self.pal.text,
        );
        let available = rect.width() - 140.0 * self.s;
        let font = FontId::proportional(13.0 * self.s);
        let subtitle = look::ellipsize(self.ui, subtitle, &font, available);
        self.painter.text(
            Pos2::new(rect.min.x + 24.0 * self.s, rect.center().y + 13.0 * self.s),
            egui::Align2::LEFT_CENTER,
            subtitle,
            font,
            self.pal.muted,
        );
        let knob = Rect::from_center_size(
            Pos2::new(rect.max.x - 52.0 * self.s, rect.center().y),
            Vec2::new(52.0 * self.s, 28.0 * self.s),
        );
        look::switch(&self.painter, knob, if on { 1.0 } else { 0.0 }, self.accent, self.pal);
        if hovered {
            self.ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        response.is_some_and(|response| response.clicked()) || accepted
    }
}

fn friend_game_name(friend: &Friend, titles: &[LibraryTitle]) -> String {
    if !friend.app_name.trim().is_empty() {
        return friend.app_name.clone();
    }
    titles
        .iter()
        .find(|title| title.title_id == friend.app_id)
        .map(|title| title.name.clone())
        .or_else(|| nexium_common::nextendo::compatible_title(friend.app_id).map(|title| title.name.to_string()))
        .unwrap_or_else(|| "a game".to_string())
}

fn age_text(seconds: i64) -> Option<String> {
    Some(match seconds.max(0) {
        0..=59 => "just now".to_string(),
        60..=3_599 => format!("{} min ago", seconds / 60),
        3_600..=86_399 => format!("{} h ago", seconds / 3_600),
        86_400..=604_799 => format!("{} d ago", seconds / 86_400),
        _ => return None,
    })
}

fn relative_time(stamp: &str) -> Option<String> {
    let then = chrono::DateTime::parse_from_rfc3339(stamp.trim())
        .ok()?
        .with_timezone(&chrono::Utc);
    let seconds = (chrono::Utc::now() - then).num_seconds();
    age_text(seconds).or_else(|| Some(then.format("%b %-d").to_string()))
}

fn title_name(title_id: u64, titles: &[LibraryTitle]) -> Option<String> {
    titles
        .iter()
        .find(|title| title.title_id == title_id)
        .map(|title| title.name.clone())
        .or_else(|| nexium_common::nextendo::compatible_title(title_id).map(|title| title.name.to_string()))
}

fn ease(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[allow(clippy::too_many_lines)]
pub fn hub_view(state: &mut HubState, frame: &HubFrame, ctx: &egui::Context, ui: &mut egui::Ui, full: Rect) -> HubAction {
    let now = ui.input(|keys| keys.time);
    let dt = ui.input(|keys| keys.stable_dt).min(0.1);
    let theme_target = if frame.light_mode { 1.0 } else { 0.0 };
    state.theme_t += (theme_target - state.theme_t) * (dt * 5.0).min(1.0);
    if (state.theme_t - theme_target).abs() < 0.002 {
        state.theme_t = theme_target;
    }
    let pal = look::palette(state.theme_t);
    let accent = look::brighten(frame.accent, 0.15);
    let base_s = (full.height() / 820.0).clamp(1.0, 2.4);
    let s = base_s * frame.zoom;
    let view = Rect::from_center_size(full.center(), full.size() * frame.zoom);

    if !state.servers_loaded {
        state.server.set(&frame.settings.server);
        state.nat.set(&frame.settings.nat);
        state.servers_loaded = true;
    }
    if !frame.state.signed_in() && matches!(state.tab, Tab::Friends | Tab::Requests | Tab::Players | Tab::AddFriend) {
        state.tab = Tab::Overview;
        state.focus_content = false;
    }

    let mut backdrop = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("nextendo_backdrop")));
    backdrop.set_opacity(frame.backdrop_opacity);
    crate::carousel::draw_backdrop(
        &backdrop,
        full,
        frame.ambient,
        now as f32,
        frame.backdrop,
        1.0,
        state.theme_t,
        None,
    );
    let scrim = if state.theme_t < 0.5 {
        Color32::from_black_alpha((150.0 * (1.0 - state.theme_t * 2.0)) as u8)
    } else {
        Color32::from_white_alpha((55.0 * (state.theme_t * 2.0 - 1.0)) as u8)
    };
    backdrop.rect_filled(full, CornerRadius::ZERO, scrim);

    let mut painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("nextendo_hub")));
    painter.set_opacity(frame.content_opacity);

    let typing = state.editing();
    let mut pad = if frame.active {
        read_pad(state, ui, frame.input, now, typing)
    } else {
        Pad::default()
    };
    let mut action = HubAction::None;
    let dialog_open = state.confirm.is_some();
    let dialog_pad = pad;
    if dialog_open {
        pad = Pad::default();
    }
    let interactive = frame.active && !dialog_open && !state.keyboard.open;

    let mx = view.width() * 0.055;
    let left = view.min.x + mx;
    let right = view.max.x - mx;
    let title_y = view.min.y + 40.0 * s;
    let title_size = 34.0 * s;
    let glyph_center = Pos2::new(left + 17.0 * s, title_y + title_size * 0.55);
    painter.circle_filled(glyph_center, 19.0 * s, look::alpha(accent, 0.16));
    look::network_glyph(&painter, glyph_center, 12.5 * s, accent);
    painter.text(
        Pos2::new(left + 48.0 * s, title_y),
        egui::Align2::LEFT_TOP,
        "Nextendo",
        FontId::proportional(title_size),
        pal.text,
    );
    let mut status_x = right;
    let status_y = title_y + title_size * 0.55;
    if let Some(latency) = frame.state.latency {
        let text = format!("{} ms", latency.as_millis());
        let r = painter.text(
            Pos2::new(status_x, status_y),
            egui::Align2::RIGHT_CENTER,
            text,
            FontId::proportional(14.0 * s),
            pal.muted,
        );
        status_x = r.min.x - 18.0 * s;
    }
    let (status_color, status_text) = match (frame.state.reachable, frame.state.players_online) {
        (Some(false), _) => (look::WARNING, "Nextendo is unreachable".to_string()),
        (_, Some(players)) => (
            look::ONLINE,
            format!("{} {} online", group(players), if players == 1 { "player" } else { "players" }),
        ),
        _ => (pal.faint, "Connecting…".to_string()),
    };
    let r = painter.text(
        Pos2::new(status_x, status_y),
        egui::Align2::RIGHT_CENTER,
        status_text,
        FontId::proportional(14.0 * s),
        pal.text,
    );
    painter.circle_filled(Pos2::new(r.min.x - 11.0 * s, status_y), 4.5 * s, status_color);

    let header_y = title_y + title_size + 20.0 * s;
    painter.hline(left..=right, header_y, Stroke::new(1.0, pal.border));
    let footer_y = view.max.y - 60.0 * s;
    painter.hline(left..=right, footer_y, Stroke::new(1.0, pal.border));

    let signed_in = frame.state.signed_in();
    let available_tabs: Vec<Tab> = TABS
        .iter()
        .copied()
        .filter(|tab| signed_in || matches!(tab, Tab::Overview | Tab::OnlinePlay))
        .collect();
    let tab_pos = available_tabs.iter().position(|tab| *tab == state.tab).unwrap_or(0);

    if pad.prev_tab && !available_tabs.is_empty() {
        state.tab = available_tabs[(tab_pos + available_tabs.len() - 1) % available_tabs.len()];
        state.focus_content = false;
        crate::ui_audio::play_move();
    }
    if pad.next_tab && !available_tabs.is_empty() {
        state.tab = available_tabs[(tab_pos + 1) % available_tabs.len()];
        state.focus_content = false;
        crate::ui_audio::play_move();
    }
    if pad.refresh && signed_in {
        action = HubAction::Refresh;
    }

    let mut content_accept = false;
    if !state.focus_content {
        let tab_pos = available_tabs.iter().position(|tab| *tab == state.tab).unwrap_or(0);
        if pad.up && tab_pos > 0 {
            state.tab = available_tabs[tab_pos - 1];
            crate::ui_audio::play_move();
        }
        if pad.down && tab_pos + 1 < available_tabs.len() {
            state.tab = available_tabs[tab_pos + 1];
            crate::ui_audio::play_move();
        }
        if pad.accept || pad.right {
            state.focus_content = true;
            state.row = 0;
            state.col = 0;
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        }
        if pad.back {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            action = HubAction::Close;
        }
    } else {
        if pad.up && state.row > 0 {
            state.row -= 1;
            state.col = 0;
            crate::ui_audio::play_move();
        }
        if pad.down {
            state.row += 1;
            state.col = 0;
            crate::ui_audio::play_move();
        }
        if pad.right {
            state.col += 1;
            crate::ui_audio::play_move();
        }
        if pad.left {
            if state.col == 0 {
                state.focus_content = false;
            } else {
                state.col -= 1;
            }
            crate::ui_audio::play_move();
        }
        if pad.back {
            state.focus_content = false;
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
        }
        content_accept = pad.accept;
    }

    let side_w = (view.width() * 0.22).max(300.0 * frame.zoom);
    let side_top = header_y + 34.0 * s;
    let item_h = 64.0 * s;
    for (index, tab) in TABS.iter().enumerate() {
        let enabled = available_tabs.contains(tab);
        let rect = Rect::from_min_size(
            Pos2::new(left, side_top + index as f32 * (item_h + 10.0 * s)),
            Vec2::new(side_w, item_h),
        );
        let response = ui.allocate_rect(rect, if enabled && interactive { Sense::click() } else { Sense::hover() });
        if enabled && response.clicked() {
            state.tab = *tab;
            state.focus_content = false;
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        }
        let selected = state.tab == *tab;
        let ring = if !state.focus_content { accent } else { pal.border };
        let rounding = CornerRadius::from(12.0 * s);
        if selected {
            painter.rect_filled(rect, rounding, pal.selected);
            painter.rect_stroke(rect, rounding, Stroke::new(1.8, ring), egui::StrokeKind::Outside);
            painter.rect_filled(
                Rect::from_min_size(rect.min + Vec2::new(6.0 * s, 12.0 * s), Vec2::new(4.0 * s, rect.height() - 24.0 * s)),
                CornerRadius::from(2.0 * s),
                ring,
            );
        } else if enabled && response.hovered() {
            painter.rect_filled(rect, rounding, pal.panel_hover);
        }
        let color = if !enabled {
            look::alpha(pal.muted, 0.45)
        } else if selected {
            pal.text
        } else {
            pal.muted
        };
        painter.text(
            Pos2::new(rect.min.x + 26.0 * s, rect.center().y),
            egui::Align2::LEFT_CENTER,
            tab.label(),
            FontId::proportional(19.0 * s),
            color,
        );
        let badge_x = rect.max.x - 26.0 * s;
        match tab {
            Tab::Friends if signed_in && frame.state.friends_online() > 0 => {
                let text = format!("{} online", frame.state.friends_online());
                painter.text(
                    Pos2::new(badge_x, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    text,
                    FontId::proportional(13.0 * s),
                    look::ONLINE,
                );
            }
            Tab::Players if signed_in && frame.state.lobby.is_some() => {
                painter.text(
                    Pos2::new(badge_x, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    "In a lobby",
                    FontId::proportional(13.0 * s),
                    look::PLAYING,
                );
            }
            Tab::Requests if signed_in && !frame.state.requests.is_empty() => {
                look::counter_badge(&painter, Pos2::new(badge_x - 8.0 * s, rect.center().y), 22.0 * s, frame.state.requests.len());
            }
            _ => {}
        }
    }

    let content_full = Rect::from_min_max(
        Pos2::new(left + side_w + 48.0 * s, side_top),
        Pos2::new(right, footer_y - 20.0 * s),
    );
    if state.tab != state.shown_tab {
        state.tab_anim = 0.0;
        state.shown_tab = state.tab;
        state.row = 0;
        state.col = 0;
        state.stop_editing();
    }
    state.tab_anim = (state.tab_anim + dt * 9.0).min(1.0);
    let eased = ease(state.tab_anim);
    let content = content_full.translate(Vec2::new((1.0 - eased) * 46.0 * s, 0.0));

    let mut content_painter = ctx
        .layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("nextendo_hub_content")))
        .with_clip_rect(content_full.expand2(Vec2::new(8.0 * s, 6.0 * s)));
    content_painter.set_opacity(eased * frame.content_opacity);

    let tab_index = state.tab.index();
    if interactive && ui.rect_contains_pointer(content_full) {
        let wheel = ui.input(|keys| keys.smooth_scroll_delta.y);
        state.scroll[tab_index] = (state.scroll[tab_index] - wheel).max(0.0);
    }
    let scroll = state.scroll[tab_index];

    let mut page = Page {
        ui,
        painter: content_painter,
        frame,
        pal,
        accent,
        s,
        time: now,
        nav: Nav {
            rows: Vec::new(),
            row: state.row,
            col: state.col,
            accept: content_accept && interactive,
            content: state.focus_content && !dialog_open,
            clip: content_full,
        },
        action: HubAction::None,
        enabled: interactive,
    };
    let origin = content.min - Vec2::new(0.0, scroll);
    let content_height = match state.tab {
        Tab::Overview => overview_page(&mut page, state, origin, content.width()),
        Tab::Friends => friends_page(&mut page, state, origin, content.width()),
        Tab::Requests => requests_page(&mut page, state, origin, content.width()),
        Tab::Players => players_page(&mut page, state, origin, content.width()),
        Tab::AddFriend => add_friend_page(&mut page, state, origin, content.width()),
        Tab::OnlinePlay => online_play_page(&mut page, state, origin, content.width()),
    };
    let rows = std::mem::take(&mut page.nav.rows);
    let page_action = std::mem::replace(&mut page.action, HubAction::None);
    let ui = page.ui;
    let max_scroll = (content_height - content_full.height()).max(0.0);
    state.scroll[tab_index] = state.scroll[tab_index].min(max_scroll);
    if page_action != HubAction::None {
        action = page_action;
    }

    if state.focus_content {
        if rows.iter().all(|row| row.is_empty()) {
            state.focus_content = false;
        } else {
            state.row = state.row.min(rows.len() - 1);
            let downward = pad.down && !pad.up;
            while rows[state.row].is_empty() {
                if downward && state.row + 1 < rows.len() {
                    state.row += 1;
                } else if state.row > 0 {
                    state.row -= 1;
                } else {
                    state.row = rows.iter().position(|row| !row.is_empty()).unwrap_or(0);
                    break;
                }
            }
            let cols = rows[state.row].len();
            state.col = state.col.min(cols.saturating_sub(1));
            if let Some(focus) = rows.get(state.row).and_then(|row| row.get(state.col)) {
                let margin = 16.0 * s;
                if focus.min.y < content_full.min.y + margin {
                    state.scroll[tab_index] = (state.scroll[tab_index] - (content_full.min.y + margin - focus.min.y)).max(0.0);
                } else if focus.max.y > content_full.max.y - margin {
                    state.scroll[tab_index] = (state.scroll[tab_index] + (focus.max.y - (content_full.max.y - margin))).min(max_scroll);
                }
            }
        }
    }

    if max_scroll > 0.0 {
        let track = Rect::from_min_max(
            Pos2::new(content_full.max.x + 10.0 * s, content_full.min.y),
            Pos2::new(content_full.max.x + 13.0 * s, content_full.max.y),
        );
        let ratio = content_full.height() / content_height.max(1.0);
        let thumb_h = (track.height() * ratio).max(30.0 * s);
        let thumb_y = track.min.y + (track.height() - thumb_h) * (state.scroll[tab_index] / max_scroll);
        painter.rect_filled(track, CornerRadius::from(2.0 * s), look::alpha(pal.border, 0.5));
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(track.min.x, thumb_y), Vec2::new(track.width(), thumb_h)),
            CornerRadius::from(2.0 * s),
            pal.muted,
        );
    }

    if state.keyboard.active() {
        let field = state.keyboard_field;
        let buffer = match field {
            Some(Field::Server) => &mut state.server.text,
            Some(Field::Nat) => &mut state.nat.text,
            _ => &mut state.code.text,
        };
        let result = state.keyboard.update(ctx, ui, buffer, frame.input, accent, frame.light_mode);
        if let crate::vkeyboard::VkResult::Accept = result {
            if field == Some(Field::FriendCode) {
                let code = state.code.text.clone();
                if !code.trim().is_empty() {
                    state.awaiting_add = true;
                    action = HubAction::AddFriend(code);
                }
            }
            state.keyboard_field = None;
        } else if let crate::vkeyboard::VkResult::Cancel = result {
            state.keyboard_field = None;
        }
    }

    let hint = if frame.input.connected {
        if state.focus_content {
            "🎮  [↑/↓] Move      [A] Select      [Y] Refresh      [L/R] Switch tab      [B] Back"
        } else {
            "🎮  [↑/↓] Choose a section      [A] Open      [L/R] Switch tab      [B] Close"
        }
    } else if typing {
        "⌨  [Enter] Confirm      [Esc] Cancel"
    } else {
        "⌨  [Esc] Back      [F5] Refresh      Click a section      Scroll wheel"
    };
    painter.text(
        Pos2::new(left, view.max.y - 30.0 * s),
        egui::Align2::LEFT_CENTER,
        hint,
        FontId::proportional(14.0 * s),
        pal.muted,
    );
    if let Some(account) = frame.state.account.as_ref().filter(|_| signed_in) {
        painter.text(
            Pos2::new(right, view.max.y - 30.0 * s),
            egui::Align2::RIGHT_CENTER,
            format!("Signed in as {}", account.username),
            FontId::proportional(14.0 * s),
            pal.muted,
        );
    }

    if dialog_open {
        if let Some(chosen) = confirm_dialog(state, ctx, ui, full, s, pal, accent, dialog_pad) {
            action = chosen;
        }
    }

    if eased < 1.0 || frame.content_opacity < 1.0 {
        ctx.request_repaint();
    }
    if frame.state.phase != Phase::SignedIn && frame.state.phase != Phase::SignedOut {
        ctx.request_repaint();
    }
    if action == HubAction::Close {
        state.stop_editing();
    }
    action
}

#[allow(clippy::too_many_arguments)]
fn confirm_dialog(
    state: &mut HubState,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    full: Rect,
    s: f32,
    pal: Palette,
    accent: Color32,
    pad: Pad,
) -> Option<HubAction> {
    let confirm = state.confirm.clone()?;
    let (title, body, label) = match &confirm {
        Confirm::SignOut => (
            "Sign out of Nextendo?".to_string(),
            "Online play, friends and presence stop until you sign in again. Your games and saves stay on this PC.".to_string(),
            "Sign out",
        ),
        Confirm::Remove { name, .. } => (
            format!("Remove {name}?"),
            "They'll be removed from your friend list on Nextendo. You can send a new request later.".to_string(),
            "Remove",
        ),
    };
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("nextendo_confirm")));
    painter.rect_filled(full, CornerRadius::ZERO, Color32::from_black_alpha(150));
    let card = Rect::from_center_size(full.center(), Vec2::new(500.0 * s, 230.0 * s));
    look::soft_shadow(&painter, card, 16.0 * s, 1.0);
    look::card(&painter, card, 16.0 * s, pal.panel, pal.border, 1.0);
    painter.text(
        card.min + Vec2::new(30.0 * s, 34.0 * s),
        egui::Align2::LEFT_CENTER,
        title,
        FontId::proportional(22.0 * s),
        pal.text,
    );
    let galley = painter.layout(body, FontId::proportional(14.5 * s), pal.muted, card.width() - 60.0 * s);
    painter.galley(card.min + Vec2::new(30.0 * s, 62.0 * s), galley, pal.muted);
    if pad.left {
        state.confirm_choice = 0;
    }
    if pad.right {
        state.confirm_choice = 1;
    }
    let button_w = 150.0 * s;
    let button_h = 44.0 * s;
    let cancel = Rect::from_min_size(
        Pos2::new(card.max.x - 30.0 * s - button_w * 2.0 - 12.0 * s, card.max.y - 30.0 * s - button_h),
        Vec2::new(button_w, button_h),
    );
    let confirm_rect = Rect::from_min_size(
        Pos2::new(card.max.x - 30.0 * s - button_w, card.max.y - 30.0 * s - button_h),
        Vec2::new(button_w, button_h),
    );
    let cancel_clicked = look::button(
        ui,
        &painter,
        cancel,
        "Cancel",
        ButtonKind::Secondary,
        accent,
        pal,
        state.confirm_choice == 0,
        true,
        15.0 * s,
    )
    .clicked();
    let confirm_clicked = look::button(
        ui,
        &painter,
        confirm_rect,
        label,
        ButtonKind::Danger,
        accent,
        pal,
        state.confirm_choice == 1,
        true,
        15.0 * s,
    )
    .clicked();
    let accepted = pad.accept && state.confirm_choice == 1;
    if cancel_clicked || pad.back || (pad.accept && state.confirm_choice == 0) {
        state.confirm = None;
        crate::ui_audio::play(crate::ui_audio::Sfx::Back);
        return None;
    }
    if confirm_clicked || accepted {
        state.confirm = None;
        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
        return Some(match confirm {
            Confirm::SignOut => HubAction::SignOut,
            Confirm::Remove { pid, .. } => HubAction::Remove(pid),
        });
    }
    None
}

fn overview_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let mut y = origin.y;
    if let Some(error) = page.frame.state.sign_in_error.clone().filter(|_| !page.frame.state.signed_in()) {
        page.banner(
            Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 48.0 * s)),
            look::DANGER,
            &error,
        );
        y += 62.0 * s;
    }
    match page.frame.state.phase.clone() {
        Phase::SignedOut => {
            y = signed_out_hero(page, origin.x, y, width);
        }
        Phase::Browser { url } => {
            y = waiting_card(
                page,
                origin.x,
                y,
                width,
                "Finish signing in with your browser",
                "Approve NeXium on the Nextendo page that just opened. This screen updates by itself once you're done.",
                Some(url),
            );
        }
        Phase::Finishing => {
            y = waiting_card(
                page,
                origin.x,
                y,
                width,
                "Signing you in…",
                "Exchanging your one-time code for a secure session.",
                None,
            );
        }
        Phase::Restoring => {
            y = waiting_card(
                page,
                origin.x,
                y,
                width,
                "Reconnecting to Nextendo…",
                "Restoring your saved session.",
                None,
            );
        }
        Phase::SignedIn => {
            y = account_overview(page, state, origin.x, y, width);
        }
    }
    y - origin.y
}

fn signed_out_hero(page: &mut Page, x: f32, mut y: f32, width: f32) -> f32 {
    let s = page.s;
    let hero = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, 248.0 * s));
    look::card(&page.painter, hero, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    let glyph_center = Pos2::new(hero.min.x + 70.0 * s, hero.min.y + 72.0 * s);
    page.painter.circle_filled(glyph_center, 40.0 * s, look::alpha(page.accent, 0.14));
    page.painter.circle_stroke(glyph_center, 40.0 * s, Stroke::new(1.5, look::alpha(page.accent, 0.6)));
    look::network_glyph(&page.painter, glyph_center, 22.0 * s, page.accent);
    let text_x = hero.min.x + 136.0 * s;
    let text_w = hero.max.x - text_x - 30.0 * s;
    page.text(
        Pos2::new(text_x, hero.min.y + 50.0 * s),
        egui::Align2::LEFT_CENTER,
        "Play online with Nextendo Network",
        24.0,
        page.pal.text,
    );
    page.paragraph(
        Pos2::new(text_x, hero.min.y + 72.0 * s),
        "Sign in with your Nextendo account to play supported games online, add friends and let them see what you're playing.",
        15.0,
        page.pal.muted,
        text_w,
    );
    let buttons_y = hero.min.y + 138.0 * s;
    let sign_in = Rect::from_min_size(Pos2::new(text_x, buttons_y), Vec2::new(250.0 * s, 48.0 * s));
    if page.button(0, sign_in, "Sign in with Nextendo", ButtonKind::Primary, true) {
        page.action = HubAction::SignIn;
    }
    let create = Rect::from_min_size(
        Pos2::new(sign_in.max.x + 14.0 * s, buttons_y),
        Vec2::new(200.0 * s, 48.0 * s),
    );
    if page.button(0, create, "Create an account", ButtonKind::Secondary, true) {
        page.action = HubAction::OpenUrl(ACCOUNT_URL.into());
    }
    page.text(
        Pos2::new(text_x, buttons_y + 70.0 * s),
        egui::Align2::LEFT_CENTER,
        "You sign in on nextendo.network in your browser. NeXium never sees your password.",
        12.5,
        page.pal.faint,
    );
    y = hero.max.y + 18.0 * s;
    let gap = 16.0 * s;
    let tile_w = (width - gap * 2.0) / 3.0;
    let tiles: [(&str, &str, u8); 3] = [
        ("Online play", "Supported games reach Nextendo's servers automatically. No hosts files, no DNS changes.", 0),
        ("Friends", "Add friends by code, accept requests and see who's online right now.", 1),
        ("Presence", "Friends see when you're online and which game you're playing.", 2),
    ];
    for (index, (title, body, icon)) in tiles.iter().enumerate() {
        let tile = Rect::from_min_size(
            Pos2::new(x + index as f32 * (tile_w + gap), y),
            Vec2::new(tile_w, 148.0 * s),
        );
        look::card(&page.painter, tile, 14.0 * s, page.pal.panel, page.pal.border, 1.0);
        let icon_center = Pos2::new(tile.min.x + 34.0 * s, tile.min.y + 36.0 * s);
        page.painter.circle_filled(icon_center, 17.0 * s, look::alpha(page.accent, 0.13));
        match icon {
            0 => look::network_glyph(&page.painter, icon_center, 10.0 * s, page.accent),
            1 => look::person_glyph(&page.painter, icon_center, 12.0 * s, page.accent),
            _ => {
                page.painter.circle_filled(icon_center, 5.0 * s, look::ONLINE);
                page.painter.circle_stroke(icon_center, 9.5 * s, Stroke::new(1.6 * s, look::alpha(look::ONLINE, 0.6)));
            }
        }
        page.text(
            Pos2::new(tile.min.x + 62.0 * s, tile.min.y + 36.0 * s),
            egui::Align2::LEFT_CENTER,
            title,
            17.0,
            page.pal.text,
        );
        page.paragraph(
            Pos2::new(tile.min.x + 20.0 * s, tile.min.y + 64.0 * s),
            body,
            13.5,
            page.pal.muted,
            tile_w - 40.0 * s,
        );
    }
    y + 148.0 * s + 12.0 * s
}

fn waiting_card(page: &mut Page, x: f32, y: f32, width: f32, title: &str, body: &str, url: Option<String>) -> f32 {
    let s = page.s;
    let height = if url.is_some() { 236.0 } else { 150.0 } * s;
    let card = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, height));
    look::card(&page.painter, card, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    let spinner_center = Pos2::new(card.min.x + 62.0 * s, card.min.y + 66.0 * s);
    page.painter.circle_stroke(spinner_center, 26.0 * s, Stroke::new(3.0 * s, look::alpha(page.accent, 0.18)));
    look::spinner(&page.painter, spinner_center, 26.0 * s, page.time, page.accent);
    let text_x = card.min.x + 120.0 * s;
    page.text(
        Pos2::new(text_x, card.min.y + 52.0 * s),
        egui::Align2::LEFT_CENTER,
        title,
        22.0,
        page.pal.text,
    );
    page.paragraph(
        Pos2::new(text_x, card.min.y + 72.0 * s),
        body,
        14.5,
        page.pal.muted,
        card.max.x - text_x - 30.0 * s,
    );
    if let Some(url) = url {
        let buttons_y = card.min.y + 150.0 * s;
        let reopen = Rect::from_min_size(Pos2::new(text_x, buttons_y), Vec2::new(220.0 * s, 46.0 * s));
        if page.button(0, reopen, "Open the page again", ButtonKind::Primary, !url.is_empty()) {
            page.action = HubAction::ReopenBrowser;
        }
        let copy = Rect::from_min_size(Pos2::new(reopen.max.x + 12.0 * s, buttons_y), Vec2::new(150.0 * s, 46.0 * s));
        let copied = page
            .ui
            .memory(|memory| memory.data.get_temp::<f64>(egui::Id::new("nextendo_link_copied")))
            .is_some_and(|at| page.time - at < 1.6);
        if page.button(0, copy, if copied { "Copied" } else { "Copy link" }, ButtonKind::Secondary, !url.is_empty()) {
            page.ui.ctx().copy_text(url.clone());
            let now = page.time;
            page.ui.memory_mut(|memory| memory.data.insert_temp(egui::Id::new("nextendo_link_copied"), now));
        }
        let cancel = Rect::from_min_size(Pos2::new(copy.max.x + 12.0 * s, buttons_y), Vec2::new(120.0 * s, 46.0 * s));
        if page.button(0, cancel, "Cancel", ButtonKind::Quiet, true) {
            page.action = HubAction::CancelSignIn;
        }
    }
    card.max.y + 12.0 * s
}

fn account_overview(page: &mut Page, state: &mut HubState, x: f32, y: f32, width: f32) -> f32 {
    let s = page.s;
    let snapshot = page.frame.state;
    let Some(account) = snapshot.account.clone() else {
        return y;
    };
    let card = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, 140.0 * s));
    look::card(&page.painter, card, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    let avatar_r = 46.0 * s;
    let avatar_center = Pos2::new(card.min.x + 30.0 * s + avatar_r, card.center().y);
    let ring = if snapshot.offline { look::WARNING } else { look::ONLINE };
    page.painter.circle_filled(avatar_center, avatar_r + 5.0 * s, page.pal.panel);
    page.avatar(avatar_center, avatar_r, account.pid, &account.username, &mut state.avatars);
    page.painter.circle_stroke(avatar_center, avatar_r + 3.0 * s, Stroke::new(3.0 * s, ring));
    let name_x = avatar_center.x + avatar_r + 26.0 * s;
    page.text(
        Pos2::new(name_x, card.center().y - 20.0 * s),
        egui::Align2::LEFT_CENTER,
        &account.username,
        26.0,
        page.pal.text,
    );
    let code = if account.friend_code.is_empty() { "No friend code yet".to_string() } else { account.friend_code.clone() };
    let code_rect = page.text(
        Pos2::new(name_x, card.center().y + 16.0 * s),
        egui::Align2::LEFT_CENTER,
        &code,
        16.0,
        page.pal.muted,
    );
    let copied = state.copied_at.is_some_and(|at| page.time - at < 1.6);
    if !account.friend_code.is_empty() {
        let copy = Rect::from_center_size(
            Pos2::new(code_rect.max.x + 22.0 * s, code_rect.center().y),
            Vec2::splat(30.0 * s),
        );
        if page.icon_button(0, copy, look::copy_glyph, "Copy friend code") {
            page.ui.ctx().copy_text(account.friend_code.clone());
            state.copied_at = Some(page.time);
        }
        if copied {
            page.text(
                Pos2::new(copy.max.x + 8.0 * s, copy.center().y),
                egui::Align2::LEFT_CENTER,
                "Copied",
                13.0,
                look::ONLINE,
            );
        }
    }
    let sign_out = Rect::from_min_size(
        Pos2::new(card.max.x - 30.0 * s - 150.0 * s, card.center().y - 22.0 * s),
        Vec2::new(150.0 * s, 44.0 * s),
    );
    if page.button(0, sign_out, "Sign out", ButtonKind::Danger, true) {
        state.confirm = Some(Confirm::SignOut);
        state.confirm_choice = 0;
    }

    let mut row_y = card.max.y + 18.0 * s;
    let (online_value, online_color, online_detail) = if !page.frame.settings.online_play {
        ("Turned off".to_string(), page.pal.muted, "Turn it on in Online Play to use Nextendo's servers.".to_string())
    } else {
        match &snapshot.game_access {
            GameAccess::Ready => (
                "Ready".to_string(),
                look::ONLINE,
                "Supported games connect to Nextendo automatically.".to_string(),
            ),
            GameAccess::Missing(reason) => ("Unavailable".to_string(), look::WARNING, reason.clone()),
            GameAccess::Unknown => ("Checking…".to_string(), page.pal.muted, "Preparing your game sign-in.".to_string()),
        }
    };
    let playing = page
        .frame
        .running_title
        .map(|title_id| {
            page.frame
                .titles
                .iter()
                .find(|title| title.title_id == title_id)
                .map(|title| title.name.clone())
                .or_else(|| nexium_common::nextendo::compatible_title(title_id).map(|title| title.name.to_string()))
                .unwrap_or_else(|| "a game".into())
        });
    let (status_value, status_color, status_detail) = if !page.frame.settings.share_presence {
        ("Hidden".to_string(), page.pal.muted, "Friends see you as offline.".to_string())
    } else if let Some(game) = playing {
        (format!("Playing {game}"), look::PLAYING, "Friends can see the game you're in.".to_string())
    } else {
        ("Online".to_string(), look::ONLINE, "Friends can see that you're online.".to_string())
    };
    let (standing_value, standing_color, standing_detail) = match &snapshot.standing {
        Some(standing) if standing.allow => ("In good standing".to_string(), look::ONLINE, "Your account can join online sessions.".to_string()),
        Some(standing) => (
            "Action needed".to_string(),
            look::WARNING,
            if standing.message.is_empty() { standing.reason.clone() } else { standing.message.clone() },
        ),
        None => ("—".to_string(), page.pal.muted, "Account status hasn't been checked yet.".to_string()),
    };
    let (network_value, network_color, network_detail) = if snapshot.offline {
        ("Offline".to_string(), look::WARNING, "NeXium keeps retrying in the background.".to_string())
    } else {
        (
            snapshot
                .latency
                .map(|latency| format!("Connected · {} ms", latency.as_millis()))
                .unwrap_or_else(|| "Connected".into()),
            look::ONLINE,
            "Your session refreshes automatically.".to_string(),
        )
    };
    let mut rows = vec![
        ("Online play", online_value, online_color, online_detail, true),
        ("Your status", status_value, status_color, status_detail, true),
    ];
    if snapshot.standing.is_some() {
        rows.push(("Account", standing_value, standing_color, standing_detail, false));
    }
    rows.push(("Connection", network_value, network_color, network_detail, false));
    let mut nav_row = 1;
    for (label, value, color, detail, link) in rows {
        let rect = Rect::from_min_size(Pos2::new(x, row_y), Vec2::new(width, 64.0 * s));
        let (focused, accepted) = if link {
            nav_row += 1;
            page.nav.slot(nav_row - 1, rect)
        } else {
            (false, false)
        };
        let hovered = link && page.row_hover(rect);
        let clicked = link && {
            let hit = page.interactive(rect);
            hit.is_positive() && page.ui.allocate_rect(hit, Sense::click()).clicked()
        };
        look::card(
            &page.painter,
            rect,
            12.0 * s,
            if hovered { page.pal.panel_hover } else { page.pal.panel },
            if focused { page.accent } else { page.pal.border },
            if focused { 1.8 } else { 1.0 },
        );
        page.text(
            Pos2::new(rect.min.x + 24.0 * s, rect.center().y - 10.0 * s),
            egui::Align2::LEFT_CENTER,
            label,
            17.0,
            page.pal.text,
        );
        let font = FontId::proportional(13.0 * s);
        let detail = look::ellipsize(page.ui, &detail, &font, rect.width() * 0.55);
        page.painter.text(
            Pos2::new(rect.min.x + 24.0 * s, rect.center().y + 13.0 * s),
            egui::Align2::LEFT_CENTER,
            detail,
            font,
            page.pal.muted,
        );
        let value_right = rect.max.x - if link { 50.0 } else { 26.0 } * s;
        let value_font = FontId::proportional(16.0 * s);
        let value = look::ellipsize(page.ui, &value, &value_font, rect.width() * 0.36);
        let value_rect = page.painter.text(
            Pos2::new(value_right, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            value,
            value_font,
            page.pal.text,
        );
        page.painter.circle_filled(Pos2::new(value_rect.min.x - 12.0 * s, rect.center().y), 4.5 * s, color);
        if link {
            page.text(
                Pos2::new(rect.max.x - 26.0 * s, rect.center().y),
                egui::Align2::CENTER_CENTER,
                "›",
                24.0,
                if focused || hovered { page.accent } else { page.pal.muted },
            );
            if clicked || accepted {
                state.tab = Tab::OnlinePlay;
                state.focus_content = true;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        }
        row_y = rect.max.y + 10.0 * s;
    }
    let buttons_y = row_y + 8.0 * s;
    let refresh = Rect::from_min_size(Pos2::new(x, buttons_y), Vec2::new(150.0 * s, 44.0 * s));
    if page.button(nav_row, refresh, "Refresh", ButtonKind::Secondary, true) {
        page.action = HubAction::Refresh;
    }
    let manage = Rect::from_min_size(Pos2::new(refresh.max.x + 12.0 * s, buttons_y), Vec2::new(250.0 * s, 44.0 * s));
    if page.button(nav_row, manage, "Manage account online  ↗", ButtonKind::Secondary, true) {
        page.action = HubAction::OpenUrl(ACCOUNT_URL.into());
    }
    buttons_y + 44.0 * s + 12.0 * s
}

fn empty_state(page: &mut Page, x: f32, y: f32, width: f32, title: &str, body: &str, person: bool) -> f32 {
    let s = page.s;
    let card = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, 210.0 * s));
    look::card(&page.painter, card, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    let icon_center = Pos2::new(card.center().x, card.min.y + 64.0 * s);
    page.painter.circle_filled(icon_center, 32.0 * s, look::alpha(page.accent, 0.12));
    if person {
        look::person_glyph(&page.painter, icon_center, 22.0 * s, page.accent);
    } else {
        look::check_glyph(&page.painter, icon_center, 26.0 * s, page.accent);
    }
    page.text(
        Pos2::new(card.center().x, card.min.y + 122.0 * s),
        egui::Align2::CENTER_CENTER,
        title,
        19.0,
        page.pal.text,
    );
    let font = FontId::proportional(14.0 * s);
    let limit = (width * 0.7).max(1.0);
    let natural = look::text_width(page.ui, body, &font);
    let lines = (natural / limit).ceil().max(1.0);
    let wrap = (natural / lines + 24.0 * s).min(limit);
    let mut job = egui::text::LayoutJob::simple(body.to_string(), font, page.pal.muted, wrap);
    job.halign = egui::Align::Center;
    let galley = page.painter.layout_job(job);
    page.painter.galley(
        Pos2::new(card.center().x, card.min.y + 140.0 * s),
        galley,
        page.pal.muted,
    );
    card.max.y + 12.0 * s
}

fn friends_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let snapshot = page.frame.state;
    let mut y = origin.y;
    let online = snapshot.friends_online();
    let summary = if snapshot.friends.is_empty() {
        "Your friends".to_string()
    } else {
        format!(
            "{} online  ·  {} {}",
            online,
            snapshot.friends.len(),
            if snapshot.friends.len() == 1 { "friend" } else { "friends" }
        )
    };
    page.text(Pos2::new(origin.x, y + 18.0 * s), egui::Align2::LEFT_CENTER, &summary, 18.0, page.pal.text);
    let refresh = Rect::from_min_size(Pos2::new(origin.x + width - 36.0 * s, y), Vec2::splat(36.0 * s));
    if page.icon_button(0, refresh, look::refresh_glyph, "Refresh") {
        page.action = HubAction::Refresh;
    }
    if let Some(last) = snapshot.last_sync {
        let ago = last.elapsed().as_secs();
        let text = if ago < 5 { "Updated just now".to_string() } else if ago < 60 { format!("Updated {ago}s ago") } else { format!("Updated {}m ago", ago / 60) };
        page.text(
            Pos2::new(refresh.min.x - 10.0 * s, refresh.center().y),
            egui::Align2::RIGHT_CENTER,
            &text,
            13.0,
            page.pal.faint,
        );
    }
    y += 52.0 * s;
    if let Some(issue) = snapshot.friends_issue.clone() {
        page.banner(Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 46.0 * s)), look::WARNING, &issue);
        y += 58.0 * s;
    }
    if !snapshot.friends_loaded {
        let center = Pos2::new(origin.x + width * 0.5, y + 60.0 * s);
        look::spinner(&page.painter, center, 20.0 * s, page.time, page.accent);
        page.text(Pos2::new(center.x, center.y + 42.0 * s), egui::Align2::CENTER_CENTER, "Loading your friends…", 15.0, page.pal.muted);
        return y + 120.0 * s - origin.y;
    }
    if snapshot.friends.is_empty() {
        y = empty_state(
            page,
            origin.x,
            y,
            width,
            "No friends yet",
            "Share your friend code, or send a request from Add Friend. Requests you receive show up under Friend Requests.",
            true,
        );
        let add = Rect::from_min_size(Pos2::new(origin.x + width * 0.5 - 100.0 * s, y + 4.0 * s), Vec2::new(200.0 * s, 44.0 * s));
        if page.button(1, add, "Add a friend", ButtonKind::Primary, true) {
            state.tab = Tab::AddFriend;
            state.focus_content = true;
        }
        return add.max.y + 12.0 * s - origin.y;
    }
    for (index, friend) in snapshot.friends.iter().enumerate() {
        let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 70.0 * s));
        friend_row(page, state, rect, friend, index + 1);
        y = rect.max.y + 8.0 * s;
    }
    y - origin.y
}

fn friend_row(page: &mut Page, state: &mut HubState, rect: Rect, friend: &Friend, row: usize) {
    let s = page.s;
    let hovered = page.row_hover(rect);
    let remove = Rect::from_min_size(
        Pos2::new(rect.max.x - 20.0 * s - 110.0 * s, rect.center().y - 19.0 * s),
        Vec2::new(110.0 * s, 38.0 * s),
    );
    let focused_row = page.nav.content && page.nav.row == row;
    look::card(
        &page.painter,
        rect,
        12.0 * s,
        if hovered || focused_row { page.pal.panel_hover } else { page.pal.panel },
        if focused_row { look::alpha(page.accent, 0.6) } else { page.pal.border },
        1.0,
    );
    let avatar_r = 22.0 * s;
    let avatar_center = Pos2::new(rect.min.x + 20.0 * s + avatar_r, rect.center().y);
    page.avatar(avatar_center, avatar_r, friend.pid, &friend.name, &mut state.avatars);
    let (dot, line, line_color) = if friend.in_game() {
        (look::PLAYING, format!("Playing {}", friend_game_name(friend, page.frame.titles)), look::PLAYING)
    } else if friend.online() {
        (look::ONLINE, "Online".to_string(), look::ONLINE)
    } else {
        let seen = friend
            .last_seen
            .as_deref()
            .and_then(relative_time)
            .map(|when| format!("Offline  ·  last online {when}"))
            .unwrap_or_else(|| "Offline".to_string());
        (page.pal.faint, seen, page.pal.muted)
    };
    look::status_dot(
        &page.painter,
        avatar_center + Vec2::new(avatar_r * 0.72, avatar_r * 0.72),
        5.0 * s,
        dot,
        if hovered || focused_row { page.pal.panel_hover } else { page.pal.panel },
    );
    let name_x = avatar_center.x + avatar_r + 18.0 * s;
    let mut name_rect = page.text(
        Pos2::new(name_x, rect.center().y - 11.0 * s),
        egui::Align2::LEFT_CENTER,
        if friend.name.is_empty() { "Unnamed player" } else { &friend.name },
        18.0,
        if friend.online() { page.pal.text } else { page.pal.muted },
    );
    if friend.favorite {
        name_rect = name_rect.union(page.text(
            Pos2::new(name_rect.max.x + 8.0 * s, name_rect.center().y),
            egui::Align2::LEFT_CENTER,
            "★",
            15.0,
            Color32::from_rgb(0xF5, 0xC1, 0x42),
        ));
    }
    if friend.staff {
        let tag_font = FontId::proportional(10.5 * s);
        let tag_w = look::text_width(page.ui, "TEAM", &tag_font) + 14.0 * s;
        let tag = Rect::from_min_size(Pos2::new(name_rect.max.x + 10.0 * s, name_rect.center().y - 9.0 * s), Vec2::new(tag_w, 18.0 * s));
        page.painter.rect_filled(tag, CornerRadius::from(9.0 * s), look::alpha(page.accent, 0.2));
        page.painter.text(tag.center(), egui::Align2::CENTER_CENTER, "TEAM", tag_font, look::brighten(page.accent, 0.3));
        name_rect = name_rect.union(tag);
    }
    if friend.console {
        let tag_font = FontId::proportional(11.0 * s);
        let tag_w = look::text_width(page.ui, "Switch", &tag_font) + 14.0 * s;
        let tag = Rect::from_min_size(Pos2::new(name_rect.max.x + 10.0 * s, name_rect.center().y - 9.0 * s), Vec2::new(tag_w, 18.0 * s));
        page.painter.rect_stroke(tag, CornerRadius::from(9.0 * s), Stroke::new(1.0, page.pal.border), egui::StrokeKind::Inside);
        page.painter.text(tag.center(), egui::Align2::CENTER_CENTER, "Switch", tag_font, page.pal.muted);
    }
    let line_font = FontId::proportional(13.5 * s);
    let line_space = remove.min.x - name_x - 70.0 * s;
    let line = look::ellipsize(page.ui, &line, &line_font, line_space.max(60.0 * s));
    page.painter.text(
        Pos2::new(name_x, rect.center().y + 13.0 * s),
        egui::Align2::LEFT_CENTER,
        line,
        line_font,
        line_color,
    );
    if friend.in_game() {
        if let Some(icon) = page.frame.titles.iter().find(|title| title.title_id == friend.app_id).and_then(|title| title.icon) {
            let icon_rect = Rect::from_center_size(Pos2::new(remove.min.x - 34.0 * s, rect.center().y), Vec2::splat(44.0 * s));
            crate::carousel::draw_rounded_image(&page.painter, icon, icon_rect, 9.0 * s, Color32::WHITE);
        }
    }
    let busy = page.frame.state.busy.contains(&Busy::Remove(friend.pid));
    if busy {
        look::spinner(&page.painter, remove.center(), 12.0 * s, page.time, page.accent);
    } else if hovered || focused_row {
        if page.button(row, remove, "Remove", ButtonKind::Danger, true) {
            state.confirm = Some(Confirm::Remove { pid: friend.pid, name: friend.name.clone() });
            state.confirm_choice = 0;
        }
    } else {
        page.nav.slot(row, remove);
    }
}

fn requests_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let snapshot = page.frame.state;
    let mut y = origin.y;
    let count = snapshot.requests.len();
    let summary = match count {
        0 => "Friend requests".to_string(),
        1 => "1 friend request".to_string(),
        count => format!("{count} friend requests"),
    };
    page.text(Pos2::new(origin.x, y + 18.0 * s), egui::Align2::LEFT_CENTER, &summary, 18.0, page.pal.text);
    y += 52.0 * s;
    if snapshot.requests.is_empty() {
        y = empty_state(
            page,
            origin.x,
            y,
            width,
            "You're all caught up",
            "When someone adds your friend code, their request appears here and NeXium lets you know.",
            false,
        );
        return y - origin.y;
    }
    let requests: Vec<Request> = snapshot.requests.clone();
    for (index, request) in requests.iter().enumerate() {
        let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 70.0 * s));
        let row = index;
        let focused_row = page.nav.content && page.nav.row == row;
        look::card(
            &page.painter,
            rect,
            12.0 * s,
            if focused_row { page.pal.panel_hover } else { page.pal.panel },
            if focused_row { look::alpha(page.accent, 0.6) } else { page.pal.border },
            1.0,
        );
        let avatar_r = 22.0 * s;
        let avatar_center = Pos2::new(rect.min.x + 20.0 * s + avatar_r, rect.center().y);
        page.avatar(avatar_center, avatar_r, request.pid, &request.name, &mut state.avatars);
        let name_x = avatar_center.x + avatar_r + 18.0 * s;
        page.text(
            Pos2::new(name_x, rect.center().y - 11.0 * s),
            egui::Align2::LEFT_CENTER,
            if request.name.is_empty() { "Unnamed player" } else { &request.name },
            18.0,
            page.pal.text,
        );
        page.text(
            Pos2::new(name_x, rect.center().y + 13.0 * s),
            egui::Align2::LEFT_CENTER,
            if request.friend_code.is_empty() { "Wants to be your friend" } else { &request.friend_code },
            13.5,
            page.pal.muted,
        );
        let decline = Rect::from_min_size(Pos2::new(rect.max.x - 20.0 * s - 112.0 * s, rect.center().y - 20.0 * s), Vec2::new(112.0 * s, 40.0 * s));
        let accept = Rect::from_min_size(Pos2::new(decline.min.x - 12.0 * s - 112.0 * s, decline.min.y), Vec2::new(112.0 * s, 40.0 * s));
        if snapshot.busy.contains(&Busy::Answer(request.pid)) {
            look::spinner(&page.painter, Pos2::new(accept.max.x + 6.0 * s, rect.center().y), 13.0 * s, page.time, page.accent);
            page.nav.slot(row, accept);
        } else {
            if page.button(row, accept, "Accept", ButtonKind::Primary, true) {
                page.action = HubAction::Answer { pid: request.pid, accept: true };
            }
            if page.button(row, decline, "Decline", ButtonKind::Secondary, true) {
                page.action = HubAction::Answer { pid: request.pid, accept: false };
            }
        }
        y = rect.max.y + 8.0 * s;
    }
    y - origin.y
}

fn players_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let snapshot = page.frame.state;
    let me = snapshot.account.as_ref().map_or(0, |account| account.pid);
    let mut y = origin.y;
    let mut row = 0usize;
    if let Some(lobby) = snapshot.lobby.clone() {
        let game = title_name(lobby.title_id, page.frame.titles).unwrap_or_else(|| "an online match".into());
        page.label(Pos2::new(origin.x + 4.0 * s, y + 10.0 * s), &format!("IN YOUR LOBBY  ·  {}", game.to_uppercase()));
        let status = match lobby.state.as_str() {
            "searching" => "Looking for players",
            "matched" => "Match in progress",
            _ => "Lobby open",
        };
        let count = if lobby.max > 0 {
            format!("{status}  ·  {}/{} players", lobby.count.max(lobby.players.len() as u32), lobby.max)
        } else {
            status.to_string()
        };
        page.text(Pos2::new(origin.x + width - 4.0 * s, y + 10.0 * s), egui::Align2::RIGHT_CENTER, &count, 13.0, look::PLAYING);
        y += 26.0 * s;
        for player in &lobby.players {
            let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 66.0 * s));
            let subtitle = if player.host { "Host".to_string() } else { "In this lobby".to_string() };
            player_row(page, state, rect, player, row, me, &subtitle);
            row += 1;
            y = rect.max.y + 8.0 * s;
        }
        y += 18.0 * s;
    }
    page.label(Pos2::new(origin.x + 4.0 * s, y + 10.0 * s), "RECENTLY MET");
    y += 26.0 * s;
    if !snapshot.players_loaded {
        let center = Pos2::new(origin.x + width * 0.5, y + 50.0 * s);
        look::spinner(&page.painter, center, 18.0 * s, page.time, page.accent);
        return y + 100.0 * s - origin.y;
    }
    let recent: Vec<_> = snapshot.recent.iter().filter(|player| player.pid != me).cloned().collect();
    if recent.is_empty() {
        y = empty_state(
            page,
            origin.x,
            y,
            width,
            "No one yet",
            "Players you meet in online matches show up here, so you can add the ones you enjoyed playing with.",
            true,
        );
        return y - origin.y;
    }
    for player in &recent {
        let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 66.0 * s));
        let game = title_name(player.title_id, page.frame.titles);
        let when = player.seen_at.as_deref().and_then(relative_time);
        let subtitle = match (game, when) {
            (Some(game), Some(when)) => format!("{game}  ·  {when}"),
            (Some(game), None) => game,
            (None, Some(when)) => format!("Met {when}"),
            (None, None) => "Met online".to_string(),
        };
        player_row(page, state, rect, player, row, me, &subtitle);
        row += 1;
        y = rect.max.y + 8.0 * s;
    }
    y - origin.y
}

fn player_row(
    page: &mut Page,
    state: &mut HubState,
    rect: Rect,
    player: &super::api::LobbyPlayer,
    row: usize,
    me: u64,
    subtitle: &str,
) {
    let s = page.s;
    let focused_row = page.nav.content && page.nav.row == row;
    let hovered = page.row_hover(rect);
    look::card(
        &page.painter,
        rect,
        12.0 * s,
        if hovered || focused_row { page.pal.panel_hover } else { page.pal.panel },
        if focused_row { look::alpha(page.accent, 0.6) } else { page.pal.border },
        1.0,
    );
    let avatar_r = 21.0 * s;
    let avatar_center = Pos2::new(rect.min.x + 20.0 * s + avatar_r, rect.center().y);
    page.avatar(avatar_center, avatar_r, player.pid, &player.name, &mut state.avatars);
    let name_x = avatar_center.x + avatar_r + 18.0 * s;
    let name = if player.name.is_empty() { "Unknown player" } else { &player.name };
    let name_rect = page.text(Pos2::new(name_x, rect.center().y - 10.0 * s), egui::Align2::LEFT_CENTER, name, 17.0, page.pal.text);
    if player.host {
        let font = FontId::proportional(11.0 * s);
        let tag_w = look::text_width(page.ui, "HOST", &font) + 14.0 * s;
        let tag = Rect::from_min_size(Pos2::new(name_rect.max.x + 10.0 * s, name_rect.center().y - 9.0 * s), Vec2::new(tag_w, 18.0 * s));
        page.painter.rect_filled(tag, CornerRadius::from(9.0 * s), look::alpha(look::PLAYING, 0.18));
        page.painter.text(tag.center(), egui::Align2::CENTER_CENTER, "HOST", font, look::PLAYING);
    }
    let subtitle_font = FontId::proportional(13.0 * s);
    let subtitle = look::ellipsize(page.ui, subtitle, &subtitle_font, rect.width() * 0.5);
    page.painter.text(Pos2::new(name_x, rect.center().y + 13.0 * s), egui::Align2::LEFT_CENTER, subtitle, subtitle_font, page.pal.muted);
    let action = Rect::from_min_size(Pos2::new(rect.max.x - 20.0 * s - 140.0 * s, rect.center().y - 19.0 * s), Vec2::new(140.0 * s, 38.0 * s));
    let is_friend = page.frame.state.friends.iter().any(|friend| friend.pid == player.pid);
    let label = |text: &str, color: Color32| {
        page.painter.text(Pos2::new(action.max.x, action.center().y), egui::Align2::RIGHT_CENTER, text, FontId::proportional(14.0 * s), color);
    };
    if player.me || player.pid == me {
        label("You", page.pal.muted);
        page.nav.slot(row, action);
    } else if is_friend {
        label("Friends", look::ONLINE);
        page.nav.slot(row, action);
    } else if state.requested.contains(&player.pid) {
        label("Request sent", page.pal.muted);
        page.nav.slot(row, action);
    } else if !player.known || player.friend_code.is_empty() {
        label("Guest player", page.pal.faint);
        page.nav.slot(row, action);
    } else if page.button(row, action, "Add friend", ButtonKind::Secondary, true) {
        state.requested.insert(player.pid);
        page.action = HubAction::AddFriend(player.friend_code.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn text_input(
    page: &mut Page,
    buffer: &mut TextBuf,
    rect: Rect,
    placeholder: &str,
    max: usize,
    row: usize,
    keyboard: &mut crate::vkeyboard::VirtualKeyboard,
    keyboard_field: &mut Option<Field>,
    field: Field,
    title: &str,
) -> (bool, bool) {
    let s = page.s;
    let (focused, accepted) = page.nav.slot(row, rect);
    let primary_clicked = page.ui.input(|keys| keys.pointer.primary_clicked());
    if primary_clicked && buffer.editing && !page.ui.rect_contains_pointer(rect) {
        buffer.editing = false;
    }
    page.painter.rect_filled(rect, CornerRadius::from(10.0 * s), page.pal.input);
    page.painter.rect_stroke(
        rect,
        CornerRadius::from(10.0 * s),
        Stroke::new(if buffer.editing || focused { 1.8 } else { 1.0 }, if buffer.editing || focused { page.accent } else { page.pal.border }),
        egui::StrokeKind::Outside,
    );
    let font = FontId::proportional(19.0 * s);
    let pad = 14.0 * s;
    let selection = look::alpha(page.accent, 0.35);
    let blink = buffer.editing && (page.time * 1.6).fract() < 0.5;
    let events = if buffer.editing { page.ui.input(|keys| keys.events.clone()) } else { Vec::new() };
    let clip = page.painter.with_clip_rect(rect.shrink2(Vec2::new(pad * 0.5, 0.0)).intersect(page.nav.clip));
    let hit = page.interactive(rect);
    let response = if hit.is_positive() {
        crate::app::text_field(
            &clip,
            page.ui,
            hit,
            rect.min.x + pad,
            rect.center().y,
            &mut buffer.text,
            &mut buffer.caret,
            &mut buffer.anchor,
            font,
            page.pal.text,
            page.pal.faint,
            selection,
            placeholder,
            false,
            max,
            buffer.editing,
            blink,
            &events,
        )
    } else {
        let (shown, color) = if buffer.text.is_empty() {
            (placeholder, page.pal.faint)
        } else {
            (buffer.text.as_str(), page.pal.text)
        };
        clip.text(Pos2::new(rect.min.x + pad, rect.center().y), egui::Align2::LEFT_CENTER, shown, font, color);
        crate::app::FieldResp::default()
    };
    if response.clicked || response.secondary_clicked {
        buffer.editing = true;
    }
    if accepted {
        if page.frame.input.connected {
            keyboard.show_titled(&buffer.text, max, title);
            *keyboard_field = Some(field);
        } else {
            buffer.editing = true;
        }
    }
    let mut commit = false;
    if buffer.editing && response.commit {
        buffer.editing = false;
        commit = true;
    }
    if buffer.editing && response.cancel {
        buffer.editing = false;
    }
    (commit, response.changed)
}

fn add_friend_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let snapshot = page.frame.state;
    let mut y = origin.y;
    let own = snapshot.account.as_ref().map(|account| account.friend_code.clone()).unwrap_or_default();
    let card = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 128.0 * s));
    look::card(&page.painter, card, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    page.label(Pos2::new(card.min.x + 28.0 * s, card.min.y + 30.0 * s), "YOUR FRIEND CODE");
    page.text(
        Pos2::new(card.min.x + 28.0 * s, card.min.y + 76.0 * s),
        egui::Align2::LEFT_CENTER,
        if own.is_empty() { "Not assigned yet" } else { &own },
        32.0,
        page.pal.text,
    );
    let copied = state.copied_at.is_some_and(|at| page.time - at < 1.6);
    let copy = Rect::from_min_size(Pos2::new(card.max.x - 28.0 * s - 150.0 * s, card.center().y - 22.0 * s), Vec2::new(150.0 * s, 44.0 * s));
    if page.button(0, copy, if copied { "Copied" } else { "Copy code" }, ButtonKind::Secondary, !own.is_empty()) {
        page.ui.ctx().copy_text(own.clone());
        state.copied_at = Some(page.time);
    }
    y = card.max.y + 18.0 * s;

    let hint = page.painter.layout(
        "Ask your friend for their code. It's on their Nextendo profile and in their emulator's Nextendo screen.".to_string(),
        FontId::proportional(13.0 * s),
        page.pal.faint,
        width - 56.0 * s,
    );
    let form = Rect::from_min_size(
        Pos2::new(origin.x, y),
        Vec2::new(width, 144.0 * s + hint.size().y.max(20.0 * s)),
    );
    look::card(&page.painter, form, 16.0 * s, page.pal.panel, page.pal.border, 1.0);
    page.label(Pos2::new(form.min.x + 28.0 * s, form.min.y + 30.0 * s), "ADD A FRIEND");
    let field_w = (form.width() - 56.0 * s - 200.0 * s).min(440.0 * s);
    let field = Rect::from_min_size(Pos2::new(form.min.x + 28.0 * s, form.min.y + 52.0 * s), Vec2::new(field_w, 52.0 * s));
    let (commit, changed) = text_input(
        page,
        &mut state.code,
        field,
        "SW-0000-0000-0000",
        20,
        1,
        &mut state.keyboard,
        &mut state.keyboard_field,
        Field::FriendCode,
        "Friend code",
    );
    if changed {
        state.awaiting_add = false;
    }
    let busy = snapshot.busy.contains(&Busy::AddFriend);
    let send = Rect::from_min_size(Pos2::new(field.max.x + 14.0 * s, field.min.y + 2.0 * s), Vec2::new(186.0 * s, 48.0 * s));
    let valid = normalize_friend_code(&state.code.text).is_some();
    if busy {
        look::spinner(&page.painter, send.center(), 14.0 * s, page.time, page.accent);
        page.nav.slot(1, send);
    } else if page.button(1, send, "Send request", ButtonKind::Primary, valid) || (commit && valid) {
        if let Some(code) = normalize_friend_code(&state.code.text) {
            state.code.set(&code);
            state.awaiting_add = true;
            page.action = HubAction::AddFriend(code);
        }
    }
    let message_top = field.max.y + 14.0 * s;
    let message_y = message_top + 10.0 * s;
    if state.awaiting_add && matches!(snapshot.add_result, Some(Ok(()))) && !state.code.editing {
        state.code.set("");
    }
    let typed = state.code.text.trim();
    let mut status_shown = true;
    match (&snapshot.add_result, state.awaiting_add) {
        (Some(Ok(())), true) => {
            look::check_glyph(&page.painter, Pos2::new(form.min.x + 36.0 * s, message_y), 16.0 * s, look::ONLINE);
            page.text(Pos2::new(form.min.x + 52.0 * s, message_y), egui::Align2::LEFT_CENTER, "Request sent. You'll see them in Friends once they accept.", 14.5, look::ONLINE);
        }
        (Some(Err(message)), true) => {
            look::cross_glyph(&page.painter, Pos2::new(form.min.x + 36.0 * s, message_y), 16.0 * s, look::DANGER);
            let font = FontId::proportional(14.5 * s);
            let message = look::ellipsize(page.ui, message, &font, form.width() - 90.0 * s);
            page.painter.text(Pos2::new(form.min.x + 52.0 * s, message_y), egui::Align2::LEFT_CENTER, message, font, look::DANGER);
        }
        _ if !state.awaiting_add && !typed.is_empty() && !valid => {
            page.text(Pos2::new(form.min.x + 28.0 * s, message_y), egui::Align2::LEFT_CENTER, "Friend codes have 12 digits, like SW-1234-5678-9012.", 14.0, page.pal.muted);
        }
        _ => status_shown = false,
    }
    if !status_shown {
        page.painter.galley(Pos2::new(form.min.x + 28.0 * s, message_top), hint, page.pal.faint);
    }
    form.max.y + 12.0 * s - origin.y
}

fn online_play_page(page: &mut Page, state: &mut HubState, origin: Pos2, width: f32) -> f32 {
    let s = page.s;
    let settings = page.frame.settings.clone();
    let mut y = origin.y;
    page.label(Pos2::new(origin.x + 4.0 * s, y + 10.0 * s), "PREFERENCES");
    y += 26.0 * s;
    let rows: [(&str, &str, bool, u8); 3] = [
        (
            "Play online through Nextendo",
            "Supported games connect to Nextendo's servers while you're signed in.",
            settings.online_play,
            0,
        ),
        (
            "Share what I'm playing",
            "Friends see when you're online and which game you're in.",
            settings.share_presence,
            1,
        ),
        (
            "Friend notifications",
            "Show a notice when friends come online or send you a request.",
            settings.friend_alerts,
            2,
        ),
    ];
    for (index, (title, subtitle, on, kind)) in rows.iter().enumerate() {
        let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 68.0 * s));
        if page.toggle_row(index, rect, title, subtitle, *on) {
            crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            page.action = match kind {
                0 => HubAction::SetOnlinePlay(!on),
                1 => HubAction::SetSharePresence(!on),
                _ => HubAction::SetFriendAlerts(!on),
            };
        }
        y = rect.max.y + 10.0 * s;
    }

    y += 18.0 * s;
    page.label(Pos2::new(origin.x + 4.0 * s, y + 10.0 * s), &format!("SUPPORTED GAMES  ·  {}", nexium_common::nextendo::COMPATIBLE_TITLES.len()));
    if let Some(players) = page.frame.state.players_online {
        page.text(
            Pos2::new(origin.x + width - 4.0 * s, y + 10.0 * s),
            egui::Align2::RIGHT_CENTER,
            &format!("{} playing right now", group(players)),
            13.0,
            look::ONLINE,
        );
    }
    y += 26.0 * s;
    let mut games: Vec<&nexium_common::nextendo::CompatibleTitle> = Vec::new();
    for title in nexium_common::nextendo::COMPATIBLE_TITLES {
        if !games.iter().any(|known| known.name == title.name) {
            games.push(title);
        }
    }
    let frame = page.frame;
    let owned = |title: &nexium_common::nextendo::CompatibleTitle| {
        nexium_common::nextendo::COMPATIBLE_TITLES
            .iter()
            .filter(|other| other.name == title.name)
            .find_map(|other| frame.titles.iter().find(|owned| owned.title_id == other.title_id))
            .cloned()
    };
    let players_for = |title: &nexium_common::nextendo::CompatibleTitle| -> u32 {
        nexium_common::nextendo::COMPATIBLE_TITLES
            .iter()
            .filter(|other| other.name == title.name)
            .map(|other| frame.state.counts.get(&other.title_id).copied().unwrap_or(0))
            .sum()
    };
    games.sort_by_key(|title| (owned(title).is_none(), std::cmp::Reverse(players_for(title)), title.name));
    let base_row = rows.len();
    for (index, title) in games.iter().enumerate() {
        let rect = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 62.0 * s));
        let library = owned(title);
        let players = players_for(title);
        let row = base_row + index;
        let focused_row = page.nav.content && page.nav.row == row;
        let hovered = page.row_hover(rect);
        look::card(
            &page.painter,
            rect,
            12.0 * s,
            if hovered || focused_row { page.pal.panel_hover } else { page.pal.panel },
            if focused_row { look::alpha(page.accent, 0.6) } else { page.pal.border },
            1.0,
        );
        let icon_rect = Rect::from_center_size(Pos2::new(rect.min.x + 18.0 * s + 21.0 * s, rect.center().y), Vec2::splat(42.0 * s));
        match library.as_ref().and_then(|owned| owned.icon) {
            Some(icon) => crate::carousel::draw_rounded_image(&page.painter, icon, icon_rect, 9.0 * s, Color32::WHITE),
            None => {
                page.painter.rect_filled(icon_rect, CornerRadius::from(9.0 * s), page.pal.input);
                look::network_glyph(&page.painter, icon_rect.center(), 11.0 * s, page.pal.faint);
            }
        }
        let text_x = icon_rect.max.x + 16.0 * s;
        let name_font = FontId::proportional(17.0 * s);
        let name = look::ellipsize(page.ui, title.name, &name_font, rect.width() * 0.5);
        page.painter.text(Pos2::new(text_x, rect.center().y - 10.0 * s), egui::Align2::LEFT_CENTER, name, name_font, page.pal.text);
        let (detail, detail_color) = match &library {
            Some(owned) if nexium_common::nextendo::version_matches(owned.title_id, &owned.version) => (
                format!("In your library  ·  v{} ready", title.version),
                look::ONLINE,
            ),
            Some(owned) => (
                format!("Needs v{}  ·  you have v{}", title.version, owned.version),
                look::WARNING,
            ),
            None => (format!("Requires v{}", title.version), page.pal.muted),
        };
        page.text(Pos2::new(text_x, rect.center().y + 13.0 * s), egui::Align2::LEFT_CENTER, &detail, 13.0, detail_color);
        let count_text = if players == 0 { "No one online".to_string() } else { format!("{} online", group(players)) };
        let mut right_edge = rect.max.x - 20.0 * s;
        if let Some(owned) = library.as_ref().filter(|owned| nexium_common::nextendo::version_matches(owned.title_id, &owned.version)) {
            let play = Rect::from_min_size(Pos2::new(right_edge - 96.0 * s, rect.center().y - 18.0 * s), Vec2::new(96.0 * s, 36.0 * s));
            let running = page.frame.running_title == Some(owned.title_id);
            if page.button(row, play, if running { "Playing" } else { "Play" }, ButtonKind::Primary, !running) {
                page.action = HubAction::Launch(owned.path.clone());
            }
            right_edge = play.min.x - 16.0 * s;
        } else {
            page.nav.slot(row, rect);
        }
        page.text(
            Pos2::new(right_edge, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            &count_text,
            14.0,
            if players > 0 { page.pal.text } else { page.pal.faint },
        );
        if players > 0 {
            let width_text = look::text_width(page.ui, &count_text, &FontId::proportional(14.0 * s));
            page.painter.circle_filled(Pos2::new(right_edge - width_text - 10.0 * s, rect.center().y), 4.0 * s, look::ONLINE);
        }
        y = rect.max.y + 8.0 * s;
    }

    y += 22.0 * s;
    page.label(Pos2::new(origin.x + 4.0 * s, y + 10.0 * s), "SERVERS");
    y += 26.0 * s;
    let server_rows = base_row + games.len();
    let card = Rect::from_min_size(Pos2::new(origin.x, y), Vec2::new(width, 206.0 * s));
    look::card(&page.painter, card, 14.0 * s, page.pal.panel, page.pal.border, 1.0);
    let field_w = (card.width() * 0.42).min(320.0 * s);
    let labels = [
        ("Game server", "Where Nintendo's game hosts are sent."),
        ("NAT check server", "The second responder used to test your connection."),
    ];
    let mut commit_any = false;
    for (index, (label, detail)) in labels.iter().enumerate() {
        let line_y = card.min.y + 46.0 * s + index as f32 * 66.0 * s;
        page.text(Pos2::new(card.min.x + 24.0 * s, line_y - 9.0 * s), egui::Align2::LEFT_CENTER, label, 16.0, page.pal.text);
        page.text(Pos2::new(card.min.x + 24.0 * s, line_y + 13.0 * s), egui::Align2::LEFT_CENTER, detail, 12.5, page.pal.muted);
        let field = Rect::from_min_size(Pos2::new(card.max.x - 24.0 * s - field_w, line_y - 23.0 * s), Vec2::new(field_w, 46.0 * s));
        let (buffer, kind) = if index == 0 { (&mut state.server, Field::Server) } else { (&mut state.nat, Field::Nat) };
        let (commit, _) = text_input(
            page,
            buffer,
            field,
            if index == 0 { "51.178.29.194" } else { "164.132.111.120" },
            15,
            server_rows + index,
            &mut state.keyboard,
            &mut state.keyboard_field,
            kind,
            label,
        );
        commit_any |= commit;
    }
    let buttons_y = card.max.y - 58.0 * s;
    let apply = Rect::from_min_size(Pos2::new(card.min.x + 24.0 * s, buttons_y), Vec2::new(150.0 * s, 42.0 * s));
    let changed = state.server.text.trim() != settings.server || state.nat.text.trim() != settings.nat;
    if page.button(server_rows + 2, apply, "Apply", ButtonKind::Primary, changed) || (commit_any && changed) {
        let parsed = (
            state.server.text.trim().parse::<std::net::Ipv4Addr>(),
            state.nat.text.trim().parse::<std::net::Ipv4Addr>(),
        );
        match parsed {
            (Ok(server), Ok(nat)) if server != nat => {
                state.server_error = None;
                page.action = HubAction::SetServers { server: server.to_string(), nat: nat.to_string() };
            }
            (Ok(_), Ok(_)) => state.server_error = Some("The two servers must be different addresses.".into()),
            _ => state.server_error = Some("Enter IPv4 addresses, like 51.178.29.194.".into()),
        }
    }
    let restore = Rect::from_min_size(Pos2::new(apply.max.x + 12.0 * s, buttons_y), Vec2::new(190.0 * s, 42.0 * s));
    if page.button(server_rows + 2, restore, "Restore defaults", ButtonKind::Secondary, true) {
        state.server.set(&nexium_common::nextendo::DEFAULT_SERVER_IP.to_string());
        state.nat.set(&nexium_common::nextendo::DEFAULT_NAT_IP.to_string());
        state.server_error = None;
        page.action = HubAction::RestoreServers;
    }
    if let Some(error) = &state.server_error {
        page.text(Pos2::new(restore.max.x + 16.0 * s, restore.center().y), egui::Align2::LEFT_CENTER, error, 13.5, look::DANGER);
    } else if !settings.online_play {
        page.text(Pos2::new(restore.max.x + 16.0 * s, restore.center().y), egui::Align2::LEFT_CENTER, "Applies when online play is on.", 13.0, page.pal.faint);
    }
    card.max.y + 12.0 * s - origin.y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_naturally() {
        assert_eq!(age_text(5).as_deref(), Some("just now"));
        assert_eq!(age_text(125).as_deref(), Some("2 min ago"));
        assert_eq!(age_text(3 * 3_600 + 10).as_deref(), Some("3 h ago"));
        assert_eq!(age_text(2 * 86_400).as_deref(), Some("2 d ago"));
        assert_eq!(age_text(30 * 86_400), None);
        assert!(relative_time("not a date").is_none());
        assert!(relative_time("2026-01-01T00:00:00Z").is_some());
    }

    #[test]
    fn every_tab_has_a_distinct_index() {
        let indices: std::collections::HashSet<usize> = TABS.iter().map(|tab| tab.index()).collect();
        assert_eq!(indices.len(), TAB_COUNT);
    }
}
