use super::HorizonApp;
use crate::nextendo::hub::{HubAction, HubFrame, HubSettings, LibraryTitle, Tab};
use crate::nextendo::toasts::{ToastKind, ToastTarget};
use crate::nextendo::widgets::{ChipTone, HeaderChip, OnlineBadge};
use crate::nextendo::{Notice, NowPlaying, Phase, PlaySession, Preferences};
use eframe::egui;
use std::time::{Duration, Instant};

const DECORATION_REFRESH: Duration = Duration::from_millis(500);

pub(super) fn preferences(settings: &crate::app_settings::NextendoSettings) -> Preferences {
    Preferences {
        online_play: settings.online_play,
        server: settings.server_ip(),
        nat: settings.nat_ip(),
        share_presence: settings.share_presence,
        friend_alerts: settings.friend_alerts,
    }
}

impl HorizonApp {
    pub(super) fn nextendo_overlay_active(&self) -> bool {
        self.show_nextendo || self.nextendo_anim > 0.0
    }

    pub(super) fn open_nextendo(&mut self, tab: Tab) {
        if !self.show_nextendo {
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }
        self.nextendo_hub.open(tab);
        self.show_nextendo = true;
        self.show_profile = false;
    }

    fn running_title_id(&mut self) -> Option<u64> {
        let running = self
            .emulation_handle
            .as_ref()
            .is_some_and(|handle| handle.is_running());
        if !running {
            return None;
        }
        let path = self.playing_path.clone()?;
        self.nextendo_catalog.info(&path).map(|info| info.title_id)
    }

    fn library_name(&self, path: &std::path::Path) -> Option<String> {
        self.library
            .index_of_path(path)
            .and_then(|index| self.library.games.get(index))
            .map(|game| game.title.clone())
    }

    pub(super) fn nextendo_tick(&mut self, ctx: &egui::Context) {
        self.nextendo_state = self.nextendo.state();
        for notice in self.nextendo.take_notices() {
            self.nextendo_notice(notice);
        }
        let running = self.running_title_id();
        if running != self.nextendo_reported_title {
            if let (Some(title_id), Some(path)) =
                (self.nextendo_reported_title, self.nextendo_session_path.take())
            {
                self.play_times.save_if_dirty();
                self.nextendo.sync_history(PlaySession {
                    title_id,
                    name: self.library_name(&path).unwrap_or_default(),
                    seconds: self.play_times.get(&path),
                    path,
                });
            }
            if running.is_some() {
                self.nextendo_session_path = self.playing_path.clone();
            }
            self.nextendo_reported_title = running;
            let now_playing = running.map(|title_id| NowPlaying {
                title_id,
                name: self
                    .playing_path
                    .clone()
                    .and_then(|path| self.library_name(&path))
                    .or_else(|| {
                        nexium_common::nextendo::compatible_title(title_id)
                            .map(|title| title.name.to_string())
                    })
                    .unwrap_or_default(),
            });
            self.nextendo.set_now_playing(now_playing);
        }
        if self
            .nextendo_decorated
            .map_or(true, |at| at.elapsed() >= DECORATION_REFRESH)
        {
            self.nextendo_decorated = Some(Instant::now());
            self.refresh_online_decorations(ctx);
        }
    }

    fn refresh_online_decorations(&mut self, ctx: &egui::Context) {
        let state = &self.nextendo_state;
        let account = state.account.clone();
        let chip = match &state.phase {
            Phase::SignedOut => HeaderChip {
                label: "Nextendo".into(),
                detail: "Sign in".into(),
                tone: ChipTone::SignedOut,
                avatar: None,
                seed: 0,
                alerts: 0,
            },
            Phase::Browser { .. } | Phase::Finishing | Phase::Restoring => HeaderChip {
                label: account
                    .as_ref()
                    .map(|account| account.username.clone())
                    .unwrap_or_else(|| "Nextendo".into()),
                detail: "Signing in…".into(),
                tone: ChipTone::Connecting,
                avatar: None,
                seed: account.as_ref().map_or(0, |account| account.pid),
                alerts: 0,
            },
            Phase::SignedIn => {
                let pid = account.as_ref().map_or(0, |account| account.pid);
                let online = state.friends_online();
                let detail = if state.offline {
                    "Offline".to_string()
                } else if online == 1 {
                    "1 friend online".to_string()
                } else if online > 1 {
                    format!("{online} friends online")
                } else {
                    "Online".to_string()
                };
                HeaderChip {
                    label: account
                        .as_ref()
                        .map(|account| account.username.clone())
                        .unwrap_or_else(|| "Nextendo".into()),
                    detail,
                    tone: if state.offline { ChipTone::Offline } else { ChipTone::Online },
                    avatar: self.nextendo_hub.avatars.texture(ctx, &state.avatars, pid),
                    seed: pid,
                    alerts: state.requests.len(),
                }
            }
        };
        self.carousel.online_chip = Some(chip);
        let mut badges = std::collections::HashMap::new();
        if self.app_settings.nextendo.online_play {
            let paths: Vec<std::path::PathBuf> = self
                .library
                .games
                .iter()
                .filter(|game| game.download.is_none())
                .map(|game| game.path.clone())
                .collect();
            for path in paths {
                let Some(info) = self.nextendo_catalog.info(&path) else {
                    continue;
                };
                let Some(compatible) = info.compatible() else {
                    continue;
                };
                badges.insert(
                    path,
                    OnlineBadge {
                        players: self.nextendo_state.counts.get(&info.title_id).copied(),
                        required: compatible.version,
                        installed: info.version.clone(),
                        version_ok: info.version_ok(),
                        enabled: true,
                    },
                );
            }
        }
        self.carousel.online_badges = badges;
    }

    fn nextendo_notice(&mut self, notice: Notice) {
        let alerts = self.app_settings.nextendo.friend_alerts;
        match notice {
            Notice::SignedIn { name } => self.nextendo_toasts.push(
                ToastKind::Success,
                "Signed in to Nextendo",
                format!("Welcome, {name}. Supported games can now play online."),
                None,
                ToastTarget::Account,
            ),
            Notice::SessionEnded { reason } => self.nextendo_toasts.push(
                ToastKind::Warning,
                "Signed out of Nextendo",
                reason,
                None,
                ToastTarget::Account,
            ),
            Notice::SignInFailed(message) => self.nextendo_toasts.push(
                ToastKind::Warning,
                "Sign-in didn't finish",
                message,
                None,
                ToastTarget::Account,
            ),
            Notice::FriendOnline { pid, name, app_id, app_name } if alerts => {
                let body = if app_id != 0 {
                    let game = if app_name.trim().is_empty() {
                        nexium_common::nextendo::compatible_title(app_id)
                            .map(|title| title.name.to_string())
                            .unwrap_or_else(|| "a game".into())
                    } else {
                        app_name
                    };
                    format!("Playing {game}")
                } else {
                    "Just came online".to_string()
                };
                self.nextendo_toasts.push(
                    ToastKind::Friend,
                    format!("{name} is online"),
                    body,
                    Some(pid),
                    ToastTarget::Friends,
                );
            }
            Notice::FriendRequest { pid, name } if alerts => self.nextendo_toasts.push(
                ToastKind::Request,
                format!("{name} sent you a friend request"),
                "Open Nextendo to accept or decline.",
                Some(pid),
                ToastTarget::Requests,
            ),
            Notice::FriendOnline { .. } | Notice::FriendRequest { .. } => {}
            Notice::RequestSent => self.nextendo_toasts.push(
                ToastKind::Success,
                "Friend request sent",
                "They'll show up in Friends once they accept.",
                None,
                ToastTarget::Friends,
            ),
            Notice::FriendAdded { pid, name } => self.nextendo_toasts.push(
                ToastKind::Success,
                format!("You're now friends with {name}"),
                "Say hi next time you're both online.",
                Some(pid),
                ToastTarget::Friends,
            ),
            Notice::FriendRemoved { name } => self.nextendo_toasts.push(
                ToastKind::Info,
                format!("{name} was removed"),
                "They're no longer on your friend list.",
                None,
                ToastTarget::Friends,
            ),
            Notice::ActionFailed(message) => self.nextendo_toasts.push(
                ToastKind::Warning,
                "Nextendo",
                message,
                None,
                ToastTarget::Account,
            ),
        }
    }

    fn save_nextendo_settings(&mut self) {
        let _ = self.app_settings.save();
        self.nextendo.configure(preferences(&self.app_settings.nextendo));
    }

    fn library_titles(&mut self, ctx: &egui::Context) -> Vec<LibraryTitle> {
        let mut titles = Vec::new();
        for index in 0..self.library.games.len() {
            let path = self.library.games[index].path.clone();
            let Some(info) = self.nextendo_catalog.info(&path) else {
                continue;
            };
            let icon = self.library.texture(ctx, index).map(|texture| texture.id());
            titles.push(LibraryTitle {
                title_id: info.title_id,
                name: self.library.games[index].title.clone(),
                version: info.version,
                path,
                icon,
            });
        }
        titles
    }

    pub(super) fn draw_nextendo_hub(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, full: egui::Rect, running: bool) {
        let target = if self.show_nextendo { 1.0 } else { 0.0 };
        let dt = ctx.input(|input| input.stable_dt).min(0.1);
        if self.nextendo_anim < target {
            self.nextendo_anim = (self.nextendo_anim + dt * 3.6).min(target);
        } else if self.nextendo_anim > target {
            self.nextendo_anim = (self.nextendo_anim - dt * 3.6).max(target);
        }
        let smooth = |value: f32| {
            let value = value.clamp(0.0, 1.0);
            value * value * (3.0 - 2.0 * value)
        };
        let progress = self.nextendo_anim.clamp(0.0, 1.0);
        let backdrop_opacity = smooth(progress / 0.4);
        let content_opacity = smooth((progress - 0.4) / 0.6);
        let titles = self.library_titles(ctx);
        let settings = HubSettings {
            online_play: self.app_settings.nextendo.online_play,
            share_presence: self.app_settings.nextendo.share_presence,
            friend_alerts: self.app_settings.nextendo.friend_alerts,
            server: self.app_settings.nextendo.server_ip().to_string(),
            nat: self.app_settings.nextendo.nat_ip().to_string(),
        };
        let state = self.nextendo_state.clone();
        let accent = self.theme_accent();
        let frame = HubFrame {
            state: &state,
            settings: &settings,
            titles: &titles,
            ambient: self.carousel.ambient_color,
            accent,
            backdrop: self.app_settings.backdrop_theme,
            light_mode: self.app_settings.light_mode,
            content_opacity,
            backdrop_opacity,
            zoom: 0.92 + 0.08 * content_opacity,
            active: self.show_nextendo && !self.modal_active() && !self.modal_active_frame_start,
            input: &self.last_input,
            running_title: self.nextendo_reported_title,
        };
        let action = crate::nextendo::hub::hub_view(&mut self.nextendo_hub, &frame, ctx, ui, full);
        if !self.show_nextendo {
            return;
        }
        match action {
            HubAction::None => {}
            HubAction::Close => {
                self.show_nextendo = false;
            }
            HubAction::SignIn | HubAction::ReopenBrowser => self.nextendo.sign_in(),
            HubAction::CancelSignIn => self.nextendo.cancel_sign_in(),
            HubAction::OpenUrl(url) => {
                crate::nextendo::oauth::open_browser(&url);
            }
            HubAction::SignOut => {
                self.nextendo.sign_out();
                self.nextendo_toasts.push(
                    ToastKind::Info,
                    "Signed out of Nextendo",
                    "Sign in again any time from the Nextendo screen.",
                    None,
                    ToastTarget::Account,
                );
            }
            HubAction::Refresh => self.nextendo.refresh(),
            HubAction::AddFriend(code) => self.nextendo.add_friend(&code),
            HubAction::Answer { pid, accept } => self.nextendo.answer_request(pid, accept),
            HubAction::Remove(pid) => self.nextendo.remove_friend(pid),
            HubAction::SetOnlinePlay(on) => {
                self.app_settings.nextendo.online_play = on;
                self.save_nextendo_settings();
                self.nextendo_decorated = None;
            }
            HubAction::SetSharePresence(on) => {
                self.app_settings.nextendo.share_presence = on;
                self.save_nextendo_settings();
            }
            HubAction::SetFriendAlerts(on) => {
                self.app_settings.nextendo.friend_alerts = on;
                self.save_nextendo_settings();
            }
            HubAction::SetServers { server, nat } => {
                self.app_settings.nextendo.server = server;
                self.app_settings.nextendo.nat = nat;
                self.save_nextendo_settings();
                self.nextendo_toasts.push(
                    ToastKind::Success,
                    "Servers updated",
                    "Games use the new addresses the next time they connect.",
                    None,
                    ToastTarget::Account,
                );
            }
            HubAction::RestoreServers => {
                self.app_settings.nextendo.server.clear();
                self.app_settings.nextendo.nat.clear();
                self.save_nextendo_settings();
            }
            HubAction::Launch(path) => {
                self.show_nextendo = false;
                self.request_launch(path.to_string_lossy().to_string(), ctx, running);
            }
        }
    }

    pub(super) fn draw_nextendo_toasts(&mut self, ctx: &egui::Context, top_offset: f32) {
        if self.nextendo_toasts.is_empty() {
            return;
        }
        let area = ctx.viewport_rect();
        let accent = self.theme_accent();
        let clicked = self.nextendo_toasts.show(
            ctx,
            area,
            top_offset,
            &mut self.nextendo_hub.avatars,
            &self.nextendo_state.avatars,
            accent,
            self.app_settings.light_mode,
        );
        if let Some(target) = clicked {
            self.open_nextendo(match target {
                ToastTarget::Account => Tab::Overview,
                ToastTarget::Friends => Tab::Friends,
                ToastTarget::Requests => Tab::Requests,
            });
        }
    }

    pub(super) fn nextendo_menu_button(&mut self, ui: &mut egui::Ui) {
        let state = &self.nextendo_state;
        let (dot, label) = match &state.phase {
            Phase::SignedIn if state.offline => (crate::nextendo::look::WARNING, "Nextendo · Offline".to_string()),
            Phase::SignedIn => {
                let online = state.friends_online();
                (
                    crate::nextendo::look::ONLINE,
                    if online > 0 {
                        format!("Nextendo · {online} online")
                    } else {
                        "Nextendo".to_string()
                    },
                )
            }
            Phase::SignedOut => (super::MUTED, "Nextendo · Sign in".to_string()),
            _ => (crate::nextendo::look::WARNING, "Nextendo · Signing in…".to_string()),
        };
        let requests = state.requests.len();
        let text = if requests > 0 { format!("{label}  ({requests})") } else { label };
        let galley = ui.painter().layout_no_wrap(text.clone(), egui::FontId::proportional(13.0), super::TEXT);
        let size = egui::vec2(galley.size().x + 26.0, 24.0);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(rect, egui::CornerRadius::same(6), egui::Color32::from_rgb(0x1E, 0x1E, 0x26));
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        ui.painter().circle_filled(egui::pos2(rect.min.x + 9.0, rect.center().y), 3.6, dot);
        ui.painter().galley(
            egui::pos2(rect.min.x + 18.0, rect.center().y - galley.size().y * 0.5),
            galley,
            super::TEXT,
        );
        if response.on_hover_text("Nextendo Network: account, friends and online play").clicked() {
            self.open_nextendo(Tab::Overview);
        }
    }
}
