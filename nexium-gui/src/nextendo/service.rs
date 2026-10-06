use super::api::{Api, ApiError, FriendList, HistoryEntry, Identity, Lobby, LobbyPlayer, PresenceUpdate, Standing};
use super::{oauth, vault};
use eframe::egui;
use nexium_common::nextendo as shared;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TICK: Duration = Duration::from_millis(250);
const FRIEND_POLL: Duration = Duration::from_secs(20);
const PRESENCE_EVERY: Duration = Duration::from_secs(60);
const PRESENCE_GAP: Duration = Duration::from_secs(4);
const COUNTS_EVERY: Duration = Duration::from_secs(60);
const HEALTH_EVERY: Duration = Duration::from_secs(45);
const STANDING_EVERY: Duration = Duration::from_secs(5 * 60);
const REFRESH_MARGIN: Duration = Duration::from_secs(120);
const GAME_TOKEN_MARGIN: Duration = Duration::from_secs(60 * 60);
const OFFLINE_RETRY: Duration = Duration::from_secs(30);
const OFFLINE_CONFIRMATIONS: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preferences {
    pub online_play: bool,
    pub server: Ipv4Addr,
    pub nat: Ipv4Addr,
    pub share_presence: bool,
    pub friend_alerts: bool,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            online_play: true,
            server: shared::DEFAULT_SERVER_IP,
            nat: shared::DEFAULT_NAT_IP,
            share_presence: true,
            friend_alerts: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NowPlaying {
    pub title_id: u64,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    SignedOut,
    Restoring,
    Browser {
        url: String,
    },
    Finishing,
    SignedIn,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GameAccess {
    #[default]
    Unknown,
    Ready,
    Missing(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Busy {
    AddFriend,
    Answer(u64),
    Remove(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Friend {
    pub pid: u64,
    pub name: String,
    pub friend_code: String,
    pub status: i32,
    pub app_id: u64,
    pub app_name: String,
    pub console: bool,
    pub last_seen: Option<String>,
    pub staff: bool,
    pub favorite: bool,
}

impl Friend {
    pub fn online(&self) -> bool {
        self.status > shared::PRESENCE_OFFLINE
    }

    pub fn in_game(&self) -> bool {
        self.online() && self.app_id != 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub pid: u64,
    pub name: String,
    pub friend_code: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Notice {
    SignedIn { name: String },
    SessionEnded { reason: String },
    SignInFailed(String),
    FriendOnline { pid: u64, name: String, app_id: u64, app_name: String },
    FriendRequest { pid: u64, name: String },
    RequestSent,
    FriendAdded { pid: u64, name: String },
    FriendRemoved { name: String },
    ActionFailed(String),
}

#[derive(Clone, Default)]
pub struct State {
    pub phase: Phase,
    pub account: Option<Identity>,
    pub friends: Vec<Friend>,
    pub requests: Vec<Request>,
    pub friends_loaded: bool,
    pub friends_issue: Option<String>,
    pub standing: Option<Standing>,
    pub game_access: GameAccess,
    pub presence_issue: Option<String>,
    pub counts: HashMap<u64, u32>,
    pub players_online: Option<u32>,
    pub latency: Option<Duration>,
    pub reachable: Option<bool>,
    pub offline: bool,
    pub avatars: HashMap<u64, Arc<Vec<u8>>>,
    pub avatar_revision: u64,
    pub busy: HashSet<Busy>,
    pub add_result: Option<Result<(), String>>,
    pub sign_in_error: Option<String>,
    pub last_sync: Option<Instant>,
    pub lobby: Option<Lobby>,
    pub recent: Vec<LobbyPlayer>,
    pub players_loaded: bool,
}

impl State {
    pub fn signed_in(&self) -> bool {
        self.phase == Phase::SignedIn
    }

    pub fn friends_online(&self) -> usize {
        self.friends.iter().filter(|friend| friend.online()).count()
    }
}

enum Command {
    SignIn,
    CancelSignIn,
    SignOut,
    RefreshNow,
    AddFriend(String),
    Answer { pid: u64, accept: bool },
    Remove(u64),
    NowPlaying(Option<NowPlaying>),
    Configure(Preferences),
    SyncHistory(PlaySession),
    Authorized(Result<oauth::Grant, oauth::AuthError>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaySession {
    pub title_id: u64,
    pub name: String,
    pub seconds: u64,
    pub path: std::path::PathBuf,
}

pub struct Nextendo {
    state: Arc<Mutex<State>>,
    commands: Sender<Command>,
    notices: Receiver<Notice>,
}

impl Nextendo {
    pub fn start(ctx: egui::Context, preferences: Preferences) -> Self {
        let state = Arc::new(Mutex::new(State::default()));
        let (commands, inbox) = mpsc::channel();
        let (notice_tx, notices) = mpsc::channel();
        let worker = Worker::new(
            state.clone(),
            inbox,
            commands.clone(),
            notice_tx,
            ctx,
            preferences,
        );
        if let Err(error) = std::thread::Builder::new()
            .name("nextendo".into())
            .spawn(move || worker.run())
        {
            log::error!("nextendo: could not start the service thread: {error}");
        }
        Self {
            state,
            commands,
            notices,
        }
    }

    pub fn state(&self) -> State {
        self.state.lock().map(|state| state.clone()).unwrap_or_default()
    }

    fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }

    pub fn sign_in(&self) {
        self.send(Command::SignIn);
    }

    pub fn cancel_sign_in(&self) {
        self.send(Command::CancelSignIn);
    }

    pub fn sign_out(&self) {
        self.send(Command::SignOut);
    }

    pub fn refresh(&self) {
        self.send(Command::RefreshNow);
    }

    pub fn add_friend(&self, friend_code: &str) {
        self.send(Command::AddFriend(friend_code.to_string()));
    }

    pub fn answer_request(&self, pid: u64, accept: bool) {
        self.send(Command::Answer { pid, accept });
    }

    pub fn remove_friend(&self, pid: u64) {
        self.send(Command::Remove(pid));
    }

    pub fn set_now_playing(&self, now_playing: Option<NowPlaying>) {
        self.send(Command::NowPlaying(now_playing));
    }

    pub fn sync_history(&self, session: PlaySession) {
        self.send(Command::SyncHistory(session));
    }

    pub fn configure(&self, preferences: Preferences) {
        self.send(Command::Configure(preferences));
    }

    pub fn take_notices(&self) -> Vec<Notice> {
        self.notices.try_iter().collect()
    }
}

struct Session {
    access: String,
    access_until: Instant,
    refresh: String,
    identity: Identity,
    scope: Option<String>,
    game: Option<(String, Instant)>,
}

struct Schedule {
    friends: Instant,
    presence: Instant,
    counts: Instant,
    health: Instant,
    standing: Instant,
    game_token: Instant,
    reconnect: Instant,
    players: Instant,
}

impl Schedule {
    fn now() -> Self {
        let now = Instant::now();
        Self {
            friends: now,
            presence: now,
            counts: now,
            health: now,
            standing: now,
            game_token: now,
            reconnect: now,
            players: now,
        }
    }
}

struct Worker {
    api: Api,
    state: Arc<Mutex<State>>,
    inbox: Receiver<Command>,
    loopback: Sender<Command>,
    notices: Sender<Notice>,
    ctx: egui::Context,
    preferences: Preferences,
    now_playing: Option<NowPlaying>,
    session: Option<Session>,
    sign_in_cancel: Option<Arc<AtomicBool>>,
    schedule: Schedule,
    known_status: HashMap<u64, i32>,
    offline_streak: HashMap<u64, u8>,
    known_requests: HashSet<u64>,
    first_poll: bool,
    published: Option<PresenceUpdate>,
    last_publish: Option<Instant>,
    avatar_queue: VecDeque<u64>,
    avatar_tried: HashSet<u64>,
    standing_supported: bool,
    last_refresh: Option<Instant>,
}

impl Worker {
    fn new(
        state: Arc<Mutex<State>>,
        inbox: Receiver<Command>,
        loopback: Sender<Command>,
        notices: Sender<Notice>,
        ctx: egui::Context,
        preferences: Preferences,
    ) -> Self {
        Self {
            api: Api::new(),
            state,
            inbox,
            loopback,
            notices,
            ctx,
            preferences,
            now_playing: None,
            session: None,
            sign_in_cancel: None,
            schedule: Schedule::now(),
            known_status: HashMap::new(),
            offline_streak: HashMap::new(),
            known_requests: HashSet::new(),
            first_poll: true,
            published: None,
            last_publish: None,
            avatar_queue: VecDeque::new(),
            avatar_tried: HashSet::new(),
            standing_supported: true,
            last_refresh: None,
        }
    }

    fn update(&self, change: impl FnOnce(&mut State)) {
        if let Ok(mut state) = self.state.lock() {
            change(&mut state);
        }
        self.ctx.request_repaint();
    }

    fn notify(&self, notice: Notice) {
        let _ = self.notices.send(notice);
        self.ctx.request_repaint();
    }

    fn run(mut self) {
        self.apply_redirect();
        self.restore();
        loop {
            match self.inbox.recv_timeout(TICK) {
                Ok(command) => self.handle(command),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.tick();
        }
        if self.session.is_some() && self.preferences.share_presence {
            let _ = self.publish(PresenceUpdate {
                status: shared::PRESENCE_OFFLINE,
                app_field: Vec::new(),
                app_id: None,
                app_name: String::new(),
            });
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::SignIn => self.begin_sign_in(),
            Command::CancelSignIn => {
                if let Some(cancel) = self.sign_in_cancel.take() {
                    cancel.store(true, Ordering::Relaxed);
                }
                if self.session.is_none() {
                    self.update(|state| state.phase = Phase::SignedOut);
                }
            }
            Command::Authorized(result) => self.complete_sign_in(result),
            Command::SignOut => self.sign_out(),
            Command::RefreshNow => {
                self.schedule = Schedule::now();
                self.avatar_tried.clear();
            }
            Command::AddFriend(code) => self.add_friend(&code),
            Command::Answer { pid, accept } => self.answer(pid, accept),
            Command::Remove(pid) => self.remove(pid),
            Command::NowPlaying(now_playing) => {
                if self.now_playing != now_playing {
                    if now_playing.is_none() {
                        shared::set_local_presence(None);
                    }
                    self.now_playing = now_playing;
                    self.schedule.presence = Instant::now();
                }
            }
            Command::SyncHistory(session) => self.push_history(session),
            Command::Configure(preferences) => {
                let stopped_sharing = self.preferences.share_presence && !preferences.share_presence;
                self.preferences = preferences;
                self.apply_redirect();
                if stopped_sharing && self.session.is_some() {
                    let _ = self.publish(PresenceUpdate {
                        status: shared::PRESENCE_OFFLINE,
                        app_field: Vec::new(),
                        app_id: None,
                        app_name: String::new(),
                    });
                }
                self.published = None;
                self.schedule.presence = Instant::now();
            }
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if now >= self.schedule.counts {
            self.schedule.counts = now + COUNTS_EVERY;
            match self.api.online_counts() {
                Ok(counts) => {
                    let total = counts.values().map(|count| *count as u64).sum::<u64>() as u32;
                    self.update(|state| {
                        state.counts = counts;
                        state.players_online = Some(total);
                    });
                }
                Err(error) => log::debug!("nextendo: online counts unavailable: {error:?}"),
            }
        }
        if now >= self.schedule.health {
            self.schedule.health = now + HEALTH_EVERY;
            let ping = self.api.ping();
            self.update(|state| match ping {
                Ok(latency) => {
                    state.latency = Some(latency);
                    state.reachable = Some(true);
                }
                Err(_) => {
                    state.latency = None;
                    state.reachable = Some(false);
                }
            });
        }
        if self.session.is_none() {
            return;
        }
        if self.session.as_ref().is_some_and(|session| session.access.is_empty()) {
            if now >= self.schedule.reconnect {
                self.schedule.reconnect = now + OFFLINE_RETRY;
                if self.refresh_session().is_ok() {
                    self.finish_sign_in(false);
                }
            }
            return;
        }
        if now >= self.schedule.game_token {
            self.refresh_game_token();
        }
        if now >= self.schedule.friends {
            self.schedule.friends = now + FRIEND_POLL;
            self.poll_friends();
        }
        if now >= self.schedule.standing && self.standing_supported {
            self.schedule.standing = now + STANDING_EVERY;
            self.check_standing();
        }
        if now >= self.schedule.players {
            self.schedule.players = now
                + if self.now_playing.is_some() {
                    Duration::from_secs(15)
                } else {
                    Duration::from_secs(60)
                };
            self.poll_players();
        }
        self.maybe_publish();
        self.fetch_avatars();
    }

    fn poll_players(&mut self) {
        let lobby = self.call(|api, token| api.lobby(token));
        let recent = self.call(|api, token| api.recent_players(token));
        let me = self.session.as_ref().map_or(0, |session| session.identity.pid);
        let mut pids: Vec<u64> = Vec::new();
        if let Ok(Some(lobby)) = &lobby {
            pids.extend(lobby.players.iter().map(|player| player.pid));
        }
        if let Ok(recent) = &recent {
            pids.extend(recent.iter().map(|player| player.pid));
        }
        for pid in pids.into_iter().filter(|pid| *pid != me) {
            self.queue_avatar(pid);
        }
        self.update(|state| {
            if let Ok(lobby) = lobby {
                state.lobby = lobby;
            }
            if let Ok(recent) = recent {
                state.recent = recent;
                state.players_loaded = true;
            }
        });
    }

    fn push_history(&mut self, session: PlaySession) {
        if self.session.is_none() || session.seconds == 0 || session.title_id == 0 {
            return;
        }
        let icon = nexium_loader::read_container_metadata(&session.path)
            .and_then(|metadata| metadata.icon_jpeg)
            .unwrap_or_default();
        let entry = HistoryEntry {
            title_id: session.title_id,
            name: session.name,
            seconds: session.seconds,
            last_played: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            icon,
        };
        match self.call(|api, token| api.sync_history(token, std::slice::from_ref(&entry))) {
            Ok(()) => log::info!("nextendo: play time synced for {:016X}", entry.title_id),
            Err(error) => log::debug!("nextendo: play time sync failed: {error:?}"),
        }
        self.schedule.players = Instant::now();
    }

    fn restore(&mut self) {
        let Some(saved) = vault::load() else {
            return;
        };
        if saved.refresh_token.is_empty() || saved.pid == 0 {
            vault::clear();
            return;
        }
        let identity = Identity {
            pid: saved.pid,
            username: saved.username.clone(),
            friend_code: saved.friend_code.clone(),
        };
        self.update(|state| {
            state.phase = Phase::Restoring;
            state.account = Some(identity.clone());
        });
        self.session = Some(Session {
            access: String::new(),
            access_until: Instant::now(),
            refresh: saved.refresh_token,
            identity,
            scope: saved.scope,
            game: None,
        });
        match self.refresh_session() {
            Ok(()) => self.finish_sign_in(false),
            Err(ApiError::InvalidGrant) => {}
            Err(error) => {
                log::info!("nextendo: restoring the session offline: {error:?}");
                self.schedule.reconnect = Instant::now() + OFFLINE_RETRY;
                self.update(|state| {
                    state.phase = Phase::SignedIn;
                    state.offline = true;
                });
            }
        }
    }

    fn begin_sign_in(&mut self) {
        if self.sign_in_cancel.is_some() {
            if let Ok(state) = self.state.lock() {
                if let Phase::Browser { url } = &state.phase {
                    if !url.is_empty() {
                        oauth::open_browser(url);
                    }
                }
            }
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.sign_in_cancel = Some(cancel.clone());
        self.update(|state| {
            state.phase = Phase::Browser { url: String::new() };
            state.sign_in_error = None;
        });
        let base = self.api.base().to_string();
        let loopback = self.loopback.clone();
        let state = self.state.clone();
        let ctx = self.ctx.clone();
        let spawned = std::thread::Builder::new()
            .name("nextendo-sign-in".into())
            .spawn(move || {
                let result = oauth::authorize(&base, &cancel, |url| {
                    if let Ok(mut state) = state.lock() {
                        state.phase = Phase::Browser { url: url.to_string() };
                    }
                    ctx.request_repaint();
                    if !oauth::open_browser(url) {
                        log::warn!("nextendo: no browser could be opened for sign-in");
                    }
                });
                let _ = loopback.send(Command::Authorized(result));
            });
        if let Err(error) = spawned {
            self.sign_in_cancel = None;
            self.update(|state| {
                state.phase = Phase::SignedOut;
                state.sign_in_error = Some(format!("NeXium couldn't start sign-in: {error}"));
            });
        }
    }

    fn complete_sign_in(&mut self, result: Result<oauth::Grant, oauth::AuthError>) {
        self.sign_in_cancel = None;
        let grant = match result {
            Ok(grant) => grant,
            Err(oauth::AuthError::Cancelled) => {
                if self.session.is_none() {
                    self.update(|state| state.phase = Phase::SignedOut);
                }
                return;
            }
            Err(error) => {
                let message = error.describe();
                self.update(|state| {
                    state.phase = Phase::SignedOut;
                    state.sign_in_error = Some(message.clone());
                });
                self.notify(Notice::SignInFailed(message));
                return;
            }
        };
        self.update(|state| state.phase = Phase::Finishing);
        let tokens = match self
            .api
            .exchange_code(&grant.code, &grant.verifier, &grant.redirect_uri)
        {
            Ok(tokens) => tokens,
            Err(error) => {
                let message = match error {
                    ApiError::InvalidGrant => {
                        "Nextendo didn't accept the sign-in code. Please try again.".to_string()
                    }
                    other => other.describe(),
                };
                self.update(|state| {
                    state.phase = Phase::SignedOut;
                    state.sign_in_error = Some(message.clone());
                });
                self.notify(Notice::SignInFailed(message));
                return;
            }
        };
        let identity = match tokens.identity.clone() {
            Some(identity) => Ok(identity),
            None => self.api.identity(&tokens.access),
        };
        let identity = match identity {
            Ok(identity) => identity,
            Err(error) => {
                let message = format!("Signed in, but your profile couldn't be loaded. {}", error.describe());
                self.update(|state| {
                    state.phase = Phase::SignedOut;
                    state.sign_in_error = Some(message.clone());
                });
                self.notify(Notice::SignInFailed(message));
                return;
            }
        };
        self.reset_friend_tracking();
        self.session = Some(Session {
            access: tokens.access,
            access_until: Instant::now() + Duration::from_secs(tokens.expires_in.max(60)),
            refresh: tokens.refresh,
            identity,
            scope: tokens.scope,
            game: None,
        });
        self.finish_sign_in(true);
    }

    fn finish_sign_in(&mut self, announce: bool) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let identity = session.identity.clone();
        self.persist();
        self.schedule = Schedule::now();
        self.queue_avatar(identity.pid);
        self.update(|state| {
            state.phase = Phase::SignedIn;
            state.account = Some(identity.clone());
            state.offline = false;
            state.sign_in_error = None;
        });
        self.refresh_game_token();
        self.sync_kernel();
        std::thread::spawn(nexium_kernel::services::nextendo_token::prepare);
        if announce {
            self.notify(Notice::SignedIn {
                name: identity.username.clone(),
            });
        }
    }

    fn persist(&self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let saved = vault::SavedSession {
            refresh_token: session.refresh.clone(),
            pid: session.identity.pid,
            username: session.identity.username.clone(),
            friend_code: session.identity.friend_code.clone(),
            scope: session.scope.clone(),
        };
        if let Err(error) = vault::save(&saved) {
            log::warn!("nextendo: could not store the session: {error}");
        }
    }

    fn sign_out(&mut self) {
        if let Some(cancel) = self.sign_in_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        if self.session.is_some() && self.preferences.share_presence {
            let _ = self.publish(PresenceUpdate {
                status: shared::PRESENCE_OFFLINE,
                app_field: Vec::new(),
                app_id: None,
                app_name: String::new(),
            });
        }
        self.session = None;
        vault::clear();
        self.clear_signed_in_state();
    }

    fn end_session(&mut self, reason: &str) {
        self.session = None;
        vault::clear();
        self.clear_signed_in_state();
        self.notify(Notice::SessionEnded {
            reason: reason.to_string(),
        });
    }

    fn clear_signed_in_state(&mut self) {
        self.reset_friend_tracking();
        self.published = None;
        self.avatar_queue.clear();
        shared::set_account(None);
        shared::set_friends(Vec::new());
        shared::set_local_presence(None);
        self.apply_redirect();
        self.update(|state| {
            let counts = std::mem::take(&mut state.counts);
            let players = state.players_online;
            let latency = state.latency;
            let reachable = state.reachable;
            *state = State {
                counts,
                players_online: players,
                latency,
                reachable,
                ..State::default()
            };
        });
    }

    fn reset_friend_tracking(&mut self) {
        self.standing_supported = true;
        self.known_status.clear();
        self.offline_streak.clear();
        self.known_requests.clear();
        self.first_poll = true;
    }

    fn apply_redirect(&self) {
        shared::set_redirect(shared::Redirect {
            enabled: self.preferences.online_play && self.session.is_some(),
            server: self.preferences.server,
            nat: self.preferences.nat,
        });
    }

    fn sync_kernel(&self) {
        let account = self.session.as_ref().map(|session| shared::LinkedAccount {
            pid: session.identity.pid,
            username: session.identity.username.clone(),
            friend_code: session.identity.friend_code.clone(),
            nex_token: session
                .game
                .as_ref()
                .map(|(token, _)| token.clone())
                .unwrap_or_default(),
        });
        shared::set_account(account);
        self.apply_redirect();
    }

    fn refresh_session(&mut self) -> Result<(), ApiError> {
        let Some(refresh) = self.session.as_ref().map(|session| session.refresh.clone()) else {
            return Err(ApiError::Unauthorized);
        };
        match self.api.refresh(&refresh) {
            Ok(tokens) => {
                self.last_refresh = Some(Instant::now());
                if let Some(session) = self.session.as_mut() {
                    session.access = tokens.access;
                    session.access_until =
                        Instant::now() + Duration::from_secs(tokens.expires_in.max(60));
                    if !tokens.refresh.is_empty() {
                        session.refresh = tokens.refresh;
                    }
                    if tokens.scope.is_some() {
                        session.scope = tokens.scope;
                    }
                }
                self.persist();
                self.update(|state| state.offline = false);
                Ok(())
            }
            Err(ApiError::InvalidGrant) => {
                self.end_session("Your Nextendo sign-in has expired. Sign in again to keep playing online.");
                Err(ApiError::InvalidGrant)
            }
            Err(error) => {
                if error == ApiError::Offline {
                    self.update(|state| state.offline = true);
                }
                Err(error)
            }
        }
    }

    fn access_token(&mut self) -> Result<String, ApiError> {
        let stale = self
            .session
            .as_ref()
            .map(|session| Instant::now() + REFRESH_MARGIN >= session.access_until)
            .ok_or(ApiError::Unauthorized)?;
        if stale {
            self.refresh_session()?;
        }
        self.session
            .as_ref()
            .map(|session| session.access.clone())
            .ok_or(ApiError::Unauthorized)
    }

    fn check_standing(&mut self) {
        let Some(game_token) = self
            .session
            .as_ref()
            .and_then(|session| session.game.as_ref())
            .map(|(token, _)| token.clone())
        else {
            self.schedule.standing = Instant::now() + OFFLINE_RETRY;
            return;
        };
        match self.api.standing(&game_token) {
            Ok(standing) => self.update(|state| state.standing = Some(standing)),
            Err(ApiError::Unauthorized) | Err(ApiError::Forbidden(_)) => {
                log::debug!("nextendo: account standing isn't available to this app");
                self.standing_supported = false;
            }
            Err(error) => log::debug!("nextendo: standing unavailable: {error:?}"),
        }
    }

    fn call<T>(&mut self, request: impl Fn(&Api, &str) -> Result<T, ApiError>) -> Result<T, ApiError> {
        let token = self.access_token()?;
        let result = match request(&self.api, &token) {
            Err(ApiError::Unauthorized)
                if self
                    .last_refresh
                    .map_or(true, |at| at.elapsed() > Duration::from_secs(60)) =>
            {
                self.refresh_session()?;
                let token = self.access_token()?;
                request(&self.api, &token)
            }
            other => other,
        };
        if matches!(result, Err(ApiError::Offline)) {
            self.update(|state| state.offline = true);
        } else if self.state.lock().is_ok_and(|state| state.offline) {
            self.update(|state| state.offline = false);
        }
        result
    }

    fn refresh_game_token(&mut self) {
        let now = Instant::now();
        let fresh = self
            .session
            .as_ref()
            .and_then(|session| session.game.as_ref())
            .is_some_and(|(_, until)| now + GAME_TOKEN_MARGIN < *until);
        if fresh {
            self.schedule.game_token = now + Duration::from_secs(10 * 60);
            return;
        }
        match self.call(|api, token| api.game_token(token)) {
            Ok(game) => {
                let until = now + Duration::from_secs(game.expires_in.max(300));
                if let Some(session) = self.session.as_mut() {
                    session.game = Some((game.token, until));
                    if let Some(identity) = game.identity.filter(|identity| identity.pid == session.identity.pid) {
                        if !identity.username.is_empty() {
                            session.identity.username = identity.username;
                        }
                        if !identity.friend_code.is_empty() {
                            session.identity.friend_code = identity.friend_code;
                        }
                    }
                }
                self.schedule.game_token = until
                    .checked_sub(GAME_TOKEN_MARGIN)
                    .unwrap_or(now + Duration::from_secs(10 * 60))
                    .max(now + Duration::from_secs(60));
                let account = self.session.as_ref().map(|session| session.identity.clone());
                self.update(|state| {
                    state.game_access = GameAccess::Ready;
                    if account.is_some() {
                        state.account = account;
                    }
                });
                self.persist();
                self.sync_kernel();
            }
            Err(ApiError::Offline) => {
                self.schedule.game_token = now + OFFLINE_RETRY;
            }
            Err(ApiError::Forbidden(_)) | Err(ApiError::Rejected { status: 403, .. }) => {
                self.schedule.game_token = now + Duration::from_secs(30 * 60);
                self.update(|state| {
                    state.game_access = GameAccess::Missing(
                        "This sign-in doesn't include online play. Sign out, sign back in and allow \"Join and host multiplayer game sessions\".".into(),
                    )
                });
            }
            Err(ApiError::InvalidGrant) => {}
            Err(error) => {
                self.schedule.game_token = now + Duration::from_secs(5 * 60);
                let message = error.describe();
                self.update(|state| state.game_access = GameAccess::Missing(message));
            }
        }
    }

    fn poll_friends(&mut self) {
        match self.call(|api, token| api.friends(token)) {
            Ok(list) => self.absorb_friends(list),
            Err(ApiError::Forbidden(_)) => self.update(|state| {
                state.friends_loaded = true;
                state.friends_issue = Some("Friends weren't shared with NeXium in this sign-in.".into());
            }),
            Err(error) => log::debug!("nextendo: friend refresh failed: {error:?}"),
        }
    }

    fn absorb_friends(&mut self, list: FriendList) {
        let announce = !self.first_poll && self.preferences.friend_alerts;
        self.first_poll = false;
        let mut notices = Vec::new();
        let mut current = HashMap::new();
        let mut views = Vec::with_capacity(list.friends.len());
        for friend in &list.friends {
            let previous = self.known_status.get(&friend.pid).copied();
            let mut status = friend.status;
            if status <= shared::PRESENCE_OFFLINE && previous.is_some_and(|previous| previous > 0) {
                let streak = self.offline_streak.entry(friend.pid).or_insert(0);
                *streak += 1;
                if *streak < OFFLINE_CONFIRMATIONS {
                    status = previous.unwrap_or_default();
                }
            } else {
                self.offline_streak.remove(&friend.pid);
                let was_offline = previous.map_or(true, |previous| previous <= 0);
                if announce && was_offline && status > 0 {
                    notices.push(Notice::FriendOnline {
                        pid: friend.pid,
                        name: friend.name.clone(),
                        app_id: friend.app_id,
                        app_name: friend.app_name.clone(),
                    });
                }
            }
            current.insert(friend.pid, status);
            views.push(Friend {
                pid: friend.pid,
                name: friend.name.clone(),
                friend_code: friend.friend_code.clone(),
                status,
                app_id: if status > 0 { friend.app_id } else { 0 },
                app_name: friend.app_name.clone(),
                console: friend.console,
                last_seen: friend.last_seen.clone(),
                staff: friend.staff,
                favorite: friend.favorite,
            });
        }
        self.known_status = current;
        let mut requests = Vec::with_capacity(list.requests.len());
        let mut known_requests = HashSet::new();
        for request in &list.requests {
            if announce && !self.known_requests.contains(&request.pid) {
                notices.push(Notice::FriendRequest {
                    pid: request.pid,
                    name: request.name.clone(),
                });
            }
            known_requests.insert(request.pid);
            requests.push(Request {
                pid: request.pid,
                name: request.name.clone(),
                friend_code: request.friend_code.clone(),
            });
        }
        self.known_requests = known_requests;
        sort_friends(&mut views);
        shared::set_friends(
            list.friends
                .iter()
                .map(|friend| shared::Friend {
                    pid: friend.pid,
                    name: friend.name.clone(),
                    status: self.known_status.get(&friend.pid).copied().unwrap_or(friend.status),
                    app_field: friend.app_field.clone(),
                    app_id: friend.app_id,
                    account_id: friend.account_id,
                    console: friend.console,
                })
                .collect(),
        );
        let mut inline_avatars = Vec::new();
        for entry in list.friends.iter().chain(list.requests.iter()) {
            if !entry.image.is_empty() {
                inline_avatars.push((entry.pid, Arc::new(entry.image.clone())));
                self.avatar_tried.insert(entry.pid);
            } else {
                self.queue_avatar(entry.pid);
            }
        }
        self.update(|state| {
            let mut changed = false;
            for (pid, image) in inline_avatars {
                if state.avatars.get(&pid).map_or(true, |current| **current != *image) {
                    state.avatars.insert(pid, image);
                    changed = true;
                }
            }
            if changed {
                state.avatar_revision += 1;
            }
            state.friends = views;
            state.requests = requests;
            state.friends_loaded = true;
            state.friends_issue = None;
            state.last_sync = Some(Instant::now());
        });
        for notice in notices {
            self.notify(notice);
        }
    }

    fn queue_avatar(&mut self, pid: u64) {
        if pid != 0 && !self.avatar_tried.contains(&pid) && !self.avatar_queue.contains(&pid) {
            self.avatar_queue.push_back(pid);
        }
    }

    fn fetch_avatars(&mut self) {
        for _ in 0..2 {
            let Some(pid) = self.avatar_queue.pop_front() else {
                return;
            };
            self.avatar_tried.insert(pid);
            match self.api.avatar(pid) {
                Ok(bytes) if !bytes.is_empty() => self.update(|state| {
                    state.avatars.insert(pid, Arc::new(bytes));
                    state.avatar_revision += 1;
                }),
                Ok(_) => {}
                Err(ApiError::Offline) => {
                    self.avatar_tried.remove(&pid);
                    return;
                }
                Err(error) => log::debug!("nextendo: no avatar for a player: {error:?}"),
            }
        }
    }

    fn desired_presence(&self) -> PresenceUpdate {
        let local = shared::local_presence();
        match &self.now_playing {
            Some(game) => PresenceUpdate {
                status: local
                    .as_ref()
                    .map_or(shared::PRESENCE_ONLINE, |presence| presence.status.max(shared::PRESENCE_ONLINE)),
                app_field: local.map(|presence| presence.app_field).unwrap_or_default(),
                app_id: Some(game.title_id),
                app_name: game.name.clone(),
            },
            None => PresenceUpdate {
                status: shared::PRESENCE_ONLINE,
                app_field: Vec::new(),
                app_id: None,
                app_name: String::new(),
            },
        }
    }

    fn publish(&mut self, update: PresenceUpdate) -> Result<(), ApiError> {
        self.call(|api, token| api.publish_presence(token, &update))
    }

    fn maybe_publish(&mut self) {
        if !self.preferences.share_presence {
            return;
        }
        let now = Instant::now();
        let desired = self.desired_presence();
        let changed = self.published.as_ref() != Some(&desired);
        let due = now >= self.schedule.presence;
        if !due && !changed {
            return;
        }
        if !due && self.last_publish.is_some_and(|last| now < last + PRESENCE_GAP) {
            return;
        }
        match self.publish(desired.clone()) {
            Ok(()) => {
                self.published = Some(desired);
                self.last_publish = Some(now);
                self.schedule.presence = now + PRESENCE_EVERY;
                self.update(|state| state.presence_issue = None);
            }
            Err(ApiError::Forbidden(_)) | Err(ApiError::Rejected { status: 403, .. }) => {
                self.schedule.presence = now + Duration::from_secs(10 * 60);
                self.published = Some(desired);
                self.update(|state| {
                    state.presence_issue =
                        Some("Your status isn't shared because this sign-in didn't include presence.".into())
                });
            }
            Err(_) => {
                self.schedule.presence = now + Duration::from_secs(15);
                self.last_publish = Some(now);
            }
        }
    }

    fn add_friend(&mut self, code: &str) {
        let Some(code) = normalize_friend_code(code) else {
            self.update(|state| {
                state.add_result = Some(Err("That doesn't look like a friend code. They look like SW-1234-5678-9012.".into()))
            });
            return;
        };
        if self
            .session
            .as_ref()
            .is_some_and(|session| normalize_friend_code(&session.identity.friend_code).as_deref() == Some(code.as_str()))
        {
            self.update(|state| state.add_result = Some(Err("That's your own friend code.".into())));
            return;
        }
        self.update(|state| {
            state.busy.insert(Busy::AddFriend);
            state.add_result = None;
        });
        let result = self.call(|api, token| api.add_friend(token, &code));
        self.update(|state| {
            state.busy.remove(&Busy::AddFriend);
            state.add_result = Some(result.clone().map_err(|error| error.describe()));
        });
        if result.is_ok() {
            self.notify(Notice::RequestSent);
            self.schedule.friends = Instant::now();
        }
    }

    fn answer(&mut self, pid: u64, accept: bool) {
        let name = self
            .state
            .lock()
            .ok()
            .and_then(|state| state.requests.iter().find(|request| request.pid == pid).map(|request| request.name.clone()))
            .unwrap_or_default();
        self.update(|state| {
            state.busy.insert(Busy::Answer(pid));
        });
        let result = self.call(|api, token| api.answer_request(token, pid, accept));
        self.update(|state| {
            state.busy.remove(&Busy::Answer(pid));
            if result.is_ok() {
                state.requests.retain(|request| request.pid != pid);
            }
        });
        match result {
            Ok(()) => {
                self.known_requests.remove(&pid);
                if accept {
                    self.notify(Notice::FriendAdded { pid, name });
                }
                self.schedule.friends = Instant::now();
            }
            Err(error) => self.notify(Notice::ActionFailed(error.describe())),
        }
    }

    fn remove(&mut self, pid: u64) {
        let name = self
            .state
            .lock()
            .ok()
            .and_then(|state| state.friends.iter().find(|friend| friend.pid == pid).map(|friend| friend.name.clone()))
            .unwrap_or_default();
        self.update(|state| {
            state.busy.insert(Busy::Remove(pid));
        });
        let result = self.call(|api, token| api.remove_friend(token, pid));
        self.update(|state| {
            state.busy.remove(&Busy::Remove(pid));
            if result.is_ok() {
                state.friends.retain(|friend| friend.pid != pid);
            }
        });
        match result {
            Ok(()) => {
                self.known_status.remove(&pid);
                self.notify(Notice::FriendRemoved { name });
                self.schedule.friends = Instant::now();
            }
            Err(error) => self.notify(Notice::ActionFailed(error.describe())),
        }
    }
}

pub fn sort_friends(friends: &mut [Friend]) {
    friends.sort_by(|a, b| {
        let rank = |friend: &Friend| (!friend.in_game(), !friend.online(), !friend.favorite);
        rank(a)
            .cmp(&rank(b))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
}

pub fn normalize_friend_code(input: &str) -> Option<String> {
    let digits: String = input.chars().filter(|ch| ch.is_ascii_digit()).collect();
    let letters: String = input
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase();
    if digits.len() != 12 || !(letters.is_empty() || letters == "SW") {
        return None;
    }
    Some(format!(
        "SW-{}-{}-{}",
        &digits[0..4],
        &digits[4..8],
        &digits[8..12]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friend_codes_normalize() {
        assert_eq!(
            normalize_friend_code("sw-1349-3479-5290").as_deref(),
            Some("SW-1349-3479-5290")
        );
        assert_eq!(
            normalize_friend_code("134934795290").as_deref(),
            Some("SW-1349-3479-5290")
        );
        assert_eq!(
            normalize_friend_code(" 1349 3479 5290 ").as_deref(),
            Some("SW-1349-3479-5290")
        );
        assert_eq!(normalize_friend_code("SW-1349-3479-529"), None);
        assert_eq!(normalize_friend_code("XX-1349-3479-5290"), None);
        assert_eq!(normalize_friend_code(""), None);
    }

    #[test]
    fn friends_sort_in_game_then_online_then_offline() {
        let friend = |name: &str, status: i32, app_id: u64| Friend {
            pid: name.len() as u64,
            name: name.into(),
            friend_code: String::new(),
            status,
            app_id,
            app_name: String::new(),
            console: false,
            last_seen: None,
            staff: false,
            favorite: name == "zed",
        };
        let mut list = vec![
            friend("zed", 0, 0),
            friend("amy", 1, 0),
            friend("bob", 2, 0x0100152000022000),
            friend("cat", 0, 0),
        ];
        sort_friends(&mut list);
        let names: Vec<&str> = list.iter().map(|friend| friend.name.as_str()).collect();
        assert_eq!(names, ["bob", "amy", "zed", "cat"]);
    }
}
