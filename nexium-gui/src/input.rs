use crate::app_settings::MotionRecenterButton;
use crate::controller_config::{ControllerConfig, GpButton, SwitchButton};
use sdl3::gamepad::{Axis, Button, Gamepad, GamepadType};
use sdl3::GamepadSubsystem;

const ALL_GP: [GpButton; 16] = [
    GpButton::South,
    GpButton::East,
    GpButton::West,
    GpButton::North,
    GpButton::L,
    GpButton::R,
    GpButton::ZL,
    GpButton::ZR,
    GpButton::Plus,
    GpButton::Minus,
    GpButton::LStick,
    GpButton::RStick,
    GpButton::Up,
    GpButton::Down,
    GpButton::Left,
    GpButton::Right,
];

const TRIGGER_THRESHOLD: i16 = 8000;

const MOTION_HINTS: [&str; 8] = [
    "SDL_JOYSTICK_HIDAPI",
    "SDL_JOYSTICK_HIDAPI_SWITCH",
    "SDL_JOYSTICK_HIDAPI_JOY_CONS",
    "SDL_JOYSTICK_HIDAPI_PS4",
    "SDL_JOYSTICK_HIDAPI_PS5",
    "SDL_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS",
    "SDL_JOYSTICK_HIDAPI_COMBINE_JOY_CONS",
    "SDL_JOYSTICK_ENHANCED_REPORTS",
];

const HIDAPI_DISABLED_NOTE: &str = "SDL HIDAPI disabled by environment; Joy-Con motion unavailable";

fn gp_pressed(pad: &Gamepad, b: GpButton) -> bool {
    match b {
        GpButton::South => pad.button(Button::South),
        GpButton::East => pad.button(Button::East),
        GpButton::West => pad.button(Button::West),
        GpButton::North => pad.button(Button::North),
        GpButton::L => pad.button(Button::LeftShoulder),
        GpButton::R => pad.button(Button::RightShoulder),
        GpButton::ZL => pad.axis(Axis::TriggerLeft) > TRIGGER_THRESHOLD,
        GpButton::ZR => pad.axis(Axis::TriggerRight) > TRIGGER_THRESHOLD,
        GpButton::Plus => pad.button(Button::Start),
        GpButton::Minus => pad.button(Button::Back),
        GpButton::LStick => pad.button(Button::LeftStick),
        GpButton::RStick => pad.button(Button::RightStick),
        GpButton::Up => pad.button(Button::DPadUp),
        GpButton::Down => pad.button(Button::DPadDown),
        GpButton::Left => pad.button(Button::DPadLeft),
        GpButton::Right => pad.button(Button::DPadRight),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InputSnapshot {
    pub connected: bool,
    pub buttons: u64,
    pub sticks: [i32; 4],
    pub raw_sticks: [f32; 4],
    pub home: bool,
    pub battery: Option<i32>,
    pub charging: bool,
    pub wired: bool,
}

impl InputSnapshot {
    pub fn new() -> Self {
        Self {
            connected: false,
            buttons: 0,
            sticks: [0; 4],
            raw_sticks: [0.0; 4],
            home: false,
            battery: None,
            charging: false,
            wired: false,
        }
    }

    pub fn to_npad(&self) -> (u64, [i32; 4]) {
        (self.buttons, self.sticks)
    }

    pub fn is(&self, btn: SwitchButton) -> bool {
        self.buttons & btn.npad_bit() != 0
    }

    pub fn lx(&self) -> f32 {
        self.sticks[0] as f32 / 30000.0
    }
    pub fn ly(&self) -> f32 {
        self.sticks[1] as f32 / 30000.0
    }
    pub fn rx(&self) -> f32 {
        self.sticks[2] as f32 / 30000.0
    }
    pub fn ry(&self) -> f32 {
        self.sticks[3] as f32 / 30000.0
    }
}

impl Default for InputSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InputBackend {
    rumble_out: crate::rumble_output::RumbleOutput,
    sdl: sdl3::Sdl,
    gamepad: GamepadSubsystem,
    pad: Option<Gamepad>,
    selected_id: Option<sdl3::joystick::JoystickId>,
    preferred: Option<String>,
    open_failure: Option<OpenFailure<sdl3::joystick::JoystickId>>,
    motion: crate::motion_input::MotionCapture,
    motion_pad: Option<Gamepad>,
    motion_scan_at: Option<std::time::Instant>,
    motion_enabled: bool,
    motion_probe: std::collections::HashMap<u32, bool>,
    motion_input_id: Option<u32>,
    motion_hints_logged: bool,
    motion_hint_note: Option<&'static str>,
    vertical_joycons: bool,
    rumble_strength: u8,
}

fn raw_joystick_id(id: sdl3::joystick::JoystickId) -> u32 {
    let raw: sdl3::sys::joystick::SDL_JoystickID = id.into();
    raw.0
}

fn connected_id(pad: Option<&Gamepad>) -> Option<u32> {
    pad.filter(|pad| pad.connected())
        .and_then(|pad| pad.id().ok())
        .map(raw_joystick_id)
}

fn pulse_or_rumble(
    out: &crate::rumble_output::RumbleOutput,
    pad: &mut Gamepad,
    strength: u8,
    low: u16,
    high: u16,
    ms: u32,
) {
    if strength == 0 {
        return;
    }
    match pad.id().ok().map(raw_joystick_id) {
        Some(id) if out.alive() => out.pulse(id, low, high, ms),
        _ => {
            let s = u32::from(strength.min(100));
            let _ = pad.set_rumble(
                (u32::from(low) * s / 100) as u16,
                (u32::from(high) * s / 100) as u16,
                ms,
            );
        }
    }
}

fn pick_pad<T: Copy + PartialEq>(
    ids: &[T],
    selected: Option<T>,
    preferred: impl Fn(T) -> bool,
    rank: impl Fn(T) -> Option<u8>,
    by_rank: bool,
) -> Option<T> {
    let key = |id: T| rank(id).unwrap_or(u8::MAX);
    let listed = selected.filter(|id| ids.contains(id));
    listed
        .filter(|id| preferred(*id))
        .or_else(|| ids.iter().copied().find(|id| preferred(*id)))
        .or_else(|| {
            listed
                .filter(|id| !by_rank || key(*id) == 0 || ids.iter().all(|other| key(*other) != 0))
        })
        .or_else(|| {
            if by_rank {
                ids.iter().copied().min_by_key(|id| key(*id))
            } else {
                ids.first().copied()
            }
        })
}

fn outranked<T: Copy>(ids: &[T], current: T, rank: impl Fn(T) -> Option<u8>) -> bool {
    rank(current) != Some(0) && ids.iter().any(|id| rank(*id) == Some(0))
}

struct OpenFailure<T> {
    id: T,
    pads: Vec<T>,
    count: u32,
    at: std::time::Instant,
}

impl<T: Copy + PartialEq> OpenFailure<T> {
    fn record(previous: Option<Self>, id: T, pads: &[T], now: std::time::Instant) -> Self {
        let count = previous
            .filter(|failure| failure.id == id && failure.pads == pads)
            .map_or(1, |failure| failure.count.saturating_add(1));
        Self {
            id,
            pads: pads.to_vec(),
            count,
            at: now,
        }
    }

    fn blocks(&self, id: T, pads: &[T], now: std::time::Instant) -> bool {
        self.id == id
            && self.pads == pads
            && now.saturating_duration_since(self.at) < open_backoff(self.count)
    }
}

fn open_backoff(count: u32) -> std::time::Duration {
    const STEPS: [u64; 5] = [1, 2, 5, 10, 30];
    std::time::Duration::from_secs(STEPS[(count as usize).clamp(1, STEPS.len()) - 1])
}

#[derive(Debug, PartialEq)]
enum PadStep<T> {
    Keep(T),
    Open(T),
    Wait,
    Close,
}

fn pad_step<T: Copy + PartialEq>(
    current: Option<T>,
    target: Option<T>,
    blocked: bool,
) -> PadStep<T> {
    match target {
        Some(id) if current == Some(id) => PadStep::Keep(id),
        Some(_) if blocked => PadStep::Wait,
        Some(id) => PadStep::Open(id),
        None => PadStep::Close,
    }
}

fn paddles_recenter(kind: GamepadType, vertical: bool) -> bool {
    vertical
        || !matches!(
            kind,
            GamepadType::NintendoSwitchJoyconLeft | GamepadType::NintendoSwitchJoyconRight
        )
}

fn recenter_buttons(choice: MotionRecenterButton) -> &'static [Button] {
    match choice {
        MotionRecenterButton::Paddles => &[
            Button::RightPaddle1,
            Button::RightPaddle2,
            Button::LeftPaddle1,
            Button::LeftPaddle2,
        ],
        MotionRecenterButton::Capture => &[Button::Misc1],
        MotionRecenterButton::Touchpad => &[Button::Touchpad],
        MotionRecenterButton::Unbound => &[],
    }
}

impl InputBackend {
    pub fn new(motion_enabled: bool, preferred: Option<String>) -> Option<Self> {
        sdl3::hint::set("SDL_JOYSTICK_THREAD", "1");
        sdl3::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
        sdl3::hint::set_with_priority(
            "SDL_JOYSTICK_HIDAPI_COMBINE_JOY_CONS",
            "1",
            &sdl3::hint::Hint::Override,
        );
        if motion_enabled {
            sdl3::hint::set_with_priority(
                "SDL_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS",
                "1",
                &sdl3::hint::Hint::Override,
            );
            sdl3::hint::set_with_priority(
                "SDL_JOYSTICK_ENHANCED_REPORTS",
                "1",
                &sdl3::hint::Hint::Override,
            );
        }
        let motion_hint_note = MOTION_HINTS[..3]
            .iter()
            .any(|name| sdl3::hint::get(name).as_deref() == Some("0"))
            .then_some(HIDAPI_DISABLED_NOTE);
        let vertical_joycons =
            sdl3::hint::get("SDL_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS").as_deref() == Some("1");
        let sdl = sdl3::init().ok()?;
        let gamepad = sdl.gamepad().ok()?;
        let rumble_out = crate::rumble_output::RumbleOutput::new();
        log::info!("SDL3 gamepad subsystem initialized");
        Some(Self {
            rumble_out,
            sdl,
            gamepad,
            pad: None,
            selected_id: None,
            preferred,
            open_failure: None,
            motion: crate::motion_input::MotionCapture::new(),
            motion_pad: None,
            motion_scan_at: None,
            motion_enabled,
            motion_probe: std::collections::HashMap::new(),
            motion_input_id: None,
            motion_hints_logged: false,
            motion_hint_note,
            vertical_joycons,
            rumble_strength: 0,
        })
    }

    fn sync_motion(&mut self) {
        if !self.motion_enabled {
            return;
        }
        let input_id = connected_id(self.pad.as_ref());
        let motion_pad_id = connected_id(self.motion_pad.as_ref());
        let input_candidate = input_id.filter(|id| {
            crate::motion_input::motion_rank_for_id(*id).is_some()
                && self.motion_probe.get(id) != Some(&false)
        });
        if input_id != self.motion_input_id {
            self.motion_input_id = input_id;
            if input_candidate.is_some()
                && self
                    .motion
                    .attached()
                    .is_some_and(|attached| Some(attached) != input_id)
            {
                self.motion.detach();
                self.motion_pad = None;
            }
            self.motion_scan_at = None;
        }
        let attached = self.motion.attached();
        if attached.is_some() && attached == input_id {
            return;
        }
        if let Some(current) = attached.filter(|id| Some(*id) == motion_pad_id) {
            if !self.motion_scan_due() {
                return;
            }
            let ids: Vec<u32> = self
                .gamepad
                .gamepads()
                .unwrap_or_default()
                .into_iter()
                .map(raw_joystick_id)
                .filter(|id| Some(*id) != input_id && self.motion_probe.get(id) != Some(&false))
                .collect();
            if !outranked(&ids, current, crate::motion_input::motion_rank_for_id) {
                return;
            }
        }
        if attached.is_some() {
            self.motion.detach();
            self.motion_pad = None;
            self.motion_scan_at = None;
        }
        if let (Some(id), Some(pad)) = (input_candidate, self.pad.as_ref()) {
            let name = pad.name().unwrap_or_default();
            let found = self.motion.attach(pad.raw(), &name);
            self.motion_probe.insert(id, found);
            if found {
                self.motion_pad = None;
                self.log_motion_hints_once();
                return;
            }
        }
        if !self.motion_scan_due() {
            return;
        }
        let Ok(ids) = self.gamepad.gamepads() else {
            return;
        };
        let listed: Vec<u32> = ids.iter().map(|id| raw_joystick_id(*id)).collect();
        self.motion_probe.retain(|id, _| listed.contains(id));
        let best = ids
            .into_iter()
            .filter(|id| {
                let raw = raw_joystick_id(*id);
                Some(raw) != input_id && self.motion_probe.get(&raw) != Some(&false)
            })
            .filter_map(|id| {
                crate::motion_input::motion_rank_for_id(raw_joystick_id(id)).map(|rank| (rank, id))
            })
            .min_by_key(|(rank, _)| *rank);
        let Some((_, id)) = best else {
            return;
        };
        let Ok(pad) = self.gamepad.open(id) else {
            self.motion_probe.insert(raw_joystick_id(id), false);
            return;
        };
        let name = pad.name().unwrap_or_default();
        let found = self.motion.attach(pad.raw(), &name);
        self.motion_probe.insert(raw_joystick_id(id), found);
        if found {
            self.motion_pad = Some(pad);
            self.log_motion_hints_once();
        }
    }

    fn motion_scan_due(&mut self) -> bool {
        let now = std::time::Instant::now();
        if self
            .motion_scan_at
            .is_some_and(|at| now.saturating_duration_since(at) < std::time::Duration::from_secs(1))
        {
            return false;
        }
        self.motion_scan_at = Some(now);
        true
    }

    fn log_motion_hints_once(&mut self) {
        if self.motion_hints_logged {
            return;
        }
        self.motion_hints_logged = true;
        let values: Vec<String> = MOTION_HINTS
            .iter()
            .map(|name| {
                format!(
                    "{}={}",
                    name,
                    sdl3::hint::get(name).unwrap_or_else(|| "(unset)".to_string())
                )
            })
            .collect();
        log::info!("motion: SDL hints {}", values.join(" "));
    }

    fn sync_host_kind(&self) {
        let kind = |id: Option<u32>| {
            id.map_or(
                nexium_core::hid_motion::HostKind::None,
                crate::motion_input::host_kind_for_id,
            )
        };
        let input = connected_id(self.pad.as_ref());
        nexium_core::hid_motion::set_host_kind(kind(self.motion.attached().or(input)));
        nexium_core::hid_motion::set_input_kind(kind(input));
    }

    fn sync_rumble(&mut self) {
        if !self
            .rumble_out
            .sync_pads(&[self.pad.as_ref(), self.motion_pad.as_ref()])
        {
            return;
        }
        for pad in [self.pad.as_mut(), self.motion_pad.as_mut()].into_iter().flatten() {
            let _ = pad.set_rumble(0, 0, 0);
            if pad.vendor_id() == Some(0x057E) {
                let neutral = crate::hd_rumble::neutral_for(pad.product_id().unwrap_or(0));
                let _ = pad.send_effect(&crate::hd_rumble::report(0, neutral, neutral));
            }
        }
    }

    fn ensure_pad(&mut self) {
        let ids = self.gamepad.gamepads().unwrap_or_default();
        let target_id = pick_pad(
            &ids,
            self.selected_id,
            |id| self.preferred.is_some() && self.gamepad.name_for_id(id).ok() == self.preferred,
            |id| crate::motion_input::motion_rank_for_id(raw_joystick_id(id)),
            self.motion_enabled,
        );

        let current_id = self
            .pad
            .as_ref()
            .filter(|p| p.connected())
            .and_then(|p| p.id().ok());

        if current_id.is_none() {
            self.pad = None;
        }
        let now = std::time::Instant::now();
        let blocked = target_id.is_some_and(|id| {
            self.open_failure
                .as_ref()
                .is_some_and(|failure| failure.blocks(id, &ids, now))
        });
        let id = match pad_step(current_id, target_id, blocked) {
            PadStep::Keep(id) => {
                self.selected_id = Some(id);
                return;
            }
            PadStep::Open(id) => id,
            PadStep::Wait => return,
            PadStep::Close => {
                self.pad = None;
                return;
            }
        };
        match self.gamepad.open(id) {
            Ok(p) => {
                log::info!(
                    "gamepad connected: {} (type {:?})",
                    p.name().unwrap_or_else(|| "unknown".to_string()),
                    p.r#type()
                );
                self.pad = Some(p);
                self.selected_id = Some(id);
                self.open_failure = None;
            }
            Err(e) => {
                let failure = OpenFailure::record(self.open_failure.take(), id, &ids, now);
                if failure.count == 1 {
                    log::warn!("gamepad open failed: {}", e);
                }
                self.open_failure = Some(failure);
                self.selected_id = current_id;
            }
        }
    }

    pub fn list_gamepads(&self) -> Vec<(sdl3::joystick::JoystickId, String)> {
        let mut list = Vec::new();
        if let Ok(ids) = self.gamepad.gamepads() {
            for id in ids {
                let name = if let Some(ref p) = self.pad {
                    if let Ok(pid) = p.id() {
                        if pid == id {
                            p.name().unwrap_or_else(|| "Gamepad".to_string())
                        } else {
                            self.gamepad
                                .name_for_id(id)
                                .unwrap_or_else(|_| "Gamepad".to_string())
                        }
                    } else {
                        self.gamepad
                            .name_for_id(id)
                            .unwrap_or_else(|_| "Gamepad".to_string())
                    }
                } else {
                    self.gamepad
                        .name_for_id(id)
                        .unwrap_or_else(|_| "Gamepad".to_string())
                };
                list.push((id, name));
            }
        }
        list
    }

    pub fn get_active_id(&self) -> Option<sdl3::joystick::JoystickId> {
        self.pad.as_ref().and_then(|p| p.id().ok())
    }

    pub fn set_active_id(&mut self, id: Option<sdl3::joystick::JoystickId>) -> Option<String> {
        if id.is_none() && self.preferred.is_none() {
            return None;
        }
        self.preferred = id.and_then(|id| self.gamepad.name_for_id(id).ok());
        self.selected_id = id.or(self.selected_id);
        self.open_failure = None;
        self.ensure_pad();
        self.preferred.clone()
    }

    pub fn poll(&mut self, cfg: &ControllerConfig, left_dz: f32, right_dz: f32) -> InputSnapshot {
        if std::env::var_os("NEXIUM_NO_GAMEPAD").is_some() {
            return InputSnapshot::new();
        }
        if let Ok(mut ep) = self.sdl.event_pump() {
            ep.pump_events();
        }
        self.gamepad.update();
        self.ensure_pad();
        self.sync_motion();
        self.sync_host_kind();
        self.motion.pump();
        self.sync_rumble();

        let mut snap = InputSnapshot::new();
        let Some(pad) = self.pad.as_ref() else {
            return snap;
        };
        if !pad.connected() {
            return snap;
        }
        snap.connected = true;

        let mut raw = 0u32;
        for b in ALL_GP {
            if gp_pressed(pad, b) {
                raw |= 1 << b.index();
            }
        }
        snap.buttons = cfg.gamepad_pressed(raw);
        snap.home = pad.button(Button::Guide);

        let power = pad.power_info();
        snap.battery = if power.percentage >= 0 {
            Some(power.percentage)
        } else {
            None
        };
        snap.charging = matches!(
            power.state,
            sdl3::joystick::PowerLevel::Charging | sdl3::joystick::PowerLevel::Charged
        );

        if let Ok(cs) = pad.connection_state() {
            match cs {
                sdl3::joystick::ConnectionState::Wired => {
                    snap.wired = true;
                    snap.charging = true;
                }
                sdl3::joystick::ConnectionState::Wireless => {
                    snap.wired = false;
                }
                _ => {}
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Some((cap, plugged)) = read_device_power_supply() {
                if snap.battery.is_none() {
                    snap.battery = cap;
                }
                if plugged {
                    snap.charging = true;
                    snap.wired = true;
                }
            }
        }

        let dz = |v: f32, limit: f32| if v.abs() < limit { 0.0 } else { v };
        let ax = |a: Axis| pad.axis(a) as f32 / 32768.0;
        let lx = ax(Axis::LeftX);
        let ly = -ax(Axis::LeftY);
        let rx = ax(Axis::RightX);
        let ry = -ax(Axis::RightY);
        snap.raw_sticks = [lx, ly, rx, ry];
        snap.sticks = [
            (dz(lx, left_dz) * 30000.0) as i32,
            (dz(ly, left_dz) * 30000.0) as i32,
            (dz(rx, right_dz) * 30000.0) as i32,
            (dz(ry, right_dz) * 30000.0) as i32,
        ];
        snap
    }

    pub fn first_pressed(&mut self) -> Option<GpButton> {
        self.gamepad.update();
        self.ensure_pad();
        let pad = self.pad.as_ref()?;
        if !pad.connected() {
            return None;
        }
        for b in ALL_GP {
            if gp_pressed(pad, b) {
                return Some(b);
            }
        }
        None
    }

    pub fn name(&self) -> Option<String> {
        self.pad.as_ref().and_then(|p| p.name())
    }

    pub fn rumble(&mut self, low_freq: u16, high_freq: u16, duration_ms: u32) {
        if let Some(pad) = self.pad.as_mut() {
            pulse_or_rumble(
                &self.rumble_out,
                pad,
                self.rumble_strength,
                low_freq,
                high_freq,
                duration_ms,
            );
        }
    }

    pub fn set_motion_enabled(&mut self, enabled: bool) {
        if self.motion_enabled == enabled {
            return;
        }
        self.motion_enabled = enabled;
        if !enabled {
            let raw = self
                .motion_pad
                .as_ref()
                .or(self.pad.as_ref())
                .map_or(std::ptr::null_mut(), |pad| pad.raw());
            self.motion.release(raw);
            self.motion_pad = None;
            self.motion_scan_at = None;
            self.motion_input_id = None;
        }
    }

    pub fn motion_source(&self) -> Option<String> {
        self.motion.attached()?;
        self.motion_pad.as_ref().or(self.pad.as_ref()).and_then(|pad| pad.name())
    }

    pub fn motion_attached(&self) -> bool {
        self.motion.attached().is_some()
    }

    pub fn motion_hint_note(&self) -> Option<&'static str> {
        self.motion_hint_note
    }

    fn recenter_pads(&self, choice: MotionRecenterButton) -> impl Iterator<Item = &Gamepad> {
        let vertical = self.vertical_joycons;
        [self.pad.as_ref(), self.motion_pad.as_ref()]
            .into_iter()
            .flatten()
            .filter(|pad| pad.connected())
            .filter(move |pad| {
                choice != MotionRecenterButton::Paddles || paddles_recenter(pad.r#type(), vertical)
            })
    }

    pub fn recenter_button_down(&self, choice: MotionRecenterButton) -> bool {
        let buttons = recenter_buttons(choice);
        self.recenter_pads(choice).any(|pad| buttons.iter().any(|button| pad.button(*button)))
    }

    pub fn recenter_button_available(&self, choice: MotionRecenterButton) -> bool {
        let buttons = recenter_buttons(choice);
        self.recenter_pads(choice).any(|pad| buttons.iter().any(|button| pad.has_button(*button)))
    }

    pub fn rumble_motion(&mut self, low_freq: u16, high_freq: u16, duration_ms: u32) {
        if self.motion.attached().is_none() {
            return;
        }
        if let Some(pad) = self.motion_pad.as_mut().or(self.pad.as_mut()) {
            pulse_or_rumble(
                &self.rumble_out,
                pad,
                self.rumble_strength,
                low_freq,
                high_freq,
                duration_ms,
            );
        }
    }

    pub fn set_guest_rumble(&mut self, live: bool, strength: u8) {
        self.rumble_strength = strength.min(100);
        self.rumble_out.set_guest(live, self.rumble_strength);
    }

    pub fn preview_rumble(&self, strength: u8) {
        self.rumble_out.preview(strength);
    }

    pub fn rumble_status_text(&self, enabled: bool) -> String {
        self.rumble_out.status_text(enabled)
    }

    pub fn shutdown_rumble(&mut self) {
        self.rumble_out.stop();
    }
}

pub fn is_pro_controller(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("pro controller") || n.contains("nintendo switch") || n.contains("switch pro")
}

#[cfg(target_os = "linux")]
fn read_device_power_supply() -> Option<(Option<i32>, bool)> {
    let entries = std::fs::read_dir("/sys/class/power_supply").ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        let typ = std::fs::read_to_string(p.join("type")).unwrap_or_default();
        if typ.trim() != "Battery" {
            continue;
        }
        let scope = std::fs::read_to_string(p.join("scope")).unwrap_or_default();
        let name = entry.file_name().to_string_lossy().to_lowercase();
        let is_peripheral = scope.trim() == "Device"
            || name.contains("controller")
            || name.contains("nintendo")
            || name.contains("joycon")
            || name.contains("sony")
            || name.contains("xbox")
            || name.contains("hid");
        if !is_peripheral {
            continue;
        }
        let cap = std::fs::read_to_string(p.join("capacity"))
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok());
        let status = std::fs::read_to_string(p.join("status")).unwrap_or_default();
        let st = status.trim();
        let plugged = st == "Charging" || st == "Full" || st == "Not charging";
        return Some((cap, plugged));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_joycon_paddles_recenter_only_in_vertical_mode() {
        for vertical in [false, true] {
            assert!(paddles_recenter(GamepadType::NintendoSwitchJoyconPair, vertical));
            assert!(paddles_recenter(GamepadType::NintendoSwitchPro, vertical));
            assert!(paddles_recenter(GamepadType::PS5, vertical));
            assert_eq!(paddles_recenter(GamepadType::NintendoSwitchJoyconLeft, vertical), vertical);
            assert_eq!(paddles_recenter(GamepadType::NintendoSwitchJoyconRight, vertical), vertical);
        }
    }

    #[test]
    fn pad_choice_follows_the_saved_pad_then_the_best_motion_pad() {
        let rank = |id: u32| -> Option<u8> {
            match id {
                1 => Some(2),
                2 => Some(5),
                3 => Some(0),
                7 => Some(1),
                _ => None,
            }
        };
        let unsaved = |_: u32| false;
        let saved = |id: u32| id == 3 || id == 5;
        assert_eq!(pick_pad(&[1, 2, 3], None, unsaved, rank, true), Some(3));
        assert_eq!(pick_pad(&[1, 2, 3], None, unsaved, rank, false), Some(1));
        assert_eq!(pick_pad(&[1, 2, 3], Some(1), unsaved, rank, true), Some(3));
        assert_eq!(pick_pad(&[1, 2, 3], Some(1), unsaved, rank, false), Some(1));
        assert_eq!(pick_pad(&[2, 1], Some(1), unsaved, rank, true), Some(1));
        assert_eq!(pick_pad(&[4, 6], Some(6), unsaved, rank, true), Some(6));
        assert_eq!(pick_pad(&[4, 1], Some(4), unsaved, rank, true), Some(4));
        assert_eq!(pick_pad(&[2, 1], Some(2), unsaved, rank, true), Some(2));
        assert_eq!(pick_pad(&[1, 7], Some(1), unsaved, rank, true), Some(1));
        assert_eq!(pick_pad(&[1, 7], None, unsaved, rank, true), Some(7));
        assert_eq!(pick_pad(&[4, 2, 3], Some(4), unsaved, rank, true), Some(3));
        assert_eq!(pick_pad(&[3, 7], Some(3), unsaved, rank, true), Some(3));
        assert_eq!(pick_pad(&[1, 2, 3], Some(2), |id: u32| id == 2, rank, true), Some(2));
        assert_eq!(pick_pad(&[1, 2, 4], Some(3), unsaved, rank, true), Some(1));
        assert_eq!(pick_pad(&[4, 6], None, unsaved, rank, true), Some(4));
        assert_eq!(pick_pad(&[1, 2, 3], Some(1), saved, rank, true), Some(3));
        assert_eq!(pick_pad(&[1, 3, 5], Some(5), saved, rank, true), Some(5));
        assert_eq!(pick_pad(&[1, 2], Some(1), saved, rank, true), Some(1));
        assert_eq!(pick_pad(&[], Some(1), saved, rank, true), None);
    }

    #[test]
    fn motion_pad_moves_only_to_the_joycon_pair() {
        let rank = |id: u32| -> Option<u8> {
            match id {
                1 => Some(2),
                2 => Some(5),
                3 => Some(0),
                7 => Some(1),
                8 => Some(2),
                _ => None,
            }
        };
        assert!(outranked(&[2, 1, 3], 1, rank));
        assert!(outranked(&[3], 7, rank));
        assert!(!outranked(&[2, 1, 7], 1, rank));
        assert!(!outranked(&[2, 1, 8], 1, rank));
        assert!(!outranked(&[2, 1, 4], 1, rank));
        assert!(!outranked(&[3, 7, 1], 3, rank));
        assert!(!outranked(&[], 1, rank));
    }

    #[test]
    fn automatic_choice_keeps_the_pad_in_use_until_the_pair_appears() {
        let rank = |id: u32| -> Option<u8> {
            match id {
                1 => Some(2),
                2 => Some(5),
                3 => Some(0),
                7 => Some(1),
                8 => Some(2),
                _ => None,
            }
        };
        let automatic = |ids: &[u32], in_use: u32, by_rank: bool| {
            let target = pick_pad(ids, Some(in_use), |_| false, rank, by_rank);
            pad_step(Some(in_use), target, false)
        };
        assert_eq!(automatic(&[1, 7], 1, true), PadStep::Keep(1));
        assert_eq!(automatic(&[8, 1], 1, true), PadStep::Keep(1));
        assert_eq!(automatic(&[2, 1, 7], 2, true), PadStep::Keep(2));
        assert_eq!(automatic(&[1, 4], 4, false), PadStep::Keep(4));
        assert_eq!(automatic(&[2, 1, 3], 2, false), PadStep::Keep(2));
        assert_eq!(automatic(&[2, 1, 7, 3], 2, true), PadStep::Open(3));
        assert_eq!(automatic(&[3, 1], 3, true), PadStep::Keep(3));
    }

    #[test]
    fn pad_in_use_is_kept_while_its_replacement_waits() {
        assert_eq!(pad_step(Some(4), Some(4), false), PadStep::Keep(4));
        assert_eq!(pad_step(Some(4), Some(4), true), PadStep::Keep(4));
        assert_eq!(pad_step(Some(4), Some(7), false), PadStep::Open(7));
        assert_eq!(pad_step(Some(4), Some(7), true), PadStep::Wait);
        assert_eq!(pad_step(None, Some(7), false), PadStep::Open(7));
        assert_eq!(pad_step(None, Some(7), true), PadStep::Wait);
        assert_eq!(pad_step(Some(4), None, false), PadStep::Close);
        assert_eq!(pad_step::<u32>(None, None, false), PadStep::Close);
    }

    #[test]
    fn failed_open_backs_off_until_the_pad_list_changes() {
        let start = std::time::Instant::now();
        let at = |secs: u64| start + std::time::Duration::from_secs(secs);
        let first = OpenFailure::record(None, 3, &[1, 3], start);
        assert_eq!(first.count, 1);
        assert!(first.blocks(3, &[1, 3], at(0)));
        assert!(!first.blocks(3, &[1, 3], at(1)));
        assert!(!first.blocks(1, &[1, 3], at(0)));
        assert!(!first.blocks(3, &[1, 3, 7], at(0)));
        let second = OpenFailure::record(Some(first), 3, &[1, 3], at(1));
        assert_eq!(second.count, 2);
        assert!(second.blocks(3, &[1, 3], at(2)));
        assert!(!second.blocks(3, &[1, 3], at(3)));
        let third = OpenFailure::record(Some(second), 3, &[1, 3], at(3));
        assert_eq!(third.count, 3);
        assert!(third.blocks(3, &[1, 3], at(7)));
        assert!(!third.blocks(3, &[1, 3], at(8)));
        assert_eq!(OpenFailure::record(Some(third), 3, &[3], at(8)).count, 1);
        let steps: Vec<u64> = (0..=7).map(|count| open_backoff(count).as_secs()).collect();
        assert_eq!(steps, [1, 1, 2, 5, 10, 30, 30, 30]);
    }
}
