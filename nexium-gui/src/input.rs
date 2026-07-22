use crate::controller_config::{ControllerConfig, GpButton, SwitchButton};
use sdl3::gamepad::{Axis, Button, Gamepad};
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
    sdl: sdl3::Sdl,
    gamepad: GamepadSubsystem,
    pad: Option<Gamepad>,
    selected_id: Option<sdl3::joystick::JoystickId>,
}

impl InputBackend {
    pub fn new() -> Option<Self> {
        sdl3::hint::set("SDL_JOYSTICK_THREAD", "1");
        sdl3::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
        let sdl = sdl3::init().ok()?;
        let gamepad = sdl.gamepad().ok()?;
        log::info!("SDL3 gamepad subsystem initialized");
        Some(Self {
            sdl,
            gamepad,
            pad: None,
            selected_id: None,
        })
    }

    fn ensure_pad(&mut self) {
        let mut target_id = self.selected_id;
        if target_id.is_none() {
            if let Ok(ids) = self.gamepad.gamepads() {
                target_id = ids.first().copied();
            }
        }

        let current_ok = if let Some(ref p) = self.pad {
            if p.connected() {
                if let Ok(curr_id) = p.id() {
                    Some(curr_id) == target_id
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if current_ok {
            return;
        }

        self.pad = None;
        if let Some(id) = target_id {
            match self.gamepad.open(id) {
                Ok(p) => {
                    log::info!(
                        "gamepad connected: {} (type {:?})",
                        p.name().unwrap_or_else(|| "unknown".to_string()),
                        p.r#type()
                    );
                    self.pad = Some(p);
                    self.selected_id = Some(id);
                }
                Err(e) => {
                    log::warn!("gamepad open failed: {}", e);
                    self.selected_id = None;
                }
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

    pub fn set_active_id(&mut self, id: sdl3::joystick::JoystickId) {
        self.selected_id = Some(id);
        self.ensure_pad();
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
        if let Some(ref mut pad) = self.pad {
            let _ = pad.set_rumble(low_freq, high_freq, duration_ms);
        }
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
