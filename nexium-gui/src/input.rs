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
    pub home: bool,
}

impl InputSnapshot {
    pub fn new() -> Self {
        Self {
            connected: false,
            buttons: 0,
            sticks: [0; 4],
            home: false,
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
    gamepad: GamepadSubsystem,
    pad: Option<Gamepad>,
}

impl InputBackend {
    pub fn new() -> Option<Self> {
        let sdl = sdl3::init().ok()?;
        let gamepad = sdl.gamepad().ok()?;
        log::info!("SDL3 gamepad subsystem initialized");
        Some(Self { gamepad, pad: None })
    }

    fn ensure_pad(&mut self) {
        let alive = self.pad.as_ref().map(|p| p.connected()).unwrap_or(false);
        if alive {
            return;
        }
        self.pad = None;
        if let Ok(ids) = self.gamepad.gamepads() {
            if let Some(&id) = ids.first() {
                match self.gamepad.open(id) {
                    Ok(p) => {
                        log::info!(
                            "gamepad connected: {} (type {:?})",
                            p.name().unwrap_or_else(|| "unknown".to_string()),
                            p.r#type()
                        );
                        self.pad = Some(p);
                    }
                    Err(e) => log::warn!("gamepad open failed: {}", e),
                }
            }
        }
    }

    pub fn poll(&mut self, cfg: &ControllerConfig) -> InputSnapshot {
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

        let dz = |v: f32| if v.abs() < 0.12 { 0.0 } else { v };
        let ax = |a: Axis| pad.axis(a) as f32 / 32768.0;
        snap.sticks = [
            (dz(ax(Axis::LeftX)) * 30000.0) as i32,
            (dz(-ax(Axis::LeftY)) * 30000.0) as i32,
            (dz(ax(Axis::RightX)) * 30000.0) as i32,
            (dz(-ax(Axis::RightY)) * 30000.0) as i32,
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
}

pub fn is_pro_controller(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("pro controller") || n.contains("nintendo switch") || n.contains("switch pro")
}
