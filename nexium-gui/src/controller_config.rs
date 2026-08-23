use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwitchButton {
    A,
    B,
    X,
    Y,
    L,
    R,
    ZL,
    ZR,
    Plus,
    Minus,
    DUp,
    DDown,
    DLeft,
    DRight,
    StickL,
    StickR,
    StickLUp,
    StickLDown,
    StickLLeft,
    StickLRight,
    StickRUp,
    StickRDown,
    StickRLeft,
    StickRRight,
}

impl SwitchButton {
    pub fn all() -> &'static [SwitchButton] {
        &[
            SwitchButton::A,
            SwitchButton::B,
            SwitchButton::X,
            SwitchButton::Y,
            SwitchButton::L,
            SwitchButton::R,
            SwitchButton::ZL,
            SwitchButton::ZR,
            SwitchButton::Plus,
            SwitchButton::Minus,
            SwitchButton::DUp,
            SwitchButton::DDown,
            SwitchButton::DLeft,
            SwitchButton::DRight,
            SwitchButton::StickL,
            SwitchButton::StickR,
            SwitchButton::StickLUp,
            SwitchButton::StickLDown,
            SwitchButton::StickLLeft,
            SwitchButton::StickLRight,
            SwitchButton::StickRUp,
            SwitchButton::StickRDown,
            SwitchButton::StickRLeft,
            SwitchButton::StickRRight,
        ]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            SwitchButton::A => "A",
            SwitchButton::B => "B",
            SwitchButton::X => "X",
            SwitchButton::Y => "Y",
            SwitchButton::L => "L",
            SwitchButton::R => "R",
            SwitchButton::ZL => "ZL",
            SwitchButton::ZR => "ZR",
            SwitchButton::Plus => "Plus (+)",
            SwitchButton::Minus => "Minus (-)",
            SwitchButton::DUp => "D-Pad Up",
            SwitchButton::DDown => "D-Pad Down",
            SwitchButton::DLeft => "D-Pad Left",
            SwitchButton::DRight => "D-Pad Right",
            SwitchButton::StickL => "Left Stick Click",
            SwitchButton::StickR => "Right Stick Click",
            SwitchButton::StickLUp => "Left Stick Up",
            SwitchButton::StickLDown => "Left Stick Down",
            SwitchButton::StickLLeft => "Left Stick Left",
            SwitchButton::StickLRight => "Left Stick Right",
            SwitchButton::StickRUp => "Right Stick Up",
            SwitchButton::StickRDown => "Right Stick Down",
            SwitchButton::StickRLeft => "Right Stick Left",
            SwitchButton::StickRRight => "Right Stick Right",
        }
    }

    pub fn npad_bit(&self) -> u64 {
        match self {
            SwitchButton::A => 1 << 0,
            SwitchButton::B => 1 << 1,
            SwitchButton::X => 1 << 2,
            SwitchButton::Y => 1 << 3,
            SwitchButton::StickL => 1 << 4,
            SwitchButton::StickR => 1 << 5,
            SwitchButton::L => 1 << 6,
            SwitchButton::R => 1 << 7,
            SwitchButton::ZL => 1 << 8,
            SwitchButton::ZR => 1 << 9,
            SwitchButton::Plus => 1 << 10,
            SwitchButton::Minus => 1 << 11,
            SwitchButton::DLeft => 1 << 12,
            SwitchButton::DUp => 1 << 13,
            SwitchButton::DRight => 1 << 14,
            SwitchButton::DDown => 1 << 15,
            SwitchButton::StickLLeft => 1 << 16,
            SwitchButton::StickLUp => 1 << 17,
            SwitchButton::StickLRight => 1 << 18,
            SwitchButton::StickLDown => 1 << 19,
            SwitchButton::StickRLeft => 1 << 20,
            SwitchButton::StickRUp => 1 << 21,
            SwitchButton::StickRRight => 1 << 22,
            SwitchButton::StickRDown => 1 << 23,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpButton {
    South,
    East,
    West,
    North,
    L,
    R,
    ZL,
    ZR,
    Plus,
    Minus,
    LStick,
    RStick,
    Up,
    Down,
    Left,
    Right,
}

impl GpButton {
    pub fn index(&self) -> u32 {
        *self as u32
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            GpButton::South => "Button ▽",
            GpButton::East => "Button ▷",
            GpButton::West => "Button ◁",
            GpButton::North => "Button △",
            GpButton::L => "L Bumper",
            GpButton::R => "R Bumper",
            GpButton::ZL => "L Trigger",
            GpButton::ZR => "R Trigger",
            GpButton::Plus => "Start",
            GpButton::Minus => "Back",
            GpButton::LStick => "L Stick Click",
            GpButton::RStick => "R Stick Click",
            GpButton::Up => "Pad Up",
            GpButton::Down => "Pad Down",
            GpButton::Left => "Pad Left",
            GpButton::Right => "Pad Right",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControllerConfig {
    pub bindings: HashMap<SwitchButton, String>,
    #[serde(default = "ControllerConfig::default_pad")]
    pub pad: HashMap<SwitchButton, GpButton>,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(SwitchButton::A, "C".to_string());
        bindings.insert(SwitchButton::B, "X".to_string());
        bindings.insert(SwitchButton::X, "V".to_string());
        bindings.insert(SwitchButton::Y, "Z".to_string());
        bindings.insert(SwitchButton::L, "Q".to_string());
        bindings.insert(SwitchButton::R, "E".to_string());
        bindings.insert(SwitchButton::ZL, "R".to_string());
        bindings.insert(SwitchButton::ZR, "T".to_string());
        bindings.insert(SwitchButton::Plus, "M".to_string());
        bindings.insert(SwitchButton::Minus, "N".to_string());
        bindings.insert(SwitchButton::DUp, "ArrowUp".to_string());
        bindings.insert(SwitchButton::DDown, "ArrowDown".to_string());
        bindings.insert(SwitchButton::DLeft, "ArrowLeft".to_string());
        bindings.insert(SwitchButton::DRight, "ArrowRight".to_string());
        bindings.insert(SwitchButton::StickLUp, "W".to_string());
        bindings.insert(SwitchButton::StickLDown, "S".to_string());
        bindings.insert(SwitchButton::StickLLeft, "A".to_string());
        bindings.insert(SwitchButton::StickLRight, "D".to_string());
        bindings.insert(SwitchButton::StickRUp, "I".to_string());
        bindings.insert(SwitchButton::StickRDown, "K".to_string());
        bindings.insert(SwitchButton::StickRLeft, "J".to_string());
        bindings.insert(SwitchButton::StickRRight, "L".to_string());
        bindings.insert(SwitchButton::StickL, "F".to_string());
        bindings.insert(SwitchButton::StickR, "G".to_string());
        Self {
            bindings,
            pad: Self::default_pad(),
        }
    }
}

impl ControllerConfig {
    pub fn config_path() -> Option<PathBuf> {
        Some(nexium_common::paths::root().join("controller.json"))
    }

    pub fn load() -> Self {
        if let Some(path) = Self::config_path() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(cfg) = serde_json::from_str::<ControllerConfig>(&s) {
                    return cfg;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::config_path() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "no config dir",
            ));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, s)
    }

    pub fn default_pad() -> HashMap<SwitchButton, GpButton> {
        let mut m = HashMap::new();
        m.insert(SwitchButton::A, GpButton::East);
        m.insert(SwitchButton::B, GpButton::South);
        m.insert(SwitchButton::X, GpButton::North);
        m.insert(SwitchButton::Y, GpButton::West);
        m.insert(SwitchButton::L, GpButton::L);
        m.insert(SwitchButton::R, GpButton::R);
        m.insert(SwitchButton::ZL, GpButton::ZL);
        m.insert(SwitchButton::ZR, GpButton::ZR);
        m.insert(SwitchButton::Plus, GpButton::Plus);
        m.insert(SwitchButton::Minus, GpButton::Minus);
        m.insert(SwitchButton::StickL, GpButton::LStick);
        m.insert(SwitchButton::StickR, GpButton::RStick);
        m.insert(SwitchButton::DUp, GpButton::Up);
        m.insert(SwitchButton::DDown, GpButton::Down);
        m.insert(SwitchButton::DLeft, GpButton::Left);
        m.insert(SwitchButton::DRight, GpButton::Right);
        m
    }

    pub fn pad_list() -> &'static [SwitchButton] {
        &[
            SwitchButton::A,
            SwitchButton::B,
            SwitchButton::X,
            SwitchButton::Y,
            SwitchButton::L,
            SwitchButton::R,
            SwitchButton::ZL,
            SwitchButton::ZR,
            SwitchButton::Plus,
            SwitchButton::Minus,
            SwitchButton::StickL,
            SwitchButton::StickR,
            SwitchButton::DUp,
            SwitchButton::DDown,
            SwitchButton::DLeft,
            SwitchButton::DRight,
        ]
    }

    pub fn pad_for(&self, btn: SwitchButton) -> Option<GpButton> {
        self.pad.get(&btn).copied()
    }

    pub fn set_pad(&mut self, btn: SwitchButton, gp: GpButton) {
        self.pad.insert(btn, gp);
    }

    pub fn gamepad_pressed(&self, raw: u32) -> u64 {
        let mut b: u64 = 0;
        for btn in SwitchButton::all() {
            if let Some(gp) = self.pad.get(btn) {
                if raw & (1 << gp.index()) != 0 {
                    b |= btn.npad_bit();
                }
            }
        }
        b
    }

    pub fn binding_for(&self, btn: SwitchButton) -> Option<&str> {
        self.bindings.get(&btn).map(|s| s.as_str())
    }

    pub fn set_binding(&mut self, btn: SwitchButton, key: String) {
        self.bindings.insert(btn, key);
    }

    pub fn buttons_pressed(&self, pressed_keys: &[String]) -> (u64, [i32; 4]) {
        let mut buttons: u64 = 0;
        let mut sticks = [0i32; 4];

        for btn in SwitchButton::all() {
            if let Some(key) = self.binding_for(*btn) {
                if pressed_keys.iter().any(|k| k.eq_ignore_ascii_case(key)) {
                    buttons |= btn.npad_bit();
                    match btn {
                        SwitchButton::StickLLeft => sticks[0] = -30000,
                        SwitchButton::StickLRight => sticks[0] = 30000,
                        SwitchButton::StickLUp => sticks[1] = 30000,
                        SwitchButton::StickLDown => sticks[1] = -30000,
                        SwitchButton::StickRLeft => sticks[2] = -30000,
                        SwitchButton::StickRRight => sticks[2] = 30000,
                        SwitchButton::StickRUp => sticks[3] = 30000,
                        SwitchButton::StickRDown => sticks[3] = -30000,
                        _ => {}
                    }
                }
            }
        }

        (buttons, sticks)
    }
}
