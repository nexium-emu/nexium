use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwitchButton {
    A, B, X, Y,
    L, R, ZL, ZR,
    Plus, Minus,
    DUp, DDown, DLeft, DRight,
    StickL, StickR,
    StickLUp, StickLDown, StickLLeft, StickLRight,
    StickRUp, StickRDown, StickRLeft, StickRRight,
}

impl SwitchButton {
    pub fn all() -> &'static [SwitchButton] {
        &[
            SwitchButton::A, SwitchButton::B, SwitchButton::X, SwitchButton::Y,
            SwitchButton::L, SwitchButton::R, SwitchButton::ZL, SwitchButton::ZR,
            SwitchButton::Plus, SwitchButton::Minus,
            SwitchButton::DUp, SwitchButton::DDown, SwitchButton::DLeft, SwitchButton::DRight,
            SwitchButton::StickL, SwitchButton::StickR,
            SwitchButton::StickLUp, SwitchButton::StickLDown, SwitchButton::StickLLeft, SwitchButton::StickLRight,
            SwitchButton::StickRUp, SwitchButton::StickRDown, SwitchButton::StickRLeft, SwitchButton::StickRRight,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControllerConfig {
    pub bindings: HashMap<SwitchButton, String>,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(SwitchButton::A, "Z".to_string());
        bindings.insert(SwitchButton::B, "X".to_string());
        bindings.insert(SwitchButton::X, "A".to_string());
        bindings.insert(SwitchButton::Y, "S".to_string());
        bindings.insert(SwitchButton::L, "Q".to_string());
        bindings.insert(SwitchButton::R, "W".to_string());
        bindings.insert(SwitchButton::ZL, "1".to_string());
        bindings.insert(SwitchButton::ZR, "2".to_string());
        bindings.insert(SwitchButton::Plus, "Enter".to_string());
        bindings.insert(SwitchButton::Minus, "Tab".to_string());
        bindings.insert(SwitchButton::DUp, "ArrowUp".to_string());
        bindings.insert(SwitchButton::DDown, "ArrowDown".to_string());
        bindings.insert(SwitchButton::DLeft, "ArrowLeft".to_string());
        bindings.insert(SwitchButton::DRight, "ArrowRight".to_string());
        bindings.insert(SwitchButton::StickLUp, "I".to_string());
        bindings.insert(SwitchButton::StickLDown, "K".to_string());
        bindings.insert(SwitchButton::StickLLeft, "J".to_string());
        bindings.insert(SwitchButton::StickLRight, "L".to_string());
        bindings.insert(SwitchButton::StickRUp, "Numpad8".to_string());
        bindings.insert(SwitchButton::StickRDown, "Numpad2".to_string());
        bindings.insert(SwitchButton::StickRLeft, "Numpad4".to_string());
        bindings.insert(SwitchButton::StickRRight, "Numpad6".to_string());
        bindings.insert(SwitchButton::StickL, "F".to_string());
        bindings.insert(SwitchButton::StickR, "G".to_string());
        Self { bindings }
    }
}

impl ControllerConfig {
    pub fn config_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("com", "NeXium", "NeXium")
            .map(|d| d.config_dir().join("controller.json"))
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
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "no config dir"));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, s)
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
