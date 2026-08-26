use crate::ui::{Canvas, TextPainter, ACCENT};
use nexium_core::hid_state::TouchInput;
use std::collections::HashMap;

pub const OVL_A: u64 = 1 << 0;
pub const OVL_B: u64 = 1 << 1;
pub const OVL_X: u64 = 1 << 2;
pub const OVL_Y: u64 = 1 << 3;
pub const OVL_L: u64 = 1 << 6;
pub const OVL_R: u64 = 1 << 7;
pub const OVL_ZL: u64 = 1 << 8;
pub const OVL_ZR: u64 = 1 << 9;
pub const OVL_PLUS: u64 = 1 << 10;
pub const OVL_MINUS: u64 = 1 << 11;
pub const OVL_DLEFT: u64 = 1 << 12;
pub const OVL_DUP: u64 = 1 << 13;
pub const OVL_DRIGHT: u64 = 1 << 14;
pub const OVL_DDOWN: u64 = 1 << 15;

const BTN_BASE: [u8; 4] = [0x0E, 0x10, 0x18, 0x66];
const BTN_HELD: [u8; 4] = [0x53, 0xC7, 0xB4, 0xB4];
const BTN_RING: [u8; 4] = [0xFF, 0xFF, 0xFF, 0x7A];
const BTN_LABEL: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xCC];
const BTN_LABEL_HELD: [u8; 4] = [0x08, 0x1A, 0x18, 0xFF];

#[derive(Clone, Copy, PartialEq)]
enum Glyph {
    Text(&'static str),
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy)]
struct Zone {
    cx: f32,
    cy: f32,
    r: f32,
    bit: u64,
    label: Glyph,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Stick,
    Button(u64),
    GuestTouch,
    Menu,
}

pub enum OverlayEvent {
    None,
    MenuTap,
}

pub struct TouchOverlay {
    pub enabled: bool,
    w: f32,
    h: f32,
    zones: Vec<Zone>,
    stick_center: (f32, f32),
    stick_r: f32,
    menu_center: (f32, f32),
    menu_r: f32,
    roles: HashMap<i32, Role>,
    stick_vec: (f32, f32),
    buttons: u64,
    guest_touch: TouchInput,
    frame_rect: (f32, f32, f32, f32),
}

impl TouchOverlay {
    pub fn new() -> Self {
        Self {
            enabled: true,
            w: 0.0,
            h: 0.0,
            zones: Vec::new(),
            stick_center: (0.0, 0.0),
            stick_r: 0.0,
            menu_center: (0.0, 0.0),
            menu_r: 0.0,
            roles: HashMap::new(),
            stick_vec: (0.0, 0.0),
            buttons: 0,
            guest_touch: TouchInput::default(),
            frame_rect: (0.0, 0.0, 1.0, 1.0),
        }
    }

    pub fn set_frame_rect(&mut self, ox: f32, oy: f32, fw: f32, fh: f32) {
        self.frame_rect = (ox, oy, fw.max(1.0), fh.max(1.0));
    }

    pub fn layout(&mut self, w: f32, h: f32) {
        if (self.w - w).abs() < 1.0 && (self.h - h).abs() < 1.0 {
            return;
        }
        self.w = w;
        self.h = h;
        let br = h * 0.072;
        let small = h * 0.052;
        let abxy = (w * 0.885, h * 0.68);
        let spread = h * 0.145;
        let dpad = (w * 0.115, h * 0.30);
        let dspread = h * 0.105;
        self.stick_center = (w * 0.145, h * 0.72);
        self.stick_r = h * 0.155;
        self.menu_center = (w * 0.5, h * 0.06);
        self.menu_r = h * 0.048;
        self.zones = vec![
            Zone {
                cx: abxy.0 + spread,
                cy: abxy.1,
                r: br,
                bit: OVL_A,
                label: Glyph::Text("A"),
            },
            Zone {
                cx: abxy.0,
                cy: abxy.1 + spread,
                r: br,
                bit: OVL_B,
                label: Glyph::Text("B"),
            },
            Zone {
                cx: abxy.0 - spread,
                cy: abxy.1,
                r: br,
                bit: OVL_X,
                label: Glyph::Text("X"),
            },
            Zone {
                cx: abxy.0,
                cy: abxy.1 - spread,
                r: br,
                bit: OVL_Y,
                label: Glyph::Text("Y"),
            },
            Zone {
                cx: dpad.0 + dspread,
                cy: dpad.1,
                r: small,
                bit: OVL_DRIGHT,
                label: Glyph::Right,
            },
            Zone {
                cx: dpad.0 - dspread,
                cy: dpad.1,
                r: small,
                bit: OVL_DLEFT,
                label: Glyph::Left,
            },
            Zone {
                cx: dpad.0,
                cy: dpad.1 - dspread,
                r: small,
                bit: OVL_DUP,
                label: Glyph::Up,
            },
            Zone {
                cx: dpad.0,
                cy: dpad.1 + dspread,
                r: small,
                bit: OVL_DDOWN,
                label: Glyph::Down,
            },
            Zone {
                cx: w * 0.048,
                cy: h * 0.10,
                r: small,
                bit: OVL_ZL,
                label: Glyph::Text("ZL"),
            },
            Zone {
                cx: w * 0.142,
                cy: h * 0.10,
                r: small,
                bit: OVL_L,
                label: Glyph::Text("L"),
            },
            Zone {
                cx: w * 0.952,
                cy: h * 0.10,
                r: small,
                bit: OVL_ZR,
                label: Glyph::Text("ZR"),
            },
            Zone {
                cx: w * 0.858,
                cy: h * 0.10,
                r: small,
                bit: OVL_R,
                label: Glyph::Text("R"),
            },
            Zone {
                cx: w * 0.575,
                cy: h * 0.07,
                r: small * 0.82,
                bit: OVL_PLUS,
                label: Glyph::Text("+"),
            },
            Zone {
                cx: w * 0.425,
                cy: h * 0.07,
                r: small * 0.82,
                bit: OVL_MINUS,
                label: Glyph::Text("-"),
            },
        ];
    }

    fn role_at(&self, x: f32, y: f32) -> Role {
        let md = ((x - self.menu_center.0).powi(2) + (y - self.menu_center.1).powi(2)).sqrt();
        if md < self.menu_r * 1.4 {
            return Role::Menu;
        }
        if self.enabled {
            for z in &self.zones {
                let d = ((x - z.cx).powi(2) + (y - z.cy).powi(2)).sqrt();
                if d < z.r * 1.35 {
                    return Role::Button(z.bit);
                }
            }
            let sd = ((x - self.stick_center.0).powi(2) + (y - self.stick_center.1).powi(2)).sqrt();
            if sd < self.stick_r * 1.5 {
                return Role::Stick;
            }
        }
        Role::GuestTouch
    }

    pub fn pointer_down(&mut self, id: i32, x: f32, y: f32) -> OverlayEvent {
        let role = self.role_at(x, y);
        self.roles.insert(id, role);
        match role {
            Role::Button(bit) => self.buttons |= bit,
            Role::Stick => self.update_stick(x, y),
            Role::GuestTouch => self.update_guest(x, y, true),
            Role::Menu => {}
        }
        OverlayEvent::None
    }

    pub fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
        match self.roles.get(&id).copied() {
            Some(Role::Stick) => self.update_stick(x, y),
            Some(Role::GuestTouch) => self.update_guest(x, y, true),
            _ => {}
        }
    }

    pub fn pointer_up(&mut self, id: i32, x: f32, y: f32) -> OverlayEvent {
        match self.roles.remove(&id) {
            Some(Role::Button(bit)) => {
                self.buttons &= !bit;
                OverlayEvent::None
            }
            Some(Role::Stick) => {
                self.stick_vec = (0.0, 0.0);
                OverlayEvent::None
            }
            Some(Role::GuestTouch) => {
                self.update_guest(x, y, false);
                OverlayEvent::None
            }
            Some(Role::Menu) => {
                let md =
                    ((x - self.menu_center.0).powi(2) + (y - self.menu_center.1).powi(2)).sqrt();
                if md < self.menu_r * 1.8 {
                    OverlayEvent::MenuTap
                } else {
                    OverlayEvent::None
                }
            }
            None => OverlayEvent::None,
        }
    }

    pub fn cancel_all(&mut self) {
        self.roles.clear();
        self.buttons = 0;
        self.stick_vec = (0.0, 0.0);
        self.guest_touch.pressed = false;
    }

    fn update_stick(&mut self, x: f32, y: f32) {
        let dx = (x - self.stick_center.0) / self.stick_r;
        let dy = (y - self.stick_center.1) / self.stick_r;
        let mag = (dx * dx + dy * dy).sqrt();
        if mag > 1.0 {
            self.stick_vec = (dx / mag, dy / mag);
        } else {
            self.stick_vec = (dx, dy);
        }
    }

    fn update_guest(&mut self, x: f32, y: f32, pressed: bool) {
        let (ox, oy, fw, fh) = self.frame_rect;
        let gx = ((x - ox) / fw * 1280.0).clamp(0.0, 1279.0);
        let gy = ((y - oy) / fh * 720.0).clamp(0.0, 719.0);
        self.guest_touch = TouchInput {
            x: gx as u32,
            y: gy as u32,
            pressed,
        };
    }

    pub fn buttons(&self) -> u64 {
        self.buttons
    }

    pub fn stick(&self) -> (f32, f32) {
        self.stick_vec
    }

    pub fn guest_touch(&self) -> TouchInput {
        self.guest_touch
    }

    pub fn render(&self, canvas: &mut Canvas, text: &mut TextPainter) {
        let h = self.h;
        if self.enabled {
            for z in &self.zones {
                let held = self.buttons & z.bit != 0;
                canvas.fill_circle(z.cx, z.cy, z.r, if held { BTN_HELD } else { BTN_BASE });
                canvas.stroke_circle(z.cx, z.cy, z.r, (z.r * 0.06).max(2.0), BTN_RING);
                let ink = if held { BTN_LABEL_HELD } else { BTN_LABEL };
                match z.label {
                    Glyph::Text(label) => {
                        let px = z.r * 0.92;
                        let tw = text.measure(label, px, false);
                        text.draw(canvas, z.cx - tw * 0.5, z.cy - px * 0.62, px, ink, label);
                    }
                    arrow => {
                        let s = z.r * 0.42;
                        let pts = match arrow {
                            Glyph::Up => [
                                (z.cx, z.cy - s),
                                (z.cx - s * 0.9, z.cy + s * 0.6),
                                (z.cx + s * 0.9, z.cy + s * 0.6),
                            ],
                            Glyph::Down => [
                                (z.cx, z.cy + s),
                                (z.cx - s * 0.9, z.cy - s * 0.6),
                                (z.cx + s * 0.9, z.cy - s * 0.6),
                            ],
                            Glyph::Left => [
                                (z.cx - s, z.cy),
                                (z.cx + s * 0.6, z.cy - s * 0.9),
                                (z.cx + s * 0.6, z.cy + s * 0.9),
                            ],
                            _ => [
                                (z.cx + s, z.cy),
                                (z.cx - s * 0.6, z.cy - s * 0.9),
                                (z.cx - s * 0.6, z.cy + s * 0.9),
                            ],
                        };
                        canvas.fill_triangle(pts, ink);
                    }
                }
            }
            let (scx, scy) = self.stick_center;
            canvas.fill_circle(scx, scy, self.stick_r, [0x0E, 0x10, 0x18, 0x3C]);
            canvas.stroke_circle(scx, scy, self.stick_r, 3.0, BTN_RING);
            let knob = (
                scx + self.stick_vec.0 * self.stick_r * 0.6,
                scy + self.stick_vec.1 * self.stick_r * 0.6,
            );
            let active = self.stick_vec.0 != 0.0 || self.stick_vec.1 != 0.0;
            canvas.fill_circle(
                knob.0,
                knob.1,
                self.stick_r * 0.44,
                if active { BTN_HELD } else { BTN_BASE },
            );
            canvas.stroke_circle(knob.0, knob.1, self.stick_r * 0.44, 2.5, BTN_RING);
        }
        let (mcx, mcy) = self.menu_center;
        canvas.fill_circle(mcx, mcy, self.menu_r, BTN_BASE);
        canvas.stroke_circle(mcx, mcy, self.menu_r, 2.0, BTN_RING);
        let bar_w = self.menu_r * 0.9;
        for i in 0..3 {
            canvas.fill_rect(
                (mcx - bar_w * 0.5) as i32,
                (mcy - h * 0.011 + i as f32 * h * 0.011) as i32,
                bar_w as i32,
                (h * 0.004).max(2.0) as i32,
                ACCENT,
            );
        }
    }
}
