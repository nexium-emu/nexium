use crate::controller_config::SwitchButton;
use crate::input::InputSnapshot;
use egui::{Color32, FontId, Rounding, Stroke, Vec2};

#[derive(Clone, Copy, PartialEq)]
enum Layer {
    Lower,
    Upper,
    Symbol,
}

#[derive(Clone, Copy, PartialEq)]
enum Special {
    Shift,
    ToggleLayer,
    Space,
    Backspace,
    Accept,
    Cancel,
}

#[derive(Clone, Copy)]
enum Key {
    Char(char),
    Fun(Special),
}

pub enum VkResult {
    None,
    Accept,
    Cancel,
}

const LOWER: [&str; 4] = ["1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.?"];
const UPPER: [&str; 4] = ["!@#$%^&*()", "QWERTYUIOP", "ASDFGHJKL\"", "ZXCVBNM;:/"];
const SYMBOL: [&str; 4] = ["+-*/=_<>[]", "{}\\|~`:;-+", "!@#$%^&()?", ".,'\"~=_<>|"];

pub struct VirtualKeyboard {
    pub open: bool,
    anim: f32,
    row: usize,
    col: usize,
    layer: Layer,
    caps: bool,
    caret: usize,
    max: usize,
    orig: String,
    prev_a: bool,
    prev_b: bool,
    prev_x: bool,
    prev_y: bool,
    prev_l: bool,
    prev_r: bool,
    prev_start: bool,
    nav_dir: u8,
    nav_since: f64,
    nav_cd: f64,
}

impl Default for VirtualKeyboard {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualKeyboard {
    pub fn new() -> Self {
        Self {
            open: false,
            anim: 0.0,
            row: 1,
            col: 0,
            layer: Layer::Lower,
            caps: false,
            caret: 0,
            max: 64,
            orig: String::new(),
            prev_a: false,
            prev_b: false,
            prev_x: false,
            prev_y: false,
            prev_l: false,
            prev_r: false,
            prev_start: false,
            nav_dir: 0,
            nav_since: 0.0,
            nav_cd: 0.0,
        }
    }

    pub fn active(&self) -> bool {
        self.open || self.anim > 0.004
    }

    pub fn show(&mut self, current: &str, max: usize) {
        self.open = true;
        self.orig = current.to_string();
        self.caret = current.chars().count();
        self.max = max;
        self.layer = Layer::Lower;
        self.caps = false;
        self.row = 1;
        self.col = 0;
        self.nav_dir = 0;
    }

    fn rows(&self) -> [&'static str; 4] {
        match self.layer {
            Layer::Symbol => SYMBOL,
            _ => {
                if self.caps {
                    UPPER
                } else {
                    LOWER
                }
            }
        }
    }

    fn fun_row(&self) -> [Key; 5] {
        [
            Key::Fun(Special::Shift),
            Key::Fun(Special::ToggleLayer),
            Key::Fun(Special::Space),
            Key::Fun(Special::Backspace),
            Key::Fun(Special::Accept),
        ]
    }

    fn key_at(&self, row: usize, col: usize) -> Option<Key> {
        if row < 4 {
            self.rows()[row].chars().nth(col).map(Key::Char)
        } else {
            self.fun_row().get(col).copied()
        }
    }

    fn row_len(&self, row: usize) -> usize {
        if row < 4 {
            self.rows()[row].chars().count()
        } else {
            5
        }
    }

    fn insert_str(&mut self, buf: &mut String, s: &str) {
        let mut chars: Vec<char> = buf.chars().collect();
        for ch in s.chars() {
            if chars.len() >= self.max {
                break;
            }
            chars.insert(self.caret.min(chars.len()), ch);
            self.caret += 1;
        }
        *buf = chars.into_iter().collect();
    }

    fn backspace(&mut self, buf: &mut String) {
        if self.caret == 0 {
            return;
        }
        let mut chars: Vec<char> = buf.chars().collect();
        if self.caret <= chars.len() {
            chars.remove(self.caret - 1);
            self.caret -= 1;
        }
        *buf = chars.into_iter().collect();
    }

    fn activate(&mut self, key: Key, buf: &mut String) -> VkResult {
        match key {
            Key::Char(c) => {
                let mut tmp = [0u8; 4];
                let s = c.encode_utf8(&mut tmp).to_string();
                self.insert_str(buf, &s);
                if self.caps && self.layer != Layer::Symbol {
                    self.caps = false;
                }
                VkResult::None
            }
            Key::Fun(sp) => match sp {
                Special::Shift => {
                    self.caps = !self.caps;
                    if self.layer == Layer::Symbol {
                        self.layer = Layer::Lower;
                    }
                    VkResult::None
                }
                Special::ToggleLayer => {
                    self.layer = if self.layer == Layer::Symbol {
                        Layer::Lower
                    } else {
                        Layer::Symbol
                    };
                    VkResult::None
                }
                Special::Space => {
                    self.insert_str(buf, " ");
                    VkResult::None
                }
                Special::Backspace => {
                    self.backspace(buf);
                    VkResult::None
                }
                Special::Accept => VkResult::Accept,
                Special::Cancel => VkResult::Cancel,
            },
        }
    }

    pub fn update(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        buf: &mut String,
        last_input: &InputSnapshot,
        accent: Color32,
        light: bool,
    ) -> VkResult {
        let target = if self.open { 1.0 } else { 0.0 };
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        self.anim += (target - self.anim) * (dt * 14.0).min(1.0);
        if self.anim < 0.004 {
            return VkResult::None;
        }
        let ease = {
            let a = self.anim.clamp(0.0, 1.0);
            a * a * (3.0 - 2.0 * a)
        };
        let now = ctx.input(|i| i.time);

        let mut result = VkResult::None;

        if self.open {
            let full = ctx.screen_rect();
            let blocker = egui::LayerId::new(egui::Order::Debug, egui::Id::new("vkb_blocker"));
            ui.with_layer_id(blocker, |ui| {
                ui.allocate_rect(full, egui::Sense::click_and_drag());
            });
            let events = ctx.input(|i| i.events.clone());
            for ev in &events {
                match ev {
                    egui::Event::Text(t) => {
                        let clean: String = t.chars().filter(|c| !c.is_control()).collect();
                        if !clean.is_empty() {
                            self.insert_str(buf, &clean);
                        }
                    }
                    egui::Event::Key {
                        key: egui::Key::Backspace,
                        pressed: true,
                        ..
                    } => self.backspace(buf),
                    egui::Event::Key {
                        key: egui::Key::Enter,
                        pressed: true,
                        ..
                    } => result = VkResult::Accept,
                    egui::Event::Key {
                        key: egui::Key::Escape,
                        pressed: true,
                        ..
                    } => result = VkResult::Cancel,
                    egui::Event::Key {
                        key: egui::Key::ArrowLeft,
                        pressed: true,
                        ..
                    } => {
                        if self.caret > 0 {
                            self.caret -= 1;
                        }
                    }
                    egui::Event::Key {
                        key: egui::Key::ArrowRight,
                        pressed: true,
                        ..
                    } => {
                        let n = buf.chars().count();
                        if self.caret < n {
                            self.caret += 1;
                        }
                    }
                    _ => {}
                }
            }

            let mut mv = (0i32, 0i32);
            let cur_dir: u8 = if last_input.is(SwitchButton::DUp) || last_input.ly() > 0.5 {
                1
            } else if last_input.is(SwitchButton::DDown) || last_input.ly() < -0.5 {
                2
            } else if last_input.is(SwitchButton::DLeft) || last_input.lx() < -0.5 {
                3
            } else if last_input.is(SwitchButton::DRight) || last_input.lx() > 0.5 {
                4
            } else {
                0
            };
            if cur_dir == 0 {
                self.nav_dir = 0;
            } else if cur_dir != self.nav_dir {
                self.nav_dir = cur_dir;
                self.nav_since = now;
                self.nav_cd = now;
                match cur_dir {
                    1 => mv.1 = -1,
                    2 => mv.1 = 1,
                    3 => mv.0 = -1,
                    _ => mv.0 = 1,
                }
            } else if now - self.nav_since >= 0.42 && now - self.nav_cd >= 0.11 {
                self.nav_cd = now;
                match cur_dir {
                    1 => mv.1 = -1,
                    2 => mv.1 = 1,
                    3 => mv.0 = -1,
                    _ => mv.0 = 1,
                }
            }
            if mv.1 != 0 {
                let nr = (self.row as i32 + mv.1).clamp(0, 4) as usize;
                if nr != self.row {
                    self.row = nr;
                    let len = self.row_len(self.row);
                    self.col = self.col.min(len.saturating_sub(1));
                    crate::ui_audio::play_move();
                }
            }
            if mv.0 != 0 {
                let len = self.row_len(self.row) as i32;
                self.col = ((self.col as i32 + mv.0).rem_euclid(len)) as usize;
                crate::ui_audio::play_move();
            }

            let a = last_input.connected && last_input.is(SwitchButton::A);
            let b = last_input.connected && last_input.is(SwitchButton::B);
            let x = last_input.connected && last_input.is(SwitchButton::X);
            let y = last_input.connected && last_input.is(SwitchButton::Y);
            let l = last_input.connected && last_input.is(SwitchButton::L);
            let r = last_input.connected && last_input.is(SwitchButton::R);
            let start = last_input.connected && last_input.is(SwitchButton::Plus);
            if start && !self.prev_start {
                result = VkResult::Accept;
            }
            if a && !self.prev_a {
                if let Some(k) = self.key_at(self.row, self.col) {
                    result = self.activate(k, buf);
                }
            }
            if b && !self.prev_b {
                self.backspace(buf);
            }
            if x && !self.prev_x {
                result = VkResult::Cancel;
            }
            if y && !self.prev_y {
                let _ = self.activate(Key::Fun(Special::Shift), buf);
            }
            if l && !self.prev_l && self.caret > 0 {
                self.caret -= 1;
            }
            if r && !self.prev_r {
                let n = buf.chars().count();
                if self.caret < n {
                    self.caret += 1;
                }
            }
            self.prev_a = a;
            self.prev_b = b;
            self.prev_x = x;
            self.prev_y = y;
            self.prev_l = l;
            self.prev_r = r;
            self.prev_start = start;
        }

        if matches!(result, VkResult::Cancel) {
            *buf = self.orig.clone();
        }
        if !matches!(result, VkResult::None) {
            self.open = false;
        }

        self.draw(ctx, ui, buf, ease, accent, light);
        result
    }

    fn draw(
        &self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        buf: &str,
        ease: f32,
        accent: Color32,
        light: bool,
    ) {
        let full = ctx.screen_rect();
        let mut p = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Debug,
            egui::Id::new("vkeyboard"),
        ));
        p.set_opacity(ease);

        let panel_h = (full.height() * 0.46).min(360.0);
        let slide = (1.0 - ease) * panel_h;
        let panel = egui::Rect::from_min_max(
            egui::pos2(full.min.x, full.max.y - panel_h + slide),
            egui::pos2(full.max.x, full.max.y + slide),
        );

        let bg = if light {
            Color32::from_rgb(0xEC, 0xEC, 0xF0)
        } else {
            Color32::from_rgb(0x1A, 0x1A, 0x22)
        };
        let key_bg = if light {
            Color32::from_rgb(0xFF, 0xFF, 0xFF)
        } else {
            Color32::from_rgb(0x2A, 0x2A, 0x34)
        };
        let key_border = if light {
            Color32::from_rgb(0xC8, 0xC8, 0xD2)
        } else {
            Color32::from_rgb(0x3C, 0x3C, 0x48)
        };
        let text = if light {
            Color32::from_rgb(0x1E, 0x1E, 0x28)
        } else {
            Color32::from_rgb(0xEC, 0xEC, 0xF0)
        };
        let muted = if light {
            Color32::from_rgb(0x70, 0x70, 0x7A)
        } else {
            Color32::from_rgb(0x8A, 0x8A, 0x98)
        };

        p.rect_filled(
            egui::Rect::from_min_max(full.min, egui::pos2(full.max.x, panel.min.y)),
            Rounding::ZERO,
            Color32::from_rgba_unmultiplied(0, 0, 0, (ease * 90.0) as u8),
        );
        p.rect_filled(
            panel,
            Rounding {
                nw: 18.0,
                ne: 18.0,
                sw: 0.0,
                se: 0.0,
            },
            bg,
        );
        p.rect_stroke(
            panel,
            Rounding {
                nw: 18.0,
                ne: 18.0,
                sw: 0.0,
                se: 0.0,
            },
            Stroke::new(1.5_f32, key_border),
        );

        let pad = 24.0;
        let preview = egui::Rect::from_min_max(
            egui::pos2(panel.min.x + pad, panel.min.y + 16.0),
            egui::pos2(panel.max.x - pad, panel.min.y + 58.0),
        );
        p.rect_filled(preview, Rounding::same(9.0), key_bg);
        p.rect_stroke(preview, Rounding::same(9.0), Stroke::new(1.5_f32, accent));
        let pv_font = FontId::proportional(20.0);
        let shown = if buf.is_empty() { "" } else { buf };
        let pre: String = buf.chars().take(self.caret).collect();
        let pre_w = ui.fonts(|f| f.layout_no_wrap(pre, pv_font.clone(), text).size().x);
        p.text(
            egui::pos2(preview.min.x + 14.0, preview.center().y),
            egui::Align2::LEFT_CENTER,
            shown,
            pv_font.clone(),
            text,
        );
        if (ctx.input(|i| i.time) * 1.6).fract() < 0.5 {
            let cx = preview.min.x + 14.0 + pre_w;
            p.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(cx, preview.center().y - 11.0),
                    egui::pos2(cx + 2.0, preview.center().y + 11.0),
                ),
                Rounding::ZERO,
                accent,
            );
        }

        let grid_top = preview.max.y + 16.0;
        let legend_h = 30.0;
        let grid_bottom = panel.max.y - legend_h - 12.0;
        let cols = 10.0;
        let rows_total = 5.0;
        let gap = 7.0;
        let grid_w = panel.width() - pad * 2.0;
        let cell_w = (grid_w - gap * (cols - 1.0)) / cols;
        let cell_h = ((grid_bottom - grid_top) - gap * (rows_total - 1.0)) / rows_total;

        for row in 0..4usize {
            let chars: Vec<char> = self.rows()[row].chars().collect();
            for (col, ch) in chars.iter().enumerate() {
                let x = panel.min.x + pad + col as f32 * (cell_w + gap);
                let y = grid_top + row as f32 * (cell_h + gap);
                let r = egui::Rect::from_min_size(egui::pos2(x, y), Vec2::new(cell_w, cell_h));
                let selected = self.open && self.row == row && self.col == col;
                p.rect_filled(r, Rounding::same(7.0), key_bg);
                if selected {
                    p.rect_stroke(r, Rounding::same(7.0), Stroke::new(2.6_f32, accent));
                } else {
                    p.rect_stroke(r, Rounding::same(7.0), Stroke::new(1.0_f32, key_border));
                }
                p.text(
                    r.center(),
                    egui::Align2::CENTER_CENTER,
                    ch.to_string(),
                    FontId::proportional(cell_h * 0.44),
                    text,
                );
            }
        }

        let fy = grid_top + 4.0 * (cell_h + gap);
        let labels = [
            (if self.caps { "SHIFT" } else { "shift" }, 1.0f32),
            (
                if self.layer == Layer::Symbol {
                    "ABC"
                } else {
                    "?12"
                },
                1.0,
            ),
            ("Space", 3.6),
            ("Del", 1.4),
            ("OK", 2.0),
        ];
        let total_weight: f32 = labels.iter().map(|(_, w)| *w).sum::<f32>();
        let total_gap = gap * (labels.len() as f32 - 1.0);
        let unit = (grid_w - total_gap) / total_weight;
        let mut fx = panel.min.x + pad;
        for (i, (lbl, w)) in labels.iter().enumerate() {
            let cw = unit * *w;
            let r = egui::Rect::from_min_size(egui::pos2(fx, fy), Vec2::new(cw, cell_h));
            let selected = self.open && self.row == 4 && self.col == i;
            let is_ok = i == 4;
            let fill = if is_ok { accent } else { key_bg };
            p.rect_filled(r, Rounding::same(7.0), fill);
            if selected {
                p.rect_stroke(r, Rounding::same(7.0), Stroke::new(2.6_f32, accent));
            } else {
                p.rect_stroke(r, Rounding::same(7.0), Stroke::new(1.0_f32, key_border));
            }
            let lc = if is_ok { Color32::WHITE } else { text };
            p.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                *lbl,
                FontId::proportional(cell_h * 0.34),
                lc,
            );
            fx += cw + gap;
        }

        let legend = "A Select    B Delete    Y Shift    X Cancel    L / R Move    Start OK";
        p.text(
            egui::pos2(panel.center().x, panel.max.y - legend_h * 0.5),
            egui::Align2::CENTER_CENTER,
            legend,
            FontId::proportional(13.0),
            muted,
        );
    }
}
