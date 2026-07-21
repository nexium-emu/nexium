use crate::homebrew::{self, ShopApp};
use crate::input::InputSnapshot;
use eframe::egui;
use egui::Color32;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

const BLUE: Color32 = Color32::from_rgb(0x1F, 0x6F, 0xE0);
const BLUE_HI: Color32 = Color32::from_rgb(0x3B, 0x8B, 0xFF);

#[derive(Clone, Copy)]
struct Pal {
    bg: Color32,
    panel: Color32,
    panel2: Color32,
    text: Color32,
    muted: Color32,
    border: Color32,
    btn_fill: Color32,
}

fn palette(light: bool) -> Pal {
    if light {
        Pal {
            bg: Color32::from_rgb(0xEC, 0xEC, 0xF1),
            panel: Color32::from_rgb(0xF5, 0xF5, 0xF9),
            panel2: Color32::from_rgb(0xE2, 0xE2, 0xE9),
            text: Color32::from_rgb(0x1E, 0x1E, 0x28),
            muted: Color32::from_rgb(0x60, 0x60, 0x6C),
            border: Color32::from_rgb(0xC6, 0xC6, 0xD0),
            btn_fill: Color32::from_rgb(0xEA, 0xEA, 0xF0),
        }
    } else {
        Pal {
            bg: Color32::from_rgb(0x14, 0x14, 0x19),
            panel: Color32::from_rgb(0x18, 0x18, 0x22),
            panel2: Color32::from_rgb(0x2C, 0x2C, 0x36),
            text: Color32::from_rgb(0xEC, 0xEC, 0xF0),
            muted: Color32::from_rgb(0x9A, 0x9A, 0xA6),
            border: Color32::from_rgb(0x32, 0x32, 0x3E),
            btn_fill: Color32::from_rgb(0x24, 0x24, 0x2E),
        }
    }
}

use crate::library::DownloadInfo;

#[derive(Default)]
struct Fetch {
    done: bool,
    apps: Vec<ShopApp>,
}

struct InstallJob {
    idx: usize,
    state: Arc<Mutex<DownloadInfo>>,
}

pub struct PendingInstall {
    pub title: String,
    pub icon: Option<Vec<u8>>,
    pub info: Arc<Mutex<DownloadInfo>>,
}

enum View {
    Grid,
    Detail(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum Dlg {
    Confirm,
    Disclaimer,
}

pub struct ShopState {
    pub open: bool,
    pub need_rescan: bool,
    anim: f32,
    fetch: Option<Arc<Mutex<Fetch>>>,
    loaded: bool,
    apps: Vec<ShopApp>,
    icons: Vec<Option<egui::TextureHandle>>,
    icon_bytes: Vec<Option<Vec<u8>>>,
    icon_jobs: Arc<Mutex<HashMap<usize, Option<Vec<u8>>>>>,
    icon_requested: HashSet<usize>,
    pub new_installs: Vec<PendingInstall>,
    search: String,
    editing: bool,
    search_kb: crate::vkeyboard::VirtualKeyboard,
    search_nav: bool,
    y_held: bool,
    filtered: Vec<usize>,
    selected: usize,
    scroll: f32,
    desc_scroll: f32,
    view: View,
    install: Option<InstallJob>,
    nav_cd: f64,
    a_held: bool,
    b_held: bool,
    dialog: Option<Dlg>,
    dialog_closing: bool,
    dialog_anim: f32,
    confirm_sel: usize,
    view_fade: f32,
    accent: Color32,
    sb_drag: bool,
    sel_px: f32,
    sel_py: f32,
    sel_init: bool,
    drag_from: Option<(f32, f32)>,
    grid_dragging: bool,
}

fn btn_fill(light: bool) -> Color32 {
    if light { Color32::from_rgb(0xEA, 0xEA, 0xF0) } else { Color32::from_rgb(0x24, 0x24, 0x2E) }
}

fn tint_toward(c: Color32, other: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(f(c.r(), other.r()), f(c.g(), other.g()), f(c.b(), other.b()))
}

impl ShopState {
    pub fn new() -> Self {
        Self {
            open: false,
            need_rescan: false,
            anim: 0.0,
            fetch: None,
            loaded: false,
            apps: Vec::new(),
            icons: Vec::new(),
            icon_bytes: Vec::new(),
            icon_jobs: Arc::new(Mutex::new(HashMap::new())),
            icon_requested: HashSet::new(),
            new_installs: Vec::new(),
            search: String::new(),
            editing: false,
            search_kb: crate::vkeyboard::VirtualKeyboard::new(),
            search_nav: false,
            y_held: false,
            filtered: Vec::new(),
            selected: 0,
            scroll: 0.0,
            desc_scroll: 0.0,
            view: View::Grid,
            install: None,
            nav_cd: 0.0,
            a_held: false,
            b_held: false,
            dialog: None,
            dialog_closing: false,
            dialog_anim: 0.0,
            confirm_sel: 1,
            view_fade: 1.0,
            accent: Color32::from_rgb(0x2F, 0xB4, 0xEF),
            sb_drag: false,
            sel_px: 0.0,
            sel_py: 0.0,
            sel_init: false,
            drag_from: None,
            grid_dragging: false,
        }
    }

    fn open_dialog(&mut self, kind: Dlg) {
        self.dialog = Some(kind);
        self.dialog_closing = false;
    }

    pub fn open(&mut self) {
        self.open = true;
        self.view = View::Grid;
        self.open_dialog(Dlg::Disclaimer);
        // swallow the still-held A/B that opened the shop so it doesn't
        // immediately fire an edge inside the shop on the same press.
        self.a_held = true;
        self.b_held = true;
        if !self.loaded && self.fetch.is_none() {
            let shared = Arc::new(Mutex::new(Fetch::default()));
            self.fetch = Some(shared.clone());
            std::thread::spawn(move || {
                let apps = homebrew::fetch_games();
                if let Ok(mut g) = shared.lock() {
                    g.apps = apps;
                    g.done = true;
                }
            });
        }
    }

    pub fn visible(&self) -> bool {
        self.open || self.anim >= 0.004
    }

    fn refilter(&mut self) {
        let q = self.search.to_lowercase();
        self.filtered = (0..self.apps.len())
            .filter(|&i| {
                q.is_empty()
                    || self.apps[i].title.to_lowercase().contains(&q)
                    || self.apps[i].author.to_lowercase().contains(&q)
            })
            .collect();
        if self.selected >= self.filtered.len() {
            self.selected = self.filtered.len().saturating_sub(1);
        }
    }

    fn start_install(&mut self, ai: usize) {
        if self.install.is_some() {
            return;
        }
        let app = self.apps[ai].clone();
        let state = Arc::new(Mutex::new(DownloadInfo::default()));
        let s2 = state.clone();
        std::thread::spawn(move || {
            let ok = install_app(&app, &s2);
            if let Ok(mut g) = s2.lock() {
                g.done = true;
                g.ok = ok;
                g.progress = 1.0;
            }
        });
        self.new_installs.push(PendingInstall {
            title: self.apps[ai].title.clone(),
            icon: self.icon_bytes.get(ai).cloned().flatten(),
            info: state.clone(),
        });
        self.install = Some(InstallJob { idx: ai, state });
    }

    fn request_icon(&mut self, i: usize) {
        if self.icon_requested.contains(&i) || i >= self.apps.len() {
            return;
        }
        self.icon_requested.insert(i);
        let app = self.apps[i].clone();
        let jobs = self.icon_jobs.clone();
        std::thread::spawn(move || {
            let bytes = homebrew::icon_bytes(&app);
            if let Ok(mut m) = jobs.lock() {
                m.insert(i, bytes);
            }
        });
    }

    pub fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, light: bool, accent: Color32, li: &InputSnapshot) {
        if !self.open && self.anim < 0.004 {
            return;
        }
        if !self.open {
            self.search_kb.open = false;
            self.search_nav = false;
            self.editing = false;
        }
        let pal = palette(light);
        let now = ctx.input(|i| i.time);
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.anim += ((if self.open { 1.0 } else { 0.0 }) - self.anim) * (dt * 14.0).min(1.0);
        let ease = { let a = self.anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };

        // ingest fetch
        if !self.loaded {
            let apps = self
                .fetch
                .as_ref()
                .and_then(|f| f.lock().ok().and_then(|g| if g.done { Some(g.apps.clone()) } else { None }));
            if let Some(apps) = apps {
                self.apps = apps;
                self.icons = vec![None; self.apps.len()];
                self.icon_bytes = vec![None; self.apps.len()];
                self.loaded = true;
                self.refilter();
            }
        }

        // build icon textures
        let ready: Vec<(usize, Vec<u8>)> = {
            let mut out = Vec::new();
            if let Ok(mut m) = self.icon_jobs.lock() {
                let keys: Vec<usize> = m.keys().copied().collect();
                for k in keys {
                    if let Some(Some(bytes)) = m.get(&k).cloned() {
                        out.push((k, bytes));
                        m.insert(k, None);
                    }
                }
            }
            out
        };
        for (i, bytes) in ready {
            if let Some(img) = image::load_from_memory(&bytes).ok().map(|im| {
                let rgba = im.to_rgba8();
                let (w, h) = rgba.dimensions();
                egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw())
            }) {
                if i < self.icons.len() {
                    self.icons[i] = Some(ctx.load_texture(format!("shop_icon_{i}"), img, egui::TextureOptions::LINEAR));
                }
            }
            if i < self.icon_bytes.len() {
                self.icon_bytes[i] = Some(bytes);
            }
        }

        let screen = ctx.screen_rect();
        let mut paint = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("shop")));
        paint.set_opacity(ease);
        paint.rect_filled(screen, egui::Rounding::ZERO, pal.bg);

        // top banner (red eShop-style)
        let bar = egui::Rect::from_min_size(screen.min, egui::Vec2::new(screen.width(), 52.0));
        paint.rect_filled(bar, egui::Rounding::ZERO, BLUE);
        paint.text(egui::pos2(bar.min.x + 24.0, bar.center().y), egui::Align2::LEFT_CENTER, "NeXium  Homebrew", egui::FontId::proportional(20.0), Color32::WHITE);

        // ---- keyboard / text ----
        let (kb_esc, kb_enter, text_events) = ctx.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::Enter), i.events.clone()));
        let kb_left = ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft));
        let kb_right = ctx.input(|i| i.key_pressed(egui::Key::ArrowRight));
        let kb_up = ctx.input(|i| i.key_pressed(egui::Key::ArrowUp));
        let kb_down = ctx.input(|i| i.key_pressed(egui::Key::ArrowDown));

        // ---- controller edges ----
        use crate::controller_config::SwitchButton;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let mut a_edge = kb_enter || (gp_a && !self.a_held);
        let mut b_edge = kb_esc || (gp_b && !self.b_held);
        self.a_held = gp_a;
        self.b_held = gp_b;
        let ready_nav = now - self.nav_cd > 0.16;
        let (mut nl, mut nr, mut nu, mut nd) = (kb_left, kb_right, kb_up, kb_down);
        if ready_nav && li.connected {
            if li.is(SwitchButton::DLeft) || li.lx() < -0.5 { nl = true; self.nav_cd = now; }
            else if li.is(SwitchButton::DRight) || li.lx() > 0.5 { nr = true; self.nav_cd = now; }
            else if li.is(SwitchButton::DUp) || li.ly() > 0.5 { nu = true; self.nav_cd = now; }
            else if li.is(SwitchButton::DDown) || li.ly() < -0.5 { nd = true; self.nav_cd = now; }
        }

        let can_focus_search = matches!(self.view, View::Grid) && self.dialog.is_none();
        let y_down = li.connected && li.is(SwitchButton::Y);
        if y_down && !self.y_held && !self.search_kb.open && can_focus_search {
            let cur = self.search.clone();
            self.search_kb.show(&cur, 40);
            self.editing = false;
        }
        self.y_held = y_down;

        if !self.editing && !self.search_kb.open && can_focus_search {
            if self.search_nav {
                if nd {
                    self.search_nav = false;
                    crate::ui_audio::play_move();
                } else if a_edge {
                    let cur = self.search.clone();
                    self.search_kb.show(&cur, 40);
                } else if b_edge {
                    self.search_nav = false;
                }
                a_edge = false;
                b_edge = false;
                nu = false;
                nd = false;
                nl = false;
                nr = false;
            } else if nu && self.selected < 5 {
                self.search_nav = true;
                nu = false;
                crate::ui_audio::play_move();
            }
        } else {
            self.search_nav = false;
        }

        // search field (top-right of banner)
        let field = egui::Rect::from_min_size(egui::pos2(screen.max.x - 320.0, 9.0), egui::Vec2::new(292.0, 34.0));
        let field_resp = ui.allocate_rect(field, egui::Sense::click());
        let field_hot = self.editing || self.search_nav || self.search_kb.open;
        paint.rect_filled(field, egui::Rounding::same(8.0), Color32::from_black_alpha(60));
        paint.rect_stroke(field, egui::Rounding::same(8.0), egui::Stroke::new(if field_hot { 2.0_f32 } else { 1.0_f32 }, if field_hot { accent } else { Color32::from_white_alpha(120) }));
        let disp = if self.search.is_empty() { "Search…".to_string() } else { self.search.clone() };
        paint.text(egui::pos2(field.min.x + 12.0, field.center().y), egui::Align2::LEFT_CENTER, &disp, egui::FontId::proportional(15.0), Color32::from_white_alpha(if self.search.is_empty() { 150 } else { 255 }));
        if li.connected {
            let badge = egui::pos2(field.max.x - 16.0, field.center().y);
            paint.circle_filled(badge, 9.0, if field_hot { accent } else { Color32::from_white_alpha(60) });
            paint.text(badge, egui::Align2::CENTER_CENTER, "Y", egui::FontId::proportional(12.0), Color32::from_rgb(0x10, 0x14, 0x1C));
        }
        if field_resp.clicked() {
            self.editing = true;
            self.search_nav = false;
        }
        if self.editing {
            for ev in &text_events {
                match ev {
                    egui::Event::Text(t) => {
                        for c in t.chars() {
                            if !c.is_control() && self.search.chars().count() < 40 {
                                self.search.push(c);
                            }
                        }
                    }
                    egui::Event::Key { key: egui::Key::Backspace, pressed: true, .. } => {
                        self.search.pop();
                    }
                    _ => {}
                }
            }
            self.refilter();
            if kb_enter || kb_esc {
                self.editing = false;
            }
        }
        if self.search_kb.active() {
            let before = self.search.clone();
            let _ = self.search_kb.update(ctx, ui, &mut self.search, li, accent, light);
            if self.search != before {
                self.refilter();
                self.selected = 0;
            }
        }
        let block_search = self.editing || self.search_kb.open;
        let (a_edge, b_edge, nl, nr, nu, nd) = if block_search {
            (false, false, false, false, false, false)
        } else {
            (a_edge, b_edge, nl, nr, nu, nd)
        };

        self.accent = accent;
        let content = egui::Rect::from_min_max(egui::pos2(screen.min.x, bar.max.y), screen.max);
        let detail_idx = if let View::Detail(i) = self.view { Some(i) } else { None };
        let dialog_before = self.dialog.is_some();
        // suppress view input while a dialog is up (so A/nav go to the dialog)
        let view_a = a_edge && !dialog_before;
        let (vnl, vnr, vnu, vnd) = if dialog_before { (false, false, false, false) } else { (nl, nr, nu, nd) };
        match detail_idx {
            None => self.grid(ui, &paint, content, pal, accent, vnl, vnr, vnu, vnd, view_a),
            Some(i) => self.detail(ui, &paint, content, pal, i, view_a, if dialog_before { 0.0 } else { ui.input(|x| x.smooth_scroll_delta.y) }, vnu, vnd),
        }

        // cross-view fade transition
        self.view_fade += (1.0 - self.view_fade) * (dt * 9.0).min(1.0);
        if self.view_fade < 0.995 {
            let a = ((1.0 - self.view_fade) * 255.0) as u8;
            paint.rect_filled(content, egui::Rounding::ZERO, Color32::from_rgba_unmultiplied(pal.bg.r(), pal.bg.g(), pal.bg.b(), a));
        }

        // dialogs (confirm / disclaimer) with the shared popup fade
        self.render_dialog(ui, ctx, screen, pal, ease, dt, detail_idx, a_edge, b_edge, nl, nr, dialog_before);

        if b_edge && !dialog_before {
            match self.view {
                View::Detail(_) => {
                    self.view = View::Grid;
                    self.desc_scroll = 0.0;
                    self.view_fade = 0.0;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
                View::Grid => {
                    self.open = false;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        }

        ctx.request_repaint();
    }

    /// Decorative auto-scrolling strip of wide game posters with a centered
    /// tagline and fade-out edges. `band` is where the strip lives (it may sit
    /// partly above `clip_rect` while sliding away); everything is clipped to
    /// `clip_rect` so it collapses cleanly as the grid scrolls up.
    fn marquee(&mut self, ui: &mut egui::Ui, paint: &egui::Painter, band: egui::Rect, clip_rect: egui::Rect, pal: Pal) {
        let band_bg = tint_toward(pal.bg, pal.panel2, 0.5);
        let clip = paint.with_clip_rect(clip_rect);
        clip.rect_filled(clip_rect, egui::Rounding::ZERO, band_bg);
        let t = ui.input(|i| i.time) as f32;
        let ph = band.height() - 34.0; // poster height
        let pw = ph * 1.72; // match the grid's landscape aspect
        let gap = 16.0;
        let step = pw + gap;
        let count = self.apps.len().min(28).max(1);
        for i in 0..count {
            self.request_icon(i);
        }
        let strip = step * count as f32;
        let speed = 26.0; // px/s, drifting left→right
        let off = (t * speed).rem_euclid(strip);
        let y = band.center().y - ph * 0.5;
        let reps = (band.width() / step).ceil() as i32 + count as i32 + 2;
        for j in 0..reps {
            let x = band.min.x - strip + off + j as f32 * step;
            if x > band.max.x + 4.0 || x + pw < band.min.x - 4.0 {
                continue;
            }
            let idx = (j as usize) % count;
            let tile = egui::Rect::from_min_size(egui::pos2(x, y), egui::Vec2::new(pw, ph));
            clip.rect_filled(tile.translate(egui::Vec2::new(0.0, 3.0)), egui::Rounding::same(9.0), Color32::from_black_alpha(55));
            clip.rect_filled(tile, egui::Rounding::same(9.0), pal.panel);
            if let Some(Some(tex)) = self.icons.get(idx) {
                crate::carousel::draw_rounded_image(&clip, tex.id(), tile, 9.0, Color32::WHITE);
            } else {
                let init = self.apps[idx].title.chars().next().unwrap_or('?').to_uppercase().to_string();
                clip.text(tile.center(), egui::Align2::CENTER_CENTER, &init, egui::FontId::proportional(ph * 0.42), pal.muted);
            }
        }
        // fade the posters out toward both edges
        let fade_w = (band.width() * 0.20).min(220.0);
        hfade(&clip, egui::Rect::from_min_max(egui::pos2(band.min.x, band.min.y), egui::pos2(band.min.x + fade_w, band.max.y)), band_bg, true);
        hfade(&clip, egui::Rect::from_min_max(egui::pos2(band.max.x - fade_w, band.min.y), egui::pos2(band.max.x, band.max.y)), band_bg, false);
        // centered tagline on a soft backing pill — rotates every 6s w/ a crossfade
        const TAGLINES: [&str; 3] = [
            "Homebrew has never been easier to get into.",
            "Enjoy Homebrew straight from the emulator.",
            "Hundreds of community games, one click away.",
        ];
        let period = 6.0_f32;
        let idx = ((t / period) as usize) % TAGLINES.len();
        let phase = t.rem_euclid(period);
        let fade = 0.6_f32;
        let ta = if phase < fade {
            phase / fade
        } else if phase > period - fade {
            (period - phase) / fade
        } else {
            1.0
        };
        let tag = TAGLINES[idx];
        let fid = egui::FontId::proportional(19.0);
        let tw = ui.fonts(|f| f.layout_no_wrap(tag.to_string(), fid.clone(), pal.text).size().x);
        let pill = egui::Rect::from_center_size(band.center(), egui::Vec2::new(tw + 56.0, 44.0));
        clip.rect_filled(pill, egui::Rounding::same(22.0), Color32::from_rgba_unmultiplied(band_bg.r(), band_bg.g(), band_bg.b(), 236));
        clip.rect_stroke(pill, egui::Rounding::same(22.0), egui::Stroke::new(1.0_f32, pal.border));
        clip.text(band.center(), egui::Align2::CENTER_CENTER, tag, fid, pal.text.gamma_multiply(ta));
        clip.line_segment([egui::pos2(band.min.x, band.max.y), egui::pos2(band.max.x, band.max.y)], egui::Stroke::new(1.0_f32, pal.border));
    }

    #[allow(clippy::too_many_arguments)]
    fn grid(&mut self, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect, pal: Pal, accent: Color32, nl: bool, nr: bool, nu: bool, nd: bool, a_edge: bool) {
        if !self.loaded {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "Loading shop…", egui::FontId::proportional(20.0), pal.muted);
            return;
        }
        let n = self.filtered.len();
        let marquee_h = 118.0;
        let cols = 5usize;
        let pad = 34.0;
        let gap = 22.0;
        let tile_w = (content.width() - pad * 2.0 - gap * (cols as f32 - 1.0)) / cols as f32;
        let tile_h = tile_w * 0.58;
        let cell_h = tile_h + 32.0;

        // nav (track whether the selection actually moved)
        let mut nav_moved = false;
        if nl && self.selected > 0 { self.selected -= 1; nav_moved = true; }
        if nr && self.selected + 1 < n { self.selected += 1; nav_moved = true; }
        if nd && self.selected + cols < n { self.selected += cols; nav_moved = true; }
        if nu && self.selected >= cols { self.selected -= cols; nav_moved = true; }
        if nav_moved { crate::ui_audio::play_move(); }

        // the marquee lives at the top of the scroll space so it slides away as
        // you scroll down and only sits at the very top.
        let rows = (n + cols - 1) / cols;
        let grid_h = rows as f32 * (cell_h + gap);
        let total_h = marquee_h + grid_h;
        let max_scroll = (total_h + pad - content.height()).max(0.0);
        let wheel = ui.input(|i| i.smooth_scroll_delta.y);
        if wheel.abs() > 0.1 {
            self.scroll = (self.scroll - wheel).clamp(0.0, max_scroll);
        }
        if nav_moved {
            let row = self.selected / cols;
            if row == 0 {
                self.scroll = 0.0; // reveal the marquee at the very top
            } else {
                let sel_top = marquee_h + row as f32 * (cell_h + gap);
                let sel_bot = sel_top + cell_h;
                let view_h = content.height() - pad - 40.0;
                if sel_top - pad < self.scroll {
                    self.scroll = (sel_top - pad).max(0.0);
                } else if sel_bot > self.scroll + view_h {
                    self.scroll = sel_bot - view_h;
                }
            }
        }
        self.scroll = self.scroll.clamp(0.0, max_scroll);

        // touch-style drag-to-scroll: press and drag anywhere over the menu
        if max_scroll > 0.0 {
            let (ptr, pdown, ppressed) = ui.input(|i| {
                (i.pointer.interact_pos(), i.pointer.primary_down(), i.pointer.primary_pressed())
            });
            let drag_area = egui::Rect::from_min_max(content.min, egui::pos2(content.max.x - 22.0, content.max.y - 40.0));
            if ppressed && !self.sb_drag && ptr.map_or(false, |p| drag_area.contains(p)) {
                self.drag_from = Some((ptr.unwrap().y, self.scroll));
                self.grid_dragging = false;
            }
            if !pdown {
                self.drag_from = None;
                self.grid_dragging = false;
            }
            if let (Some((py0, s0)), Some(p)) = (self.drag_from, ptr) {
                if (p.y - py0).abs() > 5.0 {
                    self.grid_dragging = true;
                }
                if self.grid_dragging {
                    self.scroll = (s0 - (p.y - py0)).clamp(0.0, max_scroll);
                }
            }
        } else {
            self.drag_from = None;
            self.grid_dragging = false;
        }

        // marquee, sliding up with scroll and clipped so it collapses cleanly
        if self.scroll < marquee_h {
            let vis_h = marquee_h - self.scroll;
            let band = egui::Rect::from_min_max(
                egui::pos2(content.min.x, content.min.y - self.scroll),
                egui::pos2(content.max.x, content.min.y - self.scroll + marquee_h),
            );
            let cr = egui::Rect::from_min_max(content.min, egui::pos2(content.max.x, content.min.y + vis_h));
            self.marquee(ui, paint, band, cr, pal);
        }

        let now_t = ui.input(|i| i.time) as f32;
        let x0 = content.min.x + pad;
        let y0 = content.min.y + marquee_h + pad - self.scroll;
        // clip the grid to the area below whatever of the marquee is still visible
        let grid_top = content.min.y + (marquee_h - self.scroll).max(0.0);
        let clip = paint.with_clip_rect(egui::Rect::from_min_max(egui::pos2(content.min.x, grid_top), content.max));
        // animated selection highlight that glides between tiles (like the dockbar)
        if n > 0 {
            let tvx = content.min.x + pad + (self.selected % cols) as f32 * (tile_w + gap);
            let tvy = content.min.y + marquee_h + pad + (self.selected / cols) as f32 * (cell_h + gap);
            let sdt = ui.input(|i| i.stable_dt).min(0.1);
            if !self.sel_init {
                self.sel_px = tvx;
                self.sel_py = tvy;
                self.sel_init = true;
            }
            let k = (sdt * 16.0).min(1.0);
            self.sel_px += (tvx - self.sel_px) * k;
            self.sel_py += (tvy - self.sel_py) * k;
            let sr = egui::Rect::from_min_size(egui::pos2(self.sel_px, self.sel_py - self.scroll), egui::Vec2::new(tile_w, tile_h));
            let pulse = 0.7 + 0.3 * (now_t * 3.0).sin();
            clip.rect_filled(sr.expand(6.0), egui::Rounding::same(15.0), Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (70.0 * pulse) as u8));
            clip.rect_stroke(sr.expand(4.0), egui::Rounding::same(13.0), egui::Stroke::new(2.5_f32, accent));
        }
        let mut open_detail = None;
        let filtered = self.filtered.clone();
        for (vi, ai) in filtered.iter().copied().enumerate() {
            let r = vi / cols;
            let c = vi % cols;
            let tx = x0 + c as f32 * (tile_w + gap);
            let ty = y0 + r as f32 * (cell_h + gap);
            if ty + cell_h < grid_top || ty > content.max.y {
                continue;
            }
            self.request_icon(ai);
            let tile = egui::Rect::from_min_size(egui::pos2(tx, ty), egui::Vec2::new(tile_w, tile_h));
            let sel = vi == self.selected;
            // soft drop shadow for depth
            clip.rect_filled(tile.translate(egui::Vec2::new(0.0, 4.0)).expand(1.0), egui::Rounding::same(11.0), Color32::from_black_alpha(70));
            clip.rect_filled(tile, egui::Rounding::same(10.0), pal.panel);
            if let Some(Some(tex)) = self.icons.get(ai) {
                crate::carousel::draw_rounded_image(&clip, tex.id(), tile, 10.0, Color32::WHITE);
            } else {
                clip.text(tile.center(), egui::Align2::CENTER_CENTER, &self.apps[ai].title, egui::FontId::proportional(13.0), pal.muted);
            }
            let label_col = if sel { tint_toward(pal.text, accent, 0.5) } else { pal.text };
            clip.text(egui::pos2(tile.min.x + 2.0, tile.max.y + 6.0), egui::Align2::LEFT_TOP, &self.apps[ai].title, egui::FontId::proportional(13.0), label_col);
            let resp = ui.allocate_rect(tile, egui::Sense::click());
            if !self.grid_dragging {
                if resp.double_clicked() {
                    self.selected = vi;
                    open_detail = Some(ai);
                } else if resp.clicked() {
                    self.selected = vi;
                }
            }
        }
        if n == 0 {
            let cy = (content.min.y + marquee_h + content.max.y) * 0.5;
            paint.text(egui::pos2(content.center().x, cy), egui::Align2::CENTER_CENTER, "No games found.", egui::FontId::proportional(18.0), pal.muted);
        }
        if a_edge && self.selected < n {
            open_detail = Some(self.filtered[self.selected]);
        }

        // scrollbar (draggable) on the right — manual pointer tracking so it
        // can't get stuck by egui drag-capture on the shop's overlay layer
        if max_scroll > 0.0 {
            let track = egui::Rect::from_min_size(egui::pos2(content.max.x - 16.0, content.min.y + 8.0), egui::Vec2::new(7.0, content.height() - 56.0));
            let hit = track.expand2(egui::Vec2::new(10.0, 4.0));
            let view_frac = (content.height() / total_h).clamp(0.06, 1.0);
            let thumb_h = (track.height() * view_frac).max(30.0);
            let range = (track.height() - thumb_h).max(1.0);
            let (ptr_pos, ptr_down, ptr_pressed) = ui.input(|i| {
                (i.pointer.interact_pos(), i.pointer.primary_down(), i.pointer.primary_pressed())
            });
            if ptr_pressed && ptr_pos.map_or(false, |p| hit.contains(p)) {
                self.sb_drag = true;
            }
            if !ptr_down {
                self.sb_drag = false;
            }
            if self.sb_drag {
                if let Some(p) = ptr_pos {
                    let rel = ((p.y - track.min.y - thumb_h * 0.5) / range).clamp(0.0, 1.0);
                    self.scroll = rel * max_scroll;
                }
            }
            let hot = self.sb_drag || ptr_pos.map_or(false, |p| hit.contains(p));
            paint.rect_filled(track, egui::Rounding::same(3.5), tint_toward(pal.bg, pal.panel2, 0.6));
            let thumb_y = track.min.y + range * (self.scroll / max_scroll);
            let thumb = egui::Rect::from_min_size(egui::pos2(track.min.x, thumb_y), egui::Vec2::new(7.0, thumb_h));
            paint.rect_filled(thumb, egui::Rounding::same(3.5), if hot { pal.muted } else { tint_toward(pal.panel2, pal.muted, 0.5) });
        }
        if let Some(ai) = open_detail {
            self.view = View::Detail(ai);
            self.desc_scroll = 0.0;
            self.view_fade = 0.0;
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }

        // hint bar
        let hint = if self.search_nav {
            "[A] Type   ·   [\u{2193}/B] Back to games"
        } else {
            "[A] Select   ·   [B] Close   ·   [\u{2191}/Y] Search"
        };
        hint_bar(paint, content, pal, hint);

        // homebrew site link (bottom-left) — jump straight to the real store
        let link = "Browse at hb-app.store";
        let fid = egui::FontId::proportional(13.0);
        let lw = ui.fonts(|f| f.layout_no_wrap(link.to_string(), fid.clone(), accent).size().x);
        let lrect = egui::Rect::from_min_size(egui::pos2(content.min.x + 24.0, content.max.y - 40.0), egui::Vec2::new(lw + 10.0, 40.0));
        let lresp = ui.allocate_rect(lrect, egui::Sense::click());
        let lcol = if lresp.hovered() { tint_toward(accent, Color32::WHITE, 0.35) } else { accent };
        paint.text(egui::pos2(content.min.x + 24.0, content.max.y - 20.0), egui::Align2::LEFT_CENTER, link, fid, lcol);
        if lresp.hovered() {
            paint.line_segment([egui::pos2(lrect.min.x, content.max.y - 8.0), egui::pos2(lrect.min.x + lw, content.max.y - 8.0)], egui::Stroke::new(1.0_f32, lcol));
        }
        if lresp.clicked() {
            open_url("https://hb-app.store/switch");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn detail(&mut self, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect, pal: Pal, ai: usize, a_edge: bool, wheel: f32, nu: bool, nd: bool) {
        let app = self.apps[ai].clone();
        self.request_icon(ai);
        let pad = 40.0;
        let col_split = content.min.x + content.width() * 0.68;

        // hero icon (large)
        let isz = 210.0;
        let icon_r = egui::Rect::from_min_size(egui::pos2(content.min.x + pad, content.min.y + pad), egui::Vec2::splat(isz));
        paint.rect_filled(icon_r.expand(2.0).translate(egui::Vec2::new(0.0, 5.0)), egui::Rounding::same(22.0), Color32::from_black_alpha(70));
        paint.rect_filled(icon_r.expand(2.0), egui::Rounding::same(22.0), pal.panel2);
        if let Some(Some(tex)) = self.icons.get(ai) {
            crate::carousel::draw_rounded_image(paint, tex.id(), icon_r, 20.0, Color32::WHITE);
        } else {
            paint.rect_filled(icon_r, egui::Rounding::same(20.0), pal.panel);
        }
        let tx = icon_r.max.x + 30.0;
        paint.text(egui::pos2(tx, icon_r.min.y + 22.0), egui::Align2::LEFT_TOP, &app.title, egui::FontId::proportional(34.0), pal.text);
        paint.text(egui::pos2(tx, icon_r.min.y + 70.0), egui::Align2::LEFT_TOP, &app.author, egui::FontId::proportional(18.0), pal.muted);
        paint.text(egui::pos2(tx, icon_r.min.y + 100.0), egui::Align2::LEFT_TOP, &format!("v{}   ·   {}   ·   {}", app.version, human_size(app.filesize), app.license), egui::FontId::proportional(14.0), pal.muted);
        // link straight to this game's page on the store
        let link = "View on hb-app.store";
        let lfid = egui::FontId::proportional(14.0);
        let lw = ui.fonts(|f| f.layout_no_wrap(link.to_string(), lfid.clone(), self.accent).size().x);
        let lrect = egui::Rect::from_min_size(egui::pos2(tx, icon_r.min.y + 130.0), egui::Vec2::new(lw + 10.0, 24.0));
        let lresp = ui.allocate_rect(lrect, egui::Sense::click());
        let lcol = if lresp.hovered() { tint_toward(self.accent, Color32::WHITE, 0.35) } else { self.accent };
        paint.text(egui::pos2(tx, icon_r.min.y + 130.0), egui::Align2::LEFT_TOP, link, lfid, lcol);
        if lresp.hovered() {
            paint.line_segment([egui::pos2(tx, icon_r.min.y + 150.0), egui::pos2(tx + lw, icon_r.min.y + 150.0)], egui::Stroke::new(1.0_f32, lcol));
        }
        if lresp.clicked() {
            open_url(&app.page_url());
        }

        // description panel (scrollable)
        let desc = if app.details.is_empty() { app.description.clone() } else { app.details.clone() };
        let desc_rect = egui::Rect::from_min_max(egui::pos2(content.min.x + pad, icon_r.max.y + 26.0), egui::pos2(col_split - 20.0, content.max.y - 60.0));
        paint.rect_filled(desc_rect, egui::Rounding::same(12.0), pal.panel);
        let galley = ui.fonts(|f| f.layout(desc, egui::FontId::proportional(15.0), pal.text, desc_rect.width() - 32.0));
        let text_h = galley.size().y;
        let max_ds = (text_h - (desc_rect.height() - 24.0)).max(0.0);
        if desc_rect.contains(ui.input(|i| i.pointer.hover_pos()).unwrap_or(desc_rect.center())) && wheel.abs() > 0.1 {
            self.desc_scroll = (self.desc_scroll - wheel).clamp(0.0, max_ds);
        }
        if nd { self.desc_scroll = (self.desc_scroll + 40.0).clamp(0.0, max_ds); }
        if nu { self.desc_scroll = (self.desc_scroll - 40.0).clamp(0.0, max_ds); }
        let dclip = paint.with_clip_rect(desc_rect.shrink(4.0));
        dclip.galley(egui::pos2(desc_rect.min.x + 16.0, desc_rect.min.y + 12.0 - self.desc_scroll), galley, pal.text);
        if max_ds > 0.0 {
            let track = egui::Rect::from_min_size(egui::pos2(desc_rect.max.x - 6.0, desc_rect.min.y + 6.0), egui::Vec2::new(4.0, desc_rect.height() - 12.0));
            paint.rect_filled(track, egui::Rounding::same(2.0), pal.panel2);
            let frac = self.desc_scroll / max_ds;
            let th = (track.height() * (desc_rect.height() / (text_h + 24.0))).clamp(20.0, track.height());
            let ty = track.min.y + (track.height() - th) * frac;
            paint.rect_filled(egui::Rect::from_min_size(egui::pos2(track.min.x, ty), egui::Vec2::new(4.0, th)), egui::Rounding::same(2.0), pal.muted);
        }

        // right install panel
        let (progress, done, ok) = self
            .install
            .as_ref()
            .filter(|j| j.idx == ai)
            .and_then(|j| j.state.lock().ok().map(|g| (g.progress, g.done, g.ok)))
            .unwrap_or((0.0, false, false));
        let installing = self.install.as_ref().map_or(false, |j| j.idx == ai) && !done;
        let failed = self.install.as_ref().map_or(false, |j| j.idx == ai) && done && !ok;

        let btn = egui::Rect::from_min_size(egui::pos2(col_split + 20.0, content.min.y + pad + 30.0), egui::Vec2::new(content.max.x - col_split - 60.0, 60.0));
        let btn_resp = ui.allocate_rect(btn, egui::Sense::click());
        let hovered = btn_resp.hovered();
        let idle = !installing && !(done && ok);
        // focus glow — the Install button is the detail's default focus
        if idle {
            let t = ui.input(|i| i.time) as f32;
            let pulse = 0.6 + 0.4 * (t * 3.0).sin();
            for k in 0..4 {
                let e = (4 - k) as f32 * 3.0;
                paint.rect_stroke(btn.expand(e), egui::Rounding::same(12.0 + e), egui::Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(BLUE_HI.r(), BLUE_HI.g(), BLUE_HI.b(), (40.0 * pulse) as u8)));
            }
            paint.rect_stroke(btn.expand(3.0), egui::Rounding::same(15.0), egui::Stroke::new(2.5_f32, Color32::from_rgba_unmultiplied(BLUE_HI.r(), BLUE_HI.g(), BLUE_HI.b(), (200.0 * pulse) as u8)));
        }
        let btn_col = if installing { pal.panel2 } else if done && ok { Color32::from_rgb(0x2C, 0xA0, 0x4A) } else if failed { Color32::from_rgb(0xB0, 0x3A, 0x3A) } else if hovered { BLUE_HI } else { BLUE };
        paint.rect_filled(btn, egui::Rounding::same(12.0), btn_col);
        let label = if done && ok {
            "Installed".to_string()
        } else if installing {
            format!("Installing…  {}%", (progress * 100.0) as u32)
        } else if failed {
            "Retry Install".to_string()
        } else {
            "Install".to_string()
        };
        let tw = ui.fonts(|f| f.layout_no_wrap(label.clone(), egui::FontId::proportional(20.0), Color32::WHITE).size().x);
        let lx = if done && ok { btn.center().x + 14.0 } else { btn.center().x };
        paint.text(egui::pos2(lx, btn.center().y), egui::Align2::CENTER_CENTER, &label, egui::FontId::proportional(20.0), Color32::WHITE);
        if done && ok {
            let cc = egui::pos2(lx - tw * 0.5 - 18.0, btn.center().y);
            paint.add(egui::Shape::line(vec![cc + egui::Vec2::new(-6.0, 0.0), cc + egui::Vec2::new(-2.0, 5.0), cc + egui::Vec2::new(7.0, -6.0)], egui::Stroke::new(2.6_f32, Color32::WHITE)));
        }
        if installing {
            let barr = egui::Rect::from_min_size(egui::pos2(btn.min.x, btn.max.y + 10.0), egui::Vec2::new(btn.width(), 7.0));
            paint.rect_filled(barr, egui::Rounding::same(3.5), pal.panel2);
            let mut fill = barr;
            fill.set_width(barr.width() * progress);
            paint.rect_filled(fill, egui::Rounding::same(3.5), Color32::from_rgb(0x35, 0xD0, 0x6A));
        }
        if done && ok {
            self.need_rescan = true;
        }

        // info card fills the right column under the button
        let info = egui::Rect::from_min_max(egui::pos2(btn.min.x, btn.max.y + 40.0), egui::pos2(content.max.x - 40.0, content.max.y - 60.0));
        paint.rect_filled(info, egui::Rounding::same(12.0), pal.panel);
        paint.rect_stroke(info, egui::Rounding::same(12.0), egui::Stroke::new(1.0_f32, pal.border));
        let rows = [
            ("Developer", app.author.clone()),
            ("Version", app.version.clone()),
            ("Download size", human_size(app.filesize)),
            ("License", if app.license.is_empty() { "—".into() } else { app.license.clone() }),
            ("Updated", if app.updated.is_empty() { "—".into() } else { app.updated.clone() }),
        ];
        for (ri, (k, v)) in rows.iter().enumerate() {
            let ry = info.min.y + 22.0 + ri as f32 * 40.0;
            paint.text(egui::pos2(info.min.x + 18.0, ry), egui::Align2::LEFT_CENTER, *k, egui::FontId::proportional(14.0), pal.muted);
            paint.text(egui::pos2(info.max.x - 18.0, ry), egui::Align2::RIGHT_CENTER, v, egui::FontId::proportional(14.0), pal.text);
            if ri + 1 < rows.len() {
                paint.line_segment([egui::pos2(info.min.x + 14.0, ry + 20.0), egui::pos2(info.max.x - 14.0, ry + 20.0)], egui::Stroke::new(1.0_f32, pal.panel2));
            }
        }

        // clicking Install (or pressing A) opens the confirm dialog (drawn by update)
        if self.dialog.is_none() && (btn_resp.clicked() || a_edge) && idle {
            self.confirm_sel = 1;
            self.open_dialog(Dlg::Confirm);
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }

        hint_bar(paint, content, pal, "[A] Install   ·   [B] Back   ·   scroll / ↑↓ read");
    }

    fn tmpl_btn(&self, dp: &egui::Painter, rect: egui::Rect, label: &str, selected: bool, pal: Pal, pop: f32) {
        dp.rect_filled(rect, egui::Rounding::same(10.0), pal.btn_fill);
        if selected {
            dp.rect_filled(rect, egui::Rounding::same(10.0), Color32::from_rgba_unmultiplied(self.accent.r(), self.accent.g(), self.accent.b(), 42));
            dp.rect_stroke(rect, egui::Rounding::same(10.0), egui::Stroke::new(2.6_f32, self.accent));
        } else {
            dp.rect_stroke(rect, egui::Rounding::same(10.0), egui::Stroke::new(1.2_f32, pal.border));
        }
        let col = if selected { tint_toward(self.accent, Color32::WHITE, 0.2) } else { pal.text };
        dp.text(rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(18.0 * pop), col);
    }

    /// Draws the active dialog (confirm / disclaimer) with the shared popup fade.
    #[allow(clippy::too_many_arguments)]
    fn render_dialog(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, screen: egui::Rect, pal: Pal, shop_ease: f32, dt: f32, ai_opt: Option<usize>, a_edge: bool, b_edge: bool, nl: bool, nr: bool, interactive: bool) -> bool {
        let dtarget = if self.dialog.is_some() && !self.dialog_closing { 1.0 } else { 0.0 };
        self.dialog_anim += (dtarget - self.dialog_anim) * (dt * 12.0).min(1.0);
        if self.dialog_anim < 0.004 && self.dialog_closing {
            self.dialog = None;
            self.dialog_closing = false;
        }
        let Some(kind) = self.dialog else { return false };
        let de = { let a = self.dialog_anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };
        let pop = 0.90 + 0.10 * de;
        let mut dp = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("shop_dialog")));
        dp.set_opacity(shop_ease * de);
        dp.rect_filled(screen, egui::Rounding::ZERO, Color32::from_black_alpha(195));
        let handle = interactive && !self.dialog_closing;

        match kind {
            Dlg::Confirm => {
                let title = ai_opt.map(|i| self.apps[i].title.clone()).unwrap_or_default();
                let w = 460.0 * pop;
                let h = 210.0 * pop;
                let dlg = egui::Rect::from_center_size(screen.center(), egui::Vec2::new(w, h));
                dp.rect_filled(dlg.translate(egui::Vec2::new(0.0, 10.0)), egui::Rounding::same(18.0), Color32::from_black_alpha(90));
                dp.rect_filled(dlg, egui::Rounding::same(18.0), pal.panel);
                dp.rect_stroke(dlg, egui::Rounding::same(18.0), egui::Stroke::new(1.5_f32, pal.border));
                dp.text(egui::pos2(dlg.center().x, dlg.min.y + 40.0 * pop), egui::Align2::CENTER_CENTER, "Install this game?", egui::FontId::proportional(15.0 * pop), pal.muted);
                dp.text(egui::pos2(dlg.center().x, dlg.center().y - 14.0 * pop), egui::Align2::CENTER_CENTER, &title, egui::FontId::proportional(19.0 * pop), pal.text);
                let bw = w * 0.42;
                let bh = 46.0 * pop;
                let by = dlg.max.y - bh - 18.0 * pop;
                let gap = w * 0.05;
                let no = egui::Rect::from_min_size(egui::pos2(dlg.center().x - gap * 0.5 - bw, by), egui::Vec2::new(bw, bh));
                let yes = egui::Rect::from_min_size(egui::pos2(dlg.center().x + gap * 0.5, by), egui::Vec2::new(bw, bh));
                let no_resp = ui.allocate_rect(no, egui::Sense::click());
                let yes_resp = ui.allocate_rect(yes, egui::Sense::click());
                if handle {
                    if nl && self.confirm_sel > 0 { self.confirm_sel = 0; crate::ui_audio::play_move(); }
                    if nr && self.confirm_sel < 1 { self.confirm_sel = 1; crate::ui_audio::play_move(); }
                    if no_resp.hovered() { self.confirm_sel = 0; }
                    if yes_resp.hovered() { self.confirm_sel = 1; }
                }
                self.tmpl_btn(&dp, no, "Cancel", self.confirm_sel == 0, pal, pop);
                self.tmpl_btn(&dp, yes, "Install", self.confirm_sel == 1, pal, pop);
                if handle {
                    if b_edge || no_resp.clicked() {
                        self.dialog_closing = true;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                    } else if a_edge || yes_resp.clicked() {
                        let install = yes_resp.clicked() || self.confirm_sel == 1;
                        self.dialog_closing = true;
                        if install {
                            if let Some(i) = ai_opt {
                                self.start_install(i);
                            }
                            crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
                        } else {
                            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                        }
                    }
                }
            }
            Dlg::Disclaimer => {
                let w = 660.0 * pop;
                let h = 320.0 * pop;
                let dlg = egui::Rect::from_center_size(screen.center(), egui::Vec2::new(w, h));
                dp.rect_filled(dlg.translate(egui::Vec2::new(0.0, 10.0)), egui::Rounding::same(18.0), Color32::from_black_alpha(90));
                dp.rect_filled(dlg, egui::Rounding::same(18.0), pal.panel);
                dp.rect_stroke(dlg, egui::Rounding::same(18.0), egui::Stroke::new(1.5_f32, pal.border));
                dp.text(egui::pos2(dlg.center().x, dlg.min.y + 40.0 * pop), egui::Align2::CENTER_CENTER, "Welcome to the Homebrew Shop", egui::FontId::proportional(21.0 * pop), pal.text);
                let lines = [
                    "Games are provided by the Homebrew App Store (hb-app.store/switch).",
                    "NeXium uses their public API — we don't host any of these apps.",
                    "",
                    "Many games may need additional ROMs or assets to actually run.",
                    "NeXium is an experimental, early-stage Switch emulator, so expect",
                    "bugs and games that don't work yet.",
                ];
                for (i, ln) in lines.iter().enumerate() {
                    dp.text(egui::pos2(dlg.min.x + 40.0 * pop, dlg.min.y + 78.0 * pop + i as f32 * 24.0 * pop), egui::Align2::LEFT_TOP, *ln, egui::FontId::proportional(15.0 * pop), if ln.is_empty() { pal.muted } else { pal.muted });
                }
                let bw = 220.0 * pop;
                let bh = 46.0 * pop;
                let ok = egui::Rect::from_min_size(egui::pos2(dlg.center().x - bw * 0.5, dlg.max.y - bh - 20.0 * pop), egui::Vec2::new(bw, bh));
                let ok_resp = ui.allocate_rect(ok, egui::Sense::click());
                self.tmpl_btn(&dp, ok, "Continue", true, pal, pop);
                if handle && (a_edge || b_edge || ok_resp.clicked()) {
                    self.dialog_closing = true;
                    crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
                }
            }
        }
        true
    }
}

fn hint_bar(paint: &egui::Painter, content: egui::Rect, pal: Pal, text: &str) {
    let bar = egui::Rect::from_min_max(egui::pos2(content.min.x, content.max.y - 40.0), content.max);
    // soft shadow so the solid bar doesn't hard-cut the content above it
    let shadow = egui::Rect::from_min_max(egui::pos2(bar.min.x, bar.min.y - 16.0), egui::pos2(bar.max.x, bar.min.y));
    vfade(paint, shadow, Color32::from_black_alpha(70), false);
    // opaque elevated bar — legible over the scrolling grid behind it
    let fill = tint_toward(pal.bg, pal.panel2, 0.7);
    paint.rect_filled(bar, egui::Rounding::ZERO, fill);
    paint.line_segment([bar.min, egui::pos2(bar.max.x, bar.min.y)], egui::Stroke::new(1.0_f32, pal.border));
    paint.text(egui::pos2(content.max.x - 24.0, bar.center().y), egui::Align2::RIGHT_CENTER, text, egui::FontId::proportional(13.0), pal.text);
}

fn open_url(url: &str) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

/// Horizontal gradient from `color` (solid at the outer edge) to transparent,
/// used to fade the marquee icons out at the left/right band edges.
fn hfade(paint: &egui::Painter, rect: egui::Rect, color: Color32, solid_left: bool) {
    let clear = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 0);
    let (lc, rc) = if solid_left { (color, clear) } else { (clear, color) };
    grad_quad(paint, rect, lc, rc, lc, rc);
}

/// Vertical gradient from transparent (top) to `color` (bottom) when `solid_top`
/// is false, else the reverse.
fn vfade(paint: &egui::Painter, rect: egui::Rect, color: Color32, solid_top: bool) {
    let clear = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 0);
    let (tc, bc) = if solid_top { (color, clear) } else { (clear, color) };
    grad_quad(paint, rect, tc, tc, bc, bc);
}

fn grad_quad(paint: &egui::Painter, rect: egui::Rect, lt: Color32, rt: Color32, lb: Color32, rb: Color32) {
    let mut mesh = egui::epaint::Mesh::default();
    let uv = egui::epaint::WHITE_UV;
    let v = |p: egui::Pos2, c: Color32| egui::epaint::Vertex { pos: p, uv, color: c };
    mesh.vertices.push(v(rect.left_top(), lt));
    mesh.vertices.push(v(rect.right_top(), rt));
    mesh.vertices.push(v(rect.right_bottom(), rb));
    mesh.vertices.push(v(rect.left_bottom(), lb));
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    paint.add(egui::Shape::mesh(mesh));
}

fn human_size(bytes: u64) -> String {
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        format!("{:.0} KB", kb)
    } else {
        format!("{:.1} MB", kb / 1024.0)
    }
}

fn install_app(app: &ShopApp, state: &Arc<Mutex<DownloadInfo>>) -> bool {
    let tmp = std::env::temp_dir().join(format!("nexium_shop_{}.zip", app.name));
    if let Ok(mut g) = state.lock() {
        g.progress = 0.2;
    }
    let out = std::process::Command::new("curl")
        .arg("-sSL")
        .arg("--max-time")
        .arg("240")
        .arg("-o")
        .arg(&tmp)
        .arg(app.zip_url())
        .status();
    if !matches!(out, Ok(s) if s.success()) || !tmp.exists() {
        return false;
    }
    if let Ok(mut g) = state.lock() {
        g.progress = 0.7;
    }
    let dest = nexium_common::paths::sdmc_dir();
    let _ = std::fs::create_dir_all(&dest);
    let unz = std::process::Command::new("unzip")
        .arg("-o")
        .arg(&tmp)
        .arg("-d")
        .arg(&dest)
        .status();
    let _ = std::fs::remove_file(&tmp);
    matches!(unz, Ok(s) if s.success())
}
