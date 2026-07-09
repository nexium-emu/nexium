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
}

fn palette(light: bool) -> Pal {
    if light {
        Pal {
            bg: Color32::from_rgb(0xEC, 0xEC, 0xF1),
            panel: Color32::from_rgb(0xFF, 0xFF, 0xFF),
            panel2: Color32::from_rgb(0xE2, 0xE2, 0xE9),
            text: Color32::from_rgb(0x1E, 0x1E, 0x28),
            muted: Color32::from_rgb(0x60, 0x60, 0x6C),
            border: Color32::from_rgb(0xCE, 0xCE, 0xD6),
        }
    } else {
        Pal {
            bg: Color32::from_rgb(0x14, 0x14, 0x19),
            panel: Color32::from_rgb(0x22, 0x22, 0x2A),
            panel2: Color32::from_rgb(0x2C, 0x2C, 0x36),
            text: Color32::from_rgb(0xEC, 0xEC, 0xF0),
            muted: Color32::from_rgb(0x9A, 0x9A, 0xA6),
            border: Color32::from_rgb(0x34, 0x34, 0x40),
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
    filtered: Vec<usize>,
    selected: usize,
    scroll: f32,
    desc_scroll: f32,
    view: View,
    install: Option<InstallJob>,
    nav_cd: f64,
    a_held: bool,
    b_held: bool,
    confirm_install: bool,
    confirm_sel: usize,
    view_fade: f32,
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
            filtered: Vec::new(),
            selected: 0,
            scroll: 0.0,
            desc_scroll: 0.0,
            view: View::Grid,
            install: None,
            nav_cd: 0.0,
            a_held: false,
            b_held: false,
            confirm_install: false,
            confirm_sel: 1,
            view_fade: 1.0,
        }
    }

    pub fn open(&mut self) {
        self.open = true;
        self.view = View::Grid;
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
        let a_edge = kb_enter || (gp_a && !self.a_held);
        let b_edge = kb_esc || (gp_b && !self.b_held);
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

        // search field (top-right of banner)
        let field = egui::Rect::from_min_size(egui::pos2(screen.max.x - 320.0, 9.0), egui::Vec2::new(292.0, 34.0));
        let field_resp = ui.allocate_rect(field, egui::Sense::click());
        paint.rect_filled(field, egui::Rounding::same(8.0), Color32::from_black_alpha(60));
        paint.rect_stroke(field, egui::Rounding::same(8.0), egui::Stroke::new(if self.editing { 2.0 } else { 1.0 }, if self.editing { Color32::WHITE } else { Color32::from_white_alpha(120) }));
        let disp = if self.search.is_empty() { "Search…".to_string() } else { self.search.clone() };
        paint.text(egui::pos2(field.min.x + 12.0, field.center().y), egui::Align2::LEFT_CENTER, &disp, egui::FontId::proportional(15.0), Color32::from_white_alpha(if self.search.is_empty() { 150 } else { 255 }));
        if field_resp.clicked() {
            self.editing = true;
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
            ctx.request_repaint();
            return;
        }

        let content = egui::Rect::from_min_max(egui::pos2(screen.min.x, bar.max.y), screen.max);
        let detail_idx = if let View::Detail(i) = self.view { Some(i) } else { None };
        match detail_idx {
            None => self.grid(ui, &paint, content, pal, accent, nl, nr, nu, nd, a_edge),
            Some(i) => self.detail(ui, &paint, content, pal, i, a_edge, ui.input(|x| x.smooth_scroll_delta.y), nu, nd, nl, nr),
        }

        // cross-view fade transition
        self.view_fade += (1.0 - self.view_fade) * (dt * 9.0).min(1.0);
        if self.view_fade < 0.995 {
            let a = ((1.0 - self.view_fade) * 255.0) as u8;
            paint.rect_filled(content, egui::Rounding::ZERO, Color32::from_rgba_unmultiplied(pal.bg.r(), pal.bg.g(), pal.bg.b(), a));
        }

        if b_edge {
            if self.confirm_install {
                self.confirm_install = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else {
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
        }

        ctx.request_repaint();
    }

    #[allow(clippy::too_many_arguments)]
    fn grid(&mut self, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect, pal: Pal, accent: Color32, nl: bool, nr: bool, nu: bool, nd: bool, a_edge: bool) {
        if !self.loaded {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "Loading shop…", egui::FontId::proportional(20.0), pal.muted);
            return;
        }
        let n = self.filtered.len();
        if n == 0 {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "No games found.", egui::FontId::proportional(18.0), pal.muted);
            return;
        }
        let cols = 5usize;
        let pad = 34.0;
        let gap = 22.0;
        let tile_w = (content.width() - pad * 2.0 - gap * (cols as f32 - 1.0)) / cols as f32;
        let tile_h = tile_w * 0.58;
        let cell_h = tile_h + 32.0;

        // nav
        if nl && self.selected > 0 { self.selected -= 1; crate::ui_audio::play_move(); }
        if nr && self.selected + 1 < n { self.selected += 1; crate::ui_audio::play_move(); }
        if nd && self.selected + cols < n { self.selected += cols; crate::ui_audio::play_move(); }
        if nu && self.selected >= cols { self.selected -= cols; crate::ui_audio::play_move(); }

        let rows = (n + cols - 1) / cols;
        let total_h = rows as f32 * (cell_h + gap);
        let max_scroll = (total_h - content.height() + pad * 2.0).max(0.0);
        // keep selected visible
        let sel_row = (self.selected / cols) as f32;
        let want = (sel_row * (cell_h + gap) - content.height() * 0.4).clamp(0.0, max_scroll);
        let wheel = ui.input(|i| i.smooth_scroll_delta.y);
        if wheel.abs() > 0.1 {
            self.scroll = (self.scroll - wheel).clamp(0.0, max_scroll);
        } else {
            self.scroll += (want - self.scroll) * 0.2;
        }

        let x0 = content.min.x + pad;
        let y0 = content.min.y + pad - self.scroll;
        let clip = paint.with_clip_rect(content);
        let mut open_detail = None;
        let filtered = self.filtered.clone();
        for (vi, ai) in filtered.iter().copied().enumerate() {
            let r = vi / cols;
            let c = vi % cols;
            let tx = x0 + c as f32 * (tile_w + gap);
            let ty = y0 + r as f32 * (cell_h + gap);
            if ty + cell_h < content.min.y || ty > content.max.y {
                continue;
            }
            self.request_icon(ai);
            let tile = egui::Rect::from_min_size(egui::pos2(tx, ty), egui::Vec2::new(tile_w, tile_h));
            let sel = vi == self.selected;
            if sel {
                clip.rect_filled(tile.expand(4.0), egui::Rounding::same(13.0), Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 90));
                clip.rect_stroke(tile.expand(4.0), egui::Rounding::same(13.0), egui::Stroke::new(2.5, accent));
            }
            clip.rect_filled(tile, egui::Rounding::same(10.0), pal.panel);
            if let Some(Some(tex)) = self.icons.get(ai) {
                crate::carousel::draw_rounded_image(&clip, tex.id(), tile, 10.0, Color32::WHITE);
            } else {
                clip.text(tile.center(), egui::Align2::CENTER_CENTER, &self.apps[ai].title, egui::FontId::proportional(13.0), pal.muted);
            }
            clip.text(egui::pos2(tile.min.x + 2.0, tile.max.y + 6.0), egui::Align2::LEFT_TOP, &self.apps[ai].title, egui::FontId::proportional(13.0), pal.text);
            let resp = ui.allocate_rect(tile, egui::Sense::click());
            if resp.clicked() {
                self.selected = vi;
                open_detail = Some(ai);
            }
        }
        if a_edge && self.selected < n {
            open_detail = Some(self.filtered[self.selected]);
        }
        if let Some(ai) = open_detail {
            self.view = View::Detail(ai);
            self.desc_scroll = 0.0;
            self.view_fade = 0.0;
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }

        // hint bar
        hint_bar(paint, content, pal, "[A] Select   ·   [B] Close   ·   click search to filter");
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn detail(&mut self, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect, pal: Pal, ai: usize, a_edge: bool, wheel: f32, nu: bool, nd: bool, nl: bool, nr: bool) {
        let app = self.apps[ai].clone();
        self.request_icon(ai);
        let pad = 40.0;
        let col_split = content.min.x + content.width() * 0.68;

        // hero icon
        let icon_r = egui::Rect::from_min_size(egui::pos2(content.min.x + pad, content.min.y + pad), egui::Vec2::splat(150.0));
        paint.rect_filled(icon_r.expand(2.0), egui::Rounding::same(18.0), pal.panel2);
        if let Some(Some(tex)) = self.icons.get(ai) {
            crate::carousel::draw_rounded_image(paint, tex.id(), icon_r, 16.0, Color32::WHITE);
        } else {
            paint.rect_filled(icon_r, egui::Rounding::same(16.0), pal.panel);
        }
        let tx = icon_r.max.x + 22.0;
        paint.text(egui::pos2(tx, icon_r.min.y + 12.0), egui::Align2::LEFT_TOP, &app.title, egui::FontId::proportional(28.0), pal.text);
        paint.text(egui::pos2(tx, icon_r.min.y + 52.0), egui::Align2::LEFT_TOP, &app.author, egui::FontId::proportional(16.0), pal.muted);
        paint.text(egui::pos2(tx, icon_r.min.y + 78.0), egui::Align2::LEFT_TOP, &format!("v{}   ·   {}   ·   {}", app.version, human_size(app.filesize), app.license), egui::FontId::proportional(13.0), pal.muted);

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
                paint.rect_stroke(btn.expand(e), egui::Rounding::same(12.0 + e), egui::Stroke::new(2.0, Color32::from_rgba_unmultiplied(BLUE_HI.r(), BLUE_HI.g(), BLUE_HI.b(), (40.0 * pulse) as u8)));
            }
            paint.rect_stroke(btn.expand(3.0), egui::Rounding::same(15.0), egui::Stroke::new(2.5, Color32::from_rgba_unmultiplied(BLUE_HI.r(), BLUE_HI.g(), BLUE_HI.b(), (200.0 * pulse) as u8)));
        }
        let btn_col = if installing { pal.panel2 } else if done && ok { Color32::from_rgb(0x2C, 0xA0, 0x4A) } else if failed { Color32::from_rgb(0xB0, 0x3A, 0x3A) } else if hovered { BLUE_HI } else { BLUE };
        paint.rect_filled(btn, egui::Rounding::same(12.0), btn_col);
        let label = if done && ok {
            "Installed ✓".to_string()
        } else if installing {
            format!("Installing…  {}%", (progress * 100.0) as u32)
        } else if failed {
            "Retry Install".to_string()
        } else {
            "Install".to_string()
        };
        paint.text(btn.center(), egui::Align2::CENTER_CENTER, &label, egui::FontId::proportional(20.0), Color32::WHITE);
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
        paint.rect_stroke(info, egui::Rounding::same(12.0), egui::Stroke::new(1.0, pal.border));
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
                paint.line_segment([egui::pos2(info.min.x + 14.0, ry + 20.0), egui::pos2(info.max.x - 14.0, ry + 20.0)], egui::Stroke::new(1.0, pal.panel2));
            }
        }

        // confirmation flow — matches the emulator's confirm modal (dim + panel + two nav buttons)
        if self.confirm_install {
            if nl && self.confirm_sel > 0 { self.confirm_sel = 0; crate::ui_audio::play_move(); }
            if nr && self.confirm_sel < 1 { self.confirm_sel = 1; crate::ui_audio::play_move(); }
            paint.rect_filled(content, egui::Rounding::ZERO, Color32::from_black_alpha(160));
            let dlg = egui::Rect::from_center_size(content.center(), egui::Vec2::new(460.0, 210.0));
            paint.rect_filled(dlg.translate(egui::Vec2::new(0.0, 10.0)), egui::Rounding::same(18.0), Color32::from_black_alpha(120));
            paint.rect_filled(dlg, egui::Rounding::same(18.0), pal.panel);
            paint.rect_stroke(dlg, egui::Rounding::same(18.0), egui::Stroke::new(1.5, pal.border));
            paint.text(egui::pos2(dlg.center().x, dlg.min.y + 46.0), egui::Align2::CENTER_CENTER, "Install this game?", egui::FontId::proportional(22.0), pal.text);
            paint.text(egui::pos2(dlg.center().x, dlg.min.y + 80.0), egui::Align2::CENTER_CENTER, &app.title, egui::FontId::proportional(15.0), pal.muted);
            let bw = 190.0;
            let no = egui::Rect::from_min_size(egui::pos2(dlg.center().x - bw - 10.0, dlg.max.y - 66.0), egui::Vec2::new(bw, 46.0));
            let yes = egui::Rect::from_min_size(egui::pos2(dlg.center().x + 10.0, dlg.max.y - 66.0), egui::Vec2::new(bw, 46.0));
            let no_resp = ui.allocate_rect(no, egui::Sense::click());
            let yes_resp = ui.allocate_rect(yes, egui::Sense::click());
            if no_resp.hovered() { self.confirm_sel = 0; }
            if yes_resp.hovered() { self.confirm_sel = 1; }
            for (bi, (rect, txt, base)) in [(no, "Cancel", pal.panel2), (yes, "Install", BLUE)].iter().enumerate() {
                let seld = self.confirm_sel == bi;
                paint.rect_filled(*rect, egui::Rounding::same(11.0), if seld && bi == 1 { BLUE_HI } else { *base });
                if seld {
                    paint.rect_stroke(rect.expand(3.0), egui::Rounding::same(14.0), egui::Stroke::new(2.5, BLUE_HI));
                }
                let tc = if bi == 1 { Color32::WHITE } else { pal.text };
                paint.text(rect.center(), egui::Align2::CENTER_CENTER, *txt, egui::FontId::proportional(17.0), tc);
            }
            paint.text(egui::pos2(dlg.center().x, dlg.max.y + 22.0), egui::Align2::CENTER_CENTER, "[←/→] Choose    [A] Confirm    [B] Cancel", egui::FontId::proportional(13.0), pal.muted);
            let confirm = a_edge || no_resp.clicked() || yes_resp.clicked();
            if confirm {
                let install = if no_resp.clicked() { false } else if yes_resp.clicked() { true } else { self.confirm_sel == 1 };
                self.confirm_install = false;
                if install {
                    self.start_install(ai);
                    crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
                } else {
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        } else if (btn_resp.clicked() || a_edge) && idle {
            self.confirm_install = true;
            self.confirm_sel = 1;
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }

        hint_bar(paint, content, pal, "[A] Install   ·   [B] Back   ·   scroll / ↑↓ read");
    }
}

fn hint_bar(paint: &egui::Painter, content: egui::Rect, pal: Pal, text: &str) {
    let bar = egui::Rect::from_min_max(egui::pos2(content.min.x, content.max.y - 40.0), content.max);
    paint.rect_filled(bar, egui::Rounding::ZERO, Color32::from_black_alpha(40));
    paint.line_segment([bar.min, egui::pos2(bar.max.x, bar.min.y)], egui::Stroke::new(1.0, pal.border));
    paint.text(egui::pos2(content.max.x - 24.0, bar.center().y), egui::Align2::RIGHT_CENTER, text, egui::FontId::proportional(13.0), pal.muted);
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
