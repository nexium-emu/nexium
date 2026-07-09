use crate::homebrew::{self, ShopApp};
use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Fetch {
    done: bool,
    apps: Vec<ShopApp>,
}

#[derive(Default)]
struct InstallState {
    progress: f32,
    done: bool,
    ok: bool,
    msg: String,
}

struct InstallJob {
    title: String,
    state: Arc<Mutex<InstallState>>,
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
    icon_jobs: Arc<Mutex<HashMap<usize, Option<Vec<u8>>>>>,
    icon_requested: HashSet<usize>,
    search: String,
    editing: bool,
    filtered: Vec<usize>,
    selected: usize,
    scroll: f32,
    view: View,
    install: Option<InstallJob>,
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
            icon_jobs: Arc::new(Mutex::new(HashMap::new())),
            icon_requested: HashSet::new(),
            search: String::new(),
            editing: false,
            filtered: Vec::new(),
            selected: 0,
            scroll: 0.0,
            view: View::Grid,
            install: None,
        }
    }

    pub fn open(&mut self) {
        self.open = true;
        self.view = View::Grid;
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
            .filter(|&i| q.is_empty() || self.apps[i].title.to_lowercase().contains(&q) || self.apps[i].author.to_lowercase().contains(&q))
            .collect();
        if self.selected >= self.filtered.len() {
            self.selected = self.filtered.len().saturating_sub(1);
        }
    }

    fn start_install(&mut self, idx: usize) {
        let app = self.apps[idx].clone();
        let title = app.title.clone();
        let state = Arc::new(Mutex::new(InstallState::default()));
        let s2 = state.clone();
        std::thread::spawn(move || {
            let ok = install_app(&app, &s2);
            if let Ok(mut g) = s2.lock() {
                g.done = true;
                g.ok = ok;
                g.progress = 1.0;
                if !ok && g.msg.is_empty() {
                    g.msg = "Install failed".into();
                }
            }
        });
        self.install = Some(InstallJob { title, state });
    }

    pub fn update(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if !self.open && self.anim < 0.004 {
            return;
        }
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.anim += ((if self.open { 1.0 } else { 0.0 }) - self.anim) * (dt * 14.0).min(1.0);
        let ease = { let a = self.anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };

        // ingest fetch result
        if !self.loaded {
            let apps = self
                .fetch
                .as_ref()
                .and_then(|f| f.lock().ok().and_then(|g| if g.done { Some(g.apps.clone()) } else { None }));
            if let Some(apps) = apps {
                self.apps = apps;
                self.icons = vec![None; self.apps.len()];
                self.loaded = true;
                self.refilter();
            }
        }

        // build any downloaded icon textures
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
        }

        let screen = ctx.screen_rect();
        let mut paint = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("shop")));
        paint.set_opacity(ease);
        paint.rect_filled(screen, egui::Rounding::ZERO, egui::Color32::from_rgb(0x0C, 0x0C, 0x10));

        // top bar
        let accent = egui::Color32::from_rgb(0xE6, 0x00, 0x12);
        let bar = egui::Rect::from_min_size(screen.min, egui::Vec2::new(screen.width(), 54.0));
        paint.rect_filled(bar, egui::Rounding::ZERO, egui::Color32::from_rgb(0x16, 0x16, 0x1C));
        paint.text(egui::pos2(bar.min.x + 24.0, bar.center().y), egui::Align2::LEFT_CENTER, "NeXium Homebrew Shop", egui::FontId::proportional(20.0), egui::Color32::WHITE);

        // search field
        let field = egui::Rect::from_min_size(egui::pos2(screen.center().x - 180.0, 10.0), egui::Vec2::new(360.0, 34.0));
        let field_resp = ui.allocate_rect(field, egui::Sense::click());
        paint.rect_filled(field, egui::Rounding::same(8.0), egui::Color32::from_rgb(0x24, 0x24, 0x2C));
        paint.rect_stroke(field, egui::Rounding::same(8.0), egui::Stroke::new(if self.editing { 2.0 } else { 1.0 }, if self.editing { accent } else { egui::Color32::from_gray(0x40) }));
        let disp = if self.search.is_empty() { "Search games…".to_string() } else { self.search.clone() };
        paint.text(egui::pos2(field.min.x + 12.0, field.center().y), egui::Align2::LEFT_CENTER, &disp, egui::FontId::proportional(15.0), if self.search.is_empty() { egui::Color32::from_gray(0x80) } else { egui::Color32::WHITE });
        if field_resp.clicked() {
            self.editing = true;
        }

        let content = egui::Rect::from_min_max(egui::pos2(screen.min.x, bar.max.y), screen.max);

        // ---- input ----
        let (kb_esc, kb_enter, text_events) = ctx.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::Enter), i.events.clone()));
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

        let detail_idx = if let View::Detail(i) = self.view { Some(i) } else { None };
        match detail_idx {
            None => self.draw_grid(ctx, ui, &paint, content),
            Some(i) => self.draw_detail(ctx, ui, &paint, content, i),
        }

        // close on Escape (when not editing)
        if kb_esc && !self.editing {
            match self.view {
                View::Detail(_) => self.view = View::Grid,
                View::Grid => {
                    self.open = false;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        }

        ctx.request_repaint();
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

    fn draw_grid(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect) {
        if !self.loaded {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "Loading shop…", egui::FontId::proportional(20.0), egui::Color32::from_gray(0x90));
            return;
        }
        if self.filtered.is_empty() {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "No games found.", egui::FontId::proportional(18.0), egui::Color32::from_gray(0x90));
            return;
        }
        let cols = 4usize;
        let pad = 20.0;
        let gap = 18.0;
        let tile_w = ((content.width() - pad * 2.0 - gap * (cols as f32 - 1.0)) / cols as f32).min(260.0);
        let tile_h = tile_w * 0.62;
        let cell_h = tile_h + 34.0;
        let x0 = content.min.x + pad;
        let y0 = content.min.y + pad - self.scroll;

        // wheel scroll
        let wheel = ui.input(|i| i.smooth_scroll_delta.y);
        let rows = (self.filtered.len() + cols - 1) / cols;
        let total_h = rows as f32 * (cell_h + gap);
        let max_scroll = (total_h - content.height() + pad * 2.0).max(0.0);
        if wheel.abs() > 0.1 {
            self.scroll = (self.scroll - wheel).clamp(0.0, max_scroll);
        }

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
            clip.rect_filled(tile, egui::Rounding::same(10.0), egui::Color32::from_rgb(0x1C, 0x1C, 0x24));
            if let Some(Some(tex)) = self.icons.get(ai) {
                crate::carousel::draw_rounded_image(&clip, tex.id(), tile, 10.0, egui::Color32::WHITE);
            } else {
                clip.text(tile.center(), egui::Align2::CENTER_CENTER, &self.apps[ai].title, egui::FontId::proportional(13.0), egui::Color32::from_gray(0x70));
            }
            if sel {
                clip.rect_stroke(tile, egui::Rounding::same(10.0), egui::Stroke::new(2.5, egui::Color32::from_rgb(0x4F, 0x9D, 0xFF)));
            }
            clip.text(egui::pos2(tile.min.x + 2.0, tile.max.y + 6.0), egui::Align2::LEFT_TOP, &self.apps[ai].title, egui::FontId::proportional(13.0), egui::Color32::from_gray(0xE0));
            let resp = ui.allocate_rect(tile, egui::Sense::click());
            if resp.clicked() {
                self.selected = vi;
                open_detail = Some(ai);
            }
        }
        if let Some(ai) = open_detail {
            self.view = View::Detail(ai);
            crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        }
    }

    fn draw_detail(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui, paint: &egui::Painter, content: egui::Rect, ai: usize) {
        let app = self.apps[ai].clone();
        self.request_icon(ai);
        let pad = 40.0;
        let left = egui::Rect::from_min_size(egui::pos2(content.min.x + pad, content.min.y + pad), egui::Vec2::new(content.width() * 0.5, content.height() - pad * 2.0));
        // icon
        let icon_r = egui::Rect::from_min_size(left.min, egui::Vec2::splat(140.0));
        if let Some(Some(tex)) = self.icons.get(ai) {
            crate::carousel::draw_rounded_image(paint, tex.id(), icon_r, 16.0, egui::Color32::WHITE);
        } else {
            paint.rect_filled(icon_r, egui::Rounding::same(16.0), egui::Color32::from_rgb(0x22, 0x22, 0x2A));
        }
        paint.text(egui::pos2(left.min.x, icon_r.max.y + 20.0), egui::Align2::LEFT_TOP, &app.title, egui::FontId::proportional(26.0), egui::Color32::WHITE);
        paint.text(egui::pos2(left.min.x, icon_r.max.y + 54.0), egui::Align2::LEFT_TOP, &format!("{}  ·  v{}  ·  {}", app.author, app.version, human_size(app.filesize)), egui::FontId::proportional(14.0), egui::Color32::from_gray(0x9A));

        // description (wrapped)
        let desc = if app.details.is_empty() { app.description.clone() } else { app.details.clone() };
        let galley = ui.fonts(|f| f.layout(desc, egui::FontId::proportional(15.0), egui::Color32::from_gray(0xCC), left.width()));
        paint.galley(egui::pos2(left.min.x, icon_r.max.y + 90.0), galley, egui::Color32::from_gray(0xCC));

        // install button (right side)
        let installing = self.install.is_some();
        let (progress, done, ok, msg) = self
            .install
            .as_ref()
            .and_then(|j| j.state.lock().ok().map(|g| (g.progress, g.done, g.ok, g.msg.clone())))
            .unwrap_or((0.0, false, false, String::new()));

        let btn = egui::Rect::from_min_size(egui::pos2(content.max.x - 320.0, content.min.y + pad + 40.0), egui::Vec2::new(260.0, 56.0));
        let btn_resp = ui.allocate_rect(btn, egui::Sense::click());
        let hovered = btn_resp.hovered();
        let btn_col = if installing { egui::Color32::from_gray(0x40) } else if hovered { egui::Color32::from_rgb(0xFF, 0x2A, 0x3E) } else { egui::Color32::from_rgb(0xE6, 0x00, 0x12) };
        paint.rect_filled(btn, egui::Rounding::same(10.0), btn_col);
        let label = if installing {
            if done { if ok { "Installed ✓".to_string() } else { "Failed".to_string() } } else { format!("Installing… {}%", (progress * 100.0) as u32) }
        } else {
            "Install".to_string()
        };
        paint.text(btn.center(), egui::Align2::CENTER_CENTER, &label, egui::FontId::proportional(18.0), egui::Color32::WHITE);
        if installing && !done {
            let bar = egui::Rect::from_min_size(egui::pos2(btn.min.x, btn.max.y + 8.0), egui::Vec2::new(btn.width(), 6.0));
            paint.rect_filled(bar, egui::Rounding::same(3.0), egui::Color32::from_gray(0x30));
            let mut fill = bar;
            fill.set_width(bar.width() * progress);
            paint.rect_filled(fill, egui::Rounding::same(3.0), egui::Color32::from_rgb(0x35, 0xD0, 0x6A));
        }
        if !msg.is_empty() {
            paint.text(egui::pos2(btn.center().x, btn.max.y + 24.0), egui::Align2::CENTER_TOP, &msg, egui::FontId::proportional(13.0), egui::Color32::from_gray(0xA0));
        }

        if btn_resp.clicked() && !installing {
            self.start_install(ai);
            crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
        }
        // on successful completion, mark rescan + clear job
        if done && ok {
            self.need_rescan = true;
        }

        paint.text(egui::pos2(content.min.x + pad, content.max.y - 24.0), egui::Align2::LEFT_CENTER, "[Esc / B]  Back to grid", egui::FontId::proportional(13.0), egui::Color32::from_gray(0x80));
    }
}

fn human_size(bytes: u64) -> String {
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        format!("{:.0} KB", kb)
    } else {
        format!("{:.1} MB", kb / 1024.0)
    }
}

fn install_app(app: &ShopApp, state: &Arc<Mutex<InstallState>>) -> bool {
    let Some(base) = directories::BaseDirs::new() else {
        return false;
    };
    let tmp = std::env::temp_dir().join(format!("nexium_shop_{}.zip", app.name));
    if let Ok(mut g) = state.lock() {
        g.progress = 0.15;
    }
    // download the zip
    let out = std::process::Command::new("curl")
        .arg("-sSL")
        .arg("--max-time")
        .arg("180")
        .arg("-o")
        .arg(&tmp)
        .arg(app.zip_url())
        .status();
    if !matches!(out, Ok(s) if s.success()) || !tmp.exists() {
        if let Ok(mut g) = state.lock() {
            g.msg = "Download failed".into();
        }
        return false;
    }
    if let Ok(mut g) = state.lock() {
        g.progress = 0.65;
    }
    // fortheusers zips lay files under `switch/...`; extract into the sdmc dir
    let dest = nexium_common::paths::sdmc_dir();
    let _ = std::fs::create_dir_all(&dest);
    let unz = std::process::Command::new("unzip")
        .arg("-o")
        .arg(&tmp)
        .arg("-d")
        .arg(&dest)
        .status();
    let _ = std::fs::remove_file(&tmp);
    let _ = base;
    matches!(unz, Ok(s) if s.success())
}
