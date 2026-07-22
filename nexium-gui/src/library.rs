use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct DownloadInfo {
    pub progress: f32,
    pub done: bool,
    pub ok: bool,
}

pub struct GameEntry {
    pub path: PathBuf,
    pub title: String,
    pub author: String,
    pub format: &'static str,
    pub size: u64,
    pub icon: Option<egui::ColorImage>,
    pub dominant_color: egui::Color32,
    pub download: Option<Arc<Mutex<DownloadInfo>>>,
}

pub struct Library {
    pub games: Vec<GameEntry>,
    pub textures: Vec<Option<egui::TextureHandle>>,
    pub selected: Option<usize>,
    pub loaded: bool,
    rx: Option<Receiver<Vec<GameEntry>>>,
}

impl Library {
    pub fn new() -> Self {
        Self {
            games: Vec::new(),
            textures: Vec::new(),
            selected: None,
            loaded: false,
            rx: None,
        }
    }

    pub fn rescan(&mut self, ctx: &egui::Context, extra_dirs: &[PathBuf]) {
        if self.games.is_empty() {
            self.loaded = false;
        }
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        let extra: Vec<PathBuf> = extra_dirs.to_vec();
        std::thread::spawn(move || {
            let games = scan(&extra);
            let _ = tx.send(games);
            ctx.request_repaint();
        });
        self.rx = Some(rx);
    }

    pub fn poll(&mut self) {
        if let Some(rx) = &self.rx {
            if let Ok(games) = rx.try_recv() {
                self.textures = (0..games.len()).map(|_| None).collect();
                self.games = games;
                self.loaded = true;
                self.selected = None;
                self.rx = None;
            }
        }
    }

    pub fn index_of_path(&self, path: &Path) -> Option<usize> {
        self.games.iter().position(|g| g.path == path)
    }

    pub fn move_to_front(&mut self, idx: usize) -> usize {
        if idx == 0 || idx >= self.games.len() {
            return idx.min(self.games.len().saturating_sub(1));
        }
        let game = self.games.remove(idx);
        self.games.insert(0, game);
        if idx < self.textures.len() {
            let tex = self.textures.remove(idx);
            self.textures.insert(0, tex);
        }
        if let Some(sel) = self.selected {
            self.selected = Some(if sel == idx {
                0
            } else if sel < idx {
                sel + 1
            } else {
                sel
            });
        }
        0
    }

    pub fn add_download(
        &mut self,
        title: String,
        icon: Option<egui::ColorImage>,
        info: Arc<Mutex<DownloadInfo>>,
    ) {
        let path = PathBuf::from(format!("__downloading__/{title}"));
        if self.games.iter().any(|g| g.path == path) {
            return;
        }
        let dominant = icon
            .as_ref()
            .map(sample_dominant)
            .unwrap_or(egui::Color32::from_rgb(0x2F, 0xB4, 0xEF));
        self.games.insert(
            0,
            GameEntry {
                path,
                title,
                author: String::new(),
                format: "NRO",
                size: 0,
                icon,
                dominant_color: dominant,
                download: Some(info),
            },
        );
        self.textures.insert(0, None);
        if let Some(sel) = self.selected {
            self.selected = Some(sel + 1);
        }
    }

    pub fn prune_downloads(&mut self) -> bool {
        let mut rescan = false;
        let mut i = 0;
        while i < self.games.len() {
            let remove = if let Some(dl) = &self.games[i].download {
                dl.lock().ok().map(|g| g.done).unwrap_or(false)
            } else {
                false
            };
            if remove {
                let ok = self.games[i]
                    .download
                    .as_ref()
                    .and_then(|d| d.lock().ok().map(|g| g.ok))
                    .unwrap_or(false);
                self.games.remove(i);
                if i < self.textures.len() {
                    self.textures.remove(i);
                }
                if ok {
                    rescan = true;
                }
            } else {
                i += 1;
            }
        }
        rescan
    }

    pub fn set_icon(&mut self, idx: usize, img: egui::ColorImage) {
        if idx >= self.games.len() {
            return;
        }
        self.games[idx].dominant_color = sample_dominant(&img);
        self.games[idx].icon = Some(img);
        if idx < self.textures.len() {
            self.textures[idx] = None;
        }
    }

    pub fn texture(&mut self, ctx: &egui::Context, idx: usize) -> Option<egui::TextureHandle> {
        if idx >= self.textures.len() || idx >= self.games.len() {
            return None;
        }
        if self.textures[idx].is_none() {
            if let Some(img) = self.games[idx].icon.take() {
                let handle = ctx.load_texture(
                    format!("game_icon_{idx}"),
                    img,
                    egui::TextureOptions::LINEAR,
                );
                self.textures[idx] = Some(handle);
            }
        }
        self.textures[idx].clone()
    }
}

fn scan(extra_dirs: &[PathBuf]) -> Vec<GameEntry> {
    let mut out = Vec::new();
    collect(&nexium_common::paths::nro_dir(), 0, &mut out);
    collect(
        &nexium_common::paths::sdmc_dir().join("switch"),
        0,
        &mut out,
    );
    for dir in extra_dirs {
        collect(dir, 0, &mut out);
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    let pref = |p: &Path| -> u8 {
        if p.components()
            .any(|c| c.as_os_str().eq_ignore_ascii_case("switch"))
        {
            1
        } else {
            0
        }
    };
    out.sort_by(|a, b| {
        a.title
            .to_lowercase()
            .cmp(&b.title.to_lowercase())
            .then_with(|| pref(&b.path).cmp(&pref(&a.path)))
            .then_with(|| a.path.cmp(&b.path))
    });
    out.dedup_by(|a, b| a.title.eq_ignore_ascii_case(&b.title));
    out.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    out
}

fn collect(dir: &Path, depth: usize, out: &mut Vec<GameEntry>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth < 2 {
                collect(&path, depth + 1, out);
            }
            continue;
        }
        if let Some(game) = read_entry(&path) {
            out.push(game);
        }
    }
}

fn read_entry(path: &Path) -> Option<GameEntry> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())?;
    let format = match ext.as_str() {
        "nro" => "NRO",
        "dxci" => "XCI",
        "dnsp" => "NSP",
        _ => return None,
    };
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown")
        .to_string();

    let meta = if format == "NRO" {
        nexium_loader::read_nro_metadata(path)
    } else {
        nexium_loader::read_container_metadata(path)
    };
    let (title, author, icon) = match meta {
        Some(meta) => {
            let title = if meta.title.is_empty() {
                stem.clone()
            } else {
                meta.title
            };
            let icon = meta.icon_jpeg.as_deref().and_then(decode_jpeg);
            (title, meta.author, icon)
        }
        None => (stem.clone(), String::new(), None),
    };

    let icon = custom_icon_path(path)
        .filter(|p| p.exists())
        .and_then(|p| std::fs::read(&p).ok())
        .and_then(|b| decode_jpeg(&b))
        .or(icon);

    Some(GameEntry {
        path: path.to_path_buf(),
        title,
        author,
        format,
        size,
        dominant_color: icon
            .as_ref()
            .map(sample_dominant)
            .unwrap_or(egui::Color32::from_rgb(0x2F, 0xB4, 0xEF)),
        icon,
        download: None,
    })
}

pub fn custom_icon_path(game_path: &Path) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let base = directories::BaseDirs::new()?
        .config_dir()
        .join("NeXium")
        .join("icons");
    let mut h = std::collections::hash_map::DefaultHasher::new();
    game_path.hash(&mut h);
    Some(base.join(format!("{:016x}.png", h.finish())))
}

pub fn decode_icon(bytes: &[u8]) -> Option<egui::ColorImage> {
    decode_jpeg(bytes)
}

fn decode_jpeg(bytes: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        img.as_raw(),
    ))
}

fn sample_dominant(img: &egui::ColorImage) -> egui::Color32 {
    let pixels = &img.pixels;
    if pixels.is_empty() {
        return egui::Color32::from_rgb(0x2F, 0xB4, 0xEF);
    }
    const NB: usize = 24;
    let stride = (pixels.len() / 4096).max(1);
    let mut weight = [0.0f32; NB];
    let mut sr = [0u64; NB];
    let mut sg = [0u64; NB];
    let mut sb = [0u64; NB];
    let mut cnt = [0u64; NB];
    let hue_of = |r: f32, g: f32, b: f32, mx: f32, chroma: f32| -> f32 {
        if chroma < 0.001 {
            0.0
        } else if mx == r {
            ((g - b) / chroma).rem_euclid(6.0) / 6.0
        } else if mx == g {
            ((b - r) / chroma + 2.0) / 6.0
        } else {
            ((r - g) / chroma + 4.0) / 6.0
        }
    };
    for px in pixels.iter().step_by(stride) {
        if px.a() < 40 {
            continue;
        }
        let (rf, gf, bf) = (px.r() as f32 / 255.0, px.g() as f32 / 255.0, px.b() as f32 / 255.0);
        let mx = rf.max(gf).max(bf);
        let mn = rf.min(gf).min(bf);
        let chroma = mx - mn;
        if mx < 0.16 || chroma < 0.14 {
            continue;
        }
        let sat = chroma / (mx + 0.001);
        let bucket = ((hue_of(rf, gf, bf, mx, chroma) * NB as f32) as usize) % NB;
        weight[bucket] += sat * mx;
        sr[bucket] += px.r() as u64;
        sg[bucket] += px.g() as u64;
        sb[bucket] += px.b() as u64;
        cnt[bucket] += 1;
    }
    let best = (0..NB)
        .max_by(|&i, &j| weight[i].partial_cmp(&weight[j]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    if weight[best] <= 0.0 {
        return egui::Color32::from_rgb(0x2F, 0xB4, 0xEF);
    }
    let (mut rr, mut gg, mut bb, mut nn) = (0u64, 0u64, 0u64, 0u64);
    for d in [NB - 1, 0, 1] {
        let bk = (best + d) % NB;
        rr += sr[bk];
        gg += sg[bk];
        bb += sb[bk];
        nn += cnt[bk];
    }
    let nn = nn.max(1);
    let r = (rr / nn) as f32 / 255.0;
    let g = (gg / nn) as f32 / 255.0;
    let b = (bb / nn) as f32 / 255.0;
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let chroma = mx - mn;
    let lightness = (mx + mn) * 0.5;
    let saturation = if lightness < 0.5 {
        chroma / (mx + mn + 0.001)
    } else {
        chroma / (2.0 - mx - mn + 0.001)
    };
    let boosted_s = (saturation * 2.2).min(1.0);
    let target_l = 0.48f32;
    let hue = if chroma < 0.001 {
        0.0f32
    } else if mx == r {
        ((g - b) / chroma).rem_euclid(6.0) / 6.0
    } else if mx == g {
        ((b - r) / chroma + 2.0) / 6.0
    } else {
        ((r - g) / chroma + 4.0) / 6.0
    };
    let q = if target_l < 0.5 {
        target_l * (1.0 + boosted_s)
    } else {
        target_l + boosted_s - target_l * boosted_s
    };
    let p = 2.0 * target_l - q;
    let hue_to_rgb = |mut t: f32| -> f32 {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 0.5 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };
    let fr = hue_to_rgb(hue + 1.0 / 3.0);
    let fg = hue_to_rgb(hue);
    let fb = hue_to_rgb(hue - 1.0 / 3.0);
    egui::Color32::from_rgb((fr * 255.0) as u8, (fg * 255.0) as u8, (fb * 255.0) as u8)
}
