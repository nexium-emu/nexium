use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

pub struct GameEntry {
    pub path: PathBuf,
    pub title: String,
    pub author: String,
    pub format: &'static str,
    pub size: u64,
    pub icon: Option<egui::ColorImage>,
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
        self.loaded = false;
        self.games.clear();
        self.textures.clear();
        self.selected = None;
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
                self.rx = None;
            }
        }
    }

    pub fn texture(&mut self, ctx: &egui::Context, idx: usize) -> Option<egui::TextureHandle> {
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
    collect(&nexium_common::paths::sdmc_dir().join("switch"), 0, &mut out);
    for dir in extra_dirs {
        collect(dir, 0, &mut out);
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
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

    Some(GameEntry {
        path: path.to_path_buf(),
        title,
        author,
        format,
        size,
        icon,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        img.as_raw(),
    ))
}
