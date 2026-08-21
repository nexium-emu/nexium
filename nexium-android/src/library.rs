use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

pub const ICON_SIZE: usize = 112;

pub struct GameEntry {
    pub path: PathBuf,
    pub title: String,
    pub author: String,
    pub version: String,
    pub icon: Option<Vec<u8>>,
}

pub struct LibraryScan {
    rx: Receiver<Vec<GameEntry>>,
    pub done: bool,
    pub games: Vec<GameEntry>,
}

fn rom_paths(roms_dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(roms_dir)
        .map(|it| {
            it.filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    matches!(
                        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                        Some(ref ext) if ["nro", "dnsp", "dxci", "dnca"].contains(&ext.as_str())
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    paths
}

fn decode_icon(jpeg: &[u8]) -> Option<Vec<u8>> {
    let img = image::load_from_memory(jpeg).ok()?.to_rgba8();
    let scaled = image::imageops::resize(
        &img,
        ICON_SIZE as u32,
        ICON_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    Some(scaled.into_raw())
}

fn read_entry(path: PathBuf) -> GameEntry {
    let is_nro = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("nro"))
        .unwrap_or(false);
    let meta = if is_nro {
        nexium_loader::nro::read_nro_metadata(&path)
    } else {
        nexium_loader::control::read_container_metadata(&path)
    };
    let fallback_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown")
        .to_string();
    match meta {
        Some(meta) => {
            let icon = meta.icon_jpeg.as_deref().and_then(decode_icon);
            GameEntry {
                path,
                title: if meta.title.trim().is_empty() {
                    fallback_title
                } else {
                    meta.title
                },
                author: meta.author,
                version: meta.version,
                icon,
            }
        }
        None => GameEntry {
            path,
            title: fallback_title,
            author: String::new(),
            version: String::new(),
            icon: None,
        },
    }
}

impl LibraryScan {
    pub fn start(roms_dir: PathBuf) -> Self {
        let (tx, rx) = channel();
        let _ = std::thread::Builder::new()
            .name("nexium-libscan".into())
            .spawn(move || {
                let games = rom_paths(&roms_dir).into_iter().map(read_entry).collect();
                let _ = tx.send(games);
            });
        Self {
            rx,
            done: false,
            games: Vec::new(),
        }
    }

    pub fn poll(&mut self) -> bool {
        if self.done {
            return false;
        }
        match self.rx.try_recv() {
            Ok(games) => {
                self.games = games;
                self.done = true;
                true
            }
            Err(_) => false,
        }
    }
}
