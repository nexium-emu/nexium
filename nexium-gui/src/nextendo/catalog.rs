use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TitleInfo {
    pub title_id: u64,
    pub version: String,
}

impl TitleInfo {
    pub fn compatible(&self) -> Option<&'static nexium_common::nextendo::CompatibleTitle> {
        nexium_common::nextendo::compatible_title(self.title_id)
    }

    pub fn version_ok(&self) -> bool {
        nexium_common::nextendo::version_matches(self.title_id, &self.version)
    }
}

type Known = Arc<Mutex<HashMap<PathBuf, Option<TitleInfo>>>>;

pub struct Catalog {
    known: Known,
    queued: HashSet<PathBuf>,
    jobs: Option<Sender<PathBuf>>,
}

impl Catalog {
    pub fn new(ctx: egui::Context) -> Self {
        let known: Known = Arc::new(Mutex::new(HashMap::new()));
        let (jobs, inbox) = channel::<PathBuf>();
        let results = known.clone();
        let spawned = std::thread::Builder::new()
            .name("nextendo-catalog".into())
            .spawn(move || {
                for path in inbox {
                    let info = resolve(&path);
                    if let Ok(mut known) = results.lock() {
                        known.insert(path, info);
                    }
                    ctx.request_repaint();
                }
            });
        Self {
            known,
            queued: HashSet::new(),
            jobs: spawned.ok().map(|_| jobs),
        }
    }

    pub fn info(&mut self, path: &Path) -> Option<TitleInfo> {
        if let Some(entry) = self.known.lock().ok().and_then(|known| known.get(path).cloned()) {
            return entry;
        }
        if self.queued.insert(path.to_path_buf()) {
            if let Some(jobs) = &self.jobs {
                let _ = jobs.send(path.to_path_buf());
            }
        }
        None
    }

    pub fn path_for_title(&self, title_id: u64) -> Option<PathBuf> {
        let known = self.known.lock().ok()?;
        known.iter().find_map(|(path, info)| {
            info.as_ref()
                .filter(|info| info.title_id == title_id)
                .map(|_| path.clone())
        })
    }

    pub fn forget(&mut self, path: &Path) {
        self.queued.remove(path);
        if let Ok(mut known) = self.known.lock() {
            known.remove(path);
        }
    }

    pub fn forget_all(&mut self) {
        self.queued.clear();
        if let Ok(mut known) = self.known.lock() {
            known.clear();
        }
    }
}

fn resolve(path: &Path) -> Option<TitleInfo> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(extension.as_str(), "dxci" | "dnsp") {
        return None;
    }
    let title_id = nexium_loader::read_application_title_id(path).ok().flatten()?;
    let update = nexium_loader::content::list_game_content(
        &nexium_common::paths::content_dir(),
        title_id,
    )
    .ok()
    .and_then(|content| {
        content.entries.into_iter().find(|entry| {
            entry.kind == nexium_loader::content::ContentKind::Update && entry.enabled
        })
    })
    .map(|entry| entry.display_version)
    .filter(|version| !version.trim().is_empty());
    let version = update
        .or_else(|| nexium_loader::read_container_metadata(path).map(|metadata| metadata.version))
        .unwrap_or_default();
    Some(TitleInfo { title_id, version })
}
