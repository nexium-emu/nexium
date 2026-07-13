use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Release {
    pub tag: String,
    pub date: String,
    pub commit: String,
    pub url: String,
    pub size: u64,
    pub filename: String,
}

#[derive(Clone)]
pub enum Status {
    Idle,
    Checking,
    UpToDate,
    Available(Release),
    Error(String),
}

struct Download {
    total: u64,
    path: PathBuf,
    done: bool,
    ok: bool,
}

pub struct Updater {
    status: Arc<Mutex<Status>>,
    download: Option<Arc<Mutex<Download>>>,
    downloaded: Option<PathBuf>,
}

impl Default for Updater {
    fn default() -> Self {
        Self::new()
    }
}

impl Updater {
    pub fn new() -> Self {
        Self {
            status: Arc::new(Mutex::new(Status::Idle)),
            download: None,
            downloaded: None,
        }
    }

    pub fn supported() -> bool {
        cfg!(target_os = "linux")
    }

    pub fn status(&self) -> Status {
        self.status.lock().map(|g| g.clone()).unwrap_or(Status::Idle)
    }

    pub fn available_release(&self) -> Option<Release> {
        match self.status() {
            Status::Available(r) => Some(r),
            _ => None,
        }
    }

    pub fn is_downloading(&self) -> bool {
        self.download.as_ref().map_or(false, |d| d.lock().map(|g| !g.done).unwrap_or(false))
    }

    pub fn progress(&self) -> f32 {
        let Some(d) = &self.download else { return 0.0 };
        let Ok(g) = d.lock() else { return 0.0 };
        if g.done {
            return 1.0;
        }
        if g.total == 0 {
            return 0.0;
        }
        let cur = std::fs::metadata(&g.path).map(|m| m.len()).unwrap_or(0);
        (cur as f32 / g.total as f32).clamp(0.0, 1.0)
    }

    pub fn download_result(&self) -> Option<bool> {
        let d = self.download.as_ref()?;
        let g = d.lock().ok()?;
        if g.done {
            Some(g.ok)
        } else {
            None
        }
    }

    #[cfg(target_os = "linux")]
    pub fn check(&self) {
        {
            let mut g = match self.status.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if !matches!(*g, Status::Idle) {
                return;
            }
            *g = Status::Checking;
        }
        let status = self.status.clone();
        std::thread::spawn(move || {
            let next = match fetch_latest() {
                Ok(rel) => {
                    if is_current(&rel.commit) {
                        Status::UpToDate
                    } else {
                        Status::Available(rel)
                    }
                }
                Err(e) => Status::Error(e),
            };
            if let Ok(mut g) = status.lock() {
                *g = next;
            }
        });
    }

    #[cfg(not(target_os = "linux"))]
    pub fn check(&self) {}

    #[cfg(target_os = "linux")]
    pub fn start_download(&mut self, rel: &Release) {
        let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let path = dir.join(&rel.filename);
        let dl = Arc::new(Mutex::new(Download {
            total: rel.size,
            path: path.clone(),
            done: false,
            ok: false,
        }));
        self.download = Some(dl.clone());
        self.downloaded = Some(path.clone());
        let url = rel.url.clone();
        std::thread::spawn(move || {
            let ok = std::process::Command::new("curl")
                .args(["-sL", "--fail", "-o"])
                .arg(&path)
                .arg(&url)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if let Ok(mut g) = dl.lock() {
                g.done = true;
                g.ok = ok;
            }
        });
    }

    #[cfg(not(target_os = "linux"))]
    pub fn start_download(&mut self, _rel: &Release) {}

    #[cfg(target_os = "linux")]
    pub fn install_and_relaunch(&self) -> Result<(), String> {
        let tar = self.downloaded.clone().ok_or("no download")?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let tmp = std::env::temp_dir().join("nexium_update_extract");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
        let ok = std::process::Command::new("tar")
            .arg("-xzf")
            .arg(&tar)
            .arg("-C")
            .arg(&tmp)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return Err("failed to extract archive".into());
        }
        let newbin = find_binary(&tmp, "nexium").ok_or("new binary not found in archive")?;
        backup_current(&exe)?;
        std::fs::copy(&newbin, &exe).map_err(|e| e.to_string())?;
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&exe) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&exe, perms);
            }
        }
        std::process::Command::new(&exe)
            .spawn()
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn install_and_relaunch(&self) -> Result<(), String> {
        Err("unsupported platform".into())
    }
}

#[cfg(target_os = "linux")]
fn is_current(commit: &str) -> bool {
    let hex = |s: &str| -> String {
        s.trim()
            .to_lowercase()
            .chars()
            .take_while(|c| c.is_ascii_hexdigit())
            .collect::<String>()
    };
    let ours = hex(env!("NEXIUM_GIT_HASH"));
    let theirs = hex(commit);
    if ours.is_empty() || theirs.is_empty() {
        return false;
    }
    let n = ours.len().min(theirs.len());
    ours[..n] == theirs[..n]
}

#[cfg(target_os = "linux")]
fn fetch_latest() -> Result<Release, String> {
    let out = std::process::Command::new("curl")
        .args([
            "-sL",
            "--fail",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "User-Agent: nexium-updater",
            "https://api.github.com/repos/nexium-emu/nexium-nightly/releases?per_page=10",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err("could not reach update server".into());
    }
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|_| "bad response".to_string())?;
    let json = list
        .as_array()
        .and_then(|arr| arr.iter().find(|r| !r["draft"].as_bool().unwrap_or(false)).cloned())
        .ok_or("no releases found")?;

    let tag = json["tag_name"].as_str().unwrap_or("").to_string();
    let name = json["name"].as_str().unwrap_or("").to_string();
    let body = json["body"].as_str().unwrap_or("");
    let date = json["published_at"].as_str().unwrap_or("").chars().take(10).collect::<String>();

    let commit = extract_commit(&name)
        .or_else(|| extract_commit(&tag))
        .or_else(|| extract_commit(body))
        .unwrap_or_default();

    let assets = json["assets"].as_array().cloned().unwrap_or_default();
    let asset = assets.iter().find(|a| {
        let n = a["name"].as_str().unwrap_or("").to_lowercase();
        n.contains("linux") && n.ends_with(".tar.gz")
    });
    let asset = asset.ok_or("no Linux build in latest release")?;
    let url = asset["browser_download_url"].as_str().unwrap_or("").to_string();
    let filename = asset["name"].as_str().unwrap_or("nexium-linux-x86_64.tar.gz").to_string();
    let size = asset["size"].as_u64().unwrap_or(0);
    if url.is_empty() {
        return Err("no download url".into());
    }

    Ok(Release { tag, date, commit, url, size, filename })
}

#[cfg(target_os = "linux")]
fn extract_commit(s: &str) -> Option<String> {
    let mut best = String::new();
    let mut run = String::new();
    for c in s.chars() {
        if c.is_ascii_hexdigit() {
            run.push(c.to_ascii_lowercase());
        } else {
            if run.len() >= 7 && run.len() > best.len() {
                best = run.clone();
            }
            run.clear();
        }
    }
    if run.len() >= 7 && run.len() > best.len() {
        best = run;
    }
    if best.is_empty() {
        None
    } else {
        Some(best)
    }
}

#[cfg(target_os = "linux")]
fn find_binary(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    for d in dirs {
        if let Some(found) = find_binary(&d, name) {
            return Some(found);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn backup_current(exe: &std::path::Path) -> Result<(), String> {
    let parent = exe.parent().ok_or("no parent dir")?;
    let backups = parent.join("backups");
    std::fs::create_dir_all(&backups).map_err(|e| e.to_string())?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let base = exe.file_name().and_then(|n| n.to_str()).unwrap_or("nexium");
    let dest = backups.join(format!("{}-{}", base, stamp));
    std::fs::copy(exe, &dest).map_err(|e| e.to_string())?;
    prune_backups(&backups, 20);
    Ok(())
}

#[cfg(target_os = "linux")]
fn prune_backups(dir: &std::path::Path, keep: usize) {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.is_file() {
                let t = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                Some((t, p))
            } else {
                None
            }
        })
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, p) in files.into_iter().skip(keep) {
        let _ = std::fs::remove_file(p);
    }
}
