use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[cfg(any(target_os = "linux", test))]
const UPDATE_PAYLOAD_FILES: [(&str, bool); 10] = [
    ("ffmpeg", true),
    ("THIRD_PARTY_NOTICES.txt", false),
    ("licenses/FFmpeg/LICENSE.md", false),
    ("licenses/FFmpeg/COPYING.LGPLv2.1", false),
    ("licenses/FFmpeg/BUILD.txt", false),
    ("licenses/FFmpeg/ffmpeg-9.0.1.tar.xz", false),
    ("licenses/FFmpeg/ffmpeg-9.0.1.tar.xz.asc", false),
    ("licenses/FFmpeg/ffmpeg-devel.asc", false),
    ("licenses/FFmpeg/CHANGES.diff", false),
    ("nexium", true),
];

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
        self.status
            .lock()
            .map(|g| g.clone())
            .unwrap_or(Status::Idle)
    }

    pub fn available_release(&self) -> Option<Release> {
        match self.status() {
            Status::Available(r) => Some(r),
            _ => None,
        }
    }

    pub fn is_downloading(&self) -> bool {
        self.download
            .as_ref()
            .map_or(false, |d| d.lock().map(|g| !g.done).unwrap_or(false))
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
                    if is_current(&rel.commit) && current_update_payload_complete() {
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
        let parent = exe.parent().ok_or("no parent directory")?.to_path_buf();
        install_update_payload(&tmp, &exe)?;

        std::process::Command::new(&exe)
            .current_dir(std::env::current_dir().unwrap_or(parent))
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
    let list: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|_| "bad response".to_string())?;
    let json = list
        .as_array()
        .and_then(|arr| {
            arr.iter()
                .find(|r| !r["draft"].as_bool().unwrap_or(false))
                .cloned()
        })
        .ok_or("no releases found")?;

    let tag = json["tag_name"].as_str().unwrap_or("").to_string();
    let name = json["name"].as_str().unwrap_or("").to_string();
    let body = json["body"].as_str().unwrap_or("");
    let date = json["published_at"]
        .as_str()
        .unwrap_or("")
        .chars()
        .take(10)
        .collect::<String>();

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
    let url = asset["browser_download_url"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let filename = asset["name"]
        .as_str()
        .unwrap_or("nexium-linux-x86_64.tar.gz")
        .to_string();
    let size = asset["size"].as_u64().unwrap_or(0);
    if url.is_empty() {
        return Err("no download url".into());
    }

    Ok(Release {
        tag,
        date,
        commit,
        url,
        size,
        filename,
    })
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

#[cfg(any(target_os = "linux", test))]
fn find_binary(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_dir() {
            dirs.push(path);
        } else if metadata.file_type().is_file()
            && path.file_name().and_then(|n| n.to_str()) == Some(name)
        {
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

#[cfg(any(target_os = "linux", test))]
fn validate_payload_file(
    package_root: &std::path::Path,
    relative: &str,
) -> Result<PathBuf, String> {
    use std::path::Component;

    let relative_path = std::path::Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("invalid update payload path {}", relative));
    }

    let mut current = package_root.to_path_buf();
    if let Some(parent) = relative_path.parent() {
        for component in parent.components() {
            let Component::Normal(segment) = component else {
                return Err(format!("invalid update payload path {}", relative));
            };
            current.push(segment);
            let metadata = std::fs::symlink_metadata(&current)
                .map_err(|e| format!("missing update payload {}: {}", relative, e))?;
            if !metadata.file_type().is_dir() {
                return Err(format!("invalid update payload directory {}", relative));
            }
        }
    }

    let path = package_root.join(relative_path);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|e| format!("missing update payload {}: {}", relative, e))?;
    let may_be_empty = relative == "licenses/FFmpeg/CHANGES.diff";
    if !metadata.file_type().is_file() || (!may_be_empty && metadata.len() == 0) {
        return Err(format!("invalid update payload file {}", relative));
    }
    Ok(path)
}

#[cfg(any(target_os = "linux", test))]
fn resolve_update_payload(extracted_root: &std::path::Path) -> Result<PathBuf, String> {
    let new_binary =
        find_binary(extracted_root, "nexium").ok_or("new binary not found in archive")?;
    let package_root = new_binary
        .parent()
        .ok_or("update payload has no package directory")?;
    for (relative, _) in UPDATE_PAYLOAD_FILES {
        validate_payload_file(package_root, relative)?;
    }
    Ok(package_root.to_path_buf())
}

#[cfg(any(target_os = "linux", test))]
fn installed_update_payload_complete(executable: &std::path::Path) -> bool {
    let Some(package_root) = executable.parent() else {
        return false;
    };
    for (relative, executable_file) in UPDATE_PAYLOAD_FILES {
        let path = if relative == "nexium" {
            executable.to_path_buf()
        } else {
            let Ok(path) = validate_payload_file(package_root, relative) else {
                return false;
            };
            path
        };
        let Ok(metadata) = std::fs::symlink_metadata(path) else {
            return false;
        };
        if !metadata.file_type().is_file()
            || (metadata.len() == 0 && relative != "licenses/FFmpeg/CHANGES.diff")
        {
            return false;
        }
        if executable_file {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                if metadata.permissions().mode() & 0o111 == 0 {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(target_os = "linux")]
fn current_update_payload_complete() -> bool {
    std::env::current_exe().is_ok_and(|executable| installed_update_payload_complete(&executable))
}

#[cfg(any(target_os = "linux", test))]
fn update_destination(
    executable: &std::path::Path,
    install_root: &std::path::Path,
    relative: &str,
) -> PathBuf {
    if relative == "nexium" {
        executable.to_path_buf()
    } else {
        install_root.join(relative)
    }
}

#[cfg(any(target_os = "linux", test))]
fn ensure_destination_parent(install_root: &std::path::Path, relative: &str) -> Result<(), String> {
    use std::path::Component;

    let mut current = install_root.to_path_buf();
    let Some(parent) = std::path::Path::new(relative).parent() else {
        return Ok(());
    };
    for component in parent.components() {
        let Component::Normal(segment) = component else {
            return Err(format!("invalid update destination {}", relative));
        };
        current.push(segment);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return Err(format!(
                    "update destination is not a directory: {}",
                    current.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current).map_err(|e| {
                    format!(
                        "could not create update directory {}: {}",
                        current.display(),
                        e
                    )
                })?;
            }
            Err(error) => {
                return Err(format!(
                    "could not inspect update directory {}: {}",
                    current.display(),
                    error
                ));
            }
        }
    }
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn replace_file(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::rename(source, destination)
    }
    #[cfg(not(unix))]
    {
        match std::fs::remove_file(destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        std::fs::rename(source, destination)
    }
}

#[cfg(any(target_os = "linux", test))]
fn rollback_update(installed: &[PathBuf], backed_up: &[(PathBuf, PathBuf)]) -> Vec<String> {
    let mut errors = Vec::new();
    for destination in installed.iter().rev() {
        if let Some((_, backup)) = backed_up
            .iter()
            .find(|(backed_up_destination, _)| backed_up_destination == destination)
        {
            if let Err(error) = replace_file(backup, destination) {
                errors.push(format!(
                    "could not restore {}: {}",
                    destination.display(),
                    error
                ));
            }
        } else {
            match std::fs::remove_file(destination) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => errors.push(format!(
                    "could not remove {}: {}",
                    destination.display(),
                    error
                )),
            }
        }
    }
    errors
}

#[cfg(any(target_os = "linux", test))]
fn failed_update(
    error: String,
    stage_root: &std::path::Path,
    backup_root: &std::path::Path,
    installed: &[PathBuf],
    backed_up: &[(PathBuf, PathBuf)],
) -> String {
    let rollback_errors = rollback_update(installed, backed_up);
    let _ = std::fs::remove_dir_all(stage_root);
    if rollback_errors.is_empty() {
        let _ = std::fs::remove_dir_all(backup_root);
        error
    } else {
        format!("{}; rollback failed: {}", error, rollback_errors.join("; "))
    }
}

#[cfg(any(target_os = "linux", test))]
fn install_update_payload(
    extracted_root: &std::path::Path,
    executable: &std::path::Path,
) -> Result<(), String> {
    let package_root = resolve_update_payload(extracted_root)?;
    let install_root = executable.parent().ok_or("no parent directory")?;
    for (relative, _) in UPDATE_PAYLOAD_FILES {
        ensure_destination_parent(install_root, relative)?;
        let destination = update_destination(executable, install_root, relative);
        match std::fs::symlink_metadata(&destination) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                return Err(format!(
                    "update destination is not a regular file: {}",
                    destination.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if relative == "nexium" {
                    return Err("current executable is missing".into());
                }
            }
            Err(error) => {
                return Err(format!(
                    "could not inspect update destination {}: {}",
                    destination.display(),
                    error
                ));
            }
        }
    }

    let backups_root = install_root.join("backups");
    match std::fs::symlink_metadata(&backups_root) {
        Ok(metadata) if metadata.file_type().is_dir() => {}
        Ok(_) => return Err("backup path is not a directory".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(&backups_root)
                .map_err(|e| format!("could not create backup directory: {}", e))?;
        }
        Err(error) => return Err(format!("could not inspect backup directory: {}", error)),
    }

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let unique = format!("{}-{}-{}", stamp, std::process::id(), nonce);
    let stage_root = install_root.join(format!(".nexium-update-stage-{}", unique));
    let backup_root = backups_root.join(format!("update-{}", unique));
    std::fs::create_dir(&stage_root)
        .map_err(|e| format!("could not create update staging directory: {}", e))?;

    for (relative, executable_file) in UPDATE_PAYLOAD_FILES {
        let source = package_root.join(relative);
        let staged = stage_root.join(relative);
        if let Some(parent) = staged.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                let _ = std::fs::remove_dir_all(&stage_root);
                return Err(format!("could not stage {}: {}", relative, error));
            }
        }
        if let Err(error) = std::fs::copy(&source, &staged) {
            let _ = std::fs::remove_dir_all(&stage_root);
            return Err(format!("could not stage {}: {}", relative, error));
        }
        if executable_file {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                if let Err(error) =
                    std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
                {
                    let _ = std::fs::remove_dir_all(&stage_root);
                    return Err(format!(
                        "could not set permissions on {}: {}",
                        relative, error
                    ));
                }
            }
        }
    }

    if let Err(error) = std::fs::create_dir(&backup_root) {
        let _ = std::fs::remove_dir_all(&stage_root);
        return Err(format!("could not create update backup: {}", error));
    }

    let mut backed_up = Vec::new();
    for (relative, _) in UPDATE_PAYLOAD_FILES {
        let destination = update_destination(executable, install_root, relative);
        match std::fs::symlink_metadata(&destination) {
            Ok(metadata) if metadata.file_type().is_file() => {
                let backup = backup_root.join(relative);
                if let Some(parent) = backup.parent() {
                    if let Err(error) = std::fs::create_dir_all(parent) {
                        let message =
                            format!("could not prepare backup for {}: {}", relative, error);
                        let _ = std::fs::remove_dir_all(&stage_root);
                        let _ = std::fs::remove_dir_all(&backup_root);
                        return Err(message);
                    }
                }
                if let Err(error) = std::fs::copy(&destination, &backup) {
                    let message = format!("could not back up {}: {}", relative, error);
                    let _ = std::fs::remove_dir_all(&stage_root);
                    let _ = std::fs::remove_dir_all(&backup_root);
                    return Err(message);
                }
                backed_up.push((destination, backup));
            }
            Ok(_) => {
                let message = format!("{} is no longer a regular file", relative);
                let _ = std::fs::remove_dir_all(&stage_root);
                let _ = std::fs::remove_dir_all(&backup_root);
                return Err(message);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                let message = format!("could not inspect {} for backup: {}", relative, error);
                let _ = std::fs::remove_dir_all(&stage_root);
                let _ = std::fs::remove_dir_all(&backup_root);
                return Err(message);
            }
        }
    }

    let mut installed = Vec::new();
    for (relative, _) in UPDATE_PAYLOAD_FILES {
        let staged = stage_root.join(relative);
        let destination = update_destination(executable, install_root, relative);
        if let Err(error) = replace_file(&staged, &destination) {
            let message = format!("could not install {}: {}", relative, error);
            return Err(failed_update(
                message,
                &stage_root,
                &backup_root,
                &installed,
                &backed_up,
            ));
        }
        installed.push(destination);
    }

    let _ = std::fs::remove_dir_all(&stage_root);
    prune_backups(&backups_root, 20);
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn prune_backups(dir: &std::path::Path, keep: usize) {
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_str()?;
            let metadata = std::fs::symlink_metadata(&p).ok()?;
            let is_dir = metadata.file_type().is_dir();
            if !is_dir || !name.starts_with("update-") {
                return None;
            }
            let modified = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
            Some((modified, p))
        })
        .collect();
    if entries.len() <= keep {
        return;
    }
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in entries.into_iter().skip(keep) {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempTree(PathBuf);

    impl TempTree {
        fn new(name: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "nexium-updater-{}-{}-{}",
                name,
                std::process::id(),
                nonce
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_payload(extracted_root: &std::path::Path) -> PathBuf {
        let package_root = extracted_root.join("NeXium");
        for (relative, executable_file) in UPDATE_PAYLOAD_FILES {
            let path = package_root.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            let contents = if relative == "licenses/FFmpeg/CHANGES.diff" {
                String::new()
            } else {
                format!("new:{}", relative)
            };
            std::fs::write(&path, contents).unwrap();
            if executable_file {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;

                    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
                }
            }
        }
        package_root
    }

    #[test]
    fn complete_update_payload_is_resolved() {
        let tree = TempTree::new("complete");
        let package_root = write_payload(&tree.0);
        assert_eq!(resolve_update_payload(&tree.0).unwrap(), package_root);
    }

    #[test]
    fn missing_update_payload_file_is_rejected() {
        let tree = TempTree::new("missing");
        let package_root = write_payload(&tree.0);
        std::fs::remove_file(package_root.join("ffmpeg")).unwrap();
        let error = resolve_update_payload(&tree.0).unwrap_err();
        assert!(error.contains("ffmpeg"));
    }

    #[test]
    fn empty_required_payload_file_is_rejected() {
        let tree = TempTree::new("empty");
        let package_root = write_payload(&tree.0);
        std::fs::write(package_root.join("licenses/FFmpeg/BUILD.txt"), []).unwrap();
        let error = resolve_update_payload(&tree.0).unwrap_err();
        assert!(error.contains("BUILD.txt"));
    }

    #[test]
    fn installed_payload_requires_every_sidecar() {
        let tree = TempTree::new("installed-complete");
        let package_root = write_payload(&tree.0);
        let executable = package_root.join("nexium");
        assert!(installed_update_payload_complete(&executable));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let ffmpeg = package_root.join("ffmpeg");
            std::fs::set_permissions(&ffmpeg, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert!(!installed_update_payload_complete(&executable));
            std::fs::set_permissions(ffmpeg, std::fs::Permissions::from_mode(0o755)).unwrap();
            assert!(installed_update_payload_complete(&executable));
        }

        std::fs::remove_file(package_root.join("licenses/FFmpeg/ffmpeg-devel.asc")).unwrap();
        assert!(!installed_update_payload_complete(&executable));
    }

    #[test]
    fn update_payload_replaces_all_files() {
        let tree = TempTree::new("install");
        let extracted_root = tree.0.join("archive");
        write_payload(&extracted_root);
        let install_root = tree.0.join("installed");
        std::fs::create_dir(&install_root).unwrap();
        let executable = install_root.join("nexium");
        std::fs::write(&executable, "old:nexium").unwrap();
        std::fs::write(install_root.join("ffmpeg"), "old:ffmpeg").unwrap();

        install_update_payload(&extracted_root, &executable).unwrap();

        for (relative, executable_file) in UPDATE_PAYLOAD_FILES {
            let destination = update_destination(&executable, &install_root, relative);
            let expected = if relative == "licenses/FFmpeg/CHANGES.diff" {
                String::new()
            } else {
                format!("new:{}", relative)
            };
            assert_eq!(std::fs::read_to_string(&destination).unwrap(), expected);
            if executable_file {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;

                    let mode = std::fs::metadata(destination).unwrap().permissions().mode();
                    assert_eq!(mode & 0o111, 0o111);
                }
            }
        }

        let backup = std::fs::read_dir(install_root.join("backups"))
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(backup.join("nexium")).unwrap(),
            "old:nexium"
        );
        assert_eq!(
            std::fs::read_to_string(backup.join("ffmpeg")).unwrap(),
            "old:ffmpeg"
        );
    }

    #[test]
    fn incomplete_payload_does_not_mutate_install() {
        let tree = TempTree::new("incomplete-install");
        let extracted_root = tree.0.join("archive");
        let package_root = write_payload(&extracted_root);
        std::fs::remove_file(package_root.join("licenses/FFmpeg/BUILD.txt")).unwrap();
        let install_root = tree.0.join("installed");
        std::fs::create_dir(&install_root).unwrap();
        let executable = install_root.join("nexium");
        std::fs::write(&executable, "old:nexium").unwrap();

        assert!(install_update_payload(&extracted_root, &executable).is_err());
        assert_eq!(std::fs::read_to_string(&executable).unwrap(), "old:nexium");
        assert!(!install_root.join("backups").exists());
    }

    #[test]
    fn rollback_restores_backups_and_removes_new_files() {
        let tree = TempTree::new("rollback");
        let install_root = tree.0.join("installed");
        let backup_root = tree.0.join("backup");
        std::fs::create_dir(&install_root).unwrap();
        std::fs::create_dir(&backup_root).unwrap();
        let executable = install_root.join("nexium");
        let ffmpeg = install_root.join("ffmpeg");
        let executable_backup = backup_root.join("nexium");
        std::fs::write(&executable, "new:nexium").unwrap();
        std::fs::write(&ffmpeg, "new:ffmpeg").unwrap();
        std::fs::write(&executable_backup, "old:nexium").unwrap();

        let errors = rollback_update(
            &[executable.clone(), ffmpeg.clone()],
            &[(executable.clone(), executable_backup)],
        );

        assert!(errors.is_empty());
        assert_eq!(std::fs::read_to_string(&executable).unwrap(), "old:nexium");
        assert!(!ffmpeg.exists());
    }

    #[test]
    fn backup_pruning_ignores_unowned_entries() {
        let tree = TempTree::new("prune");
        let backups = tree.0.join("backups");
        std::fs::create_dir(&backups).unwrap();
        for index in 0..4 {
            std::fs::create_dir(backups.join(format!("update-{}", index))).unwrap();
        }
        std::fs::write(backups.join("nexium-legacy"), "backup").unwrap();
        std::fs::create_dir(backups.join("personal-backup")).unwrap();
        std::fs::write(backups.join("notes.txt"), "keep").unwrap();

        prune_backups(&backups, 2);

        let updater_owned = std::fs::read_dir(&backups)
            .unwrap()
            .flatten()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("update-")
            })
            .count();
        assert_eq!(updater_owned, 2);
        assert!(backups.join("nexium-legacy").is_file());
        assert!(backups.join("personal-backup").is_dir());
        assert!(backups.join("notes.txt").is_file());
    }
}
