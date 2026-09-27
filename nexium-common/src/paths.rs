use std::path::{Path, PathBuf};

const DATA_ROOT_OVERRIDE_ENV: &str = "NEXIUM_DATA_ROOT";

fn data_root_override(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let value = value?;
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn data_root() -> PathBuf {
    if let Some(path) = data_root_override(std::env::var_os(DATA_ROOT_OVERRIDE_ENV)) {
        return path;
    }
    if let Some(dirs) = directories::BaseDirs::new() {
        return dirs.config_dir().join("NeXium");
    }
    PathBuf::from("./NeXium")
}

fn ensure(path: PathBuf) -> PathBuf {
    let _ = std::fs::create_dir_all(&path);
    path
}

pub fn root() -> PathBuf {
    ensure(data_root())
}

pub fn init() {
    let _ = (root(), sdmc_dir(), nand_dir(), nro_dir(), log_dir());
}

pub fn content_dir() -> PathBuf {
    ensure(data_root().join("content"))
}

pub fn mods_dir() -> PathBuf {
    ensure(data_root().join("mods"))
}

pub fn title_mods_dir(title_id: u64) -> PathBuf {
    ensure(mods_dir().join(format!("{title_id:016X}")))
}

pub fn mod_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(path) = std::env::var_os("NEXIUM_MODS_DIR").filter(|path| !path.is_empty()) {
        roots.push(PathBuf::from(path));
    }
    for path in [mods_dir(), sdmc_dir().join("atmosphere")] {
        if !roots.contains(&path) { roots.push(path); }
    }
    roots
}

pub fn sdmc_dir() -> PathBuf {
    ensure(data_root().join("sdmc"))
}

pub fn nand_dir() -> PathBuf {
    ensure(data_root().join("nand"))
}

pub fn nro_dir() -> PathBuf {
    ensure(data_root().join("NRO"))
}

pub fn log_dir() -> PathBuf {
    ensure(data_root().join("logs"))
}

pub fn app_name_from_nro(nro_path: &str) -> String {
    let stem = Path::new(nro_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim();
    let cleaned: String = stem
        .chars()
        .map(|c| {
            if c == '/' || c == '\\' || c == ':' || c == ' ' {
                '_'
            } else {
                c
            }
        })
        .collect();
    if cleaned.is_empty() {
        "app".to_string()
    } else {
        cleaned
    }
}

pub fn sdmc_app_dir(app: &str) -> PathBuf {
    ensure(sdmc_dir().join("switch").join(app))
}

pub fn guest_app_dir(app: &str) -> String {
    format!("sdmc:/switch/{}", app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonempty_data_root_override_is_used_verbatim() {
        let path = PathBuf::from("C:/isolated/nexium");
        assert_eq!(
            data_root_override(Some(path.clone().into_os_string())),
            Some(path)
        );
        assert_eq!(data_root_override(Some("".into())), None);
    }
}
