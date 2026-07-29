use std::path::{Path, PathBuf};

fn data_root() -> PathBuf {
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
