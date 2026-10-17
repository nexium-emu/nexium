use super::zip;
use nexium_common::nextendo;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const MAX_PACKAGE: usize = 32 * 1024 * 1024;
const MAX_ENTRIES: usize = 512;
const SPLATOON_3_REGIONS: [&str; 4] = ["ap-default", "eu-default", "jp-default", "us-default"];
const SPLATOON_3_FILES: [&str; 6] = [
    "summary.yml",
    "base.pack.zs.enc",
    "bh.pack.zs.enc",
    "wa.pack.zs.enc",
    "wb.pack.zs.enc",
    "wc.pack.zs.enc",
];

pub fn package_dir(title_id: u64) -> PathBuf {
    nexium_common::paths::root()
        .join("nextendo")
        .join("bcat")
        .join(nextendo::format_title_id(title_id))
}

fn safe_component(name: &str) -> bool {
    !name.is_empty()
        && name.len() < 32
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

pub fn validate(entries: &[zip::Entry], title_id: u64) -> Result<(), String> {
    if entries.is_empty() {
        return Err("empty package".into());
    }
    let mut seen = HashSet::new();
    let mut total = 0usize;
    for entry in entries {
        let parts: Vec<&str> = entry.name.split('/').collect();
        if parts.len() != 2 || !parts.iter().all(|part| safe_component(part)) {
            return Err(format!("unsafe package path {}", entry.name));
        }
        if !seen.insert(entry.name.to_ascii_lowercase()) {
            return Err(format!("duplicate package file {}", entry.name));
        }
        total = total
            .checked_add(entry.data.len())
            .filter(|total| *total <= MAX_PACKAGE)
            .ok_or("package too large")?;
    }
    if title_id == nextendo::SPLATOON_3 {
        for region in SPLATOON_3_REGIONS {
            for file in SPLATOON_3_FILES {
                let name = format!("{region}/{file}");
                if !entries.iter().any(|entry| entry.name == name && !entry.data.is_empty()) {
                    return Err(format!("incomplete Splatoon 3 package: {name} missing"));
                }
            }
        }
    }
    Ok(())
}

pub fn install(entries: &[zip::Entry], target: &Path) -> Result<(), String> {
    let stage = target.with_extension("stage");
    let backup = target.with_extension("backup");
    let _ = std::fs::remove_dir_all(&stage);
    if !target.exists() && backup.exists() {
        std::fs::rename(&backup, target).map_err(|error| format!("recovering {}: {error}", target.display()))?;
    }
    let result = (|| -> Result<(), String> {
        for entry in entries {
            let path = stage.join(entry.name.replace('/', std::path::MAIN_SEPARATOR_STR));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
            }
            std::fs::write(&path, &entry.data).map_err(|error| format!("{}: {error}", path.display()))?;
        }
        let _ = std::fs::remove_dir_all(&backup);
        let had_old = target.exists();
        if had_old {
            std::fs::rename(target, &backup).map_err(|error| format!("{}: {error}", target.display()))?;
        }
        if let Err(error) = std::fs::rename(&stage, target) {
            if had_old {
                let _ = std::fs::rename(&backup, target);
            }
            return Err(format!("{}: {error}", target.display()));
        }
        let _ = std::fs::remove_dir_all(&backup);
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&stage);
    result
}

fn is_installed(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| entries.flatten().any(|entry| entry.path().is_dir()))
        .unwrap_or(false)
}

fn hash_path(dir: &Path) -> PathBuf {
    dir.with_extension("sha256")
}

fn download(agent: &ureq::Agent, package: u64) -> Result<Option<Vec<u8>>, String> {
    let url = format!("{}/api/bcat/{}", nextendo::API_BASE, nextendo::format_title_id(package));
    let mut response = agent
        .get(&url)
        .header("X-Nextendo-Client-Id", super::api::CLIENT_ID)
        .call()
        .map_err(|error| format!("request failed: {error}"))?;
    match response.status().as_u16() {
        200 => {}
        204 | 404 => return Ok(None),
        status => return Err(format!("HTTP {status}")),
    }
    let body = response
        .body_mut()
        .with_config()
        .limit(MAX_PACKAGE as u64)
        .read_to_vec()
        .map_err(|error| format!("body unreadable: {error}"))?;
    Ok((!body.is_empty()).then_some(body))
}

pub fn sync(agent: &ureq::Agent, title_id: u64) -> Result<bool, String> {
    let Some(package) = nextendo::bcat_package_title(title_id) else {
        return Ok(false);
    };
    let mut bytes = download(agent, package)?;
    if bytes.is_none() && title_id == nextendo::POKEMON_SCARLET {
        bytes = download(agent, nextendo::POKEMON_VIOLET)?;
    }
    let Some(bytes) = bytes else {
        return Ok(false);
    };
    let dir = package_dir(title_id);
    let hash: String = Sha256::digest(&bytes).iter().map(|byte| format!("{byte:02x}")).collect();
    let current = std::fs::read_to_string(hash_path(&dir)).unwrap_or_default();
    if current.trim() == hash && is_installed(&dir) {
        return Ok(false);
    }
    let entries = zip::extract(&bytes, MAX_ENTRIES, MAX_PACKAGE)?;
    validate(&entries, title_id)?;
    install(&entries, &dir)?;
    if let Err(error) = std::fs::write(hash_path(&dir), &hash) {
        log::warn!("nextendo: BCAT hash for {} not saved: {error}", nextendo::format_title_id(title_id));
    }
    log::info!(
        "nextendo: BCAT package for {} installed ({} files, {} bytes)",
        nextendo::format_title_id(title_id),
        entries.len(),
        bytes.len()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, data: &[u8]) -> zip::Entry {
        zip::Entry {
            name: name.into(),
            data: data.to_vec(),
        }
    }

    #[test]
    fn packages_are_one_directory_deep_with_safe_names() {
        assert!(validate(&[entry("vsdata/VSSetting_0.byaml", b"x")], nextendo::SPLATOON_2_EU).is_ok());
        assert!(validate(&[entry("Root00/BRT.bgyml", b"x")], nextendo::NINTENDO_SWITCH_SPORTS).is_ok());
        assert!(validate(&[], nextendo::SPLATOON_2_EU).is_err());
        assert!(validate(&[entry("loose.byaml", b"x")], nextendo::SPLATOON_2_EU).is_err());
        assert!(validate(&[entry("a/b/c", b"x")], nextendo::SPLATOON_2_EU).is_err());
        assert!(validate(&[entry("../etc/passwd", b"x")], nextendo::SPLATOON_2_EU).is_err());
        assert!(validate(&[entry("vsdata/a", b"x"), entry("VSDATA/A", b"y")], nextendo::SPLATOON_2_EU).is_err());
        assert!(validate(&[entry("eu-default/summary.yml", b"x")], nextendo::SPLATOON_3).is_err());
        let complete: Vec<zip::Entry> = SPLATOON_3_REGIONS
            .iter()
            .flat_map(|region| SPLATOON_3_FILES.iter().map(move |file| entry(&format!("{region}/{file}"), b"x")))
            .collect();
        assert!(validate(&complete, nextendo::SPLATOON_3).is_ok());
    }

    #[test]
    fn install_replaces_the_previous_package_atomically() {
        let root = std::env::temp_dir().join(format!("nexium-bcat-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let target = root.join("0100F8F0000A2000");
        install(&[entry("vsdata/VSSetting_0.byaml", b"old"), entry("fesdata/Festival.byaml", b"fes")], &target).unwrap();
        assert_eq!(std::fs::read(target.join("vsdata").join("VSSetting_0.byaml")).unwrap(), b"old");
        install(&[entry("vsdata/VSSetting_0.byaml", b"new")], &target).unwrap();
        assert_eq!(std::fs::read(target.join("vsdata").join("VSSetting_0.byaml")).unwrap(), b"new");
        assert!(!target.join("fesdata").exists());
        assert!(!target.with_extension("stage").exists());
        assert!(!target.with_extension("backup").exists());
        assert!(is_installed(&target));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "needs network access and NEXIUM_DATA_ROOT"]
    fn live_switch_sports_package_installs() {
        let agent = super::super::launch::agent();
        let dir = package_dir(nextendo::NINTENDO_SWITCH_SPORTS);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(hash_path(&dir));
        assert_eq!(sync(&agent, nextendo::NINTENDO_SWITCH_SPORTS), Ok(true));
        assert!(dir.join("Root00").join("BRT.bgyml").is_file());
        assert_eq!(sync(&agent, nextendo::NINTENDO_SWITCH_SPORTS), Ok(false));
        assert_eq!(sync(&agent, nextendo::MARIO_KART_8_DELUXE), Ok(false));
        assert_eq!(sync(&agent, nextendo::POKEMON_SCARLET), Ok(true));
        assert!(package_dir(nextendo::POKEMON_SCARLET).join("raid").is_dir());
    }

    #[test]
    fn package_dirs_match_the_kernel_layout() {
        let dir = package_dir(nextendo::SPLATOON_2_US);
        assert!(dir.ends_with(Path::new("nextendo").join("bcat").join("01003BC0000A0000")));
    }
}
