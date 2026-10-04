use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::application::{Application, LoadedModule, MODULE_ORDER};
use crate::npdm::Npdm;
use crate::nso::Nso;
use crate::{AppRomfs, LayeredRomfs};

#[derive(Clone, Debug)]
pub struct ModEntry {
    pub name: String,
    pub path: PathBuf,
    pub enabled: bool,
    pub has_romfs: bool,
    pub has_exefs: bool,
}

fn directory_entries(path: &Path) -> Result<Vec<std::fs::DirEntry>, String> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Read mod directory {}: {error}", path.display())),
    };
    let mut entries = entries.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Read mod directory {}: {error}", path.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn title_directory(root: &Path, title_id: u64) -> Result<Option<PathBuf>, String> {
    let title = format!("{title_id:016X}");
    Ok(directory_entries(root)?.into_iter().find_map(|entry| {
        (entry.file_name().to_string_lossy().eq_ignore_ascii_case(&title)
            && entry.path().is_dir()).then(|| entry.path())
    }))
}

fn mod_entry(path: PathBuf, name: String) -> Option<ModEntry> {
    let has_romfs = path.join("romfs").is_dir();
    let has_exefs = path.join("exefs").is_dir();
    (has_romfs || has_exefs).then(|| ModEntry {
        enabled: !path.join(".disabled").exists(),
        name,
        path,
        has_romfs,
        has_exefs,
    })
}

pub fn discover_mods(roots: &[PathBuf], title_id: u64) -> Result<Vec<ModEntry>, String> {
    let mut legacy = Vec::new();
    let mut named = Vec::new();
    for root in roots.iter().rev() {
        for parent in [root.join("contents"), root.clone()] {
            let Some(title) = title_directory(&parent, title_id)? else { continue; };
            if let Some(entry) = mod_entry(title.clone(), format!("Legacy ({})", root.display())) {
                legacy.push(entry);
            }
            for entry in directory_entries(&title)? {
                if !entry.path().is_dir() { continue; }
                if let Some(entry) = mod_entry(entry.path(), entry.file_name().to_string_lossy().into_owned()) {
                    named.push(entry);
                }
            }
        }
    }
    named.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase())
        .then_with(|| left.name.cmp(&right.name)));
    legacy.extend(named);
    let mut seen = std::collections::HashSet::new();
    legacy.retain(|entry| seen.insert(entry.path.clone()));
    Ok(legacy)
}

pub fn set_mod_enabled(path: &Path, enabled: bool) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("Mod directory does not exist: {}", path.display()));
    }
    let marker = path.join(".disabled");
    let result = if enabled {
        match std::fs::remove_file(&marker) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            result => result,
        }
    } else {
        std::fs::write(&marker, [])
    };
    result.map_err(|error| format!("Change mod state {}: {error}", path.display()))
}

pub fn build_mod_romfs(app: &Application, mods: &[ModEntry]) -> Result<Option<AppRomfs>, String> {
    let overlays = mods.iter().filter(|entry| entry.enabled && entry.has_romfs)
        .map(|entry| entry.path.join("romfs")).collect::<Vec<_>>();
    if overlays.is_empty() {
        return Ok(app.romfs.clone().map(AppRomfs::Plain));
    }
    let layered = LayeredRomfs::build_many(app.romfs.as_ref(), &overlays)?;
    log::info!("[mods] RomFS overlays={} base_files={} overlay_files={} replacements={} virtual_bytes={}",
        overlays.len(), layered.base_file_count, layered.overlay_file_count,
        layered.replaced_file_count, layered.len());
    Ok(Some(AppRomfs::Layered(std::sync::Arc::new(layered))))
}

fn parse_build_id(name: &str) -> Option<[u8; 32]> {
    if name.is_empty() || name.len() > 64 || name.len() % 2 != 0 || !name.is_ascii() {
        return None;
    }
    let mut id = [0u8; 32];
    for (index, pair) in name.as_bytes().chunks_exact(2).enumerate() {
        let pair = std::str::from_utf8(pair).ok()?;
        id[index] = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(id)
}

fn build_id_string(id: &[u8; 32]) -> String {
    id.iter().map(|byte| format!("{byte:02X}")).collect()
}

fn exefs_files(path: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut files = BTreeMap::new();
    for entry in directory_entries(path)? {
        if !entry.path().is_file() { continue; }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if files.insert(name.clone(), entry.path()).is_some() {
            return Err(format!("Duplicate ExeFS filename {name} in {}", path.display()));
        }
    }
    Ok(files)
}

fn read_mod_file(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| format!("Read mod file {}: {error}", path.display()))
}

pub fn apply_exefs_mods(app: &mut Application, mods: &[ModEntry]) -> Result<(), String> {
    let active = mods.iter().filter(|entry| entry.enabled && entry.has_exefs)
        .map(|entry| exefs_files(&entry.path.join("exefs")).map(|files| (entry, files)))
        .collect::<Result<Vec<_>, _>>()?;
    for (entry, files) in &active {
        for &name in MODULE_ORDER {
            let Some(path) = files.get(name) else { continue; };
            let nso = Nso::parse(&read_mod_file(path)?)
                .map_err(|error| format!("Invalid ExeFS replacement {}: {error}", path.display()))?;
            if let Some(module) = app.modules.iter_mut().find(|module| module.name == name) {
                module.nso = nso;
            } else {
                app.modules.push(LoadedModule { name: name.into(), nso, load_offset: 0 });
            }
            log::info!("[mods] replaced module {name} from {}", path.display());
        }
        if let Some(path) = files.get("main.npdm") {
            let npdm = Npdm::parse(&read_mod_file(path)?)
                .map_err(|error| format!("Invalid NPDM replacement {}: {error}", path.display()))?;
            if npdm.title_id != 0 && npdm.title_id != app.title_id {
                return Err(format!("Mod {} changes title ID from {:016X} to {:016X}",
                    path.display(), app.title_id, npdm.title_id));
            }
            app.npdm = npdm;
        }
        log::debug!("[mods] resolved ExeFS replacements for {}", entry.name);
    }
    for (entry, files) in &active {
        let mut patch_count = 0usize;
        let mut matched = 0usize;
        for (name, path) in files {
            let extension = path.extension().and_then(|extension| extension.to_str())
                .unwrap_or("").to_ascii_lowercase();
            if extension == "pchtxt" {
                return Err(format!("IPSwitch text patches are not supported: {}. Use a compiled IPS patch.", path.display()));
            }
            if !matches!(extension.as_str(), "ips" | "ips32") { continue; }
            patch_count += 1;
            let stem = Path::new(name).file_stem().and_then(|stem| stem.to_str()).unwrap_or("");
            let build_id = parse_build_id(stem)
                .ok_or_else(|| format!("Patch filename must be an NSO build ID: {}", path.display()))?;
            let patch = read_mod_file(path)?;
            for module in app.modules.iter_mut().filter(|module| module.nso.build_id == build_id) {
                let count = crate::patch::apply_ips(&mut module.nso.module_image, &patch, 0x100)
                    .map_err(|error| format!("Apply {} to {}: {error}", path.display(), module.name))?;
                matched += 1;
                log::info!("[mods] patched module {} from {} bytes={count}", module.name, path.display());
            }
        }
        if patch_count != 0 && matched == 0 {
            let builds = app.modules.iter().map(|module| format!("{}={}", module.name,
                build_id_string(&module.nso.build_id))).collect::<Vec<_>>().join(", ");
            return Err(format!("No patch in mod '{}' matches this game's build IDs ({builds}). Disable the mod or install its matching game version.", entry.name));
        }
    }
    app.modules.sort_by_key(|module| MODULE_ORDER.iter().position(|name| *name == module.name).unwrap_or(usize::MAX));
    let mut offset = 0u64;
    for module in &mut app.modules {
        module.load_offset = offset;
        let aligned = u64::from(module.nso.image_size).checked_add(0xfff)
            .ok_or("Mod module size overflow")? & !0xfff;
        offset = offset.checked_add(aligned).ok_or("Mod code layout overflow")?;
    }
    app.total_code_size = offset;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!("nexium-mods-{}-{time}-{id}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, path: impl AsRef<Path>, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn mmap(bytes: &[u8]) -> Arc<memmap2::Mmap> {
        let mut map = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        map[..bytes.len()].copy_from_slice(bytes);
        Arc::new(map.make_read_only().unwrap())
    }

    fn nso_bytes(id: [u8; 32], value: u8) -> Vec<u8> {
        let mut bytes = vec![0; 0x130];
        bytes[..4].copy_from_slice(b"NSO0");
        bytes[0x40..0x60].copy_from_slice(&id);
        for index in 0..3usize {
            let header = 0x10 + index * 0x10;
            let file = 0x100 + index * 0x10;
            bytes[header..header + 4].copy_from_slice(&(file as u32).to_le_bytes());
            bytes[header + 4..header + 8].copy_from_slice(&(index as u32 * 0x1000).to_le_bytes());
            bytes[header + 8..header + 12].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[0x60 + index * 4..0x64 + index * 4].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[file..file + 0x10].fill(value);
        }
        bytes
    }

    fn application() -> Application {
        let mut id = [0; 32];
        id[0] = 0x12;
        Application {
            mmap: mmap(&[]),
            modules: vec![LoadedModule {
                name: "main".into(),
                nso: Nso::parse(&nso_bytes(id, 1)).unwrap(),
                load_offset: 0,
            }],
            total_code_size: 0x3000,
            npdm: Npdm::default_for_homebrew(),
            romfs: None,
            system_romfs: Default::default(),
            content_version: 0,
            display_version: String::new(),
            application_control: None,
            patch_romfs: None,
            add_on_content: Default::default(),
            title_id: 0x0100_0000_0000_00ab,
        }
    }

    fn patch(value: u8) -> Vec<u8> {
        patch_at(0, value)
    }

    fn patch_at(offset: u8, value: u8) -> Vec<u8> {
        let mut patch = b"PATCH".to_vec();
        patch.extend_from_slice(&[0, 1, offset, 0, 1, value]);
        patch.extend_from_slice(b"EOF");
        patch
    }

    #[test]
    fn named_mods_are_sorted_and_persistent_enables_preserve_legacy_discovery() {
        let dir = TestDir::new();
        let app = application();
        let title = format!("{:016x}", app.title_id);
        dir.write(format!("{title}/Zebra/romfs/file"), b"last");
        dir.write(format!("{title}/alpha/exefs/main"), b"module");
        dir.write(format!("contents/{title}/romfs/file"), b"legacy");
        let roots = [dir.0.clone()];
        let entries = discover_mods(&roots, app.title_id).unwrap();
        assert_eq!(entries.len(), 3);
        assert!(entries[0].name.starts_with("Legacy"));
        assert_eq!(entries[1].name, "alpha");
        assert_eq!(entries[2].name, "Zebra");
        set_mod_enabled(&entries[2].path, false).unwrap();
        assert!(!discover_mods(&roots, app.title_id).unwrap()[2].enabled);
        set_mod_enabled(&entries[2].path, true).unwrap();
        set_mod_enabled(&entries[2].path, true).unwrap();
        assert!(discover_mods(&roots, app.title_id).unwrap()[2].enabled);
        assert!(discover_mods(&roots, app.title_id + 1).unwrap().is_empty());
    }

    #[test]
    fn atmosphere_named_mods_preserve_disabled_order_and_dedup_without_scanning_asset_wrappers() {
        let dir = TestDir::new();
        let title_id = 0x0100_F2C0_115B_6000;
        let title = format!("{title_id:016x}");
        let atmosphere_title = format!("atmosphere/contents/{title}");
        dir.write(format!("{atmosphere_title}/romfs/legacy"), b"legacy");
        dir.write(format!("{atmosphere_title}/UltraCam/exefs/main"), b"module");
        dir.write(format!("{atmosphere_title}/UltraCam/romfs/config"), b"settings");
        dir.write(format!("{atmosphere_title}/Disabled/exefs/main"), b"disabled");
        dir.write(format!("{atmosphere_title}/Disabled/.disabled"), b"");
        dir.write(format!("{atmosphere_title}/RomfsLiteEX/Actor/asset"), b"asset");
        dir.write(format!("{atmosphere_title}/RomfsLiteEX/wrapper/romfs/asset"), b"nested");
        dir.write(format!("mods/{title}/exefs/main"), b"legacy-module");
        dir.write(format!("mods/{title}/alpha/romfs/asset"), b"first");
        dir.write(format!("mods/{title}/Zebra/romfs/asset"), b"last");
        let roots = [dir.0.join("mods"), dir.0.join("atmosphere"), dir.0.join("atmosphere/contents")];
        let entries = discover_mods(&roots, title_id).unwrap();
        assert_eq!(entries.len(), 6);
        assert!(entries[..2].iter().all(|entry| entry.name.starts_with("Legacy")));
        assert_eq!(entries[2..].iter().map(|entry| entry.name.as_str()).collect::<Vec<_>>(),
            ["alpha", "Disabled", "UltraCam", "Zebra"]);
        let unique = entries.iter().map(|entry| &entry.path).collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), entries.len());
        assert!(!entries[3].enabled);
        assert_eq!(entries[4].path, dir.0.join(format!("{atmosphere_title}/UltraCam")));
        assert!(entries[4].enabled && entries[4].has_romfs && entries[4].has_exefs);
        set_mod_enabled(&entries[3].path, true).unwrap();
        dir.write(format!("{atmosphere_title}/RomfsLiteEX/romfs/asset"), b"mod");
        let refreshed = discover_mods(&roots, title_id).unwrap();
        assert_eq!(refreshed.len(), 7);
        assert!(refreshed.iter().find(|entry| entry.name == "Disabled").unwrap().enabled);
        assert_eq!(refreshed[2..].iter().map(|entry| entry.name.as_str()).collect::<Vec<_>>(),
            ["alpha", "Disabled", "RomfsLiteEX", "UltraCam", "Zebra"]);
    }

    #[test]
    fn romfs_mods_merge_base_and_named_overlays_with_switching() {
        let dir = TestDir::new();
        let mut app = application();
        dir.write("base/keep", b"base");
        dir.write("base/shared", b"base-shared");
        let base = LayeredRomfs::build(None, &dir.0.join("base")).unwrap();
        let bytes = base.read(0, base.len() as usize).unwrap();
        app.romfs = Some(crate::LazyRomfs::from_section(mmap(&bytes), &crate::nca::NcaFsSection {
            index: 0, fs_type: crate::nca::NcaFsType::RomFs, hash_type: 0, encryption_type: 0,
            section_range: 0..bytes.len(), fs_data_range: 0..bytes.len(), compression: None, patch: None, sparse: false,
        }).unwrap());
        let title = format!("{:016X}", app.title_id);
        dir.write(format!("{title}/Alpha/romfs/shared"), b"alpha");
        dir.write(format!("{title}/Alpha/romfs/new/file"), b"added");
        dir.write(format!("{title}/Zebra/romfs/shared"), b"zebra");
        let roots = [dir.0.clone()];
        let entries = discover_mods(&roots, app.title_id).unwrap();
        let read = |entries: &[ModEntry]| {
            let romfs = build_mod_romfs(&app, entries).unwrap().unwrap();
            romfs.read(0, romfs.len() as usize).unwrap()
        };
        let bytes = read(&entries);
        assert_eq!(crate::romfs::romfs_file(&bytes, "/keep"), Some(&b"base"[..]));
        let hdr = crate::romfs::romfs_header(&bytes).unwrap();
        let (keep_offset, _) = crate::romfs::find_child_file(&bytes, hdr, 0, "keep").unwrap();
        let layered = build_mod_romfs(&app, &entries).unwrap().unwrap();
        assert_eq!(layered.read(keep_offset as u64 + 1, 2).unwrap(), b"as");
        assert_eq!(crate::romfs::romfs_file(&bytes, "/shared"), Some(&b"zebra"[..]));
        assert_eq!(crate::romfs::romfs_file(&bytes, "/new/file"), Some(&b"added"[..]));
        set_mod_enabled(&entries[1].path, false).unwrap();
        let entries = discover_mods(&roots, app.title_id).unwrap();
        assert_eq!(crate::romfs::romfs_file(&read(&entries), "/shared"), Some(&b"alpha"[..]));
        set_mod_enabled(&entries[0].path, false).unwrap();
        let entries = discover_mods(&roots, app.title_id).unwrap();
        assert_eq!(crate::romfs::romfs_file(&read(&entries), "/shared"), Some(&b"base-shared"[..]));
    }

    #[test]
    fn romfs_without_base_handles_file_directory_precedence() {
        let dir = TestDir::new();
        let app = application();
        let title = format!("{:016X}", app.title_id);
        dir.write(format!("{title}/Alpha/romfs/to_file/old"), b"old");
        dir.write(format!("{title}/Alpha/romfs/to_dir"), b"old");
        dir.write(format!("{title}/Zebra/romfs/to_file"), b"replacement");
        dir.write(format!("{title}/Zebra/romfs/to_dir/new"), b"new");
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        let romfs = build_mod_romfs(&app, &entries).unwrap().unwrap();
        let bytes = romfs.read(0, romfs.len() as usize).unwrap();
        assert_eq!(crate::romfs::romfs_file(&bytes, "/to_file"), Some(&b"replacement"[..]));
        assert_eq!(crate::romfs::romfs_file(&bytes, "/to_dir/new"), Some(&b"new"[..]));
        assert!(crate::romfs::romfs_file(&bytes, "/to_file/old").is_none());
    }

    #[test]
    fn exefs_replacements_add_modules_and_patch_matching_builds_in_mod_order() {
        let dir = TestDir::new();
        let mut app = application();
        let title = format!("{:016X}", app.title_id);
        let mut id = [0; 32];
        id[0] = 0xab;
        dir.write(format!("{title}/Alpha/exefs/main"), &nso_bytes(id, 2));
        dir.write(format!("{title}/Alpha/exefs/subsdk0"), &nso_bytes([0xcd; 32], 3));
        dir.write(format!("{title}/Alpha/exefs/ab.ips"), &patch_at(1, 4));
        dir.write(format!("{title}/Alpha/exefs/ff.ips"), &patch(8));
        dir.write(format!("{title}/Zebra/exefs/main"), &nso_bytes(id, 6));
        dir.write(format!("{title}/Zebra/exefs/AB.ips"), &patch(5));
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        apply_exefs_mods(&mut app, &entries).unwrap();
        assert_eq!(app.modules.iter().map(|module| module.name.as_str()).collect::<Vec<_>>(), ["main", "subsdk0"]);
        assert_eq!(app.modules[0].nso.build_id, id);
        assert_eq!(&app.modules[0].nso.module_image[..3], &[5, 4, 6]);
        assert_eq!(app.modules[1].nso.module_image[0], 3);
        assert_eq!(app.modules[1].load_offset, 0x3000);
        assert_eq!(app.total_code_size, 0x6000);
    }

    #[test]
    fn wrong_build_and_invalid_active_replacements_report_paths() {
        let dir = TestDir::new();
        let mut app = application();
        let title = format!("{:016X}", app.title_id);
        let path = dir.write(format!("{title}/WrongBuild/exefs/ab.ips"), &patch(7));
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        assert!(apply_exefs_mods(&mut app, &entries).unwrap_err().contains("WrongBuild"));
        set_mod_enabled(&entries[0].path, false).unwrap();
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        apply_exefs_mods(&mut app, &entries).unwrap();
        assert_eq!(app.modules[0].nso.module_image[0], 1);
        let replacement = path.parent().unwrap().join("main");
        std::fs::write(&replacement, b"invalid").unwrap();
        set_mod_enabled(&entries[0].path, true).unwrap();
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        let error = apply_exefs_mods(&mut app, &entries).unwrap_err();
        assert!(error.replace('\\', "/").contains(&replacement.display().to_string().replace('\\', "/")), "{error}");
    }

    #[test]
    fn build_id_matching_zero_pads_without_accepting_arbitrary_prefixes() {
        assert_eq!(parse_build_id("12"), Some(application().modules[0].nso.build_id));
        let mut longer = application().modules[0].nso.build_id;
        longer[1] = 0x34;
        assert_ne!(parse_build_id("12"), Some(longer));
        for invalid in ["", "1", "GG", "main", "１２"] {
            assert!(parse_build_id(invalid).is_none());
        }
        assert!(parse_build_id(&"11".repeat(33)).is_none());
        assert_eq!(parse_build_id(&"ab".repeat(32)), Some([0xab; 32]));
    }

    #[test]
    fn malformed_uncompressed_replacement_and_image_overflow_return_errors() {
        let dir = TestDir::new();
        let mut app = application();
        let title = format!("{:016X}", app.title_id);
        let mut malformed = nso_bytes([0xab; 32], 1);
        malformed[0x60..0x64].copy_from_slice(&0u32.to_le_bytes());
        let path = dir.write(format!("{title}/Broken/exefs/main"), &malformed);
        let entries = discover_mods(&[dir.0.clone()], app.title_id).unwrap();
        assert!(apply_exefs_mods(&mut app, &entries).unwrap_err().contains("uncompressed"));
        let mut overflow = nso_bytes([0xab; 32], 1);
        overflow[0x34..0x38].copy_from_slice(&(u32::MAX - 16).to_le_bytes());
        std::fs::write(&path, overflow).unwrap();
        assert!(apply_exefs_mods(&mut app, &entries).unwrap_err().contains("alignment"));
    }
}
