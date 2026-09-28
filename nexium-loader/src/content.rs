use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::cnmt::{Cnmt, ContentMetaType, ContentType};
use crate::container::{Nsp, PartitionFs};
use crate::nca::{Nca, NcaContentType, NcaFsType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId {
    pub title_id: u64,
    pub version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentKind {
    Update,
    Dlc,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledContent {
    pub id: ContentId,
    pub application_id: u64,
    pub kind: ContentKind,
    pub name: String,
    pub display_version: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameContent {
    pub application_id: u64,
    pub entries: Vec<InstalledContent>,
}

#[derive(Clone, Copy, Debug)]
pub struct ContentProgress {
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ContentPackage {
    pub entries: Vec<InstalledContent>,
    pub size_bytes: u64,
}

pub(crate) struct Package {
    pub mmap: Arc<Mmap>,
    nsp: Nsp,
    pub metadata: Vec<Cnmt>,
}

impl Package {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("Open {}: {error}", path.display()))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }
            .map_err(|error| format!("Map {}: {error}", path.display()))?);
        let nsp = Nsp::parse(mmap.clone()).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut metadata = Vec::new();
        for entry in nsp.ncas().entries() {
            if !entry.name.to_ascii_lowercase().ends_with(".nca") {
                continue;
            }
            let range = nsp.ncas().entry_range(entry)?;
            if range.len() as u64 != entry.size {
                return Err(format!("{} contains a truncated NCA: {}", path.display(), entry.name));
            }
            let nca = Nca::parse(mmap.clone(), range.start)
                .map_err(|error| format!("{} / {}: {error}", path.display(), entry.name))?;
            validate_nca(&nca, &range)?;
            if nca.content_type != NcaContentType::Meta {
                continue;
            }
            let section = nca.section(NcaFsType::PartitionFs)
                .ok_or_else(|| format!("Metadata NCA {} has no filesystem", entry.name))?;
            let pfs = PartitionFs::parse(mmap.clone(), section.fs_data_range.start)?;
            for cnmt_entry in pfs.entries().iter().filter(|entry| entry.name.ends_with(".cnmt")) {
                let cnmt_range = pfs.entry_range(cnmt_entry)?;
                if cnmt_range.end > section.fs_data_range.end || cnmt_range.len() as u64 != cnmt_entry.size {
                    return Err(format!("Truncated CNMT {}", cnmt_entry.name));
                }
                metadata.push(Cnmt::parse(&mmap[cnmt_range])?);
            }
        }
        if metadata.is_empty() {
            return Err(format!("{} contains no content metadata", path.display()));
        }
        let package = Self { mmap, nsp, metadata };
        for cnmt in &package.metadata {
            for record in cnmt.records.iter().filter(|record| record.content_type != ContentType::DeltaFragment) {
                package.nca(record)?;
            }
        }
        Ok(package)
    }

    pub(crate) fn nca(&self, record: &crate::cnmt::ContentRecord) -> Result<Nca, String> {
        let name = record.nca_filename();
        let entry = self.nsp.ncas().entries().iter().find(|entry| entry.name.eq_ignore_ascii_case(&name))
            .ok_or_else(|| format!("Required content {name} is missing from the package"))?;
        let range = self.nsp.ncas().entry_range(entry)?;
        if range.len() as u64 != entry.size {
            return Err(format!("Required content {name} is truncated"));
        }
        let nca = Nca::parse(self.mmap.clone(), range.start)?;
        validate_nca(&nca, &range)?;
        Ok(nca)
    }

    pub(crate) fn metadata_for(&self, entry: &InstalledContent) -> Result<&Cnmt, String> {
        self.metadata.iter().find(|cnmt| {
            cnmt.title_id == entry.id.title_id && cnmt.version == entry.id.version
                && cnmt.application_id == entry.application_id
                && kind_for(cnmt.meta_type) == Some(entry.kind)
        }).ok_or_else(|| format!("Installed package {} no longer matches {:016X} version {}", entry.path.display(), entry.id.title_id, entry.id.version))
    }
}

fn validate_nca(nca: &Nca, range: &std::ops::Range<usize>) -> Result<(), String> {
    if nca.content_size > range.len() as u64 || nca.sections.iter().any(|section| {
        section.section_range.start < range.start || section.section_range.end > range.end
    }) {
        return Err(format!("NCA {:016X} extends beyond its package entry", nca.program_id));
    }
    Ok(())
}

fn kind_for(kind: ContentMetaType) -> Option<ContentKind> {
    match kind {
        ContentMetaType::Patch => Some(ContentKind::Update),
        ContentMetaType::AddOnContent => Some(ContentKind::Dlc),
        _ => None,
    }
}

pub fn is_content_only_package(path: &Path) -> bool {
    let Ok(package) = Package::open(path) else { return false };
    package.metadata.iter().any(|metadata| kind_for(metadata.meta_type).is_some())
        && !package.metadata.iter().any(|metadata| metadata.meta_type == ContentMetaType::Application)
}

pub fn inspect_package(path: &Path) -> Result<ContentPackage, String> {
    let package = Package::open(path)?;
    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    for cnmt in &package.metadata {
        let Some(kind) = kind_for(cnmt.meta_type) else { continue };
        let id = ContentId { title_id: cnmt.title_id, version: cnmt.version };
        if !seen.insert(id) {
            return Err(format!("Duplicate content metadata {:016X} version {}", id.title_id, id.version));
        }
        if cnmt.application_id == 0 || cnmt.application_id & 0xFFF != 0 {
            return Err(format!("Content {:016X} has an invalid application ID", id.title_id));
        }
        if kind == ContentKind::Update && cnmt.find(ContentType::Program).is_none() {
            return Err(format!("Update {:016X} has no complete Program content; fragment-only updates are unsupported", id.title_id));
        }
        validate_supported_content(&package, cnmt, kind)?;
        let (name, display_version) = package_display(&package, cnmt);
        entries.push(InstalledContent {
            id,
            application_id: cnmt.application_id,
            kind,
            name,
            display_version,
            path: path.to_path_buf(),
            size_bytes: package.mmap.len() as u64,
            enabled: false,
        });
    }
    if entries.is_empty() {
        return Err(format!("{} contains no game updates or DLC", path.display()));
    }
    Ok(ContentPackage { entries, size_bytes: package.mmap.len() as u64 })
}

fn validate_supported_content(package: &Package, cnmt: &Cnmt, kind: ContentKind) -> Result<(), String> {
    for record in cnmt.records.iter().filter(|record| {
        matches!(record.content_type, ContentType::Program | ContentType::Data | ContentType::Control)
    }) {
        let nca = package.nca(record)?;
        if nca.sections.iter().any(|section| section.sparse) {
            return Err(format!("Content {:016X} uses sparse storage, which is not supported yet", cnmt.title_id));
        }
        if nca.sections.iter().any(|section| section.fs_type == NcaFsType::PartitionFs && (section.patch.is_some() || section.compression.is_some())) {
            return Err(format!("Content {:016X} uses an indirect or compressed ExeFS, which is not supported yet", cnmt.title_id));
        }
        if record.content_type == ContentType::Program {
            if nca.content_type != NcaContentType::Program || nca.section(NcaFsType::PartitionFs).is_none() {
                return Err(format!("Update {:016X} has no complete ExeFS", cnmt.title_id));
            }
        }
        if kind == ContentKind::Dlc && nca.sections.iter().any(|section| section.patch.is_some()) {
            return Err(format!("DLC {:016X} requires a separate data patch, which is not supported yet", cnmt.title_id));
        }
        if record.content_type == ContentType::Data && nca.section(NcaFsType::RomFs).is_none() {
            return Err(format!("DLC {:016X} has no RomFS data", cnmt.title_id));
        }
    }
    Ok(())
}

fn package_display(package: &Package, cnmt: &Cnmt) -> (String, String) {
    let metadata = (|| {
        let record = cnmt.find(ContentType::Control)?;
        let nca = package.nca(record).ok()?;
        let section = nca.section(NcaFsType::RomFs)?;
        let romfs = crate::LazyRomfs::from_section(package.mmap.clone(), section).ok()?;
        if romfs.len() > 64 * 1024 * 1024 { return None }
        let bytes = romfs.read(0, romfs.len() as usize).ok()?;
        let nacp = crate::romfs::romfs_file(&bytes, "/control.nacp")?;
        Some(crate::nro::parse_nacp(nacp))
    })();
    if let Some((name, _, version)) = metadata {
        (name, version)
    } else {
        (String::new(), String::new())
    }
}

fn game_dir(root: &Path, application_id: u64) -> PathBuf {
    root.join(format!("{application_id:016X}"))
}

pub fn list_game_content(root: &Path, application_id: u64) -> Result<GameContent, String> {
    let directory = game_dir(root, application_id);
    let path = directory.join("content.json");
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(GameContent { application_id, entries: Vec::new() }),
        Err(error) => return Err(format!("Read {}: {error}", path.display())),
    };
    let mut content: GameContent = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid content registry {}: {error}", path.display()))?;
    if content.application_id != application_id {
        return Err(format!("Content registry {} belongs to another game", path.display()));
    }
    let mut ids = HashSet::new();
    let mut update_selected = false;
    for entry in &mut content.entries {
        if entry.application_id != application_id || !ids.insert(entry.id) {
            return Err(format!("Content registry {} has inconsistent entries", path.display()));
        }
        let mut components = entry.path.components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(format!("Invalid managed content filename in {}", path.display()));
        }
        entry.path = directory.join("packages").join(&entry.path);
        if entry.kind == ContentKind::Update && entry.enabled {
            if update_selected { return Err("More than one update is selected".into()) }
            update_selected = true;
        }
    }
    content.entries.sort_by_key(|entry| (entry.kind == ContentKind::Dlc, entry.id.title_id, entry.id.version));
    Ok(content)
}

fn unique_name(prefix: &str, extension: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{prefix}-{nanos:x}-{:x}.{extension}", NEXT.fetch_add(1, Ordering::Relaxed))
}

fn save_content(root: &Path, content: &GameContent) -> Result<(), String> {
    let directory = game_dir(root, content.application_id);
    fs::create_dir_all(&directory).map_err(|error| format!("Create {}: {error}", directory.display()))?;
    let mut stored = content.clone();
    for entry in &mut stored.entries {
        entry.path = entry.path.file_name().map(PathBuf::from).ok_or("Invalid managed content path")?;
    }
    let bytes = serde_json::to_vec_pretty(&stored).map_err(|error| error.to_string())?;
    let temporary = directory.join(unique_name("content", "tmp"));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
            .map_err(|error| format!("Create {}: {error}", temporary.display()))?;
        file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(|error| error.to_string())?;
        drop(file);
        fs::rename(&temporary, directory.join("content.json")).map_err(|error| format!("Save content registry: {error}"))
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

pub fn install_package(
    root: &Path,
    application_id: u64,
    package: &Path,
    mut progress: impl FnMut(ContentProgress),
) -> Result<GameContent, String> {
    let inspected = inspect_package(package)?;
    if let Some(entry) = inspected.entries.iter().find(|entry| entry.application_id != application_id) {
        return Err(format!("This package belongs to game {:016X}, not {:016X}", entry.application_id, application_id));
    }
    let mut content = list_game_content(root, application_id)?;
    let directory = game_dir(root, application_id).join("packages");
    fs::create_dir_all(&directory).map_err(|error| format!("Create {}: {error}", directory.display()))?;
    let filename = unique_name(&format!("{:016X}", inspected.entries[0].id.title_id), "dnsp");
    let destination = directory.join(filename);
    let temporary = destination.with_extension("tmp");
    let result = (|| {
        let mut source = File::open(package).map_err(|error| format!("Open {}: {error}", package.display()))?;
        let mut target = OpenOptions::new().create_new(true).write(true).open(&temporary)
            .map_err(|error| format!("Create {}: {error}", temporary.display()))?;
        let mut buffer = vec![0; 1024 * 1024];
        let mut completed = 0;
        progress(ContentProgress { completed_bytes: completed, total_bytes: inspected.size_bytes });
        loop {
            let count = source.read(&mut buffer).map_err(|error| error.to_string())?;
            if count == 0 { break }
            target.write_all(&buffer[..count]).map_err(|error| error.to_string())?;
            completed += count as u64;
            progress(ContentProgress { completed_bytes: completed, total_bytes: inspected.size_bytes });
        }
        if completed != inspected.size_bytes { return Err("Source package changed during import".into()) }
        target.sync_all().map_err(|error| error.to_string())?;
        drop(target);
        let copied = inspect_package(&temporary)?;
        if copied.entries.iter().map(|entry| entry.id).collect::<Vec<_>>() != inspected.entries.iter().map(|entry| entry.id).collect::<Vec<_>>() {
            return Err("Source package metadata changed during import".into());
        }
        fs::rename(&temporary, &destination).map_err(|error| format!("Install {}: {error}", destination.display()))?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let previous = content.clone();
    let imported_update = inspected.entries.iter().filter(|entry| entry.kind == ContentKind::Update).max_by_key(|entry| entry.id.version).map(|entry| entry.id);
    if imported_update.is_some() {
        for entry in &mut content.entries {
            if entry.kind == ContentKind::Update { entry.enabled = false; }
        }
    }
    for mut entry in inspected.entries {
        content.entries.retain(|existing| existing.id != entry.id);
        if entry.kind == ContentKind::Dlc {
            for existing in &mut content.entries {
                if existing.kind == ContentKind::Dlc && existing.id.title_id == entry.id.title_id {
                    existing.enabled = false;
                }
            }
        }
        entry.path = destination.clone();
        entry.enabled = entry.kind == ContentKind::Dlc || Some(entry.id) == imported_update;
        content.entries.push(entry);
    }
    if let Err(error) = save_content(root, &content) {
        let _ = fs::remove_file(&destination);
        return Err(error);
    }
    remove_unreferenced(&previous, &content);
    list_game_content(root, application_id)
}

pub fn select_update(root: &Path, application_id: u64, id: Option<ContentId>) -> Result<GameContent, String> {
    let mut content = list_game_content(root, application_id)?;
    if let Some(id) = id {
        let entry = content.entries.iter().find(|entry| entry.id == id && entry.kind == ContentKind::Update)
            .ok_or("The selected update is no longer installed")?;
        let package = Package::open(&entry.path)?;
        package.metadata_for(entry)?;
    }
    for entry in &mut content.entries {
        if entry.kind == ContentKind::Update { entry.enabled = Some(entry.id) == id; }
    }
    save_content(root, &content)?;
    Ok(content)
}

pub fn set_dlc_enabled(root: &Path, application_id: u64, id: ContentId, enabled: bool) -> Result<GameContent, String> {
    let mut content = list_game_content(root, application_id)?;
    let entry = content.entries.iter_mut().find(|entry| entry.id == id && entry.kind == ContentKind::Dlc)
        .ok_or("The selected DLC is no longer installed")?;
    if enabled {
        let package = Package::open(&entry.path)?;
        package.metadata_for(entry)?;
    }
    entry.enabled = enabled;
    if enabled {
        for other in &mut content.entries {
            if other.kind == ContentKind::Dlc && other.id.title_id == id.title_id && other.id != id {
                other.enabled = false;
            }
        }
    }
    save_content(root, &content)?;
    Ok(content)
}

pub fn remove_content(root: &Path, application_id: u64, id: ContentId) -> Result<GameContent, String> {
    remove_contents(root, application_id, &[id])
}

pub fn remove_contents(root: &Path, application_id: u64, ids: &[ContentId]) -> Result<GameContent, String> {
    let previous = list_game_content(root, application_id)?;
    let mut content = previous.clone();
    content.entries.retain(|entry| !ids.contains(&entry.id));
    if content.entries.len() == previous.entries.len() { return Err("This content is no longer installed".into()) }
    save_content(root, &content)?;
    remove_unreferenced(&previous, &content);
    Ok(content)
}

fn remove_unreferenced(previous: &GameContent, current: &GameContent) {
    for entry in &previous.entries {
        if !current.entries.iter().any(|kept| kept.path == entry.path) {
            if let Err(error) = fs::remove_file(&entry.path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("Could not remove unused managed content {}: {error}", entry.path.display());
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const TITLE: u64 = 0x0100_1234_5678_0000;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(unique_name("nexium-content", "test"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn package(&self, name: &str, kind: ContentKind, version: u32, application_id: u64) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, package_bytes(kind, version, application_id, true)).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn pfs(files: &[(String, Vec<u8>)]) -> Vec<u8> {
        let mut names = Vec::new();
        let mut offsets = Vec::new();
        for (name, _) in files {
            offsets.push(names.len());
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        let header_size = 0x10 + files.len() * 0x18 + names.len();
        let mut result = vec![0; header_size];
        result[..4].copy_from_slice(b"PFS0");
        result[4..8].copy_from_slice(&(files.len() as u32).to_le_bytes());
        result[8..12].copy_from_slice(&(names.len() as u32).to_le_bytes());
        result[0x10 + files.len() * 0x18..].copy_from_slice(&names);
        let mut offset = 0u64;
        for (index, (_, bytes)) in files.iter().enumerate() {
            let at = 0x10 + index * 0x18;
            result[at..at + 8].copy_from_slice(&offset.to_le_bytes());
            result[at + 8..at + 16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            result[at + 16..at + 20].copy_from_slice(&(offsets[index] as u32).to_le_bytes());
            result.extend_from_slice(bytes);
            offset += bytes.len() as u64;
        }
        result
    }

    fn nca(kind: u8, fs_kind: u8, title_id: u64, body: &[u8]) -> Vec<u8> {
        let size = (0xC00 + body.len() + 0x1FF) & !0x1FF;
        let mut bytes = vec![0; size];
        bytes[0x200..0x204].copy_from_slice(b"DNCA");
        bytes[0x205] = kind;
        bytes[0x208..0x210].copy_from_slice(&(size as u64).to_le_bytes());
        bytes[0x210..0x218].copy_from_slice(&title_id.to_le_bytes());
        bytes[0x240..0x244].copy_from_slice(&6u32.to_le_bytes());
        bytes[0x244..0x248].copy_from_slice(&((size / 0x200) as u32).to_le_bytes());
        bytes[0x402] = fs_kind;
        bytes[0x403] = 1;
        bytes[0x404] = 1;
        bytes[0xC00..0xC00 + body.len()].copy_from_slice(body);
        bytes
    }

    fn package_bytes(kind: ContentKind, version: u32, application_id: u64, include_data: bool) -> Vec<u8> {
        let title_id = application_id + if kind == ContentKind::Update { 0x800 } else { 0x1001 };
        let payload = if kind == ContentKind::Update {
            nca(0, 1, application_id, &pfs(&[("main".into(), b"NSO0".to_vec())]))
        } else {
            nca(4, 0, title_id, &[0; 0x50])
        };
        let extension = if kind == ContentKind::Update { 0x18 } else { 0x10 };
        let record = 0x20 + extension;
        let mut cnmt = vec![0; record + 0x38];
        cnmt[..8].copy_from_slice(&title_id.to_le_bytes());
        cnmt[8..12].copy_from_slice(&version.to_le_bytes());
        cnmt[0xC] = if kind == ContentKind::Update { 0x81 } else { 0x82 };
        cnmt[0xE..0x10].copy_from_slice(&(extension as u16).to_le_bytes());
        cnmt[0x10..0x12].copy_from_slice(&1u16.to_le_bytes());
        cnmt[0x20..0x28].copy_from_slice(&application_id.to_le_bytes());
        cnmt[record + 0x20..record + 0x30].fill(0x42);
        cnmt[record + 0x30..record + 0x36].copy_from_slice(&(payload.len() as u64).to_le_bytes()[..6]);
        cnmt[record + 0x36] = if kind == ContentKind::Update { 1 } else { 2 };
        let meta = nca(1, 1, title_id, &pfs(&[("content.cnmt".into(), cnmt)]));
        let mut files = vec![("meta.cnmt.nca".into(), meta)];
        if include_data { files.push((format!("{}.nca", "42".repeat(16)), payload)); }
        pfs(&files)
    }

    #[test]
    fn packages_are_copied_and_enable_changes_survive_reload() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let source = temporary.package("dlc.dnsp", ContentKind::Dlc, 1, TITLE);
        let mut progress = Vec::new();
        let content = install_package(&root, TITLE, &source, |value| progress.push(value)).unwrap();
        let entry = &content.entries[0];
        assert!(entry.enabled);
        assert_ne!(entry.path, source);
        assert_eq!(fs::read(&entry.path).unwrap(), fs::read(&source).unwrap());
        assert_eq!(progress.first().unwrap().completed_bytes, 0);
        assert_eq!(progress.last().unwrap().completed_bytes, fs::metadata(&source).unwrap().len());
        let id = entry.id;
        let managed = entry.path.clone();
        set_dlc_enabled(&root, TITLE, id, false).unwrap();
        assert!(!list_game_content(&root, TITLE).unwrap().entries[0].enabled);
        remove_content(&root, TITLE, id).unwrap();
        assert!(!managed.exists());
        assert!(source.exists());
        assert!(list_game_content(&root, TITLE).unwrap().entries.is_empty());
    }

    #[test]
    fn selecting_updates_and_dlc_versions_is_exclusive() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        for (name, kind, version) in [
            ("update1.dnsp", ContentKind::Update, 65536),
            ("update2.dnsp", ContentKind::Update, 131072),
            ("dlc1.dnsp", ContentKind::Dlc, 1),
            ("dlc2.dnsp", ContentKind::Dlc, 2),
        ] {
            let source = temporary.package(name, kind, version, TITLE);
            install_package(&root, TITLE, &source, |_| {}).unwrap();
        }
        let content = list_game_content(&root, TITLE).unwrap();
        assert_eq!(content.entries.iter().filter(|entry| entry.enabled).count(), 2);
        let old_update = ContentId { title_id: TITLE + 0x800, version: 65536 };
        let old_dlc = ContentId { title_id: TITLE + 0x1001, version: 1 };
        select_update(&root, TITLE, Some(old_update)).unwrap();
        set_dlc_enabled(&root, TITLE, old_dlc, true).unwrap();
        let content = list_game_content(&root, TITLE).unwrap();
        assert_eq!(content.entries.iter().filter(|entry| entry.enabled).map(|entry| entry.id).collect::<HashSet<_>>(), HashSet::from([old_update, old_dlc]));
        select_update(&root, TITLE, None).unwrap();
        assert!(list_game_content(&root, TITLE).unwrap().entries.iter().filter(|entry| entry.kind == ContentKind::Update).all(|entry| !entry.enabled));
    }

    #[test]
    fn wrong_game_and_missing_content_leave_registry_unchanged() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let source = temporary.package("other.dnsp", ContentKind::Update, 1, TITLE + 0x2000);
        assert!(install_package(&root, TITLE, &source, |_| {}).unwrap_err().contains("belongs to game"));
        let missing = temporary.0.join("missing.dnsp");
        fs::write(&missing, package_bytes(ContentKind::Dlc, 1, TITLE, false)).unwrap();
        assert!(install_package(&root, TITLE, &missing, |_| {}).unwrap_err().contains("missing"));
        assert!(list_game_content(&root, TITLE).unwrap().entries.is_empty());
        assert!(!root.exists());
    }

    #[test]
    fn registry_cannot_remove_an_external_file() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let directory = game_dir(&root, TITLE);
        fs::create_dir_all(&directory).unwrap();
        let source = temporary.package("keep.dnsp", ContentKind::Dlc, 1, TITLE);
        let content = GameContent { application_id: TITLE, entries: inspect_package(&source).unwrap().entries };
        fs::write(directory.join("content.json"), serde_json::to_vec(&content).unwrap()).unwrap();
        assert!(remove_content(&root, TITLE, content.entries[0].id).unwrap_err().contains("filename"));
        assert!(source.exists());
    }

    #[test]
    fn removing_several_packages_updates_the_registry_once_and_keeps_sources() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let mut sources = Vec::new();
        for (name, kind, version) in [
            ("update1.dnsp", ContentKind::Update, 65536),
            ("update2.dnsp", ContentKind::Update, 131072),
            ("dlc1.dnsp", ContentKind::Dlc, 1),
            ("dlc2.dnsp", ContentKind::Dlc, 2),
        ] {
            let source = temporary.package(name, kind, version, TITLE);
            install_package(&root, TITLE, &source, |_| {}).unwrap();
            sources.push(source);
        }
        let installed = list_game_content(&root, TITLE).unwrap();
        let dlc: Vec<_> = installed.entries.iter().filter(|entry| entry.kind == ContentKind::Dlc).collect();
        let dlc_ids: Vec<_> = dlc.iter().map(|entry| entry.id).collect();
        let dlc_paths: Vec<_> = dlc.iter().map(|entry| entry.path.clone()).collect();
        assert_eq!(dlc_ids.len(), 2);
        let remaining = remove_contents(&root, TITLE, &dlc_ids).unwrap();
        assert!(remaining.entries.iter().all(|entry| entry.kind == ContentKind::Update));
        assert_eq!(list_game_content(&root, TITLE).unwrap().entries.len(), 2);
        assert!(dlc_paths.iter().all(|path| !path.exists()));
        assert!(remove_contents(&root, TITLE, &dlc_ids).unwrap_err().contains("no longer installed"));
        let update_ids: Vec<_> = remaining.entries.iter().map(|entry| entry.id).collect();
        assert!(remove_contents(&root, TITLE, &update_ids).unwrap().entries.is_empty());
        assert!(list_game_content(&root, TITLE).unwrap().entries.is_empty());
        assert!(sources.iter().all(|source| source.exists()));
    }

    #[test]
    #[ignore]
    fn installed_real_update_loads_and_can_be_disabled() {
        let base = std::env::var("NEXIUM_TEST_BASE_DNSP").expect("Set NEXIUM_TEST_BASE_DNSP");
        let update = PathBuf::from(std::env::var_os("NEXIUM_TEST_UPDATE_DNSP").expect("Set NEXIUM_TEST_UPDATE_DNSP"));
        let title = crate::read_application_title_id(Path::new(&base)).unwrap().unwrap();
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let installed = install_package(&root, title, &update, |_| {}).unwrap();
        let enabled = installed.entries.iter().find(|entry| entry.enabled && entry.kind == ContentKind::Update).unwrap();
        let application = crate::Application::load_with_content(&base, &root).unwrap();
        assert_eq!(application.title_id, title);
        assert_eq!(application.content_version, enabled.id.version);
        assert!(!application.modules.is_empty());
        let romfs = application.romfs.as_ref().unwrap();
        let header = romfs.read(0, 0x50).unwrap();
        assert_eq!(&header[..8], &0x50u64.to_le_bytes());
        assert!(application.patch_romfs.is_some());
        select_update(&root, title, None).unwrap();
        let unpatched = crate::Application::load_with_content(&base, &root).unwrap();
        assert_eq!(unpatched.title_id, title);
        assert!(unpatched.patch_romfs.is_none());
        assert_ne!(application.content_version, unpatched.content_version);
    }


    fn generated_dlc_package(application_id: u64, index: u64, required_version: u32, romfs: Option<&[u8]>) -> Vec<u8> {
        let title_id = application_id + 0x1000 + index;
        let payload = romfs.map(|bytes| nca(4, 0, title_id, bytes));
        let mut cnmt = vec![0; 0x30 + if payload.is_some() { 0x38 } else { 0 }];
        cnmt[..8].copy_from_slice(&title_id.to_le_bytes());
        cnmt[8..12].copy_from_slice(&1u32.to_le_bytes());
        cnmt[0xC] = 0x82;
        cnmt[0xE..0x10].copy_from_slice(&0x10u16.to_le_bytes());
        cnmt[0x10..0x12].copy_from_slice(&u16::from(payload.is_some()).to_le_bytes());
        cnmt[0x20..0x28].copy_from_slice(&application_id.to_le_bytes());
        cnmt[0x28..0x2C].copy_from_slice(&required_version.to_le_bytes());
        if let Some(payload) = &payload {
            cnmt[0x50..0x60].fill(0x43);
            cnmt[0x60..0x66].copy_from_slice(&(payload.len() as u64).to_le_bytes()[..6]);
            cnmt[0x66] = 2;
        }
        let meta = nca(1, 1, title_id, &pfs(&[("content.cnmt".into(), cnmt)]));
        let mut files = vec![("meta.cnmt.nca".into(), meta)];
        if let Some(payload) = payload {
            files.push((format!("{}.nca", "43".repeat(16)), payload));
        }
        pfs(&files)
    }

    #[test]
    #[ignore = "requires NEXIUM_TEST_BASE_DNSP pointing to a decrypted base game"]
    fn generated_dlc_installs_loads_disables_and_enforces_required_version() {
        let base = std::env::var("NEXIUM_TEST_BASE_DNSP").expect("Set NEXIUM_TEST_BASE_DNSP");
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        let original = crate::Application::load_with_content(&base, &root).unwrap();
        let title = original.title_id;
        let version = original.content_version;
        assert!(original.add_on_content.is_empty());
        drop(original);

        let romfs_directory = temporary.0.join("generated-romfs");
        fs::create_dir(&romfs_directory).unwrap();
        let proof = b"generated DLC data from its independently managed package";
        fs::write(romfs_directory.join("proof.bin"), proof).unwrap();
        let romfs = crate::LayeredRomfs::build(None, &romfs_directory).unwrap();
        let romfs_bytes = romfs.read(0, romfs.len() as usize).unwrap();
        drop(romfs);
        let data_source = temporary.0.join("data.dnsp");
        let unlock_source = temporary.0.join("unlock.dnsp");
        fs::write(&data_source, generated_dlc_package(title, 1, version, Some(&romfs_bytes))).unwrap();
        fs::write(&unlock_source, generated_dlc_package(title, 2, version, None)).unwrap();
        let data_id = ContentId { title_id: title + 0x1001, version: 1 };
        let unlock_id = ContentId { title_id: title + 0x1002, version: 1 };
        for source in [&data_source, &unlock_source] {
            let installed = install_package(&root, title, source, |_| {}).unwrap();
            assert!(installed.entries.iter().all(|entry| entry.path.starts_with(&root) && entry.path != *source));
            fs::remove_file(source).unwrap();
        }

        let application = crate::Application::load_with_content(&base, &root).unwrap();
        assert_eq!(application.title_id, title);
        assert_eq!(application.add_on_content.keys().copied().collect::<Vec<_>>(), vec![data_id.title_id, unlock_id.title_id]);
        assert!(application.add_on_content.get(&unlock_id.title_id).unwrap().is_none());
        let data = application.add_on_content.get(&data_id.title_id).unwrap().as_ref().unwrap().clone();
        assert!(!Arc::ptr_eq(&data.mmap, &application.mmap));
        let bytes = data.read(0, data.len() as usize).unwrap();
        assert_eq!(crate::romfs::romfs_file(&bytes, "/proof.bin").unwrap(), proof);
        drop(application);

        set_dlc_enabled(&root, title, data_id, false).unwrap();
        set_dlc_enabled(&root, title, unlock_id, false).unwrap();
        let disabled = crate::Application::load_with_content(&base, &root).unwrap();
        assert!(disabled.add_on_content.is_empty());
        assert_eq!(data.read(0, data.len() as usize).unwrap(), bytes);
        drop(disabled);
        set_dlc_enabled(&root, title, data_id, true).unwrap();

        let required = version.checked_add(1).expect("base version must allow a newer required version");
        let incompatible_source = temporary.0.join("requires-newer-version.dnsp");
        fs::write(&incompatible_source, generated_dlc_package(title, 3, required, None)).unwrap();
        install_package(&root, title, &incompatible_source, |_| {}).unwrap();
        let error = crate::Application::load_with_content(&base, &root).err().expect("incompatible DLC must fail loading");
        assert!(error.contains(&format!("{:016X}", title + 0x1003)), "{error}");
        assert!(error.contains(&format!("requires game version {required}")), "{error}");
        assert!(error.contains(&format!("selected version is {version}")), "{error}");
        set_dlc_enabled(&root, title, ContentId { title_id: title + 0x1003, version: 1 }, false).unwrap();
        let recovered = crate::Application::load_with_content(&base, &root).unwrap();
        assert_eq!(recovered.add_on_content.keys().copied().collect::<Vec<_>>(), vec![data_id.title_id]);
        let data = recovered.add_on_content[&data_id.title_id].as_ref().unwrap();
        let bytes = data.read(0, data.len() as usize).unwrap();
        assert_eq!(crate::romfs::romfs_file(&bytes, "/proof.bin").unwrap(), proof);
    }

    #[test]
    fn unsupported_storage_is_rejected_before_installation() {
        let temporary = TestDirectory::new();
        let root = temporary.0.join("installed");
        for (name, sparse) in [("sparse.dnsp", true), ("indirect-exefs.dnsp", false)] {
            let mut bytes = package_bytes(ContentKind::Update, 1, TITLE, true);
            let nca_base = bytes.windows(4).rposition(|window| window == b"DNCA").unwrap() - 0x200;
            if sparse {
                bytes[nca_base + 0x570] = 1;
            } else {
                bytes[nca_base + 0x508..nca_base + 0x510].copy_from_slice(&0x8000u64.to_le_bytes());
                bytes[nca_base + 0x510..nca_base + 0x514].copy_from_slice(b"BKTR");
                bytes[nca_base + 0x518..nca_base + 0x51C].copy_from_slice(&1u32.to_le_bytes());
            }
            let path = temporary.0.join(name);
            fs::write(&path, bytes).unwrap();
            let error = install_package(&root, TITLE, &path, |_| {}).unwrap_err();
            assert!(error.contains(if sparse { "sparse storage" } else { "indirect or compressed ExeFS" }), "{error}");
        }
        assert!(!root.exists());
    }


    #[test]
    fn content_only_detection_uses_metadata_instead_of_filenames() {
        let temporary = TestDirectory::new();
        let path = temporary.package("ordinary-game-name.dnsp", ContentKind::Update, 1, TITLE);
        assert!(is_content_only_package(&path));
        let mut bytes = fs::read(&path).unwrap();
        let mut signature = (TITLE + 0x800).to_le_bytes().to_vec();
        signature.extend_from_slice(&1u32.to_le_bytes());
        let cnmt = bytes.windows(signature.len()).position(|window| window == signature).unwrap();
        bytes[cnmt + 0xC] = 0x80;
        let base = temporary.0.join("misleading-update-name.dnsp");
        fs::write(&base, bytes).unwrap();
        assert!(!is_content_only_package(&base));
    }

}
