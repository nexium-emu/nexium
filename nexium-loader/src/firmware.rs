use std::collections::{hash_map, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::ops::Range;
use std::path::{Component, Path};
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::application::{detect, ContainerKind, LazyRomfs};
use crate::cnmt::{Cnmt, ContentMetaType, ContentType};
use crate::container::{Nsp, PartitionFs, Xci};
use crate::content::{unique_name, validate_nca};
use crate::nca::{Nca, NcaContentType, NcaFsType, DNCA_MAGIC};

pub const SYSTEM_UPDATE_TITLE_ID: u64 = 0x0100_0000_0000_0816;
pub const SYSTEM_VERSION_TITLE_ID: u64 = 0x0100_0000_0000_0809;

const REGISTRY: &str = "firmware.json";
const COPY_CHUNK: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FirmwarePackage {
    pub version: u32,
    pub display_version: String,
    pub title_count: usize,
    pub nca_count: usize,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct FirmwareScan {
    pub game: bool,
    pub packages: Vec<FirmwarePackage>,
    pub problems: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledFirmware {
    pub version: u32,
    pub display_version: String,
    pub directory: String,
    pub files: Vec<String>,
    pub size_bytes: u64,
    pub source: String,
}

#[derive(Clone, Copy, Debug)]
pub struct FirmwareProgress {
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

struct NcaFile {
    name: String,
    mmap: Arc<Mmap>,
    range: Range<usize>,
    size: u64,
}

impl NcaFile {
    fn encrypted(&self) -> bool {
        self.range.len() >= 0x204
            && crate::bin_read::u32at(&self.mmap[..], self.range.start + 0x200).ok() != Some(DNCA_MAGIC)
    }

    fn nca(&self) -> Result<Nca, String> {
        if self.range.len() as u64 != self.size {
            return Err(format!("{} is truncated", self.name));
        }
        let nca = Nca::parse(self.mmap.clone(), self.range.start).map_err(|error| format!("{}: {error}", self.name))?;
        validate_nca(&nca, &self.range).map_err(|error| format!("{}: {error}", self.name))?;
        Ok(nca)
    }

    fn metadata(&self, nca: &Nca) -> Result<Vec<Cnmt>, String> {
        let section = nca.section(NcaFsType::PartitionFs)
            .ok_or_else(|| format!("Metadata NCA {} has no filesystem", self.name))?;
        let pfs = PartitionFs::parse(self.mmap.clone(), section.fs_data_range.start)?;
        let mut metadata = Vec::new();
        for entry in pfs.entries().iter().filter(|entry| entry.name.ends_with(".cnmt")) {
            let range = pfs.entry_range(entry)?;
            if range.end > section.fs_data_range.end || range.len() as u64 != entry.size {
                return Err(format!("Truncated CNMT {} in {}", entry.name, self.name));
            }
            metadata.push(Cnmt::parse(&self.mmap[range])?);
        }
        Ok(metadata)
    }
}

struct Meta {
    file: usize,
    cnmt: Cnmt,
}

struct Catalog {
    files: Vec<NcaFile>,
    metas: Vec<Meta>,
    encrypted: usize,
    problems: Vec<String>,
}

struct Resolved {
    version: u32,
    display_version: String,
    title_count: usize,
    files: Vec<usize>,
}

impl Catalog {
    fn from_partition(mmap: &Arc<Mmap>, ncas: &PartitionFs) -> Self {
        let mut files = Vec::new();
        let mut problems = Vec::new();
        for entry in ncas.entries().iter().filter(|entry| entry.name.to_ascii_lowercase().ends_with(".nca")) {
            match ncas.entry_range(entry) {
                Ok(range) => files.push(NcaFile { name: entry.name.clone(), mmap: mmap.clone(), range, size: entry.size }),
                Err(error) => problems.push(format!("{}: {error}", entry.name)),
            }
        }
        Self::index(files, problems)
    }

    fn from_directory(directory: &Path, names: &[String]) -> Result<Self, String> {
        let mut files = Vec::with_capacity(names.len());
        for name in names {
            let mmap = map(&directory.join(name))?;
            files.push(NcaFile { name: name.clone(), range: 0..mmap.len(), size: mmap.len() as u64, mmap });
        }
        Ok(Self::index(files, Vec::new()))
    }

    fn index(files: Vec<NcaFile>, mut problems: Vec<String>) -> Self {
        let mut metas = Vec::new();
        let mut encrypted = 0;
        for (index, file) in files.iter().enumerate() {
            if file.encrypted() {
                encrypted += 1;
                continue;
            }
            match file.nca() {
                Ok(nca) if nca.content_type == NcaContentType::Meta => match file.metadata(&nca) {
                    Ok(found) => metas.extend(found.into_iter().map(|cnmt| Meta { file: index, cnmt })),
                    Err(error) => problems.push(error),
                },
                Ok(_) => {}
                Err(error) => problems.push(error),
            }
        }
        Self { files, metas, encrypted, problems }
    }

    fn updates(&self) -> Vec<&Meta> {
        let mut updates: Vec<&Meta> = self.metas.iter().filter(|meta| {
            meta.cnmt.meta_type == ContentMetaType::SystemUpdate && meta.cnmt.title_id == SYSTEM_UPDATE_TITLE_ID
        }).collect();
        updates.sort_by(|a, b| b.cnmt.version.cmp(&a.cnmt.version));
        updates.dedup_by_key(|meta| meta.cnmt.version);
        updates
    }

    fn resolve(&self, update: &Meta) -> Result<Resolved, String> {
        if update.cnmt.content_meta.is_empty() {
            return Err("its system update metadata lists no system titles".into());
        }
        let mut files = BTreeSet::from([update.file]);
        let mut missing = Vec::new();
        let mut system_version = None;
        for info in &update.cnmt.content_meta {
            let Some(meta) = self.metas.iter().find(|meta| {
                meta.cnmt.title_id == info.title_id && meta.cnmt.version == info.version && meta.cnmt.meta_type == info.meta_type
            }) else {
                missing.push(format!("{:016X} v{}", info.title_id, info.version));
                continue;
            };
            files.insert(meta.file);
            for record in meta.cnmt.records.iter().filter(|record| record.content_type != ContentType::DeltaFragment) {
                let name = record.nca_filename();
                let Some(index) = self.files.iter().position(|file| file.name.eq_ignore_ascii_case(&name)) else {
                    missing.push(format!("{:016X} content {name}", info.title_id));
                    continue;
                };
                let nca = self.files[index].nca()?;
                if !content_type_matches(record.content_type, nca.content_type) {
                    return Err(format!("{name} is not the {:?} content of {:016X}", record.content_type, info.title_id));
                }
                if nca.sections.iter().any(|section| section.patch.is_some() || section.sparse) {
                    return Err(format!("{name} uses patch or sparse storage, which firmware can't use"));
                }
                if info.title_id == SYSTEM_VERSION_TITLE_ID && record.content_type == ContentType::Data {
                    system_version = Some(index);
                }
                files.insert(index);
            }
        }
        if !missing.is_empty() {
            let more = missing.len().saturating_sub(3);
            missing.truncate(3);
            let more = if more > 0 { format!(" and {more} more") } else { String::new() };
            return Err(format!("the firmware is incomplete; missing {}{more}", missing.join(", ")));
        }
        let display_version = system_version
            .and_then(|index| system_version_name(&self.files[index]))
            .unwrap_or_else(|| version_name(update.cnmt.version));
        Ok(Resolved {
            version: update.cnmt.version,
            display_version,
            title_count: update.cnmt.content_meta.len(),
            files: files.into_iter().collect(),
        })
    }

    fn package(&self, resolved: &Resolved) -> FirmwarePackage {
        FirmwarePackage {
            version: resolved.version,
            display_version: resolved.display_version.clone(),
            title_count: resolved.title_count,
            nca_count: resolved.files.len(),
            size_bytes: resolved.files.iter().map(|&index| self.files[index].size).sum(),
        }
    }
}

fn content_type_matches(expected: ContentType, actual: NcaContentType) -> bool {
    match expected {
        ContentType::Meta => actual == NcaContentType::Meta,
        ContentType::Program => actual == NcaContentType::Program,
        ContentType::Data => matches!(actual, NcaContentType::Data | NcaContentType::PublicData),
        ContentType::Control => actual == NcaContentType::Control,
        ContentType::HtmlDocument | ContentType::LegalInformation => actual == NcaContentType::Manual,
        ContentType::DeltaFragment | ContentType::Unknown => true,
    }
}

fn system_version_name(file: &NcaFile) -> Option<String> {
    let nca = file.nca().ok()?;
    let section = nca.section(NcaFsType::RomFs)?;
    let romfs = LazyRomfs::from_section(file.mmap.clone(), section).ok()?;
    if romfs.len() > 0x10_0000 {
        return None;
    }
    let bytes = romfs.read(0, romfs.len() as usize).ok()?;
    let data = crate::romfs::romfs_file(&bytes, "/file")?;
    let display = data.get(0x68..0x80)?;
    let display = &display[..display.iter().position(|&byte| byte == 0).unwrap_or(display.len())];
    let display = String::from_utf8_lossy(display).trim().to_string();
    if !display.is_empty() {
        return Some(display);
    }
    data.get(..3).map(|number| format!("{}.{}.{}", number[0], number[1], number[2]))
}

pub fn version_name(version: u32) -> String {
    if version >> 26 >= 3 {
        format!("{}.{}.{}", version >> 26, (version >> 20) & 0x3F, (version >> 16) & 0xF)
    } else {
        format!("version {version}")
    }
}

struct Source {
    catalog: Option<Catalog>,
    game: bool,
    firmware_partition: bool,
    unreadable: Option<String>,
}

fn open(path: &Path) -> Result<Source, String> {
    let mmap = map(path)?;
    match detect(&path.to_string_lossy(), &mmap) {
        ContainerKind::Dxci => {
            let xci = Xci::parse(mmap.clone())?;
            Ok(Source {
                catalog: xci.update_ncas().map(|update| Catalog::from_partition(&mmap, update)),
                game: true,
                firmware_partition: true,
                unreadable: xci.update_error().map(str::to_string),
            })
        }
        ContainerKind::Dnsp => {
            let nsp = Nsp::parse(mmap.clone())?;
            let catalog = Catalog::from_partition(&mmap, nsp.ncas());
            let game = catalog.metas.iter().any(|meta| meta.cnmt.meta_type == ContentMetaType::Application);
            Ok(Source { catalog: Some(catalog), game, firmware_partition: false, unreadable: None })
        }
        _ => Err(format!("{} is not a decrypted .dxci or .dnsp file", path.display())),
    }
}

pub fn scan(path: &Path) -> Result<FirmwareScan, String> {
    let source = open(path)?;
    let mut scan = FirmwareScan { game: source.game, ..FirmwareScan::default() };
    if let Some(error) = &source.unreadable {
        scan.problems.push(format!("The firmware partition can't be read, so the file may be incomplete ({error})."));
    }
    let Some(catalog) = source.catalog else { return Ok(scan) };
    let updates = catalog.updates();
    for update in &updates {
        match catalog.resolve(update) {
            Ok(resolved) => scan.packages.push(catalog.package(&resolved)),
            Err(error) => scan.problems.push(format!("Firmware {} can't be installed: {error}.", version_name(update.cnmt.version))),
        }
    }
    if updates.is_empty() && source.firmware_partition {
        if catalog.encrypted > 0 {
            scan.problems.push(format!(
                "{} firmware file(s) are still encrypted. Decrypt the original .xci with NXDecrypt to use its firmware.",
                catalog.encrypted
            ));
        } else if catalog.problems.is_empty() && !catalog.files.is_empty() {
            scan.problems.push(format!(
                "The firmware partition has {} file(s) but no system update metadata, so it can't be installed.",
                catalog.files.len()
            ));
        }
        scan.problems.extend(catalog.problems.iter().take(3).cloned());
    }
    Ok(scan)
}

pub fn installed_firmware(root: &Path) -> Result<Option<InstalledFirmware>, String> {
    let path = root.join(REGISTRY);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Read {}: {error}", path.display())),
    };
    let installed: InstalledFirmware = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid firmware registry {}: {error}", path.display()))?;
    if !is_plain(&installed.directory) || !installed.files.iter().all(|name| is_plain(name)) {
        return Err(format!("Firmware registry {} has invalid entries", path.display()));
    }
    Ok(Some(installed))
}

pub fn install_firmware(
    root: &Path,
    source: &Path,
    version: u32,
    mut progress: impl FnMut(FirmwareProgress),
) -> Result<InstalledFirmware, String> {
    let catalog = open(source)?.catalog
        .ok_or_else(|| format!("{} has no readable firmware partition", source.display()))?;
    let update = catalog.updates().into_iter().find(|update| update.cnmt.version == version)
        .ok_or_else(|| format!("{} doesn't contain firmware {}", source.display(), version_name(version)))?;
    let resolved = catalog.resolve(update)?;
    let names = resolved.files.iter()
        .map(|&index| installable_name(&catalog.files[index].name))
        .collect::<Result<Vec<_>, _>>()?;
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err("The firmware contains duplicate file names".into());
    }
    let total_bytes = catalog.package(&resolved).size_bytes;
    fs::create_dir_all(root).map_err(|error| format!("Create {}: {error}", root.display()))?;
    let staging = root.join(unique_name(".staging", "tmp"));
    fs::create_dir(&staging).map_err(|error| format!("Create {}: {error}", staging.display()))?;
    let copied = (|| {
        let mut completed = 0;
        progress(FirmwareProgress { completed_bytes: completed, total_bytes });
        for (&index, name) in resolved.files.iter().zip(&names) {
            let file = &catalog.files[index];
            let path = staging.join(name);
            let mut target = OpenOptions::new().write(true).create_new(true).open(&path)
                .map_err(|error| format!("Create {}: {error}", path.display()))?;
            for chunk in file.mmap[file.range.clone()].chunks(COPY_CHUNK) {
                target.write_all(chunk).map_err(|error| format!("Write {}: {error}", path.display()))?;
                completed += chunk.len() as u64;
                progress(FirmwareProgress { completed_bytes: completed, total_bytes });
            }
            target.sync_all().map_err(|error| format!("Write {}: {error}", path.display()))?;
        }
        let copy = Catalog::from_directory(&staging, &names)?;
        let update = copy.updates().into_iter().find(|update| update.cnmt.version == version)
            .ok_or("The copied firmware lost its system update metadata")?;
        if copy.resolve(update)?.files.len() != names.len() {
            return Err("The copied firmware doesn't match its source".to_string());
        }
        Ok(())
    })();
    if let Err(error) = copied {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    let label: String = resolved.display_version.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '_' })
        .collect();
    let installed = InstalledFirmware {
        version,
        display_version: resolved.display_version.clone(),
        directory: unique_name(&label, "firmware"),
        files: names,
        size_bytes: total_bytes,
        source: source.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default(),
    };
    let destination = root.join(&installed.directory);
    if let Err(error) = fs::rename(&staging, &destination) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("Install firmware into {}: {error}", destination.display()));
    }
    if let Err(error) = save_registry(root, &installed) {
        let _ = fs::remove_dir_all(&destination);
        return Err(error);
    }
    remove_unused(root, &installed.directory);
    Ok(installed)
}

pub fn remove_firmware(root: &Path) -> Result<Option<InstalledFirmware>, String> {
    let installed = installed_firmware(root).ok().flatten();
    match fs::remove_file(root.join(REGISTRY)) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("Remove firmware registry: {error}")),
    }
    remove_unused(root, "");
    Ok(installed)
}

pub fn add_installed_system_archives(root: &Path, archives: &mut HashMap<u64, LazyRomfs>) -> Result<usize, String> {
    let Some(installed) = installed_firmware(root)? else { return Ok(0) };
    let directory = root.join(&installed.directory);
    let mut added = 0;
    for name in &installed.files {
        let path = directory.join(name);
        let archive = (|| -> Result<Option<(u64, LazyRomfs)>, String> {
            let mmap = map(&path)?;
            let nca = Nca::parse(mmap.clone(), 0)?;
            if !matches!(nca.content_type, NcaContentType::Data | NcaContentType::PublicData) {
                return Ok(None);
            }
            let Some(section) = nca.section(NcaFsType::RomFs) else { return Ok(None) };
            LazyRomfs::from_section(mmap.clone(), section).map(|romfs| Some((nca.program_id, romfs)))
        })();
        match archive {
            Ok(Some((title_id, romfs))) => {
                if let hash_map::Entry::Vacant(slot) = archives.entry(title_id) {
                    slot.insert(romfs);
                    added += 1;
                }
            }
            Ok(None) => {}
            Err(error) => log::warn!("Skipping installed firmware file {}: {error}", path.display()),
        }
    }
    Ok(added)
}

fn map(path: &Path) -> Result<Arc<Mmap>, String> {
    let file = File::open(path).map_err(|error| format!("Open {}: {error}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| format!("Map {}: {error}", path.display()))?;
    Ok(Arc::new(mmap))
}

fn installable_name(name: &str) -> Result<String, String> {
    let lower = name.to_ascii_lowercase();
    if lower.len() <= 64 && lower.ends_with(".nca") && !lower.starts_with('.')
        && lower.chars().all(|c| c.is_ascii_alphanumeric() || c == '.')
    {
        Ok(lower)
    } else {
        Err(format!("Refusing to install unexpected firmware file {name:?}"))
    }
}

fn is_plain(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn save_registry(root: &Path, installed: &InstalledFirmware) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(installed).map_err(|error| error.to_string())?;
    let temporary = root.join(unique_name("firmware", "tmp"));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
            .map_err(|error| format!("Create {}: {error}", temporary.display()))?;
        file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(|error| error.to_string())?;
        drop(file);
        fs::rename(&temporary, root.join(REGISTRY)).map_err(|error| format!("Save firmware registry: {error}"))
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

fn remove_unused(root: &Path, keep: &str) {
    let Ok(entries) = fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name.to_str().is_some_and(|name| name == keep || name == REGISTRY) {
            continue;
        }
        let result = if path.is_dir() {
            fs::remove_dir_all(&path)
        } else if path.extension().is_some_and(|extension| extension == "tmp") {
            fs::remove_file(&path)
        } else {
            continue;
        };
        if let Err(error) = result {
            log::warn!("Could not remove old firmware files {}: {error}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::tests::{nca, pfs, TestDirectory};
    use std::path::PathBuf;

    const V16: u32 = 16 << 26;
    const V17: u32 = 17 << 26;
    const FONT: u64 = 0x0100_0000_0000_0810;

    fn hex(id: [u8; 16]) -> String {
        id.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn hfs(files: &[(String, Vec<u8>)]) -> Vec<u8> {
        let mut names = Vec::new();
        let mut offsets = Vec::new();
        for (name, _) in files {
            offsets.push(names.len());
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        let mut result = vec![0; 0x10 + files.len() * 0x40];
        result[..4].copy_from_slice(b"HFS0");
        result[4..8].copy_from_slice(&(files.len() as u32).to_le_bytes());
        result[8..12].copy_from_slice(&(names.len() as u32).to_le_bytes());
        result.extend_from_slice(&names);
        let mut offset = 0u64;
        for (index, (_, bytes)) in files.iter().enumerate() {
            let at = 0x10 + index * 0x40;
            result[at..at + 8].copy_from_slice(&offset.to_le_bytes());
            result[at + 8..at + 16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            result[at + 16..at + 20].copy_from_slice(&(offsets[index] as u32).to_le_bytes());
            result[at + 20..at + 24].copy_from_slice(&0x200u32.to_le_bytes());
            offset += bytes.len() as u64;
        }
        for (_, bytes) in files {
            result.extend_from_slice(bytes);
        }
        result
    }

    fn dxci(update: &[(String, Vec<u8>)], update_last: bool) -> Vec<u8> {
        let update = ("update".to_string(), hfs(update));
        let secure = ("secure".to_string(), hfs(&[]));
        let partitions = if update_last { [secure, update] } else { [update, secure] };
        let mut bytes = vec![0; 0x200];
        bytes[0x100..0x104].copy_from_slice(b"DXCI");
        bytes[0x130..0x138].copy_from_slice(&0x200u64.to_le_bytes());
        bytes.extend_from_slice(&hfs(&partitions));
        bytes
    }

    fn romfs(name: &str, contents: &[u8]) -> Vec<u8> {
        let directory = TestDirectory::new();
        fs::write(directory.0.join(name), contents).unwrap();
        let romfs = crate::LayeredRomfs::build(None, &directory.0).unwrap();
        romfs.read(0, romfs.len() as usize).unwrap()
    }

    fn cnmt(title_id: u64, version: u32, meta_type: u8, records: &[([u8; 16], usize)], titles: &[u64]) -> Vec<u8> {
        let extended = if meta_type == 0x03 { 4 } else { 0 };
        let table = 0x20 + extended;
        let mut bytes = vec![0; table + records.len() * 0x38 + titles.len() * 0x10 + 0x20];
        bytes[..8].copy_from_slice(&title_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&version.to_le_bytes());
        bytes[0xC] = meta_type;
        bytes[0xE..0x10].copy_from_slice(&(extended as u16).to_le_bytes());
        bytes[0x10..0x12].copy_from_slice(&(records.len() as u16).to_le_bytes());
        bytes[0x12..0x14].copy_from_slice(&(titles.len() as u16).to_le_bytes());
        for (index, (id, size)) in records.iter().enumerate() {
            let at = table + index * 0x38;
            bytes[at + 0x20..at + 0x30].copy_from_slice(id);
            bytes[at + 0x30..at + 0x36].copy_from_slice(&(*size as u64).to_le_bytes()[..6]);
            bytes[at + 0x36] = 2;
        }
        for (index, title) in titles.iter().enumerate() {
            let at = table + records.len() * 0x38 + index * 0x10;
            bytes[at..at + 8].copy_from_slice(&title.to_le_bytes());
            bytes[at + 8..at + 12].copy_from_slice(&version.to_le_bytes());
            bytes[at + 0xC] = 0x02;
        }
        bytes
    }

    fn firmware(version: u32, display: &str, seed: u8) -> Vec<(String, Vec<u8>)> {
        let mut system_version = vec![0; 0x100];
        system_version[0] = (version >> 26) as u8;
        system_version[0x68..0x68 + display.len()].copy_from_slice(display.as_bytes());
        let titles = [
            (SYSTEM_VERSION_TITLE_ID, [seed; 16], nca(4, 0, SYSTEM_VERSION_TITLE_ID, &romfs("file", &system_version))),
            (FONT, [seed + 1; 16], nca(4, 0, FONT, &romfs("nintendo_udsg-r_std_003.bfttf", display.as_bytes()))),
        ];
        let mut files = Vec::new();
        for (index, (title_id, id, data)) in titles.into_iter().enumerate() {
            let metadata = cnmt(title_id, version, 0x02, &[(id, data.len())], &[]);
            files.push((format!("{}.cnmt.nca", hex([seed + 2 + index as u8; 16])), nca(1, 1, title_id, &pfs(&[("title.cnmt".into(), metadata)]))));
            files.push((format!("{}.nca", hex(id)), data));
        }
        let update = cnmt(SYSTEM_UPDATE_TITLE_ID, version, 0x03, &[], &[SYSTEM_VERSION_TITLE_ID, FONT]);
        files.push((format!("{}.cnmt.nca", hex([seed + 4; 16])), nca(1, 1, SYSTEM_UPDATE_TITLE_ID, &pfs(&[("update.cnmt".into(), update)]))));
        files
    }

    fn write(directory: &TestDirectory, name: &str, bytes: &[u8]) -> PathBuf {
        let path = directory.0.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn game_only_dxci_has_no_firmware() {
        let directory = TestDirectory::new();
        let found = scan(&write(&directory, "game.dxci", &dxci(&[], false))).unwrap();
        assert!(found.game);
        assert!(found.packages.is_empty());
        assert!(found.problems.is_empty(), "{:?}", found.problems);
    }

    #[test]
    fn firmware_in_the_update_partition_is_detected_with_its_display_version() {
        let directory = TestDirectory::new();
        let files = firmware(V17, "17.0.0", 0x10);
        let size_bytes = files.iter().map(|(_, bytes)| bytes.len() as u64).sum();
        let found = scan(&write(&directory, "game.dxci", &dxci(&files, false))).unwrap();
        assert!(found.problems.is_empty(), "{:?}", found.problems);
        assert_eq!(found.packages, [FirmwarePackage {
            version: V17,
            display_version: "17.0.0".into(),
            title_count: 2,
            nca_count: 5,
            size_bytes,
        }]);
    }

    #[test]
    fn installing_copies_only_the_selected_firmware_and_feeds_system_archives() {
        let directory = TestDirectory::new();
        let root = directory.0.join("firmware");
        let mut files = firmware(V16, "16.0.0", 0x20);
        files.extend(firmware(V17, "17.0.0", 0x10));
        let path = write(&directory, "both.dxci", &dxci(&files, false));
        let found = scan(&path).unwrap();
        assert_eq!(found.packages.iter().map(|package| package.version).collect::<Vec<_>>(), [V17, V16]);

        let mut progress = Vec::new();
        let older = install_firmware(&root, &path, V16, |value| progress.push(value)).unwrap();
        assert_eq!(older.display_version, "16.0.0");
        assert_eq!(older.files.len(), 5);
        assert_eq!(older.size_bytes, found.packages[1].size_bytes);
        assert_eq!(progress.first().unwrap().completed_bytes, 0);
        assert_eq!(progress.last().unwrap().completed_bytes, older.size_bytes);
        assert_eq!(installed_firmware(&root).unwrap(), Some(older.clone()));
        assert!(older.files.iter().all(|name| root.join(&older.directory).join(name).is_file()));

        let bundled = write(&directory, "bundled.bin", b"bundled");
        let mut archives = HashMap::from([(FONT, LazyRomfs::from_range(map(&bundled).unwrap(), 0..7).unwrap())]);
        assert_eq!(add_installed_system_archives(&root, &mut archives).unwrap(), 1);
        assert_eq!(archives[&FONT].read(0, 7).unwrap(), b"bundled");
        let version = &archives[&SYSTEM_VERSION_TITLE_ID];
        let bytes = version.read(0, version.len() as usize).unwrap();
        assert_eq!(&crate::romfs::romfs_file(&bytes, "/file").unwrap()[0x68..0x6E], b"16.0.0");
        drop(archives);

        let newer = install_firmware(&root, &path, V17, |_| {}).unwrap();
        assert_eq!(installed_firmware(&root).unwrap(), Some(newer.clone()));
        let entries: BTreeSet<_> = fs::read_dir(&root).unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(entries, BTreeSet::from([REGISTRY.to_string(), newer.directory.clone()]));
    }

    #[test]
    fn incomplete_or_truncated_firmware_is_rejected_without_leaving_files() {
        let directory = TestDirectory::new();
        let root = directory.0.join("firmware");
        let mut files = firmware(V17, "17.0.0", 0x10);
        let font = format!("{}.nca", hex([0x11; 16]));
        files.retain(|(name, _)| *name != font);
        let incomplete = write(&directory, "incomplete.dxci", &dxci(&files, false));
        let found = scan(&incomplete).unwrap();
        assert!(found.packages.is_empty());
        assert!(found.problems.iter().any(|problem| problem.contains("incomplete")), "{:?}", found.problems);
        assert!(install_firmware(&root, &incomplete, V17, |_| {}).unwrap_err().contains("incomplete"));

        let mut bytes = dxci(&firmware(V17, "17.0.0", 0x10), true);
        bytes.truncate(bytes.len() - 0x100);
        let truncated = write(&directory, "truncated.dxci", &bytes);
        let found = scan(&truncated).unwrap();
        assert!(found.packages.is_empty());
        assert!(found.problems.iter().any(|problem| problem.contains("truncated")), "{:?}", found.problems);
        assert!(install_firmware(&root, &truncated, V17, |_| {}).is_err());
        assert!(!root.exists());
    }

    #[test]
    fn encrypted_firmware_is_reported_instead_of_installed() {
        let directory = TestDirectory::new();
        let mut files = firmware(V17, "17.0.0", 0x10);
        for (_, bytes) in &mut files {
            bytes[0x200..0x204].copy_from_slice(&[0x8F, 0x12, 0xA0, 0x33]);
        }
        let found = scan(&write(&directory, "encrypted.dxci", &dxci(&files, false))).unwrap();
        assert!(found.packages.is_empty());
        assert!(found.problems.iter().any(|problem| problem.contains("encrypted") && problem.contains("NXDecrypt")), "{:?}", found.problems);
    }

    #[test]
    fn firmware_only_dnsp_is_installable_but_not_a_game() {
        let directory = TestDirectory::new();
        let path = write(&directory, "Firmware 17.0.0.dnsp", &pfs(&firmware(V17, "17.0.0", 0x10)));
        let found = scan(&path).unwrap();
        assert!(!found.game);
        assert_eq!(found.packages.len(), 1);
        assert!(crate::content::is_content_only_package(&path));
        let installed = install_firmware(&directory.0.join("firmware"), &path, V17, |_| {}).unwrap();
        assert_eq!(installed.source, "Firmware 17.0.0.dnsp");
    }

    #[test]
    fn removing_firmware_unregisters_it_and_deletes_its_files() {
        let directory = TestDirectory::new();
        let root = directory.0.join("firmware");
        let path = write(&directory, "game.dxci", &dxci(&firmware(V17, "17.0.0", 0x10), false));
        let installed = install_firmware(&root, &path, V17, |_| {}).unwrap();
        assert_eq!(remove_firmware(&root).unwrap(), Some(installed));
        assert_eq!(installed_firmware(&root).unwrap(), None);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        assert_eq!(add_installed_system_archives(&root, &mut HashMap::new()).unwrap(), 0);
        assert_eq!(remove_firmware(&root).unwrap(), None);
        assert!(path.exists());
    }

    #[test]
    fn registry_cannot_point_outside_the_firmware_directory() {
        let directory = TestDirectory::new();
        let installed = InstalledFirmware {
            version: V17,
            display_version: "17.0.0".into(),
            directory: "../outside".into(),
            files: Vec::new(),
            size_bytes: 0,
            source: String::new(),
        };
        fs::write(directory.0.join(REGISTRY), serde_json::to_vec(&installed).unwrap()).unwrap();
        assert!(installed_firmware(&directory.0).unwrap_err().contains("invalid entries"));
        assert!(add_installed_system_archives(&directory.0, &mut HashMap::new()).is_err());
    }

    #[test]
    fn system_update_versions_decode_to_firmware_numbers() {
        assert_eq!(version_name(17 << 26), "17.0.0");
        assert_eq!(version_name(16 << 26 | 1 << 20), "16.1.0");
        assert_eq!(version_name(15 << 26 | 1 << 16), "15.0.1");
        assert_eq!(version_name(450), "version 450");
    }
}
