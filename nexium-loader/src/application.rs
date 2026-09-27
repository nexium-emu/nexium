use memmap2::Mmap;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::cnmt::{Cnmt, ContentType};
use crate::container::{Nsp, PartitionFs, Xci};
use crate::nca::{Nca, NcaContentType, NcaFsSection, NcaFsType};
use crate::npdm::Npdm;
use crate::nso::Nso;

const PAGE: u64 = 0x1000;

fn page_align(v: u64) -> u64 {
    (v + (PAGE - 1)) & !(PAGE - 1)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContainerKind {
    Nro,
    Dxci,
    Dnsp,
    Nca,
    Unknown,
}

pub fn detect(path: &str, mmap: &Mmap) -> ContainerKind {
    let buf = &mmap[..];
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let nro0 = crate::bin_read::u32at(buf, 0).ok() == Some(0x304F524E)
        || crate::bin_read::u32at(buf, 16).ok() == Some(0x304F524E);
    let dxci = crate::bin_read::u32at(buf, 0x100).ok() == Some(crate::container::DXCI_MAGIC);
    let pfs0 = crate::bin_read::u32at(buf, 0).ok() == Some(crate::container::PFS0_MAGIC);
    let dnca = crate::bin_read::u32at(buf, 0x200).ok() == Some(crate::nca::DNCA_MAGIC);

    match ext.as_str() {
        "nro" if nro0 => return ContainerKind::Nro,
        "dxci" if dxci => return ContainerKind::Dxci,
        "dnsp" if pfs0 => return ContainerKind::Dnsp,
        "dnca" if dnca => return ContainerKind::Nca,
        _ => {}
    }
    if nro0 {
        ContainerKind::Nro
    } else if dxci {
        ContainerKind::Dxci
    } else if pfs0 {
        ContainerKind::Dnsp
    } else if dnca {
        ContainerKind::Nca
    } else {
        ContainerKind::Unknown
    }
}

#[derive(Clone)]
pub struct LazyRomfs {
    pub mmap: Arc<Mmap>,
    pub range: Range<usize>,
    compressed: Option<CompressedRomfs>,
    patched: Option<Arc<crate::bktr::PatchedRomfs>>,
}

#[derive(Clone, Debug)]
struct CompressedRomfs {
    entries: Vec<CompressionEntry>,
    virtual_size: u64,
}

#[derive(Clone, Copy, Debug)]
struct CompressionEntry {
    virtual_offset: u64,
    physical_offset: u64,
    compression_type: u8,
    physical_size: u32,
}

impl LazyRomfs {
    pub fn len(&self) -> u64 {
        self.compressed
            .as_ref()
            .map(|storage| storage.virtual_size)
            .unwrap_or((self.range.end - self.range.start) as u64)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.compressed.is_some() || self.patched.is_some() {
            &[]
        } else {
            &self.mmap[self.range.clone()]
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.compressed.is_some() || self.patched.is_some()
    }

    pub fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let available = self.len().saturating_sub(offset).min(size as u64) as usize;
        if available == 0 {
            return Ok(Vec::new());
        }
        let Some(storage) = &self.compressed else {
            return self.read_raw(offset, available);
        };

        let mut output = vec![0; available];
        let mut done = 0usize;
        while done < available {
            let position = offset + done as u64;
            let index = storage
                .entries
                .partition_point(|entry| entry.virtual_offset <= position)
                .checked_sub(1)
                .ok_or_else(|| format!("compressed RomFS has no entry for {position:#x}"))?;
            let entry = storage.entries[index];
            let entry_end = storage
                .entries
                .get(index + 1)
                .map(|next| next.virtual_offset)
                .unwrap_or(storage.virtual_size);
            if entry_end <= position {
                return Err(format!("invalid compressed RomFS extent at {position:#x}"));
            }
            let within = (position - entry.virtual_offset) as usize;
            let take = available
                .saturating_sub(done)
                .min((entry_end - position) as usize);
            match entry.compression_type {
                0 => {
                    let physical = entry.physical_offset.checked_add(within as u64)
                        .ok_or("physical read overflow")?;
                    output[done..done + take].copy_from_slice(&self.read_raw(physical, take)?);
                }
                1 => {}
                3 => {
                    let compressed = self.read_raw(entry.physical_offset, entry.physical_size as usize)?;
                    let virtual_size = usize::try_from(entry_end - entry.virtual_offset)
                        .map_err(|_| "virtual extent too large")?;
                    let block = lz4_flex::block::decompress(
                        &compressed,
                        virtual_size,
                    )
                    .map_err(|err| {
                        format!("compressed RomFS LZ4 decode at {position:#x}: {err}")
                    })?;
                    if block.len() != virtual_size {
                        return Err("compressed RomFS block has an unexpected decoded size".into());
                    }
                    output[done..done + take].copy_from_slice(&block[within..within + take]);
                }
                kind => return Err(format!("unsupported RomFS compression type {kind}")),
            }
            done += take;
        }
        Ok(output)
    }

    fn read_raw(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        if offset.checked_add(size as u64).is_none_or(|end| end > self.range.len() as u64) {
            return Err("RomFS physical read exceeds its storage".into());
        }
        if let Some(patched) = &self.patched {
            return patched.read(offset, size);
        }
        let start = self.range.start.checked_add(usize::try_from(offset).map_err(|_| "RomFS offset overflow")?)
            .ok_or("RomFS offset overflow")?;
        let end = start.checked_add(size).ok_or("RomFS read overflow")?;
        self.mmap.get(start..end).map(|bytes| bytes.to_vec()).ok_or_else(|| "RomFS storage is truncated".into())
    }

    pub fn from_range(mmap: Arc<Mmap>, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > mmap.len() {
            return Err("RomFS range exceeds its mapped file".into());
        }
        Ok(Self { mmap, range, compressed: None, patched: None })
    }

    pub(crate) fn from_section(mmap: Arc<Mmap>, section: &NcaFsSection) -> Result<Self, String> {
        if section.patch.is_some() {
            return Err("This update requires its base game. Install it from Updates & DLC.".into());
        }
        if section.sparse {
            return Err("Sparse RomFS storage is not supported".into());
        }
        let mut storage = Self::from_range(mmap, section.fs_data_range.clone())?;
        if let Some(info) = &section.compression {
            storage.compressed = Some(CompressedRomfs::parse(&storage, info)?);
        }
        Ok(storage)
    }

    pub(crate) fn from_patched_section(
        base_mmap: Arc<Mmap>,
        base_section: &NcaFsSection,
        patch_mmap: Arc<Mmap>,
        patch_section: &NcaFsSection,
    ) -> Result<Self, String> {
        if patch_section.patch.is_none() {
            return Self::from_section(patch_mmap, patch_section);
        }
        let patched = crate::bktr::PatchedRomfs::new(base_mmap, base_section, patch_mmap.clone(), patch_section)?;
        let mut storage = Self {
            mmap: patch_mmap,
            range: patch_section.fs_data_range.clone(),
            compressed: None,
            patched: Some(Arc::new(patched)),
        };
        if let Some(info) = &patch_section.compression {
            storage.compressed = Some(CompressedRomfs::parse(&storage, info)?);
        }
        Ok(storage)
    }
}

impl CompressedRomfs {
    fn parse(
        storage: &LazyRomfs,
        info: &crate::nca::NcaCompressionInfo,
    ) -> Result<Self, String> {
        const NODE_SIZE: usize = 0x4000;
        const ENTRY_SIZE: usize = 0x18;
        if info.entry_count == 0 || info.entry_count > i32::MAX as u32 {
            return Err("compression table has no entries".into());
        }
        let per_node = (NODE_SIZE - 0x10) / ENTRY_SIZE;
        let entry_sets = (info.entry_count as usize + per_node - 1) / per_node;
        let offsets_per_node = (NODE_SIZE - 0x10) / 8;
        let l2_nodes = if entry_sets <= offsets_per_node {
            0
        } else {
            let initial = (entry_sets + offsets_per_node - 1) / offsets_per_node;
            (entry_sets - (offsets_per_node - (initial - 1)) + offsets_per_node - 1)
                / offsets_per_node
        };
        let node_size = (1 + l2_nodes) * NODE_SIZE;
        let table_size = usize::try_from(info.bucket_size).map_err(|_| "compression table size overflow")?;
        if node_size.checked_add(entry_sets.checked_mul(NODE_SIZE).ok_or("compression table overflow")?)
            .is_none_or(|required| required > table_size)
        {
            return Err("compression table is truncated".into());
        }
        let mmap = storage.read_raw(info.bucket_offset, table_size)?;
        let table = 0;
        let virtual_size = u64::from_le_bytes(mmap[table + 8..table + 16].try_into().unwrap());
        let root_index = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        let root_count = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        if root_index != 0 || root_count == 0 || root_count > offsets_per_node
            || virtual_size == 0 || virtual_size > i64::MAX as u64
        {
            return Err("invalid compression root node".into());
        }
        let entry_base = table + node_size;
        let mut previous_set_end = 0;
        let mut entries = Vec::with_capacity(info.entry_count as usize);
        for set in 0..entry_sets {
            let set_base = entry_base + set * NODE_SIZE;
            let index = i32::from_le_bytes(mmap[set_base..set_base + 4].try_into().unwrap());
            let count = i32::from_le_bytes(mmap[set_base + 4..set_base + 8].try_into().unwrap());
            let remaining = info.entry_count as usize - entries.len();
            if index != set as i32 || count <= 0 || count as usize != remaining.min(per_node) {
                return Err(format!(
                    "invalid compression entry set {set}: index={index} count={count}"
                ));
            }
            let first_offset = u64::from_le_bytes(mmap[set_base + 0x10..set_base + 0x18].try_into().unwrap());
            let set_end = u64::from_le_bytes(mmap[set_base + 8..set_base + 16].try_into().unwrap());
            if first_offset != previous_set_end || set_end <= first_offset || set_end > virtual_size {
                return Err(format!("invalid compression entry set boundary {set}"));
            }
            previous_set_end = set_end;
            for i in 0..count as usize {
                let at = set_base + 0x10 + i * ENTRY_SIZE;
                entries.push(CompressionEntry {
                    virtual_offset: u64::from_le_bytes(mmap[at..at + 8].try_into().unwrap()),
                    physical_offset: u64::from_le_bytes(mmap[at + 8..at + 16].try_into().unwrap()),
                    compression_type: mmap[at + 16],
                    physical_size: u32::from_le_bytes(mmap[at + 20..at + 24].try_into().unwrap()),
                });
            }
        }
        if previous_set_end != virtual_size {
            return Err("compression entries do not cover virtual storage".into());
        }
        if entries.is_empty()
            || entries[0].virtual_offset != 0
            || virtual_size == 0
            || entries
                .windows(2)
                .any(|pair| pair[0].virtual_offset >= pair[1].virtual_offset)
        {
            return Err("invalid compressed RomFS entry ordering".to_string());
        }
        for (index, entry) in entries.iter().enumerate() {
            let end = entries.get(index + 1).map_or(virtual_size, |next| next.virtual_offset);
            if end <= entry.virtual_offset || end > virtual_size {
                return Err("compressed RomFS extent exceeds virtual storage".into());
            }
            let physical_size = match entry.compression_type {
                0 => end - entry.virtual_offset,
                1 => continue,
                3 if entry.physical_size != 0 => {
                    if end - entry.virtual_offset > u64::from(entry.physical_size) * 255 + 19 {
                        return Err("compressed RomFS decoded extent is too large".into());
                    }
                    u64::from(entry.physical_size)
                }
                kind => return Err(format!("invalid RomFS compression type or size {kind}")),
            };
            if entry.physical_offset.checked_add(physical_size)
                .is_none_or(|end| end > info.bucket_offset)
            {
                return Err("compressed RomFS extent exceeds its source data".into());
            }
        }
        log::info!(
            "compressed RomFS: {} extents, virtual_size={:#x}, physical_size={:#x}",
            entries.len(),
            virtual_size,
            storage.range.len()
        );
        Ok(Self {
            entries,
            virtual_size,
        })
    }
}

pub struct LoadedModule {
    pub name: String,
    pub nso: Nso,
    pub load_offset: u64,
}

pub struct Application {
    pub mmap: Arc<Mmap>,
    pub modules: Vec<LoadedModule>,
    pub total_code_size: u64,
    pub npdm: Npdm,
    pub romfs: Option<LazyRomfs>,
    pub system_romfs: HashMap<u64, LazyRomfs>,
    pub title_id: u64,
    pub content_version: u32,
    pub display_version: String,
    pub patch_romfs: Option<LazyRomfs>,
    pub add_on_content: std::collections::BTreeMap<u64, Option<LazyRomfs>>,
}

pub(crate) const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

impl Application {
    pub fn load_with_content(path: &str, content_root: &std::path::Path) -> Result<Self, String> {
        use crate::content::{ContentKind, Package};
        let mut application = Self::load(path)?;
        let base_title_id = application.title_id;
        let installed = crate::content::list_game_content(content_root, base_title_id)?;
        if let Some(entry) = installed.entries.iter().find(|entry| entry.enabled && entry.kind == ContentKind::Update) {
            let package = Package::open(&entry.path)?;
            let metadata = package.metadata_for(entry)?;
            let record = metadata.find(ContentType::Program).ok_or("Selected update has no Program content")?;
            let updated_program = package.nca(record)?;
            if updated_program.content_type != NcaContentType::Program {
                return Err("Selected update references a non-Program NCA".into());
            }
            let original_program = Self::program_for_mmap(path, application.mmap.clone())?;
            let romfs = match updated_program.section(NcaFsType::RomFs) {
                Some(updated) => match original_program.section(NcaFsType::RomFs) {
                    Some(original) => Some(LazyRomfs::from_patched_section(
                        application.mmap.clone(), original, package.mmap.clone(), updated,
                    )?),
                    None => Some(LazyRomfs::from_section(package.mmap.clone(), updated)?),
                },
                None => application.romfs.clone(),
            };
            let mut updated = Self::from_program_nca_with_romfs(package.mmap.clone(), &updated_program, romfs)?;
            if updated.npdm.title_id != 0 && updated.npdm.title_id != base_title_id {
                return Err(format!("Update program belongs to {:016X}, not {base_title_id:016X}", updated.npdm.title_id));
            }
            updated.title_id = base_title_id;
            updated.content_version = metadata.version;
            updated.display_version = if entry.display_version.is_empty() {
                application.display_version.clone()
            } else {
                entry.display_version.clone()
            };
            updated.patch_romfs = updated.romfs.clone();
            updated.system_romfs = application.system_romfs;
            application = updated;
            log::info!("Applied update {:016X} version {} from {}", metadata.title_id, metadata.version, entry.path.display());
        }
        for entry in installed.entries.iter().filter(|entry| entry.enabled && entry.kind == ContentKind::Dlc) {
            let package = Package::open(&entry.path)?;
            let metadata = package.metadata_for(entry)?;
            if metadata.required_application_version > application.content_version {
                return Err(format!("DLC {:016X} requires game version {} or newer; selected version is {}", metadata.title_id, metadata.required_application_version, application.content_version));
            }
            let romfs = metadata.find(ContentType::Data).map(|record| {
                let nca = package.nca(record)?;
                if !matches!(nca.content_type, NcaContentType::Data | NcaContentType::PublicData) {
                    return Err("DLC references a non-Data NCA".into());
                }
                let section = nca.section(NcaFsType::RomFs).ok_or("DLC Data content has no RomFS")?;
                LazyRomfs::from_section(package.mmap.clone(), section)
            }).transpose()?;
            if application.add_on_content.insert(metadata.title_id, romfs).is_some() {
                return Err(format!("Multiple versions of DLC {:016X} are enabled", metadata.title_id));
            }
            log::info!("Mounted DLC {:016X} version {} from {}", metadata.title_id, metadata.version, entry.path.display());
        }
        Ok(application)
    }

    fn program_for_mmap(path: &str, mmap: Arc<Mmap>) -> Result<Nca, String> {
        match detect(path, &mmap) {
            ContainerKind::Dxci => {
                let xci = Xci::parse(mmap.clone())?;
                Self::program_from_partition(mmap, xci.ncas())
            }
            ContainerKind::Dnsp => {
                let nsp = Nsp::parse(mmap.clone())?;
                Self::program_from_partition(mmap, nsp.ncas())
            }
            ContainerKind::Nca => Nca::parse(mmap, 0),
            _ => Err("The base game is not an application container".into()),
        }
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mmap =
            Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {}", path, e))?);
        log::info!("container {} ({} bytes)", path, mmap.len());

        let mut application = match detect(path, &mmap) {
            ContainerKind::Dxci => {
                let xci = Xci::parse(mmap.clone())?;
                let mut system_romfs = Self::collect_system_romfs(&mmap, xci.ncas());
                if let Some(update) = xci.update_ncas() {
                    system_romfs.extend(Self::collect_system_romfs(&mmap, update));
                }
                log::info!("indexed {} bundled system archive(s)", system_romfs.len());
                let mut application = Self::from_partition(mmap, xci.ncas())?;
                application.system_romfs = system_romfs;
                Ok(application)
            }
            ContainerKind::Dnsp => {
                let nsp = Nsp::parse(mmap.clone())?;
                let system_romfs = Self::collect_system_romfs(&mmap, nsp.ncas());
                let mut application = Self::from_partition(mmap, nsp.ncas())?;
                application.system_romfs = system_romfs;
                Ok(application)
            }
            ContainerKind::Nca => {
                let nca = Nca::parse(mmap.clone(), 0)?;
                Self::from_program_nca(mmap, &nca)
            }
            ContainerKind::Nro => Err("NRO is not an application container".to_string()),
            ContainerKind::Unknown => Err(format!("unrecognized container format: {}", path)),
        }?;
        application.display_version = crate::read_container_metadata(std::path::Path::new(path))
            .map(|metadata| metadata.version).unwrap_or_default();
        Ok(application)
    }

    fn from_partition(mmap: Arc<Mmap>, ncas: &PartitionFs) -> Result<Self, String> {
        let program = Self::program_from_partition(mmap.clone(), ncas)?;
        let metadata = Self::metadata_from_partition(&mmap, ncas)?;
        let mut application = Self::from_program_nca(mmap, &program)?;
        if let Some(metadata) = metadata.iter().find(|metadata| {
            metadata.meta_type == crate::cnmt::ContentMetaType::Application
                && metadata.title_id == application.title_id
        }) {
            application.content_version = metadata.version;
        }
        Ok(application)
    }

    fn program_from_partition(mmap: Arc<Mmap>, ncas: &PartitionFs) -> Result<Nca, String> {
        let mut parsed: Vec<(String, Nca)> = Vec::new();
        for e in ncas.entries() {
            if !e.name.to_ascii_lowercase().ends_with(".nca") {
                continue;
            }
            let range = ncas.entry_range(e)?;
            match Nca::parse(mmap.clone(), range.start) {
                Ok(nca) => parsed.push((e.name.clone(), nca)),
                Err(err) => log::warn!("skipping NCA {}: {}", e.name, err),
            }
        }
        if parsed.is_empty() {
            return Err("no parseable NCAs in container".to_string());
        }

        Self::resolve_program(mmap, ncas, &parsed)
    }

    fn collect_system_romfs(mmap: &Arc<Mmap>, ncas: &PartitionFs) -> HashMap<u64, LazyRomfs> {
        let mut archives = HashMap::new();
        for entry in ncas.entries() {
            if !entry.name.to_ascii_lowercase().ends_with(".nca") {
                continue;
            }
            let Ok(range) = ncas.entry_range(entry) else {
                continue;
            };
            let Ok(nca) = Nca::parse(mmap.clone(), range.start) else {
                continue;
            };
            if !matches!(
                nca.content_type,
                NcaContentType::Data | NcaContentType::PublicData
            ) {
                continue;
            }
            let Some(section) = nca.section(NcaFsType::RomFs) else {
                continue;
            };
            if section.fs_data_range.start < range.start
                || section.fs_data_range.end > range.end
                || section.fs_data_range.start > section.fs_data_range.end
            {
                log::warn!(
                    "skipping system archive {:#018x} from {}: RomFS range {:#x}..{:#x} lies outside NCA {:#x}..{:#x}",
                    nca.program_id,
                    entry.name,
                    section.fs_data_range.start,
                    section.fs_data_range.end,
                    range.start,
                    range.end
                );
                continue;
            }
            let candidate = match LazyRomfs::from_section(mmap.clone(), section) {
                Ok(storage) => storage,
                Err(error) => {
                    log::warn!("Skipping system archive {:016X}: {error}", nca.program_id);
                    continue;
                }
            };
            let should_replace = archives
                .get(&nca.program_id)
                .map(|current: &LazyRomfs| candidate.len() > current.len())
                .unwrap_or(true);
            if should_replace {
                log::debug!(
                    "system archive {:#018x} from {} romfs={} bytes",
                    nca.program_id,
                    entry.name,
                    candidate.len()
                );
                archives.insert(nca.program_id, candidate);
            }
        }
        archives
    }

    fn metadata_from_partition(mmap: &Arc<Mmap>, ncas: &PartitionFs) -> Result<Vec<Cnmt>, String> {
        let mut metadata = Vec::new();
        for entry in ncas.entries().iter().filter(|entry| entry.name.to_ascii_lowercase().ends_with(".nca")) {
            let range = ncas.entry_range(entry)?;
            let Ok(nca) = Nca::parse(mmap.clone(), range.start) else { continue };
            if nca.content_type != NcaContentType::Meta { continue }
            let Some(section) = nca.section(NcaFsType::PartitionFs) else { continue };
            let pfs = PartitionFs::parse(mmap.clone(), section.fs_data_range.start)?;
            for entry in pfs.entries().iter().filter(|entry| entry.name.ends_with(".cnmt")) {
                let bytes = pfs.entry_range(entry)?;
                if bytes.end > section.fs_data_range.end || bytes.len() as u64 != entry.size {
                    return Err(format!("Truncated metadata {}", entry.name));
                }
                metadata.push(Cnmt::parse(&mmap[bytes])?);
            }
        }
        Ok(metadata)
    }

    fn resolve_program(
        mmap: Arc<Mmap>,
        ncas: &PartitionFs,
        parsed: &[(String, Nca)],
    ) -> Result<Nca, String> {
        let mut metadata = Self::metadata_from_partition(&mmap, ncas)?;
        metadata.sort_by_key(|metadata| match metadata.meta_type {
            crate::cnmt::ContentMetaType::Application => 0,
            crate::cnmt::ContentMetaType::Patch => 1,
            _ => 2,
        });
        for cnmt in metadata {
            if let Some(record) = cnmt.find(ContentType::Program) {
                let wanted = record.nca_filename();
                let entry = ncas.entries().iter().find(|entry| entry.name.eq_ignore_ascii_case(&wanted))
                    .ok_or_else(|| format!("CNMT Program content {wanted} is missing"))?;
                let range = ncas.entry_range(entry)?;
                return Nca::parse(mmap, range.start);
            }
        }
        parsed.iter().find(|(_, nca)| nca.content_type == NcaContentType::Program)
            .map(|(_, nca)| Nca::parse(mmap.clone(), nca.nca_base))
            .unwrap_or_else(|| Err("no Program NCA found".to_string()))
    }

    fn from_program_nca(mmap: Arc<Mmap>, program: &Nca) -> Result<Self, String> {
        let romfs = program.section(NcaFsType::RomFs)
            .map(|section| LazyRomfs::from_section(mmap.clone(), section)).transpose()?;
        Self::from_program_nca_with_romfs(mmap, program, romfs)
    }

    fn from_program_nca_with_romfs(mmap: Arc<Mmap>, program: &Nca, romfs: Option<LazyRomfs>) -> Result<Self, String> {
        log::info!("selected program NCA base={:#x}", program.nca_base);
        let exefs_section = program
            .section(NcaFsType::PartitionFs)
            .ok_or("program NCA has no exefs (PartitionFs) section")?;
        let exefs = PartitionFs::parse(mmap.clone(), exefs_section.fs_data_range.start)?;
        log::info!(
            "exefs files: {:?}",
            exefs
                .entries()
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
        );

        let npdm = match exefs.find("main.npdm") {
            Some(e) => {
                let bytes = &mmap[exefs.entry_range(e)?];
                Npdm::parse(bytes).unwrap_or_else(|err| {
                    log::warn!("main.npdm parse failed ({}); using homebrew defaults", err);
                    Npdm::default_for_homebrew()
                })
            }
            None => {
                log::warn!("no main.npdm in exefs; using homebrew defaults");
                Npdm::default_for_homebrew()
            }
        };

        let mut modules = Vec::new();
        let mut load_offset: u64 = 0;
        for name in MODULE_ORDER {
            if let Some(entry) = exefs.find(name) {
                let region = &mmap[exefs.entry_range(entry)?];
                let nso = Nso::parse(region).map_err(|e| format!("{}: {}", name, e))?;
                let image_size = nso.image_size as u64;
                log::info!(
                    "module {} @ +{:#x} image={:#x} text={:#x} ro={:#x} data={:#x} bss={:#x}",
                    name,
                    load_offset,
                    image_size,
                    nso.text.decompressed_size,
                    nso.ro.decompressed_size,
                    nso.data.decompressed_size,
                    nso.bss_size
                );
                modules.push(LoadedModule {
                    name: name.to_string(),
                    nso,
                    load_offset,
                });
                load_offset = load_offset
                    .checked_add(page_align(image_size))
                    .ok_or("code layout overflow")?;
            }
        }
        if modules.is_empty() {
            return Err("exefs has no loadable NSO modules".to_string());
        }
        let total_code_size = load_offset;

        if let Some(r) = &romfs {
            log::info!(
                "romfs image {:#x}..{:#x} ({} bytes)",
                r.range.start,
                r.range.end,
                r.len()
            );
        }

        let title_id = if npdm.title_id != 0 {
            npdm.title_id
        } else {
            program.program_id
        };
        log::info!(
            "application title_id={:#018x} addr_space={:?} stack={:#x} code_size={:#x} modules={}",
            title_id,
            npdm.address_space,
            npdm.main_stack_size,
            total_code_size,
            modules.len()
        );

        Ok(Self {
            mmap,
            modules,
            total_code_size,
            npdm,
            romfs,
            system_romfs: HashMap::new(),
            title_id,
            content_version: 0,
            display_version: String::new(),
            patch_romfs: None,
            add_on_content: Default::default(),
        })
    }
}


pub fn read_application_title_id(path: &std::path::Path) -> Result<Option<u64>, String> {
    let file = std::fs::File::open(path).map_err(|error| format!("Open {}: {error}", path.display()))?;
    let mmap = Arc::new(unsafe { Mmap::map(&file) }
        .map_err(|error| format!("Map {}: {error}", path.display()))?);
    let program = match detect(&path.to_string_lossy(), &mmap) {
        ContainerKind::Nro => return Ok(None),
        ContainerKind::Dxci => {
            let xci = Xci::parse(mmap.clone())?;
            Application::program_from_partition(mmap.clone(), xci.ncas())?
        }
        ContainerKind::Dnsp => {
            let nsp = Nsp::parse(mmap.clone())?;
            Application::program_from_partition(mmap.clone(), nsp.ncas())?
        }
        ContainerKind::Nca => Nca::parse(mmap.clone(), 0)?,
        ContainerKind::Unknown => return Err(format!("Unrecognized application format: {}", path.display())),
    };
    let mut title_id = program.program_id;
    if let Some(section) = program.section(NcaFsType::PartitionFs) {
        let exefs = PartitionFs::parse(mmap.clone(), section.fs_data_range.start)?;
        if let Some(entry) = exefs.find("main.npdm") {
            if let Ok(npdm) = Npdm::parse(&mmap[exefs.entry_range(entry)?]) {
                if npdm.title_id != 0 { title_id = npdm.title_id; }
            }
        }
    }
    Ok(Some(title_id))
}
