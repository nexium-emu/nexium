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
        self.range.end == self.range.start
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.compressed.is_some() {
            &[]
        } else {
            &self.mmap[self.range.clone()]
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.compressed.is_some()
    }

    pub fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let available = self.len().saturating_sub(offset).min(size as u64) as usize;
        if available == 0 {
            return Ok(Vec::new());
        }
        let Some(storage) = &self.compressed else {
            let start = self.range.start + offset as usize;
            return Ok(self.mmap[start..start + available].to_vec());
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
                    let physical = self.range.start
                        + usize::try_from(entry.physical_offset)
                            .map_err(|_| "physical offset overflow")?
                        + within;
                    let end = physical.checked_add(take).ok_or("physical read overflow")?;
                    if end > self.range.end {
                        return Err(format!(
                            "compressed RomFS raw read {physical:#x}..{end:#x} out of range"
                        ));
                    }
                    output[done..done + take].copy_from_slice(&self.mmap[physical..end]);
                }
                1 => {}
                3 => {
                    let physical = self.range.start
                        + usize::try_from(entry.physical_offset)
                            .map_err(|_| "physical offset overflow")?;
                    let physical_end = physical
                        .checked_add(entry.physical_size as usize)
                        .ok_or("compressed read overflow")?;
                    if physical_end > self.range.end {
                        return Err(format!("compressed RomFS LZ4 read {physical:#x}..{physical_end:#x} out of range"));
                    }
                    let virtual_size = usize::try_from(entry_end - entry.virtual_offset)
                        .map_err(|_| "virtual extent too large")?;
                    let block = lz4_flex::block::decompress(
                        &self.mmap[physical..physical_end],
                        virtual_size,
                    )
                    .map_err(|err| {
                        format!("compressed RomFS LZ4 decode at {position:#x}: {err}")
                    })?;
                    output[done..done + take].copy_from_slice(&block[within..within + take]);
                }
                kind => return Err(format!("unsupported RomFS compression type {kind}")),
            }
            done += take;
        }
        Ok(output)
    }

    fn from_section(mmap: Arc<Mmap>, section: &NcaFsSection) -> Result<Self, String> {
        let compressed = section
            .compression
            .as_ref()
            .map(|info| CompressedRomfs::parse(&mmap, &section.fs_data_range, info))
            .transpose()?;
        Ok(Self {
            mmap,
            range: section.fs_data_range.clone(),
            compressed,
        })
    }
}

impl CompressedRomfs {
    fn parse(
        mmap: &Mmap,
        data: &Range<usize>,
        info: &crate::nca::NcaCompressionInfo,
    ) -> Result<Self, String> {
        const NODE_SIZE: usize = 0x4000;
        const ENTRY_SIZE: usize = 0x18;
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
        let table = data
            .start
            .checked_add(info.bucket_offset as usize)
            .ok_or("compression table overflow")?;
        let table_end = table
            .checked_add(info.bucket_size as usize)
            .ok_or("compression table overflow")?;
        if table_end > data.end || table + node_size + entry_sets * NODE_SIZE > table_end {
            return Err(format!(
                "compression table {table:#x}..{table_end:#x} is truncated"
            ));
        }
        let virtual_size = u64::from_le_bytes(mmap[table + 8..table + 16].try_into().unwrap());
        let entry_base = table + node_size;
        let mut entries = Vec::with_capacity(info.entry_count as usize);
        for set in 0..entry_sets {
            let set_base = entry_base + set * NODE_SIZE;
            let index = i32::from_le_bytes(mmap[set_base..set_base + 4].try_into().unwrap());
            let count = i32::from_le_bytes(mmap[set_base + 4..set_base + 8].try_into().unwrap());
            if index != set as i32 || count <= 0 || count as usize > per_node {
                return Err(format!(
                    "invalid compression entry set {set}: index={index} count={count}"
                ));
            }
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
        entries.truncate(info.entry_count as usize);
        if entries.is_empty()
            || entries[0].virtual_offset != 0
            || virtual_size == 0
            || entries
                .windows(2)
                .any(|pair| pair[0].virtual_offset >= pair[1].virtual_offset)
        {
            return Err("invalid compressed RomFS entry ordering".to_string());
        }
        log::info!(
            "compressed RomFS: {} extents, virtual_size={:#x}, physical_size={:#x}",
            entries.len(),
            virtual_size,
            data.end - data.start
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
}

const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

impl Application {
    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mmap =
            Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {}", path, e))?);
        log::info!("container {} ({} bytes)", path, mmap.len());

        match detect(path, &mmap) {
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
        }
    }

    fn from_partition(mmap: Arc<Mmap>, ncas: &PartitionFs) -> Result<Self, String> {
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

        let program = Self::resolve_program(mmap.clone(), ncas, &parsed)?;
        Self::from_program_nca(mmap, &program)
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
            let candidate = LazyRomfs {
                mmap: mmap.clone(),
                range: section.fs_data_range.clone(),
                compressed: None,
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

    fn resolve_program(
        mmap: Arc<Mmap>,
        ncas: &PartitionFs,
        parsed: &[(String, Nca)],
    ) -> Result<Nca, String> {
        if let Some((meta_name, meta)) = parsed.iter().find(|(n, nca)| {
            nca.content_type == NcaContentType::Meta
                || n.to_ascii_lowercase().ends_with(".cnmt.nca")
        }) {
            if let Some(section) = meta.section(NcaFsType::PartitionFs) {
                let pfs = PartitionFs::parse(mmap.clone(), section.fs_data_range.start)?;
                if let Some(cnmt_entry) = pfs.entries().iter().find(|e| e.name.ends_with(".cnmt")) {
                    let bytes = &mmap[pfs.entry_range(cnmt_entry)?];
                    let cnmt = Cnmt::parse(bytes)?;
                    log::info!(
                        "CNMT in {} title_id={:#018x} records={}",
                        meta_name,
                        cnmt.title_id,
                        cnmt.records.len()
                    );
                    if let Some(rec) = cnmt.find(ContentType::Program) {
                        let want = rec.nca_filename();
                        if let Some(entry) = ncas.find(&want) {
                            let range = ncas.entry_range(entry)?;
                            return Nca::parse(mmap.clone(), range.start);
                        }
                        log::warn!("CNMT program NCA {} not found in container", want);
                    }
                }
            }
        }

        parsed
            .iter()
            .find(|(_, nca)| nca.content_type == NcaContentType::Program)
            .map(|(_, nca)| Nca::parse(mmap.clone(), nca.nca_base))
            .unwrap_or_else(|| Err("no Program NCA found".to_string()))
    }

    fn from_program_nca(mmap: Arc<Mmap>, program: &Nca) -> Result<Self, String> {
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

        let romfs = program
            .section(NcaFsType::RomFs)
            .map(|section| LazyRomfs::from_section(mmap.clone(), section))
            .transpose()?;
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
        })
    }
}
