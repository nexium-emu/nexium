use std::ops::Range;
use std::sync::Arc;
use memmap2::Mmap;

use crate::cnmt::{Cnmt, ContentType};
use crate::container::{Nsp, PartitionFs, Xci};
use crate::nca::{Nca, NcaContentType, NcaFsType};
use crate::nso::Nso;
use crate::npdm::Npdm;

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

pub struct LazyRomfs {
    pub mmap: Arc<Mmap>,
    pub range: Range<usize>,
}

impl LazyRomfs {
    pub fn len(&self) -> u64 {
        (self.range.end - self.range.start) as u64
    }

    pub fn is_empty(&self) -> bool {
        self.range.end == self.range.start
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[self.range.clone()]
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
    pub title_id: u64,
}

const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

impl Application {
    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {}", path, e))?);
        log::info!("container {} ({} bytes)", path, mmap.len());

        match detect(path, &mmap) {
            ContainerKind::Dxci => {
                let xci = Xci::parse(mmap.clone())?;
                Self::from_partition(mmap, xci.ncas())
            }
            ContainerKind::Dnsp => {
                let nsp = Nsp::parse(mmap.clone())?;
                Self::from_partition(mmap, nsp.ncas())
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

    fn resolve_program(
        mmap: Arc<Mmap>,
        ncas: &PartitionFs,
        parsed: &[(String, Nca)],
    ) -> Result<Nca, String> {
        if let Some((meta_name, meta)) = parsed.iter().find(|(n, nca)| {
            nca.content_type == NcaContentType::Meta || n.to_ascii_lowercase().ends_with(".cnmt.nca")
        }) {
            if let Some(section) = meta.section(NcaFsType::PartitionFs) {
                let pfs = PartitionFs::parse(mmap.clone(), section.fs_data_range.start)?;
                if let Some(cnmt_entry) = pfs.entries().iter().find(|e| e.name.ends_with(".cnmt")) {
                    let bytes = &mmap[pfs.entry_range(cnmt_entry)?];
                    let cnmt = Cnmt::parse(bytes)?;
                    log::info!("CNMT in {} title_id={:#018x} records={}", meta_name, cnmt.title_id, cnmt.records.len());
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
        let exefs_section = program
            .section(NcaFsType::PartitionFs)
            .ok_or("program NCA has no exefs (PartitionFs) section")?;
        let exefs = PartitionFs::parse(mmap.clone(), exefs_section.fs_data_range.start)?;
        log::info!(
            "exefs files: {:?}",
            exefs.entries().iter().map(|e| e.name.as_str()).collect::<Vec<_>>()
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
                    name, load_offset, image_size, nso.text.decompressed_size, nso.ro.decompressed_size,
                    nso.data.decompressed_size, nso.bss_size
                );
                modules.push(LoadedModule { name: name.to_string(), nso, load_offset });
                load_offset = load_offset.checked_add(page_align(image_size)).ok_or("code layout overflow")?;
            }
        }
        if modules.is_empty() {
            return Err("exefs has no loadable NSO modules".to_string());
        }
        let total_code_size = load_offset;

        let romfs = program.section(NcaFsType::RomFs).map(|s| LazyRomfs {
            mmap: mmap.clone(),
            range: s.fs_data_range.clone(),
        });
        if let Some(r) = &romfs {
            log::info!("romfs image {:#x}..{:#x} ({} bytes)", r.range.start, r.range.end, r.len());
        }

        let title_id = if npdm.title_id != 0 { npdm.title_id } else { program.program_id };
        log::info!(
            "application title_id={:#018x} addr_space={:?} stack={:#x} code_size={:#x} modules={}",
            title_id, npdm.address_space, npdm.main_stack_size, total_code_size, modules.len()
        );

        Ok(Self { mmap, modules, total_code_size, npdm, romfs, title_id })
    }
}
