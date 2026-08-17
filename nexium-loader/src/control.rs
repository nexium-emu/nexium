use memmap2::Mmap;
use std::sync::Arc;

use crate::application::{detect, ContainerKind};
use crate::cnmt::{Cnmt, ContentType};
use crate::container::{Nsp, PartitionFs, Xci};
use crate::nca::{Nca, NcaContentType, NcaFsType};
use crate::nro::{parse_nacp, NroMetadata};
use crate::romfs::{romfs_dir_files, romfs_file, romfs_header};

pub fn read_container_metadata(path: &std::path::Path) -> Option<NroMetadata> {
    let file = std::fs::File::open(path).ok()?;
    let mmap = Arc::new(unsafe { Mmap::map(&file) }.ok()?);
    let path_str = path.to_string_lossy();

    let control = match detect(&path_str, &mmap) {
        ContainerKind::Dxci => {
            let xci = Xci::parse(mmap.clone()).ok()?;
            resolve_control(&mmap, xci.ncas())?
        }
        ContainerKind::Dnsp => {
            let nsp = Nsp::parse(mmap.clone()).ok()?;
            resolve_control(&mmap, nsp.ncas())?
        }
        _ => return None,
    };

    let section = control.section(NcaFsType::RomFs)?;
    let romfs = mmap.get(section.fs_data_range.clone())?;

    let (title, author, version) = romfs_file(romfs, "/control.nacp")
        .filter(|n| n.len() >= 0x300)
        .map(parse_nacp)
        .unwrap_or_default();
    let icon_jpeg = find_icon(romfs);

    if title.is_empty() && author.is_empty() && icon_jpeg.is_none() {
        return None;
    }
    Some(NroMetadata {
        title,
        author,
        version,
        icon_jpeg,
    })
}

fn resolve_control(mmap: &Arc<Mmap>, ncas: &PartitionFs) -> Option<Nca> {
    let mut parsed: Vec<(String, Nca)> = Vec::new();
    for e in ncas.entries() {
        if !e.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let Ok(range) = ncas.entry_range(e) else {
            continue;
        };
        if let Ok(nca) = Nca::parse(mmap.clone(), range.start) {
            parsed.push((e.name.clone(), nca));
        }
    }

    if let Some((_, meta)) = parsed.iter().find(|(n, nca)| {
        nca.content_type == NcaContentType::Meta || n.to_ascii_lowercase().ends_with(".cnmt.nca")
    }) {
        if let Some(section) = meta.section(NcaFsType::PartitionFs) {
            if let Ok(pfs) = PartitionFs::parse(mmap.clone(), section.fs_data_range.start) {
                if let Some(cnmt_entry) = pfs.entries().iter().find(|e| e.name.ends_with(".cnmt")) {
                    if let Ok(range) = pfs.entry_range(cnmt_entry) {
                        if let Ok(cnmt) = Cnmt::parse(&mmap[range]) {
                            if let Some(rec) = cnmt.find(ContentType::Control) {
                                let want = rec.nca_filename();
                                if let Some(entry) = ncas.find(&want) {
                                    if let Ok(r) = ncas.entry_range(entry) {
                                        if let Ok(nca) = Nca::parse(mmap.clone(), r.start) {
                                            return Some(nca);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    parsed
        .into_iter()
        .find(|(_, nca)| nca.content_type == NcaContentType::Control)
        .map(|(_, nca)| nca)
}

fn find_icon(romfs: &[u8]) -> Option<Vec<u8>> {
    let hdr = romfs_header(romfs)?;
    for (name, off, size) in romfs_dir_files(romfs, hdr, 0) {
        let lower = name.to_ascii_lowercase();
        if lower.starts_with("icon") && lower.ends_with(".dat") && size > 0 {
            return romfs.get(off..off.checked_add(size)?).map(|b| b.to_vec());
        }
    }
    None
}
