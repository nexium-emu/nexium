use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::application::LazyRomfs;

const INVALID: u32 = 0xffff_ffff;
const OPEN_HANDLE_CAP: usize = 64;

#[derive(Clone)]
pub enum AppRomfs {
    Plain(LazyRomfs),
    Layered(Arc<LayeredRomfs>),
}

impl AppRomfs {
    pub fn len(&self) -> u64 {
        match self {
            AppRomfs::Plain(romfs) => romfs.len(),
            AppRomfs::Layered(romfs) => romfs.len(),
        }
    }

    pub fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        match self {
            AppRomfs::Plain(romfs) => romfs.read(offset, size),
            AppRomfs::Layered(romfs) => romfs.read(offset, size),
        }
    }
}

enum ExtentSource {
    Base { offset: u64 },
    Host { index: usize },
}

struct Extent {
    start: u64,
    len: u64,
    source: ExtentSource,
}

pub struct LayeredRomfs {
    meta: Vec<u8>,
    extents: Vec<Extent>,
    total_len: u64,
    base: Option<LazyRomfs>,
    host_paths: Vec<PathBuf>,
    handles: Mutex<HandleCache>,
    pub base_file_count: usize,
    pub overlay_file_count: usize,
    pub replaced_file_count: usize,
}

struct HandleCache {
    open: HashMap<usize, Arc<File>>,
    order: VecDeque<usize>,
}

enum FileSource {
    Base { data_offset: u64, size: u64 },
    Host { path: PathBuf, size: u64 },
}

#[derive(Default)]
struct DirNode {
    dirs: BTreeMap<Vec<u8>, DirNode>,
    files: BTreeMap<Vec<u8>, FileSource>,
}

fn align4(value: usize) -> usize {
    (value + 3) & !3
}

fn align16(value: u64) -> u64 {
    (value + 15) & !15
}

fn u32le(buf: &[u8], off: usize) -> Result<u32, String> {
    buf.get(off..off + 4)
        .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .ok_or_else(|| format!("romfs metadata read out of range at {:#x}", off))
}

fn u64le(buf: &[u8], off: usize) -> Result<u64, String> {
    buf.get(off..off + 8)
        .map(|v| u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]))
        .ok_or_else(|| format!("romfs metadata read out of range at {:#x}", off))
}

fn romfs_path_hash(parent: u32, name: &[u8]) -> u32 {
    let mut hash = parent ^ 123_456_789;
    for byte in name {
        hash = hash.rotate_right(5) ^ u32::from(*byte);
    }
    hash
}

fn hash_table_entry_count(entry_count: usize) -> usize {
    if entry_count < 3 {
        return 3;
    }
    if entry_count < 19 {
        return entry_count | 1;
    }
    let mut count = entry_count;
    while [2, 3, 5, 7, 11, 13, 17]
        .into_iter()
        .any(|divisor| count % divisor == 0)
    {
        count += 1;
    }
    count
}

fn parse_base_tree(base: &LazyRomfs) -> Result<DirNode, String> {
    let header = base.read(0, 0x50)?;
    if header.len() < 0x50 || u64le(&header, 0)? != 0x50 {
        return Err("base romfs has no valid header".to_string());
    }
    let dir_table_off = u64le(&header, 0x18)?;
    let dir_table_len = u64le(&header, 0x20)? as usize;
    let file_table_off = u64le(&header, 0x38)?;
    let file_table_len = u64le(&header, 0x40)? as usize;
    let file_data_off = u64le(&header, 0x48)?;
    let dir_table = base.read(dir_table_off, dir_table_len)?;
    let file_table = base.read(file_table_off, file_table_len)?;
    if dir_table.len() != dir_table_len || file_table.len() != file_table_len {
        return Err("base romfs metadata tables truncated".to_string());
    }

    fn walk(
        dir_table: &[u8],
        file_table: &[u8],
        dir_off: u32,
        file_data_off: u64,
        base_len: u64,
    ) -> Result<DirNode, String> {
        let mut node = DirNode::default();
        let dir_abs = dir_off as usize;
        let mut child = u32le(dir_table, dir_abs + 0x08)?;
        while child != INVALID {
            let abs = child as usize;
            let name_len = u32le(dir_table, abs + 0x14)? as usize;
            let name = dir_table
                .get(abs + 0x18..abs + 0x18 + name_len)
                .ok_or("base romfs dir name out of range")?
                .to_vec();
            node.dirs.insert(
                name,
                walk(dir_table, file_table, child, file_data_off, base_len)?,
            );
            child = u32le(dir_table, abs + 0x04)?;
        }
        let mut file = u32le(dir_table, dir_abs + 0x0c)?;
        while file != INVALID {
            let abs = file as usize;
            let data_offset = file_data_off.checked_add(u64le(file_table, abs + 0x08)?)
                .ok_or("base romfs file data offset overflow")?;
            let size = u64le(file_table, abs + 0x10)?;
            if data_offset.checked_add(size).is_none_or(|end| end > base_len) {
                return Err("base romfs file data out of range".to_string());
            }
            let name_len = u32le(file_table, abs + 0x1c)? as usize;
            let name = file_table
                .get(abs + 0x20..abs + 0x20 + name_len)
                .ok_or("base romfs file name out of range")?
                .to_vec();
            node.files.insert(name, FileSource::Base { data_offset, size });
            file = u32le(file_table, abs + 0x04)?;
        }
        Ok(node)
    }

    walk(&dir_table, &file_table, 0, file_data_off, base.len())
}

fn overlay_into(node: &mut DirNode, dir: &Path, stats: &mut (usize, usize)) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("read overlay dir {}: {}", dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read overlay dir {}: {}", dir.display(), e))?;
        let meta = entry
            .metadata()
            .map_err(|e| format!("stat {}: {}", entry.path().display(), e))?;
        let name = entry.file_name();
        let name_bytes = name.to_string_lossy().as_bytes().to_vec();
        if name_bytes.is_empty() {
            continue;
        }
        if meta.is_dir() {
            node.files.remove(&name_bytes);
            overlay_into(
                node.dirs.entry(name_bytes).or_default(),
                &entry.path(),
                stats,
            )?;
        } else if meta.is_file() {
            node.dirs.remove(&name_bytes);
            stats.0 += 1;
            if node
                .files
                .insert(
                    name_bytes,
                    FileSource::Host {
                        path: entry.path(),
                        size: meta.len(),
                    },
                )
                .is_some()
            {
                stats.1 += 1;
            }
        }
    }
    Ok(())
}

struct FlatDir {
    entry_offset: u32,
    parent_offset: u32,
    name: Vec<u8>,
    sibling: u32,
    child_dir: u32,
    child_file: u32,
}

struct FlatFile {
    entry_offset: u32,
    parent_offset: u32,
    name: Vec<u8>,
    sibling: u32,
    data_offset: u64,
    size: u64,
    source: FileSource,
}

impl LayeredRomfs {
    pub fn build(base: Option<&LazyRomfs>, overlay_dir: &Path) -> Result<Self, String> {
        Self::build_many(base, &[overlay_dir.to_path_buf()])
    }

    pub fn build_many(base: Option<&LazyRomfs>, overlay_dirs: &[PathBuf]) -> Result<Self, String> {
        let mut root = match base {
            Some(base) if !base.is_empty() => parse_base_tree(base)?,
            _ => DirNode::default(),
        };
        let mut base_file_count = 0usize;
        fn count_files(node: &DirNode, total: &mut usize) {
            *total += node.files.len();
            for child in node.dirs.values() {
                count_files(child, total);
            }
        }
        count_files(&root, &mut base_file_count);

        let mut stats = (0usize, 0usize);
        for overlay_dir in overlay_dirs {
            overlay_into(&mut root, overlay_dir, &mut stats)?;
        }

        let mut dirs: Vec<FlatDir> = Vec::new();
        let mut files: Vec<FlatFile> = Vec::new();

        fn flatten(
            node: &mut DirNode,
            parent_offset: u32,
            name: Vec<u8>,
            dirs: &mut Vec<FlatDir>,
            files: &mut Vec<FlatFile>,
            next_dir_offset: &mut u32,
            next_file_offset: &mut u32,
        ) -> usize {
            let entry_offset = *next_dir_offset;
            *next_dir_offset += (0x18 + align4(name.len())) as u32;
            let index = dirs.len();
            dirs.push(FlatDir {
                entry_offset,
                parent_offset,
                name,
                sibling: INVALID,
                child_dir: INVALID,
                child_file: INVALID,
            });

            let mut prev_file: Option<usize> = None;
            for (file_name, source) in std::mem::take(&mut node.files) {
                let file_offset = *next_file_offset;
                *next_file_offset += (0x20 + align4(file_name.len())) as u32;
                let size = match &source {
                    FileSource::Base { size, .. } => *size,
                    FileSource::Host { size, .. } => *size,
                };
                let file_index = files.len();
                files.push(FlatFile {
                    entry_offset: file_offset,
                    parent_offset: entry_offset,
                    name: file_name,
                    sibling: INVALID,
                    data_offset: 0,
                    size,
                    source,
                });
                match prev_file {
                    Some(prev) => files[prev].sibling = file_offset,
                    None => dirs[index].child_file = file_offset,
                }
                prev_file = Some(file_index);
            }

            let mut prev_dir: Option<usize> = None;
            for (dir_name, mut child) in std::mem::take(&mut node.dirs) {
                let child_index = flatten(
                    &mut child,
                    entry_offset,
                    dir_name,
                    dirs,
                    files,
                    next_dir_offset,
                    next_file_offset,
                );
                let child_offset = dirs[child_index].entry_offset;
                match prev_dir {
                    Some(prev) => dirs[prev].sibling = child_offset,
                    None => dirs[index].child_dir = child_offset,
                }
                prev_dir = Some(child_index);
            }

            index
        }

        let mut next_dir_offset = 0u32;
        let mut next_file_offset = 0u32;
        flatten(
            &mut root,
            0,
            Vec::new(),
            &mut dirs,
            &mut files,
            &mut next_dir_offset,
            &mut next_file_offset,
        );

        let dir_table_size = next_dir_offset as usize;
        let file_table_size = next_file_offset as usize;
        let dir_hash_count = hash_table_entry_count(dirs.len());
        let file_hash_count = hash_table_entry_count(files.len());

        let mut data_cursor = 0u64;
        for file in files.iter_mut() {
            data_cursor = align16(data_cursor);
            file.data_offset = data_cursor;
            data_cursor += file.size;
        }

        let dir_hash_ofs = 0x50usize;
        let dir_table_ofs = dir_hash_ofs + dir_hash_count * 4;
        let file_hash_ofs = align4(dir_table_ofs + dir_table_size);
        let file_table_ofs = file_hash_ofs + file_hash_count * 4;
        let meta_end = file_table_ofs + file_table_size;
        let file_data_ofs = ((meta_end as u64 + 0x1ff) & !0x1ff) as usize;

        let mut meta = vec![0u8; meta_end];
        meta[0..8].copy_from_slice(&0x50u64.to_le_bytes());
        meta[8..16].copy_from_slice(&(dir_hash_ofs as u64).to_le_bytes());
        meta[16..24].copy_from_slice(&((dir_hash_count * 4) as u64).to_le_bytes());
        meta[24..32].copy_from_slice(&(dir_table_ofs as u64).to_le_bytes());
        meta[32..40].copy_from_slice(&(dir_table_size as u64).to_le_bytes());
        meta[40..48].copy_from_slice(&(file_hash_ofs as u64).to_le_bytes());
        meta[48..56].copy_from_slice(&((file_hash_count * 4) as u64).to_le_bytes());
        meta[56..64].copy_from_slice(&(file_table_ofs as u64).to_le_bytes());
        meta[64..72].copy_from_slice(&(file_table_size as u64).to_le_bytes());
        meta[72..80].copy_from_slice(&(file_data_ofs as u64).to_le_bytes());

        let mut dir_hash = vec![INVALID; dir_hash_count];
        for dir in &dirs {
            let parent = if dir.entry_offset == 0 { 0 } else { dir.parent_offset };
            let hash = romfs_path_hash(parent, &dir.name);
            let bucket = (hash as usize) % dir_hash_count;
            let base_off = dir_table_ofs + dir.entry_offset as usize;
            meta[base_off..base_off + 4].copy_from_slice(&dir.parent_offset.to_le_bytes());
            meta[base_off + 4..base_off + 8].copy_from_slice(&dir.sibling.to_le_bytes());
            meta[base_off + 8..base_off + 12].copy_from_slice(&dir.child_dir.to_le_bytes());
            meta[base_off + 12..base_off + 16].copy_from_slice(&dir.child_file.to_le_bytes());
            meta[base_off + 16..base_off + 20].copy_from_slice(&dir_hash[bucket].to_le_bytes());
            meta[base_off + 20..base_off + 24]
                .copy_from_slice(&(dir.name.len() as u32).to_le_bytes());
            meta[base_off + 24..base_off + 24 + dir.name.len()].copy_from_slice(&dir.name);
            dir_hash[bucket] = dir.entry_offset;
        }
        for (index, word) in dir_hash.iter().enumerate() {
            let off = dir_hash_ofs + index * 4;
            meta[off..off + 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut file_hash = vec![INVALID; file_hash_count];
        let mut extents = Vec::with_capacity(files.len());
        let mut host_paths = Vec::new();
        for file in &files {
            let hash = romfs_path_hash(file.parent_offset, &file.name);
            let bucket = (hash as usize) % file_hash_count;
            let base_off = file_table_ofs + file.entry_offset as usize;
            meta[base_off..base_off + 4].copy_from_slice(&file.parent_offset.to_le_bytes());
            meta[base_off + 4..base_off + 8].copy_from_slice(&file.sibling.to_le_bytes());
            meta[base_off + 8..base_off + 16].copy_from_slice(&file.data_offset.to_le_bytes());
            meta[base_off + 16..base_off + 24].copy_from_slice(&file.size.to_le_bytes());
            meta[base_off + 24..base_off + 28].copy_from_slice(&file_hash[bucket].to_le_bytes());
            meta[base_off + 28..base_off + 32]
                .copy_from_slice(&(file.name.len() as u32).to_le_bytes());
            meta[base_off + 32..base_off + 32 + file.name.len()].copy_from_slice(&file.name);
            file_hash[bucket] = file.entry_offset;

            if file.size > 0 {
                let source = match &file.source {
                    FileSource::Base { data_offset, .. } => ExtentSource::Base {
                        offset: *data_offset,
                    },
                    FileSource::Host { path, .. } => {
                        host_paths.push(path.clone());
                        ExtentSource::Host {
                            index: host_paths.len() - 1,
                        }
                    }
                };
                extents.push(Extent {
                    start: file_data_ofs as u64 + file.data_offset,
                    len: file.size,
                    source,
                });
            }
        }
        for (index, word) in file_hash.iter().enumerate() {
            let off = file_hash_ofs + index * 4;
            meta[off..off + 4].copy_from_slice(&word.to_le_bytes());
        }

        Ok(Self {
            meta,
            extents,
            total_len: file_data_ofs as u64 + data_cursor,
            base: base.cloned(),
            host_paths,
            handles: Mutex::new(HandleCache {
                open: HashMap::new(),
                order: VecDeque::new(),
            }),
            base_file_count,
            overlay_file_count: stats.0,
            replaced_file_count: stats.1,
        })
    }

    pub fn len(&self) -> u64 {
        self.total_len
    }

    pub fn file_count(&self) -> usize {
        self.extents.len()
    }

    fn host_handle(&self, index: usize) -> Result<Arc<File>, String> {
        let mut cache = self.handles.lock().map_err(|_| "handle cache poisoned")?;
        if let Some(file) = cache.open.get(&index) {
            return Ok(Arc::clone(file));
        }
        let path = &self.host_paths[index];
        let file = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
        let file = Arc::new(file);
        cache.open.insert(index, Arc::clone(&file));
        cache.order.push_back(index);
        while cache.order.len() > OPEN_HANDLE_CAP {
            if let Some(evict) = cache.order.pop_front() {
                cache.open.remove(&evict);
            }
        }
        Ok(file)
    }

    fn read_host(&self, index: usize, offset: u64, out: &mut [u8]) -> Result<(), String> {
        let file = self.host_handle(index)?;
        let mut done = 0usize;
        while done < out.len() {
            #[cfg(windows)]
            let got = {
                use std::os::windows::fs::FileExt;
                file.seek_read(&mut out[done..], offset + done as u64)
            };
            #[cfg(unix)]
            let got = {
                use std::os::unix::fs::FileExt;
                file.read_at(&mut out[done..], offset + done as u64)
            };
            match got {
                Ok(0) => {
                    log::warn!(
                        "layered romfs: {} shorter than indexed, zero-filling {} bytes",
                        self.host_paths[index].display(),
                        out.len() - done
                    );
                    for byte in &mut out[done..] {
                        *byte = 0;
                    }
                    return Ok(());
                }
                Ok(n) => done += n,
                Err(e) => {
                    return Err(format!(
                        "read {}: {}",
                        self.host_paths[index].display(),
                        e
                    ))
                }
            }
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let available = self.total_len.saturating_sub(offset).min(size as u64) as usize;
        let mut out = vec![0u8; available];
        if available == 0 {
            return Ok(out);
        }

        if offset < self.meta.len() as u64 {
            let meta_start = offset as usize;
            let take = (self.meta.len() - meta_start).min(available);
            out[..take].copy_from_slice(&self.meta[meta_start..meta_start + take]);
        }

        let end = offset + available as u64;
        let mut index = self
            .extents
            .partition_point(|extent| extent.start + extent.len <= offset);
        while index < self.extents.len() {
            let extent = &self.extents[index];
            if extent.start >= end {
                break;
            }
            let seg_start = extent.start.max(offset);
            let seg_end = (extent.start + extent.len).min(end);
            let out_off = (seg_start - offset) as usize;
            let within = seg_start - extent.start;
            let take = (seg_end - seg_start) as usize;
            match &extent.source {
                ExtentSource::Base { offset: base_off } => {
                    let base = self
                        .base
                        .as_ref()
                        .ok_or("layered romfs base extent without base image")?;
                    let bytes = base.read(base_off + within, take)?;
                    let got = bytes.len().min(take);
                    out[out_off..out_off + got].copy_from_slice(&bytes[..got]);
                }
                ExtentSource::Host { index: host_index } => {
                    self.read_host(*host_index, within, &mut out[out_off..out_off + take])?;
                }
            }
            index += 1;
        }
        Ok(out)
    }
}

pub fn find_overlay_dir(roots: &[PathBuf], title_id: u64) -> Option<PathBuf> {
    let tid = format!("{:016x}", title_id);
    for root in roots {
        let contents = root.join("contents");
        let Ok(entries) = std::fs::read_dir(&contents) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.eq_ignore_ascii_case(&tid) {
                let romfs = entry.path().join("romfs");
                if romfs.is_dir() {
                    return Some(romfs);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup_file(image: &[u8], path: &str) -> Option<(u64, u64)> {
        let dir_hash_ofs = u64le(image, 0x08).ok()? as usize;
        let dir_hash_len = (u64le(image, 0x10).ok()? / 4) as usize;
        let dir_table_ofs = u64le(image, 0x18).ok()? as usize;
        let file_hash_ofs = u64le(image, 0x28).ok()? as usize;
        let file_hash_len = (u64le(image, 0x30).ok()? / 4) as usize;
        let file_table_ofs = u64le(image, 0x38).ok()? as usize;
        let data_ofs = u64le(image, 0x48).ok()?;

        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let (file_name, dir_parts) = parts.split_last()?;

        let mut parent = 0u32;
        for part in dir_parts {
            let hash = romfs_path_hash(parent, part.as_bytes());
            let bucket = dir_hash_ofs + (hash as usize % dir_hash_len) * 4;
            let mut entry = u32le(image, bucket).ok()?;
            loop {
                if entry == INVALID {
                    return None;
                }
                let abs = dir_table_ofs + entry as usize;
                let entry_parent = u32le(image, abs).ok()?;
                let name_len = u32le(image, abs + 0x14).ok()? as usize;
                let name = image.get(abs + 0x18..abs + 0x18 + name_len)?;
                if entry_parent == parent && name == part.as_bytes() {
                    parent = entry;
                    break;
                }
                entry = u32le(image, abs + 0x10).ok()?;
            }
        }

        let hash = romfs_path_hash(parent, file_name.as_bytes());
        let bucket = file_hash_ofs + (hash as usize % file_hash_len) * 4;
        let mut entry = u32le(image, bucket).ok()?;
        loop {
            if entry == INVALID {
                return None;
            }
            let abs = file_table_ofs + entry as usize;
            let entry_parent = u32le(image, abs).ok()?;
            let name_len = u32le(image, abs + 0x1c).ok()? as usize;
            let name = image.get(abs + 0x20..abs + 0x20 + name_len)?;
            if entry_parent == parent && name == file_name.as_bytes() {
                let rel = u64le(image, abs + 0x08).ok()?;
                let size = u64le(image, abs + 0x10).ok()?;
                return Some((data_ofs + rel, size));
            }
            entry = u32le(image, abs + 0x18).ok()?;
        }
    }

    #[test]
    fn overlay_only_tree_round_trips() {
        let dir = std::env::temp_dir().join(format!("nexium-layered-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub/deeper")).unwrap();
        std::fs::write(dir.join("top.bin"), b"top-data").unwrap();
        std::fs::write(dir.join("sub/inner.bin"), b"inner-data-longer").unwrap();
        std::fs::write(dir.join("sub/deeper/leaf.bin"), b"leaf").unwrap();
        std::fs::write(dir.join("sub/CASE.BIN"), b"upper").unwrap();

        let layered = LayeredRomfs::build(None, &dir).unwrap();
        let image = layered.read(0, layered.len() as usize).unwrap();

        for (path, expect) in [
            ("/top.bin", b"top-data" as &[u8]),
            ("/sub/inner.bin", b"inner-data-longer"),
            ("/sub/deeper/leaf.bin", b"leaf"),
            ("/sub/CASE.BIN", b"upper"),
        ] {
            let (off, size) = lookup_file(&image, path).unwrap_or_else(|| panic!("{path} missing"));
            assert_eq!(&image[off as usize..(off + size) as usize], expect, "{path}");
            let direct = layered.read(off, size as usize).unwrap();
            assert_eq!(&direct[..], expect, "{path} direct");
        }
        assert!(lookup_file(&image, "/sub/case.bin").is_none());
        assert!(lookup_file(&image, "/missing").is_none());

        let parsed = crate::romfs::romfs_file(&image, "/sub/inner.bin").unwrap();
        assert_eq!(parsed, b"inner-data-longer");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
