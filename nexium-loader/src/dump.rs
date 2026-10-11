use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use memmap2::Mmap;

use crate::application::{Application, LazyRomfs};
use crate::bin_read::{u32at, u64at};
use crate::container::PartitionFs;
use crate::nca::{Nca, NcaFsType};

const CHUNK: usize = 4 << 20;
const HEADER_LEN: usize = 0x50;
const MAX_TABLE_LEN: u64 = 1 << 30;
const DIR_ENTRY_LEN: usize = 0x18;
const FILE_ENTRY_LEN: usize = 0x20;
const NO_ENTRY: u32 = 0xffff_ffff;
const RESERVED_NAMES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];

#[derive(Clone, Debug)]
pub struct ExefsFile {
    pub name: String,
    pub range: Range<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionStats {
    pub files: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DumpSummary {
    pub exefs: Option<SectionStats>,
    pub romfs: Option<SectionStats>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DumpProgress {
    pub files_done: u64,
    pub files_total: u64,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DumpError {
    Cancelled,
    Failed(String),
}

impl fmt::Display for DumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DumpError::Cancelled => f.write_str("Dump cancelled"),
            DumpError::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DumpError {}

impl From<String> for DumpError {
    fn from(message: String) -> Self {
        DumpError::Failed(message)
    }
}

pub(crate) fn exefs_files(mmap: &Arc<Mmap>, program: &Nca) -> Vec<ExefsFile> {
    let Some(section) = program.section(NcaFsType::PartitionFs) else {
        return Vec::new();
    };
    let exefs = match PartitionFs::parse(mmap.clone(), section.fs_data_range.start) {
        Ok(exefs) => exefs,
        Err(error) => {
            log::warn!(
                "ExeFS of program NCA at {:#x} is unreadable: {}",
                program.nca_base,
                error
            );
            return Vec::new();
        }
    };
    exefs
        .entries()
        .iter()
        .filter_map(|entry| match exefs.entry_range(entry) {
            Ok(range) => Some(ExefsFile {
                name: entry.name.clone(),
                range,
            }),
            Err(error) => {
                log::warn!("skipping ExeFS entry {}: {}", entry.name, error);
                None
            }
        })
        .collect()
}

pub fn safe_name(name: &str) -> String {
    let mut cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let kept = cleaned.trim_end_matches(['.', ' ']).len();
    cleaned.truncate(kept);
    if cleaned.is_empty() {
        return "_".to_string();
    }
    if is_reserved(cleaned.split('.').next().unwrap_or("")) {
        cleaned.insert(0, '_');
    }
    cleaned
}

fn is_reserved(stem: &str) -> bool {
    let stem = stem.trim_end().as_bytes();
    RESERVED_NAMES
        .iter()
        .any(|name| stem.eq_ignore_ascii_case(name.as_bytes()))
        || (stem.len() == 4
            && (stem[..3].eq_ignore_ascii_case(b"COM") || stem[..3].eq_ignore_ascii_case(b"LPT"))
            && matches!(stem[3], b'1'..=b'9'))
}

fn unique_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .map(|name| {
            let mut candidate = name.clone();
            let mut number = 1u32;
            while !seen.insert(candidate.to_lowercase()) {
                number += 1;
                candidate = numbered(&name, number);
            }
            candidate
        })
        .collect()
}

fn numbered(name: &str, number: u32) -> String {
    match name.rfind('.') {
        Some(dot) if dot > 0 => format!("{} ({}){}", &name[..dot], number, &name[dot..]),
        _ => format!("{name} ({number})"),
    }
}

struct Tables {
    dirs: Vec<u8>,
    files: Vec<u8>,
    data_offset: u64,
    image_len: u64,
}

struct PlannedFile {
    path: PathBuf,
    offset: u64,
    size: u64,
}

#[derive(Default)]
struct RomfsPlan {
    dirs: Vec<PathBuf>,
    files: Vec<PlannedFile>,
    bytes: u64,
}

enum Child {
    Dir(u32),
    File { offset: u64, size: u64 },
}

fn read_tables(romfs: &LazyRomfs) -> Result<Tables, String> {
    let image_len = romfs.len();
    let header = romfs.read(0, HEADER_LEN)?;
    if header.len() < HEADER_LEN || u64at(&header, 0)? != HEADER_LEN as u64 {
        return Err("RomFS header is invalid".to_string());
    }
    let table = |offset_at: usize, length_at: usize| -> Result<Vec<u8>, String> {
        let offset = u64at(&header, offset_at)?;
        let length = u64at(&header, length_at)?;
        if length > MAX_TABLE_LEN || offset.checked_add(length).is_none_or(|end| end > image_len) {
            return Err("RomFS metadata table lies outside the image".to_string());
        }
        let bytes = romfs.read(offset, length as usize)?;
        if bytes.len() as u64 != length {
            return Err("RomFS metadata table is truncated".to_string());
        }
        Ok(bytes)
    };
    Ok(Tables {
        dirs: table(0x18, 0x20)?,
        files: table(0x38, 0x40)?,
        data_offset: u64at(&header, 0x48)?,
        image_len,
    })
}

fn entry_name(table: &[u8], entry: usize, len_at: usize, name_at: usize) -> Result<String, String> {
    let len = u32at(table, entry + len_at)? as usize;
    let start = entry + name_at;
    let end = start.checked_add(len).ok_or("RomFS entry name overflows")?;
    let bytes = table
        .get(start..end)
        .ok_or("RomFS entry name is out of range")?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn sibling_chain(table: &[u8], first: u32, limit: usize) -> Result<Vec<u32>, String> {
    let mut entries = Vec::new();
    let mut cursor = first;
    while cursor != NO_ENTRY {
        if entries.len() >= limit {
            return Err("RomFS entry chain is corrupt".to_string());
        }
        entries.push(cursor);
        cursor = u32at(table, cursor as usize + 0x04)?;
    }
    Ok(entries)
}

fn plan_romfs(romfs: &LazyRomfs) -> Result<RomfsPlan, String> {
    let tables = read_tables(romfs)?;
    let dir_limit = tables.dirs.len() / DIR_ENTRY_LEN;
    let file_limit = tables.files.len() / FILE_ENTRY_LEN;
    let mut plan = RomfsPlan::default();
    let mut pending = vec![(0u32, PathBuf::new())];
    while let Some((dir, path)) = pending.pop() {
        if plan.dirs.len() >= dir_limit {
            return Err("RomFS directory table is corrupt".to_string());
        }
        let dir_at = dir as usize;
        let mut children: Vec<(String, Child)> = Vec::new();
        let first_dir = u32at(&tables.dirs, dir_at + 0x08)?;
        for entry in sibling_chain(&tables.dirs, first_dir, dir_limit)? {
            let name = entry_name(&tables.dirs, entry as usize, 0x14, 0x18)?;
            children.push((name, Child::Dir(entry)));
        }
        let first_file = u32at(&tables.dirs, dir_at + 0x0c)?;
        for entry in sibling_chain(&tables.files, first_file, file_limit)? {
            let at = entry as usize;
            let name = entry_name(&tables.files, at, 0x1c, 0x20)?;
            let offset = tables
                .data_offset
                .checked_add(u64at(&tables.files, at + 0x08)?)
                .ok_or("RomFS file offset overflows")?;
            let size = u64at(&tables.files, at + 0x10)?;
            children.push((name, Child::File { offset, size }));
        }
        let names = unique_names(children.iter().map(|(name, _)| safe_name(name)));
        plan.dirs.push(path.clone());
        for ((_, child), name) in children.into_iter().zip(names) {
            let child_path = path.join(name);
            match child {
                Child::Dir(entry) => pending.push((entry, child_path)),
                Child::File { offset, size } => {
                    if plan.files.len() >= file_limit {
                        return Err("RomFS file table is corrupt".to_string());
                    }
                    if offset
                        .checked_add(size)
                        .is_none_or(|end| end > tables.image_len)
                    {
                        return Err(format!(
                            "RomFS file {} lies outside the image; the dump may be truncated",
                            child_path.display()
                        ));
                    }
                    plan.bytes = plan.bytes.saturating_add(size);
                    plan.files.push(PlannedFile {
                        path: child_path,
                        offset,
                        size,
                    });
                }
            }
        }
    }
    Ok(plan)
}

enum Source<'a> {
    Memory(&'a [u8]),
    Romfs(&'a LazyRomfs),
}

impl<'a> Source<'a> {
    fn read(&self, offset: u64, len: usize) -> Result<Cow<'a, [u8]>, String> {
        match *self {
            Source::Memory(bytes) => {
                let start = usize::try_from(offset).map_err(|_| "source offset overflows")?;
                let end = start.checked_add(len).ok_or("source offset overflows")?;
                bytes
                    .get(start..end)
                    .map(Cow::Borrowed)
                    .ok_or_else(|| "source data is truncated".to_string())
            }
            Source::Romfs(romfs) => {
                let bytes = romfs.read(offset, len)?;
                if bytes.len() == len {
                    Ok(Cow::Owned(bytes))
                } else {
                    Err("RomFS storage is truncated".to_string())
                }
            }
        }
    }
}

struct Tracker<'a, F> {
    cancel: &'a AtomicBool,
    report: F,
    state: DumpProgress,
}

impl<F: FnMut(DumpProgress)> Tracker<'_, F> {
    fn check(&self) -> Result<(), DumpError> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(DumpError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn wrote(&mut self, bytes: u64) -> Result<(), DumpError> {
        self.state.bytes_done += bytes;
        (self.report)(self.state);
        self.check()
    }

    fn finished_file(&mut self) {
        self.state.files_done += 1;
        (self.report)(self.state);
    }
}

fn create_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| format!("{} - creating {}", error, path.display()))
}

fn write_file<F: FnMut(DumpProgress)>(
    path: &Path,
    source: &Source<'_>,
    offset: u64,
    size: u64,
    tracker: &mut Tracker<'_, F>,
) -> Result<(), DumpError> {
    tracker.check()?;
    let mut file =
        File::create(path).map_err(|error| format!("{} - creating {}", error, path.display()))?;
    let result = copy_chunks(&mut file, path, source, offset, size, tracker);
    drop(file);
    if result.is_err() {
        let _ = fs::remove_file(path);
    }
    result?;
    tracker.finished_file();
    Ok(())
}

fn copy_chunks<F: FnMut(DumpProgress)>(
    file: &mut File,
    path: &Path,
    source: &Source<'_>,
    offset: u64,
    size: u64,
    tracker: &mut Tracker<'_, F>,
) -> Result<(), DumpError> {
    let mut done = 0u64;
    while done < size {
        let take = (size - done).min(CHUNK as u64) as usize;
        let bytes = source.read(offset + done, take)?;
        file.write_all(&bytes)
            .map_err(|error| format!("{} - writing {}", error, path.display()))?;
        done += take as u64;
        tracker.wrote(take as u64)?;
    }
    Ok(())
}

fn dump_exefs<F: FnMut(DumpProgress)>(
    mmap: &Mmap,
    files: &[ExefsFile],
    dir: &Path,
    tracker: &mut Tracker<'_, F>,
) -> Result<SectionStats, DumpError> {
    create_dir(dir)?;
    let names = unique_names(files.iter().map(|file| safe_name(&file.name)));
    let source = Source::Memory(&mmap[..]);
    let mut stats = SectionStats::default();
    for (file, name) in files.iter().zip(names) {
        let size = file.range.len() as u64;
        write_file(
            &dir.join(name),
            &source,
            file.range.start as u64,
            size,
            tracker,
        )?;
        stats.files += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

fn dump_romfs<F: FnMut(DumpProgress)>(
    romfs: &LazyRomfs,
    plan: &RomfsPlan,
    dir: &Path,
    tracker: &mut Tracker<'_, F>,
) -> Result<SectionStats, DumpError> {
    for relative in &plan.dirs {
        create_dir(&dir.join(relative))?;
    }
    let source = if romfs.is_compressed() {
        Source::Romfs(romfs)
    } else {
        Source::Memory(romfs.as_slice())
    };
    for file in &plan.files {
        write_file(
            &dir.join(&file.path),
            &source,
            file.offset,
            file.size,
            tracker,
        )?;
    }
    Ok(SectionStats {
        files: plan.files.len() as u64,
        bytes: plan.bytes,
    })
}

pub fn dump_application(
    app: &Application,
    out_dir: &Path,
    cancel: &AtomicBool,
    progress: impl FnMut(DumpProgress),
) -> Result<DumpSummary, DumpError> {
    let romfs = app.romfs.as_ref().filter(|romfs| !romfs.is_empty());
    let plan = romfs.map(plan_romfs).transpose()?;
    if app.exefs_files.is_empty() && plan.is_none() {
        return Err(DumpError::Failed(
            "This title has no ExeFS or RomFS to dump.".to_string(),
        ));
    }
    let exefs_bytes: u64 = app
        .exefs_files
        .iter()
        .map(|file| file.range.len() as u64)
        .sum();
    let mut tracker = Tracker {
        cancel,
        report: progress,
        state: DumpProgress {
            files_done: 0,
            files_total: app.exefs_files.len() as u64
                + plan.as_ref().map_or(0, |plan| plan.files.len() as u64),
            bytes_done: 0,
            bytes_total: exefs_bytes + plan.as_ref().map_or(0, |plan| plan.bytes),
        },
    };
    (tracker.report)(tracker.state);
    tracker.check()?;
    let mut summary = DumpSummary::default();
    if !app.exefs_files.is_empty() {
        summary.exefs = Some(dump_exefs(
            &app.mmap,
            &app.exefs_files,
            &out_dir.join("exefs"),
            &mut tracker,
        )?);
    }
    if let (Some(romfs), Some(plan)) = (romfs, &plan) {
        summary.romfs = Some(dump_romfs(
            romfs,
            plan,
            &out_dir.join("romfs"),
            &mut tracker,
        )?);
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::tests::{pfs, TestDirectory};
    use crate::LayeredRomfs;
    use std::collections::BTreeMap;

    const TITLE: u64 = 0x0100_0000_0000_1234;

    fn mmap(bytes: &[u8]) -> Arc<Mmap> {
        let mut map = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        map[..bytes.len()].copy_from_slice(bytes);
        Arc::new(map.make_read_only().unwrap())
    }

    fn nso_bytes() -> Vec<u8> {
        let mut bytes = vec![0; 0x130];
        bytes[..4].copy_from_slice(b"NSO0");
        for index in 0..3usize {
            let header = 0x10 + index * 0x10;
            let file = 0x100 + index * 0x10;
            bytes[header..header + 4].copy_from_slice(&(file as u32).to_le_bytes());
            bytes[header + 4..header + 8].copy_from_slice(&(index as u32 * 0x1000).to_le_bytes());
            bytes[header + 8..header + 12].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[0x60 + index * 4..0x64 + index * 4].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[file..file + 0x10].fill(1);
        }
        bytes
    }

    fn program_nca(exefs: &[u8], romfs: &[u8]) -> Vec<u8> {
        const SECTOR: usize = 0x200;
        let sectors = |len: usize| len.div_ceil(SECTOR);
        let exefs_start = 0xC00 / SECTOR;
        let exefs_end = exefs_start + sectors(exefs.len());
        let romfs_end = exefs_end + sectors(romfs.len());
        let mut bytes = vec![0u8; romfs_end * SECTOR];
        let total = bytes.len() as u64;
        bytes[0x200..0x204].copy_from_slice(b"DNCA");
        bytes[0x208..0x210].copy_from_slice(&total.to_le_bytes());
        bytes[0x210..0x218].copy_from_slice(&TITLE.to_le_bytes());
        let sections = [
            (exefs_start, exefs_end, 1u8, exefs),
            (exefs_end, romfs_end, 0u8, romfs),
        ];
        for (index, (start, end, fs_type, body)) in sections.into_iter().enumerate() {
            let info = 0x240 + index * 0x10;
            bytes[info..info + 4].copy_from_slice(&(start as u32).to_le_bytes());
            bytes[info + 4..info + 8].copy_from_slice(&(end as u32).to_le_bytes());
            let header = 0x400 + index * 0x200;
            bytes[header + 2] = fs_type;
            bytes[header + 3] = 1;
            bytes[header + 4] = 1;
            bytes[start * SECTOR..start * SECTOR + body.len()].copy_from_slice(body);
        }
        bytes
    }

    fn romfs_image(source: &Path) -> Vec<u8> {
        let layered = LayeredRomfs::build(None, source).unwrap();
        layered.read(0, layered.len() as usize).unwrap()
    }

    fn lazy(bytes: &[u8]) -> LazyRomfs {
        LazyRomfs::from_range(mmap(bytes), 0..bytes.len()).unwrap()
    }

    fn read_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
        fn walk(dir: &Path, prefix: &str, out: &mut BTreeMap<String, Vec<u8>>) {
            for entry in fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
                if entry.file_type().unwrap().is_dir() {
                    walk(&entry.path(), &format!("{name}/"), out);
                } else {
                    out.insert(name, fs::read(entry.path()).unwrap());
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(root, "", &mut out);
        out
    }

    fn patch_unique(bytes: &mut [u8], from: &[u8], to: &[u8]) {
        assert_eq!(from.len(), to.len());
        let hits: Vec<usize> = bytes
            .windows(from.len())
            .enumerate()
            .filter(|(_, window)| *window == from)
            .map(|(at, _)| at)
            .collect();
        assert_eq!(hits.len(), 1, "{:?} must occur exactly once", from);
        bytes[hits[0]..hits[0] + to.len()].copy_from_slice(to);
    }

    fn sample_source(dir: &TestDirectory) -> PathBuf {
        let source = dir.0.join("source");
        fs::create_dir_all(source.join("sub/deeper")).unwrap();
        fs::create_dir_all(source.join("empty")).unwrap();
        fs::write(source.join("top.bin"), b"top").unwrap();
        fs::write(source.join("sub/inner.bin"), vec![7u8; CHUNK + 123]).unwrap();
        fs::write(source.join("sub/deeper/leaf.bin"), b"").unwrap();
        source
    }

    fn no_progress(_: DumpProgress) {}

    #[test]
    fn unsafe_names_are_neutralised() {
        for (input, expected) in [
            ("plain name.bin", "plain name.bin"),
            ("é.png", "é.png"),
            ("a:b", "a_b"),
            ("a/b\\c", "a_b_c"),
            ("what?.txt", "what_.txt"),
            ("tab\there", "tab_here"),
            ("..", "_"),
            (".", "_"),
            ("", "_"),
            ("trail. .", "trail"),
            ("con", "_con"),
            ("Aux.txt", "_Aux.txt"),
            ("NUL.tar.gz", "_NUL.tar.gz"),
            ("com1", "_com1"),
            ("LPT9.bin", "_LPT9.bin"),
            ("com10", "com10"),
            ("com0", "com0"),
            ("console", "console"),
            ("abé", "abé"),
        ] {
            assert_eq!(safe_name(input), expected, "{input:?}");
        }
    }

    #[test]
    fn duplicate_names_are_numbered_ignoring_case() {
        let names = unique_names(
            ["a.bin", "A.BIN", "a.bin", "b", "B", ".hidden", ".hidden"].map(String::from),
        );
        assert_eq!(
            names,
            [
                "a.bin",
                "A (2).BIN",
                "a (3).bin",
                "b",
                "B (2)",
                ".hidden",
                ".hidden (2)"
            ]
        );
        assert_eq!(
            unique_names(["a", "a", "a (2)"].map(String::from)),
            ["a", "a (2)", "a (2) (2)"]
        );
    }

    #[test]
    fn large_files_are_written_in_chunks_from_either_source() {
        let dir = TestDirectory::new();
        let data: Vec<u8> = (0..CHUNK * 2 + 17).map(|i| (i % 251) as u8).collect();
        let storage = lazy(&data);
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        {
            let mut tracker = Tracker {
                cancel: &cancel,
                report: |progress| events.push(progress),
                state: DumpProgress::default(),
            };
            let size = data.len() as u64;
            write_file(
                &dir.0.join("memory.bin"),
                &Source::Memory(&data),
                0,
                size,
                &mut tracker,
            )
            .unwrap();
            write_file(
                &dir.0.join("reader.bin"),
                &Source::Romfs(&storage),
                0,
                size,
                &mut tracker,
            )
            .unwrap();
            write_file(
                &dir.0.join("tail.bin"),
                &Source::Memory(&data),
                100,
                50,
                &mut tracker,
            )
            .unwrap();
        }
        assert_eq!(fs::read(dir.0.join("memory.bin")).unwrap(), data);
        assert_eq!(fs::read(dir.0.join("reader.bin")).unwrap(), data);
        assert_eq!(fs::read(dir.0.join("tail.bin")).unwrap(), &data[100..150]);
        let last = events.last().unwrap();
        assert_eq!(last.files_done, 3);
        assert_eq!(last.bytes_done, data.len() as u64 * 2 + 50);
        assert!(events
            .windows(2)
            .all(|pair| pair[0].bytes_done <= pair[1].bytes_done));
    }

    #[test]
    fn cancelling_between_chunks_removes_the_partial_file() {
        let dir = TestDirectory::new();
        let data = vec![1u8; CHUNK * 3];
        let cancel = AtomicBool::new(false);
        let mut tracker = Tracker {
            cancel: &cancel,
            report: |progress: DumpProgress| {
                if progress.bytes_done >= CHUNK as u64 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            state: DumpProgress::default(),
        };
        let result = write_file(
            &dir.0.join("big.bin"),
            &Source::Memory(&data),
            0,
            data.len() as u64,
            &mut tracker,
        );
        assert_eq!(result, Err(DumpError::Cancelled));
        assert!(!dir.0.join("big.bin").exists());
        assert_eq!(tracker.state.files_done, 0);
    }

    #[test]
    fn failed_writes_report_the_reason_first_and_leave_no_partial_file() {
        let dir = TestDirectory::new();
        let blocker = dir.0.join("blocker");
        fs::write(&blocker, b"a file, not a directory").unwrap();
        let cancel = AtomicBool::new(false);
        let mut tracker = Tracker {
            cancel: &cancel,
            report: no_progress,
            state: DumpProgress::default(),
        };
        let target = blocker.join("child.bin");
        let result = write_file(&target, &Source::Memory(b"data"), 0, 4, &mut tracker);
        let Err(DumpError::Failed(message)) = result else {
            panic!("writing below a file must fail: {result:?}");
        };
        assert!(message.contains(" - creating "), "{message}");
        assert!(
            message.ends_with(&target.display().to_string()),
            "{message}"
        );

        let short = [1u8; 8];
        let result = write_file(
            &dir.0.join("short.bin"),
            &Source::Memory(&short),
            4,
            8,
            &mut tracker,
        );
        let Err(DumpError::Failed(message)) = result else {
            panic!("reading past the source must fail: {result:?}");
        };
        assert!(message.contains("truncated"), "{message}");
        assert!(!dir.0.join("short.bin").exists());
    }

    #[test]
    fn romfs_tree_round_trips_with_empty_directories_and_files() {
        let dir = TestDirectory::new();
        let image = romfs_image(&sample_source(&dir));
        let romfs = lazy(&image);
        let plan = plan_romfs(&romfs).unwrap();
        assert_eq!(plan.files.len(), 3);
        assert_eq!(plan.dirs.len(), 4);
        let out = dir.0.join("out");
        let cancel = AtomicBool::new(false);
        let mut tracker = Tracker {
            cancel: &cancel,
            report: no_progress,
            state: DumpProgress::default(),
        };
        let stats = dump_romfs(&romfs, &plan, &out, &mut tracker).unwrap();
        assert_eq!(
            stats,
            SectionStats {
                files: 3,
                bytes: 3 + CHUNK as u64 + 123
            }
        );
        assert_eq!(read_tree(&out), read_tree(&dir.0.join("source")));
        assert!(out.join("empty").is_dir());
        assert!(out.join("sub/deeper").is_dir());
    }

    #[test]
    fn hostile_names_cannot_escape_the_output_directory() {
        let dir = TestDirectory::new();
        let source = dir.0.join("source");
        fs::create_dir_all(source.join("QQ")).unwrap();
        fs::write(source.join("QQ/file.bin"), b"inner").unwrap();
        fs::write(source.join("ZZZ"), b"outer").unwrap();
        let mut image = romfs_image(&source);
        patch_unique(&mut image, b"QQ", b"..");
        patch_unique(&mut image, b"ZZZ", b"a/b");
        let romfs = lazy(&image);
        let plan = plan_romfs(&romfs).unwrap();
        let out = dir.0.join("out/romfs");
        let cancel = AtomicBool::new(false);
        let mut tracker = Tracker {
            cancel: &cancel,
            report: no_progress,
            state: DumpProgress::default(),
        };
        dump_romfs(&romfs, &plan, &out, &mut tracker).unwrap();
        assert_eq!(
            read_tree(&dir.0.join("out")),
            BTreeMap::from([
                ("romfs/_/file.bin".to_string(), b"inner".to_vec()),
                ("romfs/a_b".to_string(), b"outer".to_vec()),
            ])
        );
    }

    #[test]
    fn colliding_names_keep_every_file() {
        let dir = TestDirectory::new();
        let source = dir.0.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("a_b"), b"first").unwrap();
        fs::write(source.join("a_x"), b"second").unwrap();
        let mut image = romfs_image(&source);
        patch_unique(&mut image, b"a_x", b"a:b");
        let romfs = lazy(&image);
        let plan = plan_romfs(&romfs).unwrap();
        let out = dir.0.join("out");
        let cancel = AtomicBool::new(false);
        let mut tracker = Tracker {
            cancel: &cancel,
            report: no_progress,
            state: DumpProgress::default(),
        };
        dump_romfs(&romfs, &plan, &out, &mut tracker).unwrap();
        let tree = read_tree(&out);
        assert_eq!(tree.len(), 2);
        let mut contents: Vec<_> = tree.values().cloned().collect();
        contents.sort();
        assert_eq!(contents, [b"first".to_vec(), b"second".to_vec()]);
    }

    #[test]
    fn corrupt_directory_graphs_are_rejected_instead_of_looping() {
        let dir = TestDirectory::new();
        let mut image = romfs_image(&sample_source(&dir));
        let dir_table = u64::from_le_bytes(image[0x18..0x20].try_into().unwrap()) as usize;
        image[dir_table + 0x08..dir_table + 0x0c].copy_from_slice(&0u32.to_le_bytes());
        let error = plan_romfs(&lazy(&image))
            .err()
            .expect("cycle must be rejected");
        assert!(error.contains("corrupt"), "{error}");

        let mut image = romfs_image(&sample_source(&dir));
        let file_table = u64::from_le_bytes(image[0x38..0x40].try_into().unwrap()) as usize;
        image[file_table + 0x04..file_table + 0x08].copy_from_slice(&0u32.to_le_bytes());
        let error = plan_romfs(&lazy(&image))
            .err()
            .expect("file cycle must be rejected");
        assert!(error.contains("corrupt"), "{error}");
    }

    #[test]
    fn files_beyond_the_image_are_rejected_during_planning() {
        let dir = TestDirectory::new();
        let mut image = romfs_image(&sample_source(&dir));
        let file_table = u64::from_le_bytes(image[0x38..0x40].try_into().unwrap()) as usize;
        image[file_table + 0x10..file_table + 0x18].copy_from_slice(&u64::MAX.to_le_bytes());
        let error = plan_romfs(&lazy(&image))
            .err()
            .expect("oversized file must be rejected");
        assert!(error.contains("outside the image"), "{error}");

        let image = romfs_image(&sample_source(&dir));
        let mut bad_header = image.clone();
        bad_header[0..8].copy_from_slice(&0x60u64.to_le_bytes());
        assert!(plan_romfs(&lazy(&bad_header)).is_err());
        assert!(plan_romfs(&lazy(&image[..0x30])).is_err());
        assert!(plan_romfs(&lazy(&image)).is_ok());
    }

    fn load_fixture(dir: &TestDirectory, name: &str, romfs: &[u8]) -> Application {
        let exefs = pfs(&[
            ("main".to_string(), nso_bytes()),
            ("main.npdm".to_string(), b"not a real npdm".to_vec()),
            ("rtld".to_string(), nso_bytes()),
        ]);
        let path = dir.0.join(name);
        fs::write(&path, program_nca(&exefs, romfs)).unwrap();
        Application::load(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn application_dump_writes_exefs_and_romfs_and_reports_progress() {
        let dir = TestDirectory::new();
        let image = romfs_image(&sample_source(&dir));
        let app = load_fixture(&dir, "writes.dnca", &image);
        assert_eq!(
            app.exefs_files
                .iter()
                .map(|file| file.name.as_str())
                .collect::<Vec<_>>(),
            ["main", "main.npdm", "rtld"]
        );
        let out = dir.0.join("dump");
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        let summary =
            dump_application(&app, &out, &cancel, |progress| events.push(progress)).unwrap();
        assert_eq!(
            summary,
            DumpSummary {
                exefs: Some(SectionStats {
                    files: 3,
                    bytes: nso_bytes().len() as u64 * 2 + 15,
                }),
                romfs: Some(SectionStats {
                    files: 3,
                    bytes: 3 + CHUNK as u64 + 123
                }),
            }
        );
        assert_eq!(fs::read(out.join("exefs/main")).unwrap(), nso_bytes());
        assert_eq!(fs::read(out.join("exefs/rtld")).unwrap(), nso_bytes());
        assert_eq!(
            fs::read(out.join("exefs/main.npdm")).unwrap(),
            b"not a real npdm"
        );
        assert_eq!(
            read_tree(&out.join("romfs")),
            read_tree(&dir.0.join("source"))
        );
        let first = events.first().unwrap();
        let last = events.last().unwrap();
        assert_eq!(first.files_done, 0);
        assert_eq!(first.files_total, 6);
        assert_eq!(last.files_done, 6);
        assert_eq!(last.bytes_done, last.bytes_total);
        assert_eq!(last.bytes_total, first.bytes_total);
    }

    #[test]
    fn application_dump_validates_before_writing_and_honours_cancel() {
        let dir = TestDirectory::new();
        let mut image = romfs_image(&sample_source(&dir));
        let file_table = u64::from_le_bytes(image[0x38..0x40].try_into().unwrap()) as usize;
        image[file_table + 0x10..file_table + 0x18].copy_from_slice(&u64::MAX.to_le_bytes());
        let broken = load_fixture(&dir, "broken.dnca", &image);
        let out = dir.0.join("broken");
        let cancel = AtomicBool::new(false);
        let result = dump_application(&broken, &out, &cancel, no_progress);
        assert!(matches!(result, Err(DumpError::Failed(_))), "{result:?}");
        assert!(!out.exists());

        let image = romfs_image(&sample_source(&dir));
        let app = load_fixture(&dir, "cancelled.dnca", &image);
        let out = dir.0.join("cancelled");
        let cancel = AtomicBool::new(true);
        assert_eq!(
            dump_application(&app, &out, &cancel, no_progress),
            Err(DumpError::Cancelled)
        );
        assert!(!out.exists());
    }
}
