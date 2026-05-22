use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn sdmc_root() -> PathBuf {
    if let Some(dirs) = directories::ProjectDirs::from("com", "NeXium", "NeXium") {
        let root = dirs.data_dir().join("sdmc");
        let _ = std::fs::create_dir_all(&root);
        return root;
    }
    PathBuf::from("./sdmc")
}

pub fn translate_path(switch_path: &str) -> PathBuf {
    let cleaned = switch_path.trim_end_matches('\0');
    let stripped = cleaned
        .strip_prefix("sdmc:/")
        .or_else(|| cleaned.strip_prefix("sdmc:"))
        .or_else(|| cleaned.strip_prefix("/"))
        .unwrap_or(cleaned);
    let stripped = stripped.trim_start_matches('/');
    sdmc_root().join(stripped.replace('\\', "/"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryType {
    Directory = 0,
    File = 1,
}

pub fn entry_type(switch_path: &str) -> Option<EntryType> {
    let host = translate_path(switch_path);
    let meta = std::fs::metadata(&host).ok()?;
    if meta.is_dir() {
        Some(EntryType::Directory)
    } else if meta.is_file() {
        Some(EntryType::File)
    } else {
        None
    }
}

pub struct OpenFile {
    file: File,
    pub size: u64,
    pub writable: bool,
}

pub struct FsHost {
    files: HashMap<u32, Arc<Mutex<OpenFile>>>,
    next_id: u32,
}

impl FsHost {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn open_file(&mut self, switch_path: &str, mode: u32) -> Result<u32, u32> {
        let host = translate_path(switch_path);
        let read = (mode & 1) != 0;
        let write = (mode & 2) != 0;
        let append = (mode & 4) != 0;

        let mut opts = OpenOptions::new();
        opts.read(read || !write);
        if write {
            opts.write(true);
            if append {
                opts.create(true).append(true);
            }
        }

        let file = opts.open(&host).map_err(|e| {
            log::debug!("fs_host: failed to open {:?}: {}", host, e);
            FS_NOT_FOUND
        })?;

        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.files.insert(id, Arc::new(Mutex::new(OpenFile { file, size, writable: write })));
        log::debug!("fs_host: opened {:?} as id={} size={}", host, id, size);
        Ok(id)
    }

    pub fn create_file(&mut self, switch_path: &str, size: u64) -> Result<(), u32> {
        let host = translate_path(switch_path);
        if let Some(parent) = host.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let f = File::create(&host).map_err(|_| FS_ACCESS_DENIED)?;
        f.set_len(size).map_err(|_| FS_ACCESS_DENIED)?;
        Ok(())
    }

    pub fn delete_file(&mut self, switch_path: &str) -> Result<(), u32> {
        std::fs::remove_file(translate_path(switch_path)).map_err(|_| FS_NOT_FOUND)
    }

    pub fn create_dir(&mut self, switch_path: &str) -> Result<(), u32> {
        std::fs::create_dir_all(translate_path(switch_path)).map_err(|_| FS_ACCESS_DENIED)
    }

    pub fn read(&self, id: u32, offset: u64, buf: &mut [u8]) -> Result<usize, u32> {
        let entry = self.files.get(&id).ok_or(FS_NOT_FOUND)?;
        let mut g = entry.lock();
        g.file.seek(SeekFrom::Start(offset)).map_err(|_| FS_IO_ERROR)?;
        let n = g.file.read(buf).map_err(|_| FS_IO_ERROR)?;
        Ok(n)
    }

    pub fn write(&self, id: u32, offset: u64, data: &[u8]) -> Result<(), u32> {
        let entry = self.files.get(&id).ok_or(FS_NOT_FOUND)?;
        let mut g = entry.lock();
        if !g.writable {
            return Err(FS_ACCESS_DENIED);
        }
        g.file.seek(SeekFrom::Start(offset)).map_err(|_| FS_IO_ERROR)?;
        g.file.write_all(data).map_err(|_| FS_IO_ERROR)?;
        if offset + data.len() as u64 > g.size {
            g.size = offset + data.len() as u64;
        }
        Ok(())
    }

    pub fn size(&self, id: u32) -> Option<u64> {
        self.files.get(&id).map(|f| f.lock().size)
    }

    pub fn close(&mut self, id: u32) {
        if let Some(f) = self.files.remove(&id) {
            let mut g = f.lock();
            let _ = g.file.flush();
        }
    }
}

impl Default for FsHost {
    fn default() -> Self {
        Self::new()
    }
}

pub const FS_NOT_FOUND: u32 = 1 | (1 << 9) | (2 << 21);
pub const FS_ACCESS_DENIED: u32 = 6 | (1 << 9) | (2 << 21);
pub const FS_IO_ERROR: u32 = 4006 | (1 << 9) | (2 << 21);

pub static FS_HOST: once_cell::sync::OnceCell<Arc<Mutex<FsHost>>> = once_cell::sync::OnceCell::new();

pub fn get_fs_host() -> Arc<Mutex<FsHost>> {
    FS_HOST
        .get_or_init(|| Arc::new(Mutex::new(FsHost::new())))
        .clone()
}

pub fn read_path_from_buffer(buf: &[u8], addr: u64, size: u64, mem_read: impl FnOnce(u64, &mut [u8]) -> bool) -> Option<String> {
    let _ = buf;
    let mut bytes = vec![0u8; size as usize];
    if !mem_read(addr, &mut bytes) {
        return None;
    }
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8(bytes[..end].to_vec()).ok()
}
