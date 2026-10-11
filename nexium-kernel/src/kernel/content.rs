use super::{
    build_ipc_response, fs_trace_enabled, fs_trace_read, ipc, mii_model_romfs, ng_word2_romfs,
    return_subsession, romfs_path_for_data_offset, Kernel,
};
use nexium_loader::{AppRomfs, LazyRomfs};
use std::borrow::Cow;

#[cfg(test)]
#[path = "content_tests.rs"]
mod tests;

const INVALID_INPUT: u32 = 0xD401;
const CONTENT_NOT_FOUND: u32 = 0x7D402;
const UNSUPPORTED: u32 = 0x177202;

fn input<const N: usize>(ctx: &ipc::IpcCtx, offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    if end > ctx.cmif_in_data_len {
        return None;
    }
    let start = ctx.cmif_in_data_off.checked_add(offset)?;
    ctx.buf.get(start..start.checked_add(N)?)?.try_into().ok()
}

fn add_on_base(application_id: u64) -> Option<u64> {
    (application_id & !0xfff).checked_add(0x1000)
}

fn add_on_indices(kernel: &Kernel, application_id: u64) -> Vec<u32> {
    if application_id & !0xfff != kernel.title_id & !0xfff {
        return Vec::new();
    }
    let Some(base) = add_on_base(application_id) else {
        return Vec::new();
    };
    kernel
        .add_on_content
        .keys()
        .filter_map(|title_id| {
            title_id
                .checked_sub(base)
                .filter(|index| *index < 0x1000)
                .map(|index| index as u32)
        })
        .collect()
}

pub(super) fn dispatch_add_on_content(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    command: u32,
) -> Option<Vec<u8>> {
    if command > 7 {
        return None;
    }
    let application_id = match command {
        0 | 4 => input::<8>(ctx, 0).map(u64::from_le_bytes),
        1 | 6 => input::<8>(ctx, 8).map(u64::from_le_bytes),
        _ => Some(kernel.title_id),
    };
    let Some(application_id) = application_id else {
        return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
    };
    match command {
        0 | 2 => {
            let count = add_on_indices(kernel, application_id).len() as u32;
            Some(build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]))
        }
        1 | 3 => {
            let (Some(offset), Some(count)) = (input::<4>(ctx, 0), input::<4>(ctx, 4)) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            let offset = u32::from_le_bytes(offset) as usize;
            let count = u32::from_le_bytes(count) as usize;
            let target = ctx
                .recv_buffers
                .iter()
                .chain(ctx.recv_statics.iter())
                .find(|buffer| buffer.size != 0 && buffer.addr != 0)
                .copied();
            let capacity = target
                .map(|buffer| (buffer.size / 4).min(u32::MAX as u64) as usize)
                .unwrap_or(0);
            let values = add_on_indices(kernel, application_id);
            let output: Vec<u8> = values
                .iter()
                .skip(offset)
                .take(count.min(capacity))
                .flat_map(|index| index.to_le_bytes())
                .collect();
            if count != 0 && offset < values.len() && target.is_none() {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            if let Some(target) = target.filter(|_| !output.is_empty()) {
                if kernel
                    .address_space
                    .write_checked(target.addr, &output)
                    .is_err()
                {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                }
            }
            let count = (output.len() / 4) as u32;
            Some(build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]))
        }
        4 | 5 => {
            let Some(base) = add_on_base(application_id) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            Some(build_ipc_response(ctx, 0, &base.to_le_bytes(), &[]))
        }
        6 | 7 => {
            let Some(index) = input::<4>(ctx, 0).map(u32::from_le_bytes) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            let result = if add_on_indices(kernel, application_id)
                .binary_search(&index)
                .is_ok()
            {
                0
            } else {
                CONTENT_NOT_FOUND
            };
            Some(build_ipc_response(ctx, result, &[], &[]))
        }
        _ => None,
    }
}

enum Storage<'a> {
    Application(&'a AppRomfs),
    Archive(&'a LazyRomfs),
    Bytes(&'a [u8]),
}

/// An owned handle on a storage that can be read without the kernel: the
/// loader storages are immutable and share their mappings and caches.
#[derive(Clone)]
enum SharedStorage {
    Application(AppRomfs),
    Archive(LazyRomfs),
}

impl SharedStorage {
    fn read_exact(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let mut bytes = off_lock::buffer(size);
        let got = match self {
            Self::Application(storage) => storage.read_into(offset, &mut bytes)?,
            Self::Archive(storage) => storage.read_into(offset, &mut bytes)?,
        };
        bytes.truncate(got);
        Ok(bytes)
    }
}

/// Large RomFS reads (file data is mmapped, possibly decompressed, and
/// copied out) can take tens of milliseconds. Holding the global kernel lock
/// that long stalls every other core and vsync, so a read of at least
/// `off_lock_read_min()` bytes runs on a worker thread while the requesting
/// guest thread is parked on its `svcSendSyncRequest`, exactly like a
/// DequeueBuffer or GPU-drain retry. The re-issued request (same thread, same
/// request) collects the bytes and copies them into guest memory under the
/// lock, so the guest sees the same reply and the same buffer contents, only
/// later. Without a guest thread to park (tests, host calls) reads run inline.
mod off_lock {
    use super::super::{rewind_svc_for_retry_after, IPC_RETRY_INTERVAL};
    use super::{Kernel, SharedStorage};
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    const DEFAULT_MIN_KIB: usize = 1024;
    const ABANDONED_AFTER: Duration = Duration::from_secs(30);

    /// `NEXIUM_FS_OFFLOCK_READ_KIB`: reads of at least this many KiB run off
    /// the kernel lock (default 1024; 0 or "off" keeps every read inline).
    pub(super) fn min_size() -> Option<usize> {
        static MIN: OnceLock<Option<usize>> = OnceLock::new();
        *MIN.get_or_init(|| {
            match std::env::var("NEXIUM_FS_OFFLOCK_READ_KIB").ok().as_deref().map(str::trim) {
                None | Some("") => Some(DEFAULT_MIN_KIB << 10),
                Some("off") | Some("0") => None,
                Some(value) => Some(value.parse::<usize>().map_or(DEFAULT_MIN_KIB, |kib| kib.max(1)) << 10),
            }
        })
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    pub(super) struct Request {
        pub port: String,
        pub offset: u64,
        pub size: usize,
        pub target: u64,
    }

    type Slot = Arc<Mutex<Option<Result<Vec<u8>, String>>>>;

    struct Pending {
        request: Request,
        result: Slot,
        started: Instant,
    }

    /// In-flight reads keyed by (kernel, guest thread).
    fn pending() -> &'static Mutex<HashMap<(usize, u32), Pending>> {
        static PENDING: OnceLock<Mutex<HashMap<(usize, u32), Pending>>> = OnceLock::new();
        PENDING.get_or_init(Default::default)
    }

    /// Off-lock read buffers are recycled so a large read neither faults in
    /// a fresh allocation on the worker nor unmaps one under the kernel lock.
    const POOL_BYTES: usize = 64 << 20;
    const POOL_BUFFERS: usize = 4;

    fn pool() -> &'static Mutex<Vec<Vec<u8>>> {
        static POOL: OnceLock<Mutex<Vec<Vec<u8>>>> = OnceLock::new();
        POOL.get_or_init(Default::default)
    }

    /// A `size`-byte buffer whose contents are unspecified.
    pub(super) fn buffer(size: usize) -> Vec<u8> {
        let reused = {
            let mut pool = pool().lock();
            pool.iter()
                .position(|buffer| buffer.capacity() >= size && buffer.capacity() / 2 <= size)
                .map(|index| pool.swap_remove(index))
        };
        let mut buffer = reused.unwrap_or_default();
        if buffer.len() >= size {
            buffer.truncate(size);
        } else {
            buffer.resize(size, 0);
        }
        buffer
    }

    /// Returns a large owned read buffer to the pool once the guest has its
    /// bytes.
    pub(super) fn recycle(bytes: std::borrow::Cow<'_, [u8]>) {
        let std::borrow::Cow::Owned(buffer) = bytes else { return };
        if min_size().is_none_or(|min| buffer.capacity() < min) {
            return;
        }
        let mut pool = pool().lock();
        let pooled: usize = pool.iter().map(Vec::capacity).sum();
        if pool.len() < POOL_BUFFERS && pooled + buffer.capacity() <= POOL_BYTES {
            pool.push(buffer);
        }
    }

    pub(super) enum Outcome {
        /// The bytes are here; reply now.
        Ready(Result<Vec<u8>, String>),
        /// The thread is parked and will re-issue the request.
        Retry,
        /// Read inline under the lock.
        Inline,
    }

    pub(super) fn read(
        kernel: &mut Kernel,
        request: Request,
        storage: impl FnOnce(&Kernel) -> Option<SharedStorage>,
    ) -> Outcome {
        let Some(min) = min_size() else {
            return Outcome::Inline;
        };
        let Some(thread) = kernel.threads.current_handle() else {
            return Outcome::Inline;
        };
        let key = (Arc::as_ptr(&kernel.address_space) as *const () as usize, thread);
        let in_flight = {
            let mut pending = pending().lock();
            match pending.get(&key) {
                Some(entry) if entry.request == request => {
                    let ready = entry.result.lock().take();
                    if let Some(result) = ready {
                        pending.remove(&key);
                        return Outcome::Ready(result);
                    }
                    Some(entry.result.clone())
                }
                Some(_) => {
                    pending.remove(&key);
                    None
                }
                None => None,
            }
        };
        if let Some(slot) = in_flight {
            if rewind_svc_for_retry_after(kernel, IPC_RETRY_INTERVAL) {
                return Outcome::Retry;
            }
            pending().lock().remove(&key);
            loop {
                if let Some(result) = slot.lock().take() {
                    return Outcome::Ready(result);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        }
        if request.size < min {
            return Outcome::Inline;
        }
        let Some(storage) = storage(kernel) else {
            return Outcome::Inline;
        };
        if !rewind_svc_for_retry_after(kernel, IPC_RETRY_INTERVAL) {
            return Outcome::Inline;
        }
        let slot: Slot = Arc::new(Mutex::new(None));
        let (offset, size, worker_slot) = (request.offset, request.size, slot.clone());
        let fallback = storage.clone();
        let spawned = std::thread::Builder::new()
            .name("nexium-fs-read".into())
            .spawn(move || {
                let result = storage.read_exact(offset, size);
                *worker_slot.lock() = Some(result);
            });
        if let Err(error) = spawned {
            log::warn!("fs: could not start an off-lock read ({error}); reading under the kernel lock");
            *slot.lock() = Some(fallback.read_exact(offset, size));
        }
        let mut pending = pending().lock();
        let now = Instant::now();
        pending.retain(|_, entry| {
            now.duration_since(entry.started) < ABANDONED_AFTER || entry.result.lock().is_none()
        });
        pending.insert(key, Pending { request, result: slot, started: now });
        Outcome::Retry
    }
}

impl<'a> Storage<'a> {
    fn len(&self) -> u64 {
        match self {
            Self::Application(storage) => storage.len(),
            Self::Archive(storage) => storage.len(),
            Self::Bytes(bytes) => bytes.len() as u64,
        }
    }

    fn shared(&self) -> Option<SharedStorage> {
        match self {
            Self::Application(storage) => Some(SharedStorage::Application((*storage).clone())),
            Self::Archive(storage) => Some(SharedStorage::Archive((*storage).clone())),
            Self::Bytes(_) => None,
        }
    }

    fn read(&self, offset: u64, size: usize) -> Result<Cow<'a, [u8]>, String> {
        match self {
            Self::Application(storage) => storage.read(offset, size).map(Cow::Owned),
            Self::Archive(storage) => storage.read(offset, size).map(Cow::Owned),
            Self::Bytes(bytes) => {
                let start = usize::try_from(offset).map_err(|_| "storage offset overflow")?;
                let end = start.checked_add(size).ok_or("storage size overflow")?;
                bytes
                    .get(start..end)
                    .map(Cow::Borrowed)
                    .ok_or_else(|| "storage read out of range".into())
            }
        }
    }
}

fn data_storage(kernel: &Kernel, title_id: u64) -> Option<Storage<'_>> {
    if let Some(storage) = kernel.add_on_content.get(&title_id) {
        return storage.as_ref().map(Storage::Archive);
    }
    if let Some(storage) = kernel.system_romfs.get(&title_id) {
        return Some(Storage::Archive(storage));
    }
    match title_id {
        0x0100_0000_0000_0802 => Some(Storage::Bytes(mii_model_romfs())),
        0x0100_0000_0000_0823 => Some(Storage::Bytes(ng_word2_romfs())),
        _ => None,
    }
}

pub(super) fn dispatch_mount(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    command: u32,
) -> Option<Vec<u8>> {
    if command == 207 {
        return Some(build_ipc_response(ctx, UNSUPPORTED, &[], &[]));
    }
    let service = match command {
        2 => "IFileSystemApplication".to_string(),
        7 | 9 => {
            let offset = if command == 7 { 8 } else { 0 };
            let Some(title_id) = input::<8>(ctx, offset).map(u64::from_le_bytes) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            if title_id != kernel.title_id
                || (command == 7 && input::<4>(ctx, 0).map(u32::from_le_bytes) != Some(6))
            {
                return Some(build_ipc_response(ctx, CONTENT_NOT_FOUND, &[], &[]));
            }
            "IFileSystemApplication".to_string()
        }
        200 => "IFsStorage".to_string(),
        201 => {
            let Some(title_id) = input::<8>(ctx, 0).map(u64::from_le_bytes) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            if title_id != kernel.title_id {
                return Some(build_ipc_response(ctx, CONTENT_NOT_FOUND, &[], &[]));
            }
            "IFsStorage".to_string()
        }
        202 => {
            let Some(title_id) = input::<8>(ctx, 8).map(u64::from_le_bytes) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            if data_storage(kernel, title_id).is_none() {
                return Some(build_ipc_response(ctx, CONTENT_NOT_FOUND, &[], &[]));
            }
            format!("IFsStorageData:{title_id:016x}")
        }
        203 => {
            if kernel.patch_romfs.is_none() {
                return Some(build_ipc_response(ctx, CONTENT_NOT_FOUND, &[], &[]));
            }
            "IFsStoragePatch".to_string()
        }
        _ => return None,
    };
    Some(return_subsession(kernel, ctx, session_handle, &service))
}

fn application_storage(kernel: &Kernel) -> Storage<'_> {
    kernel
        .application_romfs
        .as_ref()
        .map(Storage::Application)
        .unwrap_or_else(|| Storage::Bytes(kernel.nro_romfs()))
}

fn application_metadata(kernel: &mut Kernel) -> Result<&[u8], String> {
    if kernel.application_romfs.is_none() {
        return Ok(kernel.nro_romfs());
    }
    if kernel.application_romfs_metadata.is_none() {
        let storage = application_storage(kernel);
        let mut metadata = storage.read(0, 0x50)?.into_owned();
        let header = super::romfs_header(&metadata).ok_or("invalid application RomFS header")?;
        for (offset, size) in [
            (header.dir_meta_off, header.dir_meta_size),
            (header.file_meta_off, header.file_meta_size),
        ] {
            if (offset as u64)
                .checked_add(size as u64)
                .filter(|end| *end <= storage.len())
                .is_none()
            {
                return Err("application RomFS metadata exceeds storage".into());
            }
            let table = storage.read(offset as u64, size)?;
            if table.len() != size {
                return Err("truncated application RomFS metadata".into());
            }
            metadata.extend_from_slice(&table);
        }
        metadata[0x18..0x20].copy_from_slice(&0x50u64.to_le_bytes());
        metadata[0x38..0x40]
            .copy_from_slice(&(0x50u64 + header.dir_meta_size as u64).to_le_bytes());
        kernel.application_romfs_metadata = Some(metadata);
    }
    Ok(kernel.application_romfs_metadata.as_deref().unwrap())
}

pub(super) fn application_entry_type(kernel: &mut Kernel, path: &str) -> Option<u32> {
    super::romfs_entry_type(application_metadata(kernel).ok()?, path)
}

pub(super) fn application_file(kernel: &mut Kernel, path: &str) -> Option<(usize, usize)> {
    let range = super::romfs_open_file(application_metadata(kernel).ok()?, path)?;
    (range.0 as u64)
        .checked_add(range.1 as u64)
        .filter(|end| *end <= application_storage(kernel).len())
        .map(|_| range)
}

pub(super) fn read_application(
    kernel: &Kernel,
    offset: u64,
    size: usize,
) -> Result<Cow<'_, [u8]>, String> {
    match kernel.application_romfs.as_ref() {
        Some(storage) => storage.read(offset, size).map(Cow::Owned),
        None => {
            let start = usize::try_from(offset).map_err(|_| "application offset overflow")?;
            let end = start.checked_add(size).ok_or("application read overflow")?;
            kernel
                .nro_romfs()
                .get(start..end)
                .map(Cow::Borrowed)
                .ok_or_else(|| "application read out of range".into())
        }
    }
}

fn directory_entries(
    kernel: &mut Kernel,
    path: &str,
    filter: u32,
) -> Option<Vec<(String, bool, u64)>> {
    let metadata = application_metadata(kernel).ok()?;
    let header = super::romfs_header(metadata)?;
    let mut directory = 0;
    for component in super::romfs_components(path)? {
        directory = super::romfs_child_dir(metadata, header, directory, component)?;
    }
    let directory = header.dir_meta_off.checked_add(directory as usize)?;
    let mut entries = Vec::new();
    for is_directory in [true, false] {
        let (first, table, table_size, record_size, name_offset, name_length, flag) =
            if is_directory {
                (
                    8,
                    header.dir_meta_off,
                    header.dir_meta_size,
                    0x18,
                    0x18,
                    0x14,
                    1,
                )
            } else {
                (
                    12,
                    header.file_meta_off,
                    header.file_meta_size,
                    0x20,
                    0x20,
                    0x1c,
                    2,
                )
            };
        if filter & flag == 0 {
            continue;
        }
        let mut child = super::romfs_u32(metadata, directory.checked_add(first)?)?;
        let mut seen = std::collections::HashSet::new();
        while child != u32::MAX {
            if !seen.insert(child) || (child as usize).checked_add(record_size)? > table_size {
                return None;
            }
            let at = table.checked_add(child as usize)?;
            let name = super::romfs_name(metadata, at, name_offset, name_length)?.to_string();
            let size = if is_directory {
                0
            } else {
                super::romfs_u64(metadata, at.checked_add(0x10)?)?
            };
            entries.push((name, is_directory, size));
            child = super::romfs_u32(metadata, at.checked_add(4)?)?;
        }
    }
    Some(entries)
}

pub(super) fn dispatch_file_system(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    port_name: &str,
    command: u32,
) -> Option<Vec<u8>> {
    if let Some(range) = port_name.strip_prefix("IFileApplication:") {
        let range = range.split_once(':').and_then(|(offset, size)| {
            Some((
                u64::from_str_radix(offset, 16).ok()?,
                u64::from_str_radix(size, 16).ok()?,
            ))
        });
        let Some((base, length)) = range else {
            return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
        };
        let result = match command {
            0 => {
                let (Some(offset), Some(size)) = (input::<8>(ctx, 8), input::<8>(ctx, 16)) else {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                };
                let offset = i64::from_le_bytes(offset);
                let size = i64::from_le_bytes(size);
                if offset < 0 || size < 0 {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                }
                let target = ctx
                    .recv_buffers
                    .iter()
                    .chain(ctx.recv_statics.iter())
                    .find(|buffer| buffer.addr != 0 && buffer.size != 0)
                    .copied();
                let Some(target) = target else {
                    return Some(build_ipc_response(
                        ctx,
                        if size == 0 { 0 } else { INVALID_INPUT },
                        &0u64.to_le_bytes(),
                        &[],
                    ));
                };
                let offset = (offset as u64).min(length);
                let count = (size as u64).min(target.size).min(length - offset);
                let Some(start) = base.checked_add(offset) else {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                };
                let Ok(count) = usize::try_from(count) else {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                };
                let read = match read_off_lock(kernel, port_name, start, count, target.addr, |kernel| {
                    application_storage(kernel).shared()
                }) {
                    off_lock::Outcome::Retry => return Some(Vec::new()),
                    off_lock::Outcome::Ready(result) => result.map(Cow::Owned),
                    off_lock::Outcome::Inline => read_application(kernel, start, count),
                };
                let Ok(bytes) = read else {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                };
                if !bytes.is_empty()
                    && kernel
                        .address_space
                        .write_checked(target.addr, &bytes)
                        .is_err()
                {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                }
                let read = bytes.len() as u64;
                off_lock::recycle(bytes);
                build_ipc_response(ctx, 0, &read.to_le_bytes(), &[])
            }
            2 => build_ipc_response(ctx, 0, &[], &[]),
            4 => build_ipc_response(ctx, 0, &length.to_le_bytes(), &[]),
            _ => build_ipc_response(ctx, INVALID_INPUT, &[], &[]),
        };
        return Some(result);
    }
    if port_name != "IFileSystemApplication" {
        return None;
    }
    let path = super::fs_read_path(ctx, &kernel.address_space);
    let result = match command {
        7 => match application_entry_type(kernel, &path) {
            Some(kind) => build_ipc_response(ctx, 0, &kind.to_le_bytes(), &[]),
            None => build_ipc_response(ctx, 0x202, &[], &[]),
        },
        8 => match application_file(kernel, &path) {
            Some((offset, size)) => {
                if input::<4>(ctx, 0).map(u32::from_le_bytes).unwrap_or(0) & !1 != 0 {
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                }
                return_subsession(
                    kernel,
                    ctx,
                    session_handle,
                    &format!("IFileApplication:{offset:x}:{size:x}"),
                )
            }
            None => build_ipc_response(ctx, 0x202, &[], &[]),
        },
        9 => {
            let filter = input::<4>(ctx, 0).map(u32::from_le_bytes).unwrap_or(0);
            let Some(entries) = directory_entries(kernel, &path, filter) else {
                return Some(build_ipc_response(ctx, 0x202, &[], &[]));
            };
            if kernel
                .sessions
                .get(&session_handle)
                .is_some_and(|session| session.is_domain)
            {
                let object_id = super::next_domain_object_id(kernel, session_handle);
                kernel
                    .open_dir_lists
                    .insert((session_handle, object_id), (entries, 0));
                return_subsession(kernel, ctx, session_handle, "IDirectory")
            } else {
                let handle = kernel.handles.create_handle(super::HandleType::Session);
                kernel
                    .sessions
                    .insert(handle, super::Session::new(handle, "IDirectory".into()));
                kernel.open_dir_lists.insert((handle, 0), (entries, 0));
                build_ipc_response(ctx, 0, &[], &[handle])
            }
        }
        10 => build_ipc_response(ctx, 0, &[], &[]),
        11 => build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[]),
        12 => build_ipc_response(
            ctx,
            0,
            &application_storage(kernel).len().to_le_bytes(),
            &[],
        ),
        _ => build_ipc_response(ctx, INVALID_INPUT, &[], &[]),
    };
    Some(result)
}

/// The storage behind an IFsStorage port: None when `port_name` is not a
/// storage, Some(None) when its content is missing.
fn storage_for<'a>(kernel: &'a Kernel, port_name: &str) -> Option<Option<Storage<'a>>> {
    Some(if port_name == "IFsStorage" {
        Some(application_storage(kernel))
    } else if port_name == "IFsStoragePatch" {
        kernel.patch_romfs.as_ref().map(Storage::Archive)
    } else if let Some(title_id) = port_name.strip_prefix("IFsStorageData:") {
        u64::from_str_radix(title_id, 16)
            .ok()
            .and_then(|title_id| data_storage(kernel, title_id))
    } else {
        return None;
    })
}

/// Fetches `size` bytes at `offset` for a read into guest `target` off the
/// kernel lock when the read is large (see `off_lock`). `Retry` means the
/// thread was parked to re-issue the request (`ipc_retry_pending` is set and
/// the caller returns an empty response); `Inline` means read it in place.
fn read_off_lock(
    kernel: &mut Kernel,
    port_name: &str,
    offset: u64,
    size: usize,
    target: u64,
    storage: impl FnOnce(&Kernel) -> Option<SharedStorage>,
) -> off_lock::Outcome {
    let request = off_lock::Request { port: port_name.to_string(), offset, size, target };
    let outcome = off_lock::read(kernel, request, storage);
    if matches!(outcome, off_lock::Outcome::Retry) {
        kernel.ipc_retry_pending = true;
    }
    outcome
}

pub(super) fn dispatch_storage(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    port_name: &str,
    command: u32,
) -> Option<Vec<u8>> {
    let storage = storage_for(kernel, port_name)?;
    let Some(storage) = storage else {
        return Some(build_ipc_response(ctx, CONTENT_NOT_FOUND, &[], &[]));
    };
    match command {
        0 => {
            let (Some(offset), Some(size)) = (input::<8>(ctx, 0), input::<8>(ctx, 8)) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            let offset = i64::from_le_bytes(offset);
            let size = i64::from_le_bytes(size);
            if offset < 0 || size < 0 {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            let Some(target) = ctx
                .recv_buffers
                .iter()
                .chain(ctx.recv_statics.iter())
                .find(|buffer| buffer.size != 0 && buffer.addr != 0)
                .copied()
            else {
                return Some(build_ipc_response(
                    ctx,
                    if size == 0 { 0 } else { INVALID_INPUT },
                    &[],
                    &[],
                ));
            };
            let offset = offset as u64;
            let size = size as u64;
            if offset > storage.len() || size > storage.len() - offset || size > target.size {
                log::debug!(
                    "{port_name}.Read rejected offset={offset:#x} size={size:#x} storage_len={:#x} target={:#x}",
                    storage.len(),
                    target.size
                );
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            log::debug!("{port_name}.Read offset={offset:#x} size={size:#x} target={:#x}", target.size);
            let Ok(amount) = usize::try_from(size) else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            let read = match read_off_lock(kernel, port_name, offset, amount, target.addr, |kernel| {
                storage_for(kernel, port_name).flatten().and_then(|storage| storage.shared())
            }) {
                off_lock::Outcome::Retry => return Some(Vec::new()),
                off_lock::Outcome::Ready(result) => result.map(Cow::Owned),
                off_lock::Outcome::Inline => storage_for(kernel, port_name)
                    .flatten()
                    .map_or_else(|| Err("storage is no longer mounted".into()), |storage| storage.read(offset, amount)),
            };
            let bytes = match read {
                Ok(bytes) => bytes,
                Err(error) => {
                    log::error!("{port_name}.Read failed: {error}");
                    return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
                }
            };
            if port_name.starts_with("IFsStorageData:") {
                log::debug!(
                    "{port_name}.Read head={:02x?} nonzero={}",
                    &bytes[..bytes.len().min(16)],
                    bytes.iter().filter(|b| **b != 0).count()
                );
            }
            if !bytes.is_empty()
                && kernel
                    .address_space
                    .write_checked(target.addr, &bytes)
                    .is_err()
            {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            let path = if fs_trace_enabled() && port_name == "IFsStorage" {
                romfs_path_for_data_offset(kernel.nro_romfs(), offset as usize)
                    .map(|(path, file_offset, _)| {
                        format!("{path}+{:#x}", offset as usize - file_offset)
                    })
                    .unwrap_or_else(|| "<romfs-meta>".to_string())
            } else {
                port_name.to_string()
            };
            fs_trace_read(
                "IFsStorage.Read",
                &path,
                offset as usize,
                offset as i64,
                size,
                bytes.len() as u64,
            );
            off_lock::recycle(bytes);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        2 => Some(build_ipc_response(ctx, 0, &[], &[])),
        4 => Some(build_ipc_response(
            ctx,
            0,
            &storage.len().to_le_bytes(),
            &[],
        )),
        5 => {
            let (Some(operation), Some(offset), Some(size)) =
                (input::<4>(ctx, 0), input::<8>(ctx, 8), input::<8>(ctx, 16))
            else {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            };
            let operation = u32::from_le_bytes(operation);
            let offset = i64::from_le_bytes(offset);
            let size = i64::from_le_bytes(size);
            if operation > 3 || offset < 0 || size < 0 {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            let offset = offset as u64;
            let size = size as u64;
            if offset > storage.len() || size > storage.len() - offset {
                return Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[]));
            }
            Some(build_ipc_response(ctx, 0, &[0; 0x40], &[]))
        }
        _ => Some(build_ipc_response(ctx, INVALID_INPUT, &[], &[])),
    }
}
