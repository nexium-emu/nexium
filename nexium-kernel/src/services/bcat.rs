use crate::kernel::svc::{build_ipc_response, domain_object_keys, indexed_ipc_buffer};
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const NAME_SIZE: usize = 0x20;
const DIGEST_SIZE: usize = 0x10;
const ENTRY_SIZE: usize = NAME_SIZE + 8 + DIGEST_SIZE;
const MAX_FILE: u64 = 64 * 1024 * 1024;
const RESULT_FAILED_OPEN_ENTITY: u32 = 122 | (2 << 9);
const RESULT_NO_OPEN_ENTITY: u32 = 122 | (7 << 9);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Object {
    Directory(PathBuf),
    File(PathBuf),
}

#[cfg(test)]
static ROOT_OVERRIDE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

pub fn package_root(title_id: u64) -> PathBuf {
    #[cfg(test)]
    if let Some(root) = ROOT_OVERRIDE.lock().ok().and_then(|slot| slot.clone()) {
        return root.join(format!("{title_id:016X}"));
    }
    nexium_common::paths::root()
        .join("nextendo")
        .join("bcat")
        .join(format!("{title_id:016X}"))
}

pub fn decode_name(bytes: &[u8]) -> Option<String> {
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(bytes.len());
    let name = std::str::from_utf8(&bytes[..end]).ok()?;
    let valid = !name.is_empty()
        && name.len() < NAME_SIZE
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
    valid.then(|| name.to_string())
}

pub fn encode_name(name: &str) -> [u8; NAME_SIZE] {
    let mut out = [0u8; NAME_SIZE];
    let bytes = name.as_bytes();
    let len = bytes.len().min(NAME_SIZE - 1);
    out[..len].copy_from_slice(&bytes[..len]);
    out
}

pub fn digest(data: &[u8]) -> [u8; DIGEST_SIZE] {
    let hash = Sha256::digest(data);
    let mut out = [0u8; DIGEST_SIZE];
    out.copy_from_slice(&hash[..DIGEST_SIZE]);
    out
}

fn sorted_entries(dir: &Path, directories: bool) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| if directories { path.is_dir() } else { path.is_file() })
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .and_then(|name| decode_name(name.as_bytes()))
                        .is_some()
                })
                .collect()
        })
        .unwrap_or_default();
    entries.sort();
    entries
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string()
}

pub fn directory_listing(root: &Path) -> Vec<u8> {
    sorted_entries(root, true)
        .iter()
        .flat_map(|dir| encode_name(&file_name(dir)))
        .collect()
}

pub fn directory_entries(dir: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    for file in sorted_entries(dir, false) {
        let size = std::fs::metadata(&file).map(|meta| meta.len()).unwrap_or(0);
        out.extend_from_slice(&encode_name(&file_name(&file)));
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&[0u8; DIGEST_SIZE]);
    }
    out
}

pub fn read_at(path: &Path, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if offset >= size {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut out = vec![0u8; len.min((size - offset) as usize)];
    file.read_exact(&mut out)?;
    Ok(out)
}

fn raw_input(ctx: &IpcCtx, offset: usize, len: usize) -> Option<&[u8]> {
    if ctx.cmif_in_data_len < offset + len {
        return None;
    }
    ctx.buf.get(ctx.cmif_in_data_off + offset..ctx.cmif_in_data_off + offset + len)
}

fn object_id(ctx: &IpcCtx) -> u32 {
    ctx.domain.map(|domain| domain.object_id).unwrap_or(0)
}

fn current(kernel: &Kernel, ctx: &IpcCtx, session_handle: u32) -> Option<Object> {
    domain_object_keys(kernel, session_handle, object_id(ctx))
        .into_iter()
        .find_map(|key| kernel.bcat_objects.get(&key).cloned())
}

fn remember(kernel: &mut Kernel, ctx: &IpcCtx, session_handle: u32, object: Object) {
    for key in domain_object_keys(kernel, session_handle, object_id(ctx)) {
        kernel.bcat_objects.insert(key, object.clone());
    }
}

fn write_records(kernel: &Kernel, ctx: &IpcCtx, records: &[u8], record_size: usize) -> Option<usize> {
    let Some(output) = indexed_ipc_buffer(&ctx.recv_buffers, &ctx.recv_statics, 0) else {
        return Some(0);
    };
    let fit = (output.size as usize / record_size).min(records.len() / record_size);
    if fit > 0 {
        kernel
            .address_space
            .write(output.addr, &records[..fit * record_size])
            .ok()?;
    }
    Some(fit)
}

pub(crate) fn dispatch(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session_handle: u32,
    port_name: &str,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    let root = package_root(kernel.title_id);
    let ok = |ctx: &IpcCtx, data: &[u8]| Some(build_ipc_response(ctx, 0, data, &[]));
    let fail = |ctx: &IpcCtx, rc: u32| Some(build_ipc_response(ctx, rc, &[], &[]));
    match (port_name, cmd_id) {
        ("IDeliveryCacheStorageService", 10) => {
            if !root.is_dir() {
                return None;
            }
            let listing = directory_listing(&root);
            let count = write_records(kernel, ctx, &listing, NAME_SIZE)?;
            log::info!("bcat: {} delivery cache directories served from the Nextendo package", count);
            ok(ctx, &(count as u32).to_le_bytes())
        }
        ("IDeliveryCacheDirectoryService", 0) => {
            let name = raw_input(ctx, 0, NAME_SIZE).and_then(decode_name);
            let dir = name.as_ref().map(|name| root.join(name));
            match dir.filter(|dir| dir.is_dir()) {
                Some(dir) => {
                    log::info!("bcat: opened directory {} from the Nextendo package", name.unwrap_or_default());
                    remember(kernel, ctx, session_handle, Object::Directory(dir));
                    ok(ctx, &[])
                }
                None => {
                    log::debug!("bcat: directory {:?} is not in the Nextendo package", name);
                    fail(ctx, RESULT_FAILED_OPEN_ENTITY)
                }
            }
        }
        ("IDeliveryCacheDirectoryService", 1 | 2) => {
            let Some(Object::Directory(dir)) = current(kernel, ctx, session_handle) else {
                return fail(ctx, RESULT_NO_OPEN_ENTITY);
            };
            let entries = directory_entries(&dir);
            let count = if cmd_id == 1 {
                write_records(kernel, ctx, &entries, ENTRY_SIZE)?
            } else {
                entries.len() / ENTRY_SIZE
            };
            ok(ctx, &(count as u32).to_le_bytes())
        }
        ("IDeliveryCacheFileService", 0) => {
            let dir = raw_input(ctx, 0, NAME_SIZE).and_then(decode_name);
            let file = raw_input(ctx, NAME_SIZE, NAME_SIZE).and_then(decode_name);
            let path = match (&dir, &file) {
                (Some(dir), Some(file)) => Some(root.join(dir).join(file)),
                _ => None,
            };
            match path.filter(|path| path.is_file()) {
                Some(path) => {
                    log::info!(
                        "bcat: opened {}/{} from the Nextendo package",
                        dir.unwrap_or_default(),
                        file.unwrap_or_default()
                    );
                    remember(kernel, ctx, session_handle, Object::File(path));
                    ok(ctx, &[])
                }
                None => {
                    log::debug!("bcat: file {:?}/{:?} is not in the Nextendo package", dir, file);
                    fail(ctx, RESULT_FAILED_OPEN_ENTITY)
                }
            }
        }
        ("IDeliveryCacheFileService", 1) => {
            let Some(Object::File(path)) = current(kernel, ctx, session_handle) else {
                return fail(ctx, RESULT_NO_OPEN_ENTITY);
            };
            let offset = raw_input(ctx, 0, 8)
                .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                .unwrap_or(0);
            let Some(output) = indexed_ipc_buffer(&ctx.recv_buffers, &ctx.recv_statics, 0) else {
                return ok(ctx, &0u64.to_le_bytes());
            };
            let data = match read_at(&path, offset, (output.size as usize).min(MAX_FILE as usize)) {
                Ok(data) => data,
                Err(error) => {
                    log::warn!("bcat: reading {} failed: {error}", path.display());
                    return fail(ctx, RESULT_NO_OPEN_ENTITY);
                }
            };
            if kernel.address_space.write(output.addr, &data).is_err() {
                return fail(ctx, RESULT_NO_OPEN_ENTITY);
            }
            ok(ctx, &(data.len() as u64).to_le_bytes())
        }
        ("IDeliveryCacheFileService", 2) => {
            let Some(Object::File(path)) = current(kernel, ctx, session_handle) else {
                return fail(ctx, RESULT_NO_OPEN_ENTITY);
            };
            let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            ok(ctx, &size.to_le_bytes())
        }
        ("IDeliveryCacheFileService", 3) => {
            let Some(Object::File(path)) = current(kernel, ctx, session_handle) else {
                return fail(ctx, RESULT_NO_OPEN_ENTITY);
            };
            let data = std::fs::read(&path).unwrap_or_default();
            ok(ctx, &digest(&data))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::services::resolver::harness::Harness;
    use super::*;

    #[test]
    fn delivery_cache_services_serve_the_package() {
        const TITLE: u64 = 0x0100_F8F0_000A_2000;
        let root = std::env::temp_dir().join(format!("nexium-bcat-dispatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let package = root.join(format!("{TITLE:016X}"));
        std::fs::create_dir_all(package.join("vsdata")).unwrap();
        std::fs::create_dir_all(package.join("coopdata")).unwrap();
        std::fs::write(package.join("vsdata").join("VSSetting_0.byaml"), b"first").unwrap();
        std::fs::write(package.join("coopdata").join("CoopSetting.byaml"), b"coop!").unwrap();
        *ROOT_OVERRIDE.lock().unwrap() = Some(root.clone());

        let mut h = Harness::new("IDeliveryCacheStorageService");
        h.kernel.title_id = TITLE;
        let listing = h.output(0x40);
        let mut ctx = h.request(10, &[]);
        ctx.recv_buffers.push(listing);
        let response = dispatch(&mut h.kernel, &mut ctx, h.session, "IDeliveryCacheStorageService", 10).unwrap();
        assert_eq!(Harness::result(&response), 0);
        assert_eq!(&Harness::data(&response)[..4], &2u32.to_le_bytes());
        let names = h.read(listing);
        assert_eq!(&names[..8], b"coopdata");
        assert_eq!(&names[0x20..0x26], b"vsdata");

        let dir = h.session_for("IDeliveryCacheDirectoryService");
        let mut missing = h.request(0, &encode_name("nope"));
        let response = dispatch(&mut h.kernel, &mut missing, dir, "IDeliveryCacheDirectoryService", 0).unwrap();
        assert_eq!(Harness::result(&response), RESULT_FAILED_OPEN_ENTITY);
        let mut open = h.request(0, &encode_name("vsdata"));
        let response = dispatch(&mut h.kernel, &mut open, dir, "IDeliveryCacheDirectoryService", 0).unwrap();
        assert_eq!(Harness::result(&response), 0);
        let mut count = h.request(2, &[]);
        let response = dispatch(&mut h.kernel, &mut count, dir, "IDeliveryCacheDirectoryService", 2).unwrap();
        assert_eq!(&Harness::data(&response)[..4], &1u32.to_le_bytes());
        let entries = h.output(2 * ENTRY_SIZE as u64);
        let mut read = h.request(1, &[]);
        read.recv_buffers.push(entries);
        let response = dispatch(&mut h.kernel, &mut read, dir, "IDeliveryCacheDirectoryService", 1).unwrap();
        assert_eq!(&Harness::data(&response)[..4], &1u32.to_le_bytes());
        let entries = h.read(entries);
        assert_eq!(&entries[..17], b"VSSetting_0.byaml");
        assert_eq!(&entries[NAME_SIZE..NAME_SIZE + 8], &5u64.to_le_bytes());

        let file = h.session_for("IDeliveryCacheFileService");
        let mut early = h.request(2, &[]);
        let response = dispatch(&mut h.kernel, &mut early, file, "IDeliveryCacheFileService", 2).unwrap();
        assert_eq!(Harness::result(&response), RESULT_NO_OPEN_ENTITY);
        let mut raw = encode_name("vsdata").to_vec();
        raw.extend_from_slice(&encode_name("VSSetting_0.byaml"));
        let mut open = h.request(0, &raw);
        let response = dispatch(&mut h.kernel, &mut open, file, "IDeliveryCacheFileService", 0).unwrap();
        assert_eq!(Harness::result(&response), 0);
        let mut size = h.request(2, &[]);
        let response = dispatch(&mut h.kernel, &mut size, file, "IDeliveryCacheFileService", 2).unwrap();
        assert_eq!(&Harness::data(&response)[..8], &5u64.to_le_bytes());
        let data = h.output(16);
        let mut read = h.request(1, &1u64.to_le_bytes());
        read.recv_buffers.push(data);
        let response = dispatch(&mut h.kernel, &mut read, file, "IDeliveryCacheFileService", 1).unwrap();
        assert_eq!(&Harness::data(&response)[..8], &4u64.to_le_bytes());
        assert_eq!(&h.read(data)[..4], b"irst");
        let mut hash = h.request(3, &[]);
        let response = dispatch(&mut h.kernel, &mut hash, file, "IDeliveryCacheFileService", 3).unwrap();
        assert_eq!(&Harness::data(&response)[..DIGEST_SIZE], &digest(b"first"));
        let mut wrong = h.request(0, &encode_name("vsdata"));
        let response = dispatch(&mut h.kernel, &mut wrong, file, "IDeliveryCacheFileService", 0).unwrap();
        assert_eq!(Harness::result(&response), RESULT_FAILED_OPEN_ENTITY);

        *ROOT_OVERRIDE.lock().unwrap() = None;
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn names_are_nul_terminated_ascii_without_traversal() {
        assert_eq!(decode_name(b"vsdata\0junk"), Some("vsdata".into()));
        assert_eq!(decode_name(b"VSSetting_0.byaml"), Some("VSSetting_0.byaml".into()));
        assert_eq!(decode_name(b".."), None);
        assert_eq!(decode_name(b"a/b"), None);
        assert_eq!(decode_name(b""), None);
        assert_eq!(decode_name(&[b'a'; 32]), None);
        let encoded = encode_name("coopdata");
        assert_eq!(&encoded[..8], b"coopdata");
        assert_eq!(encoded[8], 0);
        assert_eq!(encode_name(&"x".repeat(40))[NAME_SIZE - 1], 0);
    }

    #[test]
    fn listings_are_sorted_fixed_records() {
        let root = std::env::temp_dir().join(format!("nexium-bcat-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("vsdata")).unwrap();
        std::fs::create_dir_all(root.join("coopdata")).unwrap();
        std::fs::write(root.join("vsdata").join("VSSetting_1.byaml"), b"second").unwrap();
        std::fs::write(root.join("vsdata").join("VSSetting_0.byaml"), b"first").unwrap();
        std::fs::write(root.join("stray.txt"), b"ignored").unwrap();
        let listing = directory_listing(&root);
        assert_eq!(listing.len(), 2 * NAME_SIZE);
        assert_eq!(&listing[..8], b"coopdata");
        assert_eq!(&listing[NAME_SIZE..NAME_SIZE + 6], b"vsdata");
        let entries = directory_entries(&root.join("vsdata"));
        assert_eq!(entries.len(), 2 * ENTRY_SIZE);
        assert_eq!(&entries[..17], b"VSSetting_0.byaml");
        assert_eq!(&entries[NAME_SIZE..NAME_SIZE + 8], &5u64.to_le_bytes());
        assert_eq!(&entries[ENTRY_SIZE..ENTRY_SIZE + 17], b"VSSetting_1.byaml");
        assert_eq!(read_at(&root.join("vsdata").join("VSSetting_0.byaml"), 1, 3).unwrap(), b"irs");
        assert_eq!(read_at(&root.join("vsdata").join("VSSetting_0.byaml"), 9, 3).unwrap(), b"");
        assert_eq!(digest(b"first"), digest(b"first"));
        assert_ne!(digest(b"first"), digest(b"second"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
