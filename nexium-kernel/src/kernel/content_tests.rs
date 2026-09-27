use super::*;
use memmap2::MmapOptions;
use nexium_ipc::{DomainIn, IpcBuffer, IpcCtx, CMIF_IN_MAGIC};
use nexium_memory::{AddressSpace, Perm};
use std::sync::Arc;

const BASE: u64 = 0x1000_0000_0000;
const TITLE: u64 = 0x0100_1234_5678_0000;

fn kernel() -> Kernel {
    let memory = Arc::new(AddressSpace::new());
    memory.map(BASE, 0x20000, Perm::RW, "content_test").unwrap();
    let mut kernel = Kernel::new(
        memory,
        BASE,
        0x1000,
        BASE + 0x10000,
        0x1000,
        BASE + 0x18000,
        0x1000,
        BASE + 0x19000,
        BASE + 0x1a000,
    );
    kernel.title_id = TITLE;
    kernel
}

fn request(command: u32, payload: &[u8]) -> IpcCtx {
    let mut bytes = vec![0; 0x100];
    bytes[..4].copy_from_slice(&4u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&((16 + payload.len().div_ceil(4) * 4) as u32 / 4).to_le_bytes());
    bytes[0x10..0x14].copy_from_slice(&CMIF_IN_MAGIC.to_le_bytes());
    bytes[0x18..0x1c].copy_from_slice(&command.to_le_bytes());
    bytes[0x20..0x20 + payload.len()].copy_from_slice(payload);
    let mut ctx = IpcCtx::parse(bytes, false).unwrap();
    ctx.cmif_in_data_len = payload.len();
    ctx
}

fn result(response: &[u8]) -> u32 {
    let start = response
        .windows(4)
        .position(|value| value == b"SFCO")
        .unwrap();
    u32::from_le_bytes(response[start + 8..start + 12].try_into().unwrap())
}

fn output(response: &[u8]) -> &[u8] {
    let start = response
        .windows(4)
        .position(|value| value == b"SFCO")
        .unwrap();
    &response[start + 16..]
}

fn storage(bytes: &[u8]) -> LazyRomfs {
    let mut mmap = MmapOptions::new().len(bytes.len() + 8).map_anon().unwrap();
    mmap[..8].fill(0xA5);
    mmap[8..].copy_from_slice(bytes);
    LazyRomfs::from_range(Arc::new(mmap.make_read_only().unwrap()), 8..8 + bytes.len()).unwrap()
}

fn receiver(ctx: &mut IpcCtx, size: u64) {
    ctx.recv_buffers.push(IpcBuffer {
        addr: BASE + 0x1000,
        size,
        mode: 0,
    });
}

fn session(kernel: &mut Kernel, domain: bool) -> (u32, Option<DomainIn>) {
    let handle = kernel
        .handles
        .create_handle(super::super::HandleType::Session);
    let mut session = super::super::Session::new(handle, "fsp-srv".into());
    if domain {
        session.convert_to_domain();
    }
    kernel.sessions.insert(handle, session);
    let domain = domain.then_some(DomainIn {
        kind: 1,
        object_id: 1,
        num_in_objects: 0,
        data_size: 16,
    });
    (handle, domain)
}

fn returned_object(
    kernel: &Kernel,
    handle: u32,
    domain: Option<DomainIn>,
    response: &[u8],
) -> (u32, Option<DomainIn>, String) {
    assert_eq!(result(response), 0);
    if let Some(mut domain) = domain {
        domain.object_id = u32::from_le_bytes(output(response)[..4].try_into().unwrap());
        let name = kernel.sessions[&handle]
            .service_for_object(domain.object_id)
            .unwrap()
            .to_string();
        (handle, Some(domain), name)
    } else {
        let handle = u32::from_le_bytes(response[12..16].try_into().unwrap());
        (handle, None, kernel.sessions[&handle].port_name.clone())
    }
}

#[test]
fn enabled_add_on_enumeration_is_sorted_paged_bounded_and_includes_unlocks() {
    let mut kernel = kernel();
    kernel
        .add_on_content
        .insert(TITLE + 0x1007, Some(storage(b"payload")));
    kernel.add_on_content.insert(TITLE + 0x1002, None);
    kernel.add_on_content.insert(TITLE + 0x1004, None);
    kernel.add_on_content.insert(TITLE + 0x3001, None);
    assert_eq!(add_on_indices(&kernel, TITLE), vec![2, 4, 7]);
    for command in [0, 2] {
        let response = dispatch_add_on_content(
            &mut kernel,
            &mut request(command, &TITLE.to_le_bytes()),
            command,
        )
        .unwrap();
        assert_eq!(result(&response), 0);
        assert_eq!(&output(&response)[..4], &3u32.to_le_bytes());
    }
    for command in [1, 3] {
        kernel
            .address_space
            .write(BASE + 0x1000, &[0xA5; 16])
            .unwrap();
        let mut payload = Vec::from(1u32.to_le_bytes());
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&TITLE.to_le_bytes());
        let mut ctx = request(command, &payload);
        receiver(&mut ctx, 6);
        let response = dispatch_add_on_content(&mut kernel, &mut ctx, command).unwrap();
        assert_eq!(&output(&response)[..4], &1u32.to_le_bytes());
        let mut actual = [0; 16];
        kernel
            .address_space
            .read(BASE + 0x1000, &mut actual)
            .unwrap();
        assert_eq!(&actual[..4], &4u32.to_le_bytes());
        assert_eq!(&actual[4..], &[0xA5; 12]);
        ctx.buf[ctx.cmif_in_data_off..ctx.cmif_in_data_off + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        let response = dispatch_add_on_content(&mut kernel, &mut ctx, command).unwrap();
        assert_eq!(&output(&response)[..4], &0u32.to_le_bytes());
    }
    let other = TITLE + 0x2000;
    let response =
        dispatch_add_on_content(&mut kernel, &mut request(0, &other.to_le_bytes()), 0).unwrap();
    assert_eq!(&output(&response)[..4], &0u32.to_le_bytes());
    assert_eq!(
        result(
            &dispatch_add_on_content(&mut kernel, &mut request(7, &2u32.to_le_bytes()), 7).unwrap()
        ),
        0
    );
    assert_ne!(
        result(
            &dispatch_add_on_content(&mut kernel, &mut request(7, &9u32.to_le_bytes()), 7).unwrap()
        ),
        0
    );
}

#[test]
fn archive_and_patch_mounts_keep_independent_storage_in_both_session_modes() {
    let mut kernel = kernel();
    kernel.application_romfs = Some(AppRomfs::Plain(storage(b"base")));
    kernel
        .system_romfs
        .insert(0x0100_0000_0000_0801, storage(b"system"));
    kernel
        .add_on_content
        .insert(TITLE + 0x1001, Some(storage(b"first-dlc")));
    kernel
        .add_on_content
        .insert(TITLE + 0x1002, Some(storage(b"second-dlc")));
    kernel.add_on_content.insert(TITLE + 0x1003, None);
    kernel.patch_romfs = Some(storage(b"resolved-patch"));
    for domain in [false, true] {
        let (handle, domain) = session(&mut kernel, domain);
        for (command, title_id, expected) in [
            (200, TITLE, &b"base"[..]),
            (202, 0x0100_0000_0000_0801, &b"system"[..]),
            (202, TITLE + 0x1001, &b"first-dlc"[..]),
            (202, TITLE + 0x1002, &b"second-dlc"[..]),
            (203, TITLE, &b"resolved-patch"[..]),
        ] {
            let mut payload = vec![0; 16];
            payload[8..].copy_from_slice(&title_id.to_le_bytes());
            let mut ctx = request(command, &payload);
            ctx.domain = domain;
            let response = dispatch_mount(&mut kernel, &mut ctx, handle, command).unwrap();
            let (_, object, port) = returned_object(&kernel, handle, domain, &response);
            let mut size_ctx = request(4, &[]);
            size_ctx.domain = object;
            let response = dispatch_storage(&mut kernel, &mut size_ctx, &port, 4).unwrap();
            assert_eq!(
                &output(&response)[..8],
                &(expected.len() as u64).to_le_bytes()
            );
            let mut payload = vec![0; 16];
            payload[8..].copy_from_slice(&(expected.len() as u64).to_le_bytes());
            let mut read = request(0, &payload);
            read.domain = object;
            receiver(&mut read, 64);
            let response = dispatch_storage(&mut kernel, &mut read, &port, 0).unwrap();
            assert_eq!(result(&response), 0);
            let mut actual = vec![0; expected.len()];
            kernel
                .address_space
                .read(BASE + 0x1000, &mut actual)
                .unwrap();
            assert_eq!(actual, expected);
        }
        for title in [TITLE + 0x1003, TITLE + 0x1004] {
            let mut payload = vec![0; 16];
            payload[8..].copy_from_slice(&title.to_le_bytes());
            assert_ne!(
                result(
                    &dispatch_mount(&mut kernel, &mut request(202, &payload), handle, 202).unwrap()
                ),
                0
            );
        }
    }
    kernel.patch_romfs = None;
    assert_ne!(
        result(&dispatch_mount(&mut kernel, &mut request(203, &[]), 0, 203).unwrap()),
        0
    );
    assert_ne!(
        result(&dispatch_mount(&mut kernel, &mut request(207, &[]), 0, 207).unwrap()),
        0
    );
}

#[test]
fn storage_reads_validate_full_requests_and_query_range_returns_fs_abi_size() {
    let mut kernel = kernel();
    kernel.application_romfs = Some(AppRomfs::Plain(storage(b"data")));
    kernel.address_space.write(BASE + 0x1000, &[0xAA; 8]).unwrap();

    let mut payload = vec![0; 16];
    payload[8..].copy_from_slice(&5u64.to_le_bytes());
    let mut read = request(0, &payload);
    receiver(&mut read, 8);
    assert_ne!(
        result(&dispatch_storage(&mut kernel, &mut read, "IFsStorage", 0).unwrap()),
        0
    );
    let mut actual = [0; 8];
    kernel.address_space.read(BASE + 0x1000, &mut actual).unwrap();
    assert_eq!(actual, [0xAA; 8]);

    payload[8..].copy_from_slice(&4u64.to_le_bytes());
    let mut read = request(0, &payload);
    receiver(&mut read, 3);
    assert_ne!(
        result(&dispatch_storage(&mut kernel, &mut read, "IFsStorage", 0).unwrap()),
        0
    );
    kernel.address_space.read(BASE + 0x1000, &mut actual).unwrap();
    assert_eq!(actual, [0xAA; 8]);

    let mut query = vec![0; 24];
    query[..4].copy_from_slice(&3u32.to_le_bytes());
    query[16..].copy_from_slice(&4u64.to_le_bytes());
    let response = dispatch_storage(
        &mut kernel,
        &mut request(5, &query),
        "IFsStorage",
        5,
    )
    .unwrap();
    assert_eq!(result(&response), 0);
    assert_eq!(output(&response).len(), 0x40);

    query[8..16].copy_from_slice(&4u64.to_le_bytes());
    query[16..24].copy_from_slice(&1u64.to_le_bytes());
    assert_ne!(
        result(&dispatch_storage(
            &mut kernel,
            &mut request(5, &query),
            "IFsStorage",
            5,
        ).unwrap()),
        0
    );
}

#[test]
fn application_file_and_directory_reads_use_effective_romfs_in_both_session_modes() {
    let mut kernel = kernel();
    let base = super::super::build_flat_romfs(vec![("change.txt".into(), b"old".to_vec())]);
    let update = super::super::build_flat_romfs(vec![
        ("change.txt".into(), b"updated".to_vec()),
        ("added.txt".into(), b"new file".to_vec()),
    ]);
    let base = storage(&base);
    kernel.nro_mmap = Some(base.mmap.clone());
    kernel.nro_romfs_range = Some(base.range.clone());
    kernel.application_romfs = Some(AppRomfs::Plain(storage(&update)));
    for domain in [false, true] {
        let (handle, domain) = session(&mut kernel, domain);
        kernel
            .address_space
            .write(BASE + 0x100, b"/added.txt\0")
            .unwrap();
        let mut ctx = request(8, &1u32.to_le_bytes());
        ctx.domain = domain;
        ctx.send_statics.push(IpcBuffer {
            addr: BASE + 0x100,
            size: 11,
            mode: 0,
        });
        let response =
            dispatch_file_system(&mut kernel, &mut ctx, handle, "IFileSystemApplication", 8)
                .unwrap();
        let (file_handle, object, port) = returned_object(&kernel, handle, domain, &response);
        let mut payload = vec![0; 24];
        payload[16..].copy_from_slice(&8u64.to_le_bytes());
        let mut read = request(0, &payload);
        read.domain = object;
        receiver(&mut read, 8);
        let response = dispatch_file_system(&mut kernel, &mut read, file_handle, &port, 0).unwrap();
        assert_eq!(result(&response), 0);
        assert_eq!(&output(&response)[..8], &8u64.to_le_bytes());
        let mut actual = [0; 8];
        kernel
            .address_space
            .read(BASE + 0x1000, &mut actual)
            .unwrap();
        assert_eq!(&actual, b"new file");
        kernel.address_space.write(BASE + 0x100, b"/\0").unwrap();
        let mut ctx = request(9, &3u32.to_le_bytes());
        ctx.domain = domain;
        ctx.send_statics.push(IpcBuffer {
            addr: BASE + 0x100,
            size: 2,
            mode: 0,
        });
        let response =
            dispatch_file_system(&mut kernel, &mut ctx, handle, "IFileSystemApplication", 9)
                .unwrap();
        let (directory_handle, object, port) = returned_object(&kernel, handle, domain, &response);
        let mut ctx = request(1, &[]);
        ctx.domain = object;
        let response = super::super::dispatch_service_v2(
            &mut kernel,
            &port,
            &mut ctx,
            directory_handle,
            &mut Vec::new(),
        );
        assert_eq!(result(&response), 0);
        assert_eq!(&output(&response)[..8], &2u64.to_le_bytes());
    }
    let range = application_file(&mut kernel, "/change.txt").unwrap();
    assert_eq!(
        &*read_application(&kernel, range.0 as u64, range.1).unwrap(),
        b"updated"
    );
    assert!(application_file(&mut kernel, "/../change.txt").is_none());
}

#[test]
fn malformed_requests_and_metadata_do_not_fabricate_content() {
    let mut kernel = kernel();
    assert_ne!(
        result(&dispatch_mount(&mut kernel, &mut request(202, &[0; 8]), 0, 202).unwrap()),
        0
    );
    assert_ne!(
        result(&dispatch_add_on_content(&mut kernel, &mut request(1, &[0; 8]), 1).unwrap()),
        0
    );
    kernel.application_romfs = Some(AppRomfs::Plain(storage(&[0; 8])));
    assert!(application_entry_type(&mut kernel, "/").is_none());
    let mut payload = vec![0; 16];
    payload[..8].copy_from_slice(&(-1i64).to_le_bytes());
    let mut ctx = request(0, &payload);
    receiver(&mut ctx, 8);
    assert_ne!(
        result(&dispatch_storage(&mut kernel, &mut ctx, "IFsStorage", 0).unwrap()),
        0
    );
}

#[test]
fn application_metadata_and_file_reads_include_layered_mods() {
    let mut kernel = kernel();
    let original = super::super::build_flat_romfs(vec![("change.txt".into(), b"updated".to_vec())]);
    let base = storage(&original);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("nexium-content-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("change.txt"), b"mod override").unwrap();
    std::fs::write(directory.join("mod-only.txt"), b"added by mod").unwrap();
    let layered = nexium_loader::LayeredRomfs::build(Some(&base), &directory).unwrap();
    kernel.application_romfs = Some(AppRomfs::Layered(Arc::new(layered)));
    for (name, expected) in [
        ("/change.txt", &b"mod override"[..]),
        ("/mod-only.txt", &b"added by mod"[..]),
    ] {
        let (offset, size) = application_file(&mut kernel, name).unwrap();
        assert_eq!(
            &*read_application(&kernel, offset as u64, size).unwrap(),
            expected
        );
    }
    assert_eq!(directory_entries(&mut kernel, "/", 2).unwrap().len(), 2);
    drop(kernel);
    std::fs::remove_dir_all(directory).unwrap();
}
