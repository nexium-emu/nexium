use super::*;
use nexium_ipc::{IpcCtx, CMIF_IN_MAGIC};
use nexium_memory::{AddressSpace, Perm};
use std::sync::Arc;

const BASE: u64 = 0x1000_0000_0000;
const IMAGE: u64 = BASE + 0x8000;
const BSS: u64 = BASE + 0xB000;

fn kernel() -> Kernel {
    let memory = Arc::new(AddressSpace::new());
    memory.map(BASE, 0x20000, Perm::RW, "ro_test").unwrap();
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
    kernel.aslr_base = BASE;
    kernel.aslr_size = 0x100_0000;
    kernel.address_space_end = BASE + 0x100_0000;
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

fn write_image(kernel: &Kernel, sizes: [u32; 3], bss_size: u32) {
    let mut header = [0u8; 0x40];
    header[0x10..0x14].copy_from_slice(b"NRO0");
    let total: u32 = sizes.iter().sum();
    header[0x18..0x1c].copy_from_slice(&total.to_le_bytes());
    let mut offset = 0u32;
    for (index, size) in sizes.iter().enumerate() {
        header[0x20 + index * 8..0x24 + index * 8].copy_from_slice(&offset.to_le_bytes());
        header[0x24 + index * 8..0x28 + index * 8].copy_from_slice(&size.to_le_bytes());
        offset += size;
    }
    header[0x38..0x3c].copy_from_slice(&bss_size.to_le_bytes());
    let mut image = vec![0u8; total as usize];
    image[..0x40].copy_from_slice(&header);
    for (index, byte) in image.iter_mut().enumerate().skip(0x40) {
        *byte = (index % 251) as u8;
    }
    kernel.address_space.write(IMAGE, &image).unwrap();
}

fn load_payload(image: u64, image_size: u64, bss: u64, bss_size: u64) -> Vec<u8> {
    let mut payload = vec![0; 40];
    payload[8..16].copy_from_slice(&image.to_le_bytes());
    payload[16..24].copy_from_slice(&image_size.to_le_bytes());
    payload[24..32].copy_from_slice(&bss.to_le_bytes());
    payload[32..40].copy_from_slice(&bss_size.to_le_bytes());
    payload
}

fn load(kernel: &mut Kernel, payload: &[u8]) -> Vec<u8> {
    dispatch(kernel, &mut request(0, payload), 0).unwrap()
}

#[test]
fn load_maps_segments_with_code_permissions_and_unload_releases_them() {
    let mut kernel = kernel();
    write_image(&kernel, [0x1000, 0x1000, 0x1000], 0x1000);
    let response = load(&mut kernel, &load_payload(IMAGE, 0x3000, BSS, 0x1000));
    assert_eq!(result(&response), 0);
    let base = u64::from_le_bytes(output(&response)[..8].try_into().unwrap());
    assert!(base >= BASE + 0x20000 && base + 0x4000 <= kernel.address_space_end);
    let regions = kernel.address_space.regions();
    let find = |va: u64| regions.iter().find(|region| region.base == va).unwrap();
    assert_eq!((find(base).perm, find(base).size), (Perm::RX, 0x1000));
    assert_eq!(find(base + 0x1000).perm, Perm::R);
    assert_eq!(find(base + 0x2000).perm, Perm::RW);
    assert_eq!(find(base + 0x3000).perm, Perm::RW);
    let info = |va: u64| super::super::memory_info_for_regions(&regions, va, kernel.address_space_end);
    assert_eq!((info(base).mem_type, info(base).perm), (0x08, Perm::RX.bits() as u32));
    assert_eq!(info(base + 0x1000).mem_type, 0x08);
    assert_eq!(info(base + 0x2000).mem_type, 0x09);
    assert_eq!(info(base + 0x3000).mem_type, 0x09);
    let mut source = vec![0u8; 0x3000];
    kernel.address_space.read(IMAGE, &mut source).unwrap();
    let mut mapped = vec![0u8; 0x3000];
    kernel.address_space.read(base, &mut mapped).unwrap();
    assert_eq!(source, mapped);
    kernel.address_space.write(base + 0x2010, b"relocated").unwrap();
    let mut through = [0u8; 9];
    kernel.address_space.read(IMAGE + 0x2010, &mut through).unwrap();
    assert_eq!(&through, b"relocated");
    kernel.address_space.write(BSS + 0x20, &[7; 4]).unwrap();
    let mut bss = [0u8; 4];
    kernel.address_space.read(base + 0x3020, &mut bss).unwrap();
    assert_eq!(bss, [7; 4]);
    assert_eq!(kernel.loaded_nros.len(), 1);
    assert_eq!(
        result(&load(&mut kernel, &load_payload(IMAGE, 0x3000, BSS, 0x1000))),
        ALREADY_LOADED
    );
    let mut unload = vec![0; 16];
    unload[8..].copy_from_slice(&base.to_le_bytes());
    assert_eq!(result(&dispatch(&mut kernel, &mut request(1, &unload), 1).unwrap()), 0);
    assert!(kernel.loaded_nros.is_empty());
    assert!(kernel
        .address_space
        .regions()
        .iter()
        .all(|region| region.base < base || region.base >= base + 0x4000));
    assert_eq!(
        result(&dispatch(&mut kernel, &mut request(1, &unload), 1).unwrap()),
        NOT_LOADED
    );
    kernel.address_space.read(IMAGE, &mut source).unwrap();
    assert_eq!(&source[0x2010..0x2019], b"relocated");
}

#[test]
fn malformed_images_and_arguments_are_rejected_without_mapping() {
    let mut kernel = kernel();
    write_image(&kernel, [0x1000, 0x1000, 0x1000], 0x1000);
    let before = kernel.address_space.regions().len();
    let end = kernel.address_space_end;
    let cases = [
        (IMAGE + 0x10, 0x3000, BSS, 0x1000, INVALID_ADDRESS),
        (IMAGE, 0x3010, BSS, 0x1000, INVALID_SIZE),
        (IMAGE, 0, BSS, 0x1000, INVALID_SIZE),
        (IMAGE, 0x3000, BSS + 1, 0x1000, INVALID_ADDRESS),
        (IMAGE, 0x3000, BSS, 0x10, INVALID_SIZE),
        (IMAGE, 0x2000, BSS, 0x1000, INVALID_NRO),
        (IMAGE, 0x3000, BSS, 0x2000, INVALID_NRO),
        (IMAGE, 0x3000, BSS, 0, INVALID_NRO),
        (end, 0x3000, BSS, 0x1000, INVALID_ADDRESS),
        (IMAGE, 0x3000, end, 0x1000, INVALID_ADDRESS),
    ];
    for (image, size, bss, bss_size, expected) in cases {
        let response = load(&mut kernel, &load_payload(image, size, bss, bss_size));
        assert_eq!(result(&response), expected, "{image:#x} {size:#x} {bss:#x} {bss_size:#x}");
    }
    kernel.address_space.write(IMAGE + 0x10, b"NRO1").unwrap();
    assert_eq!(
        result(&load(&mut kernel, &load_payload(IMAGE, 0x3000, BSS, 0x1000))),
        INVALID_NRO
    );
    kernel.address_space.write(IMAGE + 0x10, b"NRO0").unwrap();
    kernel.address_space.write(IMAGE + 0x28, &0x800u32.to_le_bytes()).unwrap();
    assert_eq!(
        result(&load(&mut kernel, &load_payload(IMAGE, 0x3000, BSS, 0x1000))),
        INVALID_NRO
    );
    assert_eq!(result(&load(&mut kernel, &[0; 8])), INVALID_NRO);
    assert_eq!(kernel.address_space.regions().len(), before);
    assert!(kernel.loaded_nros.is_empty());
}

#[test]
fn registration_commands_succeed_and_exhausted_address_space_is_reported() {
    let mut kernel = kernel();
    for command in [2, 3, 4, 10] {
        assert_eq!(
            result(&dispatch(&mut kernel, &mut request(command, &[0; 24]), command).unwrap()),
            0
        );
    }
    assert!(dispatch(&mut kernel, &mut request(11, &[]), 11).is_none());
    write_image(&kernel, [0x1000, 0, 0x1000], 0);
    kernel.aslr_size = 0x20000;
    assert_eq!(
        result(&load(&mut kernel, &load_payload(IMAGE, 0x2000, 0, 0))),
        OUT_OF_ADDRESS_SPACE
    );
    kernel.aslr_size = 0x100_0000;
    let response = load(&mut kernel, &load_payload(IMAGE, 0x2000, 0, 0));
    assert_eq!(result(&response), 0);
    let base = u64::from_le_bytes(output(&response)[..8].try_into().unwrap());
    let regions = kernel.address_space.regions();
    assert!(regions
        .iter()
        .any(|region| region.base == base && region.size == 0x1000 && region.perm == Perm::RX));
    assert!(regions
        .iter()
        .any(|region| region.base == base + 0x1000 && region.size == 0x1000 && region.perm == Perm::RW));
    assert!(regions.iter().all(|region| region.base != base + 0x2000));
}
