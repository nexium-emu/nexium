use super::{build_ipc_response, ipc, sync_process_alias_cpu, Kernel};
use nexium_memory::Perm;

#[cfg(test)]
#[path = "ro_tests.rs"]
mod tests;

const PAGE: u64 = 0x1000;
const OUT_OF_ADDRESS_SPACE: u32 = 0x416;
const ALREADY_LOADED: u32 = 0x616;
const INVALID_NRO: u32 = 0x816;
const INVALID_ADDRESS: u32 = 0x8_0216;
const INVALID_SIZE: u32 = 0x8_0416;
const NOT_LOADED: u32 = 0x8_0816;
const INVALID_PROCESS: u32 = 0x8_0E16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadedNro {
    pub image: u64,
    pub bss: u64,
    pub bss_size: u64,
    pub segments: [(u64, u64); 3],
}

impl LoadedNro {
    fn image_size(&self) -> u64 {
        self.segments.iter().map(|(_, size)| size).sum()
    }
}

fn input(ctx: &ipc::IpcCtx, offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    if end > ctx.cmif_in_data_len {
        return None;
    }
    let start = ctx.cmif_in_data_off.checked_add(offset)?;
    let bytes: [u8; 8] = ctx.buf.get(start..start.checked_add(8)?)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn aligned(value: u64) -> bool {
    value % PAGE == 0
}

fn parse_header(header: &[u8; 0x40], image_size: u64, bss_size: u64) -> Result<[(u64, u64); 3], u32> {
    let field = |offset: usize| {
        u64::from(u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap()))
    };
    if &header[0x10..0x14] != b"NRO0" || field(0x18) != image_size || field(0x38) != bss_size {
        return Err(INVALID_NRO);
    }
    let segments = [
        (field(0x20), field(0x24)),
        (field(0x28), field(0x2c)),
        (field(0x30), field(0x34)),
    ];
    let mut expected = 0;
    for (offset, size) in segments {
        if offset != expected || !aligned(size) {
            return Err(INVALID_NRO);
        }
        expected = offset.checked_add(size).ok_or(INVALID_NRO)?;
    }
    if expected != image_size {
        return Err(INVALID_NRO);
    }
    Ok(segments)
}

fn free_range(kernel: &Kernel, size: u64) -> Option<u64> {
    let start = kernel.aslr_base.max(PAGE);
    let limit = kernel
        .aslr_base
        .checked_add(kernel.aslr_size)?
        .min(kernel.address_space_end);
    let mut regions = kernel.address_space.regions();
    regions.sort_by_key(|region| region.base);
    let after_code = kernel
        .code_base
        .checked_add(kernel.code_size)?
        .checked_add(PAGE - 1)?
        & !(PAGE - 1);
    [after_code.max(start), start].into_iter().find_map(|lower| {
        let mut cursor = lower;
        for region in &regions {
            let end = region.base.checked_add(region.size)?;
            if end <= cursor {
                continue;
            }
            if region.base >= cursor.checked_add(size)? {
                break;
            }
            cursor = end.checked_add(PAGE - 1)? & !(PAGE - 1);
        }
        (cursor.checked_add(size)? <= limit).then_some(cursor)
    })
}

fn pieces(base: u64, loaded: &LoadedNro) -> Vec<(u64, u64, u64, Perm, &'static str)> {
    let kinds = [
        ("aliascode_nro_text", Perm::RX),
        ("aliascode_nro_rodata", Perm::R),
        ("aliascodedata_nro_data", Perm::RW),
    ];
    let mut pieces: Vec<_> = loaded
        .segments
        .iter()
        .zip(kinds)
        .filter(|((_, size), _)| *size != 0)
        .map(|((offset, size), (name, perm))| {
            (base + offset, loaded.image + offset, *size, perm, name)
        })
        .collect();
    if loaded.bss_size != 0 {
        pieces.push((
            base + loaded.image_size(),
            loaded.bss,
            loaded.bss_size,
            Perm::RW,
            "aliascodedata_nro_bss",
        ));
    }
    pieces
}

fn map_nro(kernel: &mut Kernel, image: u64, image_size: u64, bss: u64, bss_size: u64) -> Result<u64, u32> {
    if !aligned(image) || !aligned(bss) {
        return Err(INVALID_ADDRESS);
    }
    if image_size == 0 || !aligned(image_size) || !aligned(bss_size) {
        return Err(INVALID_SIZE);
    }
    let image_end = image.checked_add(image_size).ok_or(INVALID_ADDRESS)?;
    let bss_end = bss.checked_add(bss_size).ok_or(INVALID_ADDRESS)?;
    if image_end > kernel.address_space_end || (bss_size != 0 && bss_end > kernel.address_space_end) {
        return Err(INVALID_ADDRESS);
    }
    if nexium_memory::fastmem::direct_va_base().is_some() {
        log::warn!("[ro] NRO loading is unavailable with the direct-mapped guest address space");
        return Err(INVALID_PROCESS);
    }
    if kernel.loaded_nros.values().any(|loaded| loaded.image == image) {
        return Err(ALREADY_LOADED);
    }
    let mut header = [0u8; 0x40];
    kernel
        .address_space
        .read_checked(image, &mut header)
        .map_err(|_| INVALID_ADDRESS)?;
    let segments = parse_header(&header, image_size, bss_size)?;
    let total = image_size.checked_add(bss_size).ok_or(INVALID_SIZE)?;
    let base = free_range(kernel, total).ok_or(OUT_OF_ADDRESS_SPACE)?;
    let loaded = LoadedNro { image, bss, bss_size, segments };
    let generation = kernel.address_space.generation();
    let mut mapped = Vec::new();
    let mut failure = None;
    for (destination, source, size, perm, name) in pieces(base, &loaded) {
        match kernel.address_space.map_alias(destination, source, size, perm, name) {
            Ok(()) => mapped.push((destination, source, size)),
            Err(error) => {
                failure = Some(error.to_string());
                break;
            }
        }
    }
    if failure.is_none() {
        if let Err(error) =
            sync_process_alias_cpu(kernel.address_space.host_region_changes_since(generation))
        {
            failure = Some(error);
        }
    }
    if let Some(error) = failure {
        log::error!("[ro] mapping NRO image {image:#x} at {base:#x} failed: {error}");
        let rollback = kernel.address_space.generation();
        for (destination, source, size) in mapped.into_iter().rev() {
            let _ = kernel.address_space.unmap_alias(destination, source, size);
        }
        let _ = sync_process_alias_cpu(kernel.address_space.host_region_changes_since(rollback));
        return Err(OUT_OF_ADDRESS_SPACE);
    }
    kernel.loaded_nros.insert(base, loaded);
    log::info!(
        "[ro] mapped NRO image {image:#x} size {image_size:#x} bss {bss:#x}+{bss_size:#x} at {base:#x}"
    );
    Ok(base)
}

fn unmap_nro(kernel: &mut Kernel, base: u64) -> u32 {
    let Some(loaded) = kernel.loaded_nros.remove(&base) else {
        return NOT_LOADED;
    };
    let generation = kernel.address_space.generation();
    let mut result = 0;
    for (destination, source, size, _, _) in pieces(base, &loaded).into_iter().rev() {
        if let Err(error) = kernel.address_space.unmap_alias(destination, source, size) {
            log::error!("[ro] unmapping NRO at {base:#x} failed: {error}");
            result = INVALID_ADDRESS;
        }
    }
    if let Err(error) = sync_process_alias_cpu(kernel.address_space.host_region_changes_since(generation)) {
        log::error!("[ro] CPU unmapping for NRO at {base:#x} failed: {error}");
    }
    log::info!("[ro] unmapped NRO at {base:#x}");
    result
}

fn load(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx) -> Vec<u8> {
    let (Some(image), Some(image_size), Some(bss), Some(bss_size)) =
        (input(ctx, 8), input(ctx, 16), input(ctx, 24), input(ctx, 32))
    else {
        return build_ipc_response(ctx, INVALID_NRO, &[], &[]);
    };
    match map_nro(kernel, image, image_size, bss, bss_size) {
        Ok(base) => build_ipc_response(ctx, 0, &base.to_le_bytes(), &[]),
        Err(code) => {
            log::warn!(
                "[ro] LoadNro image={image:#x} size={image_size:#x} bss={bss:#x}+{bss_size:#x} -> {code:#x}"
            );
            build_ipc_response(ctx, code, &[], &[])
        }
    }
}

fn unload(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx) -> Vec<u8> {
    let Some(base) = input(ctx, 8) else {
        return build_ipc_response(ctx, INVALID_ADDRESS, &[], &[]);
    };
    let result = unmap_nro(kernel, base);
    build_ipc_response(ctx, result, &[], &[])
}

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, command: u32) -> Option<Vec<u8>> {
    match command {
        0 => Some(load(kernel, ctx)),
        1 => Some(unload(kernel, ctx)),
        2 | 3 | 4 | 10 => Some(build_ipc_response(ctx, 0, &[], &[])),
        _ => None,
    }
}
