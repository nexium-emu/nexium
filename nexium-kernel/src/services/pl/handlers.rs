use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn request_load(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _shared_font_type: u32,
) {
}

pub fn get_load_state(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _shared_font_type: u32,
) -> u32 {
    1
}

pub fn get_size(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, font_type: u32) -> u64 {
    kernel.ensure_font_shmem_handle();
    let (_, size) = kernel.font_offsets[font_type.min(5) as usize];
    log::debug!("pl GetSize type={} -> {}", font_type, size);
    size as u64
}

pub fn get_shared_font_in_order_of_priority_offset(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    font_type: u32,
) -> u64 {
    kernel.ensure_font_shmem_handle();
    let (offset, _) = kernel.font_offsets[font_type.min(5) as usize];
    log::debug!("pl GetOffset type={} -> {}", font_type, offset);
    offset as u64
}

pub fn get_shared_memory_native_handle(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let handle = kernel.ensure_font_shmem_handle();
    log::debug!("pl GetSharedMemoryNativeHandle -> {:#x}", handle);
    handle
}

pub fn get_shared_font_in_order_of_priority(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _language_code: u64,
    font_codes: &mut Vec<u8>,
    font_offsets: &mut Vec<u8>,
    font_sizes: &mut Vec<u8>,
) -> Vec<u8> {
    shared_font_priority(kernel, font_codes, font_offsets, font_sizes)
}

pub fn get_shared_font_in_order_of_priority_for_system(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _language_code: u64,
    font_codes: &mut Vec<u8>,
    font_offsets: &mut Vec<u8>,
    font_sizes: &mut Vec<u8>,
) -> Vec<u8> {
    shared_font_priority(kernel, font_codes, font_offsets, font_sizes)
}

fn shared_font_priority(
    kernel: &mut Kernel,
    font_codes: &mut Vec<u8>,
    font_offsets: &mut Vec<u8>,
    font_sizes: &mut Vec<u8>,
) -> Vec<u8> {
    kernel.ensure_font_shmem_handle();
    let entries: Vec<(u32, u32, u32)> = kernel
        .font_offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(offset, size))| {
            if size == 0 {
                None
            } else {
                Some((i as u32, offset, size))
            }
        })
        .take(6)
        .collect();

    for (code, offset, size) in &entries {
        font_codes.extend_from_slice(&code.to_le_bytes());
        font_offsets.extend_from_slice(&offset.to_le_bytes());
        font_sizes.extend_from_slice(&size.to_le_bytes());
    }

    let count = entries.len() as u32;
    log::debug!("pl GetSharedFontInOrderOfPriority count={}", count);

    let mut out = Vec::with_capacity(8);
    out.push(1);
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&count.to_le_bytes());
    out
}
