use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn request_load(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _shared_font_type: u32) {}

pub fn get_load_state(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _shared_font_type: u32) -> u32 { 1 }

pub fn get_size(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, font_type: u32) -> u64 {
    let (_, size) = kernel.font_offsets[font_type.min(5) as usize];
    log::debug!("pl GetSize type={} → {}", font_type, size);
    size as u64
}

pub fn get_shared_font_in_order_of_priority_offset(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, font_type: u32) -> u64 {
    let (offset, _) = kernel.font_offsets[font_type.min(5) as usize];
    log::debug!("pl GetOffset type={} → {}", font_type, offset);
    offset as u64
}

pub fn get_shared_memory_native_handle(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let handle = kernel.ensure_font_shmem_handle();
    log::debug!("pl GetSharedMemoryNativeHandle → {:#x}", handle);
    handle
}

pub fn get_shared_font_in_order_of_priority(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _language_code: u64) -> u64 { 0 }
pub fn get_shared_font_in_order_of_priority_for_system(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _language_code: u64) -> u64 { 0 }
