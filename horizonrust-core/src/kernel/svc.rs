use super::Kernel;
use crate::common::result::{SUCCESS, KERNEL_INVALID_HANDLE, KERNEL_NOT_IMPLEMENTED};
use crate::kernel::handles::HandleType;

pub fn dispatch(kernel: &mut Kernel, imm: u16) -> u32 {
    log::trace!("SVC {:#04x}", imm);
    match imm {
        0x01 => svc_set_heap_size(kernel),
        0x06 => svc_query_memory(kernel),
        0x07 => svc_exit_process(kernel),
        0x0d => svc_map_shared_memory(kernel),
        0x0e => svc_unmap_shared_memory(kernel),
        0x10 => svc_signal_event(kernel),
        0x12 => svc_wait_synchronization(kernel),
        0x13 => svc_cancel_synchronization(kernel),
        0x15 => svc_send_sync_request(kernel),
        0x16 => svc_send_sync_request_with_user_buffer(kernel),
        0x19 => svc_get_thread_id(kernel),
        0x1a => svc_break(kernel),
        0x1b => svc_output_debug_string(kernel),
        0x1f => svc_connect_to_named_port(kernel),
        0x21 => svc_get_info(kernel),
        0x25 => svc_map_physical_memory(kernel),
        0x26 => svc_unmap_physical_memory(kernel),
        0x29 => svc_create_session(kernel),
        0x2b => svc_reply_and_receive(kernel),
        0x2d => svc_create_event(kernel),
        0x31 => svc_create_shared_memory(kernel),
        0x41 => svc_map_transfer_memory(kernel),
        0x46 => svc_create_thread(kernel),
        0x47 => svc_start_thread(kernel),
        0x48 => svc_exit_thread(kernel),
        0x4a => svc_sleep_thread(kernel),
        0x50 => svc_flush_data_cache(kernel),
        _ => {
            log::warn!("unknown SVC: {:#04x}", imm);
            KERNEL_NOT_IMPLEMENTED
        }
    }
}

fn svc_set_heap_size(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcSetHeapSize");
    SUCCESS
}

fn svc_query_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcQueryMemory");
    SUCCESS
}

fn svc_exit_process(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcExitProcess");
    SUCCESS
}

fn svc_map_shared_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapSharedMemory");
    SUCCESS
}

fn svc_unmap_shared_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapSharedMemory");
    SUCCESS
}

fn svc_signal_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSignalEvent");
    SUCCESS
}

fn svc_wait_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcWaitSynchronization");
    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_send_sync_request(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequest");
    SUCCESS
}

fn svc_send_sync_request_with_user_buffer(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequestWithUserBuffer");
    SUCCESS
}

fn svc_get_thread_id(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcGetThreadId");
    SUCCESS
}

fn svc_break(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcBreak");
    SUCCESS
}

fn svc_output_debug_string(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcOutputDebugString");
    SUCCESS
}

fn svc_connect_to_named_port(kernel: &mut Kernel) -> u32 {
    log::debug!("svcConnectToNamedPort");
    SUCCESS
}

fn svc_get_info(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcGetInfo");
    SUCCESS
}

fn svc_map_physical_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapPhysicalMemory");
    SUCCESS
}

fn svc_unmap_physical_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapPhysicalMemory");
    SUCCESS
}

fn svc_create_session(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateSession");
    kernel.handles.create_handle(HandleType::Session);
    SUCCESS
}

fn svc_reply_and_receive(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReplyAndReceive");
    SUCCESS
}

fn svc_create_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateEvent");
    kernel.handles.create_handle(HandleType::Event);
    SUCCESS
}

fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateSharedMemory");
    kernel.handles.create_handle(HandleType::SharedMemory);
    SUCCESS
}

fn svc_map_transfer_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapTransferMemory");
    SUCCESS
}

fn svc_create_thread(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateThread");
    kernel.threads.create_thread(1);
    SUCCESS
}

fn svc_start_thread(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcStartThread");
    SUCCESS
}

fn svc_exit_thread(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcExitThread");
    SUCCESS
}

fn svc_sleep_thread(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcSleepThread");
    SUCCESS
}

fn svc_flush_data_cache(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcFlushDataCache");
    SUCCESS
}
