use super::Kernel;
use crate::common::result::{SUCCESS, KERNEL_NOT_IMPLEMENTED};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;

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

fn svc_set_heap_size(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSetHeapSize (X0=heap_size, X1=heap_addr_ptr)");

    if let Some(cpu) = &kernel.cpu {
        let heap_size = cpu.get_register(0);
        log::debug!("  heap_size: {:#x}", heap_size);
    }

    SUCCESS
}

fn svc_query_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcQueryMemory (X1=address)");

    let address = if let Some(cpu) = &kernel.cpu {
        cpu.get_register(1)
    } else {
        0
    };

    log::debug!("  query address: {:#x}", address);

    let memory_type: u32 = 0;
    let memory_attr: u32 = 0;
    let permission: u32 = 0x3;

    log::debug!("returning memory_type={:#x}, attr={:#x}, perm={:#x}", memory_type, memory_attr, permission);
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

fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 {
    log::debug!("svcWaitSynchronization (X0=handles[], X1=count, X2=timeout_ns)");
    log::debug!("waiting on {} handles", 1);
    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequest");

    if kernel.tls_buffer.len() < 24 {
        log::warn!("TLS buffer too small");
        return 1;
    }

    if kernel.tls_buffer[..4] != *b"SFCI" {
        log::warn!("bad SFCI magic");
        return 1;
    }

    let cmd_id = u32::from_le_bytes([
        kernel.tls_buffer[8],
        kernel.tls_buffer[9],
        kernel.tls_buffer[10],
        kernel.tls_buffer[11],
    ]);
    let token = u32::from_le_bytes([
        kernel.tls_buffer[12],
        kernel.tls_buffer[13],
        kernel.tls_buffer[14],
        kernel.tls_buffer[15],
    ]);

    let port_name = "sm:".to_string();
    let result = kernel.services.dispatch_service(&port_name, cmd_id);

    let response_header = [
        0x46, 0x43, 0x4F, 0x53, 0x01, 0x00, 0x00, 0x00,
        (result & 0xFF) as u8,
        ((result >> 8) & 0xFF) as u8,
        ((result >> 16) & 0xFF) as u8,
        ((result >> 24) & 0xFF) as u8,
        token as u8,
        (token >> 8) as u8,
        (token >> 16) as u8,
        (token >> 24) as u8,
    ];

    if kernel.tls_buffer.len() >= 16 {
        kernel.tls_buffer[..16].copy_from_slice(&response_header);
    }

    if result != SUCCESS {
        log::warn!("service dispatch returned error: {:#x}", result);
        return result;
    }

    SUCCESS
}

fn svc_send_sync_request_with_user_buffer(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequestWithUserBuffer");
    SUCCESS
}

fn svc_get_thread_id(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcGetThreadId (X1=thread_handle)");
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
    log::debug!("svcConnectToNamedPort (X1=port_name_ptr)");

    let port_name_ptr = if let Some(cpu) = &kernel.cpu {
        cpu.get_register(1)
    } else {
        0
    };

    let port_name = if port_name_ptr > 0 {
        let mut buf = [0u8; 12];
        match kernel.address_space.read(port_name_ptr, &mut buf) {
            Ok(()) => {
                let name_str = std::str::from_utf8(&buf)
                    .unwrap_or("invalid")
                    .trim_end_matches('\0')
                    .to_string();
                log::debug!("  port_name: '{}'", name_str);
                name_str
            }
            Err(_) => {
                log::warn!("failed to read port name from {:#x}", port_name_ptr);
                "sm:".to_string()
            }
        }
    } else {
        "sm:".to_string()
    };

    let handle = kernel.handles.create_handle(HandleType::Session);
    let session = Session::new(handle, port_name.clone());
    kernel.sessions.insert(handle, session);

    log::debug!("created session handle {:#x} to port '{}'", handle, port_name);
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
    log::debug!("svcCreateThread (X1=entry, X2=arg, X3=sp, X4=priority, X5=core)");

    if let Some(cpu) = &kernel.cpu {
        let entry = cpu.get_register(1);
        let arg = cpu.get_register(2);
        let sp = cpu.get_register(3);
        let priority = cpu.get_register(4);
        let core = cpu.get_register(5);

        log::debug!("  entry: {:#x}, arg: {:#x}, sp: {:#x}, priority: {}, core: {}",
                   entry, arg, sp, priority, core);
    }

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
