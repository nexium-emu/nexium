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
    log::info!("svcExitProcess - terminating process");
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
    log::debug!("svcSignalEvent (X0=event_handle)");

    if let Some(cpu) = &kernel.cpu {
        let handle = cpu.get_register(0) as u32;
        log::debug!("  signaling event handle {:#x}", handle);

        if let Some(_event) = kernel.handles.get_handle(handle) {
            kernel.event_signals.insert(handle, true);
            log::debug!("  event {:#x} signaled", handle);
            return SUCCESS;
        } else {
            log::warn!("  invalid event handle {:#x}", handle);
            return 1;
        }
    }

    SUCCESS
}

fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 {
    log::debug!("svcWaitSynchronization (X0=handles_ptr, X1=count, X2=timeout_ns)");

    if let Some(cpu) = &kernel.cpu {
        let handles_ptr = cpu.get_register(0);
        let handle_count = cpu.get_register(1);
        let timeout_ns = cpu.get_register(2);

        log::debug!("  waiting on {} handles, timeout={} ns", handle_count, timeout_ns);

        if handle_count == 0 {
            log::warn!("  invalid: handle_count is 0");
            return 1;
        }

        if handle_count > 64 {
            log::warn!("  too many handles: {}", handle_count);
            return 1;
        }

        if timeout_ns == 0 {
            log::debug!("  timeout=0 (immediate check)");
            for i in 0..handle_count {
                let mut handle_buf = [0u8; 4];
                let addr = handles_ptr + (i * 4);
                if let Ok(()) = kernel.address_space.read(addr, &mut handle_buf) {
                    let handle = u32::from_le_bytes(handle_buf);
                    if let Some(true) = kernel.event_signals.get(&handle) {
                        log::debug!("    handle {:#x} is signaled", handle);
                        if let Some(cpu_mut) = &mut kernel.cpu {
                            cpu_mut.set_register(0, i as u64);
                        }
                        return SUCCESS;
                    }
                }
            }
            return 1;
        }

        if timeout_ns == 0xFFFFFFFFFFFFFFFF {
            log::debug!("  timeout=infinite (WAIT_INFINITE)");
        } else {
            log::debug!("  timeout={} ns (~{} ms)", timeout_ns, timeout_ns / 1_000_000);
        }

        for i in 0..handle_count {
            let mut handle_buf = [0u8; 4];
            let addr = handles_ptr + (i * 4);
            if let Ok(()) = kernel.address_space.read(addr, &mut handle_buf) {
                let handle = u32::from_le_bytes(handle_buf);
                if let Some(true) = kernel.event_signals.get(&handle) {
                    log::debug!("    handle {:#x} is signaled", handle);
                    if let Some(cpu_mut) = &mut kernel.cpu {
                        cpu_mut.set_register(0, i as u64);
                    }
                    return SUCCESS;
                }
            }
        }

        return SUCCESS;
    }

    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequest");

    let (tls_addr, session_handle) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_tpidrro_el0(), cpu.get_register(0) as u32)
    } else {
        return 1;
    };

    let mut tls_buf = vec![0u8; 256];
    if let Err(_) = kernel.address_space.read(tls_addr, &mut tls_buf) {
        log::warn!("Failed to read TLS buffer from {:#x}", tls_addr);
        return 1;
    }

    if tls_buf.len() < 24 {
        log::warn!("TLS buffer too small");
        return 1;
    }

    if tls_buf[..4] != *b"SFCI" {
        log::warn!("bad SFCI magic");
        return 1;
    }

    let cmd_id = u32::from_le_bytes([
        tls_buf[8],
        tls_buf[9],
        tls_buf[10],
        tls_buf[11],
    ]);
    let token = u32::from_le_bytes([
        tls_buf[12],
        tls_buf[13],
        tls_buf[14],
        tls_buf[15],
    ]);

    let port_name = if let Some(session) = kernel.sessions.get(&session_handle) {
        session.port_name.clone()
    } else {
        log::warn!("invalid session handle {:#x}", session_handle);
        return 1;
    };

    log::debug!("  session handle {:#x} -> port '{}', cmd_id {}", session_handle, port_name, cmd_id);

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

    tls_buf[..16].copy_from_slice(&response_header);

    if let Err(_) = kernel.address_space.write(tls_addr, &tls_buf) {
        log::warn!("Failed to write TLS response to {:#x}", tls_addr);
        return 1;
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

fn svc_output_debug_string(kernel: &mut Kernel) -> u32 {
    log::debug!("svcOutputDebugString (X0=str_ptr, X1=str_len)");

    if let Some(cpu) = &kernel.cpu {
        let str_ptr = cpu.get_register(0);
        let str_len = cpu.get_register(1);

        if str_ptr > 0 && str_len > 0 && str_len < 4096 {
            let mut buf = vec![0u8; str_len as usize];
            match kernel.address_space.read(str_ptr, &mut buf) {
                Ok(()) => {
                    let output = std::str::from_utf8(&buf).unwrap_or("[invalid utf8]");
                    println!("[DEBUG] {}", output);
                    log::debug!("OutputDebugString: {}", output);
                }
                Err(e) => {
                    log::warn!("Failed to read debug string from {:#x}: {:?}", str_ptr, e);
                }
            }
        }
    }

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

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, handle as u64);
    }

    log::debug!("created session handle {:#x} to port '{}'", handle, port_name);
    SUCCESS
}

fn svc_get_info(kernel: &mut Kernel) -> u32 {
    log::debug!("svcGetInfo (X0=type, X1=handle, X2=info_id)");

    if let Some(cpu) = &kernel.cpu {
        let info_type = cpu.get_register(0);
        let handle = cpu.get_register(1);
        let info_id = cpu.get_register(2);

        log::debug!("  type={}, handle={:#x}, id={}", info_type, handle, info_id);

        match info_type {
            2 => {
                log::debug!("  GetInfo::MemoryUsage");
                if let Some(cpu_mut) = &mut kernel.cpu {
                    cpu_mut.set_register(0, kernel.heap_size);
                }
                return SUCCESS;
            }
            11 => {
                log::debug!("  GetInfo::ThreadCount");
                if let Some(cpu_mut) = &mut kernel.cpu {
                    cpu_mut.set_register(0, 1);
                }
                return SUCCESS;
            }
            _ => {
                log::debug!("  unknown info type: {}", info_type);
                return 1;
            }
        }
    }

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
    let handle = kernel.handles.create_handle(HandleType::Session);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, handle as u64);
    }
    SUCCESS
}

fn svc_reply_and_receive(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReplyAndReceive");
    SUCCESS
}

fn svc_create_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateEvent");
    let handle = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(handle, false);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, handle as u64);
    }
    log::debug!("  created event handle {:#x}", handle);
    SUCCESS
}

fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateSharedMemory");
    let handle = kernel.handles.create_handle(HandleType::SharedMemory);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, handle as u64);
    }
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
