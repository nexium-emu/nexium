use super::Kernel;
use crate::common::result::{SUCCESS, KERNEL_NOT_IMPLEMENTED};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use crate::services::IpcCtx;

pub fn dispatch(kernel: &mut Kernel, imm: u16) -> u32 {
    log::trace!("SVC {:#04x}", imm);
    match imm {
        0x01 => svc_set_heap_size(kernel),
        0x02 => svc_set_memory_permission(kernel),
        0x03 => svc_set_memory_attribute(kernel),
        0x04 => svc_map_memory(kernel),
        0x05 => svc_unmap_memory(kernel),
        0x06 => svc_query_memory(kernel),
        0x07 => svc_exit_process(kernel),
        0x08 => svc_create_thread(kernel),
        0x09 => svc_start_thread(kernel),
        0x0a => svc_exit_thread(kernel),
        0x0b => svc_sleep_thread(kernel),
        0x13 => svc_map_shared_memory(kernel),
        0x14 => svc_unmap_shared_memory(kernel),
        0x15 => svc_create_transfer_memory(kernel),
        0x16 => svc_close_handle(kernel),
        0x18 => svc_wait_synchronization(kernel),
        0x19 => svc_cancel_synchronization(kernel),
        0x1f => svc_connect_to_named_port(kernel),
        0x21 => svc_send_sync_request(kernel),
        0x26 => svc_break(kernel),
        0x27 => svc_output_debug_string(kernel),
        0x29 => svc_get_info(kernel),
        0x2c => svc_map_physical_memory(kernel),
        0x2d => svc_unmap_physical_memory(kernel),
        0x45 => svc_create_event(kernel),
        0x41 => svc_map_transfer_memory(kernel),
        _ => {
            log::warn!("unknown SVC: {:#04x}", imm);
            KERNEL_NOT_IMPLEMENTED
        }
    }
}

fn svc_set_heap_size(kernel: &mut Kernel) -> u32 {
    let size = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) } else { return 1; };
    log::debug!("svcSetHeapSize size={:#x} -> heap_base={:#x}", size, kernel.heap_base);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, kernel.heap_base);
    }
    SUCCESS
}

fn svc_set_memory_permission(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSetMemoryPermission (no-op)");
    SUCCESS
}

fn svc_set_memory_attribute(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSetMemoryAttribute (no-op)");
    SUCCESS
}

fn svc_map_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapMemory (no-op)");
    SUCCESS
}

fn svc_unmap_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapMemory (no-op)");
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

        log::warn!("svcWaitSynchronization: no handles signaled, returning TIMEOUT_ERROR");
        const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
        return TIMEOUT_ERROR;
    }

    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    let (tls_addr, session_handle) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_tpidrro_el0(), cpu.get_register(0) as u32)
    } else {
        return 1;
    };

    let mut tls_buf = vec![0u8; 0x100];
    if kernel.address_space.read(tls_addr, &mut tls_buf).is_err() {
        log::warn!("SendSyncRequest: failed to read TLS at {:#x}", tls_addr);
        return 1;
    }

    let port_name = match kernel.sessions.get(&session_handle) {
        Some(s) => s.port_name.clone(),
        None => {
            log::warn!("SendSyncRequest: invalid session handle {:#x}", session_handle);
            return 1;
        }
    };

    let (cmd_id, token, data_offset) = parse_ipc_cmd_id(&tls_buf);
    log::debug!("IPC port='{}' cmd={} handle={:#x}", port_name, cmd_id, session_handle);

    let tls_snapshot = tls_buf.clone();
    let mut pending_frames = std::mem::take(&mut kernel.pending_frames);
    let mut ipc_ctx = IpcCtx {
        tls_buf: &tls_snapshot,
        pending_frames: &mut pending_frames,
    };
    let result = kernel.services.dispatch_service(&port_name, cmd_id, &mut ipc_ctx);
    kernel.pending_frames = pending_frames;

    write_ipc_response(&mut tls_buf, data_offset, result, token);

    if kernel.address_space.write(tls_addr, &tls_buf).is_err() {
        log::warn!("SendSyncRequest: failed to write TLS response at {:#x}", tls_addr);
        return 1;
    }

    SUCCESS
}

fn parse_ipc_cmd_id(buf: &[u8]) -> (u32, u32, usize) {
    if buf.len() < 8 {
        return (0, 0, 8);
    }
    let hipc_word0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let hipc_word1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

    let num_send_statics = (hipc_word0 >> 16) & 0xF;
    let num_send_buffers = (hipc_word0 >> 20) & 0xF;
    let num_recv_buffers = (hipc_word0 >> 24) & 0xF;
    let num_exch_buffers = (hipc_word0 >> 28) & 0xF;
    let num_data_words   = (hipc_word1 >> 0)  & 0x3FF;
    let has_special      = (hipc_word1 >> 31) & 1;

    let mut cursor: usize = 8;

    if has_special != 0 && buf.len() >= cursor + 4 {
        let special = u32::from_le_bytes([buf[cursor], buf[cursor+1], buf[cursor+2], buf[cursor+3]]);
        let num_copy_handles = (special >> 0) & 0xF;
        let num_move_handles = (special >> 4) & 0xF;
        let send_pid = (special >> 8) & 1;
        cursor += 4;
        if send_pid != 0 { cursor += 8; }
        cursor += (num_copy_handles + num_move_handles) as usize * 4;
    }

    cursor += num_send_statics as usize * 8;
    cursor += (num_send_buffers + num_recv_buffers + num_exch_buffers) as usize * 12;

    cursor = (cursor + 3) & !3;

    let data_start = cursor;

    if num_data_words == 0 {
        return (0, 0, data_start);
    }

    if buf.len() < cursor + 16 {
        return (0, 0, data_start);
    }

    if &buf[cursor..cursor+4] == b"SFCI" {
        let cmd_id = u32::from_le_bytes([buf[cursor+8], buf[cursor+9], buf[cursor+10], buf[cursor+11]]);
        let token  = u32::from_le_bytes([buf[cursor+12], buf[cursor+13], buf[cursor+14], buf[cursor+15]]);
        return (cmd_id, token, data_start);
    }

    (0, 0, data_start)
}

fn write_ipc_response(buf: &mut [u8], data_offset: usize, result: u32, token: u32) {
    let hipc_resp: u64 = 0x0000_0004_0000_0000;
    buf[0..8].copy_from_slice(&hipc_resp.to_le_bytes());

    let off = (data_offset + 3) & !3;
    if buf.len() >= off + 16 {
        buf[off..off+4].copy_from_slice(b"SFCO");
        buf[off+4..off+8].copy_from_slice(&0u32.to_le_bytes());
        buf[off+8..off+12].copy_from_slice(&result.to_le_bytes());
        buf[off+12..off+16].copy_from_slice(&token.to_le_bytes());
    }
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
    let (info_type, _handle, _info_id) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1), cpu.get_register(2))
    } else {
        return 1;
    };

    let val: u64 = match info_type {
        0  => 0xF,
        1  => 0x0001_0000_0000,
        2  => kernel.code_base,
        3  => 0x4_0000_0000,
        4  => kernel.heap_base,
        5  => kernel.heap_size,
        6  => 0x80_000_000,
        7  => 0x40_000_000,
        8  => 0,
        9  => kernel.stack_base,
        10 => kernel.stack_size,
        11 => 0xCAFE_F00D_DEAD_BEEF,
        12 => kernel.code_base,
        13 => 0x40_0000_0000,
        14 => kernel.stack_base,
        15 => 0x4_000_000,
        16 => 0,
        17 => 0,
        18 => 0,
        19 => 0,
        20 => 0,
        21 => 0,
        22 => kernel.code_base,
        _  => {
            log::warn!("svcGetInfo: unknown type {}", info_type);
            if let Some(cpu) = &mut kernel.cpu { cpu.set_register(0, 0); }
            return SUCCESS;
        }
    };

    log::debug!("svcGetInfo type={} -> {:#x}", info_type, val);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, val);
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

fn svc_create_transfer_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateTransferMemory");
    SUCCESS
}

fn svc_close_handle(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCloseHandle");
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
