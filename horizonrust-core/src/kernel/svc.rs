use super::Kernel;
use crate::common::result::{SUCCESS, KERNEL_NOT_IMPLEMENTED};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use crate::ipc;

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
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_permission(kernel: &mut Kernel) -> u32 {
    let (addr, size, perm) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1), cpu.get_register(2))
    } else {
        (0, 0, 0)
    };
    log::info!("svcSetMemoryPermission addr={:#x} size={:#x} perm={:#x} (no-op)", addr, size, perm);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_attribute(kernel: &mut Kernel) -> u32 {
    let (addr, size, mask, value) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1), cpu.get_register(2), cpu.get_register(3))
    } else {
        (0, 0, 0, 0)
    };
    log::info!("svcSetMemoryAttribute addr={:#x} size={:#x} mask={:#x} value={:#x} (no-op)", addr, size, mask, value);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_map_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapMemory (no-op)");
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_memory(kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapMemory (no-op)");
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_memory(kernel: &mut Kernel) -> u32 {
    let (out_ptr, address) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(2))
    } else {
        return 1;
    };

    log::debug!("svcQueryMemory out_ptr={:#x} address={:#x}", out_ptr, address);

    let info = synthesize_memory_info(kernel, address);
    let mut buf = [0u8; 0x28];
    buf[0..8].copy_from_slice(&info.addr.to_le_bytes());
    buf[8..16].copy_from_slice(&info.size.to_le_bytes());
    buf[16..20].copy_from_slice(&info.mem_type.to_le_bytes());
    buf[20..24].copy_from_slice(&info.attr.to_le_bytes());
    buf[24..28].copy_from_slice(&info.perm.to_le_bytes());
    buf[28..32].copy_from_slice(&0u32.to_le_bytes());
    buf[32..36].copy_from_slice(&0u32.to_le_bytes());
    buf[36..40].copy_from_slice(&0u32.to_le_bytes());

    if out_ptr != 0 {
        let _ = kernel.address_space.write(out_ptr, &buf);
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, 0);
    }
    SUCCESS
}

struct SynthMemInfo {
    addr: u64,
    size: u64,
    mem_type: u32,
    attr: u32,
    perm: u32,
}

fn synthesize_memory_info(kernel: &Kernel, address: u64) -> SynthMemInfo {
    let regions = kernel.address_space.regions();
    for r in &regions {
        if address >= r.base && address < r.base + r.size {
            let mem_type = if r.name.contains("text") || r.name.contains("rodata") {
                0x10
            } else if r.name.contains("data") || r.name.contains("bss") {
                0x11
            } else if r.name.starts_with("heap") {
                0x05
            } else if r.name.starts_with("stack") {
                0x07
            } else if r.name.starts_with("shared") {
                0x12
            } else {
                0x03
            };
            return SynthMemInfo {
                addr: r.base,
                size: r.size,
                mem_type,
                attr: 0,
                perm: r.perm.bits() as u32,
            };
        }
    }

    let next_base = regions.iter()
        .map(|r| r.base)
        .filter(|&b| b > address)
        .min()
        .unwrap_or(u64::MAX);
    let page_addr = address & !0xFFF;
    let gap_size = next_base.saturating_sub(page_addr);
    SynthMemInfo {
        addr: page_addr,
        size: if gap_size == 0 { 0x10000_0000 } else { gap_size },
        mem_type: 0,
        attr: 0,
        perm: 0,
    }
}

fn svc_exit_process(kernel: &mut Kernel) -> u32 {
    log::info!("svcExitProcess - terminating process");
    kernel.process_exited = true;
    SUCCESS
}

fn svc_map_shared_memory(kernel: &mut Kernel) -> u32 {
    let (handle, addr, size, perm) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0) as u32, cpu.get_register(1), cpu.get_register(2), cpu.get_register(3) as u32)
    } else {
        return 1;
    };
    log::info!("svcMapSharedMemory handle={:#x} addr={:#x} size={:#x} perm={:#x}", handle, addr, size, perm);

    let backing = if size as usize == crate::hid_state::HID_SHMEM_SIZE {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        hid.shmem_va = Some(addr);
        log::info!("  → recognized as HID shared memory, populating with Pro Controller state");
        hid.build_initial_shmem()
    } else {
        vec![0u8; size as usize]
    };

    if kernel.address_space.write(addr, &backing).is_err() {
        let _ = kernel.address_space.map(addr, size, crate::memory::perm::Perm::RW, "shared");
        let _ = kernel.address_space.write(addr, &backing);
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_shared_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapSharedMemory");
    SUCCESS
}

fn svc_signal_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSignalEvent (X0=event_handle)");

    let handle = if let Some(cpu) = &kernel.cpu {
        cpu.get_register(0) as u32
    } else {
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, 1u64);
        }
        return 1;
    };

    log::debug!("  signaling event handle {:#x}", handle);

    if let Some(_event) = kernel.handles.get_handle(handle) {
        kernel.event_signals.insert(handle, true);
        log::debug!("  event {:#x} signaled", handle);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, SUCCESS as u64);
        }
        return SUCCESS;
    } else {
        log::warn!("  invalid event handle {:#x}", handle);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }
}

fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 {
    log::debug!("svcWaitSynchronization");

    if let Some(cpu) = &kernel.cpu {
        let handles_ptr = cpu.get_register(1);
        let handle_count = cpu.get_register(2);
        let timeout_ns = cpu.get_register(3);

        log::debug!("  handles_ptr={:#x} count={} timeout={}ns PC={:#x}",
            handles_ptr, handle_count, timeout_ns, cpu.get_pc());

        if handle_count == 0 {
            log::warn!("  invalid: handle_count is 0");
            if let Some(cpu_mut) = &mut kernel.cpu {
                cpu_mut.set_register(0, 1u64);
            }
            return 1;
        }

        if handle_count > 64 {
            log::warn!("  too many handles: {}", handle_count);
            if let Some(cpu_mut) = &mut kernel.cpu {
                cpu_mut.set_register(0, 1u64);
            }
            return 1;
        }

        if timeout_ns == 0 {
            log::debug!("  timeout=0 (immediate poll)");
            for i in 0..handle_count {
                let mut handle_buf = [0u8; 4];
                let addr = handles_ptr + (i * 4);
                if let Ok(()) = kernel.address_space.read(addr, &mut handle_buf) {
                    let handle = u32::from_le_bytes(handle_buf);
                    if let Some(true) = kernel.event_signals.get(&handle) {
                        log::debug!("    handle {:#x} is signaled", handle);
                        if let Some(cpu_mut) = &mut kernel.cpu {
                            cpu_mut.set_register(1, i as u64);
                        }
                        return SUCCESS;
                    }
                }
            }
            const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
            if let Some(cpu_mut) = &mut kernel.cpu {
                cpu_mut.set_register(0, TIMEOUT_ERROR as u64);
            }
            return TIMEOUT_ERROR;
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
                        cpu_mut.set_register(1, i as u64);
                    }
                    return SUCCESS;
                }
            }
        }

        if handle_count > 0 {
            log::debug!("  no handles signaled, returning index=0 (simulated vsync)");
            if let Some(cpu_mut) = &mut kernel.cpu {
                cpu_mut.set_register(1, 0);
            }
            return SUCCESS;
        }

        const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
        if let Some(cpu_mut) = &mut kernel.cpu {
            cpu_mut.set_register(0, TIMEOUT_ERROR as u64);
        }
        return TIMEOUT_ERROR;
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    dump_regs(kernel, "SendSync ENTRY");
    let (tls_addr, session_handle) = if let Some(cpu) = &kernel.cpu {
        let x0 = cpu.get_register(0) as u32;
        log::info!("SendSyncRequest: X0={:#x}", x0);
        (cpu.get_tpidrro_el0(), x0)
    } else {
        return 1;
    };

    let mut tls_buf = vec![0u8; 0x100];
    if kernel.address_space.read(tls_addr, &mut tls_buf).is_err() {
        log::warn!("SendSyncRequest: failed to read TLS at {:#x}", tls_addr);
        return 1;
    }
    log::info!("  TLS[0..32]: {:02x?}", &tls_buf[..32]);

    let port_name = match kernel.sessions.get(&session_handle) {
        Some(s) => s.port_name.clone(),
        None => {
            log::warn!("SendSyncRequest: invalid session handle {:#x}", session_handle);
            return 1;
        }
    };

    let hipc_header_raw = u64::from_le_bytes([tls_buf[0], tls_buf[1], tls_buf[2], tls_buf[3], tls_buf[4], tls_buf[5], tls_buf[6], tls_buf[7]]);
    let cmd_type = (hipc_header_raw & 0xFFFF) as u16;

    match cmd_type {
        2 => {
            log::info!("session Close session={:#x} service={}", session_handle, port_name);
            kernel.sessions.remove(&session_handle);
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        5 | 7 => {
            log::info!("Control cmd_type={} session={:#x} service={}", cmd_type, session_handle, port_name);
            let response = handle_control_request(kernel, session_handle, &port_name, &tls_buf);
            if !response.is_empty() {
                let mut response_buf = tls_buf.clone();
                let copy_len = response.len().min(response_buf.len());
                response_buf[..copy_len].copy_from_slice(&response[..copy_len]);
                let _ = kernel.address_space.write(tls_addr, &response_buf);
            }
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        _ => {}
    }

    let is_domain = kernel.sessions.get(&session_handle).map(|s| s.is_domain).unwrap_or(false);
    let ipc_parse_result = ipc::IpcCtx::parse(tls_buf.clone(), is_domain);
    let mut ctx = match ipc_parse_result {
        Ok(c) => c,
        Err(e) => {
            log::warn!("Failed to parse IPC message: {:?} (is_domain={})", e, is_domain);
            return 1;
        }
    };

    let cmd_id = ctx.cmif_in.cmd_id;

    let dispatch_target = if let Some(d) = ctx.domain {
        if d.kind == 2 {
            if let Some(s) = kernel.sessions.get_mut(&session_handle) {
                s.close_object(d.object_id);
            }
            log::debug!("domain Close-object session={:#x} object_id={}", session_handle, d.object_id);
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        kernel.sessions.get(&session_handle)
            .and_then(|s| s.service_for_object(d.object_id).map(String::from))
            .unwrap_or_else(|| port_name.clone())
    } else {
        port_name.clone()
    };

    log::info!("IPC request service=\"{}\" cmd={} in_data={} is_domain={}", dispatch_target, cmd_id, ctx.cmif_in_data_len, is_domain);

    if dispatch_target == "fatal:u" && cmd_id == 1 {
        if ctx.cmif_in_data_len >= 4 {
            let result = u32::from_le_bytes([
                ctx.buf[ctx.cmif_in_data_off],
                ctx.buf[ctx.cmif_in_data_off + 1],
                ctx.buf[ctx.cmif_in_data_off + 2],
                ctx.buf[ctx.cmif_in_data_off + 3],
            ]);
            let module = result & 0x1FF;
            let desc = (result >> 9) & 0x1FFF;
            log::error!("**** fatal:u ThrowFatal result={:#010x} module={} description={} ****",
                result, module, desc);
        }
    }

    let response = if dispatch_target == "sm:" {
        dispatch_sm_command_v2(kernel, &mut ctx)
    } else {
        let mut pending_frames = std::mem::take(&mut kernel.pending_frames);
        let response = dispatch_service_v2(kernel, &dispatch_target, &mut ctx, session_handle, &mut pending_frames);
        kernel.pending_frames = pending_frames;
        response
    };

    let mut response_buf = vec![0u8; 0x100];
    let copy_len = response.len().min(response_buf.len());
    response_buf[..copy_len].copy_from_slice(&response[..copy_len]);

    if kernel.address_space.write(tls_addr, &response_buf).is_err() {
        log::warn!("SendSyncRequest: failed to write TLS response at {:#x}", tls_addr);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn handle_control_request(kernel: &mut Kernel, session_handle: u32, port_name: &str, tls_buf: &[u8]) -> Vec<u8> {
    let parse_result = ipc::IpcCtx::parse(tls_buf.to_vec(), false);
    let mut ctx = match parse_result {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    match ctx.cmif_in.cmd_id {
        0 => {
            log::debug!("Control: ConvertCurrentObjectToDomain service={}", port_name);
            if let Some(session) = kernel.sessions.get_mut(&session_handle) {
                session.convert_to_domain();
            }
            build_ipc_response(&ctx, 0, &1u32.to_le_bytes(), &[])
        }
        1 => {
            log::debug!("Control: CopyFromCurrentDomain service={}", port_name);
            build_ipc_response(&ctx, 0, &[], &[])
        }
        2 | 4 => {
            let dup_handle = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(dup_handle, port_name.to_string());
            kernel.sessions.insert(dup_handle, session);
            log::debug!("Control: CloneCurrentObject service={} dup={:#x}", port_name, dup_handle);
            build_ipc_response(&mut ctx, 0, &[], &[dup_handle])
        }
        3 => {
            log::debug!("Control: QueryPointerBufferSize → 0x500 service={}", port_name);
            build_ipc_response(&mut ctx, 0, &0x500u16.to_le_bytes(), &[])
        }
        other => {
            log::debug!("Control: unknown cmd={} service={}", other, port_name);
            build_ipc_response(&mut ctx, 0, &[], &[])
        }
    }
}

fn build_ipc_response(ctx: &ipc::IpcCtx, result: u32, out_data: &[u8], move_handles: &[u32]) -> Vec<u8> {
    build_ipc_response_full(ctx, result, out_data, move_handles, &[])
}

fn build_ipc_response_full(ctx: &ipc::IpcCtx, result: u32, out_data: &[u8], move_handles: &[u32], out_objects: &[u32]) -> Vec<u8> {
    let is_domain = ctx.domain.is_some();

    let mut raw_size = 0usize;
    if is_domain {
        raw_size += 16;
    }
    raw_size += 16;
    raw_size += out_data.len();
    if is_domain {
        raw_size += out_objects.len() * 4;
    }
    let raw_padded = (raw_size + 3) & !3;

    let mut special_bytes: Vec<u8> = Vec::new();
    let has_special_header = !move_handles.is_empty();
    if has_special_header {
        let mut sh: u32 = 0;
        sh |= (move_handles.len() as u32 & 0xF) << 5;
        special_bytes.extend_from_slice(&sh.to_le_bytes());
        for h in move_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
    }

    let mut hipc: u64 = 0;
    hipc |= ((raw_padded / 4) as u64 & 0x3FF) << 32;
    if has_special_header {
        hipc |= 1u64 << 63;
    }

    let raw_data_off = (8 + special_bytes.len() + 15) & !15;
    let total = raw_data_off + raw_padded;
    let mut out = vec![0u8; total];

    out[0..8].copy_from_slice(&hipc.to_le_bytes());
    if !special_bytes.is_empty() {
        out[8..8 + special_bytes.len()].copy_from_slice(&special_bytes);
    }

    let mut p = raw_data_off;
    if is_domain {
        out[p..p + 4].copy_from_slice(&(out_objects.len() as u32).to_le_bytes());
        p += 16;
    }

    out[p..p + 4].copy_from_slice(b"SFCO");
    out[p + 4..p + 8].copy_from_slice(&1u32.to_le_bytes());
    out[p + 8..p + 12].copy_from_slice(&result.to_le_bytes());
    out[p + 12..p + 16].copy_from_slice(&ctx.cmif_in.token.to_le_bytes());
    p += 16;

    if !out_data.is_empty() && p + out_data.len() <= out.len() {
        out[p..p + out_data.len()].copy_from_slice(out_data);
        p += out_data.len();
    }

    if is_domain {
        for obj in out_objects {
            if p + 4 <= out.len() {
                out[p..p + 4].copy_from_slice(&obj.to_le_bytes());
                p += 4;
            }
        }
    }

    out
}

fn dispatch_sm_command_v2(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx) -> Vec<u8> {
    match ctx.cmif_in.cmd_id {
        0 => {
            let pid = ctx.send_pid;
            log::info!("sm:RegisterClient pid={:?}", pid);
            build_ipc_response(ctx, 0, &[], &[])
        }
        1 => {
            let name_bytes = if ctx.cmif_in_data_off + 8 <= ctx.buf.len() {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&ctx.buf[ctx.cmif_in_data_off..ctx.cmif_in_data_off + 8]);
                arr
            } else {
                [0u8; 8]
            };
            let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(&name_bytes);
            let name = String::from_utf8_lossy(trimmed).into_owned();
            log::info!("sm:GetServiceHandle name={} raw={:02x?}", name, name_bytes);

            let handle = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(handle, name.clone());
            kernel.sessions.insert(handle, session);

            build_ipc_response(ctx, 0, &[], &[handle])
        }
        2 => {
            log::info!("sm:RegisterService");
            let handle = kernel.handles.create_handle(HandleType::Session);
            build_ipc_response(ctx, 0, &[], &[handle])
        }
        3 => {
            log::info!("sm:UnregisterService");
            build_ipc_response(ctx, 0, &[], &[])
        }
        4 => {
            log::info!("sm:DetachClient");
            build_ipc_response(ctx, 0, &[], &[])
        }
        other => {
            log::warn!("sm: unknown cmd={}", other);
            build_ipc_response(ctx, 1, &[], &[])
        }
    }
}

fn dispatch_service_v2(kernel: &mut Kernel, port_name: &str, ctx: &mut ipc::IpcCtx, session_handle: u32, pending_frames: &mut Vec<crate::services::FrameOut>) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;

    if port_name == "nvdrv" || port_name == "nvdrv:a" || port_name == "nvdrv:s" || port_name == "nvdrv:t" {
        return dispatch_nvdrv_command(kernel, ctx, port_name);
    }

    if port_name == "IHOSBinderDriver" && (cmd_id == 0 || cmd_id == 3) {
        return handle_binder_transact(kernel, ctx, session_handle);
    }

    if let Some(buffer_data) = applet_buffer_response(port_name, cmd_id) {
        let target_buf = ctx.recv_buffers.iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = target_buf {
            let write_len = buffer_data.len().min(buf.size as usize);
            let _ = kernel.address_space.write(buf.addr, &buffer_data[..write_len]);
            log::info!("  wrote {} bytes to recv buf at {:#x} (avail {})", write_len, buf.addr, buf.size);
        } else {
            log::debug!("  no recv buffer/static available for {} cmd={}", port_name, cmd_id);
        }
    }

    if let Some(sub_service) = subsession_service(port_name, cmd_id) {
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if let Some(proxy_service) = applet_proxy_service(port_name, cmd_id) {
        log::info!("{} cmd={} → returning {} proxy", port_name, cmd_id, proxy_service);
        return return_subsession(kernel, ctx, session_handle, proxy_service);
    }

    if let Some((data, handle_opt)) = applet_command_response(kernel, port_name, cmd_id) {
        log::info!("{}.cmd_{} → returning data ({} bytes, handle={:?})", port_name, cmd_id, data.len(), handle_opt);
        let handles: Vec<u32> = handle_opt.into_iter().collect();
        return build_ipc_response(ctx, 0, &data, &handles);
    }

    let tls_snapshot = ctx.buf.clone();
    let mut svc_ctx = crate::services::IpcCtx {
        tls_buf: &tls_snapshot,
        pending_frames,
    };
    let (result, out_data) = kernel.services.dispatch_service(port_name, cmd_id, &mut svc_ctx);
    build_ipc_response(ctx, result, &out_data, &[])
}

const IGBP_REQUEST_BUFFER: u32 = 1;
const IGBP_DEQUEUE_BUFFER: u32 = 3;
const IGBP_QUEUE_BUFFER: u32 = 7;
const IGBP_CANCEL_BUFFER: u32 = 8;
const IGBP_QUERY: u32 = 9;
const IGBP_CONNECT: u32 = 10;
const IGBP_DISCONNECT: u32 = 11;
const IGBP_SET_PREALLOCATED_BUFFER: u32 = 14;

fn handle_binder_transact(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, _session_handle: u32) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    let (binder_id, code) = if ctx.cmif_in_data_len >= 8 {
        let off = ctx.cmif_in_data_off;
        let bid = i32::from_le_bytes([ctx.buf[off], ctx.buf[off + 1], ctx.buf[off + 2], ctx.buf[off + 3]]);
        let c = u32::from_le_bytes([ctx.buf[off + 4], ctx.buf[off + 5], ctx.buf[off + 6], ctx.buf[off + 7]]);
        (bid as u32, c)
    } else {
        (0u32, 0u32)
    };

    let mut in_parcel: Vec<u8> = Vec::new();
    let in_src = ctx.send_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied()
        .or_else(|| ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied());
    if let Some(sb) = in_src {
        in_parcel.resize(sb.size as usize, 0);
        let _ = kernel.address_space.read(sb.addr, &mut in_parcel);
    }

    let reply = igbp_handle_transact(kernel, binder_id, code, &in_parcel);

    log::info!("IHOSBinderDriver.TransactParcel{} binder={} code={} in_size={} reply_size={}",
        if cmd_id == 3 { "Auto" } else { "" }, binder_id, code, in_parcel.len(), reply.len());

    let out_dst = ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied()
        .or_else(|| ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied());
    if let Some(rb) = out_dst {
        let n = reply.len().min(rb.size as usize);
        let _ = kernel.address_space.write(rb.addr, &reply[..n]);
    }

    build_ipc_response(ctx, 0, &[], &[])
}

fn igbp_handle_transact(kernel: &mut Kernel, binder_id: u32, code: u32, in_parcel: &[u8]) -> Vec<u8> {
    let mut reader = ParcelReader::new(in_parcel);
    let _ = reader.skip_interface_token();

    match code {
        IGBP_CONNECT => {
            let _listener = reader.read_i32();
            let api = reader.read_i32().unwrap_or(0);
            let _producer_controlled = reader.read_i32();
            let (w, h) = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                bq.connected_api = api;
                (bq.width, bq.height)
            });
            log::info!("IGBP::Connect binder={} api={} {}x{}", binder_id, api, w, h);
            let mut p = ParcelBuilder::new();
            p.write_bq_buffer_output(w, h);
            p.write_u32(0);
            p.finish()
        }
        IGBP_DISCONNECT => {
            log::debug!("IGBP::Disconnect binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_SET_PREALLOCATED_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let has = reader.read_i32().unwrap_or(0);
            if has == 0 {
                return ParcelBuilder::new().finish();
            }
            let gb = parse_flattened_graphic_buffer(&mut reader);
            kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                if let Some(gb) = gb {
                    bq.set_preallocated(slot, gb);
                }
            });
            log::info!("IGBP::SetPreallocatedBuffer binder={} slot={}", binder_id, slot);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_REQUEST_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let gb = kernel.nvdrv.with_bufferqueue(binder_id, |bq| bq.request_buffer(slot).cloned());
            let mut p = ParcelBuilder::new();
            if let Some(gb) = gb {
                p.write_u32(1);
                p.write_flattened_graphic_buffer(&gb);
            } else {
                p.write_u32(0);
            }
            p.write_u32(0);
            log::info!("IGBP::RequestBuffer binder={} slot={}", binder_id, slot);
            p.finish()
        }
        IGBP_DEQUEUE_BUFFER => {
            let _async_ = reader.read_i32();
            let _w = reader.read_u32();
            let _h = reader.read_u32();
            let _fmt = reader.read_i32();
            let _usage = reader.read_u32();
            let slot = kernel.nvdrv.with_bufferqueue(binder_id, |bq| bq.dequeue());
            log::info!("IGBP::DequeueBuffer binder={} → slot={}", binder_id, slot);
            let mut p = ParcelBuilder::new();
            p.write_u32(slot);
            p.write_u32(1);
            p.write_flattened_zero_fence();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUEUE_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let gb_opt = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                bq.queue(slot);
                bq.request_buffer(slot).cloned()
            });
            log::info!("IGBP::QueueBuffer binder={} slot={}", binder_id, slot);

            if let Some(gb) = gb_opt {
                if let Some(nvmap) = kernel.nvdrv.nvmap_handles.get(&gb.nvmap_id) {
                    let addr = nvmap.address.wrapping_add(gb.buffer_offset);
                    let size = (gb.stride as usize) * (gb.height as usize) * 4;
                    let mut pixels = vec![0u8; size];
                    if kernel.address_space.read(addr, &mut pixels).is_ok() {
                        kernel.nvdrv.submit_frame(crate::nvdrv::QueuedFrame {
                            width: gb.width,
                            height: gb.height,
                            pixels,
                        });
                    }
                }
            }

            let mut p = ParcelBuilder::new();
            p.write_bq_buffer_output(1280, 720);
            p.write_u32(0);
            p.finish()
        }
        IGBP_CANCEL_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            kernel.nvdrv.with_bufferqueue(binder_id, |bq| bq.cancel(slot));
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUERY => {
            let _what = reader.read_i32().unwrap_or(0);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.write_u32(0);
            p.finish()
        }
        other => {
            log::debug!("IGBP::Unknown code={} binder={}", other, binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
    }
}

struct ParcelReader<'a> {
    data: &'a [u8],
    payload_off: usize,
    cursor: usize,
}

impl<'a> ParcelReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let payload_off = if data.len() >= 16 {
            u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize
        } else { 0 };
        Self { data, payload_off, cursor: payload_off }
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.cursor + 4 > self.data.len() { return None; }
        let v = u32::from_le_bytes([
            self.data[self.cursor], self.data[self.cursor + 1],
            self.data[self.cursor + 2], self.data[self.cursor + 3],
        ]);
        self.cursor += 4;
        Some(v)
    }

    fn read_i32(&mut self) -> Option<i32> {
        self.read_u32().map(|v| v as i32)
    }

    fn read_u64(&mut self) -> Option<u64> {
        let lo = self.read_u32()? as u64;
        let hi = self.read_u32()? as u64;
        Some(lo | (hi << 32))
    }

    fn skip_interface_token(&mut self) -> Option<()> {
        let _strict_policy = self.read_u32()?;
        let len = self.read_i32()?;
        if len <= 0 {
            return Some(());
        }
        let byte_len = ((len as usize) + 1) * 2;
        let padded = (byte_len + 3) & !3;
        self.cursor += padded;
        Some(())
    }
}

struct ParcelBuilder {
    payload: Vec<u8>,
}

impl ParcelBuilder {
    fn new() -> Self {
        Self { payload: Vec::new() }
    }

    fn write_u32(&mut self, v: u32) {
        self.payload.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bq_buffer_output(&mut self, w: u32, h: u32) {
        self.write_u32(w);
        self.write_u32(h);
        self.write_u32(0);
        self.write_u32(0);
    }

    fn write_flattened_zero_fence(&mut self) {
        self.write_u32(36);
        self.write_u32(0);
        self.write_u32(0);
        for _ in 0..4 {
            self.write_u32(0);
            self.write_u32(0);
        }
    }

    fn write_flattened_graphic_buffer(&mut self, gb: &crate::nvdrv::GraphicBuffer) {
        const NUM_INTS: u32 = 81;
        const HEADER_U32S: u32 = 10;
        let body_size = (HEADER_U32S + NUM_INTS) * 4;
        self.write_u32(body_size);
        self.write_u32(0);
        self.write_u32(0x47424652);
        self.write_u32(gb.width);
        self.write_u32(gb.height);
        self.write_u32(gb.stride);
        self.write_u32(gb.format);
        self.write_u32(gb.usage);
        self.write_u32(42);
        self.write_u32(1);
        self.write_u32(0);
        self.write_u32(NUM_INTS);
        let mut ints = [0u32; NUM_INTS as usize];
        ints[0] = 0xFFFF_FFFF;
        ints[1] = gb.nvmap_id;
        ints[2] = 0;
        ints[3] = 0xDAFF_CAFF;
        ints[4] = 42;
        ints[5] = 0;
        ints[6] = gb.usage;
        ints[7] = gb.format;
        ints[8] = gb.format;
        ints[9] = gb.stride;
        ints[10] = gb.width.saturating_mul(gb.height).saturating_mul(4);
        ints[11] = 1;
        ints[12] = 0;
        ints[13] = gb.width;
        ints[14] = gb.height;
        ints[18] = gb.stride.saturating_mul(4);
        ints[19] = gb.nvmap_id;
        ints[20] = gb.buffer_offset as u32;
        ints[21] = 0;
        ints[22] = 4;
        for v in ints {
            self.write_u32(v);
        }
    }

    fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.payload.len());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&((16 + self.payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
}

fn parse_flattened_graphic_buffer(reader: &mut ParcelReader) -> Option<crate::nvdrv::GraphicBuffer> {
    let _length = reader.read_u32()?;
    let _fd_count = reader.read_u32()?;
    let _magic = reader.read_u32();
    let width = reader.read_u32().unwrap_or(0);
    let height = reader.read_u32().unwrap_or(0);
    let stride = reader.read_u32().unwrap_or(0);
    let format = reader.read_u32().unwrap_or(0);
    let usage = reader.read_u32().unwrap_or(0);

    let _pid = reader.read_u32();
    let _refcount = reader.read_u32();
    let _num_fds = reader.read_u32();
    let num_ints = reader.read_u32().unwrap_or(0) as usize;

    let mut ints = Vec::with_capacity(num_ints);
    for _ in 0..num_ints {
        ints.push(reader.read_u32().unwrap_or(0));
    }
    let nvmap_id = ints.get(1).copied().unwrap_or(0);
    let buffer_offset = ints.get(20).copied().unwrap_or(0);

    Some(crate::nvdrv::GraphicBuffer {
        width,
        height,
        stride,
        format,
        usage,
        nvmap_id,
        buffer_offset: buffer_offset as u64,
        size: stride * height * 4,
    })
}

fn dispatch_nvdrv_command(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, port_name: &str) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    log::info!("nvdrv:{}.cmd_{}", port_name, cmd_id);

    match cmd_id {
        0 => {
            let buf_src = ctx.send_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied()
                .or_else(|| ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied());
            let path = if let Some(sb) = buf_src {
                let mut buf = vec![0u8; sb.size as usize];
                let _ = kernel.address_space.read(sb.addr, &mut buf);
                let trimmed = buf.split(|&b| b == 0).next().unwrap_or(&buf);
                String::from_utf8_lossy(trimmed).into_owned()
            } else {
                String::new()
            };
            log::info!("nvdrv:Open path='{}' (sb={:?})", path, buf_src.map(|b| (b.addr, b.size)));
            let fd = kernel.nvdrv.open(&path).unwrap_or(0);
            let mut out = Vec::new();
            out.extend_from_slice(&fd.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            build_ipc_response(ctx, 0, &out, &[])
        }
        1 | 11 | 12 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else { 0 };
            let ioctl_id = if ctx.cmif_in_data_len >= 8 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off + 4],
                    ctx.buf[ctx.cmif_in_data_off + 5],
                    ctx.buf[ctx.cmif_in_data_off + 6],
                    ctx.buf[ctx.cmif_in_data_off + 7],
                ])
            } else { 0 };

            let in_src = ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied()
                .or_else(|| ctx.send_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied());
            let mut in_data: Vec<u8> = Vec::new();
            if let Some(sb) = in_src {
                in_data.resize(sb.size as usize, 0);
                let _ = kernel.address_space.read(sb.addr, &mut in_data);
            }

            let out_dst = ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied()
                .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied());
            let out_size = out_dst.map(|b| b.size as usize).unwrap_or(0);

            if cmd_id == 1 {
                log::debug!("nvdrv:Ioctl fd={} ioctl_id={:#x} send_buf={:?} send_static={:?} recv_buf={:?}",
                    fd, ioctl_id,
                    ctx.send_buffers.iter().map(|b| (b.addr, b.size)).collect::<Vec<_>>(),
                    ctx.send_statics.iter().map(|b| (b.addr, b.size)).collect::<Vec<_>>(),
                    ctx.recv_buffers.iter().map(|b| (b.addr, b.size)).collect::<Vec<_>>());
            }

            let req = crate::nvdrv::IoctlRequest {
                fd, ioctl_id, in_data, out_size,
            };
            let addr_space = kernel.address_space.clone();
            let outcome = kernel.nvdrv.dispatch_ioctl_with_mem(req, &|addr, buf| {
                addr_space.read(addr, buf).is_ok()
            });

            if !outcome.data.is_empty() {
                if let Some(buf) = out_dst {
                    let n = outcome.data.len().min(buf.size as usize);
                    let _ = kernel.address_space.write(buf.addr, &outcome.data[..n]);
                }
            }

            build_ipc_response(ctx, 0, &outcome.result.to_le_bytes(), &[])
        }
        2 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else { 0 };
            kernel.nvdrv.close(fd);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        3 => {
            log::debug!("nvdrv:Initialize");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        4 => {
            log::debug!("nvdrv:QueryEvent");
            let h = kernel.handles.create_handle(HandleType::Event);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[h])
        }
        8 => {
            log::debug!("nvdrv:SetClientPID");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        13 => {
            log::debug!("nvdrv:GetStatus");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        other => {
            log::debug!("nvdrv: unknown cmd={}", other);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
    }
}

fn applet_buffer_response(port_name: &str, cmd_id: u32) -> Option<Vec<u8>> {
    match (port_name, cmd_id) {
        ("IApplicationDisplayService", 2020) | ("IApplicationDisplayService", 2030) | ("IManagerDisplayService", 2012) => {
            Some(build_native_window_parcel(0x100))
        }
        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => {
            Some(build_igbp_success_parcel())
        }
        _ => None,
    }
}

fn build_igbp_success_parcel() -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1280u32.to_le_bytes());
    payload.extend_from_slice(&720u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&2u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::with_capacity(16 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn build_native_window_parcel(binder_handle: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0x2u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&binder_handle.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(b"dispdrv\0");
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::with_capacity(16 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn return_subsession(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, session_handle: u32, sub_service: &str) -> Vec<u8> {
    let is_domain = kernel.sessions.get(&session_handle).map(|s| s.is_domain).unwrap_or(false);
    if is_domain {
        let object_id = if let Some(s) = kernel.sessions.get_mut(&session_handle) {
            s.alloc_domain_object(sub_service.to_string())
        } else {
            0
        };
        log::info!("→ {} sub-object id={}", sub_service, object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, sub_service.to_string());
        kernel.sessions.insert(h, session);
        log::info!("→ {} sub-session handle={:#x}", sub_service, h);
        build_ipc_response(ctx, 0, &[], &[h])
    }
}

fn subsession_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("hid", 0) => Some("IAppletResource"),
        ("IAppletResource", 0) => Some("HidSharedMemory"),
        ("IApplicationCreator", 0) => Some("IApplicationAccessor"),
        ("ILibraryAppletCreator", 0) => Some("ILibraryAppletAccessor"),
        ("time:s" | "time:u" | "time:a" | "time:r", 0) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 1) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 2) => Some("ISteadyClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 3) => Some("ITimeZoneService"),
        ("time:s" | "time:u" | "time:a" | "time:r", 4) => Some("ISystemClock"),
        ("fsp-srv", 18) => Some("IFileSystem"),
        ("fsp-srv", 51) => Some("IFileSystem"),
        ("vi:m" | "vi:s" | "vi:u", 0) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 1) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 2) => Some("IApplicationDisplayService"),
        ("vi:m" | "vi:s" | "vi:u", 3) => Some("IApplicationDisplayService"),
        ("IApplicationDisplayService", 100) => Some("IHOSBinderDriver"),
        ("IApplicationDisplayService", 101) => Some("ISystemDisplayService"),
        ("IApplicationDisplayService", 102) => Some("IManagerDisplayService"),
        ("IApplicationDisplayService", 103) => Some("IHOSBinderDriver"),
        ("appletAE" | "appletOE", 0) => Some("IApplicationProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        _ => None,
    }
}

fn applet_command_response(kernel: &mut Kernel, port_name: &str, cmd_id: u32) -> Option<(Vec<u8>, Option<u32>)> {
    match (port_name, cmd_id) {
        ("IWindowController", 1) => Some((1u64.to_le_bytes().to_vec(), None)),
        ("IWindowController", 10) => Some((Vec::new(), None)),
        ("ISelfController", 0) => Some((Vec::new(), None)),
        ("ISelfController", 1) => Some((Vec::new(), None)),
        ("ISelfController", 10) => Some((Vec::new(), None)),
        ("ISelfController", 11) => Some((Vec::new(), None)),
        ("ISelfController", 12) => Some((Vec::new(), None)),
        ("ISelfController", 16) => Some((Vec::new(), None)),
        ("ISelfController", 40) => {
            let handle = kernel.handles.create_handle(HandleType::Event);
            Some((Vec::new(), Some(handle)))
        }
        ("ISelfController", 50) => Some((1u8.to_le_bytes().to_vec(), None)),
        ("ISelfController", 91) => {
            let handle = kernel.handles.create_handle(HandleType::Event);
            Some((Vec::new(), Some(handle)))
        }
        ("ICommonStateGetter", 0) => {
            let handle = kernel.handles.create_handle(HandleType::Event);
            Some((Vec::new(), Some(handle)))
        }
        ("ICommonStateGetter", 1) => Some((0u32.to_le_bytes().to_vec(), None)),
        ("ICommonStateGetter", 5) => Some((1u8.to_le_bytes().to_vec(), None)),
        ("ICommonStateGetter", 6) => Some((0u32.to_le_bytes().to_vec(), None)),
        ("ICommonStateGetter", 8) => Some((1u8.to_le_bytes().to_vec(), None)),
        ("ICommonStateGetter", 9) => Some((1u8.to_le_bytes().to_vec(), None)),
        ("ICommonStateGetter", 60) => Some({
            let mut data = Vec::new();
            data.extend_from_slice(&1280u32.to_le_bytes());
            data.extend_from_slice(&720u32.to_le_bytes());
            (data, None)
        }),
        ("IApplicationFunctions", 1) => Some((0u8.to_le_bytes().to_vec(), None)),
        ("IApplicationFunctions", 20) => Some((Vec::new(), None)),
        ("IApplicationFunctions", 21) => Some((Vec::new(), None)),
        ("IApplicationFunctions", 22) => Some((Vec::new(), None)),
        ("IApplicationFunctions", 23) => Some((0u8.to_le_bytes().to_vec(), None)),
        ("IApplicationFunctions", 30) => Some((Vec::new(), None)),
        ("IApplicationFunctions", 40) => Some((0u32.to_le_bytes().to_vec(), None)),
        ("IApplicationFunctions", 50) => Some((Vec::new(), None)),
        ("IDebugFunctions", _) => Some((Vec::new(), None)),

        ("IApplicationDisplayService", 1010) => Some((1u64.to_le_bytes().to_vec(), None)),
        ("IApplicationDisplayService", 1011) => Some((1u64.to_le_bytes().to_vec(), None)),
        ("IApplicationDisplayService", 1020) => Some((Vec::new(), None)),
        ("IApplicationDisplayService", 2020) => {
            let parcel_size = build_native_window_parcel(0x100).len() as u64;
            Some((parcel_size.to_le_bytes().to_vec(), None))
        }
        ("IApplicationDisplayService", 2021) => Some((Vec::new(), None)),
        ("IApplicationDisplayService", 2030) => {
            let layer_id: u64 = 1;
            let parcel_size = build_native_window_parcel(0x100).len() as u64;
            let mut out = Vec::new();
            out.extend_from_slice(&layer_id.to_le_bytes());
            out.extend_from_slice(&parcel_size.to_le_bytes());
            Some((out, None))
        }
        ("IApplicationDisplayService", 2031) => Some((Vec::new(), None)),
        ("IApplicationDisplayService", 2101) => Some((Vec::new(), None)),
        ("IApplicationDisplayService", 2102) => Some((Vec::new(), None)),
        ("IApplicationDisplayService", 3000) => Some((60u64.to_le_bytes().to_vec(), None)),

        ("ISystemDisplayService", 2205) => Some((Vec::new(), None)),
        ("ISystemDisplayService", 2207) => Some((Vec::new(), None)),
        ("ISystemDisplayService", 2312) => Some((Vec::new(), None)),
        ("ISystemDisplayService", 2400) => Some((Vec::new(), None)),
        ("ISystemDisplayService", 2402) => Some((Vec::new(), None)),
        ("ISystemDisplayService", 3216) => Some((0u32.to_le_bytes().to_vec(), None)),

        ("IManagerDisplayService", 2010) => Some((1u64.to_le_bytes().to_vec(), None)),
        ("IManagerDisplayService", 2011) => Some((Vec::new(), None)),
        ("IManagerDisplayService", 2012) => {
            let layer_id: u64 = 1;
            let parcel_size = build_native_window_parcel(0x100).len() as u64;
            let mut out = Vec::new();
            out.extend_from_slice(&layer_id.to_le_bytes());
            out.extend_from_slice(&parcel_size.to_le_bytes());
            Some((out, None))
        }
        ("IManagerDisplayService", 6000) => Some((Vec::new(), None)),

        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => Some((Vec::new(), None)),
        ("IHOSBinderDriver", 1) => Some((Vec::new(), None)),
        ("IHOSBinderDriver", 2) => {
            let handle = kernel.handles.create_handle(HandleType::Event);
            Some((Vec::new(), Some(handle)))
        }

        ("ISystemClock", 0) => {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let switch_epoch = secs.saturating_sub(946_684_800);
            Some((switch_epoch.to_le_bytes().to_vec(), None))
        }
        ("ISystemClock", 2) => Some(([0u8; 0x20].to_vec(), None)),
        ("ISteadyClock", 0) => Some(([0u8; 0x18].to_vec(), None)),
        ("ITimeZoneService", 0) => Some(([0u8; 0x24].to_vec(), None)),
        ("ITimeZoneService", 101) => Some(([0u8; 0x4].to_vec(), None)),

        ("IFileSystem", _) => Some((Vec::new(), None)),
        ("fsp-srv", _) => Some((Vec::new(), None)),

        ("psm", _) => Some((Vec::new(), None)),
        ("set", _) | ("set:sys", _) => Some((Vec::new(), None)),
        ("nvdrv:a", _) | ("nvdrv", _) | ("nvdrv:s", _) | ("nvdrv:t", _) => Some((0u32.to_le_bytes().to_vec(), None)),

        _ => None,
    }
}

fn applet_proxy_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("appletAE" | "appletOE", 100) => Some("ISystemAppletProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        ("appletAE" | "appletOE", 300) => Some("IOverlayAppletProxy"),
        ("appletAE" | "appletOE", 350) => Some("IApplicationProxy"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 0) => Some("ICommonStateGetter"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 1) => Some("ISelfController"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 2) => Some("IWindowController"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 3) => Some("IAudioController"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 4) => Some("IDisplayController"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 10) => Some("IProcessWindingController"),
        ("ISystemAppletProxy" | "ILibraryAppletProxy" | "IOverlayAppletProxy" | "IApplicationProxy", 11) => Some("ILibraryAppletCreator"),
        ("ISystemAppletProxy", 20) => Some("IApplicationFunctions"),
        ("ISystemAppletProxy", 21) => Some("IHomeMenuFunctions"),
        ("ISystemAppletProxy", 22) => Some("IGlobalStateController"),
        ("ISystemAppletProxy", 23) => Some("IApplicationCreator"),
        ("IApplicationProxy", 20) => Some("IApplicationFunctions"),
        ("IApplicationProxy", 1000) => Some("IDebugFunctions"),
        ("ISystemAppletProxy", 1000) => Some("IDebugFunctions"),
        ("ILibraryAppletProxy", 1000) => Some("IDebugFunctions"),
        ("IOverlayAppletProxy", 1000) => Some("IDebugFunctions"),
        _ => None,
    }
}

fn dispatch_sm_command(kernel: &mut Kernel, cmd_id: u32, tls_buf: &[u8], cmif_data_off: usize, cmif_data_len: usize, parsed_ctx: Option<ipc::IpcCtx>) -> (u32, Vec<u8>) {
    match cmd_id {
        0 => dispatch_sm_register_client(kernel, tls_buf, cmif_data_off, cmif_data_len, parsed_ctx),
        1 => dispatch_sm_get_service_handle(kernel, tls_buf, cmif_data_off, cmif_data_len, parsed_ctx),
        2 => dispatch_sm_register_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        3 => dispatch_sm_unregister_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        _ => {
            log::warn!("unknown SM command: {}", cmd_id);
            (1, Vec::new())
        }
    }
}

fn dispatch_sm_register_client(_kernel: &mut Kernel, _tls_buf: &[u8], _cmif_data_off: usize, _cmif_data_len: usize, parsed_ctx: Option<ipc::IpcCtx>) -> (u32, Vec<u8>) {
    let pid = parsed_ctx.and_then(|ctx| ctx.send_pid);
    log::info!("SM::RegisterClient pid={:?}", pid);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_register_service(_kernel: &mut Kernel, tls_buf: &[u8], cmif_data_off: usize, _cmif_data_len: usize) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };
    log::debug!("SM::RegisterService '{}'", service_name);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_unregister_service(_kernel: &mut Kernel, tls_buf: &[u8], cmif_data_off: usize, _cmif_data_len: usize) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };
    log::debug!("SM::UnregisterService '{}'", service_name);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_get_service_handle(kernel: &mut Kernel, tls_buf: &[u8], cmif_data_off: usize, _cmif_data_len: usize, _parsed_ctx: Option<ipc::IpcCtx>) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };

    log::info!("SM::GetServiceHandle '{}' data_off={:#x}", service_name, cmif_data_off);

    let handle = kernel.handles.create_handle(HandleType::Session);
    let final_name = if !service_name.is_empty() { service_name } else { "unknown".to_string() };
    let session = Session::new(handle, final_name.clone());
    kernel.sessions.insert(handle, session);

    log::info!("SM: returning handle {:#x} for service '{}'", handle, final_name);

    let mut response = Vec::new();
    response.extend_from_slice(&handle.to_le_bytes());
    (SUCCESS, response)
}

fn write_ipc_response(buf: &mut [u8], data_offset: usize, result: u32, token: u32) {
    write_ipc_response_with_data(buf, data_offset, result, token, &[]);
}

fn write_ipc_response_with_data(buf: &mut [u8], data_offset: usize, result: u32, token: u32, out_data: &[u8]) {
    let hipc_resp: u64 = 0x0000_0004_0000_0000;
    buf[0..8].copy_from_slice(&hipc_resp.to_le_bytes());

    let off = (data_offset + 3) & !3;
    if buf.len() >= off + 16 {
        buf[off..off+4].copy_from_slice(b"SFCO");
        buf[off+4..off+8].copy_from_slice(&0u32.to_le_bytes());
        buf[off+8..off+12].copy_from_slice(&result.to_le_bytes());
        buf[off+12..off+16].copy_from_slice(&token.to_le_bytes());

        if !out_data.is_empty() && off + 16 + out_data.len() <= buf.len() {
            buf[off+16..off+16+out_data.len()].copy_from_slice(out_data);
        }
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

fn svc_break(kernel: &mut Kernel) -> u32 {
    let reason = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) } else { 0 };
    let info_va = if let Some(cpu) = &kernel.cpu { cpu.get_register(1) } else { 0 };
    let info_size = if let Some(cpu) = &kernel.cpu { cpu.get_register(2) as usize } else { 0 };

    log::warn!("svcBreak: reason={:#x}, info_va={:#x}, info_size={:#x}", reason, info_va, info_size);

    if let Some(cpu) = &kernel.cpu {
        let pc = cpu.get_pc();
        let lr = cpu.get_register(30);
        let sp = cpu.get_register(31);
        let fp = cpu.get_register(29);

        log::warn!("  PC={:#x}, LR={:#x}, SP={:#x}, FP={:#x}", pc, lr, sp, fp);

        let mut callers = Vec::new();
        let mut cur_fp = fp;
        for i in 0..8 {
            if cur_fp < 0x80_0000_0000 || cur_fp > 0xc0_0000_0000 {
                break;
            }
            let mut frame = [0u8; 16];
            if kernel.address_space.read(cur_fp, &mut frame).is_err() {
                break;
            }
            let next_fp = u64::from_le_bytes([frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7]]);
            let saved_lr = u64::from_le_bytes([frame[8], frame[9], frame[10], frame[11], frame[12], frame[13], frame[14], frame[15]]);
            callers.push((i, saved_lr));
            if next_fp == 0 || next_fp <= cur_fp {
                break;
            }
            cur_fp = next_fp;
        }

        for (i, addr) in &callers {
            log::warn!("  Stack[{}]: {:#x} (offset {:#x})", i, addr, addr.wrapping_sub(kernel.code_base));
        }

        if info_size > 0 && info_size <= 0x1000 {
            let mut info_buf = vec![0u8; info_size.min(0x80)];
            if kernel.address_space.read(info_va, &mut info_buf).is_ok() {
                log::warn!("  Info buffer: {:02x?}", &info_buf);
            }
        }
    }

    kernel.process_exited = true;
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
    log::info!("svcConnectToNamedPort (X1=port_name_ptr)");

    let port_name_ptr = if let Some(cpu) = &kernel.cpu {
        cpu.get_register(1)
    } else {
        return 1;
    };

    let port_name = if port_name_ptr > 0 {
        let mut buf = [0u8; 32];
        match kernel.address_space.read(port_name_ptr, &mut buf) {
            Ok(()) => {
                let mut len = 0;
                for (i, &byte) in buf.iter().enumerate() {
                    if byte == 0 {
                        len = i;
                        break;
                    }
                    if i == buf.len() - 1 {
                        len = buf.len();
                    }
                }
                let name_str = std::str::from_utf8(&buf[..len])
                    .unwrap_or("invalid")
                    .to_string();
                log::info!("  port_name: '{}' (len={}) PC={:#x}", name_str, len,
                    kernel.cpu.as_ref().map(|c| c.get_pc()).unwrap_or(0));
                name_str
            }
            Err(_) => {
                log::warn!("failed to read port name from {:#x}", port_name_ptr);
                return 1;
            }
        }
    } else {
        return 1;
    };

    let handle = kernel.handles.create_handle(HandleType::Session);
    let session = Session::new(handle, port_name.clone());
    kernel.sessions.insert(handle, session);

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, handle as u64);
    } else {
        log::error!("kernel.cpu is None!");
    }

    log::info!("created session handle {:#x} to port '{}'", handle, port_name);
    SUCCESS
}

fn svc_get_info(kernel: &mut Kernel) -> u32 {
    let (info_type, _handle, _sub) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(1) as u32, cpu.get_register(2), cpu.get_register(3))
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
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, 0);
                cpu.set_register(1, 0);
            }
            return SUCCESS;
        }
    };

    log::debug!("svcGetInfo type={} -> {:#x}", info_type, val);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
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

fn dump_regs(kernel: &Kernel, tag: &str) {
    if let Some(cpu) = &kernel.cpu {
        log::info!(
            "  [{}] X0={:#x} X1={:#x} X2={:#x} X3={:#x} X4={:#x} X8={:#x} X19={:#x} X30={:#x}",
            tag,
            cpu.get_register(0), cpu.get_register(1), cpu.get_register(2),
            cpu.get_register(3), cpu.get_register(4), cpu.get_register(8),
            cpu.get_register(19), cpu.get_register(30),
        );
    }
}

fn svc_create_transfer_memory(kernel: &mut Kernel) -> u32 {
    dump_regs(kernel, "CreateTmem ENTRY");
    let (addr, size, perm) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(1), cpu.get_register(2), cpu.get_register(3))
    } else {
        return 1;
    };
    let handle = kernel.handles.create_handle(HandleType::TransferMemory);
    log::info!("svcCreateTransferMemory addr={:#x} size={:#x} perm={:#x} → handle={:#x}",
        addr, size, perm, handle);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, handle as u64);
    }
    dump_regs(kernel, "CreateTmem EXIT");
    SUCCESS
}

fn svc_close_handle(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) as u32 } else { 0 };
    let kind = kernel.handles.get_handle(handle).map(|h| format!("{:?}", h.handle_type)).unwrap_or_else(|| "unknown".into());
    log::info!("svcCloseHandle handle={:#x} ({})", handle, kind);
    dump_regs(kernel, "CloseHandle ENTRY");
    kernel.handles.close_handle(handle);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
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
