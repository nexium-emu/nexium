use super::Kernel;
use nexium_common::result::{SUCCESS, KERNEL_NOT_IMPLEMENTED, KERNEL_INVALID_ADDRESS};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use nexium_ipc as ipc;

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
        0x0c => svc_get_thread_priority(kernel),
        0x0d => svc_set_thread_priority(kernel),
        0x0e => svc_get_thread_core_mask(kernel),
        0x0f => svc_set_thread_core_mask(kernel),
        0x10 => svc_get_current_processor_number(kernel),
        0x11 => svc_signal_event(kernel),
        0x12 => svc_clear_event(kernel),
        0x13 => svc_map_shared_memory(kernel),
        0x14 => svc_unmap_shared_memory(kernel),
        0x15 => svc_create_transfer_memory(kernel),
        0x16 => svc_close_handle(kernel),
        0x17 => svc_reset_signal(kernel),
        0x18 => svc_wait_synchronization(kernel),
        0x19 => svc_cancel_synchronization(kernel),
        0x1a => svc_arbitrate_lock(kernel),
        0x1b => svc_arbitrate_unlock(kernel),
        0x1c => svc_wait_process_wide_key_atomic(kernel),
        0x1d => svc_signal_process_wide_key(kernel),
        0x1e => svc_get_system_tick(kernel),
        0x1f => svc_connect_to_named_port(kernel),
        0x20 => svc_send_sync_request_light(kernel),
        0x21 => svc_send_sync_request(kernel),
        0x22 => svc_send_sync_request_with_user_buffer(kernel),
        0x23 => svc_send_async_request_with_user_buffer(kernel),
        0x24 => svc_get_process_id(kernel),
        0x25 => svc_get_thread_id(kernel),
        0x26 => svc_break(kernel),
        0x27 => svc_output_debug_string(kernel),
        0x28 => svc_return_from_exception(kernel),
        0x29 => svc_get_info(kernel),
        0x2a => svc_flush_entire_data_cache(kernel),
        0x2b => svc_flush_data_cache(kernel),
        0x2c => svc_map_physical_memory(kernel),
        0x2d => svc_unmap_physical_memory(kernel),
        0x2e => svc_get_debug_future_thread_info(kernel),
        0x2f => svc_get_last_thread_info(kernel),
        0x30 => svc_get_resource_limit_limit_value(kernel),
        0x31 => svc_get_resource_limit_current_value(kernel),
        0x32 => svc_set_thread_activity(kernel),
        0x33 => svc_get_thread_context3(kernel),
        0x34 => svc_wait_for_address(kernel),
        0x35 => svc_signal_to_address(kernel),
        0x36 => svc_synchronize_preemption_state(kernel),
        0x37 => svc_get_resource_limit_peak_value(kernel),
        0x40 => svc_create_session(kernel),
        0x41 => svc_accept_session(kernel),
        0x42 => svc_reply_and_receive_light(kernel),
        0x43 => svc_reply_and_receive(kernel),
        0x44 => svc_reply_and_receive_with_user_buffer(kernel),
        0x45 => svc_create_event(kernel),
        0x50 => svc_create_shared_memory(kernel),
        0x51 => svc_map_transfer_memory(kernel),
        0x52 => svc_unmap_transfer_memory(kernel),
        0x53 => svc_create_interrupt_event(kernel),
        0x54 => svc_query_io_mapping(kernel),
        0x5f => svc_debug_active_process(kernel),
        0x60 => svc_break_debug_process(kernel),
        0x61 => svc_terminate_debug_process(kernel),
        0x62 => svc_get_debug_event(kernel),
        0x63 => svc_continue_debug_event(kernel),
        0x64 => svc_get_process_list(kernel),
        0x65 => svc_get_thread_list(kernel),
        0x6f => svc_create_port(kernel),
        0x70 => svc_manage_named_port(kernel),
        0x71 => svc_connect_to_port(kernel),
        0x7c => svc_create_resource_limit(kernel),
        0x7d => svc_set_resource_limit_limit_value(kernel),
        0x7e => svc_call_secure_monitor(kernel),
        _ => {
            log::warn!("unknown SVC: {:#04x}", imm);
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, KERNEL_NOT_IMPLEMENTED as u64);
            }
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
    let (dst, src, size) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1), cpu.get_register(2))
    } else {
        return 1;
    };

    if size == 0 || (dst & 0xFFF) != 0 || (size & 0xFFF) != 0 {
        log::warn!("svcMapMemory: bad args dst={:#x} src={:#x} size={:#x}", dst, src, size);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, KERNEL_INVALID_ADDRESS as u64);
        }
        return KERNEL_INVALID_ADDRESS;
    }

    let map_rc = kernel.address_space.map(dst, size, nexium_memory::Perm::RW, "stack_mirror");
    let was_new = map_rc.is_ok();

    let mut buf = vec![0u8; size as usize];
    if kernel.address_space.read(src, &mut buf).is_ok() {
        let _ = kernel.address_space.write(dst, &buf);
    }

    if was_new {
        if let Some(region) = kernel.address_space.host_region_at(dst) {
            if let Some(cpu) = &mut kernel.cpu {
                let plumb = unsafe {
                    cpu.map_host(region.base, region.size, region.perm, region.host_ptr as *mut u8)
                };
                match plumb {
                    Ok(_) => log::info!(
                        "svcMapMemory dst={:#x} src={:#x} size={:#x} → mapped + copied + plumbed to dynarmic",
                        dst, src, size
                    ),
                    Err(e) => log::warn!(
                        "svcMapMemory dst={:#x} size={:#x} mapped in AS but dynarmic map_host failed: {}",
                        dst, size, e
                    ),
                }
            }
        } else {
            log::warn!("svcMapMemory dst={:#x}: AS region lookup failed after map()", dst);
        }
    } else if let Err(e) = map_rc {
        log::debug!("svcMapMemory dst={:#x} src={:#x} size={:#x} → already mapped ({:?}), refreshed contents only", dst, src, size, e);
    }

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
    if handle == 0 {
        log::error!(
            "svcMapSharedMemory called with handle=0 (addr={:#x} size={:#x}). \
             Upstream service returned no shared-memory handle. \
             Allocating zero-filled placeholder; expect downstream code to read zeros from this region.",
            addr, size
        );
    }

    if size as usize == crate::hid_state::HID_SHMEM_SIZE {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        hid.shmem_va = Some(addr);
        let ptr = hid.host_ptr();
        log::info!("  → recognized as HID shared memory, mapping host buffer directly to guest VA {:#x}", addr);
        if let Some(cpu) = &mut kernel.cpu {
            unsafe {
                if let Err(e) = cpu.map_host(addr, size, nexium_memory::perm::Perm::RW, ptr) {
                    log::warn!("failed to map HID shmem in CPU: {}", e);
                }
            }
        }
    } else {
        let backing = vec![0u8; size as usize];
        let needed_map = kernel.address_space.write(addr, &backing).is_err();
        if needed_map {
            let _ = kernel.address_space.map(addr, size, nexium_memory::perm::Perm::RW, "shared");
            let _ = kernel.address_space.write(addr, &backing);
            if let Some(region) = kernel.address_space.host_region_at(addr) {
                if let Some(cpu) = &mut kernel.cpu {
                    unsafe {
                        if let Err(e) = cpu.map_host(region.base, region.size, region.perm, region.host_ptr) {
                            log::warn!("failed to map shared mem in CPU: {}", e);
                        } else {
                            log::info!("  → registered shared mem at {:#x} with CPU", region.base);
                        }
                    }
                }
            }
        }
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
        kernel.threads.signal_handle(handle);
        log::debug!("  event {:#x} signaled (waiters woken)", handle);
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
    let (handles_addr, count, timeout_ns) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(1), (cpu.get_register(2) as u32).min(0x40), cpu.get_register(3))
    } else {
        return 1;
    };

    let mut handles: Vec<u32> = Vec::with_capacity(count as usize);
    if count > 0 && handles_addr != 0 {
        let mut buf = vec![0u8; count as usize * 4];
        if kernel.address_space.read(handles_addr, &mut buf).is_ok() {
            for i in 0..count as usize {
                let h = u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
                handles.push(h);
            }
        }
    }

    let mut vsync_idx: Option<usize> = None;
    for (i, h) in handles.iter().enumerate() {
        if kernel.vsync_handles.contains(h) {
            vsync_idx = Some(i);
            break;
        }
    }

    if timeout_ns == 0 {
        log::debug!("svcWaitSync(timeout=0) handles={:?} applet_msg_event={:?} applet_msgs_pending={} vsync_idx={:?}",
            handles, kernel.applet_message_event, kernel.applet_messages.len(), vsync_idx);
        for (i, h) in handles.iter().enumerate() {
            if let Some(msg_evt) = kernel.applet_message_event {
                if *h == msg_evt && !kernel.applet_messages.is_empty() {
                    if let Some(cpu) = &mut kernel.cpu {
                        cpu.set_register(0, SUCCESS as u64);
                        cpu.set_register(1, i as u64);
                    }
                    return SUCCESS;
                }
            }
            if let Some(slot) = kernel.event_signals.get_mut(h) {
                if *slot {
                    *slot = false;
                    if let Some(cpu) = &mut kernel.cpu {
                        cpu.set_register(0, SUCCESS as u64);
                        cpu.set_register(1, i as u64);
                    }
                    return SUCCESS;
                }
            }
        }

        {
            let state = crate::hid_state::get_hid_state();
            let mut hid = state.lock();
            if hid.shmem_va.is_some() {
                let cur = hid.input.clone();
                hid.tick(cur);
            }
        }

        const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, TIMEOUT_ERROR as u64);
            cpu.set_register(1, 0);
        }
        return TIMEOUT_ERROR;
    }

    if let Some(i) = vsync_idx {
        const VSYNC_PERIOD: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let now = std::time::Instant::now();
        let elapsed = now.saturating_duration_since(kernel.last_vsync);
        let remaining = if elapsed >= VSYNC_PERIOD { std::time::Duration::ZERO } else { VSYNC_PERIOD - elapsed };
        let allowed = if timeout_ns == u64::MAX || timeout_ns == 0 {
            remaining
        } else {
            std::time::Duration::from_nanos(timeout_ns).min(remaining)
        };
        if allowed > std::time::Duration::ZERO {
            std::thread::sleep(allowed);
        }
        kernel.last_vsync = std::time::Instant::now();
        {
            let state = crate::hid_state::get_hid_state();
            let mut hid = state.lock();
            if hid.shmem_va.is_some() {
                let cur = hid.input.clone();
                hid.tick(cur);
            }
        }
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, SUCCESS as u64);
            cpu.set_register(1, i as u64);
        }
        return SUCCESS;
    }

    for (i, h) in handles.iter().enumerate() {
        if let Some(slot) = kernel.event_signals.get_mut(h) {
            if *slot {
                *slot = false;
                if let Some(cpu) = &mut kernel.cpu {
                    cpu.set_register(0, SUCCESS as u64);
                    cpu.set_register(1, i as u64);
                }
                return SUCCESS;
            }
        }
    }

    const VSYNC_PERIOD_NS: u64 = 16_666_667;
    let cap = std::time::Duration::from_nanos(VSYNC_PERIOD_NS);
    let wait = if timeout_ns == u64::MAX {
        cap
    } else {
        std::time::Duration::from_nanos(timeout_ns).min(cap)
    };
    if wait > std::time::Duration::ZERO {
        std::thread::sleep(wait);
    }

    {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        if hid.shmem_va.is_some() {
            let cur = hid.input.clone();
            hid.tick(cur);
        }
    }

    const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, TIMEOUT_ERROR as u64);
    }
    TIMEOUT_ERROR
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_arbitrate_lock(kernel: &mut Kernel) -> u32 {
    let (_holder, mutex_addr, self_handle) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0) as u32, cpu.get_register(1), cpu.get_register(2) as u32)
    } else {
        return 1;
    };

    let _ = kernel.address_space.write(mutex_addr, &self_handle.to_le_bytes());
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_arbitrate_unlock(kernel: &mut Kernel) -> u32 {
    let mutex_addr = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) } else { return 1; };
    let _ = kernel.address_space.write(mutex_addr, &0u32.to_le_bytes());
    if let Some(woken) = kernel.threads.wake_one_on_mutex(mutex_addr) {
        log::debug!("svcArbitrateUnlock mutex={:#x} woke handle={:#x}", mutex_addr, woken);
    }
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_wait_process_wide_key_atomic(kernel: &mut Kernel) -> u32 {
    let (mutex_addr, condvar_addr, _self_handle, timeout_ns) = if let Some(cpu) = &kernel.cpu {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2) as u32,
            cpu.get_register(3),
        )
    } else {
        return 1;
    };

    let _ = kernel.address_space.write(mutex_addr, &0u32.to_le_bytes());
    if let Some(woken) = kernel.threads.wake_one_on_mutex(mutex_addr) {
        log::debug!("cond_wait: handed mutex={:#x} to handle={:#x} on release", mutex_addr, woken);
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }

    let wake_at = if (timeout_ns as i64) <= 0 || timeout_ns == u64::MAX {
        None
    } else {
        Some(std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns))
    };

    if let Some(cpu) = kernel.cpu.as_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingCondvar { mutex_addr, condvar_addr, wake_at },
        );
    }

    SUCCESS
}

fn svc_signal_process_wide_key(kernel: &mut Kernel) -> u32 {
    let (condvar_addr, count) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1) as i32)
    } else {
        return 1;
    };

    let woken = if count < 0 {
        kernel.threads.wake_all_on_condvar(condvar_addr)
    } else {
        let mut n = 0;
        for _ in 0..count {
            if kernel.threads.wake_one_on_condvar(condvar_addr).is_some() {
                n += 1;
            } else {
                break;
            }
        }
        n
    };
    log::debug!("svcSignalProcessWideKey cond={:#x} count={} woken={}", condvar_addr, count, woken);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_system_tick(kernel: &mut Kernel) -> u32 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let elapsed = EPOCH.get_or_init(std::time::Instant::now).elapsed();
    let ticks = (elapsed.as_nanos() as u64).wrapping_mul(19_200_000) / 1_000_000_000;
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, ticks);
    }
    SUCCESS
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    let (tls_addr, session_handle) = if let Some(cpu) = &kernel.cpu {
        let x0 = cpu.get_register(0) as u32;
        log::trace!("SendSyncRequest: X0={:#x}", x0);
        (cpu.get_tpidrro_el0(), x0)
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
            log::debug!("Control cmd_type={} session={:#x} service={}", cmd_type, session_handle, port_name);
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
            kernel.open_files.remove(&(session_handle, d.object_id));
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

    log::debug!("IPC request service=\"{}\" cmd={} in_data={} is_domain={}", dispatch_target, cmd_id, ctx.cmif_in_data_len, is_domain);

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

    if let Some(sub_service) = crate::services::am::proxy_subsession(port_name, cmd_id) {
        log::debug!("{} cmd={} → returning {} sub-session", port_name, cmd_id, sub_service);
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if let Some(sub_service) = subsession_service(port_name, cmd_id) {
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if let Some((rc, data, handles)) = crate::services::am::dispatch_command(kernel, port_name, cmd_id) {
        log::debug!("am.{}.cmd_{} rc={:#x} → {} bytes, {} handle(s)", port_name, cmd_id, rc, data.len(), handles.len());
        return build_ipc_response(ctx, rc, &data, &handles);
    }

    if port_name == "IFileSystem" {
        match cmd_id {
            8 => {
                let path_buf = ctx.send_statics.iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let mut path_str = String::new();
                if let Some(b) = path_buf {
                    let max = (b.size as usize).min(0x301);
                    let mut tmp = vec![0u8; max];
                    if kernel.address_space.read(b.addr, &mut tmp).is_ok() {
                        let nul = tmp.iter().position(|&c| c == 0).unwrap_or(tmp.len());
                        tmp.truncate(nul);
                        path_str = String::from_utf8_lossy(&tmp).into_owned();
                    }
                }
                let basename = std::path::Path::new(&path_str)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();

                let mmap_arc: Option<std::sync::Arc<memmap2::Mmap>> = if !basename.is_empty() {
                    kernel.homebrew_dir.as_ref().and_then(|dir| {
                        let candidate = dir.join(&basename);
                        std::fs::File::open(&candidate)
                            .ok()
                            .and_then(|f| unsafe { memmap2::Mmap::map(&f) }.ok())
                            .map(std::sync::Arc::new)
                    })
                } else {
                    None
                };

                let is_domain = kernel.sessions.get(&session_handle).map(|s| s.is_domain).unwrap_or(false);
                let new_obj_id = if is_domain {
                    kernel.sessions.get(&session_handle).map(|s| s.next_domain_object_id).unwrap_or(0)
                } else {
                    0
                };

                let mmap_len = mmap_arc.as_ref().map(|m| m.len()).unwrap_or(0);
                if let Some(m) = mmap_arc {
                    kernel.open_files.insert((session_handle, new_obj_id), m);
                }

                log::debug!("IFileSystem.OpenFile path={:?} basename={:?} mmap_len={} → IFile object_id={}", path_str, basename, mmap_len, new_obj_id);
                return return_subsession(kernel, ctx, session_handle, "IFile");
            }
            9 => {
                log::debug!("IFileSystem.OpenDirectory → IDirectory sub-session");
                kernel.dir_cursor.insert(session_handle, 0);
                return return_subsession(kernel, ctx, session_handle, "IDirectory");
            }
            7 => {
                let entry_type: u32 = 1;
                log::debug!("IFileSystem.GetEntryType → file");
                return build_ipc_response(ctx, 0, &entry_type.to_le_bytes(), &[]);
            }
            14 => {
                log::debug!("IFileSystem.GetFileTimeStampRaw → zeros");
                return build_ipc_response(ctx, 0, &[0u8; 0x20], &[]);
            }
            _ => {}
        }
    }

    if port_name == "IFile" {
        let obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let per_session = kernel.open_files.get(&(session_handle, obj_id)).cloned();
        match cmd_id {
            0 => {
                let in_off = ctx.cmif_in_data_off;
                let avail = ctx.buf.len().saturating_sub(in_off);
                if avail < 24 {
                    log::warn!("IFile.Read: short input ({} bytes)", avail);
                    return build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[]);
                }
                let offset = i64::from_le_bytes([
                    ctx.buf[in_off + 8], ctx.buf[in_off + 9], ctx.buf[in_off + 10], ctx.buf[in_off + 11],
                    ctx.buf[in_off + 12], ctx.buf[in_off + 13], ctx.buf[in_off + 14], ctx.buf[in_off + 15],
                ]);
                let read_size = u64::from_le_bytes([
                    ctx.buf[in_off + 16], ctx.buf[in_off + 17], ctx.buf[in_off + 18], ctx.buf[in_off + 19],
                    ctx.buf[in_off + 20], ctx.buf[in_off + 21], ctx.buf[in_off + 22], ctx.buf[in_off + 23],
                ]);
                let file_bytes: &[u8] = match per_session.as_ref() {
                    Some(m) => &m[..],
                    None => kernel.nro_mmap.as_ref().map(|m| &m[..]).unwrap_or(&[]),
                };
                let target = ctx.recv_buffers.iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let mut bytes_read: u64 = 0;
                if let Some(buf) = target {
                    let start = (offset.max(0) as usize).min(file_bytes.len());
                    let want = (read_size as usize).min(buf.size as usize);
                    let end = start.saturating_add(want).min(file_bytes.len());
                    let slice = &file_bytes[start..end];
                    let _ = kernel.address_space.write(buf.addr, slice);
                    bytes_read = slice.len() as u64;
                    log::debug!(
                        "IFile.Read (sess={:#x} obj={}) off={:#x} size={:#x} → {} bytes (file total {}, per_session={})",
                        session_handle, obj_id, offset, read_size, slice.len(), file_bytes.len(), per_session.is_some()
                    );
                } else {
                    log::warn!("IFile.Read: no recv buffer (off={:#x} size={:#x})", offset, read_size);
                }
                return build_ipc_response(ctx, 0, &bytes_read.to_le_bytes(), &[]);
            }
            1 => {
                log::debug!("IFile.Write → SUCCESS (discarded)");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            2 => {
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            3 => {
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let size: i64 = match per_session.as_ref() {
                    Some(m) => m.len() as i64,
                    None => kernel.nro_mmap.as_ref().map(|m| m.len() as i64).unwrap_or(0),
                };
                log::debug!("IFile.GetSize (sess={:#x} obj={}) → {} (per_session={})", session_handle, obj_id, size, per_session.is_some());
                return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
            }
            _ => {}
        }
    }

    if port_name == "IDirectory" {
        match cmd_id {
            0 => {
                let target = ctx.recv_buffers.iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let buf = match target {
                    Some(b) => b,
                    None => {
                        log::warn!("IDirectory.Read: no recv buffer");
                        return build_ipc_response(ctx, 0, &0i64.to_le_bytes(), &[]);
                    }
                };
                let max_entries = (buf.size as usize) / 0x310;
                let cursor = *kernel.dir_cursor.get(&session_handle).unwrap_or(&0);
                let entries = enumerate_homebrew_nros(&kernel.homebrew_dir);
                let remaining = entries.len().saturating_sub(cursor);
                let to_emit = remaining.min(max_entries);
                let mut payload = vec![0u8; to_emit * 0x310];
                for (i, e) in entries.iter().skip(cursor).take(to_emit).enumerate() {
                    let base = i * 0x310;
                    let name_bytes = e.name.as_bytes();
                    let name_len = name_bytes.len().min(0x300);
                    payload[base..base + name_len].copy_from_slice(&name_bytes[..name_len]);
                    payload[base + 0x301 + 3] = 1;
                    payload[base + 0x308..base + 0x310].copy_from_slice(&e.size.to_le_bytes());
                }
                if !payload.is_empty() {
                    let _ = kernel.address_space.write(buf.addr, &payload);
                }
                kernel.dir_cursor.insert(session_handle, cursor + to_emit);
                log::info!(
                    "IDirectory.Read cursor={} max_entries={} → {} of {} entries (homebrew_dir={:?})",
                    cursor, max_entries, to_emit, entries.len(), kernel.homebrew_dir
                );
                let total: i64 = to_emit as i64;
                return build_ipc_response(ctx, 0, &total.to_le_bytes(), &[]);
            }
            1 => {
                let count = enumerate_homebrew_nros(&kernel.homebrew_dir).len() as i64;
                log::info!("IDirectory.GetEntryCount → {}", count);
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            _ => {}
        }
    }

    if port_name == "IFsStorage" {
        match cmd_id {
            0 => {
                let off_lo = ctx.cmif_in_data_off;
                let read_in = &ctx.buf[off_lo..off_lo + 16];
                let offset = i64::from_le_bytes([
                    read_in[0], read_in[1], read_in[2], read_in[3],
                    read_in[4], read_in[5], read_in[6], read_in[7],
                ]);
                let read_size = u64::from_le_bytes([
                    read_in[8], read_in[9], read_in[10], read_in[11],
                    read_in[12], read_in[13], read_in[14], read_in[15],
                ]);
                let romfs = kernel.nro_romfs();
                let target = ctx.recv_buffers.iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(buf) = target {
                    let start = (offset.max(0) as usize).min(romfs.len());
                    let want = (read_size as usize).min(buf.size as usize);
                    let end = start.saturating_add(want).min(romfs.len());
                    let slice = &romfs[start..end];
                    let _ = kernel.address_space.write(buf.addr, slice);
                    log::debug!("IFsStorage.Read off={:#x} size={:#x} → {} bytes (romfs total {})", offset, read_size, slice.len(), romfs.len());
                } else {
                    log::warn!("IFsStorage.Read: no recv buffer (off={:#x} size={:#x})", offset, read_size);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let size = kernel.nro_romfs().len() as i64;
                log::debug!("IFsStorage.GetSize → {}", size);
                return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
            }
            _ => {}
        }
    }

    if port_name == "set" || port_name == "set:sys" {
        if let Some(outcome) = cmif_dispatch_set(kernel, ctx) {
            log::debug!("set.cmd_{} → {} bytes (rc={:#x}) via #[service]", cmd_id, outcome.inline_out.len(), outcome.result);
            return build_ipc_response(ctx, outcome.result, &outcome.inline_out, &[]);
        }
    }

    if port_name == "pl:u" || port_name == "pl:s" {
        if let Some(outcome) = cmif_dispatch_pl(kernel, ctx) {
            log::debug!("pl.cmd_{} → {} bytes (rc={:#x}) via #[service]", cmd_id, outcome.inline_out.len(), outcome.result);
            return build_ipc_response(ctx, outcome.result, &outcome.inline_out, &[]);
        }
    }

    if let Some((data, handle_opt)) = applet_command_response(kernel, port_name, cmd_id) {
        log::debug!("{}.cmd_{} → returning data ({} bytes, handle={:?})", port_name, cmd_id, data.len(), handle_opt);
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

    log::debug!("IHOSBinderDriver.TransactParcel{} binder={} code={} in_size={} reply_size={}",
        if cmd_id == 3 { "Auto" } else { "" }, binder_id, code, in_parcel.len(), reply.len());

    if code == IGBP_REQUEST_BUFFER || code == IGBP_DEQUEUE_BUFFER {
        let preview = &in_parcel[..in_parcel.len().min(64)];
        log::info!("IGBP in code={} (len={}): {:02x?}", code, in_parcel.len(), preview);
        let preview = &reply[..reply.len().min(96)];
        log::info!("IGBP reply code={} (len={}): {:02x?}", code, reply.len(), preview);
    }

    let out_dst = ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0).copied()
        .or_else(|| ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied());
    if let Some(rb) = out_dst {
        let n = reply.len().min(rb.size as usize);
        match kernel.address_space.write(rb.addr, &reply[..n]) {
            Ok(()) => {}
            Err(e) => log::error!("binder reply write FAILED to {:#x} ({} bytes): {:?}", rb.addr, n, e),
        }
    } else {
        log::warn!("binder transact code={} produced {}-byte reply but no recv buffer descriptor", code, reply.len());
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
                log::warn!("IGBP::SetPreallocatedBuffer slot={} has=0 — no buffer", slot);
                return ParcelBuilder::new().finish();
            }
            let mut gb = parse_flattened_graphic_buffer(&mut reader);
            if let Some(ref mut g) = gb {
                if g.nvmap_id == 0 && g.kind == 254 {
                    let tiled_size = compute_tiled_size(g.stride, g.height, g.block_height_log2);
                    let needed = (g.buffer_offset as usize).saturating_add(tiled_size);
                    let pick = kernel.nvdrv.nvmap_handles.iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) >= needed)
                        .min_by_key(|(_, h)| h.size as usize)
                        .map(|(id, _)| *id);
                    if let Some(id) = pick {
                        g.nvmap_id = id;
                        log::info!(
                            "SetPreallocatedBuffer fixup: nvmap_id=0 → {} (off={:#x} tiled_size={:#x} needed={:#x})",
                            id, g.buffer_offset, tiled_size, needed
                        );
                    }
                }
            }
            let parsed = gb.is_some();
            let (nvmap_id, w, h, off) = gb.as_ref().map(|g| (g.nvmap_id, g.width, g.height, g.buffer_offset)).unwrap_or((0, 0, 0, 0));
            kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                if let Some(gb) = gb {
                    bq.set_preallocated(slot, gb);
                }
            });
            log::info!("IGBP::SetPreallocatedBuffer binder={} slot={} parsed={} nvmap_id={} {}x{} off={:#x}", binder_id, slot, parsed, nvmap_id, w, h, off);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_REQUEST_BUFFER => {
            kernel.nvdrv.stats.request_buffer_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            kernel.nvdrv.stats.dequeue_buffer_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _async_ = reader.read_i32();
            let _w = reader.read_u32();
            let _h = reader.read_u32();
            let _fmt = reader.read_i32();
            let _usage = reader.read_u32();
            let (slot, free, deq, queued) = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                let s = bq.dequeue();
                (s, bq.free.len(), bq.dequeued.len(), bq.queued.len())
            });
            log::info!("IGBP::DequeueBuffer binder={} → slot={} (free={} deq={} queued={})", binder_id, slot, free, deq, queued);
            let mut p = ParcelBuilder::new();
            p.write_u32(slot);
            p.write_u32(1);
            p.write_flattened_zero_fence();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUEUE_BUFFER => {
            kernel.nvdrv.stats.queue_buffer_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let _has = reader.read_u32();
            let _timestamp = reader.read_u64();
            let _is_auto = reader.read_i32();
            let _crop_l = reader.read_i32();
            let _crop_t = reader.read_i32();
            let _crop_r = reader.read_i32();
            let _crop_b = reader.read_i32();
            let _scaling = reader.read_i32();
            let _transform = reader.read_i32();
            let _sticky = reader.read_u32();
            let _async = reader.read_i32();
            let swap_interval = reader.read_i32().unwrap_or(1).max(1);

            let gb_opt = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                bq.queue(slot);
                let r = bq.request_buffer(slot).cloned();
                let slot_count = bq.slots.len();
                let has_buf = bq.slots.get(slot as usize).and_then(|s| s.buffer.as_ref()).is_some();
                (r, slot_count, has_buf)
            });
            let (gb_opt, slot_count, has_buf) = gb_opt;
            log::info!("IGBP::QueueBuffer binder={} slot={} swap_interval={} slot_count={} has_buf={} gb_some={}", binder_id, slot, swap_interval, slot_count, has_buf, gb_opt.is_some());

            if let Some(gb) = gb_opt {
                let bpp: usize = 4;
                let linear_size = (gb.stride as usize) * (gb.height as usize) * bpp;
                let tiled_size = compute_tiled_size(gb.stride, gb.height, gb.block_height_log2);
                let resolved: Option<(u64, bool)> = if let Some(nvmap) = kernel.nvdrv.nvmap_handles.get(&gb.nvmap_id) {
                    let actual_size = nvmap.size as usize;
                    let is_tiled = actual_size >= tiled_size && gb.kind == 254 && gb.block_height_log2 != 0;
                    log::info!(
                        "QueueBuffer fast-path: nvmap_id={} addr={:#x} off={:#x} size={:#x} kind={} bh_log2={} tiled={} (slot={})",
                        gb.nvmap_id, nvmap.address, gb.buffer_offset, actual_size, gb.kind, gb.block_height_log2, is_tiled, slot
                    );
                    Some((nvmap.address.wrapping_add(gb.buffer_offset), is_tiled))
                } else {
                    let mut candidates: Vec<(u32, u64, u32)> = kernel.nvdrv.nvmap_handles.iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) == linear_size)
                        .map(|(id, h)| (*id, h.address, h.size))
                        .collect();
                    candidates.sort_by_key(|(id, _, _)| *id);
                    if let Some(&(id, addr, size)) = candidates.last() {
                        log::info!(
                            "QueueBuffer fallback pick newest: nvmap_id={} addr={:#x} size={:#x} (slot={} candidates={})",
                            id, addr, size, slot, candidates.len()
                        );
                        Some((addr, false))
                    } else {
                        log::warn!(
                            "QueueBuffer fallback: no exact-size candidate (linear_size={:#x} slot={} total_handles={})",
                            linear_size, slot, kernel.nvdrv.nvmap_handles.len()
                        );
                        None
                    }
                };
                if let Some((addr, is_tiled)) = resolved {
                    let read_size = if is_tiled { tiled_size } else { linear_size };
                    let mut raw = vec![0u8; read_size];
                    let mut effective_tiled = is_tiled;
                    let mut effective_addr = addr;
                    let mut effective_bh_log2 = gb.block_height_log2;
                    if kernel.address_space.read(addr, &mut raw).is_ok() {
                        if is_tiled && raw.iter().all(|&b| b == 0) {

                            let (tiled_rt_cpu, dma_bh_log2, dma_stride, dma_height) = {
                                let dma = kernel.nvdrv.gpu.maxwell_dma.lock();
                                (dma.last_tiled_dst_cpu, dma.last_tiled_dst_bh_log2, dma.last_tiled_dst_stride, dma.last_tiled_dst_height)
                            };
                            let mut found = false;
                            if tiled_rt_cpu != 0 {

                                let dma_tiled_size = compute_tiled_size(dma_stride.max(gb.stride), dma_height.max(gb.height), dma_bh_log2);
                                let mut tiled_raw = vec![0u8; dma_tiled_size.max(tiled_size)];
                                if kernel.address_space.read(tiled_rt_cpu, &mut tiled_raw).is_ok()
                                    && tiled_raw.iter().any(|&b| b != 0)
                                {
                                    log::info!(
                                        "QueueBuffer tiled-rt-redirect: slot={} slot_tiled={:#x} → rt_cpu={:#x} (gralloc_bh={} dma_bh={} dma_stride={} dma_h={})",
                                        slot, addr, tiled_rt_cpu, gb.block_height_log2, dma_bh_log2, dma_stride, dma_height
                                    );
                                    raw = tiled_raw;
                                    effective_tiled = true;
                                    effective_addr = tiled_rt_cpu;
                                    effective_bh_log2 = dma_bh_log2;
                                    found = true;
                                }
                            }
                            if !found {

                                let mut best: Option<(u32, u64)> = None;
                                for (id, h) in &kernel.nvdrv.nvmap_handles {
                                    if h.address == 0 || (h.size as usize) != linear_size { continue; }
                                    match best {
                                        None => { best = Some((*id, h.address)); }
                                        Some((best_id, _)) if *id > best_id => { best = Some((*id, h.address)); }
                                        _ => {}
                                    }
                                }
                                if let Some((id, lin_addr)) = best {
                                    let mut lin_raw = vec![0u8; linear_size];
                                    if kernel.address_space.read(lin_addr, &mut lin_raw).is_ok()
                                        && lin_raw.iter().any(|&b| b != 0)
                                    {
                                        log::info!(
                                            "QueueBuffer tiled-empty fallback nvmap_id={} addr={:#x} (slot={})",
                                            id, lin_addr, slot
                                        );
                                        raw = lin_raw;
                                        effective_tiled = false;
                                        effective_addr = lin_addr;
                                    }
                                }
                            }
                        }
                        let mut pixels = if effective_tiled {
                            unswizzle_block_linear(&raw, gb.stride, gb.height, bpp, effective_bh_log2)
                        } else {
                            raw
                        };
                        if pixels.len() < linear_size { pixels.resize(linear_size, 0); }
                        for px in pixels.chunks_exact_mut(4) { px[3] = 0xFF; }
                        let nz = pixels.iter().filter(|b| **b != 0).count();
                        let checksum: u32 = pixels.chunks_exact(4).map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).fold(0u32, |a,b| a.wrapping_add(b));
                        log::info!(
                            "QueueBuffer submit slot={} parsed_nvmap_id={} addr={:#x} {}x{} tiled={} nz={} cksum={:#x}",
                            slot, gb.nvmap_id, effective_addr, gb.width, gb.height, effective_tiled, nz, checksum
                        );
                        kernel.nvdrv.submit_frame(nexium_nvdrv::QueuedFrame {
                            width: gb.width,
                            height: gb.height,
                            pixels,
                        });
                    } else {
                        log::warn!("QueueBuffer: failed to read slot {} addr={:#x} read_size={:#x}", slot, addr, read_size);
                    }
                } else {
                    log::warn!("QueueBuffer: no nvmap candidate for size {} (slot {})", linear_size, slot);
                }
            } else {
                log::warn!("QueueBuffer: slot {} has no GraphicBuffer", slot);
            }

            let vsyncs: Vec<u32> = kernel.vsync_handles.iter().copied().collect();
            for h in vsyncs {
                kernel.event_signals.insert(h, true);
                kernel.threads.signal_handle(h);
                kernel.nvdrv.stats.vsync_signals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            kernel.last_vsync = std::time::Instant::now();

            kernel.nvdrv.pace_swap(swap_interval);

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
            let what = reader.read_i32().unwrap_or(0);
            let (w, h) = kernel.nvdrv.with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
            let value: i32 = match what {
                0 => w as i32,
                1 => h as i32,
                2 => 1,
                3 => 2,
                _ => 0,
            };
            log::debug!("IGBP::Query what={} → {}", what, value);
            let mut p = ParcelBuilder::new();
            p.write_u32(value as u32);
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
    objects_off: usize,
    objects_size: usize,
}

impl<'a> ParcelReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let (payload_off, objects_off, objects_size) = if data.len() >= 16 {
            let po = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            let os = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            let oo = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
            (po, oo, os)
        } else { (0, 0, 0) };
        Self { data, payload_off, cursor: payload_off, objects_off, objects_size }
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

    fn first_binder_handle(&self) -> Option<u32> {
        if self.objects_size < FLAT_BINDER_OBJECT_SIZE || self.objects_off == 0 {
            return None;
        }
        let end = self.objects_off.checked_add(FLAT_BINDER_OBJECT_SIZE)?;
        if end > self.data.len() { return None; }
        let obj = &self.data[self.objects_off..end];
        let handle = u32::from_le_bytes([obj[8], obj[9], obj[10], obj[11]]);
        Some(handle)
    }
}

const FLAT_BINDER_OBJECT_SIZE: usize = 24;

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

    fn write_flattened_graphic_buffer(&mut self, gb: &nexium_nvdrv::GraphicBuffer) {
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

fn parse_flattened_graphic_buffer(reader: &mut ParcelReader) -> Option<nexium_nvdrv::GraphicBuffer> {
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
    let inline_nvmap_id = ints.get(1).copied().unwrap_or(0);
    let binder_handle = reader.first_binder_handle().unwrap_or(0);
    let nvmap_id = if binder_handle != 0 {
        binder_handle
    } else if inline_nvmap_id != 0 {
        inline_nvmap_id
    } else {
        ints.get(19).copied().unwrap_or(0)
    };
    let buffer_offset = ints.get(20).copied().unwrap_or(0);
    let kind = ints.get(21).copied().unwrap_or(0);
    let block_height_log2 = ints.get(22).copied().unwrap_or(4);

    log::debug!(
        "parse_flattened_graphic_buffer: nvmap_id={} (binder_handle={} inline={}) off={:#x} kind={} bh_log2={}",
        nvmap_id, binder_handle, inline_nvmap_id, buffer_offset, kind, block_height_log2
    );

    Some(nexium_nvdrv::GraphicBuffer {
        width,
        height,
        stride,
        format,
        usage,
        kind,
        nvmap_id,
        buffer_offset: buffer_offset as u64,
        size: stride * height * 4,
        block_height_log2,
    })
}

fn dispatch_nvdrv_command(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, port_name: &str) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    log::debug!("nvdrv:{}.cmd_{}", port_name, cmd_id);

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

            let req = nexium_nvdrv::IoctlRequest {
                fd, ioctl_id, in_data, out_size,
            };
            let addr_space = kernel.address_space.clone();
            let addr_space_w = kernel.address_space.clone();
            let outcome = kernel.nvdrv.dispatch_ioctl_with_mem(
                req,
                &|addr, buf| addr_space.read(addr, buf).is_ok(),
                &|addr, buf| addr_space_w.write(addr, buf).is_ok(),
            );

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
        log::debug!("→ {} sub-object id={}", sub_service, object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, sub_service.to_string());
        kernel.sessions.insert(h, session);
        log::debug!("→ {} sub-session handle={:#x}", sub_service, h);
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
        ("fsp-srv", 200) => Some("IFsStorage"),
        ("fsp-srv", 202) => Some("IFsStorage"),
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
        ("IApplicationDisplayService", 5202) => {
            let h = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(h, false);
            kernel.vsync_handles.insert(h);
            log::info!("IApplicationDisplayService.GetDisplayVsyncEvent → vsync_handle={:#x}", h);
            Some((Vec::new(), Some(h)))
        }
        ("IApplicationDisplayService", 5203) => Some((Vec::new(), None)),

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
            kernel.event_signals.insert(handle, false);
            kernel.vsync_handles.insert(handle);
            log::info!("IHOSBinderDriver.GetNativeHandle → BinderEvent {:#x} (registered as vsync)", handle);
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

fn svc_get_thread_id(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(1) as u32 } else { 0 };
    let target = if handle == 0 || handle == 0xFFFF8000 {
        kernel.threads.current_handle().unwrap_or(kernel.main_thread_handle)
    } else {
        handle
    };
    let tid = kernel.threads.threads.get(&target).map(|t| t.tid).unwrap_or(1);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, tid);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_id(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 0x4F4F4F4F_4F4F4F4F);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_clear_event(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) as u32 } else { 0 };
    kernel.event_signals.insert(handle, false);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reset_signal(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) as u32 } else { 0 };
    kernel.event_signals.insert(handle, false);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_wait_for_address(kernel: &mut Kernel) -> u32 {
    let (addr, arb_type, value, timeout_ns) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1) as u32, cpu.get_register(2) as u32, cpu.get_register(3))
    } else {
        return 1;
    };

    let mut buf = [0u8; 4];
    let read_ok = kernel.address_space.read(addr, &mut buf).is_ok();
    let current = u32::from_le_bytes(buf);

    let should_wait = match arb_type {
        0 | 1 => current < value,
        2 => current == value,
        _ => false,
    };

    if !read_ok || !should_wait {
        const KERNEL_INVALID_STATE: u32 = 1 | (125 << 9);
        if let Some(cpu) = &mut kernel.cpu {
            cpu.set_register(0, KERNEL_INVALID_STATE as u64);
        }
        return KERNEL_INVALID_STATE;
    }

    if arb_type == 1 {
        let _ = kernel.address_space.write(addr, &current.wrapping_sub(1).to_le_bytes());
    }

    let cap = std::time::Duration::from_millis(100);
    let wait = if timeout_ns == u64::MAX || timeout_ns == 0 {
        cap
    } else {
        std::time::Duration::from_nanos(timeout_ns).min(cap)
    };
    if wait > std::time::Duration::ZERO {
        std::thread::sleep(wait);
    }

    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, KERNEL_TIMEOUT as u64);
    }
    KERNEL_TIMEOUT
}

fn svc_signal_to_address(kernel: &mut Kernel) -> u32 {
    let (addr, signal_type, value, count) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1) as u32, cpu.get_register(2) as u32, cpu.get_register(3) as i32)
    } else {
        return 1;
    };

    let mut buf = [0u8; 4];
    let _ = kernel.address_space.read(addr, &mut buf);
    let current = u32::from_le_bytes(buf);

    match signal_type {
        0 => {}
        1 if current == value => {
            let _ = kernel.address_space.write(addr, &current.wrapping_add(1).to_le_bytes());
        }
        2 if current == value => {}
        _ => {}
    }

    if count < 0 {
        kernel.threads.wake_all_on_arbiter(addr);
    } else {
        for _ in 0..count {
            if kernel.threads.wake_one_on_arbiter(addr).is_none() {
                break;
            }
        }
    }

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
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

        23 | 24 | 25 | 26 | 27 => 0,

        28 => 0x1000,

        29 => kernel.cycle_count,

        30 => 1,

        31 => 0,

        41 => 0,
        _  => {
            log::warn!("svcGetInfo: unsupported type {} — returning InvalidEnumValue (0xF001)", info_type);
            if let Some(cpu) = &mut kernel.cpu {
                cpu.set_register(0, 0xF001);
                cpu.set_register(1, 0);
            }
            return 0xF001;
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

fn svc_create_event(kernel: &mut Kernel) -> u32 {
    log::debug!("svcCreateEvent");
    let writable = kernel.handles.create_handle(HandleType::Event);
    let readable = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(writable, false);
    kernel.event_signals.insert(readable, false);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, writable as u64);
        cpu.set_register(2, readable as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    log::debug!("  created event writable={:#x} readable={:#x}", writable, readable);
    SUCCESS
}

fn svc_map_transfer_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapTransferMemory");
    SUCCESS
}

fn dump_regs(kernel: &Kernel, tag: &str) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Some(cpu) = &kernel.cpu {
        log::trace!(
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
    log::debug!("svcCloseHandle handle={:#x} ({})", handle, kind);
    dump_regs(kernel, "CloseHandle ENTRY");
    kernel.handles.close_handle(handle);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_thread(kernel: &mut Kernel) -> u32 {
    let (entry, arg, sp, priority, core) = if let Some(cpu) = &kernel.cpu {
        (
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
            cpu.get_register(4) as i32,
            cpu.get_register(5) as i32,
        )
    } else {
        return 1;
    };

    let handle = kernel.handles.create_handle(crate::kernel::handles::HandleType::Thread);
    let tls_va = kernel.threads.alloc_tls();

    let mut ctx = crate::kernel::threads::ThreadCtx::zero();
    ctx.x[0] = arg;
    ctx.sp = sp;
    ctx.pc = entry;
    ctx.tpidrro_el0 = tls_va;

    kernel.threads.add_thread(handle, ctx, tls_va, sp);
    if let Some(t) = kernel.threads.threads.get_mut(&handle) {
        t.priority = priority;
    }

    log::info!(
        "svcCreateThread entry={:#x} arg={:#x} sp={:#x} prio={} core={} -> handle={:#x} tls={:#x}",
        entry, arg, sp, priority, core, handle, tls_va
    );

    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_start_thread(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) as u32 } else { return 1; };
    log::info!("svcStartThread handle={:#x}", handle);
    kernel.threads.transition_state(handle, crate::kernel::threads::ThreadState::Ready);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_exit_thread(kernel: &mut Kernel) -> u32 {
    let current = kernel.threads.current_handle();
    log::debug!("svcExitThread current={:?}", current);
    if let Some(cpu) = kernel.cpu.as_ref() {
        kernel.threads.yield_with_state(cpu, crate::kernel::threads::ThreadState::Exited);
    }
    SUCCESS
}

fn svc_sleep_thread(kernel: &mut Kernel) -> u32 {
    let ns = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) } else { 0 };
    let signed = ns as i64;
    if signed > 0 {
        let dur = std::time::Duration::from_nanos(ns).min(std::time::Duration::from_millis(100));
        std::thread::sleep(dur);
    } else if signed == -1 {
        std::thread::yield_now();
    }
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_flush_data_cache(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_priority(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = &kernel.cpu { cpu.get_register(1) as u32 } else { return 1; };
    let prio = kernel.threads.threads.get(&handle).map(|t| t.priority).unwrap_or(0x2C);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, prio as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_priority(kernel: &mut Kernel) -> u32 {
    let (handle, priority) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0) as u32, cpu.get_register(1) as i32)
    } else {
        return 1;
    };
    if let Some(t) = kernel.threads.threads.get_mut(&handle) {
        t.priority = priority;
    }
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_core_mask(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 0);
        cpu.set_register(2, 0xF);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_core_mask(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_current_processor_number(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, 0);
    }
    0
}

fn svc_send_sync_request_light(kernel: &mut Kernel) -> u32 {
    svc_send_sync_request(kernel)
}

fn svc_send_sync_request_with_user_buffer(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendSyncRequestWithUserBuffer (treating as svcSendSyncRequest)");
    svc_send_sync_request(kernel)
}

fn svc_send_async_request_with_user_buffer(kernel: &mut Kernel) -> u32 {
    log::debug!("svcSendAsyncRequestWithUserBuffer");
    let handle = kernel.handles.create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(handle, true);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_return_from_exception(kernel: &mut Kernel) -> u32 {
    log::debug!("svcReturnFromException");
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_flush_entire_data_cache(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_future_thread_info(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_last_thread_info(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_limit_value(kernel: &mut Kernel) -> u32 {
    let limitable = if let Some(cpu) = &kernel.cpu { cpu.get_register(2) as u32 } else { 0 };
    let value: u64 = match limitable {
        0 => 0x40_000_000,
        1 => 1024,
        2 => 1024,
        3 => 8,
        4 => 0x80_000,
        5 => 64,
        _ => 0,
    };
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, value);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_current_value(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_peak_value(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_activity(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_context3(kernel: &mut Kernel) -> u32 {
    let (out_ptr, handle) = if let Some(cpu) = &kernel.cpu {
        (cpu.get_register(0), cpu.get_register(1) as u32)
    } else {
        return 1;
    };
    if out_ptr != 0 {
        if let Some(t) = kernel.threads.threads.get(&handle) {
            let mut buf = [0u8; 0x320];
            for i in 0..29 {
                let off = i * 8;
                buf[off..off + 8].copy_from_slice(&t.ctx.x[i].to_le_bytes());
            }
            buf[0xe8..0xf0].copy_from_slice(&t.ctx.sp.to_le_bytes());
            buf[0xf8..0x100].copy_from_slice(&t.ctx.pc.to_le_bytes());
            let _ = kernel.address_space.write(out_ptr, &buf);
        }
    }
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_synchronize_preemption_state(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_session(kernel: &mut Kernel) -> u32 {
    let server = kernel.handles.create_handle(crate::kernel::handles::HandleType::Session);
    let client = kernel.handles.create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_accept_session(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reply_and_receive_light(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_reply_and_receive(kernel: &mut Kernel) -> u32 {
    log::debug!("svcReplyAndReceive (stub → TIMEOUT)");
    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, KERNEL_TIMEOUT as u64);
    }
    KERNEL_TIMEOUT
}

fn svc_reply_and_receive_with_user_buffer(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::SharedMemory);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_transfer_memory(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_interrupt_event(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(h, false);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_io_mapping(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_debug_active_process(kernel: &mut Kernel) -> u32 {
    const KERNEL_INVALID_HANDLE: u32 = 1 | (114 << 9);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
    }
    KERNEL_INVALID_HANDLE
}

fn svc_break_debug_process(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_terminate_debug_process(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_event(kernel: &mut Kernel) -> u32 {
    const KERNEL_NO_DEBUG_EVENT: u32 = 1 | (140 << 9);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, KERNEL_NO_DEBUG_EVENT as u64);
    }
    KERNEL_NO_DEBUG_EVENT
}

fn svc_continue_debug_event(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_list(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, 1);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_list(kernel: &mut Kernel) -> u32 {
    let count = kernel.threads.threads.len() as u64;
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, count);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_port(kernel: &mut Kernel) -> u32 {
    let server = kernel.handles.create_handle(crate::kernel::handles::HandleType::Port);
    let client = kernel.handles.create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_manage_named_port(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_connect_to_port(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_resource_limit(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(crate::kernel::handles::HandleType::Process);
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_resource_limit_limit_value(kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = &mut kernel.cpu {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_call_secure_monitor(kernel: &mut Kernel) -> u32 {
    let smc_id = if let Some(cpu) = &kernel.cpu { cpu.get_register(0) as u32 } else { 0 };
    log::debug!("svcCallSecureMonitor smc_id={:#x} (HLE: returning success)", smc_id);
    if let Some(cpu) = &mut kernel.cpu {
        for r in 0..=7 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn compute_tiled_size(stride: u32, height: u32, block_height_log2: u32) -> usize {
    const GOB_W: usize = 64;
    const GOB_H: usize = 8;
    const GOB_SIZE: usize = 512;
    let bpp: usize = 4;
    let width_bytes = stride as usize * bpp;
    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_rows = (height as usize + rows_per_block - 1) / rows_per_block;
    gobs_per_row * block_rows * block_height * GOB_SIZE
}

fn unswizzle_block_linear(src: &[u8], stride: u32, height: u32, bpp: usize, block_height_log2: u32) -> Vec<u8> {
    const GOB_W: usize = 64;
    const GOB_H: usize = 8;
    const GOB_SIZE: usize = 512;
    let stride_px = stride as usize;
    let height = height as usize;
    let dst_stride = stride_px * bpp;
    let mut dst = vec![0u8; dst_stride * height];
    let width_bytes = stride_px * bpp;
    let block_height = 1usize << block_height_log2 as usize;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_row_stride_bytes = gobs_per_row * block_height * GOB_SIZE;
    for y in 0..height {
        let block_y = y / rows_per_block;
        let y_in_block = y - block_y * rows_per_block;
        let gob_row_in_block = y_in_block / GOB_H;
        let y_in_gob = y_in_block - gob_row_in_block * GOB_H;
        let block_row_offset = block_y * block_row_stride_bytes;
        for x in 0..stride_px {
            let byte_x = x * bpp;
            let gob_col = byte_x / GOB_W;
            let x_in_gob = byte_x - gob_col * GOB_W;
            let gob_offset = block_row_offset
                + gob_col * block_height * GOB_SIZE
                + gob_row_in_block * GOB_SIZE;
            let in_gob = ((x_in_gob >> 5) & 1) * 256
                + ((y_in_gob >> 1) & 3) * 64
                + ((x_in_gob >> 4) & 1) * 32
                + (y_in_gob & 1) * 16
                + (x_in_gob & 15);
            let src_off = gob_offset + in_gob;
            let dst_off = y * dst_stride + byte_x;
            if src_off + bpp <= src.len() && dst_off + bpp <= dst.len() {
                dst[dst_off..dst_off + bpp].copy_from_slice(&src[src_off..src_off + bpp]);
            }
        }
    }
    dst
}

struct AddressSpaceMemory<'a> {
    addr_space: &'a nexium_memory::AddressSpace,
}

impl nexium_cmif::Memory for AddressSpaceMemory<'_> {
    fn read(&self, addr: u64, dst: &mut [u8]) -> bool {
        self.addr_space.read(addr, dst).is_ok()
    }

    fn write(&self, addr: u64, src: &[u8]) -> bool {
        self.addr_space.write(addr, src).is_ok()
    }
}

fn make_cmif_ctx<'a>(
    ctx: &'a ipc::IpcCtx,
    mem: &'a AddressSpaceMemory<'a>,
    recv_buffers: &'a [nexium_cmif::CmifBuffer],
    recv_statics: &'a [nexium_cmif::CmifBuffer],
    send_buffers: &'a [nexium_cmif::CmifBuffer],
    send_statics: &'a [nexium_cmif::CmifBuffer],
) -> nexium_cmif::DispatchCtx<'a> {
    let in_off = ctx.cmif_in_data_off;
    let in_len = ctx.cmif_in_data_len;
    let end = (in_off + in_len).min(ctx.buf.len());
    nexium_cmif::DispatchCtx {
        input_data: &ctx.buf[in_off..end],
        recv_buffers,
        recv_statics,
        send_buffers,
        send_statics,
        mem,
    }
}

fn convert_buffers(src: &[ipc::IpcBuffer]) -> Vec<nexium_cmif::CmifBuffer> {
    src.iter()
        .map(|b| nexium_cmif::CmifBuffer { addr: b.addr, size: b.size })
        .collect()
}

struct HomebrewEntry {
    name: String,
    size: i64,
}

fn enumerate_homebrew_nros(dir: &Option<std::path::PathBuf>) -> Vec<HomebrewEntry> {
    let Some(dir) = dir else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<HomebrewEntry> = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let is_nro = path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("nro")).unwrap_or(false);
        if !is_nro { continue; }
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let size = entry.metadata().map(|m| m.len() as i64).unwrap_or(0);
        out.push(HomebrewEntry { name, size });
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn cmif_dispatch_set(
    kernel: &mut Kernel,
    ctx: &ipc::IpcCtx,
) -> Option<nexium_cmif::DispatchOutcome> {
    let recv_buffers = convert_buffers(&ctx.recv_buffers);
    let recv_statics = convert_buffers(&ctx.recv_statics);
    let send_buffers = convert_buffers(&ctx.send_buffers);
    let send_statics = convert_buffers(&ctx.send_statics);
    let mem = AddressSpaceMemory { addr_space: &*kernel.address_space };
    let mut cmif_ctx = make_cmif_ctx(ctx, &mem, &recv_buffers, &recv_statics, &send_buffers, &send_statics);
    kernel.services.set.dispatch_cmif(ctx.cmif_in.cmd_id, &mut cmif_ctx)
}

fn cmif_dispatch_pl(
    kernel: &mut Kernel,
    ctx: &ipc::IpcCtx,
) -> Option<nexium_cmif::DispatchOutcome> {
    let recv_buffers = convert_buffers(&ctx.recv_buffers);
    let recv_statics = convert_buffers(&ctx.recv_statics);
    let send_buffers = convert_buffers(&ctx.send_buffers);
    let send_statics = convert_buffers(&ctx.send_statics);
    let mem = AddressSpaceMemory { addr_space: &*kernel.address_space };
    let mut cmif_ctx = make_cmif_ctx(ctx, &mem, &recv_buffers, &recv_statics, &send_buffers, &send_statics);
    kernel.services.pl.dispatch_cmif(ctx.cmif_in.cmd_id, &mut cmif_ctx)
}
