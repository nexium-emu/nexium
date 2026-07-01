use super::{Kernel, MUTEX_HAS_LISTENERS};
use crate::kernel::audio_lut::NX_SRC_LUT_UP;
use crate::kernel::cpu_local::{cpu_mut, cpu_ref};
use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use crate::kernel::AudioRendererState;
use nexium_common::result::{
    KERNEL_INVALID_ADDRESS, KERNEL_NOT_IMPLEMENTED, KERNEL_TIMEOUT, SUCCESS,
};
use nexium_ipc as ipc;

fn svc_trace_capture() -> Option<(u64, u64, u64, u64, u64, u64, u64)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_SVC_TRACE")
            .ok()
            .map(|v| v.parse::<u64>().unwrap_or(1000))
            .unwrap_or(0)
    });
    if limit == 0 {
        return None;
    }
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n >= limit {
        return None;
    }
    let cpu = cpu_ref()?;
    Some((
        n,
        cpu.get_register(0),
        cpu.get_register(1),
        cpu.get_register(2),
        cpu.get_register(3),
        cpu.get_pc(),
        cpu.get_register(30),
    ))
}

pub fn dispatch(kernel: &mut Kernel, imm: u16) -> u32 {
    log::trace!("SVC {:#04x}", imm);
    struct ProfileGuard(std::time::Instant, u16);
    impl Drop for ProfileGuard {
        fn drop(&mut self) {
            crate::kernel::profile::record_svc(self.1, self.0);
        }
    }
    let _guard = if crate::kernel::profile::enabled() {
        Some(ProfileGuard(std::time::Instant::now(), imm))
    } else {
        None
    };
    let __svc_trace_args = svc_trace_capture();
    let __svc_res = match imm {
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
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, KERNEL_NOT_IMPLEMENTED as u64);
            }
            KERNEL_NOT_IMPLEMENTED
        }
    };
    if let Some((n, x0, x1, x2, x3, pc, lr)) = __svc_trace_args {
        log::info!(
            "[svc-trace #{}] svc={:#04x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} -> {:#x} pc={:#x} lr={:#x}",
            n,
            imm,
            x0,
            x1,
            x2,
            x3,
            __svc_res,
            pc,
            lr
        );
    }
    __svc_res
}

fn svc_set_heap_size(kernel: &mut Kernel) -> u32 {
    let size = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1)
    } else {
        return 1;
    };
    if size > kernel.heap_size {
        log::warn!(
            "svcSetHeapSize requested {:#x} > mapped heap {:#x}; clamping (guest may fault on overflow)",
            size,
            kernel.heap_size
        );
    }
    kernel.heap_committed = size.min(kernel.heap_size);
    log::debug!(
        "svcSetHeapSize size={:#x} -> heap_base={:#x} (heap mapped {:#x})",
        size,
        kernel.heap_base,
        kernel.heap_size
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, kernel.heap_base);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_permission(_kernel: &mut Kernel) -> u32 {
    let (addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
        )
    } else {
        (0, 0, 0)
    };
    log::info!(
        "svcSetMemoryPermission addr={:#x} size={:#x} perm={:#x} (no-op)",
        addr,
        size,
        perm
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_memory_attribute(_kernel: &mut Kernel) -> u32 {
    let (addr, size, mask, value) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        (0, 0, 0, 0)
    };
    log::info!(
        "svcSetMemoryAttribute addr={:#x} size={:#x} mask={:#x} value={:#x} (no-op)",
        addr,
        size,
        mask,
        value
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_map_memory(kernel: &mut Kernel) -> u32 {
    let (dst, src, size) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
        )
    } else {
        return 1;
    };

    if size == 0 || (dst & 0xFFF) != 0 || (size & 0xFFF) != 0 {
        log::warn!(
            "svcMapMemory: bad args dst={:#x} src={:#x} size={:#x}",
            dst,
            src,
            size
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_ADDRESS as u64);
        }
        return KERNEL_INVALID_ADDRESS;
    }

    let map_rc = kernel
        .address_space
        .map(dst, size, nexium_memory::Perm::RW, "stack_mirror");
    let was_new = map_rc.is_ok();

    let mut buf = vec![0u8; size as usize];
    if kernel.address_space.read(src, &mut buf).is_ok() {
        let _ = kernel.address_space.write(dst, &buf);
    }

    if was_new {
        if let Some(region) = kernel.address_space.host_region_at(dst) {
            if let Some(cpu) = cpu_mut() {
                let plumb = unsafe {
                    cpu.map_host(
                        region.base,
                        region.size,
                        region.perm,
                        region.host_ptr as *mut u8,
                    )
                };
                match plumb {
                    Ok(_) => log::info!(
                        "svcMapMemory dst={:#x} src={:#x} size={:#x} → mapped + copied + plumbed to dynarmic",
                        dst,
                        src,
                        size
                    ),
                    Err(e) => log::warn!(
                        "svcMapMemory dst={:#x} size={:#x} mapped in AS but dynarmic map_host failed: {}",
                        dst,
                        size,
                        e
                    ),
                }
            }
        } else {
            log::warn!(
                "svcMapMemory dst={:#x}: AS region lookup failed after map()",
                dst
            );
        }
    } else if let Err(e) = map_rc {
        log::debug!(
            "svcMapMemory dst={:#x} src={:#x} size={:#x} → already mapped ({:?}), refreshed contents only",
            dst,
            src,
            size,
            e
        );
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcUnmapMemory (no-op)");
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_memory(kernel: &mut Kernel) -> u32 {
    let (out_ptr, address) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(2))
    } else {
        return 1;
    };

    log::debug!(
        "svcQueryMemory out_ptr={:#x} address={:#x}",
        out_ptr,
        address
    );

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

    if let Some(cpu) = cpu_mut() {
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
            let mem_type = if r.name.starts_with("codestatic") {
                0x03
            } else if r.name.starts_with("codemutable") {
                0x04
            } else if r.name.contains("text") || r.name.contains("rodata") {
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

    let next_base = regions
        .iter()
        .map(|r| r.base)
        .filter(|&b| b > address)
        .min()
        .unwrap_or(u64::MAX);
    let page_addr = address & !0xFFF;
    let gap_size = next_base.saturating_sub(page_addr);

    SynthMemInfo {
        addr: page_addr,
        size: if gap_size == 0 {
            0x10000_0000
        } else {
            gap_size
        },
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
    let (handle, addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0) as u32,
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3) as u32,
        )
    } else {
        return 1;
    };
    log::info!(
        "svcMapSharedMemory handle={:#x} addr={:#x} size={:#x} perm={:#x}",
        handle,
        addr,
        size,
        perm
    );
    if handle == 0 {
        log::error!(
            "svcMapSharedMemory called with handle=0 (addr={:#x} size={:#x}). \
             Upstream service returned no shared-memory handle. \
             Allocating zero-filled placeholder; expect downstream code to read zeros from this region.",
            addr,
            size
        );
    }

    if size as usize == crate::hid_state::HID_SHMEM_SIZE {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        hid.shmem_va = Some(addr);
        let ptr = hid.host_ptr();
        log::info!(
            "  → recognized as HID shared memory, mapping host buffer directly to guest VA {:#x}",
            addr
        );
        if let Some(cpu) = cpu_mut() {
            unsafe {
                if let Err(e) = cpu.map_host(addr, size, nexium_memory::perm::Perm::RW, ptr) {
                    log::warn!("failed to map HID shmem in CPU: {}", e);
                }
            }
        }
    } else if kernel.time_shmem_handle == Some(handle) {
        log::info!(
            "  → recognized as time shared memory, mapping {} bytes at {:#x}",
            size,
            addr
        );
        let backing: Vec<u8> = kernel
            .time_shmem
            .as_deref()
            .map(|d| {
                let mut v = vec![0u8; size as usize];
                let copy_len = d.len().min(size as usize);
                v[..copy_len].copy_from_slice(&d[..copy_len]);
                v
            })
            .unwrap_or_else(|| vec![0u8; size as usize]);
        let needed_map = kernel.address_space.write(addr, &backing).is_err();
        if needed_map {
            let _ =
                kernel
                    .address_space
                    .map(addr, size, nexium_memory::perm::Perm::R, "time_shmem");
            let _ = kernel.address_space.write(addr, &backing);
        }
        if let Some(region) = kernel.address_space.host_region_at(addr) {
            if let Some(cpu) = cpu_mut() {
                unsafe {
                    if let Err(e) =
                        cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    {
                        log::warn!("failed to map time shmem in CPU: {}", e);
                    } else {
                        log::info!("  → registered time shmem at {:#x} with CPU", region.base);
                    }
                }
            }
        }
    } else if kernel.font_shmem_handle == Some(handle) {
        log::info!(
            "  → recognized as font shared memory, mapping {} bytes of font data at {:#x}",
            size,
            addr
        );
        let font_data: Vec<u8> = kernel
            .font_shmem
            .as_deref()
            .map(|d| {
                let mut v = vec![0u8; size as usize];
                let copy_len = d.len().min(size as usize);
                v[..copy_len].copy_from_slice(&d[..copy_len]);
                v
            })
            .unwrap_or_else(|| vec![0u8; size as usize]);
        let needed_map = kernel.address_space.write(addr, &font_data).is_err();
        if needed_map {
            let _ =
                kernel
                    .address_space
                    .map(addr, size, nexium_memory::perm::Perm::R, "font_shmem");
            let _ = kernel.address_space.write(addr, &font_data);
        }
        if let Some(region) = kernel.address_space.host_region_at(addr) {
            if let Some(cpu) = cpu_mut() {
                unsafe {
                    if let Err(e) =
                        cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    {
                        log::warn!("failed to map font shmem in CPU: {}", e);
                    } else {
                        log::info!("  → registered font shmem at {:#x} with CPU", region.base);
                    }
                }
            }
        }
    } else {
        let backing = vec![0u8; size as usize];
        let needed_map = kernel.address_space.write(addr, &backing).is_err();
        if needed_map {
            let _ = kernel
                .address_space
                .map(addr, size, nexium_memory::perm::Perm::RW, "shared");
            let _ = kernel.address_space.write(addr, &backing);
            if let Some(region) = kernel.address_space.host_region_at(addr) {
                if let Some(cpu) = cpu_mut() {
                    unsafe {
                        if let Err(e) =
                            cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                        {
                            log::warn!("failed to map shared mem in CPU: {}", e);
                        } else {
                            log::info!("  → registered shared mem at {:#x} with CPU", region.base);
                        }
                    }
                }
            }
        }
    }

    if let Some(cpu) = cpu_mut() {
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

    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    };

    log::debug!("  signaling event handle {:#x}", handle);

    if let Some(_event) = kernel.handles.get_handle(handle) {
        kernel.event_signals.insert(handle, true);
        kernel.threads.signal_handle(handle);
        log::debug!("  event {:#x} signaled (waiters woken)", handle);
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
        }
        return SUCCESS;
    } else {
        log::warn!("  invalid event handle {:#x}", handle);
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }
}

fn completed_thread_wait_index(kernel: &Kernel, handles: &[u32]) -> Option<usize> {
    handles.iter().position(|h| {
        if !matches!(
            kernel
                .handles
                .get_handle(*h)
                .map(|handle| handle.handle_type),
            Some(HandleType::Thread)
        ) {
            return false;
        }
        if kernel.exited_thread_handles.contains(h) {
            return true;
        }
        kernel
            .threads
            .threads
            .get(h)
            .map(|thread| matches!(thread.state, crate::kernel::threads::ThreadState::Exited))
            .unwrap_or(false)
    })
}

fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 {
    let (handles_addr, count, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1),
            (cpu.get_register(2) as u32).min(0x40),
            cpu.get_register(3),
        )
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

    for h in &handles {
        crate::kernel::profile::record_wait_handle(*h);
    }

    if let Some(i) = completed_thread_wait_index(kernel, &handles) {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
            cpu.set_register(1, i as u64);
        }
        return SUCCESS;
    }

    for (i, h) in handles.iter().enumerate() {
        if kernel.nvdrv_sync_events.contains(h) {
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
                cpu.set_register(1, i as u64);
            }
            return SUCCESS;
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
        log::trace!(
            "svcWaitSync(timeout=0) handles={:?} applet_msg_event={:?} applet_msgs_pending={} vsync_idx={:?}",
            handles,
            kernel.applet_message_event,
            kernel.applet_messages.len(),
            vsync_idx
        );
        for (i, h) in handles.iter().enumerate() {
            if let Some(msg_evt) = kernel.applet_message_event {
                if *h == msg_evt && !kernel.applet_messages.is_empty() {
                    if let Some(cpu) = cpu_mut() {
                        cpu.set_register(0, SUCCESS as u64);
                        cpu.set_register(1, i as u64);
                    }
                    return SUCCESS;
                }
            }
            if let Some(slot) = kernel.event_signals.get_mut(h) {
                if *slot {
                    *slot = false;
                    if let Some(cpu) = cpu_mut() {
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
        {
            const AUDIO_PERIOD: std::time::Duration = std::time::Duration::from_millis(20);
            use parking_lot::Mutex;
            use std::sync::OnceLock;
            static LAST_TICK: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
            let cell = LAST_TICK.get_or_init(|| Mutex::new(std::time::Instant::now()));
            let mut last = cell.lock();
            if last.elapsed() >= AUDIO_PERIOD {
                *last = std::time::Instant::now();
                let sessions_with_pending: Vec<u32> = kernel
                    .audio_out_buffers
                    .iter()
                    .filter_map(|(s, q)| if !q.is_empty() { Some(*s) } else { None })
                    .collect();
                for sess in sessions_with_pending {
                    if let Some(&ev) = kernel.audio_buffer_events.get(&sess) {
                        kernel.event_signals.insert(ev, true);
                    }
                }
            }
        }

        const TIMEOUT_ERROR: u32 = 1 | (117 << 9);
        if !kernel.threads.ready.is_empty() {
            kernel.yield_after_svc = true;
        }
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, TIMEOUT_ERROR as u64);
            cpu.set_register(1, 0);
        }
        return TIMEOUT_ERROR;
    }

    if let Some(i) = vsync_idx {
        const VSYNC_PERIOD: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let now = std::time::Instant::now();
        let elapsed = now.saturating_duration_since(kernel.last_vsync);
        let remaining = if elapsed >= VSYNC_PERIOD {
            std::time::Duration::ZERO
        } else {
            VSYNC_PERIOD - elapsed
        };
        let allowed = if timeout_ns == u64::MAX || timeout_ns == 0 {
            remaining
        } else {
            std::time::Duration::from_nanos(timeout_ns).min(remaining)
        };
        kernel.last_vsync = std::time::Instant::now() + allowed;
        {
            let state = crate::hid_state::get_hid_state();
            let mut hid = state.lock();
            if hid.shmem_va.is_some() {
                let cur = hid.input.clone();
                hid.tick(cur);
            }
        }
        let sessions_with_pending: Vec<u32> = kernel
            .audio_out_buffers
            .iter()
            .filter_map(|(s, q)| if !q.is_empty() { Some(*s) } else { None })
            .collect();
        for sess in sessions_with_pending {
            if let Some(&ev) = kernel.audio_buffer_events.get(&sess) {
                kernel.event_signals.insert(ev, true);
            }
        }
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
            cpu.set_register(1, i as u64);
        }
        if allowed > std::time::Duration::ZERO {
            if let Some(cpu) = cpu_ref() {
                let wake_at = std::time::Instant::now() + allowed;
                kernel.threads.yield_with_state(
                    cpu,
                    crate::kernel::threads::ThreadState::Sleeping { wake_at },
                );
            }
        }
        return SUCCESS;
    }

    for (i, h) in handles.iter().enumerate() {
        if let Some(msg_evt) = kernel.applet_message_event {
            if *h == msg_evt && !kernel.applet_messages.is_empty() {
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, SUCCESS as u64);
                    cpu.set_register(1, i as u64);
                }
                return SUCCESS;
            }
        }
        if let Some(slot) = kernel.event_signals.get_mut(h) {
            if *slot {
                *slot = false;
                if let Some(cpu) = cpu_mut() {
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
    if let Some(cpu) = cpu_ref() {
        let wake_at = if timeout_ns == u64::MAX {
            None
        } else {
            const VSYNC_PERIOD_NS: u64 = 16_666_667;
            let cap = std::time::Duration::from_nanos(VSYNC_PERIOD_NS);
            let wait = std::time::Duration::from_nanos(timeout_ns).min(cap);
            Some(std::time::Instant::now() + wait)
        };
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingHandle {
                handles: handles.clone(),
                wake_at,
            },
        );
    }
    if timeout_ns != u64::MAX {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, TIMEOUT_ERROR as u64);
        }
        return TIMEOUT_ERROR;
    }
    SUCCESS
}

fn svc_cancel_synchronization(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcCancelSynchronization");
    SUCCESS
}

fn svc_arbitrate_lock(kernel: &mut Kernel) -> u32 {
    let (_holder, mutex_addr, self_handle) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0) as u32,
            cpu.get_register(1),
            cpu.get_register(2) as u32,
        )
    } else {
        return 1;
    };

    let mut cur = [0u8; 4];
    let cur_word = if kernel.address_space.read(mutex_addr, &mut cur).is_ok() {
        u32::from_le_bytes(cur)
    } else {
        0
    };
    let holder = cur_word & !MUTEX_HAS_LISTENERS;
    let lr = cpu_ref().map(|c| c.get_register(30)).unwrap_or(0);

    if holder == 0 || holder == self_handle {
        let new_word = self_handle | (cur_word & MUTEX_HAS_LISTENERS);
        let _ = kernel
            .address_space
            .write(mutex_addr, &new_word.to_le_bytes());
        log::info!(
            "svcArbitrateLock mutex={:#x} self_handle={:#x} cur={:#x} → uncontended lr={:#x}",
            mutex_addr,
            self_handle,
            cur_word,
            lr
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, SUCCESS as u64);
        }
        return SUCCESS;
    }

    let new_word = cur_word | MUTEX_HAS_LISTENERS;
    let _ = kernel
        .address_space
        .write(mutex_addr, &new_word.to_le_bytes());

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    if let Some(cpu) = cpu_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingMutex { mutex_addr },
        );
    }
    log::debug!(
        "svcArbitrateLock mutex={:#x} contended (holder={:#x} self={:#x}) → parked",
        mutex_addr,
        holder,
        self_handle
    );
    SUCCESS
}

fn svc_arbitrate_unlock(kernel: &mut Kernel) -> u32 {
    let mutex_addr = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        return 1;
    };
    let woken = kernel.threads.wake_one_on_mutex(mutex_addr);
    let new_word = match woken {
        Some(h) => {
            let more = kernel.threads.has_mutex_waiters(mutex_addr);
            if more {
                h | MUTEX_HAS_LISTENERS
            } else {
                h
            }
        }
        None => 0,
    };
    let _ = kernel
        .address_space
        .write(mutex_addr, &new_word.to_le_bytes());
    if let Some(h) = woken {
        log::debug!(
            "svcArbitrateUnlock mutex={:#x} handed to handle={:#x} (word={:#x})",
            mutex_addr,
            h,
            new_word
        );
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_wait_process_wide_key_atomic(kernel: &mut Kernel) -> u32 {
    let (mutex_addr, condvar_addr, self_handle, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2) as u32,
            cpu.get_register(3),
        )
    } else {
        return 1;
    };
    let lr = cpu_ref().map(|c| c.get_register(30)).unwrap_or(0);
    log::trace!(
        "svcWaitProcessWideKeyAtomic mutex={:#x} condvar={:#x} self_handle={:#x} timeout_ns={} lr={:#x}",
        mutex_addr,
        condvar_addr,
        self_handle,
        timeout_ns,
        lr
    );

    let woken = kernel.threads.wake_one_on_mutex(mutex_addr);
    let new_word = match woken {
        Some(h) => {
            let more = kernel.threads.has_mutex_waiters(mutex_addr);
            if more {
                h | MUTEX_HAS_LISTENERS
            } else {
                h
            }
        }
        None => 0,
    };
    let _ = kernel
        .address_space
        .write(mutex_addr, &new_word.to_le_bytes());
    if let Some(h) = woken {
        log::debug!(
            "cond_wait release: mutex={:#x} handed to handle={:#x} (word={:#x})",
            mutex_addr,
            h,
            new_word
        );
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }

    let had_pending = if let Some(n) = kernel.pending_condvar_signals.get_mut(&condvar_addr) {
        *n = n.saturating_sub(1);
        let remove = *n == 0;
        if remove {
            kernel.pending_condvar_signals.remove(&condvar_addr);
        }
        true
    } else {
        false
    };
    if had_pending {
        if kernel.reacquire_condvar_mutex(self_handle, mutex_addr) {
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
        } else if let Some(cpu) = cpu_ref() {
            kernel.threads.yield_with_state(
                cpu,
                crate::kernel::threads::ThreadState::WaitingMutex { mutex_addr },
            );
        }
        return SUCCESS;
    }

    if timeout_ns == 0 {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_TIMEOUT as u64);
        }
        return KERNEL_TIMEOUT;
    }

    let wake_at = if timeout_ns == u64::MAX {
        None
    } else {
        Some(std::time::Instant::now() + std::time::Duration::from_nanos(timeout_ns))
    };

    let _ = kernel
        .address_space
        .write(condvar_addr, &1u32.to_le_bytes());

    if let Some(cpu) = cpu_ref() {
        kernel.threads.yield_with_state(
            cpu,
            crate::kernel::threads::ThreadState::WaitingCondvar {
                mutex_addr,
                condvar_addr,
                wake_at,
                spurious_wake: false,
            },
        );
    }

    SUCCESS
}

fn svc_signal_process_wide_key(kernel: &mut Kernel) -> u32 {
    let (condvar_addr, count) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0), cpu.get_register(1) as i32)
    } else {
        return 1;
    };

    let max = if count < 0 { i32::MAX } else { count };
    let mut woken = 0;
    for _ in 0..max {
        let Some((handle, mutex_addr)) = kernel.threads.peek_one_condvar_waiter(condvar_addr)
        else {
            break;
        };

        let mut cur = [0u8; 4];
        let cur_word = if kernel.address_space.read(mutex_addr, &mut cur).is_ok() {
            u32::from_le_bytes(cur)
        } else {
            0
        };
        let holder = cur_word & !MUTEX_HAS_LISTENERS;

        if holder == 0 {
            let _ = kernel
                .address_space
                .write(mutex_addr, &handle.to_le_bytes());
            kernel.threads.wake_condvar_to_ready(handle);
            log::trace!(
                "svcSignalProcessWideKey cond={:#x} → handle={:#x} mutex={:#x} (mutex was free, handed off)",
                condvar_addr,
                handle,
                mutex_addr
            );
        } else {
            let new_word = cur_word | MUTEX_HAS_LISTENERS;
            let _ = kernel
                .address_space
                .write(mutex_addr, &new_word.to_le_bytes());
            kernel
                .threads
                .wake_condvar_into_mutex_waiter(handle, mutex_addr);
            log::trace!(
                "svcSignalProcessWideKey cond={:#x} → handle={:#x} mutex={:#x} (held by {:#x}, requeued as WaitingMutex)",
                condvar_addr,
                handle,
                mutex_addr,
                holder
            );
        }
        woken += 1;
    }
    if woken == 0 && count != 0 {
        let pending = if count < 0 { 32 } else { count as u32 };
        let entry = kernel
            .pending_condvar_signals
            .entry(condvar_addr)
            .or_insert(0);
        *entry = entry.saturating_add(pending);
    }
    if !kernel.threads.has_condvar_waiters(condvar_addr) {
        let _ = kernel
            .address_space
            .write(condvar_addr, &0u32.to_le_bytes());
    }
    log::trace!(
        "svcSignalProcessWideKey cond={:#x} count={} woken={}",
        condvar_addr,
        count,
        woken
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_system_tick(_kernel: &mut Kernel) -> u32 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let elapsed = EPOCH.get_or_init(std::time::Instant::now).elapsed();
    let ticks = (elapsed.as_nanos() as u64).wrapping_mul(19_200_000) / 1_000_000_000;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, ticks);
    }
    SUCCESS
}

fn domain_group(kernel: &Kernel, session_handle: u32) -> u32 {
    kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.domain_group)
        .unwrap_or(session_handle)
}

fn domain_group_handles(kernel: &Kernel, session_handle: u32) -> Vec<u32> {
    let group = domain_group(kernel, session_handle);
    kernel
        .sessions
        .iter()
        .filter_map(|(&handle, session)| {
            if session.is_domain && session.domain_group == group {
                Some(handle)
            } else {
                None
            }
        })
        .collect()
}

fn next_domain_object_id(kernel: &Kernel, session_handle: u32) -> u32 {
    let group = domain_group(kernel, session_handle);
    kernel
        .sessions
        .values()
        .filter(|session| session.is_domain && session.domain_group == group)
        .map(|session| session.next_domain_object_id)
        .max()
        .unwrap_or(0)
}

fn alloc_domain_object(kernel: &mut Kernel, session_handle: u32, service_name: &str) -> u32 {
    let object_id = next_domain_object_id(kernel, session_handle);
    let group = domain_group(kernel, session_handle);
    for session in kernel.sessions.values_mut() {
        if session.is_domain && session.domain_group == group {
            session
                .domain_objects
                .insert(object_id, service_name.to_string());
            session.next_domain_object_id = object_id.saturating_add(1);
        }
    }
    object_id
}

fn close_domain_object(kernel: &mut Kernel, session_handle: u32, object_id: u32) -> Vec<u32> {
    let handles = domain_group_handles(kernel, session_handle);
    for handle in &handles {
        if let Some(session) = kernel.sessions.get_mut(handle) {
            session.close_object(object_id);
        }
    }
    handles
}

fn service_for_domain_object(
    kernel: &Kernel,
    session_handle: u32,
    object_id: u32,
) -> Option<String> {
    if let Some(name) = kernel
        .sessions
        .get(&session_handle)
        .and_then(|session| session.service_for_object(object_id))
    {
        return Some(name.to_string());
    }

    let group = domain_group(kernel, session_handle);
    kernel.sessions.values().find_map(|session| {
        if session.is_domain && session.domain_group == group {
            session.service_for_object(object_id).map(str::to_string)
        } else {
            None
        }
    })
}

fn domain_object_keys(kernel: &Kernel, session_handle: u32, object_id: u32) -> Vec<(u32, u32)> {
    let mut keys = vec![(session_handle, object_id)];
    for handle in domain_group_handles(kernel, session_handle) {
        if handle != session_handle {
            keys.push((handle, object_id));
        }
    }
    keys
}

fn svc_send_sync_request(kernel: &mut Kernel) -> u32 {
    let (tls_addr, session_handle) = if let Some(cpu) = cpu_ref() {
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
            log::warn!(
                "SendSyncRequest: invalid session handle {:#x}",
                session_handle
            );
            return 1;
        }
    };

    let hipc_header_raw = u64::from_le_bytes([
        tls_buf[0], tls_buf[1], tls_buf[2], tls_buf[3], tls_buf[4], tls_buf[5], tls_buf[6],
        tls_buf[7],
    ]);
    let cmd_type = (hipc_header_raw & 0xFFFF) as u16;

    match cmd_type {
        2 => {
            log::info!(
                "session Close session={:#x} service={}",
                session_handle,
                port_name
            );
            kernel.sessions.remove(&session_handle);
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        5 | 7 => {
            log::debug!(
                "Control cmd_type={} session={:#x} service={}",
                cmd_type,
                session_handle,
                port_name
            );
            let response = handle_control_request(kernel, session_handle, &port_name, &tls_buf);
            if !response.is_empty() {
                let mut response_buf = tls_buf.clone();
                let copy_len = response.len().min(response_buf.len());
                response_buf[..copy_len].copy_from_slice(&response[..copy_len]);
                let _ = kernel.address_space.write(tls_addr, &response_buf);
            }
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        _ => {}
    }

    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    let ipc_parse_result = ipc::IpcCtx::parse(tls_buf.clone(), is_domain);
    let mut ctx = match ipc_parse_result {
        Ok(c) => c,
        Err(e) => {
            log::warn!(
                "Failed to parse IPC message: {:?} (is_domain={})",
                e,
                is_domain
            );
            return 1;
        }
    };

    let cmd_id = ctx.cmif_in.cmd_id;

    let dispatch_target = if let Some(d) = ctx.domain {
        if d.kind == 2 {
            let domain_handles = close_domain_object(kernel, session_handle, d.object_id);
            for handle in domain_handles {
                kernel.open_files.remove(&(handle, d.object_id));
                kernel.file_system_roots.remove(&(handle, d.object_id));
                kernel.open_host_files.remove(&(handle, d.object_id));
                kernel.open_romfs_files.remove(&(handle, d.object_id));
                kernel.open_file_handles.remove(&(handle, d.object_id));
                kernel.open_dir_lists.remove(&(handle, d.object_id));
            }
            log::debug!(
                "domain Close-object session={:#x} object_id={}",
                session_handle,
                d.object_id
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, SUCCESS as u64);
            }
            return SUCCESS;
        }
        match service_for_domain_object(kernel, session_handle, d.object_id) {
            Some(name) => name,
            None => {
                log::warn!(
                    "domain object_id={} not found on session={:#x} (port={}) → InvalidObject 0xCE01",
                    d.object_id,
                    session_handle,
                    port_name
                );
                if let Some(cpu) = cpu_mut() {
                    cpu.set_register(0, SUCCESS as u64);
                }
                return 0xCE01;
            }
        }
    } else {
        port_name.clone()
    };

    log::trace!(
        "IPC request service=\"{}\" cmd={} in_data={} is_domain={}",
        dispatch_target,
        cmd_id,
        ctx.cmif_in_data_len,
        is_domain
    );

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
            log::error!(
                "**** fatal:u ThrowFatal result={:#010x} module={} description={} ****",
                result,
                module,
                desc
            );
        }
    }

    let info_enabled = log::max_level() >= log::LevelFilter::Info;
    let in_data_preview: Vec<u8> = if info_enabled {
        let start = ctx.cmif_in_data_off;
        let end = (start + 32).min(ctx.buf.len());
        if start < ctx.buf.len() {
            ctx.buf[start..end].to_vec()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let response = if dispatch_target == "sm:" {
        dispatch_sm_command_v2(kernel, &mut ctx)
    } else {
        let mut pending_frames = std::mem::take(&mut kernel.pending_frames);
        let response = dispatch_service_v2(
            kernel,
            &dispatch_target,
            &mut ctx,
            session_handle,
            &mut pending_frames,
        );
        kernel.pending_frames = pending_frames;
        response
    };

    if info_enabled {
        use parking_lot::Mutex;
        use std::collections::HashSet;
        use std::sync::OnceLock;
        static SEEN: OnceLock<Mutex<HashSet<(String, u32)>>> = OnceLock::new();
        let seen_cell = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let key = (dispatch_target.clone(), cmd_id);
        let is_first = seen_cell.lock().insert(key);
        if is_first {
            let resp_preview: Vec<String> = response
                .iter()
                .take(64)
                .map(|b| format!("{:02x}", b))
                .collect();
            let in_preview: Vec<String> = in_data_preview
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect();
            let reply_rc = if response.len() >= 12 {
                u32::from_le_bytes(response[8..12].try_into().unwrap_or([0; 4]))
            } else {
                0
            };
            log::info!(
                "IPC FIRST-OCCURRENCE response (compare with RustSwitch) service={} cmd={} rc={:#010x} response_len={} response_first64={} in_data_first32={}",
                dispatch_target,
                cmd_id,
                reply_rc,
                response.len(),
                resp_preview.join(","),
                in_preview.join(",")
            );
        }
    }

    let mut response_buf = vec![0u8; 0x100];
    let copy_len = response.len().min(response_buf.len());
    response_buf[..copy_len].copy_from_slice(&response[..copy_len]);

    if kernel.address_space.write(tls_addr, &response_buf).is_err() {
        log::warn!(
            "SendSyncRequest: failed to write TLS response at {:#x}",
            tls_addr
        );
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, 1u64);
        }
        return 1;
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn handle_control_request(
    kernel: &mut Kernel,
    session_handle: u32,
    port_name: &str,
    tls_buf: &[u8],
) -> Vec<u8> {
    let parse_result = ipc::IpcCtx::parse(tls_buf.to_vec(), false);
    let mut ctx = match parse_result {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    match ctx.cmif_in.cmd_id {
        0 => {
            log::debug!(
                "Control: ConvertCurrentObjectToDomain service={}",
                port_name
            );
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
            let mut session = Session::new(dup_handle, port_name.to_string());
            if let Some(orig) = kernel.sessions.get(&session_handle) {
                session.is_domain = orig.is_domain;
                session.domain_group = orig.domain_group;
                session.domain_objects = orig.domain_objects.clone();
                session.next_domain_object_id = orig.next_domain_object_id;
            }
            kernel.sessions.insert(dup_handle, session);
            log::debug!(
                "Control: CloneCurrentObject service={} dup={:#x}",
                port_name,
                dup_handle
            );
            build_ipc_response(&mut ctx, 0, &[], &[dup_handle])
        }
        3 => {
            log::debug!(
                "Control: QueryPointerBufferSize → 0x500 service={}",
                port_name
            );
            build_ipc_response(&mut ctx, 0, &0x500u16.to_le_bytes(), &[])
        }
        other => {
            log::debug!("Control: unknown cmd={} service={}", other, port_name);
            build_ipc_response(&mut ctx, 0, &[], &[])
        }
    }
}

pub(crate) fn build_ipc_response(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    move_handles: &[u32],
) -> Vec<u8> {
    build_ipc_response_full(ctx, result, out_data, move_handles, &[], &[])
}

pub(crate) fn build_ipc_response_copy(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    copy_handles: &[u32],
) -> Vec<u8> {
    build_ipc_response_full(ctx, result, out_data, &[], copy_handles, &[])
}

fn build_ipc_response_full(
    ctx: &ipc::IpcCtx,
    result: u32,
    out_data: &[u8],
    move_handles: &[u32],
    copy_handles: &[u32],
    out_objects: &[u32],
) -> Vec<u8> {
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
    let has_special_header = !move_handles.is_empty() || !copy_handles.is_empty();
    if has_special_header {
        let mut sh: u32 = 0;
        sh |= (copy_handles.len() as u32 & 0xF) << 1;
        sh |= (move_handles.len() as u32 & 0xF) << 5;
        special_bytes.extend_from_slice(&sh.to_le_bytes());
        for h in copy_handles {
            special_bytes.extend_from_slice(&h.to_le_bytes());
        }
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

fn dump_throw_context(kernel: &Kernel) {
    let cpu = match cpu_ref() {
        Some(c) => c,
        None => return,
    };
    let base = kernel.code_base;
    log::warn!(
        "[throw] ===== uncaught-exception context (code_base={:#x}) =====",
        base
    );
    for row in 0..4 {
        let r = row * 8;
        log::warn!(
            "[throw] x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x} x{:<2}={:#018x}",
            r,
            cpu.get_register(r),
            r + 1,
            cpu.get_register(r + 1),
            r + 2,
            cpu.get_register(r + 2),
            r + 3,
            cpu.get_register(r + 3),
            r + 4,
            cpu.get_register(r + 4),
            r + 5,
            cpu.get_register(r + 5),
            r + 6,
            cpu.get_register(r + 6),
            r + 7,
            cpu.get_register(r + 7)
        );
    }
    log::warn!(
        "[throw] x28={:#018x} x29(fp)={:#018x} x30(lr)={:#018x} sp={:#018x}",
        cpu.get_register(28),
        cpu.get_register(29),
        cpu.get_register(30),
        cpu.get_sp()
    );
    let read_u64 = |va: u64| -> Option<u64> {
        let mut b = [0u8; 8];
        if kernel.address_space.read(va, &mut b).is_ok() {
            Some(u64::from_le_bytes(b))
        } else {
            None
        }
    };
    let read_u32 = |va: u64| -> Option<u32> {
        let mut b = [0u8; 4];
        if kernel.address_space.read(va, &mut b).is_ok() {
            Some(u32::from_le_bytes(b))
        } else {
            None
        }
    };
    let mut fp = cpu.get_register(29);
    for depth in 0..28u32 {
        if fp == 0 || (fp & 7) != 0 {
            break;
        }
        let next_fp = match read_u64(fp) {
            Some(v) => v,
            None => break,
        };
        let ret = match read_u64(fp.wrapping_add(8)) {
            Some(v) => v,
            None => break,
        };
        let off = ret.wrapping_sub(base);
        let mut words = String::new();
        for i in 0..6u64 {
            if let Some(w) = read_u32(ret.wrapping_sub(20).wrapping_add(i * 4)) {
                words.push_str(&format!("{:08x} ", w));
            }
        }
        log::warn!(
            "[throw] #{:02} ret=+{:#x} (raw={:#x}) fp={:#x} | callsite[ret-20..ret+4]= {}",
            depth,
            off,
            ret,
            fp,
            words
        );
        if next_fp <= fp {
            break;
        }
        fp = next_fp;
    }
    log::warn!("[throw] ===== end context =====");
}

fn dispatch_service_v2(
    kernel: &mut Kernel,
    port_name: &str,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    pending_frames: &mut Vec<crate::services::FrameOut>,
) -> Vec<u8> {
    struct IpcProfileGuard(std::time::Instant, String);
    impl Drop for IpcProfileGuard {
        fn drop(&mut self) {
            crate::kernel::profile::record_ipc(&self.1, self.0);
        }
    }
    let _guard = if crate::kernel::profile::enabled() {
        Some(IpcProfileGuard(
            std::time::Instant::now(),
            port_name.to_string(),
        ))
    } else {
        None
    };
    let cmd_id = ctx.cmif_in.cmd_id;

    if port_name == "ILogService" && cmd_id == 0 {
        let sb = ctx
            .send_buffers
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .copied()
            .or_else(|| {
                ctx.send_statics
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .copied()
            });
        if let Some(b) = sb {
            let mut buf = vec![0u8; (b.size as usize).min(0x400)];
            if kernel.address_space.read(b.addr, &mut buf).is_ok() {
                let txt: String = buf
                    .iter()
                    .map(|&c| {
                        if (0x20..0x7f).contains(&c) {
                            c as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                log::warn!("[lm.Log] {}", txt);
                if txt.contains("bad_alloc") || txt.contains("uncaught") || txt.contains("abort") {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static DUMPED: AtomicBool = AtomicBool::new(false);
                    if !DUMPED.swap(true, Ordering::Relaxed) {
                        dump_throw_context(kernel);
                    }
                }
            }
        }
    }

    if port_name == "nvdrv"
        || port_name == "nvdrv:a"
        || port_name == "nvdrv:s"
        || port_name == "nvdrv:t"
    {
        return dispatch_nvdrv_command(kernel, ctx, port_name);
    }

    if port_name == "IHOSBinderDriver" && (cmd_id == 0 || cmd_id == 3) {
        return handle_binder_transact(kernel, ctx, session_handle);
    }

    if let Some(buffer_data) = applet_buffer_response(port_name, cmd_id) {
        let target_buf = ctx
            .recv_buffers
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = target_buf {
            let write_len = buffer_data.len().min(buf.size as usize);
            let _ = kernel
                .address_space
                .write(buf.addr, &buffer_data[..write_len]);
            log::info!(
                "  wrote {} bytes to recv buf at {:#x} (avail {})",
                write_len,
                buf.addr,
                buf.size
            );
        } else {
            log::debug!(
                "  no recv buffer/static available for {} cmd={}",
                port_name,
                cmd_id
            );
        }
    }

    if let Some(sub_service) = crate::services::am::proxy_subsession(port_name, cmd_id) {
        log::debug!(
            "{} cmd={} → returning {} sub-session",
            port_name,
            cmd_id,
            sub_service
        );
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if port_name == "fsp-srv" && cmd_id == 51 {
        if let Some(root) = fs_save_data_root(kernel, ctx) {
            log::debug!(
                "fsp-srv.OpenSaveDataFileSystem title_id={:#018x} root={}",
                kernel.title_id,
                root.display()
            );
            return return_file_system_with_root(kernel, ctx, session_handle, root);
        }
        return build_ipc_response(ctx, 0x202, &[], &[]);
    }

    if let Some(sub_service) = subsession_service(port_name, cmd_id) {
        return return_subsession(kernel, ctx, session_handle, sub_service);
    }

    if let Some((rc, data, handles)) =
        crate::services::am::dispatch_command(kernel, port_name, cmd_id)
    {
        log::trace!(
            "am.{}.cmd_{} rc={:#x} → {} bytes, {} handle(s) [copy]",
            port_name,
            cmd_id,
            rc,
            data.len(),
            handles.len()
        );
        return build_ipc_response_copy(ctx, rc, &data, &handles);
    }

    if port_name == "IFriendService" {
        match cmd_id {
            0 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            10101 | 10400 | 20100 | 20101 | 20200 | 22010 => {
                return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]);
            }
            10120 | 10420 => return build_ipc_response(ctx, 0, &[1], &[]),
            10601 | 10610 | 10700 => return build_ipc_response(ctx, 0, &[], &[]),
            other => {
                log::debug!("IFriendService.cmd_{} stubbed empty success", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "INfpUser" {
        match cmd_id {
            0 | 1 => return build_ipc_response(ctx, 0, &[], &[]),
            2 => return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]),
            17 | 18 | 23 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                log::debug!("nfp IUser.cmd_{} → event {:#x}", cmd_id, h);
                return build_ipc_response_copy(ctx, 0, &[], &[h]);
            }
            19 => return build_ipc_response(ctx, 0, &1u32.to_le_bytes(), &[]),
            20 | 21 => return build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[]),
            other => {
                log::debug!("nfp IUser.cmd_{} stubbed empty success", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "IFileSystem" {
        let fs_obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let path_str = fs_read_path(ctx, &kernel.address_space);
        let basename = std::path::Path::new(&path_str)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        match cmd_id {
            0 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    log::warn!(
                        "IFileSystem.CreateFile path={:?} → 0x202 PathNotFound",
                        path_str
                    );
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                if let Some(parent) = host.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&host)
                {
                    Ok(_) => {
                        log::debug!("IFileSystem.CreateFile path={:?} → SUCCESS", path_str);
                        return build_ipc_response(ctx, 0, &[], &[]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        return build_ipc_response(ctx, 0x402, &[], &[]);
                    }
                    Err(_) => return build_ipc_response(ctx, 0x402, &[], &[]),
                }
            }
            1 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_file(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            2 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::create_dir_all(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x402, &[], &[]),
                }
            }
            3 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_dir(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            4 => {
                let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                let Some(host) = host else {
                    return build_ipc_response(ctx, 0x202, &[], &[]);
                };
                match std::fs::remove_dir_all(&host) {
                    Ok(()) => return build_ipc_response(ctx, 0, &[], &[]),
                    Err(_) => return build_ipc_response(ctx, 0x202, &[], &[]),
                }
            }
            7 => {
                let entry_type: u32 = {
                    let in_homebrew = !basename.is_empty()
                        && kernel
                            .homebrew_dir
                            .as_ref()
                            .map(|d| d.join(&basename).is_file())
                            .unwrap_or(false);
                    if in_homebrew {
                        1
                    } else if let Some(host) =
                        fs_host_path(kernel, session_handle, fs_obj_id, &path_str)
                    {
                        match std::fs::metadata(&host) {
                            Ok(m) if m.is_dir() => 0,
                            Ok(_) => 1,
                            Err(_) => {
                                if let Some(ty) = romfs_entry_type(kernel.nro_romfs(), &path_str) {
                                    ty
                                } else {
                                    log::debug!(
                                        "IFileSystem.GetEntryType path={:?} → 0x202 NotFound",
                                        path_str
                                    );
                                    return build_ipc_response(ctx, 0x202, &[], &[]);
                                }
                            }
                        }
                    } else if let Some(ty) = romfs_entry_type(kernel.nro_romfs(), &path_str) {
                        ty
                    } else {
                        1
                    }
                };
                log::debug!(
                    "IFileSystem.GetEntryType path={:?} → {}",
                    path_str,
                    entry_type
                );
                return build_ipc_response(ctx, 0, &entry_type.to_le_bytes(), &[]);
            }
            8 => {
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

                let is_domain = kernel
                    .sessions
                    .get(&session_handle)
                    .map(|s| s.is_domain)
                    .unwrap_or(false);
                let new_obj_id = if is_domain {
                    next_domain_object_id(kernel, session_handle)
                } else {
                    0
                };

                if let Some(m) = mmap_arc {
                    let mmap_len = m.len();
                    kernel.open_files.insert((session_handle, new_obj_id), m);
                    log::debug!(
                        "IFileSystem.OpenFile path={:?} → IFile (NRO mmap {} bytes)",
                        path_str,
                        mmap_len
                    );
                } else {
                    let host = fs_host_path(kernel, session_handle, fs_obj_id, &path_str);
                    if let Some(host) = host {
                        if host.is_file() {
                            kernel
                                .open_host_files
                                .insert((session_handle, new_obj_id), host.clone());
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} → IFile (host {})",
                                path_str,
                                host.display()
                            );
                        } else if let Some(rf) = romfs_open_file(kernel.nro_romfs(), &path_str) {
                            kernel
                                .open_romfs_files
                                .insert((session_handle, new_obj_id), rf);
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} → IFile (romfs off={:#x} size={})",
                                path_str,
                                rf.0,
                                rf.1
                            );
                        } else {
                            log::debug!(
                                "IFileSystem.OpenFile path={:?} → 0x202 NotFound (host miss)",
                                path_str
                            );
                            return build_ipc_response(ctx, 0x202, &[], &[]);
                        }
                    } else if let Some(rf) = romfs_open_file(kernel.nro_romfs(), &path_str) {
                        kernel
                            .open_romfs_files
                            .insert((session_handle, new_obj_id), rf);
                        log::debug!(
                            "IFileSystem.OpenFile path={:?} → IFile (romfs off={:#x} size={})",
                            path_str,
                            rf.0,
                            rf.1
                        );
                    } else {
                        log::debug!("IFileSystem.OpenFile path={:?} → 0x202 NotFound", path_str);
                        return build_ipc_response(ctx, 0x202, &[], &[]);
                    }
                }
                return return_subsession(kernel, ctx, session_handle, "IFile");
            }
            9 => {
                let in_off = ctx.cmif_in_data_off;
                let filter = if ctx.cmif_in_data_len >= 4 {
                    u32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ])
                } else {
                    0
                };

                let is_domain = kernel
                    .sessions
                    .get(&session_handle)
                    .map(|s| s.is_domain)
                    .unwrap_or(false);
                let new_obj_id = if is_domain {
                    next_domain_object_id(kernel, session_handle)
                } else {
                    0
                };

                let mut entries: Vec<(String, bool, u64)> = Vec::new();
                if let Some(host) = fs_host_path(kernel, session_handle, fs_obj_id, &path_str) {
                    let _ = std::fs::create_dir_all(&host);
                    if let Ok(rd) = std::fs::read_dir(&host) {
                        for e in rd.filter_map(|e| e.ok()) {
                            let Ok(md) = e.metadata() else { continue };
                            let name = e.file_name().to_string_lossy().into_owned();
                            let is_dir = md.is_dir();
                            if is_dir && filter & 1 == 0 {
                                continue;
                            }
                            if !is_dir && filter & 2 == 0 {
                                continue;
                            }
                            entries.push((name, is_dir, if is_dir { 0 } else { md.len() }));
                        }
                    }
                }
                let is_switch_path = path_str == "/switch" || path_str == "/switch/";
                if is_switch_path {
                    if let Some(dir) = &kernel.homebrew_dir {
                        let mut seen: std::collections::HashSet<String> =
                            entries.iter().map(|(n, _, _)| n.clone()).collect();
                        if let Ok(rd) = std::fs::read_dir(dir) {
                            for e in rd.filter_map(|e| e.ok()) {
                                let Ok(md) = e.metadata() else { continue };
                                let name = e.file_name().to_string_lossy().into_owned();
                                if seen.contains(&name) {
                                    continue;
                                }
                                let is_dir = md.is_dir();
                                if is_dir && filter & 1 == 0 {
                                    continue;
                                }
                                if !is_dir && filter & 2 == 0 {
                                    continue;
                                }
                                entries.push((
                                    name.clone(),
                                    is_dir,
                                    if is_dir { 0 } else { md.len() },
                                ));
                                seen.insert(name);
                            }
                        }
                    }
                }
                log::debug!(
                    "IFileSystem.OpenDirectory path={:?} filter={:#x} → {} entries",
                    path_str,
                    filter,
                    entries.len()
                );
                kernel
                    .open_dir_lists
                    .insert((session_handle, new_obj_id), (entries, 0));
                kernel.dir_cursor.insert(session_handle, 0);
                return return_subsession(kernel, ctx, session_handle, "IDirectory");
            }
            10 => return build_ipc_response(ctx, 0, &[], &[]),
            11 | 12 => {
                let huge: u64 = 64u64 * 1024 * 1024 * 1024;
                log::debug!(
                    "IFileSystem.Get{}SpaceSize → {}",
                    if cmd_id == 11 { "Free" } else { "Total" },
                    huge
                );
                return build_ipc_response(ctx, 0, &huge.to_le_bytes(), &[]);
            }
            14 => {
                log::debug!("IFileSystem.GetFileTimeStampRaw → zeros");
                return build_ipc_response(ctx, 0, &[0u8; 0x20], &[]);
            }
            _ => {
                log::warn!("IFileSystem.cmd_{} UNHANDLED → empty SUCCESS", cmd_id);
            }
        }
    }

    if port_name == "IFile" {
        let obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let object_keys = domain_object_keys(kernel, session_handle, obj_id);
        let per_session = object_keys
            .iter()
            .find_map(|key| kernel.open_files.get(key).cloned());
        match cmd_id {
            0 => {
                let in_off = ctx.cmif_in_data_off;
                let avail = ctx.buf.len().saturating_sub(in_off);
                if avail < 24 {
                    log::warn!("IFile.Read: short input ({} bytes)", avail);
                    return build_ipc_response(ctx, 0, &0u64.to_le_bytes(), &[]);
                }
                let offset = i64::from_le_bytes([
                    ctx.buf[in_off + 8],
                    ctx.buf[in_off + 9],
                    ctx.buf[in_off + 10],
                    ctx.buf[in_off + 11],
                    ctx.buf[in_off + 12],
                    ctx.buf[in_off + 13],
                    ctx.buf[in_off + 14],
                    ctx.buf[in_off + 15],
                ]);
                let read_size = u64::from_le_bytes([
                    ctx.buf[in_off + 16],
                    ctx.buf[in_off + 17],
                    ctx.buf[in_off + 18],
                    ctx.buf[in_off + 19],
                    ctx.buf[in_off + 20],
                    ctx.buf[in_off + 21],
                    ctx.buf[in_off + 22],
                    ctx.buf[in_off + 23],
                ]);
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                let romfs_file = object_keys
                    .iter()
                    .find_map(|key| kernel.open_romfs_files.get(key).copied());
                let target = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                let mut bytes_read: u64 = 0;
                if let Some(buf) = target {
                    if let Some(host) = host_path {
                        let want = (read_size as usize).min(buf.size as usize);
                        let mmap = if let Some(m) = kernel.host_file_cache.get(&host) {
                            Some(m.clone())
                        } else {
                            match std::fs::File::open(&host)
                                .and_then(|f| unsafe { memmap2::Mmap::map(&f) })
                            {
                                Ok(m) => {
                                    let a = std::sync::Arc::new(m);
                                    kernel.host_file_cache.insert(host.clone(), a.clone());
                                    Some(a)
                                }
                                Err(_) => None,
                            }
                        };
                        if let Some(m) = mmap {
                            let start = (offset.max(0) as usize).min(m.len());
                            let end = start.saturating_add(want).min(m.len());
                            let slice = &m[start..end];
                            let _ = kernel.address_space.write(buf.addr, slice);
                            bytes_read = slice.len() as u64;
                        }
                        log::debug!(
                            "IFile.Read (host {}) off={:#x} size={:#x} → {} bytes",
                            host.display(),
                            offset,
                            read_size,
                            bytes_read
                        );
                    } else if let Some((base, size)) = romfs_file {
                        let romfs = kernel.nro_romfs();
                        let off = offset.max(0) as usize;
                        let start = base.saturating_add(off).min(romfs.len());
                        let remaining = size.saturating_sub(off);
                        let want = (read_size as usize).min(buf.size as usize).min(remaining);
                        let end = start.saturating_add(want).min(romfs.len());
                        let slice = &romfs[start..end];
                        let _ = kernel.address_space.write(buf.addr, slice);
                        bytes_read = slice.len() as u64;
                        log::debug!(
                            "IFile.Read (romfs off={:#x}) read_off={:#x} size={:#x} → {} bytes",
                            base,
                            offset,
                            read_size,
                            bytes_read
                        );
                    } else {
                        let file_bytes: &[u8] = match per_session.as_ref() {
                            Some(m) => &m[..],
                            None => kernel.nro_mmap.as_ref().map(|m| &m[..]).unwrap_or(&[]),
                        };
                        let start = (offset.max(0) as usize).min(file_bytes.len());
                        let want = (read_size as usize).min(buf.size as usize);
                        let end = start.saturating_add(want).min(file_bytes.len());
                        let slice = &file_bytes[start..end];
                        let _ = kernel.address_space.write(buf.addr, slice);
                        bytes_read = slice.len() as u64;
                        log::debug!(
                            "IFile.Read (sess={:#x} obj={}) off={:#x} size={:#x} → {} bytes",
                            session_handle,
                            obj_id,
                            offset,
                            read_size,
                            slice.len(),
                        );
                    }
                } else {
                    log::warn!(
                        "IFile.Read: no recv buffer (off={:#x} size={:#x})",
                        offset,
                        read_size
                    );
                }
                return build_ipc_response(ctx, 0, &bytes_read.to_le_bytes(), &[]);
            }
            1 => {
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                if let Some(host) = host_path {
                    use std::io::{Seek, SeekFrom, Write};
                    let in_off = ctx.cmif_in_data_off;
                    if ctx.cmif_in_data_len >= 24 {
                        let offset = i64::from_le_bytes([
                            ctx.buf[in_off + 8],
                            ctx.buf[in_off + 9],
                            ctx.buf[in_off + 10],
                            ctx.buf[in_off + 11],
                            ctx.buf[in_off + 12],
                            ctx.buf[in_off + 13],
                            ctx.buf[in_off + 14],
                            ctx.buf[in_off + 15],
                        ]);
                        let size = u64::from_le_bytes([
                            ctx.buf[in_off + 16],
                            ctx.buf[in_off + 17],
                            ctx.buf[in_off + 18],
                            ctx.buf[in_off + 19],
                            ctx.buf[in_off + 20],
                            ctx.buf[in_off + 21],
                            ctx.buf[in_off + 22],
                            ctx.buf[in_off + 23],
                        ]);
                        if let Some(send_buf) = ctx
                            .send_buffers
                            .iter()
                            .find(|b| b.size > 0 && b.addr != 0)
                            .copied()
                        {
                            let n = (send_buf.size.min(size)) as usize;
                            let mut data = vec![0u8; n];
                            if kernel.address_space.read(send_buf.addr, &mut data).is_ok() {
                                let res = std::fs::OpenOptions::new()
                                    .write(true)
                                    .create(true)
                                    .open(&host)
                                    .and_then(|mut f| {
                                        f.seek(SeekFrom::Start(offset.max(0) as u64))?;
                                        f.write_all(&data)
                                    });
                                if res.is_ok() {
                                    for key in &object_keys {
                                        kernel.open_file_handles.remove(key);
                                    }
                                    kernel.host_file_cache.remove(&host);
                                    log::debug!(
                                        "IFile.Write (host {}) off={:#x} size={} → SUCCESS",
                                        host.display(),
                                        offset,
                                        n
                                    );
                                    return build_ipc_response(ctx, 0, &[], &[]);
                                }
                            }
                        }
                    }
                    return build_ipc_response(ctx, 0x2EE602, &[], &[]);
                }
                log::debug!("IFile.Write (read-only mmap) → SUCCESS discarded");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            2 => return build_ipc_response(ctx, 0, &[], &[]),
            3 => {
                let host_path = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key).cloned());
                if let Some(host) = host_path {
                    let in_off = ctx.cmif_in_data_off;
                    if ctx.cmif_in_data_len >= 8 {
                        let new_size = u64::from_le_bytes([
                            ctx.buf[in_off],
                            ctx.buf[in_off + 1],
                            ctx.buf[in_off + 2],
                            ctx.buf[in_off + 3],
                            ctx.buf[in_off + 4],
                            ctx.buf[in_off + 5],
                            ctx.buf[in_off + 6],
                            ctx.buf[in_off + 7],
                        ]);
                        let res = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&host)
                            .and_then(|f| f.set_len(new_size));
                        if res.is_ok() {
                            kernel.host_file_cache.remove(&host);
                            return build_ipc_response(ctx, 0, &[], &[]);
                        }
                    }
                    return build_ipc_response(ctx, 0x2EE602, &[], &[]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 => {
                let size: i64 = if let Some(host) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_host_files.get(key))
                {
                    std::fs::metadata(host).map(|m| m.len() as i64).unwrap_or(0)
                } else if let Some((_, size)) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_romfs_files.get(key).copied())
                {
                    size as i64
                } else {
                    match per_session.as_ref() {
                        Some(m) => m.len() as i64,
                        None => kernel
                            .nro_mmap
                            .as_ref()
                            .map(|m| m.len() as i64)
                            .unwrap_or(0),
                    }
                };
                log::debug!(
                    "IFile.GetSize (sess={:#x} obj={}) → {}",
                    session_handle,
                    obj_id,
                    size
                );
                return build_ipc_response(ctx, 0, &size.to_le_bytes(), &[]);
            }
            _ => {
                log::warn!("IFile.cmd_{} UNHANDLED → empty SUCCESS", cmd_id);
            }
        }
    }

    if port_name == "IDirectory" {
        let obj_id = ctx.domain.map(|d| d.object_id).unwrap_or(0);
        let object_keys = domain_object_keys(kernel, session_handle, obj_id);
        match cmd_id {
            0 => {
                let target = ctx
                    .recv_buffers
                    .iter()
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

                let dir_key = object_keys
                    .iter()
                    .copied()
                    .find(|key| kernel.open_dir_lists.contains_key(key));
                if let Some((entries, cursor)) =
                    dir_key.and_then(|key| kernel.open_dir_lists.get_mut(&key))
                {
                    let remaining = entries.len().saturating_sub(*cursor);
                    let to_emit = remaining.min(max_entries);
                    let mut payload = vec![0u8; to_emit * 0x310];
                    for (i, (name, is_dir, size)) in
                        entries.iter().skip(*cursor).take(to_emit).enumerate()
                    {
                        let base = i * 0x310;
                        let name_bytes = name.as_bytes();
                        let name_len = name_bytes.len().min(0x300);
                        payload[base..base + name_len].copy_from_slice(&name_bytes[..name_len]);
                        payload[base + 0x304] = if *is_dir { 0 } else { 1 };
                        payload[base + 0x308..base + 0x310].copy_from_slice(&size.to_le_bytes());
                    }
                    *cursor += to_emit;
                    if !payload.is_empty() {
                        let _ = kernel.address_space.write(buf.addr, &payload);
                    }
                    log::debug!(
                        "IDirectory.Read (host) → {} of {} entries",
                        to_emit,
                        entries.len()
                    );
                    return build_ipc_response(ctx, 0, &(to_emit as i64).to_le_bytes(), &[]);
                }

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
                    "IDirectory.Read (homebrew_dir fallback) cursor={} → {} of {}",
                    cursor,
                    to_emit,
                    entries.len()
                );
                return build_ipc_response(ctx, 0, &(to_emit as i64).to_le_bytes(), &[]);
            }
            1 => {
                let count: i64 = if let Some((entries, _)) = object_keys
                    .iter()
                    .find_map(|key| kernel.open_dir_lists.get(key))
                {
                    entries.len() as i64
                } else {
                    enumerate_homebrew_nros(&kernel.homebrew_dir).len() as i64
                };
                log::info!("IDirectory.GetEntryCount → {}", count);
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            _ => {
                log::warn!("IDirectory.cmd_{} UNHANDLED → empty SUCCESS", cmd_id);
            }
        }
    }

    if port_name == "IFsStorage" {
        match cmd_id {
            0 => {
                let off_lo = ctx.cmif_in_data_off;
                let read_in = &ctx.buf[off_lo..off_lo + 16];
                let offset = i64::from_le_bytes([
                    read_in[0], read_in[1], read_in[2], read_in[3], read_in[4], read_in[5],
                    read_in[6], read_in[7],
                ]);
                let read_size = u64::from_le_bytes([
                    read_in[8],
                    read_in[9],
                    read_in[10],
                    read_in[11],
                    read_in[12],
                    read_in[13],
                    read_in[14],
                    read_in[15],
                ]);
                let romfs = kernel.nro_romfs();
                let target = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(buf) = target {
                    let start = (offset.max(0) as usize).min(romfs.len());
                    let want = (read_size as usize).min(buf.size as usize);
                    let end = start.saturating_add(want).min(romfs.len());
                    let slice = &romfs[start..end];
                    let _ = kernel.address_space.write(buf.addr, slice);
                    log::debug!(
                        "IFsStorage.Read off={:#x} size={:#x} bytes={} total={}",
                        offset,
                        read_size,
                        slice.len(),
                        romfs.len()
                    );
                } else {
                    log::warn!(
                        "IFsStorage.Read: no recv buffer (off={:#x} size={:#x})",
                        offset,
                        read_size
                    );
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

    if (port_name == "audren:u" || port_name == "audren:a") && cmd_id == 0 {
        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let read_u32 = |o: usize| -> u32 {
            if in_avail >= o + 4 {
                u32::from_le_bytes([
                    ctx.buf[in_off + o],
                    ctx.buf[in_off + o + 1],
                    ctx.buf[in_off + o + 2],
                    ctx.buf[in_off + o + 3],
                ])
            } else {
                0
            }
        };
        let sample_rate = {
            let v = read_u32(0);
            if v == 0 {
                48000
            } else {
                v
            }
        };
        let sample_count = {
            let v = read_u32(4);
            if v == 0 {
                240
            } else {
                v
            }
        };
        let mix_buffer_count = read_u32(8);
        let voice_count = read_u32(0x10);
        let sink_count = read_u32(0x14);
        let effect_count = read_u32(0x18);
        let revision = read_u32(0x30);

        let state = AudioRendererState {
            sample_rate,
            sample_count,
            mix_buffer_count,
            voice_count,
            sink_count,
            effect_count,
            revision,
            state: 1,
            rendering_time_limit: 100,
            voice_drop_param: 1.0,
            voice_played_samples: Vec::new(),
            voice_wbufs_consumed: Vec::new(),
            voice_last_wb_index: Vec::new(),
            voice_is_new_seen: Vec::new(),
            voice_wb_progress_frames: Vec::new(),
            voice_frac_q15: Vec::new(),
            voice_hist: Vec::new(),
        };

        let is_domain = kernel
            .sessions
            .get(&session_handle)
            .map(|s| s.is_domain)
            .unwrap_or(false);
        log::info!(
            "audren:u OpenAudioRenderer sr={} samples={} voices={} sinks={} effects={} rev={:#x} → IAudioRenderer (domain={})",
            sample_rate,
            sample_count,
            voice_count,
            sink_count,
            effect_count,
            revision,
            is_domain
        );

        if is_domain {
            let object_id = alloc_domain_object(kernel, session_handle, "IAudioRenderer");
            kernel
                .audio_renderers
                .insert((session_handle, object_id), state);
            return build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id]);
        } else {
            let h = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(h, "IAudioRenderer".to_string());
            kernel.sessions.insert(h, session);
            kernel.audio_renderers.insert((h, 0), state);
            return build_ipc_response(ctx, 0, &[], &[h]);
        }
    }
    if (port_name == "audren:u" || port_name == "audren:a") && cmd_id == 1 {
        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let rd = |o: usize| -> u64 {
            if in_avail >= o + 4 {
                u32::from_le_bytes([
                    ctx.buf[in_off + o],
                    ctx.buf[in_off + o + 1],
                    ctx.buf[in_off + o + 2],
                    ctx.buf[in_off + o + 3],
                ]) as u64
            } else {
                0
            }
        };
        let align_up = |v: u64, a: u64| (v + a - 1) & !(a - 1);
        let sample_count = {
            let v = rd(4);
            if v == 0 {
                240
            } else {
                v
            }
        };
        let mixes = rd(8);
        let sub_mixes = rd(0xC);
        let voices = rd(0x10);
        let sinks = rd(0x14);
        let effects = rd(0x18);
        const TARGET: u64 = 240;
        const MAXCH: u64 = 6;
        let mut size: u64 = 0x4000;
        size += (sub_mixes + 1) * 0xC00;
        size += voices * 0x1400;
        size += effects * 0x400;
        size += align_up(
            ((sinks + sub_mixes) * TARGET + sample_count) * 4 * (mixes + MAXCH),
            0x40,
        );
        size += (sinks + sub_mixes) * 0xC00;
        size += 0x40000;
        let computed = align_up(size, 0x1000);
        let work_buffer_size = std::env::var("NEXIUM_AUDIO_WORKBUF")
            .ok()
            .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| computed.clamp(0x20_0000, 0x80_0000));
        log::info!(
            "audren GetWorkBufferSize voices={} effects={} mixes={} → {:#x} (computed {:#x})",
            voices,
            effects,
            mixes,
            work_buffer_size,
            computed
        );
        return build_ipc_response(ctx, 0, &work_buffer_size.to_le_bytes(), &[]);
    }
    if (port_name == "audren:u" || port_name == "audren:a") && (cmd_id == 2 || cmd_id == 4) {
        return return_subsession(kernel, ctx, session_handle, "IAudioDevice");
    }

    if port_name == "IAudioRenderer" {
        let obj_id = ctx.domain.as_ref().map(|d| d.object_id).unwrap_or(0);
        let key = (session_handle, obj_id);
        let st = kernel
            .audio_renderers
            .entry(key)
            .or_insert(AudioRendererState {
                sample_rate: 48000,
                sample_count: 240,
                mix_buffer_count: 0,
                voice_count: 0,
                sink_count: 0,
                effect_count: 0,
                revision: 0,
                state: 1,
                rendering_time_limit: 100,
                voice_drop_param: 1.0,
                voice_played_samples: Vec::new(),
                voice_wbufs_consumed: Vec::new(),
                voice_last_wb_index: Vec::new(),
                voice_is_new_seen: Vec::new(),
                voice_wb_progress_frames: Vec::new(),
                voice_frac_q15: Vec::new(),
                voice_hist: Vec::new(),
            });
        match cmd_id {
            0 => {
                let v = st.sample_rate;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            1 => {
                let v = st.sample_count;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            2 => {
                let v = st.mix_buffer_count;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            3 => {
                let v = st.state;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            4 | 10 => {
                let in_buf = ctx
                    .send_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .copied();
                let out_buf = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .copied();
                let perf_buf = ctx
                    .recv_buffers
                    .iter()
                    .filter(|b| b.size > 0 && b.addr != 0)
                    .nth(1)
                    .copied();
                let revision = st.revision;
                let voice_drop_param = st.voice_drop_param;
                let frame = kernel.audio_renderer_frame_counter;

                let mut in_behavior_sz: u64 = 0;
                let mut in_mempools_sz: u64 = 0;
                let mut in_voices_sz: u64 = 0;
                let mut in_channels_sz: u64 = 0;
                let mut in_effects_sz: u64 = 0;
                let mut in_sinks_sz: u64 = 0;
                let mut in_perf_sz: u64 = 0;
                let mut mempool_in_states: Vec<u32> = Vec::new();
                if let Some(ib) = in_buf {
                    let mut hdr = [0u8; 0x40];
                    if (ib.size as usize) >= 0x40
                        && kernel.address_space.read(ib.addr, &mut hdr).is_ok()
                    {
                        let rd = |off: usize| {
                            u32::from_le_bytes([hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]])
                                as u64
                        };
                        in_behavior_sz = rd(0x04);
                        in_mempools_sz = rd(0x08);
                        in_voices_sz = rd(0x0C);
                        in_channels_sz = rd(0x10);
                        in_effects_sz = rd(0x14);
                        in_sinks_sz = rd(0x1C);
                        in_perf_sz = rd(0x20);
                    }
                    if in_mempools_sz > 0 {
                        let mempool_count = (in_mempools_sz / 0x20) as usize;
                        let mempools_off = 0x40u64 + in_behavior_sz;
                        mempool_in_states.reserve(mempool_count);
                        for i in 0..mempool_count {
                            let off = mempools_off + (i as u64) * 0x20 + 0x10;
                            let mut sb = [0u8; 4];
                            if kernel
                                .address_space
                                .read(ib.addr.wrapping_add(off), &mut sb)
                                .is_ok()
                            {
                                mempool_in_states.push(u32::from_le_bytes(sb));
                            } else {
                                mempool_in_states.push(0);
                            }
                        }
                    }
                }
                let mempool_count = mempool_in_states.len();
                let voice_count_seen = (in_voices_sz / 0x170) as usize;
                let effect_count_seen = (in_effects_sz / 0xC0) as usize;
                let mut effect_out_states: Vec<u8> = vec![4; effect_count_seen];
                if let Some(ib) = in_buf {
                    let effects_in_off =
                        0x40u64 + in_behavior_sz + in_mempools_sz + in_channels_sz + in_voices_sz;
                    for i in 0..effect_count_seen {
                        let off = effects_in_off + (i as u64) * 0xC0;
                        let mut eb = [0u8; 3];
                        if kernel
                            .address_space
                            .read(ib.addr.wrapping_add(off), &mut eb)
                            .is_ok()
                        {
                            let ty = eb[0];
                            let is_new = eb[1] != 0;
                            let enabled = eb[2] != 0;
                            effect_out_states[i] =
                                if ty != 0 && (st.state == 0 || is_new || enabled) {
                                    3
                                } else {
                                    4
                                };
                        }
                    }
                }
                if st.voice_played_samples.len() < voice_count_seen {
                    st.voice_played_samples.resize(voice_count_seen, 0);
                    st.voice_wbufs_consumed.resize(voice_count_seen, 0);
                    st.voice_last_wb_index.resize(voice_count_seen, 0);
                    st.voice_is_new_seen.resize(voice_count_seen, false);
                    st.voice_wb_progress_frames.resize(voice_count_seen, 0);
                    st.voice_frac_q15.resize(voice_count_seen, 0);
                    st.voice_hist.resize(voice_count_seen, [0.0f32; 6]);
                }

                const TARGET_FRAMES: usize = 240;
                const TARGET_SR: f32 = 48_000.0;

                const RING_HIGH_WATER_FRAMES: usize = 3_840;

                let queued_now = crate::audio_sink::host_audio_sink()
                    .map(|s| s.queued_frames())
                    .unwrap_or(0);

                let blocks_to_produce: usize = if queued_now >= RING_HIGH_WATER_FRAMES {
                    0
                } else {
                    1
                };
                let mut is_new_latched: Vec<bool> = vec![false; voice_count_seen];
                let mut big_out: Vec<f32> =
                    Vec::with_capacity(TARGET_FRAMES * 2 * blocks_to_produce);

                for _block in 0..blocks_to_produce {
                    let mut out_stereo = vec![0.0f32; TARGET_FRAMES * 2];
                    let mut block_consumed_wb = false;
                    let mut voice_snapshot: Vec<(u16, bool, bool, u32, u32, u32, bool)> =
                        vec![(0u16, false, false, 0u32, 0u32, 0u32, false); voice_count_seen];

                    fn decode_gc_adpcm(
                        data: &[u8],
                        coeffs: &[i16; 16],
                        yn0_seed: i16,
                        yn1_seed: i16,
                        count: usize,
                    ) -> Vec<i16> {
                        let mut out: Vec<i16> = Vec::with_capacity(count);
                        let mut yn0 = yn0_seed as i64;
                        let mut yn1 = yn1_seed as i64;
                        let mut pos = 0usize;
                        while out.len() < count {
                            if pos >= data.len() {
                                break;
                            }
                            let header = data[pos];
                            pos += 1;
                            let ci = ((header >> 4) & 0xF) as usize;
                            let scale = (header & 0xF) as u32;
                            let c0 = coeffs[ci * 2] as i64;
                            let c1 = coeffs[ci * 2 + 1] as i64;
                            for _ in 0..7 {
                                if out.len() >= count || pos >= data.len() {
                                    break;
                                }
                                let byte = data[pos];
                                pos += 1;
                                for nib in [(byte >> 4) & 0xF, byte & 0xF] {
                                    if out.len() >= count {
                                        break;
                                    }
                                    let code = if nib >= 8 {
                                        nib as i64 - 16
                                    } else {
                                        nib as i64
                                    };
                                    let xn = code * (1i64 << scale);
                                    let pred = c0 * yn0 + c1 * yn1;
                                    let s =
                                        (((xn << 11) + 0x400 + pred) >> 11).clamp(-0x8000, 0x7FFF);
                                    yn1 = yn0;
                                    yn0 = s;
                                    out.push(s as i16);
                                }
                            }
                        }
                        while out.len() < count {
                            out.push(0);
                        }
                        out
                    }

                    'mix: {
                        let Some(ib) = in_buf else {
                            break 'mix;
                        };
                        if (ib.size as usize) < 0x40 || voice_count_seen == 0 {
                            break 'mix;
                        }
                        let voices_off: u64 =
                            0x40 + in_behavior_sz + in_mempools_sz + in_channels_sz;
                        let voice_info_stride: u64 = 0x170;

                        for vid in 0..voice_count_seen {
                            let vinfo_off = voices_off + (vid as u64) * voice_info_stride;
                            if vinfo_off + voice_info_stride > ib.size as u64 {
                                break;
                            }
                            let v0_addr = ib.addr.wrapping_add(vinfo_off);
                            let mut v = [0u8; 0x170];
                            if kernel.address_space.read(v0_addr, &mut v).is_err() {
                                continue;
                            }

                            let is_new = v[0x008] != 0;
                            let is_in_use = v[0x009] != 0;
                            let play_state = v[0x00A];
                            let sample_format = v[0x00B];
                            let sample_rate =
                                u32::from_le_bytes([v[0x00C], v[0x00D], v[0x00E], v[0x00F]]);
                            let channel_count =
                                u32::from_le_bytes([v[0x018], v[0x019], v[0x01A], v[0x01B]]);
                            let volume =
                                f32::from_le_bytes([v[0x020], v[0x021], v[0x022], v[0x023]]);
                            let wb_count =
                                u32::from_le_bytes([v[0x03C], v[0x03D], v[0x03E], v[0x03F]]);
                            let wb_index = u16::from_le_bytes([v[0x040], v[0x041]]) as usize;
                            voice_snapshot[vid].0 = wb_index as u16;
                            voice_snapshot[vid].1 = is_new;

                            if is_in_use {
                                use std::sync::atomic::{AtomicU64, Ordering as O};
                                static SEEN_MASK: AtomicU64 = AtomicU64::new(0);
                                let bit = 1u64 << ((vid as u64) & 63);
                                let prev = SEEN_MASK.fetch_or(bit, O::Relaxed);
                                if prev & bit == 0 {
                                    let fmt_name = match sample_format {
                                        0 => "Invalid",
                                        1 => "PcmInt8",
                                        2 => "PcmInt16",
                                        3 => "PcmInt24",
                                        4 => "PcmInt32",
                                        5 => "PcmFloat",
                                        6 => "Adpcm",
                                        _ => "?",
                                    };
                                    log::trace!(
                                        "voice[{}] FIRST SEEN: fmt={}({}) ch={} sr={} vol={:.2} state={} wb_count={} wb_index={}",
                                        vid,
                                        fmt_name,
                                        sample_format,
                                        channel_count,
                                        sample_rate,
                                        volume,
                                        play_state,
                                        wb_count,
                                        wb_index
                                    );
                                }
                            }

                            if !is_in_use
                                || play_state != 0
                                || (sample_format != 2 && sample_format != 6)
                                || !(channel_count == 1 || channel_count == 2)
                                || sample_rate == 0
                                || wb_count == 0
                                || wb_index >= 4
                            {
                                continue;
                            }

                            let wb_base = 0x060 + wb_index * 0x38;
                            let wb = &v[wb_base..wb_base + 0x38];
                            let buffer_address = u64::from_le_bytes([
                                wb[0x00], wb[0x01], wb[0x02], wb[0x03], wb[0x04], wb[0x05],
                                wb[0x06], wb[0x07],
                            ]);
                            let buffer_size = u64::from_le_bytes([
                                wb[0x08], wb[0x09], wb[0x0A], wb[0x0B], wb[0x0C], wb[0x0D],
                                wb[0x0E], wb[0x0F],
                            ]);
                            let start_offset =
                                i32::from_le_bytes([wb[0x10], wb[0x11], wb[0x12], wb[0x13]]);
                            let end_offset =
                                i32::from_le_bytes([wb[0x14], wb[0x15], wb[0x16], wb[0x17]]);
                            let is_looping = wb[0x18] != 0;
                            if buffer_address == 0 || start_offset < 0 || end_offset <= start_offset
                            {
                                continue;
                            }

                            let ch = channel_count as usize;
                            let ratio = sample_rate as f32 / TARGET_SR;
                            let in_frames_needed =
                                ((TARGET_FRAMES as f32) * ratio).ceil() as usize + 3;
                            let wb_total_frames = (end_offset - start_offset) as usize;
                            let cursor =
                                (st.voice_wb_progress_frames.get(vid).copied().unwrap_or(0)
                                    as usize)
                                    .min(wb_total_frames.saturating_sub(1));
                            let in_frames = in_frames_needed;
                            let mut pcm_l = vec![0.0f32; in_frames];
                            let mut pcm_r = vec![0.0f32; in_frames];

                            if sample_format == 6 {
                                let coeff_addr = u64::from_le_bytes([
                                    v[0x048], v[0x049], v[0x04A], v[0x04B], v[0x04C], v[0x04D],
                                    v[0x04E], v[0x04F],
                                ]);
                                let ctx_addr = u64::from_le_bytes([
                                    wb[0x20], wb[0x21], wb[0x22], wb[0x23], wb[0x24], wb[0x25],
                                    wb[0x26], wb[0x27],
                                ]);
                                let mut coeff_bytes = [0u8; 32];
                                if coeff_addr == 0
                                    || kernel
                                        .address_space
                                        .read(coeff_addr, &mut coeff_bytes)
                                        .is_err()
                                {
                                    continue;
                                }
                                let mut coeffs = [0i16; 16];
                                for i in 0..16 {
                                    coeffs[i] = i16::from_le_bytes([
                                        coeff_bytes[i * 2],
                                        coeff_bytes[i * 2 + 1],
                                    ]);
                                }
                                let (mut yn0_seed, mut yn1_seed) = (0i16, 0i16);
                                if ctx_addr != 0 {
                                    let mut ctx = [0u8; 6];
                                    if kernel.address_space.read(ctx_addr, &mut ctx).is_ok() {
                                        yn0_seed = i16::from_le_bytes([ctx[2], ctx[3]]);
                                        yn1_seed = i16::from_le_bytes([ctx[4], ctx[5]]);
                                    }
                                }
                                let decode_through = start_offset as usize + cursor + in_frames;
                                let frames_needed = (decode_through + 13) / 14;
                                let bytes_needed = (frames_needed * 8).min(buffer_size as usize);
                                if buffer_address == 0 || bytes_needed < 8 {
                                    continue;
                                }
                                let mut adpcm = vec![0u8; bytes_needed];
                                if kernel
                                    .address_space
                                    .read(buffer_address, &mut adpcm)
                                    .is_err()
                                {
                                    continue;
                                }
                                let decoded = decode_gc_adpcm(
                                    &adpcm,
                                    &coeffs,
                                    yn0_seed,
                                    yn1_seed,
                                    decode_through,
                                );
                                let base = start_offset as usize + cursor;
                                for f in 0..in_frames {
                                    let s = decoded.get(base + f).copied().unwrap_or(0);
                                    pcm_l[f] = (s as f32) / 32768.0;
                                    pcm_r[f] = pcm_l[f];
                                }
                                {
                                    use std::sync::atomic::{AtomicU64, Ordering as O};
                                    static DUMPED: AtomicU64 = AtomicU64::new(0);
                                    let bit = 1u64 << ((vid as u64) & 63);
                                    if DUMPED.fetch_or(bit, O::Relaxed) & bit == 0 {
                                        let nz = decoded.iter().filter(|s| **s != 0).count();
                                        log::trace!(
                                            "voice[{}] ADPCM: decoded={} nonzero={} coeff_addr={:#x} ctx_addr={:#x} start_off={} in_frames={} sr={}",
                                            vid,
                                            decoded.len(),
                                            nz,
                                            coeff_addr,
                                            ctx_addr,
                                            start_offset,
                                            in_frames,
                                            sample_rate
                                        );
                                    }
                                }
                            } else {
                                let stride = 2 * ch;
                                let queued = (wb_count as usize).min(4);
                                let mut got = 0usize;
                                let mut k = 0usize;
                                let mut slot_cursor = cursor;
                                while got < in_frames && k < queued {
                                    let slot = (wb_index + k) % 4;
                                    let sb = 0x060 + slot * 0x38;
                                    let swb = &v[sb..sb + 0x38];
                                    let s_addr = u64::from_le_bytes([
                                        swb[0x00], swb[0x01], swb[0x02], swb[0x03], swb[0x04],
                                        swb[0x05], swb[0x06], swb[0x07],
                                    ]);
                                    let s_size = u64::from_le_bytes([
                                        swb[0x08], swb[0x09], swb[0x0A], swb[0x0B], swb[0x0C],
                                        swb[0x0D], swb[0x0E], swb[0x0F],
                                    ]);
                                    let s_start = i32::from_le_bytes([
                                        swb[0x10], swb[0x11], swb[0x12], swb[0x13],
                                    ]);
                                    let s_end = i32::from_le_bytes([
                                        swb[0x14], swb[0x15], swb[0x16], swb[0x17],
                                    ]);
                                    let s_loop = swb[0x18] != 0;
                                    if s_addr == 0 || s_start < 0 || s_end <= s_start {
                                        break;
                                    }
                                    let s_total = (s_end - s_start) as usize;
                                    if slot_cursor >= s_total {
                                        if s_loop {
                                            slot_cursor = 0;
                                        } else {
                                            k += 1;
                                            slot_cursor = 0;
                                            continue;
                                        }
                                    }
                                    let avail = s_total - slot_cursor;
                                    let want = (in_frames - got).min(avail);
                                    let boff = ((s_start as u64) + slot_cursor as u64)
                                        .wrapping_mul(stride as u64);
                                    if boff.saturating_add((want * stride) as u64) > s_size {
                                        break;
                                    }
                                    let mut buf = vec![0u8; want * stride];
                                    if kernel
                                        .address_space
                                        .read(s_addr.wrapping_add(boff), &mut buf)
                                        .is_err()
                                    {
                                        break;
                                    }
                                    for f in 0..want {
                                        let l = i16::from_le_bytes([
                                            buf[f * stride],
                                            buf[f * stride + 1],
                                        ]);
                                        pcm_l[got + f] = (l as f32) / 32768.0;
                                        if ch == 2 {
                                            let r = i16::from_le_bytes([
                                                buf[f * stride + 2],
                                                buf[f * stride + 3],
                                            ]);
                                            pcm_r[got + f] = (r as f32) / 32768.0;
                                        } else {
                                            pcm_r[got + f] = pcm_l[got + f];
                                        }
                                    }
                                    got += want;
                                    if s_loop {
                                        slot_cursor += want;
                                        if slot_cursor >= s_total {
                                            slot_cursor = 0;
                                        }
                                    } else {
                                        k += 1;
                                        slot_cursor = 0;
                                    }
                                }
                                if vid == 0 {
                                    use std::sync::atomic::{AtomicU64, Ordering as O};
                                    static DC: AtomicU64 = AtomicU64::new(0);
                                    let n = DC.fetch_add(1, O::Relaxed);
                                    if n % 256 == 0 {
                                        log::trace!(
                                            "voice[0] chain wb_count={} wb_index={} cursor={} got={} in_frames={}",
                                            wb_count,
                                            wb_index,
                                            cursor,
                                            got,
                                            in_frames
                                        );
                                    }
                                }
                                if got == 0 {
                                    continue;
                                }
                            }

                            let phist = st.voice_hist.get(vid).copied().unwrap_or([0.0f32; 6]);
                            let mut frac_q15: i32 =
                                st.voice_frac_q15.get(vid).copied().unwrap_or(0);
                            let step: i32 = ((sample_rate as f32 / TARGET_SR) * 32768.0) as i32;
                            let master = voice_drop_param.clamp(0.0, 4.0);
                            let gain = volume * master * 0.5;
                            let smp_l = |i: isize| -> f32 {
                                if i < 0 {
                                    phist[0]
                                } else {
                                    pcm_l[(i as usize).min(in_frames - 1)]
                                }
                            };
                            let smp_r = |i: isize| -> f32 {
                                if i < 0 {
                                    phist[3]
                                } else {
                                    pcm_r[(i as usize).min(in_frames - 1)]
                                }
                            };
                            let mut read_idx: usize = 0;
                            for i in 0..TARGET_FRAMES {
                                let p = ((frac_q15 >> 8) as usize & 127) * 4;
                                let c0 = NX_SRC_LUT_UP[p];
                                let c1 = NX_SRC_LUT_UP[p + 1];
                                let c2 = NX_SRC_LUT_UP[p + 2];
                                let c3 = NX_SRC_LUT_UP[p + 3];
                                let bi = read_idx as isize;
                                let ol = smp_l(bi - 1) * c0
                                    + smp_l(bi) * c1
                                    + smp_l(bi + 1) * c2
                                    + smp_l(bi + 2) * c3;
                                let orr = smp_r(bi - 1) * c0
                                    + smp_r(bi) * c1
                                    + smp_r(bi + 1) * c2
                                    + smp_r(bi + 2) * c3;
                                out_stereo[i * 2] += ol * gain;
                                out_stereo[i * 2 + 1] += orr * gain;
                                let no = frac_q15 + step;
                                read_idx += (no >> 15) as usize;
                                frac_q15 = no & 0x7fff;
                            }
                            let consumed =
                                read_idx.min(in_frames).saturating_sub(1).min(in_frames - 1);
                            let mut nh = [0.0f32; 6];
                            nh[0] = pcm_l[consumed];
                            nh[3] = pcm_r[consumed];
                            if let Some(h) = st.voice_hist.get_mut(vid) {
                                *h = nh;
                            }
                            if let Some(f) = st.voice_frac_q15.get_mut(vid) {
                                *f = frac_q15;
                            }

                            let src_frames_this_pass = read_idx as u32;
                            voice_snapshot[vid].2 = true;
                            voice_snapshot[vid].3 = src_frames_this_pass;
                            voice_snapshot[vid].4 = wb_total_frames as u32;
                            voice_snapshot[vid].5 = wb_count;
                            voice_snapshot[vid].6 = is_looping;
                        }
                    }

                    {
                        use std::sync::atomic::{AtomicBool, Ordering as O};
                        static LOGGED: AtomicBool = AtomicBool::new(false);
                        static CONSUMED_LOGGED: AtomicBool = AtomicBool::new(false);
                        let mut any_mix = false;
                        let mut any_consumed = false;
                        for vid in 0..voice_count_seen {
                            let (
                                wb_now,
                                is_new,
                                did_mix,
                                src_frames,
                                wb_total,
                                wb_count_now,
                                is_looping,
                            ) = voice_snapshot[vid];
                            if did_mix {
                                any_mix = true;
                            }
                            if is_new && !is_new_latched[vid] {
                                st.voice_played_samples[vid] = 0;
                                st.voice_wbufs_consumed[vid] = 0;
                                st.voice_last_wb_index[vid] = wb_now;
                                st.voice_is_new_seen[vid] = true;
                                is_new_latched[vid] = true;
                                if let Some(p) = st.voice_wb_progress_frames.get_mut(vid) {
                                    *p = 0;
                                }
                            } else if did_mix && wb_total > 0 {
                                st.voice_played_samples[vid] =
                                    st.voice_played_samples[vid].wrapping_add(src_frames as u64);

                                let prev_progress =
                                    st.voice_wb_progress_frames.get(vid).copied().unwrap_or(0);
                                let mut new_progress =
                                    prev_progress.wrapping_add(src_frames as u64);

                                let cap = wb_count_now.min(4) as u32;
                                let mut completed: u32 = 0;
                                if !is_looping {
                                    while new_progress >= wb_total as u64 && completed < cap {
                                        new_progress -= wb_total as u64;
                                        completed += 1;
                                    }
                                    if new_progress >= wb_total as u64 {
                                        use std::sync::atomic::{AtomicBool, Ordering as O2};
                                        static WARN_ONCE: AtomicBool = AtomicBool::new(false);
                                        if !WARN_ONCE.swap(true, O2::Relaxed) {
                                            log::warn!(
                                                "audio voice[{}] consume cap hit: residue={} wb_total={} completed={} cap={} (wb_count={})",
                                                vid,
                                                new_progress,
                                                wb_total,
                                                completed,
                                                cap,
                                                wb_count_now
                                            );
                                        }
                                    }
                                } else {
                                    while new_progress >= wb_total as u64 {
                                        new_progress -= wb_total as u64;
                                    }
                                }

                                if completed > 0 {
                                    st.voice_wbufs_consumed[vid] =
                                        st.voice_wbufs_consumed[vid].wrapping_add(completed);
                                    any_consumed = true;
                                    block_consumed_wb = true;
                                    use std::sync::atomic::{AtomicU64, Ordering as O3};
                                    static PER_VOICE_LOGGED: AtomicU64 = AtomicU64::new(0);
                                    let bit = 1u64 << ((vid as u64) & 63);
                                    let prev_mask = PER_VOICE_LOGGED.fetch_or(bit, O3::Relaxed);
                                    if prev_mask & bit == 0 {
                                        log::trace!(
                                            "voice[{}] FIRST CONSUMED BUMP: wb_index={}, samples_played={} (consumed_now={}, wb_total={}, completed={}, frame {})",
                                            vid,
                                            wb_now,
                                            st.voice_played_samples[vid],
                                            st.voice_wbufs_consumed[vid],
                                            wb_total,
                                            completed,
                                            frame
                                        );
                                    }
                                }
                                if let Some(p) = st.voice_wb_progress_frames.get_mut(vid) {
                                    *p = new_progress;
                                }
                                st.voice_last_wb_index[vid] = wb_now;
                            } else if did_mix {
                                st.voice_played_samples[vid] =
                                    st.voice_played_samples[vid].wrapping_add(TARGET_FRAMES as u64);
                                st.voice_last_wb_index[vid] = wb_now;
                            }
                        }
                        if any_mix && !LOGGED.swap(true, O::Relaxed) {
                            let mixed_ids: Vec<usize> = (0..voice_count_seen)
                                .filter(|&i| voice_snapshot[i].2)
                                .collect();
                            log::trace!(
                                "audio multi-voice MIXED first time: voice_count_seen={} mixed_voices={:?} (frame {})",
                                voice_count_seen,
                                mixed_ids,
                                frame
                            );
                        }
                        if any_consumed && !CONSUMED_LOGGED.swap(true, O::Relaxed) {
                            let states: Vec<(usize, u32, u64)> = (0..voice_count_seen)
                                .filter(|&i| voice_snapshot[i].2)
                                .map(|i| {
                                    (i, st.voice_wbufs_consumed[i], st.voice_played_samples[i])
                                })
                                .collect();
                            log::trace!(
                                "audio wavebuf FIRST CONSUMED: voices={:?} (frame {})",
                                states,
                                frame
                            );
                        }
                    }
                    big_out.extend_from_slice(&out_stereo);
                    if block_consumed_wb {
                        break;
                    }
                }

                if let Some(ob) = out_buf {
                    let rev_num = if revision >= 0x100 {
                        revision.wrapping_sub(0x3056_4552) >> 24
                    } else {
                        revision
                    };
                    let mempool_out_count = mempool_count;
                    let voice_out_count = voice_count_seen;
                    let effect_out_count = effect_count_seen;
                    let sink_out_count = (in_sinks_sz / 0x140) as usize;

                    let mempools_sz: u32 = (mempool_out_count as u32) * 0x10;
                    let voices_sz: u32 = (voice_out_count as u32) * 0x10;
                    let effect_status_size: u32 = if rev_num >= 9 { 0x90 } else { 0x10 };
                    let effects_sz: u32 = (effect_out_count as u32) * effect_status_size;
                    let sinks_sz: u32 = (sink_out_count as u32) * 0x20;
                    let perf_sz: u32 = if in_perf_sz == 0 { 0 } else { 0x10 };
                    let behaviour_sz: u32 = 0xB0;
                    let render_info_sz: u32 = if rev_num >= 5 { 0x10 } else { 0 };

                    let mempools_off = 0x40usize;
                    let voices_off = mempools_off + mempools_sz as usize;
                    let effects_off = voices_off + voices_sz as usize;
                    let sinks_off = effects_off + effects_sz as usize;
                    let perf_off = sinks_off + sinks_sz as usize;
                    let behaviour_off = perf_off + perf_sz as usize;
                    let render_info_off = behaviour_off + behaviour_sz as usize;
                    let total_size: u32 = render_info_off as u32 + render_info_sz;
                    let mut out = vec![0u8; total_size as usize];
                    out[0x00..0x04].copy_from_slice(&revision.to_le_bytes());
                    out[0x04..0x08].copy_from_slice(&behaviour_sz.to_le_bytes());
                    out[0x08..0x0C].copy_from_slice(&mempools_sz.to_le_bytes());
                    out[0x0C..0x10].copy_from_slice(&voices_sz.to_le_bytes());
                    out[0x14..0x18].copy_from_slice(&effects_sz.to_le_bytes());
                    out[0x1C..0x20].copy_from_slice(&sinks_sz.to_le_bytes());
                    out[0x20..0x24].copy_from_slice(&perf_sz.to_le_bytes());
                    out[0x28..0x2C].copy_from_slice(&render_info_sz.to_le_bytes());
                    out[0x3C..0x40].copy_from_slice(&total_size.to_le_bytes());

                    for (i, &in_state) in mempool_in_states.iter().enumerate() {
                        let new_state: u32 = match in_state {
                            4 => 5,
                            2 => 3,
                            s => s,
                        };
                        let off = mempools_off + i * 0x10;
                        out[off..off + 4].copy_from_slice(&new_state.to_le_bytes());
                    }

                    for vid in 0..voice_count_seen {
                        let off = voices_off + vid * 0x10;
                        let played = st.voice_played_samples.get(vid).copied().unwrap_or(0);
                        let consumed = st.voice_wbufs_consumed.get(vid).copied().unwrap_or(0);
                        out[off..off + 8].copy_from_slice(&played.to_le_bytes());
                        out[off + 8..off + 12].copy_from_slice(&consumed.to_le_bytes());
                    }
                    for (i, &state) in effect_out_states.iter().enumerate() {
                        let off = effects_off + i * effect_status_size as usize;
                        out[off] = state;
                    }
                    if render_info_sz != 0 {
                        out[render_info_off..render_info_off + 8]
                            .copy_from_slice(&frame.to_le_bytes());
                    }

                    let n = out.len().min(ob.size as usize);
                    let _ = kernel.address_space.write(ob.addr, &out[..n]);
                }
                if let Some(pb) = perf_buf {
                    let zero = vec![0u8; (pb.size as usize).min(0x100)];
                    let _ = kernel.address_space.write(pb.addr, &zero);
                }

                if std::env::var("NEXIUM_AUDIO_TEST_TONE").ok().as_deref() == Some("1") {
                    let base_phase = (frame as f32) * (TARGET_FRAMES as f32);
                    let phase_inc = std::f32::consts::TAU * 440.0 / TARGET_SR;
                    let total_frames = big_out.len() / 2;
                    for i in 0..total_frames {
                        let s = (((base_phase + i as f32) * phase_inc).sin()) * 0.25;
                        big_out[i * 2] = s;
                        big_out[i * 2 + 1] = s;
                    }
                }

                if let Some(sink) = crate::audio_sink::host_audio_sink() {
                    let pushed = sink.push_stereo_f32(&big_out);
                    let mix_peak = big_out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
                    use std::sync::atomic::{AtomicBool, Ordering as O};
                    static FIRST_PUSH: AtomicBool = AtomicBool::new(false);
                    static FIRST_NONZERO: AtomicBool = AtomicBool::new(false);
                    if !FIRST_PUSH.swap(true, O::Relaxed) {
                        log::info!(
                            "audio: first push to sink — pushed {} frames of {} mix_peak={:.4} (frame {})",
                            pushed,
                            TARGET_FRAMES,
                            mix_peak,
                            frame
                        );
                    }
                    if mix_peak > 0.001 && !FIRST_NONZERO.swap(true, O::Relaxed) {
                        log::info!(
                            "audio: FIRST NON-ZERO MIX — peak={:.4} pushed={}/{} (frame {})",
                            mix_peak,
                            pushed,
                            TARGET_FRAMES,
                            frame
                        );
                    }
                }

                log::trace!(
                    "IAudioRenderer.RequestUpdate{} in={:?} out={:?} perf={:?} mempools={} voices={} frame={}",
                    if cmd_id == 10 { "Auto" } else { "" },
                    in_buf.map(|b| b.size),
                    out_buf.map(|b| b.size),
                    perf_buf.map(|b| b.size),
                    mempool_count,
                    voice_count_seen,
                    frame
                );
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            5 => {
                st.state = 0;
                log::info!("IAudioRenderer.Start");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            6 => {
                st.state = 1;
                log::info!("IAudioRenderer.Stop");
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            7 => {
                let event_handle = if let Some(&h) = kernel.audio_renderer_events.get(&key) {
                    h
                } else {
                    let h = kernel.handles.create_handle(HandleType::Event);
                    kernel.event_signals.insert(h, false);
                    kernel.audio_renderer_events.insert(key, h);
                    log::info!(
                        "IAudioRenderer.QuerySystemEvent → new event handle={:#x}",
                        h
                    );
                    h
                };
                return build_ipc_response_copy(ctx, 0, &[], &[event_handle]);
            }
            8 => {
                let in_off = ctx.cmif_in_data_off;
                if ctx.cmif_in_data_len >= 4 {
                    st.rendering_time_limit = u32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            9 => {
                let v = st.rendering_time_limit;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            12 => {
                let in_off = ctx.cmif_in_data_off;
                if ctx.cmif_in_data_len >= 4 {
                    st.voice_drop_param = f32::from_le_bytes([
                        ctx.buf[in_off],
                        ctx.buf[in_off + 1],
                        ctx.buf[in_off + 2],
                        ctx.buf[in_off + 3],
                    ]);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            13 => {
                let v = st.voice_drop_param;
                return build_ipc_response(ctx, 0, &v.to_le_bytes(), &[]);
            }
            other => {
                log::warn!("IAudioRenderer.cmd_{} UNHANDLED → empty SUCCESS", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "IAudioDevice" {
        match cmd_id {
            0 | 6 | 14 => {
                let buf = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(b) = buf {
                    let mut name = vec![0u8; (b.size as usize).min(0x100)];
                    let bytes = b"AudioTvOutput";
                    let n = bytes.len().min(name.len());
                    name[..n].copy_from_slice(&bytes[..n]);
                    let _ = kernel.address_space.write(b.addr, &name);
                }
                let count: u32 = 1;
                return build_ipc_response(ctx, 0, &count.to_le_bytes(), &[]);
            }
            1 | 7 => {
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            2 | 8 => {
                let vol: f32 = 1.0;
                return build_ipc_response(ctx, 0, &vol.to_le_bytes(), &[]);
            }
            3 | 10 | 13 => {
                let buf = ctx
                    .recv_buffers
                    .iter()
                    .find(|b| b.size > 0 && b.addr != 0)
                    .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
                    .copied();
                if let Some(b) = buf {
                    let mut name = vec![0u8; (b.size as usize).min(0x100)];
                    let bytes = b"AudioTvOutput";
                    let n = bytes.len().min(name.len());
                    name[..n].copy_from_slice(&bytes[..n]);
                    let _ = kernel.address_space.write(b.addr, &name);
                }
                return build_ipc_response(ctx, 0, &[], &[]);
            }
            4 | 11 | 12 => {
                let h = kernel.handles.create_handle(HandleType::Event);
                kernel.event_signals.insert(h, true);
                return build_ipc_response(ctx, 0, &[], &[h]);
            }
            5 => {
                let ch: u32 = 2;
                return build_ipc_response(ctx, 0, &ch.to_le_bytes(), &[]);
            }
            other => {
                log::debug!("IAudioDevice.cmd_{} → empty SUCCESS", other);
                return build_ipc_response(ctx, 0, &[], &[]);
            }
        }
    }

    if port_name == "audout:u" && cmd_id == 1 {
        let name_buf = ctx
            .recv_statics
            .iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
            .copied();
        if let Some(buf) = name_buf {
            let cap = (buf.size as usize).min(0x100);
            let mut name = vec![0u8; cap];
            let bytes = b"DeviceOut";
            let n = bytes.len().min(cap);
            name[..n].copy_from_slice(&bytes[..n]);
            let _ = kernel.address_space.write(buf.addr, &name);
        }

        let in_off = ctx.cmif_in_data_off;
        let in_avail = ctx.cmif_in_data_len as usize;
        let sample_rate = if in_avail >= 4 {
            u32::from_le_bytes([
                ctx.buf[in_off],
                ctx.buf[in_off + 1],
                ctx.buf[in_off + 2],
                ctx.buf[in_off + 3],
            ])
        } else {
            0
        };
        let channel_count = if in_avail >= 6 {
            u16::from_le_bytes([ctx.buf[in_off + 4], ctx.buf[in_off + 5]])
        } else {
            0
        };
        let effective_rate = if sample_rate == 0 { 48000 } else { sample_rate };
        let effective_channels: u32 = if channel_count == 0 {
            2
        } else {
            channel_count as u32
        };

        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&effective_rate.to_le_bytes());
        out.extend_from_slice(&effective_channels.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());

        let is_domain = kernel
            .sessions
            .get(&session_handle)
            .map(|s| s.is_domain)
            .unwrap_or(false);
        log::info!(
            "audout:u OpenAudioOut sample_rate={} channels={} → IAudioOut (domain={})",
            effective_rate,
            effective_channels,
            is_domain
        );
        if is_domain {
            let object_id = alloc_domain_object(kernel, session_handle, "IAudioOut");
            return build_ipc_response_full(ctx, 0, &out, &[], &[], &[object_id]);
        } else {
            let h = kernel.handles.create_handle(HandleType::Session);
            let session = Session::new(h, "IAudioOut".to_string());
            kernel.sessions.insert(h, session);
            return build_ipc_response(ctx, 0, &out, &[h]);
        }
    }

    if port_name == "set" || port_name == "set:sys" {
        if let Some(outcome) = cmif_dispatch_set(kernel, ctx) {
            log::debug!(
                "set.cmd_{} → {} bytes (rc={:#x}) via #[service]",
                cmd_id,
                outcome.inline_out.len(),
                outcome.result
            );
            return build_ipc_response(ctx, outcome.result, &outcome.inline_out, &[]);
        }
    }

    if let Some(resp) =
        crate::services::generated::dispatch_generated(kernel, port_name, ctx, session_handle)
    {
        return resp;
    }

    if port_name == "fsp-srv" && cmd_id == 203 {
        log::debug!(
            "fsp-srv.OpenPatchDataStorageByCurrentProcess → ResultTargetNotFound (no patch)"
        );
        return build_ipc_response(ctx, 0x7D402, &[], &[]);
    }

    if port_name == "fsp-srv" && cmd_id == 1005 {
        let mode: u32 = 0;
        return build_ipc_response(ctx, 0, &mode.to_le_bytes(), &[]);
    }

    if let Some((data, handle_opt)) = applet_command_response(kernel, port_name, cmd_id) {
        log::debug!(
            "{}.cmd_{} → returning data ({} bytes, handle={:?})",
            port_name,
            cmd_id,
            data.len(),
            handle_opt
        );
        let handles: Vec<u32> = handle_opt.into_iter().collect();
        return build_ipc_response(ctx, 0, &data, &handles);
    }

    if matches!(port_name, "time:u" | "time:s" | "time:a" | "time:r") && cmd_id == 20 {
        let h = kernel.ensure_time_shmem_handle();
        log::info!("time:u GetSharedMemoryNativeHandle → handle={:#x}", h);
        return build_ipc_response(ctx, 0, &[], &[h]);
    }

    if port_name == "IAppletResource" && cmd_id == 0 {
        let h = kernel.handles.create_handle(HandleType::SharedMemory);
        log::info!(
            "IAppletResource.GetSharedMemoryHandle → hid_shmem_handle={:#x}",
            h
        );
        return build_ipc_response_copy(ctx, 0, &[], &[h]);
    }

    log::warn!(
        "dispatch_service_v2: {} cmd_{} FELL THROUGH to legacy dispatch_service (probably needs a real handler)",
        port_name,
        cmd_id
    );
    let tls_snapshot = ctx.buf.clone();
    let mut svc_ctx = crate::services::IpcCtx {
        tls_buf: &tls_snapshot,
        pending_frames,
    };
    let (result, out_data) = kernel
        .services
        .dispatch_service(port_name, cmd_id, &mut svc_ctx);
    build_ipc_response(ctx, result, &out_data, &[])
}

const IGBP_REQUEST_BUFFER: u32 = 1;
const IGBP_SET_BUFFER_COUNT: u32 = 2;
const IGBP_DEQUEUE_BUFFER: u32 = 3;
const IGBP_DETACH_BUFFER: u32 = 4;
const IGBP_DETACH_NEXT_BUFFER: u32 = 5;
const IGBP_ATTACH_BUFFER: u32 = 6;
const IGBP_QUEUE_BUFFER: u32 = 7;
const IGBP_CANCEL_BUFFER: u32 = 8;
const IGBP_QUERY: u32 = 9;
const IGBP_CONNECT: u32 = 10;
const IGBP_DISCONNECT: u32 = 11;
const IGBP_ALLOCATE_BUFFERS: u32 = 13;
const IGBP_SET_PREALLOCATED_BUFFER: u32 = 14;

fn handle_binder_transact(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    _session_handle: u32,
) -> Vec<u8> {
    let cmd_id = ctx.cmif_in.cmd_id;
    let (binder_id, code) = if ctx.cmif_in_data_len >= 8 {
        let off = ctx.cmif_in_data_off;
        let bid = i32::from_le_bytes([
            ctx.buf[off],
            ctx.buf[off + 1],
            ctx.buf[off + 2],
            ctx.buf[off + 3],
        ]);
        let c = u32::from_le_bytes([
            ctx.buf[off + 4],
            ctx.buf[off + 5],
            ctx.buf[off + 6],
            ctx.buf[off + 7],
        ]);
        (bid as u32, c)
    } else {
        (0u32, 0u32)
    };

    let mut in_parcel: Vec<u8> = Vec::new();
    let in_src = ctx
        .send_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .copied()
        .or_else(|| {
            ctx.send_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
        });
    if let Some(sb) = in_src {
        in_parcel.resize(sb.size as usize, 0);
        let _ = kernel.address_space.read(sb.addr, &mut in_parcel);
    }

    let reply = igbp_handle_transact(kernel, binder_id, code, &in_parcel);

    log::trace!(
        "IHOSBinderDriver.TransactParcel{} binder={} code={} in_size={} reply_size={}",
        if cmd_id == 3 { "Auto" } else { "" },
        binder_id,
        code,
        in_parcel.len(),
        reply.len()
    );

    if code == IGBP_REQUEST_BUFFER || code == IGBP_DEQUEUE_BUFFER {
        let preview = &in_parcel[..in_parcel.len().min(64)];
        log::trace!(
            "IGBP in code={} (len={}): {:02x?}",
            code,
            in_parcel.len(),
            preview
        );
        let preview = &reply[..reply.len().min(96)];
        log::trace!(
            "IGBP reply code={} (len={}): {:02x?}",
            code,
            reply.len(),
            preview
        );
    }

    let out_dst = ctx
        .recv_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .copied()
        .or_else(|| {
            ctx.recv_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
        });
    if let Some(rb) = out_dst {
        let n = reply.len().min(rb.size as usize);
        match kernel.address_space.write(rb.addr, &reply[..n]) {
            Ok(()) => {}
            Err(e) => log::error!(
                "binder reply write FAILED to {:#x} ({} bytes): {:?}",
                rb.addr,
                n,
                e
            ),
        }
    } else {
        log::warn!(
            "binder transact code={} produced {}-byte reply but no recv buffer descriptor",
            code,
            reply.len()
        );
    }

    build_ipc_response(ctx, 0, &[], &[])
}

fn igbp_handle_transact(
    kernel: &mut Kernel,
    binder_id: u32,
    code: u32,
    in_parcel: &[u8],
) -> Vec<u8> {
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
                log::warn!(
                    "IGBP::SetPreallocatedBuffer slot={} has=0 — no buffer",
                    slot
                );
                return ParcelBuilder::new().finish();
            }
            let mut gb = parse_flattened_graphic_buffer(&mut reader);
            if let Some(ref mut g) = gb {
                if g.nvmap_id == 0 && g.kind == 254 {
                    let tiled_size = compute_tiled_size(g.stride, g.height, g.block_height_log2);
                    let needed = (g.buffer_offset as usize).saturating_add(tiled_size);
                    let pick = kernel
                        .nvdrv
                        .nvmap_handles
                        .iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) >= needed)
                        .min_by_key(|(_, h)| h.size as usize)
                        .map(|(id, _)| *id);
                    if let Some(id) = pick {
                        g.nvmap_id = id;
                        log::info!(
                            "SetPreallocatedBuffer fixup: nvmap_id=0 → {} (off={:#x} tiled_size={:#x} needed={:#x})",
                            id,
                            g.buffer_offset,
                            tiled_size,
                            needed
                        );
                    }
                }
            }
            let parsed = gb.is_some();
            let (nvmap_id, w, h, off) = gb
                .as_ref()
                .map(|g| (g.nvmap_id, g.width, g.height, g.buffer_offset))
                .unwrap_or((0, 0, 0, 0));
            kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                if let Some(gb) = gb {
                    bq.set_preallocated(slot, gb);
                }
            });
            log::info!(
                "IGBP::SetPreallocatedBuffer binder={} slot={} parsed={} nvmap_id={} {}x{} off={:#x}",
                binder_id,
                slot,
                parsed,
                nvmap_id,
                w,
                h,
                off
            );
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_REQUEST_BUFFER => {
            kernel
                .nvdrv
                .stats
                .request_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let gb = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.request_buffer(slot).cloned());
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
            kernel
                .nvdrv
                .stats
                .dequeue_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _async_ = reader.read_i32();
            let _w = reader.read_u32();
            let _h = reader.read_u32();
            let _fmt = reader.read_i32();
            let _usage = reader.read_u32();
            let (slot, free, deq, queued) = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                let s = bq.dequeue();
                (s, bq.free.len(), bq.dequeued.len(), bq.queued.len())
            });
            log::trace!(
                "IGBP::DequeueBuffer binder={} → slot={} (free={} deq={} queued={})",
                binder_id,
                slot,
                free,
                deq,
                queued
            );
            let mut p = ParcelBuilder::new();
            p.write_u32(slot);
            p.write_u32(1);
            p.write_flattened_zero_fence();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUEUE_BUFFER => {
            kernel
                .nvdrv
                .stats
                .queue_buffer_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            let _has = reader.read_u32();
            let _timestamp = reader.read_u64();
            let _is_auto = reader.read_i32();
            let _crop_l = reader.read_i32();
            let _crop_t = reader.read_i32();
            let _crop_r = reader.read_i32();
            let _crop_b = reader.read_i32();
            let _scaling = reader.read_i32();
            let transform = reader.read_i32().unwrap_or(0) as u32;
            let _sticky = reader.read_u32();
            let _async = reader.read_i32();
            let swap_interval = reader.read_i32().unwrap_or(1).max(1);

            let gb_opt = kernel.nvdrv.with_bufferqueue(binder_id, |bq| {
                bq.queue(slot);
                let r = bq.request_buffer(slot).cloned();
                let slot_count = bq.slots.len();
                let has_buf = bq
                    .slots
                    .get(slot as usize)
                    .and_then(|s| s.buffer.as_ref())
                    .is_some();
                (r, slot_count, has_buf)
            });
            let (gb_opt, slot_count, has_buf) = gb_opt;
            log::trace!(
                "IGBP::QueueBuffer binder={} slot={} swap_interval={} transform={:#x} slot_count={} has_buf={} gb_some={}",
                binder_id,
                slot,
                swap_interval,
                transform,
                slot_count,
                has_buf,
                gb_opt.is_some()
            );
            if transform != 0 && std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                log::warn!(
                    "IGBP::QueueBuffer transform binder={} slot={} transform={:#x}",
                    binder_id,
                    slot,
                    transform
                );
            }

            if let Some(gb) = gb_opt {
                let bpp: usize = 4;
                let linear_size = (gb.stride as usize) * (gb.height as usize) * bpp;
                let tiled_size = compute_tiled_size(gb.stride, gb.height, gb.block_height_log2);
                let resolved: Option<(u64, bool)> = if let Some(nvmap) =
                    kernel.nvdrv.nvmap_handles.get(&gb.nvmap_id)
                {
                    let actual_size = nvmap.size as usize;
                    let is_tiled =
                        actual_size >= tiled_size && gb.kind == 254 && gb.block_height_log2 != 0;
                    log::trace!(
                        "QueueBuffer fast-path: nvmap_id={} addr={:#x} off={:#x} size={:#x} kind={} bh_log2={} tiled={} (slot={})",
                        gb.nvmap_id,
                        nvmap.address,
                        gb.buffer_offset,
                        actual_size,
                        gb.kind,
                        gb.block_height_log2,
                        is_tiled,
                        slot
                    );
                    Some((nvmap.address.wrapping_add(gb.buffer_offset), is_tiled))
                } else {
                    let mut candidates: Vec<(u32, u64, u32)> = kernel
                        .nvdrv
                        .nvmap_handles
                        .iter()
                        .filter(|(_, h)| h.address != 0 && (h.size as usize) == linear_size)
                        .map(|(id, h)| (*id, h.address, h.size))
                        .collect();
                    candidates.sort_by_key(|(id, _, _)| *id);
                    if let Some(&(id, addr, size)) = candidates.last() {
                        log::info!(
                            "QueueBuffer fallback pick newest: nvmap_id={} addr={:#x} size={:#x} (slot={} candidates={})",
                            id,
                            addr,
                            size,
                            slot,
                            candidates.len()
                        );
                        Some((addr, false))
                    } else {
                        log::warn!(
                            "QueueBuffer fallback: no exact-size candidate (linear_size={:#x} slot={} total_handles={})",
                            linear_size,
                            slot,
                            kernel.nvdrv.nvmap_handles.len()
                        );
                        None
                    }
                };
                if let Some(r_async) = kernel.nvdrv.renderer().cloned() {
                    let rt_worker = nexium_nvdrv::render_thread::present_thread();
                    let fq = kernel.nvdrv.frame_queue.clone();
                    let qba = kernel.nvdrv.queue_buffer_active.clone();
                    let stats = kernel.nvdrv.stats.clone();
                    let (pw, ph, pnv) = (gb.width, gb.height, gb.nvmap_id);
                    qba.store(true, std::sync::atomic::Ordering::Relaxed);
                    let present_profile = std::env::var_os("NEXIUM_NVDRV_PROFILE").is_some();
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static PRESENT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_ENQUEUED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_DROPPED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_EXECUTED: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_READY: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_EMPTY: AtomicU64 = AtomicU64::new(0);
                    static PRESENT_NS: AtomicU64 = AtomicU64::new(0);
                    let attempt = if present_profile {
                        PRESENT_ATTEMPTS.fetch_add(1, Ordering::Relaxed) + 1
                    } else {
                        0
                    };
                    let submitted = rt_worker.try_submit(Box::new(move || {
                        let t0 = std::time::Instant::now();
                        let crop = cached_present_crop(pw, ph);
                        let read_rect = crop.map(|(x0, y0, w, h)| {
                            [x0, ph.saturating_sub(y0).saturating_sub(h), w, h]
                        });
                        if let Some((read_w, read_h, bytes)) =
                            r_async.readback_target_pipelined(pnv, pw, ph, read_rect)
                        {
                            let (present_w, present_h, bytes) =
                                prepare_vulkan_present_frame(bytes, read_w, read_h, transform);
                            dump_present_frame(&bytes, present_w, present_h);
                            if std::env::var_os("NEXIUM_FRAME_PRESENT_CACHE").is_some() {
                                nexium_common::frame_present::set_last_presented(
                                    present_w,
                                    present_h,
                                    bytes.clone(),
                                );
                            }
                            fq.lock().push(nexium_nvdrv::QueuedFrame {
                                width: present_w,
                                height: present_h,
                                pixels: bytes,
                            });
                            stats
                                .frames_submitted
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if present_profile {
                                PRESENT_READY.fetch_add(1, Ordering::Relaxed);
                            }
                        } else if present_profile {
                            PRESENT_EMPTY.fetch_add(1, Ordering::Relaxed);
                        }
                        if present_profile {
                            let elapsed = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                            PRESENT_NS.fetch_add(elapsed, Ordering::Relaxed);
                            let exec = PRESENT_EXECUTED.fetch_add(1, Ordering::Relaxed) + 1;
                            if exec % 60 == 0 {
                                let total_ns = PRESENT_NS.load(Ordering::Relaxed);
                                log::warn!(
                                    "[nvprof] present_exec executed={} ready={} empty={} avg_ms={:.3}",
                                    exec,
                                    PRESENT_READY.load(Ordering::Relaxed),
                                    PRESENT_EMPTY.load(Ordering::Relaxed),
                                    total_ns as f64 / exec as f64 / 1_000_000.0
                                );
                            }
                        }
                    }));
                    if present_profile {
                        if submitted {
                            PRESENT_ENQUEUED.fetch_add(1, Ordering::Relaxed);
                        } else {
                            PRESENT_DROPPED.fetch_add(1, Ordering::Relaxed);
                        }
                        if attempt % 60 == 0 {
                            log::warn!(
                                "[nvprof] present_enqueue attempts={} enqueued={} dropped={}",
                                attempt,
                                PRESENT_ENQUEUED.load(Ordering::Relaxed),
                                PRESENT_DROPPED.load(Ordering::Relaxed)
                            );
                        }
                    }
                } else if let Some((addr, is_tiled)) = resolved {
                    let read_size = if is_tiled { tiled_size } else { linear_size };
                    let mut raw = vec![0u8; read_size];
                    let mut effective_tiled = is_tiled;
                    let mut effective_addr = addr;
                    let mut effective_bh_log2 = gb.block_height_log2;
                    if kernel.address_space.read(addr, &mut raw).is_ok() {
                        let nonzero = raw.iter().filter(|&&b| b != 0).count();
                        if nonzero > 0 && slot < 2 {
                            log::info!(
                                "QueueBuffer slot={} addr={:#x} nonzero_bytes={}/{} first16={:02x?}",
                                slot,
                                addr,
                                nonzero,
                                read_size,
                                &raw[..16.min(raw.len())]
                            );
                        }
                        if is_tiled && raw.iter().all(|&b| b == 0) {
                            let (tiled_rt_cpu, dma_bh_log2, dma_stride, dma_height) = {
                                let dma = kernel.nvdrv.gpu.maxwell_dma.lock();
                                (
                                    dma.last_tiled_dst_cpu,
                                    dma.last_tiled_dst_bh_log2,
                                    dma.last_tiled_dst_stride,
                                    dma.last_tiled_dst_height,
                                )
                            };
                            let mut found = false;
                            if tiled_rt_cpu != 0 {
                                let dma_tiled_size = compute_tiled_size(
                                    dma_stride.max(gb.stride),
                                    dma_height.max(gb.height),
                                    dma_bh_log2,
                                );
                                let mut tiled_raw = vec![0u8; dma_tiled_size.max(tiled_size)];
                                if kernel
                                    .address_space
                                    .read(tiled_rt_cpu, &mut tiled_raw)
                                    .is_ok()
                                    && tiled_raw.iter().any(|&b| b != 0)
                                {
                                    log::info!(
                                        "QueueBuffer tiled-rt-redirect: slot={} slot_tiled={:#x} → rt_cpu={:#x} (gralloc_bh={} dma_bh={} dma_stride={} dma_h={})",
                                        slot,
                                        addr,
                                        tiled_rt_cpu,
                                        gb.block_height_log2,
                                        dma_bh_log2,
                                        dma_stride,
                                        dma_height
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
                                    if h.address == 0 || (h.size as usize) != linear_size {
                                        continue;
                                    }
                                    match best {
                                        None => {
                                            best = Some((*id, h.address));
                                        }
                                        Some((best_id, _)) if *id > best_id => {
                                            best = Some((*id, h.address));
                                        }
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
                                            id,
                                            lin_addr,
                                            slot
                                        );
                                        raw = lin_raw;
                                        effective_tiled = false;
                                        effective_addr = lin_addr;
                                    }
                                }
                            }
                        }
                        let fermi_frame = kernel.nvdrv.drain_fermi2d_frame();
                        if let Some(qf) = fermi_frame.as_ref() {
                            log::info!(
                                "QueueBuffer Fermi2D-captured frame: {}x{} ({} bytes)",
                                qf.width,
                                qf.height,
                                qf.pixels.len()
                            );
                        }
                        let vk_readback = kernel
                            .nvdrv
                            .renderer()
                            .and_then(|r| r.readback_target(gb.nvmap_id, gb.width, gb.height))
                            .map(|mut bytes| {
                                let row = (gb.width as usize) * 4;
                                let h = gb.height as usize;
                                if bytes.len() >= row * h {
                                    for y in 0..h / 2 {
                                        let top = y * row;
                                        let bot = (h - 1 - y) * row;
                                        let (a, b) = bytes.split_at_mut(bot);
                                        a[top..top + row].swap_with_slice(&mut b[..row]);
                                    }
                                }
                                let (present_w, present_h, mut bytes) =
                                    maybe_crop_present_subwindow(bytes, gb.width, gb.height);
                                make_present_opaque(&mut bytes);
                                dump_present_frame(&bytes, present_w, present_h);
                                (present_w, present_h, bytes)
                            });
                        let have_gpu_frame = fermi_frame.is_some() || vk_readback.is_some();
                        let legacy_gfx = kernel
                            .nvdrv
                            .legacy_gfx
                            .load(std::sync::atomic::Ordering::Relaxed);
                        let (pixels, rgb_nz) = if have_gpu_frame {
                            (Vec::new(), 0usize)
                        } else {
                            let mut pixels = if effective_tiled {
                                unswizzle_block_linear(
                                    &raw,
                                    gb.stride,
                                    gb.height,
                                    bpp,
                                    effective_bh_log2,
                                )
                            } else {
                                raw
                            };
                            if pixels.len() < linear_size {
                                pixels.resize(linear_size, 0);
                            }
                            if legacy_gfx {
                                let row_bytes = (gb.width * (bpp as u32)) as usize;
                                let h = gb.height as usize;
                                for y in 0..h / 2 {
                                    let top = y * row_bytes;
                                    let bot = (h - 1 - y) * row_bytes;
                                    if bot + row_bytes <= pixels.len() {
                                        let (a, b) = pixels.split_at_mut(bot);
                                        a[top..top + row_bytes]
                                            .swap_with_slice(&mut b[..row_bytes]);
                                    }
                                }
                            }
                            for px in pixels.chunks_exact_mut(4) {
                                px[3] = 0xFF;
                            }
                            let rgb_nz = pixels
                                .chunks_exact(4)
                                .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                                .count();
                            (pixels, rgb_nz)
                        };
                        let (frame_w, frame_h, mut frame_pixels) = if let Some(qf) = fermi_frame {
                            (qf.width, qf.height, qf.pixels)
                        } else if let Some((w, h, bytes)) = vk_readback {
                            nexium_common::frame_present::set_last_presented(w, h, bytes.clone());
                            (w, h, bytes)
                        } else if rgb_nz >= 16 {
                            if legacy_gfx {
                                if let Some((x0, y0, w, h)) =
                                    active_bbox(&pixels, gb.width, gb.height)
                                {
                                    let area_ratio = (w as f32 * h as f32)
                                        / (gb.width as f32 * gb.height as f32);
                                    if area_ratio < 0.65 && w >= 64 && h >= 64 {
                                        let upscaled = crop_and_upscale(
                                            &pixels, gb.width, x0, y0, w, h, gb.width, gb.height,
                                        );
                                        log::info!(
                                            "QueueBuffer legacy_gfx sub-window: src=({},{}) {}x{} → upscale to {}x{}",
                                            x0,
                                            y0,
                                            w,
                                            h,
                                            gb.width,
                                            gb.height
                                        );
                                        (gb.width, gb.height, upscaled)
                                    } else {
                                        (gb.width, gb.height, pixels)
                                    }
                                } else {
                                    (gb.width, gb.height, pixels)
                                }
                            } else {
                                (gb.width, gb.height, pixels)
                            }
                        } else if let Some((w, h, sdl_pixels)) =
                            try_compose_from_sdl_surface(kernel, gb.width, gb.height)
                        {
                            log::info!(
                                "QueueBuffer SDL_Surface fallback: {}x{} (back buffer had only {} nonzero RGB pixels)",
                                w,
                                h,
                                rgb_nz
                            );
                            (w, h, sdl_pixels)
                        } else if legacy_gfx {
                            if let Some(renderer) = kernel.nvdrv.renderer() {
                                let r = renderer.clone();
                                let mut color = kernel.nvdrv.last_clear_color();
                                if color[3] < 0.5 {
                                    color[3] = 1.0;
                                }
                                let clears = kernel.nvdrv.last_clear_count();
                                if r.clear_target(gb.nvmap_id, gb.width, gb.height, color)
                                    .is_ok()
                                {
                                    if let Some(bytes) =
                                        r.readback_target(gb.nvmap_id, gb.width, gb.height)
                                    {
                                        log::info!(
                                            "QueueBuffer legacy_gfx Vulkan clear-only fallback: {}x{} color=[{:.2},{:.2},{:.2},{:.2}] clears={} → {} bytes",
                                            gb.width,
                                            gb.height,
                                            color[0],
                                            color[1],
                                            color[2],
                                            color[3],
                                            clears,
                                            bytes.len()
                                        );
                                        (gb.width, gb.height, bytes)
                                    } else {
                                        (gb.width, gb.height, pixels)
                                    }
                                } else {
                                    (gb.width, gb.height, pixels)
                                }
                            } else {
                                (gb.width, gb.height, pixels)
                            }
                        } else {
                            (gb.width, gb.height, pixels)
                        };
                        make_present_opaque(&mut frame_pixels);
                        let nz = frame_pixels.iter().filter(|b| **b != 0).count();
                        let rgb_nz = frame_pixels
                            .chunks_exact(4)
                            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                            .count();
                        let checksum: u32 = frame_pixels
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .fold(0u32, |a, b| a.wrapping_add(b));
                        log::trace!(
                            "QueueBuffer submit slot={} parsed_nvmap_id={} addr={:#x} {}x{} tiled={} nz={} rgb_nz={} cksum={:#x}",
                            slot,
                            gb.nvmap_id,
                            effective_addr,
                            frame_w,
                            frame_h,
                            effective_tiled,
                            nz,
                            rgb_nz,
                            checksum
                        );
                        {
                            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                            static FRAME_SEQ: AtomicU64 = AtomicU64::new(0);
                            static FIRST_NONBLACK: AtomicBool = AtomicBool::new(false);
                            static LAST_RGB_NZ: AtomicU64 = AtomicU64::new(0);
                            let seq = FRAME_SEQ.fetch_add(1, Ordering::Relaxed);
                            let is_first_nonblack =
                                rgb_nz > 0 && !FIRST_NONBLACK.swap(true, Ordering::Relaxed);
                            let should_dump = nexium_common::dumps::enabled()
                                && ((seq > 0 && seq % 300 == 60) || is_first_nonblack);
                            if should_dump {
                                if let Some(home) = std::env::var_os("APPDATA") {
                                    let path = std::path::PathBuf::from(home)
                                        .join("NeXium")
                                        .join("logs")
                                        .join(format!("compose-{}.bmp", seq));
                                    let _ = save_rgba_bmp(&path, frame_w, frame_h, &frame_pixels);
                                    log::warn!(
                                        "FRAME DUMP seq={} rgb_nz={} → {}",
                                        seq,
                                        rgb_nz,
                                        path.display()
                                    );
                                }
                            }
                            if is_first_nonblack {
                                log::warn!("FIRST NON-BLACK FRAME seq={} rgb_nz={}", seq, rgb_nz);
                            }
                            if seq % 60 == 0 {
                                let prev = LAST_RGB_NZ.swap(rgb_nz as u64, Ordering::Relaxed);
                                if (prev == 0) != (rgb_nz == 0) {
                                    log::warn!(
                                        "frame heartbeat seq={} rgb_nz={} (was {})",
                                        seq,
                                        rgb_nz,
                                        prev
                                    );
                                }
                            }
                        }
                        kernel.nvdrv.submit_frame(nexium_nvdrv::QueuedFrame {
                            width: frame_w,
                            height: frame_h,
                            pixels: frame_pixels,
                        });
                    } else {
                        log::warn!(
                            "QueueBuffer: failed to read slot {} addr={:#x} read_size={:#x}",
                            slot,
                            addr,
                            read_size
                        );
                    }
                } else {
                    log::warn!(
                        "QueueBuffer: no nvmap candidate for size {} (slot {})",
                        linear_size,
                        slot
                    );
                }
            } else {
                log::warn!("QueueBuffer: slot {} has no GraphicBuffer", slot);
            }

            let _ = swap_interval;

            let (qw, qh) = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
            let mut p = ParcelBuilder::new();
            p.write_bq_buffer_output(qw, qh);
            p.write_u32(0);
            p.finish()
        }
        IGBP_CANCEL_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.cancel(slot));
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_QUERY => {
            let what = reader.read_i32().unwrap_or(0);
            let (w, h) = kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| (bq.width, bq.height));
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
        IGBP_SET_BUFFER_COUNT => {
            let count = reader.read_i32().unwrap_or(0);
            log::debug!("IGBP::SetBufferCount binder={} count={}", binder_id, count);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_DETACH_BUFFER => {
            let slot = reader.read_i32().unwrap_or(0).max(0) as u32;
            kernel
                .nvdrv
                .with_bufferqueue(binder_id, |bq| bq.cancel(slot));
            log::debug!("IGBP::DetachBuffer binder={} slot={}", binder_id, slot);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.finish()
        }
        IGBP_DETACH_NEXT_BUFFER => {
            log::debug!("IGBP::DetachNextBuffer binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.write_u32(0);
            p.write_u32(0);
            p.finish()
        }
        IGBP_ATTACH_BUFFER => {
            log::debug!("IGBP::AttachBuffer binder={}", binder_id);
            let mut p = ParcelBuilder::new();
            p.write_u32(0);
            p.write_u32(0);
            p.finish()
        }
        IGBP_ALLOCATE_BUFFERS => {
            let async_ = reader.read_i32().unwrap_or(0);
            log::debug!(
                "IGBP::AllocateBuffers binder={} async={}",
                binder_id,
                async_
            );
            let mut p = ParcelBuilder::new();
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
        } else {
            (0, 0, 0)
        };
        Self {
            data,
            payload_off,
            cursor: payload_off,
            objects_off,
            objects_size,
        }
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.cursor + 4 > self.data.len() {
            return None;
        }
        let v = u32::from_le_bytes([
            self.data[self.cursor],
            self.data[self.cursor + 1],
            self.data[self.cursor + 2],
            self.data[self.cursor + 3],
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
        if end > self.data.len() {
            return None;
        }
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
        Self {
            payload: Vec::new(),
        }
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
        ints[21] = gb.kind;
        ints[22] = gb.block_height_log2;
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

fn active_bbox(pixels: &[u8], width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let w = width as usize;
    let h = height as usize;
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut min_y = h;
    let mut max_y = 0usize;
    for y in 0..h {
        let row_off = y * w * 4;
        for x in 0..w {
            let p = &pixels[row_off + x * 4..row_off + x * 4 + 3];
            if p[0] != 0 || p[1] != 0 || p[2] != 0 {
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if max_x < min_x || max_y < min_y {
        return None;
    }
    Some((
        min_x as u32,
        min_y as u32,
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
    ))
}

fn crop_and_upscale(
    src: &[u8],
    src_stride_px: u32,
    sx: u32,
    sy: u32,
    sw: u32,
    sh: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    for dy in 0..dst_h {
        let yy = sy + dy * sh / dst_h;
        for dx in 0..dst_w {
            let xx = sx + dx * sw / dst_w;
            let s = ((yy * src_stride_px + xx) * 4) as usize;
            let d = ((dy * dst_w + dx) * 4) as usize;
            out[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    out
}

fn make_present_opaque(pixels: &mut [u8]) {
    for px in pixels.chunks_exact_mut(4) {
        px[3] = 0xFF;
    }
}

fn outside_crop_has_visible(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> bool {
    let row = width as usize * 4;
    let x1 = x0.saturating_add(w);
    let y1 = y0.saturating_add(h);
    let mut visible = 0u32;
    let mut min_rgb = [255u8; 3];
    let mut max_rgb = [0u8; 3];
    for y in (0..height).step_by(4) {
        let row_off = y as usize * row;
        for x in (0..width).step_by(4) {
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                continue;
            }
            let p = row_off + x as usize * 4;
            if p + 2 >= pixels.len() {
                continue;
            }
            let rgb = [pixels[p], pixels[p + 1], pixels[p + 2]];
            if rgb[0].max(rgb[1]).max(rgb[2]) > 4 {
                visible += 1;
                for i in 0..3 {
                    min_rgb[i] = min_rgb[i].min(rgb[i]);
                    max_rgb[i] = max_rgb[i].max(rgb[i]);
                }
                if visible >= 64 {
                    let range = (max_rgb[0] - min_rgb[0])
                        .max(max_rgb[1] - min_rgb[1])
                        .max(max_rgb[2] - min_rgb[2]);
                    let hi = max_rgb[0].max(max_rgb[1]).max(max_rgb[2]);
                    let lo = min_rgb[0].min(min_rgb[1]).min(min_rgb[2]);
                    if range > 24 {
                        return true;
                    }
                    if hi.saturating_sub(lo) > 48 {
                        return true;
                    }
                }
            }
        }
    }
    if visible >= 64 {
        let range = (max_rgb[0] - min_rgb[0])
            .max(max_rgb[1] - min_rgb[1])
            .max(max_rgb[2] - min_rgb[2]);
        let hi = max_rgb[0].max(max_rgb[1]).max(max_rgb[2]);
        let lo = min_rgb[0].min(min_rgb[1]).min(min_rgb[2]);
        if range > 24 {
            return true;
        }
        if hi.saturating_sub(lo) > 48 {
            return true;
        }
    }
    false
}

fn present_crop_slot() -> &'static std::sync::Mutex<Option<(u32, u32, u32, u32, u32, u32)>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<(u32, u32, u32, u32, u32, u32)>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

fn cached_present_crop(width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    present_crop_slot().lock().ok().and_then(|slot| {
        let (dst_w, dst_h, x0, y0, w, h) = (*slot)?;
        (dst_w == width && dst_h == height).then_some((x0, y0, w, h))
    })
}

fn crop_flipped_opaque(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let src_row = src_w as usize * 4;
    let dst_row = w as usize * 4;
    let mut out = vec![0u8; h as usize * dst_row];
    if src.len() < src_h as usize * src_row {
        return out;
    }
    for dy in 0..h as usize {
        let Some(src_y) = (src_h as usize).checked_sub(1 + y0 as usize + dy) else {
            continue;
        };
        let src_off = src_y
            .saturating_mul(src_row)
            .saturating_add(x0 as usize * 4);
        let dst_off = dy * dst_row;
        if src_off + dst_row > src.len() || dst_off + dst_row > out.len() {
            continue;
        }
        out[dst_off..dst_off + dst_row].copy_from_slice(&src[src_off..src_off + dst_row]);
        for px in out[dst_off..dst_off + dst_row].chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
    }
    out
}

fn prepare_vulkan_present_frame(
    mut bytes: Vec<u8>,
    width: u32,
    height: u32,
    transform: u32,
) -> (u32, u32, Vec<u8>) {
    if let Some((x0, y0, w, h)) = cached_present_crop(width, height) {
        let mut cropped = crop_flipped_opaque(&bytes, width, height, x0, y0, w, h);
        apply_present_transform(&mut cropped, w, h, transform);
        return (w, h, cropped);
    }
    if should_flip_vulkan_present(width, height) {
        flip_present_v(&mut bytes, width, height);
    }
    let (present_w, present_h, mut bytes) = maybe_crop_present_subwindow(bytes, width, height);
    apply_present_transform(&mut bytes, present_w, present_h, transform);
    make_present_opaque(&mut bytes);
    (present_w, present_h, bytes)
}

fn should_flip_vulkan_present(width: u32, height: u32) -> bool {
    !(width == 1600 && height == 900)
}

fn apply_present_transform(bytes: &mut [u8], width: u32, height: u32, transform: u32) {
    if transform & 0x1 != 0 {
        flip_present_h(bytes, width, height);
    }
    if transform & 0x2 != 0 {
        flip_present_v(bytes, width, height);
    }
}

fn flip_present_h(bytes: &mut [u8], width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let row = w * 4;
    if w == 0 || h == 0 || bytes.len() < row * h {
        return;
    }
    for y in 0..h {
        let base = y * row;
        for x in 0..w / 2 {
            let a = base + x * 4;
            let b = base + (w - 1 - x) * 4;
            for c in 0..4 {
                bytes.swap(a + c, b + c);
            }
        }
    }
}

fn flip_present_v(bytes: &mut [u8], width: u32, height: u32) {
    let row = width as usize * 4;
    let h = height as usize;
    if row == 0 || h == 0 || bytes.len() < row * h {
        return;
    }
    for y in 0..h / 2 {
        let top = y * row;
        let bot = (h - 1 - y) * row;
        let (a, b) = bytes.split_at_mut(bot);
        a[top..top + row].swap_with_slice(&mut b[..row]);
    }
}

fn maybe_crop_present_subwindow(bytes: Vec<u8>, width: u32, height: u32) -> (u32, u32, Vec<u8>) {
    if width < 1600 || height < 900 || bytes.len() < (width as usize) * (height as usize) * 4 {
        return (width, height, bytes);
    }
    if let Ok(mut slot) = present_crop_slot().lock() {
        if let Some((dst_w, dst_h, x0, y0, w, h)) = *slot {
            if dst_w == width && dst_h == height {
                if !outside_crop_has_visible(&bytes, width, height, x0, y0, w, h) {
                    return (w, h, crop_and_upscale(&bytes, width, x0, y0, w, h, w, h));
                }
                log::info!(
                    "QueueBuffer Vulkan sub-window invalidated: cached=({},{}) {}x{} target={}x{}",
                    x0,
                    y0,
                    w,
                    h,
                    width,
                    height
                );
                *slot = None;
            }
        }
    }
    let Some((x0, y0, w, h)) = active_bbox(&bytes, width, height) else {
        return (width, height, bytes);
    };
    let area_ratio = (w as f32 * h as f32) / (width as f32 * height as f32);
    let src_aspect = w as f32 / h as f32;
    let dst_aspect = width as f32 / height as f32;
    let aspect_delta = ((src_aspect / dst_aspect) - 1.0).abs();
    let inset = x0 > 4 || y0 > 4 || x0 + w + 4 < width || y0 + h + 4 < height;
    let anchored = x0 <= 4 || y0 <= 4 || x0 + w + 4 >= width || y0 + h + 4 >= height;
    if inset
        && anchored
        && w >= 640
        && h >= 360
        && area_ratio >= 0.30
        && area_ratio <= 0.80
        && aspect_delta <= 0.05
    {
        if let Ok(mut slot) = present_crop_slot().lock() {
            *slot = Some((width, height, x0, y0, w, h));
        }
        log::info!(
            "QueueBuffer Vulkan sub-window: src=({},{}) {}x{} -> crop",
            x0,
            y0,
            w,
            h
        );
        (w, h, crop_and_upscale(&bytes, width, x0, y0, w, h, w, h))
    } else {
        (width, height, bytes)
    }
}

fn dump_present_frame(bytes: &[u8], width: u32, height: u32) {
    if std::env::var("NEXIUM_PRESENT_DUMP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        if seq % 60 == 0 {
            if let Some(home) = std::env::var_os("APPDATA") {
                let path = std::path::PathBuf::from(home)
                    .join("NeXium")
                    .join("logs")
                    .join(format!("present-{}.bmp", seq));
                if save_rgba_bmp(&path, width, height, bytes).is_ok() {
                    log::warn!("PRESENT DUMP seq={} -> {}", seq, path.display());
                }
            }
        }
    }
}

fn save_rgba_bmp(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let row_bytes = (width as usize) * 3;
    let row_padded = (row_bytes + 3) & !3;
    let pixel_bytes = row_padded * height as usize;
    let file_size = 54 + pixel_bytes;
    let mut f = std::fs::File::create(path)?;
    let mut h = Vec::with_capacity(54);
    h.extend_from_slice(b"BM");
    h.extend_from_slice(&(file_size as u32).to_le_bytes());
    h.extend_from_slice(&[0u8; 4]);
    h.extend_from_slice(&54u32.to_le_bytes());
    h.extend_from_slice(&40u32.to_le_bytes());
    h.extend_from_slice(&width.to_le_bytes());
    h.extend_from_slice(&height.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&24u16.to_le_bytes());
    h.extend_from_slice(&[0u8; 24]);
    f.write_all(&h)?;
    let mut row = vec![0u8; row_padded];
    for y in (0..height as usize).rev() {
        let src_off = y * (width as usize) * 4;
        for x in 0..width as usize {
            let s = src_off + x * 4;
            row[x * 3] = rgba.get(s + 2).copied().unwrap_or(0);
            row[x * 3 + 1] = rgba.get(s + 1).copied().unwrap_or(0);
            row[x * 3 + 2] = rgba.get(s).copied().unwrap_or(0);
        }
        f.write_all(&row)?;
    }
    Ok(())
}

fn try_compose_from_sdl_surface(
    kernel: &Kernel,
    fb_width: u32,
    fb_height: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    const CANDIDATES: &[(u32, u32)] = &[
        (1280, 720),
        (640, 360),
        (854, 480),
        (1920, 1080),
        (1280, 768),
    ];
    for h in kernel.nvdrv.nvmap_handles.values() {
        if h.address == 0 {
            continue;
        }
        let Some(&(width, height)) = CANDIDATES
            .iter()
            .find(|(w, hh)| (*w as u64) * (*hh as u64) * 4 == h.size as u64)
        else {
            continue;
        };
        let mut linear = vec![0u8; h.size as usize];
        if kernel.address_space.read(h.address, &mut linear).is_err() {
            continue;
        }
        let nz = linear.iter().filter(|b| **b != 0).count();
        if nz < 256 {
            continue;
        }
        for px in linear.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
        log::info!(
            "compose: SDL_Surface candidate nvmap_id={} cpu={:#x} {}x{} nz={}",
            h.id,
            h.address,
            width,
            height,
            nz
        );
        if width == fb_width && height == fb_height {
            return Some((width, height, linear));
        }
        let dst_w = fb_width;
        let dst_h = fb_height;
        let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
        for dy in 0..dst_h {
            let sy = (dy as u64 * height as u64 / dst_h as u64) as u32;
            for dx in 0..dst_w {
                let sx = (dx as u64 * width as u64 / dst_w as u64) as u32;
                let s = ((sy * width + sx) * 4) as usize;
                let d = ((dy * dst_w + dx) * 4) as usize;
                out[d..d + 4].copy_from_slice(&linear[s..s + 4]);
            }
        }
        return Some((dst_w, dst_h, out));
    }
    None
}

fn parse_flattened_graphic_buffer(
    reader: &mut ParcelReader,
) -> Option<nexium_nvdrv::GraphicBuffer> {
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
        nvmap_id,
        binder_handle,
        inline_nvmap_id,
        buffer_offset,
        kind,
        block_height_log2
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
    log::trace!("nvdrv:{}.cmd_{}", port_name, cmd_id);

    match cmd_id {
        0 => {
            let buf_src = ctx
                .send_statics
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
                .or_else(|| {
                    ctx.send_buffers
                        .iter()
                        .find(|b| b.size > 0 && b.addr != 0)
                        .copied()
                });
            let path = if let Some(sb) = buf_src {
                let mut buf = vec![0u8; sb.size as usize];
                let _ = kernel.address_space.read(sb.addr, &mut buf);
                let trimmed = buf.split(|&b| b == 0).next().unwrap_or(&buf);
                String::from_utf8_lossy(trimmed).into_owned()
            } else {
                String::new()
            };
            log::info!(
                "nvdrv:Open path='{}' (sb={:?})",
                path,
                buf_src.map(|b| (b.addr, b.size))
            );
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
            } else {
                0
            };
            let ioctl_id = if ctx.cmif_in_data_len >= 8 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off + 4],
                    ctx.buf[ctx.cmif_in_data_off + 5],
                    ctx.buf[ctx.cmif_in_data_off + 6],
                    ctx.buf[ctx.cmif_in_data_off + 7],
                ])
            } else {
                0
            };

            let in_src = ctx
                .send_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
                .or_else(|| {
                    ctx.send_statics
                        .iter()
                        .find(|b| b.size > 0 && b.addr != 0)
                        .copied()
                });
            let mut in_data: Vec<u8> = Vec::new();
            if let Some(sb) = in_src {
                in_data.resize(sb.size as usize, 0);
                let _ = kernel.address_space.read(sb.addr, &mut in_data);
            }

            let out_dst = ctx
                .recv_buffers
                .iter()
                .find(|b| b.size > 0 && b.addr != 0)
                .copied()
                .or_else(|| {
                    ctx.recv_statics
                        .iter()
                        .find(|b| b.size > 0 && b.addr != 0)
                        .copied()
                });
            let out_size = out_dst.map(|b| b.size as usize).unwrap_or(0);

            if cmd_id == 1 {
                log::trace!(
                    "nvdrv:Ioctl fd={} ioctl_id={:#x} send_buf={:?} send_static={:?} recv_buf={:?}",
                    fd,
                    ioctl_id,
                    ctx.send_buffers
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>(),
                    ctx.send_statics
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>(),
                    ctx.recv_buffers
                        .iter()
                        .map(|b| (b.addr, b.size))
                        .collect::<Vec<_>>()
                );
            }

            let req = nexium_nvdrv::IoctlRequest {
                fd,
                ioctl_id,
                in_data,
                out_size,
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

            let ioctl_cmd = (ioctl_id & 0xFFFF) as u16;
            if ioctl_cmd == 0x4808 || ioctl_cmd == 0x481b {
                let fence_handles: Vec<u32> = kernel.gpu_fence_events.drain().collect();
                for fh in fence_handles {
                    kernel.event_signals.insert(fh, true);
                    log::debug!(
                        "nvdrv:SubmitGPFIFO → signaling gpu_fence_event handle={:#x}",
                        fh
                    );
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
            } else {
                0
            };
            kernel.nvdrv.close(fd);
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        3 => {
            log::debug!("nvdrv:Initialize");
            build_ipc_response(ctx, 0, &0u32.to_le_bytes(), &[])
        }
        4 => {
            let fd = if ctx.cmif_in_data_len >= 4 {
                u32::from_le_bytes([
                    ctx.buf[ctx.cmif_in_data_off],
                    ctx.buf[ctx.cmif_in_data_off + 1],
                    ctx.buf[ctx.cmif_in_data_off + 2],
                    ctx.buf[ctx.cmif_in_data_off + 3],
                ])
            } else {
                0
            };
            let is_nvhost_ctrl_fd = kernel
                .nvdrv
                .files
                .get(&fd)
                .map(|f| f.device == nexium_nvdrv::NvDevice::NvhostCtrl)
                .unwrap_or(false);
            let h = kernel.handles.create_handle(HandleType::Event);
            if is_nvhost_ctrl_fd {
                kernel.event_signals.insert(h, false);
                kernel.gpu_fence_events.insert(h);
                log::debug!(
                    "nvdrv:QueryEvent fd={} (nvhost-ctrl) → fence event handle={:#x} (unsignaled, will signal on GPU submit)",
                    fd,
                    h
                );
            } else {
                kernel.event_signals.insert(h, true);
                kernel.nvdrv_sync_events.insert(h);
                log::debug!(
                    "nvdrv:QueryEvent fd={} → event handle={:#x} (COPY, always-signaled)",
                    fd,
                    h
                );
            }
            build_ipc_response_copy(ctx, 0, &0u32.to_le_bytes(), &[h])
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
        ("IApplicationDisplayService", 2020)
        | ("IApplicationDisplayService", 2030)
        | ("IManagerDisplayService", 2012) => Some(build_native_window_parcel(0x100)),
        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => Some(build_igbp_success_parcel()),
        ("ILaunchParamStorageAccessor", 11) => Some(build_launch_parameter()),
        ("acc:u0" | "acc:u1" | "acc:aa", 2) | ("acc:u0" | "acc:u1" | "acc:aa", 3) => {
            Some(build_user_id_list())
        }
        ("IProfile", 0) => Some(vec![0u8; 0x80]),
        _ => None,
    }
}

fn build_launch_parameter() -> Vec<u8> {
    let mut out = vec![0u8; 0x88];
    out[0..4].copy_from_slice(&0xC794_97CAu32.to_le_bytes());
    out[4..8].copy_from_slice(&1u32.to_le_bytes());
    out[8..24].copy_from_slice(&crate::services::am::ACCOUNT_UID);
    out
}

fn build_user_id_list() -> Vec<u8> {
    let mut out = vec![0u8; 0x80];
    out[0..16].copy_from_slice(&crate::services::am::ACCOUNT_UID);
    out
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
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&binder_handle.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(b"dispdrv\0");
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::with_capacity(16 + payload.len() + 4);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

pub(crate) fn return_subsession(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    sub_service: &str,
) -> Vec<u8> {
    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    if is_domain {
        let object_id = alloc_domain_object(kernel, session_handle, sub_service);
        log::debug!("→ {} sub-object id={}", sub_service, object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, sub_service.to_string());
        kernel.sessions.insert(h, session);
        log::debug!("→ {} sub-session handle={:#x}", sub_service, h);
        build_ipc_response(ctx, 0, &[], &[h])
    }
}

fn return_file_system_with_root(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    root: std::path::PathBuf,
) -> Vec<u8> {
    let is_domain = kernel
        .sessions
        .get(&session_handle)
        .map(|s| s.is_domain)
        .unwrap_or(false);
    if is_domain {
        let object_id = alloc_domain_object(kernel, session_handle, "IFileSystem");
        for handle in domain_group_handles(kernel, session_handle) {
            kernel
                .file_system_roots
                .insert((handle, object_id), root.clone());
        }
        log::debug!("-> IFileSystem sub-object id={}", object_id);
        build_ipc_response_full(ctx, 0, &[], &[], &[], &[object_id])
    } else {
        let h = kernel.handles.create_handle(HandleType::Session);
        let session = Session::new(h, "IFileSystem".to_string());
        kernel.sessions.insert(h, session);
        kernel.file_system_roots.insert((h, 0), root);
        log::debug!("-> IFileSystem sub-session handle={:#x}", h);
        build_ipc_response(ctx, 0, &[], &[h])
    }
}

fn subsession_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("hid", 0) => Some("IAppletResource"),
        ("IApplicationCreator", 0) => Some("IApplicationAccessor"),
        ("ILibraryAppletCreator", 0) => Some("ILibraryAppletAccessor"),
        ("time:s" | "time:u" | "time:a" | "time:r", 0) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 1) => Some("ISystemClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 2) => Some("ISteadyClock"),
        ("time:s" | "time:u" | "time:a" | "time:r", 3) => Some("ITimeZoneService"),
        ("time:s" | "time:u" | "time:a" | "time:r", 4) => Some("ISystemClock"),
        ("friend:u" | "friend:a" | "friend:s" | "friend:v" | "friend:m", 0) => {
            Some("IFriendService")
        }
        ("nfp:user", 0) => Some("INfpUser"),
        ("fsp-srv", 18) => Some("IFileSystem"),
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
        ("apm" | "apm:p", 0) => Some("IApmManager"),
        ("IApmManager", 0) => Some("IApmSession"),
        ("pctl:a" | "pctl:r" | "pctl:s" | "pctl", 0) => Some("IParentalControlService"),
        ("pctl:a" | "pctl:r" | "pctl:s" | "pctl", 1) => Some("IParentalControlService"),
        _ => None,
    }
}

fn applet_command_response(
    _kernel: &mut Kernel,
    port_name: &str,
    cmd_id: u32,
) -> Option<(Vec<u8>, Option<u32>)> {
    match (port_name, cmd_id) {
        ("IDebugFunctions", _) => Some((Vec::new(), None)),

        ("IHOSBinderDriver", 0) | ("IHOSBinderDriver", 3) => Some((Vec::new(), None)),

        ("IFileSystem", _) => Some((Vec::new(), None)),
        ("fsp-srv", _) => Some((Vec::new(), None)),

        ("psm", _) => Some((Vec::new(), None)),
        ("set", _) | ("set:sys", _) => Some((Vec::new(), None)),
        ("nvdrv:a", _) | ("nvdrv", _) | ("nvdrv:s", _) | ("nvdrv:t", _) => {
            Some((0u32.to_le_bytes().to_vec(), None))
        }

        ("IParentalControlService", cmd) => {
            let out: Vec<u8> = match cmd {
                1031 | 1061 | 1403 | 1453 | 1455 | 1458 => vec![0u8],
                1018 | 1065 => vec![1u8],
                1032 | 1039 | 1206 => 0u32.to_le_bytes().to_vec(),
                _ => Vec::new(),
            };
            Some((out, None))
        }

        _ => None,
    }
}

fn applet_proxy_service(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("appletAE" | "appletOE", 100) => Some("ISystemAppletProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        ("appletAE" | "appletOE", 300) => Some("IOverlayAppletProxy"),
        ("appletAE" | "appletOE", 350) => Some("IApplicationProxy"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            0,
        ) => Some("ICommonStateGetter"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            1,
        ) => Some("ISelfController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            2,
        ) => Some("IWindowController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            3,
        ) => Some("IAudioController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            4,
        ) => Some("IDisplayController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            10,
        ) => Some("IProcessWindingController"),
        (
            "ISystemAppletProxy"
            | "ILibraryAppletProxy"
            | "IOverlayAppletProxy"
            | "IApplicationProxy",
            11,
        ) => Some("ILibraryAppletCreator"),
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

fn dispatch_sm_command(
    kernel: &mut Kernel,
    cmd_id: u32,
    tls_buf: &[u8],
    cmif_data_off: usize,
    cmif_data_len: usize,
    parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    match cmd_id {
        0 => dispatch_sm_register_client(kernel, tls_buf, cmif_data_off, cmif_data_len, parsed_ctx),
        1 => dispatch_sm_get_service_handle(
            kernel,
            tls_buf,
            cmif_data_off,
            cmif_data_len,
            parsed_ctx,
        ),
        2 => dispatch_sm_register_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        3 => dispatch_sm_unregister_service(kernel, tls_buf, cmif_data_off, cmif_data_len),
        _ => {
            log::warn!("unknown SM command: {}", cmd_id);
            (1, Vec::new())
        }
    }
}

fn dispatch_sm_register_client(
    _kernel: &mut Kernel,
    _tls_buf: &[u8],
    _cmif_data_off: usize,
    _cmif_data_len: usize,
    parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    let pid = parsed_ctx.and_then(|ctx| ctx.send_pid);
    log::info!("SM::RegisterClient pid={:?}", pid);
    (SUCCESS, Vec::new())
}

fn dispatch_sm_register_service(
    _kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
) -> (u32, Vec<u8>) {
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

fn dispatch_sm_unregister_service(
    _kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
) -> (u32, Vec<u8>) {
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

fn dispatch_sm_get_service_handle(
    kernel: &mut Kernel,
    tls_buf: &[u8],
    cmif_data_off: usize,
    _cmif_data_len: usize,
    _parsed_ctx: Option<ipc::IpcCtx>,
) -> (u32, Vec<u8>) {
    let service_name = if tls_buf.len() >= cmif_data_off + 8 {
        let name_bytes = &tls_buf[cmif_data_off..cmif_data_off + 8];
        let trimmed = name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes);
        String::from_utf8_lossy(trimmed).into_owned()
    } else {
        String::new()
    };

    log::info!(
        "SM::GetServiceHandle '{}' data_off={:#x}",
        service_name,
        cmif_data_off
    );

    let handle = kernel.handles.create_handle(HandleType::Session);
    let final_name = if !service_name.is_empty() {
        service_name
    } else {
        "unknown".to_string()
    };
    let session = Session::new(handle, final_name.clone());
    kernel.sessions.insert(handle, session);

    log::info!(
        "SM: returning handle {:#x} for service '{}'",
        handle,
        final_name
    );

    let mut response = Vec::new();
    response.extend_from_slice(&handle.to_le_bytes());
    (SUCCESS, response)
}

fn write_ipc_response(buf: &mut [u8], data_offset: usize, result: u32, token: u32) {
    write_ipc_response_with_data(buf, data_offset, result, token, &[]);
}

fn write_ipc_response_with_data(
    buf: &mut [u8],
    data_offset: usize,
    result: u32,
    token: u32,
    out_data: &[u8],
) {
    let hipc_resp: u64 = 0x0000_0004_0000_0000;
    buf[0..8].copy_from_slice(&hipc_resp.to_le_bytes());

    let off = (data_offset + 3) & !3;
    if buf.len() >= off + 16 {
        buf[off..off + 4].copy_from_slice(b"SFCO");
        buf[off + 4..off + 8].copy_from_slice(&0u32.to_le_bytes());
        buf[off + 8..off + 12].copy_from_slice(&result.to_le_bytes());
        buf[off + 12..off + 16].copy_from_slice(&token.to_le_bytes());

        if !out_data.is_empty() && off + 16 + out_data.len() <= buf.len() {
            buf[off + 16..off + 16 + out_data.len()].copy_from_slice(out_data);
        }
    }
}

fn svc_get_thread_id(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1) as u32
    } else {
        0
    };
    let target = if handle == 0 || handle == 0xFFFF8000 {
        kernel
            .threads
            .current_handle()
            .unwrap_or(kernel.main_thread_handle)
    } else {
        handle
    };
    let tid = kernel
        .threads
        .threads
        .get(&target)
        .map(|t| t.tid)
        .unwrap_or(1);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, tid);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_id(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0x4F4F4F4F_4F4F4F4F);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_clear_event(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    kernel.event_signals.insert(handle, false);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reset_signal(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    kernel.event_signals.insert(handle, false);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_wait_for_address(kernel: &mut Kernel) -> u32 {
    let (addr, arb_type, value, timeout_ns) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1) as u32,
            cpu.get_register(2) as u32,
            cpu.get_register(3),
        )
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
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_INVALID_STATE as u64);
        }
        return KERNEL_INVALID_STATE;
    }

    if arb_type == 1 {
        let _ = kernel
            .address_space
            .write(addr, &current.wrapping_sub(1).to_le_bytes());
    }

    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if timeout_ns == 0 {
        if let Some(cpu) = cpu_mut() {
            cpu.set_register(0, KERNEL_TIMEOUT as u64);
        }
        return KERNEL_TIMEOUT;
    }

    let cap = std::time::Duration::from_millis(100);
    let wait = if timeout_ns == u64::MAX {
        cap
    } else {
        std::time::Duration::from_nanos(timeout_ns).min(cap)
    };
    if wait > std::time::Duration::ZERO {
        if let Some(cpu) = cpu_ref() {
            let wake_at = std::time::Instant::now() + wait;
            kernel.threads.yield_with_state(
                cpu,
                crate::kernel::threads::ThreadState::Sleeping { wake_at },
            );
        }
    }
    if !kernel.threads.ready.is_empty() {
        kernel.yield_after_svc = true;
    }

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_TIMEOUT as u64);
    }
    KERNEL_TIMEOUT
}

fn svc_signal_to_address(kernel: &mut Kernel) -> u32 {
    let (addr, signal_type, value, count) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(0),
            cpu.get_register(1) as u32,
            cpu.get_register(2) as u32,
            cpu.get_register(3) as i32,
        )
    } else {
        return 1;
    };

    let mut buf = [0u8; 4];
    let _ = kernel.address_space.read(addr, &mut buf);
    let current = u32::from_le_bytes(buf);

    match signal_type {
        0 => {}
        1 if current == value => {
            let _ = kernel
                .address_space
                .write(addr, &current.wrapping_add(1).to_le_bytes());
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

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_break(kernel: &mut Kernel) -> u32 {
    let reason = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        0
    };
    let info_va = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1)
    } else {
        0
    };
    let info_size = if let Some(cpu) = cpu_ref() {
        cpu.get_register(2) as usize
    } else {
        0
    };

    log::warn!(
        "svcBreak: reason={:#x}, info_va={:#x}, info_size={:#x}",
        reason,
        info_va,
        info_size
    );

    if let Some(cpu) = cpu_ref() {
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
            let next_fp = u64::from_le_bytes([
                frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7],
            ]);
            let saved_lr = u64::from_le_bytes([
                frame[8], frame[9], frame[10], frame[11], frame[12], frame[13], frame[14],
                frame[15],
            ]);
            callers.push((i, saved_lr));
            if next_fp == 0 || next_fp <= cur_fp {
                break;
            }
            cur_fp = next_fp;
        }

        for (i, addr) in &callers {
            log::warn!(
                "  Stack[{}]: {:#x} (offset {:#x})",
                i,
                addr,
                addr.wrapping_sub(kernel.code_base)
            );
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

    if let Some(cpu) = cpu_ref() {
        let str_ptr = cpu.get_register(0);
        let str_len = cpu.get_register(1);

        if str_ptr > 0 && str_len > 0 && str_len < 262_144 {
            let mut buf = vec![0u8; str_len as usize];
            match kernel.address_space.read(str_ptr, &mut buf) {
                Ok(()) => {
                    let output = std::str::from_utf8(&buf).unwrap_or("[invalid utf8]");
                    println!("[DEBUG] {}", output);
                    log::info!("OutputDebugString: {}", output);
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

    let port_name_ptr = if let Some(cpu) = cpu_ref() {
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
                log::info!(
                    "  port_name: '{}' (len={}) PC={:#x}",
                    name_str,
                    len,
                    cpu_ref().map(|c| c.get_pc()).unwrap_or(0)
                );
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

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
        cpu.set_register(1, handle as u64);
    } else {
        log::error!("kernel.cpu is None!");
    }

    log::info!(
        "created session handle {:#x} to port '{}'",
        handle,
        port_name
    );
    SUCCESS
}

fn svc_get_info(kernel: &mut Kernel) -> u32 {
    let (info_type, _handle, _sub) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1) as u32,
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        return 1;
    };

    let val: u64 = match info_type {
        0 => 0xF,
        1 => 0x0001_0000_0000,
        2 => kernel.alias_base,
        3 => kernel.alias_size,
        4 => kernel.heap_base,
        5 => kernel.heap_size,
        6 => {
            if kernel.is_application {
                kernel.total_memory
            } else {
                0x80_000_000
            }
        }
        7 => {
            if kernel.is_application {
                kernel.code_size + kernel.stack_size + kernel.heap_committed + 0x100_0000
            } else {
                0x40_000_000
            }
        }
        8 => 0,
        9 => kernel.stack_base,
        10 => kernel.stack_size,
        11 => {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let c = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut z = c
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        12 => kernel.aslr_base,
        13 => kernel.aslr_size,
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
        _ => {
            log::warn!(
                "svcGetInfo: unsupported type {} — returning InvalidEnumValue (0xF001)",
                info_type
            );
            if let Some(cpu) = cpu_mut() {
                cpu.set_register(0, 0xF001);
                cpu.set_register(1, 0);
            }
            return 0xF001;
        }
    };

    log::debug!("svcGetInfo type={} -> {:#x}", info_type, val);
    if let Some(cpu) = cpu_mut() {
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
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, writable as u64);
        cpu.set_register(2, readable as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    log::debug!(
        "  created event writable={:#x} readable={:#x}",
        writable,
        readable
    );
    SUCCESS
}

fn svc_map_transfer_memory(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcMapTransferMemory");
    SUCCESS
}

fn dump_regs(_kernel: &Kernel, tag: &str) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Some(cpu) = cpu_ref() {
        log::trace!(
            "  [{}] X0={:#x} X1={:#x} X2={:#x} X3={:#x} X4={:#x} X8={:#x} X19={:#x} X30={:#x}",
            tag,
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
            cpu.get_register(4),
            cpu.get_register(8),
            cpu.get_register(19),
            cpu.get_register(30),
        );
    }
}

fn svc_create_transfer_memory(kernel: &mut Kernel) -> u32 {
    dump_regs(kernel, "CreateTmem ENTRY");
    let (addr, size, perm) = if let Some(cpu) = cpu_ref() {
        (
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
        )
    } else {
        return 1;
    };
    let handle = kernel.handles.create_handle(HandleType::TransferMemory);
    log::info!(
        "svcCreateTransferMemory addr={:#x} size={:#x} perm={:#x} → handle={:#x}",
        addr,
        size,
        perm,
        handle
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
    }
    dump_regs(kernel, "CreateTmem EXIT");
    SUCCESS
}

fn svc_close_handle(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    let kind = kernel
        .handles
        .get_handle(handle)
        .map(|h| format!("{:?}", h.handle_type))
        .unwrap_or_else(|| "unknown".into());
    log::debug!("svcCloseHandle handle={:#x} ({})", handle, kind);
    dump_regs(kernel, "CloseHandle ENTRY");
    if let Some(closed) = kernel.handles.close_handle(handle) {
        if closed.handle_type == HandleType::Thread {
            kernel.exited_thread_handles.remove(&handle);
        }
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_thread(kernel: &mut Kernel) -> u32 {
    let (entry, arg, sp, priority, core) = if let Some(cpu) = cpu_ref() {
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

    let handle = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Thread);
    let tls_va = kernel.threads.alloc_tls();

    let mut ctx = crate::kernel::threads::ThreadCtx::zero();
    ctx.x[0] = arg;
    ctx.sp = sp;
    ctx.pc = entry;
    ctx.tpidrro_el0 = tls_va;

    kernel.threads.add_thread(handle, ctx, tls_va, sp);
    if let Some(t) = kernel.threads.threads.get_mut(&handle) {
        t.priority = priority;
        let active_cores = std::env::var("NEXIUM_CPU_CORES")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(crate::kernel::threads::NUM_CORES as i32)
            .clamp(1, crate::kernel::threads::NUM_CORES as i32);
        if (0..active_cores).contains(&core) {
            t.core = core;
        }
    }

    log::info!(
        "svcCreateThread entry={:#x} arg={:#x} sp={:#x} prio={} core={} -> handle={:#x} tls={:#x}",
        entry,
        arg,
        sp,
        priority,
        core,
        handle,
        tls_va
    );

    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_start_thread(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        return 1;
    };
    log::info!("svcStartThread handle={:#x}", handle);
    kernel
        .threads
        .transition_state(handle, crate::kernel::threads::ThreadState::Ready);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_exit_thread(kernel: &mut Kernel) -> u32 {
    let current = kernel.threads.current_handle();
    log::debug!("svcExitThread current={:?}", current);
    if let Some(handle) = current {
        kernel.exited_thread_handles.insert(handle);
        kernel.threads.signal_handle(handle);
    }
    if let Some(cpu) = cpu_ref() {
        kernel
            .threads
            .yield_with_state(cpu, crate::kernel::threads::ThreadState::Exited);
    }
    SUCCESS
}

fn svc_sleep_thread(kernel: &mut Kernel) -> u32 {
    let ns = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0)
    } else {
        0
    };
    let signed = ns as i64;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    if signed > 0 {
        let dur = std::time::Duration::from_nanos(ns);
        let wake_at = std::time::Instant::now() + dur;
        if let Some(cpu) = cpu_ref() {
            kernel.threads.yield_with_state(
                cpu,
                crate::kernel::threads::ThreadState::Sleeping { wake_at },
            );
        }
    } else if signed == 0 || signed == -1 {
        if let Some(cpu) = cpu_ref() {
            kernel
                .threads
                .yield_with_state(cpu, crate::kernel::threads::ThreadState::Ready);
        }
    }
    SUCCESS
}

fn svc_flush_data_cache(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_priority(kernel: &mut Kernel) -> u32 {
    let handle = if let Some(cpu) = cpu_ref() {
        cpu.get_register(1) as u32
    } else {
        return 1;
    };
    let prio = kernel
        .threads
        .threads
        .get(&handle)
        .map(|t| t.priority)
        .unwrap_or(0x2C);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, prio as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_priority(kernel: &mut Kernel) -> u32 {
    let (handle, priority) = if let Some(cpu) = cpu_ref() {
        (cpu.get_register(0) as u32, cpu.get_register(1) as i32)
    } else {
        return 1;
    };
    if let Some(t) = kernel.threads.threads.get_mut(&handle) {
        t.priority = priority;
    }
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_core_mask(kernel: &mut Kernel) -> u32 {
    let handle = cpu_ref().map(|cpu| cpu.get_register(0) as u32);
    let core = handle
        .and_then(|h| kernel.threads.threads.get(&h))
        .map(|t| t.core)
        .filter(|core| (0..crate::kernel::threads::NUM_CORES as i32).contains(core))
        .unwrap_or(0) as u64;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, core);
        cpu.set_register(2, 0xF);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_core_mask(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_current_processor_number(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, crate::kernel::cpu_local::current_core() as u64);
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
    let handle = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(handle, true);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, handle as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_return_from_exception(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReturnFromException");
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_flush_entire_data_cache(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_future_thread_info(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_last_thread_info(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        for r in 1..=5 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_limit_value(_kernel: &mut Kernel) -> u32 {
    let limitable = if let Some(cpu) = cpu_ref() {
        cpu.get_register(2) as u32
    } else {
        0
    };
    let value: u64 = match limitable {
        0 => 0x40_000_000,
        1 => 1024,
        2 => 1024,
        3 => 8,
        4 => 0x80_000,
        5 => 64,
        _ => 0,
    };
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, value);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_current_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_resource_limit_peak_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_thread_activity(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_context3(kernel: &mut Kernel) -> u32 {
    let (out_ptr, handle) = if let Some(cpu) = cpu_ref() {
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
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_synchronize_preemption_state(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_session(kernel: &mut Kernel) -> u32 {
    let server = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    let client = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_accept_session(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_reply_and_receive_light(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_reply_and_receive(_kernel: &mut Kernel) -> u32 {
    log::debug!("svcReplyAndReceive (stub → TIMEOUT)");
    const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_TIMEOUT as u64);
    }
    KERNEL_TIMEOUT
}

fn svc_reply_and_receive_with_user_buffer(kernel: &mut Kernel) -> u32 {
    svc_reply_and_receive(kernel)
}

fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::SharedMemory);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_unmap_transfer_memory(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_interrupt_event(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Event);
    kernel.event_signals.insert(h, false);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_query_io_mapping(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 0);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_debug_active_process(_kernel: &mut Kernel) -> u32 {
    const KERNEL_INVALID_HANDLE: u32 = 1 | (114 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_INVALID_HANDLE as u64);
    }
    KERNEL_INVALID_HANDLE
}

fn svc_break_debug_process(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_terminate_debug_process(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_debug_event(_kernel: &mut Kernel) -> u32 {
    const KERNEL_NO_DEBUG_EVENT: u32 = 1 | (140 << 9);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, KERNEL_NO_DEBUG_EVENT as u64);
    }
    KERNEL_NO_DEBUG_EVENT
}

fn svc_continue_debug_event(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_process_list(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, 1);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_get_thread_list(kernel: &mut Kernel) -> u32 {
    let count = kernel.threads.threads.len() as u64;
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, count);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_port(kernel: &mut Kernel) -> u32 {
    let server = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    let client = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, server as u64);
        cpu.set_register(2, client as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_manage_named_port(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Port);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_connect_to_port(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Session);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_create_resource_limit(kernel: &mut Kernel) -> u32 {
    let h = kernel
        .handles
        .create_handle(crate::kernel::handles::HandleType::Process);
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(1, h as u64);
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_set_resource_limit_limit_value(_kernel: &mut Kernel) -> u32 {
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn svc_call_secure_monitor(_kernel: &mut Kernel) -> u32 {
    let smc_id = if let Some(cpu) = cpu_ref() {
        cpu.get_register(0) as u32
    } else {
        0
    };
    log::debug!(
        "svcCallSecureMonitor smc_id={:#x} (HLE: returning success)",
        smc_id
    );
    if let Some(cpu) = cpu_mut() {
        for r in 0..=7 {
            cpu.set_register(r, 0);
        }
        cpu.set_register(0, SUCCESS as u64);
    }
    SUCCESS
}

fn fs_base_root() -> Option<std::path::PathBuf> {
    std::env::var_os("APPDATA")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("XDG_DATA_HOME").map(std::path::PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share"))
        })
        .map(|base| base.join("NeXium"))
}

fn fs_sd_root(kernel: &mut Kernel) -> Option<std::path::PathBuf> {
    if kernel.sd_root.is_none() {
        let root = fs_base_root()?.join("sdmc");
        if let Err(e) = std::fs::create_dir_all(&root) {
            log::warn!("fs: failed to create SD root {}: {}", root.display(), e);
            return None;
        }
        kernel.sd_root = Some(root);
    }
    kernel.sd_root.clone()
}

fn fs_save_data_root(kernel: &Kernel, ctx: &ipc::IpcCtx) -> Option<std::path::PathBuf> {
    let data_start = ctx.cmif_in_data_off.min(ctx.buf.len());
    let data_end = data_start
        .saturating_add(ctx.cmif_in_data_len)
        .min(ctx.buf.len());
    let data = &ctx.buf[data_start..data_end];
    let space_id = data.first().copied().unwrap_or(1);
    let attr_off = 8usize;
    let program_id = fs_read_le_u64(data, attr_off).unwrap_or(0);
    let system_save_data_id = fs_read_le_u64(data, attr_off + 24).unwrap_or(0);
    let save_type = data.get(attr_off + 32).copied().unwrap_or(1);
    let user_id = fs_user_id_hex(data, attr_off + 8);
    let title_id = if program_id != 0 {
        program_id
    } else {
        kernel.title_id
    };
    let base = fs_base_root()?;
    let title = format!("{:016x}", title_id);
    let root = match space_id {
        0 => base
            .join("nand")
            .join("system")
            .join("save")
            .join(format!("{:016x}", system_save_data_id))
            .join(&user_id),
        1 => match save_type {
            4 => base
                .join("nand")
                .join("temp")
                .join("0000000000000000")
                .join(&user_id)
                .join(&title),
            5 => base
                .join("nand")
                .join("user")
                .join("save")
                .join("cache")
                .join(&title),
            _ => base
                .join("nand")
                .join("user")
                .join("save")
                .join("0000000000000000")
                .join(&user_id)
                .join(&title),
        },
        2 | 4 => base
            .join("sdmc")
            .join("save")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
        3 => base
            .join("nand")
            .join("temp")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
        _ => base
            .join("nand")
            .join("user")
            .join("save")
            .join("0000000000000000")
            .join(&user_id)
            .join(&title),
    };
    if let Err(e) = std::fs::create_dir_all(&root) {
        log::warn!("fs: failed to create save root {}: {}", root.display(), e);
        return None;
    }
    Some(root)
}

fn fs_read_le_u64(data: &[u8], off: usize) -> Option<u64> {
    let bytes = data.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn fs_user_id_hex(data: &[u8], off: usize) -> String {
    let mut bytes = [0u8; 16];
    if let Some(src) = data.get(off..off.saturating_add(16)) {
        if src.len() == 16 {
            bytes.copy_from_slice(src);
        }
    }
    let mut low_bytes = [0u8; 8];
    let mut high_bytes = [0u8; 8];
    low_bytes.copy_from_slice(&bytes[0..8]);
    high_bytes.copy_from_slice(&bytes[8..16]);
    let low = u64::from_le_bytes(low_bytes);
    let high = u64::from_le_bytes(high_bytes);
    format!("{:016x}{:016x}", high, low)
}

fn fs_object_root(
    kernel: &mut Kernel,
    session_handle: u32,
    object_id: u32,
) -> Option<std::path::PathBuf> {
    let root = domain_object_keys(kernel, session_handle, object_id)
        .into_iter()
        .find_map(|key| kernel.file_system_roots.get(&key).cloned());
    root.or_else(|| fs_sd_root(kernel))
}

fn fs_host_path(
    kernel: &mut Kernel,
    session_handle: u32,
    object_id: u32,
    hos: &str,
) -> Option<std::path::PathBuf> {
    let root = fs_object_root(kernel, session_handle, object_id)?;
    fs_translate(&root, hos)
}

fn fs_translate(root: &std::path::Path, hos: &str) -> Option<std::path::PathBuf> {
    let trimmed = hos.trim_start_matches(|c: char| c.is_ascii_alphabetic() || c == ':');
    let trimmed = trimmed.trim_start_matches(|c| c == '/' || c == '\\');
    let rel = std::path::Path::new(trimmed);
    for c in rel.components() {
        if matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::Prefix(_)
                | std::path::Component::RootDir
        ) {
            return None;
        }
    }
    Some(root.join(rel))
}

fn fs_read_path(ctx: &ipc::IpcCtx, addr_space: &nexium_memory::AddressSpace) -> String {
    let buf = ctx
        .send_statics
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .or_else(|| ctx.send_buffers.iter().find(|b| b.size > 0 && b.addr != 0))
        .copied();
    let Some(b) = buf else { return String::new() };
    let n = (b.size as usize).min(0x301);
    let mut bytes = vec![0u8; n];
    if addr_space.read(b.addr, &mut bytes).is_err() {
        return String::new();
    }
    let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[derive(Clone, Copy)]
struct RomfsHeader {
    dir_meta_off: usize,
    dir_meta_size: usize,
    file_meta_off: usize,
    file_meta_size: usize,
    file_data_off: usize,
}

enum RomfsEntry {
    Dir,
    File { offset: usize, size: usize },
}

fn romfs_u32(data: &[u8], off: usize) -> Option<u32> {
    let bytes = data.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn romfs_u64(data: &[u8], off: usize) -> Option<u64> {
    let bytes = data.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn romfs_usize(data: &[u8], off: usize) -> Option<usize> {
    usize::try_from(romfs_u64(data, off)?).ok()
}

fn romfs_header(romfs: &[u8]) -> Option<RomfsHeader> {
    if romfs_u64(romfs, 0)? != 0x50 {
        return None;
    }
    Some(RomfsHeader {
        dir_meta_off: romfs_usize(romfs, 0x18)?,
        dir_meta_size: romfs_usize(romfs, 0x20)?,
        file_meta_off: romfs_usize(romfs, 0x38)?,
        file_meta_size: romfs_usize(romfs, 0x40)?,
        file_data_off: romfs_usize(romfs, 0x48)?,
    })
}

fn romfs_components(path: &str) -> Option<Vec<&str>> {
    let path = path.trim_matches(char::from(0)).trim();
    let path = path.strip_prefix("rom:").unwrap_or(path);
    let mut out = Vec::new();
    for part in path.split(|c| c == '/' || c == '\\') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return None;
        }
        out.push(part);
    }
    Some(out)
}

fn romfs_name(
    romfs: &[u8],
    entry_abs: usize,
    name_off: usize,
    name_len_off: usize,
) -> Option<&str> {
    let name_len = romfs_u32(romfs, entry_abs.checked_add(name_len_off)?)? as usize;
    let start = entry_abs.checked_add(name_off)?;
    let end = start.checked_add(name_len)?;
    std::str::from_utf8(romfs.get(start..end)?).ok()
}

fn romfs_child_dir(romfs: &[u8], hdr: RomfsHeader, dir_off: u32, name: &str) -> Option<u32> {
    let dir_rel = usize::try_from(dir_off).ok()?;
    if dir_rel >= hdr.dir_meta_size {
        return None;
    }
    let mut child = romfs_u32(
        romfs,
        hdr.dir_meta_off.checked_add(dir_rel)?.checked_add(0x08)?,
    )?;
    while child != u32::MAX {
        let rel = usize::try_from(child).ok()?;
        if rel >= hdr.dir_meta_size {
            return None;
        }
        let abs = hdr.dir_meta_off.checked_add(rel)?;
        if romfs_name(romfs, abs, 0x18, 0x14)? == name {
            return Some(child);
        }
        child = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }
    None
}

fn romfs_child_file(
    romfs: &[u8],
    hdr: RomfsHeader,
    dir_off: u32,
    name: &str,
) -> Option<(usize, usize)> {
    let dir_rel = usize::try_from(dir_off).ok()?;
    if dir_rel >= hdr.dir_meta_size {
        return None;
    }
    let mut child = romfs_u32(
        romfs,
        hdr.dir_meta_off.checked_add(dir_rel)?.checked_add(0x0c)?,
    )?;
    while child != u32::MAX {
        let rel = usize::try_from(child).ok()?;
        if rel >= hdr.file_meta_size {
            return None;
        }
        let abs = hdr.file_meta_off.checked_add(rel)?;
        if romfs_name(romfs, abs, 0x20, 0x1c)? == name {
            let rel_off = usize::try_from(romfs_u64(romfs, abs.checked_add(0x08)?)?).ok()?;
            let size = usize::try_from(romfs_u64(romfs, abs.checked_add(0x10)?)?).ok()?;
            let offset = hdr.file_data_off.checked_add(rel_off)?;
            return Some((offset, size));
        }
        child = romfs_u32(romfs, abs.checked_add(0x04)?)?;
    }
    None
}

fn romfs_find_entry(romfs: &[u8], path: &str) -> Option<RomfsEntry> {
    let hdr = romfs_header(romfs)?;
    let comps = romfs_components(path)?;
    if comps.is_empty() {
        return Some(RomfsEntry::Dir);
    }
    let mut dir = 0u32;
    for (i, name) in comps.iter().enumerate() {
        let last = i + 1 == comps.len();
        if last {
            if let Some(child_dir) = romfs_child_dir(romfs, hdr, dir, name) {
                let rel = usize::try_from(child_dir).ok()?;
                if rel < hdr.dir_meta_size {
                    return Some(RomfsEntry::Dir);
                }
            }
            if let Some((offset, size)) = romfs_child_file(romfs, hdr, dir, name) {
                return Some(RomfsEntry::File { offset, size });
            }
            return None;
        }
        dir = romfs_child_dir(romfs, hdr, dir, name)?;
    }
    None
}

fn romfs_open_file(romfs: &[u8], path: &str) -> Option<(usize, usize)> {
    match romfs_find_entry(romfs, path)? {
        RomfsEntry::File { offset, size } => Some((offset, size)),
        RomfsEntry::Dir => None,
    }
}

fn romfs_entry_type(romfs: &[u8], path: &str) -> Option<u32> {
    match romfs_find_entry(romfs, path)? {
        RomfsEntry::Dir => Some(0),
        RomfsEntry::File { .. } => Some(1),
    }
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

fn unswizzle_block_linear(
    src: &[u8],
    stride: u32,
    height: u32,
    bpp: usize,
    block_height_log2: u32,
) -> Vec<u8> {
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
            let gob_offset =
                block_row_offset + gob_col * block_height * GOB_SIZE + gob_row_in_block * GOB_SIZE;
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
        .map(|b| nexium_cmif::CmifBuffer {
            addr: b.addr,
            size: b.size,
        })
        .collect()
}

struct HomebrewEntry {
    name: String,
    size: i64,
}

fn enumerate_homebrew_nros(dir: &Option<std::path::PathBuf>) -> Vec<HomebrewEntry> {
    let Some(dir) = dir else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<HomebrewEntry> = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let is_nro = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("nro"))
            .unwrap_or(false);
        if !is_nro {
            continue;
        }
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
    let mem = AddressSpaceMemory {
        addr_space: &*kernel.address_space,
    };
    let mut cmif_ctx = make_cmif_ctx(
        ctx,
        &mem,
        &recv_buffers,
        &recv_statics,
        &send_buffers,
        &send_statics,
    );
    kernel
        .services
        .set
        .dispatch_cmif(ctx.cmif_in.cmd_id, &mut cmif_ctx)
}
