mod gpu;
mod native;
mod swizzle;
mod syncpt;
mod vi2;

use crate::kernel::handles::HandleType;
use crate::kernel::session::Session;
use crate::kernel::svc::{build_ipc_response, build_ipc_response_copy, dispatch_nvdrv_command};
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub const SUPERVISOR: &str = "nx2:gsv";
pub const NVSRV: &str = "nx2:nvsrv";
pub const SYNCPT: &str = "nx2:syncpt";
pub const GPU: &str = "nx2:gpu";
pub const VI2: &str = "nx2:vi2";
pub const CHAT: &str = "nx2:chat";
pub const CAMERA: &str = "nx2:camera";

#[derive(Default)]
pub struct Graphics {
    gpu: gpu::GpuState,
    native: native::NativeState,
    syncpt: syncpt::SyncptState,
    reported: std::collections::HashSet<(String, u32)>,
    presents: u64,
}

pub(crate) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, port_name: &str, cmd_id: u32) -> Option<Vec<u8>> {
    let reply = match port_name {
        NVSRV => Some(dispatch_nvdrv_command(kernel, ctx, port_name)),
        SUPERVISOR => supervisor(kernel, ctx, cmd_id),
        SYNCPT => syncpt::dispatch(kernel, ctx, cmd_id),
        GPU => gpu::dispatch(kernel, ctx, cmd_id),
        VI2 => vi2::dispatch(kernel, ctx, cmd_id),
        CHAT => chat(kernel, ctx, cmd_id),
        CAMERA => Some(build_ipc_response(ctx, 0, &[], &[])),
        _ if port_name.starts_with("nx2:") => None,
        _ => return None,
    };
    Some(reply.unwrap_or_else(|| {
        let first = kernel.nx2_graphics.reported.insert((port_name.to_string(), cmd_id));
        let frames = if first { backtrace(kernel) } else { Vec::new() };
        log::warn!(
            "[nx2] {port_name} cmd={cmd_id} tipc={} words={:08x?} copy={:x?} move={:x?} unhandled{}",
            ctx.tipc.is_some(),
            words(ctx),
            ctx.copy_handles,
            ctx.move_handles,
            if first { format!(" frames={frames:#x?}") } else { String::new() }
        );
        build_ipc_response(ctx, 0, &[], &[])
    }))
}

fn supervisor(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    let service = match cmd_id {
        202 => SYNCPT,
        305 => VI2,
        497 => GPU,
        1402 => NVSRV,
        1202 => CHAT,
        602 => CAMERA,
        601 | 1201 => return Some(build_ipc_response(ctx, 0, &[], &[])),
        204 => return Some(syncpt::attach_page(kernel, ctx)),
        _ => return native::dispatch(kernel, ctx, cmd_id),
    };
    Some(open_session(kernel, ctx, service))
}

fn chat(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    match cmd_id {
        1 => Some(reply_words(ctx, &[1])),
        11 => {
            let event = new_event(kernel);
            Some(build_ipc_response_copy(ctx, 0, &[], &[event]))
        }
        _ => None,
    }
}

pub fn poll(kernel: &mut Kernel) {
    if kernel.nx2.is_some() {
        syncpt::poll(kernel);
    }
}

pub(crate) fn open_session(kernel: &mut Kernel, ctx: &IpcCtx, service: &str) -> Vec<u8> {
    let handle = kernel.handles.create_handle(HandleType::Session);
    kernel.sessions.insert(handle, Session::new(handle, service.to_string()));
    log::info!("[nx2] opened {service} session {handle:#x}");
    build_ipc_response_copy(ctx, 0, &[], &[handle])
}

fn dump_frame(index: u64, width: u32, height: u32, rgba: &[u8]) {
    static TARGET: std::sync::OnceLock<Option<(std::path::PathBuf, u64, u64)>> = std::sync::OnceLock::new();
    let Some((directory, every, from)) = TARGET.get_or_init(|| {
        let directory = std::env::var_os("NEXIUM_NX2_FRAME_DUMP")?;
        let setting = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
        };
        let every = setting("NEXIUM_NX2_FRAME_DUMP_EVERY", 30).max(1);
        let from = setting("NEXIUM_NX2_FRAME_DUMP_FROM", 0);
        Some((directory.into(), every, from))
    }) else {
        return;
    };
    if index < *from || (index != 1 && index % every != 0) {
        return;
    }
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.reserve(rgba.len() / 4 * 3);
    for pixel in rgba.chunks_exact(4) {
        ppm.extend_from_slice(&pixel[..3]);
    }
    let path = directory.join(format!("frame-{index:05}.ppm"));
    if let Err(error) = std::fs::write(&path, ppm) {
        log::warn!("[nx2] could not write {}: {error}", path.display());
    }
}

fn peek_memory(kernel: &Kernel) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let (Some(spec), Some(directory)) = (std::env::var_os("NEXIUM_NX2_PEEK"), std::env::var_os("NEXIUM_NX2_FRAME_DUMP")) else {
        return;
    };
    if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    for item in spec.to_string_lossy().split(',') {
        let Some((address, length)) = item.split_once(':') else {
            continue;
        };
        let parse = |text: &str| u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
        let (Some(address), Some(length)) = (parse(address), parse(length)) else {
            continue;
        };
        let cpu = kernel.nvdrv.gpu.mappings.read().cpu_address_for(address).unwrap_or(address);
        log::info!("[nx2] peek gpu {address:#x} -> cpu {cpu:#x}");
        let mut bytes = vec![0u8; length as usize];
        let _ = kernel.address_space.read(cpu, &mut bytes);
        let path = std::path::Path::new(&directory).join(format!("peek-{address:x}.bin"));
        if let Err(error) = std::fs::write(&path, &bytes) {
            log::warn!("[nx2] could not write {}: {error}", path.display());
        }
    }
}

fn backtrace(kernel: &Kernel) -> Vec<u64> {
    let Some(cpu) = crate::kernel::cpu_local::cpu_ref() else {
        return Vec::new();
    };
    let mut frames = vec![cpu.get_register(30)];
    let mut fp = cpu.get_register(29);
    while frames.len() < 10 && fp != 0 && fp % 8 == 0 {
        let (Some(next), Some(lr)) = (kernel.debug_read_u64(fp), kernel.debug_read_u64(fp + 8)) else {
            break;
        };
        if lr == 0 {
            break;
        }
        frames.push(lr);
        if next <= fp {
            break;
        }
        fp = next;
    }
    frames
}

fn words(ctx: &IpcCtx) -> Vec<u32> {
    ctx.input_data()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn word(ctx: &IpcCtx, index: usize) -> u32 {
    words(ctx).get(index).copied().unwrap_or(0)
}

fn long(ctx: &IpcCtx, index: usize) -> u64 {
    u64::from(word(ctx, index)) | (u64::from(word(ctx, index + 1)) << 32)
}

fn reply_words(ctx: &IpcCtx, values: &[u32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    build_ipc_response(ctx, 0, &data, &[])
}

fn reply_longs(ctx: &IpcCtx, values: &[u64]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    build_ipc_response(ctx, 0, &data, &[])
}

fn new_event(kernel: &mut Kernel) -> u32 {
    let handle = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(handle, false);
    handle
}

fn signal(kernel: &mut Kernel, handle: u32) {
    kernel.event_signals.insert(handle, true);
    kernel.threads.signal_handle(handle);
}
