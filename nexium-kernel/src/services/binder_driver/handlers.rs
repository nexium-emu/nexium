use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn adjust_refcount(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_native_handle(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    binder_id: i32,
    type_id: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    if type_id == 0xF {
        kernel.register_bufferqueue_event(h, binder_id as u32);
        log::info!(
            "IHOSBinderDriver.GetNativeHandle binder={} type={:#x} -> BufferQueueEvent {:#x}",
            binder_id,
            type_id,
            h
        );
    } else {
        log::warn!(
            "IHOSBinderDriver.GetNativeHandle binder={} unsupported type={:#x} -> Event {:#x}",
            binder_id,
            type_id,
            h
        );
    }
    h
}
