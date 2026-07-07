use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn adjust_refcount(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_native_handle(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    kernel.vsync_handles.insert(h);
    log::info!(
        "IHOSBinderDriver.GetNativeHandle → BinderEvent {:#x} (registered as vsync)",
        h
    );
    h
}
