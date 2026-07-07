use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_size(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}

pub fn get_handle(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> (u64, u32) {
    let h = kernel.handles.create_handle(HandleType::TransferMemory);
    (0, h)
}
