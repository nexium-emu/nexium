use nexium_ipc::IpcCtx;

use crate::kernel::handles::HandleType;
use crate::kernel::svc::{build_ipc_response, build_ipc_response_copy, return_subsession};
use crate::kernel::Kernel;

const CLIENT: &str = "IChatService";

pub(crate) fn dispatch(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session_handle: u32,
    port_name: &str,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    match (port_name, cmd_id) {
        ("chat:u", 0) => Some(return_subsession(kernel, ctx, session_handle, CLIENT)),
        (CLIENT, 0) => Some(build_ipc_response(ctx, 0, &[0; 8], &[])),
        (CLIENT, 1) => {
            let event = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(event, false);
            Some(build_ipc_response_copy(ctx, 0, &[], &[event]))
        }
        ("chat:u" | CLIENT, _) => {
            log::warn!("{port_name} cmd {cmd_id} stubbed");
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        _ => None,
    }
}
