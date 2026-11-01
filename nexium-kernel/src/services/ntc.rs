use nexium_ipc::IpcCtx;

use crate::kernel::handles::HandleType;
use crate::kernel::svc::{build_ipc_response, build_ipc_response_copy, return_subsession};
use crate::kernel::Kernel;

const CLIENT: &str = "IEnsureNetworkClockAvailabilityService";

pub(crate) fn dispatch(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session_handle: u32,
    port_name: &str,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    match (port_name, cmd_id) {
        ("ntc", 0) => Some(return_subsession(kernel, ctx, session_handle, CLIENT)),
        ("ntc", 100 | 101) => Some(build_ipc_response(ctx, 0, &[], &[])),
        (CLIENT, 1) => {
            let event = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(event, true);
            Some(build_ipc_response_copy(ctx, 0, &[], &[event]))
        }
        (CLIENT, 4) => Some(build_ipc_response(ctx, 0, &[0], &[])),
        (CLIENT, 5) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs() as i64);
            Some(build_ipc_response(ctx, 0, &now.to_le_bytes(), &[]))
        }
        (CLIENT, _) => Some(build_ipc_response(ctx, 0, &[], &[])),
        _ => None,
    }
}
