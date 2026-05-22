use nexium_ipc::IpcCtx;
use nexium_common::result::SUCCESS;

pub trait Service {
    fn dispatch(&self, cmd_id: u32, _ctx: Option<&IpcCtx>) -> u32 {
        log::trace!("service dispatch: cmd_id={}", cmd_id);
        SUCCESS
    }

    fn get_name(&self) -> &'static str;
}

pub struct ServiceContext {
    pub session_id: u32,
    pub client_pid: u64,
    pub port_name: String,
}

impl ServiceContext {
    pub fn new(session_id: u32, port_name: String) -> Self {
        Self {
            session_id,
            client_pid: 0,
            port_name,
        }
    }
}
