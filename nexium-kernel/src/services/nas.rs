use crate::kernel::svc::{
    application_version, build_ipc_response, indexed_ipc_buffer, nextendo_identity,
    read_ipc_bytes, return_subsession,
};
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;
use std::sync::Mutex;

const PARAMETERS_SIZE: usize = 0x200;
const STATE_OFFSET: usize = 0x100;
const STATE_SIZE: usize = 0x80;
const USER_BASE_SIZE: usize = 0x68;
const SHARED_BIRTHDAY: &[u8] = b"2000-01-01";
const TOKEN_FILL_LIMIT: usize = 0x4000;
const RESOURCE_FILL_LIMIT: usize = 0x10000;

struct Authorization {
    state: [u8; STATE_SIZE],
    parameters_address: u64,
    code: String,
}

static AUTHORIZATION: Mutex<Option<Authorization>> = Mutex::new(None);

pub(crate) fn dispatch(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session_handle: u32,
    port_name: &str,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    match (port_name, cmd_id) {
        ("IManagerForApplication" | "IManagerForSystemService", 1 | 3 | 4 | 130 | 136) => {
            let identity = nextendo_identity(kernel)?;
            account(kernel, ctx, cmd_id, &identity)
        }
        ("IManagerForApplication", 150) => Some(create_authorization_request(kernel, ctx, session_handle)),
        ("IAuthorizationRequest", _) => Some(authorization_request(kernel, ctx, session_handle, cmd_id)),
        _ => None,
    }
}

pub fn user_base() -> [u8; USER_BASE_SIZE] {
    let mut base = [0u8; USER_BASE_SIZE];
    base[..SHARED_BIRTHDAY.len()].copy_from_slice(SHARED_BIRTHDAY);
    base
}

pub fn padded_token(token: &[u8], capacity: usize) -> Vec<u8> {
    let mut out = vec![0u8; capacity.min(TOKEN_FILL_LIMIT).max(token.len().min(capacity))];
    let len = token.len().min(out.len());
    out[..len].copy_from_slice(&token[..len]);
    out
}

fn account(
    kernel: &mut Kernel,
    ctx: &IpcCtx,
    cmd_id: u32,
    account: &nexium_common::nextendo::LinkedAccount,
) -> Option<Vec<u8>> {
    match cmd_id {
        1 => Some(build_ipc_response(ctx, 0, &account.pid.to_le_bytes(), &[])),
        130 | 136 => {
            if let Some(buffer) = ctx
                .recv_statics
                .iter()
                .find(|buffer| buffer.addr != 0 && buffer.size as usize >= USER_BASE_SIZE)
            {
                if kernel.address_space.write(buffer.addr, &user_base()).is_ok() {
                    log::info!("acc: NasUserBase answered with the shared birthday");
                }
            }
            if let Some(buffer) = ctx.recv_buffers.iter().find(|buffer| buffer.addr != 0 && buffer.size > 0) {
                let zeros = vec![0u8; (buffer.size as usize).min(RESOURCE_FILL_LIMIT)];
                let _ = kernel.address_space.write(buffer.addr, &zeros);
            }
            Some(build_ipc_response(ctx, 0, &account.pid.to_le_bytes(), &[]))
        }
        _ => {
            let token = id_token(kernel)?;
            let output = indexed_ipc_buffer(&ctx.recv_buffers, &ctx.recv_statics, 0)?;
            let bytes = padded_token(token.as_bytes(), output.size as usize);
            if kernel.address_space.write(output.addr, &bytes).is_err() {
                return None;
            }
            let len = token.len().min(bytes.len());
            log::info!("acc: LoadIdTokenCache served the Nextendo id_token ({} bytes)", len);
            Some(build_ipc_response(ctx, 0, &(len as u32).to_le_bytes(), &[]))
        }
    }
}

fn id_token(kernel: &Kernel) -> Option<String> {
    crate::services::nextendo_token::id_token(kernel.title_id, &application_version(kernel))
}

fn random_code() -> String {
    let mut bytes = [0u8; 16];
    if getrandom::fill(&mut bytes).is_err() {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (seed >> ((index % 16) * 8)) as u8 ^ (index as u8).wrapping_mul(0x5B);
        }
    }
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("nextendo-{hex}")
}

fn create_authorization_request(kernel: &mut Kernel, ctx: &mut IpcCtx, session_handle: u32) -> Vec<u8> {
    let parameters_buffer = indexed_ipc_buffer(&ctx.send_buffers, &ctx.send_statics, 0);
    let parameters = read_ipc_bytes(kernel, parameters_buffer, PARAMETERS_SIZE);
    let mut state = [0u8; STATE_SIZE];
    if parameters.len() >= STATE_OFFSET + STATE_SIZE {
        state.copy_from_slice(&parameters[STATE_OFFSET..STATE_OFFSET + STATE_SIZE]);
        let scope_end = parameters[..STATE_OFFSET]
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(STATE_OFFSET);
        log::info!(
            "acc: Nintendo Account authorization requested, scope '{}'",
            String::from_utf8_lossy(&parameters[..scope_end])
        );
    }
    if let Ok(mut slot) = AUTHORIZATION.lock() {
        *slot = Some(Authorization {
            state,
            parameters_address: parameters_buffer.map_or(0, |buffer| buffer.addr),
            code: random_code(),
        });
    }
    return_subsession(kernel, ctx, session_handle, "IAuthorizationRequest")
}

fn write_string(kernel: &Kernel, ctx: &IpcCtx, value: &[u8]) -> Vec<u8> {
    let written = indexed_ipc_buffer(&ctx.recv_buffers, &ctx.recv_statics, 0)
        .map(|buffer| {
            let len = value.len().min(buffer.size as usize);
            kernel
                .address_space
                .write(buffer.addr, &value[..len])
                .map_or(0, |_| len)
        })
        .unwrap_or(0);
    log::info!(
        "acc: Nintendo Account authorization returned {} of {} bytes",
        written,
        value.len()
    );
    build_ipc_response(ctx, 0, &(written as u32).to_le_bytes(), &[])
}

fn authorization_request(kernel: &mut Kernel, ctx: &mut IpcCtx, session_handle: u32, cmd_id: u32) -> Vec<u8> {
    match cmd_id {
        0 => build_ipc_response(ctx, 0, &[0u8; 16], &[]),
        10 => {
            log::info!("acc: Nintendo Account authorization invoked");
            return_subsession(kernel, ctx, session_handle, "IAsyncContext")
        }
        19 => build_ipc_response(ctx, 0, &[1u8], &[]),
        20 => {
            let code = AUTHORIZATION
                .lock()
                .ok()
                .and_then(|slot| slot.as_ref().map(|auth| auth.code.clone()))
                .unwrap_or_default();
            write_string(kernel, ctx, code.as_bytes())
        }
        21 => {
            let token = id_token(kernel).unwrap_or_default();
            write_string(kernel, ctx, token.as_bytes())
        }
        22 => {
            let mut state = [0u8; STATE_SIZE];
            if let Ok(mut slot) = AUTHORIZATION.lock() {
                if let Some(auth) = slot.as_mut() {
                    if auth.parameters_address != 0 {
                        let mut live = [0u8; STATE_SIZE];
                        if kernel
                            .address_space
                            .read(auth.parameters_address + STATE_OFFSET as u64, &mut live)
                            .is_ok()
                        {
                            auth.state = live;
                        }
                    }
                    state = auth.state;
                }
            }
            if let Some(buffer) = indexed_ipc_buffer(&ctx.recv_statics, &ctx.recv_buffers, 0) {
                let len = state.len().min(buffer.size as usize);
                let _ = kernel.address_space.write(buffer.addr, &state[..len]);
            }
            build_ipc_response(ctx, 0, &[], &[])
        }
        _ => {
            log::warn!("IAuthorizationRequest.cmd_{} is not implemented; returning success", cmd_id);
            build_ipc_response(ctx, 0, &[], &[])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_base_starts_with_the_shared_birthday() {
        let base = user_base();
        assert_eq!(base.len(), 0x68);
        assert_eq!(&base[..10], b"2000-01-01");
        assert!(base[10..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn tokens_are_zero_padded_to_the_caller_buffer() {
        let padded = padded_token(b"abc", 8);
        assert_eq!(padded, b"abc\0\0\0\0\0");
        assert_eq!(padded_token(b"abcdef", 4), b"abcd");
        assert_eq!(padded_token(b"", 0), Vec::<u8>::new());
        assert_eq!(padded_token(b"x", TOKEN_FILL_LIMIT * 2).len(), TOKEN_FILL_LIMIT);
    }

    #[test]
    fn authorization_requests_hand_out_code_state_and_async_context() {
        use crate::services::resolver::harness::Harness;
        let mut h = Harness::new("IManagerForApplication");
        let mut parameters = vec![0u8; PARAMETERS_SIZE];
        parameters[..6].copy_from_slice(b"openid");
        parameters[STATE_OFFSET..STATE_OFFSET + STATE_SIZE].fill(0x5A);
        let input = h.input(&parameters);
        let mut ctx = h.request(150, &[]);
        ctx.send_buffers.push(input);
        let response = dispatch(&mut h.kernel, &mut ctx, h.session, "IManagerForApplication", 150).unwrap();
        assert_eq!(Harness::result(&response), 0);
        let object = Harness::moved_handle(&response);
        assert_eq!(h.kernel.sessions[&object].port_name, "IAuthorizationRequest");

        let mut authorized = h.request(19, &[]);
        let response = dispatch(&mut h.kernel, &mut authorized, object, "IAuthorizationRequest", 19).unwrap();
        assert_eq!(Harness::data(&response)[0], 1);

        let mut session_id = h.request(0, &[]);
        let response = dispatch(&mut h.kernel, &mut session_id, object, "IAuthorizationRequest", 0).unwrap();
        assert_eq!(&Harness::data(&response)[..16], &[0u8; 16]);

        let code_out = h.output(64);
        let mut code = h.request(20, &[]);
        code.recv_buffers.push(code_out);
        let response = dispatch(&mut h.kernel, &mut code, object, "IAuthorizationRequest", 20).unwrap();
        assert_eq!(&Harness::data(&response)[..4], &41u32.to_le_bytes());
        assert_eq!(&h.read(code_out)[..9], b"nextendo-");

        let state_out = h.output(STATE_SIZE as u64);
        let mut state = h.request(22, &[]);
        state.recv_statics.push(state_out);
        let response = dispatch(&mut h.kernel, &mut state, object, "IAuthorizationRequest", 22).unwrap();
        assert_eq!(Harness::result(&response), 0);
        assert!(h.read(state_out).iter().all(|byte| *byte == 0x5A));
        h.kernel
            .address_space
            .write(input.addr + STATE_OFFSET as u64, &[0x33u8; STATE_SIZE])
            .unwrap();
        let mut again = h.request(22, &[]);
        again.recv_statics.push(state_out);
        dispatch(&mut h.kernel, &mut again, object, "IAuthorizationRequest", 22).unwrap();
        assert!(h.read(state_out).iter().all(|byte| *byte == 0x33));

        let mut invoke = h.request(10, &[]);
        let response = dispatch(&mut h.kernel, &mut invoke, object, "IAuthorizationRequest", 10).unwrap();
        let context = Harness::moved_handle(&response);
        assert_eq!(h.kernel.sessions[&context].port_name, "IAsyncContext");
        let mut unrelated = h.request(0, &[]);
        assert!(dispatch(&mut h.kernel, &mut unrelated, h.session, "IManagerForApplication", 0).is_none());
    }

    #[test]
    fn account_cache_answers_the_user_base_and_zeroes_the_resource() {
        use crate::services::resolver::harness::Harness;
        let mut h = Harness::new("IManagerForApplication");
        let identity = nexium_common::nextendo::LinkedAccount {
            pid: 1_800_000_001,
            ..Default::default()
        };
        let base_out = h.output(USER_BASE_SIZE as u64);
        let resource = h.output(0x40);
        h.kernel.address_space.write(resource.addr, &[0xFFu8; 0x40]).unwrap();
        let mut ctx = h.request(130, &[]);
        ctx.recv_statics.push(base_out);
        ctx.recv_buffers.push(resource);
        let response = account(&mut h.kernel, &ctx, 130, &identity).unwrap();
        assert_eq!(Harness::result(&response), 0);
        assert_eq!(&Harness::data(&response)[..8], &identity.pid.to_le_bytes());
        assert_eq!(&h.read(base_out)[..10], b"2000-01-01");
        assert!(h.read(resource).iter().all(|byte| *byte == 0));
        let mut plain = h.request(1, &[]);
        let response = account(&mut h.kernel, &mut plain, 1, &identity).unwrap();
        assert_eq!(&Harness::data(&response)[..8], &identity.pid.to_le_bytes());
    }

    #[test]
    fn authorization_codes_are_random_hex() {
        let code = random_code();
        assert!(code.starts_with("nextendo-"));
        assert_eq!(code.len(), 9 + 32);
        assert!(code[9..].chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_ne!(code, random_code());
    }
}
