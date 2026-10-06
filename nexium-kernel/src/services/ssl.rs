#![cfg_attr(target_vendor = "sony", allow(dead_code))]

use crate::services::bsd::BsdService;
use nexium_ipc::IpcCtx;
use nexium_memory::AddressSpace;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub type ObjectKey = (u32, u32);

const fn ssl_result(description: u32) -> u32 {
    123 | (description << 9)
}

pub const RESULT_NO_SOCKET: u32 = ssl_result(103);
pub const RESULT_INVALID_SOCKET: u32 = ssl_result(106);
pub const RESULT_WOULD_BLOCK: u32 = ssl_result(204);
pub const RESULT_TIMEOUT: u32 = ssl_result(205);
pub const RESULT_INTERNAL: u32 = ssl_result(999);

pub struct SslReply {
    pub rc: u32,
    pub data: Vec<u8>,
    pub retry: bool,
}

enum Step {
    Done(Vec<u8>),
    Fail(u32),
    Wait {
        timeout: Option<Duration>,
        expired: (u32, Vec<u8>),
    },
}

#[derive(Clone, Copy)]
struct PendingWait {
    cmd_id: u32,
    key: ObjectKey,
    deadline: Option<Instant>,
}

pub struct SslService {
    #[cfg(not(target_vendor = "sony"))]
    connections: HashMap<ObjectKey, tls::Connection>,
    waits: HashMap<u32, PendingWait>,
}

impl SslService {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_vendor = "sony"))]
            connections: HashMap::new(),
            waits: HashMap::new(),
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("ssl cmd: {}", cmd_id);
        0
    }

    #[cfg(not(target_vendor = "sony"))]
    pub fn dispatch_connection(
        &mut self,
        bsd: &mut BsdService,
        memory: &AddressSpace,
        ctx: &IpcCtx,
        key: ObjectKey,
        thread: u32,
    ) -> Option<SslReply> {
        let cmd_id = ctx.cmif_in.cmd_id;
        let resumed = self
            .waits
            .remove(&thread)
            .filter(|wait| wait.cmd_id == cmd_id && wait.key == key);
        let connection = self.connections.entry(key).or_insert_with(tls::Connection::new);
        let step = connection.command(cmd_id, bsd, memory, ctx);
        Some(self.finish(step, cmd_id, key, thread, resumed))
    }

    #[cfg(target_vendor = "sony")]
    pub fn dispatch_connection(
        &mut self,
        _bsd: &mut BsdService,
        _memory: &AddressSpace,
        _ctx: &IpcCtx,
        _key: ObjectKey,
        _thread: u32,
    ) -> Option<SslReply> {
        None
    }

    fn finish(
        &mut self,
        step: Step,
        cmd_id: u32,
        key: ObjectKey,
        thread: u32,
        resumed: Option<PendingWait>,
    ) -> SslReply {
        match step {
            Step::Done(data) => SslReply {
                rc: 0,
                data,
                retry: false,
            },
            Step::Fail(rc) => SslReply {
                rc,
                data: failure_payload(cmd_id),
                retry: false,
            },
            Step::Wait { timeout, expired } => {
                let wait = resumed.unwrap_or(PendingWait {
                    cmd_id,
                    key,
                    deadline: timeout.map(|timeout| Instant::now() + timeout),
                });
                if wait
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return SslReply {
                        rc: expired.0,
                        data: expired.1,
                        retry: false,
                    };
                }
                self.waits.insert(thread, wait);
                SslReply {
                    rc: RESULT_WOULD_BLOCK,
                    data: failure_payload(cmd_id),
                    retry: true,
                }
            }
        }
    }

    pub fn close_object(&mut self, key: ObjectKey, bsd: &mut BsdService) {
        #[cfg(not(target_vendor = "sony"))]
        if let Some(connection) = self.connections.remove(&key) {
            connection.close(bsd);
        }
        #[cfg(target_vendor = "sony")]
        let _ = bsd;
        self.waits.retain(|_, wait| wait.key != key);
    }

    pub fn close_handle(&mut self, handle: u32, bsd: &mut BsdService) {
        #[cfg(not(target_vendor = "sony"))]
        {
            let keys: Vec<ObjectKey> = self
                .connections
                .keys()
                .filter(|key| key.0 == handle)
                .copied()
                .collect();
            for key in keys {
                self.close_object(key, bsd);
            }
        }
        #[cfg(target_vendor = "sony")]
        let _ = bsd;
        self.waits.retain(|_, wait| wait.key.0 != handle);
    }
}

impl Default for SslService {
    fn default() -> Self {
        Self::new()
    }
}

fn failure_payload(cmd_id: u32) -> Vec<u8> {
    match cmd_id {
        0 => (-1i32).to_le_bytes().to_vec(),
        9 => vec![0u8; 8],
        10..=14 => vec![0u8; 4],
        _ => Vec::new(),
    }
}

#[cfg(not(target_vendor = "sony"))]
mod tls {
    use super::{
        Step, RESULT_INTERNAL, RESULT_INVALID_SOCKET, RESULT_NO_SOCKET, RESULT_TIMEOUT,
        RESULT_WOULD_BLOCK,
    };
    use crate::services::bsd::{
        input_u32, read_guest, recv_buffer, send_buffer, BsdService, SharedSocket, POLLERR,
        POLLHUP, POLLIN, POLLOUT,
    };
    use nexium_ipc::IpcCtx;
    use nexium_memory::AddressSpace;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::crypto::CryptoProvider;
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{ClientConfig, ClientConnection, DigitallySignedStruct, RootCertStore, SignatureScheme};
    use std::io::{self, Read, Write};
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
    const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
    const MAX_WRITE: usize = 16 * 1024 * 1024;
    const POLL_READ: u32 = 1;
    const POLL_WRITE: u32 = 2;
    const POLL_EXCEPT: u32 = 4;
    const OPTION_DO_NOT_CLOSE_SOCKET: u32 = 0;
    const OPTION_GET_SERVER_CERT_CHAIN: u32 = 1;
    const IO_MODE_NONBLOCKING: u32 = 2;
    const CERT_CHAIN_MAGIC: u64 = 0x4E4D_6843_7472_6543;

    #[derive(Debug)]
    struct NextendoServerVerifier(Arc<CryptoProvider>);

    impl ServerCertVerifier for NextendoServerVerifier {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls12_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls13_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }

    fn provider() -> Arc<CryptoProvider> {
        static PROVIDER: OnceLock<Arc<CryptoProvider>> = OnceLock::new();
        PROVIDER
            .get_or_init(|| Arc::new(rustls::crypto::ring::default_provider()))
            .clone()
    }

    fn public_roots() -> Arc<RootCertStore> {
        static ROOTS: OnceLock<Arc<RootCertStore>> = OnceLock::new();
        ROOTS
            .get_or_init(|| {
                Arc::new(RootCertStore {
                    roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
                })
            })
            .clone()
    }

    fn client_config(nextendo_peer: bool, alpn: Vec<Vec<u8>>) -> Option<Arc<ClientConfig>> {
        let builder = ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .ok()?;
        let mut config = if nextendo_peer {
            builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(NextendoServerVerifier(provider())))
                .with_no_client_auth()
        } else {
            builder
                .with_root_certificates(public_roots())
                .with_no_client_auth()
        };
        config.alpn_protocols = alpn;
        Some(Arc::new(config))
    }

    fn alpn_for(host: &str, requested: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let npln = host.contains("npln") || host.contains("gs.nintendo.net");
        if npln {
            let offered: Vec<Vec<u8>> = requested
                .iter()
                .filter(|protocol| protocol.as_slice() == b"h2" || protocol.as_slice() == b"http/1.1")
                .cloned()
                .collect();
            if !offered.is_empty() {
                return offered;
            }
        }
        vec![b"http/1.1".to_vec()]
    }

    fn server_name(host: &str, peer: Option<Ipv4Addr>) -> Option<ServerName<'static>> {
        if !host.is_empty() {
            if let Ok(ip) = host.parse::<IpAddr>() {
                return Some(ServerName::IpAddress(ip.into()));
            }
            if let Ok(name) = ServerName::try_from(host.to_string()) {
                return Some(name);
            }
        }
        peer.map(|ip| ServerName::IpAddress(IpAddr::V4(ip).into()))
    }

    fn parse_alpn_wire(wire: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut cursor = 0;
        while let Some(&len) = wire.get(cursor) {
            let len = len as usize;
            let Some(protocol) = wire.get(cursor + 1..cursor + 1 + len) else {
                break;
            };
            if len == 0 {
                break;
            }
            out.push(protocol.to_vec());
            cursor += 1 + len;
        }
        out
    }

    fn word(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn would_block(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::WouldBlock
    }

    fn tls_error(error: rustls::Error) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, error)
    }

    pub(super) struct Connection {
        socket: Option<SharedSocket>,
        fd: i32,
        release_fd: Option<i32>,
        host: String,
        verify_option: u32,
        io_mode: u32,
        do_not_close_socket: bool,
        server_cert_chain: bool,
        alpn: Vec<Vec<u8>>,
        io_timeout: Option<Duration>,
        session: Option<ClientConnection>,
        nextendo_peer: bool,
        handshake_done: bool,
        inbox: Vec<u8>,
        peer_closed: bool,
        status_logged: bool,
        close_logged: bool,
        watch: Option<crate::services::nextendo_watch::NexWatch>,
    }

    impl Connection {
        pub(super) fn new() -> Self {
            Self {
                socket: None,
                fd: -1,
                release_fd: None,
                host: String::new(),
                verify_option: 0,
                io_mode: 1,
                do_not_close_socket: false,
                server_cert_chain: false,
                alpn: Vec::new(),
                io_timeout: None,
                session: None,
                nextendo_peer: false,
                handshake_done: false,
                inbox: Vec::new(),
                peer_closed: false,
                status_logged: false,
                close_logged: false,
                watch: None,
            }
        }

        fn nonblocking(&self) -> bool {
            self.io_mode == IO_MODE_NONBLOCKING
        }

        pub(super) fn command(
            &mut self,
            cmd_id: u32,
            bsd: &mut BsdService,
            memory: &AddressSpace,
            ctx: &IpcCtx,
        ) -> Step {
            match cmd_id {
                0 => self.set_socket(bsd, ctx),
                1 => {
                    self.host = send_buffer(ctx, 0)
                        .and_then(|buffer| read_guest(memory, buffer, 0x100).ok())
                        .map(|bytes| crate::services::resolver::c_string(&bytes))
                        .unwrap_or_default()
                        .replace('%', "lp1");
                    Step::Done(Vec::new())
                }
                2 => {
                    self.verify_option = input_u32(ctx, 0).unwrap_or(0);
                    Step::Done(Vec::new())
                }
                3 => {
                    self.io_mode = input_u32(ctx, 0).unwrap_or(1);
                    Step::Done(Vec::new())
                }
                4 => Step::Done(self.fd.to_le_bytes().to_vec()),
                5 => {
                    let written = recv_buffer(ctx, 0).map_or(0, |buffer| {
                        let bytes = self.host.as_bytes();
                        let len = bytes.len().min((buffer.size as usize).saturating_sub(1));
                        let mut out = bytes[..len].to_vec();
                        out.push(0);
                        memory.write(buffer.addr, &out).map_or(0, |_| len as u32)
                    });
                    Step::Done(word(written))
                }
                6 => Step::Done(word(self.verify_option)),
                7 => Step::Done(word(self.io_mode)),
                8 => self.handshake(),
                9 => self.handshake_with_certs(memory, ctx),
                10 => self.read(memory, ctx, false),
                11 => self.write(memory, ctx),
                12 => {
                    if let Err(error) = self.drain_plaintext() {
                        log::debug!("ssl: pending check failed: {error}");
                    }
                    self.publish_pending();
                    Step::Done((self.inbox.len() as i32).to_le_bytes().to_vec())
                }
                13 => self.read(memory, ctx, true),
                14 => self.poll(ctx),
                15 => Step::Done(word(0)),
                16 => Step::Done(word(self.serialized_certs().len() as u32)),
                17 | 19 | 20 | 28 | 30 | 31 => Step::Done(Vec::new()),
                18 | 21 => Step::Done(word(0)),
                22 => {
                    let option = input_u32(ctx, 0).unwrap_or(u32::MAX);
                    let value = input_u32(ctx, 1).unwrap_or(0) != 0;
                    match option {
                        OPTION_DO_NOT_CLOSE_SOCKET => self.do_not_close_socket = value,
                        OPTION_GET_SERVER_CERT_CHAIN => self.server_cert_chain = value,
                        _ => {}
                    }
                    Step::Done(Vec::new())
                }
                23 => {
                    let value = match input_u32(ctx, 0) {
                        Some(OPTION_DO_NOT_CLOSE_SOCKET) => self.do_not_close_socket,
                        Some(OPTION_GET_SERVER_CERT_CHAIN) => self.server_cert_chain,
                        _ => false,
                    };
                    Step::Done(word(u32::from(value)))
                }
                24 => Step::Done(vec![0u8; 8]),
                25 => {
                    if let Some(buffer) = recv_buffer(ctx, 0) {
                        let _ = memory.write(buffer.addr, &vec![0u8; (buffer.size as usize).min(0x100)]);
                    }
                    Step::Done(Vec::new())
                }
                26 => {
                    self.alpn = send_buffer(ctx, 0)
                        .and_then(|buffer| read_guest(memory, buffer, 0x400).ok())
                        .map(|wire| parse_alpn_wire(&wire))
                        .unwrap_or_default();
                    Step::Done(Vec::new())
                }
                27 => self.negotiated_alpn(memory, ctx),
                34 => {
                    let millis = input_u32(ctx, 0).unwrap_or(0);
                    self.io_timeout = (millis != 0).then(|| Duration::from_millis(millis as u64));
                    Step::Done(Vec::new())
                }
                35 => Step::Done(word(
                    self.io_timeout.map_or(0, |timeout| timeout.as_millis() as u32),
                )),
                other => {
                    log::debug!("ISslConnection.cmd_{} accepted without action", other);
                    Step::Done(Vec::new())
                }
            }
        }

        fn set_socket(&mut self, bsd: &mut BsdService, ctx: &IpcCtx) -> Step {
            let Some(fd) = input_u32(ctx, 0).map(|fd| fd as i32) else {
                return Step::Fail(RESULT_INVALID_SOCKET);
            };
            let owned = if self.do_not_close_socket {
                match bsd.duplicate_socket(fd) {
                    Some(duplicate) => duplicate,
                    None => return Step::Fail(RESULT_INVALID_SOCKET),
                }
            } else {
                fd
            };
            let Some(socket) = bsd.shared_socket(owned) else {
                return Step::Fail(RESULT_INVALID_SOCKET);
            };
            self.socket = Some(socket);
            self.fd = owned;
            self.release_fd = Some(owned);
            let out = if self.do_not_close_socket { owned } else { -1 };
            log::debug!("ssl: SetSocketDescriptor fd={} -> {}", fd, out);
            Step::Done(out.to_le_bytes().to_vec())
        }

        fn peer_ip(&self) -> Option<Ipv4Addr> {
            let socket = self.socket.as_ref()?;
            let address = socket.lock().os_socket().peer_addr().ok()?;
            address.as_socket_ipv4().map(|address| *address.ip())
        }

        fn start(&mut self) -> Result<(), u32> {
            if self.socket.is_none() {
                return Err(RESULT_NO_SOCKET);
            }
            let peer = self.peer_ip();
            let mut host = self.host.clone();
            if let Some(peer) = peer {
                if host.is_empty() || host == peer.to_string() {
                    if let Some(known) = nexium_common::nextendo::host_for_ip(peer) {
                        host = known;
                    }
                }
            }
            self.nextendo_peer = peer.is_some_and(nexium_common::nextendo::is_redirected_ip);
            let config = client_config(self.nextendo_peer, alpn_for(&host, &self.alpn))
                .ok_or(RESULT_INTERNAL)?;
            let name = server_name(&host, peer).ok_or(RESULT_INTERNAL)?;
            let mut session = ClientConnection::new(config, name).map_err(|error| {
                log::warn!("ssl: could not start TLS with '{}': {}", host, error);
                RESULT_INTERNAL
            })?;
            session.set_buffer_limit(None);
            log::info!(
                "ssl: TLS handshake with '{}'{}",
                host,
                if self.nextendo_peer { " (Nextendo)" } else { "" }
            );
            if self.nextendo_peer {
                self.watch = Some(crate::services::nextendo_watch::NexWatch::new(&host));
            }
            self.host = host;
            self.session = Some(session);
            Ok(())
        }

        fn drive_handshake(&mut self) -> io::Result<bool> {
            let (Some(session), Some(socket)) = (self.session.as_mut(), self.socket.as_ref()) else {
                return Err(io::Error::from(io::ErrorKind::NotConnected));
            };
            let guard = socket.lock();
            let mut io = guard.os_socket();
            loop {
                while session.wants_write() {
                    match session.write_tls(&mut io) {
                        Ok(_) => {}
                        Err(error) if would_block(&error) => return Ok(false),
                        Err(error) => return Err(error),
                    }
                }
                if !session.is_handshaking() {
                    return Ok(true);
                }
                match session.read_tls(&mut io) {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::ConnectionAborted,
                            "server closed the connection during the handshake",
                        ))
                    }
                    Ok(_) => {
                        session.process_new_packets().map_err(tls_error)?;
                    }
                    Err(error) if would_block(&error) => return Ok(false),
                    Err(error) => return Err(error),
                }
            }
        }

        fn handshake(&mut self) -> Step {
            if self.handshake_done {
                return Step::Done(Vec::new());
            }
            if self.session.is_none() {
                if let Err(rc) = self.start() {
                    return Step::Fail(rc);
                }
            }
            match self.drive_handshake() {
                Ok(true) => {
                    self.handshake_done = true;
                    if self.nextendo_peer {
                        log::info!("ssl: TLS session with '{}' is up (Nextendo)", self.host);
                    }
                    if let Err(error) = self.drain_plaintext() {
                        log::debug!("ssl: early data drain failed: {error}");
                    }
                    self.publish_pending();
                    Step::Done(Vec::new())
                }
                Ok(false) if self.nonblocking() => Step::Fail(RESULT_WOULD_BLOCK),
                Ok(false) => Step::Wait {
                    timeout: Some(self.io_timeout.unwrap_or(HANDSHAKE_TIMEOUT)),
                    expired: (RESULT_TIMEOUT, Vec::new()),
                },
                Err(error) => {
                    log::warn!("ssl: handshake with '{}' failed: {}", self.host, error);
                    Step::Fail(RESULT_INTERNAL)
                }
            }
        }

        fn serialized_certs(&self) -> Vec<u8> {
            let certs: Vec<&[u8]> = self
                .session
                .as_ref()
                .and_then(|session| session.peer_certificates())
                .map(|certs| certs.iter().map(|cert| cert.as_ref()).collect())
                .unwrap_or_default();
            if !self.server_cert_chain {
                return certs.first().map(|cert| cert.to_vec()).unwrap_or_default();
            }
            let mut out = Vec::new();
            out.extend_from_slice(&CERT_CHAIN_MAGIC.to_le_bytes());
            out.extend_from_slice(&(certs.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            let mut offset = 16 + certs.len() * 8;
            for cert in &certs {
                out.extend_from_slice(&(cert.len() as u32).to_le_bytes());
                out.extend_from_slice(&(offset as u32).to_le_bytes());
                offset += cert.len();
            }
            for cert in &certs {
                out.extend_from_slice(cert);
            }
            out
        }

        fn handshake_with_certs(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Step {
            match self.handshake() {
                Step::Done(_) => {
                    let certs = self.serialized_certs();
                    let count = self
                        .session
                        .as_ref()
                        .and_then(|session| session.peer_certificates())
                        .map_or(0, |certs| certs.len() as u32);
                    let written = recv_buffer(ctx, 0)
                        .filter(|buffer| buffer.size as usize >= certs.len())
                        .is_some_and(|buffer| memory.write(buffer.addr, &certs).is_ok());
                    let size = if written { certs.len() as u32 } else { 0 };
                    let mut out = word(size);
                    out.extend_from_slice(&count.to_le_bytes());
                    Step::Done(out)
                }
                other => other,
            }
        }

        fn drain_plaintext(&mut self) -> io::Result<()> {
            let Some(session) = self.session.as_mut() else {
                return Ok(());
            };
            let mut chunk = [0u8; 8192];
            loop {
                match session.reader().read(&mut chunk) {
                    Ok(0) => {
                        self.peer_closed = true;
                        return Ok(());
                    }
                    Ok(len) => self.inbox.extend_from_slice(&chunk[..len]),
                    Err(error) if would_block(&error) => return Ok(()),
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                        self.peer_closed = true;
                        return Ok(());
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        fn receive(&mut self) -> io::Result<bool> {
            loop {
                self.drain_plaintext()?;
                if !self.inbox.is_empty() || self.peer_closed {
                    return Ok(true);
                }
                let (Some(session), Some(socket)) = (self.session.as_mut(), self.socket.as_ref())
                else {
                    return Err(io::Error::from(io::ErrorKind::NotConnected));
                };
                let result = {
                    let guard = socket.lock();
                    let mut io = guard.os_socket();
                    session.read_tls(&mut io)
                };
                match result {
                    Ok(0) => {
                        self.peer_closed = true;
                        return Ok(true);
                    }
                    Ok(_) => {
                        session.process_new_packets().map_err(tls_error)?;
                    }
                    Err(error) if would_block(&error) => return Ok(false),
                    Err(error) => return Err(error),
                }
            }
        }

        fn flush(&mut self) -> io::Result<bool> {
            let (Some(session), Some(socket)) = (self.session.as_mut(), self.socket.as_ref()) else {
                return Ok(true);
            };
            let guard = socket.lock();
            let mut io = guard.os_socket();
            while session.wants_write() {
                match session.write_tls(&mut io) {
                    Ok(_) => {}
                    Err(error) if would_block(&error) => return Ok(false),
                    Err(error) => return Err(error),
                }
            }
            Ok(true)
        }

        fn publish_pending(&self) {
            if let Some(socket) = self.socket.as_ref() {
                socket.lock().set_tls_pending(self.inbox.len());
            }
        }

        fn receive_up_to(&mut self, capacity: usize, peek: bool) -> Result<Vec<u8>, Step> {
            if !self.handshake_done {
                return Err(Step::Fail(RESULT_INTERNAL));
            }
            if let Err(error) = self.flush() {
                log::debug!("ssl: flush before read failed: {error}");
            }
            if self.inbox.is_empty() && !self.peer_closed {
                match self.receive() {
                    Ok(true) => {}
                    Ok(false) if self.nonblocking() => return Err(Step::Fail(RESULT_WOULD_BLOCK)),
                    Ok(false) => {
                        return Err(Step::Wait {
                            timeout: self.io_timeout,
                            expired: (RESULT_TIMEOUT, word(0)),
                        })
                    }
                    Err(error) => {
                        if self.nextendo_peer {
                            log::info!("ssl: read from '{}' failed: {}", self.host, error);
                        } else {
                            log::debug!("ssl: read from '{}' failed: {}", self.host, error);
                        }
                        self.publish_pending();
                        return Err(Step::Fail(RESULT_INTERNAL));
                    }
                }
            }
            let len = self.inbox.len().min(capacity);
            let out = if peek {
                self.inbox[..len].to_vec()
            } else {
                self.inbox.drain(..len).collect()
            };
            self.publish_pending();
            self.report_nextendo_progress(&out);
            if !peek {
                if let Some(watch) = self.watch.as_mut() {
                    watch.incoming(&out);
                }
            }
            Ok(out)
        }

        fn report_nextendo_progress(&mut self, received: &[u8]) {
            if !self.nextendo_peer {
                return;
            }
            if !self.status_logged && !received.is_empty() {
                self.status_logged = true;
                if received.starts_with(b"HTTP/") {
                    let line = received.split(|&byte| byte == b'\r' || byte == b'\n').next().unwrap_or_default();
                    log::info!("ssl: '{}' answered {}", self.host, String::from_utf8_lossy(line));
                }
            }
            if self.peer_closed && self.inbox.is_empty() && !self.close_logged {
                self.close_logged = true;
                log::info!("ssl: '{}' closed the connection", self.host);
            }
        }

        fn read(&mut self, memory: &AddressSpace, ctx: &IpcCtx, peek: bool) -> Step {
            let Some(buffer) = recv_buffer(ctx, 0) else {
                return Step::Fail(RESULT_INTERNAL);
            };
            let bytes = match self.receive_up_to(buffer.size as usize, peek) {
                Ok(bytes) => bytes,
                Err(step) => return step,
            };
            if !bytes.is_empty() && memory.write(buffer.addr, &bytes).is_err() {
                return Step::Fail(RESULT_INTERNAL);
            }
            Step::Done(word(bytes.len() as u32))
        }

        fn send(&mut self, data: &[u8]) -> Result<usize, Step> {
            if !self.handshake_done {
                return Err(Step::Fail(RESULT_INTERNAL));
            }
            match self.flush() {
                Ok(true) => {}
                Ok(false) if self.nonblocking() => return Err(Step::Fail(RESULT_WOULD_BLOCK)),
                Ok(false) => {
                    return Err(Step::Wait {
                        timeout: Some(self.io_timeout.unwrap_or(WRITE_TIMEOUT)),
                        expired: (RESULT_TIMEOUT, word(0)),
                    })
                }
                Err(error) => {
                    log::debug!("ssl: write to '{}' failed: {}", self.host, error);
                    return Err(Step::Fail(RESULT_INTERNAL));
                }
            }
            let rewritten = self.rewrite_station_urls(data);
            let payload = rewritten.as_deref().unwrap_or(data);
            let Some(session) = self.session.as_mut() else {
                return Err(Step::Fail(RESULT_INTERNAL));
            };
            if let Err(error) = session.writer().write_all(payload) {
                log::debug!("ssl: could not queue {} bytes: {}", payload.len(), error);
                return Err(Step::Fail(RESULT_INTERNAL));
            }
            if let Some(watch) = self.watch.as_mut() {
                watch.outgoing(payload);
            }
            if let Err(error) = self.flush() {
                if self.nextendo_peer {
                    log::info!("ssl: write to '{}' failed: {}", self.host, error);
                } else {
                    log::debug!("ssl: write to '{}' failed: {}", self.host, error);
                }
                return Err(Step::Fail(RESULT_INTERNAL));
            }
            Ok(data.len())
        }

        fn write(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Step {
            let Some(data) =
                send_buffer(ctx, 0).and_then(|buffer| read_guest(memory, buffer, MAX_WRITE).ok())
            else {
                return Step::Fail(RESULT_INTERNAL);
            };
            match self.send(&data) {
                Ok(len) => Step::Done(word(len as u32)),
                Err(step) => step,
            }
        }

        fn rewrite_station_urls(&self, data: &[u8]) -> Option<Vec<u8>> {
            if !self.nextendo_peer {
                return None;
            }
            let outgoing = crate::services::nextendo_nat::inspect_outgoing(
                data,
                nexium_common::nextendo::external_ip(),
            )?;
            let nat = outgoing
                .nat
                .map_or_else(String::new, |(filtering, mapping)| {
                    format!(" (natf={filtering} natm={mapping})")
                });
            if outgoing.rewritten.is_some() {
                log::info!("ssl: {} now advertises this console's public address{}", outgoing.method, nat);
            } else {
                log::info!("ssl: {} sent unchanged{}", outgoing.method, nat);
            }
            outgoing.rewritten
        }

        fn poll(&mut self, ctx: &IpcCtx) -> Step {
            let wanted = input_u32(ctx, 0).unwrap_or(0);
            let timeout_ms = input_u32(ctx, 1).unwrap_or(0);
            let Some(socket) = self.socket.clone() else {
                return Step::Done(word(POLL_EXCEPT));
            };
            if let Err(error) = self.flush() {
                log::debug!("ssl: flush during poll failed: {error}");
            }
            let _ = self.drain_plaintext();
            self.publish_pending();
            let mut ready = 0;
            if wanted & POLL_READ != 0 && (!self.inbox.is_empty() || self.peer_closed) {
                ready |= POLL_READ;
            }
            let mut events = 0u16;
            if wanted & POLL_READ != 0 {
                events |= POLLIN;
            }
            if wanted & POLL_WRITE != 0 {
                events |= POLLOUT;
            }
            if ready == 0 && events != 0 {
                let revents = BsdService::poll_socket(&socket, events);
                if wanted & POLL_READ != 0 && revents & (POLLIN | POLLHUP) != 0 {
                    ready |= POLL_READ;
                }
                if wanted & POLL_WRITE != 0 && revents & POLLOUT != 0 {
                    ready |= POLL_WRITE;
                }
                if revents & POLLERR != 0 {
                    ready |= POLL_EXCEPT;
                }
            }
            if ready != 0 || timeout_ms == 0 {
                return Step::Done(word(ready));
            }
            Step::Wait {
                timeout: (timeout_ms != u32::MAX).then(|| Duration::from_millis(timeout_ms as u64)),
                expired: (0, word(0)),
            }
        }

        fn negotiated_alpn(&self, memory: &AddressSpace, ctx: &IpcCtx) -> Step {
            let protocol = self
                .session
                .as_ref()
                .and_then(|session| session.alpn_protocol())
                .map(|protocol| protocol.to_vec());
            let Some(protocol) = protocol else {
                return Step::Done(vec![0u8; 8]);
            };
            let len = recv_buffer(ctx, 0).map_or(0, |buffer| {
                let len = protocol.len().min(buffer.size as usize);
                memory
                    .write(buffer.addr, &protocol[..len])
                    .map_or(0, |_| len)
            });
            let mut out = word(1);
            out.extend_from_slice(&(len as u32).to_le_bytes());
            Step::Done(out)
        }

        pub(super) fn close(mut self, bsd: &mut BsdService) {
            if self.handshake_done && !self.peer_closed {
                if let Some(session) = self.session.as_mut() {
                    session.send_close_notify();
                }
                let _ = self.flush();
            }
            if let Some(socket) = self.socket.as_ref() {
                socket.lock().set_tls_pending(0);
                if let Some(fd) = self.release_fd {
                    bsd.close_if_same(fd, socket);
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn alpn_wire_is_parsed_in_order() {
            assert_eq!(
                parse_alpn_wire(b"\x02h2\x08http/1.1"),
                vec![b"h2".to_vec(), b"http/1.1".to_vec()]
            );
            assert_eq!(parse_alpn_wire(b"\x05ab"), Vec::<Vec<u8>>::new());
            assert!(parse_alpn_wire(&[]).is_empty());
        }

        #[test]
        fn nex_hosts_pin_http_1_1_and_npln_honors_the_game() {
            let requested = vec![b"h2".to_vec(), b"spdy/3".to_vec()];
            assert_eq!(
                alpn_for("g2b309e01-lp1.s.n.srv.nintendo.net", &requested),
                vec![b"http/1.1".to_vec()]
            );
            assert_eq!(
                alpn_for("app.lp1.npln.srv.nintendo.net", &requested),
                vec![b"h2".to_vec()]
            );
            assert_eq!(
                alpn_for("app.lp1.npln.srv.nintendo.net", &[]),
                vec![b"http/1.1".to_vec()]
            );
        }

        #[test]
        fn server_names_fall_back_to_the_peer_address() {
            assert!(matches!(
                server_name("example.com", None),
                Some(ServerName::DnsName(_))
            ));
            assert!(matches!(
                server_name("", Some(Ipv4Addr::LOCALHOST)),
                Some(ServerName::IpAddress(_))
            ));
            assert!(matches!(
                server_name("10.0.0.1", None),
                Some(ServerName::IpAddress(_))
            ));
            assert!(server_name("", None).is_none());
        }

        #[test]
        fn both_trust_modes_build_a_config() {
            assert!(client_config(true, vec![b"http/1.1".to_vec()]).is_some());
            assert!(client_config(false, Vec::new()).is_some());
        }

        fn settle<T>(mut step: impl FnMut() -> Result<T, Step>) -> T {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                match step() {
                    Ok(value) => return value,
                    Err(Step::Fail(RESULT_WOULD_BLOCK)) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(Step::Fail(rc)) => panic!("ssl step failed with {rc:#x}"),
                    Err(_) => panic!("unexpected wait"),
                }
            }
        }

        #[test]
        #[ignore = "needs network access"]
        fn speaks_https_through_a_guest_socket() {
            use std::net::ToSocketAddrs;
            let address = ("nextendo.network", 443)
                .to_socket_addrs()
                .unwrap()
                .find(|address| address.is_ipv4())
                .unwrap();
            let socket = socket2::Socket::new(
                socket2::Domain::IPV4,
                socket2::Type::STREAM,
                Some(socket2::Protocol::TCP),
            )
            .unwrap();
            socket.connect(&address.into()).unwrap();
            let mut connection = Connection::new();
            connection.socket = Some(crate::services::bsd::test_shared_socket(socket));
            connection.host = "nextendo.network".into();
            connection.io_mode = IO_MODE_NONBLOCKING;
            settle(|| match connection.handshake() {
                Step::Done(_) => Ok(()),
                other => Err(other),
            });
            assert!(!connection.nextendo_peer);
            let request = b"GET /api/health HTTP/1.1\r\nHost: nextendo.network\r\nUser-Agent: NeXium\r\nConnection: close\r\n\r\n";
            assert_eq!(settle(|| connection.send(request)), request.len());
            let mut response = Vec::new();
            loop {
                let chunk = settle(|| connection.receive_up_to(4096, false));
                if chunk.is_empty() {
                    break;
                }
                response.extend_from_slice(&chunk);
            }
            let text = String::from_utf8_lossy(&response);
            assert!(text.starts_with("HTTP/1.1 200"), "{text}");
            assert!(text.contains(r#""ok":true"#), "{text}");
        }
    }
}
