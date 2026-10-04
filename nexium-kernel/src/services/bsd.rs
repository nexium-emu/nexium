use nexium_ipc::{IpcBuffer, IpcCtx};
use nexium_memory::AddressSpace;
use parking_lot::Mutex;
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use std::collections::HashMap;
use std::io;
use std::mem::MaybeUninit;
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_SOCKETS: usize = 128;
const MAX_MESSAGE: usize = 16 * 1024 * 1024;

const AF_INET: u32 = 2;
const SOCK_STREAM: u32 = 1;
const SOCK_DGRAM: u32 = 2;
const IPPROTO_TCP: u32 = 6;
const IPPROTO_UDP: u32 = 17;

const EINTR: u32 = 4;
const EIO: u32 = 5;
const EBADF: u32 = 9;
const EAGAIN: u32 = 11;
const EACCES: u32 = 13;
const EFAULT: u32 = 14;
const EINVAL: u32 = 22;
const EMFILE: u32 = 24;
const EPIPE: u32 = 32;
const ENOTSOCK: u32 = 88;
const EDESTADDRREQ: u32 = 89;
const EMSGSIZE: u32 = 90;
const ESOCKTNOSUPPORT: u32 = 94;
const EOPNOTSUPP: u32 = 95;
const EAFNOSUPPORT: u32 = 97;
const EADDRINUSE: u32 = 98;
const EADDRNOTAVAIL: u32 = 99;
const ENETDOWN: u32 = 100;
const ENETUNREACH: u32 = 101;
const ENETRESET: u32 = 102;
const ECONNABORTED: u32 = 103;
const ECONNRESET: u32 = 104;
const ENOBUFS: u32 = 105;
const EISCONN: u32 = 106;
const ENOTCONN: u32 = 107;
const ESHUTDOWN: u32 = 108;
const ETIMEDOUT: u32 = 110;
const ECONNREFUSED: u32 = 111;
const EHOSTDOWN: u32 = 112;
const EHOSTUNREACH: u32 = 113;
const EALREADY: u32 = 114;
const EINPROGRESS: u32 = 115;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

const MSG_PEEK: u32 = 2;
const MSG_DONTWAIT: u32 = 0x80;
const O_NONBLOCK: i32 = 0x800;
const SOCK_NONBLOCK_FLAG: u32 = 0x2000_0000;

const POLLIN: u16 = 0x01;
const POLLPRI: u16 = 0x02;
const POLLOUT: u16 = 0x04;
const POLLERR: u16 = 0x08;
const POLLHUP: u16 = 0x10;
const POLLNVAL: u16 = 0x20;
const POLLRDNORM: u16 = 0x40;
const POLLRDBAND: u16 = 0x80;
const POLLWRBAND: u16 = 0x100;

const SOL_SOCKET: u32 = 0xffff;
const SO_REUSEADDR: u32 = 0x4;
const SO_KEEPALIVE: u32 = 0x8;
const SO_BROADCAST: u32 = 0x20;
const SO_LINGER: u32 = 0x80;
const SO_NOSIGPIPE: u32 = 0x800;
const SO_SNDBUF: u32 = 0x1001;
const SO_RCVBUF: u32 = 0x1002;
const SO_SNDTIMEO: u32 = 0x1005;
const SO_RCVTIMEO: u32 = 0x1006;
const SO_ERROR: u32 = 0x1007;
const SO_TYPE: u32 = 0x1008;
const TCP_NODELAY: u32 = 0x1;

const SHUT_RD: i32 = 0;
const SHUT_WR: i32 = 1;
const SHUT_RDWR: i32 = 2;

const FCNTL_GETFL: i32 = 3;
const FCNTL_SETFL: i32 = 4;

const EFD_SEMAPHORE: u32 = 1;
const EFD_NONBLOCK: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BsdAddress {
    ip: [u8; 4],
    port: u16,
}

impl BsdAddress {
    fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 || bytes[1] as u32 != AF_INET {
            return None;
        }
        Some(Self {
            ip: [bytes[4], bytes[5], bytes[6], bytes[7]],
            port: u16::from_be_bytes([bytes[2], bytes[3]]),
        })
    }

    fn encode(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0] = 16;
        out[1] = AF_INET as u8;
        out[2..4].copy_from_slice(&self.port.to_be_bytes());
        out[4..8].copy_from_slice(&self.ip);
        out
    }

    fn to_socket_addr(self) -> SocketAddrV4 {
        SocketAddrV4::new(Ipv4Addr::from(self.ip), self.port)
    }

    fn to_sock_addr(self) -> SockAddr {
        SockAddr::from(SocketAddr::V4(self.to_socket_addr()))
    }

    fn from_socket_addr(address: SocketAddrV4) -> Self {
        Self {
            ip: address.ip().octets(),
            port: address.port(),
        }
    }

    fn from_sock_addr(address: &SockAddr) -> Option<Self> {
        address.as_socket_ipv4().map(Self::from_socket_addr)
    }

    fn any(port: u16) -> Self {
        Self { ip: [0; 4], port }
    }
}

struct HostSocket {
    socket: Socket,
    socket_type: u32,
    flags: i32,
    connecting: bool,
    pending_error: Option<u32>,
    recv_timeout: Option<Duration>,
    send_timeout: Option<Duration>,
}

impl HostSocket {
    fn new(socket: Socket, socket_type: u32, nonblocking: bool) -> io::Result<Self> {
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            socket_type,
            flags: if nonblocking { O_NONBLOCK } else { 0 },
            connecting: false,
            pending_error: None,
            recv_timeout: None,
            send_timeout: None,
        })
    }

    fn guest_blocking(&self, message_flags: u32) -> bool {
        self.flags & O_NONBLOCK == 0 && message_flags & MSG_DONTWAIT == 0
    }

    fn is_stream(&self) -> bool {
        self.socket_type == SOCK_STREAM
    }
}

type SharedSocket = Arc<Mutex<HostSocket>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EventFd {
    counter: u64,
    semaphore: bool,
    nonblocking: bool,
}

impl EventFd {
    fn new(initial: u64, flags: u32) -> Self {
        Self {
            counter: initial,
            semaphore: flags & EFD_SEMAPHORE != 0,
            nonblocking: flags & EFD_NONBLOCK != 0,
        }
    }

    fn write(&mut self, value: u64) -> Result<(), u32> {
        if value == u64::MAX {
            return Err(EINVAL);
        }
        match self.counter.checked_add(value) {
            Some(counter) if counter != u64::MAX => {
                self.counter = counter;
                Ok(())
            }
            _ => Err(EAGAIN),
        }
    }

    fn read(&mut self) -> Option<u64> {
        if self.counter == 0 {
            return None;
        }
        if self.semaphore {
            self.counter -= 1;
            Some(1)
        } else {
            Some(std::mem::take(&mut self.counter))
        }
    }

    fn poll_events(&self, events: u16) -> u16 {
        let readable = if self.counter > 0 { POLLIN | POLLRDNORM } else { 0 };
        events & (readable | POLLOUT)
    }
}

#[derive(Clone, Copy)]
struct PendingWait {
    cmd_id: u32,
    deadline: Option<Instant>,
}

pub struct BsdReply {
    pub rc: u32,
    pub data: Vec<u8>,
    pub retry: bool,
}

enum Outcome {
    Done(Vec<u8>),
    Wait {
        timeout: Option<Duration>,
        fallback: Vec<u8>,
    },
}

fn done(ret: i32, errno: u32) -> Outcome {
    Outcome::Done(bsd_result(ret, errno))
}

fn done_with_len(ret: i32, errno: u32, len: u32) -> Outcome {
    Outcome::Done(bsd_result_with_len(ret, errno, len))
}

enum ConnectStatus {
    Pending,
    Connected,
    Failed(u32),
}

pub struct BsdService {
    sockets: Vec<Option<SharedSocket>>,
    event_fds: Vec<Option<EventFd>>,
    waits: HashMap<u32, PendingWait>,
    active_wait: Option<PendingWait>,
}

impl BsdService {
    pub fn new() -> Self {
        Self {
            sockets: (0..MAX_SOCKETS).map(|_| None).collect(),
            event_fds: vec![None; MAX_SOCKETS],
            waits: HashMap::new(),
            active_wait: None,
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("bsd legacy cmd: {}", cmd_id);
        0
    }

    pub fn dispatch_ipc(&mut self, memory: &AddressSpace, ctx: &IpcCtx, thread: u32) -> BsdReply {
        let cmd_id = ctx.cmif_in.cmd_id;
        self.active_wait = self
            .waits
            .remove(&thread)
            .filter(|wait| wait.cmd_id == cmd_id);
        let outcome = match cmd_id {
            0 => Outcome::Done(0u32.to_le_bytes().to_vec()),
            1 => Outcome::Done(Vec::new()),
            2 | 3 => self.socket_command(ctx),
            5 => self.select_command(memory, ctx),
            6 => self.poll_command(memory, ctx),
            8 => self.recv_command(memory, ctx, false),
            9 => self.recv_command(memory, ctx, true),
            10 => self.send_command(memory, ctx, false),
            11 => self.send_command(memory, ctx, true),
            12 => self.accept_command(memory, ctx),
            13 => self.bind_command(memory, ctx),
            14 => self.connect_command(memory, ctx),
            15 => self.name_command(memory, ctx, true),
            16 => self.name_command(memory, ctx, false),
            17 => self.get_sock_opt_command(memory, ctx),
            18 => self.listen_command(ctx),
            19 => done(0, 0),
            20 => self.fcntl_command(ctx),
            21 => self.set_sock_opt_command(memory, ctx),
            22 => self.shutdown_command(ctx),
            23 => done(0, 0),
            24 => self.send_command(memory, ctx, false),
            25 => self.recv_command(memory, ctx, false),
            26 => self.close_command(ctx),
            27 => self.duplicate_command(ctx),
            31 => self.event_fd_command(ctx),
            _ => {
                log::warn!("bsd:u cmd_{} unsupported", cmd_id);
                done(-1, EINVAL)
            }
        };
        match outcome {
            Outcome::Done(data) => BsdReply {
                rc: 0,
                data,
                retry: false,
            },
            Outcome::Wait { timeout, fallback } => {
                let wait = self.active_wait.take().unwrap_or(PendingWait {
                    cmd_id,
                    deadline: timeout.map(|timeout| Instant::now() + timeout),
                });
                if wait
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return BsdReply {
                        rc: 0,
                        data: fallback,
                        retry: false,
                    };
                }
                self.waits.insert(thread, wait);
                BsdReply {
                    rc: 0,
                    data: fallback,
                    retry: true,
                }
            }
        }
    }

    fn socket_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(domain) = input_u32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(raw_type) = input_u32(ctx, 1) else {
            return done(-1, EINVAL);
        };
        let Some(protocol) = input_u32(ctx, 2) else {
            return done(-1, EINVAL);
        };
        let socket_type = raw_type & 0xff;
        if domain != AF_INET {
            return done(-1, EAFNOSUPPORT);
        }
        let host_type = match socket_type {
            SOCK_STREAM => Type::STREAM,
            SOCK_DGRAM => Type::DGRAM,
            _ => return done(-1, ESOCKTNOSUPPORT),
        };
        let host_protocol = match protocol {
            0 => None,
            IPPROTO_TCP => Some(Protocol::TCP),
            IPPROTO_UDP => Some(Protocol::UDP),
            _ => return done(-1, EINVAL),
        };
        let Some(fd) = self.free_slot() else {
            return done(-1, EMFILE);
        };
        let socket = match Socket::new(Domain::IPV4, host_type, host_protocol)
            .and_then(|socket| HostSocket::new(socket, socket_type, raw_type & SOCK_NONBLOCK_FLAG != 0))
        {
            Ok(socket) => socket,
            Err(error) => {
                log::warn!("bsd.Socket host failure: {}", error);
                return done(-1, errno_from(&error));
            }
        };
        self.sockets[fd] = Some(Arc::new(Mutex::new(socket)));
        log::debug!(
            "bsd.Socket fd={} domain={} type={} protocol={}",
            fd,
            domain,
            socket_type,
            protocol
        );
        done(fd as i32, 0)
    }

    fn bind_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(address) = read_address(memory, ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let socket = shared.lock();
        let mut result = socket.socket.bind(&address.to_sock_addr());
        if result
            .as_ref()
            .is_err_and(|error| error.kind() == io::ErrorKind::AddrNotAvailable)
            && address.ip != [0; 4]
        {
            result = socket
                .socket
                .bind(&BsdAddress::any(address.port).to_sock_addr());
        }
        match result {
            Ok(()) => {
                log::debug!("bsd.Bind fd={} address={:?}", fd, address);
                done(0, 0)
            }
            Err(error) => {
                log::debug!("bsd.Bind fd={} address={:?} failed: {}", fd, address, error);
                done(-1, errno_from(&error))
            }
        }
    }

    fn connect_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(address) = read_address(memory, ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let mut socket = shared.lock();
        if socket.connecting {
            return match connect_status(&socket.socket) {
                ConnectStatus::Pending => {
                    if socket.guest_blocking(0) {
                        Outcome::Wait {
                            timeout: Some(CONNECT_TIMEOUT),
                            fallback: bsd_result(-1, ETIMEDOUT),
                        }
                    } else {
                        done(-1, EALREADY)
                    }
                }
                ConnectStatus::Connected => {
                    socket.connecting = false;
                    log::debug!("bsd.Connect fd={} address={:?} established", fd, address);
                    done(0, 0)
                }
                ConnectStatus::Failed(errno) => {
                    socket.connecting = false;
                    socket.pending_error = Some(errno);
                    log::debug!(
                        "bsd.Connect fd={} address={:?} failed errno={}",
                        fd,
                        address,
                        errno
                    );
                    done(-1, errno)
                }
            };
        }
        match socket.socket.connect(&address.to_sock_addr()) {
            Ok(()) => {
                log::debug!("bsd.Connect fd={} address={:?}", fd, address);
                done(0, 0)
            }
            Err(error) if is_in_progress(&error) => {
                socket.connecting = true;
                log::debug!("bsd.Connect fd={} address={:?} in progress", fd, address);
                if socket.guest_blocking(0) {
                    Outcome::Wait {
                        timeout: Some(CONNECT_TIMEOUT),
                        fallback: bsd_result(-1, ETIMEDOUT),
                    }
                } else {
                    done(-1, EINPROGRESS)
                }
            }
            Err(error) => {
                log::debug!("bsd.Connect fd={} address={:?} failed: {}", fd, address, error);
                done(-1, errno_from(&error))
            }
        }
    }

    fn listen_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let backlog = input_i32(ctx, 1).unwrap_or(0).max(1);
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let socket = shared.lock();
        match socket.socket.listen(backlog) {
            Ok(()) => {
                log::debug!("bsd.Listen fd={} backlog={}", fd, backlog);
                done(0, 0)
            }
            Err(error) => done(-1, errno_from(&error)),
        }
    }

    fn accept_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done_with_len(-1, EINVAL, 0);
        };
        let Some(shared) = self.socket(fd) else {
            return done_with_len(-1, EBADF, 0);
        };
        let Some(new_fd) = self.free_slot() else {
            return done_with_len(-1, EMFILE, 0);
        };
        let (accepted, blocking, socket_type) = {
            let socket = shared.lock();
            (
                socket.socket.accept(),
                socket.guest_blocking(0),
                socket.socket_type,
            )
        };
        match accepted {
            Ok((host, peer)) => {
                let accepted = match HostSocket::new(host, socket_type, false) {
                    Ok(accepted) => accepted,
                    Err(error) => return done_with_len(-1, errno_from(&error), 0),
                };
                self.sockets[new_fd] = Some(Arc::new(Mutex::new(accepted)));
                let peer = BsdAddress::from_sock_addr(&peer);
                let len = match (peer, recv_buffer(ctx, 0)) {
                    (Some(peer), Some(buffer)) if buffer.size > 0 => {
                        let encoded = peer.encode();
                        let len = encoded.len().min(buffer.size as usize);
                        if memory.write(buffer.addr, &encoded[..len]).is_err() {
                            return done_with_len(-1, EFAULT, 0);
                        }
                        len as u32
                    }
                    _ => 0,
                };
                log::debug!("bsd.Accept fd={} → fd={} peer={:?}", fd, new_fd, peer);
                done_with_len(new_fd as i32, 0, len)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if blocking {
                    Outcome::Wait {
                        timeout: None,
                        fallback: bsd_result_with_len(-1, EAGAIN, 0),
                    }
                } else {
                    done_with_len(-1, EAGAIN, 0)
                }
            }
            Err(error) => done_with_len(-1, errno_from(&error), 0),
        }
    }

    fn send_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx, with_addr: bool) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let flags = if ctx.cmif_in.cmd_id == 24 {
            0
        } else {
            input_u32(ctx, 1).unwrap_or(0)
        };
        let Some(message_buffer) = send_buffer(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Ok(message) = read_guest(memory, message_buffer, MAX_MESSAGE) else {
            return done(-1, EFAULT);
        };
        let destination = if with_addr {
            match send_buffer(ctx, 1) {
                Some(buffer) if buffer.size >= 8 => match read_guest(memory, buffer, 64) {
                    Ok(bytes) => match BsdAddress::parse(&bytes) {
                        Some(address) => Some(address),
                        None => return done(-1, EINVAL),
                    },
                    Err(()) => return done(-1, EFAULT),
                },
                _ => None,
            }
        } else {
            None
        };
        if let Some(event) = self.event_fd_mut(fd) {
            let Some(value) = message.get(..8).map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap())) else {
                return done(-1, EINVAL);
            };
            return match event.write(value) {
                Ok(()) => done(8, 0),
                Err(errno) => done(-1, errno),
            };
        }
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let socket = shared.lock();
        let result = match destination {
            Some(address) => socket
                .socket
                .send_to_with_flags(&message, &address.to_sock_addr(), send_flags()),
            None => socket.socket.send_with_flags(&message, send_flags()),
        };
        match result {
            Ok(sent) => {
                log::trace!(
                    "bsd.{} fd={} destination={:?} bytes={}",
                    if with_addr { "SendTo" } else { "Send" },
                    fd,
                    destination,
                    sent
                );
                done(sent as i32, 0)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if socket.guest_blocking(flags) {
                    Outcome::Wait {
                        timeout: socket.send_timeout,
                        fallback: bsd_result(-1, EAGAIN),
                    }
                } else {
                    done(-1, EAGAIN)
                }
            }
            Err(error) => {
                log::debug!(
                    "bsd.{} fd={} destination={:?} failed: {}",
                    if with_addr { "SendTo" } else { "Send" },
                    fd,
                    destination,
                    error
                );
                done(-1, errno_from(&error))
            }
        }
    }

    fn recv_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx, with_addr: bool) -> Outcome {
        let fail = |ret: i32, errno: u32| {
            if with_addr {
                done_with_len(ret, errno, 0)
            } else {
                done(ret, errno)
            }
        };
        let Some(fd) = input_i32(ctx, 0) else {
            return fail(-1, EINVAL);
        };
        let flags = if ctx.cmif_in.cmd_id == 25 {
            0
        } else {
            input_u32(ctx, 1).unwrap_or(0)
        };
        let Some(message_buffer) = recv_buffer(ctx, 0) else {
            return fail(-1, EINVAL);
        };
        if let Some(event) = self.event_fd_mut(fd) {
            if message_buffer.size < 8 {
                return fail(-1, EINVAL);
            }
            return match event.read() {
                Some(value) => {
                    if memory.write(message_buffer.addr, &value.to_le_bytes()).is_err() {
                        return fail(-1, EFAULT);
                    }
                    if with_addr { done_with_len(8, 0, 0) } else { done(8, 0) }
                }
                None if event.nonblocking || flags & MSG_DONTWAIT != 0 => fail(-1, EAGAIN),
                None => Outcome::Wait {
                    timeout: None,
                    fallback: bsd_result(-1, EAGAIN),
                },
            };
        }
        let Some(shared) = self.socket(fd) else {
            return fail(-1, EBADF);
        };
        let socket = shared.lock();
        let capacity = (message_buffer.size as usize).min(MAX_MESSAGE);
        let mut storage: Vec<MaybeUninit<u8>> = vec![MaybeUninit::zeroed(); capacity];
        let peek = flags & MSG_PEEK != 0;
        let (result, source) = if with_addr && !socket.is_stream() {
            let result = if peek {
                socket.socket.peek_from(&mut storage)
            } else {
                socket.socket.recv_from(&mut storage)
            };
            match result {
                Ok((len, address)) => (Ok(len), BsdAddress::from_sock_addr(&address)),
                Err(error) => (Err(error), None),
            }
        } else {
            let result = if peek {
                socket.socket.peek(&mut storage)
            } else {
                socket.socket.recv(&mut storage)
            };
            (result, None)
        };
        match result {
            Ok(received) => {
                let received = received.min(capacity);
                let bytes: Vec<u8> = storage[..received]
                    .iter()
                    .map(|byte| unsafe { byte.assume_init() })
                    .collect();
                if memory.write(message_buffer.addr, &bytes).is_err() {
                    return fail(-1, EFAULT);
                }
                let addr_len = match (with_addr, source, recv_buffer(ctx, 1)) {
                    (true, Some(source), Some(buffer)) if buffer.size > 0 => {
                        let encoded = source.encode();
                        let len = encoded.len().min(buffer.size as usize);
                        if memory.write(buffer.addr, &encoded[..len]).is_err() {
                            return fail(-1, EFAULT);
                        }
                        len as u32
                    }
                    _ => 0,
                };
                log::trace!(
                    "bsd.{} fd={} source={:?} bytes={}",
                    if with_addr { "RecvFrom" } else { "Recv" },
                    fd,
                    source,
                    received
                );
                if with_addr {
                    done_with_len(received as i32, 0, addr_len)
                } else {
                    done(received as i32, 0)
                }
            }
            Err(error) if is_would_block_for(&socket, &error) => {
                if socket.guest_blocking(flags) {
                    Outcome::Wait {
                        timeout: socket.recv_timeout,
                        fallback: if with_addr {
                            bsd_result_with_len(-1, EAGAIN, 0)
                        } else {
                            bsd_result(-1, EAGAIN)
                        },
                    }
                } else {
                    fail(-1, EAGAIN)
                }
            }
            Err(error) => {
                log::debug!(
                    "bsd.{} fd={} failed: {}",
                    if with_addr { "RecvFrom" } else { "Recv" },
                    fd,
                    error
                );
                fail(-1, errno_from(&error))
            }
        }
    }

    fn poll_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(nfds) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let timeout = input_i32(ctx, 1).unwrap_or(-1);
        if nfds < 0 || nfds as usize > MAX_SOCKETS {
            return done(-1, EINVAL);
        }
        if nfds == 0 {
            if timeout > 0 {
                return Outcome::Wait {
                    timeout: Some(Duration::from_millis(timeout as u64)),
                    fallback: bsd_result(0, 0),
                };
            }
            return done(0, 0);
        }
        let Some(input_buffer) = send_buffer(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(output_buffer) = recv_buffer(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let size = nfds as usize * 8;
        if (input_buffer.size as usize) < size || (output_buffer.size as usize) < size {
            return done(-1, EINVAL);
        }
        let mut entries = vec![0u8; size];
        if memory.read(input_buffer.addr, &mut entries).is_err() {
            return done(-1, EFAULT);
        }
        let mut host_entries = Vec::with_capacity(nfds as usize);
        let mut host_index = Vec::with_capacity(nfds as usize);
        let mut ready = 0i32;
        let mut revents = vec![0u16; nfds as usize];
        for (index, entry) in entries.chunks_exact(8).enumerate() {
            let fd = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            let events = u16::from_le_bytes([entry[4], entry[5]]);
            if fd < 0 {
                continue;
            }
            if let Some(event) = self.event_fd(fd) {
                revents[index] = event.poll_events(events);
                if revents[index] != 0 {
                    ready += 1;
                }
                continue;
            }
            match self.socket(fd) {
                Some(shared) => {
                    host_entries.push(HostPollFd {
                        raw: raw_handle(&shared.lock().socket),
                        events,
                        revents: 0,
                    });
                    host_index.push(index);
                }
                None => {
                    revents[index] = POLLNVAL;
                    ready += 1;
                }
            }
        }
        if !host_entries.is_empty() {
            match host_poll(&mut host_entries) {
                Ok(_) => {
                    for (entry, &index) in host_entries.iter().zip(host_index.iter()) {
                        if entry.revents != 0 {
                            revents[index] = entry.revents;
                            ready += 1;
                        }
                    }
                }
                Err(error) => {
                    log::debug!("bsd.Poll host failure: {}", error);
                    return done(-1, errno_from(&error));
                }
            }
        }
        let wait = ready == 0 && timeout != 0;
        for (index, entry) in entries.chunks_exact_mut(8).enumerate() {
            let value = if wait { 0 } else { revents[index] };
            entry[6..8].copy_from_slice(&value.to_le_bytes());
        }
        if memory.write(output_buffer.addr, &entries).is_err() {
            return done(-1, EFAULT);
        }
        if wait {
            return Outcome::Wait {
                timeout: (timeout > 0).then(|| Duration::from_millis(timeout as u64)),
                fallback: bsd_result(0, 0),
            };
        }
        log::trace!("bsd.Poll nfds={} ready={}", nfds, ready);
        done(ready, 0)
    }

    fn select_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(nfds) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        if nfds < 0 {
            return done(-1, EINVAL);
        }
        let timeout = select_timeout(ctx);
        let nfds = (nfds as usize).min(MAX_SOCKETS);
        let mut sets: Vec<(Option<IpcBuffer>, Option<IpcBuffer>, Vec<u8>)> = Vec::with_capacity(3);
        for index in 0..3 {
            let input = send_buffer(ctx, index).filter(|buffer| buffer.size > 0);
            let output = recv_buffer(ctx, index).filter(|buffer| buffer.size > 0);
            let bits = match input {
                Some(buffer) => match read_guest(memory, buffer, 4096) {
                    Ok(bits) => bits,
                    Err(()) => return done(-1, EFAULT),
                },
                None => Vec::new(),
            };
            sets.push((input, output, bits));
        }
        let interest = [POLLIN, POLLOUT, POLLPRI];
        let mut host_entries = Vec::new();
        let mut host_index = Vec::new();
        let mut event_entries = Vec::new();
        for fd in 0..nfds {
            let mut events = 0u16;
            for (set, mask) in sets.iter().zip(interest.iter()) {
                if fd_set_contains(&set.2, fd) {
                    events |= mask;
                }
            }
            if events == 0 {
                continue;
            }
            if let Some(event) = self.event_fd(fd as i32) {
                event_entries.push((fd, events, event.poll_events(events)));
                continue;
            }
            let Some(shared) = self.socket(fd as i32) else {
                return done(-1, EBADF);
            };
            host_entries.push(HostPollFd {
                raw: raw_handle(&shared.lock().socket),
                events,
                revents: 0,
            });
            host_index.push(fd);
        }
        let mut outputs: Vec<Vec<u8>> = sets
            .iter()
            .map(|set| vec![0u8; set.1.map_or(0, |buffer| buffer.size as usize).min(4096)])
            .collect();
        let mut ready = 0i32;
        for &(fd, events, revents) in &event_entries {
            for (index, mask) in interest.iter().enumerate() {
                if events & mask != 0 && revents & mask != 0 {
                    fd_set_insert(&mut outputs[index], fd);
                    ready += 1;
                }
            }
        }
        if !host_entries.is_empty() {
            if let Err(error) = host_poll(&mut host_entries) {
                log::debug!("bsd.Select host failure: {}", error);
                return done(-1, errno_from(&error));
            }
            for (entry, &fd) in host_entries.iter().zip(host_index.iter()) {
                let readable = entry.revents & (POLLIN | POLLHUP | POLLERR) != 0;
                let writable = entry.revents & (POLLOUT | POLLERR) != 0;
                let exceptional = entry.revents & (POLLPRI | POLLERR) != 0;
                for (index, flag) in [readable, writable, exceptional].iter().enumerate() {
                    if *flag && entry.events & interest[index] != 0 {
                        fd_set_insert(&mut outputs[index], fd);
                        ready += 1;
                    }
                }
            }
        }
        let wait = ready == 0 && timeout != Some(Duration::ZERO);
        for (set, output) in sets.iter().zip(outputs.iter()) {
            if let Some(buffer) = set.1 {
                let bytes: &[u8] = if wait { &vec![0u8; output.len()] } else { output };
                if memory.write(buffer.addr, bytes).is_err() {
                    return done(-1, EFAULT);
                }
            }
        }
        if wait {
            return Outcome::Wait {
                timeout,
                fallback: bsd_result(0, 0),
            };
        }
        log::trace!("bsd.Select nfds={} ready={}", nfds, ready);
        done(ready, 0)
    }

    fn name_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx, peer: bool) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done_with_len(-1, EINVAL, 0);
        };
        let Some(shared) = self.socket(fd) else {
            return done_with_len(-1, EBADF, 0);
        };
        let result = {
            let socket = shared.lock();
            if peer {
                socket.socket.peer_addr()
            } else {
                socket.socket.local_addr()
            }
        };
        let address = match result {
            Ok(address) => BsdAddress::from_sock_addr(&address).unwrap_or(BsdAddress::any(0)),
            Err(error) => return done_with_len(-1, errno_from(&error), 0),
        };
        let Some(buffer) = recv_buffer(ctx, 0) else {
            return done_with_len(-1, EINVAL, 0);
        };
        let encoded = address.encode();
        let len = encoded.len().min(buffer.size as usize);
        if memory.write(buffer.addr, &encoded[..len]).is_err() {
            return done_with_len(-1, EFAULT, 0);
        }
        log::debug!(
            "bsd.{} fd={} address={:?}",
            if peer { "GetPeerName" } else { "GetSockName" },
            fd,
            address
        );
        done_with_len(0, 0, len as u32)
    }

    fn get_sock_opt_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done_with_len(-1, EINVAL, 0);
        };
        let level = input_u32(ctx, 1).unwrap_or(0);
        let name = input_u32(ctx, 2).unwrap_or(0);
        let Some(shared) = self.socket(fd) else {
            return done_with_len(-1, EBADF, 0);
        };
        let Some(buffer) = recv_buffer(ctx, 0) else {
            return done_with_len(-1, EINVAL, 0);
        };
        let mut socket = shared.lock();
        let value: Result<Vec<u8>, io::Error> = match (level, name) {
            (SOL_SOCKET, SO_ERROR) => match socket.pending_error.take() {
                Some(errno) => Ok(errno.to_le_bytes().to_vec()),
                None => socket
                    .socket
                    .take_error()
                    .map(|error| error.map_or(0, |error| errno_from(&error)).to_le_bytes().to_vec()),
            },
            (SOL_SOCKET, SO_TYPE) => Ok(socket.socket_type.to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_REUSEADDR) => socket
                .socket
                .reuse_address()
                .map(|value| u32::from(value).to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_KEEPALIVE) => socket
                .socket
                .keepalive()
                .map(|value| u32::from(value).to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_BROADCAST) => socket
                .socket
                .broadcast()
                .map(|value| u32::from(value).to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_SNDBUF) => socket
                .socket
                .send_buffer_size()
                .map(|value| (value as u32).to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_RCVBUF) => socket
                .socket
                .recv_buffer_size()
                .map(|value| (value as u32).to_le_bytes().to_vec()),
            (SOL_SOCKET, SO_SNDTIMEO) => Ok(encode_timeval(socket.send_timeout)),
            (SOL_SOCKET, SO_RCVTIMEO) => Ok(encode_timeval(socket.recv_timeout)),
            (SOL_SOCKET, SO_LINGER) => socket.socket.linger().map(|linger| {
                let mut out = Vec::with_capacity(8);
                out.extend_from_slice(&u32::from(linger.is_some()).to_le_bytes());
                out.extend_from_slice(
                    &(linger.map_or(0, |linger| linger.as_secs() as u32)).to_le_bytes(),
                );
                out
            }),
            (IPPROTO_TCP, TCP_NODELAY) => socket
                .socket
                .tcp_nodelay()
                .map(|value| u32::from(value).to_le_bytes().to_vec()),
            _ => {
                log::debug!(
                    "bsd.GetSockOpt fd={} level={:#x} name={:#x} unsupported",
                    fd,
                    level,
                    name
                );
                Ok(vec![0u8; 4])
            }
        };
        let value = match value {
            Ok(value) => value,
            Err(error) => return done_with_len(-1, errno_from(&error), 0),
        };
        let len = value.len().min(buffer.size as usize);
        if memory.write(buffer.addr, &value[..len]).is_err() {
            return done_with_len(-1, EFAULT, 0);
        }
        done_with_len(0, 0, len as u32)
    }

    fn set_sock_opt_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let level = input_u32(ctx, 1).unwrap_or(0);
        let name = input_u32(ctx, 2).unwrap_or(0);
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let value = match send_buffer(ctx, 0) {
            Some(buffer) => match read_guest(memory, buffer, 64) {
                Ok(value) => value,
                Err(()) => return done(-1, EFAULT),
            },
            None => Vec::new(),
        };
        let as_u32 = || -> u32 {
            value
                .get(..4)
                .map_or(0, |bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let mut socket = shared.lock();
        let result = match (level, name) {
            (SOL_SOCKET, SO_REUSEADDR) => socket.socket.set_reuse_address(as_u32() != 0),
            (SOL_SOCKET, SO_KEEPALIVE) => socket.socket.set_keepalive(as_u32() != 0),
            (SOL_SOCKET, SO_BROADCAST) => socket.socket.set_broadcast(as_u32() != 0),
            (SOL_SOCKET, SO_LINGER) => {
                let onoff = as_u32();
                let seconds = value
                    .get(4..8)
                    .map_or(0, |bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
                socket
                    .socket
                    .set_linger((onoff != 0).then(|| Duration::from_secs(seconds as u64)))
            }
            (SOL_SOCKET, SO_SNDBUF) => socket.socket.set_send_buffer_size(as_u32() as usize),
            (SOL_SOCKET, SO_RCVBUF) => socket.socket.set_recv_buffer_size(as_u32() as usize),
            (SOL_SOCKET, SO_SNDTIMEO) => {
                socket.send_timeout = decode_timeval(&value);
                Ok(())
            }
            (SOL_SOCKET, SO_RCVTIMEO) => {
                socket.recv_timeout = decode_timeval(&value);
                Ok(())
            }
            (SOL_SOCKET, SO_NOSIGPIPE) => Ok(()),
            (IPPROTO_TCP, TCP_NODELAY) => socket.socket.set_tcp_nodelay(as_u32() != 0),
            _ => {
                log::debug!(
                    "bsd.SetSockOpt fd={} level={:#x} name={:#x} len={} ignored",
                    fd,
                    level,
                    name,
                    value.len()
                );
                Ok(())
            }
        };
        match result {
            Ok(()) => {
                log::debug!(
                    "bsd.SetSockOpt fd={} level={:#x} name={:#x} value={:?}",
                    fd,
                    level,
                    name,
                    value
                );
                done(0, 0)
            }
            Err(error) => {
                log::debug!(
                    "bsd.SetSockOpt fd={} level={:#x} name={:#x} failed: {}",
                    fd,
                    level,
                    name,
                    error
                );
                done(-1, errno_from(&error))
            }
        }
    }

    fn fcntl_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(cmd) = input_i32(ctx, 1) else {
            return done(-1, EINVAL);
        };
        let arg = input_i32(ctx, 2).unwrap_or(0);
        if let Some(event) = self.event_fd_mut(fd) {
            return match cmd {
                FCNTL_GETFL => done(if event.nonblocking { O_NONBLOCK } else { 0 }, 0),
                FCNTL_SETFL => {
                    event.nonblocking = arg & O_NONBLOCK != 0;
                    done(0, 0)
                }
                _ => done(-1, EINVAL),
            };
        }
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let mut socket = shared.lock();
        match cmd {
            FCNTL_GETFL => done(socket.flags, 0),
            FCNTL_SETFL => {
                socket.flags = arg;
                log::debug!("bsd.Fcntl fd={} flags={:#x}", fd, arg);
                done(0, 0)
            }
            _ => done(-1, EINVAL),
        }
    }

    fn shutdown_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let how = input_i32(ctx, 1).unwrap_or(-1);
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let how = match how {
            SHUT_RD => Shutdown::Read,
            SHUT_WR => Shutdown::Write,
            SHUT_RDWR => Shutdown::Both,
            _ => return done(-1, EINVAL),
        };
        let socket = shared.lock();
        match socket.socket.shutdown(how) {
            Ok(()) => done(0, 0),
            Err(error) => done(-1, errno_from(&error)),
        }
    }

    fn close_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Ok(index) = usize::try_from(fd) else {
            return done(-1, EBADF);
        };
        let Some(slot) = self.sockets.get_mut(index) else {
            return done(-1, EBADF);
        };
        let closed_socket = slot.take().is_some();
        let closed_event = self.event_fds[index].take().is_some();
        if !closed_socket && !closed_event {
            return done(-1, EBADF);
        }
        log::debug!("bsd.Close fd={}", fd);
        done(0, 0)
    }

    fn duplicate_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let Some(fd) = input_i32(ctx, 0) else {
            return done(-1, EINVAL);
        };
        let Some(shared) = self.socket(fd) else {
            return done(-1, EBADF);
        };
        let Some(new_fd) = self.free_slot() else {
            return done(-1, EMFILE);
        };
        self.sockets[new_fd] = Some(shared);
        log::debug!("bsd.DuplicateSocket fd={} → fd={}", fd, new_fd);
        done(new_fd as i32, 0)
    }

    fn socket(&self, fd: i32) -> Option<SharedSocket> {
        usize::try_from(fd)
            .ok()
            .and_then(|fd| self.sockets.get(fd))
            .and_then(|slot| slot.clone())
    }

    fn event_fd(&self, fd: i32) -> Option<EventFd> {
        usize::try_from(fd).ok().and_then(|fd| self.event_fds.get(fd).copied().flatten())
    }

    fn event_fd_mut(&mut self, fd: i32) -> Option<&mut EventFd> {
        usize::try_from(fd).ok().and_then(|fd| self.event_fds.get_mut(fd)).and_then(Option::as_mut)
    }

    fn event_fd_command(&mut self, ctx: &IpcCtx) -> Outcome {
        let flags = input_u32(ctx, 0).unwrap_or(0);
        let initial = match (input_u32(ctx, 2), input_u32(ctx, 3)) {
            (Some(low), Some(high)) => u64::from(low) | (u64::from(high) << 32),
            _ => 0,
        };
        let Some(fd) = self.free_slot() else {
            return done(-1, EMFILE);
        };
        self.event_fds[fd] = Some(EventFd::new(initial, flags));
        log::debug!("bsd.EventFd flags={:#x} initial={} -> fd={}", flags, initial, fd);
        done(fd as i32, 0)
    }

    fn free_slot(&self) -> Option<usize> {
        (0..MAX_SOCKETS).find(|&fd| self.sockets[fd].is_none() && self.event_fds[fd].is_none())
    }
}

impl Default for BsdService {
    fn default() -> Self {
        Self::new()
    }
}

fn connect_status(socket: &Socket) -> ConnectStatus {
    let mut entries = [HostPollFd {
        raw: raw_handle(socket),
        events: POLLOUT,
        revents: 0,
    }];
    let revents = match host_poll(&mut entries) {
        Ok(_) => entries[0].revents,
        Err(error) => return ConnectStatus::Failed(errno_from(&error)),
    };
    if revents == 0 {
        return ConnectStatus::Pending;
    }
    match socket.take_error() {
        Ok(Some(error)) => return ConnectStatus::Failed(errno_from(&error)),
        Ok(None) => {}
        Err(error) => return ConnectStatus::Failed(errno_from(&error)),
    }
    if revents & (POLLERR | POLLHUP | POLLNVAL) != 0 {
        return ConnectStatus::Failed(ECONNREFUSED);
    }
    if revents & POLLOUT != 0 {
        return ConnectStatus::Connected;
    }
    ConnectStatus::Pending
}

fn is_in_progress(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    matches!(error.raw_os_error(), Some(code) if is_in_progress_code(code))
}

#[cfg(windows)]
fn is_in_progress_code(code: i32) -> bool {
    matches!(code, 10035 | 10036)
}

#[cfg(unix)]
fn is_in_progress_code(code: i32) -> bool {
    code == libc::EINPROGRESS || code == libc::EAGAIN
}

fn is_would_block_for(socket: &HostSocket, error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    !socket.is_stream() && error.kind() == io::ErrorKind::ConnectionReset
}

fn send_flags() -> i32 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        libc::MSG_NOSIGNAL
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        0
    }
}

fn errno_from(error: &io::Error) -> u32 {
    if let Some(code) = error.raw_os_error() {
        if let Some(errno) = errno_from_raw(code) {
            return errno;
        }
    }
    match error.kind() {
        io::ErrorKind::WouldBlock => EAGAIN,
        io::ErrorKind::ConnectionRefused => ECONNREFUSED,
        io::ErrorKind::ConnectionReset => ECONNRESET,
        io::ErrorKind::ConnectionAborted => ECONNABORTED,
        io::ErrorKind::NotConnected => ENOTCONN,
        io::ErrorKind::AddrInUse => EADDRINUSE,
        io::ErrorKind::AddrNotAvailable => EADDRNOTAVAIL,
        io::ErrorKind::BrokenPipe => EPIPE,
        io::ErrorKind::TimedOut => ETIMEDOUT,
        io::ErrorKind::InvalidInput => EINVAL,
        io::ErrorKind::PermissionDenied => EACCES,
        io::ErrorKind::Interrupted => EINTR,
        io::ErrorKind::Unsupported => EOPNOTSUPP,
        _ => EIO,
    }
}

#[cfg(windows)]
fn errno_from_raw(code: i32) -> Option<u32> {
    Some(match code {
        10004 => EINTR,
        10009 => EBADF,
        10013 => EACCES,
        10014 => EFAULT,
        10022 => EINVAL,
        10024 => EMFILE,
        10035 => EAGAIN,
        10036 => EINPROGRESS,
        10037 => EALREADY,
        10038 => ENOTSOCK,
        10039 => EDESTADDRREQ,
        10040 => EMSGSIZE,
        10044 => ESOCKTNOSUPPORT,
        10045 => EOPNOTSUPP,
        10047 => EAFNOSUPPORT,
        10048 => EADDRINUSE,
        10049 => EADDRNOTAVAIL,
        10050 => ENETDOWN,
        10051 => ENETUNREACH,
        10052 => ENETRESET,
        10053 => ECONNABORTED,
        10054 => ECONNRESET,
        10055 => ENOBUFS,
        10056 => EISCONN,
        10057 => ENOTCONN,
        10058 => ESHUTDOWN,
        10060 => ETIMEDOUT,
        10061 => ECONNREFUSED,
        10064 => EHOSTDOWN,
        10065 => EHOSTUNREACH,
        _ => return None,
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn errno_from_raw(code: i32) -> Option<u32> {
    u32::try_from(code).ok().filter(|code| *code != 0)
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
fn errno_from_raw(_code: i32) -> Option<u32> {
    None
}

struct HostPollFd {
    raw: RawHandle,
    events: u16,
    revents: u16,
}

#[cfg(windows)]
type RawHandle = std::os::windows::io::RawSocket;

#[cfg(unix)]
type RawHandle = std::os::unix::io::RawFd;

#[cfg(windows)]
fn raw_handle(socket: &Socket) -> RawHandle {
    use std::os::windows::io::AsRawSocket;
    socket.as_raw_socket()
}

#[cfg(unix)]
fn raw_handle(socket: &Socket) -> RawHandle {
    use std::os::unix::io::AsRawFd;
    socket.as_raw_fd()
}

#[cfg(windows)]
fn host_poll(entries: &mut [HostPollFd]) -> io::Result<usize> {
    use windows_sys::Win32::Networking::WinSock::{
        WSAPoll, POLLERR as W_POLLERR, POLLHUP as W_POLLHUP, POLLNVAL as W_POLLNVAL,
        POLLRDBAND as W_POLLRDBAND, POLLRDNORM as W_POLLRDNORM, POLLWRNORM as W_POLLWRNORM,
        SOCKET_ERROR, WSAPOLLFD,
    };
    let mut host: Vec<WSAPOLLFD> = entries
        .iter()
        .map(|entry| {
            let mut events = 0i16;
            if entry.events & (POLLIN | POLLRDNORM) != 0 {
                events |= W_POLLRDNORM;
            }
            if entry.events & (POLLPRI | POLLRDBAND) != 0 {
                events |= W_POLLRDBAND;
            }
            if entry.events & (POLLOUT | POLLWRBAND) != 0 {
                events |= W_POLLWRNORM;
            }
            WSAPOLLFD {
                fd: entry.raw as usize,
                events,
                revents: 0,
            }
        })
        .collect();
    let result = unsafe { WSAPoll(host.as_mut_ptr(), host.len() as u32, 0) };
    if result == SOCKET_ERROR {
        return Err(io::Error::last_os_error());
    }
    for (entry, host) in entries.iter_mut().zip(host.iter()) {
        let mut revents = 0u16;
        if host.revents & W_POLLRDNORM != 0 {
            revents |= POLLIN;
        }
        if host.revents & W_POLLRDBAND != 0 {
            revents |= POLLPRI;
        }
        if host.revents & W_POLLWRNORM != 0 {
            revents |= POLLOUT;
        }
        if host.revents & W_POLLERR != 0 {
            revents |= POLLERR;
        }
        if host.revents & W_POLLHUP != 0 {
            revents |= POLLHUP;
        }
        if host.revents & W_POLLNVAL != 0 {
            revents |= POLLNVAL;
        }
        entry.revents = revents;
    }
    Ok(result as usize)
}

#[cfg(unix)]
fn host_poll(entries: &mut [HostPollFd]) -> io::Result<usize> {
    let mut host: Vec<libc::pollfd> = entries
        .iter()
        .map(|entry| {
            let mut events = 0i16;
            if entry.events & (POLLIN | POLLRDNORM) != 0 {
                events |= libc::POLLIN;
            }
            if entry.events & (POLLPRI | POLLRDBAND) != 0 {
                events |= libc::POLLPRI;
            }
            if entry.events & (POLLOUT | POLLWRBAND) != 0 {
                events |= libc::POLLOUT;
            }
            libc::pollfd {
                fd: entry.raw,
                events,
                revents: 0,
            }
        })
        .collect();
    let result = unsafe { libc::poll(host.as_mut_ptr(), host.len() as libc::nfds_t, 0) };
    if result < 0 {
        return Err(io::Error::last_os_error());
    }
    for (entry, host) in entries.iter_mut().zip(host.iter()) {
        let mut revents = 0u16;
        if host.revents & libc::POLLIN != 0 {
            revents |= POLLIN;
        }
        if host.revents & libc::POLLPRI != 0 {
            revents |= POLLPRI;
        }
        if host.revents & libc::POLLOUT != 0 {
            revents |= POLLOUT;
        }
        if host.revents & libc::POLLERR != 0 {
            revents |= POLLERR;
        }
        if host.revents & libc::POLLHUP != 0 {
            revents |= POLLHUP;
        }
        if host.revents & libc::POLLNVAL != 0 {
            revents |= POLLNVAL;
        }
        entry.revents = revents;
    }
    Ok(result as usize)
}

fn select_timeout(ctx: &IpcCtx) -> Option<Duration> {
    if ctx.cmif_in_data_len >= 32 {
        let is_null = input_u32(ctx, 6).unwrap_or(0) & 0xff != 0;
        if is_null {
            return None;
        }
        let seconds = (input_u32(ctx, 2).unwrap_or(0) as u64)
            | ((input_u32(ctx, 3).unwrap_or(0) as u64) << 32);
        let micros = (input_u32(ctx, 4).unwrap_or(0) as u64)
            | ((input_u32(ctx, 5).unwrap_or(0) as u64) << 32);
        return Some(Duration::from_secs(seconds) + Duration::from_micros(micros));
    }
    match input_i32(ctx, 1) {
        Some(ms) if ms >= 0 => Some(Duration::from_millis(ms as u64)),
        _ => None,
    }
}

fn fd_set_contains(bits: &[u8], fd: usize) -> bool {
    bits.get(fd / 8).is_some_and(|byte| byte & (1 << (fd % 8)) != 0)
}

fn fd_set_insert(bits: &mut [u8], fd: usize) {
    if let Some(byte) = bits.get_mut(fd / 8) {
        *byte |= 1 << (fd % 8);
    }
}

fn encode_timeval(timeout: Option<Duration>) -> Vec<u8> {
    let timeout = timeout.unwrap_or(Duration::ZERO);
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&(timeout.as_secs() as i64).to_le_bytes());
    out.extend_from_slice(&(timeout.subsec_micros() as i64).to_le_bytes());
    out
}

fn decode_timeval(value: &[u8]) -> Option<Duration> {
    let timeout = if value.len() >= 16 {
        let seconds = i64::from_le_bytes(value[..8].try_into().unwrap()).max(0) as u64;
        let micros = i64::from_le_bytes(value[8..16].try_into().unwrap()).max(0) as u64;
        Duration::from_secs(seconds) + Duration::from_micros(micros)
    } else if value.len() >= 4 {
        Duration::from_millis(u32::from_le_bytes(value[..4].try_into().unwrap()) as u64)
    } else {
        Duration::ZERO
    };
    (timeout > Duration::ZERO).then_some(timeout)
}

fn read_address(memory: &AddressSpace, ctx: &IpcCtx, index: usize) -> Option<BsdAddress> {
    let buffer = send_buffer(ctx, index)?;
    let bytes = read_guest(memory, buffer, 64).ok()?;
    BsdAddress::parse(&bytes)
}

fn input_u32(ctx: &IpcCtx, index: usize) -> Option<u32> {
    let start = ctx.cmif_in_data_off.checked_add(index.checked_mul(4)?)?;
    let end = start.checked_add(4)?;
    let data_end = ctx
        .cmif_in_data_off
        .checked_add(ctx.cmif_in_data_len)?
        .min(ctx.buf.len());
    let bytes = ctx.buf.get(start..end.min(data_end))?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn input_i32(ctx: &IpcCtx, index: usize) -> Option<i32> {
    input_u32(ctx, index).map(|value| value as i32)
}

fn pick_buffer(buffers: &[IpcBuffer], statics: &[IpcBuffer], index: usize) -> Option<IpcBuffer> {
    match buffers.get(index) {
        Some(buffer) if buffer.addr != 0 => Some(*buffer),
        _ => statics.get(index).filter(|buffer| buffer.addr != 0).copied(),
    }
}

fn send_buffer(ctx: &IpcCtx, index: usize) -> Option<IpcBuffer> {
    pick_buffer(&ctx.send_buffers, &ctx.send_statics, index)
}

fn recv_buffer(ctx: &IpcCtx, index: usize) -> Option<IpcBuffer> {
    pick_buffer(&ctx.recv_buffers, &ctx.recv_statics, index)
}

fn read_guest(memory: &AddressSpace, buffer: IpcBuffer, limit: usize) -> Result<Vec<u8>, ()> {
    let size = usize::try_from(buffer.size).map_err(|_| ())?;
    if size > limit {
        return Err(());
    }
    let mut bytes = vec![0u8; size];
    memory.read(buffer.addr, &mut bytes).map_err(|_| ())?;
    Ok(bytes)
}

fn bsd_result(ret: i32, errno: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&ret.to_le_bytes());
    out.extend_from_slice(&errno.to_le_bytes());
    out
}

fn bsd_result_with_len(ret: i32, errno: u32, len: u32) -> Vec<u8> {
    let mut out = bsd_result(ret, errno);
    out.extend_from_slice(&len.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn sockaddr_round_trip() {
        let address = BsdAddress {
            ip: [127, 0, 0, 1],
            port: 19_132,
        };
        assert_eq!(BsdAddress::parse(&address.encode()), Some(address));
        assert_eq!(
            BsdAddress::from_sock_addr(&address.to_sock_addr()),
            Some(address)
        );
    }

    #[test]
    fn fd_set_bits() {
        let mut bits = vec![0u8; 16];
        fd_set_insert(&mut bits, 0);
        fd_set_insert(&mut bits, 9);
        assert!(fd_set_contains(&bits, 0));
        assert!(fd_set_contains(&bits, 9));
        assert!(!fd_set_contains(&bits, 1));
        assert!(!fd_set_contains(&bits, 200));
    }

    #[test]
    fn timeval_round_trip() {
        let timeout = Some(Duration::from_millis(1_500));
        assert_eq!(decode_timeval(&encode_timeval(timeout)), timeout);
        assert_eq!(decode_timeval(&encode_timeval(None)), None);
    }

    #[test]
    fn event_fd_counts_writes_and_drains_reads() {
        let mut event = EventFd::new(0, EFD_NONBLOCK);
        assert!(event.nonblocking);
        assert_eq!(event.poll_events(POLLIN | POLLOUT), POLLOUT);
        assert_eq!(event.read(), None);
        event.write(2).unwrap();
        event.write(3).unwrap();
        assert_eq!(event.poll_events(POLLIN), POLLIN);
        assert_eq!(event.read(), Some(5));
        assert_eq!(event.read(), None);
        assert_eq!(event.write(u64::MAX), Err(EINVAL));
    }

    #[test]
    fn event_fd_semaphore_reads_one_at_a_time() {
        let mut event = EventFd::new(2, EFD_SEMAPHORE);
        assert!(!event.nonblocking);
        assert_eq!(event.read(), Some(1));
        assert_eq!(event.read(), Some(1));
        assert_eq!(event.read(), None);
    }

    #[test]
    fn event_fds_share_the_descriptor_space() {
        let mut service = BsdService::new();
        service.event_fds[0] = Some(EventFd::new(0, 0));
        assert_eq!(service.free_slot(), Some(1));
        assert!(service.socket(0).is_none());
        assert!(service.event_fd(0).is_some());
    }

    #[test]
    fn would_block_maps_to_eagain() {
        let error = io::Error::from(io::ErrorKind::WouldBlock);
        assert_eq!(errno_from(&error), EAGAIN);
    }

    #[test]
    fn nonblocking_connect_completes_over_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let target = match listener.local_addr().unwrap() {
            SocketAddr::V4(address) => address,
            SocketAddr::V6(_) => unreachable!(),
        };
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).unwrap();
        let mut host = HostSocket::new(socket, SOCK_STREAM, false).unwrap();
        let address = BsdAddress::from_socket_addr(target);
        match host.socket.connect(&address.to_sock_addr()) {
            Ok(()) => {}
            Err(error) if is_in_progress(&error) => host.connecting = true,
            Err(error) => panic!("connect failed: {error}"),
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match connect_status(&host.socket) {
                ConnectStatus::Connected => break,
                ConnectStatus::Failed(errno) => panic!("connect failed errno={errno}"),
                ConnectStatus::Pending => {
                    assert!(Instant::now() < deadline, "connect did not complete");
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        let (_peer, _) = listener.accept().unwrap();
        let mut entries = [HostPollFd {
            raw: raw_handle(&host.socket),
            events: POLLOUT,
            revents: 0,
        }];
        assert_eq!(host_poll(&mut entries).unwrap(), 1);
        assert_ne!(entries[0].revents & POLLOUT, 0);
    }

    #[test]
    fn refused_connect_never_reports_connected() {
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let target = match probe.local_addr().unwrap() {
            SocketAddr::V4(address) => address,
            SocketAddr::V6(_) => unreachable!(),
        };
        drop(probe);
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).unwrap();
        let host = HostSocket::new(socket, SOCK_STREAM, false).unwrap();
        let address = BsdAddress::from_socket_addr(target);
        match host.socket.connect(&address.to_sock_addr()) {
            Ok(()) => return,
            Err(error) if is_in_progress(&error) => {}
            Err(_) => return,
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match connect_status(&host.socket) {
                ConnectStatus::Failed(_) => break,
                ConnectStatus::Connected => {
                    panic!("connect to a closed port was reported as connected")
                }
                ConnectStatus::Pending => {
                    assert!(Instant::now() < deadline, "refusal never surfaced");
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }

    #[test]
    fn udp_loopback_delivers_datagram() {
        let receiver = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
        receiver
            .bind(&BsdAddress::from_socket_addr(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).to_sock_addr())
            .unwrap();
        receiver.set_nonblocking(true).unwrap();
        let bound = BsdAddress::from_sock_addr(&receiver.local_addr().unwrap()).unwrap();
        let sender = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
        sender
            .send_to_with_flags(b"ping", &bound.to_sock_addr(), send_flags())
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut entries = [HostPollFd {
            raw: raw_handle(&receiver),
            events: POLLIN,
            revents: 0,
        }];
        while host_poll(&mut entries).unwrap() == 0 {
            assert!(Instant::now() < deadline, "datagram never arrived");
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut storage: Vec<MaybeUninit<u8>> = vec![MaybeUninit::zeroed(); 16];
        let (len, from) = receiver.recv_from(&mut storage).unwrap();
        assert_eq!(len, 4);
        assert!(BsdAddress::from_sock_addr(&from).is_some());
    }
}
