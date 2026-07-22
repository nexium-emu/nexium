use nexium_ipc::{IpcBuffer, IpcCtx};
use nexium_memory::AddressSpace;
use std::collections::{HashMap, VecDeque};

const MAX_SOCKETS: usize = 128;
const MAX_DATAGRAM: usize = 65_507;
const MAX_QUEUED_DATAGRAMS: usize = 4_096;

const AF_INET: u32 = 2;
const SOCK_STREAM: u32 = 1;
const SOCK_DGRAM: u32 = 2;

const EBADF: u32 = 9;
const EAGAIN: u32 = 11;
const EFAULT: u32 = 14;
const EINVAL: u32 = 22;
const EMFILE: u32 = 24;
const EMSGSIZE: u32 = 90;
const ENOTCONN: u32 = 107;

const MSG_PEEK: u32 = 2;
const MSG_DONTWAIT: u32 = 0x80;
const O_NONBLOCK: i32 = 0x800;

const POLLIN: u16 = 1;
const POLLOUT: u16 = 4;
const POLLNVAL: u16 = 32;

const SOL_SOCKET: u32 = 0xffff;
const SO_ERROR: u32 = 0x1007;
const SO_TYPE: u32 = 0x1008;

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

    fn any(port: u16) -> Self {
        Self { ip: [0; 4], port }
    }
}

#[derive(Clone)]
struct Datagram {
    source: BsdAddress,
    data: Vec<u8>,
}

#[derive(Clone)]
struct VirtualSocket {
    domain: u32,
    socket_type: u32,
    protocol: u32,
    flags: i32,
    local: Option<BsdAddress>,
    peer: Option<BsdAddress>,
    recv_queue: VecDeque<Datagram>,
    options: HashMap<(u32, u32), Vec<u8>>,
    last_error: u32,
}

impl VirtualSocket {
    fn new(domain: u32, socket_type: u32, protocol: u32) -> Self {
        Self {
            domain,
            socket_type,
            protocol,
            flags: 0,
            local: None,
            peer: None,
            recv_queue: VecDeque::new(),
            options: HashMap::new(),
            last_error: 0,
        }
    }
}

pub struct BsdService {
    sockets: Vec<Option<VirtualSocket>>,
    next_ephemeral_port: u16,
}

impl BsdService {
    pub fn new() -> Self {
        Self {
            sockets: (0..MAX_SOCKETS).map(|_| None).collect(),
            next_ephemeral_port: 49_152,
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("bsd legacy cmd: {}", cmd_id);
        0
    }

    pub fn dispatch_ipc(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> (u32, Vec<u8>, bool) {
        let cmd_id = ctx.cmif_in.cmd_id;
        let (out, wait) = match cmd_id {
            0 => (0u32.to_le_bytes().to_vec(), false),
            1 => (Vec::new(), false),
            2 | 3 => (self.socket_command(ctx), false),
            5 => (bsd_result(-1, EINVAL), false),
            6 => (self.poll_command(memory, ctx), false),
            8 => self.recv_command(memory, ctx, false),
            9 => self.recv_command(memory, ctx, true),
            10 => (self.send_command(memory, ctx, false), false),
            11 => (self.send_command(memory, ctx, true), false),
            12 => (bsd_result_with_len(-1, EAGAIN, 0), false),
            13 => (self.bind_command(memory, ctx), false),
            14 => (self.connect_command(memory, ctx), false),
            15 => (self.name_command(memory, ctx, true), false),
            16 => (self.name_command(memory, ctx, false), false),
            17 => (self.get_sock_opt_command(memory, ctx), false),
            18 => (self.listen_command(ctx), false),
            19 => (bsd_result(0, 0), false),
            20 => (self.fcntl_command(ctx), false),
            21 => (self.set_sock_opt_command(memory, ctx), false),
            22 => (self.shutdown_command(ctx), false),
            23 => (bsd_result(0, 0), false),
            24 => (self.send_command(memory, ctx, false), false),
            25 => self.recv_command(memory, ctx, false),
            26 => (self.close_command(ctx), false),
            27 => (self.duplicate_command(ctx), false),
            _ => {
                log::warn!("bsd:u cmd_{} unsupported", cmd_id);
                (bsd_result(-1, EINVAL), false)
            }
        };
        (0, out, wait)
    }

    fn socket_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(domain) = input_u32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(raw_type) = input_u32(ctx, 1) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(protocol) = input_u32(ctx, 2) else {
            return bsd_result(-1, EINVAL);
        };
        let socket_type = raw_type & 0xff;
        if domain != AF_INET || !matches!(socket_type, SOCK_STREAM | SOCK_DGRAM) {
            return bsd_result(-1, EINVAL);
        }
        let Some(fd) = self.sockets.iter().position(Option::is_none) else {
            return bsd_result(-1, EMFILE);
        };
        let mut socket = VirtualSocket::new(domain, socket_type, protocol);
        if raw_type & 0x2000_0000 != 0 {
            socket.flags |= O_NONBLOCK;
        }
        self.sockets[fd] = Some(socket);
        log::debug!(
            "bsd.Socket fd={} domain={} type={} protocol={}",
            fd,
            domain,
            socket_type,
            protocol
        );
        bsd_result(fd as i32, 0)
    }

    fn bind_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(buffer) = send_buffer(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Ok(bytes) = read_guest(memory, buffer, 16) else {
            return bsd_result(-1, EFAULT);
        };
        let Some(mut address) = BsdAddress::parse(&bytes) else {
            return bsd_result(-1, EINVAL);
        };
        if self.socket(fd).is_none() {
            return bsd_result(-1, EBADF);
        }
        if address.port == 0 {
            address.port = self.allocate_ephemeral_port();
        }
        self.socket_mut(fd).unwrap().local = Some(address);
        log::debug!("bsd.Bind fd={} address={:?}", fd, address);
        bsd_result(0, 0)
    }

    fn connect_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(buffer) = send_buffer(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Ok(bytes) = read_guest(memory, buffer, 16) else {
            return bsd_result(-1, EFAULT);
        };
        let Some(address) = BsdAddress::parse(&bytes) else {
            return bsd_result(-1, EINVAL);
        };
        if self.socket(fd).is_none() {
            return bsd_result(-1, EBADF);
        }
        self.ensure_bound(fd);
        self.socket_mut(fd).unwrap().peer = Some(address);
        log::debug!("bsd.Connect fd={} address={:?}", fd, address);
        bsd_result(0, 0)
    }

    fn send_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx, with_addr: bool) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let message_index = 0;
        let Some(message_buffer) = send_buffer(ctx, message_index) else {
            return bsd_result(-1, EINVAL);
        };
        if message_buffer.size as usize > MAX_DATAGRAM {
            return bsd_result(-1, EMSGSIZE);
        }
        let Ok(message) = read_guest(memory, message_buffer, MAX_DATAGRAM) else {
            return bsd_result(-1, EFAULT);
        };
        let destination = if with_addr {
            let Some(address_buffer) = send_buffer(ctx, 1) else {
                return bsd_result(-1, EINVAL);
            };
            let Ok(bytes) = read_guest(memory, address_buffer, 16) else {
                return bsd_result(-1, EFAULT);
            };
            let Some(address) = BsdAddress::parse(&bytes) else {
                return bsd_result(-1, EINVAL);
            };
            address
        } else {
            let Some(socket) = self.socket(fd) else {
                return bsd_result(-1, EBADF);
            };
            let Some(peer) = socket.peer else {
                return bsd_result(-1, ENOTCONN);
            };
            peer
        };
        let Some(socket_type) = self.socket(fd).map(|socket| socket.socket_type) else {
            return bsd_result(-1, EBADF);
        };
        if socket_type != SOCK_DGRAM {
            return bsd_result(-1, ENOTCONN);
        }
        self.ensure_bound(fd);
        let source = self.socket(fd).unwrap().local.unwrap();
        let source = BsdAddress {
            ip: source_ip(source.ip, destination.ip),
            port: source.port,
        };
        let mut delivered = 0usize;
        for target in self.sockets.iter_mut().flatten() {
            if target.socket_type != SOCK_DGRAM {
                continue;
            }
            let Some(bound) = target.local else {
                continue;
            };
            if !address_matches(bound, destination) {
                continue;
            }
            if target.recv_queue.len() >= MAX_QUEUED_DATAGRAMS {
                target.recv_queue.pop_front();
            }
            target.recv_queue.push_back(Datagram {
                source,
                data: message.clone(),
            });
            delivered += 1;
        }
        log::debug!(
            "bsd.{} fd={} source={:?} destination={:?} bytes={} delivered={}",
            if with_addr { "SendTo" } else { "Send" },
            fd,
            source,
            destination,
            message.len(),
            delivered
        );
        bsd_result(message.len() as i32, 0)
    }

    fn recv_command(
        &mut self,
        memory: &AddressSpace,
        ctx: &IpcCtx,
        with_addr: bool,
    ) -> (Vec<u8>, bool) {
        let Some(fd) = input_i32(ctx, 0) else {
            return (bsd_result(-1, EINVAL), false);
        };
        let flags = input_u32(ctx, 1).unwrap_or(0);
        let Some(message_buffer) = recv_buffer(ctx, 0) else {
            let out = if with_addr {
                bsd_result_with_len(-1, EINVAL, 0)
            } else {
                bsd_result(-1, EINVAL)
            };
            return (out, false);
        };
        let Some(socket) = self.socket_mut(fd) else {
            let out = if with_addr {
                bsd_result_with_len(-1, EBADF, 0)
            } else {
                bsd_result(-1, EBADF)
            };
            return (out, false);
        };
        let blocking = socket.flags & O_NONBLOCK == 0 && flags & MSG_DONTWAIT == 0;
        let packet = if flags & MSG_PEEK != 0 {
            socket.recv_queue.front().cloned()
        } else {
            socket.recv_queue.pop_front()
        };
        let Some(packet) = packet else {
            let out = if with_addr {
                bsd_result_with_len(-1, EAGAIN, 0)
            } else {
                bsd_result(-1, EAGAIN)
            };
            return (out, blocking);
        };
        let copied = packet.data.len().min(message_buffer.size as usize);
        if memory
            .write(message_buffer.addr, &packet.data[..copied])
            .is_err()
        {
            let out = if with_addr {
                bsd_result_with_len(-1, EFAULT, 0)
            } else {
                bsd_result(-1, EFAULT)
            };
            return (out, false);
        }
        let addr_len = if with_addr {
            if let Some(address_buffer) = recv_buffer(ctx, 1) {
                let encoded = packet.source.encode();
                let len = encoded.len().min(address_buffer.size as usize);
                if memory.write(address_buffer.addr, &encoded[..len]).is_err() {
                    return (bsd_result_with_len(-1, EFAULT, 0), false);
                }
                len as u32
            } else {
                0
            }
        } else {
            0
        };
        log::debug!(
            "bsd.{} fd={} source={:?} bytes={} queued={}",
            if with_addr { "RecvFrom" } else { "Recv" },
            fd,
            packet.source,
            copied,
            self.socket(fd).map_or(0, |socket| socket.recv_queue.len())
        );
        let out = if with_addr {
            bsd_result_with_len(copied as i32, 0, addr_len)
        } else {
            bsd_result(copied as i32, 0)
        };
        (out, false)
    }

    fn poll_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Vec<u8> {
        let Some(nfds) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        if nfds < 0 || nfds as usize > MAX_SOCKETS {
            return bsd_result(-1, EINVAL);
        }
        if nfds == 0 {
            return bsd_result(0, 0);
        }
        let Some(input_buffer) = send_buffer(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(output_buffer) = recv_buffer(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let size = nfds as usize * 8;
        if (input_buffer.size as usize) < size || (output_buffer.size as usize) < size {
            return bsd_result(-1, EINVAL);
        }
        let mut entries = vec![0u8; size];
        if memory.read(input_buffer.addr, &mut entries).is_err() {
            return bsd_result(-1, EFAULT);
        }
        let mut ready = 0i32;
        for entry in entries.chunks_exact_mut(8) {
            let fd = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            let events = u16::from_le_bytes([entry[4], entry[5]]);
            let revents = match self.socket(fd) {
                Some(socket) => {
                    let mut value = 0u16;
                    if events & POLLIN != 0 && !socket.recv_queue.is_empty() {
                        value |= POLLIN;
                    }
                    if events & POLLOUT != 0 {
                        value |= POLLOUT;
                    }
                    value
                }
                None => POLLNVAL,
            };
            entry[6..8].copy_from_slice(&revents.to_le_bytes());
            if revents != 0 {
                ready += 1;
            }
        }
        if memory.write(output_buffer.addr, &entries).is_err() {
            return bsd_result(-1, EFAULT);
        }
        log::trace!("bsd.Poll nfds={} ready={}", nfds, ready);
        bsd_result(ready, 0)
    }

    fn name_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx, peer: bool) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        if !peer {
            if self.socket(fd).is_none() {
                return bsd_result_with_len(-1, EBADF, 0);
            }
            self.ensure_bound(fd);
        }
        let address = match self.socket(fd) {
            Some(socket) if peer => socket.peer,
            Some(socket) => socket.local,
            None => return bsd_result_with_len(-1, EBADF, 0),
        };
        let Some(address) = address else {
            return bsd_result_with_len(-1, ENOTCONN, 0);
        };
        let Some(buffer) = recv_buffer(ctx, 0) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        let encoded = address.encode();
        let len = encoded.len().min(buffer.size as usize);
        if memory.write(buffer.addr, &encoded[..len]).is_err() {
            return bsd_result_with_len(-1, EFAULT, 0);
        }
        bsd_result_with_len(0, 0, len as u32)
    }

    fn get_sock_opt_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        let Some(level) = input_u32(ctx, 1) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        let Some(name) = input_u32(ctx, 2) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        let Some(socket) = self.socket(fd) else {
            return bsd_result_with_len(-1, EBADF, 0);
        };
        let Some(buffer) = recv_buffer(ctx, 0) else {
            return bsd_result_with_len(-1, EINVAL, 0);
        };
        let value = if level == SOL_SOCKET && name == SO_ERROR {
            socket.last_error.to_le_bytes().to_vec()
        } else if level == SOL_SOCKET && name == SO_TYPE {
            socket.socket_type.to_le_bytes().to_vec()
        } else if let Some(value) = socket.options.get(&(level, name)) {
            value.clone()
        } else {
            vec![0u8; (buffer.size as usize).min(4)]
        };
        let len = value.len().min(buffer.size as usize);
        if memory.write(buffer.addr, &value[..len]).is_err() {
            return bsd_result_with_len(-1, EFAULT, 0);
        }
        log::trace!(
            "bsd.GetSockOpt fd={} level={:#x} name={:#x} len={}",
            fd,
            level,
            name,
            len
        );
        bsd_result_with_len(0, 0, len as u32)
    }

    fn set_sock_opt_command(&mut self, memory: &AddressSpace, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(level) = input_u32(ctx, 1) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(name) = input_u32(ctx, 2) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(buffer) = send_buffer(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Ok(value) = read_guest(memory, buffer, 64) else {
            return bsd_result(-1, EFAULT);
        };
        let Some(socket) = self.socket_mut(fd) else {
            return bsd_result(-1, EBADF);
        };
        socket.options.insert((level, name), value);
        log::trace!(
            "bsd.SetSockOpt fd={} level={:#x} name={:#x}",
            fd,
            level,
            name
        );
        bsd_result(0, 0)
    }

    fn fcntl_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(cmd) = input_i32(ctx, 1) else {
            return bsd_result(-1, EINVAL);
        };
        let arg = input_i32(ctx, 2).unwrap_or(0);
        let Some(socket) = self.socket_mut(fd) else {
            return bsd_result(-1, EBADF);
        };
        match cmd {
            3 => bsd_result(socket.flags, 0),
            4 => {
                socket.flags = arg;
                bsd_result(0, 0)
            }
            _ => bsd_result(-1, EINVAL),
        }
    }

    fn listen_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        if self.socket(fd).is_none() {
            return bsd_result(-1, EBADF);
        }
        bsd_result(0, 0)
    }

    fn shutdown_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let how = input_i32(ctx, 1).unwrap_or(-1);
        if self.socket(fd).is_none() {
            return bsd_result(-1, EBADF);
        }
        if !(0..=2).contains(&how) {
            return bsd_result(-1, EINVAL);
        }
        bsd_result(0, 0)
    }

    fn close_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Ok(index) = usize::try_from(fd) else {
            return bsd_result(-1, EBADF);
        };
        let Some(slot) = self.sockets.get_mut(index) else {
            return bsd_result(-1, EBADF);
        };
        if slot.take().is_none() {
            return bsd_result(-1, EBADF);
        }
        log::debug!("bsd.Close fd={}", fd);
        bsd_result(0, 0)
    }

    fn duplicate_command(&mut self, ctx: &IpcCtx) -> Vec<u8> {
        let Some(fd) = input_i32(ctx, 0) else {
            return bsd_result(-1, EINVAL);
        };
        let Some(socket) = self.socket(fd).cloned() else {
            return bsd_result(-1, EBADF);
        };
        let Some(new_fd) = self.sockets.iter().position(Option::is_none) else {
            return bsd_result(-1, EMFILE);
        };
        self.sockets[new_fd] = Some(socket);
        bsd_result(new_fd as i32, 0)
    }

    fn socket(&self, fd: i32) -> Option<&VirtualSocket> {
        usize::try_from(fd)
            .ok()
            .and_then(|fd| self.sockets.get(fd))
            .and_then(Option::as_ref)
    }

    fn socket_mut(&mut self, fd: i32) -> Option<&mut VirtualSocket> {
        usize::try_from(fd)
            .ok()
            .and_then(|fd| self.sockets.get_mut(fd))
            .and_then(Option::as_mut)
    }

    fn ensure_bound(&mut self, fd: i32) {
        if self.socket(fd).is_some_and(|socket| socket.local.is_some()) {
            return;
        }
        let port = self.allocate_ephemeral_port();
        if let Some(socket) = self.socket_mut(fd) {
            socket.local = Some(BsdAddress::any(port));
        }
    }

    fn allocate_ephemeral_port(&mut self) -> u16 {
        for _ in 0..16_384 {
            let port = self.next_ephemeral_port;
            self.next_ephemeral_port = if port == u16::MAX { 49_152 } else { port + 1 };
            if !self
                .sockets
                .iter()
                .flatten()
                .any(|socket| socket.local.is_some_and(|address| address.port == port))
            {
                return port;
            }
        }
        self.next_ephemeral_port
    }
}

impl Default for BsdService {
    fn default() -> Self {
        Self::new()
    }
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

fn send_buffer(ctx: &IpcCtx, index: usize) -> Option<IpcBuffer> {
    ctx.send_buffers
        .iter()
        .chain(ctx.send_statics.iter())
        .filter(|buffer| buffer.addr != 0)
        .nth(index)
        .copied()
}

fn recv_buffer(ctx: &IpcCtx, index: usize) -> Option<IpcBuffer> {
    ctx.recv_buffers
        .iter()
        .chain(ctx.recv_statics.iter())
        .filter(|buffer| buffer.addr != 0)
        .nth(index)
        .copied()
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

fn address_matches(bound: BsdAddress, destination: BsdAddress) -> bool {
    bound.port == destination.port
        && (bound.ip == [0; 4]
            || bound.ip == destination.ip
            || destination.ip == [255, 255, 255, 255])
}

fn source_ip(bound: [u8; 4], destination: [u8; 4]) -> [u8; 4] {
    if bound != [0; 4] {
        bound
    } else if destination[0] == 127 {
        destination
    } else {
        [127, 0, 0, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sockaddr_round_trip() {
        let address = BsdAddress {
            ip: [127, 0, 0, 1],
            port: 19_132,
        };
        assert_eq!(BsdAddress::parse(&address.encode()), Some(address));
    }

    #[test]
    fn wildcard_address_receives_loopback() {
        assert!(address_matches(
            BsdAddress::any(19_132),
            BsdAddress {
                ip: [127, 0, 0, 1],
                port: 19_132,
            }
        ));
    }

    #[test]
    fn ephemeral_ports_do_not_collide() {
        let mut service = BsdService::new();
        service.sockets[0] = Some(VirtualSocket::new(AF_INET, SOCK_DGRAM, 0));
        service.ensure_bound(0);
        service.sockets[1] = Some(VirtualSocket::new(AF_INET, SOCK_DGRAM, 0));
        service.ensure_bound(1);
        assert_ne!(
            service.socket(0).unwrap().local,
            service.socket(1).unwrap().local
        );
    }
}
