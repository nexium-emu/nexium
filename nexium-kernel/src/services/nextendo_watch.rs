use std::collections::HashMap;
use std::time::Instant;

const MAX_BUFFER: usize = 1 << 20;
const PRUDP_LITE_MAGIC: u8 = 0x80;
const PRUDP_DATA: u16 = 2;
const PRUDP_DISCONNECT: u16 = 3;
const PRUDP_FLAG_ACK: u16 = 0x1;
const WEBSOCKET_BINARY: u8 = 0x2;
const WEBSOCKET_CLOSE: u8 = 0x8;
const TICKET_GRANTING: u16 = 10;
const NOTIFICATION_EVENTS: u16 = 14;

#[derive(Debug, PartialEq, Eq)]
enum Rmc<'a> {
    Request {
        protocol: u16,
        call: u32,
        method: u32,
        params: &'a [u8],
    },
    Success {
        call: u32,
        body: &'a [u8],
    },
    Failure {
        error: u32,
        call: u32,
    },
}

#[derive(Default)]
struct Stream {
    buffer: Vec<u8>,
    fragments: Vec<u8>,
}

pub struct NexWatch {
    host: String,
    opened: Instant,
    upgraded: bool,
    broken: bool,
    inbound: Stream,
    outbound: Stream,
    calls: HashMap<u32, (u16, u32)>,
    requests: u32,
    failures: u32,
    disconnected: bool,
}

impl NexWatch {
    pub fn new(host: &str) -> Self {
        Self {
            host: host.to_string(),
            opened: Instant::now(),
            upgraded: false,
            broken: false,
            inbound: Stream::default(),
            outbound: Stream::default(),
            calls: HashMap::new(),
            requests: 0,
            failures: 0,
            disconnected: false,
        }
    }

    pub fn outgoing(&mut self, data: &[u8]) {
        if !self.upgraded || self.broken {
            return;
        }
        self.outbound.buffer.extend_from_slice(data);
        while let Some(frame) = take_frame(&mut self.outbound.buffer) {
            match frame {
                Some((WEBSOCKET_BINARY, payload)) => {
                    for message in messages(&payload, &mut self.outbound.fragments) {
                        if let Some(Rmc::Request { protocol, call, method, .. }) = parse_rmc(&message) {
                            self.calls.insert(call, (protocol, method));
                            self.requests += 1;
                            log::info!("nex: {} called {}", self.host, describe(protocol, method));
                        }
                    }
                }
                Some(_) => {}
                None => return self.give_up(),
            }
        }
        self.check_size();
    }

    pub fn incoming(&mut self, data: &[u8]) {
        if self.broken {
            return;
        }
        self.inbound.buffer.extend_from_slice(data);
        if !self.upgraded {
            let Some(end) = find(&self.inbound.buffer, b"\r\n\r\n") else {
                return self.check_size();
            };
            if !self.inbound.buffer.starts_with(b"HTTP/1.1 101") {
                return self.give_up();
            }
            self.inbound.buffer.drain(..end + 4);
            self.upgraded = true;
        }
        while let Some(frame) = take_frame(&mut self.inbound.buffer) {
            match frame {
                Some((WEBSOCKET_BINARY, payload)) => {
                    if packets(&payload).any(|packet| packet.kind == PRUDP_DISCONNECT) && !self.disconnected {
                        self.disconnected = true;
                        log::info!("nex: {} ended the session (PRUDP disconnect)", self.host);
                    }
                    for message in messages(&payload, &mut self.inbound.fragments) {
                        self.server_message(&message);
                    }
                }
                Some((WEBSOCKET_CLOSE, payload)) => {
                    let code = payload
                        .get(..2)
                        .map_or(0, |code| u16::from_be_bytes([code[0], code[1]]));
                    log::info!("nex: {} closed the WebSocket (code {})", self.host, code);
                }
                Some(_) => {}
                None => return self.give_up(),
            }
        }
        self.check_size();
    }

    fn server_message(&mut self, message: &[u8]) {
        match parse_rmc(message) {
            Some(Rmc::Failure { error, call }) => {
                self.failures += 1;
                match self.calls.remove(&call) {
                    Some((protocol, method)) => log::info!(
                        "nex: {} rejected {} with error {:#010X}",
                        self.host,
                        describe(protocol, method),
                        error
                    ),
                    None => log::info!("nex: {} rejected call {} with error {:#010X}", self.host, call, error),
                }
            }
            Some(Rmc::Success { call, body }) => {
                if let Some((protocol, method)) = self.calls.remove(&call) {
                    if protocol == TICKET_GRANTING {
                        match login_result(method, body) {
                            Some(Ok(())) => {
                                log::info!("nex: {} accepted {}", self.host, describe(protocol, method))
                            }
                            Some(Err(result)) => log::info!(
                                "nex: {} refused {} with {:#010X}",
                                self.host,
                                describe(protocol, method),
                                result
                            ),
                            None => {}
                        }
                    }
                }
            }
            Some(Rmc::Request { protocol, params, .. }) if protocol == NOTIFICATION_EVENTS => {
                log::info!("nex: {} sent notification type {}", self.host, notification_type(params));
            }
            Some(Rmc::Request { protocol, method, .. }) => {
                log::info!("nex: {} asked this console for {}", self.host, describe(protocol, method));
            }
            None => {}
        }
    }

    fn check_size(&mut self) {
        if self.inbound.buffer.len() > MAX_BUFFER || self.outbound.buffer.len() > MAX_BUFFER {
            self.give_up();
        }
    }

    fn give_up(&mut self) {
        self.broken = true;
        self.inbound = Stream::default();
        self.outbound = Stream::default();
    }
}

impl Drop for NexWatch {
    fn drop(&mut self) {
        if self.upgraded {
            log::info!(
                "nex: session with {} closed after {}s ({} calls, {} rejected)",
                self.host,
                self.opened.elapsed().as_secs(),
                self.requests,
                self.failures
            );
        }
    }
}

fn login_result(method: u32, body: &[u8]) -> Option<Result<(), u32>> {
    if !matches!(method, 1 | 2 | 3 | 6) {
        return None;
    }
    let result = u32::from_le_bytes(body.get(4..8)?.try_into().ok()?);
    Some(if result & 0x8000_0000 != 0 { Err(result) } else { Ok(()) })
}

fn notification_type(params: &[u8]) -> u32 {
    let start = match params.get(1..5) {
        Some(size) if u32::from_le_bytes(size.try_into().unwrap()) as usize + 5 == params.len() => 5,
        _ => 0,
    };
    params
        .get(start + 8..start + 12)
        .map_or(0, |kind| u32::from_le_bytes(kind.try_into().unwrap()))
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn take_frame(buffer: &mut Vec<u8>) -> Option<Option<(u8, Vec<u8>)>> {
    let first = *buffer.first()?;
    let second = *buffer.get(1)?;
    let (len, mut pos) = match second & 0x7F {
        126 => (u16::from_be_bytes(buffer.get(2..4)?.try_into().ok()?) as usize, 4),
        127 => (u64::from_be_bytes(buffer.get(2..10)?.try_into().ok()?) as usize, 10),
        short => (short as usize, 2),
    };
    if len > MAX_BUFFER {
        return Some(None);
    }
    let mask: Option<[u8; 4]> = if second & 0x80 != 0 {
        let key = buffer.get(pos..pos + 4)?.try_into().ok()?;
        pos += 4;
        Some(key)
    } else {
        None
    };
    let mut payload = buffer.get(pos..pos + len)?.to_vec();
    if let Some(mask) = mask {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[index % 4];
        }
    }
    buffer.drain(..pos + len);
    Some(Some((first & 0x0F, payload)))
}

struct Packet<'a> {
    kind: u16,
    flags: u16,
    fragment: u8,
    body: &'a [u8],
}

fn packets(payload: &[u8]) -> impl Iterator<Item = Packet<'_>> {
    let mut rest = payload;
    std::iter::from_fn(move || {
        if rest.len() < 12 || rest[0] != PRUDP_LITE_MAGIC {
            return None;
        }
        let options = rest[1] as usize;
        let total = 12 + options + u16::from_le_bytes([rest[2], rest[3]]) as usize;
        if rest.len() < total {
            return None;
        }
        let type_flags = u16::from_le_bytes([rest[8], rest[9]]);
        let packet = Packet {
            kind: type_flags & 0xF,
            flags: type_flags >> 4,
            fragment: rest[7],
            body: &rest[12 + options..total],
        };
        rest = &rest[total..];
        Some(packet)
    })
}

fn messages(payload: &[u8], fragments: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut complete = Vec::new();
    for packet in packets(payload) {
        if packet.kind != PRUDP_DATA || packet.flags & PRUDP_FLAG_ACK != 0 || packet.body.is_empty() {
            continue;
        }
        fragments.extend_from_slice(packet.body);
        if packet.fragment == 0 {
            complete.push(std::mem::take(fragments));
        } else if fragments.len() > MAX_BUFFER {
            fragments.clear();
        }
    }
    complete
}

fn parse_rmc(message: &[u8]) -> Option<Rmc<'_>> {
    let size = u32::from_le_bytes(message.get(..4)?.try_into().ok()?) as usize;
    if size != message.len() - 4 {
        return None;
    }
    let first = *message.get(4)?;
    let (protocol, mut pos) = match first & 0x7F {
        0x7F => (u16::from_le_bytes(message.get(5..7)?.try_into().ok()?), 7),
        id => (u16::from(id), 5),
    };
    let word = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(message.get(at..at + 4)?.try_into().ok()?)) };
    if first & 0x80 != 0 {
        return Some(Rmc::Request {
            protocol,
            call: word(pos)?,
            method: word(pos + 4)?,
            params: &message[pos + 8..],
        });
    }
    let success = *message.get(pos)?;
    pos += 1;
    if success == 1 {
        Some(Rmc::Success {
            call: word(pos)?,
            body: message.get(pos + 4..)?,
        })
    } else {
        Some(Rmc::Failure {
            error: word(pos)?,
            call: word(pos + 4)?,
        })
    }
}

fn protocol_name(protocol: u16) -> Option<&'static str> {
    Some(match protocol {
        3 => "NATTraversal",
        10 => "TicketGranting",
        11 => "SecureConnection",
        14 => "NotificationEvents",
        18 => "Health",
        21 => "MatchMaking",
        50 => "MatchMakingExt",
        101 | 102 => "Friends",
        109 => "MatchmakeExtension",
        110 => "Utility",
        112 => "Ranking",
        115 => "DataStore",
        120 => "MatchmakeReferee",
        121 => "Subscriber",
        122 => "Ranking2",
        _ => return None,
    })
}

fn describe(protocol: u16, method: u32) -> String {
    match protocol_name(protocol) {
        Some(name) => format!("{name}.{method:#x}"),
        None => format!("protocol {protocol}.{method:#x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmc_request(protocol: u8, call: u32, method: u32, params: &[u8]) -> Vec<u8> {
        let mut rmc = ((9 + params.len()) as u32).to_le_bytes().to_vec();
        rmc.push(0x80 | protocol);
        rmc.extend_from_slice(&call.to_le_bytes());
        rmc.extend_from_slice(&method.to_le_bytes());
        rmc.extend_from_slice(params);
        rmc
    }

    fn rmc_failure(protocol: u8, error: u32, call: u32) -> Vec<u8> {
        let mut rmc = 10u32.to_le_bytes().to_vec();
        rmc.push(protocol);
        rmc.push(0);
        rmc.extend_from_slice(&error.to_le_bytes());
        rmc.extend_from_slice(&call.to_le_bytes());
        rmc
    }

    fn prudp(kind: u16, fragment: u8, options: &[u8], body: &[u8]) -> Vec<u8> {
        let mut packet = vec![PRUDP_LITE_MAGIC, options.len() as u8];
        packet.extend_from_slice(&(body.len() as u16).to_le_bytes());
        packet.extend_from_slice(&[0, 0, 0, fragment]);
        packet.extend_from_slice(&kind.to_le_bytes());
        packet.extend_from_slice(&[0, 0]);
        packet.extend_from_slice(options);
        packet.extend_from_slice(body);
        packet
    }

    fn frame(opcode: u8, payload: &[u8], mask: Option<[u8; 4]>) -> Vec<u8> {
        let mut out = vec![0x80 | opcode];
        let bit = if mask.is_some() { 0x80 } else { 0 };
        if payload.len() <= 125 {
            out.push(bit | payload.len() as u8);
        } else {
            out.push(bit | 126);
            out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        }
        let mut body = payload.to_vec();
        if let Some(mask) = mask {
            out.extend_from_slice(&mask);
            for (index, byte) in body.iter_mut().enumerate() {
                *byte ^= mask[index % 4];
            }
        }
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn rmc_messages_are_classified() {
        let request = rmc_request(109, 7, 0x28, &[1, 2, 3]);
        assert_eq!(
            parse_rmc(&request),
            Some(Rmc::Request { protocol: 109, call: 7, method: 0x28, params: &[1, 2, 3] })
        );
        let failure = rmc_failure(109, 0x8003_006A, 7);
        assert_eq!(parse_rmc(&failure), Some(Rmc::Failure { error: 0x8003_006A, call: 7 }));
        assert_eq!(parse_rmc(&[1, 0, 0, 0]), None);
    }

    #[test]
    fn calls_are_matched_with_their_failures_across_split_reads() {
        let mut watch = NexWatch::new("test");
        watch.incoming(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n");
        assert!(watch.upgraded);
        let request = frame(WEBSOCKET_BINARY, &prudp(PRUDP_DATA, 0, &[], &rmc_request(109, 9, 0x28, &[])), Some([9, 8, 7, 6]));
        watch.outgoing(&request);
        assert_eq!(watch.calls.get(&9), Some(&(109, 0x28)));
        let reply = frame(WEBSOCKET_BINARY, &prudp(PRUDP_DATA, 0, &[], &rmc_failure(109, 0x8001_0002, 9)), None);
        let (head, tail) = reply.split_at(5);
        watch.incoming(head);
        assert_eq!(watch.failures, 0);
        watch.incoming(tail);
        assert_eq!(watch.failures, 1);
        assert!(watch.calls.is_empty());
        assert!(!watch.broken);
    }

    #[test]
    fn fragments_are_joined_before_parsing() {
        let message = rmc_request(11, 3, 0x7, &[0u8; 40]);
        let (first, second) = message.split_at(20);
        let mut fragments = Vec::new();
        assert!(messages(&prudp(PRUDP_DATA, 1, &[0, 1, 0], first), &mut fragments).is_empty());
        let done = messages(&prudp(PRUDP_DATA, 0, &[], second), &mut fragments);
        assert_eq!(done, vec![message]);
    }

    #[test]
    fn split_logins_are_counted_and_their_result_reported() {
        let mut watch = NexWatch::new("test");
        watch.incoming(b"HTTP/1.1 101 Switching Protocols\r\n\r\n");
        let login = rmc_request(TICKET_GRANTING as u8, 4, 2, &[7u8; 1500]);
        let (first, second) = login.split_at(1000);
        watch.outgoing(&frame(WEBSOCKET_BINARY, &prudp(PRUDP_DATA, 1, &[], first), Some([1, 2, 3, 4])));
        assert_eq!(watch.requests, 0);
        watch.outgoing(&frame(WEBSOCKET_BINARY, &prudp(PRUDP_DATA, 0, &[], second), Some([1, 2, 3, 4])));
        assert_eq!(watch.requests, 1);
        assert_eq!(watch.calls.get(&4), Some(&(TICKET_GRANTING, 2)));
        let mut reply = 14u32.to_le_bytes().to_vec();
        reply.extend_from_slice(&[TICKET_GRANTING as u8, 1]);
        reply.extend_from_slice(&4u32.to_le_bytes());
        reply.extend_from_slice(&0x8002u32.to_le_bytes());
        reply.extend_from_slice(&0x8068_000Bu32.to_le_bytes());
        watch.incoming(&frame(WEBSOCKET_BINARY, &prudp(PRUDP_DATA, 0, &[], &reply), None));
        assert!(watch.calls.is_empty());
    }

    #[test]
    fn login_replies_carry_their_result_code() {
        let reply = |result: u32| [0x8002u32.to_le_bytes(), result.to_le_bytes()].concat();
        assert_eq!(login_result(2, &reply(0x8068_000B)), Some(Err(0x8068_000B)));
        assert_eq!(login_result(6, &reply(0x0001_0001)), Some(Ok(())));
        assert_eq!(login_result(4, &reply(0x8068_000B)), None);
        assert_eq!(login_result(2, &[2, 0x80]), None);
    }

    #[test]
    fn notification_types_skip_the_structure_header() {
        let mut event = 1_800_003_542u64.to_le_bytes().to_vec();
        event.extend_from_slice(&3001u32.to_le_bytes());
        event.extend_from_slice(&[0u8; 12]);
        let mut versioned = vec![0];
        versioned.extend_from_slice(&(event.len() as u32).to_le_bytes());
        versioned.extend_from_slice(&event);
        assert_eq!(notification_type(&versioned), 3001);
        assert_eq!(notification_type(&event), 3001);
    }

    #[test]
    fn a_refused_upgrade_stops_the_watch() {
        let mut watch = NexWatch::new("test");
        watch.incoming(b"HTTP/1.1 403 Forbidden\r\n\r\n");
        assert!(watch.broken);
        assert!(!watch.upgraded);
    }
}
