use std::net::Ipv4Addr;

const GAI_AGAIN: i32 = 2;
const GAI_NO_DATA: i32 = 7;
const NETDB_HOST_NOT_FOUND: i32 = 1;
const NETDB_TRY_AGAIN: i32 = 2;

const ADDRINFO_MAGIC: u32 = 0xBEEF_CAFE;
const AF_INET: u32 = 2;
const SOCK_STREAM: u32 = 1;
const SOCK_DGRAM: u32 = 2;
const IPPROTO_TCP: u32 = 6;
const IPPROTO_UDP: u32 = 17;
const SOCKADDR_IN_LEN: u32 = 16;

pub fn sfdnsres_lookup_reply(cmd_id: u32) -> Option<Vec<u8>> {
    let words: &[i32] = match cmd_id {
        2 | 6 => &[NETDB_HOST_NOT_FOUND, GAI_NO_DATA, 0],
        10 | 12 => &[0, GAI_NO_DATA, NETDB_HOST_NOT_FOUND, 0],
        _ => return None,
    };
    Some(encode_words(words))
}

fn encode_words(words: &[i32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupFailure {
    NotFound,
    TryAgain,
}

pub fn nextendo_lookup(host: &str) -> Option<Result<Ipv4Addr, LookupFailure>> {
    if !nexium_common::nextendo::redirect().enabled {
        return None;
    }
    if let Ok(ip) = host.trim().parse::<Ipv4Addr>() {
        return Some(Ok(ip));
    }
    if let Some(ip) = nexium_common::nextendo::redirect_target(host) {
        return Some(Ok(ip));
    }
    Some(Err(LookupFailure::NotFound))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hints {
    pub socket_type: u32,
    pub protocol: u32,
}

pub fn parse_hints(bytes: &[u8]) -> Hints {
    let word = |offset: usize| -> Option<u32> {
        Some(u32::from_be_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?))
    };
    if word(0) != Some(ADDRINFO_MAGIC) {
        return Hints::default();
    }
    Hints {
        socket_type: word(12).unwrap_or(0),
        protocol: word(16).unwrap_or(0),
    }
}

pub fn parse_service(service: &str) -> u16 {
    let service = service.trim();
    if let Ok(port) = service.parse::<u16>() {
        return port;
    }
    match service.to_ascii_lowercase().as_str() {
        "http" => 80,
        "https" => 443,
        _ => 0,
    }
}

fn switch_order(ip: Ipv4Addr) -> [u8; 4] {
    u32::from(ip).to_le_bytes()
}

pub fn serialize_addrinfo(host: &str, address: Ipv4Addr, port: u16, hints: Hints) -> Vec<u8> {
    let kinds: Vec<(u32, u32)> = match hints.socket_type {
        0 => vec![(SOCK_STREAM, IPPROTO_TCP), (SOCK_DGRAM, IPPROTO_UDP)],
        SOCK_STREAM => vec![(SOCK_STREAM, if hints.protocol == 0 { IPPROTO_TCP } else { hints.protocol })],
        SOCK_DGRAM => vec![(SOCK_DGRAM, if hints.protocol == 0 { IPPROTO_UDP } else { hints.protocol })],
        other => vec![(other, hints.protocol)],
    };
    let mut out = Vec::with_capacity(kinds.len() * (41 + host.len()) + 4);
    for (socket_type, protocol) in kinds {
        for word in [ADDRINFO_MAGIC, 0, AF_INET, socket_type, protocol, SOCKADDR_IN_LEN] {
            out.extend_from_slice(&word.to_be_bytes());
        }
        out.push(SOCKADDR_IN_LEN as u8);
        out.push(AF_INET as u8);
        out.extend_from_slice(&port.to_le_bytes());
        out.extend_from_slice(&switch_order(address));
        out.extend_from_slice(&[0u8; 8]);
        out.extend_from_slice(host.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(&[0u8; 4]);
    out
}

pub fn serialize_hostent(host: &str, address: Ipv4Addr) -> Vec<u8> {
    let mut out = Vec::with_capacity(host.len() + 20);
    out.extend_from_slice(host.as_bytes());
    out.push(0);
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&(AF_INET as u16).to_be_bytes());
    out.extend_from_slice(&4u16.to_be_bytes());
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(&switch_order(address));
    out
}

pub fn addrinfo_reply(cmd_id: u32, result: Result<u32, LookupFailure>) -> Vec<u8> {
    let (size, gai, netdb) = match result {
        Ok(size) => (size as i32, 0, 0),
        Err(LookupFailure::TryAgain) => (0, GAI_AGAIN, NETDB_TRY_AGAIN),
        Err(LookupFailure::NotFound) => (0, GAI_NO_DATA, NETDB_HOST_NOT_FOUND),
    };
    if cmd_id == 12 {
        encode_words(&[size, gai, netdb, 0])
    } else {
        encode_words(&[0, gai, size])
    }
}

pub fn hostent_reply(cmd_id: u32, result: Result<u32, LookupFailure>) -> Vec<u8> {
    let (size, netdb) = match result {
        Ok(size) => (size as i32, 0),
        Err(LookupFailure::TryAgain) => (0, NETDB_TRY_AGAIN),
        Err(LookupFailure::NotFound) => (0, NETDB_HOST_NOT_FOUND),
    };
    if cmd_id == 10 {
        encode_words(&[size, netdb, 0])
    } else {
        encode_words(&[netdb, 0, size])
    }
}

pub fn nsd_resolve(fqdn: &str) -> String {
    fqdn.replace('%', "lp1")
}

pub fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn lookups_report_host_not_found_in_both_layouts() {
        assert_eq!(
            words(&sfdnsres_lookup_reply(2).unwrap()),
            vec![NETDB_HOST_NOT_FOUND, GAI_NO_DATA, 0]
        );
        assert_eq!(
            words(&sfdnsres_lookup_reply(12).unwrap()),
            vec![0, GAI_NO_DATA, NETDB_HOST_NOT_FOUND, 0]
        );
        assert!(sfdnsres_lookup_reply(14).is_none());
    }

    #[test]
    fn nsd_substitutes_the_environment() {
        assert_eq!(nsd_resolve("api.%.example.com"), "api.lp1.example.com");
        assert_eq!(c_string(b"host\0junk"), "host");
    }

    #[test]
    fn addrinfo_uses_the_switch_byte_order() {
        let host = "g2b309e01-lp1.s.n.srv.nintendo.net";
        let data = serialize_addrinfo(
            host,
            Ipv4Addr::new(51, 178, 29, 194),
            443,
            Hints {
                socket_type: SOCK_STREAM,
                protocol: 0,
            },
        );
        assert_eq!(data.len(), 24 + 16 + host.len() + 1 + 4);
        assert_eq!(&data[0..4], &[0xBE, 0xEF, 0xCA, 0xFE]);
        assert_eq!(&data[8..12], &[0, 0, 0, 2]);
        assert_eq!(&data[12..16], &[0, 0, 0, 1]);
        assert_eq!(&data[16..20], &[0, 0, 0, 6]);
        assert_eq!(&data[20..24], &[0, 0, 0, 16]);
        assert_eq!(&data[24..26], &[16, 2]);
        assert_eq!(&data[26..28], &443u16.to_le_bytes());
        assert_eq!(&data[28..32], &[194, 29, 178, 51]);
        assert_eq!(&data[40..40 + host.len()], host.as_bytes());
        assert_eq!(data[40 + host.len()], 0);
        assert_eq!(&data[data.len() - 4..], &[0, 0, 0, 0]);
    }

    #[test]
    fn addrinfo_without_hints_offers_tcp_and_udp() {
        let data = serialize_addrinfo("a", Ipv4Addr::LOCALHOST, 0, Hints::default());
        assert_eq!(data.len(), 2 * (24 + 16 + 2) + 4);
        assert_eq!(&data[12..16], &SOCK_STREAM.to_be_bytes());
        assert_eq!(&data[42 + 12..42 + 16], &SOCK_DGRAM.to_be_bytes());
        assert_eq!(&data[42 + 24..42 + 26], &[16, 2]);
    }

    #[test]
    fn hints_are_read_from_the_serialized_form() {
        let mut bytes = Vec::new();
        for word in [ADDRINFO_MAGIC, 0, AF_INET, SOCK_DGRAM, IPPROTO_UDP, 0] {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        assert_eq!(
            parse_hints(&bytes),
            Hints {
                socket_type: SOCK_DGRAM,
                protocol: IPPROTO_UDP
            }
        );
        assert_eq!(parse_hints(&[0u8; 8]), Hints::default());
    }

    #[test]
    fn hostent_lists_one_address() {
        let data = serialize_hostent("nncs1-lp1.n.n.srv.nintendo.net", Ipv4Addr::new(1, 2, 3, 4));
        let name_end = data.iter().position(|&byte| byte == 0).unwrap();
        let tail = &data[name_end + 1..];
        assert_eq!(&tail[0..4], &[0, 0, 0, 0]);
        assert_eq!(&tail[4..6], &[0, 2]);
        assert_eq!(&tail[6..8], &[0, 4]);
        assert_eq!(&tail[8..12], &[0, 0, 0, 1]);
        assert_eq!(&tail[12..16], &[4, 3, 2, 1]);
    }

    #[test]
    fn reply_layouts_follow_the_command() {
        assert_eq!(words(&addrinfo_reply(6, Ok(45))), vec![0, 0, 45]);
        assert_eq!(words(&addrinfo_reply(12, Ok(45))), vec![45, 0, 0, 0]);
        assert_eq!(
            words(&addrinfo_reply(6, Err(LookupFailure::NotFound))),
            vec![0, GAI_NO_DATA, 0]
        );
        assert_eq!(words(&hostent_reply(2, Ok(20))), vec![0, 0, 20]);
        assert_eq!(words(&hostent_reply(10, Ok(20))), vec![20, 0, 0]);
    }

    #[test]
    fn services_map_to_ports() {
        assert_eq!(parse_service("443"), 443);
        assert_eq!(parse_service("https"), 443);
        assert_eq!(parse_service(""), 0);
        assert_eq!(parse_service("nonsense"), 0);
    }
}
