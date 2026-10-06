use std::net::Ipv4Addr;

const STATION_MARKER: &[u8] = b"prudp:/address=";
const WEBSOCKET_BINARY: u8 = 0x2;
const PRUDP_LITE_MAGIC: u8 = 0x80;
const PRUDP_DATA: u16 = 2;
const SECURE_CONNECTION: u8 = 0x0B;
const MATCHMAKE_EXTENSION: u8 = 0x6D;

pub struct Outgoing {
    pub method: &'static str,
    pub nat: Option<(u8, u8)>,
    pub rewritten: Option<Vec<u8>>,
}

pub fn rewrite_outgoing(frame: &[u8], external: Ipv4Addr) -> Option<Vec<u8>> {
    inspect_outgoing(frame, Some(external))?.rewritten
}

pub fn inspect_outgoing(frame: &[u8], external: Option<Ipv4Addr>) -> Option<Outgoing> {
    let first = *frame.first()?;
    if first & 0x80 == 0 || first & 0x0F != WEBSOCKET_BINARY {
        return None;
    }
    let second = *frame.get(1)?;
    let (len, mut pos) = match second & 0x7F {
        126 => (
            u16::from_be_bytes(frame.get(2..4)?.try_into().ok()?) as usize,
            4,
        ),
        127 => return None,
        short => (short as usize, 2),
    };
    let mask: Option<[u8; 4]> = if second & 0x80 != 0 {
        let key = frame.get(pos..pos + 4)?.try_into().ok()?;
        pos += 4;
        Some(key)
    } else {
        None
    };
    if frame.len() != pos + len {
        return None;
    }
    let mut packet = frame[pos..].to_vec();
    if let Some(mask) = mask {
        apply_mask(&mut packet, mask);
    }
    if packet.len() < 12 || packet[0] != PRUDP_LITE_MAGIC {
        return None;
    }
    let start = 12 + packet[1] as usize;
    let payload_len = u16::from_le_bytes([packet[2], packet[3]]) as usize;
    if packet.len() != start + payload_len {
        return None;
    }
    if u16::from_le_bytes([packet[8], packet[9]]) & 0xF != PRUDP_DATA {
        return None;
    }
    let rmc = &packet[start..];
    if rmc.len() < 13 {
        return None;
    }
    let declared = u32::from_le_bytes(rmc[0..4].try_into().ok()?) as usize;
    if declared != rmc.len() - 4 || rmc[4] & 0x80 == 0 {
        return None;
    }
    let method = method_name(rmc[4] & 0x7F, u32::from_le_bytes(rmc[9..13].try_into().ok()?))?;
    let body = &rmc[13..];
    let rewritten = external
        .and_then(|external| rewrite_stations(body, external))
        .and_then(|body| {
            let mut new_rmc = Vec::with_capacity(13 + body.len());
            new_rmc.extend_from_slice(&u32::try_from(9 + body.len()).ok()?.to_le_bytes());
            new_rmc.extend_from_slice(&rmc[4..13]);
            new_rmc.extend_from_slice(&body);
            let mut new_packet = packet[..start].to_vec();
            new_packet[2..4].copy_from_slice(&u16::try_from(new_rmc.len()).ok()?.to_le_bytes());
            new_packet.extend_from_slice(&new_rmc);
            wrap_frame(first, mask, new_packet)
        });
    Some(Outgoing {
        method,
        nat: first_station_nat(body),
        rewritten,
    })
}

fn method_name(protocol: u8, method: u32) -> Option<&'static str> {
    match (protocol, method) {
        (SECURE_CONNECTION, 0x1) => Some("SecureConnection.Register"),
        (SECURE_CONNECTION, 0x7) => Some("SecureConnection.ReplaceURL"),
        (MATCHMAKE_EXTENSION, 0x26) => Some("MatchmakeExtension.CreateMatchmakeSessionWithParam"),
        (MATCHMAKE_EXTENSION, 0x27) => Some("MatchmakeExtension.JoinMatchmakeSessionWithParam"),
        (MATCHMAKE_EXTENSION, 0x28) => Some("MatchmakeExtension.AutoMatchmakeWithParam"),
        _ => None,
    }
}

fn wrap_frame(first: u8, mask: Option<[u8; 4]>, mut payload: Vec<u8>) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.push(first);
    let mask_bit = if mask.is_some() { 0x80 } else { 0 };
    match payload.len() {
        len @ 0..=125 => out.push(mask_bit | len as u8),
        len @ 126..=0xFFFF => {
            out.push(mask_bit | 126);
            out.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => return None,
    }
    if let Some(mask) = mask {
        out.extend_from_slice(&mask);
        apply_mask(&mut payload, mask);
    }
    out.extend_from_slice(&payload);
    Some(out)
}

fn apply_mask(bytes: &mut [u8], mask: [u8; 4]) {
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte ^= mask[index % 4];
    }
}

fn first_station_nat(body: &[u8]) -> Option<(u8, u8)> {
    let marker = find(body, STATION_MARKER)?;
    let len = u16::from_le_bytes([*body.get(marker.checked_sub(2)?)?, body[marker - 1]]) as usize;
    let station = std::str::from_utf8(body.get(marker..marker + len)?).ok()?;
    let digit = |key: &str| -> Option<u8> {
        let at = station.find(key)? + key.len();
        station.as_bytes().get(at).filter(|byte| byte.is_ascii_digit()).map(|byte| byte - b'0')
    };
    Some((digit("natf=")?, digit("natm=")?))
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn rewrite_stations(body: &[u8], external: Ipv4Addr) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(body.len() + 16);
    let mut cursor = 0;
    let mut search = 0;
    let mut changed = false;
    while let Some(offset) = body.get(search..).and_then(|rest| find(rest, STATION_MARKER)) {
        let marker = search + offset;
        search = marker + STATION_MARKER.len();
        if marker < 2 {
            continue;
        }
        let len = u16::from_le_bytes([body[marker - 2], body[marker - 1]]) as usize;
        let Some(station) = body.get(marker..marker + len) else {
            continue;
        };
        if let Some(fixed) = fix_station(station, external) {
            out.extend_from_slice(&body[cursor..marker - 2]);
            out.extend_from_slice(&u16::try_from(fixed.len()).ok()?.to_le_bytes());
            out.extend_from_slice(&fixed);
            cursor = marker + len;
            changed = true;
        }
        search = search.max(marker + len);
    }
    if !changed {
        return None;
    }
    out.extend_from_slice(&body[cursor..]);
    Some(out)
}

fn is_private(address: Ipv4Addr) -> bool {
    let [a, b, ..] = address.octets();
    address.is_private() || address.is_loopback() || (a == 100 && (64..=127).contains(&b))
}

fn fix_station(station: &[u8], external: Ipv4Addr) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(station).ok()?;
    let start = text.find("address=")? + "address=".len();
    let end = start + text[start..].find(';')?;
    let address: Ipv4Addr = text[start..end].parse().ok()?;
    if !is_private(address) {
        return None;
    }
    let natf = text.find("natf=")? + "natf=".len();
    if text.as_bytes().get(natf).copied().unwrap_or(b'0') == b'0' {
        return None;
    }
    let mut fixed = String::with_capacity(text.len() + 8);
    fixed.push_str(&text[..start]);
    fixed.push_str(&external.to_string());
    fixed.push_str(&text[end..]);
    Some(fixed.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTERNAL: Ipv4Addr = Ipv4Addr::new(203, 0, 113, 7);

    fn nex_string(text: &str) -> Vec<u8> {
        let mut out = ((text.len() + 1) as u16).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out.push(0);
        out
    }

    fn rmc_request(protocol: u8, method: u32, body: &[u8]) -> Vec<u8> {
        let mut rmc = ((9 + body.len()) as u32).to_le_bytes().to_vec();
        rmc.push(0x80 | protocol);
        rmc.extend_from_slice(&7u32.to_le_bytes());
        rmc.extend_from_slice(&method.to_le_bytes());
        rmc.extend_from_slice(body);
        rmc
    }

    fn prudp_data(rmc: &[u8]) -> Vec<u8> {
        let mut packet = vec![PRUDP_LITE_MAGIC, 0];
        packet.extend_from_slice(&(rmc.len() as u16).to_le_bytes());
        packet.extend_from_slice(&[0, 0, 0, 0]);
        packet.extend_from_slice(&0x0002u16.to_le_bytes());
        packet.extend_from_slice(&[0, 0]);
        packet.extend_from_slice(rmc);
        packet
    }

    fn ws_frame(payload: &[u8], mask: Option<[u8; 4]>) -> Vec<u8> {
        let mut frame = vec![0x82];
        let bit = if mask.is_some() { 0x80 } else { 0 };
        if payload.len() <= 125 {
            frame.push(bit | payload.len() as u8);
        } else {
            frame.push(bit | 126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        }
        let mut body = payload.to_vec();
        if let Some(mask) = mask {
            frame.extend_from_slice(&mask);
            apply_mask(&mut body, mask);
        }
        frame.extend_from_slice(&body);
        frame
    }

    fn unwrap_station(frame: &[u8]) -> String {
        let text = String::from_utf8_lossy(frame).into_owned();
        let start = text.find("prudp:/").unwrap();
        let end = start + text[start..].find('\0').unwrap();
        text[start..end].to_string()
    }

    #[test]
    fn private_stations_in_register_get_the_public_address() {
        let station = "prudp:/address=192.168.1.20;port=12345;natf=1;natm=1;type=3";
        let mut body = 1u32.to_le_bytes().to_vec();
        body.extend_from_slice(&nex_string(station));
        let frame = ws_frame(&prudp_data(&rmc_request(SECURE_CONNECTION, 1, &body)), None);
        let rewritten = rewrite_outgoing(&frame, EXTERNAL).expect("rewritten");
        assert_eq!(
            unwrap_station(&rewritten),
            "prudp:/address=203.0.113.7;port=12345;natf=1;natm=1;type=3"
        );
        let payload = &rewritten[2..];
        assert_eq!(rewritten[1] as usize, payload.len());
        let rmc = &payload[12..];
        assert_eq!(
            u16::from_le_bytes([payload[2], payload[3]]) as usize,
            rmc.len()
        );
        assert_eq!(
            u32::from_le_bytes(rmc[0..4].try_into().unwrap()) as usize,
            rmc.len() - 4
        );
    }

    #[test]
    fn masked_frames_are_rewritten_and_remasked() {
        let station = "prudp:/address=10.0.0.5;port=1;natf=2;natm=0;type=3";
        let frame = ws_frame(
            &prudp_data(&rmc_request(MATCHMAKE_EXTENSION, 0x28, &nex_string(station))),
            Some([1, 2, 3, 4]),
        );
        let rewritten = rewrite_outgoing(&frame, EXTERNAL).expect("rewritten");
        let mut payload = rewritten[6..].to_vec();
        apply_mask(&mut payload, [1, 2, 3, 4]);
        assert_eq!(
            unwrap_station(&payload),
            "prudp:/address=203.0.113.7;port=1;natf=2;natm=0;type=3"
        );
    }

    #[test]
    fn local_and_public_stations_are_left_alone() {
        for station in [
            "prudp:/address=192.168.1.20;port=1;natf=0;natm=0;type=3",
            "prudp:/address=8.8.4.4;port=1;natf=1;natm=1;type=3",
        ] {
            let frame = ws_frame(
                &prudp_data(&rmc_request(SECURE_CONNECTION, 7, &nex_string(station))),
                None,
            );
            assert!(rewrite_outgoing(&frame, EXTERNAL).is_none(), "{station}");
        }
    }

    #[test]
    fn other_methods_and_non_frames_pass_through() {
        let station = "prudp:/address=192.168.1.20;port=1;natf=1;natm=1;type=3";
        let frame = ws_frame(
            &prudp_data(&rmc_request(SECURE_CONNECTION, 4, &nex_string(station))),
            None,
        );
        assert!(rewrite_outgoing(&frame, EXTERNAL).is_none());
        assert!(rewrite_outgoing(b"GET / HTTP/1.1\r\n\r\n", EXTERNAL).is_none());
        assert!(rewrite_outgoing(&[], EXTERNAL).is_none());
    }

    #[test]
    fn inspection_names_the_call_and_reads_the_nat_type() {
        let station = "prudp:/address=192.168.1.20;port=1;natf=2;natm=1;type=3";
        let frame = ws_frame(
            &prudp_data(&rmc_request(SECURE_CONNECTION, 7, &nex_string(station))),
            None,
        );
        let before = inspect_outgoing(&frame, None).expect("station call");
        assert_eq!(before.method, "SecureConnection.ReplaceURL");
        assert_eq!(before.nat, Some((2, 1)));
        assert!(before.rewritten.is_none());
        let after = inspect_outgoing(&frame, Some(EXTERNAL)).expect("station call");
        assert!(after.rewritten.is_some());

        let bare = ws_frame(
            &prudp_data(&rmc_request(MATCHMAKE_EXTENSION, 0x28, &[0u8; 12])),
            None,
        );
        let matchmake = inspect_outgoing(&bare, Some(EXTERNAL)).expect("matchmake call");
        assert_eq!(matchmake.method, "MatchmakeExtension.AutoMatchmakeWithParam");
        assert_eq!(matchmake.nat, None);
        assert!(matchmake.rewritten.is_none());
    }

    #[test]
    fn carrier_grade_nat_counts_as_private() {
        assert!(is_private(Ipv4Addr::new(100, 64, 0, 1)));
        assert!(is_private(Ipv4Addr::new(172, 20, 0, 1)));
        assert!(!is_private(Ipv4Addr::new(100, 128, 0, 1)));
    }
}
