use nexium_common::nextendo::{self, Friend, LocalPresence};

pub const FRIEND_SIZE: usize = 0x200;
pub const PROFILE_SIZE: usize = 0x100;
pub const USER_PRESENCE_SIZE: usize = 0xE0;
const NICKNAME_LIMIT: usize = 0x20;
const APP_FIELD_SIZE: usize = 0xC0;
const UID_HIGH: u64 = 0x1100_0000_0000_0000;
const NEVER_OFFLINE: i64 = i64::MAX;
const IMAGE_HOST: &str = "https://cdn-image-e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com/1";

pub struct FriendsService;

impl FriendsService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("friends cmd: {}", cmd_id);
        0
    }
}

impl Default for FriendsService {
    fn default() -> Self {
        Self::new()
    }
}

pub fn active() -> bool {
    nextendo::redirect().enabled && nextendo::account().is_some()
}

pub fn advertised_id(friend: &Friend) -> u64 {
    if friend.console && friend.account_id != 0 {
        friend.account_id
    } else {
        friend.pid
    }
}

pub fn matches_id(friend: &Friend, id: u64) -> bool {
    friend.pid == id || (friend.account_id != 0 && friend.account_id == id)
}

pub fn friend_uid(id: u64) -> [u8; 16] {
    let mut uid = [0u8; 16];
    uid[..8].copy_from_slice(&id.to_le_bytes());
    uid[8..].copy_from_slice(&UID_HIGH.to_le_bytes());
    uid
}

fn copy_nickname(out: &mut [u8], name: &str) {
    let mut end = name.len().min(NICKNAME_LIMIT);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    out[..end].copy_from_slice(&name.as_bytes()[..end]);
}

pub fn encode_friend(friend: &Friend, requested: Option<u64>) -> [u8; FRIEND_SIZE] {
    let mut out = [0u8; FRIEND_SIZE];
    let id = requested.unwrap_or_else(|| advertised_id(friend));
    out[0x00..0x10].copy_from_slice(&friend_uid(id));
    out[0x10..0x18].copy_from_slice(&id.to_le_bytes());
    copy_nickname(&mut out[0x18..0x39], &friend.name);
    out[0x40..0x48].copy_from_slice(&friend.app_id.to_le_bytes());
    out[0x48..0x50].copy_from_slice(&friend.app_id.to_le_bytes());
    out[0x50..0x58].copy_from_slice(&NEVER_OFFLINE.to_le_bytes());
    out[0x58..0x5C].copy_from_slice(&(friend.status.max(0) as u32).to_le_bytes());
    out[0x5C] = u8::from(friend.status > 0);
    let blob = friend.app_field.len().min(APP_FIELD_SIZE);
    out[0x60..0x60 + blob].copy_from_slice(&friend.app_field[..blob]);
    out[0x128] = 1;
    out
}

pub fn encode_profile(id: u64, name: &str, owner_pid: u64, account_id: u64) -> [u8; PROFILE_SIZE] {
    let mut out = [0u8; PROFILE_SIZE];
    out[0x00..0x08].copy_from_slice(&id.to_le_bytes());
    copy_nickname(&mut out[0x08..0x29], name);
    let url = if account_id != 0 {
        format!("{IMAGE_HOST}/fr_{account_id:016x}?pid={owner_pid}")
    } else {
        format!("{IMAGE_HOST}/pid_{id}?pid={owner_pid}")
    };
    let url_len = url.len().min(0x9F);
    out[0x30..0x30 + url_len].copy_from_slice(&url.as_bytes()[..url_len]);
    out[0xD0] = 1;
    out
}

pub fn profile_for(id: u64) -> Option<[u8; PROFILE_SIZE]> {
    if let Some(account) = nextendo::account().filter(|account| account.pid == id) {
        return Some(encode_profile(id, &account.username, id, 0));
    }
    let friends = nextendo::friends();
    let friend = friends.iter().find(|friend| matches_id(friend, id))?;
    Some(encode_profile(id, &friend.name, friend.pid, friend.account_id))
}

fn tokens(app_field: &[u8]) -> Vec<(usize, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < app_field.len() {
        let end = app_field[pos..]
            .iter()
            .position(|&byte| byte == 0)
            .map_or(app_field.len(), |offset| pos + offset);
        if end == pos {
            break;
        }
        out.push((pos, &app_field[pos..end]));
        pos = end + 1;
    }
    out
}

fn field<'a>(tokens: &'a [(usize, &'a [u8])], key: &[u8]) -> Option<(usize, &'a [u8])> {
    tokens
        .chunks_exact(2)
        .find(|pair| pair[0].1 == key)
        .map(|pair| pair[1])
}

fn fix_private_battle_flag(app_field: &mut [u8]) -> bool {
    if nextendo::external_ip().is_none() {
        return false;
    }
    let offset = {
        let fields = tokens(app_field);
        let private_battle = field(&fields, b"Mode").is_some_and(|(_, mode)| mode == b"cPrivate");
        match field(&fields, b"InGame") {
            Some((offset, value)) if private_battle && value == b"0" => offset,
            _ => return false,
        }
    };
    app_field[offset] = b'1';
    true
}

fn arms_session_active(app_field: &[u8]) -> bool {
    let fields = tokens(app_field);
    field(&fields, b"JoinMode").is_some_and(|(_, mode)| matches!(mode, b"1" | b"2" | b"3" | b"4"))
}

pub fn presence_from_guest(bytes: &[u8], previous: Option<&LocalPresence>) -> Option<LocalPresence> {
    if bytes.len() < USER_PRESENCE_SIZE {
        return None;
    }
    let declared = bytes[0x18];
    let mut status = previous.map_or(nextendo::PRESENCE_ONLINE, |presence| presence.status);
    match declared {
        1 => status = nextendo::PRESENCE_PLAYING,
        2 => status = nextendo::PRESENCE_ONLINE,
        _ => {}
    }
    status = status.max(nextendo::PRESENCE_ONLINE);
    let mut app_field = bytes[0x20..0x20 + APP_FIELD_SIZE].to_vec();
    if fix_private_battle_flag(&mut app_field) || arms_session_active(&app_field) {
        status = nextendo::PRESENCE_PLAYING;
    }
    Some(LocalPresence { status, app_field })
}

pub fn declare_session(open: bool) {
    let previous = nextendo::local_presence().unwrap_or_default();
    nextendo::set_local_presence(Some(LocalPresence {
        status: if open {
            nextendo::PRESENCE_PLAYING
        } else {
            nextendo::PRESENCE_ONLINE
        },
        app_field: previous.app_field,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn friend() -> Friend {
        Friend {
            pid: 1_800_000_123,
            name: "Toadette".into(),
            status: nextendo::PRESENCE_PLAYING,
            app_field: b"Mode\0cPublic\0".to_vec(),
            app_id: 0x0100152000022000,
            account_id: 0,
            console: false,
        }
    }

    #[test]
    fn friend_records_use_the_documented_offsets() {
        let record = encode_friend(&friend(), None);
        assert_eq!(&record[0x00..0x08], &1_800_000_123u64.to_le_bytes());
        assert_eq!(&record[0x08..0x10], &UID_HIGH.to_le_bytes());
        assert_eq!(&record[0x10..0x18], &1_800_000_123u64.to_le_bytes());
        assert_eq!(&record[0x18..0x20], b"Toadette");
        assert_eq!(&record[0x40..0x48], &0x0100152000022000u64.to_le_bytes());
        assert_eq!(&record[0x58..0x5C], &2u32.to_le_bytes());
        assert_eq!(record[0x5C], 1);
        assert_eq!(&record[0x60..0x6D], b"Mode\0cPublic\0");
        assert_eq!(record[0x128], 1);
    }

    #[test]
    fn consoles_are_listed_under_their_account_id() {
        let mut console = friend();
        console.console = true;
        console.account_id = 0xABCDEF;
        assert_eq!(advertised_id(&console), 0xABCDEF);
        assert!(matches_id(&console, 1_800_000_123));
        assert!(matches_id(&console, 0xABCDEF));
        let record = encode_friend(&console, Some(1_800_000_123));
        assert_eq!(&record[0x10..0x18], &1_800_000_123u64.to_le_bytes());
    }

    #[test]
    fn long_nicknames_are_cut_on_a_character_boundary() {
        let mut long = friend();
        long.name = "ééééééééééééééééééé".into();
        let record = encode_friend(&long, None);
        let nickname = &record[0x18..0x39];
        let end = nickname.iter().position(|&byte| byte == 0).unwrap();
        assert!(std::str::from_utf8(&nickname[..end]).is_ok());
        assert!(end <= NICKNAME_LIMIT);
    }

    #[test]
    fn profiles_carry_name_and_image() {
        let record = encode_profile(42, "Luigi", 42, 0);
        assert_eq!(&record[0..8], &42u64.to_le_bytes());
        assert_eq!(&record[0x08..0x0D], b"Luigi");
        assert!(record[0x30..].starts_with(b"https://cdn-image-"));
        assert_eq!(record[0xD0], 1);
    }

    #[test]
    fn guest_presence_declarations_set_the_status() {
        let mut bytes = vec![0u8; USER_PRESENCE_SIZE];
        bytes[0x18] = 1;
        bytes[0x20..0x26].copy_from_slice(b"Mode\0x");
        let presence = presence_from_guest(&bytes, None).unwrap();
        assert_eq!(presence.status, nextendo::PRESENCE_PLAYING);
        assert_eq!(&presence.app_field[..6], b"Mode\0x");
        bytes[0x18] = 2;
        let presence = presence_from_guest(&bytes, Some(&presence)).unwrap();
        assert_eq!(presence.status, nextendo::PRESENCE_ONLINE);
        bytes[0x18] = 0;
        let kept = presence_from_guest(
            &bytes,
            Some(&LocalPresence {
                status: nextendo::PRESENCE_PLAYING,
                app_field: Vec::new(),
            }),
        )
        .unwrap();
        assert_eq!(kept.status, nextendo::PRESENCE_PLAYING);
        assert!(presence_from_guest(&bytes[..0x40], None).is_none());
    }

    #[test]
    fn arms_join_mode_counts_as_playing() {
        assert!(arms_session_active(b"SessionId\x00123\x00JoinMode\x002\x00"));
        assert!(!arms_session_active(b"SessionId\x00123\x00JoinMode\x000\x00"));
        assert!(!arms_session_active(b""));
    }
}
