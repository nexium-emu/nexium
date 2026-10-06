use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub const API_BASE: &str = "https://nextendo.network";
pub const DEFAULT_SERVER_IP: Ipv4Addr = Ipv4Addr::new(51, 178, 29, 194);
pub const DEFAULT_NAT_IP: Ipv4Addr = Ipv4Addr::new(164, 132, 111, 120);

pub const PRESENCE_OFFLINE: i32 = 0;
pub const PRESENCE_ONLINE: i32 = 1;
pub const PRESENCE_PLAYING: i32 = 2;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkedAccount {
    pub pid: u64,
    pub username: String,
    pub friend_code: String,
    pub nex_token: String,
}

static ACCOUNT: RwLock<Option<LinkedAccount>> = RwLock::new(None);
static ACCOUNT_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn set_account(account: Option<LinkedAccount>) {
    if let Ok(mut slot) = ACCOUNT.write() {
        if *slot == account {
            return;
        }
        *slot = account;
    }
    ACCOUNT_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub fn account() -> Option<LinkedAccount> {
    ACCOUNT.read().ok().and_then(|slot| slot.clone())
}

pub fn account_generation() -> u64 {
    ACCOUNT_GENERATION.load(Ordering::Acquire)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Redirect {
    pub enabled: bool,
    pub server: Ipv4Addr,
    pub nat: Ipv4Addr,
}

impl Default for Redirect {
    fn default() -> Self {
        Self {
            enabled: false,
            server: DEFAULT_SERVER_IP,
            nat: DEFAULT_NAT_IP,
        }
    }
}

static REDIRECT: RwLock<Redirect> = RwLock::new(Redirect {
    enabled: false,
    server: DEFAULT_SERVER_IP,
    nat: DEFAULT_NAT_IP,
});

pub fn set_redirect(redirect: Redirect) {
    if let Ok(mut slot) = REDIRECT.write() {
        *slot = redirect;
    }
}

pub fn redirect() -> Redirect {
    REDIRECT.read().map(|slot| *slot).unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedirectTarget {
    Server,
    Nat,
}

const MK8D_NEX_HOST: &str = "g2b309e01-lp1.s.n.srv.nintendo.net";
const SERVER_DOMAINS: [&str; 5] = [
    "nintendo.net",
    "nintendo.com",
    "nintendowifi.net",
    "nintendo.co.jp",
    "demonware.net",
];

fn in_domain(host: &str, domain: &str) -> bool {
    host == domain
        || host
            .strip_suffix(domain)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

pub fn classify_host(host: &str) -> Option<RedirectTarget> {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.starts_with("nncs2-") && in_domain(&host, "n.n.srv.nintendo.net") {
        return Some(RedirectTarget::Nat);
    }
    if host == MK8D_NEX_HOST {
        return Some(RedirectTarget::Nat);
    }
    SERVER_DOMAINS
        .iter()
        .any(|domain| in_domain(&host, domain))
        .then_some(RedirectTarget::Server)
}

pub fn redirect_target(host: &str) -> Option<Ipv4Addr> {
    let redirect = redirect();
    if !redirect.enabled {
        return None;
    }
    let ip = match classify_host(host)? {
        RedirectTarget::Server => redirect.server,
        RedirectTarget::Nat => redirect.nat,
    };
    remember_host(ip, host);
    Some(ip)
}

static HOST_FOR_IP: Mutex<Option<HashMap<Ipv4Addr, String>>> = Mutex::new(None);

fn remember_host(ip: Ipv4Addr, host: &str) {
    if let Ok(mut map) = HOST_FOR_IP.lock() {
        map.get_or_insert_with(HashMap::new)
            .insert(ip, host.trim().trim_end_matches('.').to_ascii_lowercase());
    }
}

pub fn host_for_ip(ip: Ipv4Addr) -> Option<String> {
    HOST_FOR_IP
        .lock()
        .ok()
        .and_then(|map| map.as_ref().and_then(|map| map.get(&ip).cloned()))
}

static IP_FOR_PORT: Mutex<Option<HashMap<u16, Ipv4Addr>>> = Mutex::new(None);

pub fn remember_port(port: u16, ip: Ipv4Addr) {
    if port == 0 {
        return;
    }
    if let Ok(mut map) = IP_FOR_PORT.lock() {
        map.get_or_insert_with(HashMap::new).insert(port, ip);
    }
}

pub fn redirect_for_port(port: u16) -> Option<Ipv4Addr> {
    if !redirect().enabled {
        return None;
    }
    IP_FOR_PORT
        .lock()
        .ok()
        .and_then(|map| map.as_ref().and_then(|map| map.get(&port).copied()))
}

pub fn is_redirected_ip(ip: Ipv4Addr) -> bool {
    let redirect = redirect();
    redirect.enabled && (ip == redirect.server || ip == redirect.nat)
}

static EXTERNAL_IP: RwLock<Option<Ipv4Addr>> = RwLock::new(None);

pub fn set_external_ip(ip: Ipv4Addr) {
    if let Ok(mut slot) = EXTERNAL_IP.write() {
        if *slot != Some(ip) {
            log::info!("nextendo: NAT check reports a public address");
        }
        *slot = Some(ip);
    }
}

pub fn external_ip() -> Option<Ipv4Addr> {
    EXTERNAL_IP.read().ok().and_then(|slot| *slot)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Friend {
    pub pid: u64,
    pub name: String,
    pub status: i32,
    pub app_field: Vec<u8>,
    pub app_id: u64,
    pub account_id: u64,
    pub console: bool,
}

static FRIENDS: RwLock<Option<Arc<Vec<Friend>>>> = RwLock::new(None);
static FRIENDS_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn set_friends(friends: Vec<Friend>) {
    if let Ok(mut slot) = FRIENDS.write() {
        if slot.as_deref() == Some(&friends) {
            return;
        }
        *slot = Some(Arc::new(friends));
    }
    FRIENDS_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub fn friends() -> Arc<Vec<Friend>> {
    FRIENDS
        .read()
        .ok()
        .and_then(|slot| slot.clone())
        .unwrap_or_default()
}

pub fn friends_generation() -> u64 {
    FRIENDS_GENERATION.load(Ordering::Acquire)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalPresence {
    pub status: i32,
    pub app_field: Vec<u8>,
}

static LOCAL_PRESENCE: Mutex<Option<LocalPresence>> = Mutex::new(None);
static LOCAL_PRESENCE_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn set_local_presence(presence: Option<LocalPresence>) {
    if let Ok(mut slot) = LOCAL_PRESENCE.lock() {
        if *slot == presence {
            return;
        }
        *slot = presence;
    }
    LOCAL_PRESENCE_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub fn local_presence() -> Option<LocalPresence> {
    LOCAL_PRESENCE.lock().ok().and_then(|slot| slot.clone())
}

pub fn local_presence_generation() -> u64 {
    LOCAL_PRESENCE_GENERATION.load(Ordering::Acquire)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompatibleTitle {
    pub title_id: u64,
    pub name: &'static str,
    pub version: &'static str,
}

const fn title(title_id: u64, name: &'static str, version: &'static str) -> CompatibleTitle {
    CompatibleTitle {
        title_id,
        name,
        version,
    }
}

pub const COMPATIBLE_TITLES: &[CompatibleTitle] = &[
    title(0x0100152000022000, "Mario Kart 8 Deluxe", "4.0.0"),
    title(0x01006A800016E000, "Super Smash Bros. Ultimate", "13.0.5"),
    title(0x0100F8F0000A2000, "Splatoon 2", "5.5.2"),
    title(0x01003BC0000A0000, "Splatoon 2", "5.5.2"),
    title(0x01003C700009C800, "Splatoon 2", "5.5.2"),
    title(0x01006F8002326000, "Animal Crossing: New Horizons", "3.0.3"),
    title(0x0100DCA0064A6000, "Luigi's Mansion 3", "1.4.0"),
    title(0x01009B500007C000, "ARMS", "5.5.1"),
    title(0x0100BDE00862A000, "Mario Tennis Aces", "3.1.1"),
    title(0x0100C9C00E25C000, "Mario Golf: Super Rush", "4.0.0"),
    title(0x0100C2500FC20000, "Splatoon 3", "11.3.0"),
    title(0x01009B90006DC000, "Super Mario Maker 2", "3.0.3"),
    title(0x010015100B514000, "Super Mario Bros. Wonder", "1.2.1"),
    title(0x0100277011F1A000, "Super Mario Bros. 35", "1.0.2"),
    title(0x0100770008DD8000, "Monster Hunter Generations Ultimate", "1.4.0"),
    title(0x010047700D540000, "Clubhouse Games: 51 Worldwide Classics", "2.0.1"),
    title(0x0100C6F01C4F8000, "Metal Gear Solid: Peace Walker", "1.3.0"),
    title(0x01006FD0080B2000, "Overcooked! 2", "1.0.19"),
    title(0x01006FE013472000, "Mario Party Superstars", "1.1.1"),
    title(0x0100000000010000, "Super Mario Odyssey", "1.4.1"),
    title(0x01008F6008C5E000, "Pokémon Violet", "4.0.0"),
    title(0x0100A3D008C5C000, "Pokémon Scarlet", "4.0.0"),
    title(0x0100F43008C44000, "Pokémon Legends: Z-A", "2.0.2"),
    title(0x0100C9A00ECE6000, "Nintendo 64 – Nintendo Switch Online", "4.2.0"),
    title(0x010019401051C000, "Mario Strikers: Battle League", "1.3.2"),
    title(0x0100F9F00C696000, "Crash Team Racing Nitro-Fueled", "1.0.15"),
    title(0x01001B300B9BE000, "Diablo III: Eternal Collection", "2.7.7.92380"),
    title(0x0100A7C01B792000, "Minecraft Dungeons II", "1.1.1.0"),
    title(0x0100B3F000BE2000, "Pokkén Tournament DX", "1.3.3"),
    title(0x0100DE600BEEE000, "Saints Row: The Third", "1.6.1"),
];

pub fn compatible_title(title_id: u64) -> Option<&'static CompatibleTitle> {
    COMPATIBLE_TITLES
        .iter()
        .find(|entry| entry.title_id == title_id)
}

pub fn version_matches(title_id: u64, installed: &str) -> bool {
    compatible_title(title_id).map_or(true, |entry| {
        let installed = installed.trim().trim_start_matches(['v', 'V']);
        installed.is_empty() || installed == entry.version
    })
}

pub fn format_title_id(title_id: u64) -> String {
    format!("{title_id:016X}")
}

pub fn parse_title_id(text: &str) -> Option<u64> {
    let text = text.trim();
    let text = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    if text.is_empty() || text.len() > 16 {
        return None;
    }
    u64::from_str_radix(text, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nintendo_hosts_go_to_the_game_server() {
        for host in [
            "nintendo.net",
            "api.lp1.npln.srv.nintendo.net",
            "e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com",
            "conntest.nintendowifi.net",
            "stun.us.demonware.net",
            "WWW.NINTENDO.CO.JP.",
        ] {
            assert_eq!(classify_host(host), Some(RedirectTarget::Server), "{host}");
        }
    }

    #[test]
    fn nat_check_uses_two_distinct_responders() {
        assert_eq!(
            classify_host("nncs1-lp1.n.n.srv.nintendo.net"),
            Some(RedirectTarget::Server)
        );
        assert_eq!(
            classify_host("nncs2-lp1.n.n.srv.nintendo.net"),
            Some(RedirectTarget::Nat)
        );
        assert_eq!(classify_host(MK8D_NEX_HOST), Some(RedirectTarget::Nat));
    }

    #[test]
    fn lookalike_domains_are_left_alone() {
        for host in [
            "evilnintendo.net",
            "nintendo.net.example.com",
            "example.com",
            "",
            "localhost",
        ] {
            assert_eq!(classify_host(host), None, "{host}");
        }
    }

    #[test]
    fn version_gate_accepts_unknown_titles_and_exact_versions() {
        assert!(version_matches(0x0100152000022000, "4.0.0"));
        assert!(version_matches(0x0100152000022000, "v4.0.0"));
        assert!(version_matches(0x0100152000022000, ""));
        assert!(!version_matches(0x0100152000022000, "3.0.3"));
        assert!(version_matches(0x0123456789ABCDEF, "1.0.0"));
    }

    #[test]
    fn title_ids_round_trip() {
        assert_eq!(format_title_id(0x0100152000022000), "0100152000022000");
        assert_eq!(parse_title_id("0100152000022000"), Some(0x0100152000022000));
        assert_eq!(parse_title_id("0x01006a800016e000"), Some(0x01006A800016E000));
        assert_eq!(parse_title_id("zz"), None);
        assert_eq!(parse_title_id(""), None);
    }
}
