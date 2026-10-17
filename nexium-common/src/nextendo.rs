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

pub const MARIO_KART_8_DELUXE: u64 = 0x0100152000022000;
pub const SPLATOON_2_EU: u64 = 0x0100F8F0000A2000;
pub const SPLATOON_2_US: u64 = 0x01003BC0000A0000;
pub const SPLATOON_2_JP: u64 = 0x01003C700009C800;
pub const SPLATOON_3: u64 = 0x0100C2500FC20000;
pub const POKEMON_VIOLET: u64 = 0x01008F6008C5E000;
pub const POKEMON_SCARLET: u64 = 0x0100A3D008C5C000;
pub const POKEMON_LEGENDS_ZA: u64 = 0x0100F43008C44000;
pub const POKEMON_SWORD: u64 = 0x0100ABF008968000;
pub const POKEMON_SHIELD: u64 = 0x01008DB008C2C000;
pub const NINTENDO_SWITCH_SPORTS: u64 = 0x0100D2F00D5C0000;
pub const MINECRAFT_DUNGEONS_2: u64 = 0x0100A7C01B792000;
pub const PVZ_BATTLE_FOR_NEIGHBORVILLE: u64 = 0x0100C56010FD8000;
pub const DRAGON_BALL_FIGHTERZ: u64 = 0x0100A250097F0000;
pub const FRIDAY_THE_13TH: u64 = 0x010092A00C4B6000;
pub const PEACE_WALKER_US: u64 = 0x010099801C4FA000;
pub const CRASH_TEAM_RACING: u64 = 0x0100F9F00C696000;

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

static RUNNING_TITLE: AtomicU64 = AtomicU64::new(0);

pub fn set_running_title(title_id: Option<u64>) {
    RUNNING_TITLE.store(title_id.unwrap_or(0), Ordering::Release);
}

pub fn running_title() -> Option<u64> {
    match RUNNING_TITLE.load(Ordering::Acquire) {
        0 => None,
        title_id => Some(title_id),
    }
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

const NAT_HOSTS: [&str; 2] = [
    "g2b309e01-lp1.s.n.srv.nintendo.net",
    "g2df33d01-lp1.p.srv.nintendo.net",
];
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

fn title_host(host: &str, title_id: u64) -> bool {
    match title_id {
        PVZ_BATTLE_FOR_NEIGHBORVILLE => {
            host == "spring18.gosredirector.ea.com"
                || (host.starts_with("pvzgw3prodapp-") && in_domain(host, "ea.com"))
        }
        DRAGON_BALL_FIGHTERZ => host == "dbf.channel.or.jp",
        FRIDAY_THE_13TH => host == "nswitch.f13.illfonic.com",
        MINECRAFT_DUNGEONS_2 => {
            matches!(
                host,
                "83156.playfabapi.com" | "vex.minecraftservices.com" | "s2s-vex.minecraftservices.com"
            ) || in_domain(host, "xboxlive.com")
                || host == "login.live.com"
        }
        _ => false,
    }
}

pub fn classify_host(host: &str, title_id: Option<u64>) -> Option<RedirectTarget> {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.starts_with("nncs2-") && in_domain(&host, "n.n.srv.nintendo.net") {
        return Some(RedirectTarget::Nat);
    }
    if NAT_HOSTS.contains(&host.as_str()) {
        return Some(RedirectTarget::Nat);
    }
    if SERVER_DOMAINS.iter().any(|domain| in_domain(&host, domain)) {
        return Some(RedirectTarget::Server);
    }
    title_id
        .is_some_and(|title_id| title_host(&host, title_id))
        .then_some(RedirectTarget::Server)
}

pub fn redirect_target(host: &str) -> Option<Ipv4Addr> {
    let redirect = redirect();
    if !redirect.enabled {
        return None;
    }
    let ip = match classify_host(host, running_title())? {
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
    title(MARIO_KART_8_DELUXE, "Mario Kart 8 Deluxe", "4.0.0"),
    title(0x01006A800016E000, "Super Smash Bros. Ultimate", "13.0.5"),
    title(SPLATOON_2_EU, "Splatoon 2", "5.5.2"),
    title(SPLATOON_2_US, "Splatoon 2", "5.5.2"),
    title(SPLATOON_2_JP, "Splatoon 2", "5.5.2"),
    title(0x01006F8002326000, "Animal Crossing: New Horizons", "3.0.3"),
    title(0x0100DCA0064A6000, "Luigi's Mansion 3", "1.4.0"),
    title(0x01009B500007C000, "ARMS", "5.5.1"),
    title(0x0100BDE00862A000, "Mario Tennis Aces", "3.1.1"),
    title(0x0100C9C00E25C000, "Mario Golf: Super Rush", "4.0.0"),
    title(SPLATOON_3, "Splatoon 3", "11.3.0"),
    title(0x01009B90006DC000, "Super Mario Maker 2", "3.0.3"),
    title(0x010015100B514000, "Super Mario Bros. Wonder", "1.2.1"),
    title(0x0100277011F1A000, "Super Mario Bros. 35", "1.0.2"),
    title(0x0100770008DD8000, "Monster Hunter Generations Ultimate", "1.4.0"),
    title(0x010047700D540000, "Clubhouse Games: 51 Worldwide Classics", "2.0.1"),
    title(0x0100C6F01C4F8000, "Metal Gear Solid: Peace Walker", "1.3.0"),
    title(PEACE_WALKER_US, "Metal Gear Solid: Peace Walker", "1.3.0"),
    title(0x01006FD0080B2000, "Overcooked! 2", "1.0.19"),
    title(0x01006FE013472000, "Mario Party Superstars", "1.1.1"),
    title(0x0100965017338000, "Super Mario Party Jamboree", "2.3.0"),
    title(0x0100000000010000, "Super Mario Odyssey", "1.4.1"),
    title(0x010028600EBDA000, "Super Mario 3D World + Bowser's Fury", "1.2.2"),
    title(POKEMON_VIOLET, "Pokémon Violet", "4.0.0"),
    title(POKEMON_SCARLET, "Pokémon Scarlet", "4.0.0"),
    title(POKEMON_LEGENDS_ZA, "Pokémon Legends: Z-A", "2.0.2"),
    title(POKEMON_SWORD, "Pokémon Sword", "1.3.2"),
    title(POKEMON_SHIELD, "Pokémon Shield", "1.3.2"),
    title(0x0100C9A00ECE6000, "Nintendo 64 – Nintendo Switch Online", "4.2.0"),
    title(0x010019401051C000, "Mario Strikers: Battle League", "1.3.2"),
    title(CRASH_TEAM_RACING, "Crash Team Racing Nitro-Fueled", "1.0.15"),
    title(0x01001B300B9BE000, "Diablo III: Eternal Collection", "2.7.7.92380"),
    title(MINECRAFT_DUNGEONS_2, "Minecraft Dungeons II", "1.1.1.0"),
    title(0x01006BD001E06000, "Minecraft: Nintendo Switch Edition", "1.0.17"),
    title(0x0100B3F000BE2000, "Pokkén Tournament DX", "1.3.3"),
    title(0x0100DE600BEEE000, "Saints Row: The Third", "1.6.1"),
    title(NINTENDO_SWITCH_SPORTS, "Nintendo Switch Sports", "1.5.0"),
    title(DRAGON_BALL_FIGHTERZ, "Dragon Ball FighterZ", "01.50"),
    title(0x0100B04011742000, "Monster Hunter Rise", "16.0.2"),
    title(0x01000BF0152FA000, "Zombie Army 4: Dead War", "1.1.1"),
    title(FRIDAY_THE_13TH, "Friday the 13th: The Game", "1.08.00"),
    title(0x0100AD9012510000, "PAC-MAN 99", "1.1.0"),
];

pub fn compatible_title(title_id: u64) -> Option<&'static CompatibleTitle> {
    COMPATIBLE_TITLES
        .iter()
        .find(|entry| entry.title_id == title_id)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TitleSupport {
    pub title_id: u64,
    pub name: String,
    pub version: String,
    pub accepted: Vec<String>,
    pub any_version: bool,
    pub enabled: bool,
    pub from_catalog: bool,
}

impl TitleSupport {
    fn builtin(entry: &CompatibleTitle) -> Self {
        Self {
            title_id: entry.title_id,
            name: entry.name.to_string(),
            version: entry.version.to_string(),
            accepted: vec![entry.version.to_string()],
            any_version: false,
            enabled: true,
            from_catalog: false,
        }
    }

    pub fn accepts(&self, installed: &str) -> bool {
        if !self.enabled {
            return false;
        }
        if self.any_version {
            return true;
        }
        let installed = installed.trim().trim_start_matches(['v', 'V']);
        installed.is_empty() || self.accepted.iter().any(|version| version == installed)
    }
}

static CATALOG_TITLES: RwLock<Option<Arc<Vec<TitleSupport>>>> = RwLock::new(None);
static CATALOG_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn set_catalog_titles(titles: Vec<TitleSupport>) {
    if let Ok(mut slot) = CATALOG_TITLES.write() {
        if slot.as_deref() == Some(&titles) {
            return;
        }
        *slot = Some(Arc::new(titles));
    }
    CATALOG_GENERATION.fetch_add(1, Ordering::AcqRel);
}

pub fn catalog_titles() -> Arc<Vec<TitleSupport>> {
    CATALOG_TITLES
        .read()
        .ok()
        .and_then(|slot| slot.clone())
        .unwrap_or_default()
}

pub fn catalog_generation() -> u64 {
    CATALOG_GENERATION.load(Ordering::Acquire)
}

pub fn supported_title(title_id: u64) -> Option<TitleSupport> {
    if let Some(entry) = compatible_title(title_id) {
        return Some(TitleSupport::builtin(entry));
    }
    catalog_titles()
        .iter()
        .find(|entry| entry.title_id == title_id)
        .cloned()
}

pub fn supported_titles() -> Vec<TitleSupport> {
    let mut titles: Vec<TitleSupport> = COMPATIBLE_TITLES.iter().map(TitleSupport::builtin).collect();
    for entry in catalog_titles().iter() {
        if entry.enabled && compatible_title(entry.title_id).is_none() {
            titles.push(entry.clone());
        }
    }
    titles
}

pub fn title_name(title_id: u64) -> Option<String> {
    supported_title(title_id).map(|entry| entry.name)
}

pub fn version_matches(title_id: u64, installed: &str) -> bool {
    supported_title(title_id).map_or(true, |entry| entry.accepts(installed))
}

pub const BCAT_TITLES: &[u64] = &[
    SPLATOON_2_EU,
    SPLATOON_2_US,
    SPLATOON_2_JP,
    SPLATOON_3,
    POKEMON_VIOLET,
    POKEMON_SCARLET,
    POKEMON_LEGENDS_ZA,
    POKEMON_SWORD,
    POKEMON_SHIELD,
    NINTENDO_SWITCH_SPORTS,
];

pub fn bcat_package_title(title_id: u64) -> Option<u64> {
    match title_id {
        SPLATOON_2_EU | SPLATOON_2_US | SPLATOON_2_JP => Some(SPLATOON_2_EU),
        other if BCAT_TITLES.contains(&other) => Some(other),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchGuard {
    pub offset: u32,
    pub original: Vec<u8>,
    pub patched: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnlinePatch {
    pub id: String,
    pub version: u64,
    pub enabled: bool,
    pub ips: Vec<u8>,
    pub guards: Vec<PatchGuard>,
}

pub type LaunchHook = dyn Fn(u64, &str) -> Option<OnlinePatch> + Send + Sync;

static LAUNCH_HOOK: RwLock<Option<Arc<LaunchHook>>> = RwLock::new(None);

pub fn set_launch_hook(hook: Arc<LaunchHook>) {
    if let Ok(mut slot) = LAUNCH_HOOK.write() {
        *slot = Some(hook);
    }
}

pub fn prepare_launch(title_id: u64, main_build_id: &str) -> Option<OnlinePatch> {
    let hook = LAUNCH_HOOK.read().ok().and_then(|slot| slot.clone())?;
    hook(title_id, main_build_id)
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
            assert_eq!(classify_host(host, None), Some(RedirectTarget::Server), "{host}");
        }
    }

    #[test]
    fn nat_check_uses_two_distinct_responders() {
        assert_eq!(
            classify_host("nncs1-lp1.n.n.srv.nintendo.net", None),
            Some(RedirectTarget::Server)
        );
        assert_eq!(
            classify_host("nncs2-lp1.n.n.srv.nintendo.net", None),
            Some(RedirectTarget::Nat)
        );
        assert_eq!(classify_host(NAT_HOSTS[0], None), Some(RedirectTarget::Nat));
        assert_eq!(
            classify_host("g2df33d01-lp1.p.srv.nintendo.net", Some(SPLATOON_2_EU)),
            Some(RedirectTarget::Nat)
        );
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
            assert_eq!(classify_host(host, None), None, "{host}");
            assert_eq!(classify_host(host, Some(MARIO_KART_8_DELUXE)), None, "{host}");
        }
    }

    #[test]
    fn publisher_backends_are_only_redirected_for_their_title() {
        let cases = [
            ("spring18.gosredirector.ea.com", PVZ_BATTLE_FOR_NEIGHBORVILLE),
            ("pvzgw3prodapp-01.ea.com", PVZ_BATTLE_FOR_NEIGHBORVILLE),
            ("dbf.channel.or.jp", DRAGON_BALL_FIGHTERZ),
            ("nswitch.f13.illfonic.com", FRIDAY_THE_13TH),
            ("83156.playfabapi.com", MINECRAFT_DUNGEONS_2),
            ("vex.minecraftservices.com", MINECRAFT_DUNGEONS_2),
            ("s2s-vex.minecraftservices.com", MINECRAFT_DUNGEONS_2),
            ("xboxlive.com", MINECRAFT_DUNGEONS_2),
            ("xsts.auth.xboxlive.com", MINECRAFT_DUNGEONS_2),
            ("login.live.com", MINECRAFT_DUNGEONS_2),
        ];
        for (host, title_id) in cases {
            assert_eq!(classify_host(host, Some(title_id)), Some(RedirectTarget::Server), "{host}");
            assert_eq!(classify_host(host, Some(MARIO_KART_8_DELUXE)), None, "{host}");
            assert_eq!(classify_host(host, None), None, "{host}");
        }
        assert_eq!(classify_host("www.ea.com", Some(PVZ_BATTLE_FOR_NEIGHBORVILLE)), None);
        assert_eq!(classify_host("pvzgw3prodapp-01.ea.com.evil.net", Some(PVZ_BATTLE_FOR_NEIGHBORVILLE)), None);
        assert_eq!(classify_host("notxboxlive.com", Some(MINECRAFT_DUNGEONS_2)), None);
    }

    #[test]
    fn version_gate_accepts_unknown_titles_and_exact_versions() {
        assert!(version_matches(MARIO_KART_8_DELUXE, "4.0.0"));
        assert!(version_matches(MARIO_KART_8_DELUXE, "v4.0.0"));
        assert!(version_matches(MARIO_KART_8_DELUXE, ""));
        assert!(!version_matches(MARIO_KART_8_DELUXE, "3.0.3"));
        assert!(version_matches(0x0123456789ABCDEF, "1.0.0"));
    }

    #[test]
    fn catalog_titles_extend_the_built_in_table() {
        let tetris = 0x010040600C5CE000;
        let listing = TitleSupport {
            title_id: tetris,
            name: "Tetris 99".into(),
            version: "2.6.0".into(),
            accepted: vec!["2.6.0".into()],
            any_version: false,
            enabled: true,
            from_catalog: true,
        };
        let disabled = TitleSupport {
            title_id: 0x0100000000ABC000,
            name: "Withdrawn".into(),
            enabled: false,
            from_catalog: true,
            ..TitleSupport::default()
        };
        let override_attempt = TitleSupport {
            title_id: MARIO_KART_8_DELUXE,
            name: "Mario Kart 8 Deluxe".into(),
            version: "3.0.3".into(),
            accepted: vec!["3.0.3".into()],
            any_version: false,
            enabled: true,
            from_catalog: true,
        };
        set_catalog_titles(vec![listing.clone(), disabled, override_attempt]);
        assert_eq!(supported_title(tetris), Some(listing));
        assert!(version_matches(tetris, "2.6.0"));
        assert!(!version_matches(tetris, "2.5.0"));
        assert!(!version_matches(0x0100000000ABC000, "1.0.0"));
        assert!(version_matches(MARIO_KART_8_DELUXE, "4.0.0"));
        assert!(!version_matches(MARIO_KART_8_DELUXE, "3.0.3"));
        assert_eq!(title_name(tetris).as_deref(), Some("Tetris 99"));
        let titles = supported_titles();
        assert_eq!(titles.len(), COMPATIBLE_TITLES.len() + 1);
        assert!(titles.iter().any(|entry| entry.title_id == tetris && entry.from_catalog));
        set_catalog_titles(Vec::new());
        assert_eq!(supported_title(tetris), None);
    }

    #[test]
    fn any_version_listings_pass_every_build() {
        let listing = TitleSupport {
            title_id: 0x0100000000DEF000,
            name: "Open".into(),
            any_version: true,
            enabled: true,
            from_catalog: true,
            ..TitleSupport::default()
        };
        assert!(listing.accepts("9.9.9"));
        assert!(listing.accepts(""));
    }

    #[test]
    fn bcat_packages_share_one_splatoon_2_source() {
        assert_eq!(bcat_package_title(SPLATOON_2_US), Some(SPLATOON_2_EU));
        assert_eq!(bcat_package_title(SPLATOON_2_JP), Some(SPLATOON_2_EU));
        assert_eq!(bcat_package_title(SPLATOON_3), Some(SPLATOON_3));
        assert_eq!(bcat_package_title(NINTENDO_SWITCH_SPORTS), Some(NINTENDO_SWITCH_SPORTS));
        assert_eq!(bcat_package_title(MARIO_KART_8_DELUXE), None);
    }

    #[test]
    fn launch_hook_is_optional() {
        assert_eq!(prepare_launch(MARIO_KART_8_DELUXE, "ABCD"), None);
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
