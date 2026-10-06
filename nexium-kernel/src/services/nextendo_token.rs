use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs1v15::SigningKey;
use rsa::pkcs8::{DecodePrivateKey, EncodePrivateKey, LineEnding};
use rsa::signature::{SignatureEncoding, Signer};
use rsa::RsaPrivateKey;
use sha2::Sha256;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ISSUER: &str = "https://e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com";
const JKU: &str = "https://e0d67c509fb203858ebcb2fe3f88c2aa.baas.nintendo.com/1.0.0/certificates";
const AUDIENCE: &str = "ed9e2f05d286f7b8";
const KEY_ID: &str = "nextendo-baas-key-1";
const SERIAL: &str = "XAW10000000000";
const POKEMON_SCARLET: u64 = 0x0100A3D008C5C000;
const LIFETIME_SECS: u64 = 3 * 60 * 60;
const REUSE_FOR: Duration = Duration::from_secs(2 * 60 * 60);
const KEY_BITS: usize = 2048;

struct Cached {
    token: String,
    generation: u64,
    title_id: u64,
    version: String,
    minted: Instant,
}

static CACHE: Mutex<Option<Cached>> = Mutex::new(None);
static KEY: OnceLock<Option<SigningKey<Sha256>>> = OnceLock::new();

pub fn id_token(title_id: u64, version: &str) -> Option<String> {
    let account = nexium_common::nextendo::account()?;
    if account.nex_token.is_empty() {
        return None;
    }
    let generation = nexium_common::nextendo::account_generation();
    let mut cache = CACHE.lock().ok()?;
    if let Some(cached) = cache.as_ref() {
        if cached.generation == generation
            && cached.title_id == title_id
            && cached.version == version
            && cached.minted.elapsed() < REUSE_FOR
        {
            return Some(cached.token.clone());
        }
    }
    let token = build(&account.nex_token, title_id, version)?;
    log::info!("nextendo: issued a BAAS id_token for {:016X}", title_id);
    *cache = Some(Cached {
        token: token.clone(),
        generation,
        title_id,
        version: version.to_string(),
        minted: Instant::now(),
    });
    Some(token)
}

pub fn prepare() {
    let _ = signing_key();
}

fn build(nex_token: &str, title_id: u64, version: &str) -> Option<String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    sign_token(signing_key()?, nex_token, title_id, version, now)
}

fn sign_token(
    key: &SigningKey<Sha256>,
    nex_token: &str,
    title_id: u64,
    version: &str,
    now: u64,
) -> Option<String> {
    let device = random_hex(16);
    let header = format!(
        r#"{{"alg":"RS256","kid":"{KEY_ID}","typ":"id_token","jku":"{JKU}"}}"#
    );
    let mut payload = format!(
        r#"{{"sub":"{}","aud":"{AUDIENCE}","iss":"{ISSUER}","typ":"id_token","iat":{now},"exp":{},"jku":"{JKU}","jti":"{}","di":"{device}","sn":"{SERIAL}","bs:did":"{}","nintendo":{{"dt":"NX Prod 1","pc":"HAC","di":"{device}","sn":"{SERIAL}","ist":false}},"nnex":"{}""#,
        random_hex(16),
        now + LIFETIME_SECS,
        random_uuid(),
        random_hex(16),
        json_escape(nex_token),
    );
    if !version.is_empty() {
        payload.push_str(&format!(r#","tv":"{}""#, json_escape(version)));
    }
    if title_id == POKEMON_SCARLET {
        payload.push_str(&format!(r#","app_id":"{title_id:016X}""#));
    }
    payload.push_str(r#","hm":true}"#);
    let signing_input = format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(header),
        URL_SAFE_NO_PAD.encode(payload)
    );
    let signature = key.try_sign(signing_input.as_bytes()).ok()?;
    Some(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    ))
}

fn signing_key() -> Option<&'static SigningKey<Sha256>> {
    KEY.get_or_init(|| load_or_create_key().map(SigningKey::<Sha256>::new))
        .as_ref()
}

fn load_or_create_key() -> Option<RsaPrivateKey> {
    if let Ok(pem) = std::env::var("NEXTENDO_BAAS_SIGNING_KEY") {
        match parse_pem(&pem) {
            Some(key) => {
                log::info!("nextendo: using the supplied BAAS signing key");
                return Some(key);
            }
            None => log::warn!("nextendo: NEXTENDO_BAAS_SIGNING_KEY is not a valid RSA key"),
        }
    }
    if let Some(key) = supplied_key_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|pem| parse_pem(&pem))
    {
        log::info!("nextendo: using nextendo_baas.pem");
        return Some(key);
    }
    let persisted = persisted_key_path();
    if let Some(key) = std::fs::read_to_string(&persisted)
        .ok()
        .and_then(|pem| parse_pem(&pem))
    {
        return Some(key);
    }
    let mut rng = rsa::rand_core::OsRng;
    let key = match RsaPrivateKey::new(&mut rng, KEY_BITS) {
        Ok(key) => key,
        Err(error) => {
            log::error!("nextendo: could not generate a BAAS signing key: {error}");
            return None;
        }
    };
    match key.to_pkcs8_pem(LineEnding::LF) {
        Ok(pem) => {
            if let Some(parent) = persisted.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(error) = std::fs::write(&persisted, pem.as_bytes()) {
                log::warn!("nextendo: could not store the BAAS signing key: {error}");
            }
        }
        Err(error) => log::warn!("nextendo: could not encode the BAAS signing key: {error}"),
    }
    Some(key)
}

fn parse_pem(text: &str) -> Option<RsaPrivateKey> {
    let text = text.trim();
    if !text.contains("BEGIN") {
        return None;
    }
    RsaPrivateKey::from_pkcs8_pem(text)
        .ok()
        .or_else(|| RsaPrivateKey::from_pkcs1_pem(text).ok())
}

fn supplied_key_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("nextendo_baas.pem"))
}

fn persisted_key_path() -> PathBuf {
    nexium_common::paths::root()
        .join("nextendo")
        .join("baas_signing_key.pem")
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    if getrandom::fill(&mut bytes).is_err() {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (seed >> ((index % 16) * 8)) as u8 ^ (index as u8).wrapping_mul(0x9D);
        }
    }
    bytes
}

fn random_hex(len: usize) -> String {
    random_bytes::<32>()
        .iter()
        .take(len)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn random_uuid() -> String {
    let mut bytes = random_bytes::<16>();
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_handles_quotes_and_controls() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(json_escape("x\u{1}y"), "x\\u0001y");
        assert_eq!(json_escape("nx2.abc_DEF-123"), "nx2.abc_DEF-123");
    }

    #[test]
    fn uuids_are_version_four() {
        let uuid = random_uuid();
        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[14..15], "4");
        assert!(matches!(&uuid[19..20], "8" | "9" | "a" | "b"));
    }

    #[test]
    fn hex_has_the_requested_length() {
        assert_eq!(random_hex(16).len(), 32);
        assert!(random_hex(16).chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn signed_tokens_verify_and_carry_the_claims() {
        use rsa::pkcs1v15::{Signature, VerifyingKey};
        use rsa::signature::Verifier;
        let private = RsaPrivateKey::new(&mut rsa::rand_core::OsRng, 1024).unwrap();
        let key = SigningKey::<Sha256>::new(private.clone());
        let token = sign_token(&key, "nx2.test-token", POKEMON_SCARLET, "4.0.0", 1_700_000_000)
            .expect("token");
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
        let header = String::from_utf8(URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
        assert!(header.contains(r#""alg":"RS256""#));
        let payload = String::from_utf8(URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
        assert!(payload.contains(r#""nnex":"nx2.test-token""#));
        assert!(payload.contains(r#""tv":"4.0.0""#));
        assert!(payload.contains(r#""app_id":"0100A3D008C5C000""#));
        assert!(payload.contains(r#""exp":1700010800"#));
        assert!(payload.ends_with(r#""hm":true}"#));
        let verifying = VerifyingKey::<Sha256>::new(private.to_public_key());
        let signature =
            Signature::try_from(URL_SAFE_NO_PAD.decode(parts[2]).unwrap().as_slice()).unwrap();
        let signed = format!("{}.{}", parts[0], parts[1]);
        assert!(verifying.verify(signed.as_bytes(), &signature).is_ok());
    }

    #[test]
    fn pem_keys_round_trip() {
        let private = RsaPrivateKey::new(&mut rsa::rand_core::OsRng, 1024).unwrap();
        let pem = private.to_pkcs8_pem(LineEnding::LF).unwrap();
        assert_eq!(parse_pem(&pem), Some(private));
        assert_eq!(parse_pem("not a key"), None);
    }
}
