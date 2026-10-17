use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::{DateTime, Utc};
use serde_json::Value;

pub const MAX_DOCUMENT: usize = 1024 * 1024;
const MAX_VALIDITY_SECS: i64 = 24 * 60 * 60;
const CLOCK_SLACK_SECS: i64 = 5 * 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigningKey {
    Catalog,
    NplnPatches,
}

impl SigningKey {
    fn key_id(self) -> &'static str {
        match self {
            SigningKey::Catalog => "catalog-key-1",
            SigningKey::NplnPatches => "npln-patches-key-1",
        }
    }

    fn public_key(self) -> &'static str {
        match self {
            SigningKey::Catalog => "xI+atR6VesrCH89ZDZD4t3WQLcYOFEhk5F0HLMzgWk4=",
            SigningKey::NplnPatches => "LAI+Pi+GZtj3p4DpNM5ORBraw1cbZCVSMFgUmAVWNeA=",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Verified {
    pub document_type: String,
    pub revision: u64,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub payload: Value,
}

impl Verified {
    pub fn is_valid_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at > now
    }
}

pub fn canonical_base64(text: &str) -> Result<Vec<u8>, String> {
    let bytes = STANDARD
        .decode(text)
        .map_err(|_| "noncanonical base64".to_string())?;
    if STANDARD.encode(&bytes) != text {
        return Err("noncanonical base64".into());
    }
    Ok(bytes)
}

pub fn text_field(value: &Value, field: &str, max: usize, nonempty: bool) -> Result<String, String> {
    let text = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing {field}"))?;
    if text.len() > max || (nonempty && text.trim().is_empty()) || text.chars().any(char::is_control) {
        return Err(format!("invalid {field}"));
    }
    Ok(text.to_string())
}

pub fn timestamp(value: &Value, field: &str) -> Result<DateTime<Utc>, String> {
    let raw = text_field(value, field, 40, true)?;
    if !raw.ends_with('Z') {
        return Err(format!("{field} is not a UTC date"));
    }
    DateTime::parse_from_rfc3339(&raw)
        .map(|date| date.with_timezone(&Utc))
        .map_err(|_| format!("{field} is not a valid date"))
}

pub fn hex_bytes(text: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
    if text.is_empty() || text.len() % 2 != 0 || text.len() / 2 > max_bytes {
        return Err("invalid hex".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).map_err(|_| "invalid hex".to_string()))
        .collect()
}

pub fn verify(
    envelope: &[u8],
    key: SigningKey,
    expected_type: &str,
    now: DateTime<Utc>,
    allow_expired: bool,
) -> Result<Verified, String> {
    if envelope.len() > MAX_DOCUMENT {
        return Err("document too large".into());
    }
    let outer: Value = serde_json::from_slice(envelope).map_err(|_| "envelope is not JSON".to_string())?;
    if text_field(&outer, "key_id", 64, true)? != key.key_id() {
        return Err("unknown signing key".into());
    }
    let payload = canonical_base64(&text_field(&outer, "payload", MAX_DOCUMENT, true)?)?;
    let signature = canonical_base64(&text_field(&outer, "signature", 128, true)?)?;
    let public_key = canonical_base64(key.public_key())?;
    if signature.len() != 64 || public_key.len() != 32 {
        return Err("invalid signature size".into());
    }
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, &public_key)
        .verify(&payload, &signature)
        .map_err(|_| "invalid signature".to_string())?;
    let document: Value = serde_json::from_slice(&payload).map_err(|_| "payload is not JSON".to_string())?;
    if text_field(&document, "document_type", 32, true)? != expected_type
        || document.get("schema_version").and_then(Value::as_u64) != Some(1)
    {
        return Err("unsupported document".into());
    }
    let revision = document
        .get("revision")
        .and_then(Value::as_u64)
        .filter(|revision| *revision >= 1)
        .ok_or("invalid revision")?;
    let issued_at = timestamp(&document, "issued_at")?;
    let expires_at = timestamp(&document, "expires_at")?;
    let validity = (expires_at - issued_at).num_seconds();
    if issued_at > now + chrono::Duration::seconds(CLOCK_SLACK_SECS)
        || validity <= 0
        || validity > MAX_VALIDITY_SECS
        || (!allow_expired && expires_at <= now)
    {
        return Err("document outside its validity period".into());
    }
    Ok(Verified {
        document_type: expected_type.to_string(),
        revision,
        issued_at,
        expires_at,
        payload: document,
    })
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub const CLIENT_RELEASES: &str = r#"{"key_id":"catalog-key-1","payload":"eyJyZXZpc2lvbiI6MSwicmVsZWFzZXMiOltdLCJkb2N1bWVudF90eXBlIjoiY2xpZW50X3JlbGVhc2VzIiwic2NoZW1hX3ZlcnNpb24iOjEsImlzc3VlZF9hdCI6IjIwMjYtMTAtMTZUMTg6NTI6NDAuMzI5NjA4WiIsImV4cGlyZXNfYXQiOiIyMDI2LTEwLTE3VDE4OjUyOjQwLjMyOTYwOFoifQ==","signature":"eVk3uf54Gga1wrBMz1itKQDtcThSepYmfz4KzdIFm2gE8fzJyaz7vBUb884lX4Kvs4hdtqMABh0fP9ZwhT34Aw=="}"#;

    pub fn issued() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-16T19:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn a_real_signed_document_verifies() {
        let verified = verify(CLIENT_RELEASES.as_bytes(), SigningKey::Catalog, "client_releases", issued(), false).unwrap();
        assert_eq!(verified.revision, 1);
        assert_eq!(verified.payload["releases"].as_array().map(Vec::len), Some(0));
        assert!(verified.is_valid_at(issued()));
        assert!(!verified.is_valid_at(issued() + chrono::Duration::days(2)));
    }

    #[test]
    fn expiry_and_key_are_enforced() {
        let later = issued() + chrono::Duration::days(2);
        assert!(verify(CLIENT_RELEASES.as_bytes(), SigningKey::Catalog, "client_releases", later, false).is_err());
        assert!(verify(CLIENT_RELEASES.as_bytes(), SigningKey::Catalog, "client_releases", later, true).is_ok());
        assert!(verify(CLIENT_RELEASES.as_bytes(), SigningKey::NplnPatches, "client_releases", issued(), false).is_err());
        assert!(verify(CLIENT_RELEASES.as_bytes(), SigningKey::Catalog, "game_catalog", issued(), false).is_err());
    }

    #[test]
    fn tampering_breaks_the_signature() {
        let tampered = CLIENT_RELEASES.replacen("eyJyZXZpc2lvbiI6MSw", "eyJyZXZpc2lvbiI6Miw", 1);
        assert_ne!(tampered, CLIENT_RELEASES);
        assert_eq!(
            verify(tampered.as_bytes(), SigningKey::Catalog, "client_releases", issued(), false).unwrap_err(),
            "invalid signature"
        );
        let bad_signature = CLIENT_RELEASES.replacen("eVk3uf54", "eVk3uf55", 1);
        assert_eq!(
            verify(bad_signature.as_bytes(), SigningKey::Catalog, "client_releases", issued(), false).unwrap_err(),
            "invalid signature"
        );
        assert!(verify(b"{}", SigningKey::Catalog, "client_releases", issued(), false).is_err());
    }

    #[test]
    fn helpers_reject_sloppy_input() {
        assert!(canonical_base64("QUJD").is_ok());
        assert!(canonical_base64("QUJ").is_err());
        assert_eq!(hex_bytes("F4031F2A", 4).unwrap(), vec![0xF4, 0x03, 0x1F, 0x2A]);
        assert!(hex_bytes("F40", 4).is_err());
        assert!(hex_bytes("F4031F2A", 3).is_err());
        assert!(hex_bytes("zz", 4).is_err());
        let value = serde_json::json!({ "when": "2026-10-17T01:45:14.470026Z", "local": "2026-10-17T01:45:14+02:00" });
        assert!(timestamp(&value, "when").is_ok());
        assert!(timestamp(&value, "local").is_err());
    }
}
