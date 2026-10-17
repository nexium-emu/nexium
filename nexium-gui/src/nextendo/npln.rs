use super::service_catalog::{fetch, BASE};
use super::signed::{self, canonical_base64, hex_bytes, text_field, SigningKey, Verified};
use chrono::Utc;
use nexium_common::nextendo::{self, OnlinePatch, PatchGuard};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const INDEX_TYPE: &str = "npln_title_index";
const MANIFEST_TYPE: &str = "npln_patch_catalog";
const REFRESH_EVERY: Duration = Duration::from_secs(15 * 60);
const MAX_IPS: usize = 65_536;
const MAX_IMAGE: u64 = 512 * 1024 * 1024;
const NSO_HEADER: u32 = 0x100;

pub const NPLN_TITLES: &[u64] = &[
    nextendo::SPLATOON_3,
    0x010015100B514000,
    0x0100C9A00ECE6000,
    0x0100C6F01C4F8000,
    nextendo::PEACE_WALKER_US,
    nextendo::CRASH_TEAM_RACING,
    0x01006FD0080B2000,
    nextendo::POKEMON_SCARLET,
    nextendo::POKEMON_VIOLET,
    nextendo::POKEMON_LEGENDS_ZA,
    0x0100453019AA8000,
    0x0100965017338000,
    0x0100B04011742000,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TitleIndex {
    pub revision: u64,
    pub etag: String,
    pub titles: Vec<(u64, bool)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestPatch {
    pub title_id: u64,
    pub build_id: String,
    pub module: String,
    pub patch: OnlinePatch,
    pub is_final: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub revision: u64,
    pub etag: String,
    pub patches: Vec<ManifestPatch>,
}

struct StoredPatch {
    patch: OnlinePatch,
    revision: u64,
    etag: String,
    is_final: bool,
    checked: Option<Instant>,
}

#[derive(Default)]
struct State {
    loaded: bool,
    index: HashMap<u64, bool>,
    index_revision: u64,
    index_etag: String,
    index_checked: Option<Instant>,
    patches: HashMap<(u64, String), StoredPatch>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::default()))
}

fn cache_dir() -> PathBuf {
    nexium_common::paths::root().join("nextendo").join("npln-patches")
}

fn signed_tag(text: &str) -> Result<String, String> {
    if text.len() != 66 || !text.starts_with('"') || !text.ends_with('"') {
        return Err("invalid signed tag".into());
    }
    hex_bytes(&text[1..65], 32)?;
    Ok(text.to_string())
}

fn title_from_hex(text: &str) -> Result<u64, String> {
    if text.len() != 16 {
        return Err("invalid title ID".into());
    }
    let bytes = hex_bytes(text, 8)?;
    let title_id = bytes.iter().fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte));
    if title_id == 0 {
        return Err("invalid title ID".into());
    }
    Ok(title_id)
}

pub fn parse_title_index(document: &Verified) -> Result<TitleIndex, String> {
    let etag = signed_tag(&text_field(&document.payload, "index_etag", 66, true)?)?;
    let entries = document
        .payload
        .get("titles")
        .and_then(Value::as_array)
        .filter(|array| array.len() <= 10_000)
        .ok_or("invalid title index entries")?;
    let mut titles = Vec::with_capacity(entries.len());
    let mut seen = HashSet::new();
    for entry in entries {
        let title_id = title_from_hex(&text_field(entry, "title_id", 16, true)?)?;
        let enabled = entry.get("enabled").and_then(Value::as_bool).ok_or("invalid title index target")?;
        if !seen.insert(title_id) {
            return Err("duplicate title index target".into());
        }
        titles.push((title_id, enabled));
    }
    Ok(TitleIndex {
        revision: document.revision,
        etag,
        titles,
    })
}

pub fn validate_records(ips: &[u8], guards: &[PatchGuard]) -> Result<(), String> {
    if ips.len() < 15 || ips.len() > MAX_IPS || &ips[..5] != b"IPS32" || &ips[ips.len() - 4..] != b"EEOF" {
        return Err("invalid IPS32 patch".into());
    }
    let end = ips.len() - 4;
    let mut position = 5;
    let mut previous_end = 0u64;
    for guard in guards {
        if position + 6 > end {
            return Err("truncated IPS32 patch".into());
        }
        let offset = u32::from_be_bytes([ips[position], ips[position + 1], ips[position + 2], ips[position + 3]]);
        let length = usize::from(u16::from_be_bytes([ips[position + 4], ips[position + 5]]));
        position += 6;
        let valid = length != 0
            && offset >= NSO_HEADER
            && u64::from(offset) >= previous_end
            && u64::from(offset) + length as u64 <= MAX_IMAGE
            && position + length <= end
            && offset == guard.offset
            && length == guard.original.len()
            && length == guard.patched.len()
            && ips[position..position + length] == guard.patched[..];
        if !valid {
            return Err("IPS32 write does not match its guard".into());
        }
        position += length;
        previous_end = u64::from(offset) + length as u64;
    }
    if position != end || guards.is_empty() {
        return Err("unprotected IPS32 write".into());
    }
    Ok(())
}

pub fn parse_manifest(document: &Verified) -> Result<Manifest, String> {
    let payload = &document.payload;
    let etag = match payload.get("patch_etag") {
        Some(_) => signed_tag(&text_field(payload, "patch_etag", 66, true)?)?,
        None => String::new(),
    };
    let entries = payload
        .get("patches")
        .and_then(Value::as_array)
        .filter(|array| array.len() <= 256)
        .ok_or("invalid patch catalog entries")?;
    let mut seen = HashSet::new();
    let mut patches = Vec::with_capacity(entries.len());
    for entry in entries {
        let title_id = title_from_hex(&text_field(entry, "title_id", 16, true)?)?;
        let build_id = text_field(entry, "build_id", 64, true)?;
        if build_id.len() != 64 {
            return Err("invalid patch target".into());
        }
        hex_bytes(&build_id, 32)?;
        if entry.get("offset_basis").and_then(Value::as_str) != Some("nso_header_included")
            || entry.get("required_capabilities") != Some(&serde_json::json!(["npln.ips32.v1"]))
        {
            return Err("unsupported patch capability".into());
        }
        let module = text_field(entry, "module", 16, true)?;
        let id = text_field(entry, "id", 80, true)?;
        let version = entry.get("patch_version").and_then(Value::as_u64).ok_or("invalid patch version")?;
        let enabled = entry.get("enabled").and_then(Value::as_bool).ok_or("invalid patch flag")?;
        let ips = canonical_base64(&text_field(entry, "ips32_base64", signed::MAX_DOCUMENT, true)?)?;
        let is_final = entry.get("final").and_then(Value::as_bool).unwrap_or(false);
        if module != "main" || version == 0 || !seen.insert((title_id, build_id.clone(), module.clone())) {
            return Err("invalid or duplicate patch identity".into());
        }
        let digest = hex_bytes(&text_field(entry, "sha256", 64, true)?, 32)?;
        if Sha256::digest(&ips).as_slice() != digest.as_slice() {
            return Err("invalid patch hash".into());
        }
        let guard_entries = entry
            .get("guards")
            .and_then(Value::as_array)
            .filter(|array| !array.is_empty() && array.len() <= 256)
            .ok_or("missing patch guards")?;
        let mut guards = Vec::with_capacity(guard_entries.len());
        for guard in guard_entries {
            let offset = guard
                .get("offset")
                .and_then(Value::as_u64)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or("invalid guard offset")?;
            guards.push(PatchGuard {
                offset,
                original: hex_bytes(&text_field(guard, "original_hex", 131_070, true)?, 65_535)?,
                patched: hex_bytes(&text_field(guard, "patched_hex", 131_070, true)?, 65_535)?,
            });
        }
        validate_records(&ips, &guards)?;
        patches.push(ManifestPatch {
            title_id,
            build_id,
            module,
            patch: OnlinePatch {
                id,
                version,
                enabled,
                ips,
                guards,
            },
            is_final,
        });
    }
    Ok(Manifest {
        revision: document.revision,
        etag,
        patches,
    })
}

impl State {
    fn remember_index(&mut self, index: &TitleIndex) -> Result<(), String> {
        if index.revision < self.index_revision || (index.revision == self.index_revision && index.etag != self.index_etag) {
            return Err("title index revision rollback or conflict rejected".into());
        }
        for (title_id, enabled) in &index.titles {
            self.index.insert(*title_id, *enabled);
        }
        self.index_revision = index.revision;
        self.index_etag = index.etag.clone();
        Ok(())
    }

    fn remember_manifest(&mut self, manifest: &Manifest, checked: Option<Instant>) {
        for entry in &manifest.patches {
            let key = (entry.title_id, entry.build_id.clone());
            let newer = self.patches.get(&key).map_or(true, |stored| {
                manifest.revision > stored.revision
                    || (manifest.revision == stored.revision && (!manifest.etag.is_empty() || stored.etag.is_empty()))
            });
            if newer {
                self.patches.insert(
                    key,
                    StoredPatch {
                        patch: entry.patch.clone(),
                        revision: manifest.revision,
                        etag: manifest.etag.clone(),
                        is_final: entry.is_final,
                        checked,
                    },
                );
            }
        }
    }

    fn load(&mut self) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        let now = Utc::now();
        let Ok(entries) = std::fs::read_dir(cache_dir()) else {
            return;
        };
        let mut indexes = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            if path.file_name().and_then(|name| name.to_str()) == Some("title-index.json") {
                match signed::verify(&bytes, SigningKey::NplnPatches, INDEX_TYPE, now, true).and_then(|doc| parse_title_index(&doc)) {
                    Ok(index) => indexes.push(index),
                    Err(error) => log::warn!("nextendo: NPLN title index cache rejected: {error}"),
                }
                continue;
            }
            match signed::verify(&bytes, SigningKey::NplnPatches, MANIFEST_TYPE, now, true).and_then(|doc| parse_manifest(&doc)) {
                Ok(manifest) => self.remember_manifest(&manifest, None),
                Err(error) => log::warn!("nextendo: NPLN patch cache {} rejected: {error}", path.display()),
            }
        }
        indexes.sort_by_key(|index| index.revision);
        for index in &indexes {
            if let Err(error) = self.remember_index(index) {
                log::warn!("nextendo: NPLN title index cache skipped: {error}");
            }
        }
        if !self.patches.is_empty() || !self.index.is_empty() {
            log::info!(
                "nextendo: NPLN patch cache holds {} patch(es), title index revision {}",
                self.patches.len(),
                self.index_revision
            );
        }
    }

    fn is_patchable(&self, title_id: u64) -> bool {
        self.index.get(&title_id).copied().unwrap_or(false)
            || self.patches.keys().any(|(title, _)| *title == title_id)
            || NPLN_TITLES.contains(&title_id)
    }

    fn refresh_index(&mut self, agent: &ureq::Agent) {
        if self.index_checked.is_some_and(|checked| checked.elapsed() < REFRESH_EVERY) {
            return;
        }
        self.index_checked = Some(Instant::now());
        let result = (|| -> Result<(), String> {
            let fetched = fetch(agent, &format!("{BASE}/npln-title-index"), Some(&self.index_etag))?;
            if fetched.status == 304 && !self.index_etag.is_empty() {
                log::info!("nextendo: NPLN title index unchanged");
                return Ok(());
            }
            if fetched.status != 200 {
                return Err(format!("HTTP {}", fetched.status));
            }
            let document = signed::verify(&fetched.body, SigningKey::NplnPatches, INDEX_TYPE, Utc::now(), false)?;
            let index = parse_title_index(&document)?;
            if fetched.etag.as_deref().is_some_and(|etag| etag != index.etag) {
                return Err("title index response tag mismatch".into());
            }
            self.remember_index(&index)?;
            save(&cache_dir().join("title-index.json"), &fetched.body);
            log::info!(
                "nextendo: NPLN title index revision {} lists {} titles",
                index.revision,
                index.titles.len()
            );
            Ok(())
        })();
        if let Err(error) = result {
            log::warn!("nextendo: NPLN title index check failed; keeping the verified copy: {error}");
        }
    }

    fn refresh_patch(&mut self, agent: &ureq::Agent, title_id: u64, build: &str) {
        let key = (title_id, build.to_string());
        let (etag, revision) = match self.patches.get(&key) {
            Some(stored) if stored.is_final => return,
            Some(stored) if stored.checked.is_some_and(|checked| checked.elapsed() < REFRESH_EVERY) => return,
            Some(stored) => (stored.etag.clone(), stored.revision),
            None => (String::new(), 0),
        };
        let now = Instant::now();
        let result = (|| -> Result<(), String> {
            let url = format!("{BASE}/npln-patches/{:016x}/{build}", title_id);
            let fetched = fetch(agent, &url, Some(&etag))?;
            if fetched.status == 304 && !etag.is_empty() {
                log::info!("nextendo: NPLN patch for build {build} unchanged");
                return Ok(());
            }
            if fetched.status == 404 {
                log::info!("nextendo: the catalog has no NPLN patch for build {build}");
                return Ok(());
            }
            if fetched.status != 200 {
                return Err(format!("HTTP {}", fetched.status));
            }
            let document = signed::verify(&fetched.body, SigningKey::NplnPatches, MANIFEST_TYPE, Utc::now(), false)?;
            let manifest = parse_manifest(&document)?;
            let entry = match manifest.patches.as_slice() {
                [entry] if entry.title_id == title_id && entry.build_id == build && entry.module == "main" => entry,
                _ => return Err("patch response does not match the requested build".into()),
            };
            if manifest.etag.is_empty() || fetched.etag.as_deref().is_some_and(|tag| tag != manifest.etag) {
                return Err("patch response tag mismatch".into());
            }
            if manifest.revision < revision {
                return Err("patch catalog revision rollback rejected".into());
            }
            save(&cache_dir().join(format!("{:016x}-{build}.json", title_id)), &fetched.body);
            log::info!(
                "nextendo: NPLN patch {} v{} downloaded and verified for build {build}",
                entry.patch.id,
                entry.patch.version
            );
            self.remember_manifest(&manifest, Some(now));
            Ok(())
        })();
        match result {
            Ok(()) => {
                if let Some(stored) = self.patches.get_mut(&key) {
                    stored.checked = Some(now);
                }
            }
            Err(error) => log::warn!("nextendo: NPLN patch catalog refresh failed; keeping verified patches: {error}"),
        }
    }
}

fn save(path: &std::path::Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let temporary = path.with_extension("json.tmp");
    let written = std::fs::write(&temporary, bytes).and_then(|_| std::fs::rename(&temporary, path));
    if let Err(error) = written {
        log::warn!("nextendo: NPLN cache {} not saved: {error}", path.display());
    }
}

pub fn patch_for(agent: &ureq::Agent, title_id: u64, build: &str) -> Option<OnlinePatch> {
    let build = build.to_ascii_uppercase();
    if build.len() != 64 || !build.chars().all(|ch| ch.is_ascii_hexdigit()) {
        log::warn!("nextendo: cannot check the patch catalog: invalid main build ID");
        return None;
    }
    let mut state = state().lock().ok()?;
    state.load();
    state.refresh_index(agent);
    if !state.is_patchable(title_id) {
        return None;
    }
    state.refresh_patch(agent, title_id, &build);
    state.patches.get(&(title_id, build)).map(|stored| stored.patch.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLATOON_3_MANIFEST: &str = r#"{"document_type":"npln_patch_catalog","schema_version":1,"revision":4,"issued_at":"2026-10-16T18:31:40.115291Z","expires_at":"2026-10-17T18:31:40.115291Z","patch_etag":"\"1a7e3c89d64d479d83034d88e53807295956e72cf5e9c5271c4288c32bf18752\"","patches":[{"id":"splatoon3-npln-28c4287a","title_id":"0100c2500fc20000","build_id":"28C4287AEE36F7499DA60F3E68B54C70DA382D75000000000000000000000000","module":"main","enabled":true,"patch_version":1,"final":false,"required_capabilities":["npln.ips32.v1"],"offset_basis":"nso_header_included","ips32_base64":"SVBTMzIAFN2AAAT0Ax8qABThsAAEHyAD1QAVeyAABCoAgFJFRU9G","sha256":"4877c26e8487f2ceaab03a0e9047a12b20f85439c95955474db43060bd7996d2","guards":[{"offset":1367424,"original_hex":"F403002A","patched_hex":"F4031F2A"},{"offset":1368496,"original_hex":"C0020034","patched_hex":"1F2003D5"},{"offset":1407776,"original_hex":"AAE24039","patched_hex":"2A008052"}]}]}"#;

    fn document(json: &str, document_type: &str) -> Verified {
        let payload: Value = serde_json::from_str(json).unwrap();
        Verified {
            document_type: document_type.into(),
            revision: payload["revision"].as_u64().unwrap(),
            issued_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
            payload,
        }
    }

    #[test]
    fn the_live_splatoon_3_manifest_parses() {
        let manifest = parse_manifest(&document(SPLATOON_3_MANIFEST, MANIFEST_TYPE)).unwrap();
        assert_eq!(manifest.revision, 4);
        assert!(manifest.etag.starts_with("\"1a7e3c89"));
        let entry = &manifest.patches[0];
        assert_eq!(entry.title_id, nextendo::SPLATOON_3);
        assert_eq!(entry.module, "main");
        assert!(entry.patch.enabled && !entry.is_final);
        assert_eq!(entry.patch.guards.len(), 3);
        assert_eq!(entry.patch.guards[2].offset, 0x157B20);
        assert_eq!(&entry.patch.ips[..5], b"IPS32");
    }

    #[test]
    fn manifest_tampering_is_rejected() {
        for (needle, replacement) in [
            ("\"patched_hex\":\"F4031F2A\"", "\"patched_hex\":\"F4031F2B\""),
            ("4877c26e", "4877c26f"),
            ("\"module\":\"main\"", "\"module\":\"rtld\""),
            ("\"patch_version\":1", "\"patch_version\":0"),
            ("\"offset_basis\":\"nso_header_included\"", "\"offset_basis\":\"segment\""),
        ] {
            let json = SPLATOON_3_MANIFEST.replacen(needle, replacement, 1);
            assert_ne!(json, SPLATOON_3_MANIFEST, "{needle}");
            assert!(parse_manifest(&document(&json, MANIFEST_TYPE)).is_err(), "{needle}");
        }
    }

    #[test]
    fn guards_must_cover_every_record() {
        let ips = nexium_common_ips(&[(0x104, &[1, 2, 3, 4]), (0x200, &[9, 9])]);
        let guard = |offset: u32, original: &[u8], patched: &[u8]| PatchGuard {
            offset,
            original: original.to_vec(),
            patched: patched.to_vec(),
        };
        let full = vec![guard(0x104, &[0, 0, 0, 0], &[1, 2, 3, 4]), guard(0x200, &[5, 5], &[9, 9])];
        assert!(validate_records(&ips, &full).is_ok());
        assert!(validate_records(&ips, &full[..1]).is_err());
        assert!(validate_records(&ips, &[]).is_err());
        let wrong = vec![guard(0x104, &[0, 0, 0, 0], &[1, 2, 3, 5]), full[1].clone()];
        assert!(validate_records(&ips, &wrong).is_err());
        let below_header = nexium_common_ips(&[(0x10, &[1])]);
        assert!(validate_records(&below_header, &[guard(0x10, &[0], &[1])]).is_err());
    }

    fn nexium_common_ips(records: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = b"IPS32".to_vec();
        for (offset, bytes) in records {
            out.extend_from_slice(&offset.to_be_bytes());
            out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
            out.extend_from_slice(bytes);
        }
        out.extend_from_slice(b"EEOF");
        out
    }

    #[test]
    fn title_index_parses_and_rolls_forward_only() {
        let json = r#"{"document_type":"npln_title_index","schema_version":1,"revision":4,"issued_at":"2026-10-16T18:31:40Z","expires_at":"2026-10-17T18:31:40Z","index_etag":"\"2d1d27ec02203930e4b5c87287d5640856bfb40af782f1908b1175a3324c7b7f\"","titles":[{"title_id":"010015100b514000","enabled":true},{"title_id":"0100e65002bb8000","enabled":false}]}"#;
        let index = parse_title_index(&document(json, INDEX_TYPE)).unwrap();
        assert_eq!(index.revision, 4);
        assert_eq!(index.titles, vec![(0x010015100B514000, true), (0x0100E65002BB8000, false)]);
        let mut state = State::default();
        state.remember_index(&index).unwrap();
        assert!(state.is_patchable(0x010015100B514000));
        assert!(!state.is_patchable(0x0100E65002BB8000));
        assert!(state.is_patchable(nextendo::SPLATOON_3));
        assert!(!state.is_patchable(nextendo::MARIO_KART_8_DELUXE));
        let older = TitleIndex { revision: 3, ..index.clone() };
        assert!(state.remember_index(&older).is_err());
        let conflicting = TitleIndex { etag: "\"".to_string() + &"0".repeat(64) + "\"", ..index.clone() };
        assert!(state.remember_index(&conflicting).is_err());
        let duplicate = json.replacen("0100e65002bb8000", "010015100b514000", 1);
        assert!(parse_title_index(&document(&duplicate, INDEX_TYPE)).is_err());
    }

    #[test]
    #[ignore = "needs network access and NEXIUM_DATA_ROOT"]
    fn live_splatoon_3_patch_is_served() {
        let agent = super::super::launch::agent();
        let build = format!("{:0<64}", "28C4287AEE36F7499DA60F3E68B54C70DA382D75");
        let patch = patch_for(&agent, nextendo::SPLATOON_3, &build).expect("catalog patch");
        assert!(patch.enabled);
        assert_eq!(patch.guards.len(), 3);
        assert!(cache_dir().join("title-index.json").is_file());
        assert!(cache_dir().join(format!("{:016x}-{build}.json", nextendo::SPLATOON_3)).is_file());
        assert_eq!(patch_for(&agent, nextendo::SPLATOON_3, &build), Some(patch));
        assert_eq!(patch_for(&agent, nextendo::MARIO_KART_8_DELUXE, &build), None);
        assert_eq!(patch_for(&agent, nextendo::SPLATOON_3, &"A".repeat(64)), None);
    }

    #[test]
    fn newer_manifests_replace_older_patches() {
        let manifest = parse_manifest(&document(SPLATOON_3_MANIFEST, MANIFEST_TYPE)).unwrap();
        let mut state = State::default();
        state.remember_manifest(&manifest, None);
        let key = (nextendo::SPLATOON_3, manifest.patches[0].build_id.clone());
        assert_eq!(state.patches[&key].revision, 4);
        let older = Manifest { revision: 3, ..manifest.clone() };
        state.remember_manifest(&older, None);
        assert_eq!(state.patches[&key].revision, 4);
        let newer = Manifest { revision: 5, ..manifest.clone() };
        state.remember_manifest(&newer, None);
        assert_eq!(state.patches[&key].revision, 5);
    }
}
