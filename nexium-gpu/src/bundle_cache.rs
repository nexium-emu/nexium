use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

const BUNDLE_MAGIC: [u8; 8] = *b"NXBUNDL1";
const BUNDLE_VERSION: u32 = 27;
const SPIRV_MAGIC: u32 = 0x0723_0203;
const MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;
const FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum CbufIndexOrigin {
    Static,
    Constant(u32),
    Gpr(u8),
    Instruction(u32),
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CbufRead {
    pub logical_slot: u8,
    pub byte_offset: u32,
    pub index_origin: CbufIndexOrigin,
}

impl CbufRead {
    pub fn is_indexed(self) -> bool {
        !matches!(self.index_origin, CbufIndexOrigin::Static)
    }

    pub fn effective_byte_offset(self) -> Option<u32> {
        match self.index_origin {
            CbufIndexOrigin::Static => Some(self.byte_offset),
            CbufIndexOrigin::Constant(index) => Some(self.byte_offset.wrapping_add(index)),
            CbufIndexOrigin::Gpr(_) | CbufIndexOrigin::Instruction(_) => None,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BundleRecord {
    pub content_key: u64,
    pub vs_spirv: Vec<u32>,
    pub fs_spirv: Vec<u32>,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub fs_tex_ids: Vec<u32>,
    pub texture_numeric_manifest: Vec<crate::texture_manifest::TextureNumericBinding>,
    pub vs_tex_base: u32,
    pub vs_tex_count: u32,
    pub fs_sampler_arrayed: bool,
    pub vs_sampler_arrayed: bool,
    pub depth_compare_2d_mask: u32,
    pub depth_compare_cube_mask: u32,
    pub depth_compare_cube_array_mask: u32,
    pub graphics_cbuf_reads: Vec<CbufRead>,
    pub cbuf_used: u32,
    pub ssbo_descs: Vec<(u8, u32, u32)>,
    #[serde(default)]
    pub fs_tex_or_partners: Vec<(u32, u32)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BundleFile {
    magic: [u8; 8],
    version: u32,
    build_id: u64,
    records: Vec<BundleRecord>,
    failed: Vec<u64>,
}

#[derive(serde::Serialize)]
struct BorrowedBundleFile<'a> {
    magic: [u8; 8],
    version: u32,
    build_id: u64,
    records: Vec<&'a BundleRecord>,
    failed: Vec<u64>,
}

fn build_id() -> u64 {
    static ID: OnceLock<u64> = OnceLock::new();
    *ID.get_or_init(|| {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        eat(env!("CARGO_PKG_NAME").as_bytes());
        eat(env!("CARGO_PKG_VERSION").as_bytes());
        eat(&BUNDLE_VERSION.to_le_bytes());
        h
    })
}

enum WriterMsg {
    Record(Arc<BundleRecord>),
    Failed(u64),
}

pub struct BundleStore {
    records: Mutex<HashMap<u64, Arc<BundleRecord>>>,
    failed: Mutex<HashSet<u64>>,
    tx: Option<std::sync::mpsc::Sender<WriterMsg>>,
    enabled: bool,
}

fn bundles_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")?;
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        PathBuf::from(base)
            .join("NeXium")
            .join("shader_cache")
            .join(format!("{}.bundles", title)),
    )
}

fn record_valid(rec: &BundleRecord) -> bool {
    rec.vs_spirv.first() == Some(&SPIRV_MAGIC)
        && rec.fs_spirv.first() == Some(&SPIRV_MAGIC)
        && crate::texture_manifest::normalize_texture_numeric_manifest(
            rec.texture_numeric_manifest.clone(),
        )
        .is_ok_and(|manifest| manifest == rec.texture_numeric_manifest)
}

fn load_file(path: &PathBuf) -> Option<BundleFile> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_FILE_BYTES {
        log::warn!("bundle cache too large ({} bytes), ignoring", meta.len());
        return None;
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                log::warn!("bundle cache read failed: {:?}", e);
            }
            return None;
        }
    };
    match bincode::deserialize::<BundleFile>(&bytes) {
        Ok(f) if f.magic == BUNDLE_MAGIC && f.version == BUNDLE_VERSION => {
            if f.build_id != build_id() {
                log::info!("bundle cache from a different build, ignoring");
                return None;
            }
            Some(f)
        }
        Ok(f) => {
            log::warn!(
                "bundle cache magic/version mismatch (version {}), ignoring",
                f.version
            );
            None
        }
        Err(e) => {
            log::warn!("bundle cache deserialize failed: {:?}", e);
            None
        }
    }
}

fn spawn_writer(
    path: PathBuf,
    mut records: HashMap<u64, Arc<BundleRecord>>,
    mut failed: HashSet<u64>,
) -> Option<std::sync::mpsc::Sender<WriterMsg>> {
    let (tx, rx) = std::sync::mpsc::channel::<WriterMsg>();
    let spawned = std::thread::Builder::new()
        .name("nexium-bundlecache".to_string())
        .spawn(move || {
            let mut dirty = false;
            let mut oversize = false;
            let mut last_flush = std::time::Instant::now();
            let flush =
                |records: &HashMap<u64, Arc<BundleRecord>>, failed: &HashSet<u64>| -> bool {
                    let file = BorrowedBundleFile {
                        magic: BUNDLE_MAGIC,
                        version: BUNDLE_VERSION,
                        build_id: build_id(),
                        records: records.values().map(Arc::as_ref).collect(),
                        failed: failed.iter().copied().collect(),
                    };
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let tmp = path.with_extension("bundles.tmp");
                    let output = match std::fs::File::create(&tmp) {
                        Ok(output) => output,
                        Err(error) => {
                            log::warn!("bundle cache temp create failed: {:?}", error);
                            return false;
                        }
                    };
                    let mut writer = std::io::BufWriter::new(output);
                    if let Err(error) = bincode::serialize_into(&mut writer, &file) {
                        log::warn!("bundle cache serialize failed: {:?}", error);
                        drop(writer);
                        let _ = std::fs::remove_file(&tmp);
                        return false;
                    }
                    if let Err(error) = writer.flush() {
                        log::warn!("bundle cache flush failed: {:?}", error);
                        drop(writer);
                        let _ = std::fs::remove_file(&tmp);
                        return false;
                    }
                    let bytes = writer
                        .get_ref()
                        .metadata()
                        .map(|meta| meta.len())
                        .unwrap_or(0);
                    drop(writer);
                    if bytes > MAX_FILE_BYTES {
                        let _ = std::fs::remove_file(&tmp);
                        log::warn!(
                            "bundle cache exceeds {} bytes, persistence disabled",
                            MAX_FILE_BYTES
                        );
                        return true;
                    }
                    if std::fs::rename(&tmp, &path).is_ok() {
                        log::info!(
                            "bundle cache saved ({} records, {} failed, {} bytes)",
                            file.records.len(),
                            file.failed.len(),
                            bytes
                        );
                    }
                    false
                };
            loop {
                match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                    Ok(WriterMsg::Record(rec)) => {
                        if records.insert(rec.content_key, rec).is_none() {
                            dirty = true;
                        }
                    }
                    Ok(WriterMsg::Failed(key)) => {
                        if failed.insert(key) {
                            dirty = true;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        if dirty && !oversize {
                            flush(&records, &failed);
                        }
                        break;
                    }
                }
                if should_flush(dirty, oversize, last_flush.elapsed()) {
                    oversize = flush(&records, &failed);
                    dirty = false;
                    last_flush = std::time::Instant::now();
                }
            }
        })
        .is_ok();
    spawned.then_some(tx)
}

fn should_flush(dirty: bool, oversize: bool, elapsed: std::time::Duration) -> bool {
    dirty && !oversize && elapsed >= FLUSH_INTERVAL
}

pub fn bundle_store() -> &'static BundleStore {
    static STORE: OnceLock<BundleStore> = OnceLock::new();
    STORE.get_or_init(|| {
        let enabled = std::env::var_os("NEXIUM_NO_BUNDLE_CACHE").is_none();
        if !enabled {
            return BundleStore {
                records: Mutex::new(HashMap::new()),
                failed: Mutex::new(HashSet::new()),
                tx: None,
                enabled: false,
            };
        }
        let path = bundles_path();
        let loaded = path.as_ref().and_then(load_file);
        let mut records: HashMap<u64, Arc<BundleRecord>> = HashMap::new();
        let mut failed: HashSet<u64> = HashSet::new();
        if let Some(file) = loaded {
            let total = file.records.len();
            for rec in file.records {
                if record_valid(&rec) {
                    records.insert(rec.content_key, Arc::new(rec));
                }
            }
            let dropped = total - records.len();
            if dropped > 0 {
                log::warn!("bundle cache dropped {} invalid records", dropped);
            }
            failed = file.failed.into_iter().collect();
        }
        log::info!(
            "bundle cache loaded: {} records, {} failed",
            records.len(),
            failed.len()
        );
        let tx = path.and_then(|p| spawn_writer(p, records.clone(), failed.clone()));
        BundleStore {
            records: Mutex::new(records),
            failed: Mutex::new(failed),
            tx,
            enabled: true,
        }
    })
}

impl BundleStore {
    pub fn get(&self, content_key: u64) -> Option<Arc<BundleRecord>> {
        if !self.enabled {
            return None;
        }
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&content_key)
            .cloned()
    }

    pub fn insert(&self, rec: Arc<BundleRecord>) {
        if !self.enabled {
            return;
        }
        let mut guard = self.records.lock().unwrap_or_else(|e| e.into_inner());
        if guard.insert(rec.content_key, rec.clone()).is_none() {
            drop(guard);
            if let Some(tx) = &self.tx {
                let _ = tx.send(WriterMsg::Record(rec));
            }
        }
    }

    pub fn is_failed(&self, content_key: u64) -> bool {
        if !self.enabled {
            return false;
        }
        self.failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&content_key)
    }

    pub fn mark_failed(&self, content_key: u64) {
        if !self.enabled {
            return;
        }
        let mut guard = self.failed.lock().unwrap_or_else(|e| e.into_inner());
        if guard.insert(content_key) {
            drop(guard);
            if let Some(tx) = &self.tx {
                let _ = tx.send(WriterMsg::Failed(content_key));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> BundleRecord {
        BundleRecord {
            content_key: 0x1234,
            vs_spirv: vec![SPIRV_MAGIC, 1, 2],
            fs_spirv: vec![SPIRV_MAGIC, 3, 4],
            vs_cbuf_mask: 0x11,
            fs_cbuf_mask: 0x22,
            vs_hash: 0x5678,
            fs_hash: 0x9abc,
            fs_tex_ids: vec![7, 8],
            texture_numeric_manifest:
                crate::texture_manifest::normalize_texture_numeric_manifest(vec![
                    crate::texture_manifest::TextureNumericBinding::new(
                        8,
                        1,
                        nexium_spirv::TextureNumericType::Uint,
                    ),
                    crate::texture_manifest::TextureNumericBinding::new(
                        7,
                        0,
                        nexium_spirv::TextureNumericType::Float,
                    )
                    .with_image_kind(crate::texture_manifest::GraphicsTextureImageKind::CubeArray),
                    crate::texture_manifest::TextureNumericBinding::new(
                        9,
                        10,
                        nexium_spirv::TextureNumericType::Float,
                    )
                    .with_image_kind(crate::texture_manifest::GraphicsTextureImageKind::Buffer),
                ])
                .unwrap(),
            vs_tex_base: 9,
            vs_tex_count: 10,
            fs_sampler_arrayed: true,
            vs_sampler_arrayed: false,
            depth_compare_2d_mask: 1 << 1,
            depth_compare_cube_mask: 1 << 2,
            depth_compare_cube_array_mask: 0,
            graphics_cbuf_reads: vec![
                CbufRead {
                    logical_slot: 2,
                    byte_offset: 0x40,
                    index_origin: CbufIndexOrigin::Static,
                },
                CbufRead {
                    logical_slot: 6,
                    byte_offset: 0x10,
                    index_origin: CbufIndexOrigin::Constant(0xffff_fff0),
                },
                CbufRead {
                    logical_slot: 22,
                    byte_offset: 0x2aa0,
                    index_origin: CbufIndexOrigin::Gpr(37),
                },
                CbufRead {
                    logical_slot: 23,
                    byte_offset: 0x1550,
                    index_origin: CbufIndexOrigin::Instruction(73),
                },
            ],
            cbuf_used: 0x33,
            ssbo_descs: vec![(4, 5, 6)],
            fs_tex_or_partners: vec![(11, 12)],
        }
    }

    #[test]
    fn bundle_flush_waits_for_interval() {
        assert!(!should_flush(
            true,
            false,
            std::time::Duration::from_secs(1)
        ));
        assert!(!should_flush(
            true,
            false,
            FLUSH_INTERVAL - std::time::Duration::from_nanos(1)
        ));
        assert!(should_flush(true, false, FLUSH_INTERVAL));
        assert!(!should_flush(false, false, FLUSH_INTERVAL));
        assert!(!should_flush(true, true, FLUSH_INTERVAL));
    }

    #[test]
    fn borrowed_bundle_serialization_preserves_cache_format() {
        let record = sample_record();
        let owned = BundleFile {
            magic: BUNDLE_MAGIC,
            version: BUNDLE_VERSION,
            build_id: build_id(),
            records: vec![record.clone()],
            failed: vec![0xdef0],
        };
        let borrowed = BorrowedBundleFile {
            magic: BUNDLE_MAGIC,
            version: BUNDLE_VERSION,
            build_id: build_id(),
            records: vec![&record],
            failed: vec![0xdef0],
        };

        assert_eq!(
            bincode::serialize(&borrowed).unwrap(),
            bincode::serialize(&owned).unwrap()
        );
    }

    #[test]
    fn previous_bundle_version_is_rejected() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nexium-bundle-cache-{}-{}.bin",
            std::process::id(),
            nonce
        ));
        let file = BundleFile {
            magic: BUNDLE_MAGIC,
            version: BUNDLE_VERSION - 1,
            build_id: build_id(),
            records: Vec::new(),
            failed: Vec::new(),
        };

        std::fs::write(&path, bincode::serialize(&file).unwrap()).unwrap();
        let loaded = load_file(&path);
        std::fs::remove_file(&path).unwrap();

        assert!(loaded.is_none());
    }

    #[test]
    fn bundle_record_roundtrip_preserves_cbuf_and_texture_cache_identity() {
        let record = sample_record();
        let expected_identity = (
            record.content_key,
            crate::texture_manifest::texture_numeric_manifest_fingerprint(
                &record.texture_numeric_manifest,
            ),
        );

        let bytes = bincode::serialize(&record).expect("serialize BundleRecord");
        let decoded =
            bincode::deserialize::<BundleRecord>(&bytes).expect("deserialize BundleRecord");

        assert!(record_valid(&decoded));
        assert_eq!(decoded.graphics_cbuf_reads, record.graphics_cbuf_reads);
        assert_eq!(
            decoded.texture_numeric_manifest,
            record.texture_numeric_manifest
        );
        assert_eq!(
            decoded.texture_numeric_manifest[0].image_kind,
            crate::texture_manifest::GraphicsTextureImageKind::CubeArray
        );
        assert!(decoded.texture_numeric_manifest.iter().any(|binding| {
            binding.descriptor_slot == 10
                && binding.image_kind
                    == crate::texture_manifest::GraphicsTextureImageKind::Buffer
        }));
        assert_eq!(
            (
                decoded.content_key,
                crate::texture_manifest::texture_numeric_manifest_fingerprint(
                    &decoded.texture_numeric_manifest,
                ),
            ),
            expected_identity
        );
        assert_eq!(decoded.graphics_cbuf_reads[1].effective_byte_offset(), Some(0));
        assert_eq!(decoded.graphics_cbuf_reads[2].effective_byte_offset(), None);
    }
}
