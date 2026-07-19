use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

const BUNDLE_MAGIC: [u8; 8] = *b"NXBUNDL1";
const BUNDLE_VERSION: u32 = 3;
const SPIRV_MAGIC: u32 = 0x0723_0203;
const MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;
const FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(8);
const FLUSH_EVERY_N_RECORDS: u32 = 200;

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
    pub vs_tex_base: u32,
    pub vs_tex_count: u32,
    pub fs_sampler_arrayed: bool,
    pub fs_cbuf_reads: Vec<(u32, u32)>,
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
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        nexium_common::paths::root()
            .join("shader_cache")
            .join(format!("{}.bundles", title)),
    )
}

fn record_valid(rec: &BundleRecord) -> bool {
    rec.vs_spirv.first() == Some(&SPIRV_MAGIC) && rec.fs_spirv.first() == Some(&SPIRV_MAGIC)
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
            let mut since_flush: u32 = 0;
            let flush = |records: &HashMap<u64, Arc<BundleRecord>>,
                         failed: &HashSet<u64>|
             -> bool {
                let file = BundleFile {
                    magic: BUNDLE_MAGIC,
                    version: BUNDLE_VERSION,
                    build_id: build_id(),
                    records: records.values().map(|r| (**r).clone()).collect(),
                    failed: failed.iter().copied().collect(),
                };
                let Ok(bytes) = bincode::serialize(&file) else {
                    return false;
                };
                if bytes.len() as u64 > MAX_FILE_BYTES {
                    log::warn!(
                        "bundle cache exceeds {} bytes, persistence disabled",
                        MAX_FILE_BYTES
                    );
                    return true;
                }
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let tmp = path.with_extension("bundles.tmp");
                if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_ok() {
                    log::info!(
                        "bundle cache saved ({} records, {} failed, {} bytes)",
                        file.records.len(),
                        file.failed.len(),
                        bytes.len()
                    );
                }
                false
            };
            loop {
                let mut quiet = false;
                match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                    Ok(WriterMsg::Record(rec)) => {
                        if records.insert(rec.content_key, rec).is_none() {
                            dirty = true;
                            since_flush += 1;
                        }
                    }
                    Ok(WriterMsg::Failed(key)) => {
                        if failed.insert(key) {
                            dirty = true;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        quiet = true;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        if dirty && !oversize {
                            flush(&records, &failed);
                        }
                        break;
                    }
                }
                if dirty
                    && !oversize
                    && (quiet
                        || since_flush >= FLUSH_EVERY_N_RECORDS
                        || last_flush.elapsed() >= FLUSH_INTERVAL)
                {
                    oversize = flush(&records, &failed);
                    dirty = false;
                    since_flush = 0;
                    last_flush = std::time::Instant::now();
                }
            }
        })
        .is_ok();
    spawned.then_some(tx)
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
