use super::super::PipelineStats;
use super::engines::{
    Fermi2D, KeplerCompute, KeplerMemory, KeplerMemoryWriteOutcome, Maxwell3D, MaxwellDma,
    FERMI_2D_CLASS, KEPLER_COMPUTE_CLASS, KEPLER_MEMORY_CLASS, MACRO_REGISTERS_START,
    MAXWELL_DMA_CLASS,
};
use super::GpuMappings;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

fn pb_anomaly_dump_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_PB_ANOMALY_DUMP").is_some())
}

static PB_ANOMALY_PENDING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static PB_ANOMALY_REASON: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub(crate) fn note_pb_anomaly(reason: &str) {
    if !pb_anomaly_dump_enabled() || PB_ANOMALY_PENDING.load(Ordering::Acquire) {
        return;
    }
    if let Ok(mut slot) = PB_ANOMALY_REASON.lock() {
        if !PB_ANOMALY_PENDING.load(Ordering::Relaxed) {
            *slot = Some(reason.to_string());
            PB_ANOMALY_PENDING.store(true, Ordering::Release);
        }
    }
}

const PB_ANOMALY_RING_ENTRIES: usize = 8;
const PB_ANOMALY_RING_WORDS: usize = 4096;

struct PbAnomalyEntry {
    gpu_va: u64,
    state_in: (u32, u32, u32, bool, bool),
    total_words: usize,
    words: Vec<u32>,
}

struct PbAnomalyRanges {
    sequence: u64,
    total_entries: usize,
    enabled: bool,
    entries: Vec<(CommandListHeader, CommandListHeader, Option<(u64, u64)>)>,
}

static PB_ANOMALY_RANGES: std::sync::Mutex<std::collections::VecDeque<PbAnomalyRanges>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

fn record_pb_anomaly_ranges(
    raw: &[CommandListHeader],
    normalized: &[CommandListHeader],
    mappings: &GpuMappings,
    enabled: bool,
) {
    if !pb_anomaly_dump_enabled() {
        return;
    }
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let entries = raw
        .iter()
        .zip(normalized)
        .take(512)
        .map(|(raw, normalized)| (*raw, *normalized, mappings.cpu_range_for(raw.address())))
        .collect();
    if let Ok(mut ring) = PB_ANOMALY_RANGES.lock() {
        if ring.len() >= PB_ANOMALY_RING_ENTRIES {
            ring.pop_front();
        }
        ring.push_back(PbAnomalyRanges {
            sequence,
            total_entries: raw.len(),
            enabled,
            entries,
        });
    }
}

fn pb_frame_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_PB_FRAME_TRACE").is_some())
}

fn gpfifo_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_GPFIFO_TRACE").is_some())
}

fn gs_dump_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("NEXIUM_DUMP_GS").is_some()
            || std::env::var_os("NEXIUM_DUMP_SHADERS").is_some()
    })
}

pub(crate) fn async_barrier_segments_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_ASYNC_BARRIER_SEGMENTS")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

pub(crate) fn maxwell_sync_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MW3D_SYNC_DBG").is_some())
}

fn compute_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_DBG").is_some())
}

pub(crate) fn apply_kepler_memory_write(
    cache: &mut super::vk_dispatch::SsboSnapshotCache,
    mappings: &GpuMappings,
    renderer: Option<&Arc<nexium_gpu::Renderer>>,
    outcome: KeplerMemoryWriteOutcome,
) {
    invalidate_kepler_render_targets(renderer, mappings, &outcome);
    match outcome {
        KeplerMemoryWriteOutcome::NoWrite => {}
        KeplerMemoryWriteOutcome::Exact(spans) => {
            cache.invalidate_gpu_writes(mappings, &spans);
        }
        KeplerMemoryWriteOutcome::Unknown(_) => cache.clear(),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum KeplerRtInvalidation {
    Exact(Vec<(u64, u64, u64)>),
}

fn kepler_rt_invalidation(
    mappings: &GpuMappings,
    outcome: &KeplerMemoryWriteOutcome,
) -> Option<KeplerRtInvalidation> {
    match outcome {
        KeplerMemoryWriteOutcome::NoWrite => None,
        KeplerMemoryWriteOutcome::Exact(spans) | KeplerMemoryWriteOutcome::Unknown(spans) => {
            let mut exact = Vec::with_capacity(spans.len());
            for &(gpu_va, size) in spans {
                let Ok(size) = u64::try_from(size) else {
                    return None;
                };
                if size == 0 {
                    continue;
                }
                let cpu_addr = mappings.cpu_address_for(gpu_va).unwrap_or(0);
                exact.push((cpu_addr, gpu_va, size));
            }
            (!exact.is_empty()).then_some(KeplerRtInvalidation::Exact(exact))
        }
    }
}

fn invalidate_kepler_render_targets(
    renderer: Option<&Arc<nexium_gpu::Renderer>>,
    mappings: &GpuMappings,
    outcome: &KeplerMemoryWriteOutcome,
) {
    let Some(renderer) = renderer.cloned() else {
        return;
    };
    let Some(invalidation) = kepler_rt_invalidation(mappings, outcome) else {
        return;
    };
    let KeplerRtInvalidation::Exact(spans) = invalidation;
    let job = move || {
        for (cpu_addr, gpu_va, size) in spans {
            renderer.invalidate_render_target_range(cpu_addr, size, &[(gpu_va, size)]);
        }
    };
    if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
        render_thread.submit_named("kepler-guest-write-invalidate", Box::new(job));
    } else {
        job();
    }
}

pub(crate) mod kickprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub const LOCKS: usize = 0;
    pub const ELIST: usize = 1;
    pub const PBREAD: usize = 2;
    pub const M3D: usize = 3;
    pub const MACRO: usize = 4;
    pub const CBUFWB: usize = 5;
    pub const SEMACQ: usize = 6;
    pub const SEMREL: usize = 7;
    pub const BARRIER: usize = 8;
    pub const ENQ: usize = 9;
    pub const FLUSHP: usize = 10;
    pub const DMA: usize = 11;
    pub const FERMI: usize = 12;
    pub const KEPLER: usize = 13;
    pub const PULLER: usize = 14;
    pub const SMALLRT: usize = 15;
    pub const DISJOINT: usize = 16;
    pub const KM_FLUSH: usize = 16;
    pub const KCU_FLUSH: usize = 17;
    pub const KC_LAUNCH: usize = 18;
    pub const KC_SYNC: usize = 19;
    pub const KC_EXEC: usize = 20;
    pub const KC_WB: usize = 21;
    pub const KC_RESOLVE: usize = 22;
    pub const DMA_COPY: usize = 23;
    pub const DMA_FALLBACK: usize = 24;
    pub const DMA_STAGE: usize = 25;
    pub const DMA_MAP: usize = 26;
    pub const DMA_META: usize = 27;
    pub const DMA_RT_LINEAR: usize = 28;
    pub const DMA_VIRTUAL: usize = 29;
    pub const ENQ_WATCH: usize = 30;
    pub const ENQ_BUILD: usize = 31;
    pub const VK_FLUSH: usize = 32;
    pub const HOST_DRAWS: usize = 33;
    pub const DRAW_INSTANCES: usize = 34;
    pub const ENQ_PRE: usize = 35;
    pub const ENQ_TEX: usize = 36;
    pub const ENQ_CBUF: usize = 37;
    pub const ENQ_INDEX: usize = 38;
    pub const ENQ_STATE: usize = 39;
    pub const ENQ_SSBO: usize = 40;
    pub const ENQ_FINAL: usize = 41;
    pub const VKF_CUBE: usize = 42;
    pub const VKF_INVAL: usize = 43;
    pub const VKF_PREP: usize = 44;
    pub const VKF_SUBMIT: usize = 45;
    pub const CBUF_SLOTS: usize = 46;
    pub const CBUF_WW: usize = 47;
    pub const CBUF_MISS_READ: usize = 48;
    pub const CBUF_INVAL: usize = 49;
    pub const CBUF_PACK: usize = 50;
    pub const CBUF_HIT_LAST: usize = 51;
    pub const CBUF_HIT_ENTRY: usize = 52;
    pub const CBUF_HIT_COVER: usize = 53;
    pub const CBUF_MISS: usize = 54;
    pub const CBUF_MISS_DIRTY: usize = 55;
    pub const CBUF_MISS_GROW: usize = 56;
    pub const PREP_READ: usize = 57;
    pub const KICK_INVAL: usize = 58;
    pub const KICK_INVAL_N: usize = 59;
    pub const GPU_INVAL: usize = 60;
    pub const GPU_INVAL_N: usize = 61;
    pub const CBUF_MISS_COLD: usize = 62;
    pub const PREP_VERTEX: usize = 64;
    pub const PREP_TIC: usize = 65;
    pub const PREP_TEXTURE: usize = 66;
    pub const PREP_TSC: usize = 67;
    pub const PREP_TEX_CACHE_HIT: usize = 68;
    pub const PREP_TEX_CACHE_MISS: usize = 69;
    pub const PREP_TEX_CACHE_STALE: usize = 70;
    pub const PREP_TEX_CACHE_EVICT: usize = 71;
    pub const PREP_TEX_RT_CANDIDATE: usize = 72;
    pub const PREP_TEX_CUBE: usize = 73;
    pub const PREP_TEX_OTHER: usize = 74;
    pub const CBUF_ALLOC: usize = 75;
    pub const CBUF_COPY: usize = 76;
    pub const CBUF_HIT_RECENT: usize = 77;
    pub const PREP_TEX_FERMI_EXACT_CANDIDATE: usize = 78;
    pub const PREP_TEX_FERMI_RAW_SNAPSHOT: usize = 79;
    pub const PREP_TEX_FERMI_SNAPSHOT_LEASE: usize = 80;
    pub const FLUSH_SOFT_BARRIER: usize = 81;
    pub const FLUSH_SOFT_TEXTURE_INVALIDATE: usize = 82;
    pub const FLUSH_HARD_TAIL: usize = 83;
    pub const FLUSH_HARD_PULLER: usize = 84;
    pub const FLUSH_HARD_INLINE_UPLOAD: usize = 85;
    pub const FLUSH_HARD_SEMREL_ASYNC: usize = 86;
    pub const FLUSH_HARD_SEMREL_SYNC: usize = 87;
    pub const FLUSH_HARD_DMA: usize = 88;
    pub const FLUSH_HARD_FERMI: usize = 89;
    pub const FLUSH_HARD_KEPLER_MEMORY: usize = 90;
    pub const FLUSH_HARD_KEPLER_COMPUTE: usize = 91;
    pub const FERMI_DRAIN: usize = 92;
    pub const BARRIER_SEGMENT_FRAGMENT: usize = 93;
    pub const BARRIER_SEGMENT_TEXTURE_INVALIDATE: usize = 94;
    pub const BARRIER_TILED_NOOP: usize = 95;
    pub const FLUSH_ENQUEUE_RENDER_ENABLE: usize = 96;
    pub const FLUSH_ENQUEUE_DRAW_TEXTURE: usize = 97;
    pub const FLUSH_ENQUEUE_CLEAR: usize = 98;
    pub const FLUSH_ENQUEUE_RT_SIGNATURE: usize = 99;
    pub const FLUSH_ENQUEUE_RT_REBUILD: usize = 100;
    pub const FLUSH_ENQUEUE_CAPACITY: usize = 101;
    pub const FLUSH_ENQUEUE_PREP_FAIL: usize = 102;
    pub const RT_SIGNATURE_KNOWN_INDEPENDENT: usize = 103;
    pub const RT_SIGNATURE_ATTACHMENT_ALIAS: usize = 104;
    pub const RT_SIGNATURE_PRIOR_SAMPLE: usize = 105;
    pub const RT_SIGNATURE_NEXT_SAMPLE: usize = 106;
    pub const RT_SIGNATURE_SMALL_RT: usize = 107;
    pub const EPOCH_END: usize = 108;
    pub const ENTRY_PREP: usize = 109;
    pub const RESOLVE_TAIL: usize = 110;
    pub const KC_CACHE_CLEAR: usize = 111;
    pub const KM_CACHE_CLEAR: usize = 112;
    pub const ENQ_TEX_A: usize = 113;
    pub const ENQ_TEX_B: usize = 114;
    pub const ENQ_TEX_C: usize = 115;
    pub const ENQ_TEX_D: usize = 116;
    pub const KC_SNAP: usize = 117;
    pub const KC_DISPATCH: usize = 118;
    pub const FERMI_DRAIN_SKIPPED: usize = 119;
    pub const CBUF_PACK_MEMO: usize = 120;
    pub const CBUF_SLOT_STATE_HIT: usize = 121;
    pub const MIRROR_HIT: usize = 122;
    pub const MIRROR_SYNC: usize = 123;
    pub const MIRROR_STRADDLE: usize = 124;
    pub const MIRROR_EVICT: usize = 125;
    pub const MIRROR_PATCH: usize = 126;
    pub const RESIDENT_CBUF_DRAW: usize = 127;
    pub const TMPL_PROBE: usize = 128;
    pub const TMPL_L0: usize = 129;
    pub const TMPL_LRU: usize = 130;
    pub const TMPL_OBS_BUNDLE: usize = 131;
    pub const TMPL_OBS_LAYOUT: usize = 132;
    pub const TMPL_OBS_TEX: usize = 133;
    pub const TMPL_MISS: usize = 134;
    pub const ENQPRE_RT: usize = 135;
    pub const ENQPRE_SPH: usize = 136;
    pub const ENQPRE_IND: usize = 137;
    pub const ENQPRE_TEXLAY: usize = 138;
    pub const ENQPRE_BUNDLE: usize = 139;
    pub const PRES_WB: usize = 140;
    pub const PRES_SUBMIT: usize = 141;
    pub const PRES_WB_FLUSH: usize = 142;
    pub const PRES_WB_READ: usize = 143;
    pub const PRES_WB_POST: usize = 144;
    pub const VKF_CUBE_SCAN: usize = 146;
    pub const VKF_CUBE_WB: usize = 147;
    pub const M3D_PASSIVE_BULK: usize = 148;
    pub const M3D_PASSIVE_BULK_RUN: usize = 149;
    pub const M3D_PASSIVE_COLLAPSED: usize = 150;
    pub const CENSUS_SPH_STABLE: usize = 151;
    pub const CENSUS_SPH_PROBE: usize = 152;
    pub const CENSUS_TEXLAY_STABLE: usize = 153;
    pub const CENSUS_TEXLAY_PROBE: usize = 154;
    pub const CENSUS_TEXLAY_UNCACHEABLE: usize = 155;
    pub const CENSUS_PLAN_STABLE: usize = 156;
    pub const CENSUS_PLAN_PROBE: usize = 157;
    pub const CENSUS_CBBIND: usize = 158;
    pub const CENSUS_FLUSH_CHUNKS: usize = 159;
    pub const CENSUS_FLUSH_ONE_CHUNK: usize = 160;
    pub const SPH_MEMO_HIT: usize = 161;
    pub const SPH_MEMO_MISS: usize = 162;
    pub const CENSUS_TEXLAY2_STABLE: usize = 163;
    pub const CENSUS_TEXLAY2_PROBE: usize = 164;
    pub const TEXLAY_MEMO_HIT: usize = 165;
    pub const TEXLAY_MEMO_MISS: usize = 166;
    pub const TEXLAY_MEMO_FALLBACK: usize = 167;
    pub const TEXLAY_MEMO_UNCACHEABLE: usize = 168;
    pub const CENSUS_PLAN_LAYOUT_STABLE: usize = 169;
    pub const CENSUS_PLAN_LAYOUT_PROBE: usize = 170;
    pub const CBUF_ARENA_MEMO_HIT: usize = 171;
    pub const CBUF_ARENA_MEMO_MISS: usize = 172;
    pub const CBUF_BARRIER_SYNCS: usize = 173;
    pub const CBUF_BARRIER_CHUNKS: usize = 174;
    pub const CBUF_WINDOW_HIT: usize = 175;
    pub const CBUF_WINDOW_MISS: usize = 176;
    pub const KC_FRONTEND: usize = 177;
    pub const KC_RESOURCES: usize = 178;
    pub const VERTEX_STORE_WAIT: usize = 179;
    pub const COUNT: usize = 180;

    const NAMES: [&str; COUNT] = [
        "locks",
        "elist",
        "pbread",
        "m3d",
        "macro",
        "cbufwb",
        "semacq",
        "semrel",
        "barrier",
        "enq",
        "flushp",
        "dma",
        "fermi",
        "kepler",
        "puller",
        "smallrt",
        "kmflush",
        "kcuflush",
        "kclaunch",
        "kcsync",
        "kcexec",
        "kcwb",
        "kcresolve",
        "dmacopy",
        "dmafallback",
        "dmastage",
        "dmamap",
        "dmameta",
        "dmartlinear",
        "dmavirtual",
        "enqwatch",
        "enqbuild",
        "vkflush",
        "hostdraw",
        "drawinst",
        "enqpre",
        "enqtex",
        "enqcbuf",
        "enqindex",
        "enqstate",
        "enqssbo",
        "enqfinal",
        "vkfcube",
        "vkfinval",
        "vkfprep",
        "vkfsubmit",
        "cbslots",
        "cbww",
        "cbmissrd",
        "cbinval",
        "cbpack",
        "cbhitlast",
        "cbhitentry",
        "cbhitcover",
        "cbmiss",
        "cbmissdirty",
        "cbmissgrow",
        "prepread",
        "kickinval",
        "kickinvaln",
        "gpuinval",
        "gpuinvaln",
        "cbmisscold",
        "prepcbuf",
        "prepvtx",
        "preptic",
        "preptex",
        "preptsc",
        "preptexhit",
        "preptexmiss",
        "preptexstale",
        "preptexevict",
        "preptexrt",
        "preptexcube",
        "preptexother",
        "cballoc",
        "cbcopy",
        "cbhitrecent",
        "preptexfermi",
        "preptexfermraw",
        "preptexfermlease",
        "flushsoftbar",
        "flushsofttic",
        "flushhardtail",
        "flushhardpull",
        "flushhardinline",
        "flushhardsemasync",
        "flushhardsemsync",
        "flushharddma",
        "flushhardfermi",
        "flushhardkmem",
        "flushhardkcomp",
        "fermidrain",
        "barsegfrag",
        "barsegtic",
        "barriertiled",
        "flushrenable",
        "flushdrawtex",
        "flushclear",
        "flushrtsig",
        "flushrtrebuild",
        "flushcap",
        "flushprepfail",
        "rtsigindep",
        "rtsigattach",
        "rtsigprsample",
        "rtsignextsample",
        "rtsigsmallrt",
        "epochend",
        "entryprep",
        "resolvetail",
        "kcclear",
        "kmclear",
        "enqtexa",
        "enqtexb",
        "enqtexc",
        "enqtexd",
        "kcsnap",
        "kcdispatch",
        "fermidrainskip",
        "cbpackmemo",
        "cbslothit",
        "mirhit",
        "mirsync",
        "mirstraddle",
        "mirevict",
        "mirpatch",
        "rescbuf",
        "tmplprobe",
        "tmpll0",
        "tmpllru",
        "tmplobsbund",
        "tmplobslay",
        "tmplobstex",
        "tmplmiss",
        "enqprert",
        "enqpresph",
        "enqpreind",
        "enqpretexlay",
        "enqprebundle",
        "preswb",
        "pressubmit",
        "preswbflush",
        "preswbread",
        "preswbpost",
        "preptexfan",
        "vkfcubescan",
        "vkfcubewb",
        "m3dpbulk",
        "m3dpbulkrun",
        "m3dpbulkcollapse",
        "censsph",
        "censsphn",
        "censtexlay",
        "censtexlayn",
        "censtexlayunc",
        "censplan",
        "censplann",
        "censcbbind",
        "censflushchunks",
        "censflushone",
        "sphmemohit",
        "sphmemomiss",
        "censtexlay2",
        "censtexlay2n",
        "texlaymemohit",
        "texlaymemomiss",
        "texlaymemofb",
        "texlaymemounc",
        "censplanlayout",
        "censplanlayoutn",
        "cbarenamemohit",
        "cbarenamemomiss",
        "cbbarriersyncs",
        "cbbarrierchunks",
        "cbwindowhit",
        "cbwindowmiss",
        "kcfrontend",
        "kcresources",
        "vswait",
    ];

    static NS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
    static CALLS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
    static BYTES: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    static KICKS: AtomicU64 = AtomicU64::new(0);
    static WINDOW_KICKS: AtomicU64 = AtomicU64::new(0);
    static BLOCKED_NS: AtomicU64 = AtomicU64::new(0);
    static BLOCKED_CALLS: AtomicU64 = AtomicU64::new(0);
    static RENDER_BUSY_NS: AtomicU64 = AtomicU64::new(0);
    static RENDER_GROUPS: AtomicU64 = AtomicU64::new(0);
    static RENDER_JOB_NS: AtomicU64 = AtomicU64::new(0);
    static WINDOW_WALL_START_NS: AtomicU64 = AtomicU64::new(0);

    fn process_epoch() -> Instant {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    fn now_ns() -> u64 {
        process_epoch().elapsed().as_nanos() as u64
    }

    pub fn rate_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| enabled() || std::env::var_os("NEXIUM_KICK_RATE_PROFILE").is_some())
    }

    #[inline]
    pub fn rate_start() -> Option<Instant> {
        rate_enabled().then(Instant::now)
    }

    #[inline]
    pub fn add_blocked(started: Option<Instant>) {
        if let Some(started) = started {
            BLOCKED_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            BLOCKED_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn add_render_busy(started: Option<Instant>, groups: u64) {
        if let Some(started) = started {
            RENDER_BUSY_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            RENDER_GROUPS.fetch_add(groups, Ordering::Relaxed);
        }
    }

    static JOB_LABEL_NS: std::sync::Mutex<Vec<(&'static str, u64, u64)>> =
        std::sync::Mutex::new(Vec::new());

    #[inline]
    pub fn add_render_job(started: Option<Instant>, label: &'static str) {
        if let Some(started) = started {
            let ns = started.elapsed().as_nanos() as u64;
            RENDER_JOB_NS.fetch_add(ns, Ordering::Relaxed);
            let mut labels = JOB_LABEL_NS.lock().unwrap_or_else(|e| e.into_inner());
            match labels.iter_mut().find(|entry| entry.0 == label) {
                Some(entry) => {
                    entry.1 += ns;
                    entry.2 += 1;
                }
                None => labels.push((label, ns, 1)),
            }
        }
    }

    fn take_job_labels(kicks: f64) -> String {
        let mut labels =
            std::mem::take(&mut *JOB_LABEL_NS.lock().unwrap_or_else(|e| e.into_inner()));
        labels.sort_by(|a, b| b.1.cmp(&a.1));
        labels
            .iter()
            .take(6)
            .map(|(label, ns, n)| {
                format!(
                    " {}={:.2}ms/n{:.1}",
                    label,
                    *ns as f64 / kicks / 1_000_000.0,
                    *n as f64 / kicks
                )
            })
            .collect()
    }

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("NEXIUM_KICKOFF_PROFILE").is_some()
                || std::env::var_os("NEXIUM_KICK_STAGE_PROFILE").is_some();
            if on {
                log::warn!("[kickprof] armed");
            }
            on
        })
    }

    pub fn census_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("NEXIUM_MEMO_CENSUS")
                .ok()
                .is_some_and(|value| {
                    let value = value.trim();
                    value == "1"
                        || value.eq_ignore_ascii_case("true")
                        || value.eq_ignore_ascii_case("on")
                        || value.eq_ignore_ascii_case("yes")
                })
        })
    }

    #[inline]
    pub fn start() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    #[inline]
    pub fn kick_start() -> Option<Instant> {
        rate_start()
    }

    #[inline]
    pub fn add(phase: usize, started: Option<Instant>) {
        add_counted(phase, started, 1);
    }

    #[inline]
    pub fn add_sized(phase: usize, started: Option<Instant>, bytes: usize) {
        add_counted_sized(phase, started, 1, bytes);
    }

    #[inline]
    pub fn add_counted(phase: usize, started: Option<Instant>, calls: u64) {
        add_counted_sized(phase, started, calls, 0);
    }

    #[inline]
    pub fn add_counted_sized(phase: usize, started: Option<Instant>, calls: u64, bytes: usize) {
        let Some(started) = started else {
            return;
        };
        NS[phase].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[phase].fetch_add(calls, Ordering::Relaxed);
        BYTES[phase].fetch_add(bytes as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn count(phase: usize, amount: u64) {
        if enabled() {
            CALLS[phase].fetch_add(amount, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn count_sized(phase: usize, amount: u64, bytes: usize) {
        if enabled() {
            CALLS[phase].fetch_add(amount, Ordering::Relaxed);
            BYTES[phase].fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    pub fn kick_done(started: Option<Instant>) {
        let Some(started) = started else {
            return;
        };
        TOTAL_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let total_kicks = KICKS.fetch_add(1, Ordering::Relaxed) + 1;
        let window = WINDOW_KICKS.fetch_add(1, Ordering::Relaxed) + 1;
        if window < 64 {
            return;
        }
        WINDOW_KICKS.store(0, Ordering::Relaxed);
        let total = TOTAL_NS.swap(0, Ordering::Relaxed).max(1);
        let kicks = window as f64;
        let wall_now = now_ns();
        let wall_start = WINDOW_WALL_START_NS.swap(wall_now, Ordering::Relaxed);
        let wall = wall_now.saturating_sub(wall_start);
        let blocked = BLOCKED_NS.swap(0, Ordering::Relaxed);
        let blocked_calls = BLOCKED_CALLS.swap(0, Ordering::Relaxed);
        let render_busy = RENDER_BUSY_NS.swap(0, Ordering::Relaxed);
        let render_groups = RENDER_GROUPS.swap(0, Ordering::Relaxed);
        let render_job = RENDER_JOB_NS.swap(0, Ordering::Relaxed);
        let job_labels = take_job_labels(kicks);
        let mut accounted = 0u64;
        let mut parts = String::new();
        let mut raw_texture_snapshots = 0u64;
        let mut raw_texture_snapshot_bytes = 0u64;
        for i in 0..COUNT {
            let ns = NS[i].swap(0, Ordering::Relaxed);
            let n = CALLS[i].swap(0, Ordering::Relaxed);
            let bytes = BYTES[i].swap(0, Ordering::Relaxed);
            if i == PREP_TEXTURE {
                raw_texture_snapshots = n;
                raw_texture_snapshot_bytes = bytes;
            }
            if i < DISJOINT {
                accounted += ns;
            }
            if ns == 0 && n == 0 {
                continue;
            }
            parts.push_str(&format!(
                " {}={:.2}ms/{:.0}%/n{}{}",
                NAMES[i],
                ns as f64 / kicks / 1_000_000.0,
                ns as f64 * 100.0 / total as f64,
                n,
                if bytes == 0 {
                    String::new()
                } else {
                    format!("/{:.1}MiB", bytes as f64 / (1024.0 * 1024.0))
                }
            ));
        }
        let rt_alias = nexium_gpu::renderer::take_rt_alias_telemetry();
        parts.push_str(&format!(
            " texraw=n{}/{}MiB rtaliasbind=n{} rtaliasfromraw=n{}",
            raw_texture_snapshots,
            raw_texture_snapshot_bytes as f64 / (1024.0 * 1024.0),
            rt_alias.binds,
            rt_alias.binds_after_raw_snapshot,
        ));
        let other = total.saturating_sub(accounted);
        log::warn!(
            "[kickprof] kicks={} (window {}) avg_ms total={:.2} other={:.2}/{:.0}% |{}",
            total_kicks,
            window,
            total as f64 / kicks / 1_000_000.0,
            other as f64 / kicks / 1_000_000.0,
            other as f64 * 100.0 / total as f64,
            parts
        );
        log::warn!(
            "[kickwall] kicks={} wall_ms/kick={:.2} pusher_duty={:.0}% blocked_ms/kick={:.2} blocked_calls/kick={:.1} render_draw_ms/kick={:.2} render_groups/kick={:.1} render_job_ms/kick={:.2} render_share={:.0}%",
            total_kicks,
            wall as f64 / kicks / 1_000_000.0,
            total as f64 * 100.0 / wall.max(1) as f64,
            blocked as f64 / kicks / 1_000_000.0,
            blocked_calls as f64 / kicks,
            render_busy as f64 / kicks / 1_000_000.0,
            render_groups as f64 / kicks,
            render_job as f64 / kicks / 1_000_000.0,
            (render_busy + render_job) as f64 * 100.0 / wall.max(1) as f64,
        );
        log::warn!("[kickjobs] kicks={} per-kick |{}", total_kicks, job_labels);
    }
}

static UNMAPPED_PB_WARNS: AtomicU64 = AtomicU64::new(0);

fn warn_unmapped_pushbuffer(kind: &str, gpu_va: u64, mappings: &GpuMappings) {
    let n = UNMAPPED_PB_WARNS.fetch_add(1, Ordering::Relaxed);
    if n < 64 || n % 4096 == 0 {
        log::warn!(
            "pusher: {} gpu_va={:#x} not in GMMU — skipping (#{}) {}",
            kind,
            gpu_va,
            n,
            mappings.bracket(gpu_va)
        );
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CommandListHeader {
    pub address_lo: u32,
    pub address_hi_and_count: u32,
}

impl CommandListHeader {
    pub fn address(&self) -> u64 {
        ((self.address_hi_and_count as u64 & 0xFF) << 32) | (self.address_lo as u64 & 0xFFFF_FFFC)
    }

    pub fn entry_count(&self) -> u32 {
        (self.address_hi_and_count >> 10) & 0x1F_FFFF
    }

    pub fn no_prefetch(&self) -> bool {
        (self.address_hi_and_count & 0x8000_0000) != 0
    }

    pub fn not_main(&self) -> bool {
        (self.address_hi_and_count & 0x200) != 0
    }
}

pub(crate) fn decode_command_list_headers(bytes: &[u8]) -> Vec<CommandListHeader> {
    bytes
        .chunks_exact(8)
        .map(|chunk| CommandListHeader {
            address_lo: u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            address_hi_and_count: u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
        })
        .collect()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Increasing,
    NonIncreasing,
    Inline,
    IncreaseOnce,
}

impl Mode {
    fn from_bits(v: u32) -> Option<Mode> {
        Some(match v {
            1 => Mode::Increasing,
            3 => Mode::NonIncreasing,
            4 => Mode::Inline,
            5 => Mode::IncreaseOnce,
            _ => return None,
        })
    }
}

const METHOD_BIND_OBJECT: u32 = 0x00;
const METHOD_SEMAPHORE_ADDR_HIGH: u32 = 0x04;
const METHOD_SEMAPHORE_ADDR_LOW: u32 = 0x05;
const METHOD_SEMAPHORE_PAYLOAD: u32 = 0x06;
const METHOD_SEMAPHORE_OPERATION: u32 = 0x07;
const METHOD_SEMAPHORE_ACQUIRE: u32 = 0x1A;
const METHOD_SEMAPHORE_RELEASE: u32 = 0x1B;
const METHOD_SYNCPOINT_PAYLOAD: u32 = 0x1C;
const METHOD_SYNCPOINT_OPERATION: u32 = 0x1D;
const NON_PULLER_METHODS: u32 = 0x40;

const POISON_SENTINEL: u32 = 0xBEEF_2929;

pub(crate) static GPU_SEM_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(crate) fn gpu_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_NVDRV_PROFILE").is_some())
}

fn m3d_passive_bulk_value_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1"
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("on")
            || value.eq_ignore_ascii_case("yes")
    })
}

fn m3d_passive_bulk_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        m3d_passive_bulk_value_enabled(std::env::var("NEXIUM_M3D_PASSIVE_BULK").ok().as_deref())
    })
}

fn semacq_ack_stat(elapsed: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_PREP_PROFILE").is_some()) {
        return;
    }
    static NS: AtomicU64 = AtomicU64::new(0);
    static MAX_NS: AtomicU64 = AtomicU64::new(0);
    static N: AtomicU64 = AtomicU64::new(0);
    let ns = elapsed.as_nanos() as u64;
    NS.fetch_add(ns, Ordering::Relaxed);
    MAX_NS.fetch_max(ns, Ordering::Relaxed);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 512 == 0 {
        let total = NS.swap(0, Ordering::Relaxed);
        let max = MAX_NS.swap(0, Ordering::Relaxed);
        log::warn!(
            "[semacq-ack] waits={} window_avg_ms={:.2} window_max_ms={:.2}",
            n,
            total as f64 / 512.0 / 1_000_000.0,
            max as f64 / 1_000_000.0
        );
    }
}

pub(crate) fn semrel_legacy() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_SEMREL_TIMELINE").ok().as_deref(),
            Some("0") | Some("off") | Some("OFF") | Some("false") | Some("FALSE")
        )
    })
}

pub(crate) fn semrel_verify() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_SEMREL_VERIFY").is_some())
}

pub(crate) fn write_payload_fences(
    writes: &[(u64, u32)],
    mut write_gpu: impl FnMut(u64, &[u8]) -> Option<(u64, bool)>,
) {
    for &(gpu_va, payload) in writes {
        match write_gpu(gpu_va, &payload.to_le_bytes()) {
            Some((cpu, ok)) => {
                if std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some() {
                    log::info!(
                        "[syncpt] async fence release gpu_va={:#x} cpu={:#x} payload={:#x} write_ok={}",
                        gpu_va,
                        cpu,
                        payload,
                        ok
                    );
                }
            }
            None => log::warn!(
                "pusher: async fence release gpu_va={:#x} not mapped; payload={:#x} dropped",
                gpu_va,
                payload
            ),
        }
    }
}

pub(crate) fn profile_method_if_slow(
    started: Option<std::time::Instant>,
    entry_gpu_va: u64,
    word_index: usize,
    bound_class: u32,
    method: u32,
) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed();
    if elapsed >= Duration::from_millis(1) {
        log::warn!(
            "[nvprof] method entry={:#x} word={} class={:#x} method={:#x} elapsed_ms={:.3}",
            entry_gpu_va,
            word_index,
            bound_class,
            method,
            elapsed.as_secs_f64() * 1000.0,
        );
    }
}

#[derive(Default)]
struct DmaState {
    method: u32,
    subchannel: u32,
    method_count: u32,
    non_incrementing: bool,
    increment_once: bool,
}

#[derive(Default)]
struct PullerState {
    semaphore_addr_high: u32,
    semaphore_addr_low: u32,
    semaphore_payload: u32,
    syncpoint_payload: u32,
}

pub struct Pusher {
    pub syncpt_value: u32,
    pub(crate) pending_syncpt_incrs: Vec<(u32, u32)>,
    bound_classes: [u32; 8],
    state: DmaState,
    puller: PullerState,
    entries_logged: u32,
    active_entry_gpu_va: u64,
    active_entry_cpu_va: u64,
    active_word_index: usize,
    active_header: u32,
    pub(crate) prep: super::prep::PrepLane,
    engine_event_batch: Option<(u32, Vec<(u32, u32, bool)>)>,
    live_macro_scratch: Vec<u8>,
    live_macro_values: Vec<u32>,
    entry_bytes_scratch: Vec<u8>,
    anomaly_ring: std::collections::VecDeque<PbAnomalyEntry>,
    anomaly_dumped: bool,
    gpu_profile_on: bool,
    entry_words_scratch: Vec<u32>,
    #[cfg(test)]
    passive_bulk_override: Option<bool>,
    #[cfg(test)]
    passive_bulk_words: usize,
}

impl Pusher {
    pub fn new() -> Self {
        Self {
            syncpt_value: 0,
            pending_syncpt_incrs: Vec::new(),
            bound_classes: [
                0xB197,
                KEPLER_COMPUTE_CLASS,
                KEPLER_MEMORY_CLASS,
                FERMI_2D_CLASS,
                MAXWELL_DMA_CLASS,
                0,
                0,
                0,
            ],
            state: DmaState::default(),
            puller: PullerState::default(),
            entries_logged: 0,
            active_entry_gpu_va: 0,
            active_entry_cpu_va: 0,
            active_word_index: 0,
            active_header: 0,
            prep: super::prep::PrepLane::Inline(super::prep::PrepState::new()),
            engine_event_batch: None,
            live_macro_scratch: Vec::new(),
            live_macro_values: Vec::new(),
            entry_bytes_scratch: Vec::new(),
            anomaly_ring: std::collections::VecDeque::new(),
            anomaly_dumped: false,
            gpu_profile_on: gpu_profile_enabled(),
            entry_words_scratch: Vec::new(),
            #[cfg(test)]
            passive_bulk_override: None,
            #[cfg(test)]
            passive_bulk_words: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn inline_prep(&mut self) -> &mut super::prep::PrepState {
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state,
            super::prep::PrepLane::Threaded(_) => {
                panic!("prep lane is threaded; inline prep state is unavailable")
            }
        }
    }

    pub fn set_renderer(&mut self, renderer: Option<Arc<nexium_gpu::Renderer>>) {
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state.set_renderer(renderer),
            super::prep::PrepLane::Threaded(handle) => {
                handle.send(super::prep::PrepEvent::SetRenderer(renderer))
            }
        }
    }

    pub(crate) fn set_guest_memory_access(&mut self, memory: Option<super::GuestMemoryAccess>) {
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state.set_guest_memory_access(memory),
            super::prep::PrepLane::Threaded(handle) => {
                handle.send(super::prep::PrepEvent::SetGuestMemory(memory))
            }
        }
    }

    fn flush_engine_event_batch(&mut self) {
        let Some((class, methods)) = self.engine_event_batch.take() else {
            return;
        };
        if let super::prep::PrepLane::Threaded(handle) = &mut self.prep {
            handle.send(super::prep::PrepEvent::EngineMethods { class, methods });
        }
    }

    fn emit_prep(
        &mut self,
        event: super::prep::PrepEvent,
        engines: &mut super::prep::PrepEngines,
        mappings: &GpuMappings,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) -> bool {
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state.run_event(
                event, None, engines, mappings, stats, mem_read, mem_write, mem_copy,
            ),
            super::prep::PrepLane::Threaded(handle) => {
                handle.send(event);
                true
            }
        }
    }

    pub(crate) fn prep_kick_begin(&mut self) {
        self.flush_engine_event_batch();
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state.begin_ssbo_snapshot_epoch(),
            super::prep::PrepLane::Threaded(handle) => {
                handle.begin_kick();
                if handle
                    .send_recover(super::prep::PrepEvent::KickBegin)
                    .is_err()
                {
                    handle.cancel_kick();
                    log::error!("[gpu-prep] kick-begin dropped after prep thread exit");
                }
            }
        }
    }

    fn prep_entry_begin(&mut self) {
        self.flush_engine_event_batch();
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => state.begin_ssbo_snapshot_entry(),
            super::prep::PrepLane::Threaded(handle) => {
                handle.send(super::prep::PrepEvent::EntryBegin)
            }
        }
    }

    fn prep_hard_flush(
        &mut self,
        reason: usize,
        clear_ssbo: bool,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.flush_engine_event_batch();
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => {
                state.record_flush_reason(reason);
                state.flush_vk(mappings, mem_read, mem_write);
                if clear_ssbo {
                    state.ssbo_snapshot_cache.clear_ssbo_snapshots();
                }
            }
            super::prep::PrepLane::Threaded(handle) => {
                handle.send(super::prep::PrepEvent::HardFlush { reason, clear_ssbo })
            }
        }
    }

    pub(crate) fn prep_kick_end(
        &mut self,
        hard_after: bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        self.flush_engine_event_batch();
        match &mut self.prep {
            super::prep::PrepLane::Inline(state) => {
                let kp_tail = kickprof::start();
                state.resolve_pending_compute(mappings, mem_write);
                kickprof::add(kickprof::RESOLVE_TAIL, kp_tail);
                if hard_after {
                    state.record_flush_reason(kickprof::FLUSH_HARD_TAIL);
                }
                state.flush_vk_with_boundary(mappings, mem_read, mem_write, hard_after);
                let submitted = state.finish_prepared_draw_packet_tail(hard_after);
                if writeback_small_rts {
                    if let Some(r) = state.renderer.clone() {
                        let kp = kickprof::start();
                        state.writeback_small_rts(&r, mappings, mem_write);
                        kickprof::add(kickprof::SMALLRT, kp);
                    }
                }
                if !writeback_small_rts
                    && super::vk_dispatch::cpu_readable_rt_writeback_mode()
                        == super::vk_dispatch::CpuReadableRtWritebackMode::Kick
                    && super::vk_dispatch::has_pending_cpu_readable_rt_writebacks()
                {
                    if let Some(r) = state.renderer.clone() {
                        let kp = kickprof::start();
                        state.writeback_cpu_readable_rts(&r, mappings, mem_write);
                        kickprof::add(kickprof::SMALLRT, kp);
                    }
                }
                state.end_ssbo_snapshot_epoch();
                if submitted {
                    state.schedule_kick_completion(on_complete);
                }
            }
            super::prep::PrepLane::Threaded(handle) => {
                match handle.send_recover(super::prep::PrepEvent::KickEnd {
                    hard_after,
                    writeback_small_rts,
                    on_complete,
                }) {
                    Ok(()) => {}
                    Err(super::prep::PrepEvent::KickEnd { on_complete, .. }) => {
                        handle.cancel_kick();
                        log::error!("[gpu-prep] kick-end dropped after prep thread exit");
                        drop(on_complete);
                    }
                    Err(_) => unreachable!("prep kick-end returned a different event"),
                }
            }
        }
    }

    pub fn process_gpfifo(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.process_gpfifo_with_boundary(
            address,
            num_entries,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_compute,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
            mem_copy,
            true,
            writeback_small_rts,
            on_complete,
        );
    }

    pub fn process_gpfifo_soft(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.process_gpfifo_with_boundary(
            address,
            num_entries,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_compute,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
            mem_copy,
            false,
            true,
            on_complete,
        );
    }

    pub fn process_gpfifo_soft_deferred(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.process_gpfifo_with_boundary(
            address,
            num_entries,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_compute,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
            mem_copy,
            false,
            false,
            on_complete,
        );
    }

    fn process_gpfifo_with_boundary(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
        hard_after: bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.prep_kick_begin();
        let cpu_addr = match mappings.cpu_address_for(address) {
            Some(c) => c,
            None => {
                warn_unmapped_pushbuffer("GPFIFO entry list", address, mappings);
                self.prep_kick_end(
                    hard_after,
                    false,
                    on_complete,
                    mappings,
                    mem_read,
                    mem_write,
                );
                return;
            }
        };

        let bytes_needed = (num_entries as usize) * 8;
        if direct_forensics() {
            let remaining = mappings
                .cpu_range_for(address)
                .map(|(_, sz)| sz)
                .unwrap_or(0);
            if (bytes_needed as u64) > remaining {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 32 {
                    log::warn!(
                        "[el-overrun] entry_list gpu_va={:#x} num_entries={} need={:#x} remaining={:#x}",
                        address,
                        num_entries,
                        bytes_needed,
                        remaining
                    );
                }
            }
        }
        let _ = cpu_addr;
        let kp_elist = kickprof::start();
        let mut buf = vec![0u8; bytes_needed];
        read_gpu_scattered(mappings, address, &mut buf, mem_read);

        let decoded = decode_command_list_headers(&buf);
        record_gpfifo_entries(&decoded, mappings);
        kickprof::add(kickprof::ELIST, kp_elist);

        if direct_forensics() && decoded.iter().any(|e| e.entry_count() > 4096) {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let raw: Vec<String> = decoded
                    .iter()
                    .map(|e| {
                        format!(
                            "{:08x}:{:08x}(va={:#x},n={})",
                            e.address_lo,
                            e.address_hi_and_count,
                            e.address(),
                            e.entry_count()
                        )
                    })
                    .collect();
                log::warn!(
                    "[el-dump] list_va={:#x} num_entries={} {}",
                    address,
                    num_entries,
                    raw.join(" ")
                );
            }
        }

        for entry in &decoded {
            self.process_entry(
                entry,
                mappings,
                maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                stats,
                mem_read,
                mem_write,
                mem_copy,
            );
        }
        self.prep_kick_end(
            hard_after,
            writeback_small_rts,
            on_complete,
            mappings,
            mem_read,
            mem_write,
        );
    }

    pub fn process_entry(
        &mut self,
        entry: &CommandListHeader,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        self.prep_entry_begin();
        let address = entry.address();
        let word_count = entry.entry_count();
        super::watchdog::phase(super::watchdog::Phase::Entry, address);
        let profile = gpu_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let methods_before = profile.then(|| stats.methods_dispatched.load(Ordering::Relaxed));

        if gpfifo_trace_enabled() && self.entries_logged < 16 {
            log::info!(
                "gpfifo[{}]: gpu_va={:#x} word_count={} no_prefetch={} not_main={} raw_lo={:#010x} raw_hi={:#010x}",
                self.entries_logged,
                address,
                word_count,
                entry.no_prefetch(),
                entry.not_main(),
                entry.address_lo,
                entry.address_hi_and_count
            );
            self.entries_logged += 1;
        }

        if word_count == 0 {
            return;
        }

        let cpu_addr = match mappings.cpu_address_for(address) {
            Some(c) => c,
            None => {
                warn_unmapped_pushbuffer("pushbuffer", address, mappings);
                self.state.method_count = self.state.method_count.saturating_sub(word_count);
                return;
            }
        };

        let bytes_needed = (word_count as usize) * 4;
        if direct_forensics() {
            let remaining = mappings
                .cpu_range_for(address)
                .map(|(_, sz)| sz)
                .unwrap_or(0);
            if (bytes_needed as u64) > remaining {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 64 {
                    log::warn!(
                        "[pb-overrun] gpu_va={:#x} cpu={:#x} need={:#x} remaining_in_mapping={:#x} {}",
                        address,
                        cpu_addr,
                        bytes_needed,
                        remaining,
                        mappings.bracket(address)
                    );
                }
            }
            if word_count > 16384 {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 48 {
                    log::warn!(
                        "[pb-bogus-entry] gpu_va={:#x} word_count={} raw_lo={:#010x} raw_hi={:#010x} no_prefetch={} not_main={}",
                        address,
                        word_count,
                        entry.address_lo,
                        entry.address_hi_and_count,
                        entry.no_prefetch(),
                        entry.not_main()
                    );
                }
            }
        }
        let _ = cpu_addr;
        let kp_pb = kickprof::start();
        let mut buf = std::mem::take(&mut self.entry_bytes_scratch);
        buf.clear();
        buf.resize(bytes_needed, 0);
        read_gpu_scattered(mappings, address, &mut buf, mem_read);

        let mut words = std::mem::take(&mut self.entry_words_scratch);
        words.clear();
        words.reserve(word_count as usize);
        for i in 0..word_count as usize {
            let off = i * 4;
            let w = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            words.push(if w == POISON_SENTINEL { 0 } else { w });
        }
        self.entry_bytes_scratch = buf;
        kickprof::add(kickprof::PBREAD, kp_pb);

        self.active_entry_gpu_va = address;
        self.active_entry_cpu_va = cpu_addr;
        let entry_state_in = (
            self.state.method,
            self.state.subchannel,
            self.state.method_count,
            self.state.non_incrementing,
            self.state.increment_once,
        );
        if pb_anomaly_dump_enabled() {
            if self.anomaly_ring.len() >= PB_ANOMALY_RING_ENTRIES {
                self.anomaly_ring.pop_front();
            }
            self.anomaly_ring.push_back(PbAnomalyEntry {
                gpu_va: address,
                state_in: entry_state_in,
                total_words: words.len(),
                words: words[..words.len().min(PB_ANOMALY_RING_WORDS)].to_vec(),
            });
        }
        let frame_trace = pb_frame_trace_enabled();
        if frame_trace {
            log::warn!(
                "[pb-frame-in] gpu_va={:#x} words={} state_in={:?} w={:08x?}",
                address,
                word_count,
                entry_state_in,
                &words[..words.len().min(64)]
            );
        }
        self.process_commands(
            &words,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_compute,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
            mem_copy,
        );
        if frame_trace {
            log::warn!(
                "[pb-frame-out] gpu_va={:#x} state_out=({:#x},{},{},{},{})",
                address,
                self.state.method,
                self.state.subchannel,
                self.state.method_count,
                self.state.non_incrementing,
                self.state.increment_once
            );
            let mut again = vec![0u8; bytes_needed];
            read_gpu_scattered(mappings, address, &mut again, mem_read);
            let reread: Vec<u32> = (0..word_count as usize)
                .map(|i| {
                    let off = i * 4;
                    u32::from_le_bytes([again[off], again[off + 1], again[off + 2], again[off + 3]])
                })
                .map(|w| if w == POISON_SENTINEL { 0 } else { w })
                .collect();
            if reread != words {
                let diffs: Vec<String> = words
                    .iter()
                    .zip(reread.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, (a, b))| format!("{}:{:08x}->{:08x}", i, a, b))
                    .collect();
                log::error!(
                    "[pb-torn] gpu_va={:#x} words={} changed={} {}",
                    address,
                    word_count,
                    diffs.len(),
                    diffs.join(" ")
                );
            }
        }
        if direct_forensics() && self.state.method_count > 0 {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 64 {
                log::warn!(
                    "[pb-leak] entry_gpu={:#x} words={} ended with method={:#x} count_left={} noninc={} subch={}",
                    address,
                    word_count,
                    self.state.method,
                    self.state.method_count,
                    self.state.non_incrementing,
                    self.state.subchannel
                );
                log::warn!("[pb-leak-maps] {}", mappings.describe_around(address));
                if let Some(n) = std::env::var("NEXIUM_PB_DUMP_WORDS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    log::warn!(
                        "[pb-leak-words] entry_gpu={:#x} entry_state_in={:?} words[0..{}]={:08x?}",
                        address,
                        entry_state_in,
                        n.min(words.len()),
                        &words[..n.min(words.len())]
                    );
                }
            }
        }
        self.active_entry_gpu_va = 0;
        self.active_entry_cpu_va = 0;
        self.active_word_index = 0;
        self.active_header = 0;
        if let (Some(started), Some(methods_before)) = (started, methods_before) {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(5) {
                let methods = stats
                    .methods_dispatched
                    .load(Ordering::Relaxed)
                    .saturating_sub(methods_before);
                log::warn!(
                    "[nvprof] entry gpu_va={:#x} words={} methods={} elapsed_ms={:.3}",
                    address,
                    word_count,
                    methods,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        self.entry_words_scratch = words;
    }

    fn process_commands(
        &mut self,
        commands: &[u32],
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        let mut i = 0;
        let mut methods_dispatched = 0u64;
        let constbuf_upload_watch_active = constbuf_upload_watch().is_some();
        let gpu_profile_active = gpu_profile_enabled();
        #[cfg(not(test))]
        let passive_bulk_active = m3d_passive_bulk_enabled();
        #[cfg(test)]
        let passive_bulk_active = self
            .passive_bulk_override
            .unwrap_or_else(m3d_passive_bulk_enabled);
        let live_macro_batch_active =
            mme_batch_refresh_enabled() && !gpu_profile_active && !direct_forensics();
        let live_macro_slice_active = live_macro_batch_active
            && mme_slice_dispatch_enabled()
            && !mme_param_trace()
            && !mme_dispatch_trace();
        let mut live_macro_scratch = std::mem::take(&mut self.live_macro_scratch);
        let direct_forensics_on = direct_forensics();
        let mut live_macro_values = std::mem::take(&mut self.live_macro_values);
        let mut live_macro_start = 0usize;
        let mut live_macro_end = 0usize;
        let mut live_macro_valid = false;
        let anomaly_dump_on = pb_anomaly_dump_enabled();
        while i < commands.len() {
            if anomaly_dump_on && !self.anomaly_dumped && PB_ANOMALY_PENDING.load(Ordering::Acquire)
            {
                self.dump_pb_anomaly(commands, i);
            }
            let header = commands[i];

            if self.state.method_count > 0 {
                self.active_word_index = i;
                let cls = self.bound_classes[self.state.subchannel as usize & 7];
                let passive_run = (cls == 0xB197
                    && (!gpu_profile_active || passive_bulk_active)
                    && !direct_forensics_on
                    && !self.state.increment_once)
                    .then(|| {
                        let available = (commands.len() - i).min(self.state.method_count as usize);
                        let method = self.state.method;
                        let count =
                            passive_maxwell_run_len(method, available, self.state.non_incrementing);
                        (method, count)
                    })
                    .filter(|(_, count)| *count != 0);
                let passive_drain_boundary =
                    passive_run.is_some() && maxwell.has_pending_pusher_work();
                if !passive_drain_boundary {
                    if let Some((method, count)) = passive_run {
                        let kp_m3d = kickprof::start();
                        if passive_bulk_active {
                            let kp_bulk = if gpu_profile_active {
                                kickprof::start()
                            } else {
                                None
                            };
                            let collapsed = maxwell.write_passive_register_run(
                                method,
                                &commands[i..i + count],
                                self.state.non_incrementing,
                            );
                            if gpu_profile_active {
                                kickprof::add_counted(
                                    kickprof::M3D_PASSIVE_BULK,
                                    kp_bulk,
                                    count as u64,
                                );
                                kickprof::count(kickprof::M3D_PASSIVE_BULK_RUN, 1);
                                kickprof::count(kickprof::M3D_PASSIVE_COLLAPSED, collapsed as u64);
                            }
                            #[cfg(test)]
                            {
                                self.passive_bulk_words += count;
                            }
                        } else {
                            for offset in 0..count {
                                let method = if self.state.non_incrementing {
                                    method
                                } else {
                                    method.wrapping_add(offset as u32)
                                };
                                maxwell.write_register(method, commands[i + offset]);
                                maxwell.record_method(method);
                            }
                        }
                        kickprof::add_counted(kickprof::M3D, kp_m3d, count as u64);
                        self.active_word_index = i + count - 1;
                        if !self.state.non_incrementing {
                            self.state.method = method.wrapping_add(count as u32);
                        }
                        self.state.method_count -= count as u32;
                        methods_dispatched += count as u64;
                        i += count;
                        continue;
                    }
                }
                if live_macro_slice_active
                    && cls == 0xB197
                    && self.state.method >= MACRO_REGISTERS_START
                    && self.state.non_incrementing
                    && self.state.method_count > 1
                    && !maxwell.has_pending_pusher_work()
                {
                    let available = (commands.len() - i).min(self.state.method_count as usize);
                    if i >= live_macro_end {
                        live_macro_start = i;
                        live_macro_end = i + 1;
                        live_macro_valid = false;
                        if available > 1 {
                            let arg_gpu_va = self
                                .active_entry_gpu_va
                                .checked_add((self.active_word_index as u64) * 4);
                            let kp = kickprof::start();
                            let (span, valid) = arg_gpu_va.map_or((1, false), |gpu_va| {
                                read_live_words(
                                    mappings,
                                    gpu_va,
                                    available,
                                    &mut live_macro_scratch,
                                    mem_read,
                                )
                            });
                            live_macro_end = i + span;
                            live_macro_valid = valid;
                            kickprof::add_counted_sized(
                                kickprof::MACRO,
                                kp,
                                if valid { span as u64 } else { 0 },
                                if valid { span * 4 } else { 0 },
                            );
                        }
                    }
                    if live_macro_valid {
                        let span = live_macro_end - i;
                        let count = span.min(self.state.method_count.saturating_sub(1) as usize);
                        if count != 0 {
                            let byte_start = (i - live_macro_start) * 4;
                            let byte_end = byte_start + count * 4;
                            live_macro_values.clear();
                            live_macro_values.extend(
                                live_macro_scratch[byte_start..byte_end]
                                    .chunks_exact(4)
                                    .map(|bytes| {
                                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                    }),
                            );
                            let kp_m3d = kickprof::start();
                            maxwell.dispatch_macro_methods(
                                self.state.method,
                                live_macro_values.as_slice(),
                                false,
                            );
                            kickprof::add_counted(kickprof::M3D, kp_m3d, count as u64);
                            maxwell.record_methods(self.state.method, count as u64);
                            self.active_word_index = i + count - 1;
                            self.state.method_count -= count as u32;
                            methods_dispatched += count as u64;
                            i += count;
                            continue;
                        }
                    }
                }
                if suspicious_direct(cls, self.state.method, header) {
                    dump_direct_ctx(
                        self.active_entry_gpu_va,
                        self.active_entry_cpu_va,
                        self.state.method,
                        header,
                        cls,
                        self.state.method_count,
                        self.state.non_incrementing,
                        commands,
                        i,
                    );
                }
                let defer_constbuf_writeback = !passive_drain_boundary
                    && should_defer_constbuf_writeback(
                        &self.state,
                        i + 1 < commands.len(),
                        constbuf_upload_watch_active,
                    );
                let arg_gpu_va = self
                    .active_entry_gpu_va
                    .checked_add((self.active_word_index as u64) * 4);
                let prefetched_macro_arg = if live_macro_batch_active
                    && cls == 0xB197
                    && self.state.method >= MACRO_REGISTERS_START
                    && self.state.non_incrementing
                {
                    if i >= live_macro_end {
                        let available = (commands.len() - i).min(self.state.method_count as usize);
                        live_macro_start = i;
                        live_macro_end = i + 1;
                        live_macro_valid = false;
                        if available > 1 {
                            let kp = kickprof::start();
                            let (span, valid) = arg_gpu_va.map_or((1, false), |gpu_va| {
                                read_live_words(
                                    mappings,
                                    gpu_va,
                                    available,
                                    &mut live_macro_scratch,
                                    mem_read,
                                )
                            });
                            live_macro_end = i + span;
                            live_macro_valid = valid;
                            kickprof::add_counted_sized(
                                kickprof::MACRO,
                                kp,
                                if live_macro_valid { span as u64 } else { 0 },
                                if live_macro_valid { span * 4 } else { 0 },
                            );
                        }
                    }
                    if live_macro_valid {
                        let offset = (i - live_macro_start) * 4;
                        let bytes = &live_macro_scratch[offset..offset + 4];
                        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    } else {
                        None
                    }
                } else {
                    None
                };
                self.dispatch_method(
                    header,
                    arg_gpu_va,
                    prefetched_macro_arg,
                    defer_constbuf_writeback,
                    mappings,
                    maxwell,
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
                methods_dispatched += 1;
                if !self.state.non_incrementing {
                    self.state.method = self.state.method.wrapping_add(1);
                }
                if self.state.increment_once {
                    self.state.non_incrementing = true;
                }
                self.state.method_count -= 1;
                i += 1;
                continue;
            }

            if header == 0 {
                i += 1;
                continue;
            }

            let method = header & 0x1FFF;
            let subchannel = (header >> 13) & 0x7;
            let arg_count = (header >> 16) & 0x1FFF;
            let mode_bits = (header >> 29) & 0x7;
            self.active_word_index = i;
            self.active_header = header;
            let Some(mode) = Mode::from_bits(mode_bits) else {
                static UNKNOWN_MODES: AtomicU64 = AtomicU64::new(0);
                if UNKNOWN_MODES.fetch_add(1, Ordering::Relaxed) < 64 {
                    log::warn!(
                        "[pb-unknown-secop] mode={} header={:#010x} entry_gpu={:#x} word={} of {} state_in_method={:#x} subch={} lo={:08x?}",
                        mode_bits,
                        header,
                        self.active_entry_gpu_va,
                        i,
                        commands.len(),
                        self.state.method,
                        self.state.subchannel,
                        &commands[i.saturating_sub(8)..(i + 8).min(commands.len())]
                    );
                }
                note_pb_anomaly("unknown-secop");
                i += 1;
                continue;
            };

            self.state.method = method;
            self.state.subchannel = subchannel;
            self.state.method_count = arg_count;

            if direct_forensics_on && arg_count > 512 && mode != Mode::Inline {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 64 {
                    let cls = self.bound_classes[subchannel as usize & 7];
                    log::warn!(
                        "[pb-bighdr] entry_gpu={:#x} word={} of {} header={:#010x} method={:#x} count={} mode={:?} subch={} class={:#x} {}",
                        self.active_entry_gpu_va,
                        i,
                        commands.len(),
                        header,
                        method,
                        arg_count,
                        mode,
                        subchannel,
                        cls,
                        mappings.bracket(self.active_entry_gpu_va)
                    );
                }
            }

            match mode {
                Mode::Increasing => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = false;
                }
                Mode::NonIncreasing => {
                    self.state.non_incrementing = true;
                    self.state.increment_once = false;
                }
                Mode::IncreaseOnce => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = true;
                }
                Mode::Inline => {
                    self.state.method_count = 0;
                    self.state.non_incrementing = true;
                    self.state.increment_once = false;
                    self.active_word_index = i;
                    self.dispatch_method(
                        arg_count,
                        None,
                        None,
                        false,
                        mappings,
                        maxwell,
                        maxwell_dma,
                        fermi_2d,
                        kepler_compute,
                        kepler_memory,
                        stats,
                        mem_read,
                        mem_write,
                        mem_copy,
                    );
                    methods_dispatched += 1;
                }
            }
            i += 1;
        }
        if methods_dispatched != 0 {
            stats
                .methods_dispatched
                .fetch_add(methods_dispatched, Ordering::Relaxed);
        }
        live_macro_scratch.clear();
        self.live_macro_scratch = live_macro_scratch;
        live_macro_values.clear();
        self.live_macro_values = live_macro_values;
    }

    fn dump_pb_anomaly(&mut self, commands: &[u32], i: usize) {
        self.anomaly_dumped = true;
        let reason = PB_ANOMALY_REASON
            .lock()
            .ok()
            .and_then(|slot| slot.clone())
            .unwrap_or_default();
        log::error!(
            "[pb-anomaly] reason={} entry_gpu={:#x} word={} of {} state=(method={:#x} subch={} count={} noninc={} once={}) active_header={:#010x} ring_entries={}",
            reason,
            self.active_entry_gpu_va,
            i,
            commands.len(),
            self.state.method,
            self.state.subchannel,
            self.state.method_count,
            self.state.non_incrementing,
            self.state.increment_once,
            self.active_header,
            self.anomaly_ring.len()
        );
        let lo = i.saturating_sub(24);
        let hi = (i + 24).min(commands.len());
        log::error!(
            "[pb-anomaly] window[{}..{}]={:08x?}",
            lo,
            hi,
            &commands[lo..hi]
        );
        if let Ok(ranges) = PB_ANOMALY_RANGES.lock() {
            for submission in ranges.iter() {
                log::error!(
                    "[pb-anomaly-ranges] submit={} endform_enabled={} entries={} captured={}",
                    submission.sequence,
                    submission.enabled,
                    submission.total_entries,
                    submission.entries.len()
                );
                for (index, (raw, normalized, cpu_range)) in submission.entries.iter().enumerate() {
                    log::error!(
                        "[pb-anomaly-ranges] submit={} entry={} raw={:08x}:{:08x} gpu={:#x} words={} cpu_range={:#x?} normalized={:08x}:{:08x} gpu={:#x} words={} changed={}",
                        submission.sequence,
                        index,
                        raw.address_lo,
                        raw.address_hi_and_count,
                        raw.address(),
                        raw.entry_count(),
                        cpu_range,
                        normalized.address_lo,
                        normalized.address_hi_and_count,
                        normalized.address(),
                        normalized.entry_count(),
                        raw.address_lo != normalized.address_lo
                            || raw.address_hi_and_count != normalized.address_hi_and_count
                    );
                }
            }
        }
        for (index, entry) in self.anomaly_ring.iter().enumerate() {
            log::error!(
                "[pb-anomaly] ring[{}] gpu_va={:#x} state_in=(method={:#x} subch={} count={} noninc={} once={}) words={}",
                index,
                entry.gpu_va,
                entry.state_in.0,
                entry.state_in.1,
                entry.state_in.2,
                entry.state_in.3,
                entry.state_in.4,
                entry.total_words
            );
            for (line, chunk) in entry.words.chunks(16).enumerate() {
                let text: Vec<String> = chunk.iter().map(|w| format!("{:08x}", w)).collect();
                log::error!(
                    "[pb-anomaly] ring[{}] +{:04x}: {}",
                    index,
                    line * 16,
                    text.join(" ")
                );
            }
        }
    }

    fn dispatch_method(
        &mut self,
        arg: u32,
        arg_gpu_va: Option<u64>,
        prefetched_macro_arg: Option<u32>,
        defer_constbuf_writeback: bool,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        let method = self.state.method;
        let subchannel = self.state.subchannel as usize;
        let bound_class = self.bound_classes[subchannel & 7];
        let profile_started = self.gpu_profile_on.then(std::time::Instant::now);

        if method < NON_PULLER_METHODS {
            if puller_method_requires_hard_boundary(method) {
                self.prep_hard_flush(
                    kickprof::FLUSH_HARD_PULLER,
                    false,
                    mappings,
                    mem_read,
                    mem_write,
                );
            }
            let kp = kickprof::start();
            let mut engines = super::prep::PrepEngines {
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
            };
            self.handle_puller_method(
                method,
                arg,
                subchannel,
                &mut engines,
                mappings,
                stats,
                mem_read,
                mem_write,
                mem_copy,
            );
            kickprof::add(kickprof::PULLER, kp);
            profile_method_if_slow(
                profile_started,
                self.active_entry_gpu_va,
                self.active_word_index,
                bound_class,
                method,
            );
            return;
        }

        if bound_class == 0xB197 {
            let arg = if method >= MACRO_REGISTERS_START {
                let live = match (arg_gpu_va, prefetched_macro_arg) {
                    (Some(gpu_va), Some(value)) => Some((gpu_va, value)),
                    (arg_gpu_va, _) => {
                        let kp = kickprof::start();
                        let live = arg_gpu_va.and_then(|gpu_va| {
                            read_live_word(mappings, gpu_va, mem_read).map(|value| (gpu_va, value))
                        });
                        kickprof::add(kickprof::MACRO, kp);
                        live
                    }
                };
                if let Some((gpu_va, value)) = live {
                    if value != arg && mme_param_trace() {
                        use std::sync::atomic::{AtomicU32, Ordering};
                        static N: AtomicU32 = AtomicU32::new(0);
                        if N.fetch_add(1, Ordering::Relaxed) < 256 {
                            log::warn!(
                                "[mme-param-refresh] method={:#x} gpu_va={:#x} snapshot={:#010x} live={:#010x}",
                                method,
                                gpu_va,
                                arg,
                                value
                            );
                        }
                    }
                    value
                } else {
                    arg
                }
            } else {
                arg
            };
            let is_last = self.state.method_count <= 1;
            let pre_draws = maxwell.regs.draw_count;
            let pre_clears = maxwell.regs.clear_count;
            let kp_m3d = kickprof::start();
            maxwell.dispatch_method(method, arg, is_last);
            kickprof::add(kickprof::M3D, kp_m3d);
            let upload_launch = maxwell.inline_upload_launch_pending();
            if upload_launch {
                self.prep_hard_flush(
                    kickprof::FLUSH_HARD_INLINE_UPLOAD,
                    true,
                    mappings,
                    mem_read,
                    mem_write,
                );
            }
            if maxwell.has_pending_inline_uploads() {
                let methods = maxwell.take_pending_inline_uploads();
                let mut engines = super::prep::PrepEngines {
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                };
                self.emit_prep(
                    super::prep::PrepEvent::InlineUploadMethods(methods),
                    &mut engines,
                    mappings,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
            }
            let d = maxwell.regs.draw_count - pre_draws;
            let c = maxwell.regs.clear_count - pre_clears;
            if d > 0 {
                stats.maxwell3d_draws.fetch_add(d, Ordering::Relaxed);
            }
            if c > 0 {
                stats.maxwell3d_clears.fetch_add(c, Ordering::Relaxed);
            }
            maxwell.record_method(method);

            let non_constbuf_work_pending = !maxwell.pending_draws.is_empty()
                || !maxwell.regs.pending_semaphore_acquires.is_empty()
                || !maxwell.regs.pending_semaphore_writes.is_empty()
                || maxwell.regs.pending_barrier_flushes != 0
                || maxwell.regs.pending_texture_cache_invalidates != 0;
            if !non_constbuf_work_pending
                && (maxwell.regs.pending_constbuf_writes.is_empty() || defer_constbuf_writeback)
            {
                profile_method_if_slow(
                    profile_started,
                    self.active_entry_gpu_va,
                    self.active_word_index,
                    bound_class,
                    method,
                );
                return;
            }

            let constbuf_trace = constbuf_upload_watch().is_some().then(|| {
                (
                    ((maxwell.regs.constbuf_selector_addr_hi as u64) << 32)
                        | maxwell.regs.constbuf_selector_addr_lo as u64,
                    maxwell.regs.constbuf_selector_size,
                )
            });
            let constbuf_write_count = maxwell.regs.pending_constbuf_writes.len();
            let replay_constbuf_writes = if constbuf_write_count != 0
                && maxwell
                    .pending_draws
                    .iter()
                    .any(|draw| draw.constbuf_write_count < constbuf_write_count)
            {
                std::mem::take(&mut maxwell.regs.pending_constbuf_writes)
            } else {
                if constbuf_write_count != 0 {
                    let writes = std::mem::take(&mut maxwell.regs.pending_constbuf_writes);
                    maxwell
                        .regs
                        .pending_constbuf_writes
                        .reserve(constbuf_write_count);
                    let mut engines = super::prep::PrepEngines {
                        maxwell_dma,
                        fermi_2d,
                        kepler_compute,
                        kepler_memory,
                    };
                    self.emit_prep(
                        super::prep::PrepEvent::ConstbufWrites {
                            writes,
                            trace: constbuf_trace,
                        },
                        &mut engines,
                        mappings,
                        stats,
                        mem_read,
                        mem_write,
                        mem_copy,
                    );
                }
                Vec::new()
            };
            if !maxwell.regs.pending_semaphore_acquires.is_empty() {
                let kp = kickprof::start();
                let acquires = std::mem::take(&mut maxwell.regs.pending_semaphore_acquires);
                static NO_ACQUIRE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let skip =
                    *NO_ACQUIRE.get_or_init(|| std::env::var_os("NEXIUM_NO_SEM_ACQUIRE").is_some());
                if !skip {
                    static ACQUIRES: AtomicU64 = AtomicU64::new(0);
                    for (gpu_va, payload, _mode) in acquires {
                        if matches!(self.prep, super::prep::PrepLane::Threaded(_)) {
                            self.flush_engine_event_batch();
                            let super::prep::PrepLane::Threaded(handle) = &mut self.prep else {
                                unreachable!()
                            };
                            let satisfied = mappings.cpu_address_for(gpu_va).is_some_and(|cpu| {
                                let mut buf = [0u8; 4];
                                mem_read(cpu, &mut buf) && {
                                    let value = u32::from_le_bytes(buf);
                                    value == payload || (value.wrapping_sub(payload) as i32) >= 0
                                }
                            });
                            if !satisfied {
                                let (ack_tx, ack_rx) = crossbeam::channel::bounded(1);
                                handle.send(super::prep::PrepEvent::SemAcquire {
                                    gpu_va,
                                    payload,
                                    ack: ack_tx,
                                });
                                let started = std::time::Instant::now();
                                let _ = ack_rx.recv_timeout(std::time::Duration::from_secs(4));
                                semacq_ack_stat(started.elapsed());
                            }
                            continue;
                        }
                        let Some(cpu) = mappings.cpu_address_for(gpu_va) else {
                            static UNMAPPED: AtomicU64 = AtomicU64::new(0);
                            let n = UNMAPPED.fetch_add(1, Ordering::Relaxed);
                            if n < 32 {
                                log::warn!(
                                    "[sem-acquire] unmapped gpu_va={:#x} payload={:#x}",
                                    gpu_va,
                                    payload
                                );
                            }
                            continue;
                        };
                        let n = ACQUIRES.fetch_add(1, Ordering::Relaxed);
                        let start = std::time::Instant::now();
                        let mut last = 0u32;
                        loop {
                            let mut buf = [0u8; 4];
                            if mem_read(cpu, &mut buf) {
                                last = u32::from_le_bytes(buf);
                                if last == payload || (last.wrapping_sub(payload) as i32) >= 0 {
                                    if n < 8 {
                                        log::warn!(
                                            "[sem-acquire] #{} gpu_va={:#x} payload={:#x} value={:#x}",
                                            n,
                                            gpu_va,
                                            payload,
                                            last
                                        );
                                    }
                                    break;
                                }
                            }
                            if start.elapsed() >= std::time::Duration::from_secs(3) {
                                static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
                                let t = TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                                if t < 32 || t % 1024 == 0 {
                                    log::warn!(
                                        "[sem-acquire] timeout gpu_va={:#x} payload={:#x} last={:#x}",
                                        gpu_va,
                                        payload,
                                        last
                                    );
                                }
                                break;
                            }
                            nexium_common::host_wake::micro_pause();
                        }
                    }
                }
                kickprof::add(kickprof::SEMACQ, kp);
            }
            if !maxwell.pending_draws.is_empty() {
                let gs_debug = gs_dump_enabled().then(|| maxwell.gs_debug_regs());
                let draw_capacity = maxwell.pending_draws.len();
                let draws = std::mem::replace(
                    &mut maxwell.pending_draws,
                    Vec::with_capacity(draw_capacity),
                );
                if let super::prep::PrepLane::Threaded(handle) = &self.prep {
                    if let Some(recycled) = handle.try_take_recycled_draw_vec() {
                        maxwell.pending_draws = recycled;
                    }
                }
                if kickprof::enabled() {
                    kickprof::count(
                        kickprof::HOST_DRAWS,
                        draws.iter().filter(|draw| !draw.is_clear).count() as u64,
                    );
                    kickprof::count(
                        kickprof::DRAW_INSTANCES,
                        draws
                            .iter()
                            .filter(|draw| !draw.is_clear)
                            .map(|draw| draw.instance_count.max(1) as u64)
                            .sum(),
                    );
                }
                let mut engines = super::prep::PrepEngines {
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                };
                self.emit_prep(
                    super::prep::PrepEvent::Draws {
                        draws,
                        gs_debug,
                        replay_constbuf_writes,
                        constbuf_trace,
                    },
                    &mut engines,
                    mappings,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
            } else {
                debug_assert!(replay_constbuf_writes.is_empty());
            }
            if !maxwell.regs.pending_semaphore_writes.is_empty() {
                let writes = std::mem::take(&mut maxwell.regs.pending_semaphore_writes);
                let mut engines = super::prep::PrepEngines {
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                };
                self.emit_prep(
                    super::prep::PrepEvent::SemRelease(writes),
                    &mut engines,
                    mappings,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
            }
            let barrier_flushes = std::mem::take(&mut maxwell.regs.pending_barrier_flushes);
            let fragment_barriers = std::mem::take(&mut maxwell.regs.pending_fragment_barriers);
            let tiled_cache_barriers =
                std::mem::take(&mut maxwell.regs.pending_tiled_cache_barriers);
            let texture_invalidates =
                std::mem::take(&mut maxwell.regs.pending_texture_cache_invalidates);
            if barrier_flushes != 0
                || fragment_barriers != 0
                || tiled_cache_barriers != 0
                || texture_invalidates != 0
            {
                let mut engines = super::prep::PrepEngines {
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                };
                self.emit_prep(
                    super::prep::PrepEvent::Barrier {
                        barrier_flushes,
                        fragment_barriers,
                        tiled_cache_barriers,
                        texture_invalidates,
                    },
                    &mut engines,
                    mappings,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
            }
        } else if matches!(
            bound_class,
            MAXWELL_DMA_CLASS | FERMI_2D_CLASS | KEPLER_MEMORY_CLASS | KEPLER_COMPUTE_CLASS
        ) {
            let is_last = self.state.method_count <= 1;
            if matches!(self.prep, super::prep::PrepLane::Threaded(_)) {
                match &mut self.engine_event_batch {
                    Some((class, methods)) if *class == bound_class => {
                        methods.push((method, arg, is_last));
                    }
                    _ => {
                        self.flush_engine_event_batch();
                        self.engine_event_batch = Some((bound_class, vec![(method, arg, is_last)]));
                    }
                }
                profile_method_if_slow(
                    profile_started,
                    self.active_entry_gpu_va,
                    self.active_word_index,
                    bound_class,
                    method,
                );
                return;
            }
            let mut engines = super::prep::PrepEngines {
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
            };
            if !self.emit_prep(
                super::prep::PrepEvent::EngineMethod {
                    class: bound_class,
                    method,
                    arg,
                    is_last,
                },
                &mut engines,
                mappings,
                stats,
                mem_read,
                mem_write,
                mem_copy,
            ) {
                return;
            }
        } else {
            if bound_class == 0xB1C0 {
                use std::sync::atomic::{AtomicU64, Ordering as O2};
                use std::sync::{Mutex, OnceLock};
                static COMPUTE_METHODS: AtomicU64 = AtomicU64::new(0);
                let n = COMPUTE_METHODS.fetch_add(1, O2::Relaxed) + 1;
                if method == 0xAF {
                    log::warn!("[compute] LAUNCH (0xAF) arg={:#x} count={}", arg, n);
                }
                if compute_debug_enabled() {
                    static METHS: OnceLock<Mutex<std::collections::BTreeMap<u32, u64>>> =
                        OnceLock::new();
                    let m = METHS.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()));
                    if let Ok(mut map) = m.lock() {
                        *map.entry(method).or_insert(0) += 1;
                        if n % 10000 == 0 {
                            let s: Vec<String> =
                                map.iter().map(|(k, v)| format!("{:#x}:{}", k, v)).collect();
                            log::warn!(
                                "[compute-methods] n={} distinct={} [{}]",
                                n,
                                map.len(),
                                s.join(" ")
                            );
                        }
                    }
                    if n <= 90 {
                        log::warn!("[compute-seq] #{} method={:#x} arg={:#x}", n, method, arg);
                    }
                }
            }
            if bound_class != 0 {
                use std::sync::{Mutex, OnceLock};
                static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
                let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                if let Ok(mut s) = seen.lock() {
                    if s.insert(bound_class) {
                        log::warn!(
                            "[gpu-unhandled-class] subch={} class={:#x} method={:#x} arg={:#x} NOT dispatched (0xb1c0=KeplerCompute) — fence may never release",
                            subchannel,
                            bound_class,
                            method,
                            arg
                        );
                    }
                }
            }
            log::trace!(
                "pusher: subch={} class={:#x} method={:#x} arg={:#x} (unsupported class)",
                subchannel,
                bound_class,
                method,
                arg
            );
        }
        profile_method_if_slow(
            profile_started,
            self.active_entry_gpu_va,
            self.active_word_index,
            bound_class,
            method,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_puller_method(
        &mut self,
        method: u32,
        arg: u32,
        subchannel: usize,
        engines: &mut super::prep::PrepEngines,
        mappings: &GpuMappings,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        mem_copy: &dyn Fn(u64, u64, usize) -> bool,
    ) {
        match method {
            METHOD_BIND_OBJECT => {
                let class = arg & 0xFFFF;
                if is_known_gpu_class(class) {
                    self.bound_classes[subchannel & 7] = class;
                    log::debug!("puller: BindObject subch={} class={:#x}", subchannel, class);
                } else {
                    log::trace!(
                        "puller: BindObject subch={} class={:#x} rejected (unknown) keeping {:#x}",
                        subchannel,
                        class,
                        self.bound_classes[subchannel & 7]
                    );
                }
            }
            METHOD_SEMAPHORE_ADDR_HIGH => {
                if arg <= 0xFF {
                    self.puller.semaphore_addr_high = arg;
                }
            }
            METHOD_SEMAPHORE_ADDR_LOW => self.puller.semaphore_addr_low = arg,
            METHOD_SEMAPHORE_PAYLOAD => self.puller.semaphore_payload = arg,
            METHOD_SEMAPHORE_OPERATION => {
                if arg & 0xF == 0x2 {
                    let payload = self.puller.semaphore_payload;
                    let gpu_va = ((self.puller.semaphore_addr_high as u64) << 32)
                        | (self.puller.semaphore_addr_low as u64);
                    self.emit_prep(
                        super::prep::PrepEvent::PullerSemWrite {
                            gpu_va,
                            payload,
                            long: true,
                        },
                        engines,
                        mappings,
                        stats,
                        mem_read,
                        mem_write,
                        mem_copy,
                    );
                }
            }
            METHOD_SEMAPHORE_RELEASE => {
                let gpu_va = ((self.puller.semaphore_addr_high as u64) << 32)
                    | (self.puller.semaphore_addr_low as u64);
                self.emit_prep(
                    super::prep::PrepEvent::PullerSemWrite {
                        gpu_va,
                        payload: arg,
                        long: false,
                    },
                    engines,
                    mappings,
                    stats,
                    mem_read,
                    mem_write,
                    mem_copy,
                );
            }
            METHOD_SEMAPHORE_ACQUIRE => {}
            METHOD_SYNCPOINT_PAYLOAD => self.puller.syncpoint_payload = arg,
            METHOD_SYNCPOINT_OPERATION => {
                let op = arg & 0xF;
                if op == 1 {
                    let index = (arg >> 8) & 0xFF;
                    self.syncpt_value = self.syncpt_value.wrapping_add(1);
                    match self
                        .pending_syncpt_incrs
                        .iter_mut()
                        .find(|(id, _)| *id == index)
                    {
                        Some((_, count)) => *count = count.wrapping_add(1),
                        None => self.pending_syncpt_incrs.push((index, 1)),
                    }
                    log::debug!(
                        "puller: SyncpointIncrement id={} → {}",
                        index,
                        self.syncpt_value
                    );
                }
            }
            _ => {
                log::trace!("puller: method {:#x} arg={:#x}", method, arg);
            }
        }
    }
}

fn puller_method_requires_hard_boundary(method: u32) -> bool {
    matches!(
        method,
        METHOD_SEMAPHORE_OPERATION
            | METHOD_SEMAPHORE_ACQUIRE
            | METHOD_SEMAPHORE_RELEASE
            | METHOD_SYNCPOINT_OPERATION
    )
}

pub(crate) fn direct_forensics() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_FORENSICS").is_some())
}

fn passive_maxwell_run_len(method: u32, available: usize, non_incrementing: bool) -> usize {
    if non_incrementing {
        return usize::from(Maxwell3D::is_pusher_passive_method(method)) * available;
    }
    (0..available)
        .take_while(|offset| {
            Maxwell3D::is_pusher_passive_method(method.wrapping_add(*offset as u32))
        })
        .count()
}

pub(crate) fn record_gpfifo_entries(entries: &[CommandListHeader], mappings: &GpuMappings) {
    record_pb_anomaly_ranges(entries, entries, mappings, false);
}

pub(crate) fn read_gpu_scattered(
    mappings: &GpuMappings,
    gpu_va: u64,
    out: &mut [u8],
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    let mut off = 0usize;
    let mut va = gpu_va;
    while off < out.len() {
        match mappings.cpu_range_for(va) {
            Some((cpu, remain)) if remain > 0 => {
                let take = (remain as usize).min(out.len() - off);
                let _ = mem_read(cpu, &mut out[off..off + take]);
                off += take;
                va = va.wrapping_add(take as u64);
            }
            _ => {
                let page_left = (0x1000 - (va & 0xFFF)) as usize;
                let step = page_left.min(out.len() - off).max(1);
                off += step;
                va = va.wrapping_add(step as u64);
            }
        }
    }
}

fn read_live_word(
    mappings: &GpuMappings,
    gpu_va: u64,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<u32> {
    let (cpu_addr, available) = mappings.cpu_range_for(gpu_va)?;
    if available < 4 {
        return None;
    }
    let mut bytes = [0u8; 4];
    if !mem_read(cpu_addr, &mut bytes) {
        return None;
    }
    Some(u32::from_le_bytes(bytes))
}

fn read_live_words(
    mappings: &GpuMappings,
    gpu_va: u64,
    max_word_count: usize,
    out: &mut Vec<u8>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> (usize, bool) {
    let Some((cpu_addr, available)) = mappings.cpu_range_for(gpu_va) else {
        return (1, false);
    };
    let mapped_words = usize::try_from(available / 4).unwrap_or(usize::MAX);
    let word_count = max_word_count.min(mapped_words);
    if word_count <= 1 {
        return (1, false);
    }
    let Some(byte_len) = word_count.checked_mul(4) else {
        return (1, false);
    };
    out.resize(byte_len, 0);
    (word_count, mem_read(cpu_addr, out.as_mut_slice()))
}

fn mme_param_trace() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MME_PARAM_TRACE").is_some())
}

fn mme_batch_refresh_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("NEXIUM_MME_BATCH_REFRESH")
            .ok()
            .is_some_and(|value| {
                let value = value.trim();
                value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off")
                    || value.eq_ignore_ascii_case("no")
            })
    })
}

fn mme_slice_dispatch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("NEXIUM_MME_SLICE_DISPATCH")
            .ok()
            .is_some_and(|value| {
                let value = value.trim();
                value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off")
                    || value.eq_ignore_ascii_case("no")
            })
    })
}

fn mme_dispatch_trace() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MME_TRACE").is_some())
}

fn suspicious_direct(class: u32, method: u32, arg: u32) -> bool {
    if !direct_forensics() {
        return false;
    }
    if class == 0xB197 {
        let hi_reg = method == 0x582
            || method == 0x6c0
            || method == 0x8e1
            || method == 0x554
            || (method >= 0x200 && method < 0x280 && (method & 0xF) == 0);
        if hi_reg && arg > 0xFF {
            return true;
        }
        if (method == 0x47 || method == 0x48) && arg > 0x1000 {
            return true;
        }
    }
    if method == METHOD_SEMAPHORE_ADDR_HIGH && arg > 0xFF {
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn dump_direct_ctx(
    entry_gpu_va: u64,
    entry_cpu_va: u64,
    method: u32,
    arg: u32,
    class: u32,
    method_count: u32,
    non_incrementing: bool,
    commands: &[u32],
    i: usize,
) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 64 {
        return;
    }
    let full = std::env::var("NEXIUM_PB_DUMP_WORDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let (lo, hi) = match full {
        Some(n) => (0, n.min(commands.len())),
        None => (i.saturating_sub(3), (i + 6).min(commands.len())),
    };
    log::warn!(
        "[pb-garbage] entry_gpu={:#x} cpu={:#x} word={} class={:#x} method={:#x} arg={:#010x} mcount={} noninc={} words[{}..{}]={:08x?}",
        entry_gpu_va,
        entry_cpu_va,
        i,
        class,
        method,
        arg,
        method_count,
        non_incrementing,
        lo,
        hi,
        &commands[lo..hi]
    );
}

fn is_known_gpu_class(class: u32) -> bool {
    matches!(
        class,
        0xB197 | MAXWELL_DMA_CLASS | FERMI_2D_CLASS | KEPLER_MEMORY_CLASS | KEPLER_COMPUTE_CLASS
    )
}

pub(crate) fn constbuf_upload_watch() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static WATCH: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *WATCH.get_or_init(|| {
        let spec = std::env::var("NEXIUM_CBUF_UPLOAD_WATCH").ok()?;
        let (va, len) = spec.trim().split_once(':')?;
        let va = parse_u64ish(va.trim())?;
        let len = parse_u64ish(len.trim()).unwrap_or(4);
        (va != 0 && len != 0).then_some((va, len))
    })
}

fn parse_u64ish(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

pub(crate) fn contiguous_constbuf_write_run_end(writes: &[(u64, u32)], start: usize) -> usize {
    if start >= writes.len() {
        return start;
    }
    let mut end = start + 1;
    while end < writes.len()
        && writes[end - 1]
            .0
            .checked_add(std::mem::size_of::<u32>() as u64)
            == Some(writes[end].0)
    {
        end += 1;
    }
    end
}

fn should_defer_constbuf_writeback(
    state: &DmaState,
    has_next_payload_word: bool,
    upload_watch_active: bool,
) -> bool {
    has_next_payload_word
        && !upload_watch_active
        && state.method_count > 1
        && state.non_incrementing
        && (0x8E4..=0x8F3).contains(&state.method)
}

fn constbuf_replay_end(requested: usize, committed: usize, total: usize) -> usize {
    requested.min(total).max(committed)
}

pub(crate) fn constbuf_replay_group_end_by<T>(
    items: &[T],
    start: usize,
    committed: usize,
    total: usize,
    mut write_count: impl FnMut(&T) -> usize,
) -> (usize, usize) {
    if start >= items.len() {
        return (start, committed.min(total));
    }
    let boundary = constbuf_replay_end(write_count(&items[start]), committed, total);
    let mut end = start + 1;
    while end < items.len()
        && constbuf_replay_end(write_count(&items[end]), boundary, total) == boundary
    {
        end += 1;
    }
    (end, boundary)
}

impl Default for Pusher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn gpfifo_entry(address: u64, count: u32, no_prefetch: bool) -> CommandListHeader {
        CommandListHeader {
            address_lo: address as u32,
            address_hi_and_count: ((address >> 32) as u32 & 0xFF)
                | 0x200
                | (count << 10)
                | if no_prefetch { 0x8000_0000 } else { 0 },
        }
    }

    fn install_three_param_echo(maxwell: &mut Maxwell3D) {
        maxwell.dispatch_method(0x45, 0, true);
        for word in [
            0x0480_0221,
            0x0000_0A30,
            0x0000_1330,
            0x0000_1BC0,
            0x0000_0010,
        ] {
            maxwell.dispatch_method(0x46, word, true);
        }
        maxwell.dispatch_method(0x47, 0, true);
        maxwell.dispatch_method(0x48, 0, true);
    }

    fn words_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn passive_bulk_gate_requires_explicit_truthy_value() {
        for value in ["1", "true", "TRUE", "on", "On", "yes", " YES "] {
            assert!(m3d_passive_bulk_value_enabled(Some(value)));
        }
        for value in ["", "0", "false", "off", "no", "2", "enabled"] {
            assert!(!m3d_passive_bulk_value_enabled(Some(value)));
        }
        assert!(!m3d_passive_bulk_value_enabled(None));
    }

    #[test]
    fn passive_bulk_run_stops_before_side_effect_boundaries() {
        assert_eq!(passive_maxwell_run_len(0x43, 8, false), 1);
        assert_eq!(passive_maxwell_run_len(0x44, 8, false), 0);
        assert_eq!(passive_maxwell_run_len(0x45, 8, false), 0);
        assert_eq!(passive_maxwell_run_len(0x8e1, 8, false), 1);
        assert_eq!(passive_maxwell_run_len(0x8e2, 8, false), 0);
        assert_eq!(passive_maxwell_run_len(0x200, 12, true), 12);
        assert_eq!(passive_maxwell_run_len(0x44, 12, true), 0);
    }

    #[test]
    fn command_list_headers_are_decoded_into_owned_values() {
        let first = gpfifo_entry(0x12_3456_7000, 37, false);
        let second = gpfifo_entry(0x23_4567_8000, 91, true);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&first.address_lo.to_le_bytes());
        bytes.extend_from_slice(&first.address_hi_and_count.to_le_bytes());
        bytes.extend_from_slice(&second.address_lo.to_le_bytes());
        bytes.extend_from_slice(&second.address_hi_and_count.to_le_bytes());
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);

        let decoded = decode_command_list_headers(&bytes);

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].address(), first.address());
        assert_eq!(decoded[0].entry_count(), 37);
        assert!(!decoded[0].no_prefetch());
        assert_eq!(decoded[1].address(), second.address());
        assert_eq!(decoded[1].entry_count(), 91);
        assert!(decoded[1].no_prefetch());
    }

    #[test]
    fn kepler_rt_invalidation_keeps_unmapped_spans_gpu_targeted() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x100, 0x8000, 1);
        let outcome = KeplerMemoryWriteOutcome::Exact(vec![(0x1010, 0x10), (0x3000, 0x10)]);

        assert_eq!(
            kepler_rt_invalidation(&mappings, &outcome),
            Some(KeplerRtInvalidation::Exact(vec![
                (0x8010, 0x1010, 0x10),
                (0, 0x3000, 0x10)
            ]))
        );
    }

    #[test]
    fn kepler_rt_invalidation_uses_gpu_range_at_mapping_boundary() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x100, 0x8000, 1);
        let outcome = KeplerMemoryWriteOutcome::Exact(vec![(0x10f0, 0x20)]);

        assert_eq!(
            kepler_rt_invalidation(&mappings, &outcome),
            Some(KeplerRtInvalidation::Exact(vec![(0x80f0, 0x10f0, 0x20)]))
        );
    }

    #[test]
    fn kepler_rt_invalidation_ignores_empty_outcomes() {
        let mappings = GpuMappings::new();

        assert_eq!(
            kepler_rt_invalidation(&mappings, &KeplerMemoryWriteOutcome::Unknown(Vec::new())),
            None
        );
        assert_eq!(
            kepler_rt_invalidation(&mappings, &KeplerMemoryWriteOutcome::Exact(Vec::new())),
            None
        );
    }

    #[test]
    fn kepler_rt_invalidation_preserves_known_spans_from_partial_writes() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x100, 0x8000, 1);
        let outcome = KeplerMemoryWriteOutcome::Unknown(vec![(0x1010, 0x10), (0x3000, 0x10)]);

        assert_eq!(
            kepler_rt_invalidation(&mappings, &outcome),
            Some(KeplerRtInvalidation::Exact(vec![
                (0x8010, 0x1010, 0x10),
                (0, 0x3000, 0x10)
            ]))
        );
    }

    #[test]
    fn kepler_rt_invalidation_preserves_fully_resolved_spans() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x100, 0x8000, 1);
        let outcome = KeplerMemoryWriteOutcome::Exact(vec![(0x1010, 0x20)]);

        assert_eq!(
            kepler_rt_invalidation(&mappings, &outcome),
            Some(KeplerRtInvalidation::Exact(vec![(0x8010, 0x1010, 0x20)]))
        );
    }

    #[test]
    fn only_dispatchable_pushbuffer_modes_decode_to_headers() {
        assert_eq!(Mode::from_bits(0), None);
        assert_eq!(Mode::from_bits(1), Some(Mode::Increasing));
        assert_eq!(Mode::from_bits(2), None);
        assert_eq!(Mode::from_bits(3), Some(Mode::NonIncreasing));
        assert_eq!(Mode::from_bits(4), Some(Mode::Inline));
        assert_eq!(Mode::from_bits(5), Some(Mode::IncreaseOnce));
        assert_eq!(Mode::from_bits(6), None);
        assert_eq!(Mode::from_bits(7), None);
    }

    #[test]
    fn zero_nop_headers_preserve_zero_method_payloads() {
        let mut pusher = Pusher::new();
        let mappings = GpuMappings::new();
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();

        pusher.process_commands(
            &[0, 0x2002_0200, 0, 0x3344, 0, 0x2001_0202, 0x5566, 0],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0, 0x3344, 0x5566]);
        assert_eq!(pusher.state.method, 0x203);
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
    }

    fn execute_gpfifo_length_case(entries: &[CommandListHeader], inline: bool) {
        let gpu = super::super::GpuContext::new();
        let gpu_base = entries[0].address() - 0x2000;
        let cpu_base = 0x20_0000;
        let end = entries[0].address() + u64::from(entries[0].entry_count()) * 4;
        let mut memory = vec![0u8; (end - gpu_base) as usize];
        memory[0x2000..0x2008].copy_from_slice(&words_bytes(&[0x2001_0200, 0x12]));
        let tail = memory.len() - 8;
        memory[tail..].copy_from_slice(&words_bytes(&[0x2001_0201, 0x5566_7788]));
        if entries.len() > 1 {
            let second = (entries[1].address() - gpu_base) as usize;
            memory[second..second + 8].copy_from_slice(&words_bytes(&[0x2001_0202, 0x99aa_bbcc]));
        }
        for (index, entry) in entries.iter().enumerate() {
            memory[index * 8..index * 8 + 8].copy_from_slice(&words_bytes(&[
                entry.address_lo,
                entry.address_hi_and_count,
            ]));
        }
        gpu.mappings
            .write()
            .add(gpu_base, memory.len() as u64, cpu_base, 1);
        let mem_read = |cpu: u64, output: &mut [u8]| {
            let start = cpu.checked_sub(cpu_base).unwrap() as usize;
            output.copy_from_slice(&memory[start..start + output.len()]);
            true
        };
        let completions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let completion_count = Arc::clone(&completions);
        let on_complete = Some(Box::new(move || {
            completion_count.fetch_add(1, Ordering::Relaxed);
        }) as Box<dyn FnOnce() + Send>);
        if inline {
            gpu.process_inline_gpfifo(entries, mem_read, |_, _| true, |_, _, _| false, on_complete);
        } else {
            gpu.submit_gpfifo(
                gpu_base,
                entries.len() as u32,
                mem_read,
                |_, _| true,
                |_, _, _| false,
                on_complete,
            );
        }
        let maxwell = gpu.maxwell3d.lock();
        assert_eq!(&maxwell.reg_file[0x200..0x202], &[0x12, 0x5566_7788]);
        if entries.len() > 1 {
            assert_eq!(maxwell.reg_file[0x202], 0x99aa_bbcc);
        }
        assert_eq!(completions.load(Ordering::Relaxed), 1);
        assert_eq!(gpu.pusher.lock().state.method_count, 0);
    }

    #[test]
    fn gpfifo_length_is_preserved_when_it_matches_cpu_address_bits() {
        for inline in [false, true] {
            execute_gpfifo_length_case(&[gpfifo_entry(0x10_2000, 0x1800, false)], inline);
        }
    }

    #[test]
    fn gpfifo_length_is_preserved_when_it_matches_another_entry_address() {
        for inline in [false, true] {
            execute_gpfifo_length_case(
                &[
                    gpfifo_entry(0x40_2000, 0x2040, true),
                    gpfifo_entry(0x40_2040, 2, false),
                ],
                inline,
            );
        }
    }

    #[test]
    fn gpfifo_length_executes_the_upper_half_of_the_21_bit_range() {
        for (words, no_prefetch) in [(0x10_0001, false), (0x1f_ffff, true)] {
            for inline in [false, true] {
                execute_gpfifo_length_case(&[gpfifo_entry(0x10_2000, words, no_prefetch)], inline);
            }
        }
    }

    #[test]
    fn constbuf_writeback_deferral_is_bounded_to_the_current_packet_chunk() {
        let mut state = DmaState {
            method: 0x8E4,
            subchannel: 0,
            method_count: 2,
            non_incrementing: true,
            increment_once: false,
        };

        assert!(should_defer_constbuf_writeback(&state, true, false));
        assert!(!should_defer_constbuf_writeback(&state, false, false));
        assert!(!should_defer_constbuf_writeback(&state, true, true));
        state.method_count = 1;
        assert!(!should_defer_constbuf_writeback(&state, true, false));
        state.method_count = 2;
        state.non_incrementing = false;
        assert!(!should_defer_constbuf_writeback(&state, true, false));
        state.non_incrementing = true;
        state.method = 0x8F4;
        assert!(!should_defer_constbuf_writeback(&state, true, false));
    }

    #[test]
    fn constbuf_write_runs_coalesce_only_strictly_contiguous_dwords() {
        let writes = [
            (0x1000, 1),
            (0x1004, 2),
            (0x1008, 3),
            (0x1008, 4),
            (0x2000, 5),
            (0x2004, 6),
            (u64::MAX - 3, 7),
            (0, 8),
        ];

        assert_eq!(contiguous_constbuf_write_run_end(&writes, 0), 3);
        assert_eq!(contiguous_constbuf_write_run_end(&writes, 3), 4);
        assert_eq!(contiguous_constbuf_write_run_end(&writes, 4), 6);
        assert_eq!(contiguous_constbuf_write_run_end(&writes, 6), 7);
        assert_eq!(contiguous_constbuf_write_run_end(&writes, 7), 8);
        assert_eq!(
            contiguous_constbuf_write_run_end(&writes, writes.len()),
            writes.len()
        );
    }

    #[test]
    fn constbuf_replay_boundaries_are_monotonic_and_clamped() {
        assert_eq!(constbuf_replay_end(3, 0, 8), 3);
        assert_eq!(constbuf_replay_end(2, 3, 8), 3);
        assert_eq!(constbuf_replay_end(20, 3, 8), 8);
    }

    #[test]
    fn constbuf_replay_groups_only_adjacent_equal_effective_boundaries() {
        let counts = [0usize, 0, 2, 2, 1, 5, 9, 9];
        assert_eq!(
            constbuf_replay_group_end_by(&counts, 0, 0, 8, |count| *count),
            (2, 0)
        );
        assert_eq!(
            constbuf_replay_group_end_by(&counts, 2, 0, 8, |count| *count),
            (5, 2)
        );
        assert_eq!(
            constbuf_replay_group_end_by(&counts, 5, 2, 8, |count| *count),
            (6, 5)
        );
        assert_eq!(
            constbuf_replay_group_end_by(&counts, 6, 5, 8, |count| *count),
            (8, 8)
        );
        assert_eq!(
            constbuf_replay_group_end_by(&counts, counts.len(), 9, 8, |count| *count),
            (counts.len(), 8)
        );
    }

    #[test]
    fn pending_constbuf_writes_commit_contiguous_runs_in_single_host_writes() {
        let mut pusher = Pusher::new();
        let mut maxwell = Maxwell3D::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 0x100, 0x9000, 1);
        maxwell.regs.pending_constbuf_writes = vec![
            (0x5000, 0x1122_3344),
            (0x5004, 0x5566_7788),
            (0x5008, 0x99aa_bbcc),
            (0x5010, 0xddee_ff00),
        ];
        let committed = Mutex::new(Vec::<(u64, Vec<u8>)>::new());
        let mem_write = |cpu: u64, data: &[u8]| {
            committed.lock().unwrap().push((cpu, data.to_vec()));
            true
        };

        let writes = std::mem::take(&mut maxwell.regs.pending_constbuf_writes);
        pusher
            .inline_prep()
            .commit_constbuf_writes(&writes, None, &mappings, &mem_write);

        assert!(maxwell.regs.pending_constbuf_writes.is_empty());
        assert!(
            pusher
                .inline_prep()
                .constbuf_invalidation_scratch
                .capacity()
                >= 2
        );
        assert!(pusher.inline_prep().constbuf_bytes_scratch.capacity() >= 12);
        assert_eq!(
            *committed.lock().unwrap(),
            vec![
                (
                    0x9000,
                    vec![0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0xcc, 0xbb, 0xaa, 0x99,],
                ),
                (0x9010, vec![0x00, 0xff, 0xee, 0xdd]),
            ]
        );
    }

    #[test]
    fn non_incrementing_constbuf_packet_commits_one_host_write() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 0x100, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        maxwell.regs.constbuf_selector_size = 0x100;
        maxwell.regs.constbuf_selector_addr_lo = 0x5000;
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let committed = Mutex::new(Vec::<(u64, Vec<u8>)>::new());
        let mem_write = |cpu: u64, data: &[u8]| {
            committed.lock().unwrap().push((cpu, data.to_vec()));
            true
        };
        let header = (3u32 << 29) | (4 << 16) | 0x8E4;

        pusher.process_commands(
            &[header, 0x1122_3344, 0x5566_7788, 0x99AA_BBCC, 0xDDEE_FF00],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &mem_write,
            &|_, _, _| false,
        );

        assert_eq!(
            *committed.lock().unwrap(),
            vec![(
                0x9000,
                vec![
                    0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0xCC, 0xBB, 0xAA, 0x99, 0x00,
                    0xFF, 0xEE, 0xDD,
                ],
            )]
        );
        assert_eq!(maxwell.regs.constbuf_load_offset, 16);
        assert!(maxwell.regs.pending_constbuf_writes.is_empty());
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn split_constbuf_packet_commits_at_each_command_chunk_boundary() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 0x100, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        maxwell.regs.constbuf_selector_size = 0x100;
        maxwell.regs.constbuf_selector_addr_lo = 0x5000;
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let committed = Mutex::new(Vec::<(u64, Vec<u8>)>::new());
        let mem_write = |cpu: u64, data: &[u8]| {
            committed.lock().unwrap().push((cpu, data.to_vec()));
            true
        };
        let header = (3u32 << 29) | (4 << 16) | 0x8E4;

        pusher.process_commands(
            &[header, 0x1122_3344, 0x5566_7788],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method_count, 2);
        assert_eq!(maxwell.regs.constbuf_load_offset, 8);
        assert!(maxwell.regs.pending_constbuf_writes.is_empty());

        pusher.process_commands(
            &[0x99AA_BBCC, 0xDDEE_FF00],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &mem_write,
            &|_, _, _| false,
        );

        assert_eq!(
            *committed.lock().unwrap(),
            vec![
                (0x9000, vec![0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55],),
                (0x9008, vec![0xCC, 0xBB, 0xAA, 0x99, 0x00, 0xFF, 0xEE, 0xDD],),
            ]
        );
        assert_eq!(maxwell.regs.constbuf_load_offset, 16);
        assert!(maxwell.regs.pending_constbuf_writes.is_empty());
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn passive_maxwell_run_preserves_register_order_and_payload_state() {
        let mut pusher = Pusher::new();
        let mappings = GpuMappings::new();
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let header = (1 << 29) | (3 << 16) | 0x200;

        pusher.process_commands(
            &[header, 0x12, 0x3456, 0x789a],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(maxwell.reg_file[0x200], 0x12);
        assert_eq!(maxwell.reg_file[0x201], 0x3456);
        assert_eq!(maxwell.reg_file[0x202], 0x789a);
        assert_eq!(maxwell.regs.rt[0].address_hi, 0x12);
        assert_eq!(maxwell.regs.rt[0].address_lo, 0x3456);
        assert_eq!(maxwell.regs.rt[0].width, 0x789a);
        assert_eq!(pusher.state.method, 0x203);
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn passive_maxwell_run_drains_deferred_constbuf_writes_first() {
        let mut pusher = Pusher::new();
        pusher.passive_bulk_override = Some(true);
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 4, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        maxwell
            .regs
            .pending_constbuf_writes
            .push((0x5000, 0x1122_3344));
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let writes = Mutex::new(Vec::new());

        pusher.process_commands(
            &[(1 << 29) | (3 << 16) | 0x200, 0x12, 0x3456, 0x789a],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &|cpu, bytes| {
                writes.lock().unwrap().push((cpu, bytes.to_vec()));
                true
            },
            &|_, _, _| false,
        );

        assert_eq!(
            *writes.lock().unwrap(),
            vec![(0x9000, vec![0x44, 0x33, 0x22, 0x11])]
        );
        assert!(maxwell.regs.pending_constbuf_writes.is_empty());
        assert_eq!(maxwell.reg_file[0x200], 0x12);
        assert_eq!(maxwell.reg_file[0x201], 0x3456);
        assert_eq!(maxwell.reg_file[0x202], 0x789a);
        assert_eq!(pusher.passive_bulk_words, 2);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn passive_maxwell_run_drains_draw_upload_and_sync_work_first() {
        let mut pusher = Pusher::new();
        pusher.passive_bulk_override = Some(true);
        let mut mappings = GpuMappings::new();
        mappings.add(0x6000, 4, 0xa000, 1);
        mappings.add(0x7000, 4, 0xb000, 2);
        let mut maxwell = Maxwell3D::new();
        maxwell.dispatch_method(0x35e, 3, true);
        maxwell.write_register(0x60, 4);
        maxwell.regs.pending_semaphore_acquires.push((0x7000, 0, 0));
        maxwell.regs.pending_semaphore_writes.push(
            super::super::engines::maxwell3d::PendingSemaphoreWrite {
                gpu_va: 0x6000,
                payload: 0x5566_7788,
                long: false,
                ordering:
                    super::super::engines::maxwell3d::SemaphoreWriteOrdering::SyntheticCounter,
            },
        );
        maxwell.regs.pending_barrier_flushes = 2;
        maxwell.regs.pending_fragment_barriers = 1;
        maxwell.regs.pending_tiled_cache_barriers = 1;
        maxwell.regs.pending_texture_cache_invalidates = 1;
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let writes = Mutex::new(Vec::new());

        pusher.process_commands(
            &[(1 << 29) | (3 << 16) | 0x200, 0x12, 0x3456, 0x789a],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &|cpu, bytes| {
                writes.lock().unwrap().push((cpu, bytes.to_vec()));
                true
            },
            &|_, _, _| false,
        );

        assert!(!maxwell.has_pending_pusher_work());
        assert_eq!(maxwell.reg_file[0x200], 0x12);
        assert_eq!(maxwell.reg_file[0x201], 0x3456);
        assert_eq!(maxwell.reg_file[0x202], 0x789a);
        assert_eq!(pusher.passive_bulk_words, 2);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
        assert_eq!(
            *writes.lock().unwrap(),
            vec![(0xa000, vec![0x88, 0x77, 0x66, 0x55])]
        );
    }

    #[test]
    fn passive_maxwell_run_preserves_cross_subchannel_hard_boundary() {
        let mut pusher = Pusher::new();
        pusher.passive_bulk_override = Some(true);
        pusher.puller.semaphore_addr_low = 0x6000;
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 4, 0x9000, 1);
        mappings.add(0x6000, 4, 0xa000, 2);
        let mut maxwell = Maxwell3D::new();
        maxwell
            .regs
            .pending_constbuf_writes
            .push((0x5000, 0x1122_3344));
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let writes = Mutex::new(Vec::new());
        let passive_header = (1 << 29) | (3 << 16) | 0x200;
        let release_header = (1 << 29) | (1 << 16) | (7 << 13) | METHOD_SEMAPHORE_RELEASE;

        pusher.process_commands(
            &[
                passive_header,
                0x12,
                0x3456,
                0x789a,
                release_header,
                0x99aa_bbcc,
            ],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| true,
            &|cpu, bytes| {
                writes.lock().unwrap().push((cpu, bytes.to_vec()));
                true
            },
            &|_, _, _| false,
        );

        assert_eq!(pusher.passive_bulk_words, 2);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 4);
        assert_eq!(
            *writes.lock().unwrap(),
            vec![
                (0x9000, vec![0x44, 0x33, 0x22, 0x11]),
                (0xa000, vec![0xcc, 0xbb, 0xaa, 0x99]),
            ]
        );
    }

    #[test]
    fn fragmented_constbuf_run_invalidates_later_cpu_alias() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 4, 0x8000, 1);
        mappings.add(0x1004, 4, 0xa000, 2);
        mappings.add(0x3000, 4, 0xa000, 2);
        let key = crate::gpu::vk_dispatch::SsboSnapshotCacheKey {
            storage_binding: 0,
            descriptor_binding: 0,
            descriptor_offset: 0,
            descriptor_align: 4,
            descriptor_indirect: false,
            descriptor_size: 4,
            guest_addr: 0x3000,
            logical_size: 4,
            data_offset: 0,
            read_len: 4,
        };
        let reads = std::cell::Cell::new(0usize);
        let read = |_: u64, dst: &mut [u8]| {
            reads.set(reads.get() + 1);
            dst.fill(reads.get() as u8);
            true
        };
        let first = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, 0xa000, &read)
            .unwrap();
        pusher.inline_prep().commit_constbuf_writes(
            &[(0x1000, 0x1122_3344), (0x1004, 0x5566_7788)],
            None,
            &mappings,
            &|_, _| true,
        );
        let second = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, 0xa000, &read)
            .unwrap();

        assert_eq!(reads.get(), 2);
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn macro_argument_refresh_reads_the_live_pushbuffer_word() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 0x1000, 0x9000, 1);
        let expected = 0x1234_5678u32;
        let read = |cpu_addr: u64, out: &mut [u8]| {
            if cpu_addr != 0x9020 || out.len() != 4 {
                return false;
            }
            out.copy_from_slice(&expected.to_le_bytes());
            true
        };

        assert_eq!(read_live_word(&mappings, 0x5020, &read), Some(expected));
    }

    #[test]
    fn macro_argument_refresh_preserves_snapshot_when_live_read_fails() {
        let mappings = GpuMappings::new();
        let read = |_: u64, _: &mut [u8]| false;
        assert_eq!(read_live_word(&mappings, 0x5020, &read), None);
    }

    #[test]
    fn macro_argument_refresh_rejects_a_split_word_mapping() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 2, 0x9000, 1);
        mappings.add(0x5002, 2, 0xa000, 2);
        let reads = std::cell::Cell::new(0);
        let read = |_: u64, _: &mut [u8]| {
            reads.set(reads.get() + 1);
            true
        };

        assert_eq!(read_live_word(&mappings, 0x5000, &read), None);
        assert_eq!(reads.get(), 0);
    }

    #[test]
    fn contiguous_macro_batch_preserves_packet_semantics() {
        for (mode, method, non_incrementing, increment_once) in [
            (1u32, MACRO_REGISTERS_START + 3, false, false),
            (3, MACRO_REGISTERS_START, true, false),
            (5, MACRO_REGISTERS_START + 1, true, true),
        ] {
            let mut pusher = Pusher::new();
            let mappings = GpuMappings::new();
            let mut maxwell = Maxwell3D::new();
            install_three_param_echo(&mut maxwell);
            let mut maxwell_dma = MaxwellDma::new();
            let mut fermi_2d = Fermi2D::new();
            let mut kepler_compute = KeplerCompute::new();
            let mut kepler_memory = KeplerMemory::new();
            let stats = PipelineStats::default();
            let header = (mode << 29) | (3 << 16) | MACRO_REGISTERS_START;

            pusher.process_commands(
                &[header, 0x11, 0x22, 0x33],
                &mappings,
                &mut maxwell,
                &mut maxwell_dma,
                &mut fermi_2d,
                &mut kepler_compute,
                &mut kepler_memory,
                &stats,
                &|_, _| panic!("unmapped macro payload was read"),
                &|_, _| true,
                &|_, _, _| false,
            );

            assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
            assert_eq!(pusher.state.method, method);
            assert_eq!(pusher.state.method_count, 0);
            assert_eq!(pusher.state.non_incrementing, non_incrementing);
            assert_eq!(pusher.state.increment_once, increment_once);
            assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
        }
    }

    #[test]
    fn contiguous_macro_batch_reads_live_payload_once() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 16, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let header = (3u32 << 29) | (3 << 16) | MACRO_REGISTERS_START;
        let backing = Mutex::new(words_bytes(&[header, 1, 2, 3]));
        let live = words_bytes(&[0x11, 0x22, 0x33]);
        let reads = Mutex::new(Vec::new());
        let mem_read = |cpu_addr: u64, out: &mut [u8]| {
            reads.lock().unwrap().push((cpu_addr, out.len()));
            let mut backing = backing.lock().unwrap();
            let offset = cpu_addr.saturating_sub(0x9000) as usize;
            let Some(source) = backing.get(offset..offset + out.len()) else {
                return false;
            };
            out.copy_from_slice(source);
            if cpu_addr == 0x9000 && out.len() == 16 {
                backing[4..16].copy_from_slice(&live);
            }
            true
        };

        pusher.process_entry(
            &gpfifo_entry(0x5000, 4, false),
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
        assert_eq!(*reads.lock().unwrap(), vec![(0x9000, 16), (0x9004, 12)]);
        assert!(pusher.live_macro_values.capacity() >= 2);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn macro_batch_preserves_snapshot_for_split_and_failed_reads() {
        let mut pusher = Pusher::new();
        pusher.active_entry_gpu_va = 0x5000;
        let mut mappings = GpuMappings::new();
        mappings.add(0x5004, 2, 0x9000, 1);
        mappings.add(0x5006, 2, 0xA000, 2);
        mappings.add(0x5008, 4, 0xB000, 3);
        mappings.add(0x500C, 4, 0xC000, 4);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let reads = Mutex::new(Vec::new());
        let mem_read = |cpu_addr: u64, out: &mut [u8]| {
            reads.lock().unwrap().push((cpu_addr, out.len()));
            if cpu_addr == 0xB000 {
                return false;
            }
            if cpu_addr == 0xC000 && out.len() == 4 {
                out.copy_from_slice(&0x33u32.to_le_bytes());
                return true;
            }
            false
        };
        let header = (3u32 << 29) | (3 << 16) | MACRO_REGISTERS_START;

        pusher.process_commands(
            &[header, 0x11, 0x22, 0x03],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
        assert_eq!(*reads.lock().unwrap(), vec![(0xB000, 4), (0xC000, 4)]);
    }

    #[test]
    fn macro_batch_bulk_failure_retries_each_live_word() {
        let mut pusher = Pusher::new();
        pusher.active_entry_gpu_va = 0x5000;
        let mut mappings = GpuMappings::new();
        mappings.add(0x5004, 12, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let reads = Mutex::new(Vec::new());
        let live = words_bytes(&[0x11, 0x22, 0x33]);
        let mem_read = |cpu_addr: u64, out: &mut [u8]| {
            reads.lock().unwrap().push((cpu_addr, out.len()));
            if out.len() != 4 {
                return false;
            }
            let offset = cpu_addr.saturating_sub(0x9000) as usize;
            let Some(source) = live.get(offset..offset + 4) else {
                return false;
            };
            out.copy_from_slice(source);
            true
        };
        let header = (3u32 << 29) | (3 << 16) | MACRO_REGISTERS_START;

        pusher.process_commands(
            &[header, 1, 2, 3],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
        assert_eq!(
            *reads.lock().unwrap(),
            vec![(0x9000, 12), (0x9000, 4), (0x9004, 4), (0x9008, 4)]
        );
    }

    #[test]
    fn increase_once_batches_only_non_incrementing_remainder() {
        let mut pusher = Pusher::new();
        pusher.active_entry_gpu_va = 0x5000;
        let mut mappings = GpuMappings::new();
        mappings.add(0x5004, 12, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let live = words_bytes(&[0x11, 0x22, 0x33]);
        let reads = Mutex::new(Vec::new());
        let mem_read = |cpu_addr: u64, out: &mut [u8]| {
            reads.lock().unwrap().push((cpu_addr, out.len()));
            let offset = cpu_addr.saturating_sub(0x9000) as usize;
            let Some(source) = live.get(offset..offset + out.len()) else {
                return false;
            };
            out.copy_from_slice(source);
            true
        };
        let header = (5u32 << 29) | (3 << 16) | MACRO_REGISTERS_START;

        pusher.process_commands(
            &[header, 1, 2, 3],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
        assert_eq!(*reads.lock().unwrap(), vec![(0x9000, 4), (0x9004, 8)]);
        assert_eq!(pusher.state.method, MACRO_REGISTERS_START + 1);
        assert!(pusher.state.non_incrementing);
        assert!(pusher.state.increment_once);
    }

    #[test]
    fn inline_macro_argument_skips_live_refresh() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 4, 0x9000, 1);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let reads = std::cell::Cell::new(0usize);
        let header = (4u32 << 29) | (0x1234 << 16) | MACRO_REGISTERS_START;

        pusher.process_commands(
            &[header],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &|_, _| {
                reads.set(reads.get() + 1);
                true
            },
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(reads.get(), 0);
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn macro_packet_continues_across_gpfifo_entries() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 8, 0x9000, 1);
        mappings.add(0x6000, 8, 0xA000, 2);
        let mut maxwell = Maxwell3D::new();
        install_three_param_echo(&mut maxwell);
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let header = (3u32 << 29) | (3 << 16) | MACRO_REGISTERS_START;
        let first_snapshot = words_bytes(&[header, 1]);
        let second_snapshot = words_bytes(&[2, 3]);
        let second_live = words_bytes(&[0x22, 0x33]);
        let reads = Mutex::new(Vec::new());
        let mem_read = |cpu_addr: u64, out: &mut [u8]| {
            let call = {
                let mut reads = reads.lock().unwrap();
                let call = reads.len();
                reads.push((cpu_addr, out.len()));
                call
            };
            match (cpu_addr, out.len(), call) {
                (0x9000, 8, 0) => out.copy_from_slice(&first_snapshot),
                (0x9004, 4, 1) => out.copy_from_slice(&0x11u32.to_le_bytes()),
                (0xA000, 8, 2) => out.copy_from_slice(&second_snapshot),
                (0xA000, 8, 3) => out.copy_from_slice(&second_live),
                _ => return false,
            }
            true
        };

        pusher.process_entry(
            &gpfifo_entry(0x5000, 2, false),
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method_count, 2);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 1);

        pusher.process_entry(
            &gpfifo_entry(0x6000, 2, false),
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &|_, _| true,
            &|_, _, _| false,
        );

        assert_eq!(&maxwell.reg_file[0x200..0x203], &[0x11, 0x22, 0x33]);
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 3);
        assert_eq!(
            *reads.lock().unwrap(),
            vec![(0x9000, 8), (0x9004, 4), (0xA000, 8), (0xA000, 8)]
        );
    }

    #[test]
    fn puller_boundary_only_flushes_work_emitting_methods() {
        assert!(!puller_method_requires_hard_boundary(METHOD_BIND_OBJECT));
        assert!(!puller_method_requires_hard_boundary(
            METHOD_SEMAPHORE_ADDR_LOW
        ));
        assert!(!puller_method_requires_hard_boundary(
            METHOD_SEMAPHORE_PAYLOAD
        ));
        assert!(puller_method_requires_hard_boundary(
            METHOD_SEMAPHORE_OPERATION
        ));
        assert!(puller_method_requires_hard_boundary(
            METHOD_SEMAPHORE_ACQUIRE
        ));
        assert!(puller_method_requires_hard_boundary(
            METHOD_SEMAPHORE_RELEASE
        ));
        assert!(puller_method_requires_hard_boundary(
            METHOD_SYNCPOINT_OPERATION
        ));
    }

    #[test]
    fn ssbo_snapshot_cache_survives_empty_read_only_flush() {
        let mut pusher = Pusher::new();
        let key = crate::gpu::vk_dispatch::SsboSnapshotCacheKey {
            storage_binding: 0,
            descriptor_binding: 2,
            descriptor_offset: 0,
            descriptor_align: 16,
            descriptor_indirect: false,
            descriptor_size: 3 * 1024 * 1024,
            guest_addr: 0x7000_0000,
            logical_size: 3 * 1024 * 1024,
            data_offset: 0,
            read_len: 16,
        };
        let read = |_: u64, dst: &mut [u8]| {
            dst.fill(0x7b);
            true
        };
        let write = |_: u64, _: &[u8]| true;
        let mappings = GpuMappings::new();

        let before_flush = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, 0x8000_0000, &read)
            .unwrap();
        pusher.inline_prep().flush_vk(&mappings, &read, &write);
        assert!(!pusher.inline_prep().ssbo_snapshot_cache.is_empty());

        let after_flush = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, 0x8000_0000, &read)
            .unwrap();
        assert!(Arc::ptr_eq(&before_flush, &after_flush));
    }

    #[test]
    fn entry_boundary_drops_partial_non_watchable_ssbo_snapshot() {
        let mut pusher = Pusher::new();
        let key = crate::gpu::vk_dispatch::SsboSnapshotCacheKey {
            storage_binding: 0,
            descriptor_binding: 2,
            descriptor_offset: 0,
            descriptor_align: 16,
            descriptor_indirect: false,
            descriptor_size: 3 * 1024 * 1024,
            guest_addr: 0x7000_0000,
            logical_size: 3 * 1024 * 1024,
            data_offset: 0,
            read_len: 16,
        };
        let read = |_: u64, dst: &mut [u8]| {
            dst.fill(0x7b);
            true
        };

        pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, 0x8000_0000, &read)
            .unwrap();
        assert!(!pusher.inline_prep().ssbo_snapshot_cache.is_empty());

        pusher.inline_prep().begin_ssbo_snapshot_entry();
        assert!(pusher.inline_prep().ssbo_snapshot_cache.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn entry_boundary_retains_write_watched_full_aurora_snapshot() {
        const CPU_VA: u64 = 0xea_0000_0000;
        const LEN: usize = 3 * 1024 * 1024;

        let ptr = nexium_memory::fastmem::commit(CPU_VA, LEN).expect("fastmem test arena");
        unsafe { std::ptr::write_bytes(ptr, 0x6d, LEN) };
        let read = |cpu_addr: u64, dst: &mut [u8]| {
            assert_eq!(cpu_addr, CPU_VA);
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
            true
        };
        let key = crate::gpu::vk_dispatch::SsboSnapshotCacheKey {
            storage_binding: 0,
            descriptor_binding: 2,
            descriptor_offset: 0,
            descriptor_align: 16,
            descriptor_indirect: false,
            descriptor_size: LEN as u32,
            guest_addr: 0x7000_0000,
            logical_size: LEN,
            data_offset: 0,
            read_len: LEN,
        };
        let mut pusher = Pusher::new();

        let before_entry = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, CPU_VA, &read)
            .unwrap();
        assert!(!pusher.inline_prep().ssbo_snapshot_cache.is_empty());

        pusher.inline_prep().begin_ssbo_snapshot_entry();

        assert!(!pusher.inline_prep().ssbo_snapshot_cache.is_empty());
        let after_entry = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, CPU_VA, &read)
            .unwrap();
        assert!(Arc::ptr_eq(&before_entry, &after_entry));

        unsafe { ptr.add(0x1234).write_volatile(0xa7) };
        pusher.inline_prep().begin_ssbo_snapshot_entry();
        assert!(pusher.inline_prep().ssbo_snapshot_cache.is_empty());

        let after_cpu_write = pusher
            .inline_prep()
            .ssbo_snapshot_cache
            .read_or_insert(key, CPU_VA, &read)
            .unwrap();
        assert!(!Arc::ptr_eq(&before_entry, &after_cpu_write));
        assert_eq!(after_cpu_write[0x1234], 0xa7);
        nexium_memory::fastmem::decommit(ptr, LEN);
    }

    #[test]
    fn split_non_incrementing_upload_preserves_payload_state() {
        let mut pusher = Pusher::new();
        let mappings = GpuMappings::new();
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let mem_read = |_: u64, _: &mut [u8]| true;
        let mem_write = |_: u64, _: &[u8]| true;

        let mut first = vec![0x8100_0000; 9];
        first.push(0x6300_006D);
        pusher.process_commands(
            &first,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method, 0x6D);
        assert_eq!(pusher.state.method_count, 768);
        assert!(pusher.state.non_incrementing);

        let payload: Vec<u32> = (0..768).map(|i| 0x1000_0000 | i).collect();
        pusher.process_commands(
            &payload,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method_count, 0);

        pusher.process_commands(
            &[0x2001_0100, 0x1234_5678],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method, 0x101);
        assert_eq!(pusher.state.method_count, 0);
    }

    #[test]
    fn payload_spans_gpfifo_entry_boundaries_by_default() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let mem_read = |_: u64, _: &mut [u8]| true;
        let mem_write = |_: u64, _: &[u8]| true;
        let first_entry = CommandListHeader {
            address_lo: 0x4000,
            address_hi_and_count: 1 << 10,
        };
        let second_entry = CommandListHeader {
            address_lo: 0x4004,
            address_hi_and_count: 1 << 10,
        };
        pusher.state.method = 0x101;
        pusher.state.subchannel = 7;
        pusher.state.method_count = 2;

        pusher.process_entry(
            &first_entry,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method_count, 1);

        pusher.process_entry(
            &second_entry,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        assert_eq!(pusher.state.method_count, 0);
        assert_eq!(stats.methods_dispatched.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn maxwell3d_inline_upload_writes_mapped_bytes() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x6000, 0x100, 0x1000, 1);
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let memory = Arc::new(Mutex::new(vec![0u8; 0x100]));
        let read_memory = memory.clone();
        let mem_read = move |cpu: u64, out: &mut [u8]| {
            let memory = read_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(out.len()) > memory.len() {
                return false;
            }
            out.copy_from_slice(&memory[start..start + out.len()]);
            true
        };
        let write_memory = memory.clone();
        let mem_write = move |cpu: u64, data: &[u8]| {
            let mut memory = write_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(data.len()) > memory.len() {
                return false;
            }
            memory[start..start + data.len()].copy_from_slice(data);
            true
        };

        let setup = [0x200D_0060, 6, 1, 0, 0x6000, 6, 0, 6, 1, 1, 0, 0, 0, 1];
        pusher.process_commands(
            &setup,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );
        pusher.process_commands(
            &[0x6002_006D, 0x1122_3344, 0x5566],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );

        let memory = memory.lock().unwrap();
        assert_eq!(&memory[..6], &[0x44, 0x33, 0x22, 0x11, 0x66, 0x55]);
        assert_eq!(maxwell.reg_file[0x47], 0);
        assert_eq!(maxwell.reg_file[0x48], 0);
    }

    #[test]
    fn nvk_implicit_copy_subchannel_executes_dma_without_set_object() {
        let mut pusher = Pusher::new();
        assert_eq!(pusher.bound_classes[4], MAXWELL_DMA_CLASS);

        let mut mappings = GpuMappings::new();
        mappings.add(0x6000, 0x200, 0x1000, 1);
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let memory = Arc::new(Mutex::new(vec![0u8; 0x200]));
        memory.lock().unwrap()[..4].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);

        let read_memory = memory.clone();
        let mem_read = move |cpu: u64, out: &mut [u8]| {
            let memory = read_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(out.len()) > memory.len() {
                return false;
            }
            out.copy_from_slice(&memory[start..start + out.len()]);
            true
        };
        let write_memory = memory.clone();
        let mem_write = move |cpu: u64, data: &[u8]| {
            let mut memory = write_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(data.len()) > memory.len() {
                return false;
            }
            memory[start..start + data.len()].copy_from_slice(data);
            true
        };

        let setup_header = (1 << 29) | (8 << 16) | (4 << 13) | 0x100;
        let launch_header = (4 << 29) | (0x180 << 16) | (4 << 13) | 0xC0;
        pusher.process_commands(
            &[
                setup_header,
                0,
                0x6000,
                0,
                0x6100,
                4,
                4,
                4,
                1,
                launch_header,
            ],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
            &|_, _, _| false,
        );

        let memory = memory.lock().unwrap();
        assert_eq!(&memory[0x100..0x104], &[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(maxwell_dma.blit_count, 1);
    }
}
