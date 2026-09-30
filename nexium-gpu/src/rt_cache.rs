use ash::vk;
use nexium_common::fast_hash::{FastMap as HashMap, FastSet as HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

const RT_COLOR_GPU_PAGE_SHIFT: u32 = 20;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RtAliasIndexedLookupTelemetry {
    pub exact_hits: u64,
    pub exact_misses: u64,
    pub wildcard_fallbacks: u64,
    pub cpu_candidates: u64,
    pub shadow_mismatches: u64,
}

static RT_ALIAS_INDEXED_EXACT_HITS: AtomicU64 = AtomicU64::new(0);
static RT_ALIAS_INDEXED_EXACT_MISSES: AtomicU64 = AtomicU64::new(0);
static RT_ALIAS_INDEXED_WILDCARD_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static RT_ALIAS_INDEXED_CPU_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static RT_ALIAS_INDEXED_SHADOW_MISMATCHES: AtomicU64 = AtomicU64::new(0);

type RtSampleableColorAlias = (RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format);
type RtSampleableDepthAlias = (
    RtKey,
    vk::Image,
    vk::ImageView,
    vk::ImageLayout,
    vk::Format,
    vk::ImageAspectFlags,
);

fn rt_alias_indexed_lookup_value_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0"
            || value.eq_ignore_ascii_case("false")
            || value.eq_ignore_ascii_case("off")
            || value.eq_ignore_ascii_case("no")
    })
}

fn rt_alias_indexed_lookup_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        rt_alias_indexed_lookup_value_enabled(
            std::env::var("NEXIUM_RT_ALIAS_INDEXED_LOOKUP")
                .ok()
                .as_deref(),
        )
    })
}

fn rt_alias_indexed_lookup_shadow_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_RT_ALIAS_INDEXED_LOOKUP_SHADOW")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn rt_alias_indexed_lookup_telemetry_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        rt_alias_indexed_lookup_shadow_enabled()
            || std::env::var("NEXIUM_RT_ALIAS_INDEXED_LOOKUP_PROFILE")
                .ok()
                .as_deref()
                == Some("1")
    })
}

#[inline]
fn count_rt_alias_indexed(counter: &AtomicU64, value: u64) {
    if value != 0 && rt_alias_indexed_lookup_telemetry_enabled() {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

fn rt_sampleable_color_alias_equal(
    indexed: &Option<RtSampleableColorAlias>,
    legacy: &Option<RtSampleableColorAlias>,
) -> bool {
    match (indexed, legacy) {
        (Some(indexed), Some(legacy)) => {
            indexed.0.same_live_identity(legacy.0)
                && indexed.1 == legacy.1
                && indexed.2 == legacy.2
                && indexed.3 == legacy.3
                && indexed.4 == legacy.4
        }
        (None, None) => true,
        _ => false,
    }
}

fn rt_sampleable_depth_alias_equal(
    indexed: &Option<RtSampleableDepthAlias>,
    legacy: &Option<RtSampleableDepthAlias>,
) -> bool {
    match (indexed, legacy) {
        (Some(indexed), Some(legacy)) => {
            indexed.0.same_live_identity(legacy.0)
                && indexed.1 == legacy.1
                && indexed.2 == legacy.2
                && indexed.3 == legacy.3
                && indexed.4 == legacy.4
                && indexed.5 == legacy.5
        }
        (None, None) => true,
        _ => false,
    }
}

#[inline]
fn rt_alias_shadow_color_result(
    indexed: Option<RtSampleableColorAlias>,
    legacy: Option<RtSampleableColorAlias>,
) -> Option<RtSampleableColorAlias> {
    if !rt_sampleable_color_alias_equal(&indexed, &legacy) {
        count_rt_alias_indexed(&RT_ALIAS_INDEXED_SHADOW_MISMATCHES, 1);
    }
    legacy
}

#[inline]
fn rt_alias_shadow_depth_result(
    indexed: Option<RtSampleableDepthAlias>,
    legacy: Option<RtSampleableDepthAlias>,
) -> Option<RtSampleableDepthAlias> {
    if !rt_sampleable_depth_alias_equal(&indexed, &legacy) {
        count_rt_alias_indexed(&RT_ALIAS_INDEXED_SHADOW_MISMATCHES, 1);
    }
    legacy
}

pub fn take_rt_alias_indexed_lookup_telemetry() -> RtAliasIndexedLookupTelemetry {
    if !rt_alias_indexed_lookup_telemetry_enabled() {
        return RtAliasIndexedLookupTelemetry::default();
    }
    RtAliasIndexedLookupTelemetry {
        exact_hits: RT_ALIAS_INDEXED_EXACT_HITS.swap(0, Ordering::Relaxed),
        exact_misses: RT_ALIAS_INDEXED_EXACT_MISSES.swap(0, Ordering::Relaxed),
        wildcard_fallbacks: RT_ALIAS_INDEXED_WILDCARD_FALLBACKS.swap(0, Ordering::Relaxed),
        cpu_candidates: RT_ALIAS_INDEXED_CPU_CANDIDATES.swap(0, Ordering::Relaxed),
        shadow_mismatches: RT_ALIAS_INDEXED_SHADOW_MISMATCHES.swap(0, Ordering::Relaxed),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RtKey {
    pub nvmap_id: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub is_3d: bool,
    pub array_stride_bytes: u64,
    pub sample_width: u8,
    pub sample_height: u8,
    pub base_layer: u32,
    pub gpu_va: u64,
    pub cpu_addr: u64,
    pub mapping_epoch: u64,
    pub guest_size_bytes: u64,
    pub layout_signature: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RtMappingEpochTransition {
    pub gpu_va: u64,
    pub size: u64,
    pub old_epoch: u64,
    pub new_epoch: u64,
}

impl PartialEq for RtKey {
    fn eq(&self, other: &Self) -> bool {
        self.nvmap_id == other.nvmap_id
            && self.width == other.width
            && self.height == other.height
            && self.depth == other.depth
            && self.is_3d == other.is_3d
            && self.sample_width == other.sample_width
            && self.sample_height == other.sample_height
            && self.gpu_va == other.gpu_va
    }
}

impl Eq for RtKey {}

impl Hash for RtKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut folded = self.gpu_va ^ (u64::from(self.nvmap_id) << 40);
        folded = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (u64::from(self.width) | (u64::from(self.height) << 32));
        folded = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (u64::from(self.depth)
                | (u64::from(self.is_3d) << 32)
                | (u64::from(self.sample_width) << 40)
                | (u64::from(self.sample_height) << 48));
        state.write_u64(folded);
    }
}

impl RtKey {
    pub fn new(nvmap_id: u32, width: u32, height: u32, gpu_va: u64) -> Self {
        Self::with_cpu(nvmap_id, width, height, gpu_va, 0)
    }

    pub fn with_cpu(nvmap_id: u32, width: u32, height: u32, gpu_va: u64, cpu_addr: u64) -> Self {
        Self {
            nvmap_id,
            width,
            height,
            depth: 1,
            is_3d: false,
            array_stride_bytes: 0,
            sample_width: 1,
            sample_height: 1,
            base_layer: 0,
            gpu_va,
            cpu_addr,
            mapping_epoch: 0,
            guest_size_bytes: 0,
            layout_signature: 0,
        }
    }

    pub fn with_mapping_epoch(mut self, epoch: u64) -> Self {
        self.mapping_epoch = epoch;
        self
    }

    pub fn with_guest_size_bytes(mut self, bytes: u64) -> Self {
        self.guest_size_bytes = bytes;
        self
    }

    pub fn with_block_linear_layout(
        mut self,
        block_width_log2: u32,
        block_height_log2: u32,
        block_depth_log2: u32,
        tile_width_spacing: u32,
    ) -> Self {
        self.layout_signature = 1
            | (u64::from(block_width_log2 & 0xff) << 8)
            | (u64::from(block_height_log2 & 0xff) << 16)
            | (u64::from(block_depth_log2 & 0xff) << 24)
            | (u64::from(tile_width_spacing & 0xff) << 32);
        self
    }

    pub fn with_pitch_linear_layout(mut self, pitch_bytes: u32) -> Self {
        self.layout_signature = 2 | (u64::from(pitch_bytes) << 8);
        self
    }

    pub fn with_volume_depth(mut self, depth: u32) -> Self {
        self.depth = depth.max(1);
        self.is_3d = self.depth > 1;
        self
    }

    pub fn with_array_layers(mut self, layers: u32, stride_bytes: u64) -> Self {
        self.depth = layers.max(1);
        self.is_3d = false;
        self.array_stride_bytes = stride_bytes;
        self
    }

    pub fn with_sample_grid(mut self, width: u32, height: u32) -> Self {
        self.sample_width = u8::try_from(width.max(1)).unwrap_or(u8::MAX);
        self.sample_height = u8::try_from(height.max(1)).unwrap_or(u8::MAX);
        self
    }

    pub fn with_base_layer(mut self, base_layer: u32) -> Self {
        self.base_layer = base_layer;
        self
    }

    pub fn sample_grid(self) -> (u32, u32) {
        (u32::from(self.sample_width), u32::from(self.sample_height))
    }

    pub fn same_live_identity(self, other: Self) -> bool {
        self == other
            && self.array_stride_bytes == other.array_stride_bytes
            && self.base_layer == other.base_layer
            && self.cpu_addr == other.cpu_addr
            && self.mapping_epoch == other.mapping_epoch
            && self.guest_size_bytes == other.guest_size_bytes
            && self.layout_signature == other.layout_signature
    }

    pub fn same_alias_view_identity(self, other: Self) -> bool {
        self == other
            && self.array_stride_bytes == other.array_stride_bytes
            && self.base_layer == other.base_layer
            && self.cpu_addr == other.cpu_addr
            && self.mapping_epoch == other.mapping_epoch
            && self.layout_signature == other.layout_signature
    }

    pub fn render_layer_count(self) -> u32 {
        self.depth.max(1)
    }

    pub fn request(nvmap_id: u32, width: u32, height: u32) -> Self {
        Self::new(nvmap_id, width, height, 0)
    }

    pub fn label(self) -> String {
        if self.gpu_va != 0 {
            if self.is_3d {
                format!(
                    "{}:{}x{}x{}@{:x}/{}x{}s",
                    self.nvmap_id,
                    self.width,
                    self.height,
                    self.depth,
                    self.gpu_va,
                    self.sample_width,
                    self.sample_height
                )
            } else {
                format!(
                    "{}:{}x{}@{:x}/{}x{}s",
                    self.nvmap_id,
                    self.width,
                    self.height,
                    self.gpu_va,
                    self.sample_width,
                    self.sample_height
                )
            }
        } else if self.is_3d {
            format!(
                "{}:{}x{}x{}",
                self.nvmap_id, self.width, self.height, self.depth
            )
        } else {
            format!("{}:{}x{}", self.nvmap_id, self.width, self.height)
        }
    }
}

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub views: std::collections::HashMap<vk::Format, vk::ImageView>,
    sample_views: HashMap<RtSampleViewKey, vk::ImageView>,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub base_format: vk::Format,
    pub aspects: vk::ImageAspectFlags,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RtSampleViewKey {
    format: i32,
    aspect: u32,
    view_type: i32,
    components: [i32; 4],
}

impl RtSampleViewKey {
    fn new(
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
        components: vk::ComponentMapping,
    ) -> Self {
        Self {
            format: format.as_raw(),
            aspect: aspect.as_raw(),
            view_type: view_type.as_raw(),
            components: [
                components.r.as_raw(),
                components.g.as_raw(),
                components.b.as_raw(),
                components.a.as_raw(),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RtColorRegion {
    pub key: RtKey,
    pub image: vk::Image,
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub stamp: u64,
    pub src_x: u32,
    pub src_y: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RtDepthRegion {
    pub key: RtKey,
    pub image: vk::Image,
    pub layout: vk::ImageLayout,
    pub generation: u64,
}

fn dims_close(a: u32, b: u32) -> bool {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    lo != 0 && hi <= lo.saturating_mul(2)
}

pub fn is_synthetic_copy_key(key: RtKey) -> bool {
    key.nvmap_id == u32::MAX
}

pub fn same_physical_backing(a: RtKey, b: RtKey) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.depth == b.depth
        && a.is_3d == b.is_3d
        && a.sample_width == b.sample_width
        && a.sample_height == b.sample_height
        && a.nvmap_id == b.nvmap_id
        && a.cpu_addr != 0
        && a.cpu_addr == b.cpu_addr
        && metadata_equal_when_known(a.mapping_epoch, b.mapping_epoch)
        && metadata_equal_when_known(a.layout_signature, b.layout_signature)
}

fn metadata_equal_when_known(a: u64, b: u64) -> bool {
    a == 0 || b == 0 || a == b
}

fn exact_texture_alias_identity(candidate: RtKey, want: RtKey) -> bool {
    if is_synthetic_copy_key(candidate)
        || candidate.nvmap_id != want.nvmap_id
        || candidate.width != want.width
        || candidate.height != want.height
        || candidate.depth != want.depth
        || candidate.is_3d != want.is_3d
        || candidate.sample_width != want.sample_width
        || candidate.sample_height != want.sample_height
        || candidate.base_layer != want.base_layer
        || (want.mapping_epoch != 0 && candidate.mapping_epoch != want.mapping_epoch)
        || (want.layout_signature != 0 && candidate.layout_signature != want.layout_signature)
    {
        return false;
    }
    if want.gpu_va != 0 && candidate.gpu_va != want.gpu_va {
        return false;
    }
    want.cpu_addr == 0 || candidate.cpu_addr == want.cpu_addr
}

fn alias_view_metadata_changed(stored: RtKey, want: RtKey) -> bool {
    stored.base_layer != want.base_layer
        || (want.cpu_addr != 0 && stored.cpu_addr != want.cpu_addr)
        || (want.mapping_epoch != 0 && stored.mapping_epoch != want.mapping_epoch)
        || (want.layout_signature != 0 && stored.layout_signature != want.layout_signature)
}

fn color_sync_row_stride(key: RtKey, format: vk::Format) -> Option<u64> {
    let bpp = usize::try_from(rt_format_bytes(format)).ok()?;
    let size = u64::try_from(crate::texture::native_render_target_source_size(
        key.width,
        key.height,
        key.depth,
        bpp,
        key.layout_signature,
    )?).ok()?;
    let (height, depth) = match key.layout_signature & 0xff {
        1 => {
            let block_height = 8u64.checked_shl(((key.layout_signature >> 16) & 0xff) as u32)?;
            let block_depth = 1u64.checked_shl(((key.layout_signature >> 24) & 0xff) as u32)?;
            (
                u64::from(key.height).div_ceil(block_height).checked_mul(block_height)?,
                u64::from(key.depth).div_ceil(block_depth).checked_mul(block_depth)?,
            )
        }
        2 => (u64::from(key.height), u64::from(key.depth)),
        _ => return None,
    };
    size.checked_div(height)?.checked_div(depth)
}

fn color_sync_layout_compatible(candidate: RtKey, want: RtKey, format: vk::Format) -> bool {
    candidate.layout_signature == 0
        || want.layout_signature == 0
        || (candidate.layout_signature == want.layout_signature
            && color_sync_row_stride(candidate, format)
                .zip(color_sync_row_stride(want, format))
                .is_some_and(|(source, destination)| source == destination))
}

fn color_alias_sync_identity(candidate: RtKey, want: RtKey) -> bool {
    !alias_view_metadata_changed(candidate, want)
        && candidate.sample_width == 1
        && candidate.sample_height == 1
        && want.sample_width == 1
        && want.sample_height == 1
}

fn color_region_sync_identity(candidate: RtKey, want: RtKey) -> bool {
    let same_cpu_mapping = candidate.cpu_addr == 0
        || want.cpu_addr == 0
        || want.gpu_va.checked_sub(candidate.gpu_va)
            .and_then(|offset| candidate.cpu_addr.checked_add(offset)) == Some(want.cpu_addr);
    same_cpu_mapping
        && candidate.nvmap_id == want.nvmap_id
        && candidate.base_layer == want.base_layer
        && (want.mapping_epoch == 0 || candidate.mapping_epoch == want.mapping_epoch)
        && (want.layout_signature == 0 || candidate.layout_signature == want.layout_signature)
        && candidate.sample_width == 1
        && candidate.sample_height == 1
        && want.sample_width == 1
        && want.sample_height == 1
}

fn render_target_backing_changed(stored: RtKey, want: RtKey) -> bool {
    alias_view_metadata_changed(stored, want)
}

fn same_d24_depth_allocation_covering(existing: RtKey, want: RtKey) -> bool {
    let existing_row = (existing.width as u64 * 4).next_multiple_of(64);
    let wanted_row = (want.width as u64 * 4).next_multiple_of(64);
    want.gpu_va != 0
        && existing.nvmap_id == want.nvmap_id
        && existing.gpu_va == want.gpu_va
        && metadata_equal_when_known(existing.mapping_epoch, want.mapping_epoch)
        && metadata_equal_when_known(existing.layout_signature, want.layout_signature)
        && existing.sample_width == want.sample_width
        && existing.sample_height == want.sample_height
        && existing.base_layer == want.base_layer
        && existing.depth == want.depth
        && existing.is_3d == want.is_3d
        && existing.width >= want.width
        && existing.height == want.height
        && existing_row == wanted_row
}

pub struct RtCache {
    cache: HashMap<RtKey, GpuImage>,
    color_gpu_base_index: HashMap<u64, Vec<RtKey>>,
    color_cpu_base_index: HashMap<u64, Vec<RtKey>>,
    color_gpu_page_index: HashMap<u64, Vec<RtKey>>,
    color_nvmap_index: HashMap<u32, Vec<RtKey>>,
    depth_cache: HashMap<RtKey, GpuImage>,
    depth_cpu_base_index: HashMap<u64, Vec<RtKey>>,
    depth_formats: crate::depth::DepthFormats,
    depth_pack_pipeline: Option<crate::depth::DepthPackPipeline>,
    snapshots: HashMap<RtKey, GpuImage>,
    mem_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    drawn_stamp: HashMap<RtKey, u64>,
    clear_stamp: HashMap<RtKey, u64>,
    last_use: std::sync::Mutex<HashMap<RtKey, (u64, u64)>>,
    use_counter: AtomicU64,
    use_frame: u64,
    full_clear_stamp: HashMap<RtKey, u64>,
    depth_full_clear_generation: HashMap<RtKey, u64>,
    color_region_sources: HashMap<RtKey, (RtKey, vk::Image, u64)>,
    depth_region_sources: HashMap<RtKey, (RtKey, vk::Image, u64)>,
    pending_view_retirement: HashSet<(bool, RtKey)>,
    obsolete_mapping_views: HashSet<(bool, RtKey)>,
    view_retirement_epoch: u64,
    guest_stale_color: HashSet<RtKey>,
    guest_stale_depth: HashSet<RtKey>,
    present_excluded: HashSet<RtKey>,
    present_flip_y: HashMap<RtKey, bool>,
    drawn_counter: u64,
    structure_generation: u64,
    color_sampling_generation: u64,
    color_sampling_versions: Option<HashMap<u32, u64>>,
    frame_draws: HashMap<RtKey, u32>,
    frame_real_draws: HashMap<RtKey, u32>,
    depth_frame_draws: HashMap<RtKey, u32>,
    depth_generation_counter: u64,
    depth_generations: HashMap<RtKey, u64>,
    depth_shadow_generations: HashMap<RtKey, (RtKey, u64)>,
    depth_array_shadow_generations: HashMap<RtKey, Vec<(RtKey, u64)>>,
    guest_cpu_lo: u64,
    guest_cpu_hi: u64,
    guest_gpu_lo: u64,
    guest_gpu_hi: u64,
    color_guest_ranges: Vec<GuestRangeEntry>,
    color_guest_range_index: HashMap<RtKey, usize>,
    depth_guest_ranges: Vec<GuestRangeEntry>,
    depth_guest_range_index: HashMap<RtKey, usize>,
    guest_range_epoch: u64,
    guest_range_lookup: GuestRangeLookup,
    guest_hit_memo: Vec<GuestHitMemoEntry>,
}

const GUEST_HIT_MEMO_SIZE: usize = 64;

struct GuestHitMemoEntry {
    epoch: u64,
    cpu_addr: u64,
    cpu_end: u64,
    gpu_ranges: Vec<(u64, u64)>,
    color: Vec<RtKey>,
    depth: Vec<RtKey>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct GuestRangeEntry {
    key: RtKey,
    cpu_lo: u64,
    cpu_hi: u64,
    gpu_lo: u64,
    gpu_hi: u64,
}

#[derive(Default)]
struct GuestRangeLookup {
    epoch: Option<u64>,
    cpu: Vec<GuestRangeLookupEntry>,
    gpu: Vec<GuestRangeLookupEntry>,
}

struct GuestRangeLookupEntry {
    key: RtKey,
    depth: bool,
    lo: u64,
    hi: u64,
    prefix_hi: u64,
}

impl GuestRangeLookup {
    fn rebuild(&mut self, epoch: u64, color: &[GuestRangeEntry], depth: &[GuestRangeEntry]) {
        self.cpu.clear();
        self.gpu.clear();
        for (entry, depth) in color
            .iter()
            .map(|e| (e, false))
            .chain(depth.iter().map(|e| (e, true)))
        {
            for (index, lo, hi) in [
                (&mut self.cpu, entry.cpu_lo, entry.cpu_hi),
                (&mut self.gpu, entry.gpu_lo, entry.gpu_hi),
            ] {
                if lo < hi {
                    index.push(GuestRangeLookupEntry {
                        key: entry.key,
                        depth,
                        lo,
                        hi,
                        prefix_hi: 0,
                    });
                }
            }
        }
        for index in [&mut self.cpu, &mut self.gpu] {
            index.sort_unstable_by_key(|entry| entry.lo);
            let mut prefix_hi = 0;
            for entry in index {
                prefix_hi = prefix_hi.max(entry.hi);
                entry.prefix_hi = prefix_hi;
            }
        }
        self.epoch = Some(epoch);
    }

    fn hits(
        &self,
        cpu_addr: u64,
        cpu_end: u64,
        gpu_ranges: &[(u64, u64)],
    ) -> (Vec<RtKey>, Vec<RtKey>) {
        let mut color = Vec::new();
        let mut depth = Vec::new();
        let mut collect = |index: &[GuestRangeLookupEntry], lo: u64, hi: u64| {
            if lo >= hi {
                return;
            }
            let mut end = index.partition_point(|entry| entry.lo < hi);
            while end > 0 && index[end - 1].prefix_hi > lo {
                end -= 1;
                let entry = &index[end];
                if entry.hi > lo {
                    let hits = if entry.depth { &mut depth } else { &mut color };
                    if !hits.contains(&entry.key) {
                        hits.push(entry.key);
                    }
                }
            }
        };
        collect(&self.cpu, cpu_addr, cpu_end);
        for &(gpu_va, size) in gpu_ranges {
            collect(&self.gpu, gpu_va, gpu_va.saturating_add(size));
        }
        (color, depth)
    }
}

fn rt_guest_size_bytes(key: RtKey, format: vk::Format) -> Option<u64> {
    let tight_physical_size = if key.layout_signature & 0xff == 2 {
        (key.layout_signature >> 8)
            .checked_mul(u64::from(key.height))?
            .checked_mul(u64::from(key.sample_height.max(1)))?
            .checked_mul(u64::from(key.render_layer_count()))?
    } else {
        u64::from(key.width)
            .checked_mul(u64::from(key.sample_width.max(1)))?
            .checked_mul(u64::from(key.height))?
            .checked_mul(u64::from(key.sample_height.max(1)))?
            .checked_mul(u64::from(key.render_layer_count()))?
            .checked_mul(rt_format_bytes(format))?
    };
    Some(tight_physical_size.max(key.guest_size_bytes))
}

fn guest_range_entry(key: RtKey, base_format: vk::Format) -> Option<GuestRangeEntry> {
    let image_size = rt_guest_size_bytes(key, base_format)?;
    if image_size == 0 {
        return None;
    }
    let (cpu_lo, cpu_hi) = if key.cpu_addr != 0 {
        (key.cpu_addr, key.cpu_addr.saturating_add(image_size))
    } else {
        (0, 0)
    };
    let (gpu_lo, gpu_hi) = if key.gpu_va != 0 {
        (key.gpu_va, key.gpu_va.saturating_add(image_size))
    } else {
        (0, 0)
    };
    Some(GuestRangeEntry {
        key,
        cpu_lo,
        cpu_hi,
        gpu_lo,
        gpu_hi,
    })
}

fn insert_guest_range(
    ranges: &mut Vec<GuestRangeEntry>,
    index: &mut HashMap<RtKey, usize>,
    key: RtKey,
    base_format: vk::Format,
) -> bool {
    let Some(entry) = guest_range_entry(key, base_format) else {
        return false;
    };
    if let Some(&pos) = index.get(&key) {
        if ranges[pos] == entry {
            return false;
        }
        ranges[pos] = entry;
    } else {
        index.insert(key, ranges.len());
        ranges.push(entry);
    }
    true
}

fn remove_guest_range(
    ranges: &mut Vec<GuestRangeEntry>,
    index: &mut HashMap<RtKey, usize>,
    key: RtKey,
) -> bool {
    let Some(pos) = index.remove(&key) else {
        return false;
    };
    ranges.swap_remove(pos);
    if pos < ranges.len() {
        index.insert(ranges[pos].key, pos);
    }
    true
}

#[cfg(test)]
fn guest_range_hits(
    ranges: &[GuestRangeEntry],
    cpu_addr: u64,
    cpu_end: u64,
    gpu_ranges: &[(u64, u64)],
) -> Vec<RtKey> {
    ranges
        .iter()
        .filter_map(|entry| {
            let cpu_hit = entry.cpu_hi != 0 && cpu_addr < entry.cpu_hi && entry.cpu_lo < cpu_end;
            let gpu_hit = entry.gpu_hi != 0
                && gpu_ranges.iter().any(|&(gpu_va, gpu_size)| {
                    gpu_size != 0
                        && gpu_va < entry.gpu_hi
                        && entry.gpu_lo < gpu_va.saturating_add(gpu_size)
                });
            (cpu_hit || gpu_hit).then_some(entry.key)
        })
        .collect()
}

fn gpu_ranges_overlap(a_gpu_va: u64, a_size: u64, b_gpu_va: u64, b_size: u64) -> bool {
    a_size != 0
        && b_size != 0
        && a_gpu_va < b_gpu_va.saturating_add(b_size)
        && b_gpu_va < a_gpu_va.saturating_add(a_size)
}

fn transitioned_rt_key(
    key: RtKey,
    base_format: vk::Format,
    transitions: &[RtMappingEpochTransition],
    changed_gpu_ranges: &[(u64, u64)],
) -> Option<RtKey> {
    if is_synthetic_copy_key(key) || key.gpu_va == 0 || key.mapping_epoch == 0 {
        return None;
    }
    let image_size = rt_guest_size_bytes(key, base_format)?;
    if image_size == 0
        || changed_gpu_ranges
            .iter()
            .any(|&(gpu_va, size)| gpu_ranges_overlap(key.gpu_va, image_size, gpu_va, size))
    {
        return None;
    }
    let transition = transitions.iter().find(|transition| {
        transition.size != 0
            && transition.old_epoch != 0
            && transition.new_epoch != 0
            && transition.old_epoch != transition.new_epoch
            && key.mapping_epoch == transition.old_epoch
            && key.gpu_va >= transition.gpu_va
            && key.gpu_va < transition.gpu_va.saturating_add(transition.size)
    })?;
    Some(key.with_mapping_epoch(transition.new_epoch))
}

fn rekey_hash_map_value<V>(map: &mut HashMap<RtKey, V>, old: RtKey, new: RtKey) {
    if let Some((_, value)) = map.remove_entry(&old) {
        map.insert(new, value);
    }
}

fn rekey_hash_set(set: &mut HashSet<RtKey>, old: RtKey, new: RtKey) {
    if set.take(&old).is_some() {
        set.insert(new);
    }
}

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::default(),
            color_gpu_base_index: HashMap::default(),
            color_cpu_base_index: HashMap::default(),
            color_nvmap_index: HashMap::default(),
            color_gpu_page_index: HashMap::default(),
            depth_cache: HashMap::default(),
            depth_cpu_base_index: HashMap::default(),
            last_use: std::sync::Mutex::new(HashMap::default()),
            use_counter: AtomicU64::new(0),
            use_frame: 0,
            depth_formats: crate::depth::DepthFormats::default(),
            depth_pack_pipeline: None,
            snapshots: HashMap::default(),
            mem_properties: None,
            drawn_stamp: HashMap::default(),
            clear_stamp: HashMap::default(),
            full_clear_stamp: HashMap::default(),
            depth_full_clear_generation: HashMap::default(),
            color_region_sources: HashMap::default(),
            depth_region_sources: HashMap::default(),
            pending_view_retirement: HashSet::default(),
            obsolete_mapping_views: HashSet::default(),
            view_retirement_epoch: 0,
            guest_stale_color: HashSet::default(),
            guest_stale_depth: HashSet::default(),
            present_excluded: HashSet::default(),
            present_flip_y: HashMap::default(),
            drawn_counter: 0,
            structure_generation: 0,
            color_sampling_generation: 0,
            color_sampling_versions: (cfg!(test)
                || crate::texture_mips::resolved_rt_mip_memo_enabled()
                || crate::texture_mips::rt_mip_memo_enabled()).then(HashMap::default),
            frame_draws: HashMap::default(),
            frame_real_draws: HashMap::default(),
            depth_frame_draws: HashMap::default(),
            depth_generation_counter: 0,
            depth_generations: HashMap::default(),
            depth_shadow_generations: HashMap::default(),
            depth_array_shadow_generations: HashMap::default(),
            guest_cpu_lo: u64::MAX,
            guest_cpu_hi: 0,
            guest_gpu_lo: u64::MAX,
            guest_gpu_hi: 0,
            color_guest_ranges: Vec::new(),
            color_guest_range_index: HashMap::default(),
            depth_guest_ranges: Vec::new(),
            depth_guest_range_index: HashMap::default(),
            guest_range_epoch: 0,
            guest_range_lookup: GuestRangeLookup::default(),
            guest_hit_memo: Vec::new(),
        }
    }

    fn note_guest_ranges_changed(&mut self) {
        self.guest_range_epoch = self.guest_range_epoch.wrapping_add(1);
    }

    fn guest_range_hits_memoized(
        &mut self,
        cpu_addr: u64,
        cpu_end: u64,
        gpu_ranges: &[(u64, u64)],
    ) -> (Vec<RtKey>, Vec<RtKey>) {
        let epoch = self.guest_range_epoch;
        if let Some(memo) = self.guest_hit_memo.iter().find(|memo| {
            memo.epoch == epoch
                && memo.cpu_addr == cpu_addr
                && memo.cpu_end == cpu_end
                && memo.gpu_ranges == gpu_ranges
        }) {
            return (memo.color.clone(), memo.depth.clone());
        }
        if self.guest_range_lookup.epoch != Some(epoch) {
            self.guest_range_lookup.rebuild(
                epoch,
                &self.color_guest_ranges,
                &self.depth_guest_ranges,
            );
        }
        let (color, depth) = self.guest_range_lookup.hits(cpu_addr, cpu_end, gpu_ranges);
        if self.guest_hit_memo.len() >= GUEST_HIT_MEMO_SIZE {
            self.guest_hit_memo.remove(0);
        }
        self.guest_hit_memo.push(GuestHitMemoEntry {
            epoch,
            cpu_addr,
            cpu_end,
            gpu_ranges: gpu_ranges.to_vec(),
            color: color.clone(),
            depth: depth.clone(),
        });
        (color, depth)
    }

    fn index_color_lookup(&mut self, key: RtKey, format: vk::Format) {
        if is_synthetic_copy_key(key) {
            return;
        }
        let base = self.color_gpu_base_index.entry(key.gpu_va).or_default();
        if !base.contains(&key) {
            base.push(key);
        }
        if key.cpu_addr != 0 {
            let base = self.color_cpu_base_index.entry(key.cpu_addr).or_default();
            if !base.contains(&key) {
                base.push(key);
            }
        }
        let by_nvmap = self.color_nvmap_index.entry(key.nvmap_id).or_default();
        if !by_nvmap.contains(&key) {
            by_nvmap.push(key);
        }
        let Some((first_page, last_page)) = rt_color_gpu_page_span(key, format) else {
            return;
        };
        for page in first_page..=last_page {
            let bucket = self.color_gpu_page_index.entry(page).or_default();
            if !bucket.contains(&key) {
                bucket.push(key);
            }
        }
    }

    fn unindex_color_lookup(&mut self, key: RtKey, format: vk::Format) {
        if is_synthetic_copy_key(key) {
            return;
        }
        remove_lookup_key(&mut self.color_gpu_base_index, key.gpu_va, key);
        if key.cpu_addr != 0 {
            remove_lookup_key(&mut self.color_cpu_base_index, key.cpu_addr, key);
        }
        remove_lookup_key(&mut self.color_nvmap_index, key.nvmap_id, key);
        let Some((first_page, last_page)) = rt_color_gpu_page_span(key, format) else {
            return;
        };
        for page in first_page..=last_page {
            remove_lookup_key(&mut self.color_gpu_page_index, page, key);
        }
    }

    fn extend_guest_bounds(&mut self, key: RtKey, base_format: vk::Format) {
        let Some(image_size) = rt_guest_size_bytes(key, base_format) else {
            return;
        };
        if image_size == 0 {
            return;
        }
        if key.cpu_addr != 0 {
            self.guest_cpu_lo = self.guest_cpu_lo.min(key.cpu_addr);
            self.guest_cpu_hi = self
                .guest_cpu_hi
                .max(key.cpu_addr.saturating_add(image_size));
        }
        if key.gpu_va != 0 {
            self.guest_gpu_lo = self.guest_gpu_lo.min(key.gpu_va);
            self.guest_gpu_hi = self.guest_gpu_hi.max(key.gpu_va.saturating_add(image_size));
        }
    }

    pub fn adopt_external_color(
        &mut self,
        key: RtKey,
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        format: vk::Format,
        extent: vk::Extent2D,
        layout: vk::ImageLayout,
    ) -> bool {
        if is_synthetic_copy_key(key) || self.cache.contains_key(&key) {
            return false;
        }
        let mut views = std::collections::HashMap::new();
        views.insert(format, view);
        let adopted = GpuImage {
            image,
            view,
            views,
            sample_views: HashMap::default(),
            memory,
            format,
            base_format: format,
            aspects: vk::ImageAspectFlags::COLOR,
            extent,
            layout,
        };
        self.insert_color_image(key, adopted);
        self.mark_drawn(key);
        true
    }

    fn insert_color_image(&mut self, key: RtKey, image: GpuImage) {
        static PROFILE: OnceLock<bool> = OnceLock::new();
        static SMALL_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
        if *PROFILE.get_or_init(|| std::env::var_os("NEXIUM_RT_SMALL_ALLOC_PROFILE").is_some())
            && u64::from(key.width).saturating_mul(u64::from(key.height)) <= 4096
        {
            let sequence = SMALL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 32 || sequence % 128 == 0 {
                log::info!("[rt-small-alloc] sequence={sequence} key={key:?} format={:?}", image.format);
            }
        }
        self.note_color_sampling_changed(key.nvmap_id);
        self.structure_generation = self.structure_generation.wrapping_add(1);
        let format = image.format;
        debug_assert!(!self.cache.contains_key(&key));
        self.extend_guest_bounds(key, image.base_format);
        if insert_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            key,
            image.base_format,
        ) {
            self.note_guest_ranges_changed();
        }
        self.cache.insert(key, image);
        let canonical = self
            .cache
            .get_key_value(&key)
            .map(|(stored, _)| *stored)
            .unwrap();
        self.index_color_lookup(canonical, format);
    }

    fn rekey_color_state(&mut self, stored: RtKey, key: RtKey) -> bool {
        self.note_color_sampling_changed(key.nvmap_id);
        if stored != key {
            return false;
        }
        let indexed_key = self
            .color_guest_range_index
            .get(&stored)
            .and_then(|&position| self.color_guest_ranges.get(position))
            .map(|entry| entry.key)
            .unwrap_or(stored);
        let Some((canonical, image)) = self.cache.remove_entry(&stored) else {
            return false;
        };
        self.unindex_color_lookup(indexed_key, image.format);
        if remove_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            canonical,
        ) {
            self.note_guest_ranges_changed();
        }
        rekey_hash_map_value(&mut self.snapshots, canonical, key);
        rekey_hash_map_value(&mut self.drawn_stamp, canonical, key);
        rekey_hash_map_value(&mut self.clear_stamp, canonical, key);
        rekey_hash_map_value(&mut self.full_clear_stamp, canonical, key);
        self.rekey_retirement_state(false, canonical, key);
        rekey_hash_set(&mut self.guest_stale_color, canonical, key);
        rekey_hash_set(&mut self.present_excluded, canonical, key);
        rekey_hash_map_value(&mut self.present_flip_y, canonical, key);
        rekey_hash_map_value(&mut self.frame_draws, canonical, key);
        rekey_hash_map_value(&mut self.frame_real_draws, canonical, key);
        let format = image.format;
        let base_format = image.base_format;
        self.cache.insert(key, image);
        self.extend_guest_bounds(key, base_format);
        if insert_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            key,
            base_format,
        ) {
            self.note_guest_ranges_changed();
        }
        self.index_color_lookup(key, format);
        true
    }

    fn insert_depth_image(&mut self, key: RtKey, image: GpuImage) {
        self.depth_cache.insert(key, image);
        let stored = *self.depth_cache.get_key_value(&key).unwrap().0;
        if stored.cpu_addr != 0 {
            let keys = self.depth_cpu_base_index.entry(stored.cpu_addr).or_default();
            if !keys.contains(&stored) {
                keys.push(stored);
            }
        }
    }

    fn remove_depth_image(&mut self, key: RtKey) -> Option<(RtKey, GpuImage)> {
        let (stored, image) = self.depth_cache.remove_entry(&key)?;
        remove_lookup_key(&mut self.depth_cpu_base_index, stored.cpu_addr, stored);
        Some((stored, image))
    }

    fn rekey_depth_state(&mut self, stored: RtKey, key: RtKey) -> bool {
        if stored != key {
            return false;
        }
        let Some((canonical, image)) = self.remove_depth_image(stored) else {
            return false;
        };
        if remove_guest_range(
            &mut self.depth_guest_ranges,
            &mut self.depth_guest_range_index,
            canonical,
        ) {
            self.note_guest_ranges_changed();
        }
        rekey_hash_set(&mut self.guest_stale_depth, canonical, key);
        rekey_hash_map_value(&mut self.depth_generations, canonical, key);
        rekey_hash_map_value(&mut self.depth_full_clear_generation, canonical, key);
        self.rekey_retirement_state(true, canonical, key);
        let shadow_generations = std::mem::take(&mut self.depth_shadow_generations);
        for (mut shadow, (mut source, generation)) in shadow_generations {
            if shadow == canonical {
                shadow = key;
            }
            if source == canonical {
                source = key;
            }
            self.depth_shadow_generations
                .insert(shadow, (source, generation));
        }
        let base_format = image.base_format;
        self.insert_depth_image(key, image);
        self.extend_guest_bounds(key, base_format);
        if insert_guest_range(
            &mut self.depth_guest_ranges,
            &mut self.depth_guest_range_index,
            key,
            base_format,
        ) {
            self.note_guest_ranges_changed();
        }
        true
    }

    fn refresh_color_footprint(&mut self, stored: RtKey, key: RtKey) {
        if key.guest_size_bytes == 0 || stored.guest_size_bytes == key.guest_size_bytes {
            return;
        }
        self.rekey_color_state(stored, key);
    }

    fn refresh_depth_footprint(&mut self, stored: RtKey, key: RtKey) -> RtKey {
        if stored != key
            || key.guest_size_bytes == 0
            || stored.guest_size_bytes == key.guest_size_bytes
        {
            return stored;
        }
        self.rekey_depth_state(stored, key)
            .then_some(key)
            .unwrap_or(stored)
    }

    pub fn apply_mapping_epoch_transitions(
        &mut self,
        transitions: &[RtMappingEpochTransition],
        changed_gpu_ranges: &[(u64, u64)],
    ) {
        if !transitions.is_empty() {
            let color_rekeys = self
                .cache
                .iter()
                .filter_map(|(stored, image)| {
                    transitioned_rt_key(*stored, image.base_format, transitions, changed_gpu_ranges)
                        .map(|key| (*stored, key))
                })
                .collect::<Vec<_>>();
            for (stored, key) in color_rekeys {
                self.rekey_color_state(stored, key);
            }

            let depth_rekeys = self
                .depth_cache
                .iter()
                .filter_map(|(stored, image)| {
                    transitioned_rt_key(*stored, image.base_format, transitions, changed_gpu_ranges)
                        .map(|key| (*stored, key))
                })
                .collect::<Vec<_>>();
            for (stored, key) in depth_rekeys {
                self.rekey_depth_state(stored, key);
            }
        }
        self.mark_guest_written_range(0, 0, changed_gpu_ranges);
        if changed_gpu_ranges.is_empty() {
            return;
        }
        let (colors, depths) = self.guest_range_hits_memoized(0, 0, changed_gpu_ranges);
        for (depth, keys) in [(false, colors), (true, depths)] {
            self.obsolete_mapping_views.extend(keys.into_iter().filter_map(|key| {
                (key.mapping_epoch != 0
                    && key.gpu_va != 0
                    && !is_synthetic_copy_key(key)
                    && changed_gpu_ranges.iter().any(|&(base, size)| {
                        size != 0 && key.gpu_va >= base && key.gpu_va - base < size
                    }))
                .then_some((depth, key))
            }));
        }
    }

    fn remove_color_image(&mut self, key: RtKey) -> Option<(RtKey, GpuImage)> {
        self.note_color_sampling_changed(key.nvmap_id);
        self.structure_generation = self.structure_generation.wrapping_add(1);
        let indexed_key = self
            .color_guest_range_index
            .get(&key)
            .and_then(|&position| self.color_guest_ranges.get(position))
            .map(|entry| entry.key);
        let (canonical, image) = self.cache.remove_entry(&key)?;
        self.clear_stamp.remove(&canonical);
        self.full_clear_stamp.remove(&canonical);
        self.color_region_sources.retain(|key, (source, _, _)| *key != canonical && *source != canonical);
        self.pending_view_retirement.remove(&(false, canonical));
        self.obsolete_mapping_views.remove(&(false, canonical));
        self.unindex_color_lookup(indexed_key.unwrap_or(canonical), image.format);
        if remove_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            canonical,
        ) {
            self.note_guest_ranges_changed();
        }
        Some((canonical, image))
    }

    fn clear_color_lookup_index(&mut self) {
        self.color_gpu_base_index.clear();
        self.color_cpu_base_index.clear();
        self.color_gpu_page_index.clear();
        self.color_nvmap_index.clear();
    }

    fn canonical_depth_key(&self, key: RtKey) -> RtKey {
        if let Some((stored, _)) = self.depth_cache.get_key_value(&key) {
            return *stored;
        }
        self.depth_cache
            .keys()
            .copied()
            .find(|existing| same_physical_backing(*existing, key))
            .unwrap_or(key)
    }

    fn next_depth_generation(&mut self) -> u64 {
        self.depth_generation_counter = self.depth_generation_counter.wrapping_add(1).max(1);
        self.depth_generation_counter
    }

    fn forget_depth_tracking(&mut self, key: RtKey) {
        let canonical = self.canonical_depth_key(key);
        self.depth_generations.remove(&canonical);
        self.depth_full_clear_generation.remove(&canonical);
        self.depth_region_sources.retain(|key, (source, _, _)| *key != canonical && *source != canonical);
        self.pending_view_retirement.remove(&(true, canonical));
        self.obsolete_mapping_views.remove(&(true, canonical));
        self.guest_stale_depth.remove(&canonical);
        self.depth_shadow_generations.retain(|shadow, (source, _)| {
            *shadow != canonical
                && *source != canonical
                && !same_physical_backing(*source, canonical)
        });
    }

    pub(crate) fn mark_depth_written(&mut self, key: RtKey) -> u64 {
        let canonical = self.canonical_depth_key(key);
        self.guest_stale_depth.remove(&canonical);
        self.depth_region_sources.remove(&canonical);
        *self.depth_frame_draws.entry(canonical).or_insert(0) += 1;
        let generation = self.next_depth_generation();
        self.depth_generations.insert(canonical, generation);
        generation
    }

    pub(crate) fn mark_depth_synced_from(&mut self, key: RtKey, generation: u64) {
        let canonical = self.canonical_depth_key(key);
        self.guest_stale_depth.remove(&canonical);
        self.depth_full_clear_generation.remove(&canonical);
        self.depth_region_sources.remove(&canonical);
        self.depth_generations.insert(canonical, generation);
        self.depth_generation_counter = self.depth_generation_counter.max(generation);
    }

    pub(crate) fn mark_depth_cleared(&mut self, key: RtKey, full_target: bool) -> u64 {
        let generation = self.mark_depth_written(key);
        let canonical = self.canonical_depth_key(key);
        if full_target {
            self.depth_full_clear_generation.insert(canonical, generation);
            self.pending_view_retirement.insert((true, canonical));
        }
        generation
    }

    pub(crate) fn mark_depth_region_synced(&mut self, key: RtKey, source: RtKey, generation: u64) {
        self.mark_depth_synced_from(key, generation);
        let key = self.canonical_depth_key(key);
        let source = self.canonical_depth_key(source);
        if key != source {
            if let Some(image) = self.depth_cache.get(&source) {
                self.depth_region_sources.insert(key, (source, image.image, self.view_retirement_epoch));
                self.pending_view_retirement.insert((true, key));
            }
        }
    }

    pub(crate) fn depth_generation(&self, key: RtKey) -> Option<u64> {
        let canonical = self.canonical_depth_key(key);
        self.depth_generations.get(&canonical).copied()
    }

    pub(crate) fn depth_shadow_is_current(&self, source: RtKey, shadow: RtKey) -> bool {
        let source = self.canonical_depth_key(source);
        let shadow = self.canonical_depth_key(shadow);
        self.depth_cache.contains_key(&source)
            && self.depth_cache.contains_key(&shadow)
            && self
                .depth_generations
                .get(&source)
                .is_some_and(|generation| {
                    self.depth_shadow_generations.get(&shadow) == Some(&(source, *generation))
                })
    }

    pub(crate) fn mark_depth_shadow_synced(&mut self, source: RtKey, shadow: RtKey) -> bool {
        let source = self.canonical_depth_key(source);
        let shadow = self.canonical_depth_key(shadow);
        let Some(generation) = self.depth_generations.get(&source).copied() else {
            self.depth_shadow_generations.remove(&shadow);
            return false;
        };
        self.depth_shadow_generations
            .insert(shadow, (source, generation));
        true
    }

    pub(crate) fn depth_array_shadow_is_current(&self, shadow: RtKey, sources: &[RtKey]) -> bool {
        let shadow = self.canonical_depth_key(shadow);
        if !self.depth_cache.contains_key(&shadow) {
            return false;
        }
        let Some(recorded) = self.depth_array_shadow_generations.get(&shadow) else {
            return false;
        };
        recorded.len() == sources.len()
            && recorded.iter().zip(sources).all(|((recorded_key, generation), source)| {
                let source = self.canonical_depth_key(*source);
                *recorded_key == source
                    && self.depth_cache.contains_key(&source)
                    && self.depth_generations.get(&source) == Some(generation)
            })
    }

    pub(crate) fn mark_depth_array_shadow_synced(&mut self, shadow: RtKey, sources: &[RtKey]) {
        let shadow = self.canonical_depth_key(shadow);
        let recorded = sources
            .iter()
            .map(|source| {
                let source = self.canonical_depth_key(*source);
                (source, self.depth_generations.get(&source).copied().unwrap_or(0))
            })
            .collect();
        self.depth_array_shadow_generations.insert(shadow, recorded);
    }

    pub fn find_depth_layer(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        self.find_sampleable_depth(want).or_else(|| {
            self.depth_cache
                .iter()
                .find(|(key, image)| {
                    key.nvmap_id == want.nvmap_id
                        && key.gpu_va == want.gpu_va
                        && key.width == want.width
                        && key.height == want.height
                        && key.depth == 1
                        && !key.is_3d
                        && image.layout != vk::ImageLayout::UNDEFINED
                        && !self.depth_is_guest_stale(**key)
                })
                .map(|(key, image)| {
                    (*key, image.image, image.view, image.layout, image.format, image.aspects)
                })
        })
    }

    pub fn debug_depth_resolution(&self, want: RtKey) -> String {
        use std::fmt::Write as _;
        let mut s = format!(
            "want={} nvmap={} exact_layout={:?}",
            want.label(),
            want.nvmap_id,
            self.depth_cache.get(&want).map(|i| i.layout)
        );
        for (k, img) in &self.depth_cache {
            let near = k.gpu_va == want.gpu_va
                || (k.nvmap_id == want.nvmap_id
                    && dims_close(k.width, want.width)
                    && dims_close(k.height, want.height));
            if near {
                let _ = write!(
                    s,
                    " | cand={} nvmap={} layout={:?} fmt={:?} gen={:?}",
                    k.label(),
                    k.nvmap_id,
                    img.layout,
                    img.format,
                    self.depth_generation(*k)
                );
            }
        }
        for (shadow, (source, gen)) in &self.depth_shadow_generations {
            if source.gpu_va == want.gpu_va || same_physical_backing(*source, want) {
                let _ = write!(
                    s,
                    " | shadow={} src={} gen={} current={}",
                    shadow.label(),
                    source.label(),
                    gen,
                    self.depth_shadow_is_current(*source, *shadow)
                );
            }
        }
        s
    }

    pub fn get_or_create_feedback_snapshot(
        &mut self,
        device: &ash::Device,
        key: RtKey,
    ) -> Result<(vk::Image, vk::ImageView, vk::Format), String> {
        let (native_format, extent) = {
            let live = self
                .cache
                .get(&key)
                .ok_or_else(|| format!("feedback snapshot: no live image for {}", key.label()))?;
            (live.base_format, live.extent)
        };
        let recreate = match self.snapshots.get(&key) {
            Some(s) => {
                s.base_format != native_format
                    || s.extent.width != extent.width
                    || s.extent.height != extent.height
            }
            None => true,
        };
        if recreate {
            let img = self.create_image(device, key, native_format)?;
            if let Some(old) = self.snapshots.insert(key, img) {
                destroy_gpu_image(device, old);
            }
        }
        let snap = self.snapshots.get(&key).unwrap();
        Ok((snap.image, snap.view, snap.base_format))
    }

    fn mark_drawn_watch() -> Option<(u32, u32)> {
        use std::sync::OnceLock;
        static V: OnceLock<Option<(u32, u32)>> = OnceLock::new();
        *V.get_or_init(|| {
            let spec = std::env::var("NEXIUM_MARK_DRAWN_WATCH").ok()?;
            let (w, h) = spec.trim().split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
    }

    pub fn mark_drawn(&mut self, key: RtKey) -> u64 {
        self.note_color_sampling_changed(key.nvmap_id);
        self.color_region_sources.remove(&key);
        self.clear_stamp.remove(&key);
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
        self.guest_stale_color.remove(&key);
        self.present_excluded.remove(&key);
        *self.frame_draws.entry(key).or_insert(0) += 1;
        *self.frame_real_draws.entry(key).or_insert(0) += 1;
        if Self::mark_drawn_watch().is_some_and(|(w, h)| key.width == w && key.height == h) {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n < 24 || n % 1000 == 0 {
                log::warn!(
                    "[mark-drawn] n={} key={} nvmap={} va={:#x} stamp={}",
                    n,
                    key.label(),
                    key.nvmap_id,
                    key.gpu_va,
                    self.drawn_counter
                );
            }
        }
        self.drawn_counter
    }

    pub fn record_present_flip(&mut self, key: RtKey, flip_y: bool) {
        self.present_flip_y.insert(key, flip_y);
    }

    pub fn present_flip_y(&self, key: RtKey) -> Option<bool> {
        self.present_flip_y.get(&key).copied()
    }

    pub fn mark_synced_sample(&mut self, key: RtKey) -> u64 {
        self.note_color_sampling_changed(key.nvmap_id);
        self.full_clear_stamp.remove(&key);
        self.color_region_sources.remove(&key);
        self.clear_stamp.remove(&key);
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
        self.guest_stale_color.remove(&key);
        self.present_excluded.insert(key);
        self.drawn_counter
    }

    pub fn mark_synced_sample_from(&mut self, key: RtKey, source_stamp: u64) -> u64 {
        self.note_color_sampling_changed(key.nvmap_id);
        self.full_clear_stamp.remove(&key);
        self.color_region_sources.remove(&key);
        debug_assert_ne!(source_stamp, 0);
        self.clear_stamp.remove(&key);
        self.drawn_counter = self.drawn_counter.max(source_stamp);
        self.drawn_stamp.insert(key, source_stamp);
        self.guest_stale_color.remove(&key);
        self.present_excluded.insert(key);
        source_stamp
    }

    pub(crate) fn mark_color_region_synced(&mut self, key: RtKey, source: RtKey, stamp: u64) -> u64 {
        self.mark_synced_sample_from(key, stamp);
        if key != source {
            if let Some((source, image)) = self.cache.get_key_value(&source) {
                self.color_region_sources.insert(key, (*source, image.image, self.view_retirement_epoch));
                self.pending_view_retirement.insert((false, key));
            }
        }
        stamp
    }

    pub fn mark_guest_uploaded(&mut self, key: RtKey) {
        self.note_color_sampling_changed(key.nvmap_id);
        let canonical = self
            .cache
            .get_key_value(&key)
            .map(|(stored, _)| *stored)
            .unwrap_or(key);
        self.drawn_stamp.remove(&canonical);
        self.clear_stamp.remove(&canonical);
        self.full_clear_stamp.remove(&canonical);
        self.color_region_sources.retain(|key, (source, _, _)| *key != canonical && *source != canonical);
        self.guest_stale_color.remove(&canonical);
        self.present_excluded.insert(canonical);
        self.frame_draws.remove(&canonical);
        self.frame_real_draws.remove(&canonical);
    }

    pub fn mark_guest_written(&mut self, key: RtKey) {
        self.note_color_sampling_changed(key.nvmap_id);
        let stale: Vec<_> = self
            .cache
            .keys()
            .filter(|existing| **existing == key || same_physical_backing(**existing, key))
            .copied()
            .collect();
        for stale_key in stale {
            self.drawn_stamp.remove(&stale_key);
            self.clear_stamp.remove(&stale_key);
            self.full_clear_stamp.remove(&stale_key);
            self.color_region_sources.retain(|key, (source, _, _)| *key != stale_key && *source != stale_key);
            self.guest_stale_color.insert(stale_key);
            self.present_excluded.insert(stale_key);
            self.frame_draws.remove(&stale_key);
            self.frame_real_draws.remove(&stale_key);
        }
        let stale_depth: Vec<_> = self
            .depth_cache
            .keys()
            .filter(|existing| **existing == key || same_physical_backing(**existing, key))
            .copied()
            .collect();
        for stale_key in stale_depth {
            self.mark_depth_written(stale_key);
            self.depth_full_clear_generation.remove(&stale_key);
            self.depth_region_sources.retain(|key, (source, _, _)| *key != stale_key && *source != stale_key);
            self.guest_stale_depth.insert(stale_key);
        }
    }

    pub fn mark_all_guest_written(&mut self) {
        self.color_sampling_generation = self.color_sampling_generation.wrapping_add(1);
        let stale_color = self.cache.keys().copied().collect::<Vec<_>>();
        for key in stale_color {
            self.drawn_stamp.remove(&key);
            self.clear_stamp.remove(&key);
            self.full_clear_stamp.remove(&key);
            self.color_region_sources.retain(|target, (source, _, _)| *target != key && *source != key);
            self.guest_stale_color.insert(key);
            self.present_excluded.insert(key);
            self.frame_draws.remove(&key);
            self.frame_real_draws.remove(&key);
        }
        let stale_depth = self.depth_cache.keys().copied().collect::<Vec<_>>();
        for key in stale_depth {
            self.mark_depth_written(key);
            self.depth_full_clear_generation.remove(&key);
            self.depth_region_sources.retain(|target, (source, _, _)| *target != key && *source != key);
            self.guest_stale_depth.insert(key);
        }
    }

    pub fn mark_guest_written_range(
        &mut self,
        cpu_addr: u64,
        size: u64,
        gpu_ranges: &[(u64, u64)],
    ) {
        fn rt_invalidate_trace_enabled() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_RT_INVALIDATE_TRACE").is_some())
        }
        let has_cpu_range = cpu_addr != 0 && size != 0;
        let has_gpu_range = gpu_ranges.iter().any(|(_, gpu_size)| *gpu_size != 0);
        if !has_cpu_range && !has_gpu_range {
            return;
        }
        let cpu_may_overlap = has_cpu_range
            && cpu_addr < self.guest_cpu_hi
            && self.guest_cpu_lo < cpu_addr.saturating_add(size);
        let gpu_may_overlap = gpu_ranges.iter().any(|(gpu_va, gpu_size)| {
            *gpu_va < self.guest_gpu_hi && self.guest_gpu_lo < gpu_va.saturating_add(*gpu_size)
        });
        if !cpu_may_overlap && !gpu_may_overlap {
            return;
        }
        let cpu_end = if has_cpu_range {
            cpu_addr.saturating_add(size)
        } else {
            0
        };
        let (stale_color, stale_depth) =
            self.guest_range_hits_memoized(cpu_addr, cpu_end, gpu_ranges);
        if !stale_color.is_empty() && rt_invalidate_trace_enabled() {
            log::warn!(
                "[rt-invalidate] cpu={:#x}+{:#x} gpu={:x?} stale=[{}]",
                cpu_addr,
                size,
                gpu_ranges,
                stale_color.iter().map(|key| key.label()).collect::<Vec<_>>().join(",")
            );
        }
        for stale_key in stale_color {
            self.note_color_sampling_changed(stale_key.nvmap_id);
            self.drawn_stamp.remove(&stale_key);
            self.clear_stamp.remove(&stale_key);
            self.full_clear_stamp.remove(&stale_key);
            self.color_region_sources.retain(|key, (source, _, _)| *key != stale_key && *source != stale_key);
            self.guest_stale_color.insert(stale_key);
            self.present_excluded.insert(stale_key);
            self.frame_draws.remove(&stale_key);
            self.frame_real_draws.remove(&stale_key);
        }
        for stale_key in stale_depth {
            self.mark_depth_written(stale_key);
            self.depth_full_clear_generation.remove(&stale_key);
            self.depth_region_sources.retain(|key, (source, _, _)| *key != stale_key && *source != stale_key);
            self.guest_stale_depth.insert(stale_key);
        }
    }

    pub fn color_is_guest_stale(&self, key: RtKey) -> bool {
        self.guest_stale_color.contains(&key) || self.obsolete_mapping_views.contains(&(false, key))
    }

    pub(crate) fn guest_stale_counts(&self) -> (usize, usize) {
        (self.guest_stale_color.len(), self.guest_stale_depth.len())
    }

    pub(crate) fn small_color_counts(&self) -> (usize, usize, usize, usize, usize) {
        self.cache.iter()
            .filter(|(key, _)| u64::from(key.width).saturating_mul(u64::from(key.height)) <= 4096)
            .fold((0, 0, 0, 0, 0), |(total, stale, stamped, undefined, excluded), (key, image)| {
                (
                    total + 1,
                    stale + usize::from(self.guest_stale_color.contains(key)),
                    stamped + usize::from(self.writeback_stamp(*key).is_some_and(|stamp| stamp > 0)),
                    undefined + usize::from(image.layout == vk::ImageLayout::UNDEFINED),
                    excluded + usize::from(self.present_excluded.contains(key)),
                )
            })
    }

    fn note_color_sampling_changed(&mut self, nvmap_id: u32) {
        let Some(versions) = self.color_sampling_versions.as_mut() else { return; };
        if versions.len() >= 4096 && !versions.contains_key(&nvmap_id) {
            versions.clear();
            self.color_sampling_generation = self.color_sampling_generation.wrapping_add(1);
        }
        let version = versions.entry(nvmap_id).or_default();
        *version = version.wrapping_add(1);
    }

    pub(crate) fn color_sampling_generation(&self, nvmap_id: u32) -> (u64, u64) {
        (self.color_sampling_generation, self.color_sampling_versions.as_ref()
            .and_then(|versions| versions.get(&nvmap_id)).copied().unwrap_or(0))
    }

    pub fn structure_generation(&self) -> u64 {
        self.structure_generation
    }

    pub fn resolve_drawn_color_exact(
        &self,
        key: RtKey,
    ) -> Option<(vk::Image, vk::ImageLayout, vk::Format, vk::Extent2D, u64)> {
        let (stored, image) = self.cache.get_key_value(&key)?;
        if self.color_is_guest_stale(*stored) {
            return None;
        }
        let stamp = *self.drawn_stamp.get(stored)?;
        Some((image.image, image.layout, image.format, image.extent, stamp))
    }

    pub fn depth_is_guest_stale(&self, key: RtKey) -> bool {
        self.guest_stale_depth.contains(&key) || self.obsolete_mapping_views.contains(&(true, key))
    }

    pub fn reset_frame_draws(&mut self) {
        self.use_frame = self.use_frame.wrapping_add(1);
        self.frame_draws.clear();
        self.frame_real_draws.clear();
        self.depth_frame_draws.clear();
    }

    pub fn present_depth_key(
        &self,
        color: RtKey,
    ) -> Option<(RtKey, u32, u32, vk::Format, vk::ImageAspectFlags, u32)> {
        let min_w = color.width / 2;
        let min_h = color.height / 2;
        let max_w = color.width.saturating_mul(2);
        let max_h = color.height.saturating_mul(2);
        let mut best: Option<(bool, u32, u64, RtKey, u32, u32, vk::Format, vk::ImageAspectFlags)> =
            None;
        for (key, draws) in &self.depth_frame_draws {
            if *draws == 0
                || self.obsolete_mapping_views.contains(&(true, *key))
                || key.depth != 1
                || key.is_3d
                || key.sample_width > 1
                || key.sample_height > 1
            {
                continue;
            }
            let Some(img) = self.depth_cache.get(key) else {
                continue;
            };
            if !img.aspects.contains(vk::ImageAspectFlags::DEPTH)
                || img.layout == vk::ImageLayout::UNDEFINED
            {
                continue;
            }
            let (w, h) = (img.extent.width, img.extent.height);
            if w == 0 || h == 0 || w < min_w || h < min_h || w > max_w || h > max_h {
                continue;
            }
            let exact = w == color.width && h == color.height;
            let area = u64::from(w) * u64::from(h);
            let better = best
                .as_ref()
                .map_or(true, |b| (exact, *draws, area) > (b.0, b.1, b.2));
            if better {
                best = Some((exact, *draws, area, *key, w, h, img.format, img.aspects));
            }
        }
        best.map(|(_, draws, _, key, w, h, format, aspects)| (key, w, h, format, aspects, draws))
    }

    pub fn depth_share_summary(&self) -> String {
        let mut entries: Vec<String> = self
            .depth_frame_draws
            .iter()
            .map(|(key, draws)| {
                let img = self.depth_cache.get(key);
                format!(
                    "{}x{} draws={} samples={}x{} depth={} 3d={} cached={} fmt={:?} layout={:?}",
                    key.width,
                    key.height,
                    draws,
                    key.sample_width,
                    key.sample_height,
                    key.depth,
                    key.is_3d,
                    img.is_some(),
                    img.map(|i| i.format),
                    img.map(|i| i.layout)
                )
            })
            .collect();
        entries.sort();
        entries.truncate(6);
        format!(
            "{} depth targets drawn this frame, {} cached: [{}]",
            self.depth_frame_draws.len(),
            self.depth_cache.len(),
            entries.join("; ")
        )
    }

    pub fn mark_cleared(&mut self, key: RtKey, full_target: bool) {
        self.note_color_sampling_changed(key.nvmap_id);
        self.color_region_sources.remove(&key);
        self.drawn_counter += 1;
        self.clear_stamp.insert(key, self.drawn_counter);
        if full_target {
            self.full_clear_stamp.insert(key, self.drawn_counter);
            self.pending_view_retirement.insert((false, key));
            self.drawn_stamp.remove(&key);
            self.guest_stale_color.remove(&key);
        } else if self.drawn_stamp.contains_key(&key) {
            self.drawn_stamp.insert(key, self.drawn_counter);
            self.present_excluded.remove(&key);
            *self.frame_draws.entry(key).or_insert(0) += 1;
        }
    }

    pub fn resolve_present_key(&self, want: RtKey, present: bool) -> Option<RtKey> {
        if is_synthetic_copy_key(want) {
            return None;
        }
        let choose_cpu = || -> Option<(RtKey, u64)> {
            if want.cpu_addr == 0 {
                return None;
            }
            let mut best: Option<(RtKey, u64)> = None;
            for k in self.cache.keys() {
                if is_synthetic_copy_key(*k)
                    || k.width != want.width
                    || k.height != want.height
                    || k.cpu_addr != want.cpu_addr
                    || self.present_excluded.contains(k)
                {
                    continue;
                }
                if want.nvmap_id != 0 && k.nvmap_id != want.nvmap_id {
                    continue;
                }
                let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                    continue;
                };
                let replace = match best {
                    Some((best_key, best_stamp)) => {
                        stamp > best_stamp
                            || (stamp == best_stamp && *k == want && best_key != want)
                    }
                    None => true,
                };
                if replace {
                    best = Some((*k, stamp));
                }
            }
            best
        };
        let choose = |gpu_va: Option<u64>, same_nvmap: bool| -> Option<(RtKey, u64)> {
            let mut best: Option<(RtKey, u64)> = None;
            for k in self.cache.keys() {
                if is_synthetic_copy_key(*k) || k.width != want.width || k.height != want.height {
                    continue;
                }
                if let Some(gpu_va) = gpu_va {
                    if k.gpu_va != gpu_va {
                        continue;
                    }
                }
                if self.present_excluded.contains(k) {
                    continue;
                }
                if same_nvmap && want.nvmap_id != 0 && k.nvmap_id != want.nvmap_id {
                    continue;
                }
                let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                    continue;
                };
                let replace = match best {
                    Some((best_key, best_stamp)) => {
                        stamp > best_stamp
                            || (stamp == best_stamp && *k == want && best_key != want)
                    }
                    None => true,
                };
                if replace {
                    best = Some((*k, stamp));
                }
            }
            best
        };
        let mut best = choose_cpu();
        if best.is_none() && want.gpu_va != 0 {
            best = choose(Some(want.gpu_va), false);
        }
        if best.is_none() {
            best = choose(None, true);
        }
        if best.is_none() {
            best = choose(None, false);
        }
        let best_stamp = best.map(|(_, s)| s).unwrap_or(0);
        if present && want.height != 0 {
            let best_real_draws = best
                .map(|(bk, _)| self.frame_real_draws.get(&bk).copied().unwrap_or(0))
                .unwrap_or(0);
            if best_real_draws == 0 {
                let aw = want.width as f32 / want.height as f32;
                let best_key = best.map(|(k, _)| k);
                let bridge = self
                    .cache
                    .keys()
                    .filter(|k| Some(**k) != best_key)
                    .filter(|k| !is_synthetic_copy_key(**k))
                    .filter(|k| want.nvmap_id == 0 || k.nvmap_id != want.nvmap_id)
                    .filter(|k| {
                        k.height != 0
                            && ((k.width as f32 / k.height as f32) - aw).abs() <= aw * 0.12
                    })
                    .filter(|k| {
                        dims_close(k.width, want.width) && dims_close(k.height, want.height)
                    })
                    .filter(|k| !self.present_excluded.contains(k))
                    .filter_map(|k| {
                        if self.frame_real_draws.get(k).copied().unwrap_or(0) > 0 {
                            Some((*k, self.drawn_stamp.get(k).copied().unwrap_or(0)))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, s)| *s);
                if let Some((bk, _)) = bridge {
                    return Some(bk);
                }
            }
        }
        let strict_cpu = std::env::var_os("NEXIUM_PRESENT_STRICT_CPU").is_some();
        if want.height != 0 && want.gpu_va == 0 && (want.cpu_addr == 0 || !strict_cpu) {
            let aw = want.width as f32 / want.height as f32;
            let same_aspect = |k: &RtKey| {
                k.height != 0 && ((k.width as f32 / k.height as f32) - aw).abs() <= aw * 0.12
            };
            let alt = self
                .cache
                .keys()
                .filter(|k| !is_synthetic_copy_key(**k))
                .filter(|k| same_aspect(k))
                .filter(|k| dims_close(k.width, want.width) && dims_close(k.height, want.height))
                .filter(|k| !self.present_excluded.contains(k))
                .filter_map(|k| self.drawn_stamp.get(k).map(|s| (*k, *s)))
                .max_by_key(|(_, s)| *s);
            if let Some((ak, astamp)) = alt {
                let best_frame_draws = best
                    .map(|(bk, _)| self.frame_draws.get(&bk).copied().unwrap_or(0))
                    .unwrap_or(0);
                let alt_frame_draws = self.frame_draws.get(&ak).copied().unwrap_or(0);
                if alt_frame_draws > 0 && best_frame_draws == 0 {
                    return Some(ak);
                }
                let stale_gap = best_stamp.max(256) / 4;
                if astamp > best_stamp.saturating_add(stale_gap) {
                    return Some(ak);
                }
            }
            if best_stamp <= 2 {
                let by_draws = self
                    .cache
                    .keys()
                    .filter(|k| !is_synthetic_copy_key(**k))
                    .filter(|k| same_aspect(k))
                    .filter(|k| {
                        dims_close(k.width, want.width) && dims_close(k.height, want.height)
                    })
                    .filter(|k| !self.present_excluded.contains(k))
                    .filter_map(|k| {
                        let fd = self.frame_draws.get(k).copied().unwrap_or(0);
                        if fd > 0 {
                            Some((*k, fd))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, fd)| *fd);
                if let Some((ak, fd)) = by_draws {
                    if fd >= 32 {
                        return Some(ak);
                    }
                }
            }
        }
        if best_stamp > 2 {
            return best.map(|(k, _)| k);
        }
        let res = best.map(|(k, _)| k);
        if res.is_none() && std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NONE_CT: AtomicU64 = AtomicU64::new(0);
            let n = NONE_CT.fetch_add(1, Ordering::Relaxed);
            if n % 120 == 0 {
                let mut keys: Vec<String> = self
                    .cache
                    .keys()
                    .map(|k| {
                        format!(
                            "{}#{}",
                            k.label(),
                            self.drawn_stamp.get(k).copied().unwrap_or(0)
                        )
                    })
                    .collect();
                keys.sort();
                log::warn!(
                    "[present-none #{}] want={} cache=[{}]",
                    n,
                    want.label(),
                    keys.join(" ")
                );
            }
        }
        res
    }

    pub fn present_key_pinned_at_va(&self, want: RtKey) -> Option<RtKey> {
        if is_synthetic_copy_key(want) || want.gpu_va == 0 || want.nvmap_id == 0 {
            return None;
        }
        self.cache
            .keys()
            .filter(|k| {
                !is_synthetic_copy_key(**k)
                    && k.nvmap_id == want.nvmap_id
                    && k.gpu_va == want.gpu_va
                    && k.width == want.width
                    && k.height == want.height
                    && k.depth == 1
                    && !k.is_3d
                    && !self.present_excluded.contains(k)
                    && self.drawn_stamp.contains_key(k)
            })
            .max_by_key(|k| self.drawn_stamp.get(*k).copied().unwrap_or(0))
            .copied()
    }

    pub fn newest_exact_present_key_at_vas(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_vas: &[u64],
    ) -> Option<(RtKey, u64)> {
        if nvmap_id == 0 || nvmap_id == u32::MAX || width == 0 || height == 0 || gpu_vas.is_empty()
        {
            return None;
        }

        let mut best = None;
        for &gpu_va in gpu_vas {
            if gpu_va == 0 {
                continue;
            }
            let Some(candidates) = self.color_gpu_base_index.get(&gpu_va) else {
                continue;
            };
            for key in candidates {
                if is_synthetic_copy_key(*key)
                    || key.nvmap_id != nvmap_id
                    || key.width != width
                    || key.height != height
                    || key.depth != 1
                    || key.is_3d
                    || key.gpu_va != gpu_va
                    || self.present_excluded.contains(key)
                {
                    continue;
                }
                let Some(image) = self.cache.get(key) else {
                    continue;
                };
                if image.layout == vk::ImageLayout::UNDEFINED {
                    continue;
                }
                let Some(stamp) = self
                    .drawn_stamp
                    .get(key)
                    .copied()
                    .filter(|stamp| *stamp != 0)
                else {
                    continue;
                };
                if best.is_none_or(|(_, best_stamp)| stamp > best_stamp) {
                    best = Some((*key, stamp));
                }
            }
        }
        best
    }

    pub fn present_alias_vas(&self, want: RtKey) -> Vec<u64> {
        let mut vas: Vec<u64> = self
            .present_candidates(want)
            .into_iter()
            .filter(|(k, _)| k.nvmap_id == want.nvmap_id && k.gpu_va != 0)
            .map(|(k, _)| k.gpu_va)
            .collect();
        vas.sort_unstable();
        vas.dedup();
        vas
    }

    pub fn present_fallback_key(&self, want: RtKey) -> Option<RtKey> {
        if is_synthetic_copy_key(want) || want.nvmap_id == 0 {
            return None;
        }
        self.cache
            .keys()
            .filter(|k| {
                !is_synthetic_copy_key(**k)
                    && k.nvmap_id == want.nvmap_id
                    && !self.present_excluded.contains(k)
                    && dims_close(k.width, want.width)
                    && dims_close(k.height, want.height)
            })
            .max_by_key(|k| {
                let exact = (want.cpu_addr != 0 && k.cpu_addr == want.cpu_addr) as u32;
                let fd = self.frame_draws.get(k).copied().unwrap_or(0);
                let kd = (k.width as i64 - want.width as i64).abs()
                    + (k.height as i64 - want.height as i64).abs();
                (exact, fd, -kd)
            })
            .copied()
    }

    fn rekey_retirement_state(&mut self, depth: bool, old: RtKey, new: RtKey) {
        let sources = if depth { &mut self.depth_region_sources } else { &mut self.color_region_sources };
        rekey_hash_map_value(sources, old, new);
        for (source, _, _) in sources.values_mut() {
            if *source == old {
                *source = new;
            }
        }
        if self.pending_view_retirement.remove(&(depth, old)) {
            self.pending_view_retirement.insert((depth, new));
        }
        if self.obsolete_mapping_views.remove(&(depth, old)) {
            self.obsolete_mapping_views.insert((depth, new));
        }
    }

    fn view_can_reconstruct(&self, depth: bool, source: RtKey, target: RtKey) -> bool {
        let images = if depth { &self.depth_cache } else { &self.cache };
        let Some((source, source_image)) = images.get_key_value(&source) else { return false; };
        let Some((target, target_image)) = images.get_key_value(&target) else { return false; };
        let source = *source;
        let target = *target;
        source != target
            && !is_synthetic_copy_key(source)
            && !is_synthetic_copy_key(target)
            && source.gpu_va != 0
            && source.gpu_va == target.gpu_va
            && source.cpu_addr != 0
            && source.cpu_addr == target.cpu_addr
            && source.nvmap_id == target.nvmap_id
            && source.mapping_epoch != 0
            && source.mapping_epoch == target.mapping_epoch
            && source.layout_signature != 0
            && source.layout_signature == target.layout_signature
            && source.depth == 1
            && target.depth == 1
            && !source.is_3d
            && !target.is_3d
            && source.base_layer == 0
            && target.base_layer == 0
            && source.sample_width == 1
            && source.sample_height == 1
            && target.sample_width == 1
            && target.sample_height == 1
            && target.width != 0
            && target.height != 0
            && source.width >= target.width
            && source.height >= target.height
            && source_image.extent.width == source.width
            && source_image.extent.height == source.height
            && target_image.extent.width == target.width
            && target_image.extent.height == target.height
            && source_image.base_format == target_image.base_format
            && source_image.format == target_image.format
            && source_image.aspects == target_image.aspects
            && source_image.layout != vk::ImageLayout::UNDEFINED
            && color_sync_layout_compatible(source, target, source_image.base_format)
            && if depth {
                source_image.base_format == vk::Format::D32_SFLOAT
                    && source_image.format == vk::Format::D32_SFLOAT
                    && source_image.aspects == vk::ImageAspectFlags::DEPTH
                    && !self.guest_stale_depth.contains(&source)
                    && !self.guest_stale_depth.contains(&target)
            } else {
                source_image.aspects == vk::ImageAspectFlags::COLOR
                    && !self.guest_stale_color.contains(&source)
                    && !self.guest_stale_color.contains(&target)
            }
    }

    pub(crate) fn has_pending_view_retirement(&self) -> bool {
        !self.pending_view_retirement.is_empty() || !self.obsolete_mapping_views.is_empty()
    }

    fn retire_view(&mut self, depth: bool, key: RtKey, retired: &mut Vec<GpuImage>) -> (usize, u64) {
        let mut count = 0;
        let mut bytes = 0u64;
        let removed = if depth {
            self.forget_depth_tracking(key);
            self.depth_frame_draws.remove(&key);
            self.depth_array_shadow_generations.retain(|shadow, sources| {
                *shadow != key && sources.iter().all(|(source, _)| *source != key)
            });
            if remove_guest_range(&mut self.depth_guest_ranges, &mut self.depth_guest_range_index, key) {
                self.note_guest_ranges_changed();
            }
            self.structure_generation = self.structure_generation.wrapping_add(1);
            self.note_color_sampling_changed(key.nvmap_id);
            self.remove_depth_image(key).map(|(_, image)| image)
        } else {
            let removed = self.remove_color_image(key).map(|(_, image)| image);
            self.drawn_stamp.remove(&key);
            self.guest_stale_color.remove(&key);
            self.present_excluded.remove(&key);
            self.present_flip_y.remove(&key);
            self.frame_draws.remove(&key);
            self.frame_real_draws.remove(&key);
            if let Some(snapshot) = self.snapshots.remove(&key) {
                bytes = bytes.saturating_add(image_estimated_bytes(key, &snapshot));
                count += 1;
                retired.push(snapshot);
            }
            removed
        };
        if let Some(image) = removed {
            bytes = bytes.saturating_add(image_estimated_bytes(key, &image));
            count += 1;
            retired.push(image);
        }
        (count, bytes)
    }

    pub fn touch_use(&self, key: RtKey) {
        let stamp = self.use_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut uses) = self.last_use.lock() {
            uses.insert(key, (self.use_frame, stamp));
        }
    }

    pub(crate) fn over_budget(&self) -> bool {
        self.cached_bytes() > rt_cache_budget_bytes()
    }

    fn cached_bytes(&self) -> u64 {
        let [(_, color_bytes), (_, depth_bytes), (_, snapshot_bytes)] = self.memory_estimates();
        color_bytes
            .saturating_add(depth_bytes)
            .saturating_add(snapshot_bytes)
    }

    pub(crate) fn retire_over_budget(
        &mut self,
        retired: &mut Vec<GpuImage>,
        protected: &[RtKey],
    ) -> (usize, u64) {
        let budget = rt_cache_budget_bytes();
        let total = self.cached_bytes();
        if total <= budget {
            return (0, 0);
        }
        let now = self.use_frame;
        let mut candidates: Vec<((u64, u64), bool, RtKey, u64)> = Vec::new();
        if let Ok(uses) = self.last_use.lock() {
            let entries = self
                .cache
                .iter()
                .map(|(key, image)| (false, *key, image_estimated_bytes(*key, image)))
                .chain(
                    self.depth_cache
                        .iter()
                        .map(|(key, image)| (true, *key, image_estimated_bytes(*key, image))),
                );
            for (depth, key, bytes) in entries {
                if protected.contains(&key) || is_synthetic_copy_key(key) {
                    continue;
                }
                let last = uses.get(&key).copied().unwrap_or((0, 0));
                if now.saturating_sub(last.0) < RT_CACHE_MIN_IDLE_FRAMES {
                    continue;
                }
                candidates.push((last, depth, key, bytes));
            }
        }
        candidates.sort_by_key(|(last, _, _, _)| *last);
        let mut excess = total - budget;
        let mut count = 0usize;
        let mut bytes_total = 0u64;
        for (_, depth, key, bytes) in candidates {
            if excess == 0 {
                break;
            }
            let (retired_count, retired_bytes) = self.retire_view(depth, key, retired);
            count += retired_count;
            bytes_total = bytes_total.saturating_add(retired_bytes);
            excess = excess.saturating_sub(bytes.max(retired_bytes));
        }
        if let Ok(mut uses) = self.last_use.lock() {
            uses.retain(|key, _| self.cache.contains_key(key) || self.depth_cache.contains_key(key));
        }
        if count != 0 {
            static EVICTIONS: AtomicU64 = AtomicU64::new(0);
            let total_evictions =
                EVICTIONS.fetch_add(count as u64, Ordering::Relaxed) + count as u64;
            if total_evictions <= 64 || total_evictions % 512 < count as u64 {
                log::info!(
                    "[rt-cache-evict] retired={count} bytes={bytes_total} total_evictions={total_evictions} cached_bytes={} budget={budget}",
                    self.cached_bytes()
                );
            }
        }
        (count, bytes_total)
    }

    pub(crate) fn retire_redundant_views(
        &mut self,
        retired: &mut Vec<GpuImage>,
        protected: &[RtKey],
    ) -> (usize, u64) {
        if !self.has_pending_view_retirement() {
            return (0, 0);
        }
        self.view_retirement_epoch = self.view_retirement_epoch.wrapping_add(1);
        for key in protected {
            for sources in [&mut self.color_region_sources, &mut self.depth_region_sources] {
                if let Some((_, _, last_used)) = sources.get_mut(key) {
                    *last_used = self.view_retirement_epoch;
                }
            }
        }
        let obsolete = self.obsolete_mapping_views.iter().copied()
            .filter(|(_, key)| !protected.contains(key))
            .collect::<Vec<_>>();
        let mut count = 0;
        let mut bytes = 0u64;
        for (depth, key) in obsolete {
            let (retired_count, retired_bytes) = self.retire_view(depth, key, retired);
            count += retired_count;
            bytes = bytes.saturating_add(retired_bytes);
        }
        let pending = std::mem::take(&mut self.pending_view_retirement);
        let mut replacements = HashMap::<(bool, RtKey), RtKey>::default();
        for (depth, key) in pending {
            let images = if depth { &self.depth_cache } else { &self.cache };
            let sources = if depth { &self.depth_region_sources } else { &self.color_region_sources };
            let stamp = if depth { self.depth_generation(key) } else { self.writeback_stamp(key) };
            if let Some(&(source, image, _)) = sources.get(&key) {
                let source_stamp = if depth { self.depth_generation(source) } else { self.writeback_stamp(source) };
                if (depth || self.present_excluded.contains(&key))
                    && images.get(&source).is_some_and(|entry| entry.image == image)
                    && source_stamp.zip(stamp).is_some_and(|(source, target)| source >= target)
                    && self.view_can_reconstruct(depth, source, key)
                {
                    replacements.insert((depth, key), source);
                }
            }
            let cutoff = if depth {
                self.depth_full_clear_generation.get(&key)
            } else {
                self.full_clear_stamp.get(&key)
            };
            let Some(&cutoff) = cutoff else { continue; };
            let candidates = if depth {
                self.depth_cache.keys().filter(|candidate| candidate.gpu_va == key.gpu_va)
                    .copied().collect::<Vec<_>>()
            } else {
                self.color_gpu_base_index.get(&key.gpu_va).cloned().unwrap_or_default()
            };
            for target in candidates {
                if !depth && (!self.color_region_sources.contains_key(&target)
                    || !self.present_excluded.contains(&target)) {
                    continue;
                }
                let latest = if depth { self.depth_generation(target) } else { self.writeback_stamp(target) };
                if latest.unwrap_or(0) <= cutoff && self.view_can_reconstruct(depth, key, target) {
                    replacements.insert((depth, target), key);
                }
            }
        }
        let retained_sources = replacements.iter().map(|(&(depth, _), &source)| (depth, source))
            .collect::<HashSet<_>>();
        for ((depth, key), source) in replacements {
            let region_sources = if depth { &self.depth_region_sources } else { &self.color_region_sources };
            let recently_used = region_sources.get(&key).is_some_and(|(_, _, last_used)| {
                self.view_retirement_epoch.wrapping_sub(*last_used) < 64
            });
            if retained_sources.contains(&(depth, key)) || protected.contains(&key) || recently_used {
                self.pending_view_retirement.insert((depth, source));
                self.pending_view_retirement.insert((depth, key));
                continue;
            }
            let (retired_count, retired_bytes) = self.retire_view(depth, key, retired);
            count += retired_count;
            bytes = bytes.saturating_add(retired_bytes);
        }
        (count, bytes)
    }

    pub(crate) fn memory_estimates(&self) -> [(usize, u64); 3] {
        [&self.cache, &self.depth_cache, &self.snapshots].map(|cache| {
            let bytes = cache.iter().fold(0u64, |total, (key, image)| {
                total.saturating_add(
                    u64::from(image.extent.width)
                        .saturating_mul(u64::from(image.extent.height))
                        .saturating_mul(u64::from(key.depth))
                        .saturating_mul(rt_format_bytes(image.format)),
                )
            });
            (cache.len(), bytes)
        })
    }

    pub fn orphan_memory_summary<'a>(&self, retired: impl Iterator<Item = &'a GpuImage>) -> Option<String> {
        use vk::Handle;
        let live = rt_live_memory()?.lock().ok()?;
        let mut owned: HashSet<u64> = HashSet::default();
        for image in self
            .cache
            .values()
            .chain(self.depth_cache.values())
            .chain(self.snapshots.values())
        {
            owned.insert(image.memory.as_raw());
        }
        for image in retired {
            owned.insert(image.memory.as_raw());
        }
        let mut live_bytes = 0u64;
        let mut orphan_count = 0usize;
        let mut orphan_bytes = 0u64;
        let mut dims: HashMap<(u32, u32, u64), (usize, u64)> = HashMap::default();
        for (memory, (size, key)) in live.iter() {
            live_bytes = live_bytes.saturating_add(*size);
            if owned.contains(memory) {
                continue;
            }
            orphan_count += 1;
            orphan_bytes = orphan_bytes.saturating_add(*size);
            let entry = dims.entry((key.width, key.height, key.gpu_va)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(*size);
        }
        let mut dims = dims.into_iter().collect::<Vec<_>>();
        dims.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
        let top = dims
            .iter()
            .take(8)
            .map(|((w, h, va), (count, bytes))| format!("{w}x{h}@{va:x}:{count}/{}MB", bytes >> 20))
            .collect::<Vec<_>>()
            .join(" ");
        Some(format!(
            "live={} live_mb={} owned={} orphans={orphan_count} orphan_mb={} top=[{top}]",
            live.len(),
            live_bytes >> 20,
            owned.len(),
            orphan_bytes >> 20
        ))
    }

    pub fn debug_all(&self) -> Vec<(RtKey, u64)> {
        let mut out: Vec<(RtKey, u64)> = self
            .cache
            .keys()
            .map(|k| (*k, self.drawn_stamp.get(k).copied().unwrap_or(0)))
            .collect();
        out.sort_by_key(|(_, s)| *s);
        out
    }

    pub fn present_candidates(&self, want: RtKey) -> Vec<(RtKey, u64)> {
        let mut out = Vec::new();
        if is_synthetic_copy_key(want) {
            return out;
        }
        for k in self.cache.keys() {
            if is_synthetic_copy_key(*k) || self.present_excluded.contains(k) {
                continue;
            }
            if k.width != want.width || k.height != want.height {
                continue;
            }
            if want.gpu_va != 0 && k.gpu_va != want.gpu_va {
                continue;
            }
            if want.cpu_addr != 0 && k.cpu_addr != want.cpu_addr {
                continue;
            }
            let stamp = self.drawn_stamp.get(k).copied().unwrap_or(0);
            out.push((*k, stamp));
        }
        out.sort_by_key(|(_, stamp)| *stamp);
        out
    }

    pub fn find_color_screen(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        if is_synthetic_copy_key(want) {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k)
                || k.nvmap_id == want.nvmap_id
                || k.width != want.width
                || k.height != want.height
            {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            if best.as_ref().map_or(true, |(_, _, bs)| stamp > *bs) {
                best = Some((*k, img, stamp));
            }
        }
        if best.is_none() && want.height != 0 {
            let aw = want.width as f32 / want.height as f32;
            for (k, img) in &self.cache {
                if is_synthetic_copy_key(*k)
                    || k.nvmap_id == want.nvmap_id
                    || k.height == 0
                    || k.width < want.width
                {
                    continue;
                }
                let ka = k.width as f32 / k.height as f32;
                if (ka - aw).abs() > aw * 0.12 {
                    continue;
                }
                let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                    continue;
                };
                if best.as_ref().map_or(true, |(_, _, bs)| stamp > *bs) {
                    best = Some((*k, img, stamp));
                }
            }
        }
        best.map(|(k, img, _)| (k, img.image, img.view, img.layout))
    }

    pub fn has_drawn_color_at_va(&self, nvmap_id: u32, gpu_va: u64) -> bool {
        self.cache.keys().any(|k| {
            !is_synthetic_copy_key(*k)
                && k.nvmap_id == nvmap_id
                && k.gpu_va == gpu_va
                && self.drawn_stamp.contains_key(k)
        })
    }

    pub(crate) fn set_depth_formats(&mut self, formats: crate::depth::DepthFormats) {
        assert!(self.depth_cache.is_empty());
        self.depth_formats = formats;
    }

    pub(crate) fn host_depth_format(&self, guest: vk::Format) -> vk::Format {
        self.depth_formats.host(guest)
    }

    pub(crate) fn pack_depth_buffer(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pool: vk::DescriptorPool,
        buffer: vk::Buffer,
        bytes: u64,
    ) -> Result<vk::DescriptorSet, String> {
        if self.depth_pack_pipeline.is_none() {
            self.depth_pack_pipeline = Some(crate::depth::DepthPackPipeline::new(device)?);
        }
        self.depth_pack_pipeline
            .as_ref()
            .unwrap()
            .record(device, cmd, pool, buffer, bytes)
    }

    pub fn set_mem_properties(&mut self, props: vk::PhysicalDeviceMemoryProperties) {
        self.mem_properties = Some(props);
    }

    pub fn get_or_create(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<&mut GpuImage, String> {
        self.note_color_sampling_changed(key.nvmap_id);
        if self.cache.contains_key(&key) && !self.obsolete_mapping_views.contains(&(false, key)) {
            return Ok(self.cache.get_mut(&key).unwrap());
        }
        self.get_or_create_with_format(key, device, vk::Format::R8G8B8A8_UNORM)
    }

    pub fn get_existing(&mut self, key: RtKey) -> Option<&mut GpuImage> {
        if self.obsolete_mapping_views.contains(&(false, key)) {
            return None;
        }
        self.note_color_sampling_changed(key.nvmap_id);
        self.cache.get_mut(&key)
    }

    pub fn debug_depth_all(
        &self,
    ) -> Vec<(
        RtKey,
        vk::Image,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        self.depth_cache
            .iter()
            .map(|(k, img)| (*k, img.image, img.layout, img.format, img.aspects))
            .collect()
    }

    pub fn get_existing_depth(&mut self, key: RtKey) -> Option<&mut GpuImage> {
        if let Some((stored, _)) = self.depth_cache.get_key_value(&key) {
            let stored = *stored;
            if self.obsolete_mapping_views.contains(&(true, stored)) || alias_view_metadata_changed(stored, key) {
                return None;
            }
            return self.depth_cache.get_mut(&stored);
        }
        let matching = self.depth_cache.keys().copied().find(|existing| {
            same_physical_backing(*existing, key)
                && !self.obsolete_mapping_views.contains(&(true, *existing))
                && !alias_view_metadata_changed(*existing, key)
        })?;
        self.depth_cache.get_mut(&matching)
    }

    pub fn color_target_touches_va(&self, gpu_va: u64) -> bool {
        if gpu_va == 0 {
            return false;
        }
        self.color_gpu_base_index.contains_key(&gpu_va)
            || self
                .color_gpu_page_index
                .contains_key(&(gpu_va >> RT_COLOR_GPU_PAGE_SHIFT))
    }

    pub fn find_color_key_at_va(&self, nvmap_id: u32, gpu_va: u64) -> Option<RtKey> {
        self.cache
            .keys()
            .filter(|k| !is_synthetic_copy_key(**k) && k.nvmap_id == nvmap_id && k.gpu_va == gpu_va)
            .filter(|k| !self.obsolete_mapping_views.contains(&(false, **k)))
            .max_by_key(|k| self.drawn_stamp.get(*k).copied().unwrap_or(0))
            .copied()
    }

    pub fn drawn_stamp(&self, key: RtKey) -> Option<u64> {
        self.drawn_stamp.get(&key).copied()
    }

    pub fn writeback_stamp(&self, key: RtKey) -> Option<u64> {
        self.drawn_stamp.get(&key).into_iter()
            .chain(self.clear_stamp.get(&key))
            .copied().max()
    }

    fn color_entries_for_nvmap(&self, nvmap_id: u32) -> impl Iterator<Item = (&RtKey, &GpuImage)> {
        self.color_nvmap_index
            .get(&nvmap_id)
            .into_iter()
            .flatten()
            .filter(|key| !self.obsolete_mapping_views.contains(&(false, **key)))
            .filter_map(|key| self.cache.get_key_value(key))
    }

    pub fn color_keys_for_nvmap(&self, nvmap_id: u32) -> Vec<(RtKey, vk::Format, u64, u32)> {
        self.color_entries_for_nvmap(nvmap_id)
            .filter(|(k, _)| !is_synthetic_copy_key(**k))
            .map(|(k, img)| {
                (
                    *k,
                    img.format,
                    self.drawn_stamp.get(k).copied().unwrap_or(0),
                    self.frame_real_draws.get(k).copied().unwrap_or(0),
                )
            })
            .collect()
    }

    pub fn has_color_for_nvmap(&self, nvmap_id: u32) -> bool {
        self.color_nvmap_index
            .get(&nvmap_id)
            .is_some_and(|keys| !keys.is_empty())
    }

    pub fn color_exact_with_format(
        &self,
        key: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        u64,
    )> {
        let (stored, img) = self.cache.get_key_value(&key)?;
        let stored = *stored;
        if self.obsolete_mapping_views.contains(&(false, stored)) || alias_view_metadata_changed(stored, key) {
            return None;
        }
        Some((
            stored,
            img.image,
            img.view,
            img.layout,
            img.format,
            self.writeback_stamp(stored).unwrap_or(0),
        ))
    }

    pub fn drawn_color_aliases(
        &self,
        key: RtKey,
    ) -> Vec<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut out = Vec::new();
        if is_synthetic_copy_key(key) || (key.gpu_va == 0 && key.cpu_addr == 0) {
            return out;
        }
        let mut candidates = Vec::new();
        if key.gpu_va != 0 {
            if let Some(indexed) = self.color_gpu_base_index.get(&key.gpu_va) {
                candidates.extend_from_slice(indexed);
            }
        }
        if key.cpu_addr != 0 {
            if let Some(indexed) = self.color_cpu_base_index.get(&key.cpu_addr) {
                for candidate in indexed {
                    if !candidates.contains(candidate) {
                        candidates.push(*candidate);
                    }
                }
            }
        }
        for candidate in &candidates {
            let Some((stored, img)) = self.cache.get_key_value(candidate) else {
                continue;
            };
            let stored = *stored;
            let same_gpu_base =
                stored.nvmap_id == key.nvmap_id && key.gpu_va != 0 && stored.gpu_va == key.gpu_va;
            if stored == key
                || (!same_gpu_base && !same_physical_backing(stored, key))
                || !color_alias_sync_identity(stored, key)
                || !color_sync_layout_compatible(stored, key, img.format)
            {
                continue;
            }
            if self.color_is_guest_stale(stored) || img.layout == vk::ImageLayout::UNDEFINED {
                continue;
            }
            let Some(stamp) = self.writeback_stamp(stored) else {
                continue;
            };
            out.push((stored, img.image, img.layout, img.format, stamp));
        }
        out.sort_by_key(|(_, _, _, _, stamp)| *stamp);
        out
    }

    #[cfg(test)]
    fn drawn_color_aliases_full_scan(
        &self,
        key: RtKey,
    ) -> Vec<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut out = Vec::new();
        if is_synthetic_copy_key(key) || (key.gpu_va == 0 && key.cpu_addr == 0) {
            return out;
        }
        for (stored, img) in &self.cache {
            let same_gpu_base =
                stored.nvmap_id == key.nvmap_id && key.gpu_va != 0 && stored.gpu_va == key.gpu_va;
            if is_synthetic_copy_key(*stored)
                || *stored == key
                || (!same_gpu_base && !same_physical_backing(*stored, key))
                || !color_alias_sync_identity(*stored, key)
                || !color_sync_layout_compatible(*stored, key, img.format)
            {
                continue;
            }
            if self.color_is_guest_stale(*stored) || img.layout == vk::ImageLayout::UNDEFINED {
                continue;
            }
            let Some(stamp) = self.writeback_stamp(*stored) else {
                continue;
            };
            out.push((*stored, img.image, img.layout, img.format, stamp));
        }
        out.sort_by_key(|(_, _, _, _, stamp)| *stamp);
        out
    }

    pub(crate) fn color_requires_recreate(&self, key: RtKey, format: vk::Format) -> bool {
        self.cache
            .get_key_value(&key)
            .is_some_and(|(stored, image)| {
                self.obsolete_mapping_views.contains(&(false, *stored))
                    || render_target_backing_changed(*stored, key)
                    || (image.format != format && !rt_formats_compatible(image.base_format, format))
            })
    }

    pub(crate) fn depth_requires_recreate(
        &self,
        key: RtKey,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
    ) -> bool {
        let cache_key = if let Some((stored, _)) = self.depth_cache.get_key_value(&key) {
            *stored
        } else {
            self.depth_cache
                .keys()
                .copied()
                .find(|existing| same_physical_backing(*existing, key))
                .unwrap_or(key)
        };
        self.depth_cache.get(&cache_key).is_some_and(|image| {
            self.obsolete_mapping_views.contains(&(true, cache_key))
                || render_target_backing_changed(cache_key, key)
                || depth_image_requires_recreate(image.base_format, image.aspects, format, aspects)
        })
    }

    fn get_or_create_color_image(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<Option<GpuImage>, String> {
        let existing = self.cache.get_key_value(&key).map(|(stored, image)| {
            (
                *stored,
                image.format,
                image.base_format,
                self.obsolete_mapping_views.contains(&(false, *stored))
                    || render_target_backing_changed(*stored, key),
            )
        });
        match existing {
            Some((stored, current, _, false)) if current == format => {
                self.refresh_color_footprint(stored, key);
                return Ok(None);
            }
            Some((stored, current, base, false)) if rt_formats_compatible(base, format) => {
                self.refresh_color_footprint(stored, key);
                if rt_format_dbg_enabled(key.nvmap_id) {
                    log::warn!(
                        "[rt-format-view] key={} {:?}->{:?} stamp={}",
                        key.label(),
                        current,
                        format,
                        self.drawn_stamp.get(&key).copied().unwrap_or(0)
                    );
                }
                let image = self.cache.get_mut(&key).unwrap();
                if !image.views.contains_key(&format) {
                    let view = create_image_view(
                        device,
                        image.image,
                        format,
                        vk::ImageAspectFlags::COLOR,
                        key,
                    )?;
                    image.views.insert(format, view);
                }
                image.view = image.views[&format];
                image.format = format;
                return Ok(None);
            }
            Some(_) => {}
            None => {}
        }
        let replacement = self.create_image(device, key, format)?;
        let retired = if let Some((canonical, image)) = self.remove_color_image(key) {
            if rt_format_dbg_enabled(key.nvmap_id) {
                log::warn!(
                    "[rt-format-churn] key={} {:?}->{:?} stamp={}",
                    canonical.label(),
                    image.format,
                    format,
                    self.drawn_stamp.get(&canonical).copied().unwrap_or(0)
                );
            }
            self.drawn_stamp.remove(&canonical);
            self.present_excluded.remove(&canonical);
            self.present_flip_y.remove(&canonical);
            self.frame_draws.remove(&canonical);
            self.frame_real_draws.remove(&canonical);
            Some(image)
        } else {
            None
        };
        self.insert_color_image(key, replacement);
        Ok(retired)
    }

    pub fn get_or_create_with_format(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<&mut GpuImage, String> {
        self.note_color_sampling_changed(key.nvmap_id);
        if let Some(image) = self.get_or_create_color_image(key, device, format)? {
            destroy_gpu_image(device, image);
        }
        Ok(self.cache.get_mut(&key).unwrap())
    }

    pub fn get_or_create_with_format_retiring(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
        retired: &mut Vec<GpuImage>,
    ) -> Result<&mut GpuImage, String> {
        self.note_color_sampling_changed(key.nvmap_id);
        if let Some(image) = self.get_or_create_color_image(key, device, format)? {
            retired.push(image);
        }
        self.touch_use(key);
        Ok(self.cache.get_mut(&key).unwrap())
    }

    pub fn get_or_create_sample_view(
        &mut self,
        device: &ash::Device,
        key: RtKey,
        source_image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
        components: vk::ComponentMapping,
    ) -> Result<Option<vk::ImageView>, String> {
        let target = if self
            .cache
            .get(&key)
            .is_some_and(|image| image.image == source_image)
        {
            self.cache.get_mut(&key)
        } else if self
            .depth_cache
            .get(&key)
            .is_some_and(|image| image.image == source_image)
        {
            self.depth_cache.get_mut(&key)
        } else if self
            .snapshots
            .get(&key)
            .is_some_and(|image| image.image == source_image)
        {
            self.snapshots.get_mut(&key)
        } else {
            None
        };
        let Some(target) = target else {
            return Ok(None);
        };
        get_or_create_sample_view(
            device,
            target,
            source_image,
            format,
            aspect,
            view_type,
            components,
        )
        .map(Some)
    }

    pub(crate) fn create_detached_color_image(
        &self,
        device: &ash::Device,
        key: RtKey,
        format: vk::Format,
    ) -> Result<GpuImage, String> {
        self.create_image(device, key, format)
    }

    pub(crate) fn get_or_create_detached_sample_view(
        device: &ash::Device,
        image: &mut GpuImage,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
        components: vk::ComponentMapping,
    ) -> Result<vk::ImageView, String> {
        let source_image = image.image;
        get_or_create_sample_view(
            device,
            image,
            source_image,
            format,
            aspect,
            view_type,
            components,
        )
    }

    pub(crate) fn destroy_detached_image(device: &ash::Device, image: GpuImage) {
        destroy_gpu_image(device, image);
    }

    pub fn get_or_create_depth(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
    ) -> Result<(&mut GpuImage, bool), String> {
        self.get_or_create_depth_impl(key, device, format, aspects, None)
    }

    pub fn get_or_create_depth_retiring(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        retired: &mut Vec<GpuImage>,
    ) -> Result<(&mut GpuImage, bool), String> {
        self.get_or_create_depth_impl(key, device, format, aspects, Some(retired))
    }

    fn get_or_create_depth_impl(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        mut retired: Option<&mut Vec<GpuImage>>,
    ) -> Result<(&mut GpuImage, bool), String> {
        if aspects.is_empty() {
            return Err("depth image requires at least one aspect".to_string());
        }
        let cache_key = if let Some((stored, _)) = self.depth_cache.get_key_value(&key) {
            *stored
        } else {
            self.depth_cache
                .keys()
                .copied()
                .find(|existing| same_physical_backing(*existing, key))
                .unwrap_or(key)
        };
        self.touch_use(cache_key);
        let recreate = self.depth_cache.get(&cache_key).is_some_and(|image| {
            self.obsolete_mapping_views.contains(&(true, cache_key))
                || render_target_backing_changed(cache_key, key)
                || depth_image_requires_recreate(image.base_format, image.aspects, format, aspects)
        });
        if recreate {
            self.forget_depth_tracking(cache_key);
            if let Some((_, image)) = self.remove_depth_image(cache_key) {
                if remove_guest_range(
                    &mut self.depth_guest_ranges,
                    &mut self.depth_guest_range_index,
                    cache_key,
                ) {
                    self.note_guest_ranges_changed();
                }
                if let Some(retired) = retired.as_mut() {
                    retired.push(image);
                } else {
                    destroy_gpu_image(device, image);
                }
            }
        }
        let insert_key = if recreate { key } else { cache_key };
        let created = !self.depth_cache.contains_key(&insert_key);
        if created {
            let mut image = self.create_image_inner(
                device,
                insert_key,
                self.host_depth_format(format),
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                aspects,
            )?;
            image.base_format = format;
            let depth_base_format = image.base_format;
            self.insert_depth_image(insert_key, image);
            self.extend_guest_bounds(insert_key, depth_base_format);
            if insert_guest_range(
                &mut self.depth_guest_ranges,
                &mut self.depth_guest_range_index,
                insert_key,
                depth_base_format,
            ) {
                self.note_guest_ranges_changed();
            }
            self.mark_depth_written(insert_key);
        }
        let result_key = if created {
            insert_key
        } else {
            self.refresh_depth_footprint(insert_key, key)
        };
        Ok((self.depth_cache.get_mut(&result_key).unwrap(), created))
    }

    fn find_depth_physical(&self, want: RtKey, indexed: bool) -> Option<(&RtKey, &GpuImage)> {
        let eligible = |key: RtKey| {
            same_physical_backing(key, want)
                && !self.obsolete_mapping_views.contains(&(true, key))
                && !alias_view_metadata_changed(key, want)
        };
        if indexed {
            let mut candidates = self.depth_cpu_base_index.get(&want.cpu_addr)?.iter()
                .filter_map(|key| self.depth_cache.get_key_value(key))
                .filter(|(key, _)| eligible(**key));
            let first = candidates.next()?;
            if candidates.next().is_none() {
                return Some(first);
            }
        }
        self.depth_cache.iter().find(|(key, _)| eligible(**key))
    }

    pub fn find_depth(&self, want: RtKey) -> Option<RtSampleableDepthAlias> {
        self.find_depth_inner(want, true)
    }

    fn find_depth_inner(&self, want: RtKey, indexed: bool) -> Option<RtSampleableDepthAlias> {
        if let Some((stored, img)) = self.depth_cache.get_key_value(&want) {
            let stored = *stored;
            if self.obsolete_mapping_views.contains(&(true, stored)) || alias_view_metadata_changed(stored, want) {
                return None;
            }
            return Some((
                stored,
                img.image,
                img.view,
                img.layout,
                img.format,
                img.aspects,
            ));
        }
        if want.gpu_va != 0 && want.cpu_addr != 0 {
            if let Some((key, img)) = self.find_depth_physical(want, indexed) {
                return Some((
                    *key,
                    img.image,
                    img.view,
                    img.layout,
                    img.format,
                    img.aspects,
                ));
            }
        }
        if want.gpu_va != 0 {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.depth_cache {
            if k.nvmap_id != want.nvmap_id || self.obsolete_mapping_views.contains(&(true, *k)) {
                continue;
            }
            if alias_view_metadata_changed(*k, want) {
                continue;
            }
            if want.gpu_va != 0 && k.gpu_va != want.gpu_va {
                continue;
            }
            if !dims_close(k.width, want.width) || !dims_close(k.height, want.height) {
                continue;
            }
            let kd = (k.width as i64 - want.width as i64).abs()
                + (k.height as i64 - want.height as i64).abs();
            let replace = match best {
                Some((bk, _)) => {
                    kd < (bk.width as i64 - want.width as i64).abs()
                        + (bk.height as i64 - want.height as i64).abs()
                }
                None => true,
            };
            if replace {
                best = Some((*k, img));
            }
        }
        best.map(|(k, img)| (k, img.image, img.view, img.layout, img.format, img.aspects))
    }

    pub fn find_sampleable_depth(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        let (key, img) = if let Some((stored, img)) = self.depth_cache.get_key_value(&want) {
            let stored = *stored;
            if alias_view_metadata_changed(stored, want) {
                return None;
            }
            (stored, img)
        } else {
            self.depth_cache
                .iter()
                .find(|(key, _)| {
                    same_physical_backing(**key, want) && !alias_view_metadata_changed(**key, want)
                })
                .map(|(key, img)| (*key, img))?
        };
        if self.depth_is_guest_stale(key) || img.layout == vk::ImageLayout::UNDEFINED {
            return None;
        }
        Some((
            key,
            img.image,
            img.view,
            img.layout,
            img.format,
            img.aspects,
        ))
    }

    pub fn find_sampleable_depth_for_exact_alias(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        if !rt_alias_indexed_lookup_enabled() {
            return self.find_sampleable_depth_for_exact_alias_legacy(want);
        }
        let indexed = self.find_sampleable_depth_for_exact_alias_indexed(want);
        if !rt_alias_indexed_lookup_shadow_enabled() {
            return indexed;
        }
        rt_alias_shadow_depth_result(
            indexed,
            self.find_sampleable_depth_for_exact_alias_legacy(want),
        )
    }

    fn find_sampleable_depth_for_exact_alias_legacy(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        let (key, image) = self
            .depth_cache
            .iter()
            .find(|(candidate, _)| exact_texture_alias_identity(**candidate, want))
            .map(|(key, image)| (*key, image))?;
        if self.depth_is_guest_stale(key) || image.layout == vk::ImageLayout::UNDEFINED {
            return None;
        }
        Some((
            key,
            image.image,
            image.view,
            image.layout,
            image.format,
            image.aspects,
        ))
    }

    fn find_sampleable_depth_for_exact_alias_indexed(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        if want.gpu_va == 0 {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_WILDCARD_FALLBACKS, 1);
            return self.find_sampleable_depth_for_exact_alias_legacy(want);
        }
        let Some((key, image)) = self
            .depth_cache
            .get_key_value(&want)
            .filter(|(candidate, _)| exact_texture_alias_identity(**candidate, want))
            .map(|(key, image)| (*key, image))
        else {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_MISSES, 1);
            return None;
        };
        if self.depth_is_guest_stale(key) || image.layout == vk::ImageLayout::UNDEFINED {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_MISSES, 1);
            return None;
        }
        count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_HITS, 1);
        Some((
            key,
            image.image,
            image.view,
            image.layout,
            image.format,
            image.aspects,
        ))
    }

    pub(crate) fn find_sampleable_d32_region(&self, want: RtKey) -> Option<RtDepthRegion> {
        if want.gpu_va == 0
            || want.layout_signature == 0
            || want.depth != 1
            || want.is_3d
            || want.sample_width != 1
            || want.sample_height != 1
        {
            return None;
        }
        self.depth_cache
            .iter()
            .filter_map(|(key, image)| {
                if key.nvmap_id != want.nvmap_id
                    || key.gpu_va != want.gpu_va
                    || key.layout_signature == 0
                    || alias_view_metadata_changed(*key, want)
                    || key.depth != 1
                    || key.is_3d
                    || key.sample_width != 1
                    || key.sample_height != 1
                    || key.width < want.width
                    || key.height < want.height
                    || !color_sync_layout_compatible(*key, want, vk::Format::D32_SFLOAT)
                    || image.base_format != vk::Format::D32_SFLOAT
                    || image.format != vk::Format::D32_SFLOAT
                    || image.aspects != vk::ImageAspectFlags::DEPTH
                    || image.layout == vk::ImageLayout::UNDEFINED
                    || self.depth_is_guest_stale(*key)
                {
                    return None;
                }
                let generation = self.depth_generations.get(key).copied().filter(|stamp| *stamp != 0)?;
                Some(RtDepthRegion {
                    key: *key,
                    image: image.image,
                    layout: image.layout,
                    generation,
                })
            })
            .max_by_key(|region| {
                (
                    region.generation,
                    region.key.same_alias_view_identity(want),
                    std::cmp::Reverse(u64::from(region.key.width) * u64::from(region.key.height)),
                )
            })
    }

    pub fn find_d24_depth_covering(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        let (key, img) = self
            .depth_cache
            .iter()
            .filter(|(key, img)| {
                img.base_format == vk::Format::D24_UNORM_S8_UINT
                    && !self.obsolete_mapping_views.contains(&(true, **key))
                    && img.aspects.contains(vk::ImageAspectFlags::DEPTH)
                    && same_d24_depth_allocation_covering(**key, want)
            })
            .min_by_key(|(key, _)| {
                (
                    key.width as u64 * key.height as u64 - want.width as u64 * want.height as u64,
                    key.width - want.width,
                    key.height - want.height,
                )
            })?;
        Some((
            *key,
            img.image,
            img.view,
            img.layout,
            img.format,
            img.aspects,
        ))
    }

    pub fn find_current_depth_shadow(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        for (shadow, (source, _)) in &self.depth_shadow_generations {
            let covers = source.gpu_va == want.gpu_va
                && dims_close(source.width, want.width)
                && dims_close(source.height, want.height);
            if !covers || self.obsolete_mapping_views.contains(&(true, *source))
                || self.obsolete_mapping_views.contains(&(true, *shadow))
                || !self.depth_shadow_is_current(*source, *shadow) {
                continue;
            }
            if let Some(img) = self.depth_cache.get(shadow) {
                if img.layout != vk::ImageLayout::UNDEFINED {
                    return Some((
                        *shadow,
                        img.image,
                        img.view,
                        img.layout,
                        img.format,
                        img.aspects,
                    ));
                }
            }
        }
        None
    }

    pub fn find_depth_fuzzy(
        &self,
        want: RtKey,
    ) -> Option<(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::ImageLayout,
        vk::Format,
        vk::ImageAspectFlags,
    )> {
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.depth_cache {
            if self.obsolete_mapping_views.contains(&(true, *k))
                || !dims_close(k.width, want.width) || !dims_close(k.height, want.height) {
                continue;
            }
            let kd = (k.width as i64 - want.width as i64).abs()
                + (k.height as i64 - want.height as i64).abs();
            let replace = match best {
                Some((bk, _)) => {
                    kd < (bk.width as i64 - want.width as i64).abs()
                        + (bk.height as i64 - want.height as i64).abs()
                }
                None => true,
            };
            if replace {
                best = Some((*k, img));
            }
        }
        best.map(|(k, img)| (k, img.image, img.view, img.layout, img.format, img.aspects))
    }

    pub fn find_color(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        self.find_color_with_format(want)
            .map(|(k, image, view, layout, _)| (k, image, view, layout))
    }

    pub(crate) fn color_base_format(&self, key: RtKey) -> Option<vk::Format> {
        self.cache.get(&key).map(|image| image.base_format)
    }

    pub fn find_color_with_format(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if let Some((stored, img)) = self.cache.get_key_value(&want) {
            let stored = *stored;
            if self.obsolete_mapping_views.contains(&(false, stored)) || alias_view_metadata_changed(stored, want) {
                return None;
            }
            return Some((stored, img.image, img.view, img.layout, img.format));
        }
        if is_synthetic_copy_key(want) {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k) || k.nvmap_id != want.nvmap_id
                || self.obsolete_mapping_views.contains(&(false, *k)) {
                continue;
            }
            if alias_view_metadata_changed(*k, want) {
                continue;
            }
            if want.gpu_va != 0 && k.gpu_va != want.gpu_va {
                continue;
            }
            if !dims_close(k.width, want.width) || !dims_close(k.height, want.height) {
                continue;
            }
            let kd = (k.width as i64 - want.width as i64).abs()
                + (k.height as i64 - want.height as i64).abs();
            let replace = match best {
                Some((bk, _)) => {
                    kd < (bk.width as i64 - want.width as i64).abs()
                        + (bk.height as i64 - want.height as i64).abs()
                }
                None => true,
            };
            if replace {
                best = Some((*k, img));
            }
        }
        best.map(|(k, img)| (k, img.image, img.view, img.layout, img.format))
    }

    pub fn find_sampleable_color_with_format(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        let (key, img) = if let Some((stored, img)) = self.cache.get_key_value(&want) {
            let stored = *stored;
            if alias_view_metadata_changed(stored, want) {
                return None;
            }
            (stored, img)
        } else if is_synthetic_copy_key(want) {
            return None;
        } else {
            self.cache
                .iter()
                .find(|(key, _)| {
                    !is_synthetic_copy_key(**key)
                        && same_physical_backing(**key, want)
                        && !alias_view_metadata_changed(**key, want)
                })
                .map(|(key, img)| (*key, img))?
        };
        if self.color_is_guest_stale(key) || img.layout == vk::ImageLayout::UNDEFINED {
            return None;
        }
        Some((key, img.image, img.view, img.layout, img.format))
    }

    pub fn find_sampleable_color_for_exact_alias(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if !rt_alias_indexed_lookup_enabled() {
            return self.find_sampleable_color_for_exact_alias_legacy(want);
        }
        let indexed = self.find_sampleable_color_for_exact_alias_indexed(want);
        if !rt_alias_indexed_lookup_shadow_enabled() {
            return indexed;
        }
        rt_alias_shadow_color_result(
            indexed,
            self.find_sampleable_color_for_exact_alias_legacy(want),
        )
    }

    fn find_sampleable_color_for_exact_alias_legacy(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        let (key, image) = self
            .cache
            .iter()
            .find(|(candidate, _)| exact_texture_alias_identity(**candidate, want))
            .map(|(key, image)| (*key, image))?;
        if self.color_is_guest_stale(key) || image.layout == vk::ImageLayout::UNDEFINED {
            return None;
        }
        Some((key, image.image, image.view, image.layout, image.format))
    }

    fn find_sampleable_color_for_exact_alias_indexed(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if want.gpu_va == 0 {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_WILDCARD_FALLBACKS, 1);
            return self.find_sampleable_color_for_exact_alias_legacy(want);
        }
        let Some((key, image)) = self
            .cache
            .get_key_value(&want)
            .filter(|(candidate, _)| exact_texture_alias_identity(**candidate, want))
            .map(|(key, image)| (*key, image))
        else {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_MISSES, 1);
            return None;
        };
        if self.color_is_guest_stale(key) || image.layout == vk::ImageLayout::UNDEFINED {
            count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_MISSES, 1);
            return None;
        }
        count_rt_alias_indexed(&RT_ALIAS_INDEXED_EXACT_HITS, 1);
        Some((key, image.image, image.view, image.layout, image.format))
    }

    pub fn find_sampleable_color_for_exact_or_physical_alias(
        &self,
        want: RtKey,
        view_format: vk::Format,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if !rt_alias_indexed_lookup_enabled() {
            return self
                .find_sampleable_color_for_exact_or_physical_alias_legacy(want, view_format);
        }
        let indexed =
            self.find_sampleable_color_for_exact_or_physical_alias_indexed(want, view_format);
        if !rt_alias_indexed_lookup_shadow_enabled() {
            return indexed;
        }
        rt_alias_shadow_color_result(
            indexed,
            self.find_sampleable_color_for_exact_or_physical_alias_legacy(want, view_format),
        )
    }

    fn find_sampleable_color_for_exact_or_physical_alias_legacy(
        &self,
        want: RtKey,
        view_format: vk::Format,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if let Some(exact) = self.find_sampleable_color_for_exact_alias_legacy(want) {
            return Some(exact);
        }
        if is_synthetic_copy_key(want) {
            return None;
        }
        let mut candidates = self.cache.iter().filter(|(candidate, image)| {
            !is_synthetic_copy_key(**candidate)
                && same_physical_backing(**candidate, want)
                && !alias_view_metadata_changed(**candidate, want)
                && rt_formats_compatible(image.base_format, view_format)
                && !self.color_is_guest_stale(**candidate)
                && image.layout != vk::ImageLayout::UNDEFINED
        });
        let (key, image) = candidates.next()?;
        if candidates.next().is_some() {
            return None;
        }
        let key = *key;
        Some((key, image.image, image.view, image.layout, image.format))
    }

    fn find_sampleable_color_for_exact_or_physical_alias_indexed(
        &self,
        want: RtKey,
        view_format: vk::Format,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if let Some(exact) = self.find_sampleable_color_for_exact_alias_indexed(want) {
            return Some(exact);
        }
        if is_synthetic_copy_key(want) || want.cpu_addr == 0 {
            return None;
        }
        let candidates = self.color_cpu_base_index.get(&want.cpu_addr)?;
        count_rt_alias_indexed(&RT_ALIAS_INDEXED_CPU_CANDIDATES, candidates.len() as u64);
        let mut candidates = candidates.iter().filter_map(|candidate| {
            let (key, image) = self.cache.get_key_value(candidate)?;
            (!is_synthetic_copy_key(*key)
                && same_physical_backing(*key, want)
                && !alias_view_metadata_changed(*key, want)
                && rt_formats_compatible(image.base_format, view_format)
                && !self.color_is_guest_stale(*key)
                && image.layout != vk::ImageLayout::UNDEFINED)
                .then_some((*key, image))
        });
        let (key, image) = candidates.next()?;
        if candidates.next().is_some() {
            return None;
        }
        Some((key, image.image, image.view, image.layout, image.format))
    }

    pub fn find_drawn_color_at(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let candidates = self.color_gpu_base_index.get(&gpu_va)?;
        self.find_drawn_color_at_candidates(width, height, gpu_va, candidates)
    }

    pub fn find_drawn_color_for_exact_alias(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        self.find_color_for_exact_alias(want, false)
    }

    pub fn find_content_bearing_color_for_exact_alias(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        self.find_color_for_exact_alias(want, true)
    }

    fn find_color_for_exact_alias(
        &self,
        want: RtKey,
        prefer_content: bool,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut best: Option<(RtKey, &GpuImage, u64, bool)> = None;
        let mut consider = |candidate: &RtKey| {
            let Some((stored, image)) = self.cache.get_key_value(candidate) else {
                return;
            };
            let stored = *stored;
            if !exact_texture_alias_identity(stored, want) || self.color_is_guest_stale(stored) {
                return;
            }
            let Some(stamp) = self.drawn_stamp.get(&stored).copied() else {
                return;
            };
            let has_content = self.frame_real_draws.get(&stored).copied().unwrap_or(0) != 0;
            let replace = match best {
                Some((_, _, best_stamp, best_content)) if prefer_content => {
                    (has_content && !best_content)
                        || (has_content == best_content && stamp > best_stamp)
                }
                Some((_, _, best_stamp, _)) => stamp > best_stamp,
                None => true,
            };
            if replace {
                best = Some((stored, image, stamp, has_content));
            }
        };

        if want.gpu_va != 0 {
            for candidate in self.color_gpu_base_index.get(&want.gpu_va)? {
                consider(candidate);
            }
        } else {
            for candidate in self.cache.keys() {
                consider(candidate);
            }
        }
        best.map(|(key, image, stamp, _)| (key, image.image, image.layout, image.format, stamp))
    }

    fn find_drawn_color_at_candidates<'a>(
        &'a self,
        width: u32,
        height: u32,
        gpu_va: u64,
        candidates: impl IntoIterator<Item = &'a RtKey>,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for candidate in candidates {
            let Some((stored, img)) = self.cache.get_key_value(candidate) else {
                continue;
            };
            let stored = *stored;
            if is_synthetic_copy_key(stored)
                || stored.gpu_va != gpu_va
                || stored.width < width
                || stored.height < height
            {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(&stored).copied() else {
                continue;
            };
            let area = stored.width as u64 * stored.height as u64;
            let replace = match best {
                Some((best_key, _, best_stamp)) => {
                    let best_area = best_key.width as u64 * best_key.height as u64;
                    stamp > best_stamp || (stamp == best_stamp && area < best_area)
                }
                None => true,
            };
            if replace {
                best = Some((stored, img, stamp));
            }
        }
        best.map(|(k, img, stamp)| (k, img.image, img.layout, img.format, stamp))
    }

    #[cfg(test)]
    fn find_drawn_color_at_full_scan(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        self.find_drawn_color_at_candidates(width, height, gpu_va, self.cache.keys())
    }

    pub fn find_content_bearing_color_at(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut best: Option<(RtKey, &GpuImage, u64, bool)> = None;
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k)
                || k.gpu_va != gpu_va
                || k.width < width
                || k.height < height
            {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            let content = self.frame_real_draws.get(k).copied().unwrap_or(0) > 0;
            let area = k.width as u64 * k.height as u64;
            let replace = match best {
                Some((best_key, _, best_stamp, best_content)) => {
                    let best_area = best_key.width as u64 * best_key.height as u64;
                    match (content, best_content) {
                        (true, false) => true,
                        (false, true) => false,
                        _ => stamp > best_stamp || (stamp == best_stamp && area < best_area),
                    }
                }
                None => true,
            };
            if replace {
                best = Some((*k, img, stamp, content));
            }
        }
        best.map(|(k, img, stamp, _)| (k, img.image, img.layout, img.format, stamp))
    }

    pub fn find_drawn_color_at_cpu(
        &self,
        width: u32,
        height: u32,
        nvmap_id: u32,
        cpu_addr: u64,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        if cpu_addr == 0 {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k)
                || k.nvmap_id != nvmap_id
                || k.cpu_addr != cpu_addr
                || k.width < width
                || k.height < height
            {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            let area = k.width as u64 * k.height as u64;
            let replace = match best {
                Some((best_key, _, best_stamp)) => {
                    let best_area = best_key.width as u64 * best_key.height as u64;
                    stamp > best_stamp || (stamp == best_stamp && area < best_area)
                }
                None => true,
            };
            if replace {
                best = Some((*k, img, stamp));
            }
        }
        best.map(|(k, img, stamp)| (k, img.image, img.layout, img.format, stamp))
    }

    pub fn find_drawn_color_region_at(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<RtColorRegion> {
        self.find_drawn_color_region_at_inner(width, height, gpu_va, None)
    }

    pub fn find_drawn_color_region_at_excluding(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
        excluded: RtKey,
    ) -> Option<RtColorRegion> {
        self.find_drawn_color_region_at_inner(width, height, gpu_va, Some(excluded))
    }

    fn find_drawn_color_region_at_inner(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
        excluded: Option<RtKey>,
    ) -> Option<RtColorRegion> {
        if gpu_va == 0 {
            return None;
        }
        let candidates = self
            .color_gpu_page_index
            .get(&(gpu_va >> RT_COLOR_GPU_PAGE_SHIFT))?;
        self.find_drawn_color_region_at_candidates(width, height, gpu_va, excluded, candidates)
    }

    fn find_drawn_color_region_at_candidates<'a>(
        &'a self,
        width: u32,
        height: u32,
        gpu_va: u64,
        excluded: Option<RtKey>,
        candidates: impl IntoIterator<Item = &'a RtKey>,
    ) -> Option<RtColorRegion> {
        let mut best: Option<(RtKey, &GpuImage, u64, u32, u32, bool)> = None;
        for candidate in candidates {
            let Some((stored, img)) = self.cache.get_key_value(candidate) else {
                continue;
            };
            let stored = *stored;
            if is_synthetic_copy_key(stored) || excluded == Some(stored) {
                continue;
            }
            if excluded.is_some_and(|want| {
                !color_region_sync_identity(stored, want)
                    || !color_sync_layout_compatible(stored, want, img.format)
                    || (stored.layout_signature != 0
                        && want.layout_signature != 0
                        && stored.gpu_va != want.gpu_va)
            }) {
                continue;
            }
            let Some((src_x, src_y, exact)) =
                rt_region_offset(stored, img.format, width, height, gpu_va)
            else {
                continue;
            };
            let stamp = if excluded.is_some() {
                if self.color_is_guest_stale(stored) || img.layout == vk::ImageLayout::UNDEFINED {
                    continue;
                }
                self.writeback_stamp(stored)
            } else {
                self.drawn_stamp.get(&stored).copied()
            };
            let Some(stamp) = stamp else {
                continue;
            };
            let area = stored.width as u64 * stored.height as u64;
            let replace = match best {
                Some((_, _, best_stamp, _, _, _))
                    if excluded.is_some() && stamp != best_stamp =>
                {
                    stamp > best_stamp
                }
                Some((best_key, _, best_stamp, _, _, best_exact)) => {
                    let best_area = best_key.width as u64 * best_key.height as u64;
                    (exact && !best_exact)
                        || (exact == best_exact
                            && (area < best_area || (area == best_area && stamp > best_stamp)))
                }
                None => true,
            };
            if replace {
                best = Some((stored, img, stamp, src_x, src_y, exact));
            }
        }
        best.map(|(key, img, stamp, src_x, src_y, _)| RtColorRegion {
            key,
            image: img.image,
            layout: img.layout,
            format: img.format,
            stamp,
            src_x,
            src_y,
        })
    }

    #[cfg(test)]
    fn find_drawn_color_region_at_full_scan(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
        excluded: Option<RtKey>,
    ) -> Option<RtColorRegion> {
        self.find_drawn_color_region_at_candidates(
            width,
            height,
            gpu_va,
            excluded,
            self.cache.keys(),
        )
    }

    pub fn find_drawn_color_region_at_cpu(
        &self,
        width: u32,
        height: u32,
        nvmap_id: u32,
        cpu_addr: u64,
    ) -> Option<RtColorRegion> {
        if cpu_addr == 0 {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage, u64, u32, u32, bool)> = None;
        for (k, img) in self.color_entries_for_nvmap(nvmap_id) {
            if is_synthetic_copy_key(*k) {
                continue;
            }
            let Some((src_x, src_y, exact)) =
                rt_region_cpu_offset(*k, img.format, width, height, cpu_addr)
            else {
                continue;
            };
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            let area = k.width as u64 * k.height as u64;
            let replace = match best {
                Some((best_key, _, best_stamp, _, _, best_exact)) => {
                    let best_area = best_key.width as u64 * best_key.height as u64;
                    (exact && !best_exact)
                        || (exact == best_exact
                            && (area < best_area || (area == best_area && stamp > best_stamp)))
                }
                None => true,
            };
            if replace {
                best = Some((*k, img, stamp, src_x, src_y, exact));
            }
        }
        best.map(|(key, img, stamp, src_x, src_y, _)| RtColorRegion {
            key,
            image: img.image,
            layout: img.layout,
            format: img.format,
            stamp,
            src_x,
            src_y,
        })
    }

    pub fn set_color_layout(&mut self, key: RtKey, layout: vk::ImageLayout) {
        if let Some(img) = self.cache.get_mut(&key) {
            let changed = img.layout != layout;
            img.layout = layout;
            if changed { self.note_color_sampling_changed(key.nvmap_id); }
        }
    }

    pub fn color_layout(&self, key: RtKey) -> Option<vk::ImageLayout> {
        self.cache.get(&key).map(|img| img.layout)
    }

    pub fn color_extent(&self, key: RtKey) -> Option<vk::Extent2D> {
        self.cache.get(&key).map(|img| img.extent)
    }

    pub fn set_depth_layout(&mut self, key: RtKey, layout: vk::ImageLayout) {
        let image = self.depth_cache.get(&key).map(|img| img.image).or_else(|| {
            self.depth_cache
                .iter()
                .find(|(existing, _)| same_physical_backing(**existing, key))
                .map(|(_, img)| img.image)
        });
        let Some(image) = image else {
            return;
        };
        for img in self
            .depth_cache
            .values_mut()
            .filter(|img| img.image == image)
        {
            img.layout = layout;
        }
    }

    pub fn depth_layout(&self, key: RtKey) -> Option<vk::ImageLayout> {
        self.depth_cache
            .get(&key)
            .map(|img| img.layout)
            .or_else(|| {
                self.depth_cache
                    .iter()
                    .find(|(existing, _)| same_physical_backing(**existing, key))
                    .map(|(_, img)| img.layout)
            })
    }

    fn create_image(
        &self,
        device: &ash::Device,
        key: RtKey,
        format: vk::Format,
    ) -> Result<GpuImage, String> {
        self.create_image_inner(
            device,
            key,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )
    }

    fn create_image_inner(
        &self,
        device: &ash::Device,
        key: RtKey,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<GpuImage, String> {
        let mem_props = self
            .mem_properties
            .ok_or_else(|| "RtCache: memory properties not set".to_string())?;
        let extent = vk::Extent2D {
            width: key.width,
            height: key.height,
        };

        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            image_type: if key.is_3d {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            },
            format,
            extent: vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: if key.is_3d { key.depth.max(1) } else { 1 },
            },
            mip_levels: 1,
            array_layers: if key.is_3d { 1 } else { key.render_layer_count() },
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            p_next: std::ptr::null(),
            flags: (if aspect.contains(vk::ImageAspectFlags::COLOR) {
                vk::ImageCreateFlags::MUTABLE_FORMAT
            } else {
                vk::ImageCreateFlags::empty()
            }) | if key.is_3d {
                vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            },
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .map_err(|e| format!("create_image: {:?}", e))?
        };

        let req = unsafe { device.get_image_memory_requirements(image) };
        let Some(mem_type) = find_memory_type(
            &mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            unsafe { device.destroy_image(image, None) };
            return Err("RtCache: no suitable DEVICE_LOCAL memory type".to_string());
        };

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: req.size,
            memory_type_index: mem_type,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(format!("allocate_memory: {:?}", error));
            }
        };
        if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(format!("bind_image_memory: {:?}", error));
        }

        let view = match create_image_view(device, image, format, aspect, key) {
            Ok(view) => view,
            Err(error) => {
                unsafe {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                }
                return Err(error);
            }
        };
        let mut views = std::collections::HashMap::default();
        views.insert(format, view);
        note_rt_memory_alloc(memory, req.size, key);

        Ok(GpuImage {
            image,
            view,
            views,
            sample_views: HashMap::default(),
            memory,
            format,
            base_format: format,
            aspects: aspect,
            extent,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn clear(&mut self, device: &ash::Device) {
        self.color_sampling_generation = self.color_sampling_generation.wrapping_add(1);
        if let Some(pipeline) = self.depth_pack_pipeline.take() {
            pipeline.destroy(device);
        }
        self.clear_color_lookup_index();
        self.depth_cpu_base_index.clear();
        self.clear_stamp.clear();
        self.full_clear_stamp.clear();
        self.depth_full_clear_generation.clear();
        self.color_region_sources.clear();
        self.depth_region_sources.clear();
        self.pending_view_retirement.clear();
        self.obsolete_mapping_views.clear();
        self.view_retirement_epoch = 0;
        self.note_guest_ranges_changed();
        self.guest_hit_memo.clear();
        self.color_guest_ranges.clear();
        self.color_guest_range_index.clear();
        self.depth_guest_ranges.clear();
        self.depth_guest_range_index.clear();
        for (_, img) in self
            .cache
            .drain()
            .chain(self.depth_cache.drain())
            .chain(self.snapshots.drain())
        {
            destroy_gpu_image(device, img);
        }
        self.depth_generations.clear();
        self.depth_shadow_generations.clear();
    }
}

const RT_CACHE_MIN_IDLE_FRAMES: u64 = 4;

fn rt_cache_budget_bytes() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("NEXIUM_RT_CACHE_BUDGET_MB")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024)
    })
}

fn image_estimated_bytes(key: RtKey, image: &GpuImage) -> u64 {
    u64::from(image.extent.width)
        .saturating_mul(u64::from(image.extent.height))
        .saturating_mul(u64::from(key.depth))
        .saturating_mul(rt_format_bytes(image.format))
}

fn depth_image_requires_recreate(
    existing_format: vk::Format,
    existing_aspects: vk::ImageAspectFlags,
    requested_format: vk::Format,
    requested_aspects: vk::ImageAspectFlags,
) -> bool {
    existing_format != requested_format || existing_aspects != requested_aspects
}

fn get_or_create_sample_view(
    device: &ash::Device,
    image: &mut GpuImage,
    source_image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    view_type: vk::ImageViewType,
    components: vk::ComponentMapping,
) -> Result<vk::ImageView, String> {
    let cache_key = RtSampleViewKey::new(format, aspect, view_type, components);
    if let Some(view) = image.sample_views.get(&cache_key) {
        return Ok(*view);
    }

    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image: source_image,
        view_type,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: if view_type == vk::ImageViewType::TYPE_2D_ARRAY {
                vk::REMAINING_ARRAY_LAYERS
            } else {
                1
            },
        },
        components,
        p_next: std::ptr::null(),
        flags: vk::ImageViewCreateFlags::empty(),
        _marker: std::marker::PhantomData,
    };
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|error| format!("create_image_view(rt sample): {:?}", error))?
    };
    image.sample_views.insert(cache_key, view);
    Ok(view)
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    key: RtKey,
) -> Result<vk::ImageView, String> {
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if key.render_layer_count() > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: key.render_layer_count(),
        },
        components: vk::ComponentMapping::default(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("create_image_view: {:?}", e))
    }
}

fn destroy_gpu_image(device: &ash::Device, image: GpuImage) {
    unsafe {
        for view in image.sample_views.into_values() {
            device.destroy_image_view(view, None);
        }
        for view in image.views.into_values() {
            device.destroy_image_view(view, None);
        }
        device.destroy_image(image.image, None);
        note_rt_memory_free(image.memory);
        device.free_memory(image.memory, None);
    }
}

fn rt_live_memory() -> Option<&'static std::sync::Mutex<HashMap<u64, (u64, RtKey)>>> {
    static LIVE: OnceLock<Option<std::sync::Mutex<HashMap<u64, (u64, RtKey)>>>> = OnceLock::new();
    LIVE.get_or_init(|| {
        std::env::var_os("NEXIUM_GPU_MEMORY_PROFILE")
            .map(|_| std::sync::Mutex::new(HashMap::default()))
    })
    .as_ref()
}

fn note_rt_memory_alloc(memory: vk::DeviceMemory, size: u64, key: RtKey) {
    use vk::Handle;
    if let Some(live) = rt_live_memory() {
        if let Ok(mut live) = live.lock() {
            live.insert(memory.as_raw(), (size, key));
        }
    }
}

fn note_rt_memory_free(memory: vk::DeviceMemory) {
    use vk::Handle;
    if let Some(live) = rt_live_memory() {
        if let Ok(mut live) = live.lock() {
            live.remove(&memory.as_raw());
        }
    }
}

pub fn rt_formats_compatible(base: vk::Format, view: vk::Format) -> bool {
    base == view
        || rt_format_compatibility_class(base)
            .zip(rt_format_compatibility_class(view))
            .is_some_and(|(a, b)| a == b)
}

fn rt_format_compatibility_class(format: vk::Format) -> Option<u32> {
    rt_format_class_bits(format)
}

fn rt_guest_ranges_overlap(
    key: RtKey,
    image: &GpuImage,
    cpu_addr: u64,
    size: u64,
    gpu_ranges: &[(u64, u64)],
) -> bool {
    let Some(image_size) = rt_guest_size_bytes(key, image.base_format) else {
        return false;
    };
    if image_size == 0 {
        return false;
    }
    (key.cpu_addr != 0 && ranges_overlap(key.cpu_addr, image_size, cpu_addr, size))
        || (key.gpu_va != 0
            && gpu_ranges.iter().any(|(gpu_va, gpu_size)| {
                ranges_overlap(key.gpu_va, image_size, *gpu_va, *gpu_size)
            }))
}

fn ranges_overlap(left: u64, left_size: u64, right: u64, right_size: u64) -> bool {
    left_size != 0
        && right_size != 0
        && left < right.saturating_add(right_size)
        && right < left.saturating_add(left_size)
}

fn rt_color_gpu_page_span(key: RtKey, format: vk::Format) -> Option<(u64, u64)> {
    if key.gpu_va == 0 || key.width == 0 || key.height == 0 {
        return None;
    }
    let size = rt_guest_size_bytes(key, format)?;
    let end = key.gpu_va.checked_add(size.checked_sub(1)?)?;
    Some((
        key.gpu_va >> RT_COLOR_GPU_PAGE_SHIFT,
        end >> RT_COLOR_GPU_PAGE_SHIFT,
    ))
}

fn remove_lookup_key<K: std::hash::Hash + Eq>(
    index: &mut HashMap<K, Vec<RtKey>>,
    bucket: K,
    key: RtKey,
) {
    let remove_bucket = if let Some(keys) = index.get_mut(&bucket) {
        keys.retain(|candidate| *candidate != key);
        keys.is_empty()
    } else {
        false
    };
    if remove_bucket {
        index.remove(&bucket);
    }
}

fn rt_region_offset(
    key: RtKey,
    format: vk::Format,
    width: u32,
    height: u32,
    gpu_va: u64,
) -> Option<(u32, u32, bool)> {
    if key.gpu_va == 0 || width == 0 || height == 0 || key.width == 0 || key.height == 0 {
        return None;
    }
    if gpu_va < key.gpu_va || width > key.width || height > key.height {
        return None;
    }
    let offset = gpu_va.checked_sub(key.gpu_va)?;
    rt_region_from_offset(key, format, width, height, offset)
}

pub fn rt_color_region_covers(
    key: RtKey,
    format: vk::Format,
    width: u32,
    height: u32,
    gpu_va: u64,
) -> bool {
    rt_region_offset(key, format, width, height, gpu_va).is_some()
}

fn rt_region_cpu_offset(
    key: RtKey,
    format: vk::Format,
    width: u32,
    height: u32,
    cpu_addr: u64,
) -> Option<(u32, u32, bool)> {
    if key.cpu_addr == 0 || width == 0 || height == 0 || key.width == 0 || key.height == 0 {
        return None;
    }
    if cpu_addr < key.cpu_addr || width > key.width || height > key.height {
        return None;
    }
    let offset = cpu_addr.checked_sub(key.cpu_addr)?;
    rt_region_from_offset(key, format, width, height, offset)
}

fn rt_region_from_offset(
    key: RtKey,
    format: vk::Format,
    width: u32,
    height: u32,
    offset: u64,
) -> Option<(u32, u32, bool)> {
    let bpp = rt_format_bytes(format);
    if bpp == 0 || offset % bpp != 0 {
        return None;
    }
    let row = (key.width as u64).checked_mul(bpp)?;
    if row == 0 {
        return None;
    }
    let src_y = offset / row;
    let src_x_bytes = offset % row;
    if src_x_bytes % bpp != 0 {
        return None;
    }
    let src_x = src_x_bytes / bpp;
    if src_x.checked_add(width as u64)? > key.width as u64
        || src_y.checked_add(height as u64)? > key.height as u64
    {
        return None;
    }
    let exact = offset == 0 && width == key.width && height == key.height;
    Some((src_x as u32, src_y as u32, exact))
}

pub fn rt_format_bytes(format: vk::Format) -> u64 {
    match format {
        vk::Format::D32_SFLOAT_S8_UINT => 8,
        vk::Format::B5G5R5A1_UNORM_PACK16 | vk::Format::D16_UNORM => 2,
        vk::Format::A2R10G10B10_UNORM_PACK32
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT
        | vk::Format::X8_D24_UNORM_PACK32 => 4,
        _ => rt_format_class_bits(format)
            .map(|bits| (bits / 8) as u64)
            .unwrap_or(4),
    }
}

fn rt_format_class_bits(format: vk::Format) -> Option<u32> {
    match format {
        vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_UINT => Some(128),
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_UINT => Some(64),
        vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R32_SFLOAT
        | vk::Format::R32_SINT
        | vk::Format::R32_UINT
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2B10G10R10_UINT_PACK32
        | vk::Format::A2B10G10R10_SINT_PACK32
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SNORM_PACK32
        | vk::Format::A8B8G8R8_SINT_PACK32
        | vk::Format::A8B8G8R8_UINT_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B10G11R11_UFLOAT_PACK32 => Some(32),
        vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_SINT
        | vk::Format::R16_UINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_UINT
        | vk::Format::R5G6B5_UNORM_PACK16 => Some(16),
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_SINT | vk::Format::R8_UINT => {
            Some(8)
        }
        _ => None,
    }
}

fn rt_format_dbg_enabled(nvmap_id: u32) -> bool {
    if std::env::var_os("NEXIUM_RT_FORMAT_DBG").is_none() {
        return false;
    }
    match std::env::var("NEXIUM_RT_FORMAT_NVMAPS") {
        Ok(list) => list.split(',').any(|item| {
            item.trim()
                .parse::<u32>()
                .is_ok_and(|want| want == nvmap_id)
        }),
        Err(_) => true,
    }
}

pub fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        let t = &props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0 && t.property_flags.contains(required) {
            return Some(i);
        }
    }
    None
}

impl Drop for RtCache {
    fn drop(&mut self) {
        if !self.cache.is_empty() {
            log::warn!(
                "RtCache dropped with {} images still cached",
                self.cache.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_color_content_stamp_includes_full_and_partial_clears() {
        let mut cache = RtCache::new();
        let key = RtKey::with_cpu(1, 96, 90, 0x1000, 0x9000);
        let alias = RtKey::with_cpu(1, 128, 128, 0x1000, 0x9000);
        cache.insert_color_image(key,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM));
        let old_draw = cache.mark_drawn(alias);
        cache.mark_cleared(key, true);
        let full = cache.color_exact_with_format(key).unwrap().5;
        assert!(full > old_draw);
        assert_eq!(cache.drawn_stamp(key), None);
        cache.mark_cleared(key, false);
        let partial = cache.color_exact_with_format(key).unwrap().5;
        assert!(partial > full);
        assert!(cache.mark_drawn(alias) > partial);
        cache.remove_color_image(key).unwrap();
    }

    #[test]
    fn clears_order_after_older_draws_without_becoming_drawn_targets() {
        let mut cache = RtCache::new();
        let first = RtKey::with_cpu(1, 64, 64, 0x1000, 0x9000);
        let alias = RtKey::with_cpu(2, 32, 32, 0x2000, 0x9000);
        let original = cache.mark_drawn(first);
        let later = cache.mark_drawn(alias);
        cache.mark_cleared(first, true);
        let full_clear = cache.writeback_stamp(first).unwrap();
        assert!(full_clear > later && later > original);
        assert_eq!(cache.drawn_stamp(first), None);
        assert_eq!(cache.writeback_stamp(alias), Some(later));

        cache.mark_cleared(first, false);
        assert!(cache.writeback_stamp(first).unwrap() > full_clear);
        assert_eq!(cache.drawn_stamp(first), None);
        let redraw = cache.mark_drawn(first);
        assert_eq!(cache.writeback_stamp(first), Some(redraw));
        cache.mark_cleared(first, false);
        assert!(cache.writeback_stamp(first).unwrap() > redraw);
        assert_eq!(cache.writeback_stamp(first), cache.drawn_stamp(first));

        cache.mark_synced_sample_from(first, original);
        assert_eq!(cache.writeback_stamp(first), Some(original));
    }

    #[test]
    fn clear_write_order_tracks_rekeys_invalidation_and_replacement() {
        let mut cache = RtCache::new();
        let key = RtKey::with_cpu(1, 64, 64, 0x1000, 0x9000)
            .with_mapping_epoch(1);
        cache.insert_color_image(key,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM));
        cache.mark_cleared(key, true);
        let stamp = cache.writeback_stamp(key);
        let rekeyed = key.with_mapping_epoch(2);
        assert!(cache.rekey_color_state(key, rekeyed));
        assert_eq!(cache.writeback_stamp(rekeyed), stamp);
        assert_eq!(cache.clear_stamp.keys().next().unwrap().mapping_epoch, 2);

        cache.mark_guest_written_range(0x9000, 1, &[]);
        assert_eq!(cache.writeback_stamp(rekeyed), None);
        cache.mark_cleared(rekeyed, true);
        cache.mark_guest_uploaded(rekeyed);
        assert_eq!(cache.writeback_stamp(rekeyed), None);
        cache.mark_cleared(rekeyed, true);
        cache.remove_color_image(rekeyed).unwrap();
        assert_eq!(cache.writeback_stamp(rekeyed), None);
    }

    use super::{
        is_synthetic_copy_key, rt_alias_indexed_lookup_value_enabled, rt_formats_compatible,
        rt_sampleable_color_alias_equal, rt_sampleable_depth_alias_equal,
        same_d24_depth_allocation_covering, same_physical_backing, GpuImage, RtCache,
        RtColorRegion, RtKey, RtMappingEpochTransition, RtSampleViewKey,
    };
    use ash::vk;
    use ash::vk::Handle;
    use std::collections::HashMap;

    fn test_depth_image() -> GpuImage {
        GpuImage {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            views: HashMap::default(),
            sample_views: HashMap::default(),
            memory: vk::DeviceMemory::null(),
            format: vk::Format::D24_UNORM_S8_UINT,
            base_format: vk::Format::D24_UNORM_S8_UINT,
            aspects: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            extent: vk::Extent2D {
                width: 1600,
                height: 900,
            },
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }

    fn test_d32_image(key: RtKey) -> GpuImage {
        GpuImage {
            format: vk::Format::D32_SFLOAT,
            base_format: vk::Format::D32_SFLOAT,
            aspects: vk::ImageAspectFlags::DEPTH,
            extent: vk::Extent2D { width: key.width, height: key.height },
            ..test_depth_image()
        }
    }

    fn retirement_key(width: u32, height: u32) -> RtKey {
        RtKey::with_cpu(12, width, height, 0x5200_0000, 0x9000_0000)
            .with_mapping_epoch(40)
            .with_block_linear_layout(0, 4, 0, 0)
    }

    fn retirement_color(key: RtKey, handle: u64) -> GpuImage {
        GpuImage {
            image: vk::Image::from_raw(handle),
            extent: vk::Extent2D { width: key.width, height: key.height },
            ..test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM)
        }
    }

    fn retire_aged_views(cache: &mut RtCache, retired: &mut Vec<GpuImage>, protected: &[RtKey]) -> (usize, u64) {
        cache.view_retirement_epoch = cache.view_retirement_epoch.wrapping_add(64);
        cache.retire_redundant_views(retired, protected)
    }

    fn insert_retirement_depth(cache: &mut RtCache, key: RtKey, handle: u64) {
        let mut image = test_d32_image(key);
        image.image = vk::Image::from_raw(handle);
        cache.extend_guest_bounds(key, image.base_format);
        super::insert_guest_range(
            &mut cache.depth_guest_ranges, &mut cache.depth_guest_range_index, key, image.base_format,
        );
        cache.insert_depth_image(key, image);
        cache.mark_depth_written(key);
    }

    #[test]
    fn depth_cpu_index_matches_full_scan_for_exact_physical_and_wildcard_queries() {
        let mut cache = RtCache::new();
        let source = retirement_key(32, 32);
        let alias = RtKey { gpu_va: source.gpu_va + 0x10000, ..source };
        let different_layer = RtKey { gpu_va: source.gpu_va + 0x20000, base_layer: 1, ..source };
        let old_mapping = RtKey { gpu_va: source.gpu_va + 0x30000, mapping_epoch: 39, ..source };
        let different_layout = RtKey { gpu_va: source.gpu_va + 0x40000, ..source }
            .with_pitch_linear_layout(128);
        let no_cpu = RtKey { gpu_va: source.gpu_va + 0x50000, cpu_addr: 0, ..source };
        for (index, key) in [source, alias, different_layer, old_mapping, different_layout, no_cpu]
            .into_iter().enumerate()
        {
            insert_retirement_depth(&mut cache, key, index as u64 + 1);
        }
        for index in 0..256 {
            let key = RtKey { nvmap_id: 20 + index, gpu_va: 0x6000_0000 + u64::from(index) * 0x10000,
                cpu_addr: 0xa000_0000 + u64::from(index) * 0x10000, ..source };
            insert_retirement_depth(&mut cache, key, u64::from(index) + 100);
        }
        let query = RtKey { gpu_va: source.gpu_va + 0xf0000, ..source };
        let queries = [source, alias, query, different_layer, old_mapping, different_layout, no_cpu,
            query.with_mapping_epoch(0), query.with_mapping_epoch(999),
            RtKey { layout_signature: 0, ..query }, RtKey { base_layer: 1, ..query },
            RtKey { cpu_addr: 0, ..query }, RtKey { cpu_addr: 0x1234, ..query },
            RtKey { width: 31, ..query }, RtKey { sample_width: 2, ..query },
            RtKey::request(source.nvmap_id, 32, 32),
            RtKey::with_cpu(source.nvmap_id, 31, 33, 0, source.cpu_addr)];
        for obsolete in [None, Some(source), Some(alias)] {
            if let Some(key) = obsolete { cache.obsolete_mapping_views.insert((true, key)); }
            for want in queries {
                assert!(rt_sampleable_depth_alias_equal(
                    &cache.find_depth(want), &cache.find_depth_inner(want, false),
                ), "{want:?}");
            }
        }
        assert!(cache.find_depth(RtKey { cpu_addr: 0x1234, ..query }).is_none());
        assert!(!cache.depth_cpu_base_index.contains_key(&0));
    }

    #[test]
    fn depth_cpu_index_tracks_canonical_rekeys_removal_and_mapping_retirement() {
        let mut cache = RtCache::new();
        let source = retirement_key(32, 32).with_guest_size_bytes(4096);
        let alias = RtKey { gpu_va: source.gpu_va + 0x10000, ..source };
        insert_retirement_depth(&mut cache, source, 1);
        insert_retirement_depth(&mut cache, alias, 2);
        let query = RtKey { gpu_va: source.gpu_va + 0x20000, ..source };
        cache.insert_depth_image(source.with_mapping_epoch(900), test_d32_image(source));
        assert_eq!(cache.depth_cpu_base_index[&source.cpu_addr].len(), 2);
        assert!(cache.depth_cpu_base_index[&source.cpu_addr].iter()
            .any(|key| key.same_live_identity(source)));
        let remapped = RtKey { cpu_addr: source.cpu_addr + 0x40000, mapping_epoch: 41,
            guest_size_bytes: 8192, ..source };
        assert!(cache.rekey_depth_state(source, remapped));
        assert_eq!(cache.depth_cpu_base_index[&source.cpu_addr], [alias]);
        assert!(cache.depth_cpu_base_index[&remapped.cpu_addr][0].same_live_identity(remapped));
        let wanted = RtKey { gpu_va: query.gpu_va, ..remapped };
        assert!(cache.find_depth(wanted).unwrap().0.same_live_identity(remapped));
        cache.remove_depth_image(alias).unwrap();
        assert!(!cache.depth_cpu_base_index.contains_key(&source.cpu_addr));
        assert!(cache.find_depth(query).is_none());
        cache.apply_mapping_epoch_transitions(&[], &[(remapped.gpu_va, 1)]);
        assert!(cache.find_depth(wanted).is_none());
        let mut retired = Vec::new();
        assert_eq!(cache.retire_redundant_views(&mut retired, &[]).0, 1);
        assert!(cache.depth_cache.is_empty());
        assert!(cache.depth_cpu_base_index.is_empty());
    }

    #[test]
    fn rt_retirement_detaches_known_obsolete_mapping_images_and_tracking() {
        let mut cache = RtCache::new();
        let key = retirement_key(32, 32).with_guest_size_bytes(4096);
        cache.insert_color_image(key, retirement_color(key, 1));
        cache.snapshots.insert(key, retirement_color(key, 2));
        cache.mark_drawn(key);
        cache.record_present_flip(key, true);
        insert_retirement_depth(&mut cache, key, 3);
        let shadow = RtKey { gpu_va: key.gpu_va + 0x10000, ..key };
        cache.depth_shadow_generations.insert(shadow, (key, 1));
        cache.depth_array_shadow_generations.insert(shadow, vec![(key, 1)]);
        let sampling_generation = cache.color_sampling_generation(key.nvmap_id);
        cache.apply_mapping_epoch_transitions(&[], &[(key.gpu_va, 4096)]);
        assert_eq!(cache.guest_stale_counts(), (1, 1));
        assert!(cache.has_pending_view_retirement());
        let mut retired = Vec::new();
        assert_eq!(cache.retire_redundant_views(&mut retired, &[]), (3, 12288));
        assert_eq!(retired.len(), 3);
        for handle in 1..=3 {
            assert!(retired.iter().any(|image| image.image == vk::Image::from_raw(handle)));
        }
        assert_eq!(cache.memory_estimates(), [(0, 0); 3]);
        assert_eq!(cache.guest_stale_counts(), (0, 0));
        assert!(!cache.has_pending_view_retirement());
        assert!(cache.color_gpu_base_index.is_empty());
        assert!(cache.color_cpu_base_index.is_empty());
        assert!(cache.color_gpu_page_index.is_empty());
        assert!(cache.color_nvmap_index.is_empty());
        assert!(cache.color_guest_ranges.is_empty());
        assert!(cache.color_guest_range_index.is_empty());
        assert!(cache.depth_guest_ranges.is_empty());
        assert!(cache.depth_guest_range_index.is_empty());
        assert!(cache.depth_generations.is_empty());
        assert!(cache.depth_shadow_generations.is_empty());
        assert!(cache.depth_array_shadow_generations.is_empty());
        assert!(cache.present_flip_y(key).is_none());
        assert!(cache.color_sampling_generation(key.nvmap_id) != sampling_generation);
        cache.mark_guest_written_range(0, 0, &[(key.gpu_va, 4096)]);
        assert_eq!(cache.guest_stale_counts(), (0, 0));
    }

    #[test]
    fn rt_retirement_preserves_protected_obsolete_mappings_until_next_boundary() {
        let mut cache = RtCache::new();
        let key = retirement_key(32, 32);
        cache.insert_color_image(key, retirement_color(key, 1));
        cache.mark_drawn(key);
        insert_retirement_depth(&mut cache, key, 2);
        cache.apply_mapping_epoch_transitions(&[], &[(key.gpu_va, 1)]);
        let mut retired = Vec::new();
        for _ in 0..128 {
            assert_eq!(cache.retire_redundant_views(&mut retired, &[key]), (0, 0));
        }
        assert!(cache.cache.contains_key(&key));
        assert!(cache.depth_cache.contains_key(&key));
        assert_eq!(cache.retire_redundant_views(&mut retired, &[]), (2, 8192));
        assert!(!cache.has_pending_view_retirement());
        assert_eq!(retired.len(), 2);
    }

    #[test]
    fn rt_retirement_preserves_unknown_epochs_synthetic_views_and_cpu_writes() {
        let mut cache = RtCache::new();
        let known = retirement_key(32, 32).with_guest_size_bytes(4096);
        let unknown = RtKey { gpu_va: known.gpu_va + 0x10000, mapping_epoch: 0, ..known };
        let synthetic = RtKey { gpu_va: known.gpu_va + 0x20000, nvmap_id: u32::MAX, ..known };
        for (index, key) in [known, unknown, synthetic].into_iter().enumerate() {
            cache.insert_color_image(key, retirement_color(key, index as u64 + 1));
            cache.mark_drawn(key);
            insert_retirement_depth(&mut cache, key, index as u64 + 4);
        }
        cache.mark_all_guest_written();
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
        cache.apply_mapping_epoch_transitions(&[], &[(unknown.gpu_va, 0x20000)]);
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
        assert_eq!(cache.cache.len(), 3);
        assert_eq!(cache.depth_cache.len(), 3);
        assert!(!cache.has_pending_view_retirement());
    }

    #[test]
    fn rt_retirement_does_not_revive_obsolete_mapping_with_wildcard_metadata() {
        let mut cache = RtCache::new();
        let key = retirement_key(32, 32);
        cache.insert_color_image(key, retirement_color(key, 1));
        cache.mark_drawn(key);
        insert_retirement_depth(&mut cache, key, 2);
        cache.apply_mapping_epoch_transitions(&[], &[(key.gpu_va, 1)]);
        let wildcard = RtKey::new(key.nvmap_id, key.width, key.height, key.gpu_va);
        assert!(cache.color_requires_recreate(wildcard, vk::Format::R8G8B8A8_UNORM));
        assert!(cache.depth_requires_recreate(wildcard, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH));
        assert!(cache.get_existing(wildcard).is_none());
        assert!(cache.get_existing_depth(wildcard).is_none());
    }

    #[test]
    fn rt_retirement_obsolete_lookup_routes_agree_before_detachment() {
        let mut cache = RtCache::new();
        let key = retirement_key(32, 32);
        cache.insert_color_image(key, retirement_color(key, 1));
        cache.mark_drawn(key);
        insert_retirement_depth(&mut cache, key, 2);
        cache.apply_mapping_epoch_transitions(&[], &[(key.gpu_va, 1)]);
        let wildcard = RtKey::request(key.nvmap_id, key.width, key.height);
        let physical = RtKey { gpu_va: key.gpu_va + 0x10000, ..key };
        for want in [key, wildcard, physical] {
            assert!(cache.find_depth(want).is_none());
            assert!(cache.find_depth_fuzzy(want).is_none());
            assert!(cache.find_color_with_format(want).is_none());
            assert!(cache.color_exact_with_format(want).is_none());
            assert!(cache.get_existing(want).is_none());
            assert!(cache.get_existing_depth(want).is_none());
        }
        assert!(cache.find_color_key_at_va(key.nvmap_id, key.gpu_va).is_none());
        assert!(cache.present_depth_key(key).is_none());
        cache.mark_drawn(key);
        cache.mark_depth_written(key);
        assert!(cache.color_is_guest_stale(key));
        assert!(cache.depth_is_guest_stale(key));
        assert!(cache.resolve_drawn_color_exact(key).is_none());
        assert!(cache.find_sampleable_color_with_format(key).is_none());
        assert!(cache.find_sampleable_depth(key).is_none());
        assert!(cache.present_depth_key(key).is_none());
        assert!(cache.color_exact_with_format(key).is_none());
    }

    #[test]
    fn rt_retirement_obsolete_padded_depth_is_not_a_readback_source() {
        let mut cache = RtCache::new();
        let key = retirement_key(32, 32);
        insert_retirement_depth(&mut cache, key, 1);
        cache.depth_cache.get_mut(&key).unwrap().base_format = vk::Format::D24_UNORM_S8_UINT;
        cache.apply_mapping_epoch_transitions(&[], &[(key.gpu_va, 1)]);
        assert!(cache.find_d24_depth_covering(RtKey { width: 31, ..key }).is_none());
    }

    #[test]
    fn rt_retirement_preserves_promoted_backing_and_changes_inside_tail() {
        for changed_tail in [false, true] {
            let mut cache = RtCache::new();
            let key = retirement_key(32, 32).with_guest_size_bytes(0x8000);
            cache.insert_color_image(key, retirement_color(key, 1));
            cache.mark_drawn(key);
            insert_retirement_depth(&mut cache, key, 2);
            let changed = if changed_tail { vec![(key.gpu_va + 0x4000, 0x1000)] } else { vec![] };
            cache.apply_mapping_epoch_transitions(&[RtMappingEpochTransition {
                gpu_va: key.gpu_va,
                size: 0x8000,
                old_epoch: key.mapping_epoch,
                new_epoch: key.mapping_epoch + 1,
            }], &changed);
            let expected_epoch = key.mapping_epoch + u64::from(!changed_tail);
            assert_eq!(cache.cache.keys().next().unwrap().mapping_epoch, expected_epoch);
            assert_eq!(cache.depth_cache.keys().next().unwrap().mapping_epoch, expected_epoch);
            let mut retired = Vec::new();
            assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
            assert_eq!(cache.cache.len(), 1);
            assert_eq!(cache.depth_cache.len(), 1);
        }
    }

    #[test]
    fn rt_retirement_keeps_active_derived_views_and_reclaims_unused_copies() {
        let mut cache = RtCache::new();
        let source = retirement_key(1248, 720);
        let target = retirement_key(1245, 700);
        cache.insert_color_image(source, retirement_color(source, 1));
        cache.insert_color_image(target, retirement_color(target, 2));
        let stamp = cache.mark_drawn(source);
        cache.mark_color_region_synced(target, source, stamp);
        let mut retired = Vec::new();
        for _ in 0..128 {
            assert_eq!(cache.retire_redundant_views(&mut retired, &[target]), (0, 0));
        }
        for _ in 0..63 {
            assert_eq!(cache.retire_redundant_views(&mut retired, &[]), (0, 0));
        }
        assert_eq!(cache.retire_redundant_views(&mut retired, &[]).0, 1);
        assert!(cache.cache.contains_key(&source));
        assert!(!cache.has_pending_view_retirement());
    }

    #[test]
    fn rt_retirement_full_clear_preserves_authoritative_present_lookup() {
        let mut cache = RtCache::new();
        let old = retirement_key(1245, 700);
        let source = retirement_key(1248, 720);
        let unrelated = RtKey { nvmap_id: 99, gpu_va: 0x7200_0000, cpu_addr: 0xa000_0000, ..old };
        cache.insert_color_image(old, retirement_color(old, 1));
        cache.insert_color_image(source, retirement_color(source, 2));
        cache.insert_color_image(unrelated, retirement_color(unrelated, 3));
        cache.mark_drawn(old);
        cache.mark_drawn(unrelated);
        cache.mark_cleared(source, true);
        cache.mark_drawn(source);
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
        assert_eq!(cache.resolve_present_key(old, false), Some(old));
        assert!(cache.cache.contains_key(&source));
    }

    #[test]
    fn rt_retirement_derived_color_retires_snapshot_and_invalidates_sampling_memo() {
        let mut cache = RtCache::new();
        let old = retirement_key(1245, 700);
        let source = retirement_key(1248, 720);
        cache.insert_color_image(old, retirement_color(old, 1));
        cache.insert_color_image(source, retirement_color(source, 2));
        cache.snapshots.insert(old, retirement_color(old, 3));
        let stamp = cache.mark_drawn(source);
        cache.mark_color_region_synced(old, source, stamp);
        let version = cache.color_sampling_generation(old.nvmap_id);
        let mut retired = Vec::new();
        let (count, bytes) = retire_aged_views(&mut cache, &mut retired, &[]);
        assert_eq!(count, 2);
        assert_eq!(bytes, 2 * 1245 * 700 * 4);
        assert_eq!(retired.len(), 2);
        assert!(!cache.cache.contains_key(&old));
        assert!(!cache.snapshots.contains_key(&old));
        assert!(!cache.drawn_stamp.contains_key(&old));
        assert!(!cache.frame_draws.contains_key(&old));
        assert!(cache.cache.contains_key(&source));
        assert_ne!(version, cache.color_sampling_generation(old.nvmap_id));
        assert!(cache.find_drawn_color_region_at_excluding(old.width, old.height, old.gpu_va, old).is_some());
    }

    #[test]
    fn rt_retirement_partial_draw_cannot_supersede_a_newer_alias_write() {
        let mut cache = RtCache::new();
        let old = retirement_key(1245, 700);
        let source = retirement_key(1248, 720);
        for (key, handle) in [(old, 1), (source, 2)] {
            cache.insert_color_image(key, retirement_color(key, handle));
        }
        cache.mark_cleared(source, true);
        cache.mark_drawn(old);
        cache.mark_drawn(source);
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
        cache.mark_cleared(source, false);
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]), (0, 0));
        cache.mark_cleared(source, true);
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 0);
    }

    #[test]
    fn rt_retirement_preserves_cross_pitch_unknown_or_uncovered_views() {
        let source = retirement_key(1248, 720);
        let target = retirement_key(1245, 700);
        let mut unknown_layout = target;
        unknown_layout.layout_signature = 0;
        let mut cases = vec![
            retirement_key(1232, 700),
            retirement_key(1245, 721),
            unknown_layout,
            target.with_mapping_epoch(0),
            target.with_mapping_epoch(41),
        ];
        let mut different_cpu = target;
        different_cpu.cpu_addr += 64;
        cases.push(different_cpu);
        let mut layered = target;
        layered.depth = 2;
        cases.push(layered);
        let mut msaa = target;
        msaa.sample_width = 2;
        cases.push(msaa);
        for target in cases {
            let mut cache = RtCache::new();
            cache.insert_color_image(source, retirement_color(source, 1));
            cache.insert_color_image(target, retirement_color(target, 2));
            let stamp = cache.mark_drawn(source);
            cache.mark_color_region_synced(target, source, stamp);
            cache.mark_cleared(source, true);
            let mut retired = Vec::new();
            assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 0, "{target:?}");
            assert_eq!(cache.cache.len(), 2);
        }
    }

    #[test]
    fn rt_retirement_derived_copy_requires_live_source_and_no_destination_writes() {
        let source = retirement_key(1248, 720);
        let target = retirement_key(1245, 700);
        for invalidate in 0..4 {
            let mut cache = RtCache::new();
            cache.insert_color_image(source, retirement_color(source, 1));
            cache.insert_color_image(target, retirement_color(target, 2));
            let stamp = cache.mark_drawn(source);
            cache.mark_color_region_synced(target, source, stamp);
            match invalidate {
                1 => { cache.mark_drawn(target); }
                2 => { cache.cache.get_mut(&source).unwrap().image = vk::Image::from_raw(3); }
                3 => { cache.mark_guest_uploaded(source); }
                _ => { cache.mark_drawn(source); }
            }
            let mut retired = Vec::new();
            assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0,
                usize::from(invalidate == 0));
            assert!(cache.cache.contains_key(&source));
        }
    }

    #[test]
    fn rt_retirement_derived_chain_keeps_reconstructing_sources_until_next_boundary() {
        let mut cache = RtCache::new();
        let source = retirement_key(1248, 720);
        let middle = retirement_key(1247, 715);
        let target = retirement_key(1245, 700);
        for (key, handle) in [(source, 1), (middle, 2), (target, 3)] {
            cache.insert_color_image(key, retirement_color(key, handle));
        }
        let stamp = cache.mark_drawn(source);
        cache.mark_color_region_synced(middle, source, stamp);
        cache.mark_color_region_synced(target, middle, stamp);
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
        assert!(cache.cache.contains_key(&source));
        assert!(cache.cache.contains_key(&middle));
        assert!(!cache.cache.contains_key(&target));
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
        assert!(cache.cache.contains_key(&source));
        assert!(!cache.cache.contains_key(&middle));
    }

    #[test]
    fn rt_retirement_protection_and_guest_writes_invalidate_depth_clear_proof() {
        let mut cache = RtCache::new();
        let source = retirement_key(1248, 720);
        let target = retirement_key(1245, 700);
        for key in [source, target] {
            cache.insert_depth_image(key, test_d32_image(key));
        }
        cache.mark_depth_written(target);
        cache.mark_depth_cleared(source, true);
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[target]).0, 0);
        cache.mark_guest_written(source);
        cache.mark_depth_written(source);
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 0);
        cache.mark_depth_cleared(source, true);
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
    }

    #[test]
    fn rt_retirement_d32_clear_and_copy_preserve_latest_authoritative_source() {
        for derived in [false, true] {
            let mut cache = RtCache::new();
            let source = retirement_key(1248, 720);
            let target = retirement_key(1245, 700);
            cache.insert_depth_image(source, test_d32_image(source));
            cache.insert_depth_image(target, test_d32_image(target));
            cache.mark_depth_written(target);
            let generation = cache.mark_depth_cleared(source, true);
            if derived {
                cache.mark_depth_region_synced(target, source, generation);
            }
            cache.mark_depth_written(source);
            let mut retired = Vec::new();
            assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
            assert_eq!(cache.find_sampleable_d32_region(target).unwrap().key, source);
            assert!(cache.depth_generation(target).is_none());
        }
    }

    #[test]
    fn rt_retirement_mapping_transition_preserves_lineage_metadata() {
        let mut cache = RtCache::new();
        let source = retirement_key(1248, 720);
        let target = retirement_key(1245, 700);
        cache.insert_color_image(source, retirement_color(source, 1));
        cache.insert_color_image(target, retirement_color(target, 2));
        let stamp = cache.mark_drawn(source);
        cache.mark_color_region_synced(target, source, stamp);
        let moved_source = source.with_mapping_epoch(41);
        let moved_target = target.with_mapping_epoch(41);
        assert!(cache.rekey_color_state(source, moved_source));
        assert!(cache.rekey_color_state(target, moved_target));
        assert_eq!(cache.color_region_sources[&moved_target].0.mapping_epoch, 41);
        let mut retired = Vec::new();
        assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
        assert!(cache.cache.contains_key(&source));
    }

    #[test]
    fn rt_retirement_depth_later_alias_write_and_unsupported_formats_survive() {
        for unsupported in [false, true] {
            let mut cache = RtCache::new();
            let source = retirement_key(1248, 720);
            let target = retirement_key(1245, 700);
            for key in [source, target] {
                let mut image = test_d32_image(key);
                if unsupported {
                    image.base_format = vk::Format::D16_UNORM;
                    image.format = vk::Format::D16_UNORM;
                }
                cache.insert_depth_image(key, image);
            }
            cache.mark_depth_cleared(source, true);
            if !unsupported { cache.mark_depth_written(target); }
            cache.mark_depth_written(source);
            let mut retired = Vec::new();
            assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 0);
            if !unsupported {
                cache.mark_depth_cleared(source, true);
                assert_eq!(retire_aged_views(&mut cache, &mut retired, &[]).0, 1);
            }
        }
    }

    #[test]
    fn d32_region_uses_newest_padded_write_and_inherits_generation() {
        let source = RtKey::with_cpu(12, 1616, 908, 0x520000000, 0x800000)
            .with_mapping_epoch(3)
            .with_block_linear_layout(0, 4, 0, 0);
        let sampled = RtKey { width: 1614, ..source };
        let mut cache = RtCache::new();
        cache.insert_depth_image(source, test_d32_image(source));
        cache.insert_depth_image(sampled, test_d32_image(sampled));
        cache.mark_depth_written(sampled);
        let producer_write = cache.mark_depth_written(source);
        assert_eq!(cache.find_sampleable_d32_region(sampled).unwrap().key, source);

        cache.mark_depth_synced_from(sampled, producer_write);
        assert_eq!(cache.depth_generation(sampled), Some(producer_write));
        assert_eq!(cache.find_sampleable_d32_region(sampled).unwrap().key, sampled);

        let cleared = cache.mark_depth_written(source);
        assert!(cleared > producer_write);
        let region = cache.find_sampleable_d32_region(sampled).unwrap();
        assert_eq!(region.key, source);
        assert_eq!(region.generation, cleared);
        cache.mark_depth_synced_from(sampled, cleared);
        cache.mark_depth_written(sampled);
        assert_eq!(cache.find_sampleable_d32_region(sampled).unwrap().key, sampled);
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn d32_region_requires_known_matching_physical_rows() {
        let source = RtKey::with_cpu(12, 1632, 916, 0x520000000, 0x800000)
            .with_mapping_epoch(3)
            .with_block_linear_layout(0, 4, 0, 0);
        let sampled = RtKey { width: 1630, ..source };
        let mut cache = RtCache::new();
        cache.insert_depth_image(source, test_d32_image(source));
        cache.mark_depth_written(source);
        assert_eq!(cache.find_sampleable_d32_region(sampled).unwrap().key, source);
        for rejected in [
            RtKey { width: 1614, ..sampled },
            RtKey { layout_signature: 0, ..sampled },
            sampled.with_block_linear_layout(0, 3, 0, 0),
            sampled.with_mapping_epoch(4),
            sampled.with_base_layer(1),
            sampled.with_sample_grid(2, 1),
            sampled.with_array_layers(2, 0x800000),
            RtKey { gpu_va: sampled.gpu_va + 64, ..sampled },
            RtKey { cpu_addr: sampled.cpu_addr + 64, ..sampled },
        ] {
            assert!(cache.find_sampleable_d32_region(rejected).is_none(), "{rejected:?}");
        }
        cache.guest_stale_depth.insert(source);
        assert!(cache.find_sampleable_d32_region(sampled).is_none());
        cache.guest_stale_depth.remove(&source);
        cache.depth_cache.get_mut(&source).unwrap().layout = vk::ImageLayout::UNDEFINED;
        assert!(cache.find_sampleable_d32_region(sampled).is_none());
        cache.depth_cache.get_mut(&source).unwrap().layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        cache.depth_cache.get_mut(&source).unwrap().base_format = vk::Format::D24_UNORM_S8_UINT;
        assert!(cache.find_sampleable_d32_region(sampled).is_none());
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn d32_region_rejects_full_size_depth_with_a_different_pitch() {
        let source = RtKey::new(12, 1920, 1080, 0x520000000)
            .with_block_linear_layout(0, 4, 0, 0);
        let sampled = RtKey { width: 1614, height: 908, ..source };
        let mut cache = RtCache::new();
        cache.insert_depth_image(source, test_d32_image(source));
        cache.mark_depth_written(source);
        assert!(cache.find_sampleable_d32_region(sampled).is_none());
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    fn test_color_image(format: vk::Format, base_format: vk::Format) -> GpuImage {
        GpuImage {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            views: HashMap::default(),
            sample_views: HashMap::default(),
            memory: vk::DeviceMemory::null(),
            format,
            base_format,
            aspects: vk::ImageAspectFlags::COLOR,
            extent: vk::Extent2D {
                width: 640,
                height: 480,
            },
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    fn assert_indexed_color_exact_matches_legacy(cache: &RtCache, want: RtKey) {
        let indexed = cache.find_sampleable_color_for_exact_alias_indexed(want);
        let legacy = cache.find_sampleable_color_for_exact_alias_legacy(want);
        assert!(rt_sampleable_color_alias_equal(&indexed, &legacy));
    }

    fn assert_indexed_depth_exact_matches_legacy(cache: &RtCache, want: RtKey) {
        let indexed = cache.find_sampleable_depth_for_exact_alias_indexed(want);
        let legacy = cache.find_sampleable_depth_for_exact_alias_legacy(want);
        assert!(rt_sampleable_depth_alias_equal(&indexed, &legacy));
    }

    fn assert_indexed_color_physical_matches_legacy(
        cache: &RtCache,
        want: RtKey,
        view_format: vk::Format,
    ) {
        let indexed =
            cache.find_sampleable_color_for_exact_or_physical_alias_indexed(want, view_format);
        let legacy =
            cache.find_sampleable_color_for_exact_or_physical_alias_legacy(want, view_format);
        assert!(rt_sampleable_color_alias_equal(&indexed, &legacy));
    }

    fn region_signature(
        region: Option<RtColorRegion>,
    ) -> Option<(RtKey, vk::Format, u64, u32, u32)> {
        region.map(|region| {
            (
                region.key,
                region.format,
                region.stamp,
                region.src_x,
                region.src_y,
            )
        })
    }

    #[test]
    fn synthetic_copy_keys_are_only_visible_through_exact_lookup() {
        const VA: u64 = 0x5123_4000;
        let guest = RtKey::new(7, 640, 480, VA);
        let synthetic = RtKey::new(u32::MAX, 640, 480, VA);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            guest,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );
        cache.insert_color_image(
            synthetic,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );
        cache.mark_drawn(guest);
        cache.mark_drawn(synthetic);

        assert!(is_synthetic_copy_key(synthetic));
        assert!(!is_synthetic_copy_key(guest));
        assert_eq!(
            cache
                .color_exact_with_format(synthetic)
                .map(|result| result.0),
            Some(synthetic)
        );
        assert_eq!(
            cache
                .find_color_with_format(synthetic)
                .map(|result| result.0),
            Some(synthetic)
        );
        assert!(cache.drawn_stamp(synthetic).is_some());

        assert_eq!(
            cache
                .find_drawn_color_at(640, 480, VA)
                .map(|result| result.0),
            Some(guest)
        );
        assert_eq!(
            cache
                .find_content_bearing_color_at(640, 480, VA)
                .map(|result| result.0),
            Some(guest)
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(640, 480, VA)
                .map(|result| result.key),
            Some(guest)
        );
        assert_eq!(cache.resolve_present_key(guest, false), Some(guest));
        assert_eq!(cache.present_candidates(guest), vec![(guest, 1)]);
        assert_eq!(
            cache
                .find_color_screen(RtKey::request(99, 640, 480))
                .map(|result| result.0),
            Some(guest)
        );
        assert_eq!(cache.find_color_key_at_va(u32::MAX, VA), None);
        assert!(!cache.has_drawn_color_at_va(u32::MAX, VA));
        assert!(cache.color_keys_for_nvmap(u32::MAX).is_empty());
        assert!(cache.drawn_color_aliases(synthetic).is_empty());
        assert!(cache
            .find_color_with_format(RtKey::new(u32::MAX, 639, 480, VA))
            .is_none());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn newest_exact_present_key_uses_freshest_supplied_va_only() {
        const OLD_VA: u64 = 0x5400_7000_0;
        const NEW_VA: u64 = 0x5408_e000_0;
        const OTHER_VA: u64 = 0x5324_7000_0;
        let old = RtKey::new(7, 1280, 720, OLD_VA);
        let new = RtKey::new(7, 1280, 720, NEW_VA);
        let not_supplied = RtKey::new(7, 1280, 720, OTHER_VA);
        let wrong_nvmap = RtKey::new(8, 1280, 720, NEW_VA);
        let wrong_extent = RtKey::new(7, 640, 360, NEW_VA);
        let mut cache = RtCache::new();
        for key in [old, new, not_supplied, wrong_nvmap, wrong_extent] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
            cache.mark_drawn(key);
        }
        let old_stamp = cache.mark_drawn(old);

        assert_eq!(
            cache.newest_exact_present_key_at_vas(7, 1280, 720, &[0, NEW_VA, OLD_VA]),
            Some((old, old_stamp))
        );
        assert_eq!(
            cache.newest_exact_present_key_at_vas(7, 1280, 720, &[OTHER_VA]),
            Some((not_supplied, 3))
        );
        assert_eq!(
            cache.newest_exact_present_key_at_vas(7, 1280, 720, &[0xdead_beef]),
            None
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn newest_exact_present_key_rejects_excluded_undefined_and_invalid_requests() {
        const VA_A: u64 = 0x5400_7000_0;
        const VA_B: u64 = 0x5408_e000_0;
        let excluded = RtKey::new(7, 1280, 720, VA_A);
        let undefined = RtKey::new(7, 1280, 720, VA_B);
        let mut cache = RtCache::new();
        for key in [excluded, undefined] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
            cache.mark_drawn(key);
        }
        cache.mark_synced_sample(excluded);
        cache.get_existing(undefined).unwrap().layout = vk::ImageLayout::UNDEFINED;

        assert_eq!(
            cache.newest_exact_present_key_at_vas(7, 1280, 720, &[VA_A, VA_B]),
            None
        );
        assert_eq!(
            cache.newest_exact_present_key_at_vas(0, 1280, 720, &[VA_A]),
            None
        );
        assert_eq!(
            cache.newest_exact_present_key_at_vas(u32::MAX, 1280, 720, &[VA_A]),
            None
        );
        assert_eq!(
            cache.newest_exact_present_key_at_vas(7, 0, 720, &[VA_A]),
            None
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn region_lookup_can_exclude_stale_exact_destination() {
        const VA: u64 = 0x5a0e_a1000;
        let exact = RtKey::new(584, 2, 1, VA);
        let padded = RtKey::new(584, 16, 1, VA);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            exact,
            test_color_image(
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::Format::B10G11R11_UFLOAT_PACK32,
            ),
        );
        cache.insert_color_image(
            padded,
            test_color_image(
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::Format::B10G11R11_UFLOAT_PACK32,
            ),
        );
        let exact_stamp = cache.mark_synced_sample(exact);
        let padded_stamp = cache.mark_drawn(padded);

        let preferred = cache.find_drawn_color_region_at(2, 1, VA).unwrap();
        assert_eq!(preferred.key, exact);
        assert_eq!(preferred.stamp, exact_stamp);

        let source = cache
            .find_drawn_color_region_at_excluding(2, 1, VA, exact)
            .unwrap();
        assert_eq!(source.key, padded);
        assert_eq!(source.stamp, padded_stamp);
        assert_eq!((source.src_x, source.src_y), (0, 0));

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn region_sync_prefers_latest_write_over_smaller_cached_view() {
        const VA: u64 = 0x6200_3000;
        let target = RtKey::new(41, 1200, 675, VA);
        let old = RtKey::new(41, 1216, 675, VA);
        let current = RtKey::new(41, 1280, 720, VA);
        let mut cache = RtCache::new();
        for key in [target, old, current] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
        }
        let old_stamp = cache.mark_drawn(old);
        cache.mark_synced_sample_from(target, old_stamp);
        let current_stamp = cache.mark_drawn(current);
        let source = cache
            .find_drawn_color_region_at_excluding(1200, 675, VA, target)
            .unwrap();
        assert_eq!(source.key, current);
        assert_eq!(source.stamp, current_stamp);
        cache.mark_drawn(old);
        assert_eq!(
            cache.find_drawn_color_region_at_excluding(1200, 675, VA, target).unwrap().key,
            old
        );
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn alias_sync_tracks_full_clears_without_presenting_them() {
        const CPU: u64 = 0x1710_0000;
        let source = RtKey::with_cpu(43, 32, 32, 0x6200_8000, CPU);
        let target = RtKey::with_cpu(43, 32, 32, 0x6201_8000, CPU);
        let mut cache = RtCache::new();
        for key in [source, target] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
        }
        let old_stamp = cache.mark_drawn(source);
        cache.mark_synced_sample_from(target, old_stamp);
        cache.mark_cleared(source, true);
        let clear_stamp = cache.writeback_stamp(source).unwrap();
        let aliases = cache.drawn_color_aliases(target);
        assert_eq!(aliases.len(), 1);
        assert_eq!((aliases[0].0, aliases[0].4), (source, clear_stamp));
        assert!(clear_stamp > old_stamp);
        assert_eq!(aliases, cache.drawn_color_aliases_full_scan(target));
        assert_eq!(cache.drawn_stamp(source), None);
        assert_eq!(cache.present_candidates(source), vec![(source, 0)]);
        assert_eq!(
            cache.newest_exact_present_key_at_vas(43, 32, 32, &[source.gpu_va]),
            None
        );
        assert!(cache.find_drawn_color_at(32, 32, source.gpu_va).is_none());

        cache.cache.get_mut(&source).unwrap().layout = vk::ImageLayout::UNDEFINED;
        assert!(cache.drawn_color_aliases(target).is_empty());
        assert!(cache.drawn_color_aliases_full_scan(target).is_empty());
        cache.cache.get_mut(&source).unwrap().layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        cache.mark_guest_written(source);
        assert!(cache.drawn_color_aliases(target).is_empty());
        assert!(cache.drawn_color_aliases_full_scan(target).is_empty());
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn region_sync_prefers_a_new_full_clear_over_old_draws() {
        const VA: u64 = 0x6200_3000;
        let target = RtKey::new(41, 1200, 675, VA);
        let old = RtKey::new(41, 1216, 675, VA);
        let current = RtKey::new(41, 1280, 720, VA);
        let mut cache = RtCache::new();
        for key in [target, old, current] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
        }
        let old_stamp = cache.mark_drawn(old);
        cache.mark_synced_sample_from(target, old_stamp);
        cache.mark_drawn(current);
        cache.mark_cleared(current, true);
        let clear_stamp = cache.writeback_stamp(current).unwrap();
        let source = cache
            .find_drawn_color_region_at_excluding(1200, 675, VA, target)
            .unwrap();
        assert_eq!((source.key, source.stamp), (current, clear_stamp));
        assert!(clear_stamp > old_stamp);
        assert_eq!(
            region_signature(Some(source)),
            region_signature(cache.find_drawn_color_region_at_full_scan(1200, 675, VA, Some(target)))
        );
        assert_eq!(cache.find_drawn_color_region_at(1200, 675, VA).unwrap().key, target);
        assert!(cache.find_drawn_color_region_at(1280, 720, VA).is_none());

        cache.cache.get_mut(&current).unwrap().layout = vk::ImageLayout::UNDEFINED;
        assert_eq!(
            cache.find_drawn_color_region_at_excluding(1200, 675, VA, target).unwrap().key,
            old
        );
        cache.cache.get_mut(&current).unwrap().layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        cache.mark_guest_written(current);
        assert_eq!(
            cache.find_drawn_color_region_at_excluding(1200, 675, VA, target).unwrap().key,
            old
        );
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn color_sync_layout_matches_physical_row_stride() {
        let format = vk::Format::R8G8B8A8_UNORM;
        let key = RtKey::new(41, 1280, 720, 0x6200_3000);
        let tiled = key.with_block_linear_layout(0, 4, 0, 0);
        let half = RtKey { width: 640, height: 360, ..tiled };
        assert!(!super::color_sync_layout_compatible(tiled, half, format));
        assert_eq!(super::color_sync_row_stride(tiled, format), Some(5120));
        let padded = RtKey { width: 1248, height: 700, ..tiled };
        let logical = RtKey { width: 1245, ..padded };
        assert!(super::color_sync_layout_compatible(padded, logical, format));
        assert_eq!(super::color_sync_row_stride(logical, format), Some(4992));

        let block_wide = tiled.with_block_linear_layout(4, 4, 0, 0);
        let logical = RtKey { width: 1200, height: 675, ..block_wide };
        assert!(super::color_sync_layout_compatible(block_wide, logical, format));
        assert_eq!(super::color_sync_row_stride(logical, format), Some(5120));
        assert!(!super::color_sync_layout_compatible(
            tiled, RtKey { width: 1200, height: 675, ..tiled }, format
        ));

        let spaced = padded.with_block_linear_layout(0, 4, 0, 3);
        let logical = RtKey { width: 1200, height: 675, ..spaced };
        assert!(super::color_sync_layout_compatible(spaced, logical, format));
        assert!(!super::color_sync_layout_compatible(
            spaced, RtKey { height: 64, ..logical }, format
        ));
        let volume = padded.with_volume_depth(2).with_block_linear_layout(0, 4, 1, 3);
        assert!(!super::color_sync_layout_compatible(
            volume, RtKey { width: 1200, depth: 1, ..volume }, format
        ));

        let pitched = key.with_pitch_linear_layout(5120);
        assert!(super::color_sync_layout_compatible(
            pitched, RtKey { width: 640, height: 360, ..pitched }, format
        ));
        assert!(!super::color_sync_layout_compatible(
            pitched, half.with_pitch_linear_layout(2560), format
        ));
        let invalid_pitch = key.with_pitch_linear_layout(2560);
        assert!(!super::color_sync_layout_compatible(invalid_pitch, invalid_pitch, format));
        let invalid_layout = RtKey { layout_signature: 3, ..key };
        assert!(!super::color_sync_layout_compatible(invalid_layout, invalid_layout, format));
        assert!(super::color_sync_layout_compatible(
            key, RtKey { width: 640, height: 360, ..key }, format
        ));
        assert!(super::color_sync_layout_compatible(tiled, key, format));
    }

    #[test]
    fn region_sync_rejects_resized_backing_but_accepts_row_padding() {
        const VA: u64 = 0x6200_3000;
        let target = RtKey::new(41, 1245, 700, VA).with_block_linear_layout(0, 4, 0, 0);
        let padded = RtKey { width: 1248, ..target };
        let resized = RtKey { width: 1280, height: 720, ..target };
        let mut cache = RtCache::new();
        for key in [padded, resized] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            );
            cache.mark_drawn(key);
        }
        let source = cache.find_drawn_color_region_at_excluding(1245, 700, VA, target).unwrap();
        assert_eq!(source.key, padded);
        assert_eq!(
            cache.drawn_color_aliases(target).iter().map(|entry| entry.0).collect::<Vec<_>>(),
            vec![padded]
        );
        assert_eq!(cache.drawn_color_aliases(target), cache.drawn_color_aliases_full_scan(target));
        let half = RtKey { width: 640, height: 360, ..target };
        assert!(cache.find_drawn_color_region_at_excluding(640, 360, VA, half).is_none());
        assert!(cache.find_drawn_color_region_at(640, 360, VA).is_some());
        let unknown = RtKey { layout_signature: 0, ..half };
        assert_eq!(
            cache.find_drawn_color_region_at_excluding(640, 360, VA, unknown).unwrap().key,
            resized
        );
        let offset = RtKey { gpu_va: VA + 64, ..half };
        assert!(cache.find_drawn_color_region_at_excluding(640, 360, VA + 64, offset).is_none());
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn indexed_exact_base_lookup_matches_full_scan_ranking_and_aliases() {
        const VA: u64 = 0x6200_4000;
        let small = RtKey::new(41, 16, 16, VA);
        let padded = RtKey::new(41, 32, 32, VA);
        let other_nvmap = RtKey::new(42, 16, 16, VA);
        let mut cache = RtCache::new();
        for key in [small, padded, other_nvmap] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
        }
        cache.drawn_stamp.insert(small, 10);
        cache.drawn_stamp.insert(padded, 10);
        cache.drawn_stamp.insert(other_nvmap, 9);

        assert_eq!(
            cache.find_drawn_color_at(8, 8, VA),
            cache.find_drawn_color_at_full_scan(8, 8, VA)
        );
        assert_eq!(
            cache.find_drawn_color_at(8, 8, VA).map(|result| result.0),
            Some(small)
        );

        cache.drawn_stamp.insert(other_nvmap, 11);
        assert_eq!(
            cache.find_drawn_color_at(8, 8, VA),
            cache.find_drawn_color_at_full_scan(8, 8, VA)
        );
        assert_eq!(
            cache.find_drawn_color_at(8, 8, VA).map(|result| result.0),
            Some(other_nvmap)
        );
        assert_eq!(
            cache.drawn_color_aliases(small),
            cache.drawn_color_aliases_full_scan(small)
        );
        assert_eq!(
            cache
                .drawn_color_aliases(small)
                .into_iter()
                .map(|result| result.0)
                .collect::<Vec<_>>(),
            vec![padded]
        );

        cache.mark_guest_written(padded);
        assert_eq!(
            cache.drawn_color_aliases(small),
            cache.drawn_color_aliases_full_scan(small)
        );
        assert!(cache.drawn_color_aliases(small).is_empty());
        cache.drawn_counter = 20;
        cache.mark_drawn(padded);
        assert_eq!(
            cache.drawn_color_aliases(small),
            cache.drawn_color_aliases_full_scan(small)
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn indexed_alias_sync_includes_exact_cpu_backing_at_alternate_va() {
        const CPU: u64 = 0x1710_0000;
        let make = |gpu_va| {
            RtKey::with_cpu(43, 32, 32, gpu_va, CPU)
                .with_mapping_epoch(7)
                .with_block_linear_layout(0, 4, 0, 0)
        };
        let target = make(0x6200_8000);
        let alternate_va = make(0x6201_8000);
        let wrong_extent = RtKey::with_cpu(43, 16, 32, 0x6202_8000, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0);
        let wrong_layout = make(0x6203_8000).with_block_linear_layout(0, 3, 0, 0);
        let wrong_cpu = RtKey::with_cpu(43, 32, 32, 0x6204_8000, CPU + 0x10_000)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0);
        let mut cache = RtCache::new();
        for (stamp, key) in [target, alternate_va, wrong_extent, wrong_layout, wrong_cpu]
            .into_iter()
            .enumerate()
        {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
            cache.drawn_stamp.insert(key, stamp as u64 + 1);
        }

        assert_eq!(
            cache.drawn_color_aliases(target),
            cache.drawn_color_aliases_full_scan(target)
        );
        assert_eq!(
            cache
                .drawn_color_aliases(target)
                .into_iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            vec![alternate_va]
        );

        let (removed, _) = cache.remove_color_image(alternate_va).unwrap();
        assert_eq!(removed, alternate_va);
        assert!(cache.drawn_color_aliases(target).is_empty());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn synchronized_sample_inherits_source_stamp_without_new_write() {
        let source = RtKey::new(43, 32, 32, 0x6210_8000);
        let destination = RtKey::new(43, 32, 32, 0x6211_8000);
        let later = RtKey::new(43, 32, 32, 0x6212_8000);
        let mut cache = RtCache::new();
        let source_stamp = cache.mark_drawn(source);
        let counter_before_sync = cache.drawn_counter;

        assert_eq!(
            cache.mark_synced_sample_from(destination, source_stamp),
            source_stamp
        );
        assert_eq!(cache.drawn_stamp(destination), Some(source_stamp));
        assert_eq!(cache.drawn_counter, counter_before_sync);
        assert!(cache.present_excluded.contains(&destination));
        assert_eq!(cache.mark_drawn(later), counter_before_sync + 1);
    }

    #[test]
    fn color_sync_candidates_require_matching_live_view_identity() {
        const VA: u64 = 0x6200_8000;
        const CPU: u64 = 0x1700_0000;
        let make = |width| {
            RtKey::with_cpu(43, width, 32, VA, CPU)
                .with_mapping_epoch(7)
                .with_block_linear_layout(0, 4, 0, 0)
                .with_guest_size_bytes(u64::from(width) * 0x100)
        };
        let target = make(8);
        let good = make(16).with_guest_size_bytes(0x80_000);
        let wrong_layer = make(12).with_base_layer(1);
        let wrong_epoch = make(20).with_mapping_epoch(8);
        let wrong_layout = make(24).with_block_linear_layout(0, 3, 0, 0);
        let multisampled = make(28).with_sample_grid(2, 1);
        let other_cpu = RtKey::with_cpu(43, 32, 32, VA, CPU + 0x10_000)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0);
        let other_nvmap = RtKey::with_cpu(44, 36, 32, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0);
        let keys = [
            target,
            good,
            wrong_layer,
            wrong_epoch,
            wrong_layout,
            multisampled,
            other_cpu,
            other_nvmap,
        ];
        let mut cache = RtCache::new();
        for (stamp, key) in keys.into_iter().enumerate() {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
            cache.drawn_stamp.insert(key, stamp as u64 + 1);
        }

        assert_eq!(
            cache
                .drawn_color_aliases(target)
                .into_iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            vec![good]
        );
        assert_eq!(
            cache.drawn_color_aliases(target),
            cache.drawn_color_aliases_full_scan(target)
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at_excluding(8, 8, VA, target)
                .map(|region| region.key),
            Some(good)
        );
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at_excluding(8, 8, VA, target)),
            region_signature(cache.find_drawn_color_region_at_full_scan(8, 8, VA, Some(target),))
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_texture_alias_rejects_newer_wrong_backing_and_extent() {
        const VA: u64 = 0x6201_4000;
        const CPU: u64 = 0x1800_0000;
        let requested = RtKey::with_cpu(41, 16, 16, VA, CPU);
        let larger = RtKey::with_cpu(41, 32, 32, VA, CPU);
        let other_nvmap = RtKey::with_cpu(42, 16, 16, VA, CPU);
        let other_cpu = RtKey::with_cpu(41, 16, 16, VA, CPU + 0x10_000);
        let mut cache = RtCache::new();
        for key in [requested, larger, other_nvmap] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
        }
        cache.drawn_stamp.insert(requested, 1);
        cache.drawn_stamp.insert(larger, 40);
        cache.drawn_stamp.insert(other_nvmap, 50);
        assert_eq!(
            cache
                .find_drawn_color_for_exact_alias(requested)
                .map(|result| result.0),
            Some(requested)
        );
        assert!(cache.find_drawn_color_for_exact_alias(other_cpu).is_none());

        cache.drawn_stamp.remove(&requested);
        assert!(cache.find_drawn_color_for_exact_alias(requested).is_none());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_texture_alias_requires_matching_volume_shape() {
        const VA: u64 = 0x6202_4000;
        const CPU: u64 = 0x1900_0000;
        let volume = RtKey::with_cpu(51, 16, 16, VA, CPU).with_volume_depth(4);
        let layer = RtKey::with_cpu(51, 16, 16, VA, CPU);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            volume,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.drawn_stamp.insert(volume, 1);

        assert_eq!(
            cache
                .find_drawn_color_for_exact_alias(volume)
                .map(|result| result.0),
            Some(volume)
        );
        assert!(cache.find_drawn_color_for_exact_alias(layer).is_none());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_texture_alias_requires_matching_sample_grid() {
        const VA: u64 = 0x6203_4000;
        const CPU: u64 = 0x1a00_0000;
        let multisampled = RtKey::with_cpu(61, 32, 32, VA, CPU).with_sample_grid(2, 2);
        let single_sampled = RtKey::with_cpu(61, 32, 32, VA, CPU);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            multisampled,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.drawn_stamp.insert(multisampled, 1);

        assert_eq!(
            cache
                .find_drawn_color_for_exact_alias(multisampled)
                .map(|result| result.0),
            Some(multisampled)
        );
        assert!(cache
            .find_drawn_color_for_exact_alias(single_sampled)
            .is_none());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_texture_alias_requires_mapping_epoch_and_layout_identity() {
        const VA: u64 = 0x6204_4000;
        const CPU: u64 = 0x1b00_0000;
        let stored = RtKey::with_cpu(62, 32, 32, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x20_000);
        let remapped = stored.with_mapping_epoch(8);
        let relaid = stored.with_block_linear_layout(0, 3, 0, 0);
        let relayered = stored.with_base_layer(2);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.drawn_stamp.insert(stored, 1);

        assert!(cache.find_drawn_color_for_exact_alias(remapped).is_none());
        assert!(cache.find_drawn_color_for_exact_alias(relaid).is_none());
        assert!(cache.find_drawn_color_for_exact_alias(relayered).is_none());
        assert!(!stored.same_live_identity(remapped));
        assert!(!stored.same_live_identity(relaid));
        assert!(!stored.same_live_identity(relayered));
        assert!(cache.color_requires_recreate(remapped, vk::Format::R8_UNORM));
        assert!(cache.color_requires_recreate(relayered, vk::Format::R8_UNORM));
        let (removed, _) = cache.remove_color_image(remapped).unwrap();
        assert_eq!(removed.mapping_epoch, stored.mapping_epoch);
        cache.insert_color_image(
            remapped,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        assert_eq!(cache.cache.len(), 1);
        assert_eq!(
            cache
                .cache
                .get_key_value(&remapped)
                .unwrap()
                .0
                .mapping_epoch,
            remapped.mapping_epoch
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn multi_epoch_identity_transition_rekeys_all_color_state_without_recreating() {
        const VA: u64 = 0x6204_5000;
        const CPU: u64 = 0x1b04_0000;
        let stored = RtKey::with_cpu(63, 32, 32, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x4000);
        let remapped = stored.with_mapping_epoch(10);
        let mut cache = RtCache::new();
        let mut image = test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM);
        image.image = vk::Image::from_raw(0x1234);
        cache.insert_color_image(stored, image);
        let mut snapshot = test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM);
        snapshot.image = vk::Image::from_raw(0x5678);
        cache.snapshots.insert(stored, snapshot);
        let stamp = cache.mark_drawn(stored);
        cache.present_excluded.insert(stored);
        cache.record_present_flip(stored, true);

        cache.apply_mapping_epoch_transitions(
            &[
                RtMappingEpochTransition {
                    gpu_va: VA,
                    size: 0x1000,
                    old_epoch: 7,
                    new_epoch: 10,
                },
                RtMappingEpochTransition {
                    gpu_va: VA + 0x1000,
                    size: 0x7000,
                    old_epoch: 8,
                    new_epoch: 10,
                },
            ],
            &[],
        );

        let (canonical, image) = cache.cache.get_key_value(&remapped).unwrap();
        assert_eq!(canonical.mapping_epoch, 10);
        assert_eq!(image.image.as_raw(), 0x1234);
        assert_eq!(cache.drawn_stamp(remapped), Some(stamp));
        assert_eq!(cache.drawn_stamp.keys().next().unwrap().mapping_epoch, 10);
        assert_eq!(
            cache.present_excluded.iter().next().unwrap().mapping_epoch,
            10
        );
        assert_eq!(
            cache.present_flip_y.keys().next().unwrap().mapping_epoch,
            10
        );
        assert_eq!(cache.frame_draws.keys().next().unwrap().mapping_epoch, 10);
        assert_eq!(
            cache.frame_real_draws.keys().next().unwrap().mapping_epoch,
            10
        );
        assert_eq!(cache.snapshots.keys().next().unwrap().mapping_epoch, 10);
        assert_eq!(cache.snapshots[&remapped].image.as_raw(), 0x5678);
        assert_eq!(
            cache.color_guest_ranges[cache.color_guest_range_index[&remapped]]
                .key
                .mapping_epoch,
            10
        );
        assert_eq!(cache.color_gpu_base_index[&VA][0].mapping_epoch, 10);
        assert!(!cache.color_requires_recreate(remapped, vk::Format::R8_UNORM));
        assert_eq!(
            cache
                .find_drawn_color_for_exact_alias(remapped)
                .map(|entry| entry.0.mapping_epoch),
            Some(10)
        );
        assert!(cache.find_drawn_color_for_exact_alias(stored).is_none());

        cache.cache.clear();
        cache.snapshots.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn mapping_epoch_transition_rekeys_depth_tracking_and_shadow_source() {
        const VA: u64 = 0x6204_7000;
        const CPU: u64 = 0x1b0c_0000;
        let stored = RtKey::with_cpu(65, 32, 32, VA, CPU)
            .with_mapping_epoch(11)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x4000);
        let remapped = stored.with_mapping_epoch(12);
        let shadow = RtKey::with_cpu(66, 32, 32, VA + 0x10_000, CPU + 0x10_000)
            .with_mapping_epoch(20)
            .with_guest_size_bytes(0x4000);
        let mut cache = RtCache::new();
        let mut image = test_depth_image();
        image.image = vk::Image::from_raw(0x9abc);
        let base_format = image.base_format;
        cache.insert_depth_image(stored, image);
        super::insert_guest_range(
            &mut cache.depth_guest_ranges,
            &mut cache.depth_guest_range_index,
            stored,
            base_format,
        );
        cache.insert_depth_image(shadow, test_depth_image());
        let generation = cache.mark_depth_written(stored);
        assert!(cache.mark_depth_shadow_synced(stored, shadow));
        cache.guest_stale_depth.insert(stored);

        cache.apply_mapping_epoch_transitions(
            &[RtMappingEpochTransition {
                gpu_va: VA,
                size: 0x8000,
                old_epoch: 11,
                new_epoch: 12,
            }],
            &[],
        );

        let (canonical, image) = cache.depth_cache.get_key_value(&remapped).unwrap();
        assert_eq!(canonical.mapping_epoch, 12);
        assert_eq!(image.image.as_raw(), 0x9abc);
        assert_eq!(cache.depth_generation(remapped), Some(generation));
        assert!(cache.depth_is_guest_stale(remapped));
        assert!(cache.depth_shadow_is_current(remapped, shadow));
        assert_eq!(
            cache.depth_generations.keys().next().unwrap().mapping_epoch,
            12
        );
        assert_eq!(
            cache
                .depth_shadow_generations
                .values()
                .next()
                .unwrap()
                .0
                .mapping_epoch,
            12
        );
        assert_eq!(
            cache.depth_guest_ranges[cache.depth_guest_range_index[&remapped]]
                .key
                .mapping_epoch,
            12
        );
        assert!(!cache.depth_requires_recreate(
            remapped,
            vk::Format::D24_UNORM_S8_UINT,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        ));
        cache.guest_stale_depth.remove(&remapped);
        assert_eq!(
            cache
                .find_sampleable_depth_for_exact_alias(remapped)
                .map(|entry| entry.0.mapping_epoch),
            Some(12)
        );
        assert!(cache
            .find_sampleable_depth_for_exact_alias(stored)
            .is_none());

        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn mapping_epoch_transition_canonicalizes_noncanonical_depth_shadow_tracking() {
        const SOURCE_VA: u64 = 0x6205_0000;
        const SHADOW_VA: u64 = 0x6206_0000;
        let source = RtKey::with_cpu(67, 32, 32, SOURCE_VA, 0x1b20_0000)
            .with_mapping_epoch(31)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x4000);
        let shadow = RtKey::with_cpu(68, 32, 32, SHADOW_VA, 0x1b30_0000)
            .with_mapping_epoch(41)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x4000);
        let source_alias = source.with_mapping_epoch(301).with_guest_size_bytes(0x2000);
        let shadow_alias = shadow.with_mapping_epoch(401).with_guest_size_bytes(0x2000);
        let remapped_source = source.with_mapping_epoch(32);
        let remapped_shadow = shadow.with_mapping_epoch(42);
        let mut cache = RtCache::new();
        for key in [source, shadow] {
            let image = test_depth_image();
            let base_format = image.base_format;
            cache.insert_depth_image(key, image);
            super::insert_guest_range(
                &mut cache.depth_guest_ranges,
                &mut cache.depth_guest_range_index,
                key,
                base_format,
            );
        }
        assert!(cache
            .canonical_depth_key(source_alias)
            .same_live_identity(source));
        assert!(cache
            .canonical_depth_key(shadow_alias)
            .same_live_identity(shadow));
        cache.depth_generations.insert(source_alias, 17);
        cache.depth_generations.insert(shadow_alias, 23);
        cache.guest_stale_depth.insert(source_alias);
        cache.guest_stale_depth.insert(shadow_alias);
        cache
            .depth_shadow_generations
            .insert(shadow_alias, (source_alias, 17));

        cache.apply_mapping_epoch_transitions(
            &[
                RtMappingEpochTransition {
                    gpu_va: SOURCE_VA,
                    size: 0x8000,
                    old_epoch: 31,
                    new_epoch: 32,
                },
                RtMappingEpochTransition {
                    gpu_va: SHADOW_VA,
                    size: 0x8000,
                    old_epoch: 41,
                    new_epoch: 42,
                },
            ],
            &[],
        );

        let source_generation_key = cache
            .depth_generations
            .get_key_value(&remapped_source)
            .unwrap()
            .0;
        let shadow_generation_key = cache
            .depth_generations
            .get_key_value(&remapped_shadow)
            .unwrap()
            .0;
        assert!(source_generation_key.same_live_identity(remapped_source));
        assert!(shadow_generation_key.same_live_identity(remapped_shadow));
        assert!(cache
            .guest_stale_depth
            .get(&remapped_source)
            .unwrap()
            .same_live_identity(remapped_source));
        assert!(cache
            .guest_stale_depth
            .get(&remapped_shadow)
            .unwrap()
            .same_live_identity(remapped_shadow));
        let (tracked_shadow, (tracked_source, generation)) =
            cache.depth_shadow_generations.iter().next().unwrap();
        assert!(tracked_shadow.same_live_identity(remapped_shadow));
        assert!(tracked_source.same_live_identity(remapped_source));
        assert_eq!(*generation, 17);
        assert!(cache.depth_shadow_is_current(remapped_source, remapped_shadow));

        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn changed_mapping_inside_footprint_blocks_epoch_rekey_and_stales_target() {
        const VA: u64 = 0x6204_9000;
        const CPU: u64 = 0x1b14_0000;
        let stored = RtKey::with_cpu(69, 32, 32, VA, CPU)
            .with_mapping_epoch(30)
            .with_guest_size_bytes(0x8000);
        let remapped = stored.with_mapping_epoch(31);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.mark_drawn(stored);

        cache.apply_mapping_epoch_transitions(
            &[RtMappingEpochTransition {
                gpu_va: VA,
                size: 0x8000,
                old_epoch: 30,
                new_epoch: 31,
            }],
            &[(VA + 0x4000, 0x1000)],
        );

        assert_eq!(cache.cache.keys().next().unwrap().mapping_epoch, 30);
        assert!(cache.color_is_guest_stale(stored));
        assert!(cache.color_requires_recreate(remapped, vk::Format::R8_UNORM));
        assert!(cache.find_drawn_color_for_exact_alias(remapped).is_none());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn canonical_lookups_preserve_stored_metadata_and_reject_changed_backing() {
        const VA: u64 = 0x6204_6000;
        const CPU: u64 = 0x1b08_0000;
        let stored = RtKey::with_cpu(64, 32, 32, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x40_000);
        let unknown = RtKey::new(64, 32, 32, VA);
        let shorter_span = stored.with_guest_size_bytes(0x20_000);
        let remapped = stored.with_mapping_epoch(8);
        let relaid = stored.with_block_linear_layout(0, 3, 0, 0);
        let relayered = stored.with_base_layer(2);
        let other_cpu = RtKey::with_cpu(64, 32, 32, VA, CPU + 0x10_000)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x40_000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.insert_depth_image(stored, test_depth_image());

        for want in [unknown, shorter_span] {
            assert_eq!(
                cache.color_exact_with_format(want).map(|result| result.0),
                Some(stored)
            );
            assert_eq!(
                cache.find_color_with_format(want).map(|result| result.0),
                Some(stored)
            );
            assert_eq!(
                cache
                    .find_sampleable_color_with_format(want)
                    .map(|result| result.0),
                Some(stored)
            );
            assert_eq!(cache.find_depth(want).map(|result| result.0), Some(stored));
            assert_eq!(
                cache.find_sampleable_depth(want).map(|result| result.0),
                Some(stored)
            );
        }

        for want in [remapped, relaid, relayered, other_cpu] {
            assert!(cache.color_exact_with_format(want).is_none());
            assert!(cache.find_color_with_format(want).is_none());
            assert!(cache.find_sampleable_color_with_format(want).is_none());
            assert!(cache.find_depth(want).is_none());
            assert!(cache.find_sampleable_depth(want).is_none());
        }

        assert!(!cache.depth_requires_recreate(
            unknown,
            vk::Format::D24_UNORM_S8_UINT,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        ));
        assert!(!cache.color_requires_recreate(shorter_span, vk::Format::R8_UNORM));
        assert!(!cache.depth_requires_recreate(
            shorter_span,
            vk::Format::D24_UNORM_S8_UINT,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        ));
        for want in [remapped, relaid, relayered, other_cpu] {
            assert!(cache.depth_requires_recreate(
                want,
                vk::Format::D24_UNORM_S8_UINT,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            ));
        }
        assert!(cache.depth_requires_recreate(
            stored,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        ));

        cache.cache.clear();
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_texture_alias_allows_different_backing_footprint_spans() {
        const VA: u64 = 0x6204_8000;
        const CPU: u64 = 0x1b10_0000;
        let stored = RtKey::with_cpu(63, 32, 32, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x40_000);
        let sampled = stored.with_guest_size_bytes(0x20_000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.drawn_stamp.insert(stored, 1);

        assert_eq!(
            cache
                .find_drawn_color_for_exact_alias(sampled)
                .map(|result| result.0),
            Some(stored)
        );
        assert!(stored.same_alias_view_identity(sampled));
        assert!(!stored.same_live_identity(sampled));

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn guest_footprint_change_rekeys_live_color_without_losing_state() {
        const PAGE: u64 = 1 << super::RT_COLOR_GPU_PAGE_SHIFT;
        const VA: u64 = 0x6300_0000;
        const CPU: u64 = 0x1c00_0000;
        let stored = RtKey::with_cpu(66, 32, 32, VA, CPU).with_guest_size_bytes(0x1000);
        let expanded = stored.with_guest_size_bytes(PAGE + 0x1000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        let stamp = cache.mark_drawn(stored);

        cache.refresh_color_footprint(stored, expanded);

        let (canonical, image) = cache.cache.get_key_value(&expanded).unwrap();
        assert_eq!(canonical.guest_size_bytes, expanded.guest_size_bytes);
        assert_eq!(image.format, vk::Format::R8_UNORM);
        assert_eq!(cache.drawn_stamp(expanded), Some(stamp));
        assert!(cache
            .color_gpu_page_index
            .get(&((VA + PAGE) >> super::RT_COLOR_GPU_PAGE_SHIFT))
            .is_some_and(|bucket| bucket.contains(&expanded)));
        assert_eq!(
            cache.color_guest_ranges[cache.color_guest_range_index[&expanded]]
                .key
                .guest_size_bytes,
            expanded.guest_size_bytes
        );

        let (removed, _) = cache.remove_color_image(expanded).unwrap();
        assert_eq!(removed.guest_size_bytes, expanded.guest_size_bytes);
        assert!(!cache
            .color_gpu_page_index
            .contains_key(&((VA + PAGE) >> super::RT_COLOR_GPU_PAGE_SHIFT)));
    }

    #[test]
    fn depth_footprint_rekey_keeps_alternate_va_and_tracking_canonical() {
        const VA: u64 = 0x6400_0000;
        const CPU: u64 = 0x1d00_0000;
        let stored = RtKey::with_cpu(67, 32, 32, VA, CPU).with_guest_size_bytes(0x1000);
        let alternate = RtKey::with_cpu(67, 32, 32, VA + 0x20_000, CPU)
            .with_guest_size_bytes(stored.guest_size_bytes);
        let expanded = stored.with_guest_size_bytes(0x4000);
        let shadow = RtKey::with_cpu(68, 32, 32, VA + 0x40_000, CPU + 0x40_000)
            .with_guest_size_bytes(0x1000);
        let mut cache = RtCache::new();
        let stored_image = test_depth_image();
        super::insert_guest_range(
            &mut cache.depth_guest_ranges,
            &mut cache.depth_guest_range_index,
            stored,
            stored_image.base_format,
        );
        cache.insert_depth_image(stored, stored_image);
        cache.insert_depth_image(shadow, test_depth_image());
        let generation = cache.mark_depth_written(stored);
        cache.mark_depth_written(shadow);
        assert!(cache.mark_depth_shadow_synced(stored, shadow));
        cache.guest_stale_depth.insert(stored);

        assert_eq!(cache.refresh_depth_footprint(stored, alternate), stored);
        assert!(cache.depth_cache.contains_key(&stored));
        assert!(!cache.depth_cache.contains_key(&alternate));

        assert_eq!(cache.refresh_depth_footprint(stored, expanded), expanded);
        let canonical = cache.depth_cache.get_key_value(&expanded).unwrap().0;
        assert_eq!(canonical.guest_size_bytes, expanded.guest_size_bytes);
        assert_eq!(cache.depth_generation(expanded), Some(generation));
        assert!(cache.depth_is_guest_stale(expanded));
        assert!(cache.depth_shadow_is_current(expanded, shadow));

        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn guest_footprint_includes_sample_grid_and_layout_padding() {
        let sampled =
            RtKey::with_cpu(63, 16, 16, (1 << 20) - 0x100, 0x1c00_0000).with_sample_grid(2, 2);
        let padded = sampled.with_guest_size_bytes(0x2000);
        let pitch = RtKey::with_cpu(64, 256, 17, (1 << 20) - 0x2000, 0x1d00_0000)
            .with_pitch_linear_layout(256)
            .with_guest_size_bytes(256 * 17);

        assert_eq!(
            super::rt_guest_size_bytes(sampled, vk::Format::R8_UNORM),
            Some(0x400)
        );
        assert_eq!(
            super::rt_guest_size_bytes(padded, vk::Format::R8_UNORM),
            Some(0x2000)
        );
        assert_eq!(
            super::rt_color_gpu_page_span(sampled, vk::Format::R8_UNORM),
            Some((0, 1))
        );
        assert_eq!(
            super::rt_color_gpu_page_span(padded, vk::Format::R8_UNORM),
            Some((0, 1))
        );
        assert_eq!(
            super::rt_guest_size_bytes(pitch, vk::Format::R8G8B8A8_UNORM),
            Some(256 * 17)
        );
        assert_eq!(
            super::rt_color_gpu_page_span(pitch, vk::Format::R8G8B8A8_UNORM),
            Some((0, 0))
        );
    }

    #[test]
    fn guest_ranges_use_exact_render_target_format_sizes() {
        const WIDTH: u32 = 7;
        const HEIGHT: u32 = 5;
        const GPU_VA: u64 = 0x1e00_0000;
        const CPU_ADDR: u64 = 0x2e00_0000;
        let key = RtKey::with_cpu(65, WIDTH, HEIGHT, GPU_VA, CPU_ADDR);

        for (format, bytes_per_pixel) in [
            (vk::Format::B5G5R5A1_UNORM_PACK16, 2),
            (vk::Format::A2R10G10B10_UNORM_PACK32, 4),
            (vk::Format::D16_UNORM, 2),
            (vk::Format::D24_UNORM_S8_UINT, 4),
            (vk::Format::D32_SFLOAT, 4),
            (vk::Format::X8_D24_UNORM_PACK32, 4),
        ] {
            let expected_size = u64::from(WIDTH * HEIGHT) * bytes_per_pixel;
            let range = super::guest_range_entry(key, format).unwrap();

            assert_eq!(super::rt_format_bytes(format), bytes_per_pixel);
            assert_eq!(super::rt_guest_size_bytes(key, format), Some(expected_size));
            assert_eq!(
                (range.cpu_lo, range.cpu_hi),
                (CPU_ADDR, CPU_ADDR + expected_size)
            );
            assert_eq!(
                (range.gpu_lo, range.gpu_hi),
                (GPU_VA, GPU_VA + expected_size)
            );
        }
    }

    #[test]
    fn indexed_region_lookup_matches_nested_ranking_and_exclusion() {
        const VA: u64 = 0x6300_4000;
        const TIED_VA: u64 = 0x6300_8000;
        let exact = RtKey::new(51, 8, 8, VA);
        let padded = RtKey::new(51, 16, 16, VA);
        let large = RtKey::new(52, 64, 64, VA);
        let tied_old = RtKey::new(53, 32, 32, TIED_VA - 8);
        let tied_new = RtKey::new(54, 32, 32, TIED_VA - 16);
        let mut cache = RtCache::new();
        for key in [exact, padded, large, tied_old, tied_new] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
        }
        cache.drawn_stamp.insert(exact, 1);
        cache.drawn_stamp.insert(padded, 100);
        cache.drawn_stamp.insert(large, 200);
        cache.drawn_stamp.insert(tied_old, 300);
        cache.drawn_stamp.insert(tied_new, 301);

        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(8, 8, VA)),
            region_signature(cache.find_drawn_color_region_at_full_scan(8, 8, VA, None))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(8, 8, VA)
                .map(|region| region.key),
            Some(exact)
        );
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at_excluding(8, 8, VA, exact)),
            region_signature(cache.find_drawn_color_region_at_full_scan(8, 8, VA, Some(exact)))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at_excluding(8, 8, VA, exact)
                .map(|region| region.key),
            Some(padded)
        );
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(4, 4, TIED_VA)),
            region_signature(cache.find_drawn_color_region_at_full_scan(4, 4, TIED_VA, None))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(4, 4, TIED_VA)
                .map(|region| region.key),
            Some(tied_new)
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn indexed_region_lookup_covers_page_boundaries_and_volume_layers() {
        const PAGE: u64 = 1 << 20;
        let crossing = RtKey::new(61, 64, 4, 0x40 * PAGE + PAGE - 128);
        let volume = RtKey::new(62, 1024, 512, 0x50 * PAGE + PAGE - 256).with_volume_depth(4);
        let later_layer_va = volume.gpu_va + 3 * 1024 * 512;
        let later_slice = RtKey::new(63, 64, 64, later_layer_va);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            crossing,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.insert_color_image(
            volume,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.insert_color_image(
            later_slice,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.mark_drawn(crossing);
        cache.mark_drawn(volume);
        cache.mark_drawn(later_slice);

        let crossing_va = crossing.gpu_va + 128;
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(16, 1, crossing_va)),
            region_signature(cache.find_drawn_color_region_at_full_scan(16, 1, crossing_va, None))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(16, 1, crossing_va)
                .map(|region| region.key),
            Some(crossing)
        );

        let later_page = later_layer_va >> super::RT_COLOR_GPU_PAGE_SHIFT;
        assert!(cache
            .color_gpu_page_index
            .get(&later_page)
            .is_some_and(|bucket| bucket.contains(&volume)));
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(64, 64, later_layer_va)),
            region_signature(cache.find_drawn_color_region_at_full_scan(
                64,
                64,
                later_layer_va,
                None
            ))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(64, 64, later_layer_va)
                .map(|region| region.key),
            Some(later_slice)
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn color_lookup_index_tracks_format_recreation_clear_and_canonical_keys() {
        const PAGE: u64 = 1 << 20;
        let old = RtKey::with_cpu(71, 256, 2, 0x70 * PAGE + PAGE - 512, 0x1000);
        let removal_alias = RtKey::with_cpu(71, 256, 2, old.gpu_va, 0x2000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            old,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.mark_drawn(old);
        let next_page_va = old.gpu_va + 512;
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(1, 1, next_page_va)),
            region_signature(cache.find_drawn_color_region_at_full_scan(1, 1, next_page_va, None))
        );
        assert!(cache
            .find_drawn_color_region_at(1, 1, next_page_va)
            .is_none());

        let (removed_key, _) = cache.remove_color_image(removal_alias).unwrap();
        assert_eq!(removed_key.cpu_addr, old.cpu_addr);
        assert!(!cache
            .color_gpu_base_index
            .get(&old.gpu_va)
            .is_some_and(|bucket| bucket.contains(&old)));

        let recreated = RtKey::with_cpu(71, 256, 2, old.gpu_va, 0x3000);
        cache.insert_color_image(
            recreated,
            test_color_image(vk::Format::R16_UNORM, vk::Format::R16_UNORM),
        );
        cache.mark_drawn(recreated);
        assert_eq!(
            region_signature(cache.find_drawn_color_region_at(1, 1, next_page_va)),
            region_signature(cache.find_drawn_color_region_at_full_scan(1, 1, next_page_va, None))
        );
        assert_eq!(
            cache
                .find_drawn_color_region_at(1, 1, next_page_va)
                .map(|region| region.key.cpu_addr),
            Some(recreated.cpu_addr)
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
        assert!(cache.find_drawn_color_at(1, 1, old.gpu_va).is_none());
        assert!(cache
            .find_drawn_color_region_at(1, 1, next_page_va)
            .is_none());

        let after_clear = RtKey::with_cpu(71, 256, 2, old.gpu_va, 0x4000);
        cache.insert_color_image(
            after_clear,
            test_color_image(vk::Format::R16_UNORM, vk::Format::R16_UNORM),
        );
        cache.mark_drawn(after_clear);
        assert_eq!(
            cache
                .find_drawn_color_at(1, 1, old.gpu_va)
                .map(|result| result.0.cpu_addr),
            Some(after_clear.cpu_addr)
        );
        assert_eq!(
            cache.find_drawn_color_at(1, 1, old.gpu_va),
            cache.find_drawn_color_at_full_scan(1, 1, old.gpu_va)
        );

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn guest_write_range_stales_overlapping_rt_aliases() {
        const CPU: u64 = 0x10_0000;
        const VA: u64 = 0x50_0000;
        let large = RtKey::with_cpu(12, 1024, 1024, VA, CPU);
        let sampled = RtKey::new(12, 480, 272, VA);
        let cleared = RtKey::with_cpu(12, 128, 128, VA + 0x10_000, CPU + 0x10_000);
        let untouched = RtKey::with_cpu(13, 64, 64, VA + 0x40_000, CPU + 0x40_000);
        let depth = RtKey::with_cpu(12, 64, 64, VA + 0x20_000, CPU + 0x20_000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            large,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );
        cache.insert_color_image(
            sampled,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.insert_color_image(
            cleared,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        cache.insert_color_image(
            untouched,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        let depth_image = test_depth_image();
        super::insert_guest_range(
            &mut cache.depth_guest_ranges,
            &mut cache.depth_guest_range_index,
            depth,
            depth_image.base_format,
        );
        cache.insert_depth_image(depth, depth_image);
        cache.mark_drawn(large);
        cache.mark_drawn(sampled);
        cache.mark_drawn(cleared);
        cache.mark_drawn(untouched);

        cache.mark_guest_written_range(CPU, 0x30_000, &[(VA, 0x30_000)]);

        assert!(cache.color_is_guest_stale(large));
        assert!(cache.color_is_guest_stale(sampled));
        assert!(cache.color_is_guest_stale(cleared));
        assert!(!cache.color_is_guest_stale(untouched));
        assert!(cache.depth_is_guest_stale(depth));
        assert_eq!(cache.drawn_stamp(large), None);
        assert!(cache.find_color_with_format(sampled).is_some());
        assert!(cache.find_sampleable_color_with_format(sampled).is_none());

        cache.mark_drawn(large);
        cache.mark_synced_sample(sampled);
        cache.mark_cleared(cleared, true);
        cache.mark_depth_written(depth);

        assert!(!cache.color_is_guest_stale(large));
        assert!(!cache.color_is_guest_stale(sampled));
        assert!(!cache.color_is_guest_stale(cleared));
        assert!(!cache.depth_is_guest_stale(depth));
        assert!(cache.find_sampleable_color_with_format(sampled).is_some());

        cache.cache.clear();
        cache.clear_color_lookup_index();
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn indexed_guest_writes_match_exhaustive_alias_overlap() {
        let make_ranges = |count, base| {
            (0..count).map(|i| {
                let lo = base + (i * 7919) % 65536;
                let size = if i % 13 == 0 { 65536 } else { 1 + (i * 101) % 4096 };
                super::GuestRangeEntry {
                    key: RtKey::with_cpu(i as u32, 1, 1, lo, lo + 0x100000),
                    cpu_lo: lo + 0x100000,
                    cpu_hi: lo + 0x100000 + size,
                    gpu_lo: lo,
                    gpu_hi: lo + size,
                }
            }).collect::<Vec<_>>()
        };
        let color = make_ranges(257, 0x200000);
        let depth = make_ranges(125, 0x201000);
        let mut lookup = super::GuestRangeLookup::default();
        lookup.rebuild(1, &color, &depth);
        for i in 0..2048u64 {
            let lo = 0x1ff000 + (i * 127) % 0x24000;
            let size = i % 1024;
            let cpu_lo = if i % 3 == 0 { lo + 0x100000 } else { 0 };
            let cpu_hi = if cpu_lo == 0 { 0 } else { cpu_lo + size };
            let gpu = [(lo, size), (lo + 7, size / 2), (lo, size)];
            let (got_color, got_depth) = lookup.hits(cpu_lo, cpu_hi, &gpu);
            for (ranges, got) in [(&color, got_color), (&depth, got_depth)] {
                let expected: std::collections::HashSet<_> =
                    super::guest_range_hits(ranges, cpu_lo, cpu_hi, &gpu).into_iter().collect();
                let unique: std::collections::HashSet<_> = got.iter().copied().collect();
                assert_eq!(got.len(), unique.len(), "duplicate aliases");
                assert_eq!(unique, expected, "write={lo:#x}+{size:#x}");
            }
        }
        lookup.rebuild(2, &[], &[]);
        assert_eq!(lookup.hits(1, u64::MAX, &[(1, u64::MAX)]), (vec![], vec![]));
    }

    #[test]
    fn gpu_only_guest_write_range_stales_render_targets() {
        const CPU: u64 = 0x20_0000;
        const VA: u64 = 0x60_0000;
        let hit = RtKey::with_cpu(22, 64, 64, VA, CPU);
        let alias = RtKey::with_cpu(22, 128, 64, VA, CPU);
        let untouched = RtKey::with_cpu(23, 64, 64, VA + 0x20_000, CPU + 0x20_000);
        let mut cache = RtCache::new();
        for key in [hit, alias, untouched] {
            cache.insert_color_image(
                key,
                test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
            );
            cache.mark_drawn(key);
        }
        assert_eq!(
            cache
                .drawn_color_aliases(hit)
                .into_iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            vec![alias]
        );

        cache.mark_guest_written_range(0, 0, &[(VA + 0x100, 0x100)]);

        assert!(cache.color_is_guest_stale(hit));
        assert!(cache.color_is_guest_stale(alias));
        assert!(!cache.color_is_guest_stale(untouched));
        assert_eq!(cache.drawn_stamp(hit), None);
        assert_eq!(cache.drawn_stamp(alias), None);
        assert!(cache.drawn_color_aliases(hit).is_empty());
        assert!(cache.present_candidates(hit).is_empty());

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn unknown_guest_write_stales_all_render_targets() {
        let color = RtKey::with_cpu(31, 32, 32, 0x70_0000, 0x30_0000);
        let depth = RtKey::with_cpu(32, 32, 32, 0x80_0000, 0x40_0000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            color,
            test_color_image(vk::Format::R8_UNORM, vk::Format::R8_UNORM),
        );
        let depth_image = test_depth_image();
        super::insert_guest_range(
            &mut cache.depth_guest_ranges,
            &mut cache.depth_guest_range_index,
            depth,
            depth_image.base_format,
        );
        cache.insert_depth_image(depth, depth_image);
        cache.mark_drawn(color);
        cache.mark_depth_written(depth);

        cache.mark_all_guest_written();

        assert!(cache.color_is_guest_stale(color));
        assert!(cache.depth_is_guest_stale(depth));
        assert_eq!(cache.drawn_stamp(color), None);

        cache.cache.clear();
        cache.clear_color_lookup_index();
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn color_recreation_query_matches_format_view_compatibility() {
        let key = RtKey::with_cpu(7, 640, 480, 0x5123_4000, 0x1000_0000);
        let mut cache = RtCache::new();
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_UNORM));

        cache.insert_color_image(
            key,
            test_color_image(vk::Format::R8G8B8A8_UINT, vk::Format::R8G8B8A8_UNORM),
        );
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_UINT));
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_SINT));
        assert!(cache.color_requires_recreate(key, vk::Format::R16G16B16A16_UNORM));
        assert!(cache.color_requires_recreate(
            RtKey::with_cpu(7, 640, 480, key.gpu_va, 0x2000_0000),
            vk::Format::R8G8B8A8_UINT,
        ));

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn rgba8_integer_storage_view_shares_the_32_bit_rt_class() {
        assert!(rt_formats_compatible(
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::R8G8B8A8_UINT
        ));
    }

    #[test]
    fn rt_view_compatibility_follows_vulkan_bit_classes() {
        assert!(rt_formats_compatible(
            vk::Format::R32_SFLOAT,
            vk::Format::B10G11R11_UFLOAT_PACK32
        ));
        assert!(rt_formats_compatible(
            vk::Format::R16G16_SNORM,
            vk::Format::A8B8G8R8_SRGB_PACK32
        ));
        assert!(rt_formats_compatible(
            vk::Format::R32G32_SFLOAT,
            vk::Format::R16G16B16A16_SFLOAT
        ));
        assert!(rt_formats_compatible(
            vk::Format::R16_SFLOAT,
            vk::Format::R8G8_UNORM
        ));
        assert!(!rt_formats_compatible(
            vk::Format::R32_SFLOAT,
            vk::Format::R16_SFLOAT
        ));
        assert!(!rt_formats_compatible(
            vk::Format::R32G32_SFLOAT,
            vk::Format::R32_SFLOAT
        ));
    }

    #[test]
    fn depth_shadow_generation_tracks_source_mutations_and_recreation() {
        let source = RtKey::with_cpu(84, 1600, 900, 0x532c70000, 0x1000);
        let shadow = RtKey::new(84 ^ 0x4000_0000, 1600, 900, 0);
        let mut cache = RtCache::new();
        cache.insert_depth_image(source, test_depth_image());
        cache.insert_depth_image(shadow, test_depth_image());

        let first = cache.mark_depth_written(source);
        assert_eq!(cache.depth_generation(source), Some(first));
        assert!(!cache.depth_shadow_is_current(source, shadow));
        assert!(cache.mark_depth_shadow_synced(source, shadow));
        assert!(cache.depth_shadow_is_current(source, shadow));

        let second = cache.mark_depth_written(source);
        assert_ne!(first, second);
        assert!(!cache.depth_shadow_is_current(source, shadow));
        assert!(cache.mark_depth_shadow_synced(source, shadow));
        assert!(cache.depth_shadow_is_current(source, shadow));

        cache.mark_guest_written(source);
        assert!(!cache.depth_shadow_is_current(source, shadow));
        assert!(cache.mark_depth_shadow_synced(source, shadow));
        assert!(cache.depth_shadow_is_current(source, shadow));

        cache.forget_depth_tracking(source);
        assert_eq!(cache.depth_generation(source), None);
        assert!(!cache.depth_shadow_is_current(source, shadow));
    }

    #[test]
    fn indexed_exact_sampleable_alias_matches_legacy() {
        const VA: u64 = 0x5330_0000;
        const CPU: u64 = 0x8330_0000;
        let stored = RtKey::with_cpu(90, 128, 128, VA, CPU)
            .with_mapping_epoch(7)
            .with_block_linear_layout(0, 4, 0, 0)
            .with_guest_size_bytes(0x20_000);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );
        cache.insert_depth_image(stored, test_depth_image());

        for want in [
            stored,
            RtKey::with_cpu(90, 128, 128, VA, CPU),
            RtKey::request(90, 128, 128),
            stored.with_mapping_epoch(8),
            stored.with_block_linear_layout(0, 3, 0, 0),
            stored.with_base_layer(2),
            RtKey::with_cpu(90, 128, 128, VA, CPU + 0x10_000)
                .with_mapping_epoch(7)
                .with_block_linear_layout(0, 4, 0, 0),
        ] {
            assert_indexed_color_exact_matches_legacy(&cache, want);
            assert_indexed_depth_exact_matches_legacy(&cache, want);
        }

        cache.guest_stale_color.insert(stored);
        cache.guest_stale_depth.insert(stored);
        assert_indexed_color_exact_matches_legacy(&cache, stored);
        assert_indexed_depth_exact_matches_legacy(&cache, stored);
        cache.guest_stale_color.remove(&stored);
        cache.guest_stale_depth.remove(&stored);

        cache.cache.get_mut(&stored).unwrap().layout = vk::ImageLayout::UNDEFINED;
        cache.depth_cache.get_mut(&stored).unwrap().layout = vk::ImageLayout::UNDEFINED;
        assert_indexed_color_exact_matches_legacy(&cache, stored);
        assert_indexed_depth_exact_matches_legacy(&cache, stored);

        cache.cache.clear();
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn indexed_alias_lookup_defaults_on_and_requires_explicit_opt_out() {
        for enabled in [None, Some(""), Some("1"), Some("true"), Some("garbage")] {
            assert!(rt_alias_indexed_lookup_value_enabled(enabled));
        }
        for disabled in [Some("0"), Some("false"), Some("OFF"), Some(" no ")] {
            assert!(!rt_alias_indexed_lookup_value_enabled(disabled));
        }
    }

    #[test]
    fn indexed_physical_sampleable_alias_matches_legacy() {
        const CPU: u64 = 0x8338_0000;
        let stored = RtKey::with_cpu(90, 128, 128, 0x5338_0000, CPU)
            .with_mapping_epoch(9)
            .with_block_linear_layout(0, 4, 0, 0);
        let want = RtKey::with_cpu(90, 128, 128, 0x5339_0000, CPU)
            .with_mapping_epoch(9)
            .with_block_linear_layout(0, 4, 0, 0);
        let ambiguous = RtKey::with_cpu(90, 128, 128, 0x533a_0000, CPU)
            .with_mapping_epoch(9)
            .with_block_linear_layout(0, 4, 0, 0);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );

        assert_indexed_color_physical_matches_legacy(&cache, want, vk::Format::R16_SFLOAT);
        assert_indexed_color_physical_matches_legacy(&cache, want, vk::Format::R32_SFLOAT);

        cache.guest_stale_color.insert(stored);
        assert_indexed_color_physical_matches_legacy(&cache, want, vk::Format::R16_SFLOAT);
        cache.guest_stale_color.remove(&stored);
        cache.cache.get_mut(&stored).unwrap().layout = vk::ImageLayout::UNDEFINED;
        assert_indexed_color_physical_matches_legacy(&cache, want, vk::Format::R16_SFLOAT);
        cache.cache.get_mut(&stored).unwrap().layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;

        cache.insert_color_image(
            ambiguous,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );
        assert_indexed_color_physical_matches_legacy(&cache, want, vk::Format::R16_SFLOAT);

        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn sampleable_color_lookup_rejects_fuzzy_extent_aliases() {
        const VA: u64 = 0x5340_0000;
        const CPU: u64 = 0x8340_0000;
        let stored = RtKey::with_cpu(91, 128, 128, VA, CPU);
        let mismatched_extent = RtKey::with_cpu(91, 96, 96, VA, CPU);
        let same_backing = RtKey::with_cpu(91, 128, 128, VA + 0x10_000, CPU);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );

        assert_eq!(
            cache
                .find_color_with_format(mismatched_extent)
                .map(|result| result.0),
            Some(stored)
        );
        assert!(cache
            .find_sampleable_color_with_format(mismatched_extent)
            .is_none());
        assert_eq!(
            cache
                .find_sampleable_color_with_format(stored)
                .map(|result| result.0),
            Some(stored)
        );
        assert_eq!(
            cache
                .find_sampleable_color_with_format(same_backing)
                .map(|result| result.0),
            Some(stored)
        );
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_or_physical_color_alias_prefers_exact_identity() {
        const CPU: u64 = 0x8360_0000;
        let fallback = RtKey::with_cpu(93, 128, 128, 0x5360_0000, CPU);
        let exact = RtKey::with_cpu(93, 128, 128, 0x5361_0000, CPU);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            fallback,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );
        cache.insert_color_image(
            exact,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );

        assert_eq!(
            cache
                .find_sampleable_color_for_exact_or_physical_alias(exact, vk::Format::R16_SFLOAT,)
                .map(|result| result.0),
            Some(exact)
        );
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn exact_or_physical_color_alias_requires_compatible_exact_backing() {
        const CPU: u64 = 0x8370_0000;
        let stored = RtKey::with_cpu(94, 128, 128, 0x5370_0000, CPU);
        let same_backing = RtKey::with_cpu(94, 128, 128, 0x5371_0000, CPU);
        let mismatched_extent = RtKey::with_cpu(94, 96, 128, 0x5372_0000, CPU);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            stored,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );

        assert_eq!(
            cache
                .find_sampleable_color_for_exact_or_physical_alias(
                    same_backing,
                    vk::Format::R16_SFLOAT,
                )
                .map(|result| result.0),
            Some(stored)
        );
        assert!(
            cache
                .find_sampleable_color_for_exact_or_physical_alias(
                    same_backing,
                    vk::Format::R32_SFLOAT,
                )
                .is_none()
        );
        assert!(cache
            .find_sampleable_color_for_exact_or_physical_alias(
                mismatched_extent,
                vk::Format::R16_SFLOAT,
            )
            .is_none());

        cache.mark_guest_written(stored);
        assert!(
            cache
                .find_sampleable_color_for_exact_or_physical_alias(
                    same_backing,
                    vk::Format::R16_SFLOAT,
                )
                .is_none()
        );
        cache.mark_drawn(stored);

        let ambiguous = RtKey::with_cpu(94, 128, 128, 0x5373_0000, CPU);
        cache.insert_color_image(
            ambiguous,
            test_color_image(vk::Format::R16_SFLOAT, vk::Format::R16_SFLOAT),
        );
        assert!(
            cache
                .find_sampleable_color_for_exact_or_physical_alias(
                    same_backing,
                    vk::Format::R16_SFLOAT,
                )
                .is_none()
        );
        cache.cache.clear();
        cache.clear_color_lookup_index();
    }

    #[test]
    fn sampleable_depth_lookup_rejects_fuzzy_extent_aliases() {
        const VA: u64 = 0x5350_0000;
        const CPU: u64 = 0x8350_0000;
        let stored = RtKey::with_cpu(92, 128, 128, VA, CPU);
        let fuzzy_request = RtKey::request(92, 96, 96);
        let same_backing = RtKey::with_cpu(92, 128, 128, VA + 0x10_000, CPU);
        let mut cache = RtCache::new();
        cache.insert_depth_image(stored, test_depth_image());

        assert_eq!(
            cache.find_depth(fuzzy_request).map(|result| result.0),
            Some(stored)
        );
        assert!(cache.find_sampleable_depth(fuzzy_request).is_none());
        assert_eq!(
            cache.find_sampleable_depth(stored).map(|result| result.0),
            Some(stored)
        );
        assert_eq!(
            cache
                .find_sampleable_depth(same_backing)
                .map(|result| result.0),
            Some(stored)
        );
        cache.depth_cache.clear();
        cache.depth_cpu_base_index.clear();
    }

    #[test]
    fn physical_depth_alias_requires_exact_backing_and_extent() {
        let canonical = RtKey::with_cpu(7, 1600, 900, 0x1000, 0x8000);
        let alternate_va = RtKey::with_cpu(7, 1600, 900, 0x2000, 0x8000);
        assert!(same_physical_backing(canonical, alternate_va));

        assert!(!same_physical_backing(
            canonical,
            RtKey::with_cpu(8, 1600, 900, 0x2000, 0x8000),
        ));
        assert!(!same_physical_backing(
            canonical,
            RtKey::with_cpu(7, 1280, 720, 0x2000, 0x8000),
        ));
        assert!(!same_physical_backing(
            canonical,
            RtKey::with_cpu(7, 1600, 900, 0x2000, 0),
        ));
        assert!(!same_physical_backing(
            canonical,
            RtKey::with_cpu(7, 1600, 900, 0x2000, 0x8000).with_volume_depth(32),
        ));
    }

    #[test]
    fn volume_depth_participates_in_render_target_identity() {
        let d2 = RtKey::with_cpu(31, 32, 32, 0x50a900000, 0x8000);
        let d3 = d2.with_volume_depth(32);
        assert!(!d2.is_3d);
        assert_eq!(d2.render_layer_count(), 1);
        assert!(d3.is_3d);
        assert_eq!(d3.depth, 32);
        assert_eq!(d3.render_layer_count(), 32);
        assert_ne!(d2, d3);
    }

    #[test]
    fn pinned_present_key_rejects_volume_targets() {
        let d2 = RtKey::with_cpu(33, 32, 32, 0x50b900000, 0x8000);
        let d3 = d2.with_volume_depth(4);
        let mut cache = RtCache::new();
        cache.insert_color_image(
            d3,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );
        cache.mark_drawn(d3);
        assert_eq!(cache.present_key_pinned_at_va(d2), None);

        cache.insert_color_image(
            d2,
            test_color_image(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
        );
        cache.mark_drawn(d2);
        assert_eq!(cache.present_key_pinned_at_va(d2), Some(d2));
    }

    #[test]
    fn depth_view_may_cover_a_smaller_logical_extent_at_the_same_base() {
        let allocation = RtKey::new(57, 1072, 600, 0x524370000);
        assert!(same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(57, 1068, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(57, 1073, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(58, 1068, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(57, 1068, 600, 0x524371000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(57, 1056, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation,
            RtKey::new(57, 1068, 599, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation.with_sample_grid(1, 2),
            RtKey::new(57, 1068, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation.with_base_layer(1),
            RtKey::new(57, 1068, 600, 0x524370000),
        ));
        assert!(!same_d24_depth_allocation_covering(
            allocation.with_volume_depth(2),
            RtKey::new(57, 1068, 600, 0x524370000),
        ));
    }

    #[test]
    fn depth_recreation_requires_matching_format_and_aspects() {
        let depth = vk::ImageAspectFlags::DEPTH;
        let depth_stencil = depth | vk::ImageAspectFlags::STENCIL;
        assert!(!super::depth_image_requires_recreate(
            vk::Format::D32_SFLOAT,
            depth,
            vk::Format::D32_SFLOAT,
            depth,
        ));
        assert!(super::depth_image_requires_recreate(
            vk::Format::D32_SFLOAT,
            depth,
            vk::Format::D24_UNORM_S8_UINT,
            depth,
        ));
        assert!(super::depth_image_requires_recreate(
            vk::Format::D24_UNORM_S8_UINT,
            depth,
            vk::Format::D24_UNORM_S8_UINT,
            depth_stencil,
        ));
    }

    #[test]
    fn sample_view_key_distinguishes_format_aspect_type_and_swizzle() {
        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::B,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::R,
            a: vk::ComponentSwizzle::A,
        };
        let key = RtSampleViewKey::new(
            vk::Format::A8B8G8R8_UNORM_PACK32,
            vk::ImageAspectFlags::COLOR,
            vk::ImageViewType::TYPE_2D,
            components,
        );
        assert_eq!(
            key,
            RtSampleViewKey::new(
                vk::Format::A8B8G8R8_UNORM_PACK32,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                components,
            )
        );
        assert_ne!(
            key,
            RtSampleViewKey::new(
                vk::Format::B8G8R8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                components,
            )
        );
        assert_ne!(
            key,
            RtSampleViewKey::new(
                vk::Format::A8B8G8R8_UNORM_PACK32,
                vk::ImageAspectFlags::DEPTH,
                vk::ImageViewType::TYPE_2D,
                components,
            )
        );
        assert_ne!(
            key,
            RtSampleViewKey::new(
                vk::Format::A8B8G8R8_UNORM_PACK32,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_3D,
                components,
            )
        );
        assert_ne!(
            key,
            RtSampleViewKey::new(
                vk::Format::A8B8G8R8_UNORM_PACK32,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                vk::ComponentMapping::default(),
            )
        );
    }
}

#[cfg(test)]
mod allocation_cleanup_tests {
    use super::{destroy_gpu_image, RtCache, RtKey};
    use ash::vk;
    use ash::vk::Handle;
    use std::cell::RefCell;
    use std::ffi::c_void;

    #[derive(Default)]
    struct MockState {
        fail: u8,
        events: Vec<&'static str>,
    }

    thread_local! {
        static STATE: RefCell<MockState> = RefCell::new(MockState::default());
    }

    fn record(event: &'static str) -> u8 {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.events.push(event);
            state.fail
        })
    }

    unsafe extern "system" fn create_image(
        _: vk::Device,
        _: *const vk::ImageCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        image: *mut vk::Image,
    ) -> vk::Result {
        if record("create") == 1 {
            return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        unsafe { *image = vk::Image::from_raw(11) };
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn requirements(
        _: vk::Device,
        _: vk::Image,
        requirements: *mut vk::MemoryRequirements,
    ) {
        record("requirements");
        unsafe {
            *requirements = vk::MemoryRequirements {
                size: 4096,
                alignment: 256,
                memory_type_bits: 1,
            };
        }
    }

    unsafe extern "system" fn allocate_memory(
        _: vk::Device,
        _: *const vk::MemoryAllocateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        memory: *mut vk::DeviceMemory,
    ) -> vk::Result {
        if record("allocate") == 3 {
            return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        unsafe { *memory = vk::DeviceMemory::from_raw(22) };
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn bind_memory(
        _: vk::Device,
        _: vk::Image,
        _: vk::DeviceMemory,
        _: vk::DeviceSize,
    ) -> vk::Result {
        if record("bind") == 4 {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        } else {
            vk::Result::SUCCESS
        }
    }

    unsafe extern "system" fn create_view(
        _: vk::Device,
        _: *const vk::ImageViewCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        view: *mut vk::ImageView,
    ) -> vk::Result {
        if record("view") == 5 {
            return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        unsafe { *view = vk::ImageView::from_raw(33) };
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn destroy_image(
        _: vk::Device,
        _: vk::Image,
        _: *const vk::AllocationCallbacks<'_>,
    ) {
        record("destroy-image");
    }

    unsafe extern "system" fn free_memory(
        _: vk::Device,
        _: vk::DeviceMemory,
        _: *const vk::AllocationCallbacks<'_>,
    ) {
        record("free-memory");
    }

    unsafe extern "system" fn destroy_view(
        _: vk::Device,
        _: vk::ImageView,
        _: *const vk::AllocationCallbacks<'_>,
    ) {
        record("destroy-view");
    }

    fn device() -> ash::Device {
        unsafe {
            ash::Device::load_with(|name| match name.to_bytes() {
                b"vkCreateImage" => create_image as *const () as *const c_void,
                b"vkGetImageMemoryRequirements" => requirements as *const () as *const c_void,
                b"vkAllocateMemory" => allocate_memory as *const () as *const c_void,
                b"vkBindImageMemory" => bind_memory as *const () as *const c_void,
                b"vkCreateImageView" => create_view as *const () as *const c_void,
                b"vkDestroyImage" => destroy_image as *const () as *const c_void,
                b"vkFreeMemory" => free_memory as *const () as *const c_void,
                b"vkDestroyImageView" => destroy_view as *const () as *const c_void,
                _ => std::ptr::null(),
            }, vk::Device::from_raw(1))
        }
    }

    fn cache(fail: u8) -> RtCache {
        STATE.with(|state| *state.borrow_mut() = MockState { fail, events: Vec::new() });
        let mut cache = RtCache::new();
        if fail != 6 {
            let mut props = vk::PhysicalDeviceMemoryProperties::default();
            props.memory_type_count = 1;
            props.memory_types[0].property_flags = if fail == 2 {
                vk::MemoryPropertyFlags::HOST_VISIBLE
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            cache.set_mem_properties(props);
        }
        cache
    }

    #[test]
    fn color_and_depth_construction_release_every_failed_allocation() {
        let device = device();
        let key = RtKey::new(1, 32, 32, 0x10000);
        for (format, usage, aspect) in [
            (vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR),
            (vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH),
        ] {
            for (fail, expected) in [
                (1, vec!["create"]),
                (2, vec!["create", "requirements", "destroy-image"]),
                (3, vec!["create", "requirements", "allocate", "destroy-image"]),
                (4, vec!["create", "requirements", "allocate", "bind", "destroy-image", "free-memory"]),
                (5, vec!["create", "requirements", "allocate", "bind", "view", "destroy-image", "free-memory"]),
                (6, vec![]),
            ] {
                let cache = cache(fail);
                assert!(cache.create_image_inner(&device, key, format, usage, aspect).is_err());
                STATE.with(|state| assert_eq!(state.borrow().events, expected, "failure stage {fail}"));
            }
        }
    }

    #[test]
    fn successful_construction_retains_resources_until_image_destruction() {
        let device = device();
        let cache = cache(0);
        let image = cache.create_image(&device, RtKey::new(1, 32, 32, 0x10000), vk::Format::R8G8B8A8_UNORM).unwrap();
        assert_eq!(image.image, vk::Image::from_raw(11));
        assert_eq!(image.memory, vk::DeviceMemory::from_raw(22));
        assert_eq!(image.view, vk::ImageView::from_raw(33));
        STATE.with(|state| assert_eq!(state.borrow().events, vec!["create", "requirements", "allocate", "bind", "view"]));
        destroy_gpu_image(&device, image);
        STATE.with(|state| assert_eq!(state.borrow().events, vec!["create", "requirements", "allocate", "bind", "view", "destroy-view", "destroy-image", "free-memory"]));
    }
}
