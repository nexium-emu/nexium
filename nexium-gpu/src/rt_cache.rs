use ash::vk;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

const RT_COLOR_GPU_PAGE_SHIFT: u32 = 20;

#[derive(Clone, Copy, Debug)]
pub struct RtKey {
    pub nvmap_id: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub is_3d: bool,
    pub gpu_va: u64,
    pub cpu_addr: u64,
}

impl PartialEq for RtKey {
    fn eq(&self, other: &Self) -> bool {
        self.nvmap_id == other.nvmap_id
            && self.width == other.width
            && self.height == other.height
            && self.depth == other.depth
            && self.is_3d == other.is_3d
            && self.gpu_va == other.gpu_va
    }
}

impl Eq for RtKey {}

impl Hash for RtKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nvmap_id.hash(state);
        self.width.hash(state);
        self.height.hash(state);
        self.depth.hash(state);
        self.is_3d.hash(state);
        self.gpu_va.hash(state);
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
            gpu_va,
            cpu_addr,
        }
    }

    pub fn with_volume_depth(mut self, depth: u32) -> Self {
        self.depth = depth.max(1);
        self.is_3d = self.depth > 1;
        self
    }

    pub fn render_layer_count(self) -> u32 {
        if self.is_3d {
            self.depth.max(1)
        } else {
            1
        }
    }

    pub fn request(nvmap_id: u32, width: u32, height: u32) -> Self {
        Self::new(nvmap_id, width, height, 0)
    }

    pub fn label(self) -> String {
        if self.gpu_va != 0 {
            if self.is_3d {
                format!(
                    "{}:{}x{}x{}@{:x}",
                    self.nvmap_id, self.width, self.height, self.depth, self.gpu_va
                )
            } else {
                format!(
                    "{}:{}x{}@{:x}",
                    self.nvmap_id, self.width, self.height, self.gpu_va
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
    pub views: HashMap<vk::Format, vk::ImageView>,
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

fn dims_close(a: u32, b: u32) -> bool {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    lo != 0 && hi <= lo.saturating_mul(2)
}

pub(crate) fn is_synthetic_copy_key(key: RtKey) -> bool {
    key.nvmap_id == u32::MAX
}

pub fn same_physical_backing(a: RtKey, b: RtKey) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.depth == b.depth
        && a.is_3d == b.is_3d
        && a.nvmap_id == b.nvmap_id
        && a.cpu_addr != 0
        && a.cpu_addr == b.cpu_addr
}

fn same_d24_depth_allocation_covering(existing: RtKey, want: RtKey) -> bool {
    let existing_row = (existing.width as u64 * 4).next_multiple_of(64);
    let wanted_row = (want.width as u64 * 4).next_multiple_of(64);
    want.gpu_va != 0
        && existing.nvmap_id == want.nvmap_id
        && existing.gpu_va == want.gpu_va
        && existing.width >= want.width
        && existing.height == want.height
        && existing_row == wanted_row
}

pub struct RtCache {
    cache: HashMap<RtKey, GpuImage>,
    color_gpu_base_index: HashMap<u64, Vec<RtKey>>,
    color_gpu_page_index: HashMap<u64, Vec<RtKey>>,
    depth_cache: HashMap<RtKey, GpuImage>,
    snapshots: HashMap<RtKey, GpuImage>,
    mem_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    drawn_stamp: HashMap<RtKey, u64>,
    guest_stale_color: HashSet<RtKey>,
    guest_stale_depth: HashSet<RtKey>,
    present_excluded: HashSet<RtKey>,
    present_flip_y: HashMap<RtKey, bool>,
    drawn_counter: u64,
    frame_draws: HashMap<RtKey, u32>,
    frame_real_draws: HashMap<RtKey, u32>,
    depth_generation_counter: u64,
    depth_generations: HashMap<RtKey, u64>,
    depth_shadow_generations: HashMap<RtKey, (RtKey, u64)>,
    guest_cpu_lo: u64,
    guest_cpu_hi: u64,
    guest_gpu_lo: u64,
    guest_gpu_hi: u64,
    color_guest_ranges: Vec<GuestRangeEntry>,
    color_guest_range_index: HashMap<RtKey, usize>,
    depth_guest_ranges: Vec<GuestRangeEntry>,
    depth_guest_range_index: HashMap<RtKey, usize>,
}

#[derive(Clone, Copy)]
struct GuestRangeEntry {
    key: RtKey,
    cpu_lo: u64,
    cpu_hi: u64,
    gpu_lo: u64,
    gpu_hi: u64,
}

fn guest_range_entry(key: RtKey, base_format: vk::Format) -> Option<GuestRangeEntry> {
    let image_size = (key.width as u64)
        .saturating_mul(key.height as u64)
        .saturating_mul(key.render_layer_count() as u64)
        .saturating_mul(rt_format_bytes(base_format));
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
) {
    let Some(entry) = guest_range_entry(key, base_format) else {
        return;
    };
    if let Some(&pos) = index.get(&key) {
        ranges[pos] = entry;
    } else {
        index.insert(key, ranges.len());
        ranges.push(entry);
    }
}

fn remove_guest_range(
    ranges: &mut Vec<GuestRangeEntry>,
    index: &mut HashMap<RtKey, usize>,
    key: RtKey,
) {
    if let Some(pos) = index.remove(&key) {
        ranges.swap_remove(pos);
        if pos < ranges.len() {
            index.insert(ranges[pos].key, pos);
        }
    }
}

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

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            color_gpu_base_index: HashMap::new(),
            color_gpu_page_index: HashMap::new(),
            depth_cache: HashMap::new(),
            snapshots: HashMap::new(),
            mem_properties: None,
            drawn_stamp: HashMap::new(),
            guest_stale_color: HashSet::new(),
            guest_stale_depth: HashSet::new(),
            present_excluded: HashSet::new(),
            present_flip_y: HashMap::new(),
            drawn_counter: 0,
            frame_draws: HashMap::new(),
            frame_real_draws: HashMap::new(),
            depth_generation_counter: 0,
            depth_generations: HashMap::new(),
            depth_shadow_generations: HashMap::new(),
            guest_cpu_lo: u64::MAX,
            guest_cpu_hi: 0,
            guest_gpu_lo: u64::MAX,
            guest_gpu_hi: 0,
            color_guest_ranges: Vec::new(),
            color_guest_range_index: HashMap::new(),
            depth_guest_ranges: Vec::new(),
            depth_guest_range_index: HashMap::new(),
        }
    }

    fn index_color_lookup(&mut self, key: RtKey, format: vk::Format) {
        if is_synthetic_copy_key(key) {
            return;
        }
        let base = self.color_gpu_base_index.entry(key.gpu_va).or_default();
        if !base.contains(&key) {
            base.push(key);
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
        let Some((first_page, last_page)) = rt_color_gpu_page_span(key, format) else {
            return;
        };
        for page in first_page..=last_page {
            remove_lookup_key(&mut self.color_gpu_page_index, page, key);
        }
    }

    fn extend_guest_bounds(&mut self, key: RtKey, base_format: vk::Format) {
        let image_size = (key.width as u64)
            .saturating_mul(key.height as u64)
            .saturating_mul(key.render_layer_count() as u64)
            .saturating_mul(rt_format_bytes(base_format));
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

    fn insert_color_image(&mut self, key: RtKey, image: GpuImage) {
        let format = image.format;
        debug_assert!(!self.cache.contains_key(&key));
        self.extend_guest_bounds(key, image.base_format);
        insert_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            key,
            image.base_format,
        );
        self.cache.insert(key, image);
        let canonical = self
            .cache
            .get_key_value(&key)
            .map(|(stored, _)| *stored)
            .unwrap();
        self.index_color_lookup(canonical, format);
    }

    fn remove_color_image(&mut self, key: RtKey) -> Option<(RtKey, GpuImage)> {
        let (canonical, image) = self.cache.remove_entry(&key)?;
        self.unindex_color_lookup(canonical, image.format);
        remove_guest_range(
            &mut self.color_guest_ranges,
            &mut self.color_guest_range_index,
            canonical,
        );
        Some((canonical, image))
    }

    fn clear_color_lookup_index(&mut self) {
        self.color_gpu_base_index.clear();
        self.color_gpu_page_index.clear();
    }

    fn canonical_depth_key(&self, key: RtKey) -> RtKey {
        if self.depth_cache.contains_key(&key) {
            return key;
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
        let generation = self.next_depth_generation();
        self.depth_generations.insert(canonical, generation);
        generation
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
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
        self.guest_stale_color.remove(&key);
        self.present_excluded.insert(key);
        self.drawn_counter
    }

    pub fn mark_guest_written(&mut self, key: RtKey) {
        let stale: Vec<_> = self
            .cache
            .keys()
            .filter(|existing| **existing == key || same_physical_backing(**existing, key))
            .copied()
            .collect();
        for stale_key in stale {
            self.drawn_stamp.remove(&stale_key);
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
            self.guest_stale_depth.insert(stale_key);
        }
    }

    pub fn mark_guest_written_range(
        &mut self,
        cpu_addr: u64,
        size: u64,
        gpu_ranges: &[(u64, u64)],
    ) {
        if cpu_addr == 0 || size == 0 {
            return;
        }
        let cpu_may_overlap =
            cpu_addr < self.guest_cpu_hi && self.guest_cpu_lo < cpu_addr.saturating_add(size);
        let gpu_may_overlap = gpu_ranges.iter().any(|(gpu_va, gpu_size)| {
            *gpu_va < self.guest_gpu_hi && self.guest_gpu_lo < gpu_va.saturating_add(*gpu_size)
        });
        if !cpu_may_overlap && !gpu_may_overlap {
            return;
        }
        let cpu_end = cpu_addr.saturating_add(size);
        let stale_color = guest_range_hits(&self.color_guest_ranges, cpu_addr, cpu_end, gpu_ranges);
        for stale_key in stale_color {
            self.drawn_stamp.remove(&stale_key);
            self.guest_stale_color.insert(stale_key);
            self.present_excluded.insert(stale_key);
            self.frame_draws.remove(&stale_key);
            self.frame_real_draws.remove(&stale_key);
        }
        let stale_depth = guest_range_hits(&self.depth_guest_ranges, cpu_addr, cpu_end, gpu_ranges);
        for stale_key in stale_depth {
            self.mark_depth_written(stale_key);
            self.guest_stale_depth.insert(stale_key);
        }
    }

    pub fn color_is_guest_stale(&self, key: RtKey) -> bool {
        self.guest_stale_color.contains(&key)
    }

    pub fn depth_is_guest_stale(&self, key: RtKey) -> bool {
        self.guest_stale_depth.contains(&key)
    }

    pub fn reset_frame_draws(&mut self) {
        self.frame_draws.clear();
        self.frame_real_draws.clear();
    }

    pub fn mark_cleared(&mut self, key: RtKey, full_target: bool) {
        if full_target {
            self.drawn_stamp.remove(&key);
            self.guest_stale_color.remove(&key);
        } else if self.drawn_stamp.contains_key(&key) {
            self.drawn_counter += 1;
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

    pub fn set_mem_properties(&mut self, props: vk::PhysicalDeviceMemoryProperties) {
        self.mem_properties = Some(props);
    }

    pub fn get_or_create(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<&mut GpuImage, String> {
        if self.cache.contains_key(&key) {
            return Ok(self.cache.get_mut(&key).unwrap());
        }
        self.get_or_create_with_format(key, device, vk::Format::R8G8B8A8_UNORM)
    }

    pub fn get_existing(&mut self, key: RtKey) -> Option<&mut GpuImage> {
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
        if self.depth_cache.contains_key(&key) {
            return self.depth_cache.get_mut(&key);
        }
        let matching = self
            .depth_cache
            .keys()
            .copied()
            .find(|existing| same_physical_backing(*existing, key))?;
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
            .max_by_key(|k| self.drawn_stamp.get(*k).copied().unwrap_or(0))
            .copied()
    }

    pub fn drawn_stamp(&self, key: RtKey) -> Option<u64> {
        self.drawn_stamp.get(&key).copied()
    }

    pub fn color_keys_for_nvmap(&self, nvmap_id: u32) -> Vec<(RtKey, vk::Format, u64, u32)> {
        self.cache
            .iter()
            .filter(|(k, _)| !is_synthetic_copy_key(**k) && k.nvmap_id == nvmap_id)
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
        let img = self.cache.get(&key)?;
        Some((
            key,
            img.image,
            img.view,
            img.layout,
            img.format,
            self.drawn_stamp.get(&key).copied().unwrap_or(0),
        ))
    }

    pub fn drawn_color_aliases(
        &self,
        key: RtKey,
    ) -> Vec<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut out = Vec::new();
        if is_synthetic_copy_key(key) || key.gpu_va == 0 {
            return out;
        }
        let Some(candidates) = self.color_gpu_base_index.get(&key.gpu_va) else {
            return out;
        };
        for candidate in candidates {
            let Some((stored, img)) = self.cache.get_key_value(candidate) else {
                continue;
            };
            let stored = *stored;
            if stored == key || stored.nvmap_id != key.nvmap_id || stored.gpu_va != key.gpu_va {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(&stored).copied() else {
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
        if is_synthetic_copy_key(key) || key.gpu_va == 0 {
            return out;
        }
        for (stored, img) in &self.cache {
            if is_synthetic_copy_key(*stored)
                || *stored == key
                || stored.nvmap_id != key.nvmap_id
                || stored.gpu_va != key.gpu_va
            {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(stored).copied() else {
                continue;
            };
            out.push((*stored, img.image, img.layout, img.format, stamp));
        }
        out.sort_by_key(|(_, _, _, _, stamp)| *stamp);
        out
    }

    pub(crate) fn color_requires_recreate(&self, key: RtKey, format: vk::Format) -> bool {
        self.cache.get(&key).is_some_and(|image| {
            image.format != format && !rt_formats_compatible(image.base_format, format)
        })
    }

    fn get_or_create_color_image(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<Option<GpuImage>, String> {
        let existing = self
            .cache
            .get(&key)
            .map(|image| (image.format, image.base_format));
        match existing {
            Some((current, _)) if current == format => {
                return Ok(None);
            }
            Some((current, base)) if rt_formats_compatible(base, format) => {
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
        if let Some(image) = self.get_or_create_color_image(key, device, format)? {
            retired.push(image);
        }
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
        if aspects.is_empty() {
            return Err("depth image requires at least one aspect".to_string());
        }
        let cache_key = if self.depth_cache.contains_key(&key) {
            key
        } else {
            self.depth_cache
                .keys()
                .copied()
                .find(|existing| same_physical_backing(*existing, key))
                .unwrap_or(key)
        };
        let recreate = self
            .depth_cache
            .get(&cache_key)
            .is_some_and(|image| image.format != format || image.aspects != aspects);
        if recreate {
            self.forget_depth_tracking(cache_key);
            if let Some(image) = self.depth_cache.remove(&cache_key) {
                remove_guest_range(
                    &mut self.depth_guest_ranges,
                    &mut self.depth_guest_range_index,
                    cache_key,
                );
                destroy_gpu_image(device, image);
            }
        }
        let created = !self.depth_cache.contains_key(&cache_key);
        if created {
            let image = self.create_image_inner(
                device,
                cache_key,
                format,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                aspects,
            )?;
            let depth_base_format = image.base_format;
            self.depth_cache.insert(cache_key, image);
            self.extend_guest_bounds(cache_key, depth_base_format);
            insert_guest_range(
                &mut self.depth_guest_ranges,
                &mut self.depth_guest_range_index,
                cache_key,
                depth_base_format,
            );
            self.mark_depth_written(cache_key);
        }
        Ok((self.depth_cache.get_mut(&cache_key).unwrap(), created))
    }

    pub fn find_depth(
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
        if let Some(img) = self.depth_cache.get(&want) {
            return Some((
                want,
                img.image,
                img.view,
                img.layout,
                img.format,
                img.aspects,
            ));
        }
        if want.gpu_va != 0 && want.cpu_addr != 0 {
            if let Some((key, img)) = self
                .depth_cache
                .iter()
                .find(|(key, _)| same_physical_backing(**key, want))
            {
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
            if k.nvmap_id != want.nvmap_id {
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
        self.find_depth(want)
            .filter(|(key, _, _, _, _, _)| !self.depth_is_guest_stale(*key))
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
                img.format == vk::Format::D24_UNORM_S8_UINT
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
            if !covers || !self.depth_shadow_is_current(*source, *shadow) {
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

    pub fn find_color(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        self.find_color_with_format(want)
            .map(|(k, image, view, layout, _)| (k, image, view, layout))
    }

    pub fn find_color_with_format(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
        if let Some(img) = self.cache.get(&want) {
            return Some((want, img.image, img.view, img.layout, img.format));
        }
        if is_synthetic_copy_key(want) {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k) || k.nvmap_id != want.nvmap_id {
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
        self.find_color_with_format(want)
            .filter(|(key, _, _, _, _)| !self.color_is_guest_stale(*key))
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
            let Some((src_x, src_y, exact)) =
                rt_region_offset(stored, img.format, width, height, gpu_va)
            else {
                continue;
            };
            let Some(stamp) = self.drawn_stamp.get(&stored).copied() else {
                continue;
            };
            let area = stored.width as u64 * stored.height as u64;
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
        for (k, img) in &self.cache {
            if is_synthetic_copy_key(*k) || k.nvmap_id != nvmap_id {
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
            img.layout = layout;
        }
    }

    pub fn color_layout(&self, key: RtKey) -> Option<vk::ImageLayout> {
        self.cache.get(&key).map(|img| img.layout)
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
            array_layers: 1,
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
        let mem_props = self
            .mem_properties
            .ok_or_else(|| "RtCache: memory properties not set".to_string())?;
        let mem_type = find_memory_type(
            &mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| "RtCache: no suitable DEVICE_LOCAL memory type".to_string())?;

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: req.size,
            memory_type_index: mem_type,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("allocate_memory: {:?}", e))?
        };
        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| format!("bind_image_memory: {:?}", e))?;
        }

        let view = create_image_view(device, image, format, aspect, key)?;
        let mut views = HashMap::new();
        views.insert(format, view);

        Ok(GpuImage {
            image,
            view,
            views,
            sample_views: HashMap::new(),
            memory,
            format,
            base_format: format,
            aspects: aspect,
            extent,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn clear(&mut self, device: &ash::Device) {
        self.clear_color_lookup_index();
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
            layer_count: 1,
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
        view_type: if key.is_3d {
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
        device.free_memory(image.memory, None);
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
    let image_size = (key.width as u64)
        .saturating_mul(key.height as u64)
        .saturating_mul(key.render_layer_count() as u64)
        .saturating_mul(rt_format_bytes(image.base_format));
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
    let size = (key.width as u64)
        .checked_mul(key.height as u64)?
        .checked_mul(key.render_layer_count() as u64)?
        .checked_mul(rt_format_bytes(format))?;
    let end = key.gpu_va.checked_add(size.checked_sub(1)?)?;
    Some((
        key.gpu_va >> RT_COLOR_GPU_PAGE_SHIFT,
        end >> RT_COLOR_GPU_PAGE_SHIFT,
    ))
}

fn remove_lookup_key(index: &mut HashMap<u64, Vec<RtKey>>, bucket: u64, key: RtKey) {
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

fn rt_format_bytes(format: vk::Format) -> u64 {
    rt_format_class_bits(format)
        .map(|bits| (bits / 8) as u64)
        .unwrap_or(4)
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
    use super::{
        is_synthetic_copy_key, rt_formats_compatible, same_d24_depth_allocation_covering,
        same_physical_backing, GpuImage, RtCache, RtColorRegion, RtKey, RtSampleViewKey,
    };
    use ash::vk;
    use std::collections::HashMap;

    fn test_depth_image() -> GpuImage {
        GpuImage {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            views: HashMap::new(),
            sample_views: HashMap::new(),
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

    fn test_color_image(format: vk::Format, base_format: vk::Format) -> GpuImage {
        GpuImage {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            views: HashMap::new(),
            sample_views: HashMap::new(),
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
        cache.depth_cache.insert(depth, depth_image);
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
    }

    #[test]
    fn color_recreation_query_matches_format_view_compatibility() {
        let key = RtKey::new(7, 640, 480, 0x5123_4000);
        let mut cache = RtCache::new();
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_UNORM));

        cache.insert_color_image(
            key,
            test_color_image(vk::Format::R8G8B8A8_UINT, vk::Format::R8G8B8A8_UNORM),
        );
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_UINT));
        assert!(!cache.color_requires_recreate(key, vk::Format::R8G8B8A8_SINT));
        assert!(cache.color_requires_recreate(key, vk::Format::R16G16B16A16_UNORM));

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
        cache.depth_cache.insert(source, test_depth_image());
        cache.depth_cache.insert(shadow, test_depth_image());

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
