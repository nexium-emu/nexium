use ash::vk;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug)]
pub struct RtKey {
    pub nvmap_id: u32,
    pub width: u32,
    pub height: u32,
    pub gpu_va: u64,
    pub cpu_addr: u64,
}

impl PartialEq for RtKey {
    fn eq(&self, other: &Self) -> bool {
        self.nvmap_id == other.nvmap_id
            && self.width == other.width
            && self.height == other.height
            && self.gpu_va == other.gpu_va
    }
}

impl Eq for RtKey {}

impl Hash for RtKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nvmap_id.hash(state);
        self.width.hash(state);
        self.height.hash(state);
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
            gpu_va,
            cpu_addr,
        }
    }

    pub fn request(nvmap_id: u32, width: u32, height: u32) -> Self {
        Self::new(nvmap_id, width, height, 0)
    }

    pub fn label(self) -> String {
        if self.gpu_va != 0 {
            format!(
                "{}:{}x{}@{:x}",
                self.nvmap_id, self.width, self.height, self.gpu_va
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
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub base_format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
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

pub struct RtCache {
    cache: HashMap<RtKey, GpuImage>,
    depth_cache: HashMap<RtKey, GpuImage>,
    snapshots: HashMap<RtKey, GpuImage>,
    mem_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    drawn_stamp: HashMap<RtKey, u64>,
    present_excluded: HashSet<RtKey>,
    present_flip_y: HashMap<RtKey, bool>,
    drawn_counter: u64,
    frame_draws: HashMap<RtKey, u32>,
}

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            depth_cache: HashMap::new(),
            snapshots: HashMap::new(),
            mem_properties: None,
            drawn_stamp: HashMap::new(),
            present_excluded: HashSet::new(),
            present_flip_y: HashMap::new(),
            drawn_counter: 0,
            frame_draws: HashMap::new(),
        }
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

    pub fn mark_drawn(&mut self, key: RtKey) -> u64 {
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
        self.present_excluded.remove(&key);
        *self.frame_draws.entry(key).or_insert(0) += 1;
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
        self.present_excluded.insert(key);
        self.drawn_counter
    }

    pub fn reset_frame_draws(&mut self) {
        self.frame_draws.clear();
    }

    pub fn mark_cleared(&mut self, key: RtKey, full_target: bool) {
        if full_target {
            self.drawn_stamp.remove(&key);
        } else if self.drawn_stamp.contains_key(&key) {
            self.mark_drawn(key);
        }
    }

    pub fn resolve_present_key(&self, want: RtKey) -> Option<RtKey> {
        let choose_cpu = || -> Option<(RtKey, u64)> {
            if want.cpu_addr == 0 {
                return None;
            }
            let mut best: Option<(RtKey, u64)> = None;
            for k in self.cache.keys() {
                if k.width != want.width
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
                if k.width != want.width || k.height != want.height {
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
        let strict_cpu = std::env::var_os("NEXIUM_PRESENT_STRICT_CPU").is_some();
        if want.height != 0 && want.gpu_va == 0 && (want.cpu_addr == 0 || !strict_cpu) {
            let aw = want.width as f32 / want.height as f32;
            let same_aspect = |k: &RtKey| {
                k.height != 0 && ((k.width as f32 / k.height as f32) - aw).abs() <= aw * 0.12
            };
            let alt = self
                .cache
                .keys()
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
        for k in self.cache.keys() {
            if self.present_excluded.contains(k) {
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
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for (k, img) in &self.cache {
            if k.nvmap_id == want.nvmap_id || k.width != want.width || k.height != want.height {
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
                if k.nvmap_id == want.nvmap_id || k.height == 0 || k.width < want.width {
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
        self.cache
            .keys()
            .any(|k| k.nvmap_id == nvmap_id && k.gpu_va == gpu_va && self.drawn_stamp.contains_key(k))
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

    pub fn find_color_key_at_va(&self, nvmap_id: u32, gpu_va: u64) -> Option<RtKey> {
        self.cache
            .keys()
            .filter(|k| k.nvmap_id == nvmap_id && k.gpu_va == gpu_va)
            .max_by_key(|k| self.drawn_stamp.get(*k).copied().unwrap_or(0))
            .copied()
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
        if key.gpu_va == 0 {
            return out;
        }
        for (k, img) in &self.cache {
            if *k == key || k.nvmap_id != key.nvmap_id || k.gpu_va != key.gpu_va {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            out.push((*k, img.image, img.layout, img.format, stamp));
        }
        out.sort_by_key(|(_, _, _, _, stamp)| *stamp);
        out
    }

    pub fn get_or_create_with_format(
        &mut self,
        key: RtKey,
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<&mut GpuImage, String> {
        let existing = self
            .cache
            .get(&key)
            .map(|image| (image.format, image.base_format));
        match existing {
            Some((current, _)) if current == format => {
                return Ok(self.cache.get_mut(&key).unwrap());
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
                    )?;
                    image.views.insert(format, view);
                }
                image.view = image.views[&format];
                image.format = format;
                return Ok(image);
            }
            Some(_) => {}
            None => {}
        }
        if existing.is_some() {
            if let Some(image) = self.cache.remove(&key) {
                if rt_format_dbg_enabled(key.nvmap_id) {
                    log::warn!(
                        "[rt-format-churn] key={} {:?}->{:?} stamp={}",
                        key.label(),
                        image.format,
                        format,
                        self.drawn_stamp.get(&key).copied().unwrap_or(0)
                    );
                }
                destroy_gpu_image(device, image);
            }
            self.drawn_stamp.remove(&key);
            self.present_excluded.remove(&key);
            self.present_flip_y.remove(&key);
            self.frame_draws.remove(&key);
        }
        if !self.cache.contains_key(&key) {
            let image = self.create_image(device, key, format)?;
            self.cache.insert(key, image);
        }
        Ok(self.cache.get_mut(&key).unwrap())
    }

    pub fn get_or_create_depth(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<(&mut GpuImage, bool), String> {
        let created = !self.depth_cache.contains_key(&key);
        if created {
            let image = self.create_image_inner(
                device,
                key,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::DEPTH,
            )?;
            self.depth_cache.insert(key, image);
        }
        Ok((self.depth_cache.get_mut(&key).unwrap(), created))
    }

    pub fn find_depth(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        if let Some(img) = self.depth_cache.get(&want) {
            return Some((want, img.image, img.view, img.layout));
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
        best.map(|(k, img)| (k, img.image, img.view, img.layout))
    }

    pub fn find_depth_fuzzy(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
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
        best.map(|(k, img)| (k, img.image, img.view, img.layout))
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
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.cache {
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
        best.map(|(k, img)| (k, img.image, img.view, img.layout, img.format))
    }

    pub fn find_drawn_color_at(
        &self,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(RtKey, vk::Image, vk::ImageLayout, vk::Format, u64)> {
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for (k, img) in &self.cache {
            if k.gpu_va != gpu_va || k.width < width || k.height < height {
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
            if k.nvmap_id != nvmap_id
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
        if gpu_va == 0 {
            return None;
        }
        let mut best: Option<(RtKey, &GpuImage, u64, u32, u32, bool)> = None;
        for (k, img) in &self.cache {
            let Some((src_x, src_y, exact)) =
                rt_region_offset(*k, img.format, width, height, gpu_va)
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
            if k.nvmap_id != nvmap_id {
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
        if let Some(img) = self.depth_cache.get_mut(&key) {
            img.layout = layout;
        }
    }

    pub fn depth_layout(&self, key: RtKey) -> Option<vk::ImageLayout> {
        self.depth_cache.get(&key).map(|img| img.layout)
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
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            p_next: std::ptr::null(),
            flags: if aspect.contains(vk::ImageAspectFlags::COLOR) {
                vk::ImageCreateFlags::MUTABLE_FORMAT
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

        let view = create_image_view(device, image, format, aspect)?;
        let mut views = HashMap::new();
        views.insert(format, view);

        Ok(GpuImage {
            image,
            view,
            views,
            memory,
            format,
            base_format: format,
            extent,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, img) in self
            .cache
            .drain()
            .chain(self.depth_cache.drain())
            .chain(self.snapshots.drain())
        {
            destroy_gpu_image(device, img);
        }
    }
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, String> {
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
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
        for view in image.views.into_values() {
            device.destroy_image_view(view, None);
        }
        device.destroy_image(image.image, None);
        device.free_memory(image.memory, None);
    }
}

fn rt_formats_compatible(base: vk::Format, view: vk::Format) -> bool {
    base == view
        || rt_format_class_bits(base)
            .zip(rt_format_class_bits(view))
            .is_some_and(|(a, b)| a == b)
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
