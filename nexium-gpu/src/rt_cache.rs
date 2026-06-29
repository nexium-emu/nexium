use ash::vk;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct RtKey {
    pub nvmap_id: u32,
    pub width: u32,
    pub height: u32,
}

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

fn dims_close(a: u32, b: u32) -> bool {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    lo != 0 && hi <= lo.saturating_mul(2)
}

pub struct RtCache {
    cache: HashMap<RtKey, GpuImage>,
    depth_cache: HashMap<RtKey, GpuImage>,
    mem_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    drawn_stamp: HashMap<RtKey, u64>,
    drawn_counter: u64,
    frame_draws: HashMap<RtKey, u32>,
}

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            depth_cache: HashMap::new(),
            mem_properties: None,
            drawn_stamp: HashMap::new(),
            drawn_counter: 0,
            frame_draws: HashMap::new(),
        }
    }

    pub fn mark_drawn(&mut self, key: RtKey) {
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
        *self.frame_draws.entry(key).or_insert(0) += 1;
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
        let mut best: Option<(RtKey, u64)> = None;
        for k in self.cache.keys() {
            if k.width != want.width || k.height != want.height {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else {
                continue;
            };
            let replace = match best {
                Some((best_key, best_stamp)) => {
                    stamp > best_stamp || (stamp == best_stamp && *k == want && best_key != want)
                }
                None => true,
            };
            if replace {
                best = Some((*k, stamp));
            }
        }
        let best_stamp = best.map(|(_, s)| s).unwrap_or(0);
        if best_stamp <= 2 && want.height != 0 {
            let aw = want.width as f32 / want.height as f32;
            let same_aspect = |k: &RtKey| {
                k.height != 0 && ((k.width as f32 / k.height as f32) - aw).abs() <= aw * 0.12
            };
            let by_draws = self
                .cache
                .keys()
                .filter(|k| same_aspect(k))
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
            let alt = self
                .cache
                .keys()
                .filter(|k| same_aspect(k))
                .filter_map(|k| self.drawn_stamp.get(k).map(|s| (*k, *s)))
                .max_by_key(|(_, s)| *s);
            if let Some((ak, astamp)) = alt {
                if astamp > best_stamp.saturating_mul(16) {
                    return Some(ak);
                }
            }
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
                            "{}:{}x{}#{}",
                            k.nvmap_id,
                            k.width,
                            k.height,
                            self.drawn_stamp.get(k).copied().unwrap_or(0)
                        )
                    })
                    .collect();
                keys.sort();
                log::warn!(
                    "[present-none #{}] want={}:{}x{} cache=[{}]",
                    n,
                    want.nvmap_id,
                    want.width,
                    want.height,
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
            if k.width != want.width || k.height != want.height {
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
        best.map(|(k, img, _)| (k, img.image, img.view, img.layout))
    }

    pub fn set_mem_properties(&mut self, props: vk::PhysicalDeviceMemoryProperties) {
        self.mem_properties = Some(props);
    }

    pub fn get_or_create(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<&mut GpuImage, String> {
        if !self.cache.contains_key(&key) {
            let image = self.create_image(device, key)?;
            self.cache.insert(key, image);
        }
        Ok(self.cache.get_mut(&key).unwrap())
    }

    pub fn get_or_create_depth(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<&mut GpuImage, String> {
        if !self.depth_cache.contains_key(&key) {
            let image = self.create_image_inner(
                device,
                key,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::DEPTH,
            )?;
            self.depth_cache.insert(key, image);
        }
        Ok(self.depth_cache.get_mut(&key).unwrap())
    }

    pub fn find_color(
        &self,
        want: RtKey,
    ) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        if let Some(img) = self.cache.get(&want) {
            return Some((want, img.image, img.view, img.layout));
        }
        let mut best: Option<(RtKey, &GpuImage)> = None;
        for (k, img) in &self.cache {
            if k.nvmap_id != want.nvmap_id {
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

    pub fn set_color_layout(&mut self, key: RtKey, layout: vk::ImageLayout) {
        if let Some(img) = self.cache.get_mut(&key) {
            img.layout = layout;
        }
    }

    fn create_image(&self, device: &ash::Device, key: RtKey) -> Result<GpuImage, String> {
        self.create_image_inner(
            device,
            key,
            vk::Format::R8G8B8A8_UNORM,
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
            flags: Default::default(),
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
        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .map_err(|e| format!("create_image_view: {:?}", e))?
        };

        Ok(GpuImage {
            image,
            view,
            memory,
            format,
            extent,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, img) in self.cache.drain().chain(self.depth_cache.drain()) {
            unsafe {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
                device.free_memory(img.memory, None);
            }
        }
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
