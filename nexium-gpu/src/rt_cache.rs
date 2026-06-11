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
}

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            depth_cache: HashMap::new(),
            mem_properties: None,
            drawn_stamp: HashMap::new(),
            drawn_counter: 0,
        }
    }

    pub fn mark_drawn(&mut self, key: RtKey) {
        self.drawn_counter += 1;
        self.drawn_stamp.insert(key, self.drawn_counter);
    }

    pub fn find_color_screen(&self, want: RtKey) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
        let mut best: Option<(RtKey, &GpuImage, u64)> = None;
        for (k, img) in &self.cache {
            if k.nvmap_id == want.nvmap_id || k.width != want.width || k.height != want.height {
                continue;
            }
            let Some(stamp) = self.drawn_stamp.get(k).copied() else { continue };
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

    pub fn find_color(&self, want: RtKey) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout)> {
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
        let extent = vk::Extent2D { width: key.width, height: key.height };

        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
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
            device.create_image(&image_info, None)
                .map_err(|e| format!("create_image: {:?}", e))?
        };

        let req = unsafe { device.get_image_memory_requirements(image) };
        let mem_props = self.mem_properties
            .ok_or_else(|| "RtCache: memory properties not set".to_string())?;
        let mem_type = find_memory_type(
            &mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).ok_or_else(|| "RtCache: no suitable DEVICE_LOCAL memory type".to_string())?;

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: req.size,
            memory_type_index: mem_type,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let memory = unsafe {
            device.allocate_memory(&alloc_info, None)
                .map_err(|e| format!("allocate_memory: {:?}", e))?
        };
        unsafe {
            device.bind_image_memory(image, memory, 0)
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
            device.create_image_view(&view_info, None)
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
            log::warn!("RtCache dropped with {} images still cached", self.cache.len());
        }
    }
}
