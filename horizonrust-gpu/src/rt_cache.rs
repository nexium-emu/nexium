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
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

pub struct RtCache {
    cache: HashMap<RtKey, GpuImage>,
}

impl RtCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get_or_create(
        &mut self,
        key: RtKey,
        device: &ash::Device,
    ) -> Result<&GpuImage, String> {
        if !self.cache.contains_key(&key) {
            let image = self.create_image(device, key)?;
            self.cache.insert(key, image);
        }
        Ok(self.cache.get(&key).unwrap())
    }

    fn create_image(
        &self,
        device: &ash::Device,
        key: RtKey,
    ) -> Result<GpuImage, String> {
        let extent = vk::Extent2D {
            width: key.width,
            height: key.height,
        };

        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            p_next: std::ptr::null(),
            flags: Default::default(),
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let image = unsafe {
            device.create_image(&image_create_info, None)
                .map_err(|_| "Failed to create image".to_string())?
        };

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let view_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            subresource_range,
            components: vk::ComponentMapping::default(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let view = unsafe {
            device.create_image_view(&view_create_info, None)
                .map_err(|_| "Failed to create image view".to_string())?
        };

        Ok(GpuImage {
            image,
            view,
            format: vk::Format::R8G8B8A8_UNORM,
            extent,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, img) in self.cache.drain() {
            unsafe {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
            }
        }
    }
}

impl Drop for RtCache {
    fn drop(&mut self) {
        log::warn!("RtCache dropped without explicit cleanup");
    }
}
