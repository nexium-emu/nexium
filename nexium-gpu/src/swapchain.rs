use ash::vk;

pub struct FrameData {
    pub image_index: u32,
}

pub struct Swapchain {
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        log::info!("Creating swapchain {}x{}", width, height);

        Self {
            images: Vec::new(),
            image_views: Vec::new(),
            format,
            extent: vk::Extent2D { width, height },
        }
    }

    pub fn begin_frame(&self) -> Result<FrameData, String> {
        if self.images.is_empty() {
            return Err("Swapchain images not initialized".to_string());
        }

        Ok(FrameData { image_index: 0 })
    }

    pub fn end_frame(&self, _frame: FrameData) -> Result<(), String> {
        Ok(())
    }

    pub fn clear(&mut self, device: &ash::Device) {
        unsafe {
            for view in &self.image_views {
                device.destroy_image_view(*view, None);
            }
        }
        self.images.clear();
        self.image_views.clear();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        log::warn!("Swapchain dropped without explicit cleanup");
    }
}
