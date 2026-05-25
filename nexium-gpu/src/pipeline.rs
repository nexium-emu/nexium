use ash::vk;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct PipelineKey {
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub topology: u32,
    pub color_format: u32,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub vertex_layout_hash: u64,
}

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    pub layout: vk::PipelineLayout,
}

impl PipelineCache {
    pub fn new(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, String> {
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("create_pipeline_layout: {:?}", e))?
        };

        Ok(Self {
            pipelines: HashMap::new(),
            layout,
        })
    }

    pub fn get(&self, key: &PipelineKey) -> Option<vk::Pipeline> {
        self.pipelines.get(key).copied()
    }

    pub fn insert(&mut self, key: PipelineKey, pipeline: vk::Pipeline) {
        self.pipelines.insert(key, pipeline);
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        if self.layout != vk::PipelineLayout::null() {
            unsafe {
                device.destroy_pipeline_layout(self.layout, None);
            }
            self.layout = vk::PipelineLayout::null();
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if !self.pipelines.is_empty() || self.layout != vk::PipelineLayout::null() {
            log::warn!("PipelineCache dropped without explicit cleanup");
        }
    }
}
