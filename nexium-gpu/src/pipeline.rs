use ash::vk;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct PipelineKey {
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub topology: u32,
    pub color_format: u32,
}

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    layout: vk::PipelineLayout,
}

impl PipelineCache {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 0,
            p_set_layouts: std::ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|_| "Failed to create pipeline layout".to_string())?
        };

        Ok(Self {
            pipelines: HashMap::new(),
            layout,
        })
    }

    pub fn get_or_create(
        &mut self,
        key: PipelineKey,
        _device: &ash::Device,
    ) -> Result<vk::Pipeline, String> {
        if !self.pipelines.contains_key(&key) {
            return Err("Pipeline compilation not yet implemented".to_string());
        }
        Ok(*self.pipelines.get(&key).unwrap())
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        log::warn!("PipelineCache dropped without explicit cleanup");
    }
}
