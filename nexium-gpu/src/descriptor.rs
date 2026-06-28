use ash::vk;

pub const MAX_TEXTURE_DESCRIPTORS: u32 = 32;
pub const SSBO_BINDING_BASE: u32 = 3;
pub const MAX_SSBO: u32 = 8;

pub struct DescriptorSetLayout {
    pub layout: vk::DescriptorSetLayout,
}

pub struct DescriptorPool {
    pub pool: vk::DescriptorPool,
}

impl DescriptorSetLayout {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: MAX_TEXTURE_DESCRIPTORS,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: MAX_TEXTURE_DESCRIPTORS,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
        ];
        for i in 0..MAX_SSBO {
            bindings.push(vk::DescriptorSetLayoutBinding {
                binding: SSBO_BINDING_BASE + i,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|_| "Failed to create descriptor set layout".to_string())?
        };

        Ok(Self { layout })
    }
}

impl DescriptorPool {
    pub fn new(device: &ash::Device, max_sets: u32) -> Result<Self, String> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: max_sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: max_sets.saturating_mul(MAX_TEXTURE_DESCRIPTORS),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: max_sets.saturating_mul(MAX_TEXTURE_DESCRIPTORS),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_sets.saturating_mul(MAX_SSBO),
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|_| "Failed to create descriptor pool".to_string())?
        };

        Ok(Self { pool })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if self.layout != vk::DescriptorSetLayout::null() {
            log::warn!("DescriptorSetLayout dropped without explicit cleanup");
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        if self.pool != vk::DescriptorPool::null() {
            log::warn!("DescriptorPool dropped without explicit cleanup");
        }
    }
}
