use ash::vk;

pub use nexium_spirv::{
    GFX_BINDING_CBUF as CBUF_BINDING, GFX_BINDING_FLOAT_2D as IMAGE2D_BINDING,
    GFX_BINDING_FLOAT_3D as IMAGE3D_BINDING, GFX_BINDING_FLOAT_CUBE as IMAGE_CUBE_BINDING,
    GFX_BINDING_FLOAT_CUBE_ARRAY as IMAGE_CUBE_ARRAY_BINDING,
    GFX_BINDING_FLOAT_TEXEL_BUFFER as TEXEL_BUFFER_BINDING,
    GFX_BINDING_SAMPLERS as SAMPLER_BINDING, GFX_BINDING_SINT_2D as SINT_IMAGE2D_BINDING,
    GFX_BINDING_SINT_3D as SINT_IMAGE3D_BINDING, GFX_BINDING_SINT_CUBE as SINT_IMAGE_CUBE_BINDING,
    GFX_BINDING_SINT_CUBE_ARRAY as SINT_IMAGE_CUBE_ARRAY_BINDING,
    GFX_BINDING_SINT_TEXEL_BUFFER as SINT_TEXEL_BUFFER_BINDING,
    GFX_BINDING_SSBO_BASE as SSBO_BINDING_BASE, GFX_BINDING_UINT_2D as UINT_IMAGE2D_BINDING,
    GFX_BINDING_UINT_3D as UINT_IMAGE3D_BINDING, GFX_BINDING_UINT_CUBE as UINT_IMAGE_CUBE_BINDING,
    GFX_BINDING_UINT_CUBE_ARRAY as UINT_IMAGE_CUBE_ARRAY_BINDING,
    GFX_BINDING_UINT_TEXEL_BUFFER as UINT_TEXEL_BUFFER_BINDING,
};

pub const MAX_TEXTURE_DESCRIPTORS: u32 = 32;
pub const MAX_SSBO: u32 = 8;

pub const SAMPLED_IMAGE_BINDINGS: [u32; 12] = [
    IMAGE2D_BINDING,
    IMAGE3D_BINDING,
    IMAGE_CUBE_BINDING,
    IMAGE_CUBE_ARRAY_BINDING,
    UINT_IMAGE2D_BINDING,
    UINT_IMAGE3D_BINDING,
    UINT_IMAGE_CUBE_BINDING,
    UINT_IMAGE_CUBE_ARRAY_BINDING,
    SINT_IMAGE2D_BINDING,
    SINT_IMAGE3D_BINDING,
    SINT_IMAGE_CUBE_BINDING,
    SINT_IMAGE_CUBE_ARRAY_BINDING,
];

pub const TEXEL_BUFFER_BINDINGS: [u32; 3] = [
    TEXEL_BUFFER_BINDING,
    UINT_TEXEL_BUFFER_BINDING,
    SINT_TEXEL_BUFFER_BINDING,
];

pub struct DescriptorSetLayout {
    pub layout: vk::DescriptorSetLayout,
    pub texture_arrays_partially_bound: bool,
}

pub struct DescriptorPool {
    pub pool: vk::DescriptorPool,
}

fn graphics_layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding {
            binding: CBUF_BINDING,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        },
        vk::DescriptorSetLayoutBinding {
            binding: IMAGE2D_BINDING,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: MAX_TEXTURE_DESCRIPTORS,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        },
        vk::DescriptorSetLayoutBinding {
            binding: SAMPLER_BINDING,
            descriptor_type: vk::DescriptorType::SAMPLER,
            descriptor_count: MAX_TEXTURE_DESCRIPTORS,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        },
    ];
    for i in 0..MAX_SSBO {
        bindings.push(vk::DescriptorSetLayoutBinding {
            binding: SSBO_BINDING_BASE + i,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
    }
    for binding in SAMPLED_IMAGE_BINDINGS.into_iter().skip(1) {
        bindings.push(vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: MAX_TEXTURE_DESCRIPTORS,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
    }
    for binding in TEXEL_BUFFER_BINDINGS {
        bindings.push(vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            descriptor_count: MAX_TEXTURE_DESCRIPTORS,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::GEOMETRY,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
    }
    bindings.sort_unstable_by_key(|binding| binding.binding);
    bindings
}

fn graphics_layout_binding_flags(
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    texture_arrays_partially_bound: bool,
) -> Vec<vk::DescriptorBindingFlags> {
    bindings
        .iter()
        .map(|binding| {
            if texture_arrays_partially_bound
                && binding.descriptor_count == MAX_TEXTURE_DESCRIPTORS
                && matches!(
                    binding.descriptor_type,
                    vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                )
            {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
            } else {
                vk::DescriptorBindingFlags::empty()
            }
        })
        .collect()
}

fn graphics_pool_sizes(max_sets: u32) -> [vk::DescriptorPoolSize; 4] {
    [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: max_sets.saturating_mul(MAX_TEXTURE_DESCRIPTORS * 12),
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: max_sets.saturating_mul(MAX_TEXTURE_DESCRIPTORS),
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_sets.saturating_mul(MAX_SSBO + 1),
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            descriptor_count: max_sets.saturating_mul(MAX_TEXTURE_DESCRIPTORS * 3),
        },
    ]
}

#[cfg(test)]
fn graphics_write_specs() -> Vec<(u32, vk::DescriptorType, u32)> {
    let mut specs = vec![
        (CBUF_BINDING, vk::DescriptorType::STORAGE_BUFFER, 1),
        (
            SAMPLER_BINDING,
            vk::DescriptorType::SAMPLER,
            MAX_TEXTURE_DESCRIPTORS,
        ),
    ];
    specs.extend((0..MAX_SSBO).map(|slot| {
        (
            SSBO_BINDING_BASE + slot,
            vk::DescriptorType::STORAGE_BUFFER,
            1,
        )
    }));
    specs.extend(SAMPLED_IMAGE_BINDINGS.map(|binding| {
        (
            binding,
            vk::DescriptorType::SAMPLED_IMAGE,
            MAX_TEXTURE_DESCRIPTORS,
        )
    }));
    specs.extend(TEXEL_BUFFER_BINDINGS.map(|binding| {
        (
            binding,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            MAX_TEXTURE_DESCRIPTORS,
        )
    }));
    specs.sort_unstable_by_key(|spec| spec.0);
    specs
}

impl DescriptorSetLayout {
    pub fn new(device: &ash::Device, texture_arrays_partially_bound: bool) -> Result<Self, String> {
        let bindings = graphics_layout_bindings();
        let binding_flags =
            graphics_layout_binding_flags(&bindings, texture_arrays_partially_bound);
        let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            p_next: std::ptr::null(),
            binding_count: binding_flags.len() as u32,
            p_binding_flags: binding_flags.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            p_next: if texture_arrays_partially_bound {
                &binding_flags_info as *const _ as *const std::ffi::c_void
            } else {
                std::ptr::null()
            },
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|_| "Failed to create descriptor set layout".to_string())?
        };

        Ok(Self {
            layout,
            texture_arrays_partially_bound,
        })
    }
}

impl DescriptorPool {
    pub fn new(device: &ash::Device, max_sets: u32) -> Result<Self, String> {
        let pool_sizes = graphics_pool_sizes(max_sets);

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

    pub fn into_raw(mut self) -> vk::DescriptorPool {
        let pool = self.pool;
        self.pool = vk::DescriptorPool::null();
        pool
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphics_typed_image_bindings_match_the_public_abi() {
        assert_eq!(
            SAMPLED_IMAGE_BINDINGS,
            [1, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23]
        );
        assert_eq!(TEXEL_BUFFER_BINDINGS, [14, 19, 24]);

        let bindings = SAMPLED_IMAGE_BINDINGS
            .into_iter()
            .chain(TEXEL_BUFFER_BINDINGS);
        let unique = bindings.clone().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), bindings.count());
        assert!(unique.iter().all(|binding| {
            *binding != SAMPLER_BINDING
                && !(*binding >= SSBO_BINDING_BASE && *binding < SSBO_BINDING_BASE + MAX_SSBO)
        }));

        let layout = graphics_layout_bindings();
        assert_eq!(layout.len(), 25);
        assert_eq!(
            layout
                .iter()
                .map(|binding| binding.binding)
                .collect::<Vec<_>>(),
            (0..=24).collect::<Vec<_>>()
        );
        assert_eq!(
            layout
                .iter()
                .filter(|binding| binding.descriptor_type == vk::DescriptorType::SAMPLED_IMAGE)
                .count(),
            12
        );
        assert_eq!(
            layout
                .iter()
                .filter(|binding| {
                    binding.descriptor_type == vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                })
                .count(),
            3
        );
        let cbuf = &layout[CBUF_BINDING as usize];
        assert_eq!(cbuf.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(cbuf.descriptor_count, 1);

        let pool = graphics_pool_sizes(7);
        assert_eq!(pool.len(), 4);
        let count = |ty| {
            pool.iter()
                .find(|size| size.ty == ty)
                .map(|size| size.descriptor_count)
        };
        assert_eq!(count(vk::DescriptorType::SAMPLED_IMAGE), Some(7 * 32 * 12));
        assert_eq!(
            count(vk::DescriptorType::UNIFORM_TEXEL_BUFFER),
            Some(7 * 32 * 3)
        );
        assert_eq!(count(vk::DescriptorType::SAMPLER), Some(7 * 32));
        assert_eq!(count(vk::DescriptorType::STORAGE_BUFFER), Some(7 * 9));
        assert_eq!(count(vk::DescriptorType::UNIFORM_BUFFER), None);

        let writes = graphics_write_specs();
        assert_eq!(writes.len(), 25);
        assert_eq!(
            writes.iter().map(|spec| spec.0).collect::<Vec<_>>(),
            (0..=24).collect::<Vec<_>>()
        );
        assert_eq!(
            writes[CBUF_BINDING as usize],
            (CBUF_BINDING, vk::DescriptorType::STORAGE_BUFFER, 1)
        );
        for (layout, write) in layout.iter().zip(writes) {
            assert_eq!(layout.binding, write.0);
            assert_eq!(layout.descriptor_type, write.1);
            assert_eq!(layout.descriptor_count, write.2);
        }

        let binding_flags = graphics_layout_binding_flags(&layout, true);
        assert_eq!(binding_flags.len(), layout.len());
        assert_eq!(
            binding_flags
                .iter()
                .filter(|flags| flags.contains(vk::DescriptorBindingFlags::PARTIALLY_BOUND))
                .count(),
            16
        );
        for (binding, flags) in layout.iter().zip(binding_flags) {
            let texture_array = binding.descriptor_count == MAX_TEXTURE_DESCRIPTORS
                && matches!(
                    binding.descriptor_type,
                    vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                );
            assert_eq!(
                flags.contains(vk::DescriptorBindingFlags::PARTIALLY_BOUND),
                texture_array,
                "unexpected descriptor binding flags for binding {}",
                binding.binding
            );
        }
        assert!(graphics_layout_binding_flags(&layout, false)
            .iter()
            .all(|flags| flags.is_empty()));
    }
}
