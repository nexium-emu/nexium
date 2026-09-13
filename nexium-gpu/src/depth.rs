use ash::vk;

#[derive(Clone, Copy)]
pub(crate) struct DepthFormats {
    d24: vk::Format,
    d24s8: vk::Format,
}

impl Default for DepthFormats {
    fn default() -> Self {
        Self {
            d24: vk::Format::X8_D24_UNORM_PACK32,
            d24s8: vk::Format::D24_UNORM_S8_UINT,
        }
    }
}

impl DepthFormats {
    #[cfg(test)]
    pub(crate) fn float_fallback() -> Self {
        Self {
            d24: vk::Format::D32_SFLOAT,
            d24s8: vk::Format::D32_SFLOAT_S8_UINT,
        }
    }

    pub(crate) fn query(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, String> {
        Self::select(|format| unsafe {
            instance
                .get_physical_device_format_properties(physical_device, format)
                .optimal_tiling_features
        })
    }

    fn select(features: impl Fn(vk::Format) -> vk::FormatFeatureFlags) -> Result<Self, String> {
        let required = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        let select = |guest, fallback| {
            if features(guest).contains(required) {
                Ok(guest)
            } else if features(fallback).contains(required) {
                log::info!("Vulkan depth format fallback: {guest:?} -> {fallback:?}");
                Ok(fallback)
            } else {
                Err(format!("No supported host depth format for {guest:?}"))
            }
        };
        Ok(Self {
            d24: select(vk::Format::X8_D24_UNORM_PACK32, vk::Format::D32_SFLOAT)?,
            d24s8: select(
                vk::Format::D24_UNORM_S8_UINT,
                vk::Format::D32_SFLOAT_S8_UINT,
            )?,
        })
    }

    pub(crate) fn host(self, guest: vk::Format) -> vk::Format {
        match guest {
            vk::Format::X8_D24_UNORM_PACK32 => self.d24,
            vk::Format::D24_UNORM_S8_UINT => self.d24s8,
            _ => guest,
        }
    }
}

pub(crate) fn pack_d24(depth: f32) -> u32 {
    (f64::from(depth.clamp(0.0, 1.0)) * 16_777_215.0).round() as u32
}

pub(crate) fn pack_d32_readback(bytes: &mut [u8]) {
    for pixel in bytes.chunks_exact_mut(4) {
        let depth = f32::from_le_bytes(pixel.try_into().unwrap());
        pixel.copy_from_slice(&pack_d24(depth).to_le_bytes());
    }
}

pub(crate) fn unpack_texture_depth(bytes: &[u8], format: crate::texture::TicFormat) -> Vec<u8> {
    use crate::texture::TicFormat;
    if format == TicFormat::Z32 {
        return bytes.to_vec();
    }
    let high_depth = matches!(format, TicFormat::G24R8 | TicFormat::Z24S8);
    let mut out = Vec::with_capacity(bytes.len());
    for word in bytes.chunks_exact(4) {
        let packed = u32::from_le_bytes(word.try_into().unwrap());
        let depth = if high_depth {
            packed >> 8
        } else {
            packed & 0x00ff_ffff
        };
        out.extend_from_slice(&(depth as f32 / 16_777_215.0).to_le_bytes());
    }
    out
}

pub(crate) struct DepthPackPipeline {
    descriptor_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl DepthPackPipeline {
    pub(crate) fn new(device: &ash::Device) -> Result<Self, String> {
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
        }
        .map_err(|error| format!("depth pack descriptor layout: {error:?}"))?;
        let layouts = [descriptor_layout];
        let layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_layout, None) };
                return Err(format!("depth pack pipeline layout: {error:?}"));
            }
        };
        let mut result = Self {
            descriptor_layout,
            layout,
            pipeline: vk::Pipeline::null(),
        };
        let code: Vec<_> = include_bytes!(concat!(env!("OUT_DIR"), "/depth_pack_main.spv"))
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let module = match unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        } {
            Ok(module) => module,
            Err(error) => {
                result.destroy(device);
                return Err(format!("depth pack shader: {error:?}"));
            }
        };
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main"),
            )
            .layout(layout)];
        let pipelines =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { device.destroy_shader_module(module, None) };
        match pipelines {
            Ok(pipelines) => result.pipeline = pipelines[0],
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                result.destroy(device);
                return Err(format!("depth pack pipeline: {error:?}"));
            }
        }
        Ok(result)
    }

    pub(crate) fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pool: vk::DescriptorPool,
        buffer: vk::Buffer,
        bytes: u64,
    ) -> Result<vk::DescriptorSet, String> {
        if bytes == 0 || bytes % 4 != 0 {
            return Err("Invalid depth pack buffer size".into());
        }
        let layouts = [self.descriptor_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|error| format!("depth pack descriptor set: {error:?}"))?[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .range(bytes)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info)];
        let barrier = vk::BufferMemoryBarrier::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .size(bytes);
        unsafe {
            device.update_descriptor_sets(&write, &[]);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            let groups = (bytes / 4).div_ceil(64);
            let x = groups.min(65_535) as u32;
            device.cmd_dispatch(cmd, x, groups.div_ceil(u64::from(x)) as u32, 1);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
            );
        }
        Ok(set)
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_depth_preserves_low_bits_and_ignores_stencil() {
        use crate::texture::TicFormat;
        let values = [0, 1, 0x123456, 0x123457, 0x800000, 0xffffff];
        for format in [
            TicFormat::G24R8,
            TicFormat::Z24S8,
            TicFormat::X8Z24,
            TicFormat::S8Z24,
        ] {
            let high = matches!(format, TicFormat::G24R8 | TicFormat::Z24S8);
            for stencil in [0, 0xa5, 0xff] {
                let bytes: Vec<_> = values
                    .iter()
                    .flat_map(|&depth| {
                        let packed: u32 = if high {
                            depth << 8 | stencil
                        } else {
                            depth | stencil << 24
                        };
                        packed.to_le_bytes()
                    })
                    .collect();
                let unpacked = unpack_texture_depth(&bytes, format);
                let result: Vec<_> = unpacked
                    .chunks_exact(4)
                    .map(|word| pack_d24(f32::from_le_bytes(word.try_into().unwrap())))
                    .collect();
                assert_eq!(result, values, "{format:?}");
            }
        }
        let floats: Vec<_> = [0.0f32, 0.123456, 0.99999, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(unpack_texture_depth(&floats, TicFormat::Z32), floats);
    }

    #[test]
    fn depth_fallback_requires_all_image_usages() {
        let all = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        let native = DepthFormats::select(|_| all).unwrap();
        let fallback = DepthFormats::select(|format| {
            if matches!(
                format,
                vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT
            ) {
                all
            } else {
                all & !vk::FormatFeatureFlags::SAMPLED_IMAGE
            }
        })
        .unwrap();
        for (guest, host) in [
            (
                vk::Format::D24_UNORM_S8_UINT,
                vk::Format::D32_SFLOAT_S8_UINT,
            ),
            (vk::Format::X8_D24_UNORM_PACK32, vk::Format::D32_SFLOAT),
        ] {
            assert_eq!(native.host(guest), guest);
            assert_eq!(fallback.host(guest), host);
        }
        assert_eq!(fallback.host(vk::Format::D16_UNORM), vk::Format::D16_UNORM);
        assert!(DepthFormats::select(|_| vk::FormatFeatureFlags::empty()).is_err());
    }

    #[test]
    fn d24_readback_roundtrips_normalized_depth() {
        for value in (0..=0x00ff_ffff).step_by(127).chain([0x00ff_ffff]) {
            assert_eq!(pack_d24(value as f32 / 16_777_215.0), value);
        }
        let mut bytes: Vec<_> = [-1.0f32, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        pack_d32_readback(&mut bytes);
        let packed: Vec<_> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        assert_eq!(
            packed,
            [0, 0, 0x400000, 0x800000, 0xbfffff, 0xffffff, 0xffffff]
        );
    }
}
