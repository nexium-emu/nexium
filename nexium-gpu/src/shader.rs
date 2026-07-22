use ash::vk;
use std::collections::HashMap;

pub struct ShaderCompiler {
    modules: HashMap<u64, vk::ShaderModule>,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn compile_or_get(
        &mut self,
        spirv: &[u32],
        device: &ash::Device,
    ) -> Result<vk::ShaderModule, String> {
        self.compile_or_get_labeled(spirv, device, "")
    }

    pub fn compile_or_get_labeled(
        &mut self,
        spirv: &[u32],
        device: &ash::Device,
        label: &str,
    ) -> Result<vk::ShaderModule, String> {
        let hash = Self::hash_spirv(spirv);

        if let Some(&module) = self.modules.get(&hash) {
            return Ok(module);
        }

        Self::validate_module_spirv(spirv, label, hash)?;

        if std::env::var_os("NEXIUM_SHADER_MODULE_DBG").is_some() {
            log::warn!(
                "[shader-module] create {} hash={:016x} words={}",
                label,
                hash,
                spirv.len()
            );
        }

        let module_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let module = unsafe {
            device
                .create_shader_module(&module_info, None)
                .map_err(|e| {
                    format!(
                        "Failed to create shader module {} hash={:016x}: {:?}",
                        label, hash, e
                    )
                })?
        };

        self.modules.insert(hash, module);
        Ok(module)
    }

    fn validate_module_spirv(spirv: &[u32], label: &str, hash: u64) -> Result<(), String> {
        if nexium_spirv::phi_preds_consistent(spirv) {
            return Ok(());
        }
        Err(format!(
            "Rejected shader module {} hash={:016x}: invalid SPIR-V phi predecessors",
            label, hash
        ))
    }

    fn hash_spirv(spirv: &[u32]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for word in spirv {
            hash ^= *word as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn clear(&mut self, device: &ash::Device) {
        for (_, module) in self.modules.drain() {
            unsafe {
                device.destroy_shader_module(module, None);
            }
        }
    }
}

impl Drop for ShaderCompiler {
    fn drop(&mut self) {
        if !self.modules.is_empty() {
            log::warn!("ShaderCompiler dropped without explicit cleanup");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderCompiler;

    #[test]
    fn malformed_spirv_is_rejected_before_module_creation() {
        let words = [0];
        let hash = ShaderCompiler::hash_spirv(&words);
        let error = ShaderCompiler::validate_module_spirv(&words, "cached-fs", hash)
            .expect_err("malformed SPIR-V must be rejected");
        assert!(error.contains("invalid SPIR-V phi predecessors"));
    }
}
