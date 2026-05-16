use ash::vk;

pub struct CommandBuffer {
    pub handle: vk::CommandBuffer,
}

pub struct CommandRecorder {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl CommandRecorder {
    pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self, String> {
        let pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            queue_family_index: queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pool = unsafe {
            device.create_command_pool(&pool_info, None)
                .map_err(|_| "Failed to create command pool".to_string())?
        };

        Ok(Self {
            pool,
            buffers: Vec::new(),
        })
    }

    pub fn allocate_buffer(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, String> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: self.pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let buffers = unsafe {
            device.allocate_command_buffers(&alloc_info)
                .map_err(|_| "Failed to allocate command buffer".to_string())?
        };

        let buffer = buffers[0];
        self.buffers.push(buffer);
        Ok(buffer)
    }

    pub fn begin_recording(cmd_buf: vk::CommandBuffer, device: &ash::Device) -> Result<(), String> {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|_| "Failed to begin command buffer".to_string())?
        };

        Ok(())
    }

    pub fn end_recording(cmd_buf: vk::CommandBuffer, device: &ash::Device) -> Result<(), String> {
        unsafe {
            device.end_command_buffer(cmd_buf)
                .map_err(|_| "Failed to end command buffer".to_string())?
        };

        Ok(())
    }

    pub fn reset(&self, device: &ash::Device) -> Result<(), String> {
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES)
                .map_err(|_| "Failed to reset command pool".to_string())?
        };

        Ok(())
    }

    pub fn clear(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
        self.buffers.clear();
    }
}

impl Drop for CommandRecorder {
    fn drop(&mut self) {
        log::warn!("CommandRecorder dropped without explicit cleanup");
    }
}
