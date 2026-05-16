use ash::vk;
use std::ffi::CStr;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
}

impl VulkanContext {
    pub fn new() -> Result<Self, String> {
        log::info!("Initializing Vulkan context");

        let entry = unsafe {
            ash::Entry::load()
        }.map_err(|_| "Failed to load Vulkan".to_string())?;

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: unsafe { CStr::from_bytes_with_nul_unchecked(b"HorizonRust\0").as_ptr() },
            application_version: 1,
            p_engine_name: unsafe { CStr::from_bytes_with_nul_unchecked(b"HorizonRust\0").as_ptr() },
            engine_version: 1,
            api_version: vk::API_VERSION_1_3,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let instance_create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let instance = unsafe {
            entry.create_instance(&instance_create_info, None)
                .map_err(|_| "Failed to create instance".to_string())?
        };

        log::debug!("Vulkan instance created");

        let physical_devices = unsafe {
            instance.enumerate_physical_devices()
                .map_err(|_| "Failed to enumerate devices".to_string())?
        };

        if physical_devices.is_empty() {
            return Err("No physical devices found".to_string());
        }

        let physical_device = physical_devices[0];
        log::debug!("Using physical device: {:?}", physical_device);

        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let graphics_queue_family = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family found".to_string())?;

        let queue_priority = 1.0;
        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index: graphics_queue_family as u32,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
                .map_err(|_| "Failed to create device".to_string())?
        };

        log::debug!("Vulkan device created");

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family as u32, 0) };

        log::info!("Vulkan context initialized successfully");

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family: graphics_queue_family as u32,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        log::debug!("Vulkan context destroyed");
    }
}
