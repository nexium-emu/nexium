use ash::vk;
use parking_lot::RwLock;
use std::sync::OnceLock;

static PREFERRED_DEVICE: RwLock<Option<String>> = RwLock::new(None);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub id: String,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
}

impl AdapterInfo {
    pub fn label(&self) -> String {
        let kind = match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "dGPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "iGPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
            vk::PhysicalDeviceType::CPU => "Software",
            _ => "GPU",
        };
        format!("{} ({kind})", self.name)
    }
}

pub fn set_preferred_device(id: Option<String>) {
    *PREFERRED_DEVICE.write() = id;
}

pub(crate) fn preferred_device() -> Option<String> {
    PREFERRED_DEVICE.read().clone()
}

pub fn available_devices() -> &'static [AdapterInfo] {
    static DEVICES: OnceLock<Vec<AdapterInfo>> = OnceLock::new();
    DEVICES.get_or_init(|| match enumerate_devices() {
        Ok(devices) => devices,
        Err(error) => {
            log::warn!("GPU discovery failed: {error}");
            Vec::new()
        }
    })
}

#[cfg(target_vendor = "sony")]
unsafe extern "system" {
    fn vk_icdGetInstanceProcAddr(instance: vk::Instance, name: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction;
}

pub fn vulkan_entry() -> Result<ash::Entry, String> {
    #[cfg(target_vendor = "sony")]
    {
        Ok(unsafe { ash::Entry::from_static_fn(ash::StaticFn { get_instance_proc_addr: vk_icdGetInstanceProcAddr }) })
    }
    #[cfg(not(target_vendor = "sony"))]
    {
        unsafe { ash::Entry::load() }.map_err(|error| error.to_string())
    }
}

fn enumerate_devices() -> Result<Vec<AdapterInfo>, String> {
    let entry = vulkan_entry()?;
    let app = vk::ApplicationInfo::default()
        .application_name(c"NeXium GPU discovery")
        .api_version(vk::API_VERSION_1_3);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .map_err(|error| format!("create instance: {error:?}"))?;
    let result = physical_devices(&instance)
        .map(|devices| devices.into_iter().map(|(_, info)| info).collect());
    unsafe { instance.destroy_instance(None) };
    result
}

pub(crate) fn physical_devices(
    instance: &ash::Instance,
) -> Result<Vec<(vk::PhysicalDevice, AdapterInfo)>, String> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| format!("enumerate physical devices: {error:?}"))?;
    let mut available = Vec::new();
    for device in devices {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe { instance.get_physical_device_properties2(device, &mut properties) };
        let properties = properties.properties;
        let queues = unsafe { instance.get_physical_device_queue_family_properties(device) };
        if properties.api_version < vk::API_VERSION_1_3
            || !queues
                .iter()
                .any(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        {
            continue;
        }
        available.push((
            device,
            AdapterInfo {
                id: id
                    .device_uuid
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
                name: unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                device_type: properties.device_type,
            },
        ));
    }
    Ok(available)
}

pub(crate) fn select_index(devices: &[AdapterInfo], preferred: Option<&str>) -> Option<usize> {
    preferred
        .and_then(|id| devices.iter().position(|device| device.id == id))
        .or_else(|| {
            devices
                .iter()
                .position(|device| device.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        })
        .or_else(|| (!devices.is_empty()).then_some(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(id: &str, device_type: vk::PhysicalDeviceType) -> AdapterInfo {
        AdapterInfo {
            id: id.into(),
            name: "GPU".into(),
            vendor_id: 0,
            device_id: 0,
            device_type,
        }
    }

    #[test]
    fn selection_uses_identity_across_enumeration_order() {
        let integrated = device("integrated", vk::PhysicalDeviceType::INTEGRATED_GPU);
        let discrete = device("discrete", vk::PhysicalDeviceType::DISCRETE_GPU);
        for devices in [
            [integrated.clone(), discrete.clone()],
            [discrete, integrated],
        ] {
            let index = select_index(&devices, Some("integrated")).unwrap();
            assert_eq!(devices[index].id, "integrated");
            let index = select_index(&devices, None).unwrap();
            assert_eq!(devices[index].id, "discrete");
        }
    }

    #[test]
    fn unavailable_selection_falls_back_and_empty_list_is_safe() {
        let devices = [device("integrated", vk::PhysicalDeviceType::INTEGRATED_GPU)];
        assert_eq!(select_index(&devices, Some("removed")), Some(0));
        assert_eq!(select_index(&[], Some("removed")), None);
    }
}
