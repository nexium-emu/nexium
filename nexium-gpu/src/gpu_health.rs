use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

const BUDGET_LOG_INTERVAL_NS: u64 = 60_000_000_000;
const BUDGET_PRESSURE_LOG_INTERVAL_NS: u64 = 10_000_000_000;
const BUDGET_PRESSURE_PERCENT: u64 = 90;

struct GpuHealth {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    memory_budget: bool,
    fault_info: Option<vk::PFN_vkGetDeviceFaultInfoEXT>,
}

static HEALTH: OnceLock<GpuHealth> = OnceLock::new();
static DEVICE_LOST_REPORTED: AtomicBool = AtomicBool::new(false);
static LAST_BUDGET_LOG_NS: AtomicU64 = AtomicU64::new(0);
static VRAM_HEAP_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_PRESSURE_CHECK_NS: AtomicU64 = AtomicU64::new(0);
const PRESSURE_CHECK_INTERVAL_NS: u64 = 500_000_000;

pub(crate) fn install(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    memory_budget: bool,
    device_fault: bool,
) {
    let fault_info = device_fault
        .then(|| unsafe {
            instance.get_device_proc_addr(device.handle(), c"vkGetDeviceFaultInfoEXT".as_ptr())
        })
        .flatten()
        .map(|function| unsafe {
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetDeviceFaultInfoEXT>(
                function,
            )
        });
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let heap_count = (memory_properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    let vram_bytes = memory_properties.memory_heaps[..heap_count]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    VRAM_HEAP_BYTES.store(vram_bytes, Ordering::Relaxed);
    let _ = HEALTH.set(GpuHealth {
        instance: instance.clone(),
        physical_device,
        device: device.handle(),
        memory_budget,
        fault_info,
    });
}

pub(crate) fn vram_heap_bytes() -> u64 {
    VRAM_HEAP_BYTES.load(Ordering::Relaxed)
}

pub(crate) fn take_vram_pressure(now_ns: u64) -> bool {
    let last = LAST_PRESSURE_CHECK_NS.load(Ordering::Relaxed);
    if last != 0 && now_ns.saturating_sub(last) < PRESSURE_CHECK_INTERVAL_NS {
        return false;
    }
    if LAST_PRESSURE_CHECK_NS
        .compare_exchange(last, now_ns.max(1), Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    HEALTH
        .get()
        .and_then(query_budget)
        .is_some_and(|heaps| heaps.iter().any(HeapBudget::under_pressure))
}

pub(crate) fn note_result(context: &str, result: vk::Result) {
    if result == vk::Result::ERROR_DEVICE_LOST {
        report_device_lost(context);
    }
}

pub fn device_lost() -> bool {
    DEVICE_LOST_REPORTED.load(Ordering::Acquire)
}

fn report_device_lost(context: &str) {
    if DEVICE_LOST_REPORTED.swap(true, Ordering::AcqRel) {
        return;
    }
    log::error!("[device-lost] first detected in {context}");
    let Some(health) = HEALTH.get() else {
        return;
    };
    if let Some(heaps) = query_budget(health) {
        for heap in heaps {
            log::error!("[device-lost] {}", heap.describe());
        }
    }
    match health.fault_info {
        Some(function) => log_fault_info(health.device, function),
        None => log::error!("[device-lost] VK_EXT_device_fault unavailable; no fault details"),
    }
}

fn log_fault_info(device: vk::Device, function: vk::PFN_vkGetDeviceFaultInfoEXT) {
    let mut counts = vk::DeviceFaultCountsEXT::default();
    let result = unsafe { function(device, &mut counts, std::ptr::null_mut()) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        log::error!("[device-lost] vkGetDeviceFaultInfoEXT counts failed: {result:?}");
        return;
    }
    let mut addresses =
        vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
    let mut vendors =
        vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
    counts.vendor_binary_size = 0;
    let mut info = vk::DeviceFaultInfoEXT {
        p_address_infos: addresses.as_mut_ptr(),
        p_vendor_infos: vendors.as_mut_ptr(),
        p_vendor_binary_data: std::ptr::null_mut(),
        ..Default::default()
    };
    let result = unsafe { function(device, &mut counts, &mut info) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        log::error!("[device-lost] vkGetDeviceFaultInfoEXT failed: {result:?}");
        return;
    }
    log::error!("[device-lost] fault: {}", c_text(&info.description));
    for address in addresses.iter().take(counts.address_info_count as usize) {
        log::error!(
            "[device-lost] address type={:?} address={:#x} precision={:#x}",
            address.address_type,
            address.reported_address,
            address.address_precision
        );
    }
    for vendor in vendors.iter().take(counts.vendor_info_count as usize) {
        log::error!(
            "[device-lost] vendor: {} code={:#x} data={:#x}",
            c_text(&vendor.description),
            vendor.vendor_fault_code,
            vendor.vendor_fault_data
        );
    }
}

fn c_text(chars: &[std::os::raw::c_char]) -> String {
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

struct HeapBudget {
    index: usize,
    device_local: bool,
    size: u64,
    budget: u64,
    usage: u64,
}

impl HeapBudget {
    fn describe(&self) -> String {
        format!(
            "heap{} {} used={} MiB budget={} MiB size={} MiB",
            self.index,
            if self.device_local { "vram" } else { "system" },
            self.usage >> 20,
            self.budget >> 20,
            self.size >> 20
        )
    }

    fn under_pressure(&self) -> bool {
        self.device_local
            && self.budget != 0
            && self.usage.saturating_mul(100) >= self.budget.saturating_mul(BUDGET_PRESSURE_PERCENT)
    }
}

fn query_budget(health: &GpuHealth) -> Option<Vec<HeapBudget>> {
    if !health.memory_budget {
        return None;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let memory_properties = {
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            health
                .instance
                .get_physical_device_memory_properties2(health.physical_device, &mut properties);
        }
        properties.memory_properties
    };
    let count = (memory_properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    Some(
        (0..count)
            .map(|index| {
                let heap = memory_properties.memory_heaps[index];
                HeapBudget {
                    index,
                    device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                    size: heap.size,
                    budget: budget.heap_budget[index],
                    usage: budget.heap_usage[index],
                }
            })
            .collect(),
    )
}

pub(crate) fn log_memory_budget_periodically(now_ns: u64) {
    let last = LAST_BUDGET_LOG_NS.load(Ordering::Relaxed);
    if last != 0 && now_ns.saturating_sub(last) < BUDGET_PRESSURE_LOG_INTERVAL_NS {
        return;
    }
    let Some(health) = HEALTH.get() else {
        return;
    };
    let Some(heaps) = query_budget(health) else {
        return;
    };
    let pressure = heaps.iter().any(HeapBudget::under_pressure);
    if !pressure && last != 0 && now_ns.saturating_sub(last) < BUDGET_LOG_INTERVAL_NS {
        return;
    }
    if LAST_BUDGET_LOG_NS
        .compare_exchange(last, now_ns, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let summary = heaps
        .iter()
        .map(HeapBudget::describe)
        .collect::<Vec<_>>()
        .join("; ");
    if pressure {
        log::warn!("[vram] near budget: {summary}");
    } else {
        log::info!("[vram] {summary}");
    }
}

#[cfg(test)]
mod tests {
    use super::{c_text, HeapBudget};

    #[test]
    fn vram_pressure_starts_at_ninety_percent_of_the_budget() {
        let heap = |usage| HeapBudget {
            index: 0,
            device_local: true,
            size: 16 << 30,
            budget: 10 << 30,
            usage,
        };
        assert!(!heap(8 << 30).under_pressure());
        assert!(heap(9 << 30).under_pressure());
        let system = HeapBudget {
            device_local: false,
            ..heap(10 << 30)
        };
        assert!(!system.under_pressure());
    }

    #[test]
    fn fault_descriptions_stop_at_the_terminator() {
        let mut text = [0 as std::os::raw::c_char; 8];
        for (slot, byte) in text.iter_mut().zip(b"mmu\0junk") {
            *slot = *byte as std::os::raw::c_char;
        }
        assert_eq!(c_text(&text), "mmu");
    }
}
