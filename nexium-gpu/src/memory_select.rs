use ash::vk;

const STREAM_REBAR_MIN_HEAP_BYTES: u64 = 1 << 30;
const STREAM_REBAR_HEAP_SHARE_DIVISOR: u64 = 4;

static STREAM_REBAR_COMMITTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn stream_rebar_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| stream_rebar_switch(std::env::var("NEXIUM_STREAM_REBAR").ok().as_deref()))
}

fn stream_rebar_switch(value: Option<&str>) -> bool {
    !matches!(
        value.map(|value| value.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamMemoryChoice {
    pub(crate) type_index: u32,
    pub(crate) device_local: bool,
}

fn stream_rebar_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    size: u64,
    committed: u64,
) -> Option<u32> {
    let wanted = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..props.memory_type_count).find(|&index| {
        let memory_type = &props.memory_types[index as usize];
        if type_bits & (1 << index) == 0 || !memory_type.property_flags.contains(wanted) {
            return false;
        }
        let Some(heap) = props.memory_heaps.get(memory_type.heap_index as usize) else {
            return false;
        };
        heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            && heap.size >= STREAM_REBAR_MIN_HEAP_BYTES.max(size.saturating_mul(4))
            && committed.saturating_add(size) <= heap.size / STREAM_REBAR_HEAP_SHARE_DIVISOR
    })
}

pub(crate) fn stream_memory_candidates(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    size: u64,
    rebar: bool,
    committed: u64,
) -> Vec<StreamMemoryChoice> {
    let mut choices = Vec::with_capacity(2);
    if rebar {
        if let Some(type_index) = stream_rebar_type(props, type_bits, size, committed) {
            choices.push(StreamMemoryChoice { type_index, device_local: true });
        }
    }
    if let Some(type_index) = crate::rt_cache::find_memory_type(
        props,
        type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) {
        if !choices.iter().any(|choice| choice.type_index == type_index) {
            choices.push(StreamMemoryChoice {
                type_index,
                device_local: props.memory_types[type_index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            });
        }
    }
    choices
}

pub(crate) struct StreamAllocation {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) choice: StreamMemoryChoice,
    pub(crate) rebar_bytes: u64,
}

pub(crate) fn allocate_stream_memory(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
    name: &str,
) -> Result<StreamAllocation, String> {
    let committed = STREAM_REBAR_COMMITTED.load(std::sync::atomic::Ordering::Relaxed);
    let choices = stream_memory_candidates(
        props,
        requirements.memory_type_bits,
        requirements.size,
        stream_rebar_enabled(),
        committed,
    );
    let mut last_error = format!("no HOST_VISIBLE|HOST_COHERENT memory type for {name}");
    for choice in choices {
        let alloc = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: requirements.size,
            memory_type_index: choice.type_index,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(memory) => {
                let rebar_bytes = if choice.device_local { requirements.size } else { 0 };
                if rebar_bytes != 0 {
                    STREAM_REBAR_COMMITTED
                        .fetch_add(rebar_bytes, std::sync::atomic::Ordering::Relaxed);
                }
                let memory_type = &props.memory_types[choice.type_index as usize];
                log::info!(
                    "memory: {name} type={} flags={:?} heap={} bytes size={} bytes",
                    choice.type_index,
                    memory_type.property_flags,
                    props.memory_heaps[memory_type.heap_index as usize].size,
                    requirements.size,
                );
                return Ok(StreamAllocation { memory, choice, rebar_bytes });
            }
            Err(error) => {
                log::warn!(
                    "memory: {name} allocation in type {} failed: {error:?}",
                    choice.type_index
                );
                last_error = format!("allocate_memory({name}): {error:?}");
            }
        }
    }
    Err(last_error)
}

pub(crate) fn release_stream_rebar_bytes(bytes: u64) {
    if bytes != 0 {
        let _ = STREAM_REBAR_COMMITTED.try_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| Some(current.saturating_sub(bytes)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(types: &[(vk::MemoryPropertyFlags, u32)], heaps: &[(u64, vk::MemoryHeapFlags)]) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = types.len() as u32;
        for (index, &(flags, heap)) in types.iter().enumerate() {
            props.memory_types[index] = vk::MemoryType { property_flags: flags, heap_index: heap };
        }
        props.memory_heap_count = heaps.len() as u32;
        for (index, &(size, flags)) in heaps.iter().enumerate() {
            props.memory_heaps[index] = vk::MemoryHeap { size, flags };
        }
        props
    }

    const DL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const HV: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
    const HC: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_COHERENT;
    const CACHED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_CACHED;

    fn rebar_device(bar_heap_bytes: u64) -> vk::PhysicalDeviceMemoryProperties {
        props(
            &[
                (vk::MemoryPropertyFlags::empty(), 1),
                (DL, 0),
                (HV | HC, 1),
                (HV | HC | CACHED, 1),
                (DL | HV | HC, 2),
            ],
            &[
                (16 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (32 << 30, vk::MemoryHeapFlags::empty()),
                (bar_heap_bytes, vk::MemoryHeapFlags::DEVICE_LOCAL),
            ],
        )
    }

    #[test]
    fn rebar_heap_is_preferred_for_streams_with_host_fallback() {
        let device = rebar_device(16 << 30);
        let choices = stream_memory_candidates(&device, 0x1f, 256 << 20, true, 0);
        assert_eq!(
            choices,
            vec![
                StreamMemoryChoice { type_index: 4, device_local: true },
                StreamMemoryChoice { type_index: 2, device_local: false },
            ]
        );
    }

    #[test]
    fn small_bar_window_keeps_streams_in_host_memory() {
        let device = rebar_device(256 << 20);
        let choices = stream_memory_candidates(&device, 0x1f, 64 << 20, true, 0);
        assert_eq!(choices, vec![StreamMemoryChoice { type_index: 2, device_local: false }]);
    }

    #[test]
    fn disabled_switch_and_missing_type_bits_keep_first_match() {
        let device = rebar_device(16 << 30);
        assert_eq!(
            stream_memory_candidates(&device, 0x1f, 1 << 20, false, 0),
            vec![StreamMemoryChoice { type_index: 2, device_local: false }]
        );
        assert_eq!(
            stream_memory_candidates(&device, 0x0f, 1 << 20, true, 0),
            vec![StreamMemoryChoice { type_index: 2, device_local: false }]
        );
    }

    #[test]
    fn heap_share_cap_falls_back_to_host_memory() {
        let device = rebar_device(16 << 30);
        let cap = (16u64 << 30) / STREAM_REBAR_HEAP_SHARE_DIVISOR;
        assert_eq!(
            stream_memory_candidates(&device, 0x1f, 256 << 20, true, cap - (128 << 20)),
            vec![StreamMemoryChoice { type_index: 2, device_local: false }]
        );
        assert_eq!(
            stream_memory_candidates(&device, 0x1f, 256 << 20, true, cap - (256 << 20))[0],
            StreamMemoryChoice { type_index: 4, device_local: true }
        );
    }

    #[test]
    fn unified_memory_device_uses_its_only_host_visible_type() {
        let device = props(
            &[(DL | HV | HC, 0), (DL | HV | HC | CACHED, 0)],
            &[(8 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL)],
        );
        assert_eq!(
            stream_memory_candidates(&device, 0x3, 64 << 20, true, 0),
            vec![StreamMemoryChoice { type_index: 0, device_local: true }]
        );
    }

    #[test]
    fn switch_values_follow_the_default_on_convention() {
        assert!(stream_rebar_switch(None));
        assert!(stream_rebar_switch(Some("1")));
        assert!(!stream_rebar_switch(Some("0")));
        assert!(!stream_rebar_switch(Some(" Off ")));
    }
}
