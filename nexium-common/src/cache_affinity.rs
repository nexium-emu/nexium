pub struct EmulationCacheAffinity {
    #[cfg(all(windows, target_arch = "x86_64"))]
    selection: Option<windows::Selection>,
}

impl EmulationCacheAffinity {
    pub fn acquire() -> Self {
        Self {
            #[cfg(all(windows, target_arch = "x86_64"))]
            selection: windows::acquire(),
        }
    }
}

impl Drop for EmulationCacheAffinity {
    fn drop(&mut self) {
        #[cfg(all(windows, target_arch = "x86_64"))]
        if let Some(selection) = self.selection.take() {
            windows::restore(selection);
        }
    }
}

#[cfg(any(all(windows, target_arch = "x86_64"), test))]
fn preferred_cache_mask(caches: &[(u64, u32)], cores: &[u64], available: u64) -> Option<u64> {
    let [first, second] = caches else {
        return None;
    };
    if first.0 & second.0 != 0
        || first.0 | second.0 != available
        || first.0.count_ones() != second.0.count_ones()
    {
        return None;
    }
    let (larger, smaller) = if first.1 > second.1 {
        (first, second)
    } else {
        (second, first)
    };
    if smaller.1 == 0 || u64::from(larger.1) < 2 * u64::from(smaller.1) {
        return None;
    }
    let mut covered = 0;
    let mut larger_cores = 0;
    let mut smaller_cores = 0;
    for &core in cores {
        if core == 0 || core & covered != 0 {
            return None;
        }
        covered |= core;
        if core & larger.0 == core {
            larger_cores += 1;
        } else if core & smaller.0 == core {
            smaller_cores += 1;
        } else {
            return None;
        }
    }
    (covered == available && larger_cores >= 6 && larger_cores == smaller_cores).then_some(larger.0)
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod windows {
    use std::ffi::{c_char, c_void};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CacheDescriptor {
        level: u8,
        associativity: u8,
        line_size: u16,
        size: u32,
        kind: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    union ProcessorData {
        cache: CacheDescriptor,
        reserved: [u64; 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ProcessorInformation {
        mask: usize,
        relationship: u32,
        data: ProcessorData,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetActiveProcessorGroupCount() -> u16;
        fn GetLogicalProcessorInformation(
            buffer: *mut ProcessorInformation,
            length: *mut u32,
        ) -> i32;
        fn GetProcessAffinityMask(
            process: *mut c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        fn SetProcessAffinityMask(process: *mut c_void, mask: usize) -> i32;
        fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub(super) struct Selection {
        original: usize,
        selected: usize,
    }

    pub(super) fn acquire() -> Option<Selection> {
        if std::env::var("NEXIUM_WIN_CACHE_AFFINITY").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        }) || [
            "NEXIUM_WIN_CORE0_CPUSET_MASK",
            "NEXIUM_WIN_PREP_CPUSET_MASK",
            "NEXIUM_WIN_GPU_SUBMIT_CPUSET_MASK",
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some())
        {
            return None;
        }
        unsafe { acquire_windows() }
    }

    unsafe fn acquire_windows() -> Option<Selection> {
        if GetActiveProcessorGroupCount() != 1 {
            return None;
        }
        let process = GetCurrentProcess();
        let mut original = 0;
        let mut system = 0;
        if GetProcessAffinityMask(process, &mut original, &mut system) == 0 || original != system {
            return None;
        }
        let module = GetModuleHandleA(c"kernel32.dll".as_ptr());
        if module.is_null() {
            return None;
        }
        let proc = GetProcAddress(module, c"GetProcessDefaultCpuSets".as_ptr());
        if proc.is_null() {
            return None;
        }
        let get_sets: unsafe extern "system" fn(*mut c_void, *mut u32, u32, *mut u32) -> i32 =
            std::mem::transmute(proc);
        let mut required = 0;
        if get_sets(process, std::ptr::null_mut(), 0, &mut required) == 0 || required != 0 {
            return None;
        }
        let mut length = 0;
        if GetLogicalProcessorInformation(std::ptr::null_mut(), &mut length) != 0
            || std::io::Error::last_os_error().raw_os_error() != Some(122)
        {
            return None;
        }
        let item_size = std::mem::size_of::<ProcessorInformation>();
        if length == 0 || length as usize % item_size != 0 {
            return None;
        }
        let mut records = vec![
            ProcessorInformation {
                mask: 0,
                relationship: 0,
                data: ProcessorData { reserved: [0; 2] },
            };
            length as usize / item_size
        ];
        let capacity = length;
        if GetLogicalProcessorInformation(records.as_mut_ptr(), &mut length) == 0
            || length > capacity
            || length as usize % item_size != 0
        {
            return None;
        }
        records.truncate(length as usize / item_size);
        let mut caches = Vec::new();
        let mut cores = Vec::new();
        let mut packages = 0;
        for record in records {
            match record.relationship {
                0 => cores.push(record.mask as u64),
                2 => {
                    let cache = record.data.cache;
                    if cache.level == 3 && cache.kind == 0 {
                        caches.push((record.mask as u64, cache.size));
                    }
                }
                3 => packages += 1,
                _ => {}
            }
        }
        if packages != 1 {
            return None;
        }
        let selected = super::preferred_cache_mask(&caches, &cores, system as u64)? as usize;
        if SetProcessAffinityMask(process, selected) == 0 {
            log::warn!(
                "[cpu-cache] could not select larger L3 domain: {}",
                std::io::Error::last_os_error()
            );
            return None;
        }
        log::info!("[cpu-cache] emulation selected larger L3 domain mask {selected:#x}; previous mask {original:#x}");
        Some(Selection { original, selected })
    }

    pub(super) fn restore(selection: Selection) {
        unsafe {
            let process = GetCurrentProcess();
            let mut current = 0;
            let mut system = 0;
            if GetProcessAffinityMask(process, &mut current, &mut system) != 0
                && current == selection.selected
                && SetProcessAffinityMask(process, selection.original) == 0
            {
                log::warn!(
                    "[cpu-cache] could not restore process affinity: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::preferred_cache_mask;

    fn cores(count: u32) -> Vec<u64> {
        (0..count).map(|core| 3 << (core * 2)).collect()
    }

    #[test]
    fn selects_larger_cache_independent_of_domain_order() {
        let caches = [(0xffff, 96 << 20), (0xffff0000, 32 << 20)];
        assert_eq!(
            preferred_cache_mask(&caches, &cores(16), 0xffffffff),
            Some(0xffff)
        );
        assert_eq!(
            preferred_cache_mask(&[caches[1], caches[0]], &cores(16), 0xffffffff),
            Some(0xffff)
        );
        assert_eq!(
            preferred_cache_mask(
                &[(0xffff, 32 << 20), (0xffff0000, 96 << 20)],
                &cores(16),
                0xffffffff
            ),
            Some(0xffff0000)
        );
        assert_eq!(
            preferred_cache_mask(
                &[(0xfff, 96 << 20), (0xfff000, 32 << 20)],
                &cores(12),
                0xffffff
            ),
            Some(0xfff)
        );
    }

    #[test]
    fn leaves_symmetric_small_and_incomplete_topologies_unchanged() {
        let core_masks = cores(16);
        for caches in [
            vec![],
            vec![(0xffffffff, 96 << 20)],
            vec![(0xffff, 32 << 20), (0xffff0000, 32 << 20)],
            vec![(0xffff, 48 << 20), (0xffff0000, 32 << 20)],
            vec![(0xffff, 96 << 20), (0xffff0000, 0)],
            vec![(0xffff, 96 << 20), (0xffff00, 32 << 20)],
            vec![(0xff, 96 << 20), (0xff00, 32 << 20)],
        ] {
            assert_eq!(preferred_cache_mask(&caches, &core_masks, 0xffffffff), None);
        }
        let caches = [(0xffff, 96 << 20), (0xffff0000, 32 << 20)];
        assert_eq!(preferred_cache_mask(&caches, &core_masks, 0xffff), None);
        assert_eq!(
            preferred_cache_mask(&caches, &core_masks[..15], 0xffffffff),
            None
        );
        assert_eq!(
            preferred_cache_mask(&[(0xff, 96 << 20), (0xff00, 32 << 20)], &cores(8), 0xffff),
            None
        );
    }

    #[test]
    fn rejects_overlapping_or_cross_domain_cores() {
        let caches = [(0xffff, 96 << 20), (0xffff0000, 32 << 20)];
        let mut core_masks = cores(16);
        core_masks[0] = core_masks[1];
        assert_eq!(preferred_cache_mask(&caches, &core_masks, 0xffffffff), None);
        core_masks[0] = 0x10001;
        assert_eq!(preferred_cache_mask(&caches, &core_masks, 0xffffffff), None);
    }
}
