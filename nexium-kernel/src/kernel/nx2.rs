use nexium_common::result::{KERNEL_INVALID_ADDRESS, KERNEL_INVALID_COMBINATION, KERNEL_INVALID_SIZE};
use nexium_ipc as ipc;
use nexium_loader::npdm::Npdm;

use crate::services::nx2 as graphics;

use super::cpu_local::{cpu_mut, cpu_ref};
use super::svc::{build_ipc_response, return_subsession};
use super::threads::{IDEAL_CORE_NO_UPDATE, IDEAL_CORE_USE_PROCESS_VALUE, NUM_CORES};
use super::Kernel;

pub const APPLICATION_POOL_SIZE: u64 = 9809 << 20;
pub const HEAP_REGION_SIZE: u64 = 0x2_0000_0000;
pub const GPU_SHARABLE_ATTRIBUTE: u32 = 0x20;
pub const INFO_REMOTE_REGION_ADDRESS: u32 = 29;
pub const INFO_REMOTE_REGION_SIZE: u32 = 30;
pub const INFO_REMOTE_MEMORY_USAGE: u32 = 31;
pub const INFO_REMOTE_MEMORY_USAGE_PEAK: u32 = 32;
pub const INFO_PROCESS_PAGE_SIZE: u32 = 33;
const GRAPHICS_SCALING_PARAMETERS: u32 = 10;
const DEFAULT_CORE_MASK: u64 = 0x3F;
const DEFAULT_PRIORITY_MASK: u64 = 0x0FFF_FFFF_F000_0000;
const PAGE_SIZE: u64 = 0x1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nx2Process {
    pub core_mask: u64,
    pub priority_mask: u64,
}

impl Nx2Process {
    pub fn from_npdm(npdm: &Npdm) -> Option<Self> {
        if !npdm.is_switch2() {
            return None;
        }
        let thread = npdm.kernel_caps().thread_info;
        Some(Self {
            core_mask: thread.map_or(DEFAULT_CORE_MASK, |thread| thread.core_mask()),
            priority_mask: thread.map_or(DEFAULT_PRIORITY_MASK, |thread| thread.priority_mask()),
        })
    }
}

pub fn fold_core_id(core: i32) -> i32 {
    let cores = NUM_CORES as i32;
    if (cores..2 * cores).contains(&core) {
        core - cores
    } else {
        core
    }
}

pub fn fold_core_mask(mask: u64) -> u64 {
    let low = (1u64 << NUM_CORES) - 1;
    (mask & low) | (mask >> NUM_CORES & low)
}

pub fn guest_core_mask(current: (i32, u64), core_id: i32, mask: u64, process_ideal_core: i32) -> (i32, u64) {
    match core_id {
        IDEAL_CORE_USE_PROCESS_VALUE => (process_ideal_core, 1u64 << process_ideal_core),
        IDEAL_CORE_NO_UPDATE => (current.0, mask),
        _ => (core_id, mask),
    }
}

pub fn guest_current_core(mask: u64, physical: i32) -> i32 {
    (0..2 * NUM_CORES as i32)
        .find(|&core| mask & (1u64 << core) != 0 && fold_core_id(core) == physical)
        .unwrap_or(physical)
}

pub fn info(kernel: &Kernel, info_type: u32) -> Option<u64> {
    kernel.nx2?;
    let value = match info_type {
        INFO_REMOTE_REGION_ADDRESS => kernel.alias_base + kernel.alias_size / 2,
        INFO_REMOTE_REGION_SIZE => kernel.alias_size / 2,
        INFO_REMOTE_MEMORY_USAGE | INFO_REMOTE_MEMORY_USAGE_PEAK => 0,
        INFO_PROCESS_PAGE_SIZE => PAGE_SIZE,
        _ => return None,
    };
    log::debug!("[nx2] svcGetInfo type={info_type} -> {value:#x}");
    Some(value)
}

pub(crate) fn dispatch_ipc(
    kernel: &mut Kernel,
    ctx: &mut ipc::IpcCtx,
    session_handle: u32,
    port_name: &str,
    cmd_id: u32,
) -> Option<Vec<u8>> {
    kernel.nx2?;
    if (port_name, cmd_id) == ("ISelfController", 2000) {
        log::info!("[nx2] ISelfController cmd=2000 -> {}", graphics::SUPERVISOR);
        return Some(graphics::open_session(kernel, ctx, graphics::SUPERVISOR));
    }
    if let Some(reply) = graphics::dispatch(kernel, ctx, port_name, cmd_id) {
        return Some(reply);
    }
    if let Some(service) = subsession(port_name, cmd_id) {
        log::info!("[nx2] {port_name} cmd={cmd_id} -> {service}");
        return Some(return_subsession(kernel, ctx, session_handle, service));
    }
    let reply = match (port_name, cmd_id) {
        ("IGraphicsScalingParametersFunctions", 10) => GRAPHICS_SCALING_PARAMETERS.to_le_bytes().to_vec(),
        ("IGraphicsScalingParametersFunctions" | "IPerformanceFunctions", _) => vec![0; 4],
        _ => return None,
    };
    log::info!("[nx2] {port_name} cmd={cmd_id} stubbed");
    Some(build_ipc_response(ctx, 0, &reply, &[]))
}

fn subsession(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("appletOE", 10) => Some("IApplicationProxy"),
        ("IApplicationProxy", 40) => Some("IPerformanceFunctions"),
        ("IApplicationProxy", 60) => Some("IGraphicsScalingParametersFunctions"),
        ("aud:u", 2) => Some("audren:u"),
        _ => None,
    }
}

pub fn svc_set_gpu_memory_attribute(_kernel: &mut Kernel) -> u32 {
    let Some((address, size, mask, attribute)) = cpu_ref().map(|cpu| {
        (
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2) as u32,
            cpu.get_register(3) as u32,
        )
    }) else {
        return 1;
    };
    let result = if address % PAGE_SIZE != 0 {
        KERNEL_INVALID_ADDRESS
    } else if size == 0 || size % PAGE_SIZE != 0 || address.checked_add(size).is_none() {
        KERNEL_INVALID_SIZE
    } else if attribute & !mask != 0 || (mask | attribute) & !GPU_SHARABLE_ATTRIBUTE != 0 {
        KERNEL_INVALID_COMBINATION
    } else {
        0
    };
    log::debug!(
        "[nx2] svcSetGpuMemoryAttribute addr={address:#x} size={size:#x} mask={mask:#x} attr={attribute:#x} -> {result:#x}"
    );
    if let Some(cpu) = cpu_mut() {
        cpu.set_register(0, result as u64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_cores_past_the_emulated_ones_fold_onto_the_low_cores() {
        assert_eq!([0, 3, 4, 5, 7].map(fold_core_id), [0, 3, 0, 1, 3]);
        assert_eq!([-1, -2, -3, 8, 63].map(fold_core_id), [-1, -2, -3, 8, 63]);
        assert_eq!(fold_core_mask(0x3F), 0xF);
        assert_eq!(fold_core_mask(0x30), 0x3);
        assert_eq!(fold_core_mask(0x20), 0x2);
        assert_eq!(fold_core_mask(0x5), 0x5);
    }

    #[test]
    fn threads_pinned_past_the_emulated_cores_still_see_their_own_core() {
        let pinned: Vec<i32> = (0..6).map(|core| guest_current_core(1 << core, fold_core_id(core))).collect();
        assert_eq!(pinned, [0, 1, 2, 3, 4, 5]);
        assert_eq!(guest_current_core(0x3F, 1), 1);
        assert_eq!(guest_current_core(0x30, 1), 5);
        assert_eq!(guest_current_core(0x10, 2), 2);
    }

    #[test]
    fn guest_core_masks_follow_the_ideal_core_rules() {
        assert_eq!(guest_core_mask((0, 1), -1, 0x20, 0), (-1, 0x20));
        assert_eq!(guest_core_mask((4, 0x10), 5, 0x30, 0), (5, 0x30));
        assert_eq!(guest_core_mask((4, 0x10), IDEAL_CORE_NO_UPDATE, 0x30, 0), (4, 0x30));
        assert_eq!(guest_core_mask((4, 0x10), IDEAL_CORE_USE_PROCESS_VALUE, 0x30, 1), (1, 0x2));
    }
}
