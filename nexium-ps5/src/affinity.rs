use std::ffi::{c_char, c_int, CStr};
use std::sync::OnceLock;

const CPU_LEVEL_WHICH: c_int = 3;
const CPU_WHICH_TID: c_int = 1;
const ROLES: [&str; 6] = ["nexium-gpu-subm", "nexium-render", "nexium-core0", "nexium-core1", "nexium-core2", "nexium-core3"];

unsafe extern "C" {
    fn cpuset_getaffinity(level: c_int, which: c_int, id: i64, size: usize, mask: *mut u64) -> c_int;
    fn cpuset_setaffinity(level: c_int, which: c_int, id: i64, size: usize, mask: *const u64) -> c_int;
    fn sceKernelGetCurrentCpu() -> c_int;
    fn pthread_setaffinity_np(thread: usize, size: usize, mask: *const u64) -> c_int;
}

fn worker_cpus() -> &'static [u32] {
    static CPUS: OnceLock<Vec<u32>> = OnceLock::new();
    CPUS.get_or_init(|| {
        if !std::env::var("NEXIUM_PS5_PIN").is_ok_and(|v| v == "1") {
            return Vec::new();
        }
        let mut original = 0u64;
        if unsafe { cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &mut original) } != 0 {
            return Vec::new();
        }
        let mut cores = Vec::new();
        let mut cpus = Vec::new();
        let mut seen = Vec::new();
        for cpu in 0..64u32 {
            if original & (1 << cpu) == 0 {
                continue;
            }
            let one = 1u64 << cpu;
            if unsafe { cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &one) } != 0 {
                break;
            }
            let mut spins = 0;
            while unsafe { sceKernelGetCurrentCpu() } != cpu as c_int && spins < 200 {
                unsafe { libc::sched_yield() };
                spins += 1;
            }
            let amd = unsafe { core::arch::x86_64::__cpuid_count(0x8000_001e, 0) };
            let basic = unsafe { core::arch::x86_64::__cpuid_count(1, 0) };
            let core = (basic.ebx >> 24) >> 1;
            seen.push((cpu, amd.eax, core, basic.ebx >> 24));
            if !cores.contains(&core) {
                cores.push(core);
                cpus.push(cpu);
            }
        }
        unsafe { cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &original) };
        crate::klog!("affinity: allowed {original:#x}, {} physical cores, worker cpus {cpus:?}, probe {seen:?}", cores.len());
        cpus
    })
}

pub fn pin(thread: usize, name: *const c_char) {
    if name.is_null() {
        return;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_bytes();
    let Some(role) = ROLES.iter().position(|r| name.starts_with(r.as_bytes())) else {
        return;
    };
    let cpus = worker_cpus();
    if cpus.len() < ROLES.len() {
        return;
    }
    let mask = 1u64 << cpus[role];
    let rc = unsafe { pthread_setaffinity_np(thread, 8, &mask) };
    crate::klog!("affinity: {} -> cpu {} rc {rc}", ROLES[role], cpus[role]);
}
