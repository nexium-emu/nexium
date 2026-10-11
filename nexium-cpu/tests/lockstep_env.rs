//! Lockstep through the real entry points: `NEXIUM_LOCKSTEP=1` read from
//! the environment, cores built by `Cpu::new`, guest memory from an
//! `AddressSpace` (fastmem arena when available, so Rustarmic keeps its
//! inline loads while stores go through the recording hooks).  Its own test
//! binary because the environment is read once per process.

#![cfg(all(
    feature = "backend-dynarmic",
    feature = "backend-rustarmic",
    target_os = "linux",
    target_arch = "x86_64"
))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::{AddressSpace, Perm};
use std::sync::Arc;

#[test]
fn lockstep_from_environment_runs_a_program_to_its_svc() {
    let dir = std::env::temp_dir().join(format!("nexium-lockstep-env-{}", std::process::id()));
    std::env::set_var("NEXIUM_LOCKSTEP", "1");
    std::env::set_var("NEXIUM_SINGLECORE", "1");
    std::env::set_var("NEXIUM_LOCKSTEP_SLICE", "64");
    std::env::set_var("NEXIUM_LOCKSTEP_DIR", &dir);

    const CODE: u64 = 0x7f_0000_0000;
    const DATA: u64 = 0x7f_0010_0000;
    let memory = Arc::new(AddressSpace::new());
    nexium_cpu::lockstep_register_address_space(&memory);
    memory.map(CODE, 0x1000, Perm::RX, "code").unwrap();
    memory.map(DATA, 0x2000, Perm::RW, "data").unwrap();
    // x1 = 300; loop { x0 += 3; [data] = x0; x2 = [data]; x3 += x2; x1 -= 1 }
    // mrs x4, cntpct_el0; ldxr/stxr on data+8; svc #0x2a
    let words: [u32; 16] = [
        0xD280_2581, // movz x1, #300
        0x9100_0C00, // add x0, x0, #3
        0xF900_0280, // str x0, [x20]
        0xF940_0282, // ldr x2, [x20]
        0x8B02_0063, // add x3, x3, x2
        0xF100_0421, // subs x1, x1, #1
        0x54FF_FF61, // b.ne -5
        0xD53B_E024, // mrs x4, cntpct_el0
        0x9100_2285, // add x5, x20, #8
        0xC85F_7CA6, // ldxr x6, [x5]
        0x9100_04C6, // add x6, x6, #1
        0xC807_7CA6, // stxr w7, x6, [x5]
        0x35FF_FFA7, // cbnz w7, -3 (retry when a chunk boundary split the pair)
        0xD400_0541, // svc #0x2a
        0xD420_0000, // brk #0
        0xD420_0000,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    memory.write(CODE, &code).unwrap();

    let mut cpu = Cpu::new(CpuBackendKind::Rustarmic).expect("rustarmic core");
    for region in memory.host_regions() {
        unsafe {
            cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                .unwrap();
        }
    }
    // Lockstep is attached: an arena-backed RW page keeps inline loads but
    // loses inline stores, so every store reaches the recording hook.
    let view = cpu.mapping_view(DATA).expect("rustarmic keeps a per-core map");
    match view.page_direct {
        Some(direct) => assert_eq!(direct, (true, false), "{view}"),
        None => eprintln!("no fastmem arena here; inline-store check skipped"),
    }
    cpu.set_pc(CODE);
    cpu.set_register(20, DATA);
    let mut event = CpuEvent::Running;
    for _ in 0..1000 {
        event = cpu.run_with_count(10_000).event;
        if !matches!(event, CpuEvent::Running) {
            break;
        }
    }
    assert!(matches!(event, CpuEvent::Svc(0x2a)), "{event:?} at {:#x}", cpu.get_pc());
    assert_eq!(cpu.get_register(0), 900);
    assert_eq!(cpu.get_register(3), (1..=300u64).map(|i| 3 * i).sum::<u64>());
    let mut word = [0u8; 8];
    memory.read(DATA + 8, &mut word).unwrap();
    assert_eq!(u64::from_le_bytes(word), 1, "exclusive increment");
    // Nothing diverged, so no report was written.
    let reports = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
    assert_eq!(reports, 0, "unexpected lockstep reports in {}", dir.display());
}
