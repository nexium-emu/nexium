#![cfg(all(feature = "backend-dynarmic", target_arch = "x86_64"))]

//! Times Dynarmic CPU creation plus `map_host` of an application-shaped
//! region set (code, 0xCC00_0000 heap, stack, env/tls extras), the work
//! `Kernel::init_cpu` repeats once per emulated core.
//!
//! Run with `cargo test --release -p nexium-cpu --test dynarmic_init_timing -- --nocapture`.

use std::process::Command;
use std::time::{Duration, Instant};

use nexium_cpu::{Cpu, CpuBackendKind, GuestIsa};
use nexium_memory::{fastmem, AddressSpace, Perm};

const CHILD: &str = "NEXIUM_DYNARMIC_INIT_TIMING_CHILD";
const CODE_BASE: u64 = 0x800_0000;
const CODE_SIZE: u64 = 0x3de_0000;
const HEAP_BASE: u64 = 0x11_0800_0000;
const HEAP_SIZE: u64 = 0xcc00_0000;
const STACK_BASE: u64 = 0x11_d400_0000;
const STACK_SIZE: u64 = 0x100_0000;
const ENV_BASE: u64 = 0x12_5400_0000;
const ENV_SIZE: u64 = 0x11_0000;
const CORES: usize = 4;

fn application_layout() -> AddressSpace {
    let memory = AddressSpace::new();
    memory.map(CODE_BASE, CODE_SIZE, Perm::RX, "code").unwrap();
    memory
        .map_reserved(HEAP_BASE, HEAP_SIZE, Perm::RW, "heap")
        .unwrap();
    memory
        .map(STACK_BASE, STACK_SIZE, Perm::RW, "stack")
        .unwrap();
    memory.map(ENV_BASE, ENV_SIZE, Perm::RW, "extras").unwrap();
    memory
}

fn init_core(memory: &AddressSpace) -> (Cpu, Duration) {
    let start = Instant::now();
    let mut cpu = Cpu::new_with_isa(CpuBackendKind::Dynarmic, GuestIsa::AArch64).unwrap();
    for region in memory.host_regions() {
        unsafe {
            cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                .unwrap();
        }
    }
    (cpu, start.elapsed())
}

#[test]
fn dynarmic_application_init_per_core_is_fast() {
    if std::env::var_os(CHILD).is_none() {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "dynarmic_application_init_per_core_is_fast",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("NEXIUM_NCE", "0")
            .env("NEXIUM_NO_FASTMEM_ARENA", "0")
            .env("NEXIUM_DYNARMIC_NO_FASTMEM", "0")
            .env("NEXIUM_DYNARMIC_SHARED_PAGE_TABLE", "0");
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy();
            if name.starts_with("NEXIUM_WATCH_") || name.starts_with("NEXIUM_PC_UNTIL") {
                child.env_remove(key);
            }
        }
        let output = child.output().expect("isolated init timing process");
        print!("{}", String::from_utf8_lossy(&output.stdout));
        assert!(
            output.status.success(),
            "init timing child failed: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert!(fastmem::request_direct_mode(false));
    assert!(fastmem::base().is_some(), "fastmem arena must be enabled");
    let memory = application_layout();
    let mut cores = Vec::with_capacity(CORES);
    let mut worst = Duration::ZERO;
    for core in 0..CORES {
        let (cpu, elapsed) = init_core(&memory);
        println!("dynarmic init core{core}: {:.3} ms", elapsed.as_secs_f64() * 1e3);
        worst = worst.max(elapsed);
        cores.push(cpu);
    }

    let mut word = [0u8; 4];
    let stack_top = STACK_BASE + STACK_SIZE - 4;
    for (core, cpu) in cores.iter().enumerate() {
        let value = 0xfeed_0000u32 | core as u32;
        cpu.write_bytes(stack_top, &value.to_le_bytes()).unwrap();
        memory.read(stack_top, &mut word).unwrap();
        assert_eq!(u32::from_le_bytes(word), value, "core{core} write reaches host");
        cpu.read_bytes(CODE_BASE + CODE_SIZE - 4, &mut word).unwrap();
    }

    let stack = memory
        .host_regions()
        .into_iter()
        .find(|region| region.base == STACK_BASE)
        .unwrap();
    let cpu = &mut cores[0];
    unsafe {
        assert!(
            cpu.map_host(stack.base, stack.size, stack.perm, stack.host_ptr)
                .is_err(),
            "mapping over a mapped range must fail"
        );
        cpu.unmap_host(STACK_BASE, 0x1000).unwrap();
        assert!(cpu.read_bytes(STACK_BASE, &mut word).is_err());
        cpu.read_bytes(stack_top, &mut word).unwrap();
        cpu.map_host(STACK_BASE, 0x1000, stack.perm, stack.host_ptr)
            .unwrap();
        cpu.read_bytes(STACK_BASE, &mut word).unwrap();
        cpu.unmap_host(stack.base, stack.size).unwrap();
        assert!(cpu.read_bytes(stack_top, &mut word).is_err());
        assert!(cpu.unmap_host(stack.base, stack.size).is_err());
        cpu.map_host(stack.base, stack.size, stack.perm, stack.host_ptr)
            .unwrap();
        cpu.read_bytes(stack_top, &mut word).unwrap();
    }
    assert_eq!(u32::from_le_bytes(word), 0xfeed_0003);

    assert!(
        worst < Duration::from_secs(2),
        "Dynarmic per-core init took {worst:?}"
    );
}
