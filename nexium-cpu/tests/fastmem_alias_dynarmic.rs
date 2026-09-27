#![cfg(all(feature = "backend-dynarmic", target_arch = "x86_64"))]

use std::process::Command;
use std::sync::Arc;

use nexium_cpu::{CpuBackendKind, CpuCore, CpuEvent, CpuSystem, CpuSystemConfig};
use nexium_memory::{fastmem, AddressSpace, Perm};

const SOURCE: u64 = 0x20_0000;
const WRITER: u64 = 0x20_4000;
const ALIAS: u64 = 0x30_0000;
const PAGE_SIZE: u64 = 0x1000;
const ORIGINAL: u32 = 0x5280_00e0;
const PATCHED: u32 = 0x5280_0560;
const CHILD: &str = "NEXIUM_FASTMEM_ALIAS_TEST_CHILD";

fn write_instructions(memory: &AddressSpace, address: u64, words: &[u32]) {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory.write(address, &bytes).unwrap();
}

fn run_until_svc(core: &mut CpuCore, address: u64, expected: u16) {
    core.cpu_mut().set_pc(address);
    for _ in 0..16 {
        let event = core.cpu_mut().run(64).expect("guest execution").event;
        match event {
            CpuEvent::Running => {}
            CpuEvent::Svc(actual) => {
                assert_eq!(actual, expected);
                return;
            }
            other => panic!("unexpected {other:?} at {:#x}", core.cpu().get_pc()),
        }
    }
    panic!("guest did not reach SVC at {:#x}", core.cpu().get_pc());
}

#[test]
fn fastmem_alias_writes_invalidate_warmed_code_on_both_cores() {
    if std::env::var_os(CHILD).is_none() {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "fastmem_alias_writes_invalidate_warmed_code_on_both_cores",
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
        let output = child.output().expect("isolated fastmem test process");
        assert!(
            output.status.success(),
            "fastmem alias child failed: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert!(fastmem::request_direct_mode(false));
    assert!(fastmem::base().is_some(), "fastmem arena must be enabled");
    let memory = Arc::new(AddressSpace::new());
    memory
        .map(SOURCE, PAGE_SIZE, Perm::RX, "alias-source")
        .unwrap();
    memory
        .map(WRITER, PAGE_SIZE, Perm::RX, "alias-writer")
        .unwrap();
    write_instructions(&memory, SOURCE, &[ORIGINAL, 0xd400_0e01]);
    write_instructions(
        &memory,
        WRITER,
        &[
            0xb940_0001,
            0xb900_0002,
            0xb940_0003,
            0xb940_00a4,
            0xd400_0e21,
        ],
    );
    let system = CpuSystem::new(
        CpuBackendKind::Dynarmic,
        Arc::clone(&memory),
        CpuSystemConfig::default(),
    )
    .unwrap();
    let mut first = system.create_core(0).unwrap();
    let mut second = system.create_core(1).unwrap();
    for core in [&mut first, &mut second] {
        run_until_svc(core, SOURCE, 0x70);
        assert_eq!(core.cpu().get_register(0), 7);
    }

    memory
        .map_alias(ALIAS, SOURCE, PAGE_SIZE, Perm::RW, "code-alias")
        .unwrap();
    let regions = memory.host_regions();
    let source = regions.iter().find(|region| region.base == SOURCE).unwrap();
    let alias = regions.iter().find(|region| region.base == ALIAS).unwrap();
    assert_eq!(source.host_ptr, alias.host_ptr);
    assert_ne!(alias.host_ptr, fastmem::host_ptr(ALIAS).unwrap());
    first.sync_mappings().unwrap();
    second.sync_mappings().unwrap();
    first.cpu_mut().set_register(0, ALIAS);
    first.cpu_mut().set_register(2, u64::from(PATCHED));
    first.cpu_mut().set_register(5, SOURCE);
    run_until_svc(&mut first, WRITER, 0x71);
    assert_eq!(first.cpu().get_register(1), u64::from(ORIGINAL));
    assert_eq!(first.cpu().get_register(3), u64::from(PATCHED));
    assert_eq!(first.cpu().get_register(4), u64::from(PATCHED));
    let mut actual = [0u8; 4];
    memory.read(SOURCE, &mut actual).unwrap();
    assert_eq!(u32::from_le_bytes(actual), PATCHED);
    memory.read(ALIAS, &mut actual).unwrap();
    assert_eq!(u32::from_le_bytes(actual), PATCHED);

    memory.unmap_alias(ALIAS, SOURCE, PAGE_SIZE).unwrap();
    assert!(memory.read(ALIAS, &mut actual).is_err());
    unsafe {
        first.cpu_mut().unmap_host(ALIAS, PAGE_SIZE).unwrap();
    }
    for core in [&mut first, &mut second] {
        core.sync_mappings().unwrap();
        run_until_svc(core, SOURCE, 0x70);
        assert_eq!(core.cpu().get_register(0), 43);
    }
}
