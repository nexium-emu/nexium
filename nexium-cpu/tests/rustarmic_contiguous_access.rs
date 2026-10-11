#![cfg(feature = "backend-rustarmic")]


use nexium_cpu::{rustarmic::RustarmicCpu, CpuEvent, FaultSnapshot};
use nexium_memory::Perm;
use std::time::Instant;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SNAPSHOT: u64 = 0x30_0000;
const INIT: u64 = CODE + 0x400;
const DUMP: u64 = CODE + 0x600;
const SVC: u32 = 0xd400_0fc1;
const GUARD: u64 = 0x1234_5678_9abc_def0;
const NZCV: u64 = 0xa000_0000;
const FPCR: u64 = 0x0300_0000;
const FPSR: u64 = 0x0800_0091;
const PAGE: usize = 0x1000;
const SNAPSHOT_BYTES: usize = 0x400;

#[repr(align(16))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct AlignedPage([u8; PAGE]);

impl std::ops::Deref for AlignedPage {
    type Target = [u8; PAGE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for AlignedPage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn gprs() -> [u64; 31] {
    std::array::from_fn(|index| GUARD ^ index as u64)
}

fn vectors() -> [[u8; 16]; 32] {
    std::array::from_fn(|register| {
        std::array::from_fn(|byte| (register * 37 + byte * 11 + 0x93) as u8)
    })
}

fn put_words(code: &mut [u8], offset: usize, words: &[u32]) {
    assert!(offset + words.len() * 4 <= code.len());
    for (index, word) in words.iter().enumerate() {
        code[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
}

fn program(words: &[u32]) -> Box<[u8; PAGE]> {
    assert!(words.len() * 4 <= 0x400);
    let mut code = Box::new([0; PAGE]);
    put_words(&mut code[..], 0, words);
    let mut initialize = Vec::new();
    let mut dump = Vec::new();
    for register in 0..32u32 {
        initialize.push(0x3dc0_0000 | (register << 10) | (27 << 5) | register);
        dump.push(0x3d80_0000 | (register << 10) | (27 << 5) | register);
        code[0x800 + register as usize * 16..0x810 + register as usize * 16]
            .copy_from_slice(&vectors()[register as usize]);
    }
    initialize.extend([0xd51b_4218, 0xd51b_4419, 0xd51b_443a, SVC]);
    dump.extend([
        0xd53b_4218,
        0xf900_0000 | (64 << 10) | (27 << 5) | 24,
        0xd53b_4419,
        0xf900_0000 | (65 << 10) | (27 << 5) | 25,
        0xd53b_443a,
        0xf900_0000 | (66 << 10) | (27 << 5) | 26,
        SVC,
    ]);
    put_words(&mut code[..], 0x400, &initialize);
    put_words(&mut code[..], 0x600, &dump);
    code
}

fn cpu() -> RustarmicCpu {
    RustarmicCpu::new_with_engine_config(&rustarmic::EngineConfig {
        code_cache_bytes: 1024 * 1024,
        memory_mode: rustarmic::MemoryMode::PageTable,
        ..Default::default()
    })
    .unwrap()
}

fn restore_gprs(cpu: &mut RustarmicCpu, values: &[u64; 31]) {
    for (register, value) in values.iter().copied().enumerate() {
        cpu.set_register(register as u32, value);
    }
    cpu.set_sp(GUARD);
    cpu.set_tpidrro_el0(!GUARD);
}

fn terminal(cpu: &mut RustarmicCpu, start: u64, budget: u64) -> (CpuEvent, u64) {
    cpu.set_pc(start);
    let mut retired = 0;
    for _ in 0..8 {
        let (event, count) = cpu.run_with_count(budget);
        retired += count;
        if !matches!(event, CpuEvent::Running) {
            return (event, retired);
        }
    }
    panic!("no terminal event from {start:#x}, pc={:#x}", cpu.get_pc());
}

fn svc(cpu: &mut RustarmicCpu, start: u64, retired: u64) {
    let (event, actual) = terminal(cpu, start, retired + 1000);
    assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
    assert_eq!(actual, retired);
    assert!(cpu.take_fault().is_none());
}

fn initialize(cpu: &mut RustarmicCpu) {
    cpu.set_register(24, NZCV);
    cpu.set_register(25, FPCR);
    cpu.set_register(26, FPSR);
    cpu.set_register(27, CODE + 0x800);
    svc(cpu, INIT, 36);
    restore_gprs(cpu, &gprs());
}

fn access(bytes: usize, load: bool, rn: u32, rt: u32) -> u32 {
    let base = match (bytes, load) {
        (1, false) => 0x3900_0000,
        (1, true) => 0x3940_0000,
        (2, false) => 0x7900_0000,
        (2, true) => 0x7940_0000,
        (4, false) => 0xb900_0000,
        (4, true) => 0xb940_0000,
        (8, false) => 0xf900_0000,
        (8, true) => 0xf940_0000,
        (16, false) => 0x3d80_0000,
        (16, true) => 0x3dc0_0000,
        _ => panic!("unsupported test width {bytes}"),
    };
    base | (rn << 5) | rt
}

fn value(bytes: &[u8]) -> u64 {
    assert!(bytes.len() <= 8);
    let mut result = [0; 8];
    result[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(result)
}

fn write_value(bytes: usize) -> Vec<u8> {
    if bytes == 16 {
        vectors()[0].to_vec()
    } else {
        0xfedc_ba98_7654_3210u64.to_le_bytes()[..bytes].to_vec()
    }
}

fn snapshot_expected(expected: &[[u8; 16]; 32]) -> [u8; SNAPSHOT_BYTES] {
    let mut result = [0xa5; SNAPSHOT_BYTES];
    for (register, vector) in expected.iter().enumerate() {
        result[register * 16..register * 16 + 16].copy_from_slice(vector);
    }
    for (offset, status) in [(512, NZCV), (520, FPCR), (528, FPSR)] {
        result[offset..offset + 8].copy_from_slice(&status.to_le_bytes());
    }
    result
}

fn check_state(
    cpu: &mut RustarmicCpu,
    output: *mut u8,
    output_va: u64,
    expected_x: &[u64; 31],
    expected_v: &[[u8; 16]; 32],
    expected_pc: u64,
) {
    assert_eq!(cpu.get_pc(), expected_pc);
    for (register, expected) in expected_x.iter().enumerate() {
        assert_eq!(cpu.get_register(register as u32), *expected, "X{register}");
    }
    assert_eq!(cpu.get_sp(), GUARD);
    assert_eq!(cpu.get_tpidrro_el0(), !GUARD);
    let context = format!("{:?}", cpu.save_thread_context());
    unsafe {
        std::ptr::write_bytes(output, 0xa5, SNAPSHOT_BYTES);
    }
    cpu.set_register(27, output_va);
    svc(cpu, DUMP, 39);
    let actual = unsafe { std::slice::from_raw_parts(output, SNAPSHOT_BYTES) };
    assert_eq!(actual, &snapshot_expected(expected_v)[..]);
    assert_eq!(format!("{:?}", cpu.save_thread_context()), context);
    restore_gprs(cpu, expected_x);
    cpu.set_pc(expected_pc);
}

struct Fixture {
    cpu: RustarmicCpu,
    code: Box<[u8; PAGE]>,
    expected_code: Box<[u8; PAGE]>,
    snapshot: Box<[u8; SNAPSHOT_BYTES]>,
}

impl Fixture {
    fn new(words: &[u32]) -> Self {
        let mut code = program(words);
        let expected_code = code.clone();
        let mut snapshot = Box::new([0xa5; SNAPSHOT_BYTES]);
        let mut cpu = cpu();
        unsafe {
            cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
                .unwrap();
            cpu.map_host(
                SNAPSHOT,
                SNAPSHOT_BYTES as u64,
                Perm::RW,
                snapshot.as_mut_ptr(),
            )
            .unwrap();
        }
        initialize(&mut cpu);
        Self {
            cpu,
            code,
            expected_code,
            snapshot,
        }
    }

    fn check(&mut self, x: &[u64; 31], v: &[[u8; 16]; 32], pc: u64) {
        check_state(
            &mut self.cpu,
            self.snapshot.as_mut_ptr(),
            SNAPSHOT,
            x,
            v,
            pc,
        );
        assert_eq!(self.code, self.expected_code);
    }

    fn fault(&mut self, pc: u64, addr: u64, bytes: usize, write: bool) -> FaultSnapshot {
        let (event, _) = terminal(&mut self.cpu, CODE, 1000);
        assert!(matches!(event, CpuEvent::Exception(0x0e)), "{event:?}");
        let fault = self
            .cpu
            .take_fault()
            .expect("faulting access is observable");
        assert_eq!(
            (fault.pc, fault.addr, fault.size, fault.is_write),
            (pc, addr, bytes as u32, write)
        );
        assert!(self.cpu.take_fault().is_none());
        assert_eq!(self.code, self.expected_code);
        fault
    }
}

#[test]
fn every_exact_width_preserves_canaries_at_aligned_unaligned_and_region_end_addresses() {
    for bytes in [1usize, 2, 4, 8, 16] {
        for load in [true, false] {
            for offset in [0, 1, 5, 64 - bytes] {
                let mut backing: [u8; 128] = std::array::from_fn(|i| (i * 13 + 7) as u8);
                let mut expected = backing;
                let mut fixture =
                    Fixture::new(&[access(8, true, 1, 9), access(bytes, load, 2, 0), SVC]);
                unsafe {
                    fixture
                        .cpu
                        .map_host(DATA, 64, Perm::RW, backing.as_mut_ptr().add(32))
                        .unwrap();
                }
                let mut x = gprs();
                x[1] = DATA;
                x[2] = DATA + offset as u64;
                x[9] = value(&backing[32..40]);
                let mut v = vectors();
                if load {
                    if bytes == 16 {
                        v[0].copy_from_slice(&backing[32 + offset..32 + offset + bytes]);
                    } else {
                        x[0] = value(&backing[32 + offset..32 + offset + bytes]);
                    }
                } else {
                    x[0] = 0xfedc_ba98_7654_3210;
                    expected[32 + offset..32 + offset + bytes].copy_from_slice(&write_value(bytes));
                }
                fixture.cpu.set_register(1, x[1]);
                fixture.cpu.set_register(2, x[2]);
                if !load {
                    fixture.cpu.set_register(0, x[0]);
                }
                svc(&mut fixture.cpu, CODE, 3);
                assert_eq!(
                    backing, expected,
                    "width={bytes} load={load} offset={offset}"
                );
                fixture.check(&x, &v, CODE + 12);
            }
        }
    }
}

#[test]
fn cross_region_copies_and_permission_faults_keep_exact_fallback_byte_effects() {
    for bytes in [2usize, 4, 8, 16] {
        for load in [true, false] {
            for permitted in [true, false] {
                let mut first = [0x31u8; 72];
                let mut second = [0x82u8; 80];
                let mut expected_first = first;
                let mut expected_second = second;
                let split = if bytes == 16 { 7 } else { 1 };
                let addr = DATA + 8 - split as u64;
                let mut fixture =
                    Fixture::new(&[access(8, true, 1, 9), access(bytes, load, 2, 0), SVC]);
                let second_perm = if permitted {
                    Perm::RW
                } else if load {
                    Perm::W
                } else {
                    Perm::RO
                };
                unsafe {
                    fixture
                        .cpu
                        .map_host(DATA, 8, Perm::RW, first.as_mut_ptr().add(32))
                        .unwrap();
                    fixture
                        .cpu
                        .map_host(DATA + 8, 16, second_perm, second.as_mut_ptr().add(32))
                        .unwrap();
                }
                let mut x = gprs();
                x[1] = DATA;
                x[2] = addr;
                x[9] = value(&first[32..40]);
                let mut v = vectors();
                fixture.cpu.set_register(1, DATA);
                fixture.cpu.set_register(2, addr);
                if load && permitted {
                    let mut result = first[40 - split..40].to_vec();
                    result.extend_from_slice(&second[32..32 + bytes - split]);
                    if bytes == 16 {
                        v[0].copy_from_slice(&result);
                    } else {
                        x[0] = value(&result);
                    }
                } else if !load {
                    x[0] = 0xfedc_ba98_7654_3210;
                    fixture.cpu.set_register(0, x[0]);
                    let source = write_value(bytes);
                    expected_first[40 - split..40].copy_from_slice(&source[..split]);
                    if permitted {
                        expected_second[32..32 + bytes - split].copy_from_slice(&source[split..]);
                    }
                }
                if permitted {
                    svc(&mut fixture.cpu, CODE, 3);
                    fixture.check(&x, &v, CODE + 12);
                } else {
                    let fault = fixture.fault(CODE + 4, DATA + 8, bytes.min(8), !load);
                    let mut before_access = x;
                    before_access[0] = if load { gprs()[0] } else { x[0] };
                    assert_eq!(fault.regs, before_access);
                    if load {
                        if bytes == 16 {
                            v[0] = [0; 16];
                        } else {
                            x[0] = 0;
                        }
                    }
                    fixture.check(&x, &v, CODE + 4);
                }
                assert_eq!(first, expected_first);
                assert_eq!(second, expected_second);
            }
        }
    }
}

#[test]
fn vector_and_unaligned_overlap_lookups_keep_first_match_in_both_orders() {
    for bytes in [8usize, 16] {
        for load in [true, false] {
            for narrow_first in [true, false] {
                for denied in [true, false] {
                    let mut narrow = [0x37u8; 96];
                    let mut wide = [0x89u8; 128];
                    let mut expected_narrow = narrow;
                    let mut expected_wide = wide;
                    let mut fixture =
                        Fixture::new(&[access(8, true, 1, 9), access(bytes, load, 2, 0), SVC]);
                    let winner_perm = if !denied {
                        Perm::RW
                    } else if load {
                        Perm::W
                    } else {
                        Perm::RO
                    };
                    unsafe {
                        if narrow_first {
                            fixture
                                .cpu
                                .map_host(DATA + 8, 32, winner_perm, narrow.as_mut_ptr().add(32))
                                .unwrap();
                            fixture
                                .cpu
                                .map_host(DATA, 64, Perm::RW, wide.as_mut_ptr().add(32))
                                .unwrap();
                        } else {
                            fixture
                                .cpu
                                .map_host(DATA, 64, winner_perm, wide.as_mut_ptr().add(32))
                                .unwrap();
                            fixture
                                .cpu
                                .map_host(DATA + 8, 32, Perm::RW, narrow.as_mut_ptr().add(32))
                                .unwrap();
                        }
                    }
                    let addr = DATA + 13;
                    let mut x = gprs();
                    x[1] = if load && denied && !narrow_first {
                        CODE + 0x800
                    } else {
                        DATA
                    };
                    x[2] = addr;
                    x[9] = if x[1] == DATA {
                        value(&wide[32..40])
                    } else {
                        value(&vectors()[0][..8])
                    };
                    fixture.cpu.set_register(1, x[1]);
                    fixture.cpu.set_register(2, addr);
                    let mut v = vectors();
                    let source = if narrow_first {
                        &narrow[37..37 + bytes]
                    } else {
                        &wide[45..45 + bytes]
                    };
                    if load && !denied {
                        if bytes == 16 {
                            v[0].copy_from_slice(source);
                        } else {
                            x[0] = value(source);
                        }
                    } else if !load {
                        x[0] = 0xfedc_ba98_7654_3210;
                        fixture.cpu.set_register(0, x[0]);
                        if !denied {
                            if narrow_first {
                                expected_narrow[37..37 + bytes]
                                    .copy_from_slice(&write_value(bytes));
                            } else {
                                expected_wide[45..45 + bytes].copy_from_slice(&write_value(bytes));
                            }
                        }
                    }
                    if denied {
                        fixture.fault(CODE + 4, addr, bytes.min(8), !load);
                        if load {
                            if bytes == 16 {
                                v[0] = [0; 16];
                            } else {
                                x[0] = 0;
                            }
                        }
                        fixture.check(&x, &v, CODE + 4);
                    } else {
                        svc(&mut fixture.cpu, CODE, 3);
                        fixture.check(&x, &v, CODE + 12);
                    }
                    assert_eq!(narrow, expected_narrow);
                    assert_eq!(wide, expected_wide);
                }
            }
        }
    }
}

#[test]
fn cached_accesses_cannot_wrap_the_exact_address_end() {
    for bytes in [2usize, 4, 8, 16] {
        for load in [true, false] {
            let mut backing = [0x73u8; 96];
            let mut expected = backing;
            let base = u64::MAX - 31;
            let addr = u64::MAX - bytes as u64 / 2;
            let mut fixture =
                Fixture::new(&[access(8, true, 1, 9), access(bytes, load, 2, 0), SVC]);
            unsafe {
                fixture
                    .cpu
                    .map_host(base, 31, Perm::RW, backing.as_mut_ptr().add(32))
                    .unwrap();
            }
            fixture.cpu.set_register(1, base);
            fixture.cpu.set_register(2, addr);
            let mut x = gprs();
            x[1] = base;
            x[2] = addr;
            x[9] = value(&backing[32..40]);
            let mut v = vectors();
            let fault_addr = if bytes == 16 { addr + 8 } else { addr };
            if bytes == 16 {
                if load {
                    v[0][..8].fill(0x73);
                    v[0][8..].fill(0);
                } else {
                    expected[55..63].copy_from_slice(&v[0][..8]);
                }
            } else if load {
                x[0] = 0;
            }
            fixture.fault(CODE + 4, fault_addr, bytes.min(8), !load);
            fixture.check(&x, &v, CODE + 4);
            assert_eq!(backing, expected);
        }
    }
}

#[test]
fn an_earlier_short_overlap_cannot_be_skipped_for_a_later_full_width_region() {
    for load in [true, false] {
        let mut first = [0x37u8; 72];
        let mut later = [0x89u8; 96];
        let mut expected_first = first;
        let mut expected_later = later;
        let mut fixture = Fixture::new(&[access(8, true, 1, 9), access(16, load, 2, 0), SVC]);
        unsafe {
            fixture
                .cpu
                .map_host(DATA + 8, 8, Perm::RW, first.as_mut_ptr().add(32))
                .unwrap();
            fixture
                .cpu
                .map_host(DATA, 32, Perm::RW, later.as_mut_ptr().add(32))
                .unwrap();
        }
        let mut x = gprs();
        x[1] = DATA;
        x[2] = DATA + 12;
        x[9] = value(&later[32..40]);
        fixture.cpu.set_register(1, x[1]);
        fixture.cpu.set_register(2, x[2]);
        let mut v = vectors();
        if load {
            v[0][..4].copy_from_slice(&first[36..40]);
            v[0][4..].copy_from_slice(&later[48..60]);
        } else {
            expected_first[36..40].copy_from_slice(&v[0][..4]);
            expected_later[48..60].copy_from_slice(&v[0][4..]);
        }
        svc(&mut fixture.cpu, CODE, 3);
        fixture.check(&x, &v, CODE + 12);
        assert_eq!(first, expected_first);
        assert_eq!(later, expected_later);
    }
}

#[test]
fn live_pointer_size_and_permission_replacements_refresh_warmed_accesses() {
    for bytes in [8usize, 16] {
        let mut old = [0x29u8; 128];
        let mut new = [0x91u8; 128];
        let old_before = old;
        let new_before = new;
        let mut fixture = Fixture::new(&[access(bytes, true, 1, 0), SVC]);
        fixture.cpu.set_register(1, DATA + 1);
        unsafe {
            fixture
                .cpu
                .map_host(DATA, 64, Perm::RW, old.as_mut_ptr().add(32))
                .unwrap();
        }
        for replacement in [false, true] {
            if replacement {
                unsafe {
                    fixture.cpu.unmap_host(DATA, 64).unwrap();
                    fixture
                        .cpu
                        .map_host(DATA, 32, Perm::RO, new.as_mut_ptr().add(32))
                        .unwrap();
                }
            }
            for _ in 0..4 {
                svc(&mut fixture.cpu, CODE, 2);
                let mut x = gprs();
                x[1] = DATA + 1;
                let mut v = vectors();
                let byte = if replacement { 0x91 } else { 0x29 };
                if bytes == 16 {
                    v[0] = [byte; 16];
                } else {
                    x[0] = value(&[byte; 8]);
                }
                fixture.check(&x, &v, CODE + 8);
            }
        }
        fixture.cpu.set_register(1, DATA + 32 - bytes as u64 + 1);
        let addr = DATA + 32 - bytes as u64 + 1;
        let fault_addr = DATA + 32;
        fixture.fault(CODE, fault_addr, bytes.min(8), false);
        let mut x = gprs();
        x[1] = addr;
        let mut v = vectors();
        if bytes == 16 {
            v[0][..8].fill(0x91);
            v[0][8..].fill(0);
        } else {
            x[0] = 0;
        }
        fixture.check(&x, &v, CODE);
        unsafe {
            fixture.cpu.unmap_host(DATA, 32).unwrap();
            fixture
                .cpu
                .map_host(DATA, 64, Perm::W, new.as_mut_ptr().add(32))
                .unwrap();
        }
        fixture.cpu.set_register(1, DATA + 1);
        fixture.fault(CODE, DATA + 1, bytes.min(8), false);
        x[1] = DATA + 1;
        if bytes == 16 {
            v[0] = [0; 16];
        }
        fixture.check(&x, &v, CODE);
        assert_eq!(old, old_before);
        assert_eq!(new, new_before);
    }
}

#[test]
fn permission_replacements_block_warmed_stores_and_refresh_write_only_backing() {
    for bytes in [8usize, 16] {
        let mut old = [0x2au8; 128];
        let mut replacement = [0x92u8; 128];
        let mut expected_old = old;
        let mut expected_replacement = replacement;
        let source = write_value(bytes);
        expected_old[33..33 + bytes].copy_from_slice(&source);
        expected_replacement[33..33 + bytes].copy_from_slice(&source);
        let mut fixture = Fixture::new(&[access(bytes, false, 1, 0), SVC]);
        let mut x = gprs();
        x[0] = 0xfedc_ba98_7654_3210;
        x[1] = DATA + 1;
        fixture.cpu.set_register(0, x[0]);
        fixture.cpu.set_register(1, x[1]);
        unsafe {
            fixture
                .cpu
                .map_host(DATA, 64, Perm::RW, old.as_mut_ptr().add(32))
                .unwrap();
        }
        for _ in 0..4 {
            svc(&mut fixture.cpu, CODE, 2);
            fixture.check(&x, &vectors(), CODE + 8);
            assert_eq!(old, expected_old);
        }
        unsafe {
            fixture.cpu.unmap_host(DATA, 64).unwrap();
            fixture
                .cpu
                .map_host(DATA, 64, Perm::RO, old.as_mut_ptr().add(32))
                .unwrap();
        }
        fixture.fault(CODE, DATA + 1, bytes.min(8), true);
        fixture.check(&x, &vectors(), CODE);
        assert_eq!(old, expected_old);
        unsafe {
            fixture.cpu.unmap_host(DATA, 64).unwrap();
            fixture
                .cpu
                .map_host(DATA, 64, Perm::W, replacement.as_mut_ptr().add(32))
                .unwrap();
        }
        svc(&mut fixture.cpu, CODE, 2);
        fixture.check(&x, &vectors(), CODE + 8);
        assert_eq!(old, expected_old);
        assert_eq!(replacement, expected_replacement);
    }
}

#[test]
fn discarded_scalar_loads_still_report_unmapped_callbacks_at_every_width() {
    for bytes in [1usize, 2, 4, 8] {
        let mut backing = [0x47u8; 16];
        let before = backing;
        let mut fixture = Fixture::new(&[access(8, true, 1, 9), access(bytes, true, 2, 31), SVC]);
        unsafe {
            fixture
                .cpu
                .map_host(DATA, 16, Perm::RO, backing.as_mut_ptr())
                .unwrap();
        }
        fixture.cpu.set_register(1, DATA);
        fixture.cpu.set_register(2, DATA + 0x1000 + 1);
        let mut x = gprs();
        x[1] = DATA;
        x[2] = DATA + 0x1000 + 1;
        x[9] = value(&backing[..8]);
        let fault = fixture.fault(CODE + 4, x[2], bytes, false);
        assert_eq!(fault.regs, x);
        fixture.check(&x, &vectors(), CODE + 4);
        assert_eq!(backing, before);
    }
}

#[test]
fn exact_vector_store_invalidates_cached_executable_code_without_touching_canaries() {
    let patch_words = [0x5280_00e0, SVC, 0xd503_201f, 0xd503_201f];
    let mut patch = [0u8; 16];
    put_words(&mut patch, 0, &patch_words);
    let patch_before = patch;
    let mut target = [0x56u8; 80];
    put_words(
        &mut target,
        32,
        &[0x5280_0020, SVC, 0xd503_201f, 0xd503_201f],
    );
    let mut expected_target = target;
    expected_target[32..48].copy_from_slice(&patch);
    let mut fixture = Fixture::new(&[access(16, true, 3, 0), access(16, false, 1, 0), SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA, 16, Perm::RWX, target.as_mut_ptr().add(32))
            .unwrap();
        fixture
            .cpu
            .map_host(DATA + 0x1000, 16, Perm::RO, patch.as_mut_ptr())
            .unwrap();
    }
    svc(&mut fixture.cpu, DATA, 2);
    assert_eq!(fixture.cpu.get_register(0), 1);
    fixture.cpu.set_register(1, DATA);
    fixture.cpu.set_register(3, DATA + 0x1000);
    svc(&mut fixture.cpu, CODE, 3);
    assert_eq!(target, expected_target);
    let mut x = gprs();
    x[0] = 1;
    x[1] = DATA;
    x[3] = DATA + 0x1000;
    let mut v = vectors();
    v[0] = patch;
    fixture.check(&x, &v, CODE + 12);
    svc(&mut fixture.cpu, DATA, 2);
    x[0] = 7;
    fixture.check(&x, &v, DATA + 8);
    assert_eq!(target, expected_target);
    assert_eq!(patch, patch_before);
}

#[test]
fn five_hot_regions_keep_exact_vector_and_unaligned_values_across_mru_evictions() {
    for bytes in [8usize, 16] {
        let mut backing: [[u8; 64]; 5] =
            std::array::from_fn(|map| std::array::from_fn(|byte| (map * 31 + byte * 13 + 9) as u8));
        let before = backing;
        let mut fixture = Fixture::new(&[access(bytes, true, 1, 0), SVC]);
        for map in 0..5 {
            unsafe {
                fixture
                    .cpu
                    .map_host(
                        DATA + map as u64 * 0x1000,
                        32,
                        Perm::RO,
                        backing[map].as_mut_ptr().add(16),
                    )
                    .unwrap();
            }
        }
        for iteration in 0..20 {
            let map = iteration % 5;
            let mut x = gprs();
            x[1] = DATA + map as u64 * 0x1000 + 1;
            let mut v = vectors();
            if bytes == 16 {
                v[0].copy_from_slice(&backing[map][17..33]);
            } else {
                x[0] = value(&backing[map][17..25]);
            }
            fixture.cpu.set_register(1, x[1]);
            svc(&mut fixture.cpu, CODE, 2);
            fixture.check(&x, &v, CODE + 8);
            assert_eq!(backing, before);
        }
    }
}

#[test]
#[ignore = "CPU-only matched access benchmark; idle host only, --ignored --nocapture"]
fn contiguous_width_and_mapping_position_benchmark() {
    let loops = std::env::var("NEXIUM_CONTIGUOUS_BENCH_LOOPS")
        .ok()
        .map(|text| text.parse::<u64>().expect("integer loop count"))
        .unwrap_or(2_000_000);
    assert!((1..=32_000_000).contains(&loops));
    let select = std::env::var("NEXIUM_CONTIGUOUS_BENCH_CASE").unwrap_or_else(|_| "all".into());
    assert!(matches!(
        select.as_str(),
        "all" | "aligned8" | "unaligned8" | "vector16"
    ));
    for (name, bytes, offset) in [
        ("aligned8", 8usize, 8usize),
        ("unaligned8", 8, 9),
        ("vector16", 16, 9),
    ] {
        if select != "all" && select != name {
            continue;
        }
        for regions in [2usize, 66] {
            for first in [true, false] {
                let mut code = program(&[
                    access(bytes, true, 1, 0),
                    access(bytes, false, 1, 0),
                    0xd100_0442,
                    0xb5ff_ffa2,
                    SVC,
                ]);
                let code_before = code.clone();
                let mut hot = Box::new(AlignedPage(std::array::from_fn(|i| (i * 17 + 3) as u8)));
                assert_eq!(hot.as_mut_ptr() as usize & 15, 0);
                let hot_before = hot.clone();
                let mut padding = vec![0x6cu8; (regions - 2) * PAGE];
                let mut cpu = cpu();
                unsafe {
                    if first {
                        cpu.map_host(DATA, PAGE as u64, Perm::RW, hot.as_mut_ptr())
                            .unwrap();
                    }
                    cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
                        .unwrap();
                    for index in 0..regions - 2 {
                        cpu.map_host(
                            0x40_0000 + index as u64 * PAGE as u64,
                            PAGE as u64,
                            Perm::RW,
                            padding.as_mut_ptr().add(index * PAGE),
                        )
                        .unwrap();
                    }
                    if !first {
                        cpu.map_host(DATA, PAGE as u64, Perm::RW, hot.as_mut_ptr())
                            .unwrap();
                    }
                }
                initialize(&mut cpu);
                let mut x = gprs();
                x[1] = DATA + offset as u64;
                x[2] = 0;
                let mut v = vectors();
                if bytes == 16 {
                    v[0].copy_from_slice(&hot[offset..offset + 16]);
                } else {
                    x[0] = value(&hot[offset..offset + bytes]);
                }
                cpu.set_register(1, x[1]);
                cpu.set_register(2, 65_536);
                svc(&mut cpu, CODE, 65_536 * 4 + 1);
                assert_eq!(hot, hot_before);
                check_state(
                    &mut cpu,
                    unsafe { hot.as_mut_ptr().add(0x400) },
                    DATA + 0x400,
                    &x,
                    &v,
                    CODE + 20,
                );
                hot.copy_from_slice(&hot_before[..]);
                let mut times = Vec::new();
                for trial in 0..5 {
                    restore_gprs(&mut cpu, &gprs());
                    cpu.set_register(1, x[1]);
                    cpu.set_register(2, loops);
                    cpu.set_pc(CODE);
                    let started = Instant::now();
                    let (event, retired) = cpu.run_with_count(loops * 4 + 1000);
                    let elapsed = started.elapsed();
                    assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
                    assert_eq!(retired, loops * 4 + 1);
                    assert!(cpu.take_fault().is_none());
                    assert_eq!(hot, hot_before);
                    assert_eq!(code, code_before);
                    assert!(padding.iter().all(|&byte| byte == 0x6c));
                    check_state(
                        &mut cpu,
                        unsafe { hot.as_mut_ptr().add(0x400) },
                        DATA + 0x400,
                        &x,
                        &v,
                        CODE + 20,
                    );
                    hot.copy_from_slice(&hot_before[..]);
                    let ms = elapsed.as_secs_f64() * 1000.0;
                    println!("case={name} regions={regions} position={} trial={} loops={loops} retired={retired} elapsed_ms={ms:.6}",
                        if first { "first" } else { "last" }, trial + 1);
                    if ms < 100.0 {
                        println!("note=window_below_100ms; use a larger identical fixed loop count for BOTH revisions");
                    }
                    times.push(ms);
                }
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                times.sort_by(f64::total_cmp);
                println!(
                    "case={name} regions={regions} position={} mean_ms={mean:.6} median_ms={:.6}",
                    if first { "first" } else { "last" },
                    times[2]
                );
                drop(cpu);
            }
        }
    }
}
