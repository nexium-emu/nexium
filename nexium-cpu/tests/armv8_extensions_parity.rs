#![cfg(all(feature = "backend-rustarmic", feature = "backend-dynarmic"))]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::Perm;
use std::sync::{Arc, Barrier};

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const PAGE: usize = 0x1000;
const SLOT: usize = 0x800;
const SVC: u32 = 0xd400_0fc1;
const NOP: u32 = 0xd503_201f;
const MARK_X9: u32 = 0xd28b_4b49;
const MRS_X28_NZCV: u32 = 0xd53b_421c;
const INITIAL: u64 = 0x8877_6655_4433_2211;
const SECOND: u64 = 0xfedc_ba98_7654_3210;
const OPERAND: u64 = 0x0123_4567_89ab_cdef;
const TAGGED: u64 = 0x0a00_0000_0000_0000;
const BACKENDS: [CpuBackendKind; 2] = [CpuBackendKind::Dynarmic, CpuBackendKind::Rustarmic];

#[repr(align(64))]
struct AlignedPage([u8; PAGE]);

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    x: [u64; 31],
    sp: u64,
    data: Vec<u8>,
}

impl Outcome {
    fn word(&self, offset: usize) -> u64 {
        u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap())
    }
}

fn canary(reg: u32) -> u64 {
    0x1111_0000_0000_0000 | u64::from(reg)
}

fn run(backend: CpuBackendKind, words: &[u32], registers: &[(u32, u64)], data: &[(usize, u64)]) -> Outcome {
    let mut code = Box::new(AlignedPage([0; PAGE]));
    for (index, word) in words.iter().chain([SVC].iter()).enumerate() {
        code.0[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut memory = Box::new(AlignedPage([0xa5; PAGE]));
    for (offset, value) in data {
        memory.0[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    let mut cpu = Cpu::new(backend).expect("CPU initialization");
    unsafe {
        cpu.map_host(CODE, PAGE as u64, Perm::RX, code.0.as_mut_ptr()).unwrap();
        cpu.map_host(DATA, PAGE as u64, Perm::RW, memory.0.as_mut_ptr()).unwrap();
    }
    for reg in 0..31 {
        cpu.set_register(reg, canary(reg));
    }
    cpu.set_register(0, DATA + SLOT as u64);
    for (reg, value) in registers {
        cpu.set_register(*reg, *value);
    }
    cpu.set_sp(DATA + 0x400);
    cpu.set_pc(CODE);
    let event = cpu.run(10_000).expect("execution").event;
    assert!(matches!(event, CpuEvent::Svc(0x7e)), "{backend:?} {words:08x?}: {event:?}");
    Outcome {
        x: std::array::from_fn(|reg| cpu.get_register(reg as u32)),
        sp: cpu.get_sp(),
        data: memory.0.to_vec(),
    }
}

fn both(words: &[u32], registers: &[(u32, u64)], data: &[(usize, u64)]) -> Outcome {
    let [dynarmic, rustarmic] = BACKENDS.map(|backend| run(backend, words, registers, data));
    assert_eq!(dynarmic, rustarmic, "backends disagree on {words:08x?}");
    dynarmic
}

fn mask(bytes: usize) -> u64 {
    if bytes == 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 }
}

fn atomic(op: u32, size: u32, acquire: bool, release: bool, rs: u32, rn: u32, rt: u32) -> u32 {
    (size << 30) | 0x3820_0000 | (u32::from(acquire) << 23) | (u32::from(release) << 22) | (rs << 16) | (op << 12) | (rn << 5) | rt
}

fn atomic_model(op: u32, bytes: usize, old: u64, operand: u64) -> u64 {
    let bits = bytes as u32 * 8;
    let (a, b) = (old & mask(bytes), operand & mask(bytes));
    let signed = |value: u64| ((value << (64 - bits)) as i64) >> (64 - bits);
    let result = match op {
        0 => a.wrapping_add(b),
        1 => a & !b,
        2 => a ^ b,
        3 => a | b,
        4 => if signed(a) >= signed(b) { a } else { b },
        5 => if signed(a) <= signed(b) { a } else { b },
        6 => a.max(b),
        7 => a.min(b),
        _ => b,
    };
    result & mask(bytes)
}

fn cas(size: u32, acquire: bool, release: bool, rs: u32, rn: u32, rt: u32) -> u32 {
    (size << 30) | 0x08a0_7c00 | (u32::from(acquire) << 22) | (rs << 16) | (u32::from(release) << 15) | (rn << 5) | rt
}

fn casp(pair_of_x: bool, rs: u32, rn: u32, rt: u32) -> u32 {
    (u32::from(pair_of_x) << 30) | 0x0820_7c00 | (1 << 22) | (rs << 16) | (1 << 15) | (rn << 5) | rt
}

#[test]
fn lse_read_modify_write_matches_the_architectural_model() {
    for op in 0..=8 {
        for size in 0..4u32 {
            let bytes = 1usize << size;
            for (acquire, release) in [(false, false), (true, false), (false, true), (true, true)] {
                let out = both(&[atomic(op, size, acquire, release, 1, 0, 2)], &[(1, OPERAND)], &[(SLOT, INITIAL)]);
                let expected = (INITIAL & !mask(bytes)) | atomic_model(op, bytes, INITIAL, OPERAND);
                assert_eq!(out.word(SLOT), expected, "op={op} size={size}");
                assert_eq!(out.x[2], INITIAL & mask(bytes), "op={op} size={size}");
                assert_eq!(out.x[1], OPERAND);
            }
        }
    }
}

#[test]
fn lse_aliasing_and_zero_register_forms() {
    let out = both(&[atomic(0, 3, true, true, 1, 0, 1)], &[(1, OPERAND)], &[(SLOT, INITIAL)]);
    assert_eq!((out.x[1], out.word(SLOT)), (INITIAL, INITIAL.wrapping_add(OPERAND)));
    let out = both(&[atomic(0, 2, false, true, 1, 0, 31)], &[(1, 5)], &[(SLOT, INITIAL)]);
    assert_eq!(out.word(SLOT), (INITIAL & !0xffff_ffff) | ((INITIAL + 5) & 0xffff_ffff));
    let out = both(&[atomic(8, 3, false, false, 0, 0, 0)], &[], &[(SLOT, INITIAL)]);
    assert_eq!((out.x[0], out.word(SLOT)), (INITIAL, DATA + SLOT as u64));
}

#[test]
fn ldapr_loads_every_width() {
    for size in 0..4u32 {
        let out = both(&[(size << 30) | 0x38bf_c003], &[], &[(SLOT, INITIAL)]);
        assert_eq!(out.x[3], INITIAL & mask(1 << size), "size={size}");
    }
}

#[test]
fn cas_swaps_only_on_a_match_and_returns_the_old_value() {
    for size in 0..4u32 {
        let bytes = 1usize << size;
        for (acquire, release) in [(false, false), (true, true)] {
            let hit = both(&[cas(size, acquire, release, 1, 0, 2)], &[(1, INITIAL), (2, OPERAND)], &[(SLOT, INITIAL)]);
            assert_eq!(hit.word(SLOT), (INITIAL & !mask(bytes)) | (OPERAND & mask(bytes)), "size={size}");
            assert_eq!(hit.x[1], INITIAL & mask(bytes));
            let miss = both(&[cas(size, acquire, release, 1, 0, 2)], &[(1, INITIAL ^ 1), (2, OPERAND)], &[(SLOT, INITIAL)]);
            assert_eq!(miss.word(SLOT), INITIAL, "size={size}");
            assert_eq!(miss.x[1], INITIAL & mask(bytes));
        }
    }
}

#[test]
fn casp_swaps_word_and_doubleword_pairs() {
    let words = [(4, INITIAL & 0xffff_ffff), (5, INITIAL >> 32), (6, OPERAND & 0xffff_ffff), (7, OPERAND >> 32)];
    let hit = both(&[casp(false, 4, 0, 6)], &words, &[(SLOT, INITIAL)]);
    assert_eq!(hit.word(SLOT), OPERAND);
    assert_eq!((hit.x[4], hit.x[5]), (INITIAL & 0xffff_ffff, INITIAL >> 32));
    let mut wrong = words;
    wrong[1].1 ^= 1;
    let miss = both(&[casp(false, 4, 0, 6)], &wrong, &[(SLOT, INITIAL)]);
    assert_eq!(miss.word(SLOT), INITIAL);
    assert_eq!((miss.x[4], miss.x[5]), (INITIAL & 0xffff_ffff, INITIAL >> 32));

    let pairs = [(4, INITIAL), (5, SECOND), (6, OPERAND), (7, !OPERAND)];
    let data = [(SLOT, INITIAL), (SLOT + 8, SECOND)];
    let hit = both(&[casp(true, 4, 0, 6)], &pairs, &data);
    assert_eq!((hit.word(SLOT), hit.word(SLOT + 8)), (OPERAND, !OPERAND));
    assert_eq!((hit.x[4], hit.x[5]), (INITIAL, SECOND));
    let miss = both(&[casp(true, 4, 0, 6)], &[(4, INITIAL), (5, !SECOND), (6, OPERAND), (7, !OPERAND)], &data);
    assert_eq!((miss.word(SLOT), miss.word(SLOT + 8)), (INITIAL, SECOND));
    assert_eq!((miss.x[4], miss.x[5]), (INITIAL, SECOND));
}

fn branch_program(branch: u32) -> Vec<u32> {
    let mut words = vec![branch, SVC];
    words.resize(16, NOP);
    words.push(MARK_X9);
    words
}

#[test]
fn authenticated_branches_reach_their_targets() {
    let target = CODE + 0x40;
    for retaa in [0xd65f_0bff, 0xd65f_0fff] {
        let out = both(&branch_program(retaa), &[(30, target)], &[]);
        assert_eq!((out.x[9], out.x[30]), (0x5a5a, target), "{retaa:08x}");
    }
    for branch in [0xd61f_08bf, 0xd61f_0cbf, 0xd71f_08a6, 0xd71f_0ca6] {
        let out = both(&branch_program(branch), &[(5, target)], &[]);
        assert_eq!((out.x[9], out.x[30]), (0x5a5a, canary(30)), "{branch:08x}");
    }
    for call in [0xd63f_08bf, 0xd63f_0cbf, 0xd73f_08bf, 0xd73f_0cbf] {
        let out = both(&branch_program(call), &[(5, target)], &[]);
        assert_eq!((out.x[9], out.x[30]), (0x5a5a, CODE + 4), "{call:08x}");
    }
}

#[test]
fn pointer_authentication_data_ops_are_identity_and_pacga_is_deterministic() {
    let ops = [
        0xdac1_0041, 0xdac1_0441, 0xdac1_0841, 0xdac1_0c41, 0xdac1_1041, 0xdac1_1441, 0xdac1_1841, 0xdac1_1c41,
        0xdac1_23e3, 0xdac1_27e3, 0xdac1_33e3, 0xdac1_43e1, 0xdac1_47e1,
    ];
    let out = both(&ops, &[], &[]);
    for reg in [1, 2, 3] {
        assert_eq!(out.x[reg], canary(reg as u32));
    }
    let out = both(&[0x9ac2_3023], &[(1, OPERAND), (2, SECOND)], &[]);
    assert_eq!(out.x[3], (OPERAND ^ SECOND) & 0xffff_ffff_0000_0000);
}

fn ldra(key_b: bool, doublewords: i32, writeback: bool, rn: u32, rt: u32) -> u32 {
    let imm = (doublewords as u32) & 0x3ff;
    0xf820_0400 | (u32::from(key_b) << 23) | ((imm >> 9) << 22) | ((imm & 0x1ff) << 12) | (u32::from(writeback) << 11) | (rn << 5) | rt
}

#[test]
fn ldraa_and_ldrab_use_scaled_signed_offsets() {
    let data = [(SLOT + 0x18, OPERAND), (SLOT - 0x20, SECOND)];
    let out = both(&[ldra(false, 3, false, 0, 1), ldra(true, -4, true, 0, 2)], &[], &data);
    assert_eq!((out.x[1], out.x[2]), (OPERAND, SECOND));
    assert_eq!(out.x[0], DATA + SLOT as u64 - 0x20);
}

fn tag_store(opc: u32, granules: i32, mode: u32, rn: u32, rt: u32) -> u32 {
    0xd920_0000 | (opc << 22) | (((granules as u32) & 0x1ff) << 12) | (mode << 10) | (rn << 5) | rt
}

#[test]
fn memory_tagging_runs_with_tag_access_disabled() {
    let data = [(SLOT - 0x10, INITIAL), (SLOT, INITIAL), (SLOT + 0x10, INITIAL), (SLOT + 0x18, INITIAL)];
    let out = both(&[tag_store(0, 0, 0b10, 8, 8)], &[(8, DATA + SLOT as u64)], &data);
    assert_eq!(out.x[8], DATA + SLOT as u64);
    for (opc, mode, moved) in [(0, 0b01, 0x20u64), (2, 0b11, 0x20), (0, 0b10, 0)] {
        let out = both(&[tag_store(opc, 2, mode, 0, 1)], &[], &data);
        assert_eq!(out.x[0], DATA + SLOT as u64 + moved);
        assert_eq!(out.word(SLOT), INITIAL);
    }
    let out = both(&[tag_store(1, -1, 0b11, 0, 31)], &[], &data);
    assert_eq!((out.x[0], out.word(SLOT - 0x10), out.word(SLOT - 0x8), out.word(SLOT)), (DATA + SLOT as u64 - 0x10, 0, 0, INITIAL));
    let out = both(&[tag_store(3, 0, 0b10, 0, 31)], &[], &data);
    assert_eq!((out.word(SLOT), out.word(SLOT + 0x10), out.word(SLOT + 0x18)), (0, 0, 0));
    let out = both(&[0xd960_0001], &[(1, TAGGED | 0x1234)], &[]);
    assert_eq!(out.x[1], 0x1234);
    let out = both(&[0x6900_8801], &[(1, OPERAND), (2, SECOND)], &[]);
    assert_eq!((out.word(SLOT + 0x10), out.word(SLOT + 0x18), out.x[0]), (OPERAND, SECOND, DATA + SLOT as u64));
    let out = both(&[0x6980_8801], &[(1, OPERAND), (2, SECOND)], &[]);
    assert_eq!(out.x[0], DATA + SLOT as u64 + 0x10);
    let base = TAGGED | 0x1000;
    let out = both(&[0x9ac2_1001, 0x9ac2_1403, 0x9183_0c04, 0xd181_0405], &[(0, base), (2, 0x8000)], &[]);
    assert_eq!((out.x[1], out.x[3], out.x[4], out.x[5]), (0x1000, 0x8000 | (1 << 0xa), 0x1030, 0xff0));
    let out = both(&[0x9ac1_0002, 0xbac0_0023, MRS_X28_NZCV], &[(0, TAGGED | 0x3000), (1, 0x0f00_0000_0000_1000)], &[]);
    assert_eq!((out.x[2], out.x[3]), (0x2000, (-0x2000i64) as u64));
    assert_eq!(out.x[28] >> 28, 0x8);
}

#[test]
fn ldadd_stays_atomic_across_contending_cores() {
    const THREADS: usize = 4;
    const ITERATIONS: u64 = 20_000;
    let program = [atomic(0, 3, true, true, 1, 0, 2), 0xf100_0463, 0x54ff_ffc1];
    for backend in BACKENDS {
        let mut memory = Box::new(AlignedPage([0; PAGE]));
        let shared = memory.0.as_mut_ptr() as usize;
        let start = Arc::new(Barrier::new(THREADS));
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let mut code = Box::new(AlignedPage([0; PAGE]));
                    for (index, word) in program.iter().chain([SVC].iter()).enumerate() {
                        code.0[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    let mut cpu = Cpu::new(backend).expect("CPU initialization");
                    unsafe {
                        cpu.map_host(CODE, PAGE as u64, Perm::RX, code.0.as_mut_ptr()).unwrap();
                        cpu.map_host(DATA, PAGE as u64, Perm::RW, shared as *mut u8).unwrap();
                    }
                    cpu.set_register(0, DATA + SLOT as u64);
                    cpu.set_register(1, 1);
                    cpu.set_register(3, ITERATIONS);
                    cpu.set_pc(CODE);
                    start.wait();
                    loop {
                        match cpu.run(1_000_000).expect("execution").event {
                            CpuEvent::Svc(0x7e) => break,
                            CpuEvent::Running | CpuEvent::Interrupted => continue,
                            other => panic!("{backend:?}: {other:?}"),
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let counter = u64::from_le_bytes(memory.0[SLOT..SLOT + 8].try_into().unwrap());
        assert_eq!(counter, THREADS as u64 * ITERATIONS, "{backend:?}");
    }
}
