//! Lockstep tests: planted Rustarmic miscompilations must be caught at the
//! exact instruction, and correct programs must produce no divergence.

use super::*;
use crate::rustarmic::RustarmicCpu;
use rustarmic::frontend::fault_injection::FaultInjection;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;

// --- tiny assembler -------------------------------------------------------

fn movz(rd: u32, imm: u32) -> u32 {
    0xD280_0000 | (imm << 5) | rd
}
fn add_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0x9100_0000 | (imm << 10) | (rn << 5) | rd
}
fn subs_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0xF100_0000 | (imm << 10) | (rn << 5) | rd
}
fn eor(rd: u32, rn: u32, rm: u32) -> u32 {
    0xCA00_0000 | (rm << 16) | (rn << 5) | rd
}
fn sub(rd: u32, rn: u32, rm: u32) -> u32 {
    0xCB00_0000 | (rm << 16) | (rn << 5) | rd
}
fn str_x(rt: u32, rn: u32, idx: u32) -> u32 {
    0xF900_0000 | (idx << 10) | (rn << 5) | rt
}
fn ldr_x(rt: u32, rn: u32, idx: u32) -> u32 {
    0xF940_0000 | (idx << 10) | (rn << 5) | rt
}
fn b_ne(from: usize, to: usize) -> u32 {
    let off = (to as i64 - from as i64) as u32 & 0x7FFFF;
    0x5400_0001 | (off << 5)
}
fn svc(imm: u32) -> u32 {
    0xD400_0001 | (imm << 5)
}
const BRK0: u32 = 0xD420_0000;
const MRS_X0_CNTPCT: u32 = 0xD53B_E020;
const MRS_X1_CNTPCT: u32 = 0xD53B_E021;

// --- rig -------------------------------------------------------------------

struct Pages {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl Pages {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, 0x1000).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Self { ptr, layout }
    }
    fn put_words(&self, offset: usize, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), self.ptr.add(offset + i * 4), 4)
            };
        }
    }
    fn u64_at(&self, offset: usize) -> u64 {
        unsafe { (self.ptr.add(offset) as *const u64).read_unaligned() }
    }
    fn set_u64(&self, offset: usize, v: u64) {
        unsafe { (self.ptr.add(offset) as *mut u64).write_unaligned(v) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

fn out_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nexium-lockstep-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn config(dir: Option<PathBuf>) -> Config {
    Config {
        slice: 4096,
        halt_on_divergence: true,
        halt_on_benign: false,
        dyn_fast_run: true,
        fastmem_reads: true,
        start_after: 0,
        sample: 1,
        out_dir: dir,
        max_reports: u32::MAX,
    }
}

struct Rig {
    cpu: RustarmicCpu,
    _code: Pages,
    data: Pages,
}

fn rig(cfg: Config, words: &[u32], inject: Option<FaultInjection>) -> Rig {
    let mut cpu = RustarmicCpu::ls_new_with_lockstep(cfg).expect("cpu");
    assert!(cpu.ls_lockstep().is_some(), "lockstep shadow not attached");
    cpu.ls_jit_mut().fault_injection = inject;
    cpu.ls_lockstep_mut().unwrap().replay_injection = inject;
    let code = Pages::new(0x10000);
    let data = Pages::new(0x10000);
    code.put_words(0, words);
    unsafe {
        cpu.map_host(CODE, 0x10000, Perm::RX, code.ptr).unwrap();
        cpu.map_host(DATA, 0x10000, Perm::RW, data.ptr).unwrap();
    }
    cpu.set_pc(CODE);
    cpu.set_sp(DATA + 0x8000);
    cpu.set_register(20, DATA);
    Rig { cpu, _code: code, data }
}

fn run_to_event(cpu: &mut RustarmicCpu) -> (CpuEvent, u64) {
    let mut total = 0;
    for _ in 0..100_000 {
        let (event, n) = cpu.run_with_count(100_000);
        total += n;
        if !matches!(event, CpuEvent::Running) {
            return (event, total);
        }
    }
    panic!("no event after {total} instructions at pc {:#x}", cpu.get_pc());
}

fn records(cpu: &RustarmicCpu) -> &[Record] {
    &cpu.ls_lockstep().unwrap().records
}

fn class_count(cpu: &RustarmicCpu, class: Class) -> u64 {
    cpu.ls_lockstep()
        .unwrap()
        .stats
        .by_class
        .get(class.label())
        .copied()
        .unwrap_or(0)
}

/// 2000 loop iterations (10k instructions over several chunks), then the
/// planted instruction once.
fn deep_program() -> (Vec<u32>, usize) {
    let mut w = vec![movz(0, 0), movz(1, 2000), movz(5, 7)];
    let top = w.len();
    w.push(add_imm(0, 0, 1));
    w.push(add_imm(5, 5, 3));
    w.push(eor(6, 5, 0));
    w.push(str_x(6, 20, 0));
    w.push(subs_imm(1, 1, 1));
    let at = w.len();
    w.push(b_ne(at, top));
    w.push(movz(7, 100));
    let planted = w.len();
    w.push(add_imm(7, 7, 5));
    w.push(str_x(7, 20, 1));
    w.push(svc(1));
    (w, planted)
}

// --- planted bugs ------------------------------------------------------------

#[test]
fn clean_program_has_no_divergence() {
    let (words, _) = deep_program();
    let mut rig = rig(config(None), &words, None);
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Svc(1)), "{event:?}");
    assert!(records(&rig.cpu).is_empty(), "{:?}", records(&rig.cpu));
    assert_eq!(rig.cpu.get_register(7), 105);
    assert_eq!(rig.data.u64_at(8), 105);
    let stats = &rig.cpu.ls_lockstep().unwrap().stats;
    assert!(stats.checked >= 3, "{stats:?}");
    eprintln!("clean: {stats:?}");
}

#[test]
fn planted_bug_is_caught_at_the_exact_instruction() {
    let (words, planted) = deep_program();
    let inject = FaultInjection {
        mask: 0xFFFF_FFFF,
        value: words[planted],
        xor: 0x100,
    };
    let dir = out_dir("planted");
    let mut rig = rig(config(Some(dir.clone())), &words, Some(inject));
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Stalled), "{event:?}");
    let recs = records(&rig.cpu);
    assert_eq!(recs.len(), 1, "{recs:?}");
    let rec = &recs[0];
    eprintln!("{}", rec.text);
    assert_eq!(rec.class, Class::Divergence);
    assert_eq!(rec.pc, CODE + 4 * planted as u64, "located at the wrong pc");
    assert_eq!(rec.word, words[planted]);
    assert!(!rec.block_context, "the injected instruction is wrong on its own");
    assert!(rec.text.contains("x7: rustarmic=0x0000000000000169 dynarmic=0x0000000000000069"), "{}", rec.text);
    // The core is left right before the instruction.
    assert_eq!(rig.cpu.get_pc(), rec.pc);
    assert_eq!(rig.cpu.get_register(7), 100);
    // Report and a verified repro on disk.
    let report = rec.report.as_ref().expect("report file");
    assert!(std::fs::read_to_string(report).unwrap().contains("[divergence]"));
    let repro_path = rec.repro.as_ref().expect("repro file");
    assert_eq!(rec.repro_verified, Some(true));
    if let Some(keep) = std::env::var_os("NEXIUM_LOCKSTEP_KEEP_REPRO") {
        std::fs::copy(repro_path, keep).unwrap();
    }
    let repro = Repro::parse(&std::fs::read_to_string(repro_path).unwrap()).unwrap();
    assert_eq!(repro.pc, rec.pc);
    assert_eq!(repro.expect_dyn.x[7], 105);
    assert_eq!(repro.expect_rust.x[7], 105 ^ 0x100);
    // The same file through the public replay: clean JITs agree, the planted
    // one reproduces the divergence.
    let clean = replay(&repro).unwrap();
    assert!(!clean.diverged, "{}", clean.summary);
    assert!(clean.dyn_matches_recorded);
    let planted_replay = replay_with(&repro, &|cpu: &mut RustarmicCpu| {
        cpu.ls_jit_mut().fault_injection = Some(inject);
    })
    .unwrap();
    assert!(planted_replay.diverged);
    assert!(planted_replay.rust_matches_recorded);
    // Stays stalled.
    assert!(matches!(rig.cpu.run_with_count(1000).0, CpuEvent::Stalled));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn planted_bug_inside_a_hot_loop_is_caught_on_first_execution() {
    let (words, _) = deep_program();
    // `eor x6, x5, x0` runs on every iteration and its result is stored;
    // its 1st execution is the 6th instruction of the program.  (A
    // corruption that cancels out, such as toggling a bit of an
    // accumulator every iteration, leaves nothing observable at a chunk
    // boundary and is rightly not reported.)
    let inject = FaultInjection {
        mask: 0xFFFF_FFFF,
        value: eor(6, 5, 0),
        xor: 1 << 40,
    };
    for slice in [1, 7, 4096] {
        let mut cfg = config(None);
        cfg.slice = slice;
        let mut rig = rig(cfg, &words, Some(inject));
        let (event, _) = run_to_event(&mut rig.cpu);
        assert!(matches!(event, CpuEvent::Stalled), "slice {slice}: {event:?}");
        let rec = &records(&rig.cpu)[0];
        assert_eq!(rec.class, Class::Divergence);
        assert_eq!(rec.pc, CODE + 20, "slice {slice}");
        // Index within the chunk that contained it.
        assert_eq!(rec.chunk_pc + 4 * (rec.k - 1), rec.pc, "slice {slice}");
        if slice == 4096 {
            assert_eq!((rec.chunk_pc, rec.k), (CODE, 6));
        }
        assert_eq!(rig.cpu.get_register(5), 10, "slice {slice}: core must stop before the eor");
        assert_eq!((rig.cpu.get_register(0), rig.cpu.get_register(6)), (1, 0));
    }
}

#[test]
fn continue_mode_adopts_dynarmic_and_finishes_correctly() {
    let (words, planted) = deep_program();
    let inject = FaultInjection {
        mask: 0xFFFF_FFFF,
        value: words[planted],
        xor: 0x100,
    };
    let mut cfg = config(None);
    cfg.halt_on_divergence = false;
    let mut rig = rig(cfg, &words, Some(inject));
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Svc(1)), "{event:?}");
    assert_eq!(records(&rig.cpu).len(), 1);
    assert_eq!(rig.cpu.get_register(7), 105, "Dynarmic's value was adopted");
    assert_eq!(rig.data.u64_at(8), 105);
}

#[test]
fn planted_flag_bug_in_compare_is_located() {
    // subs x1, x1, #1 corrupted (x1 ^= 4) changes the loop trip count.
    let (words, _) = deep_program();
    let inject = FaultInjection {
        mask: 0xFFFF_FFFF,
        value: subs_imm(1, 1, 1),
        xor: 4,
    };
    let mut rig = rig(config(None), &words, Some(inject));
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Stalled), "{event:?}");
    let rec = &records(&rig.cpu)[0];
    assert_eq!(rec.pc, CODE + 28);
    assert!(rec.text.contains("x1: rustarmic="), "{}", rec.text);
}

// --- known differences ---------------------------------------------------------

#[test]
fn svc_and_brk_are_compared_and_delivered() {
    let words = [movz(0, 1), add_imm(0, 0, 2), svc(5), add_imm(0, 0, 4), BRK0];
    let mut rig = rig(config(None), &words, None);
    let (event, n) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Svc(5)), "{event:?}");
    assert_eq!(n, 3);
    assert_eq!(rig.cpu.get_pc(), CODE + 12);
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Exception(0x100)), "{event:?}");
    assert_eq!(rig.cpu.get_register(0), 7);
    assert!(records(&rig.cpu).is_empty(), "{:?}", records(&rig.cpu));
}

#[test]
fn timer_reads_are_masked() {
    let mut words = vec![MRS_X0_CNTPCT, movz(2, 500)];
    let top = words.len();
    words.push(subs_imm(2, 2, 1));
    let at = words.len();
    words.push(b_ne(at, top));
    words.push(MRS_X1_CNTPCT);
    words.push(sub(3, 1, 0));
    words.push(str_x(3, 20, 0));
    words.push(svc(2));
    let mut rig = rig(config(None), &words, None);
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Svc(2)), "{event:?}");
    assert!(records(&rig.cpu).is_empty(), "{:?}", records(&rig.cpu));
    assert_eq!(class_count(&rig.cpu, Class::Timer), 0);
}

#[test]
fn dc_zva_matches_on_both_jits() {
    // This dynarmic-sys build zeroes the block like Rustarmic (and hardware).
    for dyn_run in [true, false] {
        let words = [0xD50B_7434 /* dc zva, x20 */, ldr_x(1, 20, 0), svc(3)];
        let mut cfg = config(None);
        cfg.dyn_fast_run = dyn_run;
        let mut rig = rig(cfg, &words, None);
        rig.data.set_u64(0, 0x1122_3344_5566_7788);
        rig.data.set_u64(64, 0x99);
        let (event, _) = run_to_event(&mut rig.cpu);
        assert!(matches!(event, CpuEvent::Svc(3)), "{event:?}");
        assert!(records(&rig.cpu).is_empty(), "{:?}", records(&rig.cpu));
        assert_eq!(rig.data.u64_at(0), 0);
        assert_eq!(rig.data.u64_at(64), 0x99, "only one 64-byte block");
        assert_eq!(rig.cpu.get_register(1), 0);
    }
}

#[test]
fn guest_fault_is_classified_not_reported_as_divergence() {
    let words = [movz(0, 3), add_imm(0, 0, 1), ldr_x(1, 9, 0), svc(4)];
    let mut rig = rig(config(None), &words, None);
    rig.cpu.set_register(9, 0x4000_0000);
    let (event, n) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Exception(0x0E)), "{event:?}");
    assert!(n >= 2);
    assert!(records(&rig.cpu).is_empty(), "{:?}", records(&rig.cpu));
    assert_eq!(class_count(&rig.cpu, Class::GuestFault) + class_count(&rig.cpu, Class::MappingSync), 1);
    assert_eq!(rig.cpu.get_register(0), 4);
    assert_eq!(rig.cpu.take_fault().map(|f| f.addr), Some(0x4000_0000));
}

#[test]
fn fp_results_never_report_a_false_divergence() {
    // 0/0, inf*0 and sqrt(-1) produce default NaNs; 1/3 is ordinary.
    let words = [
        0x1E22_1820, // fdiv s0, s1, s2
        0x1E63_0883, // fmul d3, d4, d3
        0x1E61_C0A5, // fsqrt d5, d5
        0x1E27_18C6, // fdiv s6, s6, s7
        svc(6),
    ];
    let mut rig = rig(config(None), &words, None);
    {
        let ctx = rig.cpu.ls_ctx_mut();
        ctx.v[1] = [0, 0];
        ctx.v[2] = [0, 0];
        ctx.v[4] = [f64::INFINITY.to_bits(), 0];
        ctx.v[3] = [0, 0];
        ctx.v[5] = [(-1.0f64).to_bits(), 0];
        ctx.v[6] = [1.0f32.to_bits() as u64, 0];
        ctx.v[7] = [3.0f32.to_bits() as u64, 0];
    }
    let (event, _) = run_to_event(&mut rig.cpu);
    assert!(matches!(event, CpuEvent::Svc(6)), "{event:?}");
    for rec in records(&rig.cpu) {
        eprintln!("{}", rec.text);
        assert_ne!(rec.class, Class::Divergence, "{}", rec.text);
    }
    let lockstep = rig.cpu.ls_lockstep().unwrap();
    eprintln!("fp: {:?}", lockstep.stats.by_class);
    // Benign classes adopt Dynarmic's (ARM) results.
    assert_eq!(rig.cpu.ls_ctx().v[0][0] as u32, ARM_DEFAULT_NAN32);
    assert_eq!(rig.cpu.ls_ctx().v[3][0], ARM_DEFAULT_NAN64);
}

// --- false positives over random programs -----------------------------------

/// Random terminating programs from rustarmic's control-flow fuzzer (loops,
/// calls, indirect branches, every flag-setting form, pair/Q/writeback/
/// exclusive memory ops, SIMD/FP, optional self-modifying code), run under
/// lockstep with several chunk sizes and both Dynarmic modes: no divergence
/// may be reported, and the end state must equal the same run with
/// checking switched off.
#[test]
fn random_programs_produce_no_false_positives() {
    use super::control_flow_fuzz::{gen_program, image, GenOpts, CODE_BASE, CODE_LIMIT, MEM_SIZE, STACK_TOP};
    let cases: u64 = std::env::var("NEXIUM_LOCKSTEP_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    // One checked core and one reference core for all programs: dynarmic-sys
    // never frees a JIT, so a core per program would pile up memory.  The
    // reference has the same chunking with checking switched off, because
    // lockstep clears the exclusive monitor at every chunk start (Dynarmic
    // clears its own at every run): architecturally allowed, but it can make
    // a split LDXR/STXR pair fail where a plain run would not.
    let mut checked_cpu = RustarmicCpu::ls_new_with_lockstep(config(None)).unwrap();
    let mut reference_cpu = {
        let mut cfg = config(None);
        cfg.start_after = u64::MAX;
        RustarmicCpu::ls_new_with_lockstep(cfg).unwrap()
    };
    let ranges = [
        (0x1000, CODE_BASE, Perm::RW),
        (CODE_BASE, CODE_LIMIT, Perm::RWX),
        (CODE_LIMIT, MEM_SIZE as u64, Perm::RW),
    ];
    let mut totals: HashMap<&'static str, u64> = HashMap::new();
    let (mut checked, mut fast, mut stepped, mut instructions) = (0u64, 0u64, 0u64, 0u64);
    let mut benign_shown = 0;
    let mut not_compared = 0;
    for seed in 0..cases {
        let opts = GenOpts {
            lse: false, // Dynarmic implements ARMv8.0 (no LSE), like the Switch.
            smc: seed % 3 == 0,
        };
        let p = gen_program(seed, opts);
        let slice = [3u64, 64, 4096][(seed % 3) as usize];
        let run = |cpu: &mut RustarmicCpu| -> (Pages, CpuEvent, Stats, usize) {
            let mem = Pages::new(MEM_SIZE);
            let img = image(&p);
            unsafe { std::ptr::copy_nonoverlapping(img.as_ptr(), mem.ptr, img.len()) };
            let records_before = cpu.ls_lockstep().unwrap().records.len();
            {
                let lockstep = cpu.ls_lockstep_mut().unwrap();
                lockstep.cfg.slice = slice;
                lockstep.cfg.dyn_fast_run = seed % 4 != 1;
                lockstep.stats = Stats::default();
            }
            for &(lo, hi, perm) in &ranges {
                unsafe { cpu.map_host(lo, hi - lo, perm, mem.ptr.add(lo as usize)).unwrap() };
            }
            let ctx = cpu.ls_ctx_mut();
            ctx.pc = CODE_BASE;
            ctx.sp = STACK_TOP;
            ctx.x = p.init_x;
            ctx.v = [[0; 2]; 32];
            ctx.v[..8].copy_from_slice(&p.init_v);
            ctx.nzcv = p.init_nzcv;
            ctx.fpcr = 0;
            ctx.fpsr = 0;
            let event = loop {
                let (event, _) = cpu.run_with_count(slice);
                if !matches!(event, CpuEvent::Running) {
                    break event;
                }
            };
            for &(lo, hi, _) in &ranges {
                unsafe { cpu.unmap_host(lo, hi - lo).unwrap() };
            }
            let lockstep = cpu.ls_lockstep().unwrap();
            (mem, event, lockstep.stats.clone(), records_before)
        };
        let (mem, event, stats, records_before) = run(&mut checked_cpu);
        let (ref_mem, ref_event, _, _) = run(&mut reference_cpu);
        for rec in &checked_cpu.ls_lockstep().unwrap().records[records_before..] {
            assert!(rec.class.benign(), "seed {seed}: unexpected lockstep report\n{}", rec.text);
            if benign_shown < 4 {
                benign_shown += 1;
                eprintln!("seed {seed}: benign event (shown for reference)\n{}", rec.text);
            }
        }
        for (class, n) in &stats.by_class {
            *totals.entry(class).or_default() += n;
        }
        checked += stats.checked;
        fast += stats.fast_hits;
        stepped += stats.stepped;
        instructions += stats.retired_checked;
        // An event that ends a chunk early moves later chunk boundaries (and
        // so exclusive-monitor clears) relative to the reference.
        if stats.by_class.keys().any(|c| *c != Class::FpStatus.label()) {
            not_compared += 1;
            continue;
        }
        assert_eq!(format!("{event:?}"), format!("{ref_event:?}"), "seed {seed}");
        let (a, b) = (GuestState::from_rust(checked_cpu.ls_ctx()), GuestState::from_rust(reference_cpu.ls_ctx()));
        let mut diff = Diff::default();
        compare_states(&a, &b, &mut diff);
        diff.fpsr = None;
        assert!(diff.fields.is_empty(), "seed {seed}: checking changed the result: {}", diff.summary());
        let (ma, mb) = unsafe {
            (
                std::slice::from_raw_parts(mem.ptr, MEM_SIZE),
                std::slice::from_raw_parts(ref_mem.ptr, MEM_SIZE),
            )
        };
        if let Some(i) = (0..MEM_SIZE).find(|&i| ma[i] != mb[i]) {
            panic!(
                "seed {seed}: checking changed memory at {i:#x}: {:02x?} vs {:02x?}",
                &ma[i & !15..(i & !15) + 16],
                &mb[i & !15..(i & !15) + 16]
            );
        }
    }
    eprintln!(
        "random programs: {cases} programs, {instructions} instructions checked in {checked} chunks ({fast} by one Dynarmic run, {stepped} single-stepped); events: {totals:?}; {not_compared} programs with events not compared with the unchecked run"
    );
    for class in totals.keys() {
        assert!(
            ![Class::Divergence.label(), Class::Unsupported.label(), Class::NonReproducible.label()].contains(class),
            "{totals:?}"
        );
    }
}

// --- helpers ---------------------------------------------------------------------

#[test]
fn instruction_helpers() {
    assert_eq!(timer_read_rt(MRS_X0_CNTPCT), Some(0));
    assert_eq!(timer_read_rt(0xD53B_E043), Some(3)); // cntvct_el0
    assert_eq!(timer_read_rt(0xD53B_E003), None); // cntfrq_el0
    assert_eq!(store_exclusive_target(0xC803_7C41), Some((2, 8))); // stxr w3, x1, [x2]
    assert_eq!(store_exclusive_target(0x8803_7C41), Some((2, 4))); // stxr w3, w1, [x2]
    assert_eq!(store_exclusive_target(0xC823_0C41), Some((2, 16))); // stxp w3, x1, x3, [x2]
    assert_eq!(store_exclusive_target(0xC85F_7C41), None); // ldxr
    assert_eq!(store_exclusive_target(0xC89F_FC41), None); // stlr
    assert!(is_fp_simd_dp(0x0EA7_B31A)); // sqdmlsl (unimplemented in dynarmic)
    assert!(is_fp_simd_dp(0x1E22_1820));
    assert!(!is_fp_simd_dp(add_imm(0, 0, 1)));
    assert!(is_branch(b_ne(4, 0)) && is_branch(0xD65F_03C0) && is_branch(0x1400_0000));
    assert!(nan_only_difference([0xFFC0_0000, 0], [0x7FC0_0000, 0]));
    assert!(nan_only_difference([0xFFF8_0000_0000_0000, 1], [0x7FF8_0000_0000_0000, 1]));
    assert!(!nan_only_difference([0x3F80_0000, 0], [0x7FC0_0000, 0]));
    let mut diff = Diff::default();
    diff.fields.push(Field::V(0, [0xFFC0_0000, 0], [0x7FC0_0000, 0]));
    assert_eq!(classify(0x1E22_1820, 0, &diff), Class::FpNan);
    let mut diff = Diff::default();
    diff.fields.push(Field::V(0, [0x3F80_0000, 0], [0x3F80_0001, 0]));
    assert_eq!(classify(0x1E22_1820, 0, &diff), Class::Divergence);
    assert_eq!(classify(0x1E22_1820, 1 << 24, &diff), Class::FpcrMode);
    let mut diff = Diff::default();
    diff.fields.push(Field::X(0, 1, 2));
    assert_eq!(classify(0xD53B_0020, 0, &diff), Class::SystemRegister); // mrs x0, ctr_el0
    assert_eq!(classify(add_imm(0, 0, 1), 0, &diff), Class::Divergence);
    assert_eq!(timestamp().len(), 15);
}

#[test]
fn repro_text_round_trips() {
    let mut r = Repro {
        class: "divergence".into(),
        pc: 0x8f2_eec0,
        word: 0x9100_2268,
        insns: 3,
        code: vec![(0x8f2_eec0, vec![0x68, 0x22, 0x00, 0x91])],
        data: vec![(0x1000, vec![1, 2, 3])],
        dyn_mem: vec![(0x1000, vec![9])],
        rust_mem: vec![(0x1000, vec![8])],
        note: "n".into(),
        ..Repro::default()
    };
    r.start.x[3] = 0xdead;
    r.start.v[31] = [1, 2];
    r.start.nzcv = 0b1010;
    r.expect_dyn.pc = 0x8f2_eecc;
    r.expect_rust.fpsr = 0x10;
    let parsed = Repro::parse(&r.to_text()).unwrap();
    assert_eq!(parsed.start, r.start);
    assert_eq!(parsed.expect_dyn, r.expect_dyn);
    assert_eq!(parsed.expect_rust, r.expect_rust);
    assert_eq!(parsed.code, r.code);
    assert_eq!(parsed.data, r.data);
    assert_eq!(parsed.dyn_mem, r.dyn_mem);
    assert_eq!(parsed.rust_mem, r.rust_mem);
    assert_eq!((parsed.pc, parsed.word, parsed.insns), (r.pc, r.word, r.insns));
}

/// Replays a repro written by a real run:
/// `NEXIUM_LOCKSTEP_REPLAY=<file> cargo test -p nexium-cpu --lib lockstep::tests::replay_file -- --nocapture`
#[test]
fn replay_file() {
    let Some(path) = std::env::var_os("NEXIUM_LOCKSTEP_REPLAY") else {
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read repro");
    let repro = Repro::parse(&text).expect("parse repro");
    let outcome = replay(&repro).expect("replay");
    eprintln!(
        "replay of {}: {} instructions from pc={:#x}\n  rustarmic stop={} dynarmic stop={}\n  diverged={} dynarmic_matches_recorded={} rustarmic_matches_recorded={}\n{}",
        std::path::Path::new(&path).display(),
        repro.insns,
        repro.start.pc,
        outcome.rust_stop,
        outcome.dyn_stop,
        outcome.diverged,
        outcome.dyn_matches_recorded,
        outcome.rust_matches_recorded,
        outcome.summary
    );
}

/// Throughput of lockstep against plain Rustarmic (run with --ignored).
#[test]
#[ignore]
fn lockstep_throughput() {
    // 40 x 65535 iterations of a 9-instruction loop with two stores and a
    // load (~24M instructions), long enough to amortize JIT warm-up.
    let mut words = vec![movz(2, 40), movz(0, 0), movz(5, 7)];
    let outer = words.len();
    words.push(movz(1, 0xFFFF));
    let top = words.len();
    words.extend([
        add_imm(0, 0, 1),
        add_imm(5, 5, 3),
        eor(6, 5, 0),
        str_x(6, 20, 0),
        ldr_x(8, 20, 0),
        add_imm(9, 8, 1),
        str_x(9, 20, 1),
        subs_imm(1, 1, 1),
    ]);
    let at = words.len();
    words.push(b_ne(at, top));
    words.push(subs_imm(2, 2, 1));
    let at = words.len();
    words.push(b_ne(at, outer));
    words.push(svc(9));
    for (label, dyn_run) in [("dyn-run", true), ("step-only", false)] {
        let mut cfg = config(None);
        cfg.dyn_fast_run = dyn_run;
        let mut rig = rig(cfg, &words, None);
        let t = Instant::now();
        let (_, n) = run_to_event(&mut rig.cpu);
        let lock = t.elapsed();
        let s = &rig.cpu.ls_lockstep().unwrap().stats;
        eprintln!(
            "lockstep {label}: {n} insns in {lock:?} = {:.1} M insn/s (fast hits {} stepped {})",
            n as f64 / lock.as_secs_f64() / 1e6,
            s.fast_hits,
            s.stepped
        );
    }
    let mut plain = {
        let _quiet = suppress();
        RustarmicCpu::new().unwrap()
    };
    let code = Pages::new(0x10000);
    let data = Pages::new(0x10000);
    code.put_words(0, &words);
    unsafe {
        plain.map_host(CODE, 0x10000, Perm::RX, code.ptr).unwrap();
        plain.map_host(DATA, 0x10000, Perm::RW, data.ptr).unwrap();
    }
    plain.set_pc(CODE);
    plain.set_register(20, DATA);
    let t = Instant::now();
    let (_, n) = run_to_event(&mut plain);
    let el = t.elapsed();
    eprintln!("plain rustarmic: {n} insns in {el:?} = {:.1} M insn/s", n as f64 / el.as_secs_f64() / 1e6);
}
