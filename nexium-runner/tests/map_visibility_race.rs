#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{Cpu, CpuBackendKind, CpuEvent};
use nexium_memory::{AddressSpace, Perm, PAGE_SIZE};
use nexium_runner::boot::sync_host_region_changes;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

/// Runner core-loop slice length.
const SLICE: u64 = 200_000;

/// `ldr w1, [x0]; cbz w1, <ldr>; ldr x2, [x3]; svc #0`: spin until the u32
/// at x0 is non-zero, then load the u64 at x3 into x2.
const READER: [u32; 4] = [0xB940_0001, 0x34FF_FFE1, 0xF940_0062, 0xD400_0001];

struct Outcome {
    faults: u64,
    ok: u64,
}

/// Core 1 spins in guest code on a flag while this thread, acting as the
/// core whose SVC maps memory, maps a fresh page and fills it, then sets the
/// flag the way guest code after the SVC would.  Core 1 then reads the page.
/// With `guarded`, core 1 runs each slice under `execution_guard` and syncs
/// before it, as the runner's core loops do.
fn race(base: u64, iterations: u64, guarded: bool, slice: u64) -> Outcome {
    let space = Arc::new(AddressSpace::new());
    let code = base;
    let flag = base + PAGE_SIZE;
    let fresh = base + 0x100 * PAGE_SIZE;
    space.map(code, PAGE_SIZE, Perm::RX, "code").unwrap();
    let words: Vec<u8> = READER.iter().flat_map(|w| w.to_le_bytes()).collect();
    space.write(code, &words).unwrap();
    space.map(flag, PAGE_SIZE, Perm::RW, "flag").unwrap();

    let (go_tx, go_rx) = mpsc::channel::<Option<u64>>();
    let (done_tx, done_rx) = mpsc::channel::<Result<u64, u64>>();
    let in_run = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(AtomicU64::new(0));

    let reader = {
        let space = Arc::clone(&space);
        let in_run = Arc::clone(&in_run);
        let ready = Arc::clone(&ready);
        std::thread::spawn(move || {
            let mut cpu = Cpu::new(CpuBackendKind::Rustarmic).unwrap();
            cpu.set_continue_on_null(false);
            let mut generation = 0;
            while let Ok(Some(va)) = go_rx.recv() {
                cpu.set_pc(code);
                cpu.set_register(0, flag);
                cpu.set_register(3, va);
                cpu.set_register(2, 0);
                let result = loop {
                    let execution = guarded.then(|| space.execution_guard());
                    if space.generation() != generation {
                        sync_host_region_changes(&mut cpu, &space, &mut generation, 1).unwrap();
                    }
                    in_run.store(true, Ordering::SeqCst);
                    ready.fetch_add(1, Ordering::SeqCst);
                    let event = cpu.run_with_count(slice).event;
                    in_run.store(false, Ordering::SeqCst);
                    drop(execution);
                    match event {
                        CpuEvent::Svc(0) => break Ok(cpu.get_register(2)),
                        CpuEvent::Exception(0x0E) => {
                            break Err(cpu.take_fault().map(|f| f.addr).unwrap_or(0));
                        }
                        CpuEvent::Running => continue,
                        other => panic!("unexpected {other:?}"),
                    }
                };
                done_tx.send(result).unwrap();
            }
        })
    };

    let mut outcome = Outcome { faults: 0, ok: 0 };
    for i in 0..iterations {
        let va = fresh + (i % 0x40) * 2 * PAGE_SIZE;
        space.write(flag, &0u32.to_le_bytes()).unwrap();
        let before = ready.load(Ordering::SeqCst);
        go_tx.send(Some(va)).unwrap();
        while ready.load(Ordering::SeqCst) == before || !in_run.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        space.map(va, PAGE_SIZE, Perm::RW, "physmem").unwrap();
        let value = 0x5EED_0000_0000_0000 | i;
        space.write(va, &value.to_le_bytes()).unwrap();
        space.write(flag, &1u32.to_le_bytes()).unwrap();
        match done_rx.recv().unwrap() {
            Ok(read) => {
                assert_eq!(read, value, "iteration {i}");
                outcome.ok += 1;
            }
            Err(addr) => {
                assert_eq!(addr, va, "iteration {i}");
                outcome.faults += 1;
            }
        }
        space.unmap(va, PAGE_SIZE).unwrap();
    }
    go_tx.send(None).unwrap();
    reader.join().unwrap();
    outcome
}

#[test]
fn mapping_is_visible_on_a_core_already_in_guest_code() {
    let iterations = 2000;
    let outcome = race(0x9_5000_0000, iterations, true, SLICE);
    assert_eq!(outcome.faults, 0, "core 1 faulted on memory mapped before the flag was set");
    assert_eq!(outcome.ok, iterations);
}

/// The same race without the guard faults every time: the test reproduces
/// the bug rather than passing by luck.
#[test]
fn without_the_execution_guard_the_core_in_guest_code_faults() {
    let iterations = 50;
    let outcome = race(0x9_6000_0000, iterations, false, 1 << 40);
    assert_eq!(outcome.faults, iterations, "ok={}", outcome.ok);
}
