#![cfg(feature = "backend-rustarmic")]
use nexium_cpu::{Cpu, CpuEvent};
use nexium_memory::Perm;

const CODE_BASE: u64 = 0x10_0000;

fn run_svc(imm: u16) -> CpuEvent {
    let mut code = vec![0u8; 0x1000];
    let svc = 0xD400_0001u32 | ((imm as u32) << 5);
    code[0..4].copy_from_slice(&svc.to_le_bytes());
    code[4..8].copy_from_slice(&0xD420_0000u32.to_le_bytes());

    let mut cpu = Cpu::new_rustarmic().expect("rustarmic init");
    unsafe {
        cpu.map_host(CODE_BASE, 0x1000, Perm::R | Perm::X, code.as_mut_ptr())
            .expect("map_host");
    }
    cpu.set_pc(CODE_BASE);
    cpu.run(100_000).expect("CPU run").event
}

#[test]
fn svc_carries_immediate_0x1c() {
    assert!(matches!(run_svc(0x1c), CpuEvent::Svc(0x1c)));
}

#[test]
fn svc_carries_immediate_zero() {
    assert!(matches!(run_svc(0), CpuEvent::Svc(0)));
}

#[test]
fn svc_carries_large_immediate() {
    assert!(matches!(run_svc(0xABCD), CpuEvent::Svc(0xABCD)));
}

#[test]
fn register_roundtrip() {
    let mut cpu = Cpu::new_rustarmic().expect("rustarmic init");
    cpu.set_register(0, 0xDEAD_BEEF_CAFE_BABE);
    cpu.set_register(5, 0x1234_5678_9ABC_DEF0);
    cpu.set_pc(0x4000);
    cpu.set_sp(0x8000);
    cpu.set_tpidrro_el0(0xB0_0000_1000);
    assert_eq!(cpu.get_register(0), 0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(cpu.get_register(5), 0x1234_5678_9ABC_DEF0);
    assert_eq!(cpu.get_pc(), 0x4000);
    assert_eq!(cpu.get_sp(), 0x8000);
    assert_eq!(cpu.get_tpidrro_el0(), 0xB0_0000_1000);
}

#[test]
fn halt_handle_survives_cpu_drop() {
    let handle = {
        let cpu = Cpu::new_rustarmic().expect("rustarmic init");
        cpu.halt_handle()
    };
    handle.halt();
    assert_eq!(handle.peek_pc_lr_sp(), (0, 0, 0));
}

#[test]
fn arithmetic_through_jit() {
    let mut code = vec![0u8; 0x1000];
    let prog = [0xD280_0C80u32, 0x9100_C800u32, 0xD400_0001u32];
    for (i, w) in prog.iter().enumerate() {
        code[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let mut cpu = Cpu::new_rustarmic().expect("init");
    unsafe {
        cpu.map_host(CODE_BASE, 0x1000, Perm::R | Perm::X, code.as_mut_ptr())
            .unwrap();
    }
    cpu.set_pc(CODE_BASE);
    let event = cpu.run(100_000).expect("CPU run").event;
    assert!(
        matches!(event, CpuEvent::Svc(0)),
        "expected Svc(0), got {:?}",
        event
    );
    assert_eq!(cpu.get_register(0), 150);
}
