use std::alloc::Layout;
use std::time::Instant;

use nexium_cpu::{Cpu, CpuEvent};
use nexium_memory::Perm;

pub type Check = fn() -> Result<String, String>;

pub const CHECKS: &[(&str, Check)] = &[("dynarmic-a64", dynarmic_a64), ("dynarmic-map-cost", dynarmic_map_cost)];

fn dynarmic_map_cost() -> Result<String, String> {
    let bytes = 1536usize << 20;
    let started = Instant::now();
    let region = HostPage::new(bytes);
    let alloc_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut cpu = Cpu::new_dynarmic()?;
    let create_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut detail = vec![format!("alloc 1.5GiB zeroed {alloc_ms:.0}ms, cpu create {create_ms:.0}ms")];
    for (label, size) in [("64MiB", 64usize << 20), ("1.5GiB", bytes)] {
        let started = Instant::now();
        unsafe { cpu.map_host(DATA_VA + (8u64 << 30) * (label.len() as u64), size as u64, Perm::RW, region.ptr)? };
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        detail.push(format!("map {label}: {ms:.0}ms ({:.2}us/page)", ms * 1000.0 / (size / 4096) as f64));
    }
    Ok(detail.join("; "))
}

const CODE_VA: u64 = 0x80_0000_0000;
const DATA_VA: u64 = 0x90_0000_0000;
const PROGRAM: [u32; 7] = [
    0xd280_0000,
    0x8b01_0000,
    0xf100_0421,
    0x54ff_ffc1,
    0xf900_0040,
    0xf940_0043,
    0xd400_00e1,
];

struct HostPage {
    ptr: *mut u8,
    layout: Layout,
}

impl HostPage {
    fn new(bytes: usize) -> Self {
        let layout = Layout::from_size_align(bytes, 0x4000).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Self { ptr, layout }
    }
}

impl Drop for HostPage {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

fn run_until_svc(cpu: &mut Cpu, budget: u64) -> Result<(u16, u64), String> {
    let mut retired = 0u64;
    for _ in 0..10_000_000 {
        let result = cpu.run_with_count(budget);
        retired += result.retired;
        match result.event {
            CpuEvent::Svc(imm) => return Ok((imm, retired)),
            CpuEvent::Exception(code) => return Err(format!("guest exception {code:#x} at pc {:#x}", cpu.get_pc())),
            _ => {}
        }
    }
    Err(format!("no svc after budget loop, pc {:#x}", cpu.get_pc()))
}

fn dynarmic_a64() -> Result<String, String> {
    let code = HostPage::new(0x4000);
    let data = HostPage::new(0x4000);
    for (i, word) in PROGRAM.iter().enumerate() {
        unsafe { (code.ptr as *mut u32).add(i).write(*word) };
    }
    let started = Instant::now();
    let mut cpu = Cpu::new_dynarmic()?;
    let create_ms = started.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        cpu.map_host(CODE_VA, 0x4000, Perm::RX, code.ptr)?;
        cpu.map_host(DATA_VA, 0x4000, Perm::RW, data.ptr)?;
    }
    let mut detail = vec![format!("create={create_ms:.1}ms")];
    for n in [1000u64, 100_000_000] {
        cpu.set_pc(CODE_VA);
        cpu.set_register(1, n);
        cpu.set_register(2, DATA_VA);
        let started = Instant::now();
        let (imm, retired) = run_until_svc(&mut cpu, 1 << 30)?;
        let secs = started.elapsed().as_secs_f64();
        let x0 = cpu.get_register(0);
        let x3 = cpu.get_register(3);
        let stored = unsafe { (data.ptr as *const u64).read_volatile() };
        let expected = n * (n + 1) / 2;
        if imm != 7 || x0 != expected || x3 != expected || stored != expected {
            return Err(format!("n={n}: svc={imm} x0={x0} x3={x3} mem={stored} expected={expected}"));
        }
        let instructions = 3 * n + 5;
        detail.push(format!(
            "n={n}: ok in {:.2}ms ({:.0} M guest insn/s, retired={retired})",
            secs * 1000.0,
            instructions as f64 / secs / 1e6
        ));
    }
    Ok(detail.join("; "))
}
