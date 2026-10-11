#![cfg(feature = "backend-rustarmic")]

use nexium_cpu::{rustarmic::RustarmicCpu, CpuEvent};
use nexium_memory::Perm;

const CODE: u64 = 0x10_0000;
const DATA: u64 = 0x20_0000;
const SVC: u32 = 0xd400_0fc1;

struct Fixture {
    cpu: RustarmicCpu,
    _code: Box<[u8; 0x1000]>,
}

impl Fixture {
    fn new(words: &[u32]) -> Self {
        let mut code = Box::new([0; 0x1000]);
        for (index, word) in words.iter().enumerate() {
            code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut cpu = RustarmicCpu::new().unwrap();
        unsafe {
            cpu.map_host(CODE, 0x1000, Perm::RX, code.as_mut_ptr())
                .unwrap();
        }
        Self { cpu, _code: code }
    }

    fn run(&mut self) -> CpuEvent {
        self.cpu.set_pc(CODE);
        self.cpu.run_with_count(1000).0
    }

    fn svc(&mut self) {
        let event = self.run();
        assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
    }
}

#[test]
fn overlapping_maps_keep_first_match_even_after_accessing_later_map() {
    let mut first = [11u64, 12];
    let mut later = [21u64, 22, 23, 24];
    let first_before = first;
    let mut fixture = Fixture::new(&[
        0xf940_0020,
        0xf940_0423,
        0xf900_0829,
        SVC,
    ]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA + 8, 16, Perm::RW, first.as_mut_ptr().cast())
            .unwrap();
        fixture
            .cpu
            .map_host(DATA, 32, Perm::RW, later.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, DATA);
    fixture.cpu.set_register(9, 37);
    fixture.svc();
    assert_eq!(fixture.cpu.get_register(0), 21);
    assert_eq!(fixture.cpu.get_register(3), first_before[0]);
    assert_eq!(first, [11, 37]);
    assert_eq!(later, [21, 22, 23, 24]);
}

#[test]
fn overlapping_first_match_read_permission_cannot_be_bypassed() {
    let mut denied = [11u64];
    let mut later = [21u64, 22, 23, 24];
    let mut fixture = Fixture::new(&[0xf940_0020, 0xf940_0423, SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA + 8, 8, Perm::W, denied.as_mut_ptr().cast())
            .unwrap();
        fixture
            .cpu
            .map_host(DATA, 32, Perm::RW, later.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, DATA);
    assert!(matches!(fixture.run(), CpuEvent::Exception(0x0e)));
    let fault = fixture.cpu.take_fault().unwrap();
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (CODE + 4, DATA + 8, 8, false)
    );
    assert_eq!(denied, [11]);
    assert_eq!(later, [21, 22, 23, 24]);
}

#[test]
fn overlapping_first_match_write_permission_cannot_be_bypassed() {
    let mut denied = [11u64];
    let mut later = [21u64, 22, 23, 24];
    let mut fixture = Fixture::new(&[0xf940_0020, 0xf900_0429, SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA + 8, 8, Perm::RO, denied.as_mut_ptr().cast())
            .unwrap();
        fixture
            .cpu
            .map_host(DATA, 32, Perm::RW, later.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, DATA);
    fixture.cpu.set_register(9, 37);
    assert!(matches!(fixture.run(), CpuEvent::Exception(0x0e)));
    let fault = fixture.cpu.take_fault().unwrap();
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (CODE + 4, DATA + 8, 8, true)
    );
    assert_eq!(denied, [11]);
    assert_eq!(later, [21, 22, 23, 24]);
}

#[test]
fn removal_and_remapping_reset_indices_and_recover_disjoint_lookup() {
    let mut first = [11u64, 12];
    let mut later = [21u64, 22, 23, 24];
    let mut replacement = [41u64, 42];
    let mut fixture = Fixture::new(&[0xf940_0020, SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA + 8, 16, Perm::RW, first.as_mut_ptr().cast())
            .unwrap();
        fixture
            .cpu
            .map_host(DATA, 32, Perm::RW, later.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, DATA);
    fixture.svc();
    assert_eq!(fixture.cpu.get_register(0), 21);
    unsafe {
        fixture.cpu.unmap_host(DATA, 8).unwrap();
    }
    fixture.cpu.set_register(1, DATA + 8);
    fixture.svc();
    assert_eq!(fixture.cpu.get_register(0), 11);
    unsafe {
        fixture.cpu.unmap_host(DATA + 8, 16).unwrap();
        fixture
            .cpu
            .map_host(DATA + 8, 16, Perm::RW, replacement.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.svc();
    assert_eq!(fixture.cpu.get_register(0), 41);
    assert_eq!(first, [11, 12]);
    assert_eq!(later, [21, 22, 23, 24]);
    assert_eq!(replacement, [41, 42]);
}

#[test]
fn five_hot_disjoint_maps_preserve_values_through_cache_eviction() {
    let mut values = [11u64, 22, 33, 44, 55];
    let mut words = Vec::new();
    for index in 0..5 {
        words.push(0xf940_0000 | ((index + 1) << 5) | (index + 9));
    }
    for index in (0..5).rev() {
        words.push(0xf940_0000 | ((index + 1) << 5) | (index + 14));
    }
    words.extend([0xf900_0074, SVC]);
    let mut fixture = Fixture::new(&words);
    for index in 0..5 {
        let address = DATA + index as u64 * 0x1000;
        unsafe {
            fixture
                .cpu
                .map_host(address, 8, Perm::RW, values.as_mut_ptr().add(index).cast())
                .unwrap();
        }
        fixture.cpu.set_register(index as u32 + 1, address);
    }
    fixture.cpu.set_register(20, 99);
    fixture.svc();
    for index in 0..5 {
        assert_eq!(fixture.cpu.get_register(index + 9), (index as u64 + 1) * 11);
        assert_eq!(
            fixture.cpu.get_register(index + 14),
            (index as u64 + 1) * 11
        );
    }
    assert_eq!(values, [11, 22, 99, 44, 55]);
}

#[test]
fn cached_region_still_checks_exact_width_and_cross_region_copy() {
    let mut first = 0x0807_0605_0403_0201u64;
    let mut second = 0x100f_0e0d_0c0b_0a09u64;
    let mut fixture = Fixture::new(&[0xf940_0020, 0xf940_0043, SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(DATA, 8, Perm::RW, (&mut first as *mut u64).cast())
            .unwrap();
        fixture
            .cpu
            .map_host(DATA + 8, 8, Perm::RO, (&mut second as *mut u64).cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, DATA);
    fixture.cpu.set_register(2, DATA + 4);
    fixture.svc();
    assert_eq!(fixture.cpu.get_register(0), first);
    assert_eq!(fixture.cpu.get_register(3), 0x0c0b_0a09_0807_0605);
    unsafe {
        fixture.cpu.unmap_host(DATA + 8, 8).unwrap();
    }
    assert!(matches!(fixture.run(), CpuEvent::Exception(0x0e)));
    let fault = fixture.cpu.take_fault().unwrap();
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (CODE + 4, DATA + 8, 8, false)
    );
    assert_eq!(first, 0x0807_0605_0403_0201);
    assert_eq!(second, 0x100f_0e0d_0c0b_0a09);
}

#[test]
fn cached_region_cannot_bypass_address_end_overflow() {
    let mut values = [11u64, 22];
    let base = u64::MAX - 15;
    let mut fixture = Fixture::new(&[0xf940_0020, 0xf940_0043, SVC]);
    unsafe {
        fixture
            .cpu
            .map_host(base, 15, Perm::RW, values.as_mut_ptr().cast())
            .unwrap();
    }
    fixture.cpu.set_register(1, base);
    fixture.cpu.set_register(2, base + 8);
    assert!(matches!(fixture.run(), CpuEvent::Exception(0x0e)));
    let fault = fixture.cpu.take_fault().unwrap();
    assert_eq!(
        (fault.pc, fault.addr, fault.size, fault.is_write),
        (CODE + 4, base + 8, 8, false)
    );
    assert_eq!(values, [11, 22]);
}

#[test]
fn cached_exclusive_stores_keep_exact_width_and_canaries() {
    for (bytes, size_bits) in [(1usize, 0u32), (2, 1 << 30), (4, 2 << 30), (8, 3 << 30)] {
        let mut backing = [0x1122_3344_5566_7788u64, 0x8877_6655_4433_2211];
        let before = backing;
        let mut padding = [0u64; 8];
        let mut fixture = Fixture::new(&[
            0x085f_7c01 | size_bits,
            0x0802_7c03 | size_bits,
            SVC,
        ]);
        for index in 0..padding.len() {
            unsafe {
                fixture
                    .cpu
                    .map_host(
                        DATA + 0x1000 + index as u64 * 0x1000,
                        8,
                        Perm::RW,
                        padding.as_mut_ptr().add(index).cast(),
                    )
                    .unwrap();
            }
        }
        unsafe {
            fixture
                .cpu
                .map_host(DATA, 16, Perm::RW, backing.as_mut_ptr().cast())
                .unwrap();
        }
        fixture.cpu.set_register(0, DATA);
        fixture.cpu.set_register(3, 0xfedc_ba98_7654_3210);
        fixture.svc();
        let mask = if bytes == 8 {
            u64::MAX
        } else {
            (1u64 << (bytes * 8)) - 1
        };
        assert_eq!(fixture.cpu.get_register(1), before[0] & mask);
        assert_eq!(fixture.cpu.get_register(2), 0);
        assert_eq!(
            backing,
            [
                (before[0] & !mask) | (0xfedc_ba98_7654_3210 & mask),
                before[1]
            ]
        );
        assert_eq!(padding, [0; 8]);
    }
}
