use nexium_cpu::Cpu;
use std::cell::Cell;

thread_local! {
    static CURRENT_CPU: Cell<*mut Cpu> = const { Cell::new(std::ptr::null_mut()) };
}

#[must_use]
pub struct CpuGuard(());

impl Drop for CpuGuard {
    fn drop(&mut self) {
        CURRENT_CPU.with(|c| c.set(std::ptr::null_mut()));
    }
}

pub fn set_current_cpu(cpu: &mut Cpu) -> CpuGuard {
    CURRENT_CPU.with(|c| c.set(cpu as *mut Cpu));
    CpuGuard(())
}

#[allow(clippy::mut_from_ref)]
pub fn cpu_mut() -> Option<&'static mut Cpu> {
    CURRENT_CPU.with(|c| {
        let p = c.get();
        if p.is_null() {
            None
        } else {
            Some(unsafe { &mut *p })
        }
    })
}

pub fn cpu_ref() -> Option<&'static Cpu> {
    CURRENT_CPU.with(|c| {
        let p = c.get();
        if p.is_null() {
            None
        } else {
            Some(unsafe { &*p })
        }
    })
}
