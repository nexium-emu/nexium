use std::sync::OnceLock;
use std::sync::RwLock;
use std::time::{Duration, Instant};

type WakeHook = std::sync::Arc<dyn Fn() + Send + Sync>;

static HOOK: RwLock<Option<WakeHook>> = RwLock::new(None);

pub fn install(hook: WakeHook) {
    *HOOK.write().unwrap() = Some(hook);
}

pub fn signal() {
    let hook = HOOK.read().unwrap().clone();
    if let Some(hook) = hook {
        hook();
    }
}

pub fn mwaitx_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        if std::env::var_os("NEXIUM_NO_MWAITX").is_some() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            let leaf = std::arch::x86_64::__cpuid(0x8000_0001);
            (leaf.ecx & (1 << 29)) != 0
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

#[cfg(target_arch = "x86_64")]
unsafe fn monitorx(addr: *const u8) {
    std::arch::asm!(
        ".byte 0x0f, 0x01, 0xfa",
        in("rax") addr,
        in("ecx") 0u32,
        in("edx") 0u32,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
unsafe fn mwaitx(tsc_timeout: u32) {
    std::arch::asm!(
        "mov {saved_rbx}, rbx",
        "mov ebx, {timeout:e}",
        ".byte 0x0f, 0x01, 0xfb",
        "mov rbx, {saved_rbx}",
        saved_rbx = out(reg) _,
        timeout = in(reg) tsc_timeout,
        in("eax") 0u32,
        in("ecx") 2u32,
        options(nostack)
    );
}

const MWAITX_TSC_TIMEOUT: u32 = 100_000;

pub fn micro_pause() {
    #[cfg(target_arch = "x86_64")]
    if mwaitx_supported() {
        let sentinel = 0u64;
        unsafe {
            monitorx(&sentinel as *const u64 as *const u8);
            mwaitx(MWAITX_TSC_TIMEOUT);
        }
        return;
    }
    for _ in 0..64 {
        std::hint::spin_loop();
    }
}

pub fn wait_until(deadline: Instant) {
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining > Duration::from_micros(60) {
            micro_pause();
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn signal_without_hook_is_noop_and_hook_fires_after_install() {
        signal();
        static FIRED: AtomicUsize = AtomicUsize::new(0);
        install(std::sync::Arc::new(|| {
            FIRED.fetch_add(1, Ordering::SeqCst);
        }));
        signal();
        signal();
        assert!(FIRED.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn micro_pause_returns_promptly() {
        let start = Instant::now();
        for _ in 0..8 {
            micro_pause();
        }
        assert!(start.elapsed() < Duration::from_millis(250));
    }

    #[test]
    fn wait_until_reaches_deadline() {
        let deadline = Instant::now() + Duration::from_micros(300);
        wait_until(deadline);
        assert!(Instant::now() >= deadline);
    }
}
