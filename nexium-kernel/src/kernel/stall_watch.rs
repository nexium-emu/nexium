use parking_lot::Mutex;
use std::time::{Duration, Instant};

pub const SLOW_SVC: Duration = Duration::from_millis(25);
pub const SLOW_LOCK_WAIT: Duration = Duration::from_millis(25);

const WINDOW: Duration = Duration::from_secs(10);
const LINES_PER_WINDOW: u32 = 30;

struct Budget {
    since: Option<Instant>,
    used: u32,
    dropped: u32,
}

static BUDGET: Mutex<Budget> = Mutex::new(Budget {
    since: None,
    used: 0,
    dropped: 0,
});

fn admit() -> bool {
    let mut budget = BUDGET.lock();
    let now = Instant::now();
    if budget.since.map_or(true, |since| now.duration_since(since) >= WINDOW) {
        if budget.dropped > 0 {
            log::info!("[stall] {} more slow kernel events in the last {}s were not logged", budget.dropped, WINDOW.as_secs());
        }
        budget.since = Some(now);
        budget.used = 0;
        budget.dropped = 0;
    }
    if budget.used >= LINES_PER_WINDOW {
        budget.dropped += 1;
        return false;
    }
    budget.used += 1;
    true
}

pub fn is_ipc_svc(imm: u16) -> bool {
    matches!(imm, 0x21..=0x23)
}

pub fn slow_svc(imm: u16, held: Duration, ipc: Option<(&str, u32, u32)>) {
    if held < SLOW_SVC || !admit() {
        return;
    }
    match ipc {
        Some((target, cmd, detail)) if detail != 0 => log::info!(
            "[stall] svc {:#04x} to {} cmd {} ({:#010x}) held the kernel for {} ms",
            imm,
            target,
            cmd,
            detail,
            held.as_millis()
        ),
        Some((target, cmd, _)) => log::info!(
            "[stall] svc {:#04x} to {} cmd {} held the kernel for {} ms",
            imm,
            target,
            cmd,
            held.as_millis()
        ),
        None => log::info!("[stall] svc {:#04x} held the kernel for {} ms", imm, held.as_millis()),
    }
}

pub fn lock_wait(who: &str, waited: Duration) {
    if waited < SLOW_LOCK_WAIT || !admit() {
        return;
    }
    log::info!("[stall] {} waited {} ms for the kernel", who, waited.as_millis());
}

pub fn off_lock_drain(ioctl_id: u32, waited: Duration) {
    if waited < SLOW_SVC || !admit() {
        return;
    }
    log::info!(
        "[stall] GPU drain for ioctl {:#010x} ran outside the kernel for {} ms",
        ioctl_id,
        waited.as_millis()
    );
}

pub fn late_tick(who: &str, late: Duration) {
    if late < SLOW_LOCK_WAIT || !admit() {
        return;
    }
    log::info!("[stall] {} thread woke {} ms late (host CPU busy)", who, late.as_millis());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_ipc_svcs_carry_a_service_name() {
        assert!(is_ipc_svc(0x21));
        assert!(is_ipc_svc(0x22));
        assert!(!is_ipc_svc(0x18));
    }
}
