use nexium_cpu::Cpu;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use crate::kernel::cpu_local::current_core;

/// Number of guest CPU cores the scheduler tracks. Host threads drive cores
/// 0..N at runtime (just core 0 on the single-thread path).
pub const NUM_CORES: usize = 4;

#[derive(Copy, Clone, Debug)]
pub struct ThreadCtx {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub tpidrro_el0: u64,
}

impl ThreadCtx {
    pub fn zero() -> Self {
        Self { x: [0; 31], sp: 0, pc: 0, tpidrro_el0: 0 }
    }
}

#[derive(Clone, Debug)]
pub enum ThreadState {
    Created,
    Ready,
    Running,
    Sleeping { wake_at: Instant },
    WaitingHandle { handles: Vec<u32>, wake_at: Option<Instant> },
    WaitingMutex { mutex_addr: u64 },
    WaitingCondvar { mutex_addr: u64, condvar_addr: u64, wake_at: Option<Instant> },
    WaitingArbiter { addr: u64, value: u32, wake_at: Option<Instant> },
    Exited,
}

pub struct Thread {
    pub handle: u32,
    pub tid: u64,
    pub ctx: ThreadCtx,
    pub state: ThreadState,
    pub tls_va: u64,
    pub stack_top: u64,
    pub priority: i32,
    /// Ideal-core affinity from svcCreateThread (-2 = "any/default"). The shared
    /// ready queue lets any host core pick it up; honored once affinity matters.
    pub core: i32,
}

pub struct Threads {
    pub threads: HashMap<u32, Thread>,
    /// Currently-running guest thread per host core (indexed by current_core()).
    pub current: [Option<u32>; NUM_CORES],
    pub ready: VecDeque<u32>,
    pub next_tid: u64,
    pub next_tls_va: u64,
    pub tls_stride: u64,
    pub last_switch: Instant,
}

impl Threads {
    pub fn new(main_handle: u32, main_tls_va: u64, main_stack_top: u64, tls_pool_base: u64) -> Self {
        let mut threads = HashMap::new();
        threads.insert(
            main_handle,
            Thread {
                handle: main_handle,
                tid: 1,
                ctx: ThreadCtx::zero(),
                state: ThreadState::Running,
                tls_va: main_tls_va,
                stack_top: main_stack_top,
                priority: 0x2C,
                core: 0, // main thread pinned to core 0 (owns the boot/GPU mappings)
            },
        );
        Self {
            threads,
            current: {
                let mut c = [None; NUM_CORES];
                c[0] = Some(main_handle);
                c
            },
            ready: VecDeque::new(),
            next_tid: 2,
            next_tls_va: tls_pool_base,
            tls_stride: 0x1000,
            last_switch: Instant::now(),
        }
    }

    pub fn timeslice_expired(&self, threshold: std::time::Duration) -> bool {
        !self.ready.is_empty() && self.last_switch.elapsed() >= threshold
    }

    pub fn alloc_tls(&mut self) -> u64 {
        let va = self.next_tls_va;
        self.next_tls_va = self.next_tls_va.wrapping_add(self.tls_stride);
        va
    }

    pub fn alloc_tid(&mut self) -> u64 {
        let tid = self.next_tid;
        self.next_tid = self.next_tid.wrapping_add(1);
        tid
    }

    pub fn add_thread(&mut self, handle: u32, ctx: ThreadCtx, tls_va: u64, stack_top: u64) {
        let tid = self.alloc_tid();
        self.threads.insert(
            handle,
            Thread {
                handle,
                tid,
                ctx,
                state: ThreadState::Created,
                tls_va,
                stack_top,
                priority: 0x2C,
                core: -2,
            },
        );
    }

    pub fn current_handle(&self) -> Option<u32> {
        self.current[current_core()]
    }

    pub fn transition_state(&mut self, handle: u32, new_state: ThreadState) {
        self.ready.retain(|&h| h != handle);
        let became_ready = matches!(new_state, ThreadState::Ready);
        if let Some(t) = self.threads.get_mut(&handle) {
            t.state = new_state;
        }
        if became_ready && !self.ready.contains(&handle) {
            self.ready.push_back(handle);
        }
    }

    pub fn save_current_ctx(&mut self, cpu: &Cpu) {
        let Some(h) = self.current[current_core()] else { return };
        let Some(t) = self.threads.get_mut(&h) else { return };
        for i in 0..31 {
            t.ctx.x[i] = cpu.get_register(i as u32);
        }
        t.ctx.sp = cpu.get_register(31);
        t.ctx.pc = cpu.get_pc();
        t.ctx.tpidrro_el0 = cpu.get_tpidrro_el0();
    }

    pub fn load_thread(&self, handle: u32, cpu: &mut Cpu) {
        let Some(t) = self.threads.get(&handle) else { return };
        for i in 0..31 {
            cpu.set_register(i as u32, t.ctx.x[i]);
        }
        cpu.set_register(31, t.ctx.sp);
        cpu.set_pc(t.ctx.pc);
        cpu.set_tpidrro_el0(t.ctx.tpidrro_el0);
    }

    pub fn wake_due_sleepers(&mut self, now: Instant) {
        let mut woken = Vec::new();
        for (h, t) in self.threads.iter() {
            let due = match &t.state {
                ThreadState::Sleeping { wake_at } => *wake_at <= now,
                ThreadState::WaitingHandle { wake_at: Some(d), .. } => *d <= now,
                ThreadState::WaitingCondvar { wake_at: Some(d), .. } => *d <= now,
                ThreadState::WaitingArbiter { wake_at: Some(d), .. } => *d <= now,
                _ => false,
            };
            if due {
                woken.push(*h);
            }
        }
        for h in woken {
            if let Some(t) = self.threads.get_mut(&h) {
                match t.state {
                    ThreadState::WaitingHandle { .. }
                    | ThreadState::WaitingCondvar { .. }
                    | ThreadState::WaitingArbiter { .. } => {
                        t.ctx.x[0] = nexium_common::result::KERNEL_TIMEOUT as u64;
                    }
                    _ => {}
                }
            }
            self.transition_state(h, ThreadState::Ready);
        }
    }

    pub fn wake_one_on_mutex(&mut self, mutex_addr: u64) -> Option<u32> {
        let h = self.threads.iter().find_map(|(h, t)| match &t.state {
            ThreadState::WaitingMutex { mutex_addr: m } if *m == mutex_addr => Some(*h),
            _ => None,
        })?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn wake_one_on_condvar(&mut self, condvar_addr: u64) -> Option<u32> {
        let h = self.threads.iter().find_map(|(h, t)| match &t.state {
            ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr => Some(*h),
            _ => None,
        })?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn peek_one_condvar_waiter(&self, condvar_addr: u64) -> Option<(u32, u64)> {
        self.threads.iter().find_map(|(h, t)| match &t.state {
            ThreadState::WaitingCondvar { condvar_addr: c, mutex_addr, .. } if *c == condvar_addr => {
                Some((*h, *mutex_addr))
            }
            _ => None,
        })
    }

    pub fn wake_condvar_into_mutex_waiter(&mut self, handle: u32, mutex_addr: u64) {
        if let Some(t) = self.threads.get_mut(&handle) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(handle, ThreadState::WaitingMutex { mutex_addr });
    }

    pub fn wake_condvar_to_ready(&mut self, handle: u32) {
        if let Some(t) = self.threads.get_mut(&handle) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(handle, ThreadState::Ready);
    }

    pub fn has_mutex_waiters(&self, mutex_addr: u64) -> bool {
        self.threads.values().any(|t| matches!(&t.state, ThreadState::WaitingMutex { mutex_addr: m } if *m == mutex_addr))
    }

    pub fn wake_all_on_condvar(&mut self, condvar_addr: u64) -> usize {
        let mut woken: Vec<u32> = Vec::new();
        for (h, t) in self.threads.iter_mut() {
            if let ThreadState::WaitingCondvar { condvar_addr: c, .. } = &t.state {
                if *c == condvar_addr {
                    t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
                    woken.push(*h);
                }
            }
        }
        let count = woken.len();
        for h in woken {
            self.transition_state(h, ThreadState::Ready);
        }
        count
    }

    pub fn signal_handle(&mut self, handle: u32) {
        let mut woken: Vec<u32> = Vec::new();
        for (h, t) in self.threads.iter_mut() {
            if let ThreadState::WaitingHandle { handles, .. } = &t.state {
                if handles.contains(&handle) {
                    t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
                    if let Some(idx) = handles.iter().position(|&v| v == handle) {
                        t.ctx.x[1] = idx as u64;
                    }
                    woken.push(*h);
                }
            }
        }
        for h in woken {
            self.transition_state(h, ThreadState::Ready);
        }
    }

    pub fn wake_one_on_arbiter(&mut self, addr: u64) -> Option<u32> {
        let h = self.threads.iter().find_map(|(h, t)| match &t.state {
            ThreadState::WaitingArbiter { addr: a, .. } if *a == addr => Some(*h),
            _ => None,
        })?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn wake_all_on_arbiter(&mut self, addr: u64) -> usize {
        let mut woken: Vec<u32> = Vec::new();
        for (h, t) in self.threads.iter_mut() {
            if let ThreadState::WaitingArbiter { addr: a, .. } = &t.state {
                if *a == addr {
                    t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
                    woken.push(*h);
                }
            }
        }
        let count = woken.len();
        for h in woken {
            self.transition_state(h, ThreadState::Ready);
        }
        count
    }

    pub fn pick_next(&mut self) -> Option<u32> {
        // Sticky affinity: a core only picks ready threads pinned to it, or
        // unassigned (core < 0) threads which it then claims. Once a thread runs
        // on a core it never migrates, so its context and dynamic memory
        // mappings stay coherent on that core's Cpu (no cross-core ctx race).
        let core = current_core() as i32;
        let pos = self
            .ready
            .iter()
            .position(|h| self.threads.get(h).map_or(false, |t| t.core == core || t.core < 0))?;
        let handle = self.ready.remove(pos)?;
        if let Some(t) = self.threads.get_mut(&handle) {
            if t.core < 0 {
                t.core = core;
            }
        }
        Some(handle)
    }

    pub fn yield_current(&mut self, cpu: &Cpu) {
        if self.current[current_core()].is_some() {
            self.save_current_ctx(cpu);
            self.current[current_core()] = None;
        }
    }

    pub fn yield_with_state(&mut self, cpu: &Cpu, new_state: ThreadState) -> Option<u32> {
        let h = self.current[current_core()]?;
        self.save_current_ctx(cpu);
        let became_ready = matches!(new_state, ThreadState::Ready);
        if let Some(t) = self.threads.get_mut(&h) {
            t.state = new_state;
        }
        if became_ready && !self.ready.contains(&h) {
            self.ready.push_back(h);
        }
        self.current[current_core()] = None;
        self.last_switch = Instant::now();
        Some(h)
    }

    pub fn schedule_to(&mut self, handle: u32, cpu: &mut Cpu) {
        self.load_thread(handle, cpu);
        if let Some(t) = self.threads.get_mut(&handle) {
            t.state = ThreadState::Running;
            log::info!("[sched] core{} now running handle={:#x} pc={:#x} sp={:#x}", current_core(), handle, t.ctx.pc, t.ctx.sp);
        }
        self.current[current_core()] = Some(handle);
        self.last_switch = Instant::now();
    }

    pub fn earliest_wake(&self) -> Option<Instant> {
        self.threads
            .values()
            .filter_map(|t| match &t.state {
                ThreadState::Sleeping { wake_at } => Some(*wake_at),
                ThreadState::WaitingHandle { wake_at: Some(d), .. } => Some(*d),
                ThreadState::WaitingCondvar { wake_at: Some(d), .. } => Some(*d),
                ThreadState::WaitingArbiter { wake_at: Some(d), .. } => Some(*d),
                _ => None,
            })
            .min()
    }

    pub fn drop_exited(&mut self) {
        self.threads.retain(|_, t| !matches!(t.state, ThreadState::Exited));
    }

    pub fn ensure_thread_loaded(&mut self, cpu: &mut Cpu) -> Option<u32> {
        if let Some(h) = self.current[current_core()] {
            if let Some(t) = self.threads.get(&h) {
                if matches!(t.state, ThreadState::Running) {
                    return Some(h);
                }
            }
            self.current[current_core()] = None;
        }
        self.wake_due_sleepers(Instant::now());
        let next = self.pick_next()?;
        self.schedule_to(next, cpu);
        Some(next)
    }
}

impl Default for Threads {
    fn default() -> Self {
        Self::new(0, 0, 0, 0)
    }
}
