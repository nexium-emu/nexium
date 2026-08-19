use crate::kernel::cpu_local::current_core;
use nexium_cpu::Cpu;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

pub const NUM_CORES: usize = 4;

pub const HOS_PREEMPTION_PRIORITIES: [i32; NUM_CORES] = [59, 59, 59, 63];

pub struct CoreWakers {
    condvars: [parking_lot::Condvar; NUM_CORES],
}

impl CoreWakers {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            condvars: std::array::from_fn(|_| parking_lot::Condvar::new()),
        })
    }

    pub fn notify_core(&self, core: i32) {
        if (0..NUM_CORES as i32).contains(&core) {
            self.condvars[core as usize].notify_one();
        } else {
            self.notify_all_cores();
        }
    }

    pub fn notify_all_cores(&self) {
        for condvar in &self.condvars {
            condvar.notify_one();
        }
    }

    pub fn park_core<T>(
        &self,
        guard: &mut parking_lot::MutexGuard<'_, T>,
        core: usize,
        timeout: std::time::Duration,
    ) {
        if core < NUM_CORES {
            self.condvars[core].wait_for(guard, timeout);
        }
    }
}

pub const IDEAL_CORE_DONT_CARE: i32 = -1;
pub const IDEAL_CORE_USE_PROCESS_VALUE: i32 = -2;
pub const IDEAL_CORE_NO_UPDATE: i32 = -3;

pub fn resolve_create_thread_core(requested: i32, process_ideal_core: i32) -> Result<i32, u32> {
    let ideal = if requested == IDEAL_CORE_USE_PROCESS_VALUE {
        process_ideal_core
    } else {
        requested
    };
    if (0..NUM_CORES as i32).contains(&ideal) {
        Ok(ideal)
    } else {
        Err(nexium_common::result::KERNEL_INVALID_CORE_ID)
    }
}

#[derive(Clone, Debug)]
pub struct ThreadCtx {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub tpidrro_el0: u64,
    pub backend: Option<nexium_cpu::CpuThreadContext>,
}

impl ThreadCtx {
    pub fn zero() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            tpidrro_el0: 0,
            backend: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ThreadState {
    Created,
    Ready,
    Running,
    Sleeping {
        wake_at: Instant,
    },
    WaitingHandle {
        handles: Vec<u32>,
        wake_at: Option<Instant>,
    },
    WaitingMutex {
        mutex_addr: u64,
        owner_handle: u32,
        tag: u32,
    },
    WaitingCondvar {
        mutex_addr: u64,
        condvar_addr: u64,
        wake_at: Option<Instant>,
        spurious_wake: bool,
    },
    WaitingArbiter {
        addr: u64,
        value: u32,
        wake_at: Option<Instant>,
    },
    Exited,
}

pub struct Thread {
    pub handle: u32,
    pub tid: u64,
    pub ctx: ThreadCtx,
    pub state: ThreadState,
    pub tls_va: u64,
    pub stack_top: u64,
    pub entry_arg: u64,
    pub priority: i32,
    pub core: i32,
    pub ideal_core: i32,
    pub affinity_mask: u64,
    pub wait_cancelled: bool,
    user_preemption_pending: bool,
}

pub struct Threads {
    pub threads: HashMap<u32, Thread>,
    pub current: [Option<u32>; NUM_CORES],
    pub ready: VecDeque<u32>,
    pub next_tid: u64,
    pub next_tls_va: u64,
    pub tls_stride: u64,
    pub tls_pool_base: u64,
    pub free_tls: Vec<u64>,
    pub last_switch: Instant,
    per_core_last_switch: [Instant; NUM_CORES],
    pub wakers: Arc<CoreWakers>,
}

impl Threads {
    pub fn new(
        main_handle: u32,
        main_tls_va: u64,
        main_stack_top: u64,
        tls_pool_base: u64,
    ) -> Self {
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
                entry_arg: 0,
                priority: 0x2C,
                core: 0,
                ideal_core: 0,
                affinity_mask: 1,
                wait_cancelled: false,
                user_preemption_pending: false,
            },
        );
        let now = Instant::now();
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
            tls_pool_base,
            free_tls: Vec::new(),
            last_switch: now,
            per_core_last_switch: [now; NUM_CORES],
            wakers: CoreWakers::new(),
        }
    }

    pub fn timeslice_expired(&self, threshold: std::time::Duration) -> bool {
        !self.ready.is_empty() && self.last_switch.elapsed() >= threshold
    }

    pub fn hos_timeslice_expired(&self, threshold: std::time::Duration) -> bool {
        self.hos_timeslice_expired_on_core(current_core(), threshold)
    }

    fn hos_timeslice_expired_on_core(&self, core: usize, threshold: std::time::Duration) -> bool {
        let Some(Some(handle)) = self.current.get(core) else {
            return false;
        };
        self.effective_priority(*handle) == HOS_PREEMPTION_PRIORITIES[core]
            && self.has_ready_for_core(core as i32)
            && self.per_core_last_switch[core].elapsed() >= threshold
    }

    fn record_switch(&mut self) {
        let now = Instant::now();
        let core = current_core();
        self.last_switch = now;
        self.per_core_last_switch[core] = now;
    }

    pub fn alloc_tls(&mut self) -> u64 {
        if let Some(va) = self.free_tls.pop() {
            return va;
        }
        let va = self.next_tls_va;
        self.next_tls_va = self.next_tls_va.wrapping_add(self.tls_stride);
        va
    }

    pub fn alloc_tid(&mut self) -> u64 {
        let tid = self.next_tid;
        self.next_tid = self.next_tid.wrapping_add(1);
        tid
    }

    pub fn add_thread(
        &mut self,
        handle: u32,
        ctx: ThreadCtx,
        tls_va: u64,
        stack_top: u64,
        entry_arg: u64,
    ) {
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
                entry_arg,
                priority: 0x2C,
                core: -2,
                ideal_core: IDEAL_CORE_USE_PROCESS_VALUE,
                affinity_mask: 0,
                wait_cancelled: false,
                user_preemption_pending: false,
            },
        );
    }

    pub fn current_handle(&self) -> Option<u32> {
        self.current[current_core()]
    }

    pub fn current_user_preemption_pending(&self) -> bool {
        self.current_handle()
            .and_then(|handle| self.threads.get(&handle))
            .is_some_and(|thread| thread.user_preemption_pending)
    }

    pub fn mark_current_user_preemption_pending(&mut self) -> bool {
        let Some(handle) = self.current_handle() else {
            return false;
        };
        let Some(thread) = self.threads.get_mut(&handle) else {
            return false;
        };
        let was_pending = thread.user_preemption_pending;
        thread.user_preemption_pending = true;
        !was_pending
    }

    pub fn take_current_user_preemption_pending(&mut self) -> bool {
        let Some(handle) = self.current_handle() else {
            return false;
        };
        self.threads
            .get_mut(&handle)
            .is_some_and(|thread| std::mem::take(&mut thread.user_preemption_pending))
    }

    pub fn transition_state(&mut self, handle: u32, new_state: ThreadState) {
        self.ready.retain(|&h| h != handle);
        let Some(t) = self.threads.get_mut(&handle) else {
            return;
        };
        let became_ready = matches!(new_state, ThreadState::Ready);
        let exited = matches!(new_state, ThreadState::Exited);
        let core = t.core;
        t.state = new_state;
        if exited {
            t.user_preemption_pending = false;
        }
        if became_ready && !self.ready.contains(&handle) {
            self.ready.push_back(handle);
            self.wakers.notify_core(core);
        }
    }

    pub fn save_current_ctx(&mut self, cpu: &Cpu) {
        let Some(h) = self.current[current_core()] else {
            return;
        };
        let Some(t) = self.threads.get_mut(&h) else {
            return;
        };
        for i in 0..31 {
            t.ctx.x[i] = cpu.get_register(i as u32);
        }
        t.ctx.sp = cpu.get_register(31);
        t.ctx.pc = cpu.get_pc();
        t.ctx.tpidrro_el0 = cpu.get_tpidrro_el0();
        if let Err(e) = cpu.save_thread_context(&mut t.ctx.backend) {
            log::error!("failed to save thread context for {:#x}: {}", h, e);
        }
    }

    pub fn load_thread(&self, handle: u32, cpu: &mut Cpu) {
        let Some(t) = self.threads.get(&handle) else {
            return;
        };
        let result = match t.ctx.backend.as_ref() {
            Some(context) => cpu.restore_thread_context(context),
            None => cpu.reset_thread_context(),
        };
        if let Err(e) = result {
            log::error!("failed to restore thread context for {:#x}: {}", handle, e);
        }
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
                ThreadState::WaitingHandle {
                    wake_at: Some(d), ..
                } => *d <= now,
                ThreadState::WaitingArbiter {
                    wake_at: Some(d), ..
                } => *d <= now,
                _ => false,
            };
            if due {
                woken.push(*h);
            }
        }
        for h in woken {
            if let Some(t) = self.threads.get_mut(&h) {
                match t.state {
                    ThreadState::WaitingHandle { .. } | ThreadState::WaitingArbiter { .. } => {
                        t.ctx.x[0] = nexium_common::result::KERNEL_TIMEOUT as u64;
                    }
                    _ => {}
                }
            }
            self.transition_state(h, ThreadState::Ready);
        }
    }

    pub fn peek_one_mutex_waiter(&self, mutex_addr: u64) -> Option<(u32, u32, bool)> {
        let h = self
            .threads
            .iter()
            .filter_map(|(h, t)| match &t.state {
                ThreadState::WaitingMutex { mutex_addr: m, .. } if *m == mutex_addr => {
                    Some((*h, t.priority))
                }
                _ => None,
            })
            .min_by_key(|(h, priority)| (*priority, *h))
            .map(|(h, _)| h)?;
        let tag = match self.threads.get(&h).map(|t| &t.state) {
            Some(ThreadState::WaitingMutex { tag, .. }) => *tag,
            _ => h,
        };
        let has_more = self.threads.iter().any(|(other_h, t)| {
            *other_h != h
                && matches!(
                    &t.state,
                    ThreadState::WaitingMutex { mutex_addr: m, .. } if *m == mutex_addr
                )
        });
        Some((h, tag, has_more))
    }

    pub fn commit_wake_mutex_waiter(&mut self, mutex_addr: u64, h: u32) {
        for (other_h, t) in self.threads.iter_mut() {
            if *other_h == h {
                continue;
            }
            if let ThreadState::WaitingMutex {
                mutex_addr: m,
                owner_handle: o,
                ..
            } = &mut t.state
            {
                if *m == mutex_addr {
                    *o = h;
                }
            }
        }
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
    }

    pub fn wake_one_on_mutex_owned(
        &mut self,
        mutex_addr: u64,
        _owner_handle: u32,
    ) -> Option<(u32, u32, bool)> {
        let (h, tag, has_more) = self.peek_one_mutex_waiter(mutex_addr)?;
        self.commit_wake_mutex_waiter(mutex_addr, h);
        Some((h, tag, has_more))
    }

    pub fn wake_one_on_condvar(&mut self, condvar_addr: u64) -> Option<u32> {
        let h = self.threads.iter().find_map(|(h, t)| match &t.state {
            ThreadState::WaitingCondvar {
                condvar_addr: c, ..
            } if *c == condvar_addr => Some(*h),
            _ => None,
        })?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn peek_one_condvar_waiter(&self, condvar_addr: u64) -> Option<(u32, u64)> {
        self.threads
            .iter()
            .filter_map(|(h, t)| match &t.state {
                ThreadState::WaitingCondvar {
                    condvar_addr: c,
                    mutex_addr,
                    ..
                } if *c == condvar_addr => Some((*h, t.priority, *mutex_addr)),
                _ => None,
            })
            .min_by_key(|(h, priority, _)| (*priority, *h))
            .map(|(h, _, mutex_addr)| (h, mutex_addr))
    }

    pub fn wake_condvar_into_mutex_waiter(
        &mut self,
        handle: u32,
        mutex_addr: u64,
        owner_handle: u32,
        tag: u32,
    ) {
        if let Some(t) = self.threads.get_mut(&handle) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(
            handle,
            ThreadState::WaitingMutex {
                mutex_addr,
                owner_handle,
                tag,
            },
        );
    }

    pub fn wake_condvar_to_ready(&mut self, handle: u32) {
        if let Some(t) = self.threads.get_mut(&handle) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(handle, ThreadState::Ready);
    }

    pub fn has_mutex_waiters_for_owner(&self, mutex_addr: u64, owner_handle: u32) -> bool {
        self.threads.values().any(|t| {
            matches!(
                &t.state,
                ThreadState::WaitingMutex {
                    mutex_addr: m,
                    owner_handle: o,
                    ..
                } if *m == mutex_addr && *o == owner_handle
            )
        })
    }

    pub fn has_condvar_waiters(&self, condvar_addr: u64) -> bool {
        self.threads.values().any(|t| matches!(&t.state, ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr))
    }

    pub fn has_mutex_waiters(&self, mutex_addr: u64) -> bool {
        self.threads.values().any(|t| matches!(&t.state, ThreadState::WaitingMutex { mutex_addr: m, .. } if *m == mutex_addr))
    }

    pub fn wake_all_on_condvar(&mut self, condvar_addr: u64) -> usize {
        let mut woken: Vec<u32> = Vec::new();
        for (h, t) in self.threads.iter_mut() {
            if let ThreadState::WaitingCondvar {
                condvar_addr: c, ..
            } = &t.state
            {
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

    pub fn cancel_synchronization(&mut self, handle: u32) -> Option<bool> {
        let waiting = matches!(
            self.threads.get(&handle).map(|thread| &thread.state),
            Some(ThreadState::WaitingHandle { .. })
        );
        let thread = self.threads.get_mut(&handle)?;
        if waiting {
            thread.wait_cancelled = false;
            thread.ctx.x[0] = nexium_common::result::KERNEL_CANCELLED as u64;
            self.transition_state(handle, ThreadState::Ready);
        } else {
            thread.wait_cancelled = true;
        }
        Some(waiting)
    }

    pub fn take_wait_cancelled(&mut self, handle: u32) -> bool {
        let Some(thread) = self.threads.get_mut(&handle) else {
            return false;
        };
        std::mem::take(&mut thread.wait_cancelled)
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

    pub fn effective_priority(&self, handle: u32) -> i32 {
        let base = self
            .threads
            .get(&handle)
            .map(|t| t.priority)
            .unwrap_or(i32::MAX);
        let inherited = self
            .threads
            .values()
            .filter_map(|t| match &t.state {
                ThreadState::WaitingMutex { owner_handle, .. } if *owner_handle == handle => {
                    Some(t.priority)
                }
                _ => None,
            })
            .min();
        match inherited {
            Some(p) => base.min(p),
            None => base,
        }
    }

    pub fn thread_core_mask(&self, handle: u32) -> Option<(i32, u64)> {
        self.threads
            .get(&handle)
            .map(|t| (t.ideal_core, t.affinity_mask))
    }

    pub fn set_thread_core_mask(
        &mut self,
        handle: u32,
        core_id: i32,
        affinity_mask: u64,
        process_ideal_core: i32,
        active_cores: i32,
    ) -> Result<(), u32> {
        use nexium_common::result::{
            KERNEL_INVALID_COMBINATION, KERNEL_INVALID_CORE_ID, KERNEL_INVALID_HANDLE,
        };
        let virtual_core_mask: u64 = (1u64 << NUM_CORES) - 1;
        let (ideal_update, affinity_mask) = if core_id == IDEAL_CORE_USE_PROCESS_VALUE {
            (process_ideal_core, 1u64 << process_ideal_core)
        } else {
            if affinity_mask & !virtual_core_mask != 0 {
                return Err(KERNEL_INVALID_CORE_ID);
            }
            if affinity_mask == 0 {
                return Err(KERNEL_INVALID_COMBINATION);
            }
            if (0..NUM_CORES as i32).contains(&core_id) {
                if affinity_mask & (1u64 << core_id) == 0 {
                    return Err(KERNEL_INVALID_COMBINATION);
                }
            } else if core_id != IDEAL_CORE_DONT_CARE && core_id != IDEAL_CORE_NO_UPDATE {
                return Err(KERNEL_INVALID_CORE_ID);
            }
            (core_id, affinity_mask)
        };
        let thread = self.threads.get_mut(&handle).ok_or(KERNEL_INVALID_HANDLE)?;
        if ideal_update == IDEAL_CORE_NO_UPDATE {
            if thread.ideal_core >= 0 && affinity_mask & (1u64 << thread.ideal_core) == 0 {
                return Err(KERNEL_INVALID_COMBINATION);
            }
        } else {
            thread.ideal_core = ideal_update;
        }
        thread.affinity_mask = affinity_mask;
        if matches!(thread.state, ThreadState::Created) {
            let preferred = if thread.ideal_core >= 0 {
                thread.ideal_core
            } else {
                affinity_mask.trailing_zeros() as i32
            };
            thread.core = preferred.min(active_cores.max(1) - 1);
        } else if thread.core >= 0 && affinity_mask & (1u64 << thread.core) == 0 {
            log::debug!(
                "SetThreadCoreMask handle={:#x}: started thread stays on sticky core {} outside new guest mask {:#x}",
                handle,
                thread.core,
                affinity_mask
            );
        }
        Ok(())
    }

    pub fn pick_next(&mut self) -> Option<u32> {
        let core = current_core() as i32;
        let pos = self
            .ready
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                self.threads
                    .get(h)
                    .map_or(false, |t| t.core == core || t.core < 0)
            })
            .min_by_key(|(idx, h)| (self.effective_priority(**h), *idx))
            .map(|(idx, _)| idx)?;
        let handle = self.ready.remove(pos)?;
        if let Some(t) = self.threads.get_mut(&handle) {
            if t.core < 0 {
                t.core = core;
            }
        }
        Some(handle)
    }

    pub fn has_ready_for_core(&self, core: i32) -> bool {
        self.ready.iter().any(|h| {
            self.threads
                .get(h)
                .map_or(false, |t| t.core == core || t.core < 0)
        })
    }

    pub fn has_higher_priority_ready_for_core(&self, core: usize) -> bool {
        let Some(current) = self.current.get(core).copied().flatten() else {
            return false;
        };
        let current_priority = self.effective_priority(current);
        self.ready.iter().any(|handle| {
            self.threads.get(handle).is_some_and(|thread| {
                (thread.core == core as i32 || thread.core < 0)
                    && self.effective_priority(*handle) < current_priority
            })
        })
    }

    fn enqueue_ready(&mut self, handle: u32, front: bool) {
        if self.ready.contains(&handle) {
            return;
        }
        if front {
            self.ready.push_front(handle);
        } else {
            self.ready.push_back(handle);
        }
    }

    pub fn yield_current(&mut self, cpu: &Cpu) {
        if self.current[current_core()].is_some() {
            self.save_current_ctx(cpu);
            self.current[current_core()] = None;
        }
    }

    pub fn yield_with_state(&mut self, cpu: &Cpu, new_state: ThreadState) -> Option<u32> {
        self.yield_with_state_at_back(cpu, new_state)
    }

    pub fn preempt_current(&mut self, cpu: &Cpu) -> Option<u32> {
        self.yield_with_state_position(cpu, ThreadState::Ready, true)
    }

    fn yield_with_state_at_back(&mut self, cpu: &Cpu, new_state: ThreadState) -> Option<u32> {
        self.yield_with_state_position(cpu, new_state, false)
    }

    fn yield_with_state_position(
        &mut self,
        cpu: &Cpu,
        new_state: ThreadState,
        ready_front: bool,
    ) -> Option<u32> {
        let h = self.current[current_core()]?;
        self.save_current_ctx(cpu);
        let became_ready = matches!(new_state, ThreadState::Ready);
        let exited = matches!(new_state, ThreadState::Exited);
        if let Some(t) = self.threads.get_mut(&h) {
            t.state = new_state;
            if exited {
                t.user_preemption_pending = false;
            }
        }
        if became_ready {
            self.enqueue_ready(h, ready_front);
        }
        self.current[current_core()] = None;
        self.record_switch();
        Some(h)
    }

    pub fn schedule_to(&mut self, handle: u32, cpu: &mut Cpu) {
        self.load_thread(handle, cpu);
        if let Some(t) = self.threads.get_mut(&handle) {
            t.state = ThreadState::Running;
            log::trace!(
                "[sched] core{} now running handle={:#x} pc={:#x} sp={:#x}",
                current_core(),
                handle,
                t.ctx.pc,
                t.ctx.sp
            );
        }
        self.current[current_core()] = Some(handle);
        self.record_switch();
    }

    pub fn earliest_wake(&self) -> Option<Instant> {
        self.threads
            .values()
            .filter_map(|t| match &t.state {
                ThreadState::Sleeping { wake_at } => Some(*wake_at),
                ThreadState::WaitingHandle {
                    wake_at: Some(d), ..
                } => Some(*d),
                ThreadState::WaitingCondvar {
                    wake_at: Some(d), ..
                } => Some(*d),
                ThreadState::WaitingArbiter {
                    wake_at: Some(d), ..
                } => Some(*d),
                _ => None,
            })
            .min()
    }

    pub fn drop_exited(&mut self) {
        let pool_base = self.tls_pool_base;
        let free_tls = &mut self.free_tls;
        self.threads.retain(|_, t| {
            if matches!(t.state, ThreadState::Exited) {
                if pool_base != 0 && t.tls_va >= pool_base {
                    free_tls.push(t.tls_va);
                }
                false
            } else {
                true
            }
        });
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

#[cfg(test)]
mod tests {
    use super::{
        resolve_create_thread_core, ThreadCtx, ThreadState, Threads, IDEAL_CORE_DONT_CARE,
        IDEAL_CORE_NO_UPDATE, IDEAL_CORE_USE_PROCESS_VALUE,
    };
    use nexium_common::result::{
        KERNEL_INVALID_COMBINATION, KERNEL_INVALID_CORE_ID, KERNEL_INVALID_HANDLE,
    };
    use std::time::{Duration, Instant};

    fn add_ready_thread(threads: &mut Threads, handle: u32, priority: i32) {
        threads.add_thread(handle, ThreadCtx::zero(), 0, 0, 0);
        threads.threads.get_mut(&handle).unwrap().priority = priority;
        threads.transition_state(handle, ThreadState::Ready);
    }

    fn add_created_thread(threads: &mut Threads, handle: u32) {
        threads.add_thread(handle, ThreadCtx::zero(), 0, 0, 0);
    }

    fn core_state(threads: &Threads, handle: u32) -> (i32, u64, i32) {
        let t = threads.threads.get(&handle).unwrap();
        (t.ideal_core, t.affinity_mask, t.core)
    }

    #[test]
    fn create_thread_core_resolution() {
        assert_eq!(resolve_create_thread_core(2, 0), Ok(2));
        assert_eq!(
            resolve_create_thread_core(IDEAL_CORE_USE_PROCESS_VALUE, 1),
            Ok(1)
        );
        assert_eq!(
            resolve_create_thread_core(IDEAL_CORE_DONT_CARE, 0),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            resolve_create_thread_core(IDEAL_CORE_NO_UPDATE, 0),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            resolve_create_thread_core(4, 0),
            Err(KERNEL_INVALID_CORE_ID)
        );
    }

    #[test]
    fn set_core_mask_validates_inputs() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_created_thread(&mut threads, 0x200);
        assert_eq!(
            threads.set_thread_core_mask(0x999, 0, 1, 0, 4),
            Err(KERNEL_INVALID_HANDLE)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x999, 0, 0, 0, 4),
            Err(KERNEL_INVALID_COMBINATION)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x999, 0, 0x10, 0, 4),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x200, 0, 0x11, 0, 4),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x200, 0, 0, 0, 4),
            Err(KERNEL_INVALID_COMBINATION)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x200, 2, 0b0011, 0, 4),
            Err(KERNEL_INVALID_COMBINATION)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x200, 4, 0b0011, 0, 4),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            threads.set_thread_core_mask(0x200, -4, 0b0011, 0, 4),
            Err(KERNEL_INVALID_CORE_ID)
        );
        assert_eq!(
            core_state(&threads, 0x200),
            (IDEAL_CORE_USE_PROCESS_VALUE, 0, -2)
        );
    }

    #[test]
    fn set_core_mask_applies_before_start_and_sticks_after() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_created_thread(&mut threads, 0x200);
        assert_eq!(threads.set_thread_core_mask(0x200, 2, 0b0100, 0, 4), Ok(()));
        assert_eq!(core_state(&threads, 0x200), (2, 0b0100, 2));

        threads.transition_state(0x200, ThreadState::Ready);
        assert_eq!(threads.set_thread_core_mask(0x200, 1, 0b0010, 0, 4), Ok(()));
        assert_eq!(core_state(&threads, 0x200), (1, 0b0010, 2));
    }

    #[test]
    fn set_core_mask_resolves_special_ideals() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_created_thread(&mut threads, 0x200);

        assert_eq!(
            threads.set_thread_core_mask(0x200, IDEAL_CORE_USE_PROCESS_VALUE, 0xdead, 1, 4),
            Ok(())
        );
        assert_eq!(core_state(&threads, 0x200), (1, 0b0010, 1));

        assert_eq!(
            threads.set_thread_core_mask(0x200, IDEAL_CORE_DONT_CARE, 0b1100, 0, 4),
            Ok(())
        );
        assert_eq!(
            core_state(&threads, 0x200),
            (IDEAL_CORE_DONT_CARE, 0b1100, 2)
        );

        assert_eq!(
            threads.set_thread_core_mask(0x200, IDEAL_CORE_NO_UPDATE, 0b1000, 0, 4),
            Ok(())
        );
        assert_eq!(
            core_state(&threads, 0x200),
            (IDEAL_CORE_DONT_CARE, 0b1000, 3)
        );
    }

    #[test]
    fn mask_only_update_must_contain_the_preserved_ideal() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_created_thread(&mut threads, 0x200);
        assert_eq!(threads.set_thread_core_mask(0x200, 0, 0b0001, 0, 4), Ok(()));

        assert_eq!(
            threads.set_thread_core_mask(0x200, IDEAL_CORE_NO_UPDATE, 0b1000, 0, 4),
            Err(KERNEL_INVALID_COMBINATION)
        );
        assert_eq!(core_state(&threads, 0x200), (0, 0b0001, 0));

        assert_eq!(
            threads.set_thread_core_mask(0x200, IDEAL_CORE_NO_UPDATE, 0b0011, 0, 4),
            Ok(())
        );
        assert_eq!(core_state(&threads, 0x200), (0, 0b0011, 0));
    }

    #[test]
    fn set_core_mask_clamps_execution_core_to_active_cores() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_created_thread(&mut threads, 0x200);
        assert_eq!(threads.set_thread_core_mask(0x200, 3, 0b1000, 0, 2), Ok(()));
        assert_eq!(core_state(&threads, 0x200), (3, 0b1000, 1));
    }

    #[test]
    fn get_core_mask_returns_stored_guest_state() {
        let threads = Threads::new(0x100, 0, 0, 0);
        assert_eq!(threads.thread_core_mask(0x100), Some((0, 1)));
        assert_eq!(threads.thread_core_mask(0xdead), None);
    }

    #[test]
    fn user_preemption_pending_is_current_thread_state() {
        let mut threads = Threads::new(0x100, 0x10_0000, 0, 0);

        assert!(!threads.current_user_preemption_pending());
        assert!(threads.mark_current_user_preemption_pending());
        assert!(threads.current_user_preemption_pending());
        assert!(!threads.mark_current_user_preemption_pending());
        assert!(threads.take_current_user_preemption_pending());
        assert!(!threads.take_current_user_preemption_pending());
        assert!(!threads.current_user_preemption_pending());
    }

    #[test]
    fn hos_timeslice_only_rotates_the_designated_priority_for_each_core() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 59);
        threads.threads.get_mut(&0x200).unwrap().core = 0;
        threads.per_core_last_switch[0] = Instant::now() - Duration::from_millis(20);

        threads.threads.get_mut(&0x100).unwrap().priority = 44;
        assert!(!threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));

        threads.threads.get_mut(&0x100).unwrap().priority = 63;
        assert!(!threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));

        threads.threads.get_mut(&0x100).unwrap().priority = 59;
        assert!(threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));

        threads.current[0] = None;
        threads.current[3] = Some(0x100);
        let current = threads.threads.get_mut(&0x100).unwrap();
        current.core = 3;
        current.priority = 59;
        threads.threads.get_mut(&0x200).unwrap().core = 3;
        threads.per_core_last_switch[3] = Instant::now() - Duration::from_millis(20);
        assert!(!threads.hos_timeslice_expired_on_core(3, Duration::from_millis(10)));

        threads.threads.get_mut(&0x100).unwrap().priority = 63;
        assert!(threads.hos_timeslice_expired_on_core(3, Duration::from_millis(10)));
    }

    #[test]
    fn hos_timeslice_uses_a_per_core_clock_and_requires_ready_work() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        threads.threads.get_mut(&0x100).unwrap().priority = 59;
        add_ready_thread(&mut threads, 0x200, 59);
        threads.threads.get_mut(&0x200).unwrap().core = 0;

        threads.per_core_last_switch[0] = Instant::now();
        threads.per_core_last_switch[1] = Instant::now() - Duration::from_millis(20);
        assert!(!threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));

        threads.per_core_last_switch[0] = Instant::now() - Duration::from_millis(20);
        assert!(threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));

        threads.ready.clear();
        assert!(!threads.hos_timeslice_expired_on_core(0, Duration::from_millis(10)));
    }

    #[test]
    fn higher_priority_ready_check_is_strict_and_core_local() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 44);
        add_ready_thread(&mut threads, 0x201, 60);
        assert!(!threads.has_higher_priority_ready_for_core(0));

        add_ready_thread(&mut threads, 0x202, 30);
        threads.threads.get_mut(&0x202).unwrap().core = 1;
        assert!(!threads.has_higher_priority_ready_for_core(0));

        threads.threads.get_mut(&0x202).unwrap().core = 0;
        assert!(threads.has_higher_priority_ready_for_core(0));
    }

    #[test]
    fn involuntary_preemption_compares_inherited_priority() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 30);
        threads.add_thread(0x300, ThreadCtx::zero(), 0, 0, 0);
        threads.threads.get_mut(&0x300).unwrap().priority = 20;
        threads.transition_state(
            0x300,
            ThreadState::WaitingMutex {
                mutex_addr: 0x1000,
                owner_handle: 0x100,
                tag: 0x300,
            },
        );
        assert_eq!(threads.effective_priority(0x100), 20);
        assert!(!threads.has_higher_priority_ready_for_core(0));

        add_ready_thread(&mut threads, 0x201, 10);
        assert!(threads.has_higher_priority_ready_for_core(0));
    }

    #[test]
    fn involuntary_preemption_keeps_current_ahead_of_equal_priority_peers() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 44);
        add_ready_thread(&mut threads, 0x300, 30);
        threads.enqueue_ready(0x100, true);

        assert_eq!(threads.pick_next(), Some(0x300));
        assert_eq!(threads.pick_next(), Some(0x100));
        assert_eq!(threads.pick_next(), Some(0x200));
    }

    #[test]
    fn voluntary_yield_moves_current_behind_equal_priority_peers() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 44);
        threads.enqueue_ready(0x100, false);

        assert_eq!(threads.pick_next(), Some(0x200));
        assert_eq!(threads.pick_next(), Some(0x100));
    }

    #[test]
    fn scheduler_picks_priority_then_fifo_with_core_affinity() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 0);
        threads.threads.get_mut(&0x200).unwrap().core = 1;

        let fifo_handles: Vec<u32> = (0..9).map(|index| 0x210 + index).collect();
        for (index, &handle) in fifo_handles.iter().enumerate() {
            add_ready_thread(&mut threads, handle, 0x2c);
            if index % 2 == 0 {
                threads.threads.get_mut(&handle).unwrap().core = 0;
            }
        }
        add_ready_thread(&mut threads, 0x300, 1);

        assert_eq!(threads.pick_next(), Some(0x300));
        for &handle in &fifo_handles {
            assert_eq!(threads.pick_next(), Some(handle));
            assert_eq!(threads.threads.get(&handle).unwrap().core, 0);
        }
        assert_eq!(
            threads.ready.iter().copied().collect::<Vec<_>>(),
            vec![0x200]
        );
    }

    #[test]
    fn pick_next_uses_inherited_priority() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 0x30);
        add_ready_thread(&mut threads, 0x201, 0x20);
        threads.add_thread(0x300, ThreadCtx::zero(), 0, 0, 0);
        threads.threads.get_mut(&0x300).unwrap().priority = 0x10;
        threads.transition_state(
            0x300,
            ThreadState::WaitingMutex {
                mutex_addr: 0x1000,
                owner_handle: 0x200,
                tag: 0x300,
            },
        );

        assert_eq!(threads.pick_next(), Some(0x200));
        assert_eq!(threads.pick_next(), Some(0x201));
    }

    #[test]
    fn transition_state_ignores_unknown_handles() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        threads.transition_state(0xdead, ThreadState::Ready);
        assert!(threads.ready.is_empty());
    }

    #[test]
    fn cancel_synchronization_wakes_waiting_thread() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        threads.transition_state(
            0x100,
            ThreadState::WaitingHandle {
                handles: Vec::new(),
                wake_at: None,
            },
        );

        assert_eq!(threads.cancel_synchronization(0x100), Some(true));
        let thread = threads.threads.get(&0x100).unwrap();
        assert!(matches!(thread.state, ThreadState::Ready));
        assert_eq!(
            thread.ctx.x[0],
            nexium_common::result::KERNEL_CANCELLED as u64
        );
        assert!(!thread.wait_cancelled);
    }

    #[test]
    fn early_cancel_is_consumed_by_next_wait_once() {
        let mut threads = Threads::new(0x100, 0, 0, 0);

        assert_eq!(threads.cancel_synchronization(0x100), Some(false));
        assert!(threads.take_wait_cancelled(0x100));
        assert!(!threads.take_wait_cancelled(0x100));
    }
}
