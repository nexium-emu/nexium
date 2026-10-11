use crate::kernel::cpu_local::current_core;
use nexium_cpu::Cpu;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

pub const NUM_CORES: usize = 4;

pub const HOS_PREEMPTION_PRIORITIES: [i32; NUM_CORES] = [59, 59, 59, 63];

pub struct CoreWakers {
    wakes: [nexium_common::host_wake::Futex; NUM_CORES],
}

impl CoreWakers {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            wakes: std::array::from_fn(|_| nexium_common::host_wake::Futex::new()),
        })
    }

    pub fn notify_core(&self, core: i32) {
        if (0..NUM_CORES as i32).contains(&core) {
            self.wakes[core as usize].notify_all();
        } else {
            self.notify_all_cores();
        }
    }

    pub fn notify_all_cores(&self) {
        for wake in &self.wakes {
            wake.notify_all();
        }
    }

    pub fn park_core<T>(
        &self,
        guard: &mut parking_lot::MutexGuard<'_, T>,
        core: usize,
        timeout: std::time::Duration,
    ) {
        if let Some(wake) = self.wakes.get(core) {
            let generation = wake.generation();
            parking_lot::MutexGuard::unlocked(guard, || {
                let _ = wake.wait(generation, timeout);
            });
        }
    }

    pub fn park_core_until<T>(
        &self,
        guard: &mut parking_lot::MutexGuard<'_, T>,
        core: usize,
        deadline: Instant,
    ) {
        if let Some(wake) = self.wakes.get(core) {
            let generation = wake.generation();
            parking_lot::MutexGuard::unlocked(guard, || {
                let _ = wake.wait_until(generation, deadline);
            });
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
    /// True while the handle sits in `Threads::ready`. Maintained by every
    /// queue mutation in this file so membership tests are O(1).
    in_ready: bool,
}

/// Secondary indexes over the blocking `ThreadState`s, so wake-ups look at the
/// threads parked on one object instead of scanning the whole thread table.
///
/// Every list is kept in wait-start (FIFO) order. The index is updated at the
/// only places a thread's state is assigned (`Threads::set_state`) and where a
/// mutex waiter's recorded owner is rewritten, and every lookup re-validates
/// candidates against the thread's actual state, so a stale entry can never
/// produce a wrong wake.
#[derive(Default)]
struct WaitIndex {
    mutex: HashMap<u64, Vec<u32>>,
    mutex_owner: HashMap<u32, Vec<u32>>,
    condvar: HashMap<u64, Vec<u32>>,
    arbiter: HashMap<u64, Vec<u32>>,
    handle: HashMap<u32, Vec<u32>>,
    /// Lower bound on the earliest `wake_at` of any timed wait. `None` means
    /// no thread has a deadline. It may be earlier than the true minimum (a
    /// timed waiter that was woken early leaves it behind), never later.
    next_deadline: Option<Instant>,
}

fn index_push<K: std::hash::Hash + Eq>(map: &mut HashMap<K, Vec<u32>>, key: K, handle: u32) {
    let list = map.entry(key).or_default();
    if !list.contains(&handle) {
        list.push(handle);
    }
}

fn index_drop<K: std::hash::Hash + Eq>(map: &mut HashMap<K, Vec<u32>>, key: &K, handle: u32) {
    if let Some(list) = map.get_mut(key) {
        list.retain(|&h| h != handle);
        if list.is_empty() {
            map.remove(key);
        }
    }
}

impl WaitIndex {
    fn note_deadline(&mut self, wake_at: Instant) {
        self.next_deadline = Some(match self.next_deadline {
            Some(current) => current.min(wake_at),
            None => wake_at,
        });
    }

    fn insert(&mut self, handle: u32, state: &ThreadState) {
        match state {
            ThreadState::Sleeping { wake_at } => self.note_deadline(*wake_at),
            ThreadState::WaitingHandle { handles, wake_at } => {
                for waited in handles {
                    index_push(&mut self.handle, *waited, handle);
                }
                if let Some(wake_at) = wake_at {
                    self.note_deadline(*wake_at);
                }
            }
            ThreadState::WaitingMutex {
                mutex_addr,
                owner_handle,
                ..
            } => {
                index_push(&mut self.mutex, *mutex_addr, handle);
                index_push(&mut self.mutex_owner, *owner_handle, handle);
            }
            ThreadState::WaitingCondvar {
                condvar_addr,
                wake_at,
                ..
            } => {
                index_push(&mut self.condvar, *condvar_addr, handle);
                if let Some(wake_at) = wake_at {
                    self.note_deadline(*wake_at);
                }
            }
            ThreadState::WaitingArbiter { addr, wake_at, .. } => {
                index_push(&mut self.arbiter, *addr, handle);
                if let Some(wake_at) = wake_at {
                    self.note_deadline(*wake_at);
                }
            }
            ThreadState::Created
            | ThreadState::Ready
            | ThreadState::Running
            | ThreadState::Exited => {}
        }
    }

    fn remove(&mut self, handle: u32, state: &ThreadState) {
        match state {
            ThreadState::WaitingHandle { handles, .. } => {
                for waited in handles {
                    index_drop(&mut self.handle, waited, handle);
                }
            }
            ThreadState::WaitingMutex {
                mutex_addr,
                owner_handle,
                ..
            } => {
                index_drop(&mut self.mutex, mutex_addr, handle);
                index_drop(&mut self.mutex_owner, owner_handle, handle);
            }
            ThreadState::WaitingCondvar { condvar_addr, .. } => {
                index_drop(&mut self.condvar, condvar_addr, handle);
            }
            ThreadState::WaitingArbiter { addr, .. } => {
                index_drop(&mut self.arbiter, addr, handle);
            }
            ThreadState::Created
            | ThreadState::Ready
            | ThreadState::Running
            | ThreadState::Sleeping { .. }
            | ThreadState::Exited => {}
        }
    }
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
    waits: WaitIndex,
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
                in_ready: false,
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
            waits: WaitIndex::default(),
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

    /// Wall-clock slice check for one core: something else is runnable there
    /// and the core has not switched threads for at least `threshold`.
    pub fn core_timeslice_expired(&self, core: usize, threshold: std::time::Duration) -> bool {
        core < NUM_CORES
            && self.per_core_last_switch[core].elapsed() >= threshold
            && self.has_ready_for_core(core as i32)
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
                in_ready: false,
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

    /// The single place a thread's state is assigned; keeps `waits` in step.
    /// Returns false for unknown handles.
    fn set_state(&mut self, handle: u32, new_state: ThreadState) -> bool {
        let Some(t) = self.threads.get_mut(&handle) else {
            return false;
        };
        let old_state = std::mem::replace(&mut t.state, new_state);
        if matches!(t.state, ThreadState::Exited) {
            t.user_preemption_pending = false;
        }
        self.waits.remove(handle, &old_state);
        self.waits.insert(handle, &t.state);
        true
    }

    fn remove_from_ready(&mut self, handle: u32) {
        let Some(t) = self.threads.get_mut(&handle) else {
            return;
        };
        if std::mem::take(&mut t.in_ready) {
            self.ready.retain(|&h| h != handle);
        }
    }

    pub fn transition_state(&mut self, handle: u32, new_state: ThreadState) {
        self.remove_from_ready(handle);
        let became_ready = matches!(new_state, ThreadState::Ready);
        if !self.set_state(handle, new_state) {
            return;
        }
        if became_ready {
            let Some(t) = self.threads.get_mut(&handle) else {
                return;
            };
            t.in_ready = true;
            let core = t.core;
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
            t.ctx.x[i] = cpu.get_thread_register(i as u32);
        }
        t.ctx.sp = cpu.get_thread_register(31);
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
            cpu.set_thread_register(i as u32, t.ctx.x[i]);
        }
        cpu.set_thread_register(31, t.ctx.sp);
        cpu.set_pc(t.ctx.pc);
        cpu.set_tpidrro_el0(t.ctx.tpidrro_el0);
    }

    /// Whether any timed wait (sleep, handle, condvar or arbiter timeout) may
    /// have reached its deadline. False means a scan would find nothing.
    pub fn timers_due(&self, now: Instant) -> bool {
        self.waits.next_deadline.is_some_and(|deadline| deadline <= now)
    }

    pub fn wake_due_sleepers(&mut self, now: Instant) {
        if !self.timers_due(now) {
            return;
        }
        let mut woken = Vec::new();
        // Recomputed over every deadline this pass leaves in place, including
        // condvar timeouts (those are resolved by `Kernel::wake_due_sleepers`).
        let mut next_deadline: Option<Instant> = None;
        for (h, t) in self.threads.iter() {
            let (deadline, wakes_here) = match &t.state {
                ThreadState::Sleeping { wake_at } => (Some(*wake_at), true),
                ThreadState::WaitingHandle {
                    wake_at: Some(d), ..
                } => (Some(*d), true),
                ThreadState::WaitingArbiter {
                    wake_at: Some(d), ..
                } => (Some(*d), true),
                ThreadState::WaitingCondvar {
                    wake_at: Some(d), ..
                } => (Some(*d), false),
                _ => (None, false),
            };
            let Some(deadline) = deadline else {
                continue;
            };
            if wakes_here && deadline <= now {
                woken.push(*h);
            } else {
                next_deadline = Some(next_deadline.map_or(deadline, |d| d.min(deadline)));
            }
        }
        self.waits.next_deadline = next_deadline;
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

    /// Threads currently blocked on `mutex_addr`, in wait-start order.
    fn mutex_waiters(&self, mutex_addr: u64) -> impl Iterator<Item = (u32, &Thread)> + '_ {
        self.waits
            .mutex
            .get(&mutex_addr)
            .into_iter()
            .flatten()
            .filter_map(move |h| {
                let t = self.threads.get(h)?;
                matches!(
                    &t.state,
                    ThreadState::WaitingMutex { mutex_addr: m, .. } if *m == mutex_addr
                )
                .then_some((*h, t))
            })
    }

    /// Threads currently blocked on `condvar_addr`, in wait-start order.
    fn condvar_waiters(&self, condvar_addr: u64) -> impl Iterator<Item = (u32, &Thread)> + '_ {
        self.waits
            .condvar
            .get(&condvar_addr)
            .into_iter()
            .flatten()
            .filter_map(move |h| {
                let t = self.threads.get(h)?;
                matches!(
                    &t.state,
                    ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr
                )
                .then_some((*h, t))
            })
    }

    /// Threads currently blocked on arbiter address `addr`, in wait-start order.
    fn arbiter_waiters(&self, addr: u64) -> impl Iterator<Item = (u32, &Thread)> + '_ {
        self.waits
            .arbiter
            .get(&addr)
            .into_iter()
            .flatten()
            .filter_map(move |h| {
                let t = self.threads.get(h)?;
                matches!(
                    &t.state,
                    ThreadState::WaitingArbiter { addr: a, .. } if *a == addr
                )
                .then_some((*h, t))
            })
    }

    pub fn peek_one_mutex_waiter(&self, mutex_addr: u64) -> Option<(u32, u32, bool)> {
        let mut count = 0usize;
        let mut best: Option<(i32, u32)> = None;
        for (h, t) in self.mutex_waiters(mutex_addr) {
            count += 1;
            let key = (t.priority, h);
            if best.map_or(true, |b| key < b) {
                best = Some(key);
            }
        }
        let (_, h) = best?;
        let tag = match self.threads.get(&h).map(|t| &t.state) {
            Some(ThreadState::WaitingMutex { tag, .. }) => *tag,
            _ => h,
        };
        Some((h, tag, count > 1))
    }

    pub fn commit_wake_mutex_waiter(&mut self, mutex_addr: u64, h: u32) {
        let waiter_count = self.waits.mutex.get(&mutex_addr).map_or(0, Vec::len);
        for idx in 0..waiter_count {
            let Some(other_h) = self
                .waits
                .mutex
                .get(&mutex_addr)
                .and_then(|list| list.get(idx))
                .copied()
            else {
                break;
            };
            if other_h == h {
                continue;
            }
            let Some(t) = self.threads.get_mut(&other_h) else {
                continue;
            };
            if let ThreadState::WaitingMutex {
                mutex_addr: m,
                owner_handle: o,
                ..
            } = &mut t.state
            {
                if *m == mutex_addr && *o != h {
                    let previous_owner = std::mem::replace(o, h);
                    index_drop(&mut self.waits.mutex_owner, &previous_owner, other_h);
                    index_push(&mut self.waits.mutex_owner, h, other_h);
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
        let h = self.condvar_waiters(condvar_addr).next().map(|(h, _)| h)?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn peek_one_condvar_waiter(&self, condvar_addr: u64) -> Option<(u32, u64)> {
        self.condvar_waiters(condvar_addr)
            .filter_map(|(h, t)| match &t.state {
                ThreadState::WaitingCondvar { mutex_addr, .. } => {
                    Some((h, t.priority, *mutex_addr))
                }
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
        self.mutex_waiters(mutex_addr).any(|(_, t)| {
            matches!(
                &t.state,
                ThreadState::WaitingMutex { owner_handle: o, .. } if *o == owner_handle
            )
        })
    }

    pub fn has_condvar_waiters(&self, condvar_addr: u64) -> bool {
        self.condvar_waiters(condvar_addr).next().is_some()
    }

    pub fn has_mutex_waiters(&self, mutex_addr: u64) -> bool {
        self.mutex_waiters(mutex_addr).next().is_some()
    }

    pub fn wake_all_on_condvar(&mut self, condvar_addr: u64) -> usize {
        // Taking the list is safe: every valid entry is woken below, which is
        // exactly the set `set_state` would have removed one by one.
        let candidates = self.waits.condvar.remove(&condvar_addr).unwrap_or_default();
        let mut count = 0;
        for h in candidates {
            let Some(t) = self.threads.get_mut(&h) else {
                continue;
            };
            if !matches!(
                &t.state,
                ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr
            ) {
                continue;
            }
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
            count += 1;
            self.transition_state(h, ThreadState::Ready);
        }
        count
    }

    pub fn signal_handle(&mut self, handle: u32) {
        let Some(candidates) = self.waits.handle.remove(&handle) else {
            return;
        };
        for h in candidates {
            let Some(t) = self.threads.get_mut(&h) else {
                continue;
            };
            let ThreadState::WaitingHandle { handles, .. } = &t.state else {
                continue;
            };
            let Some(idx) = handles.iter().position(|&v| v == handle) else {
                continue;
            };
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
            t.ctx.x[1] = idx as u64;
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

    /// Number of threads blocked on arbiter address `addr`.
    pub fn arbiter_waiter_count(&self, addr: u64) -> usize {
        self.arbiter_waiters(addr).count()
    }

    pub fn wake_one_on_arbiter(&mut self, addr: u64) -> Option<u32> {
        let h = self.arbiter_waiters(addr).next().map(|(h, _)| h)?;
        if let Some(t) = self.threads.get_mut(&h) {
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
        }
        self.transition_state(h, ThreadState::Ready);
        Some(h)
    }

    pub fn wake_all_on_arbiter(&mut self, addr: u64) -> usize {
        let candidates = self.waits.arbiter.remove(&addr).unwrap_or_default();
        let mut count = 0;
        for h in candidates {
            let Some(t) = self.threads.get_mut(&h) else {
                continue;
            };
            if !matches!(&t.state, ThreadState::WaitingArbiter { addr: a, .. } if *a == addr) {
                continue;
            }
            t.ctx.x[0] = nexium_common::result::SUCCESS as u64;
            count += 1;
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
        // Waiter priorities are read live (SetThreadPriority writes them
        // directly), only the set of waiters comes from the index.
        let inherited = self
            .waits
            .mutex_owner
            .get(&handle)
            .into_iter()
            .flatten()
            .filter_map(|waiter| match self.threads.get(waiter) {
                Some(t) => match &t.state {
                    ThreadState::WaitingMutex { owner_handle, .. } if *owner_handle == handle => {
                        Some(t.priority)
                    }
                    _ => None,
                },
                None => None,
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
        let pos = self.next_ready_position(core)?;
        let handle = self.ready.remove(pos)?;
        if let Some(t) = self.threads.get_mut(&handle) {
            t.in_ready = false;
            if t.core != core {
                t.core = core;
            }
        }
        Some(handle)
    }

    /// Ready-queue index `pick_next` would take on `core`, without taking it.
    fn next_ready_position(&self, core: i32) -> Option<usize> {
        self.best_ready_position(|t| t.core == core || t.core < 0)
            .or_else(|| {
                self.best_ready_position(|t| {
                    t.core != core && (0..NUM_CORES as i32).contains(&core) && t.affinity_mask & (1u64 << core) != 0
                })
            })
    }

    fn best_ready_position(&self, eligible: impl Fn(&Thread) -> bool) -> Option<usize> {
        self.ready
            .iter()
            .enumerate()
            .filter(|(_, h)| self.threads.get(h).is_some_and(&eligible))
            .min_by_key(|(idx, h)| (self.effective_priority(**h), *idx))
            .map(|(idx, _)| idx)
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
        let Some(t) = self.threads.get_mut(&handle) else {
            return;
        };
        if std::mem::replace(&mut t.in_ready, true) {
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
        self.set_state(h, new_state);
        if became_ready {
            self.enqueue_ready(h, ready_front);
        }
        self.current[current_core()] = None;
        self.record_switch();
        Some(h)
    }

    pub fn schedule_to(&mut self, handle: u32, cpu: &mut Cpu) {
        self.load_thread(handle, cpu);
        self.set_state(handle, ThreadState::Running);
        if let Some(t) = self.threads.get(&handle) {
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
        resolve_create_thread_core, CoreWakers, ThreadCtx, ThreadState, Threads,
        IDEAL_CORE_DONT_CARE, IDEAL_CORE_NO_UPDATE, IDEAL_CORE_USE_PROCESS_VALUE, NUM_CORES,
    };
    use nexium_common::result::{
        KERNEL_INVALID_COMBINATION, KERNEL_INVALID_CORE_ID, KERNEL_INVALID_HANDLE,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn core_wakers_notify_selected_core_and_broadcast_generations() {
        let wakers = CoreWakers::new();
        let original: [u32; NUM_CORES] = std::array::from_fn(|core| wakers.wakes[core].generation());
        wakers.notify_core(2);
        for core in 0..NUM_CORES {
            assert_eq!(
                wakers.wakes[core].generation(),
                original[core].wrapping_add(u32::from(core == 2)),
            );
        }
        wakers.notify_all_cores();
        for core in 0..NUM_CORES {
            assert_eq!(
                wakers.wakes[core].generation(),
                original[core].wrapping_add(1 + u32::from(core == 2)),
            );
        }
        wakers.notify_core(-1);
        for core in 0..NUM_CORES {
            assert_eq!(
                wakers.wakes[core].generation(),
                original[core].wrapping_add(2 + u32::from(core == 2)),
            );
        }
    }

    #[test]
    fn core_wakers_release_mutex_publish_ready_and_reacquire_without_lost_wake() {
        let mutex = std::sync::Arc::new(parking_lot::Mutex::new(false));
        let wakers = CoreWakers::new();
        let mut guard = mutex.lock();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let worker = {
            let mutex = std::sync::Arc::clone(&mutex);
            let wakers = std::sync::Arc::clone(&wakers);
            std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                let mut ready = mutex.lock();
                *ready = true;
                wakers.notify_core(1);
            })
        };
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let started = Instant::now();
        wakers.park_core(&mut guard, 1, Duration::from_secs(4));
        assert!(*guard);
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(mutex.try_lock().is_none());
        worker.join().unwrap();
    }

    #[test]
    fn core_wakers_timeout_preserves_guard_and_invalid_core_does_not_wait() {
        let mutex = parking_lot::Mutex::new(0x1234u32);
        let wakers = CoreWakers::new();
        let mut guard = mutex.lock();
        let started = Instant::now();
        wakers.park_core(&mut guard, 0, Duration::from_millis(2));
        assert!(started.elapsed() >= Duration::from_millis(2));
        let deadline = Instant::now() + Duration::from_millis(2);
        wakers.park_core_until(&mut guard, 0, deadline);
        assert!(Instant::now() >= deadline);
        assert_eq!(*guard, 0x1234);
        assert!(mutex.try_lock().is_none());
        let generations: [u32; NUM_CORES] = std::array::from_fn(|core| wakers.wakes[core].generation());
        let started = Instant::now();
        wakers.park_core(&mut guard, NUM_CORES, Duration::from_secs(4));
        wakers.park_core_until(&mut guard, NUM_CORES, Instant::now() + Duration::from_secs(4));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(*guard, 0x1234);
        assert!(mutex.try_lock().is_none());
        for core in 0..NUM_CORES {
            assert_eq!(wakers.wakes[core].generation(), generations[core]);
        }
    }

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
    fn idle_core_takes_ready_threads_its_affinity_allows() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 0x2c);
        add_ready_thread(&mut threads, 0x201, 0x2b);
        add_ready_thread(&mut threads, 0x202, 0x2a);
        for (handle, mask) in [(0x200, 0b0110u64), (0x201, 0b0110), (0x202, 0b1000)] {
            let thread = threads.threads.get_mut(&handle).unwrap();
            thread.core = 1;
            thread.affinity_mask = mask;
        }
        threads.threads.get_mut(&0x202).unwrap().core = 3;

        assert_eq!(threads.pick_next(), None);
        threads.threads.get_mut(&0x200).unwrap().affinity_mask = 0b0111;
        threads.threads.get_mut(&0x201).unwrap().affinity_mask = 0b0111;
        assert_eq!(threads.pick_next(), Some(0x201));
        assert_eq!(threads.threads.get(&0x201).unwrap().core, 0);
        assert_eq!(threads.pick_next(), Some(0x200));
        assert_eq!(threads.pick_next(), None);
        assert_eq!(threads.ready.iter().copied().collect::<Vec<_>>(), vec![0x202]);
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

    fn add_waiting(threads: &mut Threads, handle: u32, priority: i32, state: ThreadState) {
        threads.add_thread(handle, ThreadCtx::zero(), 0, 0, 0);
        threads.threads.get_mut(&handle).unwrap().priority = priority;
        threads.transition_state(handle, state);
    }

    fn state_of(threads: &Threads, handle: u32) -> &ThreadState {
        &threads.threads.get(&handle).unwrap().state
    }

    #[test]
    fn signal_handle_wakes_exactly_the_waiters_of_that_handle() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_waiting(&mut threads, 0x200, 30, ThreadState::WaitingHandle { handles: vec![0x10, 0x20], wake_at: None });
        add_waiting(&mut threads, 0x201, 30, ThreadState::WaitingHandle { handles: vec![0x20], wake_at: None });
        add_waiting(&mut threads, 0x202, 30, ThreadState::WaitingHandle { handles: vec![0x30], wake_at: None });

        threads.signal_handle(0x20);
        assert!(matches!(state_of(&threads, 0x200), ThreadState::Ready));
        assert!(matches!(state_of(&threads, 0x201), ThreadState::Ready));
        assert!(matches!(state_of(&threads, 0x202), ThreadState::WaitingHandle { .. }));
        assert_eq!(threads.threads[&0x200].ctx.x[1], 1);
        assert_eq!(threads.threads[&0x201].ctx.x[1], 0);

        // 0x200 no longer waits on 0x10 once woken through 0x20.
        threads.signal_handle(0x10);
        assert!(matches!(state_of(&threads, 0x200), ThreadState::Ready));
        assert_eq!(threads.ready.iter().filter(|&&h| h == 0x200).count(), 1);
        threads.signal_handle(0x30);
        assert!(matches!(state_of(&threads, 0x202), ThreadState::Ready));
    }

    #[test]
    fn mutex_handoff_picks_priority_then_handle_and_tracks_inheritance() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        let mutex = 0x5000;
        let wait = |owner| ThreadState::WaitingMutex { mutex_addr: mutex, owner_handle: owner, tag: 0 };
        add_waiting(&mut threads, 0x203, 20, wait(0x100));
        add_waiting(&mut threads, 0x202, 10, wait(0x100));
        add_waiting(&mut threads, 0x201, 10, wait(0x100));
        assert_eq!(threads.effective_priority(0x100), 10);
        assert!(threads.has_mutex_waiters_for_owner(mutex, 0x100));

        let (h, _, more) = threads.peek_one_mutex_waiter(mutex).unwrap();
        assert_eq!((h, more), (0x201, true));
        threads.commit_wake_mutex_waiter(mutex, h);
        // Remaining waiters now block on the new owner, who inherits from them.
        assert_eq!(threads.effective_priority(0x100), 0x2C);
        assert_eq!(threads.effective_priority(0x201), 10);
        assert!(threads.has_mutex_waiters_for_owner(mutex, 0x201));
        assert!(!threads.has_mutex_waiters_for_owner(mutex, 0x100));

        // A live priority change on a waiter is reflected immediately.
        threads.threads.get_mut(&0x203).unwrap().priority = 5;
        assert_eq!(threads.effective_priority(0x201), 5);

        let (h, _, more) = threads.peek_one_mutex_waiter(mutex).unwrap();
        assert_eq!((h, more), (0x203, true));
        threads.commit_wake_mutex_waiter(mutex, h);
        let (h, _, more) = threads.peek_one_mutex_waiter(mutex).unwrap();
        assert_eq!((h, more), (0x202, false));
        threads.commit_wake_mutex_waiter(mutex, h);
        assert!(threads.peek_one_mutex_waiter(mutex).is_none());
        assert!(!threads.has_mutex_waiters(mutex));
    }

    #[test]
    fn condvar_and_arbiter_queues_are_per_address() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        let cv = |addr| ThreadState::WaitingCondvar { mutex_addr: 0x9000, condvar_addr: addr, wake_at: None, spurious_wake: false };
        let arb = |addr| ThreadState::WaitingArbiter { addr, value: 0, wake_at: None };
        add_waiting(&mut threads, 0x200, 40, cv(0x1000));
        add_waiting(&mut threads, 0x201, 30, cv(0x1000));
        add_waiting(&mut threads, 0x202, 30, cv(0x2000));
        add_waiting(&mut threads, 0x203, 30, arb(0x3000));
        add_waiting(&mut threads, 0x204, 30, arb(0x3000));
        add_waiting(&mut threads, 0x205, 30, arb(0x4000));

        assert_eq!(threads.peek_one_condvar_waiter(0x1000), Some((0x201, 0x9000)));
        assert_eq!(threads.wake_all_on_condvar(0x1000), 2);
        assert!(!threads.has_condvar_waiters(0x1000));
        assert!(threads.has_condvar_waiters(0x2000));
        assert_eq!(threads.wake_one_on_condvar(0x2000), Some(0x202));
        assert_eq!(threads.wake_one_on_condvar(0x2000), None);

        assert_eq!(threads.arbiter_waiter_count(0x3000), 2);
        // Wake-one on an arbiter is first-come first-served.
        assert_eq!(threads.wake_one_on_arbiter(0x3000), Some(0x203));
        assert_eq!(threads.arbiter_waiter_count(0x3000), 1);
        assert_eq!(threads.wake_all_on_arbiter(0x3000), 1);
        assert_eq!(threads.wake_all_on_arbiter(0x3000), 0);
        assert_eq!(threads.arbiter_waiter_count(0x4000), 1);

        // A thread that re-waits on the same address after waking is queued again.
        threads.transition_state(0x203, arb(0x3000));
        assert_eq!(threads.arbiter_waiter_count(0x3000), 1);
    }

    #[test]
    fn deadline_cache_skips_until_a_timer_is_due() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        let now = Instant::now();
        assert!(!threads.timers_due(now));
        add_waiting(&mut threads, 0x200, 30, ThreadState::Sleeping { wake_at: now + Duration::from_secs(60) });
        add_waiting(&mut threads, 0x201, 30, ThreadState::WaitingArbiter { addr: 0x10, value: 0, wake_at: Some(now + Duration::from_millis(5)) });
        add_waiting(&mut threads, 0x202, 30, ThreadState::WaitingCondvar { mutex_addr: 0, condvar_addr: 0x20, wake_at: Some(now + Duration::from_millis(1)), spurious_wake: false });
        assert!(!threads.timers_due(now));

        // Condvar deadlines are resolved by the kernel, not here, but keep the
        // cache armed so that pass still runs.
        let later = now + Duration::from_millis(10);
        assert!(threads.timers_due(later));
        threads.wake_due_sleepers(later);
        assert!(matches!(state_of(&threads, 0x201), ThreadState::Ready));
        assert_eq!(threads.threads[&0x201].ctx.x[0], nexium_common::result::KERNEL_TIMEOUT as u64);
        assert!(matches!(state_of(&threads, 0x200), ThreadState::Sleeping { .. }));
        assert!(matches!(state_of(&threads, 0x202), ThreadState::WaitingCondvar { .. }));
        assert!(threads.timers_due(later));

        threads.transition_state(0x202, ThreadState::Ready);
        threads.wake_due_sleepers(later);
        assert!(!threads.timers_due(later));
        assert!(threads.timers_due(now + Duration::from_secs(61)));
        threads.wake_due_sleepers(now + Duration::from_secs(61));
        assert!(matches!(state_of(&threads, 0x200), ThreadState::Ready));
        assert!(!threads.timers_due(now + Duration::from_secs(3600)));
    }

    #[test]
    fn ready_queue_membership_stays_unique() {
        let mut threads = Threads::new(0x100, 0, 0, 0);
        add_ready_thread(&mut threads, 0x200, 30);
        threads.transition_state(0x200, ThreadState::Ready);
        assert_eq!(threads.ready.iter().filter(|&&h| h == 0x200).count(), 1);
        threads.transition_state(0x200, ThreadState::Exited);
        assert!(!threads.ready.contains(&0x200));
        threads.transition_state(0x200, ThreadState::Ready);
        assert_eq!(threads.ready.iter().filter(|&&h| h == 0x200).count(), 1);
    }
}

/// Differential test of the indexed thread table against a reference model.
///
/// `Model` is the pre-WaitIndex implementation (linear scans over the whole
/// table) ported verbatim, with one deliberate change: where the original
/// picked or woke waiters in `HashMap` iteration order (wake-one / wake-all on
/// condvar and arbiter, `signal_handle`), the model uses FIFO by wait start,
/// the documented new behaviour. Where the original *and* the new code both
/// iterate the thread `HashMap` (timeout scans), the model is handed the real
/// table's iteration order so that both see the same order.
#[cfg(all(test, feature = "backend-rustarmic"))]
mod model_tests {
    use super::{ThreadCtx, ThreadState, Threads, NUM_CORES};
    use crate::kernel::cpu_local::set_current_cpu;
    use nexium_common::result::{KERNEL_CANCELLED, KERNEL_TIMEOUT, SUCCESS};
    use nexium_cpu::Cpu;
    use std::collections::{HashMap, VecDeque};
    use std::time::{Duration, Instant};

    // ---------------------------------------------------------------- model

    #[derive(Clone)]
    struct MThread {
        state: ThreadState,
        priority: i32,
        core: i32,
        ideal_core: i32,
        affinity_mask: u64,
        wait_cancelled: bool,
        x0: u64,
        x1: u64,
        /// Sequence number of the last state assignment (= wait start).
        seq: u64,
    }

    #[derive(Default)]
    struct Model {
        threads: HashMap<u32, MThread>,
        ready: VecDeque<u32>,
        current: [Option<u32>; NUM_CORES],
        seq: u64,
    }

    impl Model {
        fn new(main: u32) -> Self {
            let mut model = Model::default();
            model.threads.insert(
                main,
                MThread {
                    state: ThreadState::Running,
                    priority: 0x2C,
                    core: 0,
                    ideal_core: 0,
                    affinity_mask: 1,
                    wait_cancelled: false,
                    x0: 0,
                    x1: 0,
                    seq: 0,
                },
            );
            model.current[0] = Some(main);
            model
        }

        fn add_thread(&mut self, handle: u32) {
            self.threads.insert(
                handle,
                MThread {
                    state: ThreadState::Created,
                    priority: 0x2C,
                    core: -2,
                    ideal_core: super::IDEAL_CORE_USE_PROCESS_VALUE,
                    affinity_mask: 0,
                    wait_cancelled: false,
                    x0: 0,
                    x1: 0,
                    seq: 0,
                },
            );
        }

        fn set_state(&mut self, h: u32, state: ThreadState) {
            self.seq += 1;
            let seq = self.seq;
            if let Some(t) = self.threads.get_mut(&h) {
                t.state = state;
                t.seq = seq;
            }
        }

        /// Handles matching `pred`, in wait-start order.
        fn fifo(&self, pred: impl Fn(&ThreadState) -> bool) -> Vec<u32> {
            let mut v: Vec<(u64, u32)> = self
                .threads
                .iter()
                .filter(|(_, t)| pred(&t.state))
                .map(|(h, t)| (t.seq, *h))
                .collect();
            v.sort_unstable();
            v.into_iter().map(|(_, h)| h).collect()
        }

        fn transition_state(&mut self, handle: u32, new_state: ThreadState) {
            self.ready.retain(|&h| h != handle);
            if !self.threads.contains_key(&handle) {
                return;
            }
            let became_ready = matches!(new_state, ThreadState::Ready);
            self.set_state(handle, new_state);
            if became_ready && !self.ready.contains(&handle) {
                self.ready.push_back(handle);
            }
        }

        fn wake_due_sleepers(&mut self, now: Instant, order: &[u32]) {
            let mut woken = Vec::new();
            for h in order {
                let Some(t) = self.threads.get(h) else { continue };
                let due = match &t.state {
                    ThreadState::Sleeping { wake_at } => *wake_at <= now,
                    ThreadState::WaitingHandle { wake_at: Some(d), .. } => *d <= now,
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
                        ThreadState::WaitingHandle { .. } | ThreadState::WaitingArbiter { .. } => {
                            t.x0 = KERNEL_TIMEOUT as u64;
                        }
                        _ => {}
                    }
                }
                self.transition_state(h, ThreadState::Ready);
            }
        }

        /// The condvar-timeout half of `Kernel::wake_due_sleepers` (plain
        /// timeout path: no pending signal, no spurious wake).
        fn kernel_condvar_timeouts(&mut self, now: Instant, order: &[u32]) {
            let timed_out: Vec<u32> = order
                .iter()
                .copied()
                .filter(|h| {
                    matches!(
                        self.threads.get(h).map(|t| &t.state),
                        Some(ThreadState::WaitingCondvar { wake_at: Some(d), .. }) if *d <= now
                    )
                })
                .collect();
            for h in timed_out {
                self.threads.get_mut(&h).unwrap().x0 = KERNEL_TIMEOUT as u64;
                self.transition_state(h, ThreadState::Ready);
            }
        }

        fn peek_one_mutex_waiter(&self, mutex_addr: u64) -> Option<(u32, u32, bool)> {
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

        fn commit_wake_mutex_waiter(&mut self, mutex_addr: u64, h: u32) {
            for (other_h, t) in self.threads.iter_mut() {
                if *other_h == h {
                    continue;
                }
                if let ThreadState::WaitingMutex { mutex_addr: m, owner_handle: o, .. } =
                    &mut t.state
                {
                    if *m == mutex_addr {
                        *o = h;
                    }
                }
            }
            if let Some(t) = self.threads.get_mut(&h) {
                t.x0 = SUCCESS as u64;
            }
            self.transition_state(h, ThreadState::Ready);
        }

        fn wake_one_on_condvar(&mut self, condvar_addr: u64) -> Option<u32> {
            let h = *self
                .fifo(|s| matches!(s, ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr))
                .first()?;
            self.threads.get_mut(&h).unwrap().x0 = SUCCESS as u64;
            self.transition_state(h, ThreadState::Ready);
            Some(h)
        }

        fn peek_one_condvar_waiter(&self, condvar_addr: u64) -> Option<(u32, u64)> {
            self.threads
                .iter()
                .filter_map(|(h, t)| match &t.state {
                    ThreadState::WaitingCondvar { condvar_addr: c, mutex_addr, .. }
                        if *c == condvar_addr =>
                    {
                        Some((*h, t.priority, *mutex_addr))
                    }
                    _ => None,
                })
                .min_by_key(|(h, priority, _)| (*priority, *h))
                .map(|(h, _, mutex_addr)| (h, mutex_addr))
        }

        fn wake_condvar_into_mutex_waiter(&mut self, handle: u32, mutex_addr: u64, owner_handle: u32, tag: u32) {
            if let Some(t) = self.threads.get_mut(&handle) {
                t.x0 = SUCCESS as u64;
            }
            self.transition_state(handle, ThreadState::WaitingMutex { mutex_addr, owner_handle, tag });
        }

        fn wake_condvar_to_ready(&mut self, handle: u32) {
            if let Some(t) = self.threads.get_mut(&handle) {
                t.x0 = SUCCESS as u64;
            }
            self.transition_state(handle, ThreadState::Ready);
        }

        fn has_mutex_waiters_for_owner(&self, mutex_addr: u64, owner_handle: u32) -> bool {
            self.threads.values().any(|t| {
                matches!(
                    &t.state,
                    ThreadState::WaitingMutex { mutex_addr: m, owner_handle: o, .. }
                        if *m == mutex_addr && *o == owner_handle
                )
            })
        }

        fn has_condvar_waiters(&self, condvar_addr: u64) -> bool {
            self.threads.values().any(|t| matches!(&t.state, ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr))
        }

        fn has_mutex_waiters(&self, mutex_addr: u64) -> bool {
            self.threads.values().any(|t| matches!(&t.state, ThreadState::WaitingMutex { mutex_addr: m, .. } if *m == mutex_addr))
        }

        fn wake_all_on_condvar(&mut self, condvar_addr: u64) -> usize {
            let woken = self.fifo(|s| matches!(s, ThreadState::WaitingCondvar { condvar_addr: c, .. } if *c == condvar_addr));
            for h in &woken {
                self.threads.get_mut(h).unwrap().x0 = SUCCESS as u64;
            }
            for &h in &woken {
                self.transition_state(h, ThreadState::Ready);
            }
            woken.len()
        }

        fn signal_handle(&mut self, handle: u32) {
            let woken = self.fifo(|s| matches!(s, ThreadState::WaitingHandle { handles, .. } if handles.contains(&handle)));
            for h in &woken {
                let t = self.threads.get_mut(h).unwrap();
                t.x0 = SUCCESS as u64;
                if let ThreadState::WaitingHandle { handles, .. } = &t.state {
                    if let Some(idx) = handles.iter().position(|&v| v == handle) {
                        t.x1 = idx as u64;
                    }
                }
            }
            for h in woken {
                self.transition_state(h, ThreadState::Ready);
            }
        }

        fn cancel_synchronization(&mut self, handle: u32) -> Option<bool> {
            let waiting = matches!(
                self.threads.get(&handle).map(|thread| &thread.state),
                Some(ThreadState::WaitingHandle { .. })
            );
            let thread = self.threads.get_mut(&handle)?;
            if waiting {
                thread.wait_cancelled = false;
                thread.x0 = KERNEL_CANCELLED as u64;
                self.transition_state(handle, ThreadState::Ready);
            } else {
                thread.wait_cancelled = true;
            }
            Some(waiting)
        }

        fn take_wait_cancelled(&mut self, handle: u32) -> bool {
            let Some(thread) = self.threads.get_mut(&handle) else {
                return false;
            };
            std::mem::take(&mut thread.wait_cancelled)
        }

        fn arbiter_waiter_count(&self, addr: u64) -> usize {
            self.threads
                .values()
                .filter(|t| matches!(&t.state, ThreadState::WaitingArbiter { addr: a, .. } if *a == addr))
                .count()
        }

        fn wake_one_on_arbiter(&mut self, addr: u64) -> Option<u32> {
            let h = *self
                .fifo(|s| matches!(s, ThreadState::WaitingArbiter { addr: a, .. } if *a == addr))
                .first()?;
            self.threads.get_mut(&h).unwrap().x0 = SUCCESS as u64;
            self.transition_state(h, ThreadState::Ready);
            Some(h)
        }

        fn wake_all_on_arbiter(&mut self, addr: u64) -> usize {
            let woken = self.fifo(|s| matches!(s, ThreadState::WaitingArbiter { addr: a, .. } if *a == addr));
            for h in &woken {
                self.threads.get_mut(h).unwrap().x0 = SUCCESS as u64;
            }
            for &h in &woken {
                self.transition_state(h, ThreadState::Ready);
            }
            woken.len()
        }

        fn effective_priority(&self, handle: u32) -> i32 {
            let base = self.threads.get(&handle).map(|t| t.priority).unwrap_or(i32::MAX);
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

        fn set_thread_core_mask(
            &mut self,
            handle: u32,
            core_id: i32,
            affinity_mask: u64,
            process_ideal_core: i32,
            active_cores: i32,
        ) -> Result<(), u32> {
            use super::{IDEAL_CORE_DONT_CARE, IDEAL_CORE_NO_UPDATE, IDEAL_CORE_USE_PROCESS_VALUE};
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
            }
            Ok(())
        }

        fn best_ready_position(&self, eligible: impl Fn(&MThread) -> bool) -> Option<usize> {
            self.ready
                .iter()
                .enumerate()
                .filter(|(_, h)| self.threads.get(h).is_some_and(&eligible))
                .min_by_key(|(idx, h)| (self.effective_priority(**h), *idx))
                .map(|(idx, _)| idx)
        }

        fn next_ready_position(&self, core: i32) -> Option<usize> {
            self.best_ready_position(|t| t.core == core || t.core < 0).or_else(|| {
                self.best_ready_position(|t| {
                    t.core != core
                        && (0..NUM_CORES as i32).contains(&core)
                        && t.affinity_mask & (1u64 << core) != 0
                })
            })
        }

        fn pick_next(&mut self, core: i32) -> Option<u32> {
            let pos = self.next_ready_position(core)?;
            let handle = self.ready.remove(pos)?;
            if let Some(t) = self.threads.get_mut(&handle) {
                t.core = core;
            }
            Some(handle)
        }

        fn has_ready_for_core(&self, core: i32) -> bool {
            self.ready
                .iter()
                .any(|h| self.threads.get(h).map_or(false, |t| t.core == core || t.core < 0))
        }

        fn has_higher_priority_ready_for_core(&self, core: usize) -> bool {
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

        fn yield_with_state_position(
            &mut self,
            core: usize,
            new_state: ThreadState,
            ready_front: bool,
            saved: (u64, u64),
        ) -> Option<u32> {
            let h = self.current[core]?;
            if let Some(t) = self.threads.get_mut(&h) {
                t.x0 = saved.0;
                t.x1 = saved.1;
            }
            let became_ready = matches!(new_state, ThreadState::Ready);
            self.set_state(h, new_state);
            if became_ready {
                self.enqueue_ready(h, ready_front);
            }
            self.current[core] = None;
            Some(h)
        }

        fn schedule_to(&mut self, core: usize, handle: u32) {
            self.set_state(handle, ThreadState::Running);
            self.current[core] = Some(handle);
        }

        fn earliest_wake(&self) -> Option<Instant> {
            self.threads.values().filter_map(|t| deadline_of(&t.state)).min()
        }

        fn drop_exited(&mut self) {
            self.threads.retain(|_, t| !matches!(t.state, ThreadState::Exited));
        }
    }

    fn deadline_of(state: &ThreadState) -> Option<Instant> {
        match state {
            ThreadState::Sleeping { wake_at } => Some(*wake_at),
            ThreadState::WaitingHandle { wake_at: Some(d), .. }
            | ThreadState::WaitingCondvar { wake_at: Some(d), .. }
            | ThreadState::WaitingArbiter { wake_at: Some(d), .. } => Some(*d),
            _ => None,
        }
    }

    // ------------------------------------------------------------------ rng

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn chance(&mut self, percent: u64) -> bool {
            self.below(100) < percent
        }
        fn pick<T: Copy>(&mut self, items: &[T]) -> Option<T> {
            if items.is_empty() {
                None
            } else {
                Some(items[self.below(items.len() as u64) as usize])
            }
        }
    }

    // -------------------------------------------------------------- harness

    const MAIN: u32 = 0x100;
    const MUTEXES: [u64; 4] = [0x1000, 0x1004, 0x1008, 0x100c];
    const CONDVARS: [u64; 3] = [0x2000, 0x2004, 0x2008];
    const ARBITERS: [u64; 3] = [0x3000, 0x3004, 0x3008];
    const MAX_LIVE_THREADS: usize = 24;

    struct Harness {
        real: Threads,
        model: Model,
        cpu: Cpu,
        rng: Rng,
        now: Instant,
        /// Guest-side mutex words: mutex address -> owning thread.
        owners: HashMap<u64, u32>,
        next_handle: u32,
        events: Vec<u32>,
        closed_events: Vec<u32>,
        trace: VecDeque<String>,
        seed: u64,
        step: usize,
        /// Wake events that the index-backed paths produced (for coverage).
        stats: HashMap<&'static str, usize>,
    }

    impl Harness {
        fn new(seed: u64) -> Self {
            let mut real = Threads::new(MAIN, 0, 0, 0);
            real.threads.get_mut(&MAIN).unwrap().priority = 0x2C;
            Harness {
                real,
                model: Model::new(MAIN),
                cpu: Cpu::new_rustarmic().expect("rustarmic cpu"),
                rng: Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5eed),
                now: Instant::now(),
                owners: HashMap::new(),
                next_handle: 0x200,
                events: (0..6).map(|i| 0x9000 + i).collect(),
                closed_events: Vec::new(),
                trace: VecDeque::new(),
                seed,
                step: 0,
                stats: HashMap::new(),
            }
        }

        fn note(&mut self, text: String) {
            if self.trace.len() == 60 {
                self.trace.pop_front();
            }
            self.trace.push_back(text);
        }

        fn bump(&mut self, key: &'static str) {
            *self.stats.entry(key).or_default() += 1;
        }

        fn fail(&self, what: String) -> ! {
            panic!(
                "seed {} step {}: {}\nrecent ops:\n  {}",
                self.seed,
                self.step,
                what,
                self.trace.iter().cloned().collect::<Vec<_>>().join("\n  ")
            );
        }

        fn agree<T: PartialEq + std::fmt::Debug>(&self, what: &str, real: T, model: T) {
            if real != model {
                self.fail(format!("{what}: real {real:?} != model {model:?}"));
            }
        }

        fn running(&self) -> Vec<(usize, u32)> {
            (0..NUM_CORES)
                .filter_map(|core| self.real.current[core].map(|h| (core, h)))
                .collect()
        }

        fn live(&self) -> Vec<u32> {
            let mut v: Vec<u32> = self
                .model
                .threads
                .iter()
                .filter(|(_, t)| !matches!(t.state, ThreadState::Exited))
                .map(|(h, _)| *h)
                .collect();
            v.sort_unstable();
            v
        }

        fn deadline(&mut self) -> Instant {
            // Includes deadlines that are already due and exact ties.
            match self.rng.below(10) {
                0 => self.now,
                1 => self.now - Duration::from_micros(self.rng.below(50)),
                _ => self.now + Duration::from_micros(self.rng.below(4000)),
            }
        }

        fn maybe_deadline(&mut self) -> Option<Instant> {
            if self.rng.chance(50) {
                Some(self.deadline())
            } else {
                None
            }
        }

        /// Current thread on `core` blocks/yields with `state`, the way an SVC
        /// does through `Threads::yield_with_state` / `preempt_current`.
        fn yield_current(&mut self, core: usize, state: ThreadState, front: bool) {
            let saved = (self.rng.next(), self.rng.next());
            self.cpu.set_thread_register(0, saved.0);
            self.cpu.set_thread_register(1, saved.1);
            let _guard = set_current_cpu(&mut self.cpu, core);
            let real = if front {
                self.real.preempt_current(&self.cpu)
            } else {
                self.real.yield_with_state(&self.cpu, state.clone())
            };
            let model = self.model.yield_with_state_position(core, state, front, saved);
            drop(_guard);
            self.agree("yield returned handle", real, model);
        }

        fn release_mutex(&mut self, mutex: u64) {
            let real = self.real.peek_one_mutex_waiter(mutex);
            let model = self.model.peek_one_mutex_waiter(mutex);
            self.agree("peek_one_mutex_waiter", real, model);
            match real {
                Some((h, _, _)) => {
                    self.real.commit_wake_mutex_waiter(mutex, h);
                    self.model.commit_wake_mutex_waiter(mutex, h);
                    self.owners.insert(mutex, h);
                    self.bump("mutex handoff");
                }
                None => {
                    self.owners.remove(&mutex);
                }
            }
        }

        fn kernel_wake(&mut self, now: Instant) {
            if self.model.threads.values().any(|t| deadline_of(&t.state).is_some_and(|d| d <= now)) {
                self.bump("timeout wake");
            }
            // Order the original and the new code both iterate in.
            let order: Vec<u32> = self.real.threads.keys().copied().collect();
            // Real: `Kernel::wake_due_sleepers` behind its `timers_due` gate.
            if self.real.timers_due(now) {
                let timed_out: Vec<u32> = self
                    .real
                    .threads
                    .iter()
                    .filter_map(|(h, t)| match &t.state {
                        ThreadState::WaitingCondvar { wake_at: Some(d), .. } if *d <= now => Some(*h),
                        _ => None,
                    })
                    .collect();
                for h in timed_out {
                    self.real.threads.get_mut(&h).unwrap().ctx.x[0] = KERNEL_TIMEOUT as u64;
                    self.real.transition_state(h, ThreadState::Ready);
                }
                self.real.wake_due_sleepers(now);
            }
            // Model: the original, ungated scans.
            self.model.kernel_condvar_timeouts(now, &order);
            self.model.wake_due_sleepers(now, &order);
        }

        fn op(&mut self) {
            let running = self.running();
            let acting = self.rng.pick(&running);
            let choice = self.rng.below(100);
            match (choice, acting) {
                // -- scheduling
                (0..=11, _) => {
                    let core = self.rng.below(NUM_CORES as u64) as usize;
                    if self.real.current[core].is_some() {
                        return;
                    }
                    let _guard = set_current_cpu(&mut self.cpu, core);
                    let real = self.real.pick_next();
                    let model = self.model.pick_next(core as i32);
                    drop(_guard);
                    self.note(format!("pick_next core{core} -> {real:x?}"));
                    self.agree("pick_next", real, model);
                    if let Some(h) = real {
                        let _guard = set_current_cpu(&mut self.cpu, core);
                        self.real.schedule_to(h, &mut self.cpu);
                        drop(_guard);
                        self.model.schedule_to(core, h);
                    }
                }
                (12..=15, Some((core, h))) => {
                    let front = self.rng.chance(50);
                    self.note(format!("{} {h:#x} core{core}", if front { "preempt" } else { "yield" }));
                    self.yield_current(core, ThreadState::Ready, front);
                }
                (16..=19, Some((core, h))) => {
                    let wake_at = self.deadline();
                    self.note(format!("sleep {h:#x}"));
                    self.yield_current(core, ThreadState::Sleeping { wake_at }, false);
                }
                // -- mutexes
                (20..=27, Some((core, h))) => {
                    let mutex = self.rng.pick(&MUTEXES).unwrap();
                    match self.owners.get(&mutex).copied() {
                        None => {
                            self.note(format!("lock {mutex:#x} by {h:#x} (free)"));
                            self.owners.insert(mutex, h);
                        }
                        Some(owner) if owner == h => {}
                        Some(owner) => {
                            self.note(format!("lock {mutex:#x} by {h:#x} contended owner {owner:#x}"));
                            self.bump("mutex contended");
                            self.yield_current(
                                core,
                                ThreadState::WaitingMutex { mutex_addr: mutex, owner_handle: owner, tag: h },
                                false,
                            );
                        }
                    }
                }
                (28..=34, Some((_, h))) => {
                    let owned: Vec<u64> = MUTEXES
                        .iter()
                        .copied()
                        .filter(|m| self.owners.get(m) == Some(&h))
                        .collect();
                    if let Some(mutex) = self.rng.pick(&owned) {
                        self.note(format!("unlock {mutex:#x} by {h:#x}"));
                        self.release_mutex(mutex);
                    }
                }
                (35..=36, _) => {
                    // Release a mutex whose owner can no longer do it (exited,
                    // or blocked): exercises handoff away from dead owners.
                    let stuck: Vec<u64> = MUTEXES
                        .iter()
                        .copied()
                        .filter(|m| {
                            self.owners.get(m).is_some_and(|o| {
                                !matches!(
                                    self.model.threads.get(o).map(|t| &t.state),
                                    Some(ThreadState::Running)
                                )
                            })
                        })
                        .collect();
                    if let Some(mutex) = self.rng.pick(&stuck) {
                        self.note(format!("force-unlock {mutex:#x}"));
                        self.release_mutex(mutex);
                    }
                }
                // -- condition variables
                (37..=43, Some((core, h))) => {
                    let mutex = self.rng.pick(&MUTEXES).unwrap();
                    let condvar = self.rng.pick(&CONDVARS).unwrap();
                    match self.owners.get(&mutex).copied() {
                        Some(owner) if owner != h => return,
                        _ => {}
                    }
                    self.owners.insert(mutex, h);
                    self.release_mutex(mutex);
                    let wake_at = self.maybe_deadline();
                    self.note(format!("condvar wait {h:#x} cv {condvar:#x} mutex {mutex:#x} timeout {}", wake_at.is_some()));
                    self.yield_current(
                        core,
                        ThreadState::WaitingCondvar { mutex_addr: mutex, condvar_addr: condvar, wake_at, spurious_wake: false },
                        false,
                    );
                }
                (44..=49, _) => {
                    // svcSignalProcessWideKey
                    let condvar = self.rng.pick(&CONDVARS).unwrap();
                    let count = self.rng.below(4) as i32 - 1;
                    let max = if count <= 0 { i32::MAX } else { count };
                    self.note(format!("condvar signal {condvar:#x} count {count}"));
                    for _ in 0..max {
                        let real = self.real.peek_one_condvar_waiter(condvar);
                        let model = self.model.peek_one_condvar_waiter(condvar);
                        self.agree("peek_one_condvar_waiter", real, model);
                        let Some((waiter, mutex)) = real else { break };
                        match self.owners.get(&mutex).copied() {
                            None => {
                                self.owners.insert(mutex, waiter);
                                self.real.wake_condvar_to_ready(waiter);
                                self.model.wake_condvar_to_ready(waiter);
                                self.bump("condvar to ready");
                            }
                            Some(holder) => {
                                self.real.wake_condvar_into_mutex_waiter(waiter, mutex, holder, waiter);
                                self.model.wake_condvar_into_mutex_waiter(waiter, mutex, holder, waiter);
                                self.bump("condvar requeue");
                            }
                        }
                    }
                }
                (50..=52, _) => {
                    let condvar = self.rng.pick(&CONDVARS).unwrap();
                    if self.rng.chance(50) {
                        let real = self.real.wake_one_on_condvar(condvar);
                        let model = self.model.wake_one_on_condvar(condvar);
                        self.note(format!("wake_one_on_condvar {condvar:#x} -> {real:x?}"));
                        self.agree("wake_one_on_condvar", real, model);
                    } else {
                        let real = self.real.wake_all_on_condvar(condvar);
                        let model = self.model.wake_all_on_condvar(condvar);
                        self.note(format!("wake_all_on_condvar {condvar:#x} -> {real}"));
                        self.agree("wake_all_on_condvar", real, model);
                    }
                }
                // -- address arbiter
                (53..=58, Some((core, h))) => {
                    let addr = self.rng.pick(&ARBITERS).unwrap();
                    let wake_at = self.maybe_deadline();
                    let value = self.rng.below(3) as u32;
                    self.note(format!("arbiter wait {h:#x} addr {addr:#x}"));
                    self.yield_current(core, ThreadState::WaitingArbiter { addr, value, wake_at }, false);
                }
                (59..=63, _) => {
                    let addr = self.rng.pick(&ARBITERS).unwrap();
                    self.agree(
                        "arbiter_waiter_count",
                        self.real.arbiter_waiter_count(addr),
                        self.model.arbiter_waiter_count(addr),
                    );
                    let count = self.rng.below(4) as i32 - 1;
                    self.note(format!("arbiter signal {addr:#x} count {count}"));
                    if count <= 0 {
                        let real = self.real.wake_all_on_arbiter(addr);
                        let model = self.model.wake_all_on_arbiter(addr);
                        self.agree("wake_all_on_arbiter", real, model);
                    } else {
                        for _ in 0..count {
                            let real = self.real.wake_one_on_arbiter(addr);
                            let model = self.model.wake_one_on_arbiter(addr);
                            self.agree("wake_one_on_arbiter", real, model);
                            if real.is_none() {
                                break;
                            }
                            self.bump("arbiter wake one");
                        }
                    }
                }
                // -- handles
                (64..=70, Some((core, h))) => {
                    let threads = self.live();
                    let n = 1 + self.rng.below(4) as usize;
                    let mut handles = Vec::new();
                    for _ in 0..n {
                        let pick = match self.rng.below(10) {
                            0..=5 => self.rng.pick(&self.events.clone()),
                            6..=7 => self.rng.pick(&threads),
                            8 => handles.last().copied(), // duplicate entry
                            _ => self.rng.pick(&self.closed_events.clone()),
                        };
                        if let Some(p) = pick {
                            handles.push(p);
                        }
                    }
                    let real = self.real.take_wait_cancelled(h);
                    let model = self.model.take_wait_cancelled(h);
                    self.agree("take_wait_cancelled", real, model);
                    if real {
                        self.note(format!("wait handles {h:#x}: cancelled early"));
                        self.bump("early cancel consumed");
                        return;
                    }
                    let wake_at = self.maybe_deadline();
                    self.note(format!("wait handles {h:#x} {handles:x?} timeout {}", wake_at.is_some()));
                    self.yield_current(core, ThreadState::WaitingHandle { handles, wake_at }, false);
                }
                (71..=75, _) => {
                    let pool: Vec<u32> = if self.rng.chance(80) {
                        self.events.clone()
                    } else {
                        self.closed_events.clone()
                    };
                    if let Some(ev) = self.rng.pick(&pool) {
                        self.note(format!("signal {ev:#x}"));
                        let waiters = self
                            .model
                            .fifo(|s| matches!(s, ThreadState::WaitingHandle { handles, .. } if handles.contains(&ev)))
                            .len();
                        if waiters >= 2 {
                            self.bump("multi-waiter signal");
                        }
                        self.real.signal_handle(ev);
                        self.model.signal_handle(ev);
                    }
                }
                (76, _) => {
                    // Close an event while threads may be waiting on it; the
                    // kernel leaves those waiters alone.
                    if self.events.len() > 2 {
                        let idx = self.rng.below(self.events.len() as u64) as usize;
                        let ev = self.events.remove(idx);
                        self.closed_events.push(ev);
                        let fresh = 0x9000 + self.events.len() as u32 + self.closed_events.len() as u32 * 0x10;
                        self.events.push(fresh);
                        self.note(format!("close {ev:#x}, new event {fresh:#x}"));
                    }
                }
                (77..=78, _) => {
                    let mut targets = self.live();
                    targets.push(0xdead);
                    let target = self.rng.pick(&targets).unwrap();
                    let real = self.real.cancel_synchronization(target);
                    let model = self.model.cancel_synchronization(target);
                    self.note(format!("cancel {target:#x} -> {real:?}"));
                    self.agree("cancel_synchronization", real, model);
                }
                // -- clock
                (79..=85, _) => {
                    self.now += Duration::from_micros(self.rng.below(2500));
                    let now = self.now;
                    self.note(format!("advance clock (kernel wake) +{:?}", now));
                    self.kernel_wake(now);
                }
                (86, _) => {
                    // `Threads::wake_due_sleepers` alone, as `ensure_thread_loaded`
                    // and the core loops call it.
                    self.now += Duration::from_micros(self.rng.below(2500));
                    let now = self.now;
                    let order: Vec<u32> = self.real.threads.keys().copied().collect();
                    self.note("advance clock (threads wake)".to_string());
                    self.real.wake_due_sleepers(now);
                    self.model.wake_due_sleepers(now, &order);
                }
                // -- thread lifecycle and attributes
                (87..=89, _) => {
                    if self.live().len() >= MAX_LIVE_THREADS {
                        return;
                    }
                    let h = self.next_handle;
                    self.next_handle += 1;
                    let priority = self.rng.below(64) as i32;
                    let ideal = self.rng.below(NUM_CORES as u64) as i32;
                    self.real.add_thread(h, ThreadCtx::zero(), 0, 0, 0);
                    self.model.add_thread(h);
                    // as svcCreateThread
                    {
                        let t = self.real.threads.get_mut(&h).unwrap();
                        t.priority = priority;
                        t.ideal_core = ideal;
                        t.affinity_mask = 1u64 << ideal;
                        t.core = ideal;
                    }
                    {
                        let t = self.model.threads.get_mut(&h).unwrap();
                        t.priority = priority;
                        t.ideal_core = ideal;
                        t.affinity_mask = 1u64 << ideal;
                        t.core = ideal;
                    }
                    self.note(format!("create {h:#x} prio {priority} core {ideal}"));
                    if self.rng.chance(85) {
                        self.real.transition_state(h, ThreadState::Ready);
                        self.model.transition_state(h, ThreadState::Ready);
                    }
                }
                (90, _) => {
                    let created: Vec<u32> = self
                        .live()
                        .into_iter()
                        .filter(|h| matches!(self.model.threads[h].state, ThreadState::Created))
                        .collect();
                    if let Some(h) = self.rng.pick(&created) {
                        self.note(format!("start {h:#x}"));
                        self.real.transition_state(h, ThreadState::Ready);
                        self.model.transition_state(h, ThreadState::Ready);
                    }
                }
                (91, Some((core, h))) if h != MAIN || self.rng.chance(10) => {
                    // svcExitThread, possibly while owning mutexes with waiters.
                    self.note(format!("exit {h:#x}"));
                    self.real.signal_handle(h);
                    self.model.signal_handle(h);
                    self.yield_current(core, ThreadState::Exited, false);
                }
                (92, _) => {
                    // A thread torn down while blocked (or ready).
                    let current: Vec<u32> = self.real.current.iter().flatten().copied().collect();
                    let victims: Vec<u32> = self
                        .live()
                        .into_iter()
                        .filter(|h| !current.contains(h))
                        .collect();
                    if let Some(h) = self.rng.pick(&victims) {
                        self.note(format!("terminate {h:#x} in {:?}", self.model.threads[&h].state));
                        self.bump("terminate");
                        self.real.signal_handle(h);
                        self.model.signal_handle(h);
                        self.real.transition_state(h, ThreadState::Exited);
                        self.model.transition_state(h, ThreadState::Exited);
                    }
                }
                (93..=95, _) => {
                    let threads = self.live();
                    if let Some(h) = self.rng.pick(&threads) {
                        let priority = self.rng.below(64) as i32;
                        self.note(format!("set priority {h:#x} = {priority}"));
                        self.real.threads.get_mut(&h).unwrap().priority = priority;
                        self.model.threads.get_mut(&h).unwrap().priority = priority;
                    }
                }
                (96..=97, _) => {
                    let mut threads = self.live();
                    threads.push(0xdead);
                    let h = self.rng.pick(&threads).unwrap();
                    let core_id = self.rng.below(8) as i32 - 3;
                    let mask = self.rng.below(20);
                    let active = 1 + self.rng.below(NUM_CORES as u64) as i32;
                    let real = self.real.set_thread_core_mask(h, core_id, mask, 0, active);
                    let model = self.model.set_thread_core_mask(h, core_id, mask, 0, active);
                    self.note(format!("core mask {h:#x} core {core_id} mask {mask:#b} -> {real:?}"));
                    self.agree("set_thread_core_mask", real, model);
                }
                (98, _) => {
                    self.note("drop_exited".to_string());
                    self.real.drop_exited();
                    self.model.drop_exited();
                }
                _ => {}
            }
        }

        fn check(&self) {
            let real = &self.real;
            let model = &self.model;
            let mut keys: Vec<u32> = real.threads.keys().copied().collect();
            let mut model_keys: Vec<u32> = model.threads.keys().copied().collect();
            keys.sort_unstable();
            model_keys.sort_unstable();
            self.agree("thread set", &keys, &model_keys);

            for &h in &keys {
                let r = &real.threads[&h];
                let m = &model.threads[&h];
                self.agree(&format!("state of {h:#x}"), format!("{:?}", r.state), format!("{:?}", m.state));
                self.agree(&format!("x0 of {h:#x}"), r.ctx.x[0], m.x0);
                self.agree(&format!("x1 of {h:#x}"), r.ctx.x[1], m.x1);
                self.agree(
                    &format!("attributes of {h:#x}"),
                    (r.priority, r.core, r.ideal_core, r.affinity_mask, r.wait_cancelled),
                    (m.priority, m.core, m.ideal_core, m.affinity_mask, m.wait_cancelled),
                );
                self.agree(
                    &format!("effective priority of {h:#x}"),
                    real.effective_priority(h),
                    model.effective_priority(h),
                );
                self.agree(&format!("in_ready of {h:#x}"), r.in_ready, real.ready.contains(&h));
            }
            self.agree("effective priority of unknown", real.effective_priority(0xdead), model.effective_priority(0xdead));
            self.agree("ready queue", real.ready.iter().collect::<Vec<_>>(), model.ready.iter().collect::<Vec<_>>());
            self.agree("current", real.current, model.current);
            for core in 0..NUM_CORES {
                let r = real.next_ready_position(core as i32).map(|pos| real.ready[pos]);
                let m = model.next_ready_position(core as i32).map(|pos| model.ready[pos]);
                self.agree(&format!("next pick on core{core}"), r, m);
                self.agree(
                    &format!("has_ready_for_core({core})"),
                    real.has_ready_for_core(core as i32),
                    model.has_ready_for_core(core as i32),
                );
                self.agree(
                    &format!("has_higher_priority_ready_for_core({core})"),
                    real.has_higher_priority_ready_for_core(core),
                    model.has_higher_priority_ready_for_core(core),
                );
            }
            let mut owners_to_probe: Vec<u32> = keys.clone();
            owners_to_probe.push(0xdead);
            for mutex in MUTEXES {
                self.agree("has_mutex_waiters", real.has_mutex_waiters(mutex), model.has_mutex_waiters(mutex));
                self.agree(
                    "peek_one_mutex_waiter",
                    real.peek_one_mutex_waiter(mutex),
                    model.peek_one_mutex_waiter(mutex),
                );
                for &owner in &owners_to_probe {
                    self.agree(
                        "has_mutex_waiters_for_owner",
                        real.has_mutex_waiters_for_owner(mutex, owner),
                        model.has_mutex_waiters_for_owner(mutex, owner),
                    );
                }
            }
            for condvar in CONDVARS {
                self.agree("has_condvar_waiters", real.has_condvar_waiters(condvar), model.has_condvar_waiters(condvar));
                self.agree(
                    "peek_one_condvar_waiter",
                    real.peek_one_condvar_waiter(condvar),
                    model.peek_one_condvar_waiter(condvar),
                );
            }
            for addr in ARBITERS {
                self.agree("arbiter_waiter_count", real.arbiter_waiter_count(addr), model.arbiter_waiter_count(addr));
            }
            self.agree("earliest_wake", real.earliest_wake(), model.earliest_wake());
            self.check_index();
        }

        /// WaitIndex invariants: each waiting thread is listed exactly under
        /// the keys its state implies, in wait-start order, and nowhere else;
        /// the cached deadline is never later than the true earliest one.
        fn check_index(&self) {
            let real = &self.real;
            let model = &self.model;
            let mut mutex: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut owner: HashMap<u32, Vec<u32>> = HashMap::new();
            let mut condvar: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut arbiter: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut handle: HashMap<u32, Vec<u32>> = HashMap::new();
            let mut by_seq: Vec<(u64, u32)> = model.threads.iter().map(|(h, t)| (t.seq, *h)).collect();
            by_seq.sort_unstable();
            for (_, h) in by_seq {
                match &real.threads[&h].state {
                    ThreadState::WaitingMutex { mutex_addr, owner_handle, .. } => {
                        mutex.entry(*mutex_addr).or_default().push(h);
                        owner.entry(*owner_handle).or_default().push(h);
                    }
                    ThreadState::WaitingCondvar { condvar_addr, .. } => {
                        condvar.entry(*condvar_addr).or_default().push(h);
                    }
                    ThreadState::WaitingArbiter { addr, .. } => {
                        arbiter.entry(*addr).or_default().push(h);
                    }
                    ThreadState::WaitingHandle { handles, .. } => {
                        for waited in handles {
                            let list = handle.entry(*waited).or_default();
                            if !list.contains(&h) {
                                list.push(h);
                            }
                        }
                    }
                    _ => {}
                }
            }
            self.agree("index: mutex lists (FIFO)", sorted_map(&real.waits.mutex), sorted_map(&mutex));
            self.agree("index: condvar lists (FIFO)", sorted_map(&real.waits.condvar), sorted_map(&condvar));
            self.agree("index: arbiter lists (FIFO)", sorted_map(&real.waits.arbiter), sorted_map(&arbiter));
            self.agree("index: handle lists (FIFO)", sorted_map(&real.waits.handle), sorted_map(&handle));
            let as_sets = |m: &HashMap<u32, Vec<u32>>| {
                let mut v: Vec<(u32, Vec<u32>)> = m
                    .iter()
                    .map(|(k, l)| {
                        let mut l = l.clone();
                        l.sort_unstable();
                        (*k, l)
                    })
                    .collect();
                v.sort_unstable();
                v
            };
            self.agree("index: mutex owner sets", as_sets(&real.waits.mutex_owner), as_sets(&owner));
            let true_min = real.threads.values().filter_map(|t| deadline_of(&t.state)).min();
            if let Some(min) = true_min {
                match real.waits.next_deadline {
                    Some(cached) if cached <= min => {}
                    other => self.fail(format!(
                        "index: cached next_deadline {other:?} later than earliest deadline {min:?} (missed wake-up)"
                    )),
                }
            }
            let mut seen = std::collections::HashSet::new();
            for h in &real.ready {
                if !seen.insert(*h) {
                    self.fail(format!("ready queue holds {h:#x} twice"));
                }
            }
        }
    }

    fn sorted_map<K: Ord + Copy + std::hash::Hash>(m: &HashMap<K, Vec<u32>>) -> Vec<(K, Vec<u32>)> {
        let mut v: Vec<(K, Vec<u32>)> = m.iter().map(|(k, l)| (*k, l.clone())).collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }

    fn env_or(name: &str, default: u64) -> u64 {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    fn run(seed: u64, steps: usize) -> HashMap<&'static str, usize> {
        let mut harness = Harness::new(seed);
        harness.check();
        for step in 0..steps {
            harness.step = step;
            harness.op();
            harness.check();
        }
        harness.stats
    }

    /// NEXIUM_THREAD_MODEL_SEEDS / NEXIUM_THREAD_MODEL_STEPS scale the run.
    #[test]
    fn indexed_thread_table_matches_linear_scan_reference() {
        let seeds = env_or("NEXIUM_THREAD_MODEL_SEEDS", 48);
        let steps = env_or("NEXIUM_THREAD_MODEL_STEPS", 2500) as usize;
        let first = env_or("NEXIUM_THREAD_MODEL_FIRST_SEED", 1);
        let mut totals: HashMap<&'static str, usize> = HashMap::new();
        for seed in first..first + seeds {
            for (k, v) in run(seed, steps) {
                *totals.entry(k).or_default() += v;
            }
        }
        // The random walk must actually reach the interesting paths.
        for key in [
            "mutex contended",
            "mutex handoff",
            "condvar to ready",
            "condvar requeue",
            "arbiter wake one",
            "early cancel consumed",
            "terminate",
            "timeout wake",
            "multi-waiter signal",
        ] {
            assert!(totals.get(key).copied().unwrap_or(0) > 0, "never exercised: {key} ({totals:?})");
        }
    }
}
