use crossbeam::channel::{
    bounded, Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, TrySendError,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError, Weak};
use std::time::{Duration, Instant};

use crate::gpu::vk_dispatch::PreparedDrawBatch;

pub type RenderJob = Box<dyn FnOnce() + Send + 'static>;
pub(crate) type PresentSubmitter = Box<dyn FnOnce(&'static str, RenderJob) + Send + 'static>;

enum RenderWork {
    Job(&'static str, RenderJob),
    Draw(PreparedDrawBatch),
    DrawGroup(Vec<PreparedDrawBatch>),
}

pub struct RenderThread {
    tx: Sender<RenderWork>,
    pending: Arc<AtomicUsize>,
    draw_tail: Mutex<Option<Weak<AtomicBool>>>,
    draw_work_budget: Arc<DrawWorkBudget>,
}

enum PresentWork {
    Direct {
        label: &'static str,
        job: RenderJob,
    },
    Ordered {
        label: &'static str,
        pending: Arc<AtomicUsize>,
        limit: usize,
        job: RenderJob,
        submit: PresentSubmitter,
    },
    Shutdown,
}

pub struct PresentThread {
    tx: Sender<PresentWork>,
    stop: Arc<AtomicBool>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl PresentThread {
    fn new_named(name: &str) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_name = name.to_owned();
        let handle = std::thread::Builder::new()
            .name(worker_name)
            .spawn(move || present_worker(rx, worker_stop))
            .expect("spawn present dispatch thread");
        Self {
            tx,
            stop,
            handle: Mutex::new(Some(handle)),
        }
    }

    pub fn submit_named(&self, label: &'static str, job: RenderJob) -> bool {
        self.tx.send(PresentWork::Direct { label, job }).is_ok()
    }

    pub(crate) fn submit_ordered_named(
        &self,
        pending: Arc<AtomicUsize>,
        limit: usize,
        label: &'static str,
        job: RenderJob,
        submit: PresentSubmitter,
    ) -> bool {
        self.tx
            .send(PresentWork::Ordered {
                label,
                pending,
                limit,
                job,
                submit,
            })
            .is_ok()
    }
}

impl Drop for PresentThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.tx.send(PresentWork::Shutdown);
        if let Some(handle) = self
            .handle
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }
}

fn submit_present_to_renderer(label: &'static str, job: RenderJob) {
    if let Some(render_thread) = maybe_render_thread() {
        render_thread.submit_named(label, job);
    } else {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
            log::error!("[present-job] job panicked; dispatcher continuing");
        }
    }
}

fn present_worker(rx: Receiver<PresentWork>, stop: Arc<AtomicBool>) {
    while let Ok(work) = rx.recv() {
        if stop.load(Ordering::Acquire) {
            break;
        }
        match work {
            PresentWork::Direct { label, job } => submit_present_to_renderer(label, job),
            PresentWork::Ordered {
                label,
                pending,
                limit,
                job,
                submit,
            } => {
                if !crate::reserve_ordered_present_slot_until(&pending, limit, || {
                    stop.load(Ordering::Acquire)
                }) {
                    break;
                }
                let job = crate::guarded_present_job(pending, job);
                submit(label, job);
            }
            PresentWork::Shutdown => break,
        }
    }
}

const DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION: usize = 16;
const MAX_DRAW_GROUPS_PER_SUBMISSION_ENV: &str = "NEXIUM_RENDER_MAX_GROUPS_PER_SUBMISSION";
const DRAW_GATHER_GRACE: Duration = Duration::from_micros(200);
const DEFAULT_PENDING_DRAW_GROUP_BUDGET: usize = 256;
const DEFAULT_PENDING_DRAW_SNAPSHOT_BUDGET_BYTES: usize = 512 * 1024 * 1024;
const DRAW_BACKPRESSURE_LOG_INTERVAL: Duration = Duration::from_secs(3);
const DRAW_QUEUE_TELEMETRY_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrawWorkCost {
    groups: usize,
    snapshot_bytes: usize,
}

impl DrawWorkCost {
    fn for_draws(draws: &[PreparedDrawBatch]) -> Self {
        Self {
            groups: draws.len(),
            snapshot_bytes: draws.iter().fold(0usize, |total, draw| {
                total.saturating_add(draw.snapshot_retained_bytes_upper_bound())
            }),
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        Self {
            groups: self.groups.saturating_add(other.groups),
            snapshot_bytes: self.snapshot_bytes.saturating_add(other.snapshot_bytes),
        }
    }
}

struct DrawWorkBudgetState {
    outstanding: DrawWorkCost,
    peak: DrawWorkCost,
    closed: bool,
    next_telemetry: Instant,
}

struct DrawWorkBudget {
    group_limit: usize,
    snapshot_byte_limit: usize,
    state: Mutex<DrawWorkBudgetState>,
    available: Condvar,
}

impl DrawWorkBudget {
    fn new(group_limit: usize, snapshot_byte_limit: usize) -> Self {
        Self {
            group_limit,
            snapshot_byte_limit,
            state: Mutex::new(DrawWorkBudgetState {
                outstanding: DrawWorkCost::default(),
                peak: DrawWorkCost::default(),
                closed: false,
                next_telemetry: Instant::now() + DRAW_QUEUE_TELEMETRY_INTERVAL,
            }),
            available: Condvar::new(),
        }
    }

    fn enabled(&self) -> bool {
        self.group_limit != 0 || self.snapshot_byte_limit != 0
    }

    fn dimension_fits(outstanding: usize, incoming: usize, limit: usize) -> bool {
        if limit == 0 {
            return true;
        }
        if incoming > limit {
            return outstanding == 0;
        }
        outstanding
            .checked_add(incoming)
            .is_some_and(|total| total <= limit)
    }

    fn fits(&self, outstanding: DrawWorkCost, incoming: DrawWorkCost) -> bool {
        Self::dimension_fits(outstanding.groups, incoming.groups, self.group_limit)
            && Self::dimension_fits(
                outstanding.snapshot_bytes,
                incoming.snapshot_bytes,
                self.snapshot_byte_limit,
            )
    }

    #[cfg(test)]
    fn reserve(&self, incoming: DrawWorkCost, label: &'static str, timeout: Duration) -> bool {
        let started = Instant::now();
        let deadline = Some(started.checked_add(timeout).unwrap_or(started));
        self.reserve_until(incoming, label, started, deadline)
    }

    fn reserve_blocking(&self, incoming: DrawWorkCost, label: &'static str) -> bool {
        self.reserve_until(incoming, label, Instant::now(), None)
    }

    fn reserve_until(
        &self,
        incoming: DrawWorkCost,
        label: &'static str,
        started: Instant,
        deadline: Option<Instant>,
    ) -> bool {
        if !self.enabled() || incoming.groups == 0 {
            return true;
        }
        let mut next_report = started + DRAW_BACKPRESSURE_LOG_INTERVAL;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            if state.closed {
                return false;
            }
            if self.fits(state.outstanding, incoming) {
                state.outstanding = state.outstanding.saturating_add(incoming);
                state.peak.groups = state.peak.groups.max(state.outstanding.groups);
                state.peak.snapshot_bytes = state
                    .peak
                    .snapshot_bytes
                    .max(state.outstanding.snapshot_bytes);
                let now = Instant::now();
                if now >= state.next_telemetry {
                    log::info!(
                        "[render-queue] outstanding_groups={} group_budget={} outstanding_snapshot_mib={:.1} snapshot_budget_mib={:.1} peak_groups={} peak_snapshot_mib={:.1}",
                        state.outstanding.groups,
                        self.group_limit,
                        state.outstanding.snapshot_bytes as f64 / (1024.0 * 1024.0),
                        self.snapshot_byte_limit as f64 / (1024.0 * 1024.0),
                        state.peak.groups,
                        state.peak.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    );
                    state.next_telemetry = now + DRAW_QUEUE_TELEMETRY_INTERVAL;
                }
                return true;
            }

            let now = Instant::now();
            if deadline.is_some_and(|deadline| now >= deadline) {
                log::warn!(
                    "[render-backpressure] phase=budget-timeout label={} incoming_groups={} incoming_snapshot_mib={:.1} outstanding_groups={} group_budget={} outstanding_snapshot_mib={:.1} snapshot_budget_mib={:.1} waited_ms={:.3}",
                    label,
                    incoming.groups,
                    incoming.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    state.outstanding.groups,
                    self.group_limit,
                    state.outstanding.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    self.snapshot_byte_limit as f64 / (1024.0 * 1024.0),
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                return false;
            }
            let wait_until = deadline.map_or(next_report, |deadline| deadline.min(next_report));
            let wait = wait_until.saturating_duration_since(now);
            let (next_state, wait_result) = self
                .available
                .wait_timeout(state, wait)
                .unwrap_or_else(|error| error.into_inner());
            state = next_state;
            if wait_result.timed_out() {
                let now = Instant::now();
                if deadline.is_some_and(|deadline| now >= deadline) {
                    log::warn!(
                        "[render-backpressure] phase=budget-timeout label={} incoming_groups={} incoming_snapshot_mib={:.1} outstanding_groups={} group_budget={} outstanding_snapshot_mib={:.1} snapshot_budget_mib={:.1} waited_ms={:.3}",
                        label,
                        incoming.groups,
                        incoming.snapshot_bytes as f64 / (1024.0 * 1024.0),
                        state.outstanding.groups,
                        self.group_limit,
                        state.outstanding.snapshot_bytes as f64 / (1024.0 * 1024.0),
                        self.snapshot_byte_limit as f64 / (1024.0 * 1024.0),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                    return false;
                }
                log::warn!(
                    "[render-backpressure] phase=budget label={} incoming_groups={} incoming_snapshot_mib={:.1} outstanding_groups={} group_budget={} outstanding_snapshot_mib={:.1} snapshot_budget_mib={:.1} waited_ms={:.3}",
                    label,
                    incoming.groups,
                    incoming.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    state.outstanding.groups,
                    self.group_limit,
                    state.outstanding.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    self.snapshot_byte_limit as f64 / (1024.0 * 1024.0),
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                next_report = now + DRAW_BACKPRESSURE_LOG_INTERVAL;
            }
        }
    }

    fn release(&self, completed: DrawWorkCost) {
        if !self.enabled() || completed.groups == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(groups) = state.outstanding.groups.checked_sub(completed.groups) else {
            log::error!(
                "[render-backpressure] release group underflow completed={} outstanding={} budget={}",
                completed.groups,
                state.outstanding.groups,
                self.group_limit,
            );
            state.outstanding = DrawWorkCost::default();
            drop(state);
            self.available.notify_all();
            return;
        };
        let Some(snapshot_bytes) = state
            .outstanding
            .snapshot_bytes
            .checked_sub(completed.snapshot_bytes)
        else {
            log::error!(
                "[render-backpressure] release snapshot underflow completed={} outstanding={} budget={}",
                completed.snapshot_bytes,
                state.outstanding.snapshot_bytes,
                self.snapshot_byte_limit,
            );
            state.outstanding = DrawWorkCost::default();
            drop(state);
            self.available.notify_all();
            return;
        };
        state.outstanding = DrawWorkCost {
            groups,
            snapshot_bytes,
        };
        drop(state);
        self.available.notify_all();
    }

    fn close(&self) {
        if !self.enabled() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.closed = true;
        drop(state);
        self.available.notify_all();
    }

    #[cfg(test)]
    fn outstanding(&self) -> DrawWorkCost {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .outstanding
    }
}

struct DrawWorkCompletion<'a> {
    pending: &'a AtomicUsize,
    budget: &'a DrawWorkBudget,
    cost: DrawWorkCost,
}

impl Drop for DrawWorkCompletion<'_> {
    fn drop(&mut self) {
        self.pending.fetch_sub(self.cost.groups, Ordering::Release);
        self.budget.release(self.cost);
    }
}

struct DrawWorkerBudgetGuard(Arc<DrawWorkBudget>);

impl Drop for DrawWorkerBudgetGuard {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[inline]
fn render_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_RENDER_PROFILE").is_some())
}

fn draw_gather_grace() -> Duration {
    static GRACE: OnceLock<Duration> = OnceLock::new();
    *GRACE.get_or_init(|| {
        let micros = std::env::var("NEXIUM_RENDER_GATHER_GRACE_US")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|&value| (50..=2_000).contains(&value))
            .unwrap_or(DRAW_GATHER_GRACE.as_micros() as u64);
        Duration::from_micros(micros)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawGatherEndReason {
    Hard,
    Job,
    Compatibility,
    Renderer,
    Count,
    Ring,
    Deadline,
    Disconnected,
}

impl DrawGatherEndReason {
    const COUNT: usize = 8;

    const fn index(self) -> usize {
        self as usize
    }
}

static DRAW_GATHER_END_COUNTS: [AtomicUsize; DrawGatherEndReason::COUNT] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

fn draw_group_fit_end_reason(
    max_group_count: usize,
    group_count: usize,
    ring_bytes: u64,
    same_renderer: bool,
    next_ring_bytes: u64,
) -> Option<DrawGatherEndReason> {
    if !same_renderer {
        return Some(DrawGatherEndReason::Renderer);
    }
    if group_count >= max_group_count {
        return Some(DrawGatherEndReason::Count);
    }
    if !ring_bytes
        .checked_add(next_ring_bytes)
        .is_some_and(|sum| sum <= nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES)
    {
        return Some(DrawGatherEndReason::Ring);
    }
    None
}

#[cfg(test)]
fn draw_group_fits(
    max_group_count: usize,
    group_count: usize,
    ring_bytes: u64,
    same_renderer: bool,
    next_ring_bytes: u64,
) -> bool {
    draw_group_fit_end_reason(
        max_group_count,
        group_count,
        ring_bytes,
        same_renderer,
        next_ring_bytes,
    )
    .is_none()
}

fn sealed_candidate_end_reason(is_job: bool) -> DrawGatherEndReason {
    if is_job {
        DrawGatherEndReason::Job
    } else {
        DrawGatherEndReason::Hard
    }
}

fn rejected_draw_end_reason(fit_end_reason: Option<DrawGatherEndReason>) -> DrawGatherEndReason {
    fit_end_reason.unwrap_or(DrawGatherEndReason::Compatibility)
}

fn recv_group_candidate<T>(
    rx: &Receiver<T>,
    gather_deadline: Instant,
    hard_after: impl Fn() -> bool,
) -> Result<T, DrawGatherEndReason> {
    if hard_after() {
        return Err(DrawGatherEndReason::Hard);
    }
    match rx.try_recv() {
        Ok(received) => Ok(received),
        Err(TryRecvError::Disconnected) => Err(DrawGatherEndReason::Disconnected),
        Err(TryRecvError::Empty) => {
            if hard_after() {
                return Err(DrawGatherEndReason::Hard);
            }
            let remaining = gather_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(DrawGatherEndReason::Deadline);
            }
            match rx.recv_timeout(remaining) {
                Ok(received) => Ok(received),
                Err(RecvTimeoutError::Timeout) => Err(DrawGatherEndReason::Deadline),
                Err(RecvTimeoutError::Disconnected) => Err(DrawGatherEndReason::Disconnected),
            }
        }
    }
}

fn recv_draw_candidate(
    rx: &Receiver<RenderWork>,
    queued_draws: &mut VecDeque<PreparedDrawBatch>,
    gather_deadline: Instant,
    group_count: usize,
    max_group_count: usize,
    hard_after: impl Fn() -> bool,
) -> Result<RenderWork, DrawGatherEndReason> {
    if hard_after() {
        return Err(DrawGatherEndReason::Hard);
    }
    match pop_front_below_group_limit(queued_draws, group_count, max_group_count) {
        Ok(Some(draw)) => return Ok(RenderWork::Draw(draw)),
        Ok(None) => {}
        Err(reason) => return Err(reason),
    }
    recv_group_candidate(rx, gather_deadline, hard_after)
}

fn enqueue_draw_group_fifo<T>(queued: &mut VecDeque<T>, group: Vec<T>) {
    queued.extend(group);
}

fn pop_front_below_group_limit<T>(
    queued: &mut VecDeque<T>,
    group_count: usize,
    max_group_count: usize,
) -> Result<Option<T>, DrawGatherEndReason> {
    if group_count >= max_group_count {
        Err(DrawGatherEndReason::Count)
    } else {
        Ok(queued.pop_front())
    }
}

fn seal_draw_tail_locked(draw_tail: &mut Option<Weak<AtomicBool>>) {
    if let Some(flag) = draw_tail.take().and_then(|flag| flag.upgrade()) {
        flag.store(true, Ordering::Release);
    }
}

fn lock_until_timeout<'a, T>(
    mutex: &'a Mutex<T>,
    started: Instant,
    timeout: Duration,
) -> Option<MutexGuard<'a, T>> {
    loop {
        match mutex.try_lock() {
            Ok(guard) => return Some(guard),
            Err(TryLockError::Poisoned(error)) => return Some(error.into_inner()),
            Err(TryLockError::WouldBlock) => {
                let remaining = timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return None;
                }
                std::thread::sleep(remaining.min(Duration::from_millis(1)));
            }
        }
    }
}

fn retain_received_if_unsealed<T>(received: T, hard_after: impl FnOnce() -> bool) -> Result<T, T> {
    if hard_after() {
        Err(received)
    } else {
        Ok(received)
    }
}

fn execute_job(label: &'static str, job: RenderJob, worker_pending: &AtomicUsize) {
    let profile = render_profile_enabled();
    let started = profile.then(std::time::Instant::now);
    let busy_started = crate::gpu::pusher::kickprof::rate_start();
    crate::gpu::watchdog::render_phase(label, 0);
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
        log::error!("[render-job] job panicked; worker continuing");
    }
    crate::gpu::pusher::kickprof::add_render_job(busy_started, label);
    if let Some(started) = started {
        if label == "clear" {
            static CLEAR_JOBS: AtomicUsize = AtomicUsize::new(0);
            let n = CLEAR_JOBS.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 128 == 0 {
                log::warn!("[render-clear] jobs={}", n);
            }
        }
        let elapsed = started.elapsed();
        if elapsed >= std::time::Duration::from_millis(10) {
            log::warn!(
                "[render-job] worker={} label={} elapsed_ms={:.3}",
                std::thread::current().name().unwrap_or("nexium-render"),
                label,
                elapsed.as_secs_f64() * 1000.0,
            );
        }
    }
    worker_pending.fetch_sub(1, Ordering::Release);
}

fn execute_draw_groups(
    draws: Vec<PreparedDrawBatch>,
    worker_pending: &AtomicUsize,
    draw_work_budget: &DrawWorkBudget,
) {
    let cost = DrawWorkCost::for_draws(&draws);
    let group_count = cost.groups;
    let _completion = DrawWorkCompletion {
        pending: worker_pending,
        budget: draw_work_budget,
        cost,
    };
    let profile = render_profile_enabled();
    let started = profile.then(std::time::Instant::now);
    let busy_started = crate::gpu::pusher::kickprof::rate_start();
    let timeline_started = crate::kick_timeline_enabled().then(std::time::Instant::now);
    crate::gpu::watchdog::render_phase("draw-groups", group_count as u64);
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::gpu::vk_dispatch::execute_prepared_draw_batches(draws)
    }))
    .is_err()
    {
        log::error!("[render-job] grouped draw batch panicked; worker continuing");
    }
    crate::gpu::pusher::kickprof::add_render_busy(busy_started, group_count as u64);
    if let Some(timeline_started) = timeline_started {
        log::warn!(
            "[ktl] us={} rbatch groups={} ms={:.2}",
            crate::timeline_us(),
            group_count,
            timeline_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    if let Some(started) = started {
        let elapsed = started.elapsed();
        if elapsed >= std::time::Duration::from_millis(10) {
            log::warn!(
                "[render-job] worker={} elapsed_ms={:.3} draw_groups={}",
                std::thread::current().name().unwrap_or("nexium-render"),
                elapsed.as_secs_f64() * 1000.0,
                group_count,
            );
        }
    }
}

fn profile_draw_gather(group_count: usize, hard_after: bool, end_reason: DrawGatherEndReason) {
    if !render_profile_enabled() {
        return;
    }
    static SUBMISSIONS: AtomicUsize = AtomicUsize::new(0);
    static GROUPS: AtomicUsize = AtomicUsize::new(0);
    static HARD: AtomicUsize = AtomicUsize::new(0);
    static SINGLES: AtomicUsize = AtomicUsize::new(0);
    GROUPS.fetch_add(group_count, Ordering::Relaxed);
    HARD.fetch_add(usize::from(hard_after), Ordering::Relaxed);
    SINGLES.fetch_add(usize::from(group_count == 1), Ordering::Relaxed);
    DRAW_GATHER_END_COUNTS[end_reason.index()].fetch_add(1, Ordering::Relaxed);
    let submissions = SUBMISSIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if submissions % 64 == 0 {
        let groups = GROUPS.swap(0, Ordering::Relaxed);
        let hard = HARD.swap(0, Ordering::Relaxed);
        let singles = SINGLES.swap(0, Ordering::Relaxed);
        log::warn!(
            "[render-gather-window] submissions=64 groups={} avg_groups={:.2} hard={} singles={}",
            groups,
            groups as f64 / 64.0,
            hard,
            singles,
        );
        let hard_end =
            DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Hard.index()].swap(0, Ordering::Relaxed);
        let job_end =
            DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Job.index()].swap(0, Ordering::Relaxed);
        let compatibility_end = DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Compatibility.index()]
            .swap(0, Ordering::Relaxed);
        let renderer_end = DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Renderer.index()]
            .swap(0, Ordering::Relaxed);
        let count_end =
            DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Count.index()].swap(0, Ordering::Relaxed);
        let ring_end =
            DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Ring.index()].swap(0, Ordering::Relaxed);
        let deadline_end = DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Deadline.index()]
            .swap(0, Ordering::Relaxed);
        let disconnected_end = DRAW_GATHER_END_COUNTS[DrawGatherEndReason::Disconnected.index()]
            .swap(0, Ordering::Relaxed);
        log::warn!(
            "[render-gather-reasons] submissions=64 hard={} job={} compat={} renderer={} count={} ring={} deadline={} disconnected={}",
            hard_end,
            job_end,
            compatibility_end,
            renderer_end,
            count_end,
            ring_end,
            deadline_end,
            disconnected_end,
        );
    }
}

fn render_worker(
    rx: Receiver<RenderWork>,
    worker_pending: Arc<AtomicUsize>,
    draw_work_budget: Arc<DrawWorkBudget>,
) {
    let _budget_guard = DrawWorkerBudgetGuard(Arc::clone(&draw_work_budget));
    crate::gpu::watchdog::register_render_thread();
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
        }
        let _ = SetThreadPriority(GetCurrentThread(), 1);
    }
    let max_groups_per_submission = render_max_groups_per_submission();
    let mut lookahead = None;
    let mut queued_draws = VecDeque::new();
    loop {
        let work = match lookahead.take() {
            Some(work) => work,
            None => {
                if let Some(draw) = queued_draws.pop_front() {
                    RenderWork::Draw(draw)
                } else {
                    crate::gpu::watchdog::render_phase("idle-recv", 0);
                    let idle_started = crate::kick_timeline_enabled().then(Instant::now);
                    match rx.recv() {
                        Ok(work) => {
                            if let Some(idle_started) = idle_started {
                                let waited = idle_started.elapsed();
                                if waited >= Duration::from_micros(300) {
                                    log::warn!(
                                        "[ktl] us={} ridle ms={:.2}",
                                        crate::timeline_us(),
                                        waited.as_secs_f64() * 1000.0
                                    );
                                }
                            }
                            work
                        }
                        Err(_) => break,
                    }
                }
            }
        };
        match work {
            RenderWork::Job(label, job) => execute_job(label, job, &worker_pending),
            RenderWork::DrawGroup(group) => {
                enqueue_draw_group_fifo(&mut queued_draws, group);
            }
            RenderWork::Draw(first) => {
                let profile = render_profile_enabled();
                let gather_started = profile.then(Instant::now);
                let mut compatibility_elapsed = Duration::ZERO;
                let mut compatibility_checks = 0usize;
                let mut compatibility_rejects = 0usize;
                let mut ring_bytes = first.ring_upper_bytes();
                let mut compatibility = first.compatibility().clone();
                let mut draws = vec![first];
                let gather_deadline = Instant::now() + draw_gather_grace();
                let end_reason = loop {
                    if draws.last().is_some_and(PreparedDrawBatch::hard_after) {
                        break DrawGatherEndReason::Hard;
                    }
                    let next = match recv_draw_candidate(
                        &rx,
                        &mut queued_draws,
                        gather_deadline,
                        draws.len(),
                        max_groups_per_submission,
                        || draws.last().is_some_and(PreparedDrawBatch::hard_after),
                    ) {
                        Ok(next) => next,
                        Err(reason) => break reason,
                    };
                    let next = match retain_received_if_unsealed(next, || {
                        draws.last().is_some_and(PreparedDrawBatch::hard_after)
                    }) {
                        Ok(next) => next,
                        Err(next) => {
                            let reason =
                                sealed_candidate_end_reason(matches!(&next, RenderWork::Job(_, _)));
                            lookahead = Some(next);
                            break reason;
                        }
                    };
                    match next {
                        RenderWork::Draw(next_draw) => {
                            let fit_end_reason = draw_group_fit_end_reason(
                                max_groups_per_submission,
                                draws.len(),
                                ring_bytes,
                                draws[0].same_renderer(&next_draw),
                                next_draw.ring_upper_bytes(),
                            );
                            let fits = fit_end_reason.is_none();
                            let compatible = if fits {
                                let started = profile.then(Instant::now);
                                compatibility_checks += 1;
                                let compatible =
                                    compatibility.compatible_with_later(next_draw.compatibility());
                                compatibility_rejects += usize::from(!compatible);
                                if let Some(started) = started {
                                    compatibility_elapsed += started.elapsed();
                                }
                                compatible
                            } else {
                                false
                            };
                            if !compatible {
                                lookahead = Some(RenderWork::Draw(next_draw));
                                break rejected_draw_end_reason(fit_end_reason);
                            }
                            ring_bytes = ring_bytes.saturating_add(next_draw.ring_upper_bytes());
                            compatibility.extend(next_draw.compatibility());
                            draws.push(next_draw);
                        }
                        RenderWork::DrawGroup(group) => {
                            enqueue_draw_group_fifo(&mut queued_draws, group);
                            continue;
                        }
                        next => {
                            lookahead = Some(next);
                            break DrawGatherEndReason::Job;
                        }
                    }
                };
                if let Some(gather_started) = gather_started {
                    let gather_elapsed = gather_started.elapsed();
                    if draws.len() == max_groups_per_submission
                        || compatibility_elapsed >= Duration::from_millis(1)
                    {
                        log::warn!(
                            "[render-gather] groups={} gather_ms={:.3} compat_ms={:.3} checks={} rejects={}",
                            draws.len(),
                            gather_elapsed.as_secs_f64() * 1000.0,
                            compatibility_elapsed.as_secs_f64() * 1000.0,
                            compatibility_checks,
                            compatibility_rejects,
                        );
                    }
                }
                profile_draw_gather(
                    draws.len(),
                    draws.last().is_some_and(PreparedDrawBatch::hard_after),
                    end_reason,
                );
                execute_draw_groups(draws, &worker_pending, &draw_work_budget);
            }
        }
    }
}

impl RenderThread {
    fn new() -> Self {
        Self::new_named("nexium-render")
    }

    fn new_named(name: &str) -> Self {
        let queue_depth = render_queue_depth();
        let (tx, rx) = bounded::<RenderWork>(queue_depth);
        let pending = Arc::new(AtomicUsize::new(0));
        let worker_pending = pending.clone();
        let pending_group_budget = render_pending_group_budget();
        let pending_snapshot_budget = render_pending_snapshot_budget_bytes();
        log::info!(
            "[render-queue] worker={} message_depth={} submission_group_limit={} group_budget={} snapshot_budget_mib={:.1}",
            name,
            queue_depth,
            render_max_groups_per_submission(),
            pending_group_budget,
            pending_snapshot_budget as f64 / (1024.0 * 1024.0),
        );
        let draw_work_budget = Arc::new(DrawWorkBudget::new(
            pending_group_budget,
            pending_snapshot_budget,
        ));
        let worker_draw_work_budget = Arc::clone(&draw_work_budget);
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || render_worker(rx, worker_pending, worker_draw_work_budget))
            .expect("spawn render worker thread");
        RenderThread {
            tx,
            pending,
            draw_tail: Mutex::new(None),
            draw_work_budget,
        }
    }

    pub fn submit(&self, job: RenderJob) {
        self.submit_named("unnamed", job);
    }

    pub fn submit_named(&self, label: &'static str, job: RenderJob) {
        let profile = render_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let mut draw_tail = self.draw_tail.lock().unwrap();
        seal_draw_tail_locked(&mut draw_tail);
        self.pending.fetch_add(1, Ordering::AcqRel);
        if self.tx.send(RenderWork::Job(label, job)).is_err() {
            self.pending.fetch_sub(1, Ordering::Release);
        }
        drop(draw_tail);
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} blocked_ms={:.3}",
                    label,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
    }

    pub fn try_submit(&self, job: RenderJob) -> bool {
        let mut draw_tail = self.draw_tail.lock().unwrap();
        seal_draw_tail_locked(&mut draw_tail);
        self.pending.fetch_add(1, Ordering::AcqRel);
        match self.tx.try_send(RenderWork::Job("unnamed", job)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.pending.fetch_sub(1, Ordering::Release);
                false
            }
        }
    }

    pub fn submit_timeout(&self, job: RenderJob, timeout: std::time::Duration) -> bool {
        self.submit_timeout_named("unnamed", job, timeout)
    }

    pub fn submit_timeout_named(
        &self,
        label: &'static str,
        job: RenderJob,
        timeout: std::time::Duration,
    ) -> bool {
        let profile = render_profile_enabled();
        let started = Instant::now();
        let Some(mut draw_tail) = lock_until_timeout(&self.draw_tail, started, timeout) else {
            if profile && started.elapsed() >= Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} blocked_ms={:.3}",
                    label,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return false;
        };
        seal_draw_tail_locked(&mut draw_tail);
        self.pending.fetch_add(1, Ordering::AcqRel);
        let remaining = timeout.saturating_sub(started.elapsed());
        let submitted = match self.tx.send_timeout(RenderWork::Job(label, job), remaining) {
            Ok(()) => true,
            Err(SendTimeoutError::Timeout(_)) | Err(SendTimeoutError::Disconnected(_)) => {
                self.pending.fetch_sub(1, Ordering::Release);
                false
            }
        };
        if profile {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} blocked_ms={:.3}",
                    label,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        submitted
    }

    pub(crate) fn submit_draw_group_named(
        &self,
        label: &'static str,
        draws: Vec<PreparedDrawBatch>,
    ) -> bool {
        if draws.is_empty() {
            return true;
        }
        let profile = render_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let hard_after = draws.last().is_some_and(PreparedDrawBatch::hard_after);
        let hard_after_handle = draws
            .last()
            .and_then(|draw| (!hard_after).then(|| draw.hard_after_handle()));
        let cost = DrawWorkCost::for_draws(&draws);
        let blocked_started = crate::gpu::pusher::kickprof::rate_start();
        crate::gpu::watchdog::phase(crate::gpu::watchdog::Phase::RenderWait, cost.groups as u64);
        let phase_started = crate::gpu::pusher::kickprof::start();
        let mut draw_tail = self.draw_tail.lock().unwrap();
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::BLK_LOCK, phase_started);
        self.pending.fetch_add(cost.groups, Ordering::AcqRel);
        let phase_started = crate::gpu::pusher::kickprof::start();
        let reserved = self.draw_work_budget.reserve_blocking(cost, label);
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::BLK_RESERVE, phase_started);
        let submitted = if !reserved {
            self.pending.fetch_sub(cost.groups, Ordering::Release);
            if hard_after {
                seal_draw_tail_locked(&mut draw_tail);
            }
            false
        } else {
            let work = RenderWork::DrawGroup(draws);
            let phase_started = crate::gpu::pusher::kickprof::start();
            let sent = self.tx.send(work).is_ok();
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::BLK_SEND, phase_started);
            if sent {
                if hard_after {
                    *draw_tail = None;
                } else {
                    *draw_tail = hard_after_handle;
                }
                true
            } else {
                self.draw_work_budget.release(cost);
                self.pending.fetch_sub(cost.groups, Ordering::Release);
                if hard_after {
                    seal_draw_tail_locked(&mut draw_tail);
                }
                false
            }
        };
        drop(draw_tail);
        crate::gpu::pusher::kickprof::add_blocked(blocked_started);
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} groups={} snapshot_mib={:.1} blocked_ms={:.3}",
                    label,
                    cost.groups,
                    cost.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        submitted
    }

    pub(crate) fn submit_draw_group_then_job_named(
        &self,
        label: &'static str,
        draws: Vec<PreparedDrawBatch>,
        job_label: &'static str,
        job: RenderJob,
    ) -> bool {
        let profile = render_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let hard_after = draws.last().is_some_and(PreparedDrawBatch::hard_after);
        let hard_after_handle = draws
            .last()
            .and_then(|draw| (!hard_after).then(|| draw.hard_after_handle()));
        let cost = DrawWorkCost::for_draws(&draws);
        let blocked_started = crate::gpu::pusher::kickprof::rate_start();
        let phase_started = crate::gpu::pusher::kickprof::start();
        let mut draw_tail = self.draw_tail.lock().unwrap();
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::BLK_LOCK, phase_started);
        self.pending.fetch_add(1, Ordering::AcqRel);

        if cost.groups != 0 {
            self.pending.fetch_add(cost.groups, Ordering::AcqRel);
            let phase_started = crate::gpu::pusher::kickprof::start();
            let reserved = self.draw_work_budget.reserve_blocking(cost, label);
            crate::gpu::pusher::kickprof::add(
                crate::gpu::pusher::kickprof::BLK_RESERVE,
                phase_started,
            );
            if !reserved {
                self.pending.fetch_sub(cost.groups, Ordering::Release);
                self.pending.fetch_sub(1, Ordering::Release);
                if hard_after {
                    seal_draw_tail_locked(&mut draw_tail);
                }
                return false;
            }
            let phase_started = crate::gpu::pusher::kickprof::start();
            let sent = self.tx.send(RenderWork::DrawGroup(draws));
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::BLK_SEND, phase_started);
            if sent.is_err() {
                self.draw_work_budget.release(cost);
                self.pending.fetch_sub(cost.groups, Ordering::Release);
                self.pending.fetch_sub(1, Ordering::Release);
                if hard_after {
                    seal_draw_tail_locked(&mut draw_tail);
                }
                return false;
            }
            if hard_after {
                *draw_tail = None;
            } else {
                *draw_tail = hard_after_handle;
            }
        }

        seal_draw_tail_locked(&mut draw_tail);
        let submitted = if self.tx.send(RenderWork::Job(job_label, job)).is_ok() {
            true
        } else {
            self.pending.fetch_sub(1, Ordering::Release);
            false
        };
        drop(draw_tail);
        crate::gpu::pusher::kickprof::add_blocked(blocked_started);

        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} job_label={} groups={} snapshot_mib={:.1} blocked_ms={:.3}",
                    label,
                    job_label,
                    cost.groups,
                    cost.snapshot_bytes as f64 / (1024.0 * 1024.0),
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        submitted
    }

    pub(crate) fn flush_draw_chunk_named(
        &self,
        label: &'static str,
        draws: Vec<PreparedDrawBatch>,
    ) -> bool {
        self.submit_draw_group_named(label, draws)
    }

    pub(crate) fn finish(&self, timeout: std::time::Duration) -> bool {
        let started = Instant::now();
        if self.is_idle() {
            return true;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let remaining = timeout.saturating_sub(started.elapsed());
        if !self.submit_timeout_named(
            "scheduler-finish",
            Box::new(move || {
                let _ = tx.send(());
            }),
            remaining,
        ) {
            return false;
        }
        rx.recv_timeout(timeout.saturating_sub(started.elapsed()))
            .is_ok()
    }

    pub(crate) fn finish_until(&self, interrupted: impl Fn() -> bool) -> bool {
        if interrupted() {
            return false;
        }
        if self.is_idle() {
            return true;
        }
        let started = Instant::now();
        let poll_interval = Duration::from_millis(10);
        let report_interval = Duration::from_secs(3);
        let mut next_report = report_interval;
        let mut report_wait = |stage: &str| {
            let elapsed = started.elapsed();
            if elapsed >= next_report {
                log::warn!(
                    "[render-sync] finish {} pending={} waited_ms={:.1}",
                    stage,
                    self.pending.load(Ordering::Acquire),
                    elapsed.as_secs_f64() * 1000.0,
                );
                next_report = elapsed.saturating_add(report_interval);
            }
        };
        let mut draw_tail = loop {
            if interrupted() {
                return false;
            }
            if let Some(guard) = lock_until_timeout(&self.draw_tail, Instant::now(), poll_interval)
            {
                break guard;
            }
            report_wait("draw-tail lock");
        };
        if interrupted() {
            return false;
        }
        seal_draw_tail_locked(&mut draw_tail);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut work = RenderWork::Job(
            "scheduler-finish",
            Box::new(move || {
                let _ = tx.send(());
            }),
        );
        self.pending.fetch_add(1, Ordering::AcqRel);
        loop {
            if interrupted() {
                self.pending.fetch_sub(1, Ordering::Release);
                return false;
            }
            match self.tx.send_timeout(work, poll_interval) {
                Ok(()) => break,
                Err(SendTimeoutError::Timeout(returned)) => {
                    work = returned;
                    report_wait("enqueue");
                }
                Err(SendTimeoutError::Disconnected(_)) => {
                    self.pending.fetch_sub(1, Ordering::Release);
                    log::error!("[render-sync] finish enqueue disconnected");
                    return false;
                }
            }
        }
        drop(draw_tail);
        loop {
            if interrupted() {
                return false;
            }
            match rx.recv_timeout(poll_interval) {
                Ok(()) => return true,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    report_wait("acknowledgement");
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    log::error!("[render-sync] finish acknowledgement disconnected");
                    return false;
                }
            }
        }
    }

    pub(crate) fn seal_draw_tail(&self) {
        let mut draw_tail = self.draw_tail.lock().unwrap();
        seal_draw_tail_locked(&mut draw_tail);
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.pending.load(Ordering::Acquire) == 0
    }
}

fn render_queue_depth() -> usize {
    std::env::var("NEXIUM_RENDER_QUEUE_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|depth| *depth > 0)
        .unwrap_or(256)
}

fn max_groups_per_submission_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|groups| groups.max(1))
        .unwrap_or(DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION)
}

fn render_max_groups_per_submission() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        max_groups_per_submission_from_value(
            std::env::var(MAX_DRAW_GROUPS_PER_SUBMISSION_ENV)
                .ok()
                .as_deref(),
        )
    })
}

fn pending_group_budget_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PENDING_DRAW_GROUP_BUDGET)
}

fn pending_snapshot_budget_bytes_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(|mib| mib.checked_mul(1024 * 1024))
        .unwrap_or(DEFAULT_PENDING_DRAW_SNAPSHOT_BUDGET_BYTES)
}

fn render_pending_group_budget() -> usize {
    pending_group_budget_from_value(
        std::env::var("NEXIUM_RENDER_PENDING_GROUP_BUDGET")
            .ok()
            .as_deref(),
    )
}

fn render_pending_snapshot_budget_bytes() -> usize {
    pending_snapshot_budget_bytes_from_value(
        std::env::var("NEXIUM_RENDER_PENDING_SNAPSHOT_MIB")
            .ok()
            .as_deref(),
    )
}

fn async_render_enabled() -> bool {
    match std::env::var("NEXIUM_ASYNC_RENDER").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") | Some("no")
        | Some("NO") => false,
        _ => true,
    }
}

pub fn maybe_render_thread() -> Option<&'static RenderThread> {
    static RT: OnceLock<Option<RenderThread>> = OnceLock::new();
    RT.get_or_init(|| {
        if async_render_enabled() {
            log::info!(
                "nexium-nvdrv: async render thread ENABLED (set NEXIUM_ASYNC_RENDER=0 to disable)"
            );
            Some(RenderThread::new())
        } else {
            log::info!("nexium-nvdrv: async render thread DISABLED (NEXIUM_ASYNC_RENDER=0)");
            None
        }
    })
    .as_ref()
}

pub fn present_thread() -> &'static PresentThread {
    static PT: OnceLock<PresentThread> = OnceLock::new();
    PT.get_or_init(|| PresentThread::new_named("nexium-present-dispatch"))
}

#[cfg(test)]
mod tests {
    use super::{
        draw_group_fit_end_reason, draw_group_fits, enqueue_draw_group_fifo,
        max_groups_per_submission_from_value, pending_group_budget_from_value,
        pending_snapshot_budget_bytes_from_value, pop_front_below_group_limit,
        recv_group_candidate, rejected_draw_end_reason, retain_received_if_unsealed,
        seal_draw_tail_locked, sealed_candidate_end_reason, DrawGatherEndReason, DrawWorkBudget,
        DrawWorkCompletion, DrawWorkCost, PresentThread, RenderThread, RenderWork,
        DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION, DEFAULT_PENDING_DRAW_GROUP_BUDGET,
        DEFAULT_PENDING_DRAW_SNAPSHOT_BUDGET_BYTES, DRAW_GATHER_GRACE,
        MAX_DRAW_GROUPS_PER_SUBMISSION_ENV,
    };
    use crossbeam::channel::bounded;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    fn cost(groups: usize, snapshot_bytes: usize) -> DrawWorkCost {
        DrawWorkCost {
            groups,
            snapshot_bytes,
        }
    }

    fn wait_until_idle(worker: &RenderThread) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.pending.load(std::sync::atomic::Ordering::Acquire) != 0
            && Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        assert!(
            worker.pending.load(std::sync::atomic::Ordering::Acquire) == 0,
            "render worker did not become idle"
        );
    }

    #[test]
    fn ordered_present_enqueue_returns_without_waiting_for_an_inflight_slot() {
        let dispatcher = PresentThread::new_named("nexium-present-nonblocking-test");
        let pending = Arc::new(AtomicUsize::new(1));
        let (ran_tx, ran_rx) = mpsc::channel();

        let started = Instant::now();
        assert!(dispatcher.submit_ordered_named(
            Arc::clone(&pending),
            1,
            "ordered-present-nonblocking-test",
            Box::new(move || ran_tx.send(()).unwrap()),
            Box::new(|_label, job| job()),
        ));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "guest-side QueueBuffer submission waited for an in-flight present slot"
        );
        assert!(ran_rx.recv_timeout(Duration::from_millis(20)).is_err());

        pending.fetch_sub(1, Ordering::Release);
        ran_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn pending_draw_budgets_are_finite_by_default_and_accept_explicit_overrides() {
        assert_eq!(
            pending_group_budget_from_value(None),
            DEFAULT_PENDING_DRAW_GROUP_BUDGET
        );
        assert_eq!(pending_group_budget_from_value(Some("128")), 128);
        assert_eq!(pending_group_budget_from_value(Some("0")), 0);
        assert_eq!(
            pending_group_budget_from_value(Some("invalid")),
            DEFAULT_PENDING_DRAW_GROUP_BUDGET
        );
        assert_eq!(
            pending_snapshot_budget_bytes_from_value(None),
            DEFAULT_PENDING_DRAW_SNAPSHOT_BUDGET_BYTES
        );
        assert_eq!(
            pending_snapshot_budget_bytes_from_value(Some("64")),
            64 * 1024 * 1024
        );
        assert_eq!(pending_snapshot_budget_bytes_from_value(Some("0")), 0);
        assert_eq!(
            pending_snapshot_budget_bytes_from_value(Some("invalid")),
            DEFAULT_PENDING_DRAW_SNAPSHOT_BUDGET_BYTES
        );

        let unlimited = DrawWorkBudget::new(0, 0);
        assert!(!unlimited.enabled());
        assert!(unlimited.reserve(
            cost(usize::MAX, usize::MAX),
            "unlimited-test",
            Duration::from_secs(1)
        ));
        unlimited.release(cost(usize::MAX, usize::MAX));
        assert_eq!(unlimited.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn renderer_submission_group_limit_defaults_to_sixteen_and_parses_env_values() {
        assert_eq!(
            MAX_DRAW_GROUPS_PER_SUBMISSION_ENV,
            "NEXIUM_RENDER_MAX_GROUPS_PER_SUBMISSION"
        );
        assert_eq!(
            max_groups_per_submission_from_value(None),
            DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION
        );
        assert_eq!(max_groups_per_submission_from_value(Some("1")), 1);
        assert_eq!(max_groups_per_submission_from_value(Some(" 32 ")), 32);
        assert_eq!(max_groups_per_submission_from_value(Some("0")), 1);
        assert_eq!(
            max_groups_per_submission_from_value(Some("invalid")),
            DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION
        );
        assert_eq!(
            max_groups_per_submission_from_value(Some("")),
            DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION
        );
    }

    #[test]
    fn oversized_draw_group_splits_fifo_and_releases_accounting_per_chunk() {
        const LIMIT: usize = 3;
        let mut queued = VecDeque::new();
        enqueue_draw_group_fifo(&mut queued, (0usize..8).collect());

        let budget = DrawWorkBudget::new(16, 1024);
        let total = cost(8, 80);
        assert!(budget.reserve(total, "split", Duration::from_secs(1)));
        let pending = AtomicUsize::new(total.groups);
        let mut chunks = Vec::new();

        while !queued.is_empty() {
            let mut chunk = Vec::new();
            loop {
                match pop_front_below_group_limit(&mut queued, chunk.len(), LIMIT) {
                    Ok(Some(draw)) => chunk.push(draw),
                    Ok(None) | Err(DrawGatherEndReason::Count) => break,
                    Err(reason) => panic!("unexpected split reason: {reason:?}"),
                }
            }
            assert!(!chunk.is_empty());
            let chunk_cost = cost(chunk.len(), chunk.len() * 10);
            {
                let _completion = DrawWorkCompletion {
                    pending: &pending,
                    budget: &budget,
                    cost: chunk_cost,
                };
            }
            chunks.push(chunk);
        }

        assert_eq!(chunks, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]);
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn pending_draw_budget_blocks_until_completed_work_releases_capacity() {
        let budget = Arc::new(DrawWorkBudget::new(128, 1024));
        assert!(budget.reserve(cost(64, 512), "first", Duration::from_secs(1)));
        assert!(budget.reserve(cost(64, 512), "second", Duration::from_secs(1)));
        assert_eq!(budget.outstanding(), cost(128, 1024));

        let waiter_budget = Arc::clone(&budget);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            let reserved = waiter_budget.reserve_blocking(cost(1, 1), "blocking-no-drop-waiter");
            if reserved {
                waiter_budget.release(cost(1, 1));
            }
            done_tx.send(reserved).unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());

        budget.release(cost(64, 512));
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(2)), Ok(true));
        waiter.join().unwrap();
        assert_eq!(budget.outstanding(), cost(64, 512));
        budget.release(cost(64, 512));
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn pending_draw_budget_respects_reservation_timeout() {
        let budget = DrawWorkBudget::new(64, 64);
        assert!(budget.reserve(cost(64, 64), "full", Duration::from_secs(1)));

        let started = Instant::now();
        assert!(!budget.reserve(cost(1, 1), "timeout", Duration::from_millis(20)));
        assert!(started.elapsed() >= Duration::from_millis(15));
        assert_eq!(budget.outstanding(), cost(64, 64));

        budget.release(cost(64, 64));
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn pending_draw_budget_admits_oversized_packet_only_when_empty() {
        let budget = DrawWorkBudget::new(64, 64);
        assert!(budget.fits(DrawWorkCost::default(), cost(96, 96)));
        assert!(!budget.fits(cost(1, 0), cost(96, 96)));
        assert!(!budget.fits(cost(0, 1), cost(96, 96)));
        assert!(!budget.fits(cost(usize::MAX, 0), cost(1, 0)));
        assert!(budget.reserve(cost(96, 96), "oversized", Duration::from_secs(1)));
        assert_eq!(budget.outstanding(), cost(96, 96));
        budget.release(cost(96, 96));
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn closing_pending_draw_budget_unblocks_blocking_waiter_without_reserving() {
        let budget = Arc::new(DrawWorkBudget::new(64, 64));
        assert!(budget.reserve(cost(64, 64), "full", Duration::from_secs(1)));
        let waiter_budget = Arc::clone(&budget);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            done_tx
                .send(waiter_budget.reserve_blocking(cost(1, 1), "closed-waiter"))
                .unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        budget.close();
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(2)), Ok(false));
        waiter.join().unwrap();
        budget.release(cost(64, 64));
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn draw_work_completion_releases_pending_and_budget_during_unwind() {
        let budget = DrawWorkBudget::new(128, 4096);
        let work = cost(7, 2048);
        assert!(budget.reserve(work, "panic", Duration::from_secs(1)));
        let pending = AtomicUsize::new(7);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _completion = DrawWorkCompletion {
                pending: &pending,
                budget: &budget,
                cost: work,
            };
            panic!("test unwind");
        }));
        assert!(result.is_err());
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(budget.outstanding(), DrawWorkCost::default());
    }

    #[test]
    fn pending_jobs_tracks_queued_in_flight_and_idle_transitions() {
        let worker = RenderThread::new_named("nexium-render-pending-test");
        assert!(worker.is_idle());
        assert_eq!(worker.pending.load(std::sync::atomic::Ordering::Acquire), 0);

        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        assert!(worker.try_submit(Box::new(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })));
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(!worker.is_idle());
        assert_ne!(worker.pending.load(std::sync::atomic::Ordering::Acquire), 0);

        assert!(worker.try_submit(Box::new(|| {})));
        assert_eq!(worker.pending.load(std::sync::atomic::Ordering::Acquire), 2);

        release_tx.send(()).unwrap();
        wait_until_idle(&worker);
        assert!(worker.is_idle());
    }

    #[test]
    fn draw_group_then_job_with_empty_group_seals_tail_and_tracks_job() {
        let (tx, rx) = bounded(1);
        let tail = Arc::new(AtomicBool::new(false));
        let worker = RenderThread {
            tx,
            pending: Arc::new(AtomicUsize::new(0)),
            draw_tail: Mutex::new(Some(Arc::downgrade(&tail))),
            draw_work_budget: Arc::new(DrawWorkBudget::new(0, 0)),
        };
        let (ran_tx, ran_rx) = mpsc::channel();

        assert!(worker.submit_draw_group_then_job_named(
            "empty-draw-group",
            Vec::new(),
            "paired-job",
            Box::new(move || ran_tx.send(()).unwrap()),
        ));
        assert!(tail.load(Ordering::Acquire));
        assert!(worker.draw_tail.lock().unwrap().is_none());
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);

        match rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            RenderWork::Job(label, job) => {
                assert_eq!(label, "paired-job");
                super::execute_job(label, job, worker.pending.as_ref());
            }
            _ => panic!("unexpected render work"),
        }
        ran_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(worker.is_idle());
    }

    #[test]
    fn draw_group_then_job_failure_restores_job_pending_count() {
        let (tx, rx) = bounded(1);
        drop(rx);
        let tail = Arc::new(AtomicBool::new(false));
        let worker = RenderThread {
            tx,
            pending: Arc::new(AtomicUsize::new(0)),
            draw_tail: Mutex::new(Some(Arc::downgrade(&tail))),
            draw_work_budget: Arc::new(DrawWorkBudget::new(0, 0)),
        };

        assert!(!worker.submit_draw_group_then_job_named(
            "empty-draw-group",
            Vec::new(),
            "disconnected-job",
            Box::new(|| {}),
        ));
        assert!(tail.load(Ordering::Acquire));
        assert!(worker.draw_tail.lock().unwrap().is_none());
        assert_eq!(worker.pending.load(Ordering::Acquire), 0);
    }

    #[test]
    fn timed_job_submission_bounds_draw_tail_lock_wait() {
        let (tx, _rx) = bounded(1);
        let worker = Arc::new(RenderThread {
            tx,
            pending: Arc::new(AtomicUsize::new(0)),
            draw_tail: Mutex::new(None),
            draw_work_budget: Arc::new(DrawWorkBudget::new(0, 0)),
        });
        let held_tail = worker.draw_tail.lock().unwrap();
        let submitted_worker = Arc::clone(&worker);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let timeout = Duration::from_millis(30);
        let submitter = std::thread::spawn(move || {
            let started = Instant::now();
            started_tx.send(()).unwrap();
            let submitted = submitted_worker.submit_timeout_named(
                "held-draw-tail-test",
                Box::new(|| {}),
                timeout,
            );
            done_tx.send((submitted, started.elapsed())).unwrap();
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let (submitted, elapsed) = done_rx
            .recv_timeout(Duration::from_millis(200))
            .expect("timed submission blocked indefinitely on draw_tail");
        assert!(!submitted);
        assert!(elapsed >= Duration::from_millis(20));
        assert!(elapsed < Duration::from_millis(200));
        assert_eq!(worker.pending.load(Ordering::Acquire), 0);

        drop(held_tail);
        submitter.join().unwrap();
    }

    #[test]
    fn timed_job_send_uses_budget_remaining_after_draw_tail_wait() {
        let (tx, _rx) = bounded(1);
        tx.send(RenderWork::Job("occupied", Box::new(|| {})))
            .unwrap();
        let worker = Arc::new(RenderThread {
            tx,
            pending: Arc::new(AtomicUsize::new(1)),
            draw_tail: Mutex::new(None),
            draw_work_budget: Arc::new(DrawWorkBudget::new(0, 0)),
        });
        let held_tail = worker.draw_tail.lock().unwrap();
        let submitted_worker = Arc::clone(&worker);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let timeout = Duration::from_millis(400);
        let submitter = std::thread::spawn(move || {
            let started = Instant::now();
            started_tx.send(()).unwrap();
            let submitted = submitted_worker.submit_timeout_named(
                "remaining-send-budget-test",
                Box::new(|| {}),
                timeout,
            );
            done_tx.send((submitted, started.elapsed())).unwrap();
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        drop(held_tail);
        let (submitted, elapsed) = done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(!submitted);
        assert!(elapsed >= Duration::from_millis(350));
        assert!(
            elapsed < Duration::from_millis(500),
            "send_timeout restarted the full timeout after lock acquisition: {elapsed:?}"
        );
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        submitter.join().unwrap();
    }

    fn finish_test_worker() -> (
        Arc<RenderThread>,
        crossbeam::channel::Receiver<RenderWork>,
        Arc<AtomicBool>,
    ) {
        let (tx, rx) = bounded(1);
        tx.send(RenderWork::Job("predecessor", Box::new(|| {})))
            .unwrap();
        let tail = Arc::new(AtomicBool::new(false));
        let worker = Arc::new(RenderThread {
            tx,
            pending: Arc::new(AtomicUsize::new(1)),
            draw_tail: Mutex::new(Some(Arc::downgrade(&tail))),
            draw_work_budget: Arc::new(DrawWorkBudget::new(0, 0)),
        });
        (worker, rx, tail)
    }

    fn run_finish_test_job(worker: &RenderThread, work: RenderWork) {
        let RenderWork::Job(label, job) = work else {
            panic!("unexpected render work");
        };
        super::execute_job(label, job, worker.pending.as_ref());
    }

    fn wait_for_finish_reservation(worker: &RenderThread) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.pending.load(Ordering::Acquire) != 2 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(worker.pending.load(Ordering::Acquire), 2);
    }

    #[test]
    fn cancellable_finish_retries_backpressure_and_waits_for_one_fifo_marker() {
        let (worker, rx, tail) = finish_test_worker();
        let waiter_worker = Arc::clone(&worker);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            done_tx.send(waiter_worker.finish_until(|| false)).unwrap();
        });
        wait_for_finish_reservation(&worker);
        assert!(done_rx.recv_timeout(Duration::from_millis(40)).is_err());
        assert!(tail.load(Ordering::Acquire));
        assert_eq!(worker.pending.load(Ordering::Acquire), 2);

        run_finish_test_job(&worker, rx.recv_timeout(Duration::from_secs(1)).unwrap());
        let marker = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(40)).is_err());
        assert!(worker.draw_tail.try_lock().unwrap().is_none());
        assert!(rx.is_empty());
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        run_finish_test_job(&worker, marker);

        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)), Ok(true));
        waiter.join().unwrap();
        assert!(worker.is_idle());
        assert!(rx.is_empty());
    }

    #[test]
    fn cancellable_finish_stops_while_draw_tail_is_locked() {
        let (worker, rx, tail) = finish_test_worker();
        let held_tail = worker.draw_tail.lock().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waiter_cancelled = Arc::clone(&cancelled);
        let waiter_worker = Arc::clone(&worker);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            done_tx
                .send(waiter_worker.finish_until(|| waiter_cancelled.load(Ordering::Acquire)))
                .unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(40)).is_err());
        cancelled.store(true, Ordering::Release);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)), Ok(false));
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        assert!(!tail.load(Ordering::Acquire));
        drop(held_tail);
        waiter.join().unwrap();
        run_finish_test_job(&worker, rx.recv_timeout(Duration::from_secs(1)).unwrap());
        assert!(worker.is_idle());
        assert!(rx.is_empty());
    }

    #[test]
    fn cancellable_finish_releases_unaccepted_marker_on_cancel() {
        let (worker, rx, tail) = finish_test_worker();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waiter_cancelled = Arc::clone(&cancelled);
        let waiter_worker = Arc::clone(&worker);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            done_tx
                .send(waiter_worker.finish_until(|| waiter_cancelled.load(Ordering::Acquire)))
                .unwrap();
        });
        wait_for_finish_reservation(&worker);
        cancelled.store(true, Ordering::Release);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)), Ok(false));
        waiter.join().unwrap();
        assert!(tail.load(Ordering::Acquire));
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        run_finish_test_job(&worker, rx.recv_timeout(Duration::from_secs(1)).unwrap());
        assert!(worker.is_idle());
        assert!(rx.is_empty());
    }

    #[test]
    fn cancellable_finish_preserves_accepted_marker_on_cancel() {
        let (worker, rx, _) = finish_test_worker();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waiter_cancelled = Arc::clone(&cancelled);
        let waiter_worker = Arc::clone(&worker);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            done_tx
                .send(waiter_worker.finish_until(|| waiter_cancelled.load(Ordering::Acquire)))
                .unwrap();
        });
        wait_for_finish_reservation(&worker);
        run_finish_test_job(&worker, rx.recv_timeout(Duration::from_secs(1)).unwrap());
        let marker = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        cancelled.store(true, Ordering::Release);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)), Ok(false));
        waiter.join().unwrap();
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        run_finish_test_job(&worker, marker);
        assert!(worker.is_idle());
        assert!(rx.is_empty());
    }

    #[test]
    fn cancellable_finish_reports_enqueue_disconnect_and_restores_accounting() {
        let (worker, rx, tail) = finish_test_worker();
        let predecessor = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(rx);
        assert!(!worker.finish_until(|| false));
        assert!(tail.load(Ordering::Acquire));
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        run_finish_test_job(&worker, predecessor);
        assert!(worker.is_idle());
    }

    #[test]
    fn cancellable_finish_reports_dropped_acknowledgement() {
        let (worker, rx, _) = finish_test_worker();
        let waiter_worker = Arc::clone(&worker);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            done_tx.send(waiter_worker.finish_until(|| false)).unwrap();
        });
        wait_for_finish_reservation(&worker);
        run_finish_test_job(&worker, rx.recv_timeout(Duration::from_secs(1)).unwrap());
        let marker = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(marker);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(1)), Ok(false));
        waiter.join().unwrap();
        assert_eq!(worker.pending.load(Ordering::Acquire), 1);
        worker.pending.fetch_sub(1, Ordering::Release);
        assert!(worker.is_idle());
        assert!(rx.is_empty());
    }

    #[test]
    fn draw_group_limits_require_renderer_identity_count_and_ring_budget() {
        let safe = nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES;
        let limit = DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION;
        assert!(draw_group_fits(limit, 1, safe - 1, true, 1));
        assert!(!draw_group_fits(limit, 1, safe, true, 1));
        assert!(!draw_group_fits(limit, 1, 0, false, 1));
        assert!(!draw_group_fits(limit, limit, 0, true, 0));
        assert!(!draw_group_fits(limit, 1, u64::MAX, true, 1));
    }

    #[test]
    fn draw_group_fit_reason_distinguishes_each_limit() {
        let safe = nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES;
        let limit = DEFAULT_MAX_DRAW_GROUPS_PER_SUBMISSION;
        assert_eq!(
            draw_group_fit_end_reason(limit, 1, 0, false, 0),
            Some(DrawGatherEndReason::Renderer)
        );
        assert_eq!(
            draw_group_fit_end_reason(limit, limit, 0, true, 0),
            Some(DrawGatherEndReason::Count)
        );
        assert_eq!(
            draw_group_fit_end_reason(limit, 1, safe, true, 1),
            Some(DrawGatherEndReason::Ring)
        );
        assert_eq!(
            draw_group_fit_end_reason(limit, 1, u64::MAX, true, 1),
            Some(DrawGatherEndReason::Ring)
        );
        assert_eq!(draw_group_fit_end_reason(limit, 1, safe - 1, true, 1), None);
    }

    #[test]
    fn candidate_reason_distinguishes_sealing_job_and_compatibility() {
        assert_eq!(sealed_candidate_end_reason(true), DrawGatherEndReason::Job);
        assert_eq!(
            sealed_candidate_end_reason(false),
            DrawGatherEndReason::Hard
        );
        assert_eq!(
            rejected_draw_end_reason(None),
            DrawGatherEndReason::Compatibility
        );
        assert_eq!(
            rejected_draw_end_reason(Some(DrawGatherEndReason::Ring)),
            DrawGatherEndReason::Ring
        );
    }

    #[test]
    fn soft_draw_gather_accepts_work_arriving_before_deadline() {
        let (tx, rx) = bounded(1);
        let sender = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            tx.send(7u32).unwrap();
        });
        let received =
            recv_group_candidate(&rx, Instant::now() + Duration::from_millis(100), || false);
        assert_eq!(received, Ok(7));
        sender.join().unwrap();
    }

    #[test]
    fn hard_draw_tail_skips_gather_without_consuming_work() {
        let (tx, rx) = bounded(1);
        tx.send(7u32).unwrap();
        assert_eq!(
            recv_group_candidate(&rx, Instant::now() + Duration::from_secs(1), || true),
            Err(DrawGatherEndReason::Hard)
        );
        assert_eq!(rx.try_recv(), Ok(7));
        assert!(DRAW_GATHER_GRACE >= Duration::from_micros(100));
        assert!(DRAW_GATHER_GRACE <= Duration::from_micros(250));
    }

    #[test]
    fn soft_draw_gather_preserves_disconnected_state() {
        let (tx, rx) = bounded::<u32>(1);
        drop(tx);
        assert_eq!(
            recv_group_candidate(&rx, Instant::now() + Duration::from_secs(1), || false),
            Err(DrawGatherEndReason::Disconnected)
        );
    }

    #[test]
    fn soft_draw_gather_reports_expired_deadline_without_resampling_hard_state() {
        let (_tx, rx) = bounded::<u32>(1);
        let checks = AtomicUsize::new(0);
        let result = recv_group_candidate(&rx, Instant::now(), || {
            checks.fetch_add(1, Ordering::AcqRel) >= 2
        });
        assert_eq!(result, Err(DrawGatherEndReason::Deadline));
        assert_eq!(checks.load(Ordering::Acquire), 2);
    }

    #[test]
    fn sealing_a_soft_tail_marks_it_hard_and_clears_tracking() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut tail = Some(Arc::downgrade(&flag));
        seal_draw_tail_locked(&mut tail);
        assert!(flag.load(Ordering::Acquire));
        assert!(tail.is_none());
    }

    #[test]
    fn sealing_between_initial_check_and_receive_recheck_rejects_next() {
        let flag = Arc::new(AtomicBool::new(false));
        let worker_flag = flag.clone();
        let (initial_tx, initial_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            assert!(!worker_flag.load(Ordering::Acquire));
            initial_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            retain_received_if_unsealed(7u32, || worker_flag.load(Ordering::Acquire))
        });
        initial_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let mut tail = Some(Arc::downgrade(&flag));
        seal_draw_tail_locked(&mut tail);
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), Err(7));
    }
}
