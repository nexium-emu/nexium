use crossbeam::channel::{
    bounded, Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, TrySendError,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::gpu::vk_dispatch::PreparedDrawBatch;

pub type RenderJob = Box<dyn FnOnce() + Send + 'static>;

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

const MAX_DRAW_GROUPS_PER_SUBMISSION: usize = 256;
const DRAW_GATHER_GRACE: Duration = Duration::from_micros(200);
const DEFAULT_PENDING_DRAW_GROUP_BUDGET: usize = 0;
const DRAW_BACKPRESSURE_LOG_INTERVAL: Duration = Duration::from_secs(3);

#[derive(Default)]
struct DrawWorkBudgetState {
    outstanding: usize,
    closed: bool,
}

struct DrawWorkBudget {
    limit: usize,
    state: Mutex<DrawWorkBudgetState>,
    available: Condvar,
}

impl DrawWorkBudget {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            state: Mutex::new(DrawWorkBudgetState::default()),
            available: Condvar::new(),
        }
    }

    fn enabled(&self) -> bool {
        self.limit != 0
    }

    fn fits(&self, outstanding: usize, incoming: usize) -> bool {
        if incoming > self.limit {
            return outstanding == 0;
        }
        outstanding
            .checked_add(incoming)
            .is_some_and(|total| total <= self.limit)
    }

    fn reserve(&self, incoming: usize, label: &'static str, timeout: Duration) -> bool {
        if !self.enabled() || incoming == 0 {
            return true;
        }
        let started = Instant::now();
        let deadline = started.checked_add(timeout).unwrap_or(started);
        let mut next_report = started + DRAW_BACKPRESSURE_LOG_INTERVAL;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            if state.closed {
                return false;
            }
            if self.fits(state.outstanding, incoming) {
                state.outstanding = state.outstanding.saturating_add(incoming);
                return true;
            }

            let now = Instant::now();
            if now >= deadline {
                log::warn!(
                    "[render-backpressure] phase=budget-timeout label={} chunks={} outstanding={} budget={} waited_ms={:.3}",
                    label,
                    incoming,
                    state.outstanding,
                    self.limit,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                return false;
            }
            let wait_until = std::cmp::min(next_report, deadline);
            let wait = wait_until.saturating_duration_since(now);
            let (next_state, wait_result) = self
                .available
                .wait_timeout(state, wait)
                .unwrap_or_else(|error| error.into_inner());
            state = next_state;
            if wait_result.timed_out() {
                if Instant::now() >= deadline {
                    log::warn!(
                        "[render-backpressure] phase=budget-timeout label={} chunks={} outstanding={} budget={} waited_ms={:.3}",
                        label,
                        incoming,
                        state.outstanding,
                        self.limit,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                    return false;
                }
                log::warn!(
                    "[render-backpressure] phase=budget label={} chunks={} outstanding={} budget={} waited_ms={:.3}",
                    label,
                    incoming,
                    state.outstanding,
                    self.limit,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                next_report = Instant::now() + DRAW_BACKPRESSURE_LOG_INTERVAL;
            }
        }
    }

    fn release(&self, completed: usize) {
        if !self.enabled() || completed == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        match state.outstanding.checked_sub(completed) {
            Some(remaining) => state.outstanding = remaining,
            None => {
                log::error!(
                    "[render-backpressure] release underflow completed={} outstanding={} budget={}",
                    completed,
                    state.outstanding,
                    self.limit,
                );
                state.outstanding = 0;
            }
        }
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
    fn outstanding(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .outstanding
    }
}

struct DrawWorkCompletion<'a> {
    pending: &'a AtomicUsize,
    budget: &'a DrawWorkBudget,
    units: usize,
}

impl Drop for DrawWorkCompletion<'_> {
    fn drop(&mut self) {
        self.pending.fetch_sub(self.units, Ordering::Release);
        self.budget.release(self.units);
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
    group_count: usize,
    ring_bytes: u64,
    same_renderer: bool,
    next_ring_bytes: u64,
) -> Option<DrawGatherEndReason> {
    if !same_renderer {
        return Some(DrawGatherEndReason::Renderer);
    }
    if group_count >= MAX_DRAW_GROUPS_PER_SUBMISSION {
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
    group_count: usize,
    ring_bytes: u64,
    same_renderer: bool,
    next_ring_bytes: u64,
) -> bool {
    draw_group_fit_end_reason(group_count, ring_bytes, same_renderer, next_ring_bytes).is_none()
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
    hard_after: impl Fn() -> bool,
) -> Result<RenderWork, DrawGatherEndReason> {
    if hard_after() {
        return Err(DrawGatherEndReason::Hard);
    }
    if let Some(draw) = queued_draws.pop_front() {
        return Ok(RenderWork::Draw(draw));
    }
    recv_group_candidate(rx, gather_deadline, hard_after)
}

fn seal_draw_tail_locked(draw_tail: &mut Option<Weak<AtomicBool>>) {
    if let Some(flag) = draw_tail.take().and_then(|flag| flag.upgrade()) {
        flag.store(true, Ordering::Release);
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
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
        log::error!("[render-job] job panicked; worker continuing");
    }
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
    let group_count = draws.len();
    let budget_enabled = draw_work_budget.enabled();
    let _completion = budget_enabled.then(|| DrawWorkCompletion {
        pending: worker_pending,
        budget: draw_work_budget,
        units: group_count,
    });
    let profile = render_profile_enabled();
    let started = profile.then(std::time::Instant::now);
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::gpu::vk_dispatch::execute_prepared_draw_batches(draws)
    }))
    .is_err()
    {
        log::error!("[render-job] grouped draw batch panicked; worker continuing");
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
    if !budget_enabled {
        worker_pending.fetch_sub(group_count, Ordering::Release);
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
    let mut lookahead = None;
    let mut queued_draws = VecDeque::new();
    loop {
        let work = match lookahead.take() {
            Some(work) => work,
            None => {
                if let Some(draw) = queued_draws.pop_front() {
                    RenderWork::Draw(draw)
                } else {
                    match rx.recv() {
                        Ok(work) => work,
                        Err(_) => break,
                    }
                }
            }
        };
        match work {
            RenderWork::Job(label, job) => execute_job(label, job, &worker_pending),
            RenderWork::DrawGroup(group) => {
                queued_draws.extend(group);
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
                    let next =
                        match recv_draw_candidate(&rx, &mut queued_draws, gather_deadline, || {
                            draws.last().is_some_and(PreparedDrawBatch::hard_after)
                        }) {
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
                            queued_draws.extend(group);
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
                    if draws.len() == MAX_DRAW_GROUPS_PER_SUBMISSION
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
        let draw_work_budget = Arc::new(DrawWorkBudget::new(render_pending_group_budget()));
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
        let started = profile.then(std::time::Instant::now);
        let mut draw_tail = self.draw_tail.lock().unwrap();
        seal_draw_tail_locked(&mut draw_tail);
        self.pending.fetch_add(1, Ordering::AcqRel);
        let submitted = match self.tx.send_timeout(RenderWork::Job(label, job), timeout) {
            Ok(()) => true,
            Err(SendTimeoutError::Timeout(_)) | Err(SendTimeoutError::Disconnected(_)) => {
                self.pending.fetch_sub(1, Ordering::Release);
                false
            }
        };
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
        submitted
    }

    pub(crate) fn submit_draw_group_timeout_named(
        &self,
        label: &'static str,
        draws: Vec<PreparedDrawBatch>,
        timeout: std::time::Duration,
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
        let draw_count = draws.len();
        let mut draw_tail = self.draw_tail.lock().unwrap();
        self.pending.fetch_add(draw_count, Ordering::AcqRel);
        let submitted = if self.draw_work_budget.enabled() {
            let budget_started = Instant::now();
            if !self.draw_work_budget.reserve(draw_count, label, timeout) {
                self.pending.fetch_sub(draw_count, Ordering::Release);
                if hard_after {
                    seal_draw_tail_locked(&mut draw_tail);
                }
                false
            } else {
                let remaining = timeout.saturating_sub(budget_started.elapsed());
                let sent = match self
                    .tx
                    .send_timeout(RenderWork::DrawGroup(draws), remaining)
                {
                    Ok(()) => true,
                    Err(SendTimeoutError::Timeout(_) | SendTimeoutError::Disconnected(_)) => false,
                };
                if sent {
                    if hard_after {
                        *draw_tail = None;
                    } else {
                        *draw_tail = hard_after_handle;
                    }
                    true
                } else {
                    self.draw_work_budget.release(draw_count);
                    self.pending.fetch_sub(draw_count, Ordering::Release);
                    if hard_after {
                        seal_draw_tail_locked(&mut draw_tail);
                    }
                    false
                }
            }
        } else {
            match self.tx.send_timeout(RenderWork::DrawGroup(draws), timeout) {
                Ok(()) => {
                    if hard_after {
                        *draw_tail = None;
                    } else {
                        *draw_tail = hard_after_handle;
                    }
                    true
                }
                Err(SendTimeoutError::Timeout(_) | SendTimeoutError::Disconnected(_)) => {
                    self.pending.fetch_sub(draw_count, Ordering::Release);
                    if hard_after {
                        seal_draw_tail_locked(&mut draw_tail);
                    }
                    false
                }
            }
        };
        drop(draw_tail);
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} chunks={} blocked_ms={:.3}",
                    label,
                    draw_count,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        submitted
    }

    pub(crate) fn flush_draw_chunk_timeout_named(
        &self,
        label: &'static str,
        draws: Vec<PreparedDrawBatch>,
        timeout: std::time::Duration,
    ) -> bool {
        self.submit_draw_group_timeout_named(label, draws, timeout)
    }

    pub(crate) fn finish(&self, timeout: std::time::Duration) -> bool {
        if self.is_idle() {
            return true;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        if !self.submit_timeout_named(
            "scheduler-finish",
            Box::new(move || {
                let _ = tx.send(());
            }),
            timeout,
        ) {
            return false;
        }
        rx.recv_timeout(timeout).is_ok()
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

fn pending_group_budget_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PENDING_DRAW_GROUP_BUDGET)
}

fn render_pending_group_budget() -> usize {
    pending_group_budget_from_value(
        std::env::var("NEXIUM_RENDER_PENDING_GROUP_BUDGET")
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

pub fn present_thread() -> &'static RenderThread {
    if dedicated_present_thread() {
        static PT: OnceLock<RenderThread> = OnceLock::new();
        return PT.get_or_init(|| RenderThread::new_named("nexium-present"));
    }
    if let Some(rt) = maybe_render_thread() {
        return rt;
    }
    static PT: OnceLock<RenderThread> = OnceLock::new();
    PT.get_or_init(|| RenderThread::new_named("nexium-present"))
}

fn dedicated_present_thread() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_DEDICATED_PRESENT_THREAD")
                .ok()
                .as_deref(),
            Some("0")
                | Some("false")
                | Some("FALSE")
                | Some("off")
                | Some("OFF")
                | Some("no")
                | Some("NO")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        draw_group_fit_end_reason, draw_group_fits, pending_group_budget_from_value,
        recv_group_candidate, rejected_draw_end_reason, retain_received_if_unsealed,
        seal_draw_tail_locked, sealed_candidate_end_reason, DrawGatherEndReason, DrawWorkBudget,
        DrawWorkCompletion, RenderThread, DEFAULT_PENDING_DRAW_GROUP_BUDGET, DRAW_GATHER_GRACE,
        MAX_DRAW_GROUPS_PER_SUBMISSION,
    };
    use crossbeam::channel::bounded;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

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
    fn pending_group_budget_defaults_to_exact_legacy_mode_and_accepts_opt_in_limit() {
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

        let legacy = DrawWorkBudget::new(0);
        assert!(!legacy.enabled());
        assert!(legacy.reserve(usize::MAX, "legacy-test", Duration::from_secs(1)));
        legacy.release(usize::MAX);
        assert_eq!(legacy.outstanding(), 0);
    }

    #[test]
    fn pending_group_budget_blocks_until_completed_work_releases_capacity() {
        let budget = Arc::new(DrawWorkBudget::new(128));
        assert!(budget.reserve(64, "first", Duration::from_secs(1)));
        assert!(budget.reserve(64, "second", Duration::from_secs(1)));
        assert_eq!(budget.outstanding(), 128);

        let waiter_budget = Arc::clone(&budget);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            let reserved = waiter_budget.reserve(1, "waiter", Duration::from_secs(1));
            if reserved {
                waiter_budget.release(1);
            }
            done_tx.send(reserved).unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());

        budget.release(64);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(2)), Ok(true));
        waiter.join().unwrap();
        assert_eq!(budget.outstanding(), 64);
        budget.release(64);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn pending_group_budget_respects_reservation_timeout() {
        let budget = DrawWorkBudget::new(64);
        assert!(budget.reserve(64, "full", Duration::from_secs(1)));

        let started = Instant::now();
        assert!(!budget.reserve(1, "timeout", Duration::from_millis(20)));
        assert!(started.elapsed() >= Duration::from_millis(15));
        assert_eq!(budget.outstanding(), 64);

        budget.release(64);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn pending_group_budget_admits_oversized_packet_only_when_empty() {
        let budget = DrawWorkBudget::new(64);
        assert!(budget.fits(0, 96));
        assert!(!budget.fits(1, 96));
        assert!(!budget.fits(usize::MAX, 1));
        assert!(budget.reserve(96, "oversized", Duration::from_secs(1)));
        assert_eq!(budget.outstanding(), 96);
        budget.release(96);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn closing_pending_group_budget_unblocks_waiter_without_reserving() {
        let budget = Arc::new(DrawWorkBudget::new(64));
        assert!(budget.reserve(64, "full", Duration::from_secs(1)));
        let waiter_budget = Arc::clone(&budget);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            done_tx
                .send(waiter_budget.reserve(1, "closed-waiter", Duration::from_secs(1)))
                .unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        budget.close();
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(2)), Ok(false));
        waiter.join().unwrap();
        budget.release(64);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn draw_work_completion_releases_pending_and_budget_during_unwind() {
        let budget = DrawWorkBudget::new(128);
        assert!(budget.reserve(7, "panic", Duration::from_secs(1)));
        let pending = AtomicUsize::new(7);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _completion = DrawWorkCompletion {
                pending: &pending,
                budget: &budget,
                units: 7,
            };
            panic!("test unwind");
        }));
        assert!(result.is_err());
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(budget.outstanding(), 0);
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
    fn draw_group_limits_require_renderer_identity_count_and_ring_budget() {
        let safe = nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES;
        assert!(draw_group_fits(1, safe - 1, true, 1));
        assert!(!draw_group_fits(1, safe, true, 1));
        assert!(!draw_group_fits(1, 0, false, 1));
        assert!(!draw_group_fits(MAX_DRAW_GROUPS_PER_SUBMISSION, 0, true, 0));
        assert!(!draw_group_fits(1, u64::MAX, true, 1));
    }

    #[test]
    fn draw_group_fit_reason_distinguishes_each_limit() {
        let safe = nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES;
        assert_eq!(
            draw_group_fit_end_reason(1, 0, false, 0),
            Some(DrawGatherEndReason::Renderer)
        );
        assert_eq!(
            draw_group_fit_end_reason(MAX_DRAW_GROUPS_PER_SUBMISSION, 0, true, 0),
            Some(DrawGatherEndReason::Count)
        );
        assert_eq!(
            draw_group_fit_end_reason(1, safe, true, 1),
            Some(DrawGatherEndReason::Ring)
        );
        assert_eq!(
            draw_group_fit_end_reason(1, u64::MAX, true, 1),
            Some(DrawGatherEndReason::Ring)
        );
        assert_eq!(draw_group_fit_end_reason(1, safe - 1, true, 1), None);
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
