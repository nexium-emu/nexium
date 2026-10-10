use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct GraphicBuffer {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
    pub usage: u32,
    pub kind: u32,
    pub nvmap_id: u32,
    pub buffer_offset: u64,
    pub size: u32,
    pub block_height_log2: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SlotState {
    #[default]
    Free,
    Dequeued,
    Queued,
    Acquired,
}

#[derive(Clone, Debug, Default)]
pub struct Slot {
    pub buffer: Option<GraphicBuffer>,
    pub state: SlotState,
    pub queued: bool,
    pub last_swap_interval: u32,
    pub frame_number: u64,
}

pub const BUFFER_HISTORY_LEN: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferHistoryEntry {
    pub frame_number: u64,
    pub queued_at: std::time::Instant,
    pub presented_at: Option<std::time::Instant>,
    pub state: SlotState,
}

#[derive(Clone, Debug)]
pub struct QueuedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub present_at: Option<std::time::Instant>,
    pub depth: Option<nexium_gpu::PresentDepth>,
}

pub struct BufferQueue {
    pub binder_id: u32,
    pub slots: Vec<Slot>,
    pub free: VecDeque<u32>,
    pub dequeued: Vec<u32>,
    pub queued: VecDeque<u32>,
    pub acquired: Vec<u32>,
    pub width: u32,
    pub height: u32,
    pub last_queued: Option<u32>,
    pub connected_api: i32,
    next_swap_deadline: Option<std::time::Instant>,
    availability_generation: Arc<AtomicU64>,
    frame_counter: u64,
    history: VecDeque<BufferHistoryEntry>,
    pending_wake: bool,
}

impl BufferQueue {
    pub fn new(binder_id: u32) -> Self {
        Self::with_generation(binder_id, Arc::new(AtomicU64::new(0)))
    }

    pub fn with_generation(binder_id: u32, availability_generation: Arc<AtomicU64>) -> Self {
        Self {
            binder_id,
            slots: vec![Slot::default(); 8],
            free: VecDeque::new(),
            dequeued: Vec::new(),
            queued: VecDeque::new(),
            acquired: Vec::new(),
            width: 1280,
            height: 720,
            last_queued: None,
            connected_api: 0,
            next_swap_deadline: None,
            availability_generation,
            frame_counter: 0,
            history: VecDeque::with_capacity(BUFFER_HISTORY_LEN),
            pending_wake: false,
        }
    }

    pub fn schedule_swap(
        &mut self,
        now: std::time::Instant,
        swap_interval: i32,
    ) -> Option<std::time::Instant> {
        self.schedule_swap_with_limit(now, swap_interval, nexium_common::speed_limit::pacing())
    }

    fn schedule_swap_with_limit(
        &mut self,
        now: std::time::Instant,
        swap_interval: i32,
        limited: bool,
    ) -> Option<std::time::Instant> {
        if !limited || swap_interval <= 0 {
            self.next_swap_deadline = None;
            return None;
        }

        const VSYNC_PERIOD: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let period = VSYNC_PERIOD.saturating_mul(swap_interval.clamp(1, 4) as u32);
        let target = self
            .next_swap_deadline
            .and_then(|previous| previous.checked_add(period))
            .unwrap_or_else(|| now.checked_add(period).unwrap_or(now));

        if target <= now {
            self.next_swap_deadline = Some(now);
            None
        } else {
            self.next_swap_deadline = Some(target);
            Some(target)
        }
    }

    pub fn set_preallocated(&mut self, slot: u32, buf: GraphicBuffer) {
        let idx = slot as usize;
        if idx >= self.slots.len() {
            self.slots.resize(idx + 1, Slot::default());
        }
        self.width = buf.width.max(self.width);
        self.height = buf.height.max(self.height);
        self.slots[idx].buffer = Some(buf);

        if self.slots[idx].state == SlotState::Free && !self.free.contains(&slot) {
            self.free.push_back(slot);
            self.notify_availability_changed();
        }
    }

    pub fn clear_preallocated(&mut self, slot: u32) {
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            return;
        };
        *entry = Slot::default();
        self.free.retain(|&candidate| candidate != slot);
        self.dequeued.retain(|&candidate| candidate != slot);
        self.queued.retain(|&candidate| candidate != slot);
        self.acquired.retain(|&candidate| candidate != slot);
        if self.last_queued == Some(slot) {
            self.last_queued = None;
        }
        self.pending_wake = true;
        self.notify_availability_changed();
    }

    pub fn wake_pending(&self) -> bool {
        self.pending_wake
    }

    pub fn take_wake(&mut self) -> bool {
        std::mem::take(&mut self.pending_wake)
    }

    pub fn request_buffer(&self, slot: u32) -> Option<&GraphicBuffer> {
        self.slots
            .get(slot as usize)
            .and_then(|s| s.buffer.as_ref())
    }

    pub fn slot_state(&self, slot: u32) -> Option<SlotState> {
        self.slots.get(slot as usize).map(|slot| slot.state)
    }

    pub fn has_free_slot(&self) -> bool {
        self.free.iter().copied().any(|slot| {
            self.slots
                .get(slot as usize)
                .is_some_and(|entry| entry.state == SlotState::Free && entry.buffer.is_some())
        })
    }

    pub fn try_dequeue(&mut self) -> Option<u32> {
        while let Some(slot) = self.free.pop_front() {
            let Some(state) = self.slots.get(slot as usize).map(|slot| slot.state) else {
                continue;
            };

            if state != SlotState::Free {
                continue;
            }

            self.set_state(slot, SlotState::Dequeued);
            self.dequeued.push(slot);
            self.notify_availability_changed();
            return Some(slot);
        }

        None
    }

    pub fn dequeue(&mut self) -> u32 {
        self.try_dequeue().unwrap_or(0)
    }

    pub fn queue(&mut self, slot: u32) -> bool {
        if self.slot_state(slot) != Some(SlotState::Dequeued) {
            return false;
        }

        self.dequeued.retain(|&s| s != slot);
        self.set_state(slot, SlotState::Queued);
        self.last_queued = Some(slot);
        if !self.queued.contains(&slot) {
            self.queued.push_back(slot);
        }
        self.record_queued_frame(slot, SlotState::Queued);
        true
    }

    pub fn acquire(&mut self) -> Option<u32> {
        while let Some(slot) = self.queued.pop_front() {
            if self.slot_state(slot) != Some(SlotState::Queued) {
                continue;
            }

            self.set_state(slot, SlotState::Acquired);
            self.acquired.push(slot);
            self.update_history(slot, |entry| entry.state = SlotState::Acquired);
            return Some(slot);
        }

        None
    }

    pub fn queue_and_acquire(&mut self, slot: u32) -> bool {
        if self.slot_state(slot) != Some(SlotState::Dequeued) {
            return false;
        }
        self.dequeued.retain(|&candidate| candidate != slot);
        self.queued.retain(|&candidate| candidate != slot);
        self.set_state(slot, SlotState::Acquired);
        self.last_queued = Some(slot);
        if !self.acquired.contains(&slot) {
            self.acquired.push(slot);
        }
        self.record_queued_frame(slot, SlotState::Acquired);
        true
    }

    pub fn release(&mut self, slot: u32) -> bool {
        if self.slot_state(slot) != Some(SlotState::Acquired) {
            return false;
        }

        self.acquired.retain(|&s| s != slot);
        self.set_state(slot, SlotState::Free);
        if !self.free.contains(&slot) {
            self.free.push_back(slot);
        }
        let now = std::time::Instant::now();
        self.update_history(slot, |entry| {
            entry.presented_at.get_or_insert(now);
        });
        self.notify_availability_changed();
        true
    }

    pub fn history(&self, count: usize) -> Vec<BufferHistoryEntry> {
        self.history.iter().rev().take(count).copied().collect()
    }

    fn record_queued_frame(&mut self, slot: u32, state: SlotState) {
        self.frame_counter += 1;
        let frame_number = self.frame_counter;
        if let Some(entry) = self.slots.get_mut(slot as usize) {
            entry.frame_number = frame_number;
        }
        if self.history.len() == BUFFER_HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back(BufferHistoryEntry {
            frame_number,
            queued_at: std::time::Instant::now(),
            presented_at: None,
            state,
        });
    }

    fn update_history(&mut self, slot: u32, update: impl FnOnce(&mut BufferHistoryEntry)) {
        let Some(frame_number) = self.slots.get(slot as usize).map(|entry| entry.frame_number) else {
            return;
        };
        if let Some(entry) = self.history.iter_mut().find(|entry| entry.frame_number == frame_number) {
            update(entry);
        }
    }

    pub fn cancel(&mut self, slot: u32) -> bool {
        if self.slot_state(slot) != Some(SlotState::Dequeued) {
            return false;
        }

        self.dequeued.retain(|&s| s != slot);
        self.set_state(slot, SlotState::Free);
        if !self.free.contains(&slot) {
            self.free.push_back(slot);
        }
        self.notify_availability_changed();
        true
    }

    fn notify_availability_changed(&self) {
        self.availability_generation.fetch_add(1, Ordering::Release);
        nexium_common::host_wake::signal();
    }

    fn set_state(&mut self, slot: u32, state: SlotState) {
        if let Some(entry) = self.slots.get_mut(slot as usize) {
            entry.state = state;
            entry.queued = state == SlotState::Queued;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferQueue, GraphicBuffer, SlotState};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn buffer() -> GraphicBuffer {
        GraphicBuffer {
            width: 1280,
            height: 720,
            ..GraphicBuffer::default()
        }
    }

    #[test]
    fn queued_and_acquired_slots_are_not_reused_by_producer() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());
        queue.set_preallocated(1, buffer());

        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(queue.queue(0));
        assert_eq!(queue.slot_state(0), Some(SlotState::Queued));

        assert_eq!(queue.try_dequeue(), Some(1));
        assert_eq!(queue.try_dequeue(), None);

        assert_eq!(queue.acquire(), Some(0));
        assert_eq!(queue.slot_state(0), Some(SlotState::Acquired));
        assert_eq!(queue.try_dequeue(), None);

        assert!(queue.release(0));
        assert_eq!(queue.try_dequeue(), Some(0));
    }

    #[test]
    fn clearing_a_preallocated_slot_frees_it_and_wakes_waiters() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());
        queue.set_preallocated(1, buffer());
        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(queue.queue_and_acquire(0));
        assert_eq!(queue.try_dequeue(), Some(1));
        assert!(!queue.has_free_slot());

        queue.clear_preallocated(0);
        queue.clear_preallocated(1);
        assert_eq!(queue.slot_state(0), Some(SlotState::Free));
        assert_eq!(queue.slot_state(1), Some(SlotState::Free));
        assert!(queue.request_buffer(0).is_none());
        assert!(queue.acquired.is_empty() && queue.dequeued.is_empty());
        assert!(!queue.has_free_slot());
        assert!(queue.wake_pending());
        assert!(queue.take_wake());
        assert!(!queue.wake_pending());
        assert!(!queue.release(0));

        queue.set_preallocated(0, buffer());
        assert!(queue.has_free_slot());
        assert_eq!(queue.try_dequeue(), Some(0));
    }

    #[test]
    fn buffer_history_lists_recent_frames_newest_first() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());
        queue.set_preallocated(1, buffer());
        assert!(queue.history(4).is_empty());

        for _ in 0..10 {
            let slot = queue.try_dequeue().unwrap();
            assert!(queue.queue_and_acquire(slot));
            assert!(queue.history(1)[0].presented_at.is_none());
            assert!(queue.release(slot));
        }
        let history = queue.history(32);
        assert_eq!(history.len(), super::BUFFER_HISTORY_LEN);
        assert_eq!(history.iter().map(|entry| entry.frame_number).collect::<Vec<_>>(), (3..=10).rev().collect::<Vec<_>>());
        assert!(history.iter().all(|entry| entry.presented_at.is_some() && entry.state == SlotState::Acquired));
        assert_eq!(queue.history(2).len(), 2);
    }

    #[test]
    fn consumer_acquires_queued_slots_in_fifo_order() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(3, buffer());
        queue.set_preallocated(5, buffer());

        assert_eq!(queue.try_dequeue(), Some(3));
        assert_eq!(queue.try_dequeue(), Some(5));
        assert!(queue.queue(3));
        assert!(queue.queue(5));

        assert_eq!(queue.acquire(), Some(3));
        assert_eq!(queue.acquire(), Some(5));
        assert_eq!(queue.acquire(), None);
    }

    #[test]
    fn invalid_transitions_do_not_change_ownership() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());

        assert!(!queue.queue(0));
        assert!(!queue.release(0));
        assert!(!queue.cancel(0));

        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(!queue.release(0));
        assert!(queue.queue(0));
        assert!(!queue.queue(0));
        assert!(!queue.cancel(0));

        assert_eq!(queue.acquire(), Some(0));
        assert!(!queue.cancel(0));
        assert!(queue.release(0));
        assert!(!queue.release(0));
    }

    #[test]
    fn cancel_returns_only_a_dequeued_slot_to_free_pool() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(2, buffer());

        assert_eq!(queue.try_dequeue(), Some(2));
        assert!(queue.cancel(2));
        assert_eq!(queue.slot_state(2), Some(SlotState::Free));
        assert_eq!(queue.try_dequeue(), Some(2));
    }

    #[test]
    fn replacing_an_in_flight_buffer_does_not_make_slot_free() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());
        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(queue.queue(0));

        let mut replacement = buffer();
        replacement.buffer_offset = 0x870000;
        queue.set_preallocated(0, replacement);

        assert_eq!(queue.slot_state(0), Some(SlotState::Queued));
        assert_eq!(queue.try_dequeue(), None);
        assert_eq!(queue.request_buffer(0).unwrap().buffer_offset, 0x870000);
    }

    #[test]
    fn synchronous_queue_acquire_does_not_consume_an_older_queued_slot() {
        let mut queue = BufferQueue::new(1);
        queue.set_preallocated(0, buffer());
        queue.set_preallocated(1, buffer());
        assert_eq!(queue.try_dequeue(), Some(0));
        assert_eq!(queue.try_dequeue(), Some(1));
        assert!(queue.queue(0));

        assert!(queue.queue_and_acquire(1));
        assert_eq!(queue.slot_state(0), Some(SlotState::Queued));
        assert_eq!(queue.slot_state(1), Some(SlotState::Acquired));
        assert_eq!(queue.acquire(), Some(0));
    }

    #[test]
    fn availability_generation_tracks_producer_visible_transitions() {
        let generation = Arc::new(AtomicU64::new(0));
        let mut queue = BufferQueue::with_generation(1, Arc::clone(&generation));

        assert!(!queue.has_free_slot());
        queue.set_preallocated(0, buffer());
        assert!(queue.has_free_slot());
        assert_eq!(generation.load(Ordering::Acquire), 1);

        queue.set_preallocated(0, buffer());
        assert_eq!(generation.load(Ordering::Acquire), 1);

        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(!queue.has_free_slot());
        assert_eq!(generation.load(Ordering::Acquire), 2);

        assert!(!queue.release(0));
        assert_eq!(generation.load(Ordering::Acquire), 2);
        assert!(queue.cancel(0));
        assert!(queue.has_free_slot());
        assert_eq!(generation.load(Ordering::Acquire), 3);
    }

    #[test]
    fn swap_interval_one_advances_on_the_vsync_cadence() {
        let mut queue = BufferQueue::new(1);
        let now = Instant::now();
        let period = Duration::from_nanos(16_666_667);

        let first = queue.schedule_swap(now, 1).unwrap();
        assert_eq!(first, now + period);
        assert_eq!(queue.schedule_swap(first, 1), Some(first + period));
    }

    #[test]
    fn swap_interval_two_waits_for_two_vsync_periods() {
        let mut queue = BufferQueue::new(1);
        let now = Instant::now();
        let two_periods = Duration::from_nanos(33_333_334);

        assert_eq!(queue.schedule_swap(now, 2), Some(now + two_periods));
    }

    #[test]
    fn swap_interval_zero_disables_and_resets_pacing() {
        let mut queue = BufferQueue::new(1);
        let now = Instant::now();
        let period = Duration::from_nanos(16_666_667);

        assert!(queue.schedule_swap(now, 1).is_some());
        assert_eq!(queue.schedule_swap(now, 0), None);
        assert_eq!(queue.schedule_swap(now, 1), Some(now + period));
    }

    #[test]
    fn unlocking_clears_pacing_and_relocking_starts_a_fresh_deadline() {
        let mut queue = BufferQueue::new(1);
        let now = Instant::now();
        let period = Duration::from_nanos(16_666_667);
        assert_eq!(queue.schedule_swap_with_limit(now, 2, true), Some(now + period * 2));
        for frame in 0..100 {
            assert_eq!(queue.schedule_swap_with_limit(now + Duration::from_micros(frame), 2, false), None);
        }
        let resumed = now + Duration::from_millis(1);
        assert_eq!(queue.schedule_swap_with_limit(resumed, 2, true), Some(resumed + period * 2));
        assert_eq!(queue.schedule_swap_with_limit(resumed, 1, false), None);
        assert_eq!(queue.schedule_swap_with_limit(resumed, 1, true), Some(resumed + period));
    }

    #[test]
    fn late_swap_resets_instead_of_catching_up() {
        let mut queue = BufferQueue::new(1);
        let now = Instant::now();
        let period = Duration::from_nanos(16_666_667);
        let first = queue.schedule_swap(now, 1).unwrap();
        let late = first + period + Duration::from_nanos(1);

        assert_eq!(queue.schedule_swap(late, 1), None);
        assert_eq!(queue.schedule_swap(late, 1), Some(late + period));
    }
}
