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
}

#[derive(Clone, Debug)]
pub struct QueuedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub present_at: Option<std::time::Instant>,
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
        }
    }

    pub fn schedule_swap(
        &mut self,
        now: std::time::Instant,
        swap_interval: i32,
    ) -> Option<std::time::Instant> {
        if swap_interval <= 0 {
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
        true
    }

    pub fn acquire(&mut self) -> Option<u32> {
        while let Some(slot) = self.queued.pop_front() {
            if self.slot_state(slot) != Some(SlotState::Queued) {
                continue;
            }

            self.set_state(slot, SlotState::Acquired);
            self.acquired.push(slot);
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
        self.notify_availability_changed();
        true
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
