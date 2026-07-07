use std::collections::VecDeque;

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

#[derive(Clone, Debug, Default)]
pub struct Slot {
    pub buffer: Option<GraphicBuffer>,
    pub queued: bool,
    pub last_swap_interval: u32,
}

#[derive(Clone, Debug)]
pub struct QueuedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct BufferQueue {
    pub binder_id: u32,
    pub slots: Vec<Slot>,
    pub free: VecDeque<u32>,
    pub dequeued: Vec<u32>,
    pub queued: VecDeque<u32>,
    pub width: u32,
    pub height: u32,
    pub last_queued: Option<u32>,
    pub connected_api: i32,
}

impl BufferQueue {
    pub fn new(binder_id: u32) -> Self {
        Self {
            binder_id,
            slots: vec![Slot::default(); 8],
            free: VecDeque::new(),
            dequeued: Vec::new(),
            queued: VecDeque::new(),
            width: 1280,
            height: 720,
            last_queued: None,
            connected_api: 0,
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
        if !self.free.contains(&slot) && !self.dequeued.contains(&slot) {
            self.free.push_back(slot);
        }
    }

    pub fn request_buffer(&self, slot: u32) -> Option<&GraphicBuffer> {
        self.slots
            .get(slot as usize)
            .and_then(|s| s.buffer.as_ref())
    }

    pub fn dequeue(&mut self) -> u32 {
        if let Some(slot) = self.free.pop_front() {
            if let Some(s) = self.slots.get_mut(slot as usize) {
                s.queued = false;
            }
            self.dequeued.push(slot);
            return slot;
        }
        if let Some(slot) = self.queued.pop_front() {
            if let Some(s) = self.slots.get_mut(slot as usize) {
                s.queued = false;
            }
            self.dequeued.push(slot);
            return slot;
        }
        0
    }

    pub fn queue(&mut self, slot: u32) -> bool {
        let idx = slot as usize;
        if idx < self.slots.len() {
            self.slots[idx].queued = false;
        }
        self.dequeued.retain(|&s| s != slot);
        self.last_queued = Some(slot);
        if !self.free.contains(&slot) {
            self.free.push_back(slot);
        }
        true
    }

    pub fn cancel(&mut self, slot: u32) {
        self.dequeued.retain(|&s| s != slot);
        if !self.free.contains(&slot) {
            self.free.push_back(slot);
        }
    }
}
