use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct GraphicBuffer {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
    pub usage: u32,
    pub nvmap_id: u32,
    pub buffer_offset: u64,
    pub size: u32,
}

impl Default for GraphicBuffer {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            stride: 1280,
            format: 1,
            usage: 0,
            nvmap_id: 0,
            buffer_offset: 0,
            size: 1280 * 720 * 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BufferSlot {
    pub buffer: GraphicBuffer,
    pub is_acquired: bool,
}

#[derive(Clone, Debug)]
pub struct QueuedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct BufferQueue {
    pub binder_id: u32,
    pub width: u32,
    pub height: u32,
    pub slots: HashMap<u32, BufferSlot>,
    pub next_slot: u32,
    pub max_slots: u32,
    pub connected_api: i32,
}

impl BufferQueue {
    pub fn new(binder_id: u32) -> Self {
        Self {
            binder_id,
            width: 1280,
            height: 720,
            slots: HashMap::new(),
            next_slot: 0,
            max_slots: 2,
            connected_api: 0,
        }
    }

    pub fn set_preallocated(&mut self, slot: u32, gb: GraphicBuffer) {
        self.slots.insert(slot, BufferSlot { buffer: gb, is_acquired: false });
        if slot >= self.next_slot {
            self.next_slot = slot;
        }
    }

    pub fn request_buffer(&self, slot: u32) -> Option<&GraphicBuffer> {
        self.slots.get(&slot).map(|s| &s.buffer)
    }

    pub fn dequeue(&mut self) -> u32 {
        for (id, slot) in &mut self.slots {
            if !slot.is_acquired {
                slot.is_acquired = true;
                return *id;
            }
        }
        let id = self.next_slot;
        self.next_slot = self.next_slot.wrapping_add(1);
        self.slots.insert(id, BufferSlot {
            buffer: GraphicBuffer::default(),
            is_acquired: true,
        });
        id
    }

    pub fn queue(&mut self, slot: u32) {
        if let Some(s) = self.slots.get_mut(&slot) {
            s.is_acquired = false;
        }
    }

    pub fn cancel(&mut self, slot: u32) {
        if let Some(s) = self.slots.get_mut(&slot) {
            s.is_acquired = false;
        }
    }
}
