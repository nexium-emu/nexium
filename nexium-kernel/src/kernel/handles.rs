#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HandleType {
    Session,
    Event,
    SharedMemory,
    TransferMemory,
    Port,
    Thread,
    Process,
}

#[derive(Copy, Clone, Debug)]
pub struct Handle {
    pub value: u32,
    pub handle_type: HandleType,
}

pub struct HandleTable {
    handles: Vec<Handle>,
    next_handle: u32,
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
            next_handle: 0x100,
        }
    }

    pub fn create_handle(&mut self, handle_type: HandleType) -> u32 {
        let value = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1);
        self.handles.push(Handle { value, handle_type });
        value
    }

    pub fn get_handle(&self, value: u32) -> Option<&Handle> {
        self.handles.iter().find(|h| h.value == value)
    }

    pub fn close_handle(&mut self, value: u32) -> Option<Handle> {
        if let Some(pos) = self.handles.iter().position(|h| h.value == value) {
            Some(self.handles.remove(pos))
        } else {
            None
        }
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}
