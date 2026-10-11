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
    /// Keyed by handle value. Values come from a monotonically increasing
    /// counter, so lookup and close are O(1) and allocation order is unchanged.
    handles: std::collections::HashMap<u32, Handle>,
    /// Entries whose value was still live in `handles` when the counter
    /// wrapped onto it, oldest first. The original table was a `Vec` searched
    /// front to back, so the oldest entry answers lookups and closing it
    /// exposes the next one; this keeps exactly that behaviour.
    shadowed: Vec<Handle>,
    next_handle: u32,
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            handles: std::collections::HashMap::new(),
            shadowed: Vec::new(),
            next_handle: 0x100,
        }
    }

    pub fn create_handle(&mut self, handle_type: HandleType) -> u32 {
        let value = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1);
        let handle = Handle { value, handle_type };
        match self.handles.entry(value) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(handle);
            }
            // Only reachable after a 2^32 wrap: the oldest live entry keeps
            // answering for the value, as the original first-match lookup did.
            std::collections::hash_map::Entry::Occupied(_) => self.shadowed.push(handle),
        }
        value
    }

    pub fn get_handle(&self, value: u32) -> Option<&Handle> {
        self.handles.get(&value)
    }

    pub fn close_handle(&mut self, value: u32) -> Option<Handle> {
        let closed = self.handles.remove(&value)?;
        if let Some(pos) = self.shadowed.iter().position(|h| h.value == value) {
            let next = self.shadowed.remove(pos);
            self.handles.insert(value, next);
        }
        Some(closed)
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Handle, HandleTable, HandleType};

    #[test]
    fn handle_values_are_sequential_and_never_reused() {
        let mut table = HandleTable::new();
        let a = table.create_handle(HandleType::Event);
        let b = table.create_handle(HandleType::Session);
        assert_eq!((a, b), (0x100, 0x101));
        assert_eq!(table.get_handle(a).map(|h| h.handle_type), Some(HandleType::Event));
        assert_eq!(table.close_handle(a).map(|h| h.value), Some(a));
        assert!(table.get_handle(a).is_none());
        assert!(table.close_handle(a).is_none());
        assert_eq!(table.create_handle(HandleType::Thread), 0x102);
        assert_eq!(table.get_handle(b).map(|h| h.handle_type), Some(HandleType::Session));
    }

    /// The original `Vec` table, verbatim.
    struct VecTable {
        handles: Vec<Handle>,
        next_handle: u32,
    }

    impl VecTable {
        fn create_handle(&mut self, handle_type: HandleType) -> u32 {
            let value = self.next_handle;
            self.next_handle = self.next_handle.wrapping_add(1);
            self.handles.push(Handle { value, handle_type });
            value
        }

        fn get_handle(&self, value: u32) -> Option<&Handle> {
            self.handles.iter().find(|h| h.value == value)
        }

        fn close_handle(&mut self, value: u32) -> Option<Handle> {
            if let Some(pos) = self.handles.iter().position(|h| h.value == value) {
                Some(self.handles.remove(pos))
            } else {
                None
            }
        }
    }

    fn key(h: Option<&Handle>) -> Option<(u32, HandleType)> {
        h.map(|h| (h.value, h.handle_type))
    }

    #[test]
    fn handle_table_matches_original_vec_table() {
        const TYPES: [HandleType; 7] = [
            HandleType::Session,
            HandleType::Event,
            HandleType::SharedMemory,
            HandleType::TransferMemory,
            HandleType::Port,
            HandleType::Thread,
            HandleType::Process,
        ];
        let mut wraps = 0;
        for seed in 1..=300u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rand = |n: u64| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % n
            };
            let mut real = HandleTable::new();
            let mut model = VecTable { handles: Vec::new(), next_handle: 0x100 };
            let mut seen: Vec<u32> = Vec::new();
            for step in 0..3000 {
                match rand(100) {
                    0..=44 => {
                        let ty = TYPES[rand(TYPES.len() as u64) as usize];
                        let a = real.create_handle(ty);
                        let b = model.create_handle(ty);
                        assert_eq!(a, b, "seed {seed} step {step}: create");
                        seen.push(a);
                    }
                    45..=74 => {
                        let value = match rand(4) {
                            0 => rand(0x400) as u32,
                            _ if !seen.is_empty() => seen[rand(seen.len() as u64) as usize],
                            _ => 0x100,
                        };
                        assert_eq!(
                            key(real.close_handle(value).as_ref()),
                            key(model.close_handle(value).as_ref()),
                            "seed {seed} step {step}: close {value:#x}"
                        );
                    }
                    75..=96 => {
                        let value = if seen.is_empty() || rand(5) == 0 {
                            rand(u32::MAX as u64) as u32
                        } else {
                            seen[rand(seen.len() as u64) as usize]
                        };
                        assert_eq!(
                            key(real.get_handle(value)),
                            key(model.get_handle(value)),
                            "seed {seed} step {step}: get {value:#x}"
                        );
                    }
                    _ => {
                        // Emulate the counter wrapping onto earlier values (or
                        // reaching u32::MAX), which would otherwise take 2^32
                        // creations.
                        let next = match rand(3) {
                            0 if !seen.is_empty() => seen[rand(seen.len() as u64) as usize],
                            1 => u32::MAX - rand(3) as u32,
                            _ => 0x100 + rand(0x40) as u32,
                        };
                        real.next_handle = next;
                        model.next_handle = next;
                        wraps += 1;
                    }
                }
                for &value in seen.iter().rev().take(16) {
                    assert_eq!(
                        key(real.get_handle(value)),
                        key(model.get_handle(value)),
                        "seed {seed} step {step}: lookup {value:#x}"
                    );
                }
            }
            let mut live: Vec<(u32, HandleType)> = model.handles.iter().map(|h| (h.value, h.handle_type)).collect();
            live.sort_by_key(|(v, _)| *v);
            for (value, _) in &live {
                assert_eq!(key(real.get_handle(*value)), key(model.get_handle(*value)));
            }
            // Drain everything through close: identical sequence of results.
            for (value, _) in live {
                assert_eq!(
                    key(real.close_handle(value).as_ref()),
                    key(model.close_handle(value).as_ref())
                );
            }
            assert!(real.handles.is_empty() && real.shadowed.is_empty());
        }
        assert!(wraps > 0);
    }
}
