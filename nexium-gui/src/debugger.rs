use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

pub struct LogHistory {
    entries: VecDeque<String>,
    max_entries: usize,
}

impl LogHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
        }
    }

    pub fn add_entry(&mut self, entry: String) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

pub struct DebuggerState {
    pub memory_address: u64,
    pub memory_address_input: String,
    pub memory_size: u32,
    pub show_memory: bool,
    pub show_registers: bool,
    pub show_disasm: bool,
    pub show_logs: bool,
    pub show_performance: bool,
    pub show_wait_tree: bool,
    pub log_history: Arc<Mutex<LogHistory>>,
}

impl DebuggerState {
    pub fn new() -> Self {
        Self {
            memory_address: 0x80000000,
            memory_address_input: "0x80000000".to_string(),
            memory_size: 256,
            show_memory: false,
            show_registers: false,
            show_disasm: false,
            show_logs: false,
            show_performance: false,
            show_wait_tree: false,
            log_history: Arc::new(Mutex::new(LogHistory::new(1000))),
        }
    }

    pub fn add_log(&self, message: String) {
        self.log_history.lock().add_entry(message);
    }

    pub fn toggle_memory(&mut self) {
        self.show_memory = !self.show_memory;
    }

    pub fn toggle_registers(&mut self) {
        self.show_registers = !self.show_registers;
    }

    pub fn toggle_disasm(&mut self) {
        self.show_disasm = !self.show_disasm;
    }

    pub fn toggle_logs(&mut self) {
        self.show_logs = !self.show_logs;
    }

    pub fn toggle_performance(&mut self) {
        self.show_performance = !self.show_performance;
    }

    pub fn toggle_wait_tree(&mut self) {
        self.show_wait_tree = !self.show_wait_tree;
    }
}

impl Default for DebuggerState {
    fn default() -> Self {
        Self::new()
    }
}
