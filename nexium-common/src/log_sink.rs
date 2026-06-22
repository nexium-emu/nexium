use log::{LevelFilter, Metadata, Record};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct BufferedLogger {
    buffer: Arc<Mutex<VecDeque<String>>>,
}

impl BufferedLogger {
    pub fn new(capacity: usize) -> (Self, Arc<Mutex<VecDeque<String>>>) {
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));
        (
            Self {
                buffer: buffer.clone(),
            },
            buffer,
        )
    }

    pub fn init(self, level: LevelFilter) -> Result<(), log::SetLoggerError> {
        log::set_boxed_logger(Box::new(self))?;
        log::set_max_level(level);
        Ok(())
    }
}

impl log::Log for BufferedLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let line = format!("[{:>5}] {}", record.level(), record.args());

        if let Ok(mut buf) = self.buffer.lock() {
            if buf.len() >= buf.capacity() {
                buf.pop_front();
            }
            buf.push_back(line.clone());
        }

        eprintln!("{}", line);
    }

    fn flush(&self) {}
}
