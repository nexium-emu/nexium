use log::{LevelFilter, Metadata, Record};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub struct FileLogger {
    file: Arc<Mutex<BufWriter<std::fs::File>>>,
    buffer: Arc<Mutex<VecDeque<String>>>,
}

impl FileLogger {
    pub fn new(capacity: usize) -> Result<(Self, Arc<Mutex<VecDeque<String>>>), String> {
        let log_dir = directories::BaseDirs::new()
            .ok_or_else(|| "Failed to get base directories".to_string())?
            .config_dir()
            .join("NeXium")
            .join("logs");

        fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create log directory: {}", e))?;

        let log_path = log_dir.join(format!(
            "nexium-{}.log",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        ));

        let file = std::fs::File::create(log_path)
            .map_err(|e| format!("Failed to create log file: {}", e))?;
        let buffered = BufWriter::with_capacity(64 * 1024, file);
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));

        let logger = Self {
            file: Arc::new(Mutex::new(buffered)),
            buffer: buffer.clone(),
        };

        let flush_file = Arc::clone(&logger.file);
        FLUSH_RUNNING.store(true, Ordering::Relaxed);
        thread::spawn(move || {
            while FLUSH_RUNNING.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
                if let Ok(mut f) = flush_file.lock() {
                    let _ = f.flush();
                }
            }
        });

        Ok((logger, buffer))
    }

    pub fn init(self, level: LevelFilter) -> Result<(), log::SetLoggerError> {
        log::set_boxed_logger(Box::new(self))?;
        log::set_max_level(level);
        Ok(())
    }
}

static FLUSH_RUNNING: AtomicBool = AtomicBool::new(false);

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        if metadata.level() > log::max_level() {
            return false;
        }
        if metadata.level() == log::Level::Trace {
            return false;
        }
        let t = metadata.target();
        if metadata.level() >= log::Level::Info && (t.starts_with("wgpu") || t.starts_with("naga"))
        {
            return false;
        }
        if t.starts_with("dynarmic_sys") || t.starts_with("dynarmic_sys_mythrax") || t.contains("dynarmic") {
            if metadata.level() > log::Level::Error {
                return false;
            }
        }
        true
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        if record.level() >= log::Level::Debug {
            let msg = format!("{}", record.args());
            if msg.starts_with("[Dynarmic]") || msg.starts_with("dynarmic SVC ") {
                return;
            }
            if msg.starts_with("IFile.Read") || msg.starts_with("IFsStorage.Read") {
                return;
            }
        }

        let line = format!("[{:>5}] {}", record.level(), record.args());

        if let Ok(mut buf) = self.buffer.lock() {
            if buf.len() >= buf.capacity() {
                buf.pop_front();
            }
            buf.push_back(line.clone());
        }

        if let Ok(mut f) = self.file.lock() {
            let timestamp = chrono::Local::now().format("%H:%M:%S%.3f");
            let _ = writeln!(f, "[{}] {}", timestamp, line);
        }
    }

    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}
