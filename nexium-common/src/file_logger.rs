use log::{LevelFilter, Metadata, Record};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::thread;
use std::time::Duration;

pub struct FileLogger {
    file: Arc<Mutex<BufWriter<std::fs::File>>>,
    buffer: Arc<Mutex<VecDeque<String>>>,
    level: Mutex<LevelFilter>,
    trace_targets: Vec<String>,
}

fn trace_targets_from_env() -> Vec<String> {
    std::env::var("NEXIUM_LOG_TRACE_TARGETS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|target| target.trim().to_string())
                .filter(|target| !target.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

impl FileLogger {
    pub fn new(capacity: usize) -> Result<(Self, Arc<Mutex<VecDeque<String>>>), String> {
        let log_dir = crate::paths::root().join("logs");

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
            level: Mutex::new(LevelFilter::Info),
            trace_targets: trace_targets_from_env(),
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
        *self.level.lock().unwrap_or_else(|e| e.into_inner()) = level;
        let global = if self.trace_targets.is_empty() {
            level
        } else {
            LevelFilter::Trace
        };
        let fault_file = Arc::clone(&self.file);
        log::set_boxed_logger(Box::new(self))?;
        let _ = FAULT_LOG_FILE.set(fault_file);
        log::set_max_level(global);
        Ok(())
    }

    pub fn flush_on_fault() -> bool {
        FAULT_LOG_FILE
            .get()
            .is_some_and(|file| try_flush_file(file))
    }

    fn traced_target(&self, target: &str) -> bool {
        self.trace_targets
            .iter()
            .any(|prefix| target.starts_with(prefix.as_str()))
    }
}

static FLUSH_RUNNING: AtomicBool = AtomicBool::new(false);
static FAULT_LOG_FILE: OnceLock<Arc<Mutex<BufWriter<std::fs::File>>>> = OnceLock::new();

fn try_flush_file(file: &Mutex<BufWriter<std::fs::File>>) -> bool {
    let mut writer = match file.try_lock() {
        Ok(writer) => writer,
        Err(TryLockError::Poisoned(error)) => error.into_inner(),
        Err(TryLockError::WouldBlock) => return false,
    };
    writer.flush().is_ok()
}

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let t = metadata.target();
        if !self.trace_targets.is_empty() && self.traced_target(t) {
            return true;
        }
        if metadata.level() > *self.level.lock().unwrap_or_else(|e| e.into_inner()) {
            return false;
        }
        if metadata.level() == log::Level::Trace {
            return false;
        }
        if metadata.level() >= log::Level::Info && (t.starts_with("wgpu") || t.starts_with("naga"))
        {
            return false;
        }
        if t.starts_with("dynarmic_sys")
            || t.starts_with("dynarmic_sys_mythrax")
            || t.contains("dynarmic")
        {
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
            if record.level() <= log::Level::Warn {
                let _ = f.flush();
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{try_flush_file, FileLogger};
    use log::{Level, LevelFilter, Log, Record};
    use std::collections::VecDeque;
    use std::io::BufWriter;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn test_logger() -> (FileLogger, std::path::PathBuf) {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "nexium-file-logger-{}-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed),
        ));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        (
            FileLogger {
                file: Arc::new(Mutex::new(BufWriter::with_capacity(64 * 1024, file))),
                buffer: Arc::new(Mutex::new(VecDeque::with_capacity(16))),
                level: Mutex::new(LevelFilter::Info),
                trace_targets: Vec::new(),
            },
            path,
        )
    }

    #[test]
    fn warning_and_error_flush_prior_buffered_records() {
        for level in [Level::Warn, Level::Error] {
            let (logger, path) = test_logger();
            logger.log(
                &Record::builder()
                    .level(Level::Info)
                    .args(format_args!("before fault"))
                    .build(),
            );
            assert!(std::fs::read(&path).unwrap().is_empty());
            logger.log(
                &Record::builder()
                    .level(level)
                    .args(format_args!("fault marker"))
                    .build(),
            );
            let contents = std::fs::read_to_string(&path).unwrap();
            assert!(contents.contains("before fault"));
            assert!(contents.contains("fault marker"));
            drop(logger);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn fault_flush_skips_held_writer_and_flushes_when_available() {
        let (logger, path) = test_logger();
        logger.log(
            &Record::builder()
                .level(Level::Info)
                .args(format_args!("before fault"))
                .build(),
        );
        let guard = logger.file.lock().unwrap();
        assert!(!try_flush_file(&logger.file));
        assert!(std::fs::read(&path).unwrap().is_empty());
        drop(guard);
        assert!(try_flush_file(&logger.file));
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("before fault"));
        drop(logger);
        std::fs::remove_file(path).unwrap();
    }
}
