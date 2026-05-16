use std::fs;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use log::{Record, Metadata, LevelFilter};

pub struct FileLogger {
    file: Arc<Mutex<std::fs::File>>,
    buffer: Arc<Mutex<VecDeque<String>>>,
}

impl FileLogger {
    pub fn new(capacity: usize) -> Result<(Self, Arc<Mutex<VecDeque<String>>>), String> {
        let log_dir = directories::ProjectDirs::from("", "", "HorizonRust")
            .ok_or_else(|| "Failed to get project directories".to_string())?
            .data_dir()
            .parent()
            .ok_or_else(|| "Failed to get data parent".to_string())?
            .join("HorizonRust")
            .join("logs");

        fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create log directory: {}", e))?;

        let log_path = log_dir.join(format!(
            "horizonrust-{}.log",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        ));

        let file = std::fs::File::create(log_path)
            .map_err(|e| format!("Failed to create log file: {}", e))?;
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));

        Ok((
            Self {
                file: Arc::new(Mutex::new(file)),
                buffer: buffer.clone(),
            },
            buffer,
        ))
    }

    pub fn init(self, level: LevelFilter) -> Result<(), log::SetLoggerError> {
        log::set_boxed_logger(Box::new(self))?;
        log::set_max_level(level);
        Ok(())
    }
}

impl log::Log for FileLogger {
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

        if let Ok(mut f) = self.file.lock() {
            let timestamp = chrono::Local::now().format("%H:%M:%S%.3f");
            let _ = writeln!(f, "[{}] {}", timestamp, line);
            let _ = f.flush();
        }

        eprintln!("{}", line);
    }

    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}
