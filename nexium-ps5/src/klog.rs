use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::sys;

static STDOUT: AtomicBool = AtomicBool::new(true);
static FILE: Mutex<Option<File>> = Mutex::new(None);
static START: OnceLock<Instant> = OnceLock::new();

pub const PREFIX: &str = "[nexium-ps5]";

pub fn set_stdout(enabled: bool) {
    STDOUT.store(enabled, Ordering::Relaxed);
}

pub fn open_file(path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    *FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(file);
    Ok(())
}

pub fn line(text: &str) {
    let start = START.get_or_init(Instant::now);
    let elapsed = start.elapsed();
    let stamped = format!(
        "{PREFIX} {:>4}.{:03} {text}\n",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    if let Ok(c) = CString::new(stamped.replace('\0', " ")) {
        unsafe {
            sys::sceKernelDebugOutText(0, c.as_ptr());
        }
    }
    if STDOUT.load(Ordering::Relaxed) {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(stamped.as_bytes());
        let _ = out.flush();
    }
    if let Some(file) = FILE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        let _ = file.write_all(stamped.as_bytes());
        let _ = file.flush();
    }
}

#[macro_export]
macro_rules! klog {
    ($($arg:tt)*) => {
        $crate::klog::line(&format!($($arg)*))
    };
}

struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            line(&format!("{} {}: {}", record.level(), record.target(), record.args()));
        }
    }

    fn flush(&self) {}
}

static LOGGER: Logger = Logger;

pub fn install(level: log::LevelFilter) {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level);
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        line(&format!("PANIC in thread {name}: {info}"));
    }));
}
