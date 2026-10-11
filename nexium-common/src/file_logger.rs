use log::{Level, LevelFilter, Metadata, Record};
use std::cell::RefCell;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock, RwLock, TryLockError};
use std::thread;
use std::time::Duration;

/// Lines the in-app viewer queue holds before further lines are dropped from
/// the viewer. Dropped lines still reach the log file.
pub const DEFAULT_UI_QUEUE_LINES: usize = 65_536;

pub struct FileLogger {
    file: Arc<Mutex<BufWriter<std::fs::File>>>,
    ui_tx: Option<SyncSender<String>>,
    ui_dropped: Arc<AtomicU64>,
    filter: Arc<LogFilter>,
}

/// A per-target level override, matched as a prefix of the record target
/// (`nexium_kernel` covers `nexium_kernel::kernel::svc`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetLevel {
    pub target: String,
    pub level: LevelFilter,
}

/// Runtime log filter shared between the logger and the UI. The global level
/// and the per-target rules can change at any time; the change applies to the
/// next record on every thread.
pub struct LogFilter {
    id: usize,
    default: AtomicUsize,
    generation: AtomicU64,
    rules: RwLock<Arc<[TargetLevel]>>,
    installed: AtomicBool,
}

static NEXT_FILTER_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static RULE_CACHE: RefCell<(usize, u64, Arc<[TargetLevel]>)> =
        RefCell::new((0, u64::MAX, Arc::from(Vec::new())));
}

impl LogFilter {
    pub fn new(default: LevelFilter) -> Self {
        Self {
            id: NEXT_FILTER_ID.fetch_add(1, Ordering::Relaxed),
            default: AtomicUsize::new(default as usize),
            generation: AtomicU64::new(0),
            rules: RwLock::new(Arc::from(Vec::new())),
            installed: AtomicBool::new(false),
        }
    }

    pub fn default_level(&self) -> LevelFilter {
        level_filter_from_usize(self.default.load(Ordering::Relaxed))
    }

    pub fn rules(&self) -> Vec<TargetLevel> {
        self.rules
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .to_vec()
    }

    pub fn set_default_level(&self, level: LevelFilter) {
        self.default.store(level as usize, Ordering::Relaxed);
        self.sync_max_level();
    }

    pub fn set_rules(&self, rules: Vec<TargetLevel>) {
        *self.rules.write().unwrap_or_else(|e| e.into_inner()) = Arc::from(rules);
        self.generation.fetch_add(1, Ordering::Release);
        self.sync_max_level();
    }

    /// The most verbose level any target can currently log at.
    pub fn max_level(&self) -> LevelFilter {
        self.rules
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|rule| rule.level)
            .fold(self.default_level(), Ord::max)
    }

    /// The level that applies to `target`, and whether a per-target rule
    /// chose it.
    pub fn level_for(&self, target: &str) -> (LevelFilter, bool) {
        let default = self.default_level();
        let generation = self.generation.load(Ordering::Acquire);
        if generation == 0 {
            return (default, false);
        }
        let key = self.id;
        RULE_CACHE
            .try_with(|cache| {
                let mut cache = cache.try_borrow_mut().ok()?;
                if cache.0 != key || cache.1 != generation {
                    let rules = Arc::clone(&self.rules.read().unwrap_or_else(|e| e.into_inner()));
                    *cache = (key, generation, rules);
                }
                Some(resolve_target_level(default, &cache.2, target))
            })
            .ok()
            .flatten()
            .unwrap_or_else(|| {
                let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
                resolve_target_level(default, &rules, target)
            })
    }

    fn sync_max_level(&self) {
        if self.installed.load(Ordering::Relaxed) {
            log::set_max_level(self.max_level());
        }
    }
}

/// Longest matching prefix wins; among equal prefixes the later rule wins.
pub fn resolve_target_level(
    default: LevelFilter,
    rules: &[TargetLevel],
    target: &str,
) -> (LevelFilter, bool) {
    let mut best: Option<&TargetLevel> = None;
    for rule in rules {
        if target.starts_with(rule.target.as_str())
            && best.map_or(true, |b| rule.target.len() >= b.target.len())
        {
            best = Some(rule);
        }
    }
    best.map_or((default, false), |rule| (rule.level, true))
}

pub fn parse_level_filter(value: &str) -> Option<LevelFilter> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" | "warning" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" => Some(LevelFilter::Debug),
        "trace" => Some(LevelFilter::Trace),
        _ => None,
    }
}

/// Parses `target=level` pairs separated by commas, semicolons or whitespace.
/// Returns the rules that parsed and a message for each entry that did not.
pub fn parse_target_levels(spec: &str) -> (Vec<TargetLevel>, Vec<String>) {
    let mut rules = Vec::new();
    let mut errors = Vec::new();
    for item in spec
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let Some((target, level)) = item.split_once('=') else {
            errors.push(format!("'{item}' needs the form target=level"));
            continue;
        };
        let target = target.trim();
        if target.is_empty() {
            errors.push(format!("'{item}' has no target"));
            continue;
        }
        match parse_level_filter(level) {
            Some(level) => rules.push(TargetLevel {
                target: target.to_string(),
                level,
            }),
            None => errors.push(format!("'{item}' has an unknown level")),
        }
    }
    (rules, errors)
}

pub fn format_target_levels(rules: &[TargetLevel]) -> String {
    rules
        .iter()
        .map(|rule| format!("{}={}", rule.target, rule.level.as_str().to_ascii_lowercase()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn level_filter_from_usize(value: usize) -> LevelFilter {
    match value {
        0 => LevelFilter::Off,
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    }
}

fn trace_targets_from_env() -> Vec<TargetLevel> {
    std::env::var("NEXIUM_LOG_TRACE_TARGETS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|target| target.trim().to_string())
                .filter(|target| !target.is_empty())
                .map(|target| TargetLevel {
                    target,
                    level: LevelFilter::Trace,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The receiving end of the in-app viewer queue, plus what the viewer needs
/// to know about the logger behind it.
pub struct LogFeed {
    rx: Receiver<String>,
    dropped: Arc<AtomicU64>,
    filter: Arc<LogFilter>,
    path: Option<PathBuf>,
}

impl LogFeed {
    /// A feed with no logger behind it; lines sent on the returned sender
    /// show up in the feed.
    pub fn detached(capacity: usize) -> (SyncSender<String>, Self) {
        let (tx, rx) = std::sync::mpsc::sync_channel(capacity.max(1));
        (
            tx,
            Self {
                rx,
                dropped: Arc::new(AtomicU64::new(0)),
                filter: Arc::new(LogFilter::new(LevelFilter::Info)),
                path: None,
            },
        )
    }

    pub fn try_recv(&self) -> Option<String> {
        self.rx.try_recv().ok()
    }

    /// Lines dropped from the viewer queue since the last call.
    pub fn take_dropped(&self) -> u64 {
        self.dropped.swap(0, Ordering::Relaxed)
    }

    pub fn filter(&self) -> &Arc<LogFilter> {
        &self.filter
    }

    pub fn log_path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

impl FileLogger {
    /// Creates the log file and a viewer queue that holds up to
    /// `ui_queue_lines` lines until the UI drains them.
    pub fn new(ui_queue_lines: usize) -> Result<(Self, LogFeed), String> {
        let log_dir = crate::paths::root().join("logs");

        fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create log directory: {}", e))?;

        let log_path = log_dir.join(format!(
            "nexium-{}.log",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        ));

        let (logger, feed) = Self::with_path(&log_path, ui_queue_lines)?;

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

        let _ = LOG_PATH.set(log_path);
        Ok((logger, feed))
    }

    /// A logger writing to `log_path` with a viewer queue of
    /// `ui_queue_lines`. Unlike [`FileLogger::new`] it starts no flush
    /// thread and does not become the session log file.
    pub fn with_path(log_path: &Path, ui_queue_lines: usize) -> Result<(Self, LogFeed), String> {
        let file = std::fs::File::create(log_path)
            .map_err(|e| format!("Failed to create log file: {}", e))?;
        let buffered = BufWriter::with_capacity(64 * 1024, file);
        let (tx, rx) = std::sync::mpsc::sync_channel(ui_queue_lines.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let filter = Arc::new(LogFilter::new(LevelFilter::Info));
        let env_rules = trace_targets_from_env();
        if !env_rules.is_empty() {
            filter.set_rules(env_rules);
        }
        Ok((
            Self {
                file: Arc::new(Mutex::new(buffered)),
                ui_tx: Some(tx),
                ui_dropped: Arc::clone(&dropped),
                filter: Arc::clone(&filter),
            },
            LogFeed {
                rx,
                dropped,
                filter,
                path: Some(log_path.to_path_buf()),
            },
        ))
    }

    pub fn init(self, level: LevelFilter) -> Result<(), log::SetLoggerError> {
        self.filter.default.store(level as usize, Ordering::Relaxed);
        let fault_file = Arc::clone(&self.file);
        let filter = Arc::clone(&self.filter);
        log::set_boxed_logger(Box::new(self))?;
        let _ = FAULT_LOG_FILE.set(fault_file);
        filter.installed.store(true, Ordering::Relaxed);
        log::set_max_level(filter.max_level());
        let _ = INSTALLED_FILTER.set(filter);
        Ok(())
    }

    pub fn flush_on_fault() -> bool {
        FAULT_LOG_FILE
            .get()
            .is_some_and(|file| try_flush_file(file))
    }

    /// Whether a record passes, and whether a per-target rule decided it.
    fn admit(&self, metadata: &Metadata) -> Option<bool> {
        let target = metadata.target();
        let (level, explicit) = self.filter.level_for(target);
        if metadata.level() > level {
            return None;
        }
        if !explicit && builtin_noise(target, metadata.level()) {
            return None;
        }
        Some(explicit)
    }
}

/// Targets that flood the log unless a per-target rule asks for them.
fn builtin_noise(target: &str, level: Level) -> bool {
    if level >= Level::Info && (target.starts_with("wgpu") || target.starts_with("naga")) {
        return true;
    }
    target.contains("dynarmic") && level > Level::Error
}

fn noisy_message(message: &str) -> bool {
    message.starts_with("[Dynarmic]")
        || message.starts_with("dynarmic SVC ")
        || message.starts_with("IFile.Read")
        || message.starts_with("IFsStorage.Read")
}

/// One log line as written to the file and shown in the viewer:
/// `[HH:MM:SS.mmm] [LEVEL] [target] message`.
pub fn format_line(timestamp: &str, level: Level, target: &str, message: &str) -> String {
    if target.is_empty() {
        format!("[{}] [{:>5}] {}", timestamp, level, message)
    } else {
        format!("[{}] [{:>5}] [{}] {}", timestamp, level, target, message)
    }
}

static FLUSH_RUNNING: AtomicBool = AtomicBool::new(false);
static FAULT_LOG_FILE: OnceLock<Arc<Mutex<BufWriter<std::fs::File>>>> = OnceLock::new();
static INSTALLED_FILTER: OnceLock<Arc<LogFilter>> = OnceLock::new();
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// The filter of the installed `FileLogger`, if one is installed.
pub fn installed_filter() -> Option<&'static Arc<LogFilter>> {
    INSTALLED_FILTER.get()
}

/// The file the logger writes this session to.
pub fn current_log_path() -> Option<&'static Path> {
    LOG_PATH.get().map(PathBuf::as_path)
}

/// Changes the global level live, through the installed `FileLogger` when
/// there is one.
pub fn set_global_level(level: LevelFilter) {
    match installed_filter() {
        Some(filter) => filter.set_default_level(level),
        None => log::set_max_level(level),
    }
}

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
        self.admit(metadata).is_some()
    }

    fn log(&self, record: &Record) {
        let Some(explicit) = self.admit(record.metadata()) else {
            return;
        };

        let message = record.args().to_string();
        if !explicit && record.level() >= Level::Debug && noisy_message(&message) {
            return;
        }

        let timestamp = chrono::Local::now().format("%H:%M:%S%.3f").to_string();
        let line = format_line(&timestamp, record.level(), record.target(), &message);

        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{}", line);
            if record.level() <= Level::Warn {
                let _ = f.flush();
            }
        }

        if let Some(tx) = &self.ui_tx {
            match tx.try_send(line) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => {}
                Err(TrySendError::Full(_)) => {
                    self.ui_dropped.fetch_add(1, Ordering::Relaxed);
                }
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
    use super::{
        format_line, format_target_levels, parse_target_levels, resolve_target_level,
        try_flush_file, FileLogger, LogFilter, TargetLevel,
    };
    use log::{Level, LevelFilter, Log, Metadata, Record};
    use std::io::BufWriter;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::Receiver;
    use std::sync::{Arc, Mutex};

    fn test_logger() -> (FileLogger, std::path::PathBuf) {
        let (logger, path, _) = test_logger_with_queue(16);
        (logger, path)
    }

    fn test_logger_with_queue(lines: usize) -> (FileLogger, std::path::PathBuf, Receiver<String>) {
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
        let (tx, rx) = std::sync::mpsc::sync_channel(lines);
        (
            FileLogger {
                file: Arc::new(Mutex::new(BufWriter::with_capacity(64 * 1024, file))),
                ui_tx: Some(tx),
                ui_dropped: Arc::new(AtomicU64::new(0)),
                filter: Arc::new(LogFilter::new(LevelFilter::Info)),
            },
            path,
            rx,
        )
    }

    fn rule(target: &str, level: LevelFilter) -> TargetLevel {
        TargetLevel {
            target: target.to_string(),
            level,
        }
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

    #[test]
    fn longest_prefix_rule_wins() {
        let rules = [
            rule("nexium_kernel", LevelFilter::Debug),
            rule("nexium_kernel::kernel::svc", LevelFilter::Trace),
            rule("nexium_nvdrv", LevelFilter::Warn),
        ];
        let info = LevelFilter::Info;
        assert_eq!(
            resolve_target_level(info, &rules, "nexium_kernel::kernel::svc"),
            (LevelFilter::Trace, true)
        );
        assert_eq!(
            resolve_target_level(info, &rules, "nexium_kernel::services::fsp"),
            (LevelFilter::Debug, true)
        );
        assert_eq!(
            resolve_target_level(info, &rules, "nexium_nvdrv::gpu"),
            (LevelFilter::Warn, true)
        );
        assert_eq!(
            resolve_target_level(info, &rules, "nexium_gpu::renderer"),
            (LevelFilter::Info, false)
        );
    }

    #[test]
    fn later_rule_wins_on_equal_prefix() {
        let rules = [
            rule("nexium_gpu", LevelFilter::Debug),
            rule("nexium_gpu", LevelFilter::Error),
        ];
        assert_eq!(
            resolve_target_level(LevelFilter::Info, &rules, "nexium_gpu::x"),
            (LevelFilter::Error, true)
        );
    }

    #[test]
    fn target_level_spec_round_trips_and_reports_errors() {
        let (rules, errors) =
            parse_target_levels("nexium_kernel=debug, nexium_nvdrv=TRACE;bad nope=loud =info");
        assert_eq!(
            rules,
            vec![
                rule("nexium_kernel", LevelFilter::Debug),
                rule("nexium_nvdrv", LevelFilter::Trace),
            ]
        );
        assert_eq!(errors.len(), 3);
        assert_eq!(
            format_target_levels(&rules),
            "nexium_kernel=debug, nexium_nvdrv=trace"
        );
    }

    #[test]
    fn filter_changes_apply_to_the_next_record() {
        let filter = LogFilter::new(LevelFilter::Info);
        assert_eq!(filter.level_for("nexium_kernel"), (LevelFilter::Info, false));
        filter.set_rules(vec![rule("nexium_kernel", LevelFilter::Trace)]);
        assert_eq!(filter.level_for("nexium_kernel::svc"), (LevelFilter::Trace, true));
        assert_eq!(filter.max_level(), LevelFilter::Trace);
        filter.set_rules(Vec::new());
        assert_eq!(filter.level_for("nexium_kernel::svc"), (LevelFilter::Info, false));
        filter.set_default_level(LevelFilter::Warn);
        assert_eq!(filter.max_level(), LevelFilter::Warn);
    }

    #[test]
    fn per_target_rule_admits_debug_and_overrides_builtin_noise() {
        let (logger, path, rx) = test_logger_with_queue(16);
        let debug_meta = Metadata::builder()
            .level(Level::Debug)
            .target("nexium_kernel::svc")
            .build();
        assert!(!logger.enabled(&debug_meta));
        logger
            .filter
            .set_rules(vec![rule("nexium_kernel", LevelFilter::Debug)]);
        assert!(logger.enabled(&debug_meta));
        let wgpu_info = Metadata::builder().level(Level::Info).target("wgpu_core").build();
        assert!(!logger.enabled(&wgpu_info));
        logger.filter.set_rules(vec![rule("wgpu", LevelFilter::Info)]);
        assert!(logger.enabled(&wgpu_info));
        logger.log(
            &Record::builder()
                .level(Level::Info)
                .target("wgpu_core")
                .args(format_args!("adapter"))
                .build(),
        );
        let line = rx.try_recv().unwrap();
        assert!(line.ends_with("[ INFO] [wgpu_core] adapter"), "{line}");
        drop(logger);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn full_viewer_queue_drops_without_blocking() {
        let (logger, path, rx) = test_logger_with_queue(1);
        for n in 0..3 {
            logger.log(
                &Record::builder()
                    .level(Level::Info)
                    .args(format_args!("line {n}"))
                    .build(),
            );
        }
        assert!(rx.try_recv().unwrap().ends_with("line 0"));
        assert!(rx.try_recv().is_err());
        assert_eq!(logger.ui_dropped.load(Ordering::Relaxed), 2);
        logger.flush();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("line 2"));
        drop(logger);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn line_format_includes_target_when_present() {
        assert_eq!(
            format_line("12:00:00.000", Level::Warn, "nexium_gpu", "hi"),
            "[12:00:00.000] [ WARN] [nexium_gpu] hi"
        );
        assert_eq!(
            format_line("12:00:00.000", Level::Info, "", "hi"),
            "[12:00:00.000] [ INFO] hi"
        );
    }
}
