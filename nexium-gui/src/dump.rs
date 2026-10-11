use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use nexium_loader::dump::{self, DumpError, DumpProgress, DumpSummary};
use nexium_loader::Application;

const REPAINT_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Default)]
struct Shared {
    cancel: AtomicBool,
    files_done: AtomicU64,
    files_total: AtomicU64,
    bytes_done: AtomicU64,
    bytes_total: AtomicU64,
}

impl Shared {
    fn store(&self, progress: DumpProgress) {
        self.files_done
            .store(progress.files_done, Ordering::Relaxed);
        self.files_total
            .store(progress.files_total, Ordering::Relaxed);
        self.bytes_done
            .store(progress.bytes_done, Ordering::Relaxed);
        self.bytes_total
            .store(progress.bytes_total, Ordering::Relaxed);
    }

    fn load(&self) -> DumpProgress {
        DumpProgress {
            files_done: self.files_done.load(Ordering::Relaxed),
            files_total: self.files_total.load(Ordering::Relaxed),
            bytes_done: self.bytes_done.load(Ordering::Relaxed),
            bytes_total: self.bytes_total.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done(DumpSummary),
    Cancelled,
    Failed(String),
}

struct Finished {
    destination: Option<PathBuf>,
    result: Result<DumpSummary, DumpError>,
}

pub struct DumpJob {
    title: String,
    shared: Arc<Shared>,
    receiver: Receiver<Finished>,
    destination: Option<PathBuf>,
    outcome: Option<(Outcome, Instant)>,
    pub card: Option<egui::Rect>,
}

impl DumpJob {
    pub fn start(game: PathBuf, title: String, parent: PathBuf, ctx: &egui::Context) -> Self {
        let shared = Arc::new(Shared::default());
        let (sender, receiver) = mpsc::channel();
        let worker = shared.clone();
        let repaint = ctx.clone();
        let name = title.clone();
        std::thread::spawn(move || {
            let finished = run(&game, &name, &parent, &worker, &repaint);
            let _ = sender.send(finished);
            repaint.request_repaint();
        });
        Self {
            title,
            shared,
            receiver,
            destination: None,
            outcome: None,
            card: None,
        }
    }

    pub fn poll(&mut self) -> Option<&Outcome> {
        if self.outcome.is_some() {
            return None;
        }
        let finished = match self.receiver.try_recv() {
            Ok(finished) => finished,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Finished {
                destination: None,
                result: Err(DumpError::Failed(
                    "The dump stopped unexpectedly.".to_string(),
                )),
            },
        };
        self.destination = finished.destination;
        let outcome = match finished.result {
            Ok(summary) => Outcome::Done(summary),
            Err(DumpError::Cancelled) => Outcome::Cancelled,
            Err(DumpError::Failed(message)) => Outcome::Failed(message),
        };
        self.outcome = Some((outcome, Instant::now()));
        self.outcome.as_ref().map(|(outcome, _)| outcome)
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn destination(&self) -> Option<&Path> {
        self.destination.as_deref()
    }

    pub fn is_running(&self) -> bool {
        self.outcome.is_none()
    }

    pub fn outcome(&self) -> Option<(&Outcome, Duration)> {
        self.outcome
            .as_ref()
            .map(|(outcome, at)| (outcome, at.elapsed()))
    }

    pub fn progress(&self) -> DumpProgress {
        self.shared.load()
    }

    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelling(&self) -> bool {
        self.is_running() && self.shared.cancel.load(Ordering::Relaxed)
    }
}

fn run(game: &Path, title: &str, parent: &Path, shared: &Shared, ctx: &egui::Context) -> Finished {
    let loaded = Application::load_with_content(
        &game.to_string_lossy(),
        &nexium_common::paths::content_dir(),
    );
    let app = match loaded {
        Ok(app) => app,
        Err(message) => {
            log::error!("dump of {} failed to load: {}", game.display(), message);
            return Finished {
                destination: None,
                result: Err(DumpError::Failed(message)),
            };
        }
    };
    let destination = parent.join(format!(
        "{} [{:016X}]",
        dump::safe_name(title),
        app.title_id
    ));
    log::info!("dumping {} to {}", game.display(), destination.display());
    let mut last_repaint = Instant::now();
    let result = dump::dump_application(&app, &destination, &shared.cancel, |progress| {
        shared.store(progress);
        if last_repaint.elapsed() >= REPAINT_INTERVAL {
            last_repaint = Instant::now();
            ctx.request_repaint();
        }
    });
    match &result {
        Ok(summary) => log::info!("dump finished: {:?}", summary),
        Err(error) => log::warn!("dump of {} ended: {}", game.display(), error),
    }
    Finished {
        destination: Some(destination),
        result,
    }
}
