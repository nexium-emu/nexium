use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText, Stroke};
use nexium_loader::firmware::{self, FirmwarePackage, FirmwareScan, InstalledFirmware};

const WARNING: Color32 = Color32::from_rgb(231, 177, 87);
const ERROR: Color32 = Color32::from_rgb(240, 119, 119);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Then {
    Boot,
    Select,
    Nothing,
}

pub enum Outcome {
    Open,
    Done(Then, PathBuf),
}

enum Event {
    Scanned(Result<(FirmwareScan, Option<InstalledFirmware>), String>),
    Progress(f32),
    Installed(Result<InstalledFirmware, String>),
    Removed(Result<Option<InstalledFirmware>, String>),
}

enum Stage {
    Scanning,
    Ask { scan: FirmwareScan, installed: Option<InstalledFirmware>, choice: usize },
    Installing { package: FirmwarePackage, progress: Option<f32> },
    ConfirmRemove { installed: InstalledFirmware },
    Removing { display_version: String },
    Report { title: String, lines: Vec<String>, error: bool },
}

pub struct FirmwarePrompt {
    path: PathBuf,
    then: Then,
    accent: Color32,
    stage: Stage,
    pending: Option<Receiver<Event>>,
    started: Instant,
    held_buttons: u64,
}

pub fn may_contain_firmware(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dxci") || extension.eq_ignore_ascii_case("dnsp"))
}

impl FirmwarePrompt {
    pub fn new(path: PathBuf, then: Then, accent: Color32, ctx: &egui::Context) -> Self {
        let (sender, receiver) = mpsc::channel();
        let source = path.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let scanned = firmware::scan(&source).map(|scan| {
                let installed = firmware::installed_firmware(&nexium_common::paths::firmware_dir()).ok().flatten();
                (scan, installed)
            });
            let _ = sender.send(Event::Scanned(scanned));
            ctx.request_repaint();
        });
        Self {
            path,
            then,
            accent,
            stage: Stage::Scanning,
            pending: Some(receiver),
            started: Instant::now(),
            held_buttons: u64::MAX,
        }
    }

    pub fn remove(installed: InstalledFirmware, accent: Color32) -> Self {
        Self {
            path: PathBuf::new(),
            then: Then::Nothing,
            accent,
            stage: Stage::ConfirmRemove { installed },
            pending: None,
            started: Instant::now(),
            held_buttons: u64::MAX,
        }
    }

    fn start_removal(&mut self, display_version: String, ctx: &egui::Context) {
        let (sender, receiver) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = sender.send(Event::Removed(firmware::remove_firmware(&nexium_common::paths::firmware_dir())));
            ctx.request_repaint();
        });
        self.pending = Some(receiver);
        self.stage = Stage::Removing { display_version };
    }

    fn poll(&mut self) -> Option<Outcome> {
        let receiver = self.pending.take()?;
        loop {
            match receiver.try_recv() {
                Ok(Event::Progress(fraction)) => {
                    if let Stage::Installing { progress, .. } = &mut self.stage {
                        *progress = Some(fraction);
                    }
                }
                Ok(Event::Scanned(result)) => return self.scanned(result),
                Ok(Event::Installed(result)) => {
                    self.stage = match result {
                        Ok(installed) => Stage::Report {
                            title: format!("Firmware {} installed", installed.display_version),
                            lines: vec![
                                format!("{} files  ·  {}", installed.files.len(), format_size(installed.size_bytes)),
                                "Games use it from their next launch.".into(),
                            ],
                            error: false,
                        },
                        Err(error) => Stage::Report {
                            title: "Firmware wasn't installed".into(),
                            lines: vec![error, "Your installed firmware was not changed.".into()],
                            error: true,
                        },
                    };
                    return None;
                }
                Ok(Event::Removed(result)) => {
                    self.stage = match result {
                        Ok(Some(removed)) => Stage::Report {
                            title: format!("Firmware {} removed", removed.display_version),
                            lines: vec![
                                "Games no longer use it.".into(),
                                "The file you installed it from was not touched.".into(),
                            ],
                            error: false,
                        },
                        Ok(None) => Stage::Report {
                            title: "No firmware is installed".into(),
                            lines: vec!["There was nothing to remove.".into()],
                            error: false,
                        },
                        Err(error) => Stage::Report {
                            title: "Firmware wasn't removed".into(),
                            lines: vec![error],
                            error: true,
                        },
                    };
                    return None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    self.pending = Some(receiver);
                    return None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.stage = Stage::Report {
                        title: "The firmware task stopped unexpectedly".into(),
                        lines: vec!["Please try again.".into()],
                        error: true,
                    };
                    return None;
                }
            }
        }
    }

    fn scanned(&mut self, result: Result<(FirmwareScan, Option<InstalledFirmware>), String>) -> Option<Outcome> {
        let (scan, installed) = match result {
            Ok(found) => found,
            Err(error) => {
                self.stage = Stage::Report { title: "Couldn't check this file for firmware".into(), lines: vec![error], error: true };
                return None;
            }
        };
        if !scan.game && !scan.packages.is_empty() {
            self.then = Then::Nothing;
        }
        let newest = scan.packages.first().map(|package| package.version);
        let current = installed.as_ref().map(|installed| installed.version);
        let nothing_new = newest.is_none_or(|newest| current.is_some_and(|current| current >= newest));
        if self.then != Then::Nothing && scan.problems.is_empty() && nothing_new {
            return Some(Outcome::Done(self.then, self.path.clone()));
        }
        if scan.packages.is_empty() {
            let name = self.path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            self.stage = if scan.problems.is_empty() {
                let line = if scan.game {
                    format!("{name} only contains a game. There is no firmware in it to install.")
                } else {
                    format!("{name} doesn't contain any firmware.")
                };
                Stage::Report { title: "No firmware found".into(), lines: vec![line], error: false }
            } else {
                Stage::Report { title: "This file's firmware can't be installed".into(), lines: scan.problems, error: true }
            };
            return None;
        }
        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
        self.stage = Stage::Ask { scan, installed, choice: 0 };
        None
    }

    fn install(&mut self, package: FirmwarePackage, ctx: &egui::Context) {
        let (sender, receiver) = mpsc::channel();
        let source = self.path.clone();
        let version = package.version;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut last = Instant::now();
            let result = firmware::install_firmware(&nexium_common::paths::firmware_dir(), &source, version, |progress| {
                if last.elapsed() >= Duration::from_millis(80) || progress.completed_bytes == progress.total_bytes {
                    let fraction = if progress.total_bytes == 0 {
                        1.0
                    } else {
                        (progress.completed_bytes as f64 / progress.total_bytes as f64).clamp(0.0, 1.0) as f32
                    };
                    let _ = sender.send(Event::Progress(fraction));
                    ctx.request_repaint();
                    last = Instant::now();
                }
            });
            let _ = sender.send(Event::Installed(result));
            ctx.request_repaint();
        });
        self.pending = Some(receiver);
        self.stage = Stage::Installing { package, progress: None };
    }

    pub fn show(&mut self, ctx: &egui::Context, input: &crate::input::InputSnapshot, game_running: bool) -> Outcome {
        use crate::controller_config::SwitchButton;
        if let Some(outcome) = self.poll() {
            return outcome;
        }
        if self.pending.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        if matches!(self.stage, Stage::Scanning) && self.started.elapsed() < Duration::from_millis(300) {
            return Outcome::Open;
        }
        let pressed = input.buttons & !self.held_buttons;
        self.held_buttons = input.buttons;
        let edge = |button: SwitchButton| pressed & button.npad_bit() != 0;
        let accent = self.accent;
        let then = self.then;
        let path = &self.path;
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let mut outcome = None;
        let mut install = None;
        let mut remove = None;
        let frame = egui::Frame::popup(&ctx.global_style())
            .corner_radius(18)
            .inner_margin(22)
            .stroke(Stroke::new(1.0, accent.gamma_multiply(0.45)));
        let modal = egui::Modal::new(egui::Id::new("firmware_prompt"))
            .frame(frame)
            .backdrop_color(Color32::from_black_alpha(175))
            .show(ctx, |ui| {
                ui.set_width((ctx.viewport_rect().width() - 64.0).clamp(280.0, 560.0));
                ui.spacing_mut().item_spacing = egui::vec2(10.0, 10.0);
                match &mut self.stage {
                    Stage::Scanning => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Checking {name} for firmware..."));
                        });
                    }
                    Stage::Ask { scan, installed, choice } => {
                        ui.label(RichText::new("FIRMWARE FOUND").size(11.0).strong().color(accent));
                        ui.label(RichText::new(if scan.game {
                            "This .dxci contains firmware. Do you want to install the firmware as well?"
                        } else {
                            "This package contains firmware. Do you want to install it?"
                        }).size(18.0).strong());
                        ui.weak(&name);
                        if scan.packages.len() > 1 {
                            ui.label("It contains more than one firmware version. Choose which one to install:");
                            if edge(SwitchButton::DUp) { *choice = choice.saturating_sub(1); }
                            if edge(SwitchButton::DDown) { *choice = (*choice + 1).min(scan.packages.len() - 1); }
                            for (index, package) in scan.packages.iter().enumerate() {
                                if ui.radio(*choice == index, package_label(package)).clicked() {
                                    *choice = index;
                                }
                            }
                        } else {
                            let package = &scan.packages[0];
                            ui.label(RichText::new(format!("Firmware {}", package.display_version)).size(22.0).strong().color(accent));
                            ui.weak(format!("{} system titles  ·  {} files  ·  {}", package.title_count, package.nca_count, format_size(package.size_bytes)));
                        }
                        let package = &scan.packages[*choice];
                        let (note, downgrade) = installed_note(installed.as_ref(), package);
                        if downgrade { ui.colored_label(WARNING, note); } else { ui.weak(note); }
                        for problem in &scan.problems {
                            ui.colored_label(ERROR, problem);
                        }
                        if game_running {
                            ui.colored_label(WARNING, "Close the running game to install firmware.");
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            if input.connected {
                                ui.weak(if scan.packages.len() > 1 { "Up / Down choose  ·  A install  ·  B skip" } else { "A install  ·  B skip" });
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let skip = if then == Then::Nothing { "Cancel" } else { "Skip firmware" };
                                if ui.button(skip).clicked() || edge(SwitchButton::B) {
                                    outcome = Some(Outcome::Done(then, path.clone()));
                                }
                                let button = egui::Button::new(RichText::new("Install firmware").strong()).fill(accent.gamma_multiply(0.25));
                                if ui.add_enabled(!game_running, button).clicked() || (!game_running && edge(SwitchButton::A)) {
                                    install = Some(package.clone());
                                }
                            });
                        });
                    }
                    Stage::Installing { package, progress } => {
                        ui.label(RichText::new("INSTALLING FIRMWARE").size(11.0).strong().color(accent));
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Installing firmware {} from {name}...", package.display_version));
                        });
                        if let Some(progress) = progress {
                            ui.add(egui::ProgressBar::new(*progress).show_percentage().fill(accent));
                        }
                        ui.weak("Keep NeXium open until this finishes.");
                    }
                    Stage::ConfirmRemove { installed } => {
                        ui.label(RichText::new("REMOVE FIRMWARE").size(11.0).strong().color(accent));
                        ui.label(RichText::new(format!("Remove firmware {}?", installed.display_version)).size(18.0).strong());
                        ui.weak(format!("{} files  ·  {}  ·  installed from {}", installed.files.len(), format_size(installed.size_bytes), installed.source));
                        ui.weak("Games stop using its system archives, such as fonts. The file you installed it from is not affected.");
                        if game_running {
                            ui.colored_label(WARNING, "Close the running game to remove firmware.");
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            if input.connected {
                                ui.weak("A remove  ·  B keep");
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Keep it").clicked() || edge(SwitchButton::B) {
                                    outcome = Some(Outcome::Done(Then::Nothing, path.clone()));
                                }
                                let button = egui::Button::new(RichText::new("Remove firmware").strong()).fill(ERROR.gamma_multiply(0.25));
                                if ui.add_enabled(!game_running, button).clicked() || (!game_running && edge(SwitchButton::A)) {
                                    remove = Some(installed.display_version.clone());
                                }
                            });
                        });
                    }
                    Stage::Removing { display_version } => {
                        ui.label(RichText::new("REMOVING FIRMWARE").size(11.0).strong().color(accent));
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Removing firmware {display_version}..."));
                        });
                    }
                    Stage::Report { title, lines, error } => {
                        ui.label(RichText::new(title.as_str()).size(18.0).strong().color(if *error { ERROR } else { accent }));
                        for line in lines.iter() {
                            ui.label(line.as_str());
                        }
                        ui.separator();
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let primary = match then {
                                Then::Boot => "Continue to game",
                                Then::Select => "Continue",
                                Then::Nothing => "Done",
                            };
                            if ui.button(primary).clicked() || edge(SwitchButton::A) {
                                outcome = Some(Outcome::Done(then, path.clone()));
                            }
                            if then != Then::Nothing && (ui.button("Cancel").clicked() || edge(SwitchButton::B)) {
                                outcome = Some(Outcome::Done(Then::Nothing, path.clone()));
                            }
                            if then == Then::Nothing && edge(SwitchButton::B) {
                                outcome = Some(Outcome::Done(Then::Nothing, path.clone()));
                            }
                        });
                    }
                }
            });
        let busy = matches!(self.stage, Stage::Installing { .. } | Stage::Removing { .. });
        if outcome.is_none() && modal.should_close() && !busy {
            outcome = Some(Outcome::Done(Then::Nothing, self.path.clone()));
        }
        if let Some(package) = install {
            self.install(package, ctx);
        }
        if let Some(display_version) = remove {
            self.start_removal(display_version, ctx);
        }
        outcome.unwrap_or(Outcome::Open)
    }
}

fn installed_note(installed: Option<&InstalledFirmware>, package: &FirmwarePackage) -> (String, bool) {
    match installed {
        None => ("No firmware is installed yet.".into(), false),
        Some(current) if current.version < package.version => {
            (format!("Replaces your installed firmware {}.", current.display_version), false)
        }
        Some(current) if current.version == package.version => {
            (format!("Firmware {} is already installed. Installing it again replaces it.", current.display_version), false)
        }
        Some(current) => {
            (format!("Your installed firmware {} is newer. Installing this one downgrades it.", current.display_version), true)
        }
    }
}

fn package_label(package: &FirmwarePackage) -> String {
    format!("Firmware {}  ·  {} system titles  ·  {}", package.display_version, package.title_count, format_size(package.size_bytes))
}

fn format_size(bytes: u64) -> String {
    let (size, unit) = if bytes >= 1 << 30 {
        (bytes as f64 / (1u64 << 30) as f64, "GB")
    } else if bytes >= 1 << 20 {
        (bytes as f64 / (1u64 << 20) as f64, "MB")
    } else {
        (bytes as f64 / 1024.0, "KB")
    };
    format!("{size:.1} {unit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(version: u32) -> FirmwarePackage {
        FirmwarePackage {
            version,
            display_version: firmware::version_name(version),
            title_count: 2,
            nca_count: 5,
            size_bytes: 3 << 20,
        }
    }

    fn installed(version: u32) -> InstalledFirmware {
        InstalledFirmware {
            version,
            display_version: firmware::version_name(version),
            directory: "installed".into(),
            files: Vec::new(),
            size_bytes: 0,
            source: String::new(),
        }
    }

    fn prompt(then: Then) -> FirmwarePrompt {
        FirmwarePrompt {
            path: PathBuf::from("game.dxci"),
            then,
            accent: Color32::LIGHT_BLUE,
            stage: Stage::Scanning,
            pending: None,
            started: Instant::now(),
            held_buttons: 0,
        }
    }

    fn scan(game: bool, versions: &[u32], problems: &[&str]) -> FirmwareScan {
        FirmwareScan {
            game,
            packages: versions.iter().map(|&version| package(version)).collect(),
            problems: problems.iter().map(|problem| problem.to_string()).collect(),
        }
    }

    #[test]
    fn game_flow_continues_untouched_when_there_is_nothing_new_to_install() {
        let mut game_only = prompt(Then::Boot);
        assert!(matches!(game_only.scanned(Ok((scan(true, &[], &[]), None))), Some(Outcome::Done(Then::Boot, _))));
        let mut update_package = prompt(Then::Boot);
        assert!(matches!(update_package.scanned(Ok((scan(false, &[], &[]), None))), Some(Outcome::Done(Then::Boot, _))));
        let mut already_installed = prompt(Then::Select);
        let outcome = already_installed.scanned(Ok((scan(true, &[17 << 26], &[]), Some(installed(17 << 26)))));
        assert!(matches!(outcome, Some(Outcome::Done(Then::Select, _))));
    }

    #[test]
    fn newer_firmware_asks_before_the_game_continues() {
        let mut newer = prompt(Then::Boot);
        assert!(newer.scanned(Ok((scan(true, &[17 << 26, 16 << 26], &[]), Some(installed(16 << 26))))).is_none());
        assert!(matches!(newer.stage, Stage::Ask { choice: 0, .. }));
        let mut broken = prompt(Then::Boot);
        assert!(broken.scanned(Ok((scan(true, &[], &["incomplete"]), None))).is_none());
        assert!(matches!(&broken.stage, Stage::Report { error: true, lines, .. } if lines == &["incomplete"]));
    }

    #[test]
    fn firmware_packages_never_continue_into_a_game_launch() {
        let mut package = prompt(Then::Boot);
        assert!(package.scanned(Ok((scan(false, &[17 << 26], &[]), Some(installed(17 << 26))))).is_none());
        assert_eq!(package.then, Then::Nothing);
        assert!(matches!(package.stage, Stage::Ask { .. }));
        let mut dropped = prompt(Then::Nothing);
        assert!(dropped.scanned(Ok((scan(true, &[], &[]), None))).is_none());
        assert!(matches!(&dropped.stage, Stage::Report { error: false, title, lines } if title == "No firmware found" && lines[0].contains("only contains a game")));
    }

    #[test]
    fn installed_note_flags_downgrades() {
        assert!(!installed_note(None, &package(17 << 26)).1);
        assert!(!installed_note(Some(&installed(16 << 26)), &package(17 << 26)).1);
        assert!(installed_note(Some(&installed(17 << 26)), &package(17 << 26)).0.contains("already installed"));
        assert!(installed_note(Some(&installed(18 << 26)), &package(17 << 26)).1);
    }

    #[test]
    fn removal_asks_first_and_reports_the_result() {
        let mut prompt = FirmwarePrompt::remove(installed(17 << 26), Color32::LIGHT_BLUE);
        assert!(matches!(&prompt.stage, Stage::ConfirmRemove { installed } if installed.display_version == "17.0.0"));
        assert!(prompt.pending.is_none());
        let (sender, receiver) = mpsc::channel();
        prompt.pending = Some(receiver);
        prompt.stage = Stage::Removing { display_version: "17.0.0".into() };
        sender.send(Event::Removed(Ok(Some(installed(17 << 26))))).unwrap();
        assert!(prompt.poll().is_none());
        assert!(matches!(&prompt.stage, Stage::Report { error: false, title, .. } if title == "Firmware 17.0.0 removed"));
        let (sender, receiver) = mpsc::channel();
        prompt.pending = Some(receiver);
        sender.send(Event::Removed(Err("in use".into()))).unwrap();
        assert!(prompt.poll().is_none());
        assert!(matches!(&prompt.stage, Stage::Report { error: true, lines, .. } if lines == &["in use"]));
    }

    #[test]
    fn lost_worker_reports_an_error_instead_of_hanging() {
        let (sender, receiver) = mpsc::channel::<Event>();
        let mut prompt = prompt(Then::Boot);
        prompt.pending = Some(receiver);
        drop(sender);
        assert!(prompt.poll().is_none());
        assert!(prompt.pending.is_none());
        assert!(matches!(prompt.stage, Stage::Report { error: true, .. }));
    }
}
