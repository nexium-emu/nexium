use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};

use eframe::egui::{self, Color32, RichText, Stroke};
use nexium_loader::content::{self, ContentId, ContentKind, GameContent, InstalledContent};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Updates,
    Dlc,
}

enum Action {
    Refresh,
    Install(Vec<PathBuf>),
    Update(Option<ContentId>),
    Dlc(ContentId, bool),
    Remove(ContentId),
}

enum Event {
    Loaded(GameContent),
    Progress(String, Option<f32>),
    Finished(String, Vec<String>),
}

pub struct ContentManager {
    game: PathBuf,
    title: String,
    cover: Option<egui::TextureHandle>,
    accent: Color32,
    tab: Tab,
    loaded: Option<GameContent>,
    pending: Option<Receiver<Event>>,
    progress: Option<f32>,
    status: String,
    errors: Vec<String>,
    query: String,
    removal: Option<InstalledContent>,
    selected: usize,
    held_buttons: u64,
}

impl ContentManager {
    pub fn new(
        game: PathBuf,
        title: String,
        cover: Option<egui::TextureHandle>,
        accent: Color32,
        ctx: &egui::Context,
    ) -> Self {
        let accent = if u16::from(accent.r()) + u16::from(accent.g()) + u16::from(accent.b()) < 240
        {
            Color32::from_rgb(47, 180, 239)
        } else {
            accent
        };
        let mut manager = Self {
            game,
            title,
            cover,
            accent,
            tab: Tab::Updates,
            loaded: None,
            pending: None,
            progress: None,
            status: String::new(),
            errors: Vec::new(),
            query: String::new(),
            removal: None,
            selected: 0,
            held_buttons: u64::MAX,
        };
        manager.start(Action::Refresh, ctx);
        manager
    }

    fn start(&mut self, action: Action, ctx: &egui::Context) {
        if self.pending.is_some() {
            return;
        }
        self.errors.clear();
        self.progress = None;
        self.status = match &action {
            Action::Refresh => "Reading installed content...",
            Action::Install(_) => "Checking package...",
            Action::Remove(_) => "Removing installed content...",
            _ => "Saving your selection...",
        }
        .into();
        let game = self.game.clone();
        let application_id = self.loaded.as_ref().map(|loaded| loaded.application_id);
        let ctx = ctx.clone();
        let (sender, receiver) = mpsc::channel();
        self.pending = Some(receiver);
        std::thread::spawn(move || {
            let send = |event| {
                let _ = sender.send(event);
                ctx.request_repaint();
            };
            let result = (|| {
                let root = nexium_common::paths::content_dir();
                let application_id = match application_id {
                    Some(id) => id,
                    None => nexium_loader::read_application_title_id(&game)?
                        .ok_or("This file has no game title ID for updates or DLC.")?,
                };
                let loaded = match action {
                    Action::Refresh => content::list_game_content(&root, application_id)?,
                    Action::Update(id) => content::select_update(&root, application_id, id)?,
                    Action::Dlc(id, enabled) => {
                        content::set_dlc_enabled(&root, application_id, id, enabled)?
                    }
                    Action::Remove(id) => content::remove_content(&root, application_id, id)?,
                    Action::Install(paths) => {
                        let mut errors = Vec::new();
                        let mut installed = 0;
                        for (index, path) in paths.iter().enumerate() {
                            let name = path.file_name().unwrap_or_default().to_string_lossy();
                            let label = format!("{} of {}  ·  {name}", index + 1, paths.len());
                            send(Event::Progress(format!("Checking {label}"), None));
                            let mut last = std::time::Instant::now();
                            match content::install_package(
                                &root,
                                application_id,
                                path,
                                |progress| {
                                    if last.elapsed() >= std::time::Duration::from_millis(80)
                                        || progress.completed_bytes == progress.total_bytes
                                    {
                                        let fraction = (progress.total_bytes > 0).then(|| {
                                            (progress.completed_bytes as f64
                                                / progress.total_bytes as f64)
                                                .clamp(0.0, 1.0)
                                                as f32
                                        });
                                        send(Event::Progress(
                                            format!("Installing {label}"),
                                            fraction,
                                        ));
                                        last = std::time::Instant::now();
                                    }
                                },
                            ) {
                                Ok(loaded) => {
                                    installed += 1;
                                    send(Event::Loaded(loaded));
                                }
                                Err(error) => errors.push(format!("{name}: {error}")),
                            }
                        }
                        let message = match installed {
                            0 => "No new content installed.".into(),
                            1 => "Installed 1 package. Your content is ready for the next launch."
                                .into(),
                            count => {
                                format!("Installed {count} packages. Ready for the next launch.")
                            }
                        };
                        send(Event::Finished(message, errors));
                        return Ok::<(), String>(());
                    }
                };
                send(Event::Loaded(loaded));
                send(Event::Finished(String::new(), Vec::new()));
                Ok(())
            })();
            if let Err(error) = result {
                send(Event::Finished(String::new(), vec![error]));
            }
        });
    }

    fn poll(&mut self) {
        let mut finished = false;
        if let Some(receiver) = &self.pending {
            loop {
                match receiver.try_recv() {
                    Ok(Event::Loaded(loaded)) => self.loaded = Some(loaded),
                    Ok(Event::Progress(label, fraction)) => {
                        self.status = label;
                        self.progress = fraction;
                    }
                    Ok(Event::Finished(message, errors)) => {
                        self.status = message;
                        self.errors = errors;
                        self.progress = None;
                        finished = true;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.errors =
                            vec!["The content task stopped unexpectedly. Please try again.".into()];
                        self.status.clear();
                        finished = true;
                        break;
                    }
                }
            }
        }
        if finished {
            self.pending = None;
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        input: &crate::input::InputSnapshot,
        game_running: bool,
    ) -> bool {
        use crate::controller_config::SwitchButton;
        self.poll();
        let pressed = input.buttons & !self.held_buttons;
        self.held_buttons = input.buttons;
        let edge = |button: SwitchButton| pressed & button.npad_bit() != 0;
        let busy = self.pending.is_some();
        let editable = !busy && !game_running && self.loaded.is_some();
        let mut close = edge(SwitchButton::B) && !busy && self.removal.is_none();
        if !busy
            && self.removal.is_some()
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.removal = None;
        }
        if edge(SwitchButton::B) && !busy {
            self.removal = None;
        }
        if self.removal.is_none() && (edge(SwitchButton::L) || edge(SwitchButton::R)) {
            self.tab = if self.tab == Tab::Updates {
                Tab::Dlc
            } else {
                Tab::Updates
            };
            self.selected = 0;
        }
        let mut pick_files = edge(SwitchButton::X) && editable && self.removal.is_none();
        let mut action = None;
        let entries = self
            .loaded
            .as_ref()
            .map(|loaded| loaded.entries.as_slice())
            .unwrap_or(&[]);
        let updates = entries
            .iter()
            .filter(|entry| entry.kind == ContentKind::Update)
            .count();
        let dlcs = entries
            .iter()
            .filter(|entry| entry.kind == ContentKind::Dlc)
            .count();
        let enabled_dlcs = entries
            .iter()
            .filter(|entry| entry.kind == ContentKind::Dlc && entry.enabled)
            .count();
        let active_update = entries
            .iter()
            .find(|entry| entry.kind == ContentKind::Update && entry.enabled)
            .map(version_label)
            .unwrap_or_else(|| "Base game".into());
        let accent = self.accent;
        let frame = egui::Frame::popup(&ctx.global_style())
            .corner_radius(18)
            .inner_margin(22)
            .stroke(Stroke::new(1.0, accent.gamma_multiply(0.45)));
        let modal = egui::Modal::new(egui::Id::new("game_content_manager"))
            .frame(frame)
            .backdrop_color(Color32::from_black_alpha(175))
            .show(ctx, |ui| {
                ui.set_width((ctx.viewport_rect().width() - 64.0).clamp(280.0, 720.0));
                ui.spacing_mut().item_spacing = egui::vec2(10.0, 10.0);
                ui.horizontal(|ui| {
                    let cover_size = egui::vec2(72.0, 72.0);
                    if let Some(cover) = &self.cover {
                        ui.add(egui::Image::new((cover.id(), cover_size)).corner_radius(12));
                    } else {
                        let (rect, _) = ui.allocate_exact_size(cover_size, egui::Sense::hover());
                        ui.painter().rect_filled(rect, 12, accent.gamma_multiply(0.2));
                        ui.painter().rect_stroke(rect.shrink(17.0), 7, Stroke::new(2.0, accent), egui::StrokeKind::Inside);
                        ui.painter().line_segment([rect.center_top() + egui::vec2(0.0, 23.0), rect.center_bottom() - egui::vec2(0.0, 23.0)], Stroke::new(2.0, accent));
                    }
                    ui.add_space(4.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("UPDATES & DLC").size(11.0).strong().color(accent));
                        ui.label(RichText::new(&self.title).size(23.0).strong());
                        ui.weak(format!("{active_update}  ·  {enabled_dlcs} DLC enabled"));
                    });
                });
                ui.add_space(3.0);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.tab == Tab::Updates, format!("Updates  {updates}")).clicked() {
                        self.tab = Tab::Updates;
                        self.selected = 0;
                    }
                    if ui.selectable_label(self.tab == Tab::Dlc, format!("DLC  {dlcs}")).clicked() {
                        self.tab = Tab::Dlc;
                        self.selected = 0;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(editable && self.removal.is_none(), egui::Button::new(RichText::new("Install files...").strong()).fill(accent.gamma_multiply(0.25))).clicked() {
                            pick_files = true;
                        }
                    });
                });
                if game_running {
                    notice(ui, "Close the running game to install or change its content.", Color32::from_rgb(231, 177, 87));
                } else {
                    ui.weak("Add decrypted .dnsp or .nsp files. Changes apply on the next launch.");
                }
                if self.tab == Tab::Dlc && dlcs > 5 {
                    ui.add(egui::TextEdit::singleline(&mut self.query).hint_text("Find DLC...").desired_width(f32::INFINITY));
                }
                let rows = visible_entries(self.loaded.as_ref(), self.tab, &self.query);
                let base_row = usize::from(self.tab == Tab::Updates && self.loaded.is_some());
                let row_count = rows.len() + base_row;
                if edge(SwitchButton::DUp) { self.selected = self.selected.saturating_sub(1); }
                if edge(SwitchButton::DDown) { self.selected = self.selected.saturating_add(1); }
                self.selected = self.selected.min(row_count.saturating_sub(1));
                let navigating = edge(SwitchButton::DUp) || edge(SwitchButton::DDown);
                let row_editable = editable && self.removal.is_none();
                egui::ScrollArea::vertical()
                    .id_salt("installed_content_rows")
                    .min_scrolled_height(120.0)
                    .max_height((ctx.viewport_rect().height() - 395.0).clamp(150.0, 340.0))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if base_row != 0 {
                            let active = !self.loaded.as_ref().unwrap().entries.iter().any(|entry| entry.kind == ContentKind::Update && entry.enabled);
                            let selected = self.selected == 0 && input.connected;
                            let row = row_frame(ui, active, selected, accent).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.add_enabled(row_editable, egui::RadioButton::new(active, "Base game")).clicked()
                                        || (row_editable && selected && edge(SwitchButton::A)) {
                                        action = Some(Action::Update(None));
                                    }
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if active { badge(ui, "ACTIVE", accent); }
                                    });
                                });
                                ui.weak("Launch without an installed update.");
                            });
                            if selected && navigating { row.response.scroll_to_me(Some(egui::Align::Center)); }
                        }
                        for (index, entry) in rows.iter().enumerate() {
                            let selected = self.selected == index + base_row && input.connected;
                            ui.push_id((entry.id.title_id, entry.id.version), |ui| {
                                let row = row_frame(ui, entry.enabled, selected, accent).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let mut enabled = entry.enabled;
                                        let label = if entry.kind == ContentKind::Update { version_label(entry) } else { content_name(entry) };
                                        let clicked = if entry.kind == ContentKind::Update {
                                            ui.add_enabled(row_editable, egui::RadioButton::new(enabled, RichText::new(&label).strong())).clicked()
                                        } else {
                                            ui.add_enabled(row_editable, egui::Checkbox::new(&mut enabled, RichText::new(&label).strong())).changed()
                                        };
                                        let controller_toggle = row_editable && selected && edge(SwitchButton::A);
                                        if clicked || controller_toggle {
                                            action = Some(if entry.kind == ContentKind::Update {
                                                Action::Update(Some(entry.id))
                                            } else {
                                                Action::Dlc(entry.id, if controller_toggle { !entry.enabled } else { enabled })
                                            });
                                        }
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            if ui.add_enabled(row_editable, egui::Button::new("Remove").small()).clicked() {
                                                self.removal = Some(entry.clone());
                                            }
                                            if entry.enabled { badge(ui, if entry.kind == ContentKind::Update { "ACTIVE" } else { "ENABLED" }, accent); }
                                        });
                                    });
                                    ui.horizontal(|ui| {
                                        ui.weak(format_size(entry.size_bytes));
                                        if entry.kind == ContentKind::Dlc { ui.weak(version_label(entry)); }
                                        if entry.kind == ContentKind::Update && !entry.name.trim().is_empty() {
                                            ui.weak(&entry.name);
                                        }
                                    });
                                    ui.collapsing("Details", |ui| {
                                        ui.monospace(format!("Title ID  {:016X}", entry.id.title_id));
                                        ui.weak(format!("Content version {}", entry.id.version));
                                        ui.label(entry.path.to_string_lossy().as_ref());
                                    });
                                });
                                if selected && navigating { row.response.scroll_to_me(Some(egui::Align::Center)); }
                            });
                        }
                        if rows.is_empty() && !busy {
                            ui.add_space(14.0);
                            ui.vertical_centered(|ui| {
                                ui.label(RichText::new(if !self.query.is_empty() && self.tab == Tab::Dlc {
                                    "No matching DLC"
                                } else if self.tab == Tab::Updates { "Your base game is ready" } else { "More to play, all in one place" }).size(17.0).strong());
                                ui.weak(if self.tab == Tab::Updates { "Drop an update here, or choose Install files." } else { "Drop DLC here, or choose Install files." });
                            });
                            ui.add_space(14.0);
                        }
                    });
                if let Some(entry) = self.removal.clone() {
                    egui::Frame::new().fill(ui.visuals().faint_bg_color).inner_margin(12).corner_radius(10).show(ui, |ui| {
                        ui.label(RichText::new(format!("Remove {}?", if entry.kind == ContentKind::Update { version_label(&entry) } else { content_name(&entry) })).strong());
                        ui.weak("Removes the installed copy from NeXium. Your original file stays where it is.");
                        ui.horizontal(|ui| {
                            if ui.add_enabled(editable, egui::Button::new("Remove installed copy")).clicked() {
                                action = Some(Action::Remove(entry.id));
                                self.removal = None;
                            }
                            if ui.button("Keep it").clicked() { self.removal = None; }
                        });
                    });
                }
                if busy {
                    ui.horizontal(|ui| { ui.spinner(); ui.label(&self.status); });
                    if let Some(progress) = self.progress { ui.add(egui::ProgressBar::new(progress).show_percentage().fill(accent)); }
                } else if !self.status.is_empty() {
                    notice(ui, &self.status, accent);
                }
                if !self.errors.is_empty() {
                    let error_color = Color32::from_rgb(240, 119, 119);
                    egui::ScrollArea::vertical().id_salt("content_errors").max_height(90.0).show(ui, |ui| {
                        for error in &self.errors { ui.colored_label(error_color, error); }
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if input.connected { ui.weak("L / R tabs  ·  A select  ·  X install  ·  B done"); }
                    else if let Some(loaded) = &self.loaded { ui.weak(format!("{:016X}", loaded.application_id)); }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(!busy, egui::Button::new("Done").min_size(egui::vec2(76.0, 32.0))).clicked() { close = true; }
                        if ui.add_enabled(!busy, egui::Button::new("Refresh")).clicked() { action = Some(Action::Refresh); }
                    });
                });
            });
        let dropped: Vec<_> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .collect()
        });
        if !dropped.is_empty() {
            if editable && self.removal.is_none() {
                action = Some(Action::Install(dropped));
            } else if !busy {
                self.errors = vec!["Close the running game before installing content.".into()];
            }
        }
        if pick_files {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title("Install updates and DLC")
                .add_filter("Decrypted game content", &["dnsp", "nsp"])
                .pick_files()
            {
                action = Some(Action::Install(paths));
            }
        }
        if let Some(action) = action {
            self.start(action, ctx);
            close = false;
        }
        if self.pending.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.pending.is_some() || !(close || (modal.should_close() && self.removal.is_none()))
    }
}

fn version_label(entry: &InstalledContent) -> String {
    if entry.display_version.trim().is_empty() {
        format!("Version {}", entry.id.version)
    } else {
        format!("Version {}", entry.display_version.trim())
    }
}

fn content_name(entry: &InstalledContent) -> String {
    if entry.name.trim().is_empty() {
        format!("DLC {:016X}", entry.id.title_id)
    } else {
        entry.name.clone()
    }
}

fn visible_entries(loaded: Option<&GameContent>, tab: Tab, query: &str) -> Vec<InstalledContent> {
    let query = query.trim().to_lowercase();
    let kind = if tab == Tab::Updates {
        ContentKind::Update
    } else {
        ContentKind::Dlc
    };
    let mut entries: Vec<_> = loaded
        .into_iter()
        .flat_map(|game| &game.entries)
        .filter(|entry| {
            entry.kind == kind
                && (tab == Tab::Updates
                    || query.is_empty()
                    || content_name(entry).to_lowercase().contains(&query)
                    || format!("{:016x}", entry.id.title_id).contains(&query))
        })
        .cloned()
        .collect();
    entries.sort_by(|a, b| {
        if tab == Tab::Updates {
            b.id.version.cmp(&a.id.version)
        } else {
            content_name(a)
                .to_lowercase()
                .cmp(&content_name(b).to_lowercase())
                .then_with(|| b.id.version.cmp(&a.id.version))
        }
    });
    entries
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

fn row_frame(ui: &egui::Ui, active: bool, selected: bool, accent: Color32) -> egui::Frame {
    egui::Frame::new()
        .inner_margin(12)
        .corner_radius(10)
        .fill(if active {
            accent.gamma_multiply(0.08)
        } else {
            ui.visuals().faint_bg_color
        })
        .stroke(Stroke::new(
            if selected { 1.5 } else { 1.0 },
            if selected {
                accent
            } else if active {
                accent.gamma_multiply(0.35)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke.color
            },
        ))
}

fn badge(ui: &mut egui::Ui, label: &str, accent: Color32) {
    egui::Frame::new()
        .fill(accent.gamma_multiply(0.15))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(7, 3))
        .show(ui, |ui| {
            ui.label(RichText::new(label).size(10.0).strong().color(accent));
        });
}

fn notice(ui: &mut egui::Ui, message: &str, color: Color32) {
    ui.colored_label(color, message);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(title_id: u64, version: u32, kind: ContentKind, name: &str) -> InstalledContent {
        InstalledContent {
            id: ContentId { title_id, version },
            application_id: 0x0100_0000_0000_0000,
            kind,
            name: name.into(),
            display_version: String::new(),
            path: PathBuf::from("installed/package.dnsp"),
            size_bytes: 4096,
            enabled: true,
        }
    }

    fn manager(receiver: Receiver<Event>) -> ContentManager {
        ContentManager {
            game: PathBuf::from("game.dnsp"),
            title: "Game".into(),
            cover: None,
            accent: Color32::LIGHT_BLUE,
            tab: Tab::Updates,
            loaded: None,
            pending: Some(receiver),
            progress: None,
            status: String::new(),
            errors: Vec::new(),
            query: String::new(),
            removal: None,
            selected: 0,
            held_buttons: 0,
        }
    }

    #[test]
    fn content_rows_use_metadata_versions_and_keep_dlc_identity() {
        let mut older = entry(0x1000, 1, ContentKind::Update, "older");
        older.display_version = "9.0".into();
        let mut newer = entry(0x1000, 2, ContentKind::Update, "newer");
        newer.display_version = "1.0".into();
        let game = GameContent {
            application_id: 0x0100_0000_0000_0000,
            entries: vec![
                older,
                entry(0x2002, 3, ContentKind::Dlc, "Adventure Pack"),
                newer,
                entry(0x2001, 1, ContentKind::Dlc, "Adventure Pack"),
            ],
        };
        let updates = visible_entries(Some(&game), Tab::Updates, "adventure");
        assert_eq!(
            updates
                .iter()
                .map(|entry| entry.id.version)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        let dlc = visible_entries(Some(&game), Tab::Dlc, " ADVENTURE ");
        assert_eq!(dlc.len(), 2);
        assert_ne!(dlc[0].id, dlc[1].id);
        let exact = visible_entries(Some(&game), Tab::Dlc, "2001");
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].id.title_id, 0x2001);
    }

    #[test]
    fn batch_error_keeps_successful_install_and_releases_busy_state() {
        let (sender, receiver) = mpsc::channel();
        let mut manager = manager(receiver);
        sender
            .send(Event::Progress("Copying".into(), Some(0.5)))
            .unwrap();
        sender
            .send(Event::Loaded(GameContent {
                application_id: 0x0100_0000_0000_0000,
                entries: vec![entry(0x1000, 2, ContentKind::Update, "Update")],
            }))
            .unwrap();
        sender
            .send(Event::Finished(
                "Installed 1 package.".into(),
                vec!["Wrong game".into()],
            ))
            .unwrap();
        drop(sender);
        manager.poll();
        assert!(manager.pending.is_none());
        assert!(manager.progress.is_none());
        assert_eq!(manager.loaded.as_ref().unwrap().entries[0].id.version, 2);
        assert_eq!(manager.errors, ["Wrong game"]);
        assert_eq!(manager.status, "Installed 1 package.");
    }

    #[test]
    fn lost_worker_does_not_leave_manager_permanently_busy() {
        let (sender, receiver) = mpsc::channel();
        let mut manager = manager(receiver);
        drop(sender);
        manager.poll();
        assert!(manager.pending.is_none());
        assert_eq!(manager.errors.len(), 1);
        assert!(manager.errors[0].contains("stopped unexpectedly"));
    }
}
