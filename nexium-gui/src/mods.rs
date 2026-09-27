use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

use eframe::egui;
use nexium_loader::mods::{discover_mods, set_mod_enabled, ModEntry};

struct GameMods {
    title_id: u64,
    directory: PathBuf,
    entries: Vec<ModEntry>,
}

pub struct ModManager {
    game: PathBuf,
    pending: Option<Receiver<Result<GameMods, String>>>,
    loaded: Option<GameMods>,
    error: Option<String>,
    held_buttons: u64,
    selected: usize,
}

impl ModManager {
    pub fn new(game: PathBuf, ctx: &egui::Context) -> Self {
        let mut manager = Self {
            game,
            pending: None,
            loaded: None,
            error: None,
            held_buttons: u64::MAX,
            selected: 0,
        };
        manager.refresh(ctx);
        manager
    }

    fn refresh(&mut self, ctx: &egui::Context) {
        let game = self.game.clone();
        let ctx = ctx.clone();
        let (sender, receiver) = mpsc::channel();
        self.pending = Some(receiver);
        self.error = None;
        std::thread::spawn(move || {
            let result = (|| {
                let title_id = nexium_loader::read_application_title_id(&game)?
                    .ok_or("This file has no application title ID for game mods.")?;
                let entries = discover_mods(&nexium_common::paths::mod_roots(), title_id)?;
                Ok(GameMods {
                    title_id,
                    directory: nexium_common::paths::title_mods_dir(title_id),
                    entries,
                })
            })();
            let _ = sender.send(result);
            ctx.request_repaint();
        });
    }

    pub fn show(&mut self, ctx: &egui::Context, input: &crate::input::InputSnapshot) -> bool {
        use crate::controller_config::SwitchButton;
        let pressed = input.buttons & !self.held_buttons;
        self.held_buttons = input.buttons;
        let edge = |button: SwitchButton| pressed & button.npad_bit() != 0;
        if edge(SwitchButton::B) {
            return false;
        }
        if let Some(receiver) = &self.pending {
            match receiver.try_recv() {
                Ok(result) => {
                    self.pending = None;
                    match result {
                        Ok(loaded) => self.loaded = Some(loaded),
                        Err(error) => self.error = Some(error),
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending = None;
                    self.error = Some("Unable to read the game's mod folders.".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let mut refresh = edge(SwitchButton::X) && self.pending.is_none();
        let mut close = false;
        let modal = egui::Modal::new(egui::Id::new("game_mod_manager")).show(ctx, |ui| {
            ui.set_width((ctx.viewport_rect().width() - 48.0).clamp(280.0, 600.0));
            ui.heading("Game Mods");
            ui.label(
                self.game
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .as_ref(),
            );
            if let Some(loaded) = &mut self.loaded {
                ui.monospace(format!("Title ID: {:016X}", loaded.title_id));
                ui.label("Changes apply next time you launch this game.");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Open Mods Folder").clicked() {
                        let result = std::fs::create_dir_all(&loaded.directory)
                            .map_err(|error| error.to_string())
                            .and_then(|_| open_directory(&loaded.directory));
                        if let Err(error) = result {
                            self.error = Some(error);
                        }
                    }
                    if ui
                        .add_enabled(self.pending.is_none(), egui::Button::new("Refresh"))
                        .clicked()
                    {
                        refresh = true;
                    }
                });
                ui.add_space(4.0);
                ui.label("Put each mod in its own folder containing romfs and/or exefs.");
                ui.monospace(format!("{:016X}/My Mod/romfs/", loaded.title_id));
                ui.label("ExeFS supports NSO replacements and build-ID-named IPS/IPS32 patches.");
                ui.separator();
                if loaded.entries.is_empty() {
                    ui.label("No mods found. Add a mod folder, then press Refresh.");
                }
                let navigating = edge(SwitchButton::DUp) || edge(SwitchButton::DDown);
                if edge(SwitchButton::DUp) {
                    self.selected = self.selected.saturating_sub(1);
                }
                if edge(SwitchButton::DDown) {
                    self.selected = self.selected.saturating_add(1);
                }
                self.selected = self.selected.min(loaded.entries.len().saturating_sub(1));
                let editable = self.pending.is_none() && !refresh;
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for (index, entry) in loaded.entries.iter_mut().enumerate() {
                            ui.push_id(&entry.path, |ui| {
                                ui.horizontal(|ui| {
                                    let mut enabled = entry.enabled;
                                    let checkbox = ui.add_enabled(
                                        editable,
                                        egui::Checkbox::new(&mut enabled, &entry.name),
                                    );
                                    if index == self.selected && input.connected {
                                        ui.painter().rect_stroke(
                                            checkbox.rect.expand(2.0),
                                            3.0,
                                            ui.visuals().selection.stroke,
                                            egui::StrokeKind::Outside,
                                        );
                                        if navigating {
                                            checkbox.scroll_to_me(Some(egui::Align::Center));
                                        }
                                    }
                                    let controller_toggle =
                                        editable && index == self.selected && edge(SwitchButton::A);
                                    if controller_toggle {
                                        enabled = !enabled;
                                    }
                                    if checkbox.changed() || controller_toggle {
                                        match set_mod_enabled(&entry.path, enabled) {
                                            Ok(()) => entry.enabled = enabled,
                                            Err(error) => self.error = Some(error),
                                        }
                                    }
                                    ui.weak(match (entry.has_romfs, entry.has_exefs) {
                                        (true, true) => "RomFS + ExeFS",
                                        (true, false) => "RomFS",
                                        _ => "ExeFS",
                                    });
                                    if ui.small_button("Open").clicked() {
                                        if let Err(error) = open_directory(&entry.path) {
                                            self.error = Some(error);
                                        }
                                    }
                                });
                            });
                        }
                    });
                ui.separator();
                ui.weak("Mods load in the order shown. Later mods take priority for shared files.");
            }
            if input.connected {
                ui.weak("D-pad: select   A: toggle   X: refresh   B: close");
            }
            if self.pending.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Reading mods...");
                });
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(240, 100, 100), error);
                if self.loaded.is_none() && ui.button("Retry").clicked() {
                    refresh = true;
                }
            }
            ui.add_space(8.0);
            close = ui.button("Close").clicked();
        });
        if refresh {
            self.refresh(ctx);
        }
        !close && !modal.should_close()
    }
}

fn open_directory(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let program = "explorer.exe";
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let program = "xdg-open";
    std::process::Command::new(program)
        .arg(path)
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Could not open {}: {error}", path.display()))
}
