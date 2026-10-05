use std::time::Duration;

use crate::library::{GameEntry, ICON_SIZE};
use crate::ui::{Canvas, TextPainter, ACCENT, BG, DANGER, PANEL, PANEL_HI, TEXT, TEXT_DIM};

pub const W: usize = 1920;
pub const H: usize = 1080;
pub const LIST_ROWS: usize = 5;
const ROW_H: i32 = 140;
const ROW_GAP: i32 = 18;
const LIST_TOP: i32 = 170;
const CROSS_BLUE: [u8; 4] = [0x7C, 0xB4, 0xFF, 0xFF];
const CIRCLE_RED: [u8; 4] = [0xFF, 0x6B, 0x6B, 0xFF];
const TRIANGLE_GREEN: [u8; 4] = [0x45, 0xE0, 0xA8, 0xFF];

#[derive(Clone, Copy, PartialEq)]
pub enum Glyph {
    Cross,
    Circle,
    Triangle,
    Options,
    UpDown,
    LeftRight,
}

pub struct View {
    pub canvas: Canvas,
    painter: TextPainter,
    logo: Option<(usize, usize, Vec<u8>)>,
    big_icon: Option<Vec<u8>>,
}

fn decode_logo() -> Option<(usize, usize, Vec<u8>)> {
    let img = image::load_from_memory(include_bytes!("../../branding/png/logo-128.png")).ok()?.to_rgba8();
    let img = image::imageops::resize(&img, 88, 88, image::imageops::FilterType::Triangle);
    Some((88, 88, img.into_raw()))
}

pub fn size_label(bytes: Option<u64>) -> String {
    match bytes {
        Some(n) if n >= 1 << 30 => format!("{:.2} GB", n as f64 / (1u64 << 30) as f64),
        Some(n) if n >= 1 << 20 => format!("{:.0} MB", n as f64 / (1u64 << 20) as f64),
        Some(n) => format!("{:.0} KB", (n as f64 / 1024.0).max(1.0)),
        None => String::new(),
    }
}

fn bar(c: &mut Canvas, a: (f32, f32), b: (f32, f32), width: f32, color: [u8; 4]) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt().max(0.001);
    let (nx, ny) = (-dy / len * width * 0.5, dx / len * width * 0.5);
    let q = [(a.0 + nx, a.1 + ny), (a.0 - nx, a.1 - ny), (b.0 - nx, b.1 - ny), (b.0 + nx, b.1 + ny)];
    c.fill_triangle([q[0], q[1], q[2]], color);
    c.fill_triangle([q[0], q[2], q[3]], color);
}

fn wrap(painter: &TextPainter, text: &str, px: f32, max_w: f32) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.lines() {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            let candidate = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
            if painter.measure(&candidate, px, true) > max_w && !line.is_empty() {
                lines.push(std::mem::replace(&mut line, word.to_string()));
            } else {
                line = candidate;
            }
        }
        lines.push(line);
    }
    lines
}

pub fn scroll_for(cursor: usize, scroll: usize) -> usize {
    if cursor < scroll {
        cursor
    } else if cursor >= scroll + LIST_ROWS {
        cursor + 1 - LIST_ROWS
    } else {
        scroll
    }
}

impl View {
    pub fn new() -> Option<Self> {
        Some(Self { canvas: Canvas::new(W, H), painter: TextPainter::new()?, logo: decode_logo(), big_icon: None })
    }

    pub fn set_big_icon(&mut self, icon: Option<&[u8]>) {
        self.big_icon = icon.and_then(|px| {
            let img = image::RgbaImage::from_raw(ICON_SIZE as u32, ICON_SIZE as u32, px.to_vec())?;
            Some(image::imageops::resize(&img, 256, 256, image::imageops::FilterType::Triangle).into_raw())
        });
    }

    fn center(&mut self, y: f32, px: f32, color: [u8; 4], text: &str) {
        let w = self.painter.measure(text, px, false);
        self.painter.draw(&mut self.canvas, (W as f32 - w) / 2.0, y, px, color, text);
    }

    fn header(&mut self, subtitle: &str) {
        let c = &mut self.canvas;
        c.clear(BG);
        c.fill_rect(0, 0, W as i32, 128, PANEL);
        c.fill_rect(0, 128, W as i32, 3, ACCENT);
        if let Some((w, h, px)) = &self.logo {
            c.blit_rgba(56, 20, *w, *h, px);
        }
        self.painter.draw(c, 166.0, 18.0, 52.0, TEXT, "NeXium");
        self.painter.draw(c, 168.0, 82.0, 24.0, TEXT_DIM, subtitle);
    }

    fn glyph(&mut self, g: Glyph, cx: f32, cy: f32) -> f32 {
        let c = &mut self.canvas;
        let r = 19.0;
        if g == Glyph::Options {
            let label = "OPTIONS";
            let w = self.painter.measure(label, 16.0, false) + 22.0;
            c.fill_rounded((cx - r) as i32, (cy - 15.0) as i32, w as i32, 30, 15, PANEL_HI);
            self.painter.draw(c, cx - r + 11.0, cy - 10.0, 16.0, TEXT, label);
            return w;
        }
        c.fill_circle(cx, cy, r, PANEL_HI);
        match g {
            Glyph::Cross => {
                bar(c, (cx - 8.0, cy - 8.0), (cx + 8.0, cy + 8.0), 3.2, CROSS_BLUE);
                bar(c, (cx - 8.0, cy + 8.0), (cx + 8.0, cy - 8.0), 3.2, CROSS_BLUE);
            }
            Glyph::Circle => c.stroke_circle(cx, cy, 8.5, 3.0, CIRCLE_RED),
            Glyph::Triangle => {
                c.fill_triangle([(cx, cy - 10.0), (cx - 10.0, cy + 7.0), (cx + 10.0, cy + 7.0)], TRIANGLE_GREEN);
                c.fill_triangle([(cx, cy - 4.5), (cx - 5.2, cy + 4.0), (cx + 5.2, cy + 4.0)], PANEL_HI);
            }
            Glyph::UpDown => {
                c.fill_triangle([(cx, cy - 12.0), (cx - 7.0, cy - 3.0), (cx + 7.0, cy - 3.0)], TEXT);
                c.fill_triangle([(cx, cy + 12.0), (cx - 7.0, cy + 3.0), (cx + 7.0, cy + 3.0)], TEXT);
            }
            Glyph::LeftRight => {
                c.fill_triangle([(cx - 12.0, cy), (cx - 3.0, cy - 7.0), (cx - 3.0, cy + 7.0)], TEXT);
                c.fill_triangle([(cx + 12.0, cy), (cx + 3.0, cy - 7.0), (cx + 3.0, cy + 7.0)], TEXT);
            }
            Glyph::Options => {}
        }
        2.0 * r
    }

    fn hints(&mut self, items: &[(Glyph, &str)]) {
        let y = H as f32 - 56.0;
        self.canvas.fill_rect(0, H as i32 - 110, W as i32, 110, PANEL);
        let mut x = 72.0;
        for (g, label) in items {
            let w = self.glyph(*g, x + 19.0, y);
            x += w + 14.0;
            x += self.painter.draw(&mut self.canvas, x, y - 16.0, 26.0, TEXT, label) + 46.0;
        }
    }

    pub fn library(&mut self, games: &[GameEntry], sizes: &[Option<u64>], done: bool, cursor: usize, scroll: usize) {
        let count = games.len();
        let subtitle = if done {
            format!("{count} game{}", if count == 1 { "" } else { "s" })
        } else {
            "Scanning your library…".to_string()
        };
        self.header(&subtitle);
        if done && count == 0 {
            let c = &mut self.canvas;
            c.fill_rounded(360, 300, 1200, 420, 24, PANEL);
            self.painter.draw(c, 420.0, 350.0, 44.0, TEXT, "No games yet");
            self.painter.draw(c, 420.0, 430.0, 28.0, TEXT_DIM, "Copy decrypted Switch games (.nsp .xci .dnsp .dxci .nro) into");
            self.painter.draw_mono(c, 420.0, 490.0, 28.0, ACCENT, "/data/homebrew/PPSA99640/data/games/");
            self.painter.draw(c, 420.0, 550.0, 28.0, TEXT_DIM, "with PS5Upload or FTP, then press Triangle to rescan.");
        }
        for row in 0..LIST_ROWS {
            let index = scroll + row;
            let Some(game) = games.get(index) else { break };
            let y = LIST_TOP + row as i32 * (ROW_H + ROW_GAP);
            let size = size_label(sizes.get(index).copied().flatten());
            self.row(y, index == cursor, game, &size);
        }
        if count > LIST_ROWS {
            let track = LIST_ROWS as i32 * (ROW_H + ROW_GAP) - ROW_GAP;
            let thumb = (track * LIST_ROWS as i32 / count as i32).max(40);
            let pos = (track - thumb) * scroll as i32 / (count - LIST_ROWS) as i32;
            self.canvas.fill_rounded(1830, LIST_TOP, 8, track, 4, PANEL);
            self.canvas.fill_rounded(1830, LIST_TOP + pos, 8, thumb, 4, ACCENT);
        }
        let mut hints = vec![(Glyph::UpDown, "Select")];
        if count > 0 {
            hints.push((Glyph::Cross, "Play"));
        }
        hints.extend([(Glyph::Triangle, "Rescan"), (Glyph::Options, "Settings"), (Glyph::Circle, "Quit")]);
        self.hints(&hints);
    }

    fn row(&mut self, y: i32, selected: bool, game: &GameEntry, size: &str) {
        let c = &mut self.canvas;
        c.fill_rounded(72, y, 1740, ROW_H, 18, if selected { PANEL_HI } else { PANEL });
        if selected {
            c.fill_rounded(72, y, 8, ROW_H, 4, ACCENT);
        }
        let ix = 104;
        let iy = y + (ROW_H - ICON_SIZE as i32) / 2;
        match &game.icon {
            Some(px) => c.blit_rgba(ix, iy, ICON_SIZE, ICON_SIZE, px),
            None => {
                c.fill_rounded(ix, iy, ICON_SIZE as i32, ICON_SIZE as i32, 12, BG);
                let initial: String = game.title.chars().next().map(|ch| ch.to_uppercase().collect()).unwrap_or_default();
                let w = self.painter.measure(&initial, 54.0, false);
                self.painter.draw(c, ix as f32 + (ICON_SIZE as f32 - w) / 2.0, iy as f32 + 24.0, 54.0, TEXT_DIM, &initial);
            }
        }
        let tx = (ix + ICON_SIZE as i32 + 36) as f32;
        self.painter.draw(c, tx, y as f32 + 16.0, 36.0, TEXT, &game.title);
        let meta = match (game.author.is_empty(), game.version.is_empty()) {
            (false, false) => format!("{}  ·  v{}", game.author, game.version),
            (false, true) => game.author.clone(),
            (true, false) => format!("v{}", game.version),
            (true, true) => String::new(),
        };
        self.painter.draw(c, tx, y as f32 + 62.0, 24.0, TEXT_DIM, &meta);
        let file = game.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.painter.draw_mono(c, tx, y as f32 + 98.0, 19.0, TEXT_DIM, &format!("{file}   {size}"));
    }

    pub fn dialog(&mut self, title: &str, body: &str, buttons: &[(Glyph, &str)]) {
        self.canvas.dim(140);
        let c = &mut self.canvas;
        c.fill_rounded(610, 380, 700, 300, 26, PANEL);
        self.painter.draw(c, 670.0, 420.0, 38.0, TEXT, title);
        self.painter.draw(c, 670.0, 480.0, 24.0, TEXT_DIM, body);
        let mut x = 670.0;
        for (g, label) in buttons {
            let w = self.glyph(*g, x + 19.0, 610.0);
            x += w + 12.0;
            x += self.painter.draw(&mut self.canvas, x, 594.0, 26.0, TEXT, label) + 40.0;
        }
    }

    pub fn settings(
        &mut self,
        docked: bool,
        volume: f32,
        show_fps: bool,
        stable_resolution: bool,
        normal_gpu_accuracy: bool,
        cursor: usize,
    ) {
        self.header("Settings");
        let items: [(&str, String, &str); 5] = [
            ("Console mode", if docked { "Docked (TV)".into() } else { "Handheld".into() }, "Docked renders games at 1080p; handheld at 720p."),
            ("Volume", format!("{:.0}%", volume * 100.0), "Game audio level, 0–200%."),
            ("Performance overlay", if show_fps { "On".into() } else { "Off".into() }, "Shows the game's frame rate in the corner."),
            (
                "Stable resolution",
                if stable_resolution { "On".into() } else { "Off".into() },
                "Keeps dynamic-resolution games at full size to save memory. Takes effect next launch.",
            ),
            (
                "GPU accuracy",
                if normal_gpu_accuracy { "Normal".into() } else { "High".into() },
                "Normal reports GPU work done when it is submitted. Faster; a few games may glitch. Takes effect next launch.",
            ),
        ];
        for (i, (label, value, help)) in items.iter().enumerate() {
            let y = 170 + i as i32 * 132;
            let selected = i == cursor;
            let c = &mut self.canvas;
            c.fill_rounded(260, y, 1400, 116, 18, if selected { PANEL_HI } else { PANEL });
            if selected {
                c.fill_rounded(260, y, 8, 116, 4, ACCENT);
            }
            self.painter.draw(c, 310.0, y as f32 + 22.0, 34.0, TEXT, label);
            self.painter.draw(c, 310.0, y as f32 + 72.0, 22.0, TEXT_DIM, help);
            let vw = self.painter.measure(value, 32.0, false);
            self.painter.draw(c, 1610.0 - vw, y as f32 + 38.0, 32.0, ACCENT, value);
            if i == 1 {
                let x = 1610 - vw as i32 - 340;
                c.fill_rounded(x, y + 56, 300, 10, 5, BG);
                c.fill_rounded(x, y + 56, ((volume / 2.0 * 300.0) as i32).max(10), 10, 5, ACCENT);
            }
        }
        let about = format!(
            "NeXium {} for PS5  ·  Dynarmic JIT  ·  RADV Vulkan  ·  saves in /data/homebrew/PPSA99640/data/NeXium",
            env!("CARGO_PKG_VERSION")
        );
        self.painter.draw(&mut self.canvas, 260.0, 852.0, 22.0, TEXT_DIM, &about);
        self.painter.draw(&mut self.canvas, 260.0, 888.0, 22.0, TEXT_DIM, "In a game, press Options + Touch Pad together for the pause menu.");
        self.hints(&[(Glyph::UpDown, "Select"), (Glyph::LeftRight, "Change"), (Glyph::Circle, "Back")]);
    }

    pub fn loading(&mut self, title: &str, elapsed: Duration) {
        self.canvas.clear(BG);
        if let Some(icon) = &self.big_icon {
            self.canvas.blit_rgba(W as i32 / 2 - 128, 250, 256, 256, icon);
        } else if let Some((w, h, px)) = &self.logo {
            self.canvas.blit_rgba(W as i32 / 2 - *w as i32 / 2, 330, *w, *h, px);
        }
        self.center(560.0, 48.0, TEXT, title);
        self.center(640.0, 28.0, TEXT_DIM, &format!("Starting…  {:.0}s", elapsed.as_secs_f32()));
        let phase = elapsed.as_secs_f32() * 3.0;
        for i in 0..8 {
            let a = phase + i as f32 * std::f32::consts::TAU / 8.0;
            let alpha = (60 + i * 24) as u8;
            self.canvas.fill_circle(W as f32 / 2.0 + a.cos() * 28.0, 760.0 + a.sin() * 28.0, 6.0, [ACCENT[0], ACCENT[1], ACCENT[2], alpha]);
        }
        self.center(900.0, 22.0, TEXT_DIM, "Press Options + Touch Pad together in a game for the pause menu");
    }

    pub fn pause(&mut self, backdrop: &[u8], title: &str, fps: f32, cursor: usize) {
        if backdrop.len() == self.canvas.pixels.len() {
            self.canvas.pixels.copy_from_slice(backdrop);
        } else {
            self.canvas.clear(BG);
        }
        self.canvas.dim(150);
        let c = &mut self.canvas;
        c.fill_rounded(660, 260, 600, 520, 28, PANEL);
        self.painter.draw(c, 720.0, 300.0, 40.0, TEXT, "Paused");
        self.painter.draw(c, 720.0, 360.0, 24.0, TEXT_DIM, title);
        self.painter.draw(c, 720.0, 396.0, 22.0, TEXT_DIM, &format!("{fps:.0} FPS before pausing"));
        for (i, label) in ["Resume", "Quit to library"].iter().enumerate() {
            let y = 470 + i as i32 * 110;
            let c = &mut self.canvas;
            c.fill_rounded(710, y, 500, 86, 16, if i == cursor { PANEL_HI } else { BG });
            if i == cursor {
                c.fill_rounded(710, y, 8, 86, 4, if i == 1 { DANGER } else { ACCENT });
            }
            self.painter.draw(c, 750.0, y as f32 + 24.0, 32.0, TEXT, label);
        }
        self.hints(&[(Glyph::UpDown, "Select"), (Glyph::Cross, "Confirm"), (Glyph::Circle, "Resume")]);
    }

    pub fn error(&mut self, title: &str, message: &str) {
        self.header("Something went wrong");
        let c = &mut self.canvas;
        c.fill_rounded(260, 220, 1400, 560, 24, PANEL);
        c.fill_rounded(260, 220, 8, 560, 4, DANGER);
        self.painter.draw(c, 320.0, 260.0, 40.0, TEXT, title);
        let mut y = 340.0;
        for line in wrap(&self.painter, message, 22.0, 1280.0).into_iter().take(12) {
            self.painter.draw_mono(&mut self.canvas, 320.0, y, 22.0, TEXT_DIM, &line);
            y += 34.0;
        }
        self.hints(&[(Glyph::Circle, "Back to library")]);
    }

    pub fn message(&mut self, title: &str, body: &str) {
        self.canvas.clear(BG);
        self.center(480.0, 40.0, TEXT, title);
        self.center(550.0, 24.0, TEXT_DIM, body);
    }

    pub fn fps_overlay(&mut self, pixels: &mut Vec<u8>, width: u32, height: u32, fps: f32) {
        let mut overlay = Canvas { w: width as usize, h: height as usize, pixels: std::mem::take(pixels) };
        let label = format!("{fps:.0} FPS");
        let scale = overlay.h as f32 / 720.0;
        let px = 22.0 * scale;
        let w = self.painter.measure(&label, px, true) + 20.0 * scale;
        overlay.fill_rounded((12.0 * scale) as i32, (12.0 * scale) as i32, w as i32, (36.0 * scale) as i32, (8.0 * scale) as i32, [0, 0, 0, 0xA0]);
        self.painter.draw_mono(&mut overlay, 22.0 * scale, 17.0 * scale, px, TEXT, &label);
        *pixels = overlay.pixels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn save(view: &View, name: &str) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/ps5-ui-preview");
        std::fs::create_dir_all(&dir).unwrap();
        let img = image::RgbaImage::from_raw(W as u32, H as u32, view.canvas.pixels.clone()).unwrap();
        img.save(dir.join(format!("{name}.png"))).unwrap();
    }

    fn icon(seed: u8) -> Vec<u8> {
        let mut px = vec![0u8; ICON_SIZE * ICON_SIZE * 4];
        for y in 0..ICON_SIZE {
            for x in 0..ICON_SIZE {
                let i = (y * ICON_SIZE + x) * 4;
                px[i] = (x * 2) as u8 ^ seed;
                px[i + 1] = (y * 2) as u8;
                px[i + 2] = seed.wrapping_mul(3);
                px[i + 3] = 255;
            }
        }
        px
    }

    fn games() -> Vec<GameEntry> {
        let entry = |t: &str, a: &str, v: &str, f: &str, i: Option<u8>| GameEntry {
            path: PathBuf::from(format!("/app0/data/games/{f}")),
            title: t.into(),
            author: a.into(),
            version: v.into(),
            icon: i.map(icon),
        };
        vec![
            entry("Cave Story+", "Nicalis", "1.2.0", "cavestory.dnsp", Some(30)),
            entry("Celeste", "Maddy Makes Games", "1.4.0", "celeste.dxci", Some(90)),
            entry("NeXium PS5 test", "NeXium", "1.0.0", "nxtest.nro", None),
            entry("Mario Kart 8 Deluxe", "Nintendo", "3.0.3", "mk8d.dxci", Some(160)),
            entry("Metroid Dread", "Nintendo", "2.1.0", "dread.dxci", Some(200)),
            entry("Puyo Puyo Tetris 2", "SEGA", "", "ppt2.dxci", Some(240)),
        ]
    }

    #[test]
    fn render_preview() {
        let mut view = View::new().expect("fonts");
        let list = games();
        let sizes = [Some(141_411_168), Some(2_029_368_576), Some(268_485), Some(12_453_274_624), Some(7_985_954_816), Some(1_714_521_088)];
        view.library(&list, &sizes, true, 1, 0);
        save(&view, "1-library");
        view.library(&list, &sizes, true, 5, scroll_for(5, 0));
        save(&view, "2-library-scrolled");
        view.library(&[], &[], true, 0, 0);
        save(&view, "3-library-empty");
        view.library(&list, &sizes, true, 0, 0);
        view.dialog("Quit NeXium?", "You'll return to the PS5 home screen.", &[(Glyph::Cross, "Quit"), (Glyph::Circle, "Cancel")]);
        save(&view, "4-quit-dialog");
        view.settings(true, 1.2, false, true, true, 4);
        save(&view, "5-settings");
        view.set_big_icon(list[0].icon.as_deref());
        view.loading("Cave Story+", Duration::from_secs(3));
        save(&view, "6-loading");
        let backdrop: Vec<u8> = (0..W * H).flat_map(|i| [(i % W / 8) as u8, (i / W / 5) as u8, 90, 255]).collect();
        view.pause(&backdrop, "Cave Story+", 60.0, 1);
        save(&view, "7-pause");
        view.error("Celeste stopped", "Boot failed: mmap /app0/data/games/celeste.dxci: Cannot allocate memory (os error 12)");
        save(&view, "8-error");
        let mut frame = backdrop[..1280 * 720 * 4].to_vec();
        view.fps_overlay(&mut frame, 1280, 720, 59.9);
        assert_eq!(frame.len(), 1280 * 720 * 4);
    }
}
