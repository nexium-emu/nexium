use ab_glyph::{Font, FontRef, Glyph, ScaleFont};
use std::collections::HashMap;

pub const BG: [u8; 4] = [0x10, 0x10, 0x18, 0xFF];
pub const PANEL: [u8; 4] = [0x1A, 0x1B, 0x26, 0xFF];
pub const PANEL_HI: [u8; 4] = [0x24, 0x26, 0x36, 0xFF];
pub const ACCENT: [u8; 4] = [0x53, 0xC7, 0xB4, 0xFF];
pub const TEXT: [u8; 4] = [0xEC, 0xEC, 0xF2, 0xFF];
pub const TEXT_DIM: [u8; 4] = [0x8A, 0x8C, 0x9C, 0xFF];
pub const DANGER: [u8; 4] = [0xE0, 0x6C, 0x75, 0xFF];

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u8>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            pixels: vec![0u8; w * h * 4],
        }
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        if self.w != w || self.h != h {
            self.w = w;
            self.h = h;
            self.pixels = vec![0u8; w * h * 4];
        }
    }

    pub fn clear(&mut self, color: [u8; 4]) {
        for px in self.pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&color);
        }
    }

    #[inline]
    fn blend_px(&mut self, x: usize, y: usize, color: [u8; 4], alpha: u16) {
        if x >= self.w || y >= self.h || alpha == 0 {
            return;
        }
        let idx = (y * self.w + x) * 4;
        let a = (alpha * color[3] as u16) / 255;
        if a == 0 {
            return;
        }
        let inv = 255 - a;
        let dst = &mut self.pixels[idx..idx + 4];
        for c in 0..3 {
            dst[c] = ((color[c] as u16 * a + dst[c] as u16 * inv) / 255) as u8;
        }
        dst[3] = 0xFF;
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 4]) {
        let x0 = (x.max(0) as usize).min(self.w);
        let y0 = (y.max(0) as usize).min(self.h);
        let x1 = ((x + w).max(0) as usize).min(self.w);
        let y1 = ((y + h).max(0) as usize).min(self.h);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        if color[3] == 0xFF {
            for row in y0..y1 {
                let start = (row * self.w + x0) * 4;
                let end = (row * self.w + x1) * 4;
                for px in self.pixels[start..end].chunks_exact_mut(4) {
                    px.copy_from_slice(&color);
                }
            }
        } else {
            for row in y0..y1 {
                for col in x0..x1 {
                    self.blend_px(col, row, color, 255);
                }
            }
        }
    }

    pub fn fill_rounded(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32, color: [u8; 4]) {
        let r = r.min(w / 2).min(h / 2).max(0);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.w as i32);
        let y1 = (y + h).min(self.h as i32);
        for row in y0..y1 {
            for col in x0..x1 {
                let dx = if col < x + r {
                    x + r - col
                } else if col >= x + w - r {
                    col - (x + w - r - 1)
                } else {
                    0
                };
                let dy = if row < y + r {
                    y + r - row
                } else if row >= y + h - r {
                    row - (y + h - r - 1)
                } else {
                    0
                };
                if dx > 0 && dy > 0 {
                    let d2 = (dx * dx + dy * dy) as f32;
                    let rf = r as f32 + 0.5;
                    if d2 > rf * rf {
                        continue;
                    }
                    let d = d2.sqrt();
                    if d > rf - 1.0 {
                        let cov = ((rf - d).clamp(0.0, 1.0) * 255.0) as u16;
                        self.blend_px(col as usize, row as usize, color, cov);
                        continue;
                    }
                }
                self.blend_px(col as usize, row as usize, color, 255);
            }
        }
    }

    pub fn fill_circle(&mut self, cx: f32, cy: f32, r: f32, color: [u8; 4]) {
        let x0 = ((cx - r - 1.0).max(0.0)) as usize;
        let y0 = ((cy - r - 1.0).max(0.0)) as usize;
        let x1 = ((cx + r + 2.0).min(self.w as f32)) as usize;
        let y1 = ((cy + r + 2.0).min(self.h as f32)) as usize;
        for row in y0..y1 {
            for col in x0..x1 {
                let dx = col as f32 + 0.5 - cx;
                let dy = row as f32 + 0.5 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d <= r - 0.5 {
                    self.blend_px(col, row, color, 255);
                } else if d < r + 0.5 {
                    let cov = ((r + 0.5 - d).clamp(0.0, 1.0) * 255.0) as u16;
                    self.blend_px(col, row, color, cov);
                }
            }
        }
    }

    pub fn stroke_circle(&mut self, cx: f32, cy: f32, r: f32, width: f32, color: [u8; 4]) {
        let outer = r + width * 0.5;
        let inner = r - width * 0.5;
        let x0 = ((cx - outer - 1.0).max(0.0)) as usize;
        let y0 = ((cy - outer - 1.0).max(0.0)) as usize;
        let x1 = ((cx + outer + 2.0).min(self.w as f32)) as usize;
        let y1 = ((cy + outer + 2.0).min(self.h as f32)) as usize;
        for row in y0..y1 {
            for col in x0..x1 {
                let dx = col as f32 + 0.5 - cx;
                let dy = row as f32 + 0.5 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                let cov = ((outer + 0.5 - d).clamp(0.0, 1.0)
                    * (d - inner + 0.5).clamp(0.0, 1.0)
                    * 255.0) as u16;
                if cov > 0 {
                    self.blend_px(col, row, color, cov);
                }
            }
        }
    }

    pub fn fill_triangle(&mut self, pts: [(f32, f32); 3], color: [u8; 4]) {
        let min_x = pts.iter().fold(f32::MAX, |a, p| a.min(p.0)).floor().max(0.0) as usize;
        let max_x = (pts.iter().fold(f32::MIN, |a, p| a.max(p.0)).ceil() + 1.0)
            .min(self.w as f32)
            .max(0.0) as usize;
        let min_y = pts.iter().fold(f32::MAX, |a, p| a.min(p.1)).floor().max(0.0) as usize;
        let max_y = (pts.iter().fold(f32::MIN, |a, p| a.max(p.1)).ceil() + 1.0)
            .min(self.h as f32)
            .max(0.0) as usize;
        let edge = |a: (f32, f32), b: (f32, f32), px: f32, py: f32| {
            (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
        };
        for row in min_y..max_y {
            for col in min_x..max_x {
                let px = col as f32 + 0.5;
                let py = row as f32 + 0.5;
                let w0 = edge(pts[0], pts[1], px, py);
                let w1 = edge(pts[1], pts[2], px, py);
                let w2 = edge(pts[2], pts[0], px, py);
                let inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                    || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
                if inside {
                    self.blend_px(col, row, color, 255);
                }
            }
        }
    }

    pub fn blit_rgba(&mut self, x: i32, y: i32, src_w: usize, src_h: usize, src: &[u8]) {
        if src.len() < src_w * src_h * 4 {
            return;
        }
        for row in 0..src_h {
            let dy = y + row as i32;
            if dy < 0 || dy as usize >= self.h {
                continue;
            }
            let col0 = (-x).max(0) as usize;
            let col1 = if x >= self.w as i32 {
                0
            } else {
                src_w.min((self.w as i32 - x) as usize)
            };
            for col in col0..col1 {
                let dx = (x + col as i32) as usize;
                let s = (row * src_w + col) * 4;
                let a = src[s + 3];
                if a == 0xFF {
                    let d = (dy as usize * self.w + dx) * 4;
                    self.pixels[d..d + 4].copy_from_slice(&[
                        src[s],
                        src[s + 1],
                        src[s + 2],
                        0xFF,
                    ]);
                } else if a != 0 {
                    let color = [src[s], src[s + 1], src[s + 2], 0xFF];
                    self.blend_px(dx, dy as usize, color, a as u16);
                }
            }
        }
    }

    pub fn dim(&mut self, alpha: u8) {
        let inv = 255 - alpha as u16;
        for px in self.pixels.chunks_exact_mut(4) {
            for c in 0..3 {
                px[c] = ((px[c] as u16 * inv) / 255) as u8;
            }
        }
    }
}

struct GlyphEntry {
    w: usize,
    h: usize,
    left: i32,
    top: i32,
    coverage: Vec<u8>,
}

pub struct TextPainter {
    font: FontRef<'static>,
    mono: FontRef<'static>,
    cache: HashMap<(char, u32, bool), GlyphEntry>,
}

impl TextPainter {
    pub fn new() -> Option<Self> {
        Some(Self {
            font: FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT).ok()?,
            mono: FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR).ok()?,
            cache: HashMap::new(),
        })
    }

    fn ensure_glyph(&mut self, ch: char, px: f32, mono: bool) {
        let key = (ch, px.to_bits(), mono);
        if !self.cache.contains_key(&key) {
            let font = if mono { &self.mono } else { &self.font };
            let scaled = font.as_scaled(px);
            let glyph: Glyph = scaled.scaled_glyph(ch);
            let entry = match font.outline_glyph(glyph) {
                Some(outline) => {
                    let bounds = outline.px_bounds();
                    let w = bounds.width().ceil() as usize;
                    let h = bounds.height().ceil() as usize;
                    let mut coverage = vec![0u8; w * h];
                    outline.draw(|gx, gy, c| {
                        let idx = gy as usize * w + gx as usize;
                        if idx < coverage.len() {
                            coverage[idx] = (c * 255.0) as u8;
                        }
                    });
                    GlyphEntry {
                        w,
                        h,
                        left: bounds.min.x as i32,
                        top: bounds.min.y as i32,
                        coverage,
                    }
                }
                None => GlyphEntry {
                    w: 0,
                    h: 0,
                    left: 0,
                    top: 0,
                    coverage: Vec::new(),
                },
            };
            self.cache.insert(key, entry);
        }
    }

    fn advance(&self, ch: char, px: f32, mono: bool) -> f32 {
        let font = if mono { &self.mono } else { &self.font };
        let scaled = font.as_scaled(px);
        scaled.h_advance(scaled.scaled_glyph(ch).id)
    }

    pub fn measure(&self, text: &str, px: f32, mono: bool) -> f32 {
        text.chars().map(|ch| self.advance(ch, px, mono)).sum()
    }

    pub fn draw(
        &mut self,
        canvas: &mut Canvas,
        x: f32,
        y: f32,
        px: f32,
        color: [u8; 4],
        text: &str,
    ) -> f32 {
        self.draw_impl(canvas, x, y, px, color, text, false)
    }

    pub fn draw_mono(
        &mut self,
        canvas: &mut Canvas,
        x: f32,
        y: f32,
        px: f32,
        color: [u8; 4],
        text: &str,
    ) -> f32 {
        self.draw_impl(canvas, x, y, px, color, text, true)
    }

    fn draw_impl(
        &mut self,
        canvas: &mut Canvas,
        x: f32,
        y: f32,
        px: f32,
        color: [u8; 4],
        text: &str,
        mono: bool,
    ) -> f32 {
        let ascent = {
            let font = if mono { &self.mono } else { &self.font };
            font.as_scaled(px).ascent()
        };
        let mut pen = x;
        for ch in text.chars() {
            let adv = self.advance(ch, px, mono);
            self.ensure_glyph(ch, px, mono);
            if let Some(entry) = self.cache.get(&(ch, px.to_bits(), mono)) {
                let gx0 = (pen as i32) + entry.left;
                let gy0 = (y + ascent) as i32 + entry.top;
                for row in 0..entry.h {
                    for col in 0..entry.w {
                        let c = entry.coverage[row * entry.w + col] as u16;
                        if c > 0 {
                            let dx = gx0 + col as i32;
                            let dy = gy0 + row as i32;
                            if dx >= 0 && dy >= 0 {
                                canvas.blend_px(dx as usize, dy as usize, color, c);
                            }
                        }
                    }
                }
            }
            pen += adv;
        }
        pen - x
    }
}
