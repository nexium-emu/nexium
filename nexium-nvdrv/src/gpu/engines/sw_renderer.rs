use super::maxwell3d::{DrawCall, DrawTextureCall, RenderTarget};
use super::maxwell_dma::MaxwellDma;
use super::super::GpuMappings;

pub fn execute_draws(
    draws: &[DrawCall],
    mappings: &GpuMappings,
    maxwell_dma: &mut MaxwellDma,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    for draw in draws {
        if let Some(dt) = draw.draw_texture {
            execute_draw_texture(&dt, mappings, maxwell_dma, mem_read, mem_write);
        } else if draw.is_clear {
            execute_clear(draw, mappings, mem_write);
        } else if draw.vertex_count > 0 || draw.index_count > 0 {
            execute_draw(draw, mappings, mem_read, mem_write);
        }
    }
}

fn execute_draw_texture(
    dt: &DrawTextureCall,
    mappings: &GpuMappings,
    maxwell_dma: &mut MaxwellDma,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    if dt.texture_id > dt.tic_pool_limit {
        return;
    }
    let tic_gpu_va = dt.tic_pool_gpu_va + (dt.texture_id as u64) * 0x20;
    let Some(tic_cpu) = mappings.cpu_address_for(tic_gpu_va) else { return };
    let mut tic = [0u8; 32];
    if !mem_read(tic_cpu, &mut tic) { return; }
    let w1 = u32::from_le_bytes(tic[4..8].try_into().unwrap());
    let w2 = u32::from_le_bytes(tic[8..12].try_into().unwrap());
    let w4 = u32::from_le_bytes(tic[16..20].try_into().unwrap());
    let w5 = u32::from_le_bytes(tic[20..24].try_into().unwrap());

    let src_gpu_va = (w1 as u64) | (((w2 & 0xFFFF) as u64) << 32);
    let tile_height_log2 = (w2 >> 22) & 0x7;
    let tic_w_minus_1 = w4 & 0xFFFF;
    let tic_h_minus_1 = w5 & 0xFFFF;
    let mut tex_width = (tic_w_minus_1 + 1) as usize;
    let mut tex_height = (tic_h_minus_1 + 1) as usize;
    if tex_width == 1 || tex_width > 8192 { tex_width = dt.dst_width as usize; }
    if tex_height == 1 || tex_height > 8192 { tex_height = dt.dst_height as usize; }

    if maxwell_dma.draw_texture_blits < 4 {
        log::info!(
            "DrawTexture[{}]: tex_id={} tic_pool={:#x} src_gpu={:#x} src={}x{} bh={}",
            maxwell_dma.draw_texture_blits, dt.texture_id, dt.tic_pool_gpu_va,
            src_gpu_va, tex_width, tex_height, tile_height_log2
        );
    }

    let Some(src_cpu) = mappings.cpu_address_for(src_gpu_va) else { return };

    if maxwell_dma.last_tiled_dst_cpu == 0 { return; }
    let dst_cpu = maxwell_dma.last_tiled_dst_cpu;
    let dst_bh_log2 = maxwell_dma.last_tiled_dst_bh_log2;
    let dst_w_bytes = maxwell_dma.last_tiled_dst_stride.max(1) as usize;
    let dst_h = maxwell_dma.last_tiled_dst_height.max(1) as usize;

    let blit_w = (dt.dst_width as usize).min(tex_width).min(dst_w_bytes / 4);
    let blit_h = (dt.dst_height as usize).min(tex_height).min(dst_h);

    let src_pitch = tex_width * 4;
    let linear = unswizzle_block_linear_local(src_cpu, src_pitch, tex_width, tex_height, tile_height_log2, mem_read);

    for y in 0..blit_h {
        for x in 0..blit_w {
            let off = y * src_pitch + x * 4;
            if off + 4 > linear.len() { continue; }
            let dst_x = dt.dst_x as usize + x;
            let dst_y = dt.dst_y as usize + y;
            write_tiled_pixel_bytes(dst_cpu, dst_x, dst_y, dst_w_bytes, dst_bh_log2,
                [linear[off], linear[off+1], linear[off+2], linear[off+3]], mem_write);
        }
    }
    maxwell_dma.draw_texture_blits = maxwell_dma.draw_texture_blits.wrapping_add(1);
}

fn unswizzle_block_linear_local(
    src_cpu: u64,
    pitch: usize,
    width: usize,
    height: usize,
    bh_log2: u32,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Vec<u8> {
    let width_bytes = width * 4;
    let total_tiled = {
        const GOB_W: usize = 64;
        const GOB_H: usize = 8;
        const GOB_SIZE: usize = 512;
        let bh = 1usize << bh_log2;
        let rows_per_block = bh * GOB_H;
        let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
        let block_rows = (height + rows_per_block - 1) / rows_per_block;
        gobs_per_row * block_rows * bh * GOB_SIZE
    };
    let mut tiled = vec![0u8; total_tiled];
    mem_read(src_cpu, &mut tiled);
    let mut linear = vec![0u8; pitch * height];
    for y in 0..height {
        for x in 0..width {
            let off = tiled_offset(x * 4, y, width_bytes, bh_log2);
            if off + 4 > tiled.len() { continue; }
            let dst_off = y * pitch + x * 4;
            linear[dst_off..dst_off + 4].copy_from_slice(&tiled[off..off + 4]);
        }
    }
    linear
}

fn write_tiled_pixel_bytes(
    rt_cpu: u64,
    px: usize,
    py: usize,
    width_bytes: usize,
    bh_log2: u32,
    pixel: [u8; 4],
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let byte_x = px * 4;
    let off = tiled_offset(byte_x, py, width_bytes, bh_log2);
    mem_write(rt_cpu + off as u64, &pixel);
}

#[derive(Copy, Clone)]
struct TicInfo {
    src_cpu: u64,
    bh_log2: u32,
    width: usize,
    height: usize,
}

fn is_float2_format(format: u32) -> bool {
    let size = format & 0x3F;
    let type_ = format >> 6;
    size == 0x04 && type_ == 7
}

fn read_tic_entry(
    pool_va: u64,
    idx: u64,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<TicInfo> {
    let tic_va = pool_va + idx * 0x20;
    let tic_cpu = mappings.cpu_address_for(tic_va)?;
    let mut tic = [0u8; 32];
    if !mem_read(tic_cpu, &mut tic) { return None; }
    let w1 = u32::from_le_bytes(tic[4..8].try_into().unwrap());
    let w2 = u32::from_le_bytes(tic[8..12].try_into().unwrap());
    let w3 = u32::from_le_bytes(tic[12..16].try_into().unwrap());
    let w4 = u32::from_le_bytes(tic[16..20].try_into().unwrap());
    let w5 = u32::from_le_bytes(tic[20..24].try_into().unwrap());
    let src_gpu = (w1 as u64) | (((w2 & 0xFFFF) as u64) << 32);
    if src_gpu == 0 { return None; }
    let src_cpu = mappings.cpu_address_for(src_gpu)?;
    let bh_log2 = (w3 >> 3) & 0x7;
    let width = ((w4 & 0xFFFF) + 1) as usize;
    let height = ((w5 & 0xFFFF) + 1) as usize;
    if width == 0 || height == 0 { return None; }
    Some(TicInfo { src_cpu, bh_log2, width, height })
}

fn try_resolve_bindless(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<TicInfo> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<u64, Vec<nexium_shader::FsTexId>>> = RefCell::new(HashMap::new());
    }

    let fs_cpu = mappings.cpu_address_for(draw.fs_shader_gpu_va)?;
    let cb_cpu = mappings.cpu_address_for(draw.fs_bindless_cb_addr)?;

    let ids = CACHE.with(|c| c.borrow().get(&draw.fs_shader_gpu_va).cloned()).unwrap_or_else(|| {
        const SPH: u64 = 48;
        const CODE_SIZE: usize = 4096;
        let mut code = vec![0u8; CODE_SIZE];
        if !mem_read(fs_cpu + SPH, &mut code) { return Vec::new(); }
        let result = nexium_shader::extract_fs_tex_ids(&code, 15);
        CACHE.with(|c| { c.borrow_mut().insert(draw.fs_shader_gpu_va, result.clone()); });
        result
    });

    for id in &ids {
        match *id {
            nexium_shader::FsTexId::ImmediateTic(idx) => {
                if draw.tic_pool_gpu_va != 0 {
                    if let Some(t) = read_tic_entry(draw.tic_pool_gpu_va, idx as u64, mappings, mem_read) {
                        return Some(t);
                    }
                }
            }
            nexium_shader::FsTexId::BindlessCbufOffset(off) => {
                if (off as usize) + 4 > draw.fs_bindless_cb_size as usize { continue; }
                let mut bytes = [0u8; 4];
                if !mem_read(cb_cpu + off as u64, &mut bytes) { continue; }
                let handle = u32::from_le_bytes(bytes);
                let tic_idx = (handle & 0xFFFFF) as u64;
                if tic_idx == 0 || tic_idx > draw.tic_pool_limit as u64 { continue; }
                if let Some(t) = read_tic_entry(draw.tic_pool_gpu_va, tic_idx, mappings, mem_read) {
                    return Some(t);
                }
            }
        }
    }
    None
}

fn resolve_tic(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<TicInfo> {
    if draw.fs_shader_gpu_va != 0 && draw.fs_bindless_cb_addr != 0 {
        if let Some(t) = try_resolve_bindless(draw, mappings, mem_read) {
            return Some(t);
        }
    }

    if draw.tic_pool_gpu_va == 0 { return None; }
    let limit = (draw.tic_pool_limit + 1).min(64) as u64;
    for idx in 0..limit {
        if let Some(t) = read_tic_entry(draw.tic_pool_gpu_va, idx, mappings, mem_read) {
            return Some(t);
        }
    }
    None
}

fn sample_tiled_pixel(
    info: &TicInfo,
    u: f32,
    v: f32,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> [u8; 4] {
    let x = ((u.clamp(0.0, 1.0) * info.width as f32) as usize).min(info.width.saturating_sub(1));
    let y = ((v.clamp(0.0, 1.0) * info.height as f32) as usize).min(info.height.saturating_sub(1));
    let width_bytes = info.width * 4;
    let off = tiled_offset(x * 4, y, width_bytes, info.bh_log2);
    let mut pixel = [0u8, 0, 0, 0xFF];
    mem_read(info.src_cpu + off as u64, &mut pixel);
    pixel
}

fn rt_gpu_va(rt: &RenderTarget) -> u64 {
    ((rt.address_hi as u64) << 32) | rt.address_lo as u64
}

fn block_height_log2(rt: &RenderTarget) -> u32 {
    (rt.tile_mode >> 4) & 0xF
}

fn tiled_offset(x: usize, y: usize, width_bytes: usize, bh_log2: u32) -> usize {
    const GOB_W: usize = 64;
    const GOB_H: usize = 8;
    const GOB_SIZE: usize = 512;
    let block_height = 1usize << bh_log2;
    let rows_per_block = block_height * GOB_H;
    let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
    let block_row_stride = gobs_per_row * block_height * GOB_SIZE;

    let block_y = y / rows_per_block;
    let y_in_block = y % rows_per_block;
    let gob_row = y_in_block / GOB_H;
    let y_in_gob = y_in_block % GOB_H;

    let gob_col = x / GOB_W;
    let x_in_gob = x % GOB_W;

    let gob_base = block_y * block_row_stride
        + gob_col * block_height * GOB_SIZE
        + gob_row * GOB_SIZE;

    let in_gob = ((x_in_gob >> 5) & 1) * 256
        + ((y_in_gob >> 1) & 3) * 64
        + ((x_in_gob >> 4) & 1) * 32
        + (y_in_gob & 1) * 16
        + (x_in_gob & 15);

    gob_base + in_gob
}

fn write_pixel_tiled(
    rt_cpu: u64,
    px: usize,
    py: usize,
    width: u32,
    bh_log2: u32,
    r: u8, g: u8, b: u8, a: u8,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let width_bytes = (width as usize) * 4;
    let byte_x = px * 4;
    let off = tiled_offset(byte_x, py, width_bytes, bh_log2);
    mem_write(rt_cpu + off as u64, &[r, g, b, a]);
}

fn execute_clear(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let rt = &draw.rt[0];
    let gpu_va = rt_gpu_va(rt);
    if gpu_va == 0 { return; }
    let Some(rt_cpu) = mappings.cpu_address_for(gpu_va) else { return };
    let bh_log2 = block_height_log2(rt);
    let w = rt.width as usize;
    let h = rt.height as usize;
    let r = (draw.clear_color.r.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (draw.clear_color.g.clamp(0.0, 1.0) * 255.0) as u8;
    let b = (draw.clear_color.b.clamp(0.0, 1.0) * 255.0) as u8;
    let a = (draw.clear_color.a.clamp(0.0, 1.0) * 255.0) as u8;
    for py in 0..h {
        for px in 0..w {
            write_pixel_tiled(rt_cpu, px, py, rt.width, bh_log2, r, g, b, a, mem_write);
        }
    }
}

struct Vertex {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    u: f32,
    v: f32,
}

fn read_f32(mem_read: &dyn Fn(u64, &mut [u8]) -> bool, addr: u64) -> f32 {
    let mut buf = [0u8; 4];
    mem_read(addr, &mut buf);
    f32::from_le_bytes(buf)
}

fn read_u16(mem_read: &dyn Fn(u64, &mut [u8]) -> bool, addr: u64) -> u16 {
    let mut buf = [0u8; 2];
    mem_read(addr, &mut buf);
    u16::from_le_bytes(buf)
}

fn decode_attrib(
    format: u32,
    offset: u32,
    vb_cpu: u64,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    vertex_stride: u32,
    vertex_index: u32,
) -> Option<[f32; 4]> {
    let base = vb_cpu + (vertex_index * vertex_stride + offset) as u64;
    let size = format & 0x3F;
    let type_ = format >> 6;

    match (size, type_) {
        (0x01, 7) | (0x12, 7) => Some([
            read_f32(mem_read, base),
            read_f32(mem_read, base + 4),
            read_f32(mem_read, base + 8),
            read_f32(mem_read, base + 12),
        ]),
        (0x04, 7) => Some([
            read_f32(mem_read, base),
            read_f32(mem_read, base + 4),
            0.0, 1.0,
        ]),
        (0x0A, 2) => {
            let mut buf = [0u8; 4];
            mem_read(base, &mut buf);
            Some([buf[0] as f32 / 255.0, buf[1] as f32 / 255.0, buf[2] as f32 / 255.0, buf[3] as f32 / 255.0])
        }
        (0x09, 7) => {
            let u = read_u16(mem_read, base) as f32 / 32768.0 - 1.0;
            let v = read_u16(mem_read, base + 2) as f32 / 32768.0 - 1.0;
            Some([u, v, 0.0, 1.0])
        }
        _ => None,
    }
}

fn read_vertex(
    draw: &DrawCall,
    vertex_index: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Vertex {
    let mut pos = [0.0f32; 4];
    let mut col = [1.0f32; 4];
    let mut uv = [0.0f32; 2];

    for attrib in &draw.vertex_attribs {
        if attrib.format == 0 { continue; }
        let buf_idx = attrib.buffer as usize;
        if buf_idx >= draw.vertex_buffers.len() { continue; }
        let vb = &draw.vertex_buffers[buf_idx];
        let vb_gpu = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
        if vb_gpu == 0 { continue; }
        let Some(vb_cpu) = mappings.cpu_address_for(vb_gpu) else { continue };
        let stride = if vb.stride > 0 { vb.stride } else { 16 };

        let vals = match decode_attrib(attrib.format, attrib.offset, vb_cpu, mem_read, stride, vertex_index) {
            Some(v) => v,
            None => continue,
        };

        if attrib.offset == 0 {
            pos = vals;
        } else if is_float2_format(attrib.format) {
            uv = [vals[0], vals[1]];
        } else {
            col = vals;
        }
    }

    Vertex { x: pos[0], y: pos[1], r: col[0], g: col[1], b: col[2], a: col[3], u: uv[0], v: uv[1] }
}

fn viewport_transform(v: &Vertex, draw: &DrawCall, rt_w: u32, rt_h: u32) -> (f32, f32) {
    let vp = &draw.viewport;
    let (vw, vh) = (vp.width, vp.height);
    let (vx, vy) = (vp.x, vp.y);
    let (sw, sh) = (rt_w as f32, rt_h as f32);

    if vw > 0.0 && vh > 0.0 {
        let sx = v.x * (vw * 0.5) + vx + vw * 0.5;
        let sy = v.y * (vh * 0.5) + vy + vh * 0.5;
        (sx, sy)
    } else {
        ((v.x * 0.5 + 0.5) * sw, (v.y * 0.5 + 0.5) * sh)
    }
}

fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

fn rasterize_triangle(
    v0: &Vertex, v1: &Vertex, v2: &Vertex,
    draw: &DrawCall,
    rt: &RenderTarget,
    rt_cpu: u64,
    bh_log2: u32,
    tic: Option<&TicInfo>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let w = rt.width;
    let h = rt.height;
    let (x0, y0) = viewport_transform(v0, draw, w, h);
    let (x1, y1) = viewport_transform(v1, draw, w, h);
    let (x2, y2) = viewport_transform(v2, draw, w, h);

    if w == 0 || h == 0 { return; }
    let min_x = x0.min(x1).min(x2).max(0.0) as usize;
    let min_y = y0.min(y1).min(y2).max(0.0) as usize;
    let max_x = (x0.max(x1).max(x2).ceil() as usize).min(w as usize - 1);
    let max_y = (y0.max(y1).max(y2).ceil() as usize).min(h as usize - 1);
    let area = edge(x0, y0, x1, y1, x2, y2);
    if area.abs() < 1e-6 { return; }

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
            let w0 = edge(x1, y1, x2, y2, fx, fy);
            let w1 = edge(x2, y2, x0, y0, fx, fy);
            let w2 = edge(x0, y0, x1, y1, fx, fy);
            let inside = if area >= 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if !inside { continue; }

            let b0 = w0 / area;
            let b1 = w1 / area;
            let b2 = w2 / area;

            let pixel = if let Some(t) = tic {
                let u = b0 * v0.u + b1 * v1.u + b2 * v2.u;
                let v = b0 * v0.v + b1 * v1.v + b2 * v2.v;
                sample_tiled_pixel(t, u, v, mem_read)
            } else {
                let r = (b0 * v0.r + b1 * v1.r + b2 * v2.r).clamp(0.0, 1.0);
                let g = (b0 * v0.g + b1 * v1.g + b2 * v2.g).clamp(0.0, 1.0);
                let b = (b0 * v0.b + b1 * v1.b + b2 * v2.b).clamp(0.0, 1.0);
                let a = (b0 * v0.a + b1 * v1.a + b2 * v2.a).clamp(0.0, 1.0);
                [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8]
            };

            write_pixel_tiled(rt_cpu, px, py, w, bh_log2, pixel[0], pixel[1], pixel[2], pixel[3], mem_write);
        }
    }
}

fn execute_draw(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let rt = &draw.rt[0];
    let gpu_va = rt_gpu_va(rt);
    if gpu_va == 0 || rt.width == 0 || rt.height == 0 { return; }
    let Some(rt_cpu) = mappings.cpu_address_for(gpu_va) else { return };
    let bh_log2 = block_height_log2(rt);

    let has_uv_attr = draw.vertex_attribs.iter().any(|a| a.format != 0 && is_float2_format(a.format));
    let has_color_attr = draw.vertex_attribs.iter().any(|a| {
        a.format != 0 && a.offset != 0 && (a.format & 0x3F) == 0x2F
    });

    if !has_uv_attr && !has_color_attr { return; }

    let tic = if has_uv_attr { resolve_tic(draw, mappings, mem_read) } else { None };
    if has_uv_attr && tic.is_none() && !has_color_attr { return; }

    thread_local! {
        static LOG_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    LOG_COUNT.with(|c| {
        let n = c.get();
        if n < 8 {
            let vb0 = &draw.vertex_buffers[0];
            log::info!(
                "execute_draw[{}]: topo={} vcount={} idx={} first={} rt={}x{} tile={:#x} vb0={:#x} stride={} has_uv={} tic={}",
                n, draw.topology, draw.vertex_count, draw.indexed, draw.first_vertex,
                rt.width, rt.height, rt.tile_mode,
                ((vb0.address_hi as u64) << 32) | vb0.address_lo as u64, vb0.stride,
                has_uv_attr, tic.is_some()
            );
            c.set(n + 1);
        }
    });

    let count = if draw.indexed { draw.index_count } else { draw.vertex_count };
    if count == 0 { return; }

    let indices: Vec<u32> = (draw.first_vertex..draw.first_vertex + count).collect();
    let tic_ref = tic.as_ref();

    let rast = |i0: usize, i1: usize, i2: usize| {
        if i0 >= indices.len() || i1 >= indices.len() || i2 >= indices.len() { return; }
        let v0 = read_vertex(draw, indices[i0], mappings, mem_read);
        let v1 = read_vertex(draw, indices[i1], mappings, mem_read);
        let v2 = read_vertex(draw, indices[i2], mappings, mem_read);
        rasterize_triangle(&v0, &v1, &v2, draw, rt, rt_cpu, bh_log2, tic_ref, mem_read, mem_write);
    };

    match draw.topology {
        4 => {
            let mut i = 0;
            while i + 2 < indices.len() { rast(i, i + 1, i + 2); i += 3; }
        }
        5 => {
            let mut i = 0;
            while i + 2 < indices.len() { rast(i, i + 1, i + 2); i += 1; }
        }
        6 => {
            if indices.len() >= 3 {
                for i in 1..indices.len() - 1 { rast(0, i, i + 1); }
            }
        }
        7 => {
            let mut i = 0;
            while i + 3 < indices.len() {
                rast(i, i + 1, i + 2);
                rast(i, i + 2, i + 3);
                i += 4;
            }
        }
        _ => {
            let mut i = 0;
            while i + 2 < indices.len() { rast(i, i + 1, i + 2); i += 3; }
        }
    }
}
