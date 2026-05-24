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
        log::trace!("DrawTexture: texture_id {} > limit {} — skipping", dt.texture_id, dt.tic_pool_limit);
        return;
    }
    let tic_gpu_va = dt.tic_pool_gpu_va + (dt.texture_id as u64) * 0x20;
    let Some(tic_cpu) = mappings.cpu_address_for(tic_gpu_va) else {
        log::trace!("DrawTexture: TIC gpu_va {:#x} not mapped", tic_gpu_va);
        return;
    };
    let mut tic = [0u8; 32];
    if !mem_read(tic_cpu, &mut tic) {
        log::trace!("DrawTexture: TIC read failed at cpu {:#x}", tic_cpu);
        return;
    }
    let w0 = u32::from_le_bytes(tic[0..4].try_into().unwrap());
    let w1 = u32::from_le_bytes(tic[4..8].try_into().unwrap());
    let w2 = u32::from_le_bytes(tic[8..12].try_into().unwrap());
    let w3 = u32::from_le_bytes(tic[12..16].try_into().unwrap());
    let w4 = u32::from_le_bytes(tic[16..20].try_into().unwrap());
    let w5 = u32::from_le_bytes(tic[20..24].try_into().unwrap());
    let w6 = u32::from_le_bytes(tic[24..28].try_into().unwrap());
    let w7 = u32::from_le_bytes(tic[28..32].try_into().unwrap());

    let format = w0 & 0x7F;
    let src_gpu_va = (w1 as u64) | (((w2 & 0xFFFF) as u64) << 32);

    let tile_height_log2 = (w2 >> 22) & 0x7;

    let tic_w_minus_1 = w4 & 0xFFFF;
    let tic_h_minus_1 = w5 & 0xFFFF;
    let mut tex_width = (tic_w_minus_1 + 1) as usize;
    let mut tex_height = (tic_h_minus_1 + 1) as usize;
    if tex_width == 1 || tex_width > 8192 {
        tex_width = dt.dst_width as usize;
    }
    if tex_height == 1 || tex_height > 8192 {
        tex_height = dt.dst_height as usize;
    }

    if maxwell_dma.draw_texture_blits < 4 {
        log::info!(
            "DrawTexture[{}]: tex_id={} tic_pool_va={:#x} tic_va={:#x} tic_cpu={:#x} TIC=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
            maxwell_dma.draw_texture_blits, dt.texture_id, dt.tic_pool_gpu_va, tic_gpu_va, tic_cpu,
            w0, w1, w2, w3, w4, w5, w6, w7
        );
        log::info!(
            "  parsed: src_gpu={:#x} fmt={:#x} src={}x{} bh={}",
            src_gpu_va, format, tex_width, tex_height, tile_height_log2
        );

        if src_gpu_va == 0 && dt.texture_id < 8 {
            for probe_id in 0..8 {
                let probe_va = dt.tic_pool_gpu_va + probe_id * 0x20;
                if let Some(pcpu) = mappings.cpu_address_for(probe_va) {
                    let mut p = [0u8; 16];
                    if mem_read(pcpu, &mut p) {
                        let p0 = u32::from_le_bytes(p[0..4].try_into().unwrap());
                        let p1 = u32::from_le_bytes(p[4..8].try_into().unwrap());
                        let p2 = u32::from_le_bytes(p[8..12].try_into().unwrap());
                        let p3 = u32::from_le_bytes(p[12..16].try_into().unwrap());
                        if p0 | p1 | p2 | p3 != 0 {
                            log::info!("  TIC probe[{}]: {:08x} {:08x} {:08x} {:08x}", probe_id, p0, p1, p2, p3);
                        }
                    }
                }
            }
        }
    }

    let Some(src_cpu) = mappings.cpu_address_for(src_gpu_va) else {
        log::trace!("DrawTexture: src gpu_va {:#x} not mapped", src_gpu_va);
        return;
    };

    if maxwell_dma.last_tiled_dst_cpu == 0 {
        log::trace!("DrawTexture: no last_tiled_dst_cpu — skipping (need a prior MaxwellDma blit to establish destination)");
        return;
    }
    let dst_cpu = maxwell_dma.last_tiled_dst_cpu;
    let dst_bh_log2 = maxwell_dma.last_tiled_dst_bh_log2;
    let dst_w_bytes = maxwell_dma.last_tiled_dst_stride.max(1) as usize;
    let dst_h = maxwell_dma.last_tiled_dst_height.max(1) as usize;

    let blit_w = (dt.dst_width as usize).min(tex_width).min(dst_w_bytes / 4);
    let blit_h = (dt.dst_height as usize).min(tex_height).min(dst_h);

    if maxwell_dma.draw_texture_blits < 8 {
        log::info!(
            "DrawTexture[{}]: tex_id={} TIC=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
            maxwell_dma.draw_texture_blits, dt.texture_id, w0, w1, w2, w3, w4, w5, w6, w7
        );
        log::info!(
            "  → src_gpu={:#x} src_cpu={:#x} fmt={:#x} src={}x{} bh={} dst_cpu={:#x} dst_bh={} dst_stride={} dst_h={} rect=({},{} {}x{})",
            src_gpu_va, src_cpu, format, tex_width, tex_height, tile_height_log2,
            dst_cpu, dst_bh_log2, dst_w_bytes, dst_h,
            dt.dst_x, dt.dst_y, blit_w, blit_h
        );
    }

    let src_pitch = tex_width * 4;
    let linear = unswizzle_block_linear_local(
        src_cpu, src_pitch, tex_width, tex_height, tile_height_log2, mem_read,
    );

    for y in 0..blit_h {
        for x in 0..blit_w {
            let off = y * src_pitch + x * 4;
            if off + 4 > linear.len() { continue; }
            let r = linear[off];
            let g = linear[off + 1];
            let b = linear[off + 2];
            let a = linear[off + 3];
            let dst_x = dt.dst_x as usize + x;
            let dst_y = dt.dst_y as usize + y;
            write_tiled_pixel_bytes(dst_cpu, dst_x, dst_y, dst_w_bytes, dst_bh_log2, [r, g, b, a], mem_write);
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

fn decode_fs_tex_ids_cached(
    fs_gpu_va: u64,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Vec<nexium_shader::FsTexId> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<u64, Vec<nexium_shader::FsTexId>>> = RefCell::new(HashMap::new());
    }
    if fs_gpu_va == 0 {
        return Vec::new();
    }
    if let Some(cached) = CACHE.with(|c| c.borrow().get(&fs_gpu_va).cloned()) {
        return cached;
    }
    let Some(fs_cpu) = mappings.cpu_address_for(fs_gpu_va) else {
        return Vec::new();
    };

    const FS_SPH_BYTES: u64 = 48;
    const CODE_BYTES: usize = 8 * 1024;
    let mut code = vec![0u8; CODE_BYTES];
    if !mem_read(fs_cpu + FS_SPH_BYTES, &mut code) {
        return Vec::new();
    }
    let ids = nexium_shader::extract_fs_tex_ids(&code, 15);
    CACHE.with(|c| { c.borrow_mut().insert(fs_gpu_va, ids.clone()); });
    ids
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

#[derive(Default)]
struct BlitStats {
    total: u32,
    rt_invalid: u32,
    rt_unmapped: u32,
    tic_pool_zero: u32,
    no_best: u32,
    blit_read_fail: u32,
    cb_unmapped: u32,
    no_cb_tracked: u32,
    fs_tex_ids_empty: u32,
    fs_tex_ids_found: u32,
    success_pool_match: u32,
    success_pool_fallback: u32,
    success_bindless: u32,
    last_src_cksum: u32,
    src_cksum_changed: u32,
    src_cksum_same: u32,
}

fn try_blit_bound_texture(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> bool {
    thread_local! {
        static BLIT_LOG_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static BLIT_STATS: std::cell::RefCell<BlitStats> = std::cell::RefCell::new(BlitStats::default());
    }
    let log_this = BLIT_LOG_COUNT.with(|c| {
        let n = c.get();
        if n < 6 { c.set(n + 1); true } else { false }
    });
    BLIT_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.total = s.total.wrapping_add(1);
        if s.total % 60 == 0 {
            log::info!(
                "try_blit summary @{}: rt_invalid={} rt_unmap={} tic_pool0={} no_best={} blit_fail={} cb_unmap={} no_cb={} fs_ids_empty={} fs_ids_found={} ok_pool={} ok_pool_fb={} ok_bindless={} src_changed={} src_same={}",
                s.total, s.rt_invalid, s.rt_unmapped, s.tic_pool_zero, s.no_best, s.blit_read_fail,
                s.cb_unmapped, s.no_cb_tracked, s.fs_tex_ids_empty, s.fs_tex_ids_found,
                s.success_pool_match, s.success_pool_fallback, s.success_bindless,
                s.src_cksum_changed, s.src_cksum_same,
            );
        }
    });

    let rt = &draw.rt[0];
    let rt_gpu = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    if log_this {
        log::info!(
            "try_blit_bound_texture: enter rt_gpu={:#x} rt={}x{} tile={:#x} tic_pool={:#x} limit={}",
            rt_gpu, rt.width, rt.height, rt.tile_mode, draw.tic_pool_gpu_va, draw.tic_pool_limit
        );
    }
    if rt_gpu == 0 || rt.width == 0 || rt.height == 0 {
        BLIT_STATS.with(|s| s.borrow_mut().rt_invalid += 1);
        if log_this { log::info!("  bail: rt invalid"); }
        return false;
    }
    let Some(rt_cpu) = mappings.cpu_address_for(rt_gpu) else {
        BLIT_STATS.with(|s| s.borrow_mut().rt_unmapped += 1);
        if log_this { log::info!("  bail: rt_gpu {:#x} not mapped", rt_gpu); }
        return false;
    };
    let rt_bh_log2 = (rt.tile_mode >> 4) & 0xF;
    let rt_w = rt.width as usize;
    let rt_h = rt.height as usize;

    if draw.tic_pool_gpu_va == 0 {
        BLIT_STATS.with(|s| s.borrow_mut().tic_pool_zero += 1);
        if log_this { log::info!("  bail: tic_pool_gpu_va is 0 (TIC pool registers never reached Maxwell3D)"); }
        return false;
    }

    let mut best: Option<(u32, u32, u32, u64, u32)> = None;
    let mut path: u8 = 0;
    let limit = (draw.tic_pool_limit + 1).min(128);

    if log_this {
        if let Some(pool_cpu) = mappings.cpu_address_for(draw.tic_pool_gpu_va) {
            for probe in 0..4u64 {
                let mut tic = [0u8; 32];
                if mem_read(pool_cpu + probe * 0x20, &mut tic) {
                    let w: [u32; 8] = [
                        u32::from_le_bytes(tic[ 0.. 4].try_into().unwrap()),
                        u32::from_le_bytes(tic[ 4.. 8].try_into().unwrap()),
                        u32::from_le_bytes(tic[ 8..12].try_into().unwrap()),
                        u32::from_le_bytes(tic[12..16].try_into().unwrap()),
                        u32::from_le_bytes(tic[16..20].try_into().unwrap()),
                        u32::from_le_bytes(tic[20..24].try_into().unwrap()),
                        u32::from_le_bytes(tic[24..28].try_into().unwrap()),
                        u32::from_le_bytes(tic[28..32].try_into().unwrap()),
                    ];
                    log::info!("  TIC dump [{}] @cpu={:#x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
                        probe, pool_cpu + probe * 0x20, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                }
            }
        }
    }

    for idx in 0..limit {
        let tic_gpu = draw.tic_pool_gpu_va + (idx as u64) * 0x20;
        let Some(tic_cpu) = mappings.cpu_address_for(tic_gpu) else { continue };
        let mut tic = [0u8; 32];
        if !mem_read(tic_cpu, &mut tic) { continue; }
        let w0 = u32::from_le_bytes(tic[0..4].try_into().unwrap());
        let w1 = u32::from_le_bytes(tic[4..8].try_into().unwrap());
        let w2 = u32::from_le_bytes(tic[8..12].try_into().unwrap());
        let w3 = u32::from_le_bytes(tic[12..16].try_into().unwrap());
        let w4 = u32::from_le_bytes(tic[16..20].try_into().unwrap());
        let w5 = u32::from_le_bytes(tic[20..24].try_into().unwrap());
        if w0 == 0 && w1 == 0 && w2 == 0 { continue; }

        let src_gpu = (w1 as u64) | (((w2 & 0xFFFF) as u64) << 32);
        if src_gpu == 0 { continue; }
        let Some(src_cpu) = mappings.cpu_address_for(src_gpu) else { continue; };

        let hdr_version = (w2 >> 21) & 0x7;
        let bh_log2 = (w3 >> 3) & 0x7;
        let tex_w = ((w4 & 0xFFFF) + 1) as u32;
        let tex_height = ((w5 & 0xFFFF) + 1) as u32;

        let matches_rt = tex_w == rt.width && tex_height == rt.height;
        if matches_rt {
            best = Some((idx, tex_w, tex_height, src_cpu, bh_log2));
            path = 1;
            log::trace!(
                "try_blit_bound_texture: chose TIC[{}] hdr_v={} src_gpu={:#x} src_cpu={:#x} {}x{} bh={}",
                idx, hdr_version, src_gpu, src_cpu, tex_w, tex_height, bh_log2
            );
            break;
        }

        if best.is_none() {
            best = Some((idx, tex_w, tex_height, src_cpu, bh_log2));
            path = 2;
        }
    }

    if best.is_none() {

        let fs_tex_ids = decode_fs_tex_ids_cached(
            draw.fs_shader_gpu_va,
            mappings,
            mem_read,
        );
        BLIT_STATS.with(|s| {
            let mut s = s.borrow_mut();
            if fs_tex_ids.is_empty() { s.fs_tex_ids_empty += 1; } else { s.fs_tex_ids_found += 1; }
        });
        if log_this {
            log::info!(
                "  TIC pool empty → SASS-decoded FS shader gpu={:#x}: fs_tex_ids = {:?}",
                draw.fs_shader_gpu_va, fs_tex_ids
            );
        }
        if draw.fs_bindless_cb_addr != 0 {
            if let Some(cb_cpu) = mappings.cpu_address_for(draw.fs_bindless_cb_addr) {
                for id in &fs_tex_ids {
                    let off = match *id {
                        nexium_shader::FsTexId::BindlessCbufOffset(o) => o as usize,

                        nexium_shader::FsTexId::ImmediateTic(_) => continue,
                    };
                    if off + 4 > (draw.fs_bindless_cb_size as usize) { continue; }
                    let mut handle_bytes = [0u8; 4];
                    if !mem_read(cb_cpu + off as u64, &mut handle_bytes) { continue; }
                    let handle = u32::from_le_bytes(handle_bytes);
                    let candidate_tic = handle & 0xFFFFF;
                    if candidate_tic == 0 || candidate_tic > draw.tic_pool_limit { continue; }
                    let cand_va = draw.tic_pool_gpu_va + (candidate_tic as u64) * 0x20;
                    let Some(cand_cpu) = mappings.cpu_address_for(cand_va) else { continue };
                    let mut tic = [0u8; 32];
                    if !mem_read(cand_cpu, &mut tic) { continue; }
                    let t1 = u32::from_le_bytes(tic[4..8].try_into().unwrap());
                    let t2 = u32::from_le_bytes(tic[8..12].try_into().unwrap());
                    let cand_src_va = (t1 as u64) | (((t2 & 0xFFFF) as u64) << 32);
                    if cand_src_va == 0 { continue; }
                    let Some(cand_src_cpu) = mappings.cpu_address_for(cand_src_va) else { continue };
                    let t3 = u32::from_le_bytes(tic[12..16].try_into().unwrap());
                    let t4 = u32::from_le_bytes(tic[16..20].try_into().unwrap());
                    let t5 = u32::from_le_bytes(tic[20..24].try_into().unwrap());
                    let cand_bh = (t3 >> 3) & 0x7;
                    let cand_w = (t4 & 0xFFFF) + 1;
                    let cand_h = (t5 & 0xFFFF) + 1;
                    if log_this {
                        log::info!(
                            "    cbuf+{:#x} = {:#010x} → TIC[{}] src={:#x} {}x{} bh={}",
                            off, handle, candidate_tic, cand_src_va, cand_w, cand_h, cand_bh
                        );
                    }
                    if cand_w == rt.width && cand_h == rt.height {
                        best = Some((candidate_tic, cand_w, cand_h, cand_src_cpu, cand_bh));
                        path = 3;
                        break;
                    }
                    if best.is_none() {
                        best = Some((candidate_tic, cand_w, cand_h, cand_src_cpu, cand_bh));
                        path = 3;
                    }
                }
            } else {
                BLIT_STATS.with(|s| s.borrow_mut().cb_unmapped += 1);
                if log_this { log::info!("  fs_bindless_cb gpu={:#x} not mapped", draw.fs_bindless_cb_addr); }
            }
        } else {
            BLIT_STATS.with(|s| s.borrow_mut().no_cb_tracked += 1);
            if log_this { log::info!("  no FS bindless cbuf tracked"); }
        }
    }
    let Some((idx, tex_w, tex_h, src_cpu, src_bh_log2)) = best else {
        BLIT_STATS.with(|s| s.borrow_mut().no_best += 1);
        return false;
    };
    if log_this {
        log::info!(
            "  picked TIC[{}] src_cpu={:#x} tex={}x{} src_bh={} → blit to rt_cpu={:#x} rt={}x{} rt_bh={}",
            idx, src_cpu, tex_w, tex_h, src_bh_log2, rt_cpu, rt_w, rt_h, rt_bh_log2
        );
    }

    if src_bh_log2 == rt_bh_log2 {
        let tiled_size = {
            const GOB_W: usize = 64;
            const GOB_H: usize = 8;
            const GOB_SIZE: usize = 512;
            let bh = 1usize << rt_bh_log2;
            let width_bytes = rt_w * 4;
            let gobs_per_row = (width_bytes + GOB_W - 1) / GOB_W;
            let rows_per_block = bh * GOB_H;
            let block_rows = (rt_h + rows_per_block - 1) / rows_per_block;
            gobs_per_row * block_rows * bh * GOB_SIZE
        };
        let mut buf = vec![0u8; tiled_size];
        if !mem_read(src_cpu, &mut buf) {
            BLIT_STATS.with(|s| s.borrow_mut().blit_read_fail += 1);
            return false;
        }
        let mut src_cksum: u32 = 0;
        for chunk in buf.chunks_exact(4).step_by(64) {
            let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            src_cksum = src_cksum.wrapping_mul(31).wrapping_add(v);
        }
        let mut rt_pre = vec![0u8; tiled_size];
        let rt_pre_cksum: u32 = if mem_read(rt_cpu, &mut rt_pre) {
            let mut c: u32 = 0;
            for chunk in rt_pre.chunks_exact(4).step_by(64) {
                let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                c = c.wrapping_mul(31).wrapping_add(v);
            }
            c
        } else {
            0
        };
        BLIT_STATS.with(|s| {
            let mut s = s.borrow_mut();
            if s.last_src_cksum != 0 {
                if s.last_src_cksum == src_cksum { s.src_cksum_same += 1; } else { s.src_cksum_changed += 1; }
            }
            s.last_src_cksum = src_cksum;
            if s.total <= 6 || s.total % 60 == 0 {
                log::info!(
                    "  blit cksum: src({:#x})={:#010x} rt_pre({:#x})={:#010x} differ={}",
                    src_cpu, src_cksum, rt_cpu, rt_pre_cksum, src_cksum != rt_pre_cksum
                );
            }
        });
        let ok = mem_write(rt_cpu, &buf);
        if ok {
            BLIT_STATS.with(|s| {
                let mut s = s.borrow_mut();
                match path {
                    1 => s.success_pool_match += 1,
                    2 => s.success_pool_fallback += 1,
                    3 => s.success_bindless += 1,
                    _ => {}
                }
            });
        }
        return ok;
    }

    let width_bytes = rt_w * 4;
    for y in 0..rt_h {
        for x in 0..rt_w {
            let src_off = tiled_offset(x * 4, y, width_bytes, src_bh_log2);
            let dst_off = tiled_offset(x * 4, y, width_bytes, rt_bh_log2);
            let mut pixel = [0u8; 4];
            if !mem_read(src_cpu + src_off as u64, &mut pixel) { continue; }
            mem_write(rt_cpu + dst_off as u64, &pixel);
        }
    }
    BLIT_STATS.with(|s| {
        let mut s = s.borrow_mut();
        match path {
            1 => s.success_pool_match += 1,
            2 => s.success_pool_fallback += 1,
            3 => s.success_bindless += 1,
            _ => {}
        }
    });
    true
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
    let pixel = [r, g, b, a];
    mem_write(rt_cpu + off as u64, &pixel);
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

struct Vertex {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
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
    let size = (format >> 0) & 0x3F;
    let type_ = (format >> 6) & 0x7;

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
            let v = read_f32(mem_read, base + 2) as f32 / 32768.0 - 1.0;
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
        } else {
            col = vals;
        }
    }

    Vertex { x: pos[0], y: pos[1], r: col[0], g: col[1], b: col[2], a: col[3] }
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
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let w = rt.width;
    let h = rt.height;
    let (x0, y0) = viewport_transform(v0, draw, w, h);
    let (x1, y1) = viewport_transform(v1, draw, w, h);
    let (x2, y2) = viewport_transform(v2, draw, w, h);

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
            let r = (b0 * v0.r + b1 * v1.r + b2 * v2.r).clamp(0.0, 1.0);
            let g = (b0 * v0.g + b1 * v1.g + b2 * v2.g).clamp(0.0, 1.0);
            let b = (b0 * v0.b + b1 * v1.b + b2 * v2.b).clamp(0.0, 1.0);
            let a = (b0 * v0.a + b1 * v1.a + b2 * v2.a).clamp(0.0, 1.0);

            write_pixel_tiled(
                rt_cpu, px, py, w, bh_log2,
                (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8,
                mem_write,
            );
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

    thread_local! {
        static EXEC_DRAW_LOG_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    EXEC_DRAW_LOG_COUNT.with(|c| {
        let n = c.get();
        if n < 4 {
            let vb0 = &draw.vertex_buffers[0];
            let vb0_gpu = ((vb0.address_hi as u64) << 32) | vb0.address_lo as u64;
            let attr0 = &draw.vertex_attribs[0];
            log::info!(
                "execute_draw[{}]: topology={} vcount={} indexed={} first={} | rt[0] gpu={:#x} {}x{} fmt={:#x} tile={:#x} | vb[0] gpu={:#x} stride={} | attr[0] buf={} off={} fmt={:#x}",
                n, draw.topology, draw.vertex_count, draw.indexed, draw.first_vertex,
                gpu_va, rt.width, rt.height, rt.format, rt.tile_mode,
                vb0_gpu, vb0.stride, attr0.buffer, attr0.offset, attr0.format,
            );
            c.set(n + 1);
        }
    });

    let has_color_attr = draw.vertex_attribs.iter().any(|a| {
        a.format != 0 && a.offset != 0 && (a.format & 0x3F) == 0x2F
    });
    if !has_color_attr && draw.vertex_count == 6 && (draw.topology == 4 || draw.topology == 5) {
        if try_blit_bound_texture(draw, mappings, mem_read, mem_write) {
            return;
        }

        return;
    }
    if !has_color_attr {
        return;
    }

    if gpu_va == 0 || rt.width == 0 || rt.height == 0 { return; }
    let Some(rt_cpu) = mappings.cpu_address_for(gpu_va) else { return };
    let bh_log2 = block_height_log2(rt);

    let count = if draw.indexed { draw.index_count } else { draw.vertex_count };
    if count == 0 { return; }

    let indices: Vec<u32> = if draw.indexed {

        (draw.first_vertex..draw.first_vertex + count).collect()
    } else {
        (draw.first_vertex..draw.first_vertex + count).collect()
    };

    match draw.topology {
        4 => {

            let mut i = 0;
            while i + 2 < indices.len() {
                let v0 = read_vertex(draw, indices[i], mappings, mem_read);
                let v1 = read_vertex(draw, indices[i+1], mappings, mem_read);
                let v2 = read_vertex(draw, indices[i+2], mappings, mem_read);
                rasterize_triangle(&v0, &v1, &v2, draw, rt, rt_cpu, bh_log2, mem_write);
                i += 3;
            }
        }
        5 => {

            let mut i = 0;
            while i + 2 < indices.len() {
                let v0 = read_vertex(draw, indices[i], mappings, mem_read);
                let v1 = read_vertex(draw, indices[i+1], mappings, mem_read);
                let v2 = read_vertex(draw, indices[i+2], mappings, mem_read);
                rasterize_triangle(&v0, &v1, &v2, draw, rt, rt_cpu, bh_log2, mem_write);
                i += 1;
            }
        }
        6 => {

            if indices.len() >= 3 {
                let pivot = read_vertex(draw, indices[0], mappings, mem_read);
                for i in 1..indices.len()-1 {
                    let v1 = read_vertex(draw, indices[i], mappings, mem_read);
                    let v2 = read_vertex(draw, indices[i+1], mappings, mem_read);
                    rasterize_triangle(&pivot, &v1, &v2, draw, rt, rt_cpu, bh_log2, mem_write);
                }
            }
        }
        7 => {

            let mut i = 0;
            while i + 3 < indices.len() {
                let v0 = read_vertex(draw, indices[i], mappings, mem_read);
                let v1 = read_vertex(draw, indices[i+1], mappings, mem_read);
                let v2 = read_vertex(draw, indices[i+2], mappings, mem_read);
                let v3 = read_vertex(draw, indices[i+3], mappings, mem_read);
                rasterize_triangle(&v0, &v1, &v2, draw, rt, rt_cpu, bh_log2, mem_write);
                rasterize_triangle(&v0, &v2, &v3, draw, rt, rt_cpu, bh_log2, mem_write);
                i += 4;
            }
        }
        _ => {

            let mut i = 0;
            while i + 2 < indices.len() {
                let v0 = read_vertex(draw, indices[i], mappings, mem_read);
                let v1 = read_vertex(draw, indices[i+1], mappings, mem_read);
                let v2 = read_vertex(draw, indices[i+2], mappings, mem_read);
                rasterize_triangle(&v0, &v1, &v2, draw, rt, rt_cpu, bh_log2, mem_write);
                i += 3;
            }
        }
    }
}
