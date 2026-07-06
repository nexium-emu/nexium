use ash::vk;
use std::sync::Arc;

use super::engines::maxwell3d::{DrawCall, RenderTarget, VertexBuffer};
use super::engines::Maxwell3D;
use super::GpuMappings;

use nexium_gpu::draw::{
    BlendAttachmentState, BlendState, DepthState, DrawState, Maxwell3dDrawCall, VertexAttr,
    VertexBinding, VertexBufferBinding, VertexLayout,
};
use nexium_gpu::rt_cache::RtKey;

const SPH_SIZE: usize = 0x50;
const MAX_SASS_BYTES: usize = 16 * 1024;
const PACKED_CBUF_SLOTS: usize = 32;
const PACKED_CBUF_SLOT_SIZE: usize = 2048;

fn nvprof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_NVDRV_PROFILE").is_ok())
}

fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn next_gpu_op_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static GPU_OP_SEQ: AtomicU64 = AtomicU64::new(0);
    GPU_OP_SEQ.fetch_add(1, Ordering::Relaxed)
}

fn unimplemented_samples(cfg: &nexium_shader::Cfg) -> Vec<String> {
    let mut samples = Vec::new();
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            if let nexium_shader::IrOp::Unimplemented { opcode, raw } = inst.op {
                samples.push(format!("{:?}:{:#x}", opcode, raw));
                if samples.len() >= 8 {
                    return samples;
                }
            }
        }
    }
    samples
}

fn collect_cbuf_reads(cfg: &nexium_shader::Cfg, stage_base: u32) -> Vec<(u32, u32)> {
    let mut reads = Vec::new();
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            if let nexium_shader::IrOp::LoadCbuf {
                binding,
                byte_offset,
            } = inst.op
            {
                reads.push((stage_base + ((binding as u32) & 0xF), byte_offset));
            }
        }
    }
    reads.sort_unstable();
    reads.dedup();
    if reads.len() > 32 {
        reads.truncate(32);
    }
    reads
}

fn tic_can_alias_render_target(format: nexium_gpu::texture::TicFormat) -> bool {
    matches!(
        format,
        nexium_gpu::texture::TicFormat::R32G32B32A32
            | nexium_gpu::texture::TicFormat::A8B8G8R8
            | nexium_gpu::texture::TicFormat::A2B10G10R10
            | nexium_gpu::texture::TicFormat::R8G8B8A8
            | nexium_gpu::texture::TicFormat::R5G6B5
            | nexium_gpu::texture::TicFormat::A1R5G5B5
            | nexium_gpu::texture::TicFormat::A4R4G4B4
            | nexium_gpu::texture::TicFormat::R8
            | nexium_gpu::texture::TicFormat::R8G8
            | nexium_gpu::texture::TicFormat::R16
            | nexium_gpu::texture::TicFormat::R16G16
            | nexium_gpu::texture::TicFormat::R32
            | nexium_gpu::texture::TicFormat::Z32
            | nexium_gpu::texture::TicFormat::B10G11R11
            | nexium_gpu::texture::TicFormat::Unknown(3)
            | nexium_gpu::texture::TicFormat::Unknown(47)
    )
}

pub fn enqueue_draws(
    draws: &[DrawCall],
    batch: &mut Vec<Maxwell3dDrawCall>,
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    renderer: &Arc<nexium_gpu::Renderer>,
    maxwell_dma: &mut super::engines::MaxwellDma,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    for draw in draws {
        if !draw.is_clear {
            if render_enable_needs_ordered_read(draw) {
                flush_accum(batch, renderer, mappings, mem_read);
            }
            if !render_enabled(draw, mappings, mem_read) {
                continue;
            }
        }
        if draw.draw_texture.is_some() {
            flush_accum(batch, renderer, mappings, mem_read);
            match prepare_draw_texture_job(draw, mappings, mem_read) {
                Ok(job) => {
                    if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
                        let r = renderer.clone();
                        rt_thread.submit(Box::new(move || {
                            if let Err(e) = execute_draw_texture_job(&r, &job) {
                                log::debug!("vk_dispatch: DrawTexture failed: {}", e);
                            }
                        }));
                    } else if let Err(e) = execute_draw_texture_job(renderer, &job) {
                        log::debug!("vk_dispatch: DrawTexture failed: {}", e);
                    }
                }
                Err(e) => {
                    log::debug!("vk_dispatch: DrawTexture prepare failed: {}", e);
                    super::engines::sw_renderer::execute_draws(
                        std::slice::from_ref(draw),
                        mappings,
                        maxwell_dma,
                        mem_read,
                        mem_write,
                    );
                }
            }
            continue;
        }
        if draw.is_clear {
            flush_accum(batch, renderer, mappings, mem_read);
            if let Err(e) = execute_one(draw, mappings, maxwell, renderer, mem_read) {
                log::debug!("vk_dispatch: clear failed: {}", e);
            }
            continue;
        }
        match execute_one(draw, mappings, maxwell, renderer, mem_read) {
            Ok(None) => {}
            Ok(Some(call)) => {
                if batch.last().is_some_and(|last| {
                    last.rt_key != call.rt_key
                        || last.color_rt_keys != call.color_rt_keys
                        || last.color_rt_formats != call.color_rt_formats
                        || last.depth_key != call.depth_key
                }) {
                    flush_accum(batch, renderer, mappings, mem_read);
                }
                batch.push(call);
                if batch.len() >= 256 {
                    flush_accum(batch, renderer, mappings, mem_read);
                }
            }
            Err(e) => {
                flush_accum(batch, renderer, mappings, mem_read);
                log::debug!("vk_dispatch: sw fallback: {}", e);
                super::engines::sw_renderer::execute_draws(
                    std::slice::from_ref(draw),
                    mappings,
                    maxwell_dma,
                    mem_read,
                    mem_write,
                );
            }
        }
    }
}

fn render_enable_needs_ordered_read(draw: &DrawCall) -> bool {
    draw.render_enable_override == 0 && matches!(draw.render_enable_mode, 2 | 3 | 4)
}

fn render_enabled(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> bool {
    let enabled = match draw.render_enable_override {
        1 => true,
        2 => false,
        _ => match draw.render_enable_mode {
            0 => false,
            1 => true,
            2 | 3 | 4 => {
                let Some(cpu) = mappings.cpu_address_for(draw.render_enable_addr) else {
                    log_render_enable_miss(draw, "unmapped");
                    return true;
                };
                let mut b = [0u8; 24];
                if !mem_read(cpu, &mut b) {
                    log_render_enable_miss(draw, "read-failed");
                    return true;
                }
                let initial_sequence = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let initial_mode = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let current_sequence = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
                let current_mode = u32::from_le_bytes([b[20], b[21], b[22], b[23]]);
                match draw.render_enable_mode {
                    2 => initial_sequence != 0 && initial_mode != 0,
                    3 => initial_sequence == current_sequence && initial_mode == current_mode,
                    4 => initial_sequence != current_sequence || initial_mode != current_mode,
                    _ => true,
                }
            }
            _ => {
                log_render_enable_miss(draw, "unknown-mode");
                true
            }
        },
    };
    if !enabled {
        log_render_enable_skip(draw);
    }
    enabled
}

fn log_render_enable_miss(draw: &DrawCall, reason: &str) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        log::debug!(
            "[render-enable] {} addr={:#x} mode={} override={} fs={:#x}",
            reason,
            draw.render_enable_addr,
            draw.render_enable_mode,
            draw.render_enable_override,
            draw.fs_shader_gpu_va
        );
    }
}

fn log_render_enable_skip(draw: &DrawCall) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 {
        log::debug!(
            "[render-enable] skip addr={:#x} mode={} override={} fs={:#x}",
            draw.render_enable_addr,
            draw.render_enable_mode,
            draw.render_enable_override,
            draw.fs_shader_gpu_va
        );
    }
}

pub fn flush_accum(
    batch: &mut Vec<Maxwell3dDrawCall>,
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    if batch.is_empty() {
        return;
    }
    let profile = nvprof_enabled();
    let t0 = std::time::Instant::now();
    let rt_thread = crate::render_thread::maybe_render_thread();
    let read_guest = |gpu_va: u64, len: usize| -> Option<Vec<u8>> {
        let cpu = mappings.cpu_address_for(gpu_va)?;
        let mut buf = vec![0u8; len];
        if mem_read(cpu, &mut buf) {
            Some(buf)
        } else {
            None
        }
    };
    let before = batch.len();
    let flushed = flush_batch(batch, renderer, rt_thread, &read_guest);
    if profile {
        log::warn!(
            "[nvprof] flush_accum before={} flushed={} total_ms={:.3}",
            before,
            flushed,
            elapsed_ms(t0)
        );
    }
}

fn rt_keys_uniform(calls: &[Maxwell3dDrawCall]) -> bool {
    match calls.first() {
        Some(f) => calls.iter().all(|c| {
            c.rt_key == f.rt_key
                && c.color_rt_keys == f.color_rt_keys
                && c.color_rt_formats == f.color_rt_formats
                && c.depth_key == f.depth_key
        }),
        None => true,
    }
}

fn flush_batch(
    batch: &mut Vec<Maxwell3dDrawCall>,
    renderer: &Arc<nexium_gpu::Renderer>,
    rt_thread: Option<&crate::render_thread::RenderThread>,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) -> usize {
    if batch.is_empty() {
        return 0;
    }
    let n = batch.len();
    let uniform = rt_keys_uniform(batch);
    match rt_thread {
        Some(rt) => {
            if uniform {
                submit_draw_batch_async(batch, renderer, rt, read_guest);
            } else {
                for c in batch.iter() {
                    submit_draw_batch_async(std::slice::from_ref(c), renderer, rt, read_guest);
                }
            }
        }
        None => {
            if uniform {
                if let Err(e) = renderer.execute_draws(batch, read_guest) {
                    log::debug!("vk_dispatch: execute_draws failed: {}", e);
                }
            } else {
                for c in batch.iter() {
                    let _ = renderer.execute_draw(c, read_guest);
                }
            }
        }
    }
    batch.clear();
    n
}

#[derive(Clone)]
struct DrawTextureJob {
    nvmap_id: u32,
    width: u32,
    height: u32,
    gpu_va: u64,
    dst_x: f32,
    dst_y: f32,
    dst_width: f32,
    dst_height: f32,
    src_x: f32,
    src_y: f32,
    src_width: f32,
    src_height: f32,
    src_rgba: Vec<u8>,
    src_tex_width: u32,
    src_tex_height: u32,
}

fn prepare_draw_texture_job(
    draw: &DrawCall,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<DrawTextureJob, String> {
    let dt = draw
        .draw_texture
        .ok_or_else(|| "DrawTexture call missing".to_string())?;
    if dt.texture_id > dt.tic_pool_limit {
        return Err(format!(
            "texture id {} beyond TIC limit {}",
            dt.texture_id, dt.tic_pool_limit
        ));
    }
    let rt_slot = draw_color_rt_slot(draw);
    let rt = &draw.rt[rt_slot];
    if rt.width == 0 || rt.height == 0 {
        return Err(format!("RT[{}] has zero extent", rt_slot));
    }
    let rt_gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings
        .nvmap_id_for(rt_gpu_va)
        .ok_or_else(|| format!("RT gpu_va={:#x} not mapped", rt_gpu_va))?;
    let tic_addr = dt
        .tic_pool_gpu_va
        .wrapping_add((dt.texture_id as u64).saturating_mul(32));
    let tic_cpu = mappings
        .cpu_address_for(tic_addr)
        .ok_or_else(|| format!("TIC gpu_va={:#x} not mapped", tic_addr))?;
    let mut tic_raw = [0u8; 32];
    if !mem_read(tic_cpu, &mut tic_raw) {
        return Err(format!("TIC read failed at {:#x}", tic_cpu));
    }
    let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw)
        .ok_or_else(|| format!("TIC parse failed for texture {}", dt.texture_id))?;
    let pitch_size = tic.format.linear_size(tic.width, tic.height);
    let read_size = if tic.is_block_linear {
        tic.format
            .block_linear_size(tic.width, tic.height, tic.block_height_log2)
            .max(pitch_size)
    } else {
        pitch_size
    };
    let tex_cpu = mappings
        .cpu_address_for(tic.gpu_va)
        .ok_or_else(|| format!("texture gpu_va={:#x} not mapped", tic.gpu_va))?;
    let mut raw = vec![0u8; read_size];
    if !mem_read(tex_cpu, &mut raw) {
        return Err(format!("texture read failed at {:#x}", tex_cpu));
    }
    let linear = if tic.is_block_linear {
        let (storage_width, storage_height, bpp) = tic.format.storage_extent(tic.width, tic.height);
        nexium_gpu::texture::unswizzle_block_linear(
            &raw,
            storage_width,
            storage_height,
            bpp,
            tic.block_height_log2,
        )
    } else {
        raw
    };
    let mut src_rgba =
        nexium_gpu::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
    apply_swizzle_rgba(&mut src_rgba, tic.swizzle);
    if std::env::var_os("NEXIUM_DRAW_TEXTURE_LOG").is_some() {
        log::warn!(
            "DrawTexture prepare: rt={} {}x{} dst=({},{} {}x{}) src_tex={} {}x{} fmt={:?} bl={} src=({},{} {}x{})",
            nvmap_id,
            rt.width,
            rt.height,
            dt.dst_x,
            dt.dst_y,
            dt.dst_width,
            dt.dst_height,
            dt.texture_id,
            tic.width,
            tic.height,
            tic.format,
            tic.is_block_linear,
            dt.src_x,
            dt.src_y,
            dt.src_width,
            dt.src_height
        );
    }
    Ok(DrawTextureJob {
        nvmap_id,
        width: rt.width,
        height: rt.height,
        gpu_va: rt_gpu_va,
        dst_x: dt.dst_x,
        dst_y: dt.dst_y,
        dst_width: dt.dst_width,
        dst_height: dt.dst_height,
        src_x: dt.src_x,
        src_y: dt.src_y,
        src_width: if dt.src_width.abs() >= 1.0 {
            dt.src_width
        } else {
            dt.dst_width
        },
        src_height: if dt.src_height.abs() >= 1.0 {
            dt.src_height
        } else {
            dt.dst_height
        },
        src_rgba,
        src_tex_width: tic.width,
        src_tex_height: tic.height,
    })
}

fn apply_swizzle_rgba(rgba: &mut [u8], swizzle: [nexium_gpu::texture::SwizzleSource; 4]) {
    fn component(src: nexium_gpu::texture::SwizzleSource, p: &[u8]) -> u8 {
        match src {
            nexium_gpu::texture::SwizzleSource::Zero => 0,
            nexium_gpu::texture::SwizzleSource::R => p[0],
            nexium_gpu::texture::SwizzleSource::G => p[1],
            nexium_gpu::texture::SwizzleSource::B => p[2],
            nexium_gpu::texture::SwizzleSource::A => p[3],
            nexium_gpu::texture::SwizzleSource::One => 255,
            nexium_gpu::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    for p in rgba.chunks_exact_mut(4) {
        let old = [p[0], p[1], p[2], p[3]];
        p[0] = component(swizzle[0], &old);
        p[1] = component(swizzle[1], &old);
        p[2] = component(swizzle[2], &old);
        p[3] = component(swizzle[3], &old);
    }
}

fn execute_draw_texture_job(
    renderer: &Arc<nexium_gpu::Renderer>,
    job: &DrawTextureJob,
) -> Result<(), String> {
    let mut dst = renderer
        .readback_target_at(job.nvmap_id, job.width, job.height, job.gpu_va)
        .unwrap_or_else(|| {
            vec![
                0;
                (job.width as usize)
                    .saturating_mul(job.height as usize)
                    .saturating_mul(4)
            ]
        });
    let width = job.width as i32;
    let height = job.height as i32;
    let mut dst_x = job.dst_x.round() as i32;
    let mut dst_y = job.dst_y.round() as i32;
    let mut dst_w = job.dst_width.round() as i32;
    let mut dst_h = job.dst_height.round() as i32;
    if dst_w < 0 {
        dst_x += dst_w;
        dst_w = -dst_w;
    }
    if dst_h < 0 {
        dst_y += dst_h;
        dst_h = -dst_h;
    }
    if dst_w <= 0 || dst_h <= 0 {
        return Ok(());
    }
    for y in 0..dst_h {
        let ty = dst_y + y;
        if ty < 0 || ty >= height {
            continue;
        }
        let fy = (y as f32 + 0.5) / dst_h as f32;
        let syf = job.src_y + fy * job.src_height;
        let sy = syf
            .floor()
            .clamp(0.0, job.src_tex_height.saturating_sub(1) as f32) as i32;
        for x in 0..dst_w {
            let tx = dst_x + x;
            if tx < 0 || tx >= width {
                continue;
            }
            let fx = (x as f32 + 0.5) / dst_w as f32;
            let sxf = job.src_x + fx * job.src_width;
            let sx = sxf
                .floor()
                .clamp(0.0, job.src_tex_width.saturating_sub(1) as f32) as i32;
            let src_off = ((sy as usize * job.src_tex_width as usize + sx as usize) * 4) as usize;
            let dst_off = ((ty as usize * job.width as usize + tx as usize) * 4) as usize;
            if src_off + 4 > job.src_rgba.len() || dst_off + 4 > dst.len() {
                continue;
            }
            let sa = job.src_rgba[src_off + 3] as u32;
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                dst[dst_off..dst_off + 4].copy_from_slice(&job.src_rgba[src_off..src_off + 4]);
                continue;
            }
            let inv = 255 - sa;
            for c in 0..3 {
                let s = job.src_rgba[src_off + c] as u32;
                let d = dst[dst_off + c] as u32;
                dst[dst_off + c] = ((s * sa + d * inv + 127) / 255) as u8;
            }
            let da = dst[dst_off + 3] as u32;
            dst[dst_off + 3] = (sa + (da * inv + 127) / 255).min(255) as u8;
        }
    }
    renderer.upload_target_rgba(job.nvmap_id, job.width, job.height, job.gpu_va, &dst)
}

fn snapshot_read_once(
    snapshot: &mut std::collections::HashMap<u64, Vec<u8>>,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
    gpu_va: u64,
    len: usize,
) -> usize {
    if len == 0 {
        return 0;
    }
    if snapshot
        .get(&gpu_va)
        .map(|data| data.len() >= len)
        .unwrap_or(false)
    {
        return 0;
    }
    if let Some(data) = read_guest(gpu_va, len) {
        let n = data.len();
        snapshot.insert(gpu_va, data);
        n
    } else {
        0
    }
}

fn submit_draw_batch_async(
    batch: &[Maxwell3dDrawCall],
    renderer: &Arc<nexium_gpu::Renderer>,
    rt: &crate::render_thread::RenderThread,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) {
    let profile = nvprof_enabled();
    let t_snapshot = std::time::Instant::now();
    let mut snapshot: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    let mut snapshot_reads = 0usize;
    let mut snapshot_bytes = 0usize;
    let mut tic_summ: Vec<String> = Vec::new();
    for call in batch {
        for binding in &call.vertex_bindings {
            if binding.stride == 0 {
                continue;
            }
            let stride = binding.stride as u64;
            let start_vertex = if call.state.indexed {
                0
            } else {
                call.first_vertex
            };
            let vertex_span = if call.state.indexed {
                call.first_vertex.saturating_add(call.vertex_count)
            } else {
                call.vertex_count
            };
            let start_byte = stride.saturating_mul(start_vertex as u64);
            let mut vbytes = stride.saturating_mul(vertex_span as u64);
            if binding.size > 0 {
                if start_byte >= binding.size {
                    continue;
                }
                vbytes = vbytes.min(binding.size - start_byte);
            }
            let vbytes = vbytes as usize;
            if vbytes == 0 {
                continue;
            }
            let addr = binding.addr.wrapping_add(start_byte);
            let n = snapshot_read_once(&mut snapshot, read_guest, addr, vbytes);
            snapshot_reads += usize::from(n != 0);
            snapshot_bytes += n;
        }
        if call.cbuf_size > 0 && call.cbuf_addr != 0 {
            let n = snapshot_read_once(
                &mut snapshot,
                read_guest,
                call.cbuf_addr,
                call.cbuf_size as usize,
            );
            snapshot_reads += usize::from(n != 0);
            snapshot_bytes += n;
        }
        if !call.fs_tex_ids.is_empty() && call.tic_pool_gpu_va != 0 {
            for &tex_id in &call.fs_tex_ids {
                if tex_id > call.tic_pool_limit {
                    continue;
                }
                let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
                if let Some(tic_raw) = read_guest(tic_addr, 32) {
                    snapshot_reads += 1;
                    snapshot_bytes += tic_raw.len();
                    if let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) {
                        if std::env::var_os("NEXIUM_TEXTYPE_DBG").is_some() {
                            use std::sync::{Mutex, OnceLock};
                            static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> =
                                OnceLock::new();
                            let s =
                                SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                            if let Ok(mut set) = s.lock() {
                                if set.insert(tex_id) {
                                    log::warn!(
                                        "[textype] tic{} type={} base={} depth={} norm={} {:?} {}x{} bl={}",
                                        tex_id,
                                        tic.texture_type,
                                        tic.base_layer,
                                        tic.depth,
                                        tic.normalized_coords,
                                        tic.format,
                                        tic.width,
                                        tic.height,
                                        tic.is_block_linear
                                    );
                                }
                            }
                        }
                        if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                            tic_summ.push(format!(
                                "tic{}={:?} {}x{}x{} base={} type={} norm={} bl={} bh={} va={:#x}",
                                tex_id,
                                tic.format,
                                tic.width,
                                tic.height,
                                tic.depth,
                                tic.base_layer,
                                tic.texture_type,
                                tic.normalized_coords,
                                tic.is_block_linear,
                                tic.block_height_log2,
                                tic.gpu_va
                            ));
                        }
                        let pitch = tic.format.linear_size(tic.width, tic.height);
                        let layer_size = if tic.is_block_linear {
                            tic.format
                                .block_linear_size(tic.width, tic.height, tic.block_height_log2)
                                .max(pitch)
                        } else {
                            pitch
                        };
                        let layer_count = if tic.texture_type == 5 {
                            tic.base_layer.saturating_add(tic.depth).max(1)
                        } else if tic.texture_type == 2 {
                            tic.depth.max(1)
                        } else {
                            1
                        };
                        let read_size = layer_size.saturating_mul(layer_count as usize);
                        let n =
                            snapshot_read_once(&mut snapshot, read_guest, tic.gpu_va, read_size);
                        snapshot_reads += usize::from(n != 0);
                        snapshot_bytes += n;
                    }
                    snapshot.insert(tic_addr, tic_raw);
                }
            }
        }
        if !call.fs_sampler_ids.is_empty() && call.tsc_pool_gpu_va != 0 {
            for &tsc_id in &call.fs_sampler_ids {
                if tsc_id > call.tsc_pool_limit {
                    continue;
                }
                let tsc_addr = call.tsc_pool_gpu_va.wrapping_add((tsc_id as u64) * 32);
                let n = snapshot_read_once(&mut snapshot, read_guest, tsc_addr, 32);
                snapshot_reads += usize::from(n != 0);
                snapshot_bytes += n;
            }
        }
    }
    let snapshot_ms = if profile { elapsed_ms(t_snapshot) } else { 0.0 };
    let snapshot_entries = snapshot.len();
    let n_calls = batch.len();
    let calls = batch.to_vec();
    let r = renderer.clone();
    let diag = std::env::var_os("NEXIUM_PRESENT_KEYS").is_some();
    let job = Box::new(move || {
        {
            let mut ring = submit_ring().lock().unwrap();
            let mut fss: Vec<String> = calls
                .iter()
                .map(|c| format!("{:#x}", c.fs_gpu_va))
                .collect();
            fss.dedup();
            ring.push_back(format!(
                "vs={:#x} fs=[{}] calls={} rt={}",
                calls[0].vs_gpu_va,
                fss.join(","),
                calls.len(),
                calls[0].rt_key.label()
            ));
            while ring.len() > 24 {
                ring.pop_front();
            }
        }
        let res = r.execute_draws(&calls, move |addr: u64, len: usize| {
            snapshot
                .get(&addr)
                .filter(|b| b.len() >= len)
                .map(|b| b[..len].to_vec())
        });
        if let Err(e) = res {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CT: AtomicU64 = AtomicU64::new(0);
            let n = CT.fetch_add(1, Ordering::Relaxed);
            if n < 6 {
                let c = &calls[0];
                log::warn!(
                    "[draw-fail #{}] err={} n_calls={} rt={}:{}x{} vtx={} idx={:?} tex_ids={:?} tsc_ids={:?} cbuf_sz={} tics=[{}]",
                    n, e, calls.len(), c.rt_key.nvmap_id, c.rt_key.width, c.rt_key.height,
                    c.vertex_count, c.index_count, c.fs_tex_ids, c.fs_sampler_ids, c.cbuf_size,
                    if diag { tic_summ.join(" | ") } else { String::new() }
                );
                if n == 0 {
                    let ring = submit_ring().lock().unwrap();
                    for (i, entry) in ring.iter().enumerate() {
                        log::warn!("[draw-fail-ring] {}: {}", i, entry);
                    }
                }
            }
        }
    }) as crate::render_thread::RenderJob;
    if !rt.submit_timeout(job, std::time::Duration::from_secs(3)) {
        log::warn!(
            "[render-saturated] dropped draw batch (calls={}) after 3s; render thread blocked",
            n_calls
        );
    }
    if profile {
        log::warn!(
            "[nvprof] draw_batch calls={} snapshot_entries={} reads={} bytes={} snapshot_ms={:.3}",
            batch.len(),
            snapshot_entries,
            snapshot_reads,
            snapshot_bytes,
            snapshot_ms
        );
    }
}

struct ShaderBundle {
    vs_spirv: std::sync::Arc<Vec<u32>>,
    vs_cbuf_mask: u32,
    vs_hash: u64,
    fs_spirv: std::sync::Arc<Vec<u32>>,
    fs_cbuf_mask: u32,
    fs_hash: u64,
    fs_tex_ids: Vec<u32>,
    fs_sampler_arrayed: bool,
    fs_cbuf_reads: Vec<(u32, u32)>,
    cbuf_used: u32,
    ssbo_descs: Vec<nexium_shader::StorageBufferAddr>,
}

#[allow(clippy::type_complexity)]
fn shader_bundle_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<
        (u64, u64, u32, u32, u32, u32, u32, u32, u32, u32),
        std::sync::Arc<ShaderBundle>,
    >,
> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<
                (u64, u64, u32, u32, u32, u32, u32, u32, u32, u32),
                std::sync::Arc<ShaderBundle>,
            >,
        >,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn shader_failed_set() -> &'static std::sync::Mutex<
    std::collections::HashSet<(u64, u64, u32, u32, u32, u32, u32, u32, u32, u32)>,
> {
    use std::sync::OnceLock;
    static FAILED: OnceLock<
        std::sync::Mutex<
            std::collections::HashSet<(u64, u64, u32, u32, u32, u32, u32, u32, u32, u32)>,
        >,
    > = OnceLock::new();
    FAILED.get_or_init(|| {
        // Silence the default panic hook for shader-emit panics we catch_unwind,
        // so a few thousand structurizer failures don't flood stderr.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info.to_string();
            if msg.contains("nexium-spirv") || msg.contains("NestedBlock") {
                if std::env::var_os("NEXIUM_SHADER_PANIC").is_some() {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static PC: AtomicU64 = AtomicU64::new(0);
                    if PC.fetch_add(1, Ordering::Relaxed) < 16 {
                        log::warn!("[shader-panic] {}", msg);
                    }
                }
                return;
            }
            prev(info);
        }));
        std::sync::Mutex::new(std::collections::HashSet::new())
    })
}

fn shader_panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic".to_string()
    }
}

fn depth_disabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("NEXIUM_NO_DEPTH").ok().as_deref() == Some("1"))
}

fn small_rt_registry() -> &'static std::sync::Mutex<std::collections::HashMap<RtKey, u32>> {
    static R: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<RtKey, u32>>> =
        std::sync::OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn submit_ring() -> &'static std::sync::Mutex<std::collections::VecDeque<String>> {
    static R: std::sync::OnceLock<std::sync::Mutex<std::collections::VecDeque<String>>> =
        std::sync::OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()))
}

pub fn guest_probe(mappings: &GpuMappings, mem_read: &dyn Fn(u64, &mut [u8]) -> bool) {
    use std::sync::{Mutex, OnceLock};
    static CFG: OnceLock<Vec<(u64, usize)>> = OnceLock::new();
    static LAST: OnceLock<Mutex<Vec<Vec<u8>>>> = OnceLock::new();
    static ARMED: OnceLock<()> = OnceLock::new();
    static SNAPSHOT_TARGET: OnceLock<Option<Option<u64>>> = OnceLock::new();
    if let Ok(spec) = std::env::var("NEXIUM_WATCH_WRITE_GPU") {
        if ARMED.get().is_none() {
            if let Some((va, len)) = spec.trim().split_once(':') {
                let va = u64::from_str_radix(va.trim().trim_start_matches("0x"), 16).unwrap_or(0);
                let len =
                    u64::from_str_radix(len.trim().trim_start_matches("0x"), 16).unwrap_or(0x60);
                if va != 0 {
                    if let Some(cpu) = mappings.cpu_address_for(va) {
                        let armed = if std::env::var("NEXIUM_WATCH_PAGE_PROTECT")
                            .ok()
                            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                            .unwrap_or(false)
                        {
                            nexium_memory::fastmem::watch_arm(cpu, len)
                        } else {
                            nexium_memory::fastmem::watch_mark(cpu, len)
                        };
                        if armed {
                            let _ = ARMED.set(());
                            log::warn!(
                                "[watch-write] ARMED gpu_va={:#x} cpu_va={:#x} len={:#x}",
                                va,
                                cpu,
                                len
                            );
                        } else {
                            log::warn!(
                                "[watch-write] arm FAILED gpu_va={:#x} cpu_va={:#x}",
                                va,
                                cpu
                            );
                            let _ = ARMED.set(());
                        }
                    }
                }
            }
        }
    }
    let cfg = CFG.get_or_init(|| {
        std::env::var("NEXIUM_GUEST_PROBE")
            .map(|v| {
                v.split(',')
                    .filter_map(|part| {
                        let (va, len) = part.trim().split_once(':')?;
                        let va =
                            u64::from_str_radix(va.trim().trim_start_matches("0x"), 16).ok()?;
                        let len = usize::from_str_radix(len.trim().trim_start_matches("0x"), 16)
                            .ok()?
                            .min(256);
                        Some((va, len))
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    if cfg.is_empty() {
        return;
    }
    let snapshot_target = SNAPSHOT_TARGET.get_or_init(|| {
        let spec = std::env::var("NEXIUM_GUEST_PROBE_THREAD_SNAPSHOT").ok()?;
        let spec = spec.trim();
        if spec.is_empty()
            || spec == "1"
            || spec.eq_ignore_ascii_case("true")
            || spec.eq_ignore_ascii_case("all")
        {
            return Some(None);
        }
        let va = u64::from_str_radix(spec.trim_start_matches("0x"), 16).ok()?;
        Some(Some(va))
    });
    let last = LAST.get_or_init(|| Mutex::new(vec![Vec::new(); cfg.len()]));
    let mut last = last.lock().unwrap();
    for (i, (va, len)) in cfg.iter().enumerate() {
        let Some(cpu) = mappings.cpu_address_for(*va) else {
            continue;
        };
        let mut buf = vec![0u8; *len];
        if !mem_read(cpu, &mut buf) {
            continue;
        }
        if last[i] != buf {
            let floats: Vec<String> = buf
                .chunks_exact(4)
                .map(|c| {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    format!("{:.3}", v)
                })
                .collect();
            log::warn!(
                "[guest-probe] va={:#x} len={:#x} changed: [{}]",
                va,
                len,
                floats.join(",")
            );
            let snapshot_match = match *snapshot_target {
                Some(Some(target_va)) => target_va == *va,
                Some(None) => true,
                None => false,
            };
            if buf.iter().any(|b| *b != 0) && snapshot_match {
                nexium_memory::fastmem::mark_guest_probe_event(*va);
            }
            last[i] = buf;
        }
    }
}

pub fn writeback_small_rts(
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let pending: Vec<(RtKey, u32)> = {
        let mut reg = small_rt_registry().lock().unwrap();
        if reg.is_empty() {
            return;
        }
        reg.drain().collect()
    };
    if let Some(rt) = crate::render_thread::maybe_render_thread() {
        let (tx, rx) = std::sync::mpsc::channel();
        if !rt.try_submit(Box::new(move || {
            let _ = tx.send(());
        })) {
            let mut reg = small_rt_registry().lock().unwrap();
            for (key, tile_mode) in pending {
                if reg.len() < 64 {
                    reg.insert(key, tile_mode);
                }
            }
            return;
        }
        if rx
            .recv_timeout(std::time::Duration::from_millis(250))
            .is_err()
        {
            let mut reg = small_rt_registry().lock().unwrap();
            for (key, tile_mode) in pending {
                if reg.len() < 64 {
                    reg.insert(key, tile_mode);
                }
            }
            return;
        }
    }
    for (key, tile_mode) in pending {
        let Some((kw, kh, bpp, mut raw)) = renderer.readback_target_raw(key.nvmap_id, key.gpu_va)
        else {
            continue;
        };
        let Some((cpu, limit)) = mappings.cpu_range_for(key.gpu_va) else {
            continue;
        };
        let width_bytes = kw as usize * bpp;
        if kh >= 2 && raw.len() >= width_bytes * kh as usize {
            let h = kh as usize;
            for y in 0..h / 2 {
                let (top, bot) = raw.split_at_mut((h - 1 - y) * width_bytes);
                top[y * width_bytes..(y + 1) * width_bytes]
                    .swap_with_slice(&mut bot[..width_bytes]);
            }
        }
        if (tile_mode >> 12) & 1 == 1 {
            let n = raw.len().min(limit as usize);
            mem_write(cpu, &raw[..n]);
        } else {
            let bh_log2 = (tile_mode >> 4) & 0x7;
            let tiled = super::engines::maxwell_dma::swizzle_block_linear(
                &raw,
                width_bytes,
                kh as usize,
                width_bytes,
                width_bytes,
                kh as usize,
                bh_log2,
                0,
                0,
            );
            let n = tiled.len().min(limit as usize);
            mem_write(cpu, &tiled[..n]);
        }
        log::debug!(
            "[rt-writeback] {} bpp={} tile={:#x}",
            key.label(),
            bpp,
            tile_mode
        );
    }
}

fn execute_one(
    draw: &DrawCall,
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    renderer: &Arc<nexium_gpu::Renderer>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<Option<Maxwell3dDrawCall>, String> {
    let rt_slot = if draw.is_clear {
        ((draw.clear_mask >> 6) & 0xF).min(7) as usize
    } else {
        draw_color_rt_slot(draw)
    };
    let rt = &draw.rt[rt_slot];
    if rt.width == 0 || rt.height == 0 {
        return Err(format!("RT[{}] has zero extent", rt_slot));
    }
    let rt_gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings
        .nvmap_id_for(rt_gpu_va)
        .ok_or_else(|| format!("RT gpu_va={:#x} not mapped", rt_gpu_va))?;
    let rt_key = RtKey::with_cpu(
        nvmap_id,
        rt.width,
        rt.height,
        rt_gpu_va,
        mappings.cpu_address_for(rt_gpu_va).unwrap_or(0),
    );
    let rt_format = map_rt_format_for_key(rt.format, rt_key);
    if !draw.is_clear && (rt.width as u64) * (rt.height as u64) <= 16384 && {
        static SKIP: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        let skip = SKIP.get_or_init(|| {
            std::env::var("NEXIUM_NO_SMALL_RT_WB_NVMAPS")
                .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
                .unwrap_or_default()
        });
        !skip.contains(&rt_key.nvmap_id)
    } {
        small_rt_registry()
            .lock()
            .unwrap()
            .insert(rt_key, rt.tile_mode);
    }
    let no_depth = depth_disabled();
    let zeta_key = zeta_rt_key(draw, mappings, rt_key);

    if draw.is_clear {
        let op_seq = next_gpu_op_seq();
        let mask = draw.clear_mask;
        let want_color_clear = mask == 0 || (mask & 0b11_1100) != 0;
        let want_depth_clear = (mask & 0x1) != 0 && draw.zeta_enable;
        let clear_scissor = if draw.clear_control & 0x100 != 0 {
            scissor_rect(draw, rt.width, rt.height)
        } else {
            None
        };
        let color = [
            draw.clear_color.r,
            draw.clear_color.g,
            draw.clear_color.b,
            draw.clear_color.a,
        ];
        let do_depth = want_depth_clear && !no_depth;
        let depth_clear_key = zeta_key.unwrap_or(rt_key);
        trace_clear(
            draw,
            op_seq,
            nvmap_id,
            rt,
            rt_gpu_va,
            clear_scissor,
            color,
            want_color_clear,
            do_depth,
        );
        if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
            let r = renderer.clone();
            let cdepth = draw.clear_depth;
            let (w, h) = (rt.width, rt.height);
            let (dw, dh, dva, dnv) = (
                depth_clear_key.width,
                depth_clear_key.height,
                depth_clear_key.gpu_va,
                depth_clear_key.nvmap_id,
            );
            rt_thread.submit(Box::new(move || {
                if want_color_clear {
                    if let Some(rect) = clear_scissor {
                        let _ = r.clear_target_rect_with_format(
                            nvmap_id, w, h, rt_gpu_va, color, rect, rt_format,
                        );
                    } else {
                        let _ =
                            r.clear_target_with_format(nvmap_id, w, h, rt_gpu_va, color, rt_format);
                    }
                }
                if do_depth {
                    let _ = r.clear_depth(dnv, dw, dh, dva, cdepth);
                }
            }));
        } else {
            if want_color_clear {
                if let Some(rect) = clear_scissor {
                    renderer.clear_target_rect_with_format(
                        nvmap_id, rt.width, rt.height, rt_gpu_va, color, rect, rt_format,
                    )?;
                } else {
                    renderer.clear_target_with_format(
                        nvmap_id, rt.width, rt.height, rt_gpu_va, color, rt_format,
                    )?;
                }
            }
            if do_depth {
                renderer.clear_depth(
                    depth_clear_key.nvmap_id,
                    depth_clear_key.width,
                    depth_clear_key.height,
                    depth_clear_key.gpu_va,
                    draw.clear_depth,
                )?;
            }
        }
        return Ok(None);
    }

    let program_region = ((maxwell.regs.program_region_va_hi as u64) << 32)
        | maxwell.regs.program_region_va_lo as u64;

    let vs_prog = &maxwell.regs.shader_programs[1];
    let fs_prog = &maxwell.regs.shader_programs[5];
    let vs_active = vs_prog.enabled || vs_prog.address_lo != 0;
    let fs_active = fs_prog.enabled || fs_prog.address_lo != 0;
    if !vs_active || !fs_active {
        return Err("VS or FS program disabled".to_string());
    }

    let vs_addr = program_region.wrapping_add(vs_prog.address_lo as u64);
    let fs_addr = program_region.wrapping_add(fs_prog.address_lo as u64);
    let (vptx_scale_z, vptx_translate_z) =
        if draw.viewport.scale_z != 0.0 || draw.viewport.translate_z != 0.0 {
            (draw.viewport.scale_z, draw.viewport.translate_z)
        } else {
            (1.0, 0.0)
        };
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                "viewport Z transform (first draw): captured scale_z={} translate_z={} \
                 -> applied {}/{} (OpenGL conv ~0.5/0.5; D3D/identity ~1.0/0.0)",
                draw.viewport.scale_z,
                draw.viewport.translate_z,
                vptx_scale_z,
                vptx_translate_z,
            );
        }
    }

    let ps_key = if draw.topology == 0 {
        draw.point_size.to_bits()
    } else {
        0
    };
    let surface_clip = draw.surface_clip.effective(rt.width, rt.height);
    let render_area = if !draw.viewport_transform_en {
        (surface_clip.width, surface_clip.height)
    } else {
        (rt.width, rt.height)
    };
    let window_ndc = if !draw.viewport_transform_en && render_area.0 > 0 && render_area.1 > 0 {
        Some((2.0 / render_area.0 as f32, 2.0 / render_area.1 as f32))
    } else {
        None
    };
    let win_key = if window_ndc.is_some() {
        (render_area.0 << 16) | (render_area.1 & 0xFFFF)
    } else {
        0
    };
    let mut uint_attr_mask: u32 = 0;
    let mut sint_attr_mask: u32 = 0;
    for (loc, attrib) in draw.vertex_attribs.iter().enumerate() {
        if attrib.format == 0 || attrib.constant || loc >= 32 {
            continue;
        }
        let type_ = attrib.format >> 6;
        if type_ == 3 {
            sint_attr_mask |= 1 << loc;
        } else if type_ == 4 {
            uint_attr_mask |= 1 << loc;
        }
    }
    let fs_sph = fetch_sph(fs_addr, mappings, mem_read);
    let fs_input_map = fs_sph.map(ps_generic_input_map).unwrap_or([0; 32]);
    let fs_output_map = fs_sph.map(ps_output_map).unwrap_or(0);
    let color_rts = active_color_rts(draw, mappings, rt_key, rt_format, fs_output_map);
    let color_rt_locations = color_rts
        .iter()
        .map(|(location, _, _)| *location)
        .collect::<Vec<_>>();
    let color_rt_keys = color_rts
        .iter()
        .map(|(_, key, _)| *key)
        .collect::<Vec<_>>();
    let color_rt_formats = color_rts
        .iter()
        .map(|(_, _, format)| *format)
        .collect::<Vec<_>>();
    let color_output_count = (color_rt_formats.len() as u32).clamp(1, 8);
    if std::env::var_os("NEXIUM_SHADER_MAP_DBG").is_some() {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u64, u32, u32, u32)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if seen.lock().unwrap().insert((
            vs_addr,
            fs_addr,
            color_output_count,
            fs_output_map,
            draw.rt_control,
        )) {
            let keys = color_rt_keys
                .iter()
                .map(|key| key.label())
                .collect::<Vec<_>>()
                .join(",");
            let formats = color_rt_formats
                .iter()
                .map(|format| format!("{:?}", format))
                .collect::<Vec<_>>()
                .join(",");
            let imap = fs_input_map
                .iter()
                .map(|v| format!("{:02x}", v))
                .collect::<Vec<_>>()
                .join("");
            log::warn!(
                "[shader-map] vs={:#x} fs={:#x} outputs={} omap={:#010x} imap={} rtctl={:#x} keys=[{}] fmts=[{}]",
                vs_addr,
                fs_addr,
                color_output_count,
                fs_output_map,
                imap,
                draw.rt_control,
                keys,
                formats
            );
        }
    }
    let shader_key = (
        vs_addr,
        fs_addr,
        vptx_scale_z.to_bits(),
        vptx_translate_z.to_bits(),
        ps_key,
        win_key,
        uint_attr_mask,
        sint_attr_mask,
        color_output_count,
        fs_output_map,
    );
    if shader_failed_set().lock().unwrap().contains(&shader_key) {
        return Err("shader previously failed to emit".to_string());
    }
    let bundle = {
        let cache = shader_bundle_cache();
        let mut guard = cache.lock().unwrap();
        if let Some(b) = guard.get(&shader_key) {
            b.clone()
        } else {
            let vs_sass = fetch_sass(vs_addr, mappings, mem_read)
                .ok_or_else(|| "VS SASS read failed".to_string())?;
            let fs_sass = fetch_sass(fs_addr, mappings, mem_read)
                .ok_or_else(|| "FS SASS read failed".to_string())?;

            let mut vs_cfg = nexium_shader::build_cfg(&vs_sass);
            let fs_cfg = nexium_shader::build_cfg(&fs_sass);
            let fs_cbuf_reads = collect_cbuf_reads(&fs_cfg, 16);
            let ssbo_descs = if nexium_shader::shader_uses_ldg(&vs_sass) {
                nexium_shader::collect_storage_buffers(&mut vs_cfg)
            } else {
                Vec::new()
            };
            let num_ssbo = (ssbo_descs.len() as u32).min(8);
            if vs_cfg.unimplemented != 0 || fs_cfg.unimplemented != 0 {
                log::warn!(
                    "shader unimplemented: vs_addr={:#x} fs_addr={:#x} vs={} {:?} fs={} {:?}",
                    vs_addr,
                    fs_addr,
                    vs_cfg.unimplemented,
                    unimplemented_samples(&vs_cfg),
                    fs_cfg.unimplemented,
                    unimplemented_samples(&fs_cfg),
                );
                if std::env::var_os("NEXIUM_SHADERDBG").is_some() {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 20 {
                        log::warn!(
                            "[shaderdbg] region={:#x} vs_lo={:#x} fs_lo={:#x} cpu(vs)={:?} cpu(fs)={:?} fs_sass[0..16]={:02x?} | {}",
                            program_region,
                            vs_prog.address_lo,
                            fs_prog.address_lo,
                            mappings.cpu_address_for(vs_addr),
                            mappings.cpu_address_for(fs_addr),
                            &fs_sass[..16.min(fs_sass.len())],
                            mappings.describe_around(fs_addr),
                        );
                    }
                }
            } else {
                log::debug!(
                    "shader translated: vs_addr={:#x} fs_addr={:#x} all ops covered",
                    vs_addr,
                    fs_addr,
                );
            }

            let (fs_spirv, fs_cbuf_mask, mut fs_tex_ids, fs_cbuf_used, fs_sampler_arrayed) =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let fs_debug_targets = parse_env_u64_list("NEXIUM_FS_DEBUG_TARGET");
                    let fs_debug_active =
                        fs_debug_targets.is_empty() || fs_debug_targets.contains(&fs_addr);
                    nexium_spirv::emit_fragment_full_with_input_map_meta_outputs_debug(
                        &fs_cfg,
                        fs_input_map,
                        color_output_count,
                        fs_output_map,
                        fs_debug_active,
                    )
                })) {
                    Ok(v) => v,
                    Err(panic) => {
                        shader_failed_set().lock().unwrap().insert(shader_key);
                        return Err(format!(
                            "FS SPIR-V emit panicked vs_addr={:#x} fs_addr={:#x}: {}",
                            vs_addr,
                            fs_addr,
                            shader_panic_message(panic)
                        ));
                    }
                };

            {
                let fs_walk_len = shader_cfg_code_len(&fs_cfg).min(fs_sass.len());
                let fs_walk_sass = if fs_walk_len == 0 {
                    fs_sass.as_slice()
                } else {
                    &fs_sass[..fs_walk_len]
                };
                let walked = nexium_shader::extract_fs_tex_ids(fs_walk_sass, 15);
                let before = fs_tex_ids.len();
                let mut bindless = 0usize;
                for id in walked {
                    match id {
                        nexium_shader::FsTexId::ImmediateTic(idx) => {
                            if !fs_tex_ids.contains(&idx) {
                                fs_tex_ids.push(idx);
                            }
                        }
                        nexium_shader::FsTexId::BindlessCbufOffset(_) => {
                            bindless += 1;
                        }
                    }
                }
                let walker_imm = fs_tex_ids.len() - before;
                log::debug!(
                    "fs_tex_ids: spirv-emitter={} walker-imm={} bindless-skipped={} final={:?} \
                     tic_pool=0x{:x} (limit={:#x}) fs_sass_len={}",
                    before,
                    walker_imm,
                    bindless,
                    fs_tex_ids,
                    draw.tic_pool_gpu_va,
                    draw.tic_pool_limit,
                    fs_walk_sass.len(),
                );
            }

            let vs_tex_base: u32 = {
                let vs_walked = nexium_shader::extract_fs_tex_ids(vs_sass.as_slice(), 15);
                let mut vs_tex_ids: Vec<u32> = Vec::new();
                let mut vs_bindless = 0usize;
                for id in vs_walked {
                    match id {
                        nexium_shader::FsTexId::ImmediateTic(idx) => {
                            if !vs_tex_ids.contains(&idx) {
                                vs_tex_ids.push(idx);
                            }
                        }
                        nexium_shader::FsTexId::BindlessCbufOffset(_) => vs_bindless += 1,
                    }
                }
                if !vs_tex_ids.is_empty() || vs_bindless != 0 {
                    log::warn!(
                        "[vs-tex] vs={:#x} fs={:#x} vs_imm={:?} vs_bindless={} fs_tex_ids={:?}",
                        vs_addr, fs_addr, vs_tex_ids, vs_bindless, fs_tex_ids
                    );
                }
                if std::env::var_os("NEXIUM_VS_TEX").is_some() && !vs_tex_ids.is_empty() {
                    let base = fs_tex_ids.len() as u32;
                    vs_tex_ids.sort_unstable();
                    for id in &vs_tex_ids {
                        fs_tex_ids.push(*id);
                    }
                    base
                } else {
                    0
                }
            };

            {
                let want_addrs = parse_env_u64_list("NEXIUM_DUMP_FS");
                if want_addrs.contains(&fs_addr) {
                    let fs_dis = nexium_shader::disassemble(&fs_sass)
                        .into_iter()
                        .map(|line| line.to_string_compact())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/target_fs_{:x}.txt", fs_addr),
                        &fs_dis,
                    );
                }
            }
            {
                let want_addrs = parse_env_u64_list("NEXIUM_DUMP_VS");
                if want_addrs.contains(&vs_addr) {
                    let vs_dis = nexium_shader::disassemble(&vs_sass)
                        .into_iter()
                        .map(|line| line.to_string_compact())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/target_vs_{:x}.txt", vs_addr),
                        &vs_dis,
                    );
                    log::warn!(
                        "[dump-vs] vs={:#x} vp_scale_z={} vp_translate_z={} applied={}/{} vp_en={}",
                        vs_addr,
                        draw.viewport.scale_z,
                        draw.viewport.translate_z,
                        vptx_scale_z,
                        vptx_translate_z,
                        draw.viewport_transform_en
                    );
                }
            }
            let required_outputs = nexium_spirv::scan_input_locations(&fs_spirv);
            let (vs_spirv, vs_cbuf_mask, vs_cbuf_used) =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nexium_spirv::emit_vertex_with_bindings_opts(
                        &vs_cfg,
                        &required_outputs,
                        nexium_spirv::VertexOptions {
                            vptx_scale_z,
                            vptx_translate_z,
                            point_size: if draw.topology == 0 {
                                Some(draw.point_size)
                            } else {
                                None
                            },
                            window_ndc,
                            num_ssbo,
                            uint_attr_mask,
                            sint_attr_mask,
                            tex_slot_base: vs_tex_base,
                            ..Default::default()
                        },
                    )
                })) {
                    Ok(v) => v,
                    Err(panic) => {
                        shader_failed_set().lock().unwrap().insert(shader_key);
                        return Err(format!(
                            "VS SPIR-V emit panicked vs_addr={:#x} fs_addr={:#x}: {}",
                            vs_addr,
                            fs_addr,
                            shader_panic_message(panic)
                        ));
                    }
                };

            let vs_hash = nexium_gpu::renderer::hash_spirv(&vs_spirv);
            let fs_hash = nexium_gpu::renderer::hash_spirv(&fs_spirv);
            if std::env::var_os("NEXIUM_SHADER_MODULE_DBG").is_some() {
                log::warn!(
                    "[shader-module-map] vs_addr={:#x} vs_hash={:016x} fs_addr={:#x} fs_hash={:016x}",
                    vs_addr,
                    vs_hash,
                    fs_addr,
                    fs_hash
                );
            }
            if std::env::var_os("NEXIUM_DUMP_SPIRV").is_some() {
                let fnv = |words: &[u32]| {
                    let mut h: u64 = 1469598103934665603;
                    for w in words {
                        h ^= *w as u64;
                        h = h.wrapping_mul(1099511628211);
                    }
                    h
                };
                log::info!(
                    "[spv-map] vs_addr={:#x} vs_spv={:016x} fs_addr={:#x} fs_spv={:016x}",
                    vs_addr,
                    fnv(&vs_spirv),
                    fs_addr,
                    fnv(&fs_spirv)
                );
            }
            let b = std::sync::Arc::new(ShaderBundle {
                vs_spirv: std::sync::Arc::new(vs_spirv),
                vs_cbuf_mask,
                vs_hash,
                fs_spirv: std::sync::Arc::new(fs_spirv),
                fs_cbuf_mask,
                fs_hash,
                fs_tex_ids,
                fs_sampler_arrayed,
                fs_cbuf_reads,
                cbuf_used: vs_cbuf_used.max(fs_cbuf_used),
                ssbo_descs,
            });
            guard.insert(shader_key, b.clone());
            if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static DN: AtomicU32 = AtomicU32::new(0);
                let dk = DN.fetch_add(1, Ordering::Relaxed);
                let target_fs = std::env::var("NEXIUM_PROBE_SHADE_FS")
                    .ok()
                    .and_then(|v| parse_env_u64(&v));
                let target_vs = std::env::var("NEXIUM_PROBE_SHADE_VS")
                    .ok()
                    .and_then(|v| parse_env_u64(&v));
                let target = target_fs.is_some_and(|v| v == fs_addr)
                    || target_vs.is_some_and(|v| v == vs_addr);
                if dk < 24 || target {
                    let tag = if target {
                        format!("target_{:x}_{:x}", vs_addr, fs_addr)
                    } else {
                        dk.to_string()
                    };
                    let vb: Vec<u8> = b.vs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let fb: Vec<u8> = b.fs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let _ =
                        std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_vs.spv", tag), &vb);
                    let _ =
                        std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_fs.spv", tag), &fb);
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_vs.sass", tag),
                        &vs_sass,
                    );
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_fs.sass", tag),
                        &fs_sass,
                    );
                    let vs_dis = nexium_shader::disassemble(&vs_sass)
                        .into_iter()
                        .map(|line| line.to_string_compact())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let fs_dis = nexium_shader::disassemble(&fs_sass)
                        .into_iter()
                        .map(|line| line.to_string_compact())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_vs.txt", tag),
                        vs_dis,
                    );
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_fs.txt", tag),
                        fs_dis,
                    );
                    if target {
                        let _ = std::fs::write(
                            format!("C:/Users/Mythrax/Desktop/sh_{}_fs.cfg.txt", tag),
                            shader_cfg_dump(&fs_cfg),
                        );
                    }
                    log::warn!(
                        "[shdump] #{} vs_addr={:#x} fs_addr={:#x} vs_mask={:#x} fs_mask={:#x} vs_bytes={} fs_bytes={} ntex={}",
                        tag,
                        vs_addr,
                        fs_addr,
                        b.vs_cbuf_mask,
                        b.fs_cbuf_mask,
                        b.vs_spirv.len(),
                        b.fs_spirv.len(),
                        b.fs_tex_ids.len()
                    );
                }
            }
            b
        }
    };
    let vs_spirv = bundle.vs_spirv.clone();
    let fs_spirv = bundle.fs_spirv.clone();
    let vs_cbuf_mask = bundle.vs_cbuf_mask;
    let fs_cbuf_mask = bundle.fs_cbuf_mask;
    let mut fs_tex_ids = bundle.fs_tex_ids.clone();
    let fs_sampler_arrayed = bundle.fs_sampler_arrayed;

    let layout = build_vertex_layout(draw)?;
    let topology = map_topology(draw.topology)
        .ok_or_else(|| format!("unsupported topology {}", draw.topology))?;

    let (cbuf_addr, cbuf_size) = resolve_cbuf(draw, &maxwell.regs.cbuf_binds);

    if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        if k % 2000 == 0 {
            let stage = |s: usize| -> String {
                maxwell.regs.cbuf_binds[s]
                    .iter()
                    .enumerate()
                    .filter(|(_, (a, sz))| *a != 0 && *sz > 0)
                    .map(|(i, (a, sz))| format!("[{}]={:#x}/{}", i, a, sz))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            log::warn!(
                "[cbuf] draw{} vs_mask={:#x} fs_mask={:#x} resolved={:#x}/{} last_cb={:#x}/{} | VS:{} | FS:{}",
                k,
                vs_cbuf_mask,
                fs_cbuf_mask,
                cbuf_addr,
                cbuf_size,
                draw.last_constbuf_addr,
                draw.last_constbuf_size,
                stage(0),
                stage(4)
            );
        }
    }

    log::trace!(
        "cbuf_resolve: addr={:#x} size={} cb_binds_nonzero={}",
        cbuf_addr,
        cbuf_size,
        maxwell
            .regs
            .cbuf_binds
            .iter()
            .flat_map(|stage| stage.iter())
            .filter(|(a, s)| *a != 0 && *s > 0)
            .count(),
    );

    let shader_fs_tex_ids = fs_tex_ids.clone();
    let mut fs_tex_remap: Vec<String> = Vec::with_capacity(fs_tex_ids.len());
    let mut fs_sampler_ids: Vec<u32> = vec![0u32; fs_tex_ids.len()];
    if !fs_tex_ids.is_empty() {
        let via_header_index = maxwell.regs.sampler_binding == 1;
        let tex_cb_index = choose_texture_cb_index(
            &maxwell.regs.cbuf_binds[4],
            maxwell.regs.bindless_texture_const_buffer_slot,
            maxwell.regs.tex_cb_index,
            &fs_tex_ids,
            draw.tic_pool_gpu_va,
            draw.tic_pool_limit,
            via_header_index,
            mappings,
            mem_read,
        );
        let (tcb_addr, tcb_size) = maxwell.regs.cbuf_binds[4][tex_cb_index.min(15)];
        for (i, unit_slot) in fs_tex_ids.iter_mut().enumerate() {
            let shader_id = *unit_slot;
            let off = (shader_id as u64).saturating_mul(4);
            let mut remap = format!(
                "s{}:{} cb{} base={:#x}/{} off={:#x}",
                i, shader_id, tex_cb_index, tcb_addr, tcb_size, off
            );
            if tcb_addr == 0 {
                remap.push_str(" no-tcb");
                fs_tex_remap.push(remap);
                continue;
            }
            if off + 4 > tcb_size as u64 {
                remap.push_str(" out-of-range");
                fs_tex_remap.push(remap);
                continue;
            }
            let Some(cpu) = mappings.cpu_address_for(tcb_addr.wrapping_add(off)) else {
                remap.push_str(" unmapped");
                fs_tex_remap.push(remap);
                continue;
            };
            let mut bytes = [0u8; 4];
            if mem_read(cpu, &mut bytes) {
                let handle = u32::from_le_bytes(bytes);
                let (tic, tsc) = split_texture_handle(handle, via_header_index);
                if handle != 0
                    && texture_tic_readable(
                        tic,
                        draw.tic_pool_gpu_va,
                        draw.tic_pool_limit,
                        mappings,
                        mem_read,
                    )
                {
                    if std::env::var_os("NEXIUM_TEX_BIND_LOG").is_some() {
                        log::warn!(
                            "tex_handle: cb{} off={:#x} handle={:#x} -> TIC {} TSC {}",
                            tex_cb_index,
                            off,
                            handle,
                            tic,
                            tsc
                        );
                    }
                    *unit_slot = tic;
                    fs_sampler_ids[i] = tsc;
                    remap.push_str(&format!(" handle={:#x}->tic{} tsc{}", handle, tic, tsc));
                } else {
                    remap.push_str(&format!(
                        " handle={:#x} invalid tic{} tsc{}",
                        handle, tic, tsc
                    ));
                }
            } else {
                remap.push_str(" read-fail");
            }
            fs_tex_remap.push(remap);
        }
    }

    let mut sampled_rt_fuzzy = false;
    let mut sampled_rt_keys: Vec<RtKey> = Vec::new();
    let mut sampled_rt_slots: Vec<Option<RtKey>> = vec![None; fs_tex_ids.len()];
    if std::env::var_os("NEXIUM_TEXDBG").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static D: AtomicU64 = AtomicU64::new(0);
        let d = D.fetch_add(1, Ordering::Relaxed);
        if d < 60 {
            log::warn!(
                "[texdbg-draw #{}] tic_pool={:#x} limit={} fs_tex_ids={:?}",
                d,
                draw.tic_pool_gpu_va,
                draw.tic_pool_limit,
                &fs_tex_ids
            );
        }
    }
    if draw.tic_pool_gpu_va != 0 {
        for (slot, tex_id) in fs_tex_ids.iter().enumerate() {
            if *tex_id == u32::MAX || *tex_id > draw.tic_pool_limit {
                continue;
            }
            let tic_addr = draw.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
            let Some(cpu) = mappings.cpu_address_for(tic_addr) else {
                continue;
            };
            let mut tic_raw = [0u8; 32];
            if !mem_read(cpu, &mut tic_raw) {
                continue;
            }
            let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) else {
                continue;
            };
            if tic.width >= 512 && tic.height >= 256 && std::env::var_os("NEXIUM_TEXDBG").is_some()
            {
                use std::sync::atomic::{AtomicU64, Ordering};
                static N: AtomicU64 = AtomicU64::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 200 {
                    log::warn!(
                        "[texdbg #{}] slot={} fmt={:?} {}x{}x{} base={} type={} norm={} gpu_va={:#x} nvmap={:?} can_alias_rt={} swizzle={:?}",
                        n,
                        slot,
                        tic.format,
                        tic.width,
                        tic.height,
                        tic.depth,
                        tic.base_layer,
                        tic.texture_type,
                        tic.normalized_coords,
                        tic.gpu_va,
                        mappings.nvmap_id_for(tic.gpu_va),
                        tic_can_alias_render_target(tic.format),
                        tic.swizzle
                    );
                }
            }
            if !tic_can_alias_render_target(tic.format) {
                continue;
            }
            let Some(nv) = mappings.nvmap_id_for(tic.gpu_va) else {
                continue;
            };
            let key = RtKey::with_cpu(
                nv,
                tic.width,
                tic.height,
                tic.gpu_va,
                mappings.cpu_address_for(tic.gpu_va).unwrap_or(0),
            );
            sampled_rt_slots[slot] = Some(key);
            if !sampled_rt_keys.contains(&key) {
                sampled_rt_keys.push(key);
            }
            if tic.width >= 512 && tic.height >= 256 {
                let mid = tic.gpu_va
                    + (tic.width as u64 * 4) * (tic.height as u64 / 2)
                    + (tic.width as u64 * 2);
                if let Some(mcpu) = mappings.cpu_address_for(mid) {
                    let mut probe = [0u8; 64];
                    if mem_read(mcpu, &mut probe) && probe.iter().all(|b| *b == 0) {
                        sampled_rt_fuzzy = true;
                    }
                }
            }
        }
    }
    let sampled_rt_key = sampled_rt_keys.first().copied();

    let vertex_bindings = vertex_buffer_bindings(&draw.vertex_buffers, &layout);
    let vertex_addr = vertex_bindings
        .first()
        .map(|b| b.addr)
        .ok_or_else(|| "no vertex buffer bound".to_string())?;

    if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        let probe_stride = std::env::var("NEXIUM_PROBE_SHADE_EVERY")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1500)
            .max(1);
        if k % probe_stride == 0 && layout.attrs.len() >= 5 {
            let vcpu = mappings.cpu_address_for(vertex_addr).unwrap_or(0);
            if vcpu != 0 {
                for a in &layout.attrs {
                    let stride = layout
                        .bindings
                        .iter()
                        .find(|b| b.binding == a.binding)
                        .map(|b| b.stride as u64)
                        .unwrap_or(0);
                    if stride == 0 {
                        continue;
                    }
                    let mut mx = [0f32; 4];
                    for vi in 0..96u64 {
                        let mut buf = [0u8; 16];
                        if mem_read(vcpu + a.offset as u64 + vi * stride, &mut buf) {
                            for c in 0..4 {
                                let f = f32::from_le_bytes([
                                    buf[c * 4],
                                    buf[c * 4 + 1],
                                    buf[c * 4 + 2],
                                    buf[c * 4 + 3],
                                ]);
                                if f.is_finite() && f.abs() <= 4.0 && f > mx[c] {
                                    mx[c] = f;
                                }
                            }
                        }
                    }
                    log::warn!(
                        "[shade] draw{} nattr={} loc={} fmt={:?} off={} maxRGBA=[{:.3} {:.3} {:.3} {:.3}]",
                        k,
                        layout.attrs.len(),
                        a.location,
                        a.format,
                        a.offset,
                        mx[0],
                        mx[1],
                        mx[2],
                        mx[3]
                    );
                }
                let vstride = layout.bindings.iter().map(|b| b.stride).max().unwrap_or(0) as usize;
                let n = vstride.min(80);
                if n >= 4 {
                    let mut raw = vec![0u8; n];
                    if mem_read(vcpu, &mut raw) {
                        let floats: Vec<String> = raw
                            .chunks_exact(4)
                            .map(|c| format!("{:.3}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                            .collect();
                        log::warn!(
                            "[vraw] draw{} stride={} v0=[{}]",
                            k,
                            vstride,
                            floats.join(" ")
                        );
                    }
                }
            }
        }
    }

    let depth_test = !no_depth && draw.depth_test_enable && draw.zeta_enable;
    let depth_write = !no_depth && draw.depth_write_enable && draw.zeta_enable;
    let depth_key = if depth_test || depth_write {
        zeta_key.or(Some(rt_key))
    } else {
        None
    };

    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if draw.depth_test_enable && !LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                "depth: first depth-test draw — zeta_enable={} fmt={:#x} {}x{} \
                 test={} write={} func={:#x}->{:?} use_depth={}",
                draw.zeta_enable,
                draw.zeta.format,
                draw.zeta.width,
                draw.zeta.height,
                draw.depth_test_enable,
                draw.depth_write_enable,
                draw.depth_func,
                map_compare_op(draw.depth_func),
                depth_key.is_some(),
            );
        }
    }

    let (fallback_cbuf_addr, fallback_cbuf_size) = {
        let (a, s) = resolve_vs_cbuf(&maxwell.regs.cbuf_binds);
        if a != 0 && std::env::var_os("NEXIUM_VS_CBUF").is_some() {
            (a, s)
        } else {
            (cbuf_addr, cbuf_size)
        }
    };
    let fallback_cbuf_size = fallback_cbuf_size.min(bundle.cbuf_used);
    let cbuf_data = pack_cbuf_data(
        &maxwell.regs.cbuf_binds,
        vs_cbuf_mask,
        fs_cbuf_mask,
        mappings,
        mem_read,
    )
    .or_else(|| {
        if fallback_cbuf_addr == 0 || fallback_cbuf_size == 0 {
            return None;
        }
        mappings
            .cpu_address_for(fallback_cbuf_addr)
            .and_then(|cpu| {
                let mut buf = vec![0u8; fallback_cbuf_size as usize];
                if mem_read(cpu, &mut buf) {
                    Some(buf)
                } else {
                    None
                }
            })
    });
    let (call_cbuf_addr, call_cbuf_size) = if let Some(data) = &cbuf_data {
        (0, data.len() as u32)
    } else {
        (fallback_cbuf_addr, fallback_cbuf_size)
    };

    trace_grade_discover(
        draw,
        rt_key,
        &color_rt_keys,
        &color_rt_formats,
        vs_addr,
        fs_addr,
        fs_output_map,
        &fs_input_map,
        vs_cbuf_mask,
        fs_cbuf_mask,
        &maxwell.regs.cbuf_binds,
        mappings,
        mem_read,
        &fs_tex_ids,
        &sampled_rt_slots,
        &bundle.fs_cbuf_reads,
    );

    let (out_index_data, out_index_count, out_index_type, eff_vertex_count) =
        if draw.indexed && draw.index_count > 0 && draw.index_gpu_va != 0 {
            let isz: usize = match draw.index_format {
                0 => 1,
                2 => 4,
                _ => 2,
            };
            let icount = draw.index_count as usize;
            let start = draw
                .index_gpu_va
                .wrapping_add((draw.index_first as u64) * isz as u64);
            let raw = mappings.cpu_address_for(start).and_then(|cpu| {
                let mut b = vec![0u8; icount * isz];
                if mem_read(cpu, &mut b) {
                    Some(b)
                } else {
                    None
                }
            });
            match raw {
                Some(bytes) => {
                    let (data, out_count, max_idx, itype) = match draw.index_format {
                        2 => {
                            let mut values = Vec::with_capacity(icount);
                            let mut mx = 0u32;
                            for c in bytes.chunks_exact(4) {
                                let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                values.push(v);
                                if v != 0xFFFF_FFFF && v > mx {
                                    mx = v;
                                }
                            }
                            if draw.topology == 7 {
                                let expanded = expand_quad_indices(&values);
                                let mut out = Vec::with_capacity(expanded.len() * 4);
                                for v in &expanded {
                                    out.extend_from_slice(&v.to_le_bytes());
                                }
                                (out, expanded.len(), mx, vk::IndexType::UINT32)
                            } else {
                                let count = values.len();
                                (bytes, count, mx, vk::IndexType::UINT32)
                            }
                        }
                        0 => {
                            let mut values = Vec::with_capacity(bytes.len());
                            let mut mx = 0u32;
                            for &b in &bytes {
                                values.push(b as u32);
                                if b as u32 > mx {
                                    mx = b as u32;
                                }
                            }
                            let values = if draw.topology == 7 {
                                expand_quad_indices(&values)
                            } else {
                                values
                            };
                            let mut wide = Vec::with_capacity(values.len() * 2);
                            for v in &values {
                                wide.extend_from_slice(&(*v as u16).to_le_bytes());
                            }
                            (wide, values.len(), mx, vk::IndexType::UINT16)
                        }
                        _ => {
                            let mut values = Vec::with_capacity(bytes.len() / 2);
                            let mut mx = 0u32;
                            for c in bytes.chunks_exact(2) {
                                let v = u16::from_le_bytes([c[0], c[1]]) as u32;
                                values.push(v);
                                if v != 0xFFFF && v > mx {
                                    mx = v;
                                }
                            }
                            if draw.topology == 7 {
                                let expanded = expand_quad_indices(&values);
                                let mut out = Vec::with_capacity(expanded.len() * 2);
                                for v in &expanded {
                                    out.extend_from_slice(&(*v as u16).to_le_bytes());
                                }
                                (out, expanded.len(), mx, vk::IndexType::UINT16)
                            } else {
                                let count = values.len();
                                (bytes, count, mx, vk::IndexType::UINT16)
                            }
                        }
                    };
                    (Some(data), Some(out_count as u32), itype, max_idx + 1)
                }
                None => (None, None, vk::IndexType::UINT16, draw.vertex_count),
            }
        } else {
            (None, None, vk::IndexType::UINT16, draw.vertex_count)
        };
    let is_indexed = out_index_count.is_some();

    let attachments: [BlendAttachmentState; 8] = std::array::from_fn(|rt| {
        let (
            blend_raw_src,
            blend_raw_dst,
            blend_raw_eq,
            blend_raw_src_alpha,
            blend_raw_dst_alpha,
            blend_raw_eq_alpha,
        ) = if maxwell.regs.blend_per_target_enabled {
            (
                maxwell.regs.blend_pt_src_rgb[rt],
                maxwell.regs.blend_pt_dst_rgb[rt],
                maxwell.regs.blend_pt_eq_rgb[rt],
                maxwell.regs.blend_pt_src_alpha[rt],
                maxwell.regs.blend_pt_dst_alpha[rt],
                maxwell.regs.blend_pt_eq_alpha[rt],
            )
        } else {
            (
                maxwell.regs.blend_src_rgb,
                maxwell.regs.blend_dst_rgb,
                maxwell.regs.blend_eq_rgb,
                maxwell.regs.blend_src_alpha,
                maxwell.regs.blend_dst_alpha,
                maxwell.regs.blend_eq_alpha,
            )
        };
        let mask_rt = if maxwell.regs.color_mask_common {
            0
        } else {
            rt
        };
        let shader_mask = map_output_component_mask(fragment_output_mask(fs_output_map, rt as u32));
        BlendAttachmentState {
            enabled: maxwell.regs.blend_enable[rt] && std::env::var("NEXIUM_NO_BLEND").is_err(),
            src_factor: map_blend_factor(blend_raw_src),
            dst_factor: map_blend_factor(blend_raw_dst),
            op: map_blend_op(blend_raw_eq),
            src_alpha_factor: map_blend_factor(blend_raw_src_alpha),
            dst_alpha_factor: map_blend_factor(blend_raw_dst_alpha),
            alpha_op: map_blend_op(blend_raw_eq_alpha),
            color_write_mask: map_color_write_mask(maxwell.regs.color_masks[mask_rt]) & shader_mask,
        }
    });
    let compact_attachments: [BlendAttachmentState; 8] = std::array::from_fn(|index| {
        let location = color_rt_locations.get(index).copied().unwrap_or(index).min(7);
        attachments[location]
    });
    let blend_state = BlendState {
        enabled: compact_attachments[0].enabled,
        src_factor: compact_attachments[0].src_factor,
        dst_factor: compact_attachments[0].dst_factor,
        op: compact_attachments[0].op,
        src_alpha_factor: compact_attachments[0].src_alpha_factor,
        dst_alpha_factor: compact_attachments[0].dst_alpha_factor,
        alpha_op: compact_attachments[0].alpha_op,
        color_write_mask: compact_attachments[0].color_write_mask,
        attachments: compact_attachments,
    };

    trace_menu_draw(
        draw,
        &layout,
        mappings,
        mem_read,
        nvmap_id,
        rt_slot,
        rt,
        vs_addr,
        fs_addr,
        vertex_addr,
        eff_vertex_count,
        out_index_count.unwrap_or(0),
        is_indexed,
        &shader_fs_tex_ids,
        &fs_tex_ids,
        &fs_sampler_ids,
        &fs_tex_remap,
        &sampled_rt_slots,
        depth_test,
        depth_write,
        &blend_state,
        vs_cbuf_mask,
        fs_cbuf_mask,
        cbuf_data.as_deref(),
        &maxwell.regs.cbuf_binds,
    );

    trace_draw(
        draw,
        &layout,
        mappings,
        mem_read,
        nvmap_id,
        rt,
        &color_rt_keys,
        &color_rt_locations,
        vertex_addr,
        eff_vertex_count,
        out_index_count.unwrap_or(0),
        is_indexed,
        &fs_tex_ids,
        &sampled_rt_slots,
        &color_rt_formats,
        depth_test,
        depth_write,
        (
            blend_state.enabled,
            maxwell.regs.blend_per_target_enabled,
            attachments[0].src_factor.as_raw() as u32,
            attachments[0].dst_factor.as_raw() as u32,
            attachments[0].op.as_raw() as u32,
            attachments[0].src_alpha_factor.as_raw() as u32,
            attachments[0].dst_alpha_factor.as_raw() as u32,
            attachments[0].alpha_op.as_raw() as u32,
        ),
        &blend_state,
        fs_output_map,
        &maxwell.regs.color_masks,
        maxwell.regs.color_mask_common,
        vs_cbuf_mask,
        fs_cbuf_mask,
        &maxwell.regs.cbuf_binds,
        cbuf_data.as_deref(),
        &bundle.fs_cbuf_reads,
    );
    trace_cbuf_watch(
        draw,
        nvmap_id,
        rt_key,
        vs_addr,
        fs_addr,
        vs_cbuf_mask,
        fs_cbuf_mask,
        &maxwell.regs.cbuf_binds,
        cbuf_data.as_deref(),
        &fs_tex_ids,
        &sampled_rt_slots,
        &bundle.fs_cbuf_reads,
    );

    let ssbo_dbg = std::env::var_os("NEXIUM_SSBO_DBG").is_some();
    if ssbo_dbg && !bundle.ssbo_descs.is_empty() {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<std::collections::HashSet<u64>>> = OnceLock::new();
        let s = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
        if let Ok(mut set) = s.lock() {
            if set.insert(vs_addr) {
                let descs: Vec<String> = bundle
                    .ssbo_descs
                    .iter()
                    .map(|d| format!("bind{}@{:#x}/al{}", d.cbuf_binding, d.cbuf_offset, d.align))
                    .collect();
                log::warn!(
                    "[ssbo] vs={:#x} num_ssbo={} descs=[{}]",
                    vs_addr,
                    bundle.ssbo_descs.len(),
                    descs.join(" ")
                );
            }
        }
    }
    let mut ssbo_data: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut ssbo_meta: Vec<(u64, usize, bool)> = Vec::new();
    for (idx, d) in bundle.ssbo_descs.iter().enumerate() {
        let mut bytes: Vec<u8> = vec![0u8; 16];
        let (cb_va, _cb_sz) = maxwell.regs.cbuf_binds[0][(d.cbuf_binding as usize).min(15)];
        let mut dbg_base: u64 = 0;
        let mut dbg_slack: usize = 0;
        let mut dbg_size: u32 = 0;
        let mut dbg_readok = false;
        if cb_va != 0 {
            if let Some(desc_cpu) =
                mappings.cpu_address_for(cb_va.wrapping_add(d.cbuf_offset as u64))
            {
                let mut desc = [0u8; 16];
                if mem_read(desc_cpu, &mut desc) {
                    if ssbo_dbg {
                        log::warn!("[ssbo-raw] vs={:#x} desc16={:02x?}", vs_addr, desc);
                    }
                    let base_lo = u32::from_le_bytes([desc[0], desc[1], desc[2], desc[3]]) as u64;
                    let base_hi = u32::from_le_bytes([desc[4], desc[5], desc[6], desc[7]]) as u64;
                    let size = u32::from_le_bytes([desc[8], desc[9], desc[10], desc[11]]);
                    let base = (base_hi << 32) | base_lo;
                    dbg_base = base;
                    dbg_size = size;
                    let align = (d.align.max(1)) as u64;
                    let aligned = base & !(align - 1);
                    let slack = (base - aligned) as usize;
                    dbg_slack = slack;
                    if ssbo_dbg {
                        log::warn!(
                            "[ssbo-map] vs={:#x} base={:#x} mapped={} any32={:x?} {}",
                            vs_addr,
                            base,
                            mappings.cpu_address_for(aligned).is_some(),
                            mappings.cpu_address_for_any32(aligned),
                            mappings.bracket(aligned)
                        );
                    }
                    if base != 0 {
                        let (buf_cpu, remaining) = match mappings.cpu_range_for(aligned) {
                            Some((cpu, rem)) => (Some(cpu), rem),
                            None => match mappings.cpu_address_for_any32(aligned) {
                                Some((_, cpu, rem)) => (Some(cpu), rem),
                                None => (None, 0),
                            },
                        };
                        if let Some(buf_cpu) = buf_cpu {
                            let want = if size != 0 {
                                (size as usize) + slack
                            } else {
                                0x40000
                            };
                            let read_size = want.min(remaining as usize).clamp(16, 8 * 1024 * 1024);
                            let mut b = vec![0u8; read_size];
                            if mem_read(buf_cpu, &mut b) {
                                bytes = b;
                                dbg_readok = true;
                            }
                        }
                    }
                }
            }
        }
        if ssbo_dbg {
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<std::collections::HashSet<(u64, u32)>>> = OnceLock::new();
            let s = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
            if let Ok(mut set) = s.lock() {
                if set.insert((vs_addr, idx as u32)) {
                    log::warn!(
                        "[ssbo] vs={:#x} ssbo[{}] cb_va={:#x} base={:#x} size={:#x} read_ok={} bytes={}",
                        vs_addr,
                        idx,
                        cb_va,
                        dbg_base,
                        dbg_size,
                        dbg_readok,
                        bytes.len()
                    );
                }
            }
        }
        ssbo_data.push((idx as u32, bytes));
        ssbo_meta.push((dbg_base, dbg_slack, dbg_readok));
    }
    trace_font_ssbo(
        vs_addr,
        fs_addr,
        &ssbo_data,
        &ssbo_meta,
        cbuf_data.as_deref(),
        &fs_tex_ids,
        draw.tic_pool_gpu_va,
        draw.tic_pool_limit,
        mappings,
        mem_read,
        eff_vertex_count,
        out_index_count.unwrap_or(0),
        out_index_type,
        out_index_data.as_deref(),
    );

    let call = Maxwell3dDrawCall {
        vs_spirv,
        fs_spirv,
        vs_gpu_va: vs_addr,
        fs_gpu_va: fs_addr,
        vs_hash: bundle.vs_hash,
        fs_hash: bundle.fs_hash,
        vs_cbuf_mask,
        fs_cbuf_mask,
        fs_tex_ids,
        vertex_layout: layout,
        cbuf_addr: call_cbuf_addr,
        cbuf_size: call_cbuf_size,
        cbuf_data,
        vertex_addr,
        vertex_bindings,
        vertex_count: eff_vertex_count,
        first_vertex: draw.first_vertex,
        instance_count: draw.instance_count.max(1),
        first_instance: draw.first_instance,
        index_addr: None,
        index_count: out_index_count,
        index_type: out_index_type,
        index_data: out_index_data,
        quad_expand: draw.topology == 7 && out_index_count.is_none(),
        rt_key,
        color_rt_keys,
        color_rt_formats,
        rt_format,
        vp_rect: guest_viewport_rect(
            draw,
            rt.width as f32,
            rt.height as f32,
            signed_viewport_nvmap(rt_key.nvmap_id),
        ),
        scissor: scissor_rect(draw, rt.width, rt.height),
        state: DrawState {
            topology,
            vertex_count: eff_vertex_count,
            index_count: out_index_count.unwrap_or(0),
            indexed: is_indexed,
        },
        blend: blend_state,
        depth: DepthState {
            test_enabled: depth_test,
            write_enabled: depth_write,
            compare_op: map_compare_op(draw.depth_func),
        },
        depth_clamp_enabled: draw.viewport_clip_control.depth_clamp_enabled(),
        depth_key,
        sampled_rt_key,
        sampled_rt_keys,
        sampled_rt_slots,
        sampled_rt_fuzzy,
        clear: false,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        tic_pool_gpu_va: draw.tic_pool_gpu_va,
        tic_pool_limit: draw.tic_pool_limit,
        tsc_pool_gpu_va: draw.tsc_pool_gpu_va,
        tsc_pool_limit: draw.tsc_pool_limit,
        fs_sampler_ids,
        fs_sampler_arrayed,
        cull_test_enable: draw.cull_test_enable,
        cull_face: draw.cull_face,
        front_face: if draw.window_origin.flip_y() {
            flip_front_face(draw.front_face)
        } else {
            draw.front_face
        },
        poly_offset_enable: draw.poly_offset_fill_enable,
        poly_offset_units: draw.poly_offset_units,
        poly_offset_factor: draw.poly_offset_factor,
        ssbo_data,
        flip_y: draw.window_origin.flip_y(),
    };

    Ok(Some(call))
}

fn menu_draw_dbg_targets() -> Option<Vec<u32>> {
    use std::sync::OnceLock;
    static TARGETS: OnceLock<Option<Vec<u32>>> = OnceLock::new();
    TARGETS
        .get_or_init(|| {
            if std::env::var_os("NEXIUM_MENU_DRAW_DBG").is_none() {
                return None;
            }
            let raw = std::env::var("NEXIUM_MENU_DRAW_NVMAP").unwrap_or_else(|_| "16".to_string());
            let trimmed = raw.trim();
            if trimmed.eq_ignore_ascii_case("all") || trimmed == "*" {
                return Some(Vec::new());
            }
            let targets = trimmed
                .split(',')
                .filter_map(|v| v.trim().parse::<u32>().ok())
                .collect::<Vec<_>>();
            if targets.is_empty() {
                Some(vec![16])
            } else {
                Some(targets)
            }
        })
        .clone()
}

fn rt_control_target(raw: u32, index: usize) -> usize {
    (((raw >> (4 + index * 3)) & 0x7) as usize).min(7)
}

fn rt_key_for_target(rt: &RenderTarget, mappings: &GpuMappings) -> Option<RtKey> {
    if rt.width == 0 || rt.height == 0 {
        return None;
    }
    let gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings.nvmap_id_for(gpu_va)?;
    Some(RtKey::with_cpu(
        nvmap_id,
        rt.width,
        rt.height,
        gpu_va,
        mappings.cpu_address_for(gpu_va).unwrap_or(0),
    ))
}

fn zeta_rt_key(draw: &DrawCall, mappings: &GpuMappings, fallback: RtKey) -> Option<RtKey> {
    if !draw.zeta_enable {
        return None;
    }
    let gpu_va = ((draw.zeta.address_hi as u64) << 32) | draw.zeta.address_lo as u64;
    if gpu_va == 0 {
        return None;
    }
    let nvmap_id = mappings.nvmap_id_for(gpu_va)?;
    let width = fallback.width;
    let height = fallback.height;
    if width == 0 || height == 0 {
        return None;
    }
    Some(RtKey::with_cpu(
        nvmap_id,
        width,
        height,
        gpu_va,
        mappings.cpu_address_for(gpu_va).unwrap_or(0),
    ))
}

fn active_color_rts(
    draw: &DrawCall,
    mappings: &GpuMappings,
    fallback_key: RtKey,
    fallback_format: vk::Format,
    output_map: u32,
) -> Vec<(usize, RtKey, vk::Format)> {
    let count = (draw.rt_control & 0xf).min(8) as usize;
    let mut keys = Vec::new();
    if count != 0 {
        for index in active_fragment_output_locations(count, output_map) {
            let slot = rt_control_target(draw.rt_control, index);
            if let Some((key, format)) = draw.rt.get(slot).and_then(|rt| {
                rt_key_for_target(rt, mappings)
                    .map(|key| (key, map_rt_format_for_key(rt.format, key)))
            }) {
                keys.push((index, key, format));
            }
        }
    }
    if keys.is_empty() {
        keys.push((0, fallback_key, fallback_format));
    }
    keys
}

fn active_fragment_output_locations(count: usize, output_map: u32) -> Vec<usize> {
    if output_map == 0 {
        return (0..count).collect();
    }
    (0..count)
        .filter(|index| fragment_output_mask(output_map, *index as u32) != 0)
        .collect()
}

fn draw_color_rt_slot(draw: &DrawCall) -> usize {
    let slot = rt_control_target(draw.rt_control, 0);
    let count = (draw.rt_control & 0xf).min(8);
    if count != 0 && slot < draw.rt.len() && draw.rt[slot].width != 0 && draw.rt[slot].height != 0 {
        slot
    } else {
        0
    }
}

fn map_rt_format(format: u32) -> vk::Format {
    match format {
        0xC0 | 0xC3 => vk::Format::R32G32B32A32_SFLOAT,
        0xC1 | 0xC4 => vk::Format::R32G32B32A32_SINT,
        0xC2 | 0xC5 => vk::Format::R32G32B32A32_UINT,
        0xC6 => vk::Format::R16G16B16A16_UNORM,
        0xC7 => vk::Format::R16G16B16A16_SNORM,
        0xC8 => vk::Format::R16G16B16A16_SINT,
        0xC9 => vk::Format::R16G16B16A16_UINT,
        0xCA | 0xCE => vk::Format::R16G16B16A16_SFLOAT,
        0xCB => vk::Format::R32G32_SFLOAT,
        0xCC => vk::Format::R32G32_SINT,
        0xCD => vk::Format::R32G32_UINT,
        0xCF | 0xE6 => vk::Format::B8G8R8A8_UNORM,
        0xD1 => vk::Format::A2B10G10R10_UNORM_PACK32,
        0xD2 => vk::Format::A2B10G10R10_UINT_PACK32,
        0xD5 | 0xF9 => vk::Format::A8B8G8R8_UNORM_PACK32,
        0xD6 | 0xFA => vk::Format::A8B8G8R8_SRGB_PACK32,
        0xD7 => vk::Format::A8B8G8R8_SNORM_PACK32,
        0xD8 => vk::Format::A8B8G8R8_SINT_PACK32,
        0xD9 => vk::Format::A8B8G8R8_UINT_PACK32,
        0xDA => vk::Format::R16G16_UNORM,
        0xDB => vk::Format::R16G16_SNORM,
        0xDC => vk::Format::R16G16_SINT,
        0xDD => vk::Format::R16G16_UINT,
        0xDE => vk::Format::R16G16_SFLOAT,
        0xDF => vk::Format::A2R10G10B10_UNORM_PACK32,
        0xE0 => vk::Format::B10G11R11_UFLOAT_PACK32,
        0xE3 => vk::Format::R32_SINT,
        0xE4 => vk::Format::R32_UINT,
        0xE5 => vk::Format::R32_SFLOAT,
        0xE8 => vk::Format::R5G6B5_UNORM_PACK16,
        0xEA => vk::Format::R8G8_UNORM,
        0xEB => vk::Format::R8G8_SNORM,
        0xEC => vk::Format::R8G8_SINT,
        0xED => vk::Format::R8G8_UINT,
        0xEE => vk::Format::R16_UNORM,
        0xEF => vk::Format::R16_SNORM,
        0xF0 => vk::Format::R16_SINT,
        0xF1 => vk::Format::R16_UINT,
        0xF2 => vk::Format::R16_SFLOAT,
        0xF3 => vk::Format::R8_UNORM,
        0xF4 => vk::Format::R8_SNORM,
        0xF5 => vk::Format::R8_SINT,
        0xF6 => vk::Format::R8_UINT,
        _ => vk::Format::R8G8B8A8_UNORM,
    }
}

fn map_rt_format_for_key(format: u32, _key: RtKey) -> vk::Format {
    map_rt_format(format)
}

fn menu_draw_dbg_color_only() -> bool {
    use std::sync::OnceLock;
    static COLOR_ONLY: OnceLock<bool> = OnceLock::new();
    *COLOR_ONLY.get_or_init(|| {
        if std::env::var_os("NEXIUM_MENU_DRAW_DBG").is_none() {
            return false;
        }
        std::env::var_os("NEXIUM_MENU_DRAW_COLOR_ONLY").is_some()
    })
}

fn trace_font_ssbo(
    vs_addr: u64,
    fs_addr: u64,
    ssbo_data: &[(u32, Vec<u8>)],
    ssbo_meta: &[(u64, usize, bool)],
    cbuf_data: Option<&[u8]>,
    fs_tex_ids: &[u32],
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    vertex_count: u32,
    index_count: u32,
    index_type: vk::IndexType,
    index_data: Option<&[u8]>,
) {
    if std::env::var_os("NEXIUM_FONT_SSBO_DBG").is_none() || vs_addr != 0x400660030 {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u64> = OnceLock::new();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_FONT_SSBO_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(64)
    });
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    if seq >= limit {
        return;
    }
    let c3 = font_cbuf_summary(cbuf_data);
    let indices = font_indices(vertex_count, index_count, index_type, index_data);
    for (idx, (_, bytes)) in ssbo_data.iter().enumerate() {
        let (base, slack, read_ok) = ssbo_meta.get(idx).copied().unwrap_or((0, 0, false));
        let mut verts = Vec::new();
        for v in indices.iter().take(12) {
            let off = slack.saturating_add((*v as usize).saturating_mul(64));
            verts.push(font_vertex_words(bytes, off));
        }
        log::warn!(
            "[font-ssbo] #{} vs={:#x} fs={:#x} tex={:?} v={} i={} idx={:?} ssbo={} base={:#x} slack={} read_ok={} bytes={} c3=[{}] {}",
            seq,
            vs_addr,
            fs_addr,
            fs_tex_ids,
            vertex_count,
            index_count,
            indices,
            idx,
            base,
            slack,
            read_ok,
            bytes.len(),
            c3,
            verts.join(" ")
        );
        font_tex_probe(
            seq,
            fs_tex_ids,
            tic_pool_gpu_va,
            tic_pool_limit,
            mappings,
            mem_read,
            bytes,
            slack,
            &indices,
        );
    }
}

fn font_indices(
    vertex_count: u32,
    index_count: u32,
    index_type: vk::IndexType,
    index_data: Option<&[u8]>,
) -> Vec<u32> {
    if index_count == 0 {
        return (0..vertex_count.min(12)).collect();
    }
    let Some(data) = index_data else {
        return (0..vertex_count.min(12)).collect();
    };
    let mut out = Vec::new();
    match index_type {
        vk::IndexType::UINT32 => {
            for c in data.chunks_exact(4).take(index_count as usize).take(12) {
                out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        _ => {
            for c in data.chunks_exact(2).take(index_count as usize).take(12) {
                out.push(u16::from_le_bytes([c[0], c[1]]) as u32);
            }
        }
    }
    out
}

fn font_vertex_words(bytes: &[u8], off: usize) -> String {
    let mut vals = Vec::new();
    for i in 0..16usize {
        let p = off.saturating_add(i * 4);
        if p + 4 > bytes.len() {
            vals.push("out".to_string());
            continue;
        }
        let raw = u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
        vals.push(format!("{:#010x}/{:.4}", raw, f32::from_bits(raw)));
    }
    format!("v{}=[{}]", off / 64, vals.join(","))
}

fn font_cbuf_summary(cbuf_data: Option<&[u8]>) -> String {
    let Some(data) = cbuf_data else {
        return "none".to_string();
    };
    let base = 3usize * PACKED_CBUF_SLOT_SIZE;
    let offsets = [
        0x0usize, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38,
        0x3c, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x58, 0x5c,
    ];
    let mut out = Vec::new();
    for off in offsets {
        let p = base + off;
        if p + 4 > data.len() {
            continue;
        }
        let raw = u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
        out.push(format!(
            "{:#x}={:.4}/{:#010x}",
            off,
            f32::from_bits(raw),
            raw
        ));
    }
    out.join(" ")
}

fn font_tex_probe(
    seq: u64,
    fs_tex_ids: &[u32],
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ssbo: &[u8],
    slack: usize,
    indices: &[u32],
) {
    if std::env::var_os("NEXIUM_FONT_TEX_PROBE").is_none() || tic_pool_gpu_va == 0 {
        return;
    }
    let start = std::env::var("NEXIUM_FONT_TEX_PROBE_START")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let end = std::env::var("NEXIUM_FONT_TEX_PROBE_END")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(u64::MAX);
    if seq < start || seq > end {
        return;
    }
    let target_tic = std::env::var("NEXIUM_FONT_TEX_PROBE_TIC")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    for (slot, tex_id) in fs_tex_ids.iter().enumerate() {
        if *tex_id == u32::MAX || *tex_id > tic_pool_limit {
            continue;
        }
        if target_tic.is_some_and(|target| target != *tex_id) {
            continue;
        }
        let Some((tic, rgba, layers, layer_size)) =
            font_read_texture(tic_pool_gpu_va, *tex_id, mappings, mem_read)
        else {
            log::warn!("[font-tex] #{} slot={} tic{} unreadable", seq, slot, tex_id);
            continue;
        };
        let mut seen = Vec::new();
        let mut samples = Vec::new();
        for idx in indices {
            if seen.contains(idx) {
                continue;
            }
            seen.push(*idx);
            if samples.len() >= 6 {
                break;
            }
            let off = slack.saturating_add((*idx as usize).saturating_mul(64));
            let Some(words) = font_vertex_raw_words(ssbo, off) else {
                continue;
            };
            let u0 = f32::from_bits(words[4]);
            let v0 = f32::from_bits(words[5]);
            let u1 = f32::from_bits(words[6]);
            let v1 = f32::from_bits(words[7]);
            let layer = (words[10] & 0xffff).min(layers.saturating_sub(1));
            samples.push(font_tex_sample_summary(
                *idx,
                &rgba,
                layer_size,
                tic.width,
                tic.height,
                layer,
                tic.swizzle,
                u0,
                v0,
                u1,
                v1,
            ));
        }
        log::warn!(
            "[font-tex] #{} slot={} tic{} fmt={:?} {}x{}x{} type={} base={} norm={} bl={} bh={} va={:#x} nvmap={:?} swz={:?} samples={}",
            seq,
            slot,
            tex_id,
            tic.format,
            tic.width,
            tic.height,
            tic.depth,
            tic.texture_type,
            tic.base_layer,
            tic.normalized_coords,
            tic.is_block_linear,
            tic.block_height_log2,
            tic.gpu_va,
            mappings.nvmap_id_for(tic.gpu_va),
            tic.swizzle,
            samples.join(" ")
        );
        if target_tic.is_some() {
            break;
        }
    }
}

fn font_read_texture(
    tic_pool_gpu_va: u64,
    tex_id: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<(nexium_gpu::texture::TicEntry, Vec<u8>, u32, usize)> {
    let tic_addr = tic_pool_gpu_va.wrapping_add((tex_id as u64).saturating_mul(32));
    let tic_cpu = mappings.cpu_address_for(tic_addr)?;
    let mut tic_raw = [0u8; 32];
    if !mem_read(tic_cpu, &mut tic_raw) {
        return None;
    }
    let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw)?;
    let pitch_size = tic.format.linear_size(tic.width, tic.height);
    let layers = font_tic_layer_count(&tic);
    let layer_read_size = font_tic_layer_read_size(&tic, pitch_size);
    let read_size = layer_read_size.saturating_mul(layers as usize);
    let (tex_cpu, remaining) = mappings.cpu_range_for(tic.gpu_va)?;
    let read_size = read_size.min(remaining as usize);
    let mut raw = vec![0u8; read_size];
    if !mem_read(tex_cpu, &mut raw) {
        return None;
    }
    let rgba = font_decode_texture_layers(&raw, &tic, pitch_size, layer_read_size, layers);
    let layer_size = tic.width as usize * tic.height as usize * 4;
    Some((tic, rgba, layers, layer_size))
}

fn font_tic_is_arrayed(tic: &nexium_gpu::texture::TicEntry) -> bool {
    tic.texture_type == 5
}

fn font_tic_layer_count(tic: &nexium_gpu::texture::TicEntry) -> u32 {
    if font_tic_is_arrayed(tic) {
        tic.base_layer.saturating_add(tic.depth).max(1)
    } else {
        1
    }
}

fn font_tic_layer_read_size(tic: &nexium_gpu::texture::TicEntry, pitch_size: usize) -> usize {
    if tic.is_block_linear {
        tic.format
            .block_linear_size(tic.width, tic.height, tic.block_height_log2)
            .max(pitch_size)
    } else {
        pitch_size
    }
}

fn font_decode_texture_layers(
    raw: &[u8],
    tic: &nexium_gpu::texture::TicEntry,
    pitch_size: usize,
    layer_read_size: usize,
    layers: u32,
) -> Vec<u8> {
    let layer_rgba_size = tic.width as usize * tic.height as usize * 4;
    let mut out = Vec::with_capacity(layer_rgba_size.saturating_mul(layers as usize));
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    for layer in 0..layers as usize {
        let start = layer.saturating_mul(layer_read_size);
        let end = (start + layer_read_size).min(raw.len());
        let layer_raw = if start < raw.len() {
            &raw[start..end]
        } else {
            &[]
        };
        let linear = if effective_block_linear {
            let (storage_width, storage_height, bpp) =
                tic.format.storage_extent(tic.width, tic.height);
            nexium_gpu::texture::unswizzle_block_linear(
                layer_raw,
                storage_width,
                storage_height,
                bpp,
                tic.block_height_log2,
            )
        } else if layer_raw.len() >= pitch_size {
            layer_raw[..pitch_size].to_vec()
        } else {
            layer_raw.to_vec()
        };
        let mut decoded =
            nexium_gpu::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        out.extend(decoded);
    }
    out.resize(layer_rgba_size.saturating_mul(layers as usize), 0);
    out
}

fn font_vertex_raw_words(bytes: &[u8], off: usize) -> Option<[u32; 16]> {
    if off + 64 > bytes.len() {
        return None;
    }
    let mut words = [0u32; 16];
    for (i, word) in words.iter_mut().enumerate() {
        let p = off + i * 4;
        *word = u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
    }
    Some(words)
}

fn font_tex_sample_summary(
    idx: u32,
    rgba: &[u8],
    layer_size: usize,
    width: u32,
    height: u32,
    layer: u32,
    swizzle: [nexium_gpu::texture::SwizzleSource; 4],
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
) -> String {
    let direct = font_tex_region_stats(
        rgba, layer_size, width, height, layer, swizzle, u0, v0, u1, v1, false,
    );
    let flipped = font_tex_region_stats(
        rgba, layer_size, width, height, layer, swizzle, u0, v0, u1, v1, true,
    );
    format!(
        "v{} uv=({:.4},{:.4})-({:.4},{:.4}) l{} dir[{}] flip[{}]",
        idx, u0, v0, u1, v1, layer, direct, flipped
    )
}

fn font_tex_region_stats(
    rgba: &[u8],
    layer_size: usize,
    width: u32,
    height: u32,
    layer: u32,
    swizzle: [nexium_gpu::texture::SwizzleSource; 4],
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    flip_v: bool,
) -> String {
    if width == 0
        || height == 0
        || !u0.is_finite()
        || !v0.is_finite()
        || !u1.is_finite()
        || !v1.is_finite()
    {
        return "bad".to_string();
    }
    let x0 = font_tex_coord_to_px(u0.min(u1), width);
    let x1 = font_tex_coord_to_px(u0.max(u1), width).max(x0);
    let (a, b) = if flip_v {
        (1.0 - v0, 1.0 - v1)
    } else {
        (v0, v1)
    };
    let y0 = font_tex_coord_to_px(a.min(b), height);
    let y1 = font_tex_coord_to_px(a.max(b), height).max(y0);
    let mut raw0 = FontTexStats::default();
    let mut raw3 = FontTexStats::default();
    let mut map0 = FontTexStats::default();
    let mut map3 = FontTexStats::default();
    let base = layer as usize * layer_size;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = base + ((y as usize * width as usize + x as usize) * 4);
            if p + 4 > rgba.len() {
                continue;
            }
            let src = [rgba[p], rgba[p + 1], rgba[p + 2], rgba[p + 3]];
            let mapped = font_apply_swizzle(src, swizzle);
            raw0.add(src[0]);
            raw3.add(src[3]);
            map0.add(mapped[0]);
            map3.add(mapped[3]);
        }
    }
    format!(
        "px={}..{}/{}..{} r0={} r3={} m0={} m3={}",
        x0, x1, y0, y1, raw0, raw3, map0, map3
    )
}

fn font_tex_coord_to_px(v: f32, extent: u32) -> u32 {
    let max = extent.saturating_sub(1) as f32;
    (v.clamp(0.0, 1.0) * max).round() as u32
}

fn font_apply_swizzle(src: [u8; 4], swizzle: [nexium_gpu::texture::SwizzleSource; 4]) -> [u8; 4] {
    fn one(src: [u8; 4], s: nexium_gpu::texture::SwizzleSource) -> u8 {
        match s {
            nexium_gpu::texture::SwizzleSource::Zero => 0,
            nexium_gpu::texture::SwizzleSource::R => src[0],
            nexium_gpu::texture::SwizzleSource::G => src[1],
            nexium_gpu::texture::SwizzleSource::B => src[2],
            nexium_gpu::texture::SwizzleSource::A => src[3],
            nexium_gpu::texture::SwizzleSource::One => 255,
            nexium_gpu::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    [
        one(src, swizzle[0]),
        one(src, swizzle[1]),
        one(src, swizzle[2]),
        one(src, swizzle[3]),
    ]
}

#[derive(Default)]
struct FontTexStats {
    count: u64,
    nonzero: u64,
    sum: u64,
    min: u8,
    max: u8,
}

impl FontTexStats {
    fn add(&mut self, v: u8) {
        if self.count == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.count += 1;
        self.sum += v as u64;
        if v != 0 {
            self.nonzero += 1;
        }
    }
}

impl std::fmt::Display for FontTexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            return write!(f, "empty");
        }
        write!(
            f,
            "{:.1}/{:.3}/{}..{}",
            self.sum as f64 / self.count as f64,
            self.nonzero as f64 / self.count as f64,
            self.min,
            self.max
        )
    }
}

fn menu_draw_dbg_limit() -> u64 {
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("NEXIUM_MENU_DRAW_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(4000)
    })
}

fn menu_draw_dbg_start() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<u64> = OnceLock::new();
    *START.get_or_init(|| {
        std::env::var("NEXIUM_MENU_DRAW_START")
            .ok()
            .and_then(|v| parse_env_u64(&v))
            .unwrap_or(0)
    })
}

fn menu_draw_dbg_fs() -> Option<u64> {
    use std::sync::OnceLock;
    static FS: OnceLock<Option<u64>> = OnceLock::new();
    *FS.get_or_init(|| {
        std::env::var("NEXIUM_MENU_DRAW_FS")
            .ok()
            .and_then(|v| parse_env_u64(&v))
    })
}

fn parse_env_u64(v: &str) -> Option<u64> {
    let s = v.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

fn trace_menu_draw(
    draw: &DrawCall,
    layout: &VertexLayout,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    nvmap_id: u32,
    rt_slot: usize,
    rt: &RenderTarget,
    vs_addr: u64,
    fs_addr: u64,
    vertex_addr: u64,
    vertex_count: u32,
    index_count: u32,
    indexed: bool,
    shader_fs_tex_ids: &[u32],
    fs_tex_ids: &[u32],
    fs_sampler_ids: &[u32],
    fs_tex_remap: &[String],
    sampled_rt_slots: &[Option<RtKey>],
    depth_test: bool,
    depth_write: bool,
    blend: &BlendState,
    vs_cbuf_mask: u32,
    fs_cbuf_mask: u32,
    cbuf_data: Option<&[u8]>,
    cbuf_binds: &[[(u64, u32); 16]; 5],
) {
    let Some(targets) = menu_draw_dbg_targets() else {
        return;
    };
    if !targets.is_empty() && !targets.contains(&nvmap_id) {
        return;
    }
    if menu_draw_dbg_color_only() && blend.color_write_mask.is_empty() {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    if seq < menu_draw_dbg_start() {
        return;
    }
    if let Some(fs) = menu_draw_dbg_fs() {
        if fs_addr != fs {
            return;
        }
    }
    let logged = LOGGED.fetch_add(1, Ordering::Relaxed);
    if logged >= menu_draw_dbg_limit() {
        return;
    }
    let rt_gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let pos = position_bounds(layout, mappings, mem_read, vertex_addr, vertex_count)
        .unwrap_or_else(|| "n/a".to_string());
    let attr = vertex_attr_sample(layout, mappings, mem_read, vertex_addr, vertex_count);
    let vp = guest_viewport_rect(
        draw,
        rt.width as f32,
        rt.height as f32,
        signed_viewport_nvmap(nvmap_id),
    );
    let clip = draw.surface_clip.effective(rt.width, rt.height);
    let mut tex = Vec::new();
    for (slot, tex_id) in fs_tex_ids.iter().enumerate() {
        if *tex_id == u32::MAX || draw.tic_pool_gpu_va == 0 || *tex_id > draw.tic_pool_limit {
            tex.push(format!("s{}:tic{}=invalid", slot, tex_id));
            continue;
        }
        let tic_addr = draw.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
        let mut raw = [0u8; 32];
        let entry = mappings
            .cpu_address_for(tic_addr)
            .filter(|cpu| mem_read(*cpu, &mut raw))
            .and_then(|_| nexium_gpu::texture::TicEntry::parse(&raw));
        match entry {
            Some(tic) => tex.push(format!(
                "s{}:tic{} {:?} {}x{}x{} base={} type={} norm={} bl={} bh={} va={:#x} nvmap={:?} swz={:?} sampled={:?}",
                slot,
                tex_id,
                tic.format,
                tic.width,
                tic.height,
                tic.depth,
                tic.base_layer,
                tic.texture_type,
                tic.normalized_coords,
                tic.is_block_linear,
                tic.block_height_log2,
                tic.gpu_va,
                mappings.nvmap_id_for(tic.gpu_va),
                tic.swizzle,
                sampled_rt_slots.get(slot).copied().flatten()
            )),
            None => tex.push(format!("s{}:tic{}=unreadable", slot, tex_id)),
        }
    }
    let mut tsc = Vec::new();
    for (slot, tsc_id) in fs_sampler_ids.iter().enumerate() {
        if draw.tsc_pool_gpu_va == 0 || *tsc_id > draw.tsc_pool_limit {
            continue;
        }
        let tsc_addr = draw.tsc_pool_gpu_va.wrapping_add((*tsc_id as u64) * 32);
        let mut raw = [0u8; 32];
        let entry = mappings
            .cpu_address_for(tsc_addr)
            .filter(|cpu| mem_read(*cpu, &mut raw))
            .and_then(|_| nexium_gpu::texture::TscEntry::parse(&raw));
        if let Some(ts) = entry {
            tsc.push(format!(
                "s{}:tsc{} wrap=({:?},{:?},{:?}) filt=({:?},{:?},{:?})",
                slot,
                tsc_id,
                ts.wrap_u,
                ts.wrap_v,
                ts.wrap_p,
                ts.mag_filter,
                ts.min_filter,
                ts.mip_filter
            ));
        }
    }
    log::warn!(
        "[menu-draw] #{} rt={} slot={} rt_va={:#x} rtctl={:#x} {}x{} fmt={:#x} tile={:#x} depth={} stride={} base={} vs={:#x} fs={:#x} topo={} first={} inst={}/{} v={} i={} indexed={} pos={} attrs={} {} vp_en={} vp={:?} clip=({},{} {}x{}) depth={}/{} cull={} ff={:#x} blend={} {:?}/{:?}/{:?}/{:?} cw={:#x} masks={:#x}/{:#x} shader_tex_ids={:?} tex_ids={:?} tsc_ids={:?} remap=[{}] tex=[{}] tsc=[{}]",
        seq,
        nvmap_id,
        rt_slot,
        rt_gpu_va,
        draw.rt_control,
        rt.width,
        rt.height,
        rt.format,
        rt.tile_mode,
        rt.depth,
        rt.layer_stride,
        rt.base_layer,
        vs_addr,
        fs_addr,
        draw.topology,
        draw.first_vertex,
        draw.instance_count,
        draw.first_instance,
        vertex_count,
        index_count,
        indexed,
        pos,
        layout.attrs.len(),
        attr,
        draw.viewport_transform_en,
        vp,
        clip.x,
        clip.y,
        clip.width,
        clip.height,
        depth_test,
        depth_write,
        draw.cull_test_enable,
        draw.front_face,
        blend.enabled,
        blend.src_factor,
        blend.dst_factor,
        blend.src_alpha_factor,
        blend.dst_alpha_factor,
        blend.color_write_mask.as_raw(),
        vs_cbuf_mask,
        fs_cbuf_mask,
        shader_fs_tex_ids,
        fs_tex_ids,
        fs_sampler_ids,
        fs_tex_remap.join(" | "),
        tex.join(" | "),
        tsc.join(" | "),
    );
    if std::env::var_os("NEXIUM_MENU_CBUF_DBG").is_some() {
        log::warn!(
            "[menu-cbuf] #{} vs={:#x} {}",
            seq,
            vs_addr,
            menu_cbuf_sample(cbuf_data, cbuf_binds)
        );
    }
}

fn menu_cbuf_sample(cbuf_data: Option<&[u8]>, cbuf_binds: &[[(u64, u32); 16]; 5]) -> String {
    let Some(data) = cbuf_data else {
        return "none".to_string();
    };
    let offsets = [
        0x0u32, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38,
        0x3c, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68, 0x6c, 0x70, 0x74,
        0x78, 0x7c, 0x80, 0x84, 0x88, 0x8c, 0xe0, 0xe4, 0xe8, 0xec, 0xf0, 0xf4, 0xf8, 0xfc, 0x100,
        0x104, 0x108, 0x10c, 0x110, 0x114, 0x118, 0x11c, 0x1a0, 0x1a4, 0x1a8, 0x1ac, 0x1b0, 0x1b4,
        0x1c0, 0x1c4, 0x1d0, 0x1d4, 0x1d8, 0x1dc, 0x1e0, 0x1e4, 0x1e8, 0x1ec, 0x1f0, 0x1f4, 0x1f8,
        0x1fc, 0x200, 0x204, 0x208, 0x20c, 0x210, 0x214, 0x218, 0x21c, 0x22c,
    ];
    let logical_slot = 3usize;
    let base = logical_slot * PACKED_CBUF_SLOT_SIZE;
    let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot as u32);
    let mut vals = Vec::new();
    for off in offsets {
        let idx = base + off as usize;
        if idx + 4 > data.len() {
            vals.push(format!("{:#x}=out", off));
            continue;
        }
        let b = [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]];
        vals.push(format!(
            "{:#x}={:.6}/{:#010x}",
            off,
            f32::from_le_bytes(b),
            u32::from_le_bytes(b)
        ));
    }
    format!("s3({:#x}/{}) [{}]", addr, size, vals.join(" "))
}

fn guest_viewport_rect(
    draw: &DrawCall,
    rt_w: f32,
    rt_h: f32,
    force_signed: bool,
) -> Option<[f32; 4]> {
    if !draw.viewport_transform_en {
        let clip = draw.surface_clip.effective(rt_w as u32, rt_h as u32);
        let x = clip.x as f32;
        let y = clip.y as f32;
        let w = clip.width as f32;
        let h = clip.height as f32;
        if x <= 0.5 && y <= 0.5 && w >= rt_w - 0.5 && h >= rt_h - 0.5 {
            return None;
        }
        if w < 1.0 || h < 1.0 {
            return None;
        }
        return Some([x, y, w, h]);
    }
    let signed = force_signed || signed_viewport_rect();
    let sx = if signed {
        draw.viewport.scale_x
    } else {
        draw.viewport.scale_x.abs()
    };
    let sy = if signed {
        draw.viewport.scale_y
    } else {
        draw.viewport.scale_y.abs()
    };
    if sx == 0.0 || sy == 0.0 {
        return None;
    }
    let x = draw.viewport.translate_x - sx;
    let mut y = draw.viewport.translate_y - sy;
    let w = sx * 2.0;
    let mut h = sy * 2.0;
    if draw.window_origin.lower_left() {
        let clip = draw.surface_clip.effective(rt_w as u32, rt_h as u32);
        y += clip.height as f32;
        h = -h;
    }
    if draw.viewport.y_negate() {
        y += h;
        h = -h;
    }
    let min_y = y.min(y + h);
    let max_y = y.max(y + h);
    if (!signed || (w > 0.0 && h > 0.0))
        && !draw.window_origin.lower_left()
        && !draw.viewport.y_negate()
        && x <= 0.5
        && min_y <= 0.5
        && (x + w) >= rt_w - 0.5
        && max_y >= rt_h - 0.5
    {
        return None;
    }
    if w.abs() < 1.0 || h.abs() < 1.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some([x, y, w, h])
}

fn signed_viewport_rect() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_UNSIGNED_VIEWPORT").is_none())
}

fn signed_viewport_nvmap(nvmap_id: u32) -> bool {
    use std::sync::OnceLock;
    static IDS: OnceLock<Vec<u64>> = OnceLock::new();
    IDS.get_or_init(|| parse_env_u64_list("NEXIUM_SIGNED_VIEWPORT_NVMAPS"))
        .contains(&(nvmap_id as u64))
}

fn scissor_rect(draw: &DrawCall, rt_w: u32, rt_h: u32) -> Option<[i32; 4]> {
    if !draw.scissor.enabled {
        return None;
    }
    let clip = draw.surface_clip.effective(rt_w, rt_h);
    let x0 = draw.scissor.min_x.min(rt_w);
    let x1 = draw.scissor.max_x.min(rt_w);
    if x1 <= x0 {
        return None;
    }
    let (y0, y1) = if draw.window_origin.lower_left() {
        let clip_h = clip.height as i32;
        (
            (clip_h - draw.scissor.max_y as i32).max(0) as u32,
            (clip_h - draw.scissor.min_y as i32).max(0) as u32,
        )
    } else {
        (draw.scissor.min_y.min(rt_h), draw.scissor.max_y.min(rt_h))
    };
    let y0 = y0.min(rt_h);
    let y1 = y1.min(rt_h);
    if y1 <= y0 {
        return None;
    }
    Some([x0 as i32, y0 as i32, (x1 - x0) as i32, (y1 - y0) as i32])
}

fn flip_front_face(v: u32) -> u32 {
    match v {
        0x0900 => 0x0901,
        0x0901 => 0x0900,
        _ => v,
    }
}

fn map_color_write_mask(raw: u32) -> vk::ColorComponentFlags {
    let mut mask = vk::ColorComponentFlags::empty();
    if (raw & 0x1) != 0 {
        mask |= vk::ColorComponentFlags::R;
    }
    if (raw & 0x10) != 0 {
        mask |= vk::ColorComponentFlags::G;
    }
    if (raw & 0x100) != 0 {
        mask |= vk::ColorComponentFlags::B;
    }
    if (raw & 0x1000) != 0 {
        mask |= vk::ColorComponentFlags::A;
    }
    mask
}

fn map_output_component_mask(raw: u32) -> vk::ColorComponentFlags {
    let mut mask = vk::ColorComponentFlags::empty();
    if (raw & 0x1) != 0 {
        mask |= vk::ColorComponentFlags::R;
    }
    if (raw & 0x2) != 0 {
        mask |= vk::ColorComponentFlags::G;
    }
    if (raw & 0x4) != 0 {
        mask |= vk::ColorComponentFlags::B;
    }
    if (raw & 0x8) != 0 {
        mask |= vk::ColorComponentFlags::A;
    }
    mask
}

fn fragment_output_mask(output_map: u32, location: u32) -> u32 {
    if output_map == 0 {
        return 0xF;
    }
    if location >= 8 {
        return 0;
    }
    (output_map >> (location * 4)) & 0xF
}

fn map_blend_factor(v: u32) -> vk::BlendFactor {
    match v {
        0x4000 => vk::BlendFactor::ZERO,
        0x4001 => vk::BlendFactor::ONE,
        0x4300 => vk::BlendFactor::SRC_COLOR,
        0x4301 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        0x4302 => vk::BlendFactor::SRC_ALPHA,
        0x4303 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        0x4304 => vk::BlendFactor::DST_ALPHA,
        0x4305 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        0x4306 => vk::BlendFactor::DST_COLOR,
        0x4307 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        0x4308 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        0xC001 => vk::BlendFactor::CONSTANT_COLOR,
        0xC002 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        0xC003 => vk::BlendFactor::CONSTANT_ALPHA,
        0xC004 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        0xC900 => vk::BlendFactor::SRC1_COLOR,
        0xC901 => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        0xC902 => vk::BlendFactor::SRC1_ALPHA,
        0xC903 => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
        0x0001 => vk::BlendFactor::ZERO,
        0x0002 => vk::BlendFactor::ONE,
        0x0003 => vk::BlendFactor::SRC_COLOR,
        0x0004 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        0x0005 => vk::BlendFactor::SRC_ALPHA,
        0x0006 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        0x0007 => vk::BlendFactor::DST_ALPHA,
        0x0008 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        0x0009 => vk::BlendFactor::DST_COLOR,
        0x000A => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        0x000B => vk::BlendFactor::SRC_ALPHA_SATURATE,
        0x000E => vk::BlendFactor::CONSTANT_COLOR,
        0x000F => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        0x0010 => vk::BlendFactor::SRC1_COLOR,
        0x0011 => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        0x0012 => vk::BlendFactor::SRC1_ALPHA,
        0x0013 => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
        _ => {
            log::warn!(
                "map_blend_factor: unknown encoding {:#x}, defaulting to ONE",
                v
            );
            vk::BlendFactor::ONE
        }
    }
}

fn map_blend_op(v: u32) -> vk::BlendOp {
    match v {
        0x8006 => vk::BlendOp::ADD,
        0x800A => vk::BlendOp::SUBTRACT,
        0x800B => vk::BlendOp::REVERSE_SUBTRACT,
        0x8007 => vk::BlendOp::MIN,
        0x8008 => vk::BlendOp::MAX,
        0x0001 => vk::BlendOp::ADD,
        0x0002 => vk::BlendOp::SUBTRACT,
        0x0003 => vk::BlendOp::REVERSE_SUBTRACT,
        0x0004 => vk::BlendOp::MIN,
        0x0005 => vk::BlendOp::MAX,
        _ => {
            log::warn!("map_blend_op: unknown encoding {:#x}, defaulting to ADD", v);
            vk::BlendOp::ADD
        }
    }
}

fn map_compare_op(v: u32) -> vk::CompareOp {
    match v {
        0x200 | 0x1 => vk::CompareOp::NEVER,
        0x201 | 0x2 => vk::CompareOp::LESS,
        0x202 | 0x3 => vk::CompareOp::EQUAL,
        0x203 | 0x4 => vk::CompareOp::LESS_OR_EQUAL,
        0x204 | 0x5 => vk::CompareOp::GREATER,
        0x205 | 0x6 => vk::CompareOp::NOT_EQUAL,
        0x206 | 0x7 => vk::CompareOp::GREATER_OR_EQUAL,
        0x207 | 0x8 => vk::CompareOp::ALWAYS,
        _ => {
            log::warn!(
                "map_compare_op: unknown depth func {:#x}, defaulting to LESS_OR_EQUAL",
                v
            );
            vk::CompareOp::LESS_OR_EQUAL
        }
    }
}

#[derive(Clone)]
struct DrawTraceConfig {
    enabled: bool,
    start: u64,
    end: u64,
    fs: Vec<u64>,
    rt: Vec<u32>,
    rt_va: Vec<u64>,
    sampled_rt_only: bool,
}

fn draw_trace_config() -> DrawTraceConfig {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<DrawTraceConfig> = OnceLock::new();
    CONFIG
        .get_or_init(|| DrawTraceConfig {
            enabled: std::env::var_os("NEXIUM_DRAW_TRACE").is_some(),
            start: std::env::var("NEXIUM_DRAW_TRACE_START")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            end: std::env::var("NEXIUM_DRAW_TRACE_END")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(u64::MAX),
            fs: parse_env_u64_list("NEXIUM_DRAW_TRACE_FS"),
            rt: parse_env_u64_list("NEXIUM_DRAW_TRACE_RT")
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            rt_va: parse_env_u64_list("NEXIUM_DRAW_TRACE_RT_VA"),
            sampled_rt_only: std::env::var_os("NEXIUM_DRAW_TRACE_SAMPLED_RT").is_some(),
        })
        .clone()
}

fn parse_env_u64_list(name: &str) -> Vec<u64> {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|part| parse_env_u64(part.trim()))
                .collect()
        })
        .unwrap_or_default()
}

fn trace_grade_discover(
    draw: &DrawCall,
    fallback_key: RtKey,
    color_rt_keys: &[RtKey],
    color_rt_formats: &[vk::Format],
    vs_addr: u64,
    fs_addr: u64,
    fs_output_map: u32,
    fs_input_map: &[u8; 32],
    vs_cbuf_mask: u32,
    fs_cbuf_mask: u32,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    fs_tex_ids: &[u32],
    sampled_rt_slots: &[Option<RtKey>],
    fs_cbuf_reads: &[(u32, u32)],
) {
    if std::env::var_os("NEXIUM_GRADE_DISCOVER").is_none() {
        return;
    }
    let keys = if color_rt_keys.is_empty() {
        std::slice::from_ref(&fallback_key)
    } else {
        color_rt_keys
    };
    let tiny = keys
        .iter()
        .any(|key| key.height <= 64 || key.width <= 128 || key.width.saturating_mul(key.height) <= 0x4000);
    if !tiny {
        return;
    }
    let slot19_offset = 0x120usize;
    let slot19_len = 0x60usize;
    let (slot19_addr, slot19_size) = cbuf_bind_for_slot(cbuf_binds, 19);
    let slot19_bytes = if slot19_addr != 0 && slot19_size as usize > slot19_offset {
        let len = slot19_len.min(slot19_size as usize - slot19_offset);
        mappings
            .cpu_address_for(slot19_addr.wrapping_add(slot19_offset as u64))
            .and_then(|cpu| {
                let mut bytes = vec![0u8; len];
                mem_read(cpu, &mut bytes).then_some(bytes)
            })
    } else {
        None
    };
    let hash_bytes = |bytes: &[u8]| -> u64 {
        bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ (*b as u64)).wrapping_mul(0x100000001b3)
        })
    };
    let key_hash = keys.iter().fold(0xcbf29ce484222325u64, |h, key| {
        let h = (h ^ key.nvmap_id as u64).wrapping_mul(0x100000001b3);
        let h = (h ^ key.width as u64).wrapping_mul(0x100000001b3);
        let h = (h ^ key.height as u64).wrapping_mul(0x100000001b3);
        (h ^ key.gpu_va).wrapping_mul(0x100000001b3)
    });
    let slot19_hash = slot19_bytes.as_deref().map(hash_bytes).unwrap_or(0);
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u64, u32, u32, u64, u64)>>> = OnceLock::new();
    static HITS: AtomicU64 = AtomicU64::new(0);
    let seen_key = (
        vs_addr,
        fs_addr,
        draw.rt_control,
        fs_output_map,
        key_hash,
        slot19_hash,
    );
    let first = SEEN
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut seen| seen.insert(seen_key))
        .unwrap_or(true);
    if !first {
        return;
    }
    let hit = HITS.fetch_add(1, Ordering::Relaxed);
    let cap = std::env::var("NEXIUM_GRADE_DISCOVER_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(256);
    if hit >= cap {
        return;
    }
    let rts = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let format = color_rt_formats
                .get(index)
                .map(|format| format!("{:?}", format))
                .unwrap_or_else(|| "?".to_string());
            format!("{}:{} fmt={}", index, key.label(), format)
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let imap = fs_input_map
        .iter()
        .map(|v| format!("{:02x}", v))
        .collect::<Vec<_>>()
        .join("");
    let slot19 = match slot19_bytes.as_deref() {
        Some(bytes) => format!(
            "floats=[{}] raw={}",
            cbuf_watch_float_preview(bytes),
            cbuf_watch_hex_preview(bytes)
        ),
        None if slot19_addr == 0 || slot19_size == 0 => "unbound".to_string(),
        None => "unreadable".to_string(),
    };
    let reads = fs_cbuf_reads
        .iter()
        .filter_map(|(slot, offset)| (*slot == 19).then(|| format!("{:#x}", offset)))
        .collect::<Vec<_>>()
        .join(",");
    let fs_binds = (16..32u32)
        .filter_map(|slot| {
            let (addr, size) = cbuf_bind_for_slot(cbuf_binds, slot);
            (addr != 0 && size != 0).then(|| format!("{}={:#x}/{}", slot, addr, size))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let sampled = sampled_rt_slots
        .iter()
        .enumerate()
        .filter_map(|(slot, key)| key.map(|key| format!("s{}={}", slot, key.label())))
        .collect::<Vec<_>>()
        .join(",");
    let tics = tic_trace_summary(draw, fs_tex_ids, mappings, mem_read);
    log::warn!(
        "[grade-discover] #{} vs={:#x} fs={:#x} rtctl={:#x} omap={:#010x} imap={} masks={:#x}/{:#x} active19={} slot19={:#x}/{} {} rts=[{}] tex={:?} sampled=[{}] reads19=[{}] fs_binds=[{}] tics=[{}] topo={} v={}",
        hit,
        vs_addr,
        fs_addr,
        draw.rt_control,
        fs_output_map,
        imap,
        vs_cbuf_mask,
        fs_cbuf_mask,
        ((vs_cbuf_mask | fs_cbuf_mask) & (1u32 << 19)) != 0,
        slot19_addr,
        slot19_size,
        slot19,
        rts,
        fs_tex_ids,
        sampled,
        reads,
        fs_binds,
        tics,
        draw.topology,
        draw.vertex_count,
    );
}

fn trace_draw(
    draw: &DrawCall,
    layout: &VertexLayout,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    nvmap_id: u32,
    rt: &RenderTarget,
    color_rt_keys: &[RtKey],
    color_rt_locations: &[usize],
    vertex_addr: u64,
    vertex_count: u32,
    index_count: u32,
    indexed: bool,
    fs_tex_ids: &[u32],
    sampled_rt_slots: &[Option<RtKey>],
    color_rt_formats: &[vk::Format],
    depth_test: bool,
    depth_write: bool,
    blend_raw: (bool, bool, u32, u32, u32, u32, u32, u32),
    blend: &BlendState,
    fs_output_map: u32,
    color_masks: &[u32; 8],
    color_mask_common: bool,
    vs_cbuf_mask: u32,
    fs_cbuf_mask: u32,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    cbuf_data: Option<&[u8]>,
    fs_cbuf_reads: &[(u32, u32)],
) {
    let op_seq = next_gpu_op_seq();
    let cfg = draw_trace_config();
    if !cfg.enabled {
        return;
    }
    if !cfg.fs.is_empty() && !cfg.fs.contains(&draw.fs_shader_gpu_va) {
        return;
    }
    if !cfg.rt.is_empty() {
        if !cfg.rt.contains(&nvmap_id)
            && !color_rt_keys
                .iter()
                .any(|key| cfg.rt.contains(&key.nvmap_id))
        {
            return;
        }
    }
    let rt_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    if !cfg.rt_va.is_empty() {
        if !cfg.rt_va.contains(&rt_va)
            && !color_rt_keys
                .iter()
                .any(|key| cfg.rt_va.contains(&key.gpu_va))
        {
            return;
        }
    }
    if cfg.sampled_rt_only && sampled_rt_slots.iter().all(|slot| slot.is_none()) {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static DRAW_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = DRAW_SEQ.fetch_add(1, Ordering::Relaxed);
    if seq < cfg.start || seq > cfg.end {
        return;
    }
    let pos = position_bounds(layout, mappings, mem_read, vertex_addr, vertex_count);
    let attr = vertex_attr_sample(layout, mappings, mem_read, vertex_addr, vertex_count);
    let vp = guest_viewport_rect(
        draw,
        rt.width as f32,
        rt.height as f32,
        signed_viewport_nvmap(nvmap_id),
    );
    let clip = draw.surface_clip.effective(rt.width, rt.height);
    let cbuf_full = std::env::var_os("NEXIUM_DRAW_TRACE_CBUF_FULL").is_some();
    let cbuf =
        if cbuf_full || matches!(draw.fs_shader_gpu_va, 0x20330 | 0x40430) || fs_tex_ids.is_empty()
        {
            if cbuf_full {
                cbuf_sample(cbuf_data, vs_cbuf_mask | fs_cbuf_mask, &[], cbuf_binds)
            } else {
                cbuf_sample(
                    cbuf_data,
                    vs_cbuf_mask | fs_cbuf_mask,
                    fs_cbuf_reads,
                    cbuf_binds,
                )
            }
        } else {
            String::new()
        };
    let tics = tic_trace_summary(draw, fs_tex_ids, mappings, mem_read);
    let color_keys = color_rt_keys
        .iter()
        .map(|key| key.label())
        .collect::<Vec<_>>()
        .join(",");
    log::warn!(
        "[drawtrace] op={} #{} rt={} va={:#x} keys=[{}] {}x{} topo={} first={} v={} i={} indexed={} pos={} \
         vp_en={} vp={:?} scale=({:.3},{:.3},{:.3}) trans=({:.3},{:.3},{:.3}) clip=({},{} {}x{}) origin={:#x} ll={} fy={} sw={:#x}/{} \
         depth={}/{} clamp={} vclip={:#x}/{} func={:#x} zeta={} cull={} ff={:#x} \
         tex={:?} tics=[{}] sampled={:?} \
         blend={} per={} rgb=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         a=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         vb={:#x} attrs={} {} cbuf={:#x}/{} masks={:#x}/{:#x} {} fs={:#x}",
        op_seq,
        seq,
        nvmap_id,
        rt_va,
        color_keys,
        rt.width,
        rt.height,
        draw.topology,
        draw.first_vertex,
        vertex_count,
        index_count,
        indexed,
        pos.unwrap_or_else(|| "n/a".to_string()),
        draw.viewport_transform_en,
        vp,
        draw.viewport.scale_x,
        draw.viewport.scale_y,
        draw.viewport.scale_z,
        draw.viewport.translate_x,
        draw.viewport.translate_y,
        draw.viewport.translate_z,
        clip.x,
        clip.y,
        clip.width,
        clip.height,
        draw.window_origin.raw,
        draw.window_origin.lower_left(),
        draw.window_origin.flip_y(),
        draw.viewport.swizzle,
        draw.viewport.y_swizzle(),
        depth_test,
        depth_write,
        draw.viewport_clip_control.depth_clamp_enabled(),
        draw.viewport_clip_control.raw,
        draw.viewport_clip_control.geometry_clip(),
        draw.depth_func,
        draw.zeta_enable,
        draw.cull_test_enable,
        draw.front_face,
        fs_tex_ids,
        tics,
        sampled_rt_slots,
        blend_raw.0,
        blend_raw.1,
        blend_raw.2,
        blend_raw.3,
        blend_raw.4,
        blend.src_factor,
        blend.dst_factor,
        blend.op,
        blend_raw.5,
        blend_raw.6,
        blend_raw.7,
        blend.src_alpha_factor,
        blend.dst_alpha_factor,
        blend.alpha_op,
        vertex_addr,
        layout.attrs.len(),
        attr,
        draw.last_constbuf_addr,
        draw.last_constbuf_size,
        vs_cbuf_mask,
        fs_cbuf_mask,
        cbuf,
        draw.fs_shader_gpu_va,
    );
    let attachments = color_rt_keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let location = color_rt_locations.get(index).copied().unwrap_or(index);
            let slot = rt_control_target(draw.rt_control, location);
            let mask_index = if color_mask_common { 0 } else { location.min(7) };
            let shader_mask = fragment_output_mask(fs_output_map, location as u32);
            let maxwell_mask = color_masks[mask_index];
            let final_mask = blend.attachments[index.min(7)].color_write_mask.as_raw();
            let format = color_rt_formats
                .get(index)
                .map(|format| format!("{:?}", format))
                .unwrap_or_else(|| "?".to_string());
            format!(
                "{}:loc{} slot{} {} fmt={} omap={:x} cmask={:#x} final={:#x} blend={}",
                index,
                location,
                slot,
                key.label(),
                format,
                shader_mask,
                maxwell_mask,
                final_mask,
                blend.attachments[index.min(7)].enabled
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    log::warn!(
        "[drawtrace-att] op={} #{} fs={:#x} rtctl={:#x} common={} [{}]",
        op_seq,
        seq,
        draw.fs_shader_gpu_va,
        draw.rt_control,
        color_mask_common,
        attachments
    );
}

fn tic_trace_summary(
    draw: &DrawCall,
    fs_tex_ids: &[u32],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> String {
    if draw.tic_pool_gpu_va == 0 || fs_tex_ids.is_empty() {
        return String::new();
    }
    let mut out = Vec::new();
    for (slot, tex_id) in fs_tex_ids.iter().enumerate().take(8) {
        if *tex_id == u32::MAX || *tex_id > draw.tic_pool_limit {
            out.push(format!("s{}:tic{}=invalid", slot, tex_id));
            continue;
        }
        let tic_addr = draw.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
        let Some(cpu) = mappings.cpu_address_for(tic_addr) else {
            out.push(format!("s{}:tic{}=unmapped", slot, tex_id));
            continue;
        };
        let mut raw = [0u8; 32];
        if !mem_read(cpu, &mut raw) {
            out.push(format!("s{}:tic{}=unread", slot, tex_id));
            continue;
        }
        let Some(tic) = nexium_gpu::texture::TicEntry::parse(&raw) else {
            out.push(format!("s{}:tic{}=parse", slot, tex_id));
            continue;
        };
        out.push(format!(
            "s{}:tic{} {:?}/{:?} {}x{}x{} ty{} base{} norm{} bl{} va{:#x} nv{:?} swz{:?}",
            slot,
            tex_id,
            tic.format,
            tic.component_types,
            tic.width,
            tic.height,
            tic.depth,
            tic.texture_type,
            tic.base_layer,
            tic.normalized_coords,
            tic.is_block_linear,
            tic.gpu_va,
            mappings.nvmap_id_for(tic.gpu_va),
            tic.swizzle
        ));
    }
    out.join(" | ")
}

fn clear_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_CLEAR_TRACE").is_some())
}

fn trace_clear(
    draw: &DrawCall,
    op_seq: u64,
    nvmap_id: u32,
    rt: &RenderTarget,
    rt_gpu_va: u64,
    clear_scissor: Option<[i32; 4]>,
    color: [f32; 4],
    want_color_clear: bool,
    do_depth: bool,
) {
    if !clear_trace_enabled() {
        return;
    }
    let clip = draw.surface_clip.effective(rt.width, rt.height);
    log::warn!(
        "[cleartrace] op={} rt={} va={:#x} {}x{} mask={:#x} color={} depth={} clear_depth={:.6} \
         ctrl={:#x} scissor_en={} rect={:?} clip=({},{} {}x{}) origin={:#x} ll={} fy={} zeta={}",
        op_seq,
        nvmap_id,
        rt_gpu_va,
        rt.width,
        rt.height,
        draw.clear_mask,
        want_color_clear,
        do_depth,
        draw.clear_depth,
        draw.clear_control,
        draw.scissor.enabled,
        clear_scissor,
        clip.x,
        clip.y,
        clip.width,
        clip.height,
        draw.window_origin.raw,
        draw.window_origin.lower_left(),
        draw.window_origin.flip_y(),
        draw.zeta_enable,
    );
    log::warn!(
        "[cleartrace] op={} rgba=({:.3},{:.3},{:.3},{:.3}) scissor=({},{})->({},{})",
        op_seq,
        color[0],
        color[1],
        color[2],
        color[3],
        draw.scissor.min_x,
        draw.scissor.min_y,
        draw.scissor.max_x,
        draw.scissor.max_y,
    );
}

fn cbuf_sample(
    cbuf_data: Option<&[u8]>,
    used_mask: u32,
    cbuf_reads: &[(u32, u32)],
    cbuf_binds: &[[(u64, u32); 16]; 5],
) -> String {
    let Some(data) = cbuf_data else {
        return "cbuf_sample=none".to_string();
    };
    if !cbuf_reads.is_empty() {
        let max_reads = std::env::var("NEXIUM_DRAW_TRACE_CBUF_READS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12)
            .clamp(1, 64);
        let mut vals = Vec::new();
        for &(logical_slot, byte_offset) in cbuf_reads.iter().take(max_reads) {
            let off = logical_slot as usize * PACKED_CBUF_SLOT_SIZE + byte_offset as usize;
            let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot);
            if off + 4 > data.len() {
                vals.push(format!(
                    "s{}({:#x}/{})+{:#x}=out",
                    logical_slot, addr, size, byte_offset
                ));
                continue;
            }
            let b = [data[off], data[off + 1], data[off + 2], data[off + 3]];
            vals.push(format!(
                "s{}({:#x}/{})+{:#x}={:.3}/{:#010x}",
                logical_slot,
                addr,
                size,
                byte_offset,
                f32::from_le_bytes(b),
                u32::from_le_bytes(b)
            ));
        }
        return format!("cbuf_reads={}", vals.join(" "));
    }
    let mut slots = Vec::new();
    for logical_slot in 0..PACKED_CBUF_SLOTS {
        if (used_mask & (1u32 << logical_slot)) == 0 {
            continue;
        }
        let off = logical_slot * PACKED_CBUF_SLOT_SIZE;
        if off + 16 > data.len() {
            continue;
        }
        let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot as u32);
        let dump_bytes = if std::env::var_os("NEXIUM_DRAW_TRACE_CBUF_FULL").is_some() {
            let max = std::env::var("NEXIUM_DRAW_TRACE_CBUF_MAX")
                .ok()
                .and_then(|v| {
                    let v = v.trim();
                    usize::from_str_radix(v.trim_start_matches("0x"), 16)
                        .ok()
                        .or_else(|| v.parse().ok())
                })
                .unwrap_or(256);
            PACKED_CBUF_SLOT_SIZE.min(max)
        } else {
            16
        };
        if off + dump_bytes > data.len() {
            continue;
        }
        let mut vals = Vec::new();
        for (i, c) in data[off..off + dump_bytes].chunks_exact(4).enumerate() {
            vals.push(format!(
                "{:#x}:{:.3}",
                i * 4,
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            ));
        }
        slots.push(format!(
            "s{}({:#x}/{})=[{}]",
            logical_slot,
            addr,
            size,
            vals.join(",")
        ));
        if slots.len() >= 4 {
            break;
        }
    }
    if slots.is_empty() {
        "cbuf_sample=[]".to_string()
    } else {
        format!("cbuf_sample={}", slots.join(" "))
    }
}

#[derive(Clone, Copy)]
struct CbufWatchRegion {
    slot: usize,
    offset: usize,
    len: usize,
}

fn cbuf_watch_regions() -> Option<&'static [CbufWatchRegion]> {
    use std::sync::OnceLock;
    static REGIONS: OnceLock<Option<Vec<CbufWatchRegion>>> = OnceLock::new();
    REGIONS
        .get_or_init(|| {
            let spec = std::env::var("NEXIUM_CBUF_WATCH").ok()?;
            let spec = spec.trim();
            if spec.is_empty() || spec == "0" || spec.eq_ignore_ascii_case("false") {
                return None;
            }
            if spec == "1"
                || spec.eq_ignore_ascii_case("true")
                || spec.eq_ignore_ascii_case("grade")
            {
                return Some(vec![CbufWatchRegion {
                    slot: 19,
                    offset: 0x120,
                    len: 0x60,
                }]);
            }
            let regions = spec
                .split(',')
                .filter_map(|part| {
                    let mut fields = part.trim().split(':');
                    let slot = parse_env_u64(fields.next()?)? as usize;
                    let offset = parse_env_u64(fields.next().unwrap_or("0"))? as usize;
                    let len = fields
                        .next()
                        .and_then(parse_env_u64)
                        .unwrap_or(0x40)
                        .min(0x800) as usize;
                    (slot < PACKED_CBUF_SLOTS && len != 0).then_some(CbufWatchRegion {
                        slot,
                        offset,
                        len,
                    })
                })
                .collect::<Vec<_>>();
            (!regions.is_empty()).then_some(regions)
        })
        .as_deref()
}

fn trace_cbuf_watch(
    draw: &DrawCall,
    nvmap_id: u32,
    rt_key: RtKey,
    vs_addr: u64,
    fs_addr: u64,
    vs_cbuf_mask: u32,
    fs_cbuf_mask: u32,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    cbuf_data: Option<&[u8]>,
    fs_tex_ids: &[u32],
    sampled_rt_slots: &[Option<RtKey>],
    fs_cbuf_reads: &[(u32, u32)],
) {
    let Some(regions) = cbuf_watch_regions() else {
        return;
    };
    let Some(data) = cbuf_data else {
        return;
    };
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, usize, usize, usize, u64)>>> = OnceLock::new();
    static LAST: OnceLock<Mutex<HashMap<(usize, usize, usize, u64), Vec<u8>>>> = OnceLock::new();
    static HITS: AtomicU64 = AtomicU64::new(0);
    let cap = std::env::var("NEXIUM_CBUF_WATCH_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512);
    let fs_filter = parse_env_u64_list("NEXIUM_CBUF_WATCH_FS");
    if !fs_filter.is_empty() && !fs_filter.contains(&fs_addr) {
        return;
    }
    let bind_filter = parse_env_u64_list("NEXIUM_CBUF_WATCH_BIND");
    let include_inactive = std::env::var_os("NEXIUM_CBUF_WATCH_INACTIVE").is_some();
    let used_mask = vs_cbuf_mask | fs_cbuf_mask;
    for region in regions {
        let active = (used_mask & (1u32 << region.slot)) != 0;
        if !active && !include_inactive {
            continue;
        }
        let off = region.slot * PACKED_CBUF_SLOT_SIZE + region.offset;
        if off >= data.len() {
            continue;
        }
        let (addr, size) = cbuf_bind_for_slot(cbuf_binds, region.slot as u32);
        let bind_len = size as usize;
        if bind_len <= region.offset {
            continue;
        }
        let len = region
            .len
            .min(data.len() - off)
            .min(bind_len - region.offset);
        if len == 0 {
            continue;
        }
        let bytes = data[off..off + len].to_vec();
        if !bind_filter.is_empty() && !bind_filter.contains(&addr) {
            continue;
        }
        let seen_key = (fs_addr, region.slot, region.offset, len, addr);
        let value_key = (region.slot, region.offset, len, addr);
        let first = {
            let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
            seen.lock().map(|mut seen| seen.insert(seen_key)).unwrap_or(false)
        };
        let changed = {
            let last = LAST.get_or_init(|| Mutex::new(HashMap::new()));
            match last.lock() {
                Ok(mut last) => {
                    let changed = last.get(&value_key).map(|old| old != &bytes).unwrap_or(true);
                    if changed {
                        last.insert(value_key, bytes.clone());
                    }
                    changed
                }
                Err(_) => true,
            }
        };
        if !first && !changed {
            continue;
        }
        let hit = HITS.fetch_add(1, Ordering::Relaxed);
        if hit >= cap {
            return;
        }
        let read_hits = fs_cbuf_reads
            .iter()
            .filter_map(|(slot, byte_offset)| {
                let byte_offset = *byte_offset as usize;
                (*slot as usize == region.slot
                    && byte_offset >= region.offset
                    && byte_offset < region.offset + len)
                    .then(|| format!("{:#x}", byte_offset))
            })
            .collect::<Vec<_>>()
            .join(",");
        let sampled = sampled_rt_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, key)| key.map(|key| format!("s{}={}", slot, key.label())))
            .collect::<Vec<_>>()
            .join(",");
        log::warn!(
            "[cbuf-watch] #{} rt={} key={} vs={:#x} fs={:#x} slot={} bind={:#x}/{} off={:#x} len={:#x} active={} first={} changed={} reads=[{}] tex={:?} sampled=[{}] topo={} v={} floats=[{}] raw={}",
            hit,
            nvmap_id,
            rt_key.label(),
            vs_addr,
            fs_addr,
            region.slot,
            addr,
            size,
            region.offset,
            len,
            active,
            first,
            changed,
            read_hits,
            fs_tex_ids,
            sampled,
            draw.topology,
            draw.vertex_count,
            cbuf_watch_float_preview(&bytes),
            cbuf_watch_hex_preview(&bytes),
        );
    }
}

fn cbuf_watch_float_preview(bytes: &[u8]) -> String {
    bytes
        .chunks_exact(4)
        .take(32)
        .map(|c| format!("{:.3}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect::<Vec<_>>()
        .join(",")
}

fn cbuf_watch_hex_preview(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(64)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("")
}

fn cbuf_bind_for_slot(cbuf_binds: &[[(u64, u32); 16]; 5], logical_slot: u32) -> (u64, u32) {
    let stage = if logical_slot < 16 { 0 } else { 4 };
    let binding = (logical_slot & 15) as usize;
    cbuf_binds[stage][binding]
}

fn choose_texture_cb_index(
    cbuf_binds: &[(u64, u32); 16],
    bindless_slot: u32,
    tex_cb_slot: u32,
    shader_ids: &[u32],
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> usize {
    let primary = (bindless_slot as usize).min(15);
    let fallback = (tex_cb_slot as usize).min(15);
    let mut best = primary;
    let mut best_score = texture_cb_score(
        cbuf_binds,
        best,
        shader_ids,
        tic_pool_gpu_va,
        tic_pool_limit,
        via_header_index,
        mappings,
        mem_read,
    );
    for slot in [fallback, 15usize] {
        if slot == best {
            continue;
        }
        let score = texture_cb_score(
            cbuf_binds,
            slot,
            shader_ids,
            tic_pool_gpu_va,
            tic_pool_limit,
            via_header_index,
            mappings,
            mem_read,
        );
        if score > best_score {
            best = slot;
            best_score = score;
        }
    }
    best
}

fn texture_cb_score(
    cbuf_binds: &[(u64, u32); 16],
    slot: usize,
    shader_ids: &[u32],
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> u32 {
    let mut score = 0u32;
    for shader_id in shader_ids {
        let Some(handle) = read_texture_handle(cbuf_binds, slot, *shader_id, mappings, mem_read)
        else {
            continue;
        };
        if handle == 0 {
            continue;
        }
        let (tic, _) = split_texture_handle(handle, via_header_index);
        if texture_tic_readable(tic, tic_pool_gpu_va, tic_pool_limit, mappings, mem_read) {
            score = score.saturating_add(1);
        }
    }
    score
}

fn read_texture_handle(
    cbuf_binds: &[(u64, u32); 16],
    slot: usize,
    shader_id: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<u32> {
    let (addr, size) = cbuf_binds[slot.min(15)];
    let off = (shader_id as u64).saturating_mul(4);
    if addr == 0 || off + 4 > size as u64 {
        return None;
    }
    let cpu = mappings.cpu_address_for(addr.wrapping_add(off))?;
    let mut bytes = [0u8; 4];
    if mem_read(cpu, &mut bytes) {
        Some(u32::from_le_bytes(bytes))
    } else {
        None
    }
}

fn split_texture_handle(raw: u32, via_header_index: bool) -> (u32, u32) {
    if via_header_index {
        (raw, raw)
    } else {
        (raw & 0x000F_FFFF, raw >> 20)
    }
}

fn texture_tic_readable(
    tic: u32,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> bool {
    if tic_pool_gpu_va == 0 || tic > tic_pool_limit {
        return false;
    }
    let tic_addr = tic_pool_gpu_va.wrapping_add((tic as u64).saturating_mul(32));
    let Some(cpu) = mappings.cpu_address_for(tic_addr) else {
        return false;
    };
    let mut raw = [0u8; 32];
    mem_read(cpu, &mut raw) && nexium_gpu::texture::TicEntry::parse(&raw).is_some()
}

fn shader_cfg_dump(cfg: &nexium_shader::Cfg) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "blocks={} unimplemented={}\n",
        cfg.blocks.len(),
        cfg.unimplemented
    ));
    for block in &cfg.blocks {
        out.push_str(&format!(
            "\nblock {} {:#x}..{:#x} {:?}\n",
            block.id, block.start_offset, block.end_offset, block.branch
        ));
        out.push_str(&block.program.to_string());
    }
    out
}

fn shader_cfg_code_len(cfg: &nexium_shader::Cfg) -> usize {
    cfg.blocks.iter().map(|b| b.end_offset).max().unwrap_or(0)
}

fn position_bounds(
    layout: &VertexLayout,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    vertex_addr: u64,
    vertex_count: u32,
) -> Option<String> {
    let attr = layout.attrs.iter().find(|a| a.location == 0)?;
    let stride = layout
        .bindings
        .iter()
        .find(|b| b.binding == attr.binding)
        .map(|b| b.stride)?;
    if stride == 0 {
        return None;
    }
    let base_cpu = mappings.cpu_address_for(vertex_addr)?;
    let n = vertex_count.clamp(1, 64) as u64;
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut seen = 0u32;
    for vi in 0..n {
        let addr = base_cpu + attr.offset as u64 + vi * stride as u64;
        if let Some((x, y)) = read_position_xy(attr.format, addr, mem_read) {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            seen += 1;
        }
    }
    if seen == 0 {
        None
    } else {
        Some(format!(
            "[{:.2},{:.2}]..[{:.2},{:.2}] n={}",
            min_x, min_y, max_x, max_y, seen
        ))
    }
}

fn read_position_xy(
    format: vk::Format,
    addr: u64,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<(f32, f32)> {
    match format {
        vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32B32_SFLOAT
        | vk::Format::R32G32B32A32_SFLOAT => {
            let mut buf = [0u8; 8];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            let x = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            let y = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            if x.is_finite() && y.is_finite() {
                Some((x, y))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn vertex_attr_sample(
    layout: &VertexLayout,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    vertex_addr: u64,
    vertex_count: u32,
) -> String {
    let Some(base_cpu) = mappings.cpu_address_for(vertex_addr) else {
        return "attr_sample=unmapped".to_string();
    };
    let mut parts = Vec::new();
    let attr_limit = std::env::var("NEXIUM_DRAW_TRACE_ATTR_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(3);
    for attr in layout.attrs.iter().filter(|a| a.location <= attr_limit) {
        let Some(stride) = layout
            .bindings
            .iter()
            .find(|b| b.binding == attr.binding)
            .map(|b| b.stride)
        else {
            continue;
        };
        if stride == 0 {
            parts.push(format!("l{} const {:?}", attr.location, attr.format));
            continue;
        }
        let mut vals = Vec::new();
        for vi in 0..vertex_count.min(4) {
            let addr = base_cpu + attr.offset as u64 + vi as u64 * stride as u64;
            if let Some(v) = read_attr_vec4(attr.format, addr, mem_read) {
                vals.push(format!("({:.3},{:.3},{:.3},{:.3})", v[0], v[1], v[2], v[3]));
            }
        }
        parts.push(format!(
            "l{} b{}+{} {:?} [{}]",
            attr.location,
            attr.binding,
            attr.offset,
            attr.format,
            vals.join(",")
        ));
    }
    if parts.is_empty() {
        "attr_sample=[]".to_string()
    } else {
        format!("attr_sample={}", parts.join(" "))
    }
}

fn read_attr_vec4(
    format: vk::Format,
    addr: u64,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<[f32; 4]> {
    match format {
        vk::Format::R32G32B32A32_SFLOAT => {
            let mut buf = [0u8; 16];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                f32::from_le_bytes(buf[0..4].try_into().ok()?),
                f32::from_le_bytes(buf[4..8].try_into().ok()?),
                f32::from_le_bytes(buf[8..12].try_into().ok()?),
                f32::from_le_bytes(buf[12..16].try_into().ok()?),
            ])
        }
        vk::Format::R32G32B32_SFLOAT => {
            let mut buf = [0u8; 12];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                f32::from_le_bytes(buf[0..4].try_into().ok()?),
                f32::from_le_bytes(buf[4..8].try_into().ok()?),
                f32::from_le_bytes(buf[8..12].try_into().ok()?),
                1.0,
            ])
        }
        vk::Format::R32G32_SFLOAT => {
            let mut buf = [0u8; 8];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                f32::from_le_bytes(buf[0..4].try_into().ok()?),
                f32::from_le_bytes(buf[4..8].try_into().ok()?),
                0.0,
                1.0,
            ])
        }
        vk::Format::R8G8B8A8_UNORM => {
            let mut buf = [0u8; 4];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                buf[0] as f32 / 255.0,
                buf[1] as f32 / 255.0,
                buf[2] as f32 / 255.0,
                buf[3] as f32 / 255.0,
            ])
        }
        vk::Format::R8G8B8_UNORM => {
            let mut buf = [0u8; 3];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                buf[0] as f32 / 255.0,
                buf[1] as f32 / 255.0,
                buf[2] as f32 / 255.0,
                1.0,
            ])
        }
        vk::Format::R16G16B16A16_UNORM => {
            let mut buf = [0u8; 8];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                u16::from_le_bytes(buf[0..2].try_into().ok()?) as f32 / 65535.0,
                u16::from_le_bytes(buf[2..4].try_into().ok()?) as f32 / 65535.0,
                u16::from_le_bytes(buf[4..6].try_into().ok()?) as f32 / 65535.0,
                u16::from_le_bytes(buf[6..8].try_into().ok()?) as f32 / 65535.0,
            ])
        }
        vk::Format::R16G16_UNORM => {
            let mut buf = [0u8; 4];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([
                u16::from_le_bytes(buf[0..2].try_into().ok()?) as f32 / 65535.0,
                u16::from_le_bytes(buf[2..4].try_into().ok()?) as f32 / 65535.0,
                0.0,
                1.0,
            ])
        }
        vk::Format::R8G8_UNORM => {
            let mut buf = [0u8; 2];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([buf[0] as f32 / 255.0, buf[1] as f32 / 255.0, 0.0, 1.0])
        }
        vk::Format::R8_UNORM => {
            let mut buf = [0u8; 1];
            if !mem_read(addr, &mut buf) {
                return None;
            }
            Some([buf[0] as f32 / 255.0, 0.0, 0.0, 1.0])
        }
        _ => None,
    }
}

fn fetch_sass(
    gpu_va: u64,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<Vec<u8>> {
    let cpu = mappings.cpu_address_for(gpu_va)?;
    let mut buf = vec![0u8; MAX_SASS_BYTES];
    if !mem_read(cpu + SPH_SIZE as u64, &mut buf) {
        return None;
    }
    Some(buf)
}

fn fetch_sph(
    gpu_va: u64,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<[u8; SPH_SIZE]> {
    let cpu = mappings.cpu_address_for(gpu_va)?;
    let mut buf = [0u8; SPH_SIZE];
    if !mem_read(cpu, &mut buf) {
        return None;
    }
    Some(buf)
}

fn ps_generic_input_map(sph: [u8; SPH_SIZE]) -> [u8; 32] {
    let mut map = [0u8; 32];
    map.copy_from_slice(&sph[0x18..0x38]);
    map
}

fn ps_output_map(sph: [u8; SPH_SIZE]) -> u32 {
    u32::from_le_bytes([sph[0x48], sph[0x49], sph[0x4A], sph[0x4B]])
}

fn build_vertex_layout(draw: &DrawCall) -> Result<VertexLayout, String> {
    let mut bindings: Vec<VertexBinding> = Vec::new();
    let mut attrs: Vec<VertexAttr> = Vec::new();
    let mut seen_bindings: std::collections::HashSet<u32> = std::collections::HashSet::new();

    if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        let nenabled = draw.vertex_attribs.iter().filter(|a| a.format != 0).count();
        if k % 1500 == 0 && nenabled >= 5 {
            for (loc, a) in draw.vertex_attribs.iter().enumerate() {
                if a.format != 0 {
                    log::warn!(
                        "[vattr] draw{} loc={} buf={} off={} fmt={:#x} constant={}",
                        k,
                        loc,
                        a.buffer,
                        a.offset,
                        a.format,
                        a.constant
                    );
                }
            }
        }
    }

    let skip_const = std::env::var_os("NEXIUM_SKIP_CONST_ATTR").is_some();
    const WHITE_BINDING: u32 = 15;
    let mut need_white = false;
    for (loc, attrib) in draw.vertex_attribs.iter().enumerate() {
        if attrib.format == 0 {
            continue;
        }
        if attrib.constant {
            if skip_const {
                continue;
            }
            need_white = true;
            attrs.push(VertexAttr {
                location: loc as u32,
                binding: WHITE_BINDING,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            });
            continue;
        }
        let format = map_attrib_format(attrib.format)
            .ok_or_else(|| format!("attrib {}: unsupported format {:#x}", loc, attrib.format))?;
        let binding = attrib.buffer;
        if !seen_bindings.contains(&binding) {
            let mut stride = draw
                .vertex_buffers
                .get(binding as usize)
                .map(|vb| vb.stride)
                .unwrap_or(0);
            if stride == 0 {
                let mut packed = 0u32;
                for a in draw.vertex_attribs.iter() {
                    if a.format != 0 && !a.constant && a.buffer == binding {
                        packed = packed.max(a.offset + attrib_format_bytes(a.format));
                    }
                }
                stride = packed;
            }
            if stride == 0 {
                return Err(format!(
                    "attrib {}: binding {} has zero stride",
                    loc, binding
                ));
            }
            bindings.push(VertexBinding { binding, stride });
            seen_bindings.insert(binding);
        }
        attrs.push(VertexAttr {
            location: loc as u32,
            binding,
            format,
            offset: attrib.offset,
        });
    }

    if attrs.is_empty() {
        return Err("no enabled vertex attributes".to_string());
    }
    if need_white {
        bindings.push(VertexBinding {
            binding: WHITE_BINDING,
            stride: 0,
        });
    }
    Ok(VertexLayout { bindings, attrs })
}

fn map_attrib_format(format: u32) -> Option<vk::Format> {
    let size = format & 0x3F;
    let type_ = format >> 6;
    match (size, type_) {
        (0x01, 7) => Some(vk::Format::R32G32B32A32_SFLOAT),
        (0x01, 3) => Some(vk::Format::R32G32B32A32_SINT),
        (0x01, 4) => Some(vk::Format::R32G32B32A32_UINT),

        (0x02, 7) => Some(vk::Format::R32G32B32_SFLOAT),
        (0x02, 3) => Some(vk::Format::R32G32B32_SINT),
        (0x02, 4) => Some(vk::Format::R32G32B32_UINT),

        (0x03, 7) => Some(vk::Format::R16G16B16A16_SFLOAT),
        (0x03, 2) => Some(vk::Format::R16G16B16A16_UNORM),
        (0x03, 1) => Some(vk::Format::R16G16B16A16_SNORM),
        (0x03, 3) => Some(vk::Format::R16G16B16A16_SINT),
        (0x03, 4) => Some(vk::Format::R16G16B16A16_UINT),
        (0x03, 6) => Some(vk::Format::R16G16B16A16_SSCALED),
        (0x03, 5) => Some(vk::Format::R16G16B16A16_USCALED),

        (0x04, 7) => Some(vk::Format::R32G32_SFLOAT),
        (0x04, 3) => Some(vk::Format::R32G32_SINT),
        (0x04, 4) => Some(vk::Format::R32G32_UINT),

        (0x05, 3) => Some(vk::Format::R16G16B16_SINT),
        (0x05, 4) => Some(vk::Format::R16G16B16_UINT),

        (0x0A, 2) => Some(vk::Format::R8G8B8A8_UNORM),
        (0x0A, 1) => Some(vk::Format::R8G8B8A8_SNORM),
        (0x0A, 3) => Some(vk::Format::R8G8B8A8_SINT),
        (0x0A, 4) => Some(vk::Format::R8G8B8A8_UINT),
        (0x0A, 6) => Some(vk::Format::R8G8B8A8_SSCALED),
        (0x0A, 5) | (0x0A, 7) => Some(vk::Format::R8G8B8A8_USCALED),

        (0x0F, 7) => Some(vk::Format::R16G16_SFLOAT),
        (0x0F, 2) => Some(vk::Format::R16G16_UNORM),
        (0x0F, 1) => Some(vk::Format::R16G16_SNORM),
        (0x0F, 3) => Some(vk::Format::R16G16_SINT),
        (0x0F, 4) => Some(vk::Format::R16G16_UINT),
        (0x0F, 6) => Some(vk::Format::R16G16_SSCALED),
        (0x0F, 5) => Some(vk::Format::R16G16_USCALED),

        (0x12, 7) => Some(vk::Format::R32_SFLOAT),
        (0x12, 3) => Some(vk::Format::R32_SINT),
        (0x12, 4) => Some(vk::Format::R32_UINT),

        (0x13, 2) => Some(vk::Format::R8G8B8_UNORM),
        (0x13, 1) => Some(vk::Format::R8G8B8_SNORM),
        (0x13, 3) => Some(vk::Format::R8G8B8_SINT),
        (0x13, 4) => Some(vk::Format::R8G8B8_UINT),
        (0x13, 6) => Some(vk::Format::R8G8B8_SSCALED),
        (0x13, 5) => Some(vk::Format::R8G8B8_USCALED),

        (0x18, 2) => Some(vk::Format::R8G8_UNORM),
        (0x18, 1) => Some(vk::Format::R8G8_SNORM),
        (0x18, 3) => Some(vk::Format::R8G8_SINT),
        (0x18, 4) => Some(vk::Format::R8G8_UINT),
        (0x18, 6) => Some(vk::Format::R8G8_SSCALED),
        (0x18, 5) => Some(vk::Format::R8G8_USCALED),

        (0x1B, 7) => Some(vk::Format::R16_SFLOAT),
        (0x1B, 2) => Some(vk::Format::R16_UNORM),
        (0x1B, 1) => Some(vk::Format::R16_SNORM),
        (0x1B, 3) => Some(vk::Format::R16_SINT),
        (0x1B, 4) => Some(vk::Format::R16_UINT),
        (0x1B, 6) => Some(vk::Format::R16_SSCALED),
        (0x1B, 5) => Some(vk::Format::R16_USCALED),

        (0x1D, 2) => Some(vk::Format::R8_UNORM),
        (0x1D, 1) => Some(vk::Format::R8_SNORM),
        (0x1D, 3) => Some(vk::Format::R8_SINT),
        (0x1D, 4) => Some(vk::Format::R8_UINT),
        (0x1D, 6) => Some(vk::Format::R8_SSCALED),
        (0x1D, 5) => Some(vk::Format::R8_USCALED),

        (0x30, 2) => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        (0x30, 1) => Some(vk::Format::A2B10G10R10_SNORM_PACK32),
        (0x30, 3) => Some(vk::Format::A2B10G10R10_SINT_PACK32),
        (0x30, 4) => Some(vk::Format::A2B10G10R10_UINT_PACK32),
        (0x30, 6) => Some(vk::Format::A2B10G10R10_SSCALED_PACK32),
        (0x30, 5) => Some(vk::Format::A2B10G10R10_USCALED_PACK32),

        (0x31, 7) => Some(vk::Format::B10G11R11_UFLOAT_PACK32),

        _ => None,
    }
}

fn attrib_format_bytes(format: u32) -> u32 {
    match format & 0x3F {
        0x01 => 16,
        0x02 => 12,
        0x03 => 8,
        0x04 => 8,
        0x05 => 6,
        0x0A => 4,
        0x0F => 4,
        0x12 => 4,
        0x13 => 3,
        0x18 => 2,
        0x1B => 2,
        0x1D => 1,
        0x30 => 4,
        0x31 => 4,
        _ => 0,
    }
}

fn map_topology(t: u32) -> Option<vk::PrimitiveTopology> {
    match t {
        0 => Some(vk::PrimitiveTopology::POINT_LIST),
        1 => Some(vk::PrimitiveTopology::LINE_LIST),
        3 => Some(vk::PrimitiveTopology::LINE_STRIP),
        4 => Some(vk::PrimitiveTopology::TRIANGLE_LIST),
        5 => Some(vk::PrimitiveTopology::TRIANGLE_STRIP),
        6 => Some(vk::PrimitiveTopology::TRIANGLE_FAN),
        7 => Some(vk::PrimitiveTopology::TRIANGLE_LIST),
        _ => None,
    }
}

fn expand_quad_indices(indices: &[u32]) -> Vec<u32> {
    let quads = indices.len() / 4;
    let mut out = Vec::with_capacity(quads * 6);
    for q in 0..quads {
        let base = q * 4;
        out.extend_from_slice(&[
            indices[base],
            indices[base + 1],
            indices[base + 2],
            indices[base],
            indices[base + 2],
            indices[base + 3],
        ]);
    }
    out
}

fn pack_cbuf_data(
    cbuf_binds: &[[(u64, u32); 16]; 5],
    vs_mask: u32,
    fs_mask: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<Vec<u8>> {
    let used = vs_mask | fs_mask;
    if used == 0 {
        return None;
    }
    let mut out = vec![0u8; PACKED_CBUF_SLOTS * PACKED_CBUF_SLOT_SIZE];
    let mut any = false;
    for logical_slot in 0..PACKED_CBUF_SLOTS {
        if (used & (1u32 << logical_slot)) == 0 {
            continue;
        }
        let stage = if logical_slot < 16 { 0 } else { 4 };
        let binding = logical_slot & 15;
        let (addr, size) = cbuf_binds[stage][binding];
        if addr == 0 || size == 0 {
            continue;
        }
        let Some(cpu) = mappings.cpu_address_for(addr) else {
            continue;
        };
        let len = (size as usize).min(PACKED_CBUF_SLOT_SIZE);
        let off = logical_slot * PACKED_CBUF_SLOT_SIZE;
        if mem_read(cpu, &mut out[off..off + len]) {
            any = true;
        }
    }
    if any {
        Some(out)
    } else {
        None
    }
}

fn resolve_cbuf(draw: &DrawCall, cbuf_binds: &[[(u64, u32); 16]; 5]) -> (u64, u32) {
    if draw.last_constbuf_addr != 0 && draw.last_constbuf_size > 0 {
        return (draw.last_constbuf_addr, draw.last_constbuf_size);
    }
    for stage in cbuf_binds.iter() {
        for &(addr, size) in stage.iter() {
            if addr != 0 && size > 0 {
                return (addr, size);
            }
        }
    }
    (0, 0)
}

fn resolve_vs_cbuf(cbuf_binds: &[[(u64, u32); 16]; 5]) -> (u64, u32) {
    for &(addr, size) in cbuf_binds[0].iter().rev() {
        if addr != 0 && size > 0 {
            return (addr, size);
        }
    }
    (0, 0)
}

fn vertex_buffer_bindings(
    vertex_buffers: &[VertexBuffer; 32],
    layout: &VertexLayout,
) -> Vec<VertexBufferBinding> {
    layout
        .bindings
        .iter()
        .filter(|b| b.stride > 0)
        .filter_map(|b| {
            let vb = vertex_buffers.get(b.binding as usize)?;
            let va = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
            let end = ((vb.end_hi as u64) << 32) | vb.end_lo as u64;
            let size = if end == 0 {
                0
            } else {
                end.saturating_add(1).saturating_sub(va)
            };
            (va != 0).then_some(VertexBufferBinding {
                binding: b.binding,
                addr: va,
                stride: b.stride,
                size,
            })
        })
        .collect()
}
