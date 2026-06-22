use ash::vk;
use std::sync::Arc;

use super::engines::maxwell3d::{DrawCall, RenderTarget, VertexBuffer};
use super::engines::Maxwell3D;
use super::GpuMappings;

use nexium_gpu::draw::{
    BlendState, DepthState, DrawState, Maxwell3dDrawCall, VertexAttr, VertexBinding, VertexLayout,
};
use nexium_gpu::rt_cache::RtKey;

const SPH_SIZE: usize = 0x50;
const MAX_SASS_BYTES: usize = 16 * 1024;
const PACKED_CBUF_SLOTS: usize = 32;
const PACKED_CBUF_SLOT_SIZE: usize = 2048;

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
        nexium_gpu::texture::TicFormat::A8B8G8R8 | nexium_gpu::texture::TicFormat::R8G8B8A8
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
                if batch.last().is_some_and(|last| last.rt_key != call.rt_key) {
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

pub fn flush_accum(
    batch: &mut Vec<Maxwell3dDrawCall>,
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    if batch.is_empty() {
        return;
    }
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
    flush_batch(batch, renderer, rt_thread, &read_guest);
}

fn rt_keys_uniform(calls: &[Maxwell3dDrawCall]) -> bool {
    match calls.first() {
        Some(f) => calls.iter().all(|c| c.rt_key == f.rt_key),
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
    let rt = &draw.rt[0];
    if rt.width == 0 || rt.height == 0 {
        return Err("RT[0] has zero extent".to_string());
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
        .readback_target(job.nvmap_id, job.width, job.height)
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
    renderer.upload_target_rgba(job.nvmap_id, job.width, job.height, &dst)
}

fn submit_draw_batch_async(
    batch: &[Maxwell3dDrawCall],
    renderer: &Arc<nexium_gpu::Renderer>,
    rt: &crate::render_thread::RenderThread,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) {
    let mut snapshot: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    for call in batch {
        let stride = call
            .vertex_layout
            .bindings
            .iter()
            .find(|b| b.stride > 0)
            .map(|b| b.stride as u64)
            .unwrap_or(0);
        let vbytes = stride.saturating_mul(call.vertex_count as u64) as usize;
        if vbytes > 0 {
            if let Some(d) = read_guest(call.vertex_addr, vbytes) {
                snapshot.insert(call.vertex_addr, d);
            }
        }
        if call.cbuf_size > 0 && call.cbuf_addr != 0 {
            if let Some(d) = read_guest(call.cbuf_addr, call.cbuf_size as usize) {
                snapshot.insert(call.cbuf_addr, d);
            }
        }
        if !call.fs_tex_ids.is_empty() && call.tic_pool_gpu_va != 0 {
            for &tex_id in &call.fs_tex_ids {
                if tex_id > call.tic_pool_limit {
                    continue;
                }
                let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
                if let Some(tic_raw) = read_guest(tic_addr, 32) {
                    if let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) {
                        let pitch = tic.format.linear_size(tic.width, tic.height);
                        let read_size = if tic.is_block_linear {
                            tic.format
                                .block_linear_size(tic.width, tic.height, tic.block_height_log2)
                                .max(pitch)
                        } else {
                            pitch
                        };
                        if let Some(d) = read_guest(tic.gpu_va, read_size) {
                            snapshot.insert(tic.gpu_va, d);
                        }
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
                if let Some(tsc_raw) = read_guest(tsc_addr, 32) {
                    snapshot.insert(tsc_addr, tsc_raw);
                }
            }
        }
    }
    let calls = batch.to_vec();
    let r = renderer.clone();
    rt.submit(Box::new(move || {
        let _ = r.execute_draws(&calls, move |addr: u64, len: usize| {
            snapshot
                .get(&addr)
                .filter(|b| b.len() >= len)
                .map(|b| b[..len].to_vec())
        });
    }));
}

struct ShaderBundle {
    vs_spirv: std::sync::Arc<Vec<u32>>,
    vs_cbuf_mask: u32,
    vs_hash: u64,
    fs_spirv: std::sync::Arc<Vec<u32>>,
    fs_cbuf_mask: u32,
    fs_hash: u64,
    fs_tex_ids: Vec<u32>,
    fs_cbuf_reads: Vec<(u32, u32)>,
    cbuf_used: u32,
}

#[allow(clippy::type_complexity)]
fn shader_bundle_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<(u64, u64, u32, u32, u32, u32), std::sync::Arc<ShaderBundle>>,
> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<(u64, u64, u32, u32, u32, u32), std::sync::Arc<ShaderBundle>>,
        >,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn depth_disabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("NEXIUM_NO_DEPTH").ok().as_deref() == Some("1"))
}

fn execute_one(
    draw: &DrawCall,
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    renderer: &Arc<nexium_gpu::Renderer>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<Option<Maxwell3dDrawCall>, String> {
    let rt = &draw.rt[0];
    if rt.width == 0 || rt.height == 0 {
        return Err("RT[0] has zero extent".to_string());
    }
    let rt_gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings
        .nvmap_id_for(rt_gpu_va)
        .ok_or_else(|| format!("RT gpu_va={:#x} not mapped", rt_gpu_va))?;
    let rt_key = RtKey {
        nvmap_id,
        width: rt.width,
        height: rt.height,
    };
    let no_depth = depth_disabled();

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
        trace_clear(draw, op_seq, nvmap_id, rt, clear_scissor, color, want_color_clear, do_depth);
        if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
            let r = renderer.clone();
            let cdepth = draw.clear_depth;
            let (w, h) = (rt.width, rt.height);
            rt_thread.submit(Box::new(move || {
                if want_color_clear {
                    if let Some(rect) = clear_scissor {
                        let _ = r.clear_target_rect(nvmap_id, w, h, color, rect);
                    } else {
                        let _ = r.clear_target(nvmap_id, w, h, color);
                    }
                }
                if do_depth {
                    let _ = r.clear_depth(nvmap_id, w, h, cdepth);
                }
            }));
        } else {
            if want_color_clear {
                if let Some(rect) = clear_scissor {
                    renderer.clear_target_rect(nvmap_id, rt.width, rt.height, color, rect)?;
                } else {
                    renderer.clear_target(nvmap_id, rt.width, rt.height, color)?;
                }
            }
            if do_depth {
                renderer.clear_depth(nvmap_id, rt.width, rt.height, draw.clear_depth)?;
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

    let (vptx_scale_z, vptx_translate_z) = if !no_depth && draw.viewport.scale_z != 0.0 {
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
    let shader_key = (
        vs_addr,
        fs_addr,
        vptx_scale_z.to_bits(),
        vptx_translate_z.to_bits(),
        ps_key,
        win_key,
    );
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

            let vs_cfg = nexium_shader::build_cfg(&vs_sass);
            let fs_cfg = nexium_shader::build_cfg(&fs_sass);
            let fs_cbuf_reads = collect_cbuf_reads(&fs_cfg, 16);
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
            } else {
                log::debug!(
                    "shader translated: vs_addr={:#x} fs_addr={:#x} all ops covered",
                    vs_addr,
                    fs_addr,
                );
            }

            let (fs_spirv, fs_cbuf_mask, mut fs_tex_ids, fs_cbuf_used) =
                nexium_spirv::emit_fragment_full(&fs_cfg);

            {
                let walked = nexium_shader::extract_fs_tex_ids(&fs_sass, 15);
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
                    fs_sass.len(),
                );
            }

            let required_outputs = nexium_spirv::scan_input_locations(&fs_spirv);
            let (vs_spirv, vs_cbuf_mask, vs_cbuf_used) =
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
                        ..Default::default()
                    },
                );

            let vs_hash = nexium_gpu::renderer::hash_spirv(&vs_spirv);
            let fs_hash = nexium_gpu::renderer::hash_spirv(&fs_spirv);
            let b = std::sync::Arc::new(ShaderBundle {
                vs_spirv: std::sync::Arc::new(vs_spirv),
                vs_cbuf_mask,
                vs_hash,
                fs_spirv: std::sync::Arc::new(fs_spirv),
                fs_cbuf_mask,
                fs_hash,
                fs_tex_ids,
                fs_cbuf_reads,
                cbuf_used: vs_cbuf_used.max(fs_cbuf_used),
            });
            guard.insert(shader_key, b.clone());
            if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static DN: AtomicU32 = AtomicU32::new(0);
                let dk = DN.fetch_add(1, Ordering::Relaxed);
                if dk < 24 {
                    let vb: Vec<u8> = b.vs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let fb: Vec<u8> = b.fs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let _ =
                        std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_vs.spv", dk), &vb);
                    let _ =
                        std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_fs.spv", dk), &fb);
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_vs.sass", dk),
                        &vs_sass,
                    );
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_fs.sass", dk),
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
                        format!("C:/Users/Mythrax/Desktop/sh_{}_vs.txt", dk),
                        vs_dis,
                    );
                    let _ = std::fs::write(
                        format!("C:/Users/Mythrax/Desktop/sh_{}_fs.txt", dk),
                        fs_dis,
                    );
                    log::warn!(
                        "[shdump] #{} vs_addr={:#x} fs_addr={:#x} vs_mask={:#x} fs_mask={:#x} vs_bytes={} fs_bytes={} ntex={}",
                        dk, vs_addr, fs_addr, b.vs_cbuf_mask, b.fs_cbuf_mask, b.vs_spirv.len(), b.fs_spirv.len(), b.fs_tex_ids.len()
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
                k, vs_cbuf_mask, fs_cbuf_mask, cbuf_addr, cbuf_size,
                draw.last_constbuf_addr, draw.last_constbuf_size, stage(0), stage(4)
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

    let mut fs_sampler_ids: Vec<u32> = vec![0u32; fs_tex_ids.len()];
    if !fs_tex_ids.is_empty() {
        let tex_cb_index = maxwell.regs.tex_cb_index as usize;
        let (tcb_addr, tcb_size) = maxwell.regs.cbuf_binds[4][tex_cb_index.min(15)];
        if tcb_addr != 0 {
            for (i, unit_slot) in fs_tex_ids.iter_mut().enumerate() {
                let off = (*unit_slot as u64).saturating_mul(4);
                if off + 4 > tcb_size as u64 {
                    continue;
                }
                let Some(cpu) = mappings.cpu_address_for(tcb_addr.wrapping_add(off)) else {
                    continue;
                };
                let mut bytes = [0u8; 4];
                if mem_read(cpu, &mut bytes) {
                    let handle = u32::from_le_bytes(bytes);
                    let tic = handle & 0x000F_FFFF;
                    let tsc = handle >> 20;
                    if handle != 0 && tic <= draw.tic_pool_limit {
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
                    }
                }
            }
        }
    }

    let mut sampled_rt_fuzzy = false;
    let mut sampled_rt_keys: Vec<RtKey> = Vec::new();
    let mut sampled_rt_slots: Vec<Option<RtKey>> = vec![None; fs_tex_ids.len()];
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
            if !tic_can_alias_render_target(tic.format) {
                continue;
            }
            let Some(nv) = mappings.nvmap_id_for(tic.gpu_va) else {
                continue;
            };
            let key = RtKey {
                nvmap_id: nv,
                width: tic.width,
                height: tic.height,
            };
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

    let vertex_addr = first_vertex_buffer_address(&draw.vertex_buffers, &layout)
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
                        k, layout.attrs.len(), a.location, a.format, a.offset, mx[0], mx[1], mx[2], mx[3]
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
        Some(rt_key)
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

    let (
        blend_raw_src,
        blend_raw_dst,
        blend_raw_eq,
        blend_raw_src_alpha,
        blend_raw_dst_alpha,
        blend_raw_eq_alpha,
    ) = if maxwell.regs.blend_per_target_enabled {
        (
            maxwell.regs.blend_pt_src_rgb[0],
            maxwell.regs.blend_pt_dst_rgb[0],
            maxwell.regs.blend_pt_eq_rgb[0],
            maxwell.regs.blend_pt_src_alpha[0],
            maxwell.regs.blend_pt_dst_alpha[0],
            maxwell.regs.blend_pt_eq_alpha[0],
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
    let blend_state = BlendState {
        enabled: maxwell.regs.blend_enable[0] && std::env::var("NEXIUM_NO_BLEND").is_err(),
        src_factor: map_blend_factor(blend_raw_src),
        dst_factor: map_blend_factor(blend_raw_dst),
        op: map_blend_op(blend_raw_eq),
        src_alpha_factor: map_blend_factor(blend_raw_src_alpha),
        dst_alpha_factor: map_blend_factor(blend_raw_dst_alpha),
        alpha_op: map_blend_op(blend_raw_eq_alpha),
    };

    trace_draw(
        draw,
        &layout,
        mappings,
        mem_read,
        nvmap_id,
        rt,
        vertex_addr,
        eff_vertex_count,
        out_index_count.unwrap_or(0),
        is_indexed,
        &fs_tex_ids,
        &sampled_rt_slots,
        depth_test,
        depth_write,
        (
            blend_state.enabled,
            maxwell.regs.blend_per_target_enabled,
            blend_raw_src,
            blend_raw_dst,
            blend_raw_eq,
            blend_raw_src_alpha,
            blend_raw_dst_alpha,
            blend_raw_eq_alpha,
        ),
        &blend_state,
        vs_cbuf_mask,
        fs_cbuf_mask,
        &maxwell.regs.cbuf_binds,
        cbuf_data.as_deref(),
        &bundle.fs_cbuf_reads,
    );

    let call = Maxwell3dDrawCall {
        vs_spirv,
        fs_spirv,
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
        vertex_count: eff_vertex_count,
        index_addr: None,
        index_count: out_index_count,
        index_type: out_index_type,
        index_data: out_index_data,
        quad_expand: draw.topology == 7 && out_index_count.is_none(),
        rt_key,
        rt_format: vk::Format::R8G8B8A8_UNORM,
        vp_rect: guest_viewport_rect(draw, rt.width as f32, rt.height as f32),
        scissor: None,
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
    };

    Ok(Some(call))
}

fn guest_viewport_rect(draw: &DrawCall, rt_w: f32, rt_h: f32) -> Option<[f32; 4]> {
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
    let sx = draw.viewport.scale_x.abs();
    let sy = draw.viewport.scale_y.abs();
    if sx <= 0.0 || sy <= 0.0 {
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
    let min_y = y.min(y + h);
    let max_y = y.max(y + h);
    if !draw.window_origin.lower_left()
        && x <= 0.5
        && min_y <= 0.5
        && (x + w) >= rt_w - 0.5
        && max_y >= rt_h - 0.5
    {
        return None;
    }
    if w < 1.0 || h.abs() < 1.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some([x, y, w, h])
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

#[derive(Clone, Copy)]
struct DrawTraceConfig {
    enabled: bool,
    start: u64,
    end: u64,
}

fn draw_trace_config() -> DrawTraceConfig {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<DrawTraceConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| DrawTraceConfig {
        enabled: std::env::var_os("NEXIUM_DRAW_TRACE").is_some(),
        start: std::env::var("NEXIUM_DRAW_TRACE_START")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        end: std::env::var("NEXIUM_DRAW_TRACE_END")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(u64::MAX),
    })
}

fn trace_draw(
    draw: &DrawCall,
    layout: &VertexLayout,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    nvmap_id: u32,
    rt: &RenderTarget,
    vertex_addr: u64,
    vertex_count: u32,
    index_count: u32,
    indexed: bool,
    fs_tex_ids: &[u32],
    sampled_rt_slots: &[Option<RtKey>],
    depth_test: bool,
    depth_write: bool,
    blend_raw: (bool, bool, u32, u32, u32, u32, u32, u32),
    blend: &BlendState,
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static DRAW_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = DRAW_SEQ.fetch_add(1, Ordering::Relaxed);
    if seq < cfg.start || seq > cfg.end {
        return;
    }
    let pos = position_bounds(layout, mappings, mem_read, vertex_addr, vertex_count);
    let attr = vertex_attr_sample(layout, mappings, mem_read, vertex_addr, vertex_count);
    let vp = guest_viewport_rect(draw, rt.width as f32, rt.height as f32);
    let clip = draw.surface_clip.effective(rt.width, rt.height);
    let cbuf = if matches!(draw.fs_shader_gpu_va, 0x20330 | 0x40430) || fs_tex_ids.is_empty() {
        if std::env::var_os("NEXIUM_DRAW_TRACE_CBUF_FULL").is_some() {
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
    log::warn!(
        "[drawtrace] op={} #{} rt={} {}x{} topo={} first={} v={} i={} indexed={} pos={} \
         vp_en={} vp={:?} clip=({},{} {}x{}) origin={:#x} ll={} fy={} \
         depth={}/{} clamp={} vclip={:#x}/{} func={:#x} zeta={} cull={} ff={:#x} \
         tex={:?} sampled={:?} \
         blend={} per={} rgb=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         a=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         vb={:#x} attrs={} {} cbuf={:#x}/{} masks={:#x}/{:#x} {} fs={:#x}",
        op_seq,
        seq,
        nvmap_id,
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
        clip.x,
        clip.y,
        clip.width,
        clip.height,
        draw.window_origin.raw,
        draw.window_origin.lower_left(),
        draw.window_origin.flip_y(),
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
        "[cleartrace] op={} rt={} {}x{} mask={:#x} color={} depth={} clear_depth={:.6} \
         ctrl={:#x} scissor_en={} rect={:?} clip=({},{} {}x{}) origin={:#x} ll={} fy={} zeta={}",
        op_seq,
        nvmap_id,
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
        let mut vals = Vec::new();
        for &(logical_slot, byte_offset) in cbuf_reads.iter().take(12) {
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
            PACKED_CBUF_SLOT_SIZE.min(64)
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

fn cbuf_bind_for_slot(cbuf_binds: &[[(u64, u32); 16]; 5], logical_slot: u32) -> (u64, u32) {
    let stage = if logical_slot < 16 { 0 } else { 4 };
    let binding = (logical_slot & 15) as usize;
    cbuf_binds[stage][binding]
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
                vals.push(format!(
                    "({:.3},{:.3},{:.3},{:.3})",
                    v[0], v[1], v[2], v[3]
                ));
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

fn first_vertex_buffer_address(
    vertex_buffers: &[VertexBuffer; 32],
    layout: &VertexLayout,
) -> Option<u64> {
    let first_binding = layout
        .bindings
        .iter()
        .find(|b| b.stride > 0)
        .or_else(|| layout.bindings.first())?
        .binding as usize;
    let vb = vertex_buffers.get(first_binding)?;
    let va = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
    if va == 0 {
        None
    } else {
        Some(va)
    }
}
