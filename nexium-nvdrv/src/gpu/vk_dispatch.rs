use std::sync::Arc;
use ash::vk;

use super::engines::Maxwell3D;
use super::engines::maxwell3d::{DrawCall, VertexBuffer};
use super::GpuMappings;

use nexium_gpu::draw::{Maxwell3dDrawCall, VertexAttr, VertexBinding, VertexLayout, DrawState, BlendState, DepthState};
use nexium_gpu::rt_cache::RtKey;

const SPH_SIZE: usize = 0x50;
const MAX_SASS_BYTES: usize = 16 * 1024;

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
            super::engines::sw_renderer::execute_draws(
                std::slice::from_ref(draw), mappings, maxwell_dma, mem_read, mem_write,
            );
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
                    std::slice::from_ref(draw), mappings, maxwell_dma, mem_read, mem_write,
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
        if mem_read(cpu, &mut buf) { Some(buf) } else { None }
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
            let tex_id = call.fs_tex_ids[0];
            if tex_id <= call.tic_pool_limit {
                let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
                if let Some(tic_raw) = read_guest(tic_addr, 32) {
                    if let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) {
                        let bpp = tic.format.src_bpp();
                        let pitch = (tic.width as usize) * (tic.height as usize) * bpp;
                        let read_size = if tic.is_block_linear {
                            nexium_gpu::texture::block_linear_byte_size(
                                tic.width,
                                tic.height,
                                bpp,
                                tic.block_height_log2,
                            )
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
            let tsc_id = call.fs_sampler_ids[0];
            if tsc_id <= call.tsc_pool_limit {
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
    cbuf_used: u32,
}

#[allow(clippy::type_complexity)]
fn shader_bundle_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<(u64, u64, u32, u32, u32, u32), std::sync::Arc<ShaderBundle>>>
{
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<(u64, u64, u32, u32, u32, u32), std::sync::Arc<ShaderBundle>>>,
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
    let rt_key = RtKey { nvmap_id, width: rt.width, height: rt.height };
    let no_depth = depth_disabled();

    if draw.is_clear {
        let mask = draw.clear_mask;
        let want_color_clear = mask == 0 || (mask & 0b11_1100) != 0;
        let want_depth_clear = (mask & 0x1) != 0 && draw.zeta_enable;
        let color = [draw.clear_color.r, draw.clear_color.g, draw.clear_color.b, draw.clear_color.a];
        let do_depth = want_depth_clear && !no_depth;
        if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
            let r = renderer.clone();
            let cdepth = draw.clear_depth;
            let (w, h) = (rt.width, rt.height);
            rt_thread.submit(Box::new(move || {
                if want_color_clear {
                    let _ = r.clear_target(nvmap_id, w, h, color);
                }
                if do_depth {
                    let _ = r.clear_depth(nvmap_id, w, h, cdepth);
                }
            }));
        } else {
            if want_color_clear {
                renderer.clear_target(nvmap_id, rt.width, rt.height, color)?;
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
    if !vs_prog.enabled || !fs_prog.enabled {
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
                draw.viewport.scale_z, draw.viewport.translate_z,
                vptx_scale_z, vptx_translate_z,
            );
        }
    }

    let ps_key = if draw.topology == 0 { draw.point_size.to_bits() } else { 0 };
    let window_ndc = if !draw.viewport_transform_en && rt.width > 0 && rt.height > 0 {
        Some((2.0 / rt.width as f32, 2.0 / rt.height as f32))
    } else {
        None
    };
    let win_key = if window_ndc.is_some() { (rt.width << 16) | (rt.height & 0xFFFF) } else { 0 };
    let shader_key = (vs_addr, fs_addr, vptx_scale_z.to_bits(), vptx_translate_z.to_bits(), ps_key, win_key);
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
                    before, walker_imm, bindless, fs_tex_ids,
                    draw.tic_pool_gpu_va, draw.tic_pool_limit,
                    fs_sass.len(),
                );
            }

            let required_outputs = nexium_spirv::scan_input_locations(&fs_spirv);
            let (vs_spirv, vs_cbuf_mask, vs_cbuf_used) = nexium_spirv::emit_vertex_with_bindings_opts(
                &vs_cfg,
                &required_outputs,
                nexium_spirv::VertexOptions {
                    vptx_scale_z,
                    vptx_translate_z,
                    point_size: if draw.topology == 0 { Some(draw.point_size) } else { None },
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
                    let _ = std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_vs.spv", dk), &vb);
                    let _ = std::fs::write(format!("C:/Users/Mythrax/Desktop/sh_{}_fs.spv", dk), &fb);
                    log::warn!(
                        "[shdump] #{} vs_mask={:#x} fs_mask={:#x} vs_bytes={} fs_bytes={} ntex={}",
                        dk, b.vs_cbuf_mask, b.fs_cbuf_mask, b.vs_spirv.len(), b.fs_spirv.len(), b.fs_tex_ids.len()
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
        cbuf_addr, cbuf_size,
        maxwell.regs.cbuf_binds.iter()
            .flat_map(|stage| stage.iter())
            .filter(|(a, s)| *a != 0 && *s > 0)
            .count(),
    );

    let mut fs_sampler_ids: Vec<u32> = vec![0u32; fs_tex_ids.len()];
    if draw.fs_bindless_cb_addr != 0 && draw.fs_bindless_cb_size as usize >= 128 && !fs_tex_ids.is_empty() {
        if let Some(cbuf_cpu) = mappings.cpu_address_for(draw.fs_bindless_cb_addr) {
            let mut cbuf_head = [0u8; 128];
            if mem_read(cbuf_cpu, &mut cbuf_head) {
                const HANDLE_TIC_MASK: u32 = 0x000F_FFFF;
                const INVALID_TIC: u32 = 0x000F_FFFF;
                for (i, unit_slot) in fs_tex_ids.iter_mut().enumerate() {
                    let off = (*unit_slot as usize) * 4;
                    if off + 4 <= cbuf_head.len() {
                        let handle = u32::from_le_bytes([
                            cbuf_head[off],
                            cbuf_head[off + 1],
                            cbuf_head[off + 2],
                            cbuf_head[off + 3],
                        ]);
                        let tic = handle & HANDLE_TIC_MASK;
                        let tsc = handle >> 20;
                        let was = *unit_slot;
                        if tic != INVALID_TIC && tic <= draw.tic_pool_limit {
                            log::trace!(
                                "bindless: unit {} -> handle {:#x} -> TIC {} TSC {}",
                                was, handle, tic, tsc
                            );
                            *unit_slot = tic;
                            fs_sampler_ids[i] = tsc;
                        } else {
                            log::trace!(
                                "bindless: unit {} -> handle {:#x} (invalid/no-tex)",
                                was, handle
                            );
                            *unit_slot = u32::MAX;
                            fs_sampler_ids[i] = tsc;
                        }
                    }
                }
            } else {
                log::warn!(
                    "bindless: mem_read failed at cpu={:#x} (gpu_va={:#x})",
                    cbuf_cpu, cbuf_addr
                );
            }
        } else {
            log::warn!(
                "bindless: cbuf gpu_va={:#x} not mapped to CPU address",
                cbuf_addr
            );
        }
    }

    let sampled_rt_key = if !fs_tex_ids.is_empty() && draw.tic_pool_gpu_va != 0 {
        let tex_id = fs_tex_ids[0];
        if tex_id != u32::MAX && tex_id <= draw.tic_pool_limit {
            let tic_addr = draw.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
            mappings.cpu_address_for(tic_addr).and_then(|cpu| {
                let mut tic_raw = [0u8; 32];
                if mem_read(cpu, &mut tic_raw) {
                    nexium_gpu::texture::TicEntry::parse(&tic_raw).and_then(|tic| {
                        mappings.nvmap_id_for(tic.gpu_va).map(|nv| RtKey {
                            nvmap_id: nv,
                            width: tic.width,
                            height: tic.height,
                        })
                    })
                } else {
                    None
                }
            })
        } else {
            None
        }
    } else {
        None
    };

    let vertex_addr = first_vertex_buffer_address(&draw.vertex_buffers, &layout)
        .ok_or_else(|| "no vertex buffer bound".to_string())?;

    if std::env::var_os("NEXIUM_PROBE_SHADE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        if k % 1500 == 0 && layout.attrs.len() >= 5 {
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
                                    buf[c * 4], buf[c * 4 + 1], buf[c * 4 + 2], buf[c * 4 + 3],
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
                        log::warn!("[vraw] draw{} stride={} v0=[{}]", k, vstride, floats.join(" "));
                    }
                }
            }
        }
    }

    let depth_test = !no_depth && draw.depth_test_enable && draw.zeta_enable;
    let depth_write = !no_depth && draw.depth_write_enable && draw.zeta_enable;
    let depth_key = if depth_test || depth_write { Some(rt_key) } else { None };

    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if draw.depth_test_enable && !LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                "depth: first depth-test draw — zeta_enable={} fmt={:#x} {}x{} \
                 test={} write={} func={:#x}->{:?} use_depth={}",
                draw.zeta_enable, draw.zeta.format, draw.zeta.width, draw.zeta.height,
                draw.depth_test_enable, draw.depth_write_enable,
                draw.depth_func, map_compare_op(draw.depth_func),
                depth_key.is_some(),
            );
        }
    }

    let (uni_cbuf_addr, uni_cbuf_size) = {
        let (a, s) = resolve_vs_cbuf(&maxwell.regs.cbuf_binds);
        if a != 0 && std::env::var_os("NEXIUM_VS_CBUF").is_some() {
            (a, s)
        } else {
            (cbuf_addr, cbuf_size)
        }
    };
    let uni_cbuf_size = uni_cbuf_size.min(bundle.cbuf_used);
    let cbuf_data = if uni_cbuf_addr != 0 && uni_cbuf_size > 0 {
        mappings.cpu_address_for(uni_cbuf_addr).and_then(|cpu| {
            let mut buf = vec![0u8; uni_cbuf_size as usize];
            if mem_read(cpu, &mut buf) { Some(buf) } else { None }
        })
    } else {
        None
    };

    let call = Maxwell3dDrawCall {
        vs_spirv,
        fs_spirv,
        vs_hash: bundle.vs_hash,
        fs_hash: bundle.fs_hash,
        vs_cbuf_mask,
        fs_cbuf_mask,
        fs_tex_ids,
        vertex_layout: layout,
        cbuf_addr: uni_cbuf_addr,
        cbuf_size: uni_cbuf_size,
        cbuf_data,
        vertex_addr,
        vertex_count: draw.vertex_count,
        index_addr: None,
        index_count: None,
        index_type: vk::IndexType::UINT16,
        rt_key,
        rt_format: vk::Format::R8G8B8A8_UNORM,
        vp_rect: guest_viewport_rect(draw, rt.width as f32, rt.height as f32),
        state: DrawState {
            topology,
            vertex_count: draw.vertex_count,
            index_count: 0,
            indexed: false,
        },
        blend: BlendState {
            enabled: maxwell.regs.blend_enable[0] && std::env::var("NEXIUM_NO_BLEND").is_err(),
            src_factor: map_blend_factor(maxwell.regs.blend_src_rgb),
            dst_factor: map_blend_factor(maxwell.regs.blend_dst_rgb),
            op: map_blend_op(maxwell.regs.blend_eq_rgb),
        },
        depth: DepthState {
            test_enabled: depth_test,
            write_enabled: depth_write,
            compare_op: map_compare_op(draw.depth_func),
        },
        depth_key,
        sampled_rt_key,
        clear: false,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        tic_pool_gpu_va: draw.tic_pool_gpu_va,
        tic_pool_limit: draw.tic_pool_limit,
        tsc_pool_gpu_va: draw.tsc_pool_gpu_va,
        tsc_pool_limit: draw.tsc_pool_limit,
        fs_sampler_ids,
        cull_test_enable: draw.cull_test_enable,
        cull_face: draw.cull_face,
        front_face: draw.front_face,
        poly_offset_enable: draw.poly_offset_fill_enable,
        poly_offset_units: draw.poly_offset_units,
        poly_offset_factor: draw.poly_offset_factor,
    };

    Ok(Some(call))
}

fn guest_viewport_rect(draw: &DrawCall, rt_w: f32, rt_h: f32) -> Option<[f32; 4]> {
    if !draw.viewport_transform_en {
        return None;
    }
    let sx = draw.viewport.scale_x.abs();
    let sy = draw.viewport.scale_y.abs();
    if sx <= 0.0 || sy <= 0.0 {
        return None;
    }
    let x = draw.viewport.translate_x - sx;
    let y = draw.viewport.translate_y - sy;
    let w = sx * 2.0;
    let h = sy * 2.0;
    if x <= 0.5 && y <= 0.5 && (x + w) >= rt_w - 0.5 && (y + h) >= rt_h - 0.5 {
        return None;
    }
    if w < 1.0 || h < 1.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some([x, y, w, h])
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
        _ => {
            log::warn!("map_blend_factor: unknown encoding {:#x}, defaulting to ONE", v);
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
            log::warn!("map_compare_op: unknown depth func {:#x}, defaulting to LESS_OR_EQUAL", v);
            vk::CompareOp::LESS_OR_EQUAL
        }
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
                        k, loc, a.buffer, a.offset, a.format, a.constant
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
            let stride = draw
                .vertex_buffers
                .get(binding as usize)
                .map(|vb| vb.stride)
                .unwrap_or(0);
            if stride == 0 {
                return Err(format!("attrib {}: binding {} has zero stride", loc, binding));
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

fn map_topology(t: u32) -> Option<vk::PrimitiveTopology> {
    match t {
        0 => Some(vk::PrimitiveTopology::POINT_LIST),
        1 => Some(vk::PrimitiveTopology::LINE_LIST),
        3 => Some(vk::PrimitiveTopology::LINE_STRIP),
        4 => Some(vk::PrimitiveTopology::TRIANGLE_LIST),
        5 => Some(vk::PrimitiveTopology::TRIANGLE_STRIP),
        6 => Some(vk::PrimitiveTopology::TRIANGLE_FAN),
        _ => None,
    }
}

fn resolve_cbuf(
    draw: &DrawCall,
    cbuf_binds: &[[(u64, u32); 16]; 5],
) -> (u64, u32) {
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
    let first_binding = layout.bindings.iter().find(|b| b.stride > 0).or_else(|| layout.bindings.first())?.binding as usize;
    let vb = vertex_buffers.get(first_binding)?;
    let va = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
    if va == 0 { None } else { Some(va) }
}
