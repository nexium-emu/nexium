use ash::vk;
use std::sync::Arc;

use super::engines::maxwell3d::{DrawCall, RenderTarget, VertexBuffer, WindowOrigin};
use super::engines::Maxwell3D;
use super::formats::map_surface_format;
use super::GpuMappings;

use nexium_gpu::draw::{
    BlendAttachmentState, BlendState, DepthState, DrawState, Maxwell3dDrawCall,
    StencilFaceState as GpuStencilFaceState, StencilState, VertexAttr, VertexBinding,
    VertexBufferBinding, VertexLayout,
};
use nexium_gpu::bundle_cache::{CbufIndexOrigin, CbufRead};
use nexium_gpu::rt_cache::RtKey;
use nexium_gpu::texture_manifest::{
    normalize_texture_numeric_manifest, texture_numeric_manifest_fingerprint,
    GraphicsTextureImageKind, TextureNumericBinding,
};

const SPH_SIZE: usize = 0x50;
const MAX_SASS_BYTES: usize = 64 * 1024;
const PACKED_CBUF_SLOTS: usize = nexium_spirv::GFX_CBUF_SLOTS as usize;

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

fn has_unimplemented_brx(cfg: &nexium_shader::Cfg) -> bool {
    cfg.blocks.iter().any(|block| {
        block.program.instructions.iter().any(|inst| {
            matches!(
                &inst.op,
                nexium_shader::IrOp::Unimplemented {
                    opcode: nexium_shader::Opcode::BRX,
                    ..
                }
            )
        })
    })
}

fn known_cbuf_index_bits(
    value: nexium_shader::IrValue,
    defs: &std::collections::HashMap<u32, &nexium_shader::IrInst>,
    depth: u8,
) -> Option<u32> {
    if depth >= 24 {
        return None;
    }
    let resolve = |value| known_cbuf_index_bits(value, defs, depth + 1);
    match value {
        nexium_shader::IrValue::Zero => Some(0),
        nexium_shader::IrValue::ImmU32(value) => Some(value),
        nexium_shader::IrValue::ImmF32(value) => Some(value.to_bits()),
        nexium_shader::IrValue::GprIn(_) => None,
        nexium_shader::IrValue::Inst(value) => {
            let inst = defs.get(&value.0)?;
            if inst.pred.is_some() {
                return None;
            }
            match &inst.op {
                nexium_shader::IrOp::Mov(value) => resolve(*value),
                nexium_shader::IrOp::IAdd {
                    a,
                    b,
                    neg_a,
                    neg_b,
                } => {
                    let mut a = resolve(*a)?;
                    let mut b = resolve(*b)?;
                    if *neg_a {
                        a = 0u32.wrapping_sub(a);
                    }
                    if *neg_b {
                        b = 0u32.wrapping_sub(b);
                    }
                    Some(a.wrapping_add(b))
                }
                nexium_shader::IrOp::IMul { a, b } => {
                    Some(resolve(*a)?.wrapping_mul(resolve(*b)?))
                }
                nexium_shader::IrOp::IScAdd {
                    a,
                    b,
                    shift,
                    neg_a,
                    neg_b,
                } => {
                    let mut a = resolve(*a)?;
                    let mut b = resolve(*b)?;
                    if *neg_a {
                        a = 0u32.wrapping_sub(a);
                    }
                    if *neg_b {
                        b = 0u32.wrapping_sub(b);
                    }
                    Some(a.wrapping_shl(u32::from(*shift)).wrapping_add(b))
                }
                _ => None,
            }
        }
    }
}

fn collect_cbuf_reads(cfg: &nexium_shader::Cfg, stage_base: u32) -> Vec<CbufRead> {
    let defs = cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
        .filter_map(|inst| inst.result.map(|result| (result.0, inst)))
        .collect::<std::collections::HashMap<_, _>>();
    let mut reads = Vec::new();
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            match inst.op {
                nexium_shader::IrOp::LoadCbuf {
                    binding,
                    byte_offset,
                } => reads.push(CbufRead {
                    logical_slot: (stage_base + u32::from(binding)) as u8,
                    byte_offset,
                    index_origin: CbufIndexOrigin::Static,
                }),
                nexium_shader::IrOp::LoadCbufIndexed {
                    binding,
                    byte_offset,
                    index,
                    ..
                } => reads.push(CbufRead {
                    logical_slot: (stage_base + u32::from(binding)) as u8,
                    byte_offset,
                    index_origin: known_cbuf_index_bits(index, &defs, 0)
                        .map(CbufIndexOrigin::Constant)
                        .unwrap_or_else(|| match index {
                            nexium_shader::IrValue::GprIn(register) => {
                                CbufIndexOrigin::Gpr(register)
                            }
                            nexium_shader::IrValue::Inst(value) => {
                                CbufIndexOrigin::Instruction(value.0)
                            }
                            nexium_shader::IrValue::Zero
                            | nexium_shader::IrValue::ImmU32(_)
                            | nexium_shader::IrValue::ImmF32(_) => {
                                unreachable!("immediate cbuf indices are always known")
                            }
                        }),
                }),
                nexium_shader::IrOp::LoadStorage {
                    cbuf_binding,
                    cbuf_offset,
                    ..
                } => {
                    for byte_offset in [cbuf_offset, cbuf_offset.wrapping_add(4)] {
                        reads.push(CbufRead {
                            logical_slot: (stage_base + u32::from(cbuf_binding)) as u8,
                            byte_offset,
                            index_origin: CbufIndexOrigin::Static,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    reads.sort_unstable();
    reads.dedup();
    reads
}

fn collect_graphics_cbuf_reads(
    vs_cfg: &nexium_shader::Cfg,
    fs_cfg: &nexium_shader::Cfg,
) -> Vec<CbufRead> {
    let mut reads = collect_cbuf_reads(vs_cfg, 0);
    reads.extend(collect_cbuf_reads(fs_cfg, 16));
    reads.sort_unstable();
    reads.dedup();
    reads
}

fn packed_cbuf_slot(data: &[u8], logical_slot: usize) -> Option<&[u8]> {
    if logical_slot >= PACKED_CBUF_SLOTS {
        return None;
    }
    let directory = logical_slot.checked_mul(8)?;
    let base_word = u32::from_le_bytes(data.get(directory..directory + 4)?.try_into().ok()?);
    let word_count = u32::from_le_bytes(data.get(directory + 4..directory + 8)?.try_into().ok()?);
    if word_count == 0 {
        return Some(&[]);
    }
    let start = usize::try_from(base_word).ok()?.checked_mul(4)?;
    let len = usize::try_from(word_count).ok()?.checked_mul(4)?;
    let end = start.checked_add(len)?;
    data.get(start..end)
}

fn packed_cbuf_word(data: &[u8], logical_slot: usize, byte_offset: usize) -> Option<u32> {
    let slot = packed_cbuf_slot(data, logical_slot)?;
    let bytes = slot.get(byte_offset..byte_offset.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn copy_mapped_cbuf_bytes(
    mappings: &GpuMappings,
    gpu_addr: u64,
    dst: &mut [u8],
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> bool {
    let Some(gpu_end) = gpu_addr.checked_add(dst.len() as u64) else {
        return false;
    };
    let mut cursor = 0usize;
    let mut copied_any = false;
    while cursor < dst.len() {
        let current_gpu = gpu_addr + cursor as u64;
        let Some((cpu_addr, available)) = mappings.cpu_range_for(current_gpu) else {
            let next_mapping = mappings
                .iter()
                .filter_map(|mapping| {
                    (mapping.gpu_va > current_gpu && mapping.gpu_va < gpu_end)
                        .then_some(mapping.gpu_va)
                })
                .min()
                .unwrap_or(gpu_end);
            cursor = (next_mapping - gpu_addr) as usize;
            continue;
        };
        let readable_len = usize::try_from(available.min((dst.len() - cursor) as u64))
            .unwrap_or(dst.len() - cursor);
        if readable_len == 0 {
            break;
        }
        let mut readable = vec![0u8; readable_len];
        if mem_read(cpu_addr, &mut readable) {
            dst[cursor..cursor + readable_len].copy_from_slice(&readable);
            copied_any = true;
        }
        cursor += readable_len;
    }
    copied_any
}

fn tic_can_alias_render_target(format: nexium_gpu::texture::TicFormat) -> bool {
    matches!(
        format,
        nexium_gpu::texture::TicFormat::R32G32B32A32
            | nexium_gpu::texture::TicFormat::R16G16B16A16
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
            | nexium_gpu::texture::TicFormat::Z24S8
            | nexium_gpu::texture::TicFormat::G24R8
            | nexium_gpu::texture::TicFormat::B10G11R11
            | nexium_gpu::texture::TicFormat::Unknown(47)
    )
}

fn tic_can_alias_render_target_view(tic: &nexium_gpu::texture::TicEntry) -> bool {
    !tic.is_buffer()
        && !matches!(tic.texture_type, 3 | 8)
        && tic_can_alias_render_target(tic.format)
}

fn tic_snapshot_layer_count(tic: &nexium_gpu::texture::TicEntry) -> u32 {
    match tic.texture_type {
        3 => 6,
        8 => tic.depth.saturating_mul(6).max(6),
        5 => tic.depth.max(1),
        2 => tic.depth.max(1),
        _ => 1,
    }
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
                flush_accum(batch, renderer, mappings, mem_read, mem_write);
            }
            if !render_enabled(draw, mappings, mem_read) {
                bump_draw_drop(
                    0,
                    &format!(
                        "re mode={} addr={:#x}",
                        draw.render_enable_mode, draw.render_enable_addr
                    ),
                );
                continue;
            }
        }
        if draw.draw_texture.is_some() {
            flush_accum(batch, renderer, mappings, mem_read, mem_write);
            match prepare_draw_texture_job(draw, mappings, mem_read) {
                Ok(job) => {
                    if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
                        let r = renderer.clone();
                        rt_thread.submit_named(
                            "draw-texture",
                            Box::new(move || {
                                if let Err(e) = execute_draw_texture_job(&r, &job) {
                                    log::debug!("vk_dispatch: DrawTexture failed: {}", e);
                                }
                            }),
                        );
                    } else if let Err(e) = execute_draw_texture_job(renderer, &job) {
                        log::debug!("vk_dispatch: DrawTexture failed: {}", e);
                    }
                }
                Err(e) => {
                    log::debug!("vk_dispatch: DrawTexture prepare failed: {}", e);
                    bump_draw_drop(1, &e);
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
            flush_accum(batch, renderer, mappings, mem_read, mem_write);
            if let Err(e) = execute_one(draw, mappings, maxwell, maxwell_dma, renderer, mem_read) {
                log::debug!("vk_dispatch: clear failed: {}", e);
                bump_draw_drop(3, &e);
            }
            continue;
        }
        match execute_one(draw, mappings, maxwell, maxwell_dma, renderer, mem_read) {
            Ok(None) => {}
            Ok(Some(call)) => {
                let flush_prior = batch.last().is_some_and(|last| {
                    last.rt_key != call.rt_key
                        || last.color_rt_keys != call.color_rt_keys
                        || last.color_rt_formats != call.color_rt_formats
                        || last.depth_key != call.depth_key
                        || last.depth_format != call.depth_format
                        || last.depth_aspects != call.depth_aspects
                });
                register_small_rt_after_prior_work(
                    call.rt_key,
                    call.small_rt_tile_mode,
                    || {
                        if flush_prior {
                            flush_accum(batch, renderer, mappings, mem_read, mem_write);
                        }
                    },
                );
                batch.push(call);
                if batch.len() >= 256 {
                    flush_accum(batch, renderer, mappings, mem_read, mem_write);
                }
            }
            Err(e) => {
                flush_accum(batch, renderer, mappings, mem_read, mem_write);
                log::debug!("vk_dispatch: sw fallback: {}", e);
                bump_draw_drop(2, &e);
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
    strict_cond_render()
        && draw.render_enable_override == 0
        && matches!(draw.render_enable_mode, 2 | 3 | 4)
}

fn strict_cond_render() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_STRICT_COND_RENDER").is_some())
}

fn bump_draw_drop(kind: usize, detail: &str) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTS: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    COUNTS[kind.min(3)].fetch_add(1, Ordering::Relaxed);
    let total: u64 = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    if total <= 8 || total % 64 == 0 {
        log::info!(
            "[draw-drop] re_skip={} dtex_fail={} sw_fallback={} clear_fail={} last={}",
            COUNTS[0].load(Ordering::Relaxed),
            COUNTS[1].load(Ordering::Relaxed),
            COUNTS[2].load(Ordering::Relaxed),
            COUNTS[3].load(Ordering::Relaxed),
            detail
        );
    }
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
                if !strict_cond_render() {
                    log_render_enable_miss(draw, "conditional-fail-open");
                    return true;
                }
                let Some(cpu) = mappings.cpu_address_for(draw.render_enable_addr) else {
                    log_render_enable_miss(draw, "unmapped");
                    return true;
                };
                let mut b = [0u8; 24];
                if !mem_read(cpu, &mut b) {
                    log_render_enable_miss(draw, "read-failed");
                    return true;
                }
                let initial = u64::from_le_bytes(b[0..8].try_into().unwrap());
                let current = u64::from_le_bytes(b[16..24].try_into().unwrap());
                match draw.render_enable_mode {
                    2 => initial != 0,
                    3 => initial == current,
                    4 => initial != current,
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

fn read_gpu_strict(
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    gpu_va: u64,
    len: usize,
) -> Option<Vec<u8>> {
    let mut data = vec![0u8; len];
    let mut offset = 0usize;
    while offset < data.len() {
        let Some(va) = gpu_va.checked_add(offset as u64) else {
            trace_gpu_read_failure(mappings, gpu_va, len, gpu_va, None, 0, "overflow");
            return None;
        };
        let Some((cpu, remaining)) = mappings.cpu_range_for(va) else {
            trace_gpu_read_failure(mappings, gpu_va, len, va, None, 0, "unmapped");
            return None;
        };
        let take = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(data.len() - offset);
        if take == 0 {
            trace_gpu_read_failure(
                mappings,
                gpu_va,
                len,
                va,
                Some(cpu),
                remaining,
                "empty-range",
            );
            return None;
        }
        if !mem_read(cpu, &mut data[offset..offset + take]) {
            trace_gpu_read_failure(mappings, gpu_va, len, va, Some(cpu), remaining, "cpu-read");
            return None;
        }
        offset += take;
    }
    Some(data)
}

fn trace_gpu_read_failure(
    mappings: &GpuMappings,
    gpu_va: u64,
    len: usize,
    failed_va: u64,
    cpu: Option<u64>,
    remaining: u64,
    reason: &str,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static TARGETS: OnceLock<Vec<u64>> = OnceLock::new();
    static HITS: AtomicU64 = AtomicU64::new(0);
    let targets = TARGETS.get_or_init(|| parse_env_u64_list("NEXIUM_GPU_READ_FAIL_VA"));
    if targets.is_empty()
        || !targets
            .iter()
            .any(|target| *target >= gpu_va && *target < gpu_va.saturating_add(len as u64))
    {
        return;
    }
    let hit = HITS.fetch_add(1, Ordering::Relaxed);
    if hit >= 32 {
        return;
    }
    log::warn!(
        "[gpu-read-fail] #{} reason={} request={:#x}+{:#x} failed={:#x} cpu={:?} remaining={:#x} {}",
        hit,
        reason,
        gpu_va,
        len,
        failed_va,
        cpu,
        remaining,
        mappings.describe_around(failed_va)
    );
}

pub fn flush_accum(
    batch: &mut Vec<Maxwell3dDrawCall>,
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    if batch.is_empty() {
        return;
    }
    let profile = nvprof_enabled();
    let t0 = std::time::Instant::now();
    let rt_thread = crate::render_thread::maybe_render_thread();
    let read_guest = |gpu_va: u64, len: usize| -> Option<Vec<u8>> {
        read_gpu_strict(mappings, mem_read, gpu_va, len)
    };
    writeback_cube_sample_dependencies(batch, renderer, mappings, mem_read, mem_write);
    let before = batch.len();
    let flushed = flush_batch(batch, renderer, rt_thread, mappings, &read_guest);
    if profile {
        log::warn!(
            "[nvprof] flush_accum before={} flushed={} total_ms={:.3}",
            before,
            flushed,
            elapsed_ms(t0)
        );
    }
}

fn same_rt_bindings(a: &Maxwell3dDrawCall, b: &Maxwell3dDrawCall) -> bool {
    a.rt_key == b.rt_key
        && a.color_rt_keys == b.color_rt_keys
        && a.color_rt_formats == b.color_rt_formats
        && a.depth_key == b.depth_key
}

fn uniform_rt_run_end(calls: &[Maxwell3dDrawCall], start: usize) -> usize {
    let first = &calls[start];
    let mut end = start + 1;
    while end < calls.len() && same_rt_bindings(first, &calls[end]) {
        end += 1;
    }
    end
}

fn graphics_ring_chunk_ranges_for_costs(costs: &[u64], safe_budget: u64) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < costs.len() {
        let mut end = start + 1;
        let first = costs[start];
        if first <= safe_budget {
            let mut used = first;
            while end < costs.len() {
                let Some(next) = used.checked_add(costs[end]) else {
                    break;
                };
                if next > safe_budget {
                    break;
                }
                used = next;
                end += 1;
            }
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

fn graphics_ring_chunk_ranges(calls: &[Maxwell3dDrawCall]) -> Vec<(usize, usize)> {
    let costs = calls
        .iter()
        .map(nexium_gpu::renderer::graphics_draw_ring_bytes_upper_bound)
        .collect::<Vec<_>>();
    graphics_ring_chunk_ranges_for_costs(
        &costs,
        nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES,
    )
}

fn flush_batch(
    batch: &mut Vec<Maxwell3dDrawCall>,
    renderer: &Arc<nexium_gpu::Renderer>,
    rt_thread: Option<&crate::render_thread::RenderThread>,
    mappings: &GpuMappings,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) -> usize {
    if batch.is_empty() {
        return 0;
    }
    let n = batch.len();
    let mut start = 0usize;
    while start < batch.len() {
        let end = uniform_rt_run_end(batch, start);
        let run = &batch[start..end];
        for (chunk_start, chunk_end) in graphics_ring_chunk_ranges(run) {
            let chunk = &run[chunk_start..chunk_end];
            let chunk_bytes = chunk.iter().fold(0u64, |total, call| {
                total.saturating_add(
                    nexium_gpu::renderer::graphics_draw_ring_bytes_upper_bound(call),
                )
            });
            if chunk_bytes > nexium_gpu::renderer::GRAPHICS_RING_CAPACITY_BYTES {
                log::warn!(
                    "graphics draw upload footprint {:#x} exceeds ring capacity {:#x}; submission will fail closed",
                    chunk_bytes,
                    nexium_gpu::renderer::GRAPHICS_RING_CAPACITY_BYTES,
                );
            }
            match rt_thread {
                Some(rt) => submit_draw_batch_async(chunk, renderer, rt, mappings, read_guest),
                None => {
                    if let Err(e) = renderer.execute_draws(chunk, read_guest) {
                        log::debug!("vk_dispatch: execute_draws failed: {}", e);
                    }
                }
            }
        }
        start = end;
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
        width: (rt.width / msaa_samples(draw.multisample_mode).0).max(1),
        height: (rt.height / msaa_samples(draw.multisample_mode).1).max(1),
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

fn snapshot_texture_once(
    snapshot: &mut std::collections::HashMap<u64, Vec<u8>>,
    generations: &mut std::collections::HashMap<(u64, usize), u64>,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
    gpu_va: u64,
    len: usize,
) -> usize {
    if snapshot.get(&gpu_va).is_some_and(|data| data.len() >= len) {
        generations
            .entry((gpu_va, len))
            .or_insert_with(|| nexium_gpu::tex_invalidate::region_gen_range(gpu_va, len as u64));
        return 0;
    }
    for attempt in 0..3 {
        let before = nexium_gpu::tex_invalidate::region_gen_range(gpu_va, len as u64);
        let Some(data) = read_guest(gpu_va, len) else {
            return 0;
        };
        let after = nexium_gpu::tex_invalidate::region_gen_range(gpu_va, len as u64);
        if before == after || attempt == 2 {
            let n = data.len();
            snapshot.insert(gpu_va, data);
            generations.insert((gpu_va, len), before);
            return n;
        }
    }
    0
}

#[derive(Clone, Copy)]
struct MovieDrawTraceConfig {
    enabled: bool,
    all_bound: bool,
    start: u64,
    limit: u64,
    all_limit: u64,
}

fn movie_draw_trace_config() -> MovieDrawTraceConfig {
    static CONFIG: std::sync::OnceLock<MovieDrawTraceConfig> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| MovieDrawTraceConfig {
        enabled: std::env::var_os("NEXIUM_MOVIE_DRAW_TRACE").is_some(),
        all_bound: std::env::var_os("NEXIUM_MOVIE_DRAW_TRACE_ALL").is_some(),
        start: std::env::var("NEXIUM_MOVIE_DRAW_TRACE_START")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(512),
        limit: std::env::var("NEXIUM_MOVIE_DRAW_TRACE_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(96),
        all_limit: std::env::var("NEXIUM_MOVIE_DRAW_TRACE_ALL_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256),
    })
}

static MOVIE_TIC_CENSUS_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn movie_draw_trace_candidate(tic: &nexium_gpu::texture::TicEntry, sampled_rt_alias: bool) -> bool {
    use nexium_gpu::texture::TicFormat;

    if !matches!(
        tic.format,
        TicFormat::A8B8G8R8 | TicFormat::R8G8B8A8 | TicFormat::R8G8 | TicFormat::R8
    ) {
        return false;
    }

    let near = |width: u32, height: u32, target_width: u32, target_height: u32, tolerance: u32| {
        width.abs_diff(target_width) <= tolerance && height.abs_diff(target_height) <= tolerance
    };
    let full_movie = near(tic.width, tic.height, 1280, 720, 64);
    let chroma_movie = matches!(tic.format, TicFormat::R8G8 | TicFormat::R8)
        && near(tic.width, tic.height, 640, 360, 32);
    let presentation_sized = !sampled_rt_alias
        && (near(tic.width, tic.height, 1600, 900, 64)
            || near(tic.width, tic.height, 1920, 1080, 64));
    let aspect_scaled = tic.width >= 1024
        && tic.width <= 2048
        && tic.height >= 540
        && tic.height <= 1152
        && u64::from(tic.width) * 10 >= u64::from(tic.height) * 16
        && u64::from(tic.width) * 10 <= u64::from(tic.height) * 19;

    full_movie || chroma_movie || presentation_sized || (!sampled_rt_alias && aspect_scaled)
}

fn movie_draw_sample_stats(data: &[u8]) -> (usize, usize, u64) {
    const SAMPLE_BUDGET: usize = 65_536;
    if data.is_empty() {
        return (0, 0, 0);
    }
    let step = (data.len() / SAMPLE_BUDGET).max(1);
    let mut sampled = 0usize;
    let mut nonzero = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in data.iter().step_by(step).take(SAMPLE_BUDGET) {
        sampled += 1;
        nonzero += usize::from(byte != 0);
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    (sampled, nonzero, hash)
}

fn video_tic_cpu_targets() -> &'static std::sync::Mutex<Vec<(u64, u64)>> {
    static TARGETS: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, u64)>>> =
        std::sync::OnceLock::new();
    TARGETS.get_or_init(|| std::sync::Mutex::new(Vec::with_capacity(8)))
}

pub(crate) fn register_video_tic_cpu_target(cpu_va: u64, size: u64) {
    let Ok(mut targets) = video_tic_cpu_targets().lock() else {
        return;
    };
    targets.retain(|(existing, _)| *existing != cpu_va);
    targets.push((cpu_va, size));
    while targets.len() > 8 {
        targets.remove(0);
    }
}

fn trace_tic_cpu_targets(
    mappings: &GpuMappings,
    call: &Maxwell3dDrawCall,
    slot: usize,
    tex_id: u32,
    tic: &nexium_gpu::texture::TicEntry,
    read_size: usize,
    snapshot: &std::collections::HashMap<u64, Vec<u8>>,
) {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    };

    static TARGETS: OnceLock<Vec<u64>> = OnceLock::new();
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(u64, u64, u32, usize)>>> =
        OnceLock::new();
    static COUNT: AtomicU64 = AtomicU64::new(0);

    let configured_targets = TARGETS.get_or_init(|| parse_env_u64_list("NEXIUM_TIC_CPU_TRACE"));
    let video_targets = if std::env::var_os("NEXIUM_TIC_VIDEO_BACKING_TRACE").is_some() {
        video_tic_cpu_targets()
            .lock()
            .map(|targets| targets.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    if configured_targets.is_empty() && video_targets.is_empty() {
        return;
    }
    let span = std::env::var("NEXIUM_TIC_CPU_TRACE_SPAN")
        .ok()
        .and_then(|value| parse_env_u64(&value))
        .filter(|span| *span != 0)
        .unwrap_or(0x15_1800);
    let Some(mapping) = mappings
        .iter()
        .filter(|mapping| {
            tic.gpu_va >= mapping.gpu_va && tic.gpu_va < mapping.gpu_va.saturating_add(mapping.size)
        })
        .last()
    else {
        return;
    };
    let offset = tic.gpu_va - mapping.gpu_va;
    let cpu_va = mapping.cpu_addr.saturating_add(offset);
    let available = mapping.size.saturating_sub(offset);
    let mapped_len = u64::try_from(read_size).unwrap_or(u64::MAX).min(available);
    let cpu_end = cpu_va.saturating_add(mapped_len);
    let target = configured_targets
        .iter()
        .copied()
        .map(|target| (target, span))
        .chain(video_targets)
        .find(|(target, target_span)| {
            let target_end = target.saturating_add(*target_span);
            cpu_va < target_end && *target < cpu_end
        });
    let Some((target, _)) = target else {
        return;
    };
    let key = (call.fs_gpu_va, tic.gpu_va, tex_id, slot);
    let Ok(mut seen) = SEEN
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
    else {
        return;
    };
    if !seen.insert(key) {
        return;
    }
    let sequence = COUNT.fetch_add(1, Ordering::Relaxed);
    if sequence >= 128 {
        return;
    }
    let data = snapshot.get(&tic.gpu_va).map(Vec::as_slice).unwrap_or(&[]);
    let (sampled, nonzero, hash) = movie_draw_sample_stats(data);
    log::warn!(
        "[tic-cpu-target] #{} target={:#x} fs={:#x} vs={:#x} slot={} tic={} gpu={:#x} cpu={:#x} nvmap={} map={:#x}+{:#x} fmt={:?} {}x{}x{} type={} bl={} bytes={}/{} sample={} nz={} hash={:#x}",
        sequence,
        target,
        call.fs_gpu_va,
        call.vs_gpu_va,
        slot,
        tex_id,
        tic.gpu_va,
        cpu_va,
        mapping.nvmap_id,
        mapping.gpu_va,
        mapping.size,
        tic.format,
        tic.width,
        tic.height,
        tic.depth,
        tic.texture_type,
        tic.is_block_linear,
        data.len(),
        read_size,
        sampled,
        nonzero,
        hash,
    );
}

fn trace_movie_bound_tic(
    config: MovieDrawTraceConfig,
    census_count: u64,
    present_key: Option<RtKey>,
    call: &Maxwell3dDrawCall,
    slot: usize,
    tex_id: u32,
    tic_addr: u64,
    tic_raw: &[u8],
    tic: Option<&nexium_gpu::texture::TicEntry>,
    read_size: Option<usize>,
    snapshot: &std::collections::HashMap<u64, Vec<u8>>,
    snapshot_generations: &std::collections::HashMap<(u64, usize), u64>,
) {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    };

    if !config.all_bound || census_count < config.start || config.all_limit == 0 {
        return;
    }
    let Some(present_key) = present_key else {
        return;
    };

    static MOVIE_BOUND_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    static MOVIE_BOUND_TICS: OnceLock<
        Mutex<std::collections::HashSet<(RtKey, u64, u32, u32, [u8; 32])>>,
    > = OnceLock::new();

    let mut raw_descriptor = [0u8; 32];
    let raw_len = tic_raw.len().min(raw_descriptor.len());
    raw_descriptor[..raw_len].copy_from_slice(&tic_raw[..raw_len]);
    let trace_key = (
        present_key,
        call.tic_pool_gpu_va,
        slot as u32,
        tex_id,
        raw_descriptor,
    );
    let Ok(mut seen) = MOVIE_BOUND_TICS
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
    else {
        return;
    };
    if MOVIE_BOUND_SEQUENCE.load(Ordering::Relaxed) >= config.all_limit || !seen.insert(trace_key) {
        return;
    }
    let sequence = MOVIE_BOUND_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    drop(seen);

    let w0 = u32::from_le_bytes(raw_descriptor[0..4].try_into().unwrap());
    let w2 = u32::from_le_bytes(raw_descriptor[8..12].try_into().unwrap());
    let w3 = u32::from_le_bytes(raw_descriptor[12..16].try_into().unwrap());
    let raw_format = w0 & 0x7f;
    let header_version = (w2 >> 21) & 0x7;
    let pitch = if matches!(header_version, 1 | 2) {
        (w3 & 0xffff) << 5
    } else {
        0
    };
    let raw_words = raw_descriptor
        .chunks_exact(4)
        .map(|word| format!("{:08x}", u32::from_le_bytes(word.try_into().unwrap())))
        .collect::<Vec<_>>()
        .join(":");
    let color_rts = call
        .color_rt_keys
        .iter()
        .map(|key| key.label())
        .collect::<Vec<_>>()
        .join(",");
    let (data, requested_size, generation) = match (tic, read_size) {
        (Some(tic), Some(read_size)) => (
            snapshot.get(&tic.gpu_va).map(Vec::as_slice).unwrap_or(&[]),
            read_size,
            snapshot_generations
                .get(&(tic.gpu_va, read_size))
                .copied()
                .unwrap_or(0),
        ),
        _ => (&[][..], 0, 0),
    };
    let (sampled, nonzero, hash) = movie_draw_sample_stats(data);
    let parsed = tic
        .map(|tic| {
            format!(
                "fmt={:?} ctype={:?} swizzle={:?} tex={}x{}x{} type={} base={} norm={} srgb={} va={:#x} bl={} block={}/{}/{}",
                tic.format,
                tic.component_types,
                tic.swizzle,
                tic.width,
                tic.height,
                tic.depth,
                tic.texture_type,
                tic.base_layer,
                tic.normalized_coords,
                tic.is_srgb,
                tic.gpu_va,
                tic.is_block_linear,
                tic.block_width_log2,
                tic.block_height_log2,
                tic.block_depth_log2,
            )
        })
        .unwrap_or_else(|| "parse=failed".to_string());
    log::warn!(
        "[movie-bound] #{} census={} fs={:#x} vs={:#x} slot={} tic={} pool={:#x} desc={:#x} raw_len={} raw={} raw_fmt={:#x} header={} pitch={} {} bytes={}/{} gen={} sample={} nz={} hash={:#x} present={} rt={} colors=[{}] v={} inst={} indexed={} clear={}",
        sequence,
        census_count,
        call.fs_gpu_va,
        call.vs_gpu_va,
        slot,
        tex_id,
        call.tic_pool_gpu_va,
        tic_addr,
        tic_raw.len(),
        raw_words,
        raw_format,
        header_version,
        pitch,
        parsed,
        data.len(),
        requested_size,
        generation,
        sampled,
        nonzero,
        hash,
        present_key.label(),
        call.rt_key.label(),
        color_rts,
        call.vertex_count,
        call.instance_count,
        call.state.indexed,
        call.clear,
    );
}

fn submit_draw_batch_async(
    batch: &[Maxwell3dDrawCall],
    renderer: &Arc<nexium_gpu::Renderer>,
    rt: &crate::render_thread::RenderThread,
    mappings: &GpuMappings,
    read_guest: &dyn Fn(u64, usize) -> Option<Vec<u8>>,
) {
    let profile = nvprof_enabled();
    let t_snapshot = std::time::Instant::now();
    let mut snapshot: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    let mut snapshot_generations: std::collections::HashMap<(u64, usize), u64> =
        std::collections::HashMap::new();
    let mut snapshot_reads = 0usize;
    let mut snapshot_bytes = 0usize;
    let mut tic_summ: Vec<String> = Vec::new();
    let movie_draw_trace = movie_draw_trace_config();
    let recent_present_keys = if movie_draw_trace.all_bound {
        nexium_gpu::renderer::movie_trace_present_keys()
    } else {
        Vec::new()
    };
    for call in batch {
        let movie_present_key = recent_present_keys.iter().copied().find(|present_key| {
            call.rt_key == *present_key
                || call
                    .color_rt_keys
                    .iter()
                    .any(|color_key| color_key == present_key)
        });
        for binding in &call.vertex_bindings {
            let Some((addr, vbytes)) = nexium_gpu::draw::vertex_binding_read_range(call, binding)
            else {
                continue;
            };
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
            for (slot, &tex_id) in call.fs_tex_ids.iter().enumerate() {
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
                        let layer_count = tic_snapshot_layer_count(&tic);
                        let read_size =
                            nexium_gpu::texture::texture_guest_size_bytes(&tic, layer_count)
                                .unwrap_or_else(|| layer_size.saturating_mul(layer_count as usize));
                        let n = snapshot_texture_once(
                            &mut snapshot,
                            &mut snapshot_generations,
                            read_guest,
                            tic.gpu_va,
                            read_size,
                        );
                        snapshot_reads += usize::from(n != 0);
                        snapshot_bytes += n;
                        trace_tic_cpu_targets(
                            mappings, call, slot, tex_id, &tic, read_size, &snapshot,
                        );
                        let w0 = u32::from_le_bytes(tic_raw[0..4].try_into().unwrap());
                        let w2 = u32::from_le_bytes(tic_raw[8..12].try_into().unwrap());
                        let w3 = u32::from_le_bytes(tic_raw[12..16].try_into().unwrap());
                        let raw_format = w0 & 0x7f;
                        let header_version = (w2 >> 21) & 0x7;
                        let pitch = if matches!(header_version, 1 | 2) {
                            (w3 & 0xffff) << 5
                        } else {
                            0
                        };
                        let movie_texture = matches!(header_version, 1 | 2 | 3 | 4)
                            && matches!(tic.texture_type, 1 | 7)
                            && tic.width >= 256
                            && tic.height >= 128;
                        let video_tic_trace = std::env::var_os("NEXIUM_VIDEO_DMA_TRACE").is_some();
                        if movie_texture
                            && (video_tic_trace
                                || movie_draw_trace.enabled
                                || movie_draw_trace.all_bound)
                        {
                            use std::sync::{
                                atomic::{AtomicU64, Ordering},
                                Mutex, OnceLock,
                            };
                            static VIDEO_TIC_SEQUENCE: AtomicU64 = AtomicU64::new(0);
                            static VIDEO_TICS: OnceLock<
                                Mutex<std::collections::HashSet<(u32, [u8; 32])>>,
                            > = OnceLock::new();
                            let raw_descriptor: [u8; 32] = tic_raw.as_slice().try_into().unwrap();
                            let key = (tex_id, raw_descriptor);
                            let is_new = VIDEO_TICS
                                .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
                                .lock()
                                .is_ok_and(|mut set| set.insert(key));
                            let sequence =
                                is_new.then(|| VIDEO_TIC_SEQUENCE.fetch_add(1, Ordering::Relaxed));
                            let census_count = VIDEO_TIC_SEQUENCE.load(Ordering::Relaxed);
                            MOVIE_TIC_CENSUS_COUNT.fetch_max(census_count, Ordering::Relaxed);
                            let opening_movie_size = tic.width == 1280 && tic.height == 720;
                            if video_tic_trace {
                                if let Some(sequence) = sequence
                                    .filter(|sequence| *sequence < 512 || opening_movie_size)
                                {
                                    let data =
                                        snapshot.get(&tic.gpu_va).map(Vec::as_slice).unwrap_or(&[]);
                                    let sample = &data[..data.len().min(4096)];
                                    let nonzero = sample.iter().filter(|&&byte| byte != 0).count();
                                    let checksum = sample.iter().fold(0u64, |sum, &byte| {
                                        sum.wrapping_add(byte as u64).wrapping_mul(31)
                                    });
                                    let generation = snapshot_generations
                                        .get(&(tic.gpu_va, read_size))
                                        .copied()
                                        .unwrap_or(0);
                                    log::warn!(
                                        "[video-tic] #{} fs={:#x} tic={} raw_fmt={:#x} {:?} header={} pitch={} {}x{} va={:#x} bl={} bh={} bytes={}/{} gen={} nz={} ck={:#x}",
                                        sequence,
                                        call.fs_gpu_va,
                                        tex_id,
                                        raw_format,
                                        tic.format,
                                        header_version,
                                        pitch,
                                        tic.width,
                                        tic.height,
                                        tic.gpu_va,
                                        tic.is_block_linear,
                                        tic.block_height_log2,
                                        data.len(),
                                        read_size,
                                        generation,
                                        nonzero,
                                        checksum
                                    );
                                }
                            }

                            let sampled_rt = call.sampled_rt_slots.get(slot).copied().flatten();
                            if movie_draw_trace.enabled
                                && census_count >= movie_draw_trace.start
                                && movie_draw_trace_candidate(&tic, sampled_rt.is_some())
                            {
                                static MOVIE_DRAW_SEQUENCE: AtomicU64 = AtomicU64::new(0);
                                static MOVIE_DRAWS: OnceLock<
                                    Mutex<
                                        std::collections::HashSet<(u64, u64, u32, u32, [u8; 32])>,
                                    >,
                                > = OnceLock::new();
                                let trace_key = (
                                    call.fs_gpu_va,
                                    call.tic_pool_gpu_va,
                                    slot as u32,
                                    tex_id,
                                    raw_descriptor,
                                );
                                let unique = MOVIE_DRAWS
                                    .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
                                    .lock()
                                    .is_ok_and(|mut set| set.insert(trace_key));
                                let trace_sequence = unique
                                    .then(|| MOVIE_DRAW_SEQUENCE.fetch_add(1, Ordering::Relaxed));
                                if let Some(trace_sequence) = trace_sequence
                                    .filter(|sequence| *sequence < movie_draw_trace.limit)
                                {
                                    let data =
                                        snapshot.get(&tic.gpu_va).map(Vec::as_slice).unwrap_or(&[]);
                                    let (sampled, nonzero, hash) = movie_draw_sample_stats(data);
                                    let generation = snapshot_generations
                                        .get(&(tic.gpu_va, read_size))
                                        .copied()
                                        .unwrap_or(0);
                                    let color_rts = call
                                        .color_rt_keys
                                        .iter()
                                        .map(|key| key.label())
                                        .collect::<Vec<_>>()
                                        .join(",");
                                    let sampled_rt = sampled_rt
                                        .map(|key| key.label())
                                        .unwrap_or_else(|| "-".to_string());
                                    let raw_words = tic_raw
                                        .chunks_exact(4)
                                        .map(|word| {
                                            format!(
                                                "{:08x}",
                                                u32::from_le_bytes(word.try_into().unwrap())
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join(":");
                                    log::warn!(
                                        "[movie-draw] #{} census={} fs={:#x} vs={:#x} slot={} tic={} pool={:#x} raw={} raw_fmt={:#x} fmt={:?} ctype={:?} swizzle={:?} header={} pitch={} tex={}x{}x{} type={} base={} norm={} srgb={} va={:#x} bl={} block={}/{}/{} bytes={}/{} gen={} sample={} nz={} hash={:#x} sampled_rt={} rt={} colors=[{}] v={} inst={} indexed={} clear={}",
                                        trace_sequence,
                                        census_count,
                                        call.fs_gpu_va,
                                        call.vs_gpu_va,
                                        slot,
                                        tex_id,
                                        call.tic_pool_gpu_va,
                                        raw_words,
                                        raw_format,
                                        tic.format,
                                        tic.component_types,
                                        tic.swizzle,
                                        header_version,
                                        pitch,
                                        tic.width,
                                        tic.height,
                                        tic.depth,
                                        tic.texture_type,
                                        tic.base_layer,
                                        tic.normalized_coords,
                                        tic.is_srgb,
                                        tic.gpu_va,
                                        tic.is_block_linear,
                                        tic.block_width_log2,
                                        tic.block_height_log2,
                                        tic.block_depth_log2,
                                        data.len(),
                                        read_size,
                                        generation,
                                        sampled,
                                        nonzero,
                                        hash,
                                        sampled_rt,
                                        call.rt_key.label(),
                                        color_rts,
                                        call.vertex_count,
                                        call.instance_count,
                                        call.state.indexed,
                                        call.clear,
                                    );
                                }
                            }
                        }
                        trace_movie_bound_tic(
                            movie_draw_trace,
                            MOVIE_TIC_CENSUS_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                            movie_present_key,
                            call,
                            slot,
                            tex_id,
                            tic_addr,
                            &tic_raw,
                            Some(&tic),
                            Some(read_size),
                            &snapshot,
                            &snapshot_generations,
                        );
                    } else {
                        trace_movie_bound_tic(
                            movie_draw_trace,
                            MOVIE_TIC_CENSUS_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                            movie_present_key,
                            call,
                            slot,
                            tex_id,
                            tic_addr,
                            &tic_raw,
                            None,
                            None,
                            &snapshot,
                            &snapshot_generations,
                        );
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
        let res = r.execute_draws_with_texture_generations(
            &calls,
            move |addr: u64, len: usize| {
                snapshot
                    .get(&addr)
                    .filter(|b| b.len() >= len)
                    .map(|b| b[..len].to_vec())
            },
            move |addr: u64, len: usize| {
                snapshot_generations
                    .get(&(addr, len))
                    .copied()
                    .unwrap_or_else(|| {
                        nexium_gpu::tex_invalidate::region_gen_range(addr, len as u64)
                    })
            },
        );
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
    if !rt.submit_timeout_named("draw-batch", job, std::time::Duration::from_secs(3)) {
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
    texture_numeric_manifest: Vec<TextureNumericBinding>,
    fs_tex_or_partners: std::collections::HashMap<u32, u32>,
    vs_tex_base: u32,
    vs_tex_count: u32,
    fs_sampler_arrayed: bool,
    vs_sampler_arrayed: bool,
    depth_compare_2d_mask: u32,
    depth_compare_cube_mask: u32,
    depth_compare_cube_array_mask: u32,
    graphics_cbuf_reads: Vec<CbufRead>,
    cbuf_used: u32,
    ssbo_descs: Vec<nexium_shader::StorageBufferAddr>,
}

fn bundle_content_key(
    vs_sass: &[u8],
    fs_sass: &[u8],
    fs_sph: Option<&[u8; SPH_SIZE]>,
    scalars: &[u32],
) -> u64 {
    fn eat(h: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }
    let mut h: u64 = 0xcbf29ce484222325;
    eat(&mut h, &(vs_sass.len() as u64).to_le_bytes());
    eat(&mut h, vs_sass);
    eat(&mut h, &(fs_sass.len() as u64).to_le_bytes());
    eat(&mut h, fs_sass);
    if let Some(sph) = fs_sph {
        eat(&mut h, sph.as_slice());
    }
    for s in scalars {
        eat(&mut h, &s.to_le_bytes());
    }
    h
}

fn bundle_l2_stat(hit: bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static HITS: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);
    if hit {
        HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    let h = HITS.load(Ordering::Relaxed);
    let m = MISSES.load(Ordering::Relaxed);
    if (h + m) % 32 == 0 {
        log::info!("[bundle-cache] l2_hits={} translated={}", h, m);
    }
}

type ShaderKey = (u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, u32, u64);

fn y_direction_key(lower_left: bool) -> u32 {
    (lower_left as u32) << 31
}

fn layer_output_shader_key(slot: Option<u32>) -> u32 {
    slot.map_or(0, |slot| ((slot & 0xff) + 1) << 18)
}

fn recognized_layer_output_slot(
    draw: &DrawCall,
    rt: &RenderTarget,
    vs_address_lo: u32,
    gs_address_lo: u32,
    gs_active: bool,
) -> Option<u32> {
    let volume_depth = render_target_volume_depth(rt);
    if std::env::var_os("NEXIUM_NO_LAYERED_LUT").is_some()
        || !gs_active
        || vs_address_lo != 0x88c30
        || gs_address_lo != 0x89f30
        || volume_depth <= 1
        || draw.instance_count.max(1) != volume_depth
    {
        return None;
    }
    Some(0x90)
}

fn shader_numeric_key(
    uint_output_mask: u32,
    sint_output_mask: u32,
) -> u32 {
    (uint_output_mask & 0xff) | ((sint_output_mask & 0xff) << 8)
}

#[derive(Clone, Debug, Default)]
struct FragmentTextureNumericMetadata {
    texel_fetches: Option<Vec<(u32, u8)>>,
    texture_ids: Vec<u32>,
    descriptor_ids: Vec<u32>,
    sampled_ids: Vec<u32>,
    buffer_candidates: Vec<u32>,
    image_kinds: Vec<(u32, GraphicsTextureImageKind)>,
    sampler_arrayed: bool,
    depth_compare_2d_ids: Vec<u32>,
    depth_compare_cube_ids: Vec<u32>,
    depth_compare_cube_array_ids: Vec<u32>,
    or_partners: std::collections::HashMap<u32, u32>,
}

#[derive(Clone, Debug)]
struct GraphicsTextureLayout {
    fs_ids: Vec<u32>,
    vs_tex_base: u32,
    vs_tex_count: u32,
    manifest: Vec<TextureNumericBinding>,
    fs_texel_buffer_mask: u32,
    vs_texel_buffer_mask: u32,
    fs_sampler_arrayed: bool,
    vs_sampler_arrayed: bool,
    depth_compare_2d_mask: u32,
    depth_compare_cube_mask: u32,
    depth_compare_cube_array_mask: u32,
}

impl GraphicsTextureLayout {
    fn texel_buffer_mask(&self) -> u32 {
        self.fs_texel_buffer_mask | self.vs_texel_buffer_mask
    }
}

fn spirv_texture_manifest_for_stage(
    manifest: &[TextureNumericBinding],
    descriptor_slot_base: u32,
    descriptor_count: u32,
) -> Vec<nexium_spirv::GraphicsTextureResource> {
    let descriptor_slot_end = descriptor_slot_base.saturating_add(descriptor_count);
    manifest
        .iter()
        .filter(|binding| {
            binding.descriptor_slot >= descriptor_slot_base
                && binding.descriptor_slot < descriptor_slot_end
        })
        .map(|binding| {
            nexium_spirv::GraphicsTextureResource::new(
                binding.shader_id,
                binding.descriptor_slot,
                binding.spirv_type(),
            )
            .with_image_kind(binding.spirv_image_kind())
        })
        .collect()
}

fn shader_resource_fingerprint(
    indirect_tables: u64,
    fs_texel_buffer_mask: u32,
    vs_texel_buffer_mask: u32,
    texture_manifest: u64,
    texture_view_metadata: u64,
) -> u64 {
    indirect_tables
        ^ (fs_texel_buffer_mask as u64)
            .wrapping_mul(0x9e37_79b1_85eb_ca87)
            .rotate_left(17)
        ^ (vs_texel_buffer_mask as u64)
            .wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            .rotate_left(41)
        ^ texture_manifest
            .wrapping_mul(0x1656_67b1_9e37_79f9)
            .rotate_left(29)
        ^ texture_view_metadata
            .wrapping_mul(0x85eb_ca77_c2b2_ae63)
            .rotate_left(11)
}

fn texture_view_metadata_fingerprint(layout: &GraphicsTextureLayout) -> u64 {
    let stage_shapes = (layout.fs_sampler_arrayed as u64)
        | ((layout.vs_sampler_arrayed as u64) << 1);
    stage_shapes.wrapping_mul(0x27d4_eb2f_1656_67c5)
        ^ (layout.depth_compare_2d_mask as u64)
            .wrapping_mul(0x85eb_ca77)
            .rotate_left(7)
        ^ (layout.depth_compare_cube_mask as u64)
            .wrapping_mul(0x9e37_79b1)
            .rotate_left(23)
        ^ (layout.depth_compare_cube_array_mask as u64)
            .wrapping_mul(0xc2b2_ae35)
            .rotate_left(47)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndirectTableKey {
    binding: u8,
    offset: u32,
    entries: u8,
}

fn indirect_table_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<u64, Vec<IndirectTableKey>>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<u64, Vec<IndirectTableKey>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn collect_indirect_tables(cfg: &nexium_shader::Cfg) -> Vec<IndirectTableKey> {
    let mut tables = cfg
        .blocks
        .iter()
        .filter_map(|block| match block.branch {
            nexium_shader::BranchKind::Indirect {
                cbuf_binding,
                cbuf_offset,
                table_entries,
                ..
            } => Some(IndirectTableKey {
                binding: cbuf_binding,
                offset: cbuf_offset,
                entries: table_entries,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    tables.sort_unstable_by_key(|table| (table.binding, table.offset, table.entries));
    tables.dedup();
    tables
}

fn indirect_table_fingerprint(
    vs_tables: &[IndirectTableKey],
    fs_tables: &[IndirectTableKey],
    cbuf_binds: &[[(u64, u32); 16]; 5],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> u64 {
    if vs_tables.is_empty() && fs_tables.is_empty() {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (stage, tables) in [(0usize, vs_tables), (4usize, fs_tables)] {
        for table in tables {
            for byte in [stage as u8, table.binding, table.entries] {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
            for byte in table.offset.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
            for index in 0..table.entries as u32 {
                let value = table
                    .offset
                    .checked_add(index.saturating_mul(4))
                    .and_then(|offset| {
                        read_stage_cbuf_u32(
                            stage,
                            table.binding,
                            offset,
                            cbuf_binds,
                            mappings,
                            mem_read,
                        )
                    })
                    .unwrap_or(u32::MAX);
                for byte in value.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x100_0000_01b3);
                }
            }
        }
    }
    hash
}

#[allow(clippy::type_complexity)]
fn shader_bundle_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<ShaderKey, std::sync::Arc<ShaderBundle>>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<ShaderKey, std::sync::Arc<ShaderBundle>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn fragment_texture_numeric_metadata_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<u64, FragmentTextureNumericMetadata>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<u64, FragmentTextureNumericMetadata>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn vertex_texture_numeric_metadata_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<u64, FragmentTextureNumericMetadata>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<u64, FragmentTextureNumericMetadata>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn shader_failed_set() -> &'static std::sync::Mutex<std::collections::HashSet<ShaderKey>> {
    use std::sync::OnceLock;
    static FAILED: OnceLock<std::sync::Mutex<std::collections::HashSet<ShaderKey>>> =
        OnceLock::new();
    FAILED.get_or_init(|| {
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

fn water_no_ztest() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("NEXIUM_WATER_NO_ZTEST").ok().as_deref() == Some("1"))
}

fn instancing_disabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        let v = std::env::var_os("NEXIUM_NO_INSTANCING").is_some();
        if v {
            log::info!(
                "nexium-nvdrv: instancing DISABLED (NEXIUM_NO_INSTANCING); all bindings forced to VERTEX rate"
            );
        }
        v
    })
}

fn zeta_dbg_enabled() -> bool {
    use std::sync::OnceLock;
    static Z: OnceLock<bool> = OnceLock::new();
    *Z.get_or_init(|| std::env::var_os("NEXIUM_ZETA_DBG").is_some())
}

fn trace_zeta_key(
    draw: &DrawCall,
    rt_key: RtKey,
    zeta_key: Option<RtKey>,
    depth_key: Option<RtKey>,
    depth_test: bool,
    depth_write: bool,
) {
    if !zeta_dbg_enabled() {
        return;
    }
    let op_seq = next_gpu_op_seq();
    let zeta_va = ((draw.zeta.address_hi as u64) << 32) | draw.zeta.address_lo as u64;
    log::warn!(
        "[zetadbg] op={} zeta_va={:#x} zeta={}x{} fmt={:#x} rt={} func={:#x}->{:?} \
         test={} write={} stencil={} two_side={} front=[{:#x},{:#x},{:#x},{:#x} ref={:#x} cmp={:#x} wr={:#x}] \
         back=[{:#x},{:#x},{:#x},{:#x} ref={:#x} cmp={:#x} wr={:#x}] zeta_key={} depth_key={}",
        op_seq,
        zeta_va,
        draw.zeta.width,
        draw.zeta.height,
        draw.zeta.format,
        rt_key.label(),
        draw.depth_func,
        map_compare_op(draw.depth_func),
        depth_test,
        depth_write,
        draw.stencil_enable,
        draw.stencil_two_side_enable,
        draw.stencil_front.fail_op,
        draw.stencil_front.depth_fail_op,
        draw.stencil_front.depth_pass_op,
        draw.stencil_front.compare_op,
        draw.stencil_front.reference,
        draw.stencil_front.compare_mask,
        draw.stencil_front.write_mask,
        draw.stencil_back.fail_op,
        draw.stencil_back.depth_fail_op,
        draw.stencil_back.depth_pass_op,
        draw.stencil_back.compare_op,
        draw.stencil_back.reference,
        draw.stencil_back.compare_mask,
        draw.stencil_back.write_mask,
        zeta_key
            .map(|k| k.label())
            .unwrap_or_else(|| "none".to_string()),
        depth_key
            .map(|k| k.label())
            .unwrap_or_else(|| "none".to_string()),
    );
    log::warn!(
        "[zetadbg-cull] op={} cull_en={} cull_face={:#x} front_face={:#x} flip_y={} zeta_en={}",
        op_seq,
        draw.cull_test_enable,
        draw.cull_face,
        draw.front_face,
        draw.window_origin.triangle_rast_flip(),
        draw.zeta_enable,
    );
}

fn small_rt_registry() -> &'static std::sync::Mutex<std::collections::HashMap<RtKey, u32>> {
    static R: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<RtKey, u32>>> =
        std::sync::OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn register_small_rt_after_prior_work(
    key: RtKey,
    tile_mode: Option<u32>,
    prior_work: impl FnOnce(),
) {
    prior_work();
    let Some(tile_mode) = tile_mode else {
        return;
    };
    small_rt_registry()
        .lock()
        .unwrap()
        .insert(key, tile_mode);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuestRange {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuestWriteChunk {
    gpu_va: u64,
    cpu_addr: u64,
    data_offset: usize,
    len: usize,
}

#[derive(Debug, Default)]
struct GuestWriteResult {
    complete: bool,
    written: Vec<GuestWriteChunk>,
}

#[derive(Debug, Default)]
struct CubeSampleDependencies {
    ranges: Vec<GuestRange>,
    texture_bases: Vec<u64>,
}

fn normalize_guest_ranges(mut ranges: Vec<GuestRange>) -> Vec<GuestRange> {
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut normalized: Vec<GuestRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if range.start >= range.end {
            continue;
        }
        if let Some(previous) = normalized.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        normalized.push(range);
    }
    normalized
}

fn aliased_guest_ranges(mappings: &GpuMappings, range: GuestRange) -> Vec<GuestRange> {
    if range.start >= range.end {
        return Vec::new();
    }
    let mut aliases = vec![range];
    let mut cursor = range.start;
    while cursor < range.end {
        let Some((cpu_addr, remaining)) = mappings.cpu_range_for(cursor) else {
            break;
        };
        let take = remaining.min(range.end - cursor);
        if take == 0 {
            break;
        }
        for (alias, len) in mappings.gpu_regions_for_cpu_range(cpu_addr, take) {
            if let Some(end) = alias.checked_add(len) {
                aliases.push(GuestRange { start: alias, end });
            }
        }
        let Some(next) = cursor.checked_add(take) else {
            break;
        };
        cursor = next;
    }
    aliases.sort_unstable_by_key(|range| (range.start, range.end));
    aliases.dedup();
    aliases
}

fn guest_aliases_for_address(mappings: &GpuMappings, gpu_va: u64) -> Vec<u64> {
    let mut aliases = vec![gpu_va];
    if let Some(cpu_addr) = mappings.cpu_address_for(gpu_va) {
        aliases.extend(
            mappings
                .gpu_regions_for_cpu_range(cpu_addr, 1)
                .into_iter()
                .map(|(alias, _)| alias),
        );
    }
    aliases.sort_unstable();
    aliases.dedup();
    aliases
}

fn plan_guest_write_chunks(
    mappings: &GpuMappings,
    gpu_va: u64,
    len: usize,
) -> Option<Vec<GuestWriteChunk>> {
    let mut chunks = Vec::new();
    let mut offset = 0usize;
    while offset < len {
        let va = gpu_va.checked_add(offset as u64)?;
        let (cpu_addr, remaining) = mappings.cpu_range_for(va)?;
        let take = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(len - offset);
        if take == 0 {
            return None;
        }
        chunks.push(GuestWriteChunk {
            gpu_va: va,
            cpu_addr,
            data_offset: offset,
            len: take,
        });
        offset = offset.checked_add(take)?;
    }
    Some(chunks)
}

fn write_guest_strict(
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    gpu_va: u64,
    data: &[u8],
) -> GuestWriteResult {
    let Some(chunks) = plan_guest_write_chunks(mappings, gpu_va, data.len()) else {
        return GuestWriteResult::default();
    };
    let mut result = GuestWriteResult {
        complete: true,
        written: Vec::with_capacity(chunks.len()),
    };
    for chunk in chunks {
        let end = chunk.data_offset + chunk.len;
        if !mem_write(chunk.cpu_addr, &data[chunk.data_offset..end]) {
            result.complete = false;
            break;
        }
        result.written.push(chunk);
    }
    result
}

fn guest_write_alias_ranges(
    mappings: &GpuMappings,
    chunks: &[GuestWriteChunk],
) -> Vec<GuestRange> {
    let mut aliases = Vec::new();
    for chunk in chunks {
        let len = chunk.len as u64;
        for (alias, available) in mappings.gpu_regions_for_cpu_range(chunk.cpu_addr, len) {
            if let Some(end) = alias.checked_add(available) {
                aliases.push(GuestRange { start: alias, end });
            }
        }
        if let Some(end) = chunk.gpu_va.checked_add(len) {
            aliases.push(GuestRange {
                start: chunk.gpu_va,
                end,
            });
        }
    }
    aliases.sort_unstable_by_key(|range| (range.start, range.end));
    aliases.dedup();
    aliases
}

fn invalidate_guest_write_chunks(
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    chunks: &[GuestWriteChunk],
) {
    for range in guest_write_alias_ranges(mappings, chunks) {
        nexium_gpu::tex_invalidate::bump_region(range.start, range.end - range.start);
        renderer.invalidate_texture_address(range.start);
    }
}

fn collect_cube_sample_ranges(
    batch: &[Maxwell3dDrawCall],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> CubeSampleDependencies {
    let mut ranges = Vec::new();
    let mut texture_bases = Vec::new();
    for call in batch {
        if call.tic_pool_gpu_va == 0 {
            continue;
        }
        for &tex_id in &call.fs_tex_ids {
            if tex_id == u32::MAX || tex_id > call.tic_pool_limit {
                continue;
            }
            let tic_addr = call
                .tic_pool_gpu_va
                .wrapping_add(u64::from(tex_id).saturating_mul(32));
            let Some(raw) = read_gpu_strict(mappings, mem_read, tic_addr, 32) else {
                continue;
            };
            let Some(tic) = nexium_gpu::texture::TicEntry::parse(&raw) else {
                continue;
            };
            if !matches!(tic.texture_type, 3 | 8) {
                continue;
            }
            let layers = tic_snapshot_layer_count(&tic);
            let pitch = tic.format.linear_size(tic.width, tic.height);
            let layer_size = if tic.is_block_linear {
                tic.format
                    .block_linear_size(tic.width, tic.height, tic.block_height_log2)
                    .max(pitch)
            } else {
                pitch
            };
            let read_size = nexium_gpu::texture::texture_guest_size_bytes(&tic, layers)
                .unwrap_or_else(|| layer_size.saturating_mul(layers as usize));
            let Ok(read_size) = u64::try_from(read_size) else {
                continue;
            };
            let Some(end) = tic.gpu_va.checked_add(read_size) else {
                continue;
            };
            let range = GuestRange {
                start: tic.gpu_va,
                end,
            };
            ranges.extend(aliased_guest_ranges(mappings, range));
            texture_bases.extend(guest_aliases_for_address(mappings, tic.gpu_va));
        }
    }
    texture_bases.sort_unstable();
    texture_bases.dedup();
    CubeSampleDependencies {
        ranges: normalize_guest_ranges(ranges),
        texture_bases,
    }
}

fn small_rt_starts_in_ranges(key: RtKey, ranges: &[GuestRange]) -> bool {
    ranges
        .iter()
        .any(|range| key.gpu_va >= range.start && key.gpu_va < range.end)
}

fn restore_small_rt_entries(entries: Vec<(RtKey, u32)>) {
    if entries.is_empty() {
        return;
    }
    let mut registry = small_rt_registry().lock().unwrap();
    for (key, tile_mode) in entries {
        registry.insert(key, tile_mode);
    }
}

fn writeback_small_rt_entries(
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    pending: Vec<(RtKey, u32)>,
    wait_timeout: std::time::Duration,
    marker_label: &'static str,
) -> usize {
    if pending.is_empty() {
        return 0;
    }
    if let Some(rt) = crate::render_thread::maybe_render_thread() {
        let (tx, rx) = std::sync::mpsc::channel();
        if !rt.submit_timeout_named(
            marker_label,
            Box::new(move || {
                let _ = tx.send(());
            }),
            wait_timeout,
        ) || rx.recv_timeout(wait_timeout).is_err()
        {
            restore_small_rt_entries(pending);
            return 0;
        }
    }

    let mut written = 0usize;
    let mut unresolved = Vec::new();
    for (key, tile_mode) in pending {
        let Some((kw, kh, bpp, mut raw)) = renderer.readback_target_raw_key(key) else {
            unresolved.push((key, tile_mode));
            continue;
        };
        let width_bytes = kw as usize * bpp;
        if kh >= 2 && raw.len() >= width_bytes * kh as usize {
            let height = kh as usize;
            for y in 0..height / 2 {
                let (top, bottom) = raw.split_at_mut((height - 1 - y) * width_bytes);
                top[y * width_bytes..(y + 1) * width_bytes]
                    .swap_with_slice(&mut bottom[..width_bytes]);
            }
        }
        let guest_bytes = if (tile_mode >> 12) & 1 == 1 {
            raw
        } else {
            let block_height_log2 = (tile_mode >> 4) & 0x7;
            super::engines::maxwell_dma::swizzle_block_linear(
                &raw,
                width_bytes,
                kh as usize,
                width_bytes,
                width_bytes,
                kh as usize,
                block_height_log2,
                0,
                0,
            )
        };
        let bytes_to_write = guest_bytes.len();
        if bytes_to_write == 0 {
            unresolved.push((key, tile_mode));
            continue;
        }
        let write = write_guest_strict(mappings, mem_write, key.gpu_va, &guest_bytes);
        invalidate_guest_write_chunks(renderer, mappings, &write.written);
        if !write.complete {
            unresolved.push((key, tile_mode));
            continue;
        }
        written += 1;
        log::debug!(
            "[rt-writeback] {} bpp={} tile={:#x} bytes={:#x}",
            key.label(),
            bpp,
            tile_mode,
            bytes_to_write,
        );
    }
    restore_small_rt_entries(unresolved);
    written
}

fn writeback_cube_sample_dependencies(
    batch: &[Maxwell3dDrawCall],
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    let dependencies = collect_cube_sample_ranges(batch, mappings, mem_read);
    if dependencies.ranges.is_empty() {
        return;
    }
    let current_targets = batch
        .iter()
        .flat_map(|call| {
            std::iter::once(call.rt_key)
                .chain(call.color_rt_keys.iter().copied())
                .chain(call.depth_key)
        })
        .collect::<std::collections::HashSet<_>>();
    let pending = {
        let mut registry = small_rt_registry().lock().unwrap();
        let mut selected = Vec::new();
        registry.retain(|key, tile_mode| {
            let matches = !current_targets.contains(key)
                && small_rt_starts_in_ranges(*key, &dependencies.ranges);
            if matches {
                selected.push((*key, *tile_mode));
            }
            !matches
        });
        selected
    };
    if pending.is_empty() {
        return;
    }
    let pending_count = pending.len();
    let written = writeback_small_rt_entries(
        renderer,
        mappings,
        mem_write,
        pending,
        std::time::Duration::from_secs(5),
        "cube-rt-dependency",
    );
    if written != 0 {
        for base in &dependencies.texture_bases {
            renderer.invalidate_texture_address(*base);
        }
    }
    if std::env::var_os("NEXIUM_CUBE_RT_SYNC_TRACE").is_some() {
        let range_labels = dependencies
            .ranges
            .iter()
            .map(|range| format!("{:#x}..{:#x}", range.start, range.end))
            .collect::<Vec<_>>()
            .join(",");
        log::warn!(
            "[cube-rt-sync] fs={:#x} ranges=[{}] pending={} written={}",
            batch.first().map(|call| call.fs_gpu_va).unwrap_or(0),
            range_labels,
            pending_count,
            written,
        );
    }
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
    let _ = writeback_small_rt_entries(
        renderer,
        mappings,
        mem_write,
        pending,
        std::time::Duration::from_millis(250),
        "small-rt-writeback",
    );
}

pub fn sync_render_thread() -> bool {
    let Some(rt) = crate::render_thread::maybe_render_thread() else {
        return true;
    };
    let (tx, rx) = std::sync::mpsc::channel();
    if !rt.try_submit(Box::new(move || {
        let _ = tx.send(());
    })) {
        log::warn!("compute image sync marker could not be queued");
        return false;
    }
    if rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
        log::warn!("compute image sync marker timed out");
        return false;
    }
    true
}

fn execute_one(
    draw: &DrawCall,
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    maxwell_dma: &super::engines::MaxwellDma,
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
    let (msx, msy) = msaa_samples(draw.multisample_mode);
    let rt_key = RtKey::with_cpu(
        nvmap_id,
        (rt.width / msx).max(1),
        (rt.height / msy).max(1),
        rt_gpu_va,
        mappings.cpu_address_for(rt_gpu_va).unwrap_or(0),
    )
    .with_volume_depth(render_target_volume_depth(rt));
    let rt_format = map_rt_format_for_key(rt.format, rt_key);
    let small_rt_tile_mode = (!draw.is_clear
        && !rt_key.is_3d
        && (rt.width as u64) * (rt.height as u64) <= 16384
        && {
        static SKIP: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        let skip = SKIP.get_or_init(|| {
            std::env::var("NEXIUM_NO_SMALL_RT_WB_NVMAPS")
                .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
                .unwrap_or_default()
        });
        !skip.contains(&rt_key.nvmap_id)
    })
    .then_some(rt.tile_mode);
    if std::env::var_os("NEXIUM_SMALL_RT_GATE_TRACE").is_some()
        && rt.width <= 256
        && rt.height <= 1536
    {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(RtKey, bool)>>> = OnceLock::new();
        let entry = (rt_key, small_rt_tile_mode.is_some());
        if SEEN
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap()
            .insert(entry)
        {
            log::warn!(
                "[small-rt-gate] key={} rtdims={}x{} vol={} is_3d={} clear={} tile={:#x} registered={}",
                rt_key.label(),
                rt.width,
                rt.height,
                render_target_volume_depth(rt),
                rt_key.is_3d,
                draw.is_clear,
                rt.tile_mode,
                small_rt_tile_mode.is_some(),
            );
        }
    }
    let no_depth = depth_disabled();
    let zeta_key = zeta_rt_key(draw, mappings, rt_key);

    if draw.is_clear {
        let op_seq = next_gpu_op_seq();
        let mask = draw.clear_mask;
        let want_color_clear = mask == 0 || (mask & 0b11_1100) != 0;
        let want_depth_clear = (mask & 0x1) != 0 && draw.zeta_enable;
        let want_stencil_clear = (mask & 0x2) != 0 && draw.zeta_enable;
        let clear_scissor = if draw.clear_control & 0x100 != 0 {
            scissor_rect(draw, rt_key.width, rt_key.height)
        } else {
            None
        };
        let color = [
            draw.clear_color.r,
            draw.clear_color.g,
            draw.clear_color.b,
            draw.clear_color.a,
        ];
        let do_depth = want_depth_clear && !no_depth && zeta_key.is_some();
        let do_stencil = want_stencil_clear && !no_depth && zeta_key.is_some();
        let depth_clear_key = zeta_key;
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
            do_stencil,
        );
        if let Some(rt_thread) = crate::render_thread::maybe_render_thread() {
            let r = renderer.clone();
            let cdepth = draw.clear_depth;
            let cstencil = draw.clear_stencil;
            let (w, h) = (rt_key.width, rt_key.height);
            let (depth_format, depth_aspects) = map_zeta_format(draw.zeta.format);
            let mut clear_aspects = vk::ImageAspectFlags::empty();
            if do_depth {
                clear_aspects |= vk::ImageAspectFlags::DEPTH;
            }
            if do_stencil {
                clear_aspects |= vk::ImageAspectFlags::STENCIL;
            }
            rt_thread.submit_named(
                "clear",
                Box::new(move || {
                    if want_color_clear {
                        if let Some(rect) = clear_scissor {
                            let _ = r.clear_target_rect_with_format(
                                nvmap_id, w, h, rt_gpu_va, color, rect, rt_format,
                            );
                        } else {
                            let _ = r.clear_target_with_format(
                                nvmap_id, w, h, rt_gpu_va, color, rt_format,
                            );
                        }
                    }
                    if let Some(depth_clear_key) = depth_clear_key {
                        if !clear_aspects.is_empty() {
                            let _ = r.clear_depth_stencil(
                                depth_clear_key,
                                depth_format,
                                depth_aspects,
                                clear_aspects,
                                cdepth,
                                cstencil,
                            );
                        }
                    }
                }),
            );
        } else {
            if want_color_clear {
                if let Some(rect) = clear_scissor {
                    renderer.clear_target_rect_with_format(
                        nvmap_id,
                        rt_key.width,
                        rt_key.height,
                        rt_gpu_va,
                        color,
                        rect,
                        rt_format,
                    )?;
                } else {
                    renderer.clear_target_with_format(
                        nvmap_id,
                        rt_key.width,
                        rt_key.height,
                        rt_gpu_va,
                        color,
                        rt_format,
                    )?;
                }
            }
            if let Some(depth_clear_key) = depth_clear_key {
                let (depth_format, depth_aspects) = map_zeta_format(draw.zeta.format);
                let mut clear_aspects = vk::ImageAspectFlags::empty();
                if do_depth {
                    clear_aspects |= vk::ImageAspectFlags::DEPTH;
                }
                if do_stencil {
                    clear_aspects |= vk::ImageAspectFlags::STENCIL;
                }
                if !clear_aspects.is_empty() {
                    renderer.clear_depth_stencil(
                        depth_clear_key,
                        depth_format,
                        depth_aspects,
                        clear_aspects,
                        draw.clear_depth,
                        draw.clear_stencil,
                    )?;
                }
            }
        }
        if std::env::var_os("NEXIUM_WATER_FORENSICS").is_some() {
            log::warn!(
                "[wf-clear] color={} rt={}x{}@{:#x} depth={} dkey={} cd={}",
                want_color_clear,
                rt.width,
                rt.height,
                rt_gpu_va,
                do_depth,
                depth_clear_key
                    .map(|key| key.label())
                    .unwrap_or_else(|| "none".to_string()),
                draw.clear_depth
            );
        }
        return Ok(None);
    }

    let program_region = ((maxwell.regs.program_region_va_hi as u64) << 32)
        | maxwell.regs.program_region_va_lo as u64;

    let vs_prog = &maxwell.regs.shader_programs[1];
    let gs_prog = &maxwell.regs.shader_programs[4];
    let fs_prog = &maxwell.regs.shader_programs[5];
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            let progs = maxwell
                .regs
                .shader_programs
                .iter()
                .enumerate()
                .map(|(i, p)| format!("[{}] en={} lo={:#x}", i, p.enabled, p.address_lo))
                .collect::<Vec<_>>()
                .join(" ");
            log::info!("[progmap] {}", progs);
        }
    }
    let vs_active = vs_prog.enabled || vs_prog.address_lo != 0;
    let gs_active = gs_prog.enabled || gs_prog.address_lo != 0;
    let fs_active = fs_prog.enabled || fs_prog.address_lo != 0;
    {
        use std::sync::OnceLock;
        static TARGET: OnceLock<Option<u64>> = OnceLock::new();
        let target = TARGET.get_or_init(|| {
            std::env::var("NEXIUM_PROGMAP_FS")
                .ok()
                .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        });
        if let Some(t) = target {
            let fs_addr = program_region.wrapping_add(fs_prog.address_lo as u64);
            if fs_addr == *t || fs_prog.address_lo as u64 == *t {
                let progs = maxwell
                    .regs
                    .shader_programs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| format!("[{}] en={} lo={:#x}", i, p.enabled, p.address_lo))
                    .collect::<Vec<_>>()
                    .join(" ");
                log::info!("[progmap-fs] fs={:#x} {}", fs_addr, progs);
            }
        }
    }
    if !vs_active || !fs_active {
        return Err("VS or FS program disabled".to_string());
    }
    let vsa_prog = &maxwell.regs.shader_programs[0];
    let vsa_active = {
        use std::sync::OnceLock;
        static OFF: OnceLock<bool> = OnceLock::new();
        !*OFF.get_or_init(|| std::env::var_os("NEXIUM_NO_DUAL_VS").is_some())
            && (vsa_prog.enabled || vsa_prog.address_lo != 0)
            && vsa_prog.address_lo != vs_prog.address_lo
    };
    let vsa_key = if vsa_active { vsa_prog.address_lo } else { 0 };
    let layer_output_slot =
        recognized_layer_output_slot(draw, rt, vs_prog.address_lo, gs_prog.address_lo, gs_active);
    if let Some(slot) = layer_output_slot {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                "promoting PPS pass-through GS to VS Layer output slot={:#x} rt={}",
                slot,
                rt_key.label()
            );
        }
    }

    let vs_addr = program_region.wrapping_add(vs_prog.address_lo as u64);
    let gs_addr = program_region.wrapping_add(gs_prog.address_lo as u64);
    let fs_addr = program_region.wrapping_add(fs_prog.address_lo as u64);
    if gs_active && env_dump_shader("NEXIUM_DUMP_GS", gs_addr) {
        dump_geometry_shader_once(
            gs_addr,
            program_region,
            gs_prog,
            draw,
            maxwell,
            mappings,
            mem_read,
        );
    }
    let guest_vptx =
        if draw.viewport.scale_z != 0.0 || draw.viewport.translate_z != 0.0 {
            (draw.viewport.scale_z, draw.viewport.translate_z)
        } else {
            (1.0, 0.0)
        };
    let depth_clip_control = std::env::var("NEXIUM_DEPTH_CLIP_CTL")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let apply_z_remap = draw.depth_mode == 0 && !depth_clip_control;
    let (vptx_scale_z, vptx_translate_z) = if apply_z_remap {
        (1.0, 0.0)
    } else {
        guest_vptx
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
    let surface_clip = draw.surface_clip.effective(rt_key.width, rt_key.height);
    let render_area = if !draw.viewport_transform_en {
        (surface_clip.width, surface_clip.height)
    } else {
        (rt_key.width, rt_key.height)
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
    let color_rt_keys = color_rts.iter().map(|(_, key, _)| *key).collect::<Vec<_>>();
    let color_rt_formats = color_rts
        .iter()
        .map(|(_, _, format)| *format)
        .collect::<Vec<_>>();
    let (fragment_uint_output_mask, fragment_sint_output_mask) =
        fragment_output_numeric_masks(&color_rts);
    let via_header_index = maxwell.regs.sampler_binding == 1;
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
    let alpha_test_key: u32 = if draw.alpha_test_enabled {
        (draw.alpha_test_func & 0xFFFF) | 0x8000_0000
    } else {
        0
    };
    let (cached_vs_tables, cached_fs_tables) = {
        let cache = indirect_table_cache().lock().unwrap();
        (
            cache.get(&vs_addr).cloned().unwrap_or_default(),
            cache.get(&fs_addr).cloned().unwrap_or_default(),
        )
    };
    let initial_indirect_fingerprint = indirect_table_fingerprint(
        &cached_vs_tables,
        &cached_fs_tables,
        &maxwell.regs.cbuf_binds,
        mappings,
        mem_read,
    );
    let cached_fs_texture_metadata = fragment_texture_numeric_metadata_cache()
        .lock()
        .unwrap()
        .get(&fs_addr)
        .cloned();
    let cached_vs_texture_metadata = vertex_texture_numeric_metadata_cache()
        .lock()
        .unwrap()
        .get(&vs_addr)
        .cloned();
    let texture_metadata_cached =
        cached_fs_texture_metadata.is_some() && cached_vs_texture_metadata.is_some();
    let mut texture_layout = match (
        cached_fs_texture_metadata.as_ref(),
        cached_vs_texture_metadata.as_ref(),
    ) {
        (Some(fs_metadata), Some(vs_metadata)) => Some(
            graphics_texture_layout_from_metadata(
                fs_metadata,
                vs_metadata,
                &maxwell.regs.cbuf_binds,
                maxwell.regs.bindless_texture_const_buffer_slot,
                maxwell.regs.tex_cb_index,
                draw.tic_pool_gpu_va,
                draw.tic_pool_limit,
                via_header_index,
                mappings,
                mem_read,
            )
            .map_err(|error| {
                format!(
                    "graphics texture manifest resolution failed vs_addr={vs_addr:#x} fs_addr={fs_addr:#x}: {error}"
                )
            })?,
        ),
        _ => None,
    };
    let initial_numeric_key = shader_numeric_key(
        fragment_uint_output_mask,
        fragment_sint_output_mask,
    ) | (if texture_metadata_cached { 0 } else { 3 << 16 })
        | y_direction_key(draw.window_origin.lower_left())
        | layer_output_shader_key(layer_output_slot);
    let initial_manifest_fingerprint = texture_layout
        .as_ref()
        .map(|layout| texture_numeric_manifest_fingerprint(&layout.manifest))
        .unwrap_or_else(|| texture_numeric_manifest_fingerprint(&[]));
    let initial_view_metadata_fingerprint = texture_layout
        .as_ref()
        .map(texture_view_metadata_fingerprint)
        .unwrap_or(0);
    let mut shader_key = (
        vs_addr ^ ((vsa_key as u64) << 40),
        fs_addr,
        vptx_scale_z.to_bits(),
        vptx_translate_z.to_bits(),
        ps_key,
        win_key,
        uint_attr_mask,
        sint_attr_mask,
        color_output_count,
        fs_output_map
            ^ (alpha_test_key as u32).rotate_left(16)
            ^ draw.alpha_test_ref.rotate_left(8),
        initial_numeric_key,
        shader_resource_fingerprint(
            initial_indirect_fingerprint,
            texture_layout
                .as_ref()
                .map_or(0, |layout| layout.fs_texel_buffer_mask),
            texture_layout
                .as_ref()
                .map_or(0, |layout| layout.vs_texel_buffer_mask),
            initial_manifest_fingerprint,
            initial_view_metadata_fingerprint,
        ),
    );
    if shader_failed_set().lock().unwrap().contains(&shader_key) {
        return Err("shader previously failed to emit".to_string());
    }
    let bundle = 'bundle: {
        let cache = shader_bundle_cache();
        let mut guard = cache.lock().unwrap();
        if let Some(b) = guard.get(&shader_key) {
            break 'bundle b.clone();
        } else {
            let vs_sass = match fetch_sass(vs_addr, mappings, mem_read) {
                Some(s) => s,
                None => {
                    sass_read_diag("VS", vs_addr, program_region, vs_prog.address_lo, mappings);
                    return Err("VS SASS read failed".to_string());
                }
            };
            let vs_sass = if vsa_active {
                let vsa_addr = program_region.wrapping_add(vsa_prog.address_lo as u64);
                let merged = fetch_sass(vsa_addr, mappings, mem_read)
                    .and_then(|a| nexium_shader::merge_dual_vertex_sass(&a, &vs_sass));
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                match merged {
                    Some(m) => {
                        if N.fetch_add(1, Ordering::Relaxed) < 16 {
                            log::warn!(
                                "[dual-vs] merged VertexA {:#x} + VertexB {:#x} ({} bytes)",
                                vsa_addr,
                                vs_addr,
                                m.len()
                            );
                        }
                        m
                    }
                    None => {
                        if N.fetch_add(1, Ordering::Relaxed) < 16 {
                            log::warn!(
                                "[dual-vs] merge FAILED for VertexA {:#x} + VertexB {:#x}, using VertexB only",
                                vsa_addr,
                                vs_addr
                            );
                        }
                        vs_sass
                    }
                }
            } else {
                vs_sass
            };
            let fs_sass = match fetch_sass(fs_addr, mappings, mem_read) {
                Some(s) => s,
                None => {
                    sass_read_diag("FS", fs_addr, program_region, fs_prog.address_lo, mappings);
                    return Err("FS SASS read failed".to_string());
                }
            };

            let rt_dims_dump = {
                use std::sync::OnceLock;
                static SPEC: OnceLock<Option<(u32, u32)>> = OnceLock::new();
                SPEC.get_or_init(|| {
                    std::env::var("NEXIUM_DUMP_FS_RT").ok().and_then(|s| {
                        let (w, h) = s.split_once('x')?;
                        Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
                    })
                })
                .map(|(w, h)| rt.width == w && rt.height == h)
                .unwrap_or(false)
            };
            if env_dump_shader("NEXIUM_DUMP_FS", fs_addr) || rt_dims_dump {
                let path = shader_dump_path(&format!("target_fs_{:x}.txt", fs_addr));
                let _ = std::fs::write(
                    &path,
                    format_shader_dump(
                        &fs_sass,
                        fs_sph.as_ref(),
                        Some(fs_input_map),
                        fs_output_map,
                    ),
                );
                log::warn!("[dump-fs] wrote {}", path.display());
            }
            if env_dump_shader("NEXIUM_DUMP_VS", vs_addr) || rt_dims_dump {
                let vs_sph = fetch_sph(vs_addr, mappings, mem_read);
                let path = shader_dump_path(&format!("target_vs_{:x}.txt", vs_addr));
                let _ = std::fs::write(
                    &path,
                    format_shader_dump(&vs_sass, vs_sph.as_ref(), None, 0),
                );
                log::warn!(
                    "[dump-vs] wrote {} vp_scale_z={} vp_translate_z={} applied={}/{} vp_en={}",
                    path.display(),
                    draw.viewport.scale_z,
                    draw.viewport.translate_z,
                    vptx_scale_z,
                    vptx_translate_z,
                    draw.viewport_transform_en
                );
            }

            let mut vs_cfg = nexium_shader::build_cfg_with_cbuf(&vs_sass, |binding, offset| {
                read_stage_cbuf_u32(
                    0,
                    binding,
                    offset,
                    &maxwell.regs.cbuf_binds,
                    mappings,
                    mem_read,
                )
            });
            let fs_cfg =
                nexium_shader::build_fragment_cfg_with_cbuf(&fs_sass, |binding, offset| {
                    read_stage_cbuf_u32(
                        4,
                        binding,
                        offset,
                        &maxwell.regs.cbuf_binds,
                        mappings,
                        mem_read,
                    )
                });
            let vs_tables = collect_indirect_tables(&vs_cfg);
            let fs_tables = collect_indirect_tables(&fs_cfg);
            {
                let mut cache = indirect_table_cache().lock().unwrap();
                cache.insert(vs_addr, vs_tables.clone());
                cache.insert(fs_addr, fs_tables.clone());
            }
            let mut fs_texture_numeric_metadata = fragment_texture_numeric_metadata(&fs_cfg)?;
            let mut vs_texture_numeric_metadata = fragment_texture_numeric_metadata(&vs_cfg)?;
            let fs_walk_len = shader_cfg_code_len(&fs_cfg).min(fs_sass.len());
            let fs_walk_sass = if fs_walk_len == 0 {
                fs_sass.as_slice()
            } else {
                &fs_sass[..fs_walk_len]
            };
            let (fs_walker_imm, fs_bindless) = append_walked_texture_ids(
                &mut fs_texture_numeric_metadata,
                nexium_shader::extract_fs_tex_ids(fs_walk_sass, 15),
            );
            let (vs_walker_imm, vs_bindless) = append_walked_texture_ids(
                &mut vs_texture_numeric_metadata,
                nexium_shader::extract_fs_tex_ids(vs_sass.as_slice(), 15),
            );
            fragment_texture_numeric_metadata_cache()
                .lock()
                .unwrap()
                .insert(fs_addr, fs_texture_numeric_metadata.clone());
            vertex_texture_numeric_metadata_cache()
                .lock()
                .unwrap()
                .insert(vs_addr, vs_texture_numeric_metadata.clone());
            let resolved_texture_layout = graphics_texture_layout_from_metadata(
                &fs_texture_numeric_metadata,
                &vs_texture_numeric_metadata,
                &maxwell.regs.cbuf_binds,
                maxwell.regs.bindless_texture_const_buffer_slot,
                maxwell.regs.tex_cb_index,
                draw.tic_pool_gpu_va,
                draw.tic_pool_limit,
                via_header_index,
                mappings,
                mem_read,
            )
            .map_err(|error| {
                format!(
                    "graphics texture manifest resolution failed vs_addr={vs_addr:#x} fs_addr={fs_addr:#x}: {error}"
                )
            })?;
            if resolved_texture_layout.fs_texel_buffer_mask != 0 {
                log::info!(
                    "[texel-buffer] fs={:#x} mask={:#010x} shader_ids={:?}",
                    fs_addr,
                    resolved_texture_layout.fs_texel_buffer_mask,
                    fs_texture_numeric_metadata.descriptor_ids
                );
            }
            if resolved_texture_layout.vs_texel_buffer_mask != 0 {
                log::info!(
                    "[texel-buffer] vs={:#x} mask={:#010x} shader_ids={:?}",
                    vs_addr,
                    resolved_texture_layout.vs_texel_buffer_mask,
                    vs_texture_numeric_metadata.descriptor_ids
                );
            }
            log::debug!(
                "graphics texture layout: vs={vs_addr:#x} fs={fs_addr:#x} fs_direct={} fs_walker={} fs_bindless={} vs_direct={} vs_walker={} vs_bindless={} ids={:?} manifest={:?}",
                fs_texture_numeric_metadata.texture_ids.len(),
                fs_walker_imm,
                fs_bindless,
                vs_texture_numeric_metadata.texture_ids.len(),
                vs_walker_imm,
                vs_bindless,
                resolved_texture_layout.fs_ids,
                resolved_texture_layout.manifest,
            );
            let manifest_fingerprint =
                texture_numeric_manifest_fingerprint(&resolved_texture_layout.manifest);
            let view_metadata_fingerprint =
                texture_view_metadata_fingerprint(&resolved_texture_layout);
            texture_layout = Some(resolved_texture_layout);
            shader_key.10 = shader_numeric_key(
                fragment_uint_output_mask,
                fragment_sint_output_mask,
            ) | y_direction_key(draw.window_origin.lower_left())
                | layer_output_shader_key(layer_output_slot);
            shader_key.11 = shader_resource_fingerprint(
                indirect_table_fingerprint(
                    &vs_tables,
                    &fs_tables,
                    &maxwell.regs.cbuf_binds,
                    mappings,
                    mem_read,
                ),
                texture_layout
                    .as_ref()
                    .map_or(0, |layout| layout.fs_texel_buffer_mask),
                texture_layout
                    .as_ref()
                    .map_or(0, |layout| layout.vs_texel_buffer_mask),
                manifest_fingerprint,
                view_metadata_fingerprint,
            );
            if shader_failed_set().lock().unwrap().contains(&shader_key) {
                return Err("shader previously failed to emit".to_string());
            }
            if let Some(bundle) = guard.get(&shader_key) {
                break 'bundle bundle.clone();
            }

            let fs_debug_targets_key = parse_env_u64_list("NEXIUM_FS_DEBUG_TARGET");
            let fs_debug_active_key =
                fs_debug_targets_key.is_empty() || fs_debug_targets_key.contains(&fs_addr);
            let active_texture_layout = texture_layout
                .as_ref()
                .ok_or_else(|| "graphics texture layout was not resolved".to_string())?;
            let manifest_fingerprint =
                texture_numeric_manifest_fingerprint(&active_texture_layout.manifest);
            let view_metadata_fingerprint =
                texture_view_metadata_fingerprint(active_texture_layout);
            let content_key = bundle_content_key(
                &vs_sass,
                &fs_sass,
                fs_sph.as_ref(),
                &[
                    vptx_scale_z.to_bits(),
                    vptx_translate_z.to_bits(),
                    ps_key,
                    win_key,
                    y_direction_key(draw.window_origin.lower_left()),
                    uint_attr_mask,
                    sint_attr_mask,
                    color_output_count,
                    fs_output_map,
                    fs_debug_active_key as u32,
                    alpha_test_key,
                    draw.alpha_test_ref,
                    fragment_uint_output_mask,
                    fragment_sint_output_mask,
                    layer_output_slot.unwrap_or(0),
                    active_texture_layout.fs_texel_buffer_mask,
                    active_texture_layout.vs_texel_buffer_mask,
                    active_texture_layout.fs_sampler_arrayed as u32,
                    active_texture_layout.vs_sampler_arrayed as u32,
                    active_texture_layout.depth_compare_2d_mask,
                    active_texture_layout.depth_compare_cube_mask,
                    active_texture_layout.depth_compare_cube_array_mask,
                    manifest_fingerprint as u32,
                    (manifest_fingerprint >> 32) as u32,
                    view_metadata_fingerprint as u32,
                    (view_metadata_fingerprint >> 32) as u32,
                    shader_key.11 as u32,
                    (shader_key.11 >> 32) as u32,
                ],
            );
            let l2_store = nexium_gpu::bundle_cache::bundle_store();
            if has_unimplemented_brx(&vs_cfg) || has_unimplemented_brx(&fs_cfg) {
                shader_failed_set().lock().unwrap().insert(shader_key);
                l2_store.mark_failed(content_key);
                return Err(format!(
                    "unsupported BRX shader rejected vs_addr={:#x} fs_addr={:#x}",
                    vs_addr, fs_addr
                ));
            }
            if l2_store.is_failed(content_key) {
                shader_failed_set().lock().unwrap().insert(shader_key);
                return Err("shader previously failed to emit".to_string());
            }
            'translate: {
                if let Some(rec) = l2_store.get(content_key) {
                    bundle_l2_stat(true);
                    let b = std::sync::Arc::new(ShaderBundle {
                        vs_spirv: std::sync::Arc::new(rec.vs_spirv.clone()),
                        vs_cbuf_mask: rec.vs_cbuf_mask,
                        vs_hash: rec.vs_hash,
                        fs_spirv: std::sync::Arc::new(rec.fs_spirv.clone()),
                        fs_cbuf_mask: rec.fs_cbuf_mask,
                        fs_hash: rec.fs_hash,
                        fs_tex_ids: rec.fs_tex_ids.clone(),
                        texture_numeric_manifest: rec.texture_numeric_manifest.clone(),
                        fs_tex_or_partners: rec.fs_tex_or_partners.iter().copied().collect(),
                        vs_tex_base: rec.vs_tex_base,
                        vs_tex_count: rec.vs_tex_count,
                        fs_sampler_arrayed: rec.fs_sampler_arrayed,
                        vs_sampler_arrayed: rec.vs_sampler_arrayed,
                        depth_compare_2d_mask: rec.depth_compare_2d_mask,
                        depth_compare_cube_mask: rec.depth_compare_cube_mask,
                        depth_compare_cube_array_mask: rec.depth_compare_cube_array_mask,
                        graphics_cbuf_reads: rec.graphics_cbuf_reads.clone(),
                        cbuf_used: rec.cbuf_used,
                        ssbo_descs: rec
                            .ssbo_descs
                            .iter()
                            .map(|&(cbuf_binding, cbuf_offset, align)| {
                                nexium_shader::StorageBufferAddr {
                                    cbuf_binding,
                                    cbuf_offset,
                                    align,
                                }
                            })
                            .collect(),
                    });
                    guard.insert(shader_key, b.clone());
                    break 'translate b;
                }
                bundle_l2_stat(false);
                let _translate_guard = nexium_common::shader_progress::guard();

                {
                    let vs_stg = nexium_shader::shader_uses_stg(&vs_sass);
                    let fs_stg = nexium_shader::shader_uses_stg(&fs_sass);
                    if vs_stg || fs_stg {
                        log::warn!(
                            "[stg-shader] vs={:#x} fs={:#x} vs_stg={} fs_stg={}",
                            vs_addr,
                            fs_addr,
                            vs_stg,
                            fs_stg
                        );
                    }
                }
                let ssbo_descs = if nexium_shader::shader_uses_ldg(&vs_sass) {
                    nexium_shader::collect_storage_buffers(&mut vs_cfg)
                } else {
                    Vec::new()
                };
                let graphics_cbuf_reads = collect_graphics_cbuf_reads(&vs_cfg, &fs_cfg);
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

                let (
                    fs_spirv,
                    fs_cbuf_mask,
                    emitted_fs_tex_ids,
                    fs_cbuf_used,
                    fs_sampler_arrayed,
                ) =
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let fs_debug_targets = parse_env_u64_list("NEXIUM_FS_DEBUG_TARGET");
                        let fs_debug_active =
                            fs_debug_targets.is_empty() || fs_debug_targets.contains(&fs_addr);
                        nexium_spirv::emit_fragment_full_with_options(
                            &fs_cfg,
                            fs_input_map,
                            color_output_count,
                            fs_output_map,
                            fs_debug_active,
                            if draw.alpha_test_enabled {
                                draw.alpha_test_func
                            } else {
                                0
                            },
                            if draw.alpha_test_enabled {
                                draw.alpha_test_ref
                            } else {
                                0
                            },
                            nexium_spirv::FragmentOptions {
                                uint_output_mask: fragment_uint_output_mask,
                                sint_output_mask: fragment_sint_output_mask,
                                texture_numeric_manifest: spirv_texture_manifest_for_stage(
                                    &active_texture_layout.manifest,
                                    0,
                                    fs_texture_numeric_metadata.texture_ids.len() as u32,
                                ),
                                texel_buffer_mask: active_texture_layout.fs_texel_buffer_mask,
                                y_negate: draw.window_origin.lower_left(),
                            },
                        )
                    })) {
                        Ok(v) => v,
                        Err(panic) => {
                            shader_failed_set().lock().unwrap().insert(shader_key);
                            l2_store.mark_failed(content_key);
                            return Err(format!(
                                "FS SPIR-V emit panicked vs_addr={:#x} fs_addr={:#x}: {}",
                                vs_addr,
                                fs_addr,
                                shader_panic_message(panic)
                            ));
                        }
                    };
                if emitted_fs_tex_ids != fs_texture_numeric_metadata.texture_ids {
                    return Err(format!(
                        "FS texture descriptor layout mismatch fs_addr={fs_addr:#x}: emitter={emitted_fs_tex_ids:?} metadata={:?}",
                        fs_texture_numeric_metadata.texture_ids
                    ));
                }
                if fs_sampler_arrayed != active_texture_layout.fs_sampler_arrayed {
                    return Err(format!(
                        "FS texture view layout mismatch fs_addr={fs_addr:#x}: emitter_arrayed={fs_sampler_arrayed} metadata_arrayed={}",
                        active_texture_layout.fs_sampler_arrayed
                    ));
                }
                let fs_tex_ids = active_texture_layout.fs_ids.clone();
                let vs_tex_base = active_texture_layout.vs_tex_base;
                let vs_tex_count = active_texture_layout.vs_tex_count;

                let required_outputs = nexium_spirv::scan_input_locations(&fs_spirv);
                let (vs_spirv, vs_cbuf_mask, vs_cbuf_used) =
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        nexium_spirv::emit_vertex_with_bindings_opts(
                            &vs_cfg,
                            &required_outputs,
                            nexium_spirv::VertexOptions {
                                vptx_scale_z,
                                vptx_translate_z,
                                apply_z_remap,
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
                                texture_numeric_manifest: spirv_texture_manifest_for_stage(
                                    &active_texture_layout.manifest,
                                    vs_tex_base,
                                    vs_texture_numeric_metadata.texture_ids.len() as u32,
                                ),
                                texel_buffer_mask: active_texture_layout.vs_texel_buffer_mask,
                                layer_output_slot,
                                ..Default::default()
                            },
                        )
                    })) {
                        Ok(v) => v,
                        Err(panic) => {
                            shader_failed_set().lock().unwrap().insert(shader_key);
                            l2_store.mark_failed(content_key);
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
                    texture_numeric_manifest: active_texture_layout.manifest.clone(),
                    fs_tex_or_partners: fs_cfg.bindless_or_partners.clone(),
                    vs_tex_base,
                    vs_tex_count,
                    fs_sampler_arrayed,
                    vs_sampler_arrayed: active_texture_layout.vs_sampler_arrayed,
                    depth_compare_2d_mask: active_texture_layout.depth_compare_2d_mask,
                    depth_compare_cube_mask: active_texture_layout.depth_compare_cube_mask,
                    depth_compare_cube_array_mask: active_texture_layout
                        .depth_compare_cube_array_mask,
                    graphics_cbuf_reads,
                    cbuf_used: vs_cbuf_used.max(fs_cbuf_used),
                    ssbo_descs,
                });
                guard.insert(shader_key, b.clone());
                l2_store.insert(std::sync::Arc::new(
                    nexium_gpu::bundle_cache::BundleRecord {
                        content_key,
                        vs_spirv: (*b.vs_spirv).clone(),
                        fs_spirv: (*b.fs_spirv).clone(),
                        vs_cbuf_mask: b.vs_cbuf_mask,
                        fs_cbuf_mask: b.fs_cbuf_mask,
                        vs_hash: b.vs_hash,
                        fs_hash: b.fs_hash,
                        fs_tex_ids: b.fs_tex_ids.clone(),
                        texture_numeric_manifest: b.texture_numeric_manifest.clone(),
                        vs_tex_base: b.vs_tex_base,
                        vs_tex_count: b.vs_tex_count,
                        fs_sampler_arrayed: b.fs_sampler_arrayed,
                        vs_sampler_arrayed: b.vs_sampler_arrayed,
                        depth_compare_2d_mask: b.depth_compare_2d_mask,
                        depth_compare_cube_mask: b.depth_compare_cube_mask,
                        depth_compare_cube_array_mask: b.depth_compare_cube_array_mask,
                        graphics_cbuf_reads: b.graphics_cbuf_reads.clone(),
                        cbuf_used: b.cbuf_used,
                        ssbo_descs: b
                            .ssbo_descs
                            .iter()
                            .map(|d| (d.cbuf_binding, d.cbuf_offset, d.align))
                            .collect(),
                        fs_tex_or_partners: b
                            .fs_tex_or_partners
                            .iter()
                            .map(|(&k, &v)| (k, v))
                            .collect(),
                    },
                ));
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
                    let probe_count = std::env::var("NEXIUM_PROBE_SHADE_COUNT")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(24);
                    if dk < probe_count || target {
                        let tag = if target {
                            format!("target_{:x}_{:x}", vs_addr, fs_addr)
                        } else {
                            format!("{}_{:x}_{:x}", dk, vs_addr, fs_addr)
                        };
                        let dir = std::env::var("NEXIUM_PROBE_SHADE_DIR")
                            .unwrap_or_else(|_| "C:/Users/Mythrax/Desktop".to_string());
                        let _ = std::fs::create_dir_all(&dir);
                        let vb: Vec<u8> = b.vs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let fb: Vec<u8> = b.fs_spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let _ = std::fs::write(format!("{}/sh_{}_vs.spv", dir, tag), &vb);
                        let _ = std::fs::write(format!("{}/sh_{}_fs.spv", dir, tag), &fb);
                        let _ = std::fs::write(format!("{}/sh_{}_vs.sass", dir, tag), &vs_sass);
                        let _ = std::fs::write(format!("{}/sh_{}_fs.sass", dir, tag), &fs_sass);
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
                        let _ = std::fs::write(format!("{}/sh_{}_vs.txt", dir, tag), vs_dis);
                        let _ = std::fs::write(format!("{}/sh_{}_fs.txt", dir, tag), fs_dis);
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
        }
    };
    let vs_spirv = bundle.vs_spirv.clone();
    let fs_spirv = if fs_solid_probe_matches(fs_addr) {
        std::sync::Arc::new(solid_red_fs_spirv())
    } else {
        bundle.fs_spirv.clone()
    };
    let vs_cbuf_mask = bundle.vs_cbuf_mask;
    let fs_cbuf_mask = bundle.fs_cbuf_mask;
    let mut fs_tex_ids = bundle.fs_tex_ids.clone();
    let vs_tex_base = bundle.vs_tex_base;
    let vs_tex_count = bundle.vs_tex_count;
    let fs_sampler_arrayed = bundle.fs_sampler_arrayed;
    let vs_sampler_arrayed = bundle.vs_sampler_arrayed;
    let depth_compare_2d_mask = bundle.depth_compare_2d_mask;
    let depth_compare_cube_mask = bundle.depth_compare_cube_mask;
    let depth_compare_cube_array_mask = bundle.depth_compare_cube_array_mask;
    let active_texture_layout = texture_layout
        .as_ref()
        .ok_or_else(|| "graphics texture layout missing after shader resolution".to_string())?;
    if bundle.texture_numeric_manifest != active_texture_layout.manifest
        || bundle.fs_tex_ids != active_texture_layout.fs_ids
        || bundle.vs_tex_base != active_texture_layout.vs_tex_base
        || bundle.vs_tex_count != active_texture_layout.vs_tex_count
        || bundle.fs_sampler_arrayed != active_texture_layout.fs_sampler_arrayed
        || bundle.vs_sampler_arrayed != active_texture_layout.vs_sampler_arrayed
        || bundle.depth_compare_2d_mask != active_texture_layout.depth_compare_2d_mask
        || bundle.depth_compare_cube_mask != active_texture_layout.depth_compare_cube_mask
        || bundle.depth_compare_cube_array_mask
            != active_texture_layout.depth_compare_cube_array_mask
    {
        return Err(format!(
            "graphics texture cache identity mismatch vs_addr={vs_addr:#x} fs_addr={fs_addr:#x} bundle_manifest={:?} resolved_manifest={:?}",
            bundle.texture_numeric_manifest, active_texture_layout.manifest
        ));
    }
    let texture_numeric_manifest = bundle.texture_numeric_manifest.clone();
    let texel_buffer_mask = active_texture_layout.texel_buffer_mask();

    let vs_input_locations = nexium_spirv::scan_input_locations(&vs_spirv);
    let layout = build_vertex_layout(draw, &vs_input_locations)?;
    let topology = map_topology(draw.topology)
        .ok_or_else(|| format!("unsupported topology {}", draw.topology))?;

    let (cbuf_addr, cbuf_size) = resolve_cbuf(draw, &maxwell.regs.cbuf_binds);

    {
        use std::sync::{Mutex, OnceLock};
        static SPEC: OnceLock<Option<(u32, u32)>> = OnceLock::new();
        static SEEN: OnceLock<Mutex<std::collections::HashSet<(u64, u64)>>> = OnceLock::new();
        let spec = SPEC.get_or_init(|| {
            std::env::var("NEXIUM_DUMP_FS_RT").ok().and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
            })
        });
        if let Some((w, h)) = spec {
            if rt.width == *w && rt.height == *h {
                let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                if seen.lock().unwrap().insert((vs_addr, fs_addr)) {
                    for (label, addr) in [("vs", vs_addr), ("fs", fs_addr)] {
                        if let Some(sass) = fetch_sass(addr, mappings, mem_read) {
                            let dis = nexium_shader::disassemble(&sass)
                                .into_iter()
                                .map(|line| line.to_string_compact())
                                .collect::<Vec<_>>()
                                .join("\n");
                            let path =
                                shader_dump_path(&format!("rtdump_{}_{:x}.txt", label, addr));
                            let _ = std::fs::write(&path, dis);
                        }
                    }
                    log::warn!(
                        "[dump-fs-rt] {}x{} vs={:#x} fs={:#x} v={} inst={}",
                        rt.width,
                        rt.height,
                        vs_addr,
                        fs_addr,
                        draw.vertex_count,
                        draw.instance_count
                    );
                }
            }
        }
    }

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
    let split_vs_stage = std::env::var("NEXIUM_VS_TEX_STAGE_REMAP")
        .map(|v| v != "0")
        .unwrap_or(true);
    if !fs_tex_ids.is_empty() {
        let fs_end = if split_vs_stage && vs_tex_count != 0 {
            (vs_tex_base as usize).min(fs_tex_ids.len())
        } else {
            fs_tex_ids.len()
        };
        if fs_end != 0 {
            remap_texture_ids_for_stage(
                "fs",
                &maxwell.regs.cbuf_binds[4],
                maxwell.regs.bindless_texture_const_buffer_slot,
                maxwell.regs.tex_cb_index,
                &mut fs_tex_ids[..fs_end],
                &mut fs_sampler_ids[..fs_end],
                0,
                &mut fs_tex_remap,
                draw.tic_pool_gpu_va,
                draw.tic_pool_limit,
                via_header_index,
                mappings,
                mem_read,
                &bundle.fs_tex_or_partners,
            );
        }
        if split_vs_stage && vs_tex_count != 0 {
            let vs_start = (vs_tex_base as usize).min(fs_tex_ids.len());
            let vs_end = vs_start
                .saturating_add(vs_tex_count as usize)
                .min(fs_tex_ids.len());
            if vs_start < vs_end {
                remap_texture_ids_for_stage(
                    "vs",
                    &maxwell.regs.cbuf_binds[0],
                    maxwell.regs.bindless_texture_const_buffer_slot,
                    maxwell.regs.tex_cb_index,
                    &mut fs_tex_ids[vs_start..vs_end],
                    &mut fs_sampler_ids[vs_start..vs_end],
                    vs_start,
                    &mut fs_tex_remap,
                    draw.tic_pool_gpu_va,
                    draw.tic_pool_limit,
                    via_header_index,
                    mappings,
                    mem_read,
                    &bundle.fs_tex_or_partners,
                );
            }
        }
    }
    trace_vs_tex_remap(
        vs_addr,
        fs_addr,
        vs_tex_base,
        vs_tex_count,
        &shader_fs_tex_ids,
        &fs_tex_ids,
        &fs_sampler_ids,
        &fs_tex_remap,
        maxwell.regs.bindless_texture_const_buffer_slot,
        maxwell.regs.tex_cb_index,
        via_header_index,
        split_vs_stage,
        draw.tic_pool_gpu_va,
        draw.tic_pool_limit,
    );
    if fs_remap_trace(fs_addr, bundle.fs_hash)
        && (!fs_tex_remap.is_empty() || !bundle.texture_numeric_manifest.is_empty())
    {
        let numeric_bindings = bundle
            .texture_numeric_manifest
            .iter()
            .map(|binding| {
                let tic_id = fs_tex_ids
                    .get(binding.descriptor_slot as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                format!(
                    "slot{} shader={:#x} tic={:#x} type={:?} kind={:?}",
                    binding.descriptor_slot,
                    binding.shader_id,
                    tic_id,
                    binding.spirv_type(),
                    binding.image_kind,
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        log::info!(
            "[remap-trace] fs={:#x} fs_hash={:016x} shader_tex={:?} final_tex={:?} samplers={:?} bindless_slot={} tex_cb_slot={} via_header={} tic_pool={:#x} limit={} numeric=[{}] remap=[{}]",
            fs_addr,
            bundle.fs_hash,
            shader_fs_tex_ids,
            fs_tex_ids,
            fs_sampler_ids,
            maxwell.regs.bindless_texture_const_buffer_slot,
            maxwell.regs.tex_cb_index,
            via_header_index,
            draw.tic_pool_gpu_va,
            draw.tic_pool_limit,
            numeric_bindings,
            fs_tex_remap.join(" | ")
        );
    }

    trace_matching_tic_pool_entries(
        fs_addr,
        bundle.fs_hash,
        draw.tic_pool_gpu_va,
        draw.tic_pool_limit,
        mappings,
        mem_read,
    );

    let mut sampled_rt_keys: Vec<RtKey> = Vec::new();
    let mut sampled_rt_slots: Vec<Option<RtKey>> = vec![None; fs_tex_ids.len()];
    let mut sampled_rt_copy_sources: Vec<Option<RtKey>> = vec![None; fs_tex_ids.len()];
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
                        tic_can_alias_render_target_view(&tic),
                        tic.swizzle
                    );
                }
            }
            if !tic_can_alias_render_target_view(&tic) {
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
            )
            .with_volume_depth(if tic.texture_type == 2 {
                tic.depth.max(1)
            } else {
                1
            });
            sampled_rt_slots[slot] = Some(key);
            if let Some((source, stamp)) = maxwell_dma.rt_copy_source(tic.gpu_va) {
                if renderer.render_target_stamp(source) == Some(stamp) {
                    sampled_rt_copy_sources[slot] = Some(source);
                }
            }
            if !sampled_rt_keys.contains(&key) {
                sampled_rt_keys.push(key);
            }
        }
    }
    let sampled_rt_key = sampled_rt_keys.first().copied();

    let vertex_bindings = vertex_buffer_bindings(&draw.vertex_buffers, &layout);
    let vertex_addr = vertex_bindings.first().map(|b| b.addr).unwrap_or(0);

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

    let water_probe = water_no_ztest() && maxwell.regs.blend_enable[0];
    let (depth_test, depth_write, stencil_test) = effective_depth_states(
        no_depth || water_probe,
        draw.zeta_enable,
        draw.depth_test_enable,
        draw.depth_write_enable,
        draw.stencil_enable,
    );
    let depth_key = if depth_test || stencil_test {
        zeta_key
    } else {
        None
    };

    trace_zeta_key(draw, rt_key, zeta_key, depth_key, depth_test, depth_write);

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
    let _fallback_cbuf_size = fallback_cbuf_size.min(bundle.cbuf_used);
    let cbuf_data = Some(pack_cbuf_data(
        &maxwell.regs.cbuf_binds,
        vs_cbuf_mask,
        fs_cbuf_mask,
        mappings,
        mem_read,
    ));
    let (call_cbuf_addr, call_cbuf_size) = if let Some(data) = &cbuf_data {
        (0, data.len() as u32)
    } else {
        (fallback_cbuf_addr, _fallback_cbuf_size)
    };

    if std::env::var_os("NEXIUM_WATER_FORENSICS").is_some() {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN_OMAP: OnceLock<Mutex<HashSet<(u64, u32, u32)>>> = OnceLock::new();
        let k = (fs_addr, fs_output_map, maxwell.regs.color_masks[0]);
        if SEEN_OMAP
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap()
            .insert(k)
        {
            log::warn!("[wf-omap] fs={:#x} omap={:#x} cmask0={:#x}", k.0, k.1, k.2);
        }
    }

    trace_water_forensics(
        draw,
        rt_key,
        &color_rt_keys,
        vs_addr,
        fs_addr,
        vs_cbuf_mask,
        fs_cbuf_mask,
        &maxwell.regs.cbuf_binds,
        (
            maxwell.regs.blend_enable[0],
            if maxwell.regs.blend_per_target_enabled {
                maxwell.regs.blend_pt_src_rgb[0]
            } else {
                maxwell.regs.blend_src_rgb
            },
            if maxwell.regs.blend_per_target_enabled {
                maxwell.regs.blend_pt_dst_rgb[0]
            } else {
                maxwell.regs.blend_dst_rgb
            },
        ),
        mappings,
        mem_read,
    );

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
        &bundle.graphics_cbuf_reads,
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
            let raw = read_gpu_strict(mappings, mem_read, start, icount * isz);
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
        let location = color_rt_locations
            .get(index)
            .copied()
            .unwrap_or(index)
            .min(7);
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
        vs_addr,
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
        &bundle.graphics_cbuf_reads,
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
        &bundle.graphics_cbuf_reads,
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

    let (depth_format, depth_aspects) = if depth_key.is_some() {
        map_zeta_format(draw.zeta.format)
    } else {
        (vk::Format::UNDEFINED, vk::ImageAspectFlags::empty())
    };
    let stencil_front = map_stencil_face(draw.stencil_front);
    let stencil_back = if draw.stencil_two_side_enable {
        map_stencil_face(draw.stencil_back)
    } else {
        stencil_front
    };
    let (front_face, present_flip_y) =
        maxwell_draw_orientation(draw.window_origin, draw.front_face);

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
        texture_numeric_manifest,
        texel_buffer_mask,
        vs_tex_base,
        vs_tex_count,
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
        small_rt_tile_mode,
        color_rt_keys,
        color_rt_formats,
        rt_format,
        vp_rect: guest_viewport_rect(
            draw,
            rt_key.width as f32,
            rt_key.height as f32,
            signed_viewport_nvmap(rt_key.nvmap_id),
        ),
        scissor: scissor_rect(draw, rt_key.width, rt_key.height),
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
        depth_mode: draw.depth_mode,
        depth_format,
        depth_aspects,
        stencil: StencilState {
            enabled: stencil_test,
            front: stencil_front,
            back: stencil_back,
        },
        depth_clamp_enabled: draw.viewport_clip_control.depth_clamp_enabled(),
        depth_key,
        clear_depth_hint: draw.clear_depth,
        clear_stencil_hint: draw.clear_stencil,
        sampled_rt_key,
        sampled_rt_keys,
        sampled_rt_slots,
        sampled_rt_copy_sources,
        clear: false,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        tic_pool_gpu_va: draw.tic_pool_gpu_va,
        tic_pool_limit: draw.tic_pool_limit,
        tsc_pool_gpu_va: draw.tsc_pool_gpu_va,
        tsc_pool_limit: draw.tsc_pool_limit,
        fs_sampler_ids,
        fs_sampler_arrayed,
        vs_sampler_arrayed,
        depth_compare_2d_mask,
        depth_compare_cube_mask,
        depth_compare_cube_array_mask,
        cull_test_enable: draw.cull_test_enable && std::env::var_os("NEXIUM_NO_CULL").is_none(),
        cull_face: draw.cull_face,
        front_face,
        poly_offset_enable: draw.poly_offset_fill_enable,
        poly_offset_units: draw.poly_offset_units,
        poly_offset_factor: draw.poly_offset_factor,
        ssbo_data,
        present_flip_y,
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

pub(crate) fn msaa_samples(mode: u32) -> (u32, u32) {
    match mode {
        1 | 5 => (2, 1),
        2 | 8 | 9 => (2, 2),
        3 | 4 | 10 | 11 => (4, 2),
        6 => (4, 4),
        _ => (1, 1),
    }
}

fn render_target_volume_depth(rt: &RenderTarget) -> u32 {
    let defines_depth_size = ((rt.tile_mode >> 16) & 1) != 0;
    let depth = (rt.depth & 0xffff).max(1);
    if defines_depth_size && rt.base_layer == 0 && depth > 1 {
        depth
    } else {
        1
    }
}

fn rt_key_for_target(
    rt: &RenderTarget,
    mappings: &GpuMappings,
    samples: (u32, u32),
) -> Option<RtKey> {
    if rt.width == 0 || rt.height == 0 {
        return None;
    }
    let gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings.nvmap_id_for(gpu_va)?;
    Some(
        RtKey::with_cpu(
            nvmap_id,
            (rt.width / samples.0).max(1),
            (rt.height / samples.1).max(1),
            gpu_va,
            mappings.cpu_address_for(gpu_va).unwrap_or(0),
        )
        .with_volume_depth(render_target_volume_depth(rt)),
    )
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
    let (msx, msy) = msaa_samples(draw.multisample_mode);
    let (width, height) = if draw.zeta.width != 0 && draw.zeta.height != 0 {
        (
            (draw.zeta.width / msx).max(1),
            (draw.zeta.height / msy).max(1),
        )
    } else {
        let clip = draw.surface_clip.effective(fallback.width, fallback.height);
        (clip.width, clip.height)
    };
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
                rt_key_for_target(rt, mappings, msaa_samples(draw.multisample_mode))
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

fn fragment_output_numeric_masks(color_rts: &[(usize, RtKey, vk::Format)]) -> (u32, u32) {
    let mut uint_mask = 0u32;
    let mut sint_mask = 0u32;
    for (location, _, format) in color_rts {
        if *location >= 32 {
            continue;
        }
        let bit = 1u32 << *location;
        if vk_format_is_uint(*format) {
            uint_mask |= bit;
        } else if vk_format_is_sint(*format) {
            sint_mask |= bit;
        }
    }
    (uint_mask, sint_mask)
}

fn vk_format_is_uint(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_UINT
            | vk::Format::R8G8_UINT
            | vk::Format::R8G8B8_UINT
            | vk::Format::R8G8B8A8_UINT
            | vk::Format::R16_UINT
            | vk::Format::R16G16_UINT
            | vk::Format::R16G16B16_UINT
            | vk::Format::R16G16B16A16_UINT
            | vk::Format::R32_UINT
            | vk::Format::R32G32_UINT
            | vk::Format::R32G32B32_UINT
            | vk::Format::R32G32B32A32_UINT
            | vk::Format::A2B10G10R10_UINT_PACK32
            | vk::Format::A8B8G8R8_UINT_PACK32
    )
}

fn vk_format_is_sint(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SINT
            | vk::Format::R8G8_SINT
            | vk::Format::R8G8B8_SINT
            | vk::Format::R8G8B8A8_SINT
            | vk::Format::R16_SINT
            | vk::Format::R16G16_SINT
            | vk::Format::R16G16B16_SINT
            | vk::Format::R16G16B16A16_SINT
            | vk::Format::R32_SINT
            | vk::Format::R32G32_SINT
            | vk::Format::R32G32B32_SINT
            | vk::Format::R32G32B32A32_SINT
            | vk::Format::A8B8G8R8_SINT_PACK32
    )
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
    map_surface_format(format)
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
    let offsets = [
        0x0usize, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38,
        0x3c, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x58, 0x5c,
    ];
    let mut out = Vec::new();
    for off in offsets {
        let Some(raw) = packed_cbuf_word(data, 3, off) else {
            continue;
        };
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
        tic.depth.max(1)
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
    let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot as u32);
    let mut vals = Vec::new();
    for off in offsets {
        let Some(raw) = packed_cbuf_word(data, logical_slot, off as usize) else {
            vals.push(format!("{:#x}=out", off));
            continue;
        };
        vals.push(format!(
            "{:#x}={:.6}/{:#010x}",
            off,
            f32::from_bits(raw),
            raw
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

fn maxwell_draw_orientation(window_origin: WindowOrigin, front_face: u32) -> (u32, bool) {
    let front_face = if window_origin.triangle_rast_flip() {
        flip_front_face(front_face)
    } else {
        front_face
    };

    (front_face, false)
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

fn effective_depth_states(
    suppressed: bool,
    zeta_enabled: bool,
    depth_test_enabled: bool,
    depth_write_enabled: bool,
    stencil_enabled: bool,
) -> (bool, bool, bool) {
    let depth_test = !suppressed && zeta_enabled && depth_test_enabled;
    let depth_write = depth_test && depth_write_enabled;
    let stencil_test = !suppressed && zeta_enabled && stencil_enabled;
    (depth_test, depth_write, stencil_test)
}

fn map_zeta_format(v: u32) -> (vk::Format, vk::ImageAspectFlags) {
    match v {
        0x14 => (
            vk::Format::D24_UNORM_S8_UINT,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        ),
        0x13 => (vk::Format::D16_UNORM, vk::ImageAspectFlags::DEPTH),
        _ => (vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH),
    }
}

fn map_stencil_op(v: u32) -> vk::StencilOp {
    match v {
        1 | 0x1e00 => vk::StencilOp::KEEP,
        2 | 0 => vk::StencilOp::ZERO,
        3 | 0x1e01 => vk::StencilOp::REPLACE,
        4 | 0x1e02 => vk::StencilOp::INCREMENT_AND_CLAMP,
        5 | 0x1e03 => vk::StencilOp::DECREMENT_AND_CLAMP,
        6 | 0x150a => vk::StencilOp::INVERT,
        7 | 0x8507 => vk::StencilOp::INCREMENT_AND_WRAP,
        8 | 0x8508 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => {
            log::warn!(
                "map_stencil_op: unknown encoding {:#x}, defaulting to KEEP",
                v
            );
            vk::StencilOp::KEEP
        }
    }
}

fn map_stencil_face(face: super::engines::maxwell3d::StencilFaceState) -> GpuStencilFaceState {
    GpuStencilFaceState {
        fail_op: map_stencil_op(face.fail_op),
        pass_op: map_stencil_op(face.depth_pass_op),
        depth_fail_op: map_stencil_op(face.depth_fail_op),
        compare_op: map_compare_op(face.compare_op),
        compare_mask: face.compare_mask,
        write_mask: face.write_mask,
        reference: face.reference,
    }
}

#[derive(Clone)]
struct DrawTraceConfig {
    enabled: bool,
    op_start: u64,
    op_end: u64,
    start: u64,
    end: u64,
    width: Option<u32>,
    height: Option<u32>,
    fs: Vec<u64>,
    rt: Vec<u32>,
    rt_va: Vec<u64>,
    sampled_rt_only: bool,
    min_v: u32,
}

fn draw_trace_config() -> DrawTraceConfig {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<DrawTraceConfig> = OnceLock::new();
    CONFIG
        .get_or_init(|| DrawTraceConfig {
            enabled: std::env::var_os("NEXIUM_DRAW_TRACE").is_some(),
            op_start: std::env::var("NEXIUM_DRAW_TRACE_OP_START")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            op_end: std::env::var("NEXIUM_DRAW_TRACE_OP_END")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(u64::MAX),
            start: std::env::var("NEXIUM_DRAW_TRACE_START")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            end: std::env::var("NEXIUM_DRAW_TRACE_END")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(u64::MAX),
            width: std::env::var("NEXIUM_DRAW_TRACE_WIDTH")
                .ok()
                .and_then(|v| v.parse().ok()),
            height: std::env::var("NEXIUM_DRAW_TRACE_HEIGHT")
                .ok()
                .and_then(|v| v.parse().ok()),
            fs: parse_env_u64_list("NEXIUM_DRAW_TRACE_FS"),
            rt: parse_env_u64_list("NEXIUM_DRAW_TRACE_RT")
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            rt_va: parse_env_u64_list("NEXIUM_DRAW_TRACE_RT_VA"),
            sampled_rt_only: std::env::var_os("NEXIUM_DRAW_TRACE_SAMPLED_RT").is_some(),
            min_v: std::env::var("NEXIUM_DRAW_TRACE_MIN_V")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
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

fn shader_dump_path(filename: &str) -> std::path::PathBuf {
    let base = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".config/NeXium/dump"))
        .filter(|p| std::fs::create_dir_all(p).is_ok())
        .unwrap_or_else(std::env::temp_dir);
    base.join(filename)
}

fn env_is_all(name: &str) -> bool {
    std::env::var(name).map_or(false, |v| v.trim().eq_ignore_ascii_case("all"))
}

fn env_dump_shader(name: &str, addr: u64) -> bool {
    env_is_all(name)
        || env_is_all("NEXIUM_DUMP_SHADERS")
        || parse_env_u64_list(name).contains(&addr)
}

fn dump_geometry_shader_once(
    addr: u64,
    program_region: u64,
    program: &super::engines::maxwell3d::ShaderProgram,
    draw: &DrawCall,
    maxwell: &Maxwell3D,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static DUMPED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let dumped = DUMPED.get_or_init(|| Mutex::new(HashSet::new()));
    if dumped.lock().unwrap().contains(&addr) {
        return;
    }

    let Some(sph) = fetch_sph(addr, mappings, mem_read) else {
        sass_read_diag("GS-SPH", addr, program_region, program.address_lo, mappings);
        return;
    };
    let Some(sass) = fetch_sass(addr, mappings, mem_read) else {
        sass_read_diag("GS", addr, program_region, program.address_lo, mappings);
        return;
    };

    let mut dump = format_shader_dump(&sass, Some(&sph), None, 0);
    dump.push_str(&format_geometry_shader_metadata(
        addr,
        program_region,
        program,
        draw,
        maxwell,
        &sph,
        &sass,
    ));
    let path = shader_dump_path(&format!("target_gs_{:x}.txt", addr));
    match std::fs::write(&path, dump) {
        Ok(()) => {
            dumped.lock().unwrap().insert(addr);
            log::warn!("[dump-gs] wrote {}", path.display());
        }
        Err(err) => log::warn!("[dump-gs] failed to write {}: {}", path.display(), err),
    }
}

fn format_geometry_shader_metadata(
    addr: u64,
    program_region: u64,
    program: &super::engines::maxwell3d::ShaderProgram,
    draw: &DrawCall,
    maxwell: &Maxwell3D,
    sph: &[u8; SPH_SIZE],
    sass: &[u8],
) -> String {
    let word = |offset: usize| {
        u32::from_le_bytes([
            sph[offset],
            sph[offset + 1],
            sph[offset + 2],
            sph[offset + 3],
        ])
    };
    let common0 = word(0);
    let common2 = word(8);
    let common3 = word(12);
    let common4 = word(16);
    let rt_slot = draw_color_rt_slot(draw);
    let rt = &draw.rt[rt_slot];
    let post_vtg_masks = (0..8)
        .map(|index| {
            maxwell
                .reg_file
                .get(0x490 + index)
                .copied()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let cbufs = maxwell.regs.cbuf_binds[3]
        .iter()
        .enumerate()
        .filter(|(_, (gpu_va, size))| *gpu_va != 0 && *size != 0)
        .map(|(slot, (gpu_va, size))| format!("{}={:#x}+{:#x}", slot, gpu_va, size))
        .collect::<Vec<_>>();

    let mut attr_ops = Vec::new();
    let mut generic_loads = Vec::new();
    let mut layer_writes = Vec::new();
    let mut out_ops = Vec::new();
    for instruction in nexium_shader::walk_instructions(sass) {
        let raw = instruction.decoded.raw;
        match instruction.decoded.opcode {
            nexium_shader::Opcode::ALD | nexium_shader::Opcode::AST => {
                let slot = ((raw >> 20) & 0x3ff) as u32;
                let reg = (raw & 0xff) as u8;
                let index_reg = ((raw >> 8) & 0xff) as u8;
                let vertex_reg = ((raw >> 39) & 0xff) as u8;
                let elements = ((raw >> 47) & 0x3) as u32 + 1;
                let op = instruction.decoded.opcode;
                attr_ops.push(format!(
                    "+{:04x} {:?} reg=R{} slot={:#x} elems={} index=R{} vertex=R{}",
                    instruction.byte_offset, op, reg, slot, elements, index_reg, vertex_reg
                ));
                if op == nexium_shader::Opcode::ALD && (0x80..0x100).contains(&slot) {
                    for element in 0..elements {
                        generic_loads.push((
                            instruction.byte_offset,
                            reg.wrapping_add(element as u8),
                            slot + element * 4,
                        ));
                    }
                } else if op == nexium_shader::Opcode::AST
                    && slot <= 0x64
                    && slot + elements * 4 > 0x64
                {
                    layer_writes.push((
                        instruction.byte_offset,
                        reg.wrapping_add(((0x64 - slot) / 4) as u8),
                    ));
                }
            }
            nexium_shader::Opcode::OUT_reg
            | nexium_shader::Opcode::OUT_cbuf
            | nexium_shader::Opcode::OUT_imm => {
                let emit = (raw >> 39) & 1 != 0;
                let cut = (raw >> 40) & 1 != 0;
                out_ops.push(format!(
                    "+{:04x} {:?} emit={} cut={}",
                    instruction.byte_offset, instruction.decoded.opcode, emit, cut
                ));
            }
            _ => {}
        }
    }
    let layer_routes = layer_writes
        .iter()
        .map(|(store_offset, reg)| {
            let source = generic_loads
                .iter()
                .rev()
                .find(|(load_offset, load_reg, _)| load_offset < store_offset && load_reg == reg)
                .map(|(load_offset, _, slot)| format!("a[{:#x}]@+{:04x}", slot, load_offset))
                .unwrap_or_else(|| "unresolved".to_string());
            format!("{} -> a[0x64] via R{}@+{:04x}", source, reg, store_offset)
        })
        .collect::<Vec<_>>();

    format!(
        "\ngs_metadata:\n\
         addr={:#x} region={:#x} offset={:#x} enabled={} gprs={}\n\
         sph_type={} version={} shader_type={} geometry_passthrough={} invocations={} output_topology={} max_output_vertices={}\n\
         imap_systemb={:#04x} imap_layer={} omap_systemb={:#04x} omap_layer={}\n\
         imap_generics={}\n\
         omap_generics={}\n\
         draw_topology={} instances={} first_instance={}\n\
         rt_slot={} rt={:#x}x{:#x} fmt={:#x} tile={:#x} block={}x{}x{} pitch_linear={} dim_control={} depth={} volume={} array_pitch={:#x} base_layer={}\n\
         post_vtg_attrib_skip_mask={}\n\
         gs_cbufs={}\n\
         attr_ops={}\n\
         out_ops={}\n\
         layer_route_candidates={}\n",
        addr,
        program_region,
        program.address_lo,
        program.enabled,
        program.gpr_count,
        common0 & 0x1f,
        (common0 >> 5) & 0x1f,
        (common0 >> 10) & 0xf,
        (common0 >> 24) & 1,
        common2 >> 24,
        (common3 >> 24) & 0xf,
        common4 & 0xfff,
        sph[0x17],
        (sph[0x17] >> 1) & 1,
        sph[0x35],
        (sph[0x35] >> 1) & 1,
        sph[0x18..0x28]
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<Vec<_>>()
            .join(" "),
        sph[0x36..0x46]
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<Vec<_>>()
            .join(" "),
        draw.topology,
        draw.instance_count,
        draw.first_instance,
        rt_slot,
        rt.width,
        rt.height,
        rt.format,
        rt.tile_mode,
        rt.tile_mode & 0xf,
        (rt.tile_mode >> 4) & 0xf,
        (rt.tile_mode >> 8) & 0xf,
        (rt.tile_mode >> 12) & 1,
        (rt.tile_mode >> 16) & 1,
        rt.depth & 0xffff,
        (rt.depth >> 16) & 1,
        rt.layer_stride,
        rt.base_layer,
        post_vtg_masks
            .iter()
            .map(|mask| format!("{:08x}", mask))
            .collect::<Vec<_>>()
            .join(" "),
        if cbufs.is_empty() {
            "none".to_string()
        } else {
            cbufs.join(" ")
        },
        if attr_ops.is_empty() {
            "none".to_string()
        } else {
            attr_ops.join(" | ")
        },
        if out_ops.is_empty() {
            "none".to_string()
        } else {
            out_ops.join(" | ")
        },
        if layer_routes.is_empty() {
            "none".to_string()
        } else {
            layer_routes.join(" | ")
        },
    )
}

fn format_shader_dump(
    sass: &[u8],
    sph: Option<&[u8; SPH_SIZE]>,
    input_map: Option<[u8; 32]>,
    output_map: u32,
) -> String {
    let dis = nexium_shader::disassemble(sass)
        .into_iter()
        .map(|line| line.to_string_compact())
        .collect::<Vec<_>>()
        .join("\n");
    let raw_hex = sass
        .chunks(8)
        .enumerate()
        .map(|(i, w)| {
            let mut q = [0u8; 8];
            q[..w.len()].copy_from_slice(w);
            format!("{:03x}: {:016x}", i * 8, u64::from_le_bytes(q))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = format!("{}\n\nraw:\n{}\n", dis, raw_hex);
    if let Some(sph) = sph {
        out.push_str("\nsph:\n");
        for (i, chunk) in sph.chunks(16).enumerate() {
            let hex = chunk
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("{:02x}: {}\n", i * 16, hex));
        }
    }
    if let Some(imap) = input_map {
        out.push_str(&format!("\nps_input_map (generic 0..31): {:02x?}\n", imap));
        out.push_str(&format!("ps_output_map: {:#010x}\n", output_map));
    }
    out
}

fn trace_water_forensics(
    draw: &DrawCall,
    fallback_key: RtKey,
    color_rt_keys: &[RtKey],
    vs_addr: u64,
    fs_addr: u64,
    vs_cbuf_mask: u32,
    fs_cbuf_mask: u32,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    blend: (bool, u32, u32),
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    if std::env::var_os("NEXIUM_WATER_FORENSICS").is_none() {
        return;
    }
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    static PREV: OnceLock<Mutex<HashMap<(u64, u32), Vec<u8>>>> = OnceLock::new();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let key = color_rt_keys.first().copied().unwrap_or(fallback_key);
    let fnv = |bytes: &[u8]| -> u64 {
        bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ (*b as u64)).wrapping_mul(0x100000001b3)
        })
    };
    let vb = &draw.vertex_buffers[0];
    let vb_addr = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
    let mut vb_bytes: Option<Vec<u8>> = None;
    let vb_hash = if vb_addr != 0 {
        mappings
            .cpu_address_for(vb_addr)
            .and_then(|cpu| {
                let mut b = vec![0u8; 512];
                mem_read(cpu, &mut b).then(|| {
                    let h = fnv(&b);
                    vb_bytes = Some(b);
                    h
                })
            })
            .unwrap_or(0)
    } else {
        0
    };
    if blend.0 && vb.stride == 40 {
        if let Some(cpu) = mappings.cpu_address_for(vb_addr) {
            let count = draw.index_count.max(draw.vertex_count).min(4096) as usize;
            let mut buf = vec![0u8; (count * 40).min(163840)];
            if mem_read(cpu, &mut buf) {
                let (mut gmin, mut gmax, mut neg) = (f32::MAX, f32::MIN, 0usize);
                for v in buf.chunks_exact(40) {
                    let g = f32::from_le_bytes([v[36], v[37], v[38], v[39]]);
                    if g.is_finite() {
                        gmin = gmin.min(g);
                        gmax = gmax.max(g);
                        if g < 0.0 {
                            neg += 1;
                        }
                    }
                }
                let attrs: Vec<String> = draw
                    .vertex_attribs
                    .iter()
                    .take(4)
                    .enumerate()
                    .map(|(i, a)| {
                        format!(
                            "L{}:b{},o{},f{:#x},c{}",
                            i, a.buffer, a.offset, a.format, a.constant as u8
                        )
                    })
                    .collect();
                log::warn!(
                    "[wf-gate] d={} fs={:#x} verts={} gate=[{:.2}..{:.2}] neg={} ifirst={} attrs={}",
                    SEQ.load(Ordering::Relaxed),
                    fs_addr,
                    buf.len() / 40,
                    gmin,
                    gmax,
                    neg,
                    draw.index_first,
                    attrs.join(" ")
                );
            }
        }
    }
    if blend.0 {
        let floats: Vec<String> = vb_bytes
            .as_deref()
            .unwrap_or(&[])
            .chunks_exact(4)
            .take(24)
            .map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if f.abs() < 1e6 && (f == 0.0 || f.abs() > 1e-6) {
                    format!("{:.2}", f)
                } else {
                    format!("{:x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                }
            })
            .collect();
        log::warn!(
            "[wf-blend] d={} fs={:#x} ic={} stride={} vp={:?} sc={:?} vb0={:#x} f=[{}]",
            SEQ.load(Ordering::Relaxed),
            fs_addr,
            draw.index_count,
            vb.stride,
            draw.viewport,
            draw.scissor,
            vb_addr,
            floats.join(",")
        );
    }
    let mut parts: Vec<String> = Vec::new();
    let mask = (vs_cbuf_mask | fs_cbuf_mask) as u64;
    for logical in 0..32u32 {
        if mask & (1u64 << logical) == 0 {
            continue;
        }
        let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical);
        if addr == 0 || size == 0 {
            continue;
        }
        let len = (size as usize).min(384);
        let Some(cpu) = mappings.cpu_address_for(addr) else {
            continue;
        };
        let mut bytes = vec![0u8; len];
        if !mem_read(cpu, &mut bytes) {
            continue;
        }
        let h = fnv(&bytes);
        let mut prev_map = PREV
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        let slot_key = (fs_addr, logical);
        let mut delta = String::new();
        match prev_map.get(&slot_key) {
            Some(old) if old.as_slice() != bytes.as_slice() => {
                let n = old.len().min(bytes.len()) / 4;
                let mut changed: Vec<(usize, u32, u32)> = Vec::new();
                for w in 0..n {
                    let ow = u32::from_le_bytes([
                        old[w * 4],
                        old[w * 4 + 1],
                        old[w * 4 + 2],
                        old[w * 4 + 3],
                    ]);
                    let nw = u32::from_le_bytes([
                        bytes[w * 4],
                        bytes[w * 4 + 1],
                        bytes[w * 4 + 2],
                        bytes[w * 4 + 3],
                    ]);
                    if ow != nw {
                        changed.push((w, ow, nw));
                    }
                }
                let shown: Vec<String> = changed
                    .iter()
                    .take(8)
                    .map(|(w, o, v)| format!("{}:{:x}->{:x}", w, o, v))
                    .collect();
                delta = format!("~{}[{}]", changed.len(), shown.join(","));
            }
            None => {
                let words: Vec<String> = bytes
                    .chunks_exact(4)
                    .take(96)
                    .map(|c| format!("{:x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                log::warn!(
                    "[wf-full] fs={:#x} c{} va={:#x} sz={} words={}",
                    fs_addr,
                    logical,
                    addr,
                    size,
                    words.join(",")
                );
            }
            _ => {}
        }
        prev_map.insert(slot_key, bytes);
        parts.push(format!("c{}={:08x}{}", logical, h & 0xffffffff, delta));
    }
    let z0 = vb_bytes
        .as_deref()
        .filter(|b| b.len() >= 12)
        .map(|b| f32::from_le_bytes([b[8], b[9], b[10], b[11]]))
        .unwrap_or(f32::NAN);
    let bbox = vb_bytes
        .as_deref()
        .filter(|_| vb.stride >= 12 && vb.stride <= 64)
        .map(|b| {
            let s = vb.stride as usize;
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for v in b.chunks_exact(s) {
                let x = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                let y = f32::from_le_bytes([v[4], v[5], v[6], v[7]]);
                if x.is_finite() && y.is_finite() && x.abs() < 1e5 && y.abs() < 1e5 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
            format!("[{:.0},{:.0}..{:.0},{:.0}]", x0, y0, x1, y1)
        })
        .unwrap_or_default();
    log::warn!(
        "[wf] d={} vs={:#x} fs={:#x} rt={} v={} ic={} z0={:.3} bb={} bl={}:{}/{} z={}{}f{} vh={:08x} {}",
        seq,
        vs_addr,
        fs_addr,
        key.label(),
        draw.vertex_count,
        draw.index_count,
        z0,
        bbox,
        blend.0 as u8,
        blend.1,
        blend.2,
        draw.depth_test_enable as u8,
        draw.depth_write_enable as u8,
        draw.depth_func,
        vb_hash & 0xffffffff,
        parts.join(" ")
    );
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
    graphics_cbuf_reads: &[CbufRead],
) {
    if std::env::var_os("NEXIUM_GRADE_DISCOVER").is_none() {
        return;
    }
    let keys = if color_rt_keys.is_empty() {
        std::slice::from_ref(&fallback_key)
    } else {
        color_rt_keys
    };
    let tiny = keys.iter().any(|key| {
        key.height <= 64 || key.width <= 128 || key.width.saturating_mul(key.height) <= 0x4000
    });
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
    let reads = graphics_cbuf_reads
        .iter()
        .filter_map(|read| {
            (read.logical_slot == 19).then(|| {
                let origin = cbuf_read_origin_label(*read);
                match read.effective_byte_offset() {
                    Some(effective) => format!(
                        "{} base_offset={:#x} effective={:#x}",
                        origin, read.byte_offset, effective
                    ),
                    None => format!(
                        "{} base_offset={:#x} effective=dynamic",
                        origin, read.byte_offset
                    ),
                }
            })
        })
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
    vs_addr: u64,
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
    graphics_cbuf_reads: &[CbufRead],
) {
    let op_seq = next_gpu_op_seq();
    {
        use std::sync::{Mutex, OnceLock};
        static SPEC: OnceLock<Option<(u32, u32)>> = OnceLock::new();
        static SEEN: OnceLock<Mutex<std::collections::HashSet<(u64, u64)>>> = OnceLock::new();
        let spec = SPEC.get_or_init(|| {
            std::env::var("NEXIUM_DUMP_FS_RT").ok().and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
            })
        });
        if let Some((w, h)) = spec {
            if rt.width == *w && rt.height == *h {
                let fs_va = draw.fs_shader_gpu_va;
                let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                if seen.lock().unwrap().insert((vs_addr, fs_va)) {
                    for (label, addr) in [("vs", vs_addr), ("fs", fs_va)] {
                        if let Some(sass) = fetch_sass(addr, mappings, mem_read) {
                            let dis = nexium_shader::disassemble(&sass)
                                .into_iter()
                                .map(|line| line.to_string_compact())
                                .collect::<Vec<_>>()
                                .join("\n");
                            let path =
                                shader_dump_path(&format!("rtdump_{}_{:x}.txt", label, addr));
                            let _ = std::fs::write(&path, dis);
                        }
                    }
                    log::warn!(
                        "[dump-fs-rt] {}x{} vs={:#x} fs={:#x} v={} idx={}",
                        rt.width,
                        rt.height,
                        vs_addr,
                        fs_va,
                        vertex_count,
                        index_count
                    );
                }
            }
        }
    }
    let cfg = draw_trace_config();
    if !cfg.enabled {
        return;
    }
    if op_seq < cfg.op_start || op_seq > cfg.op_end {
        return;
    }
    if cfg.width.is_some_and(|width| width != rt.width)
        || cfg.height.is_some_and(|height| height != rt.height)
    {
        return;
    }
    if cfg.min_v != 0 && vertex_count.max(index_count) < cfg.min_v {
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
                    graphics_cbuf_reads,
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
    let binds = layout
        .bindings
        .iter()
        .map(|b| {
            format!(
                "b{}:s{}:d{}:{}",
                b.binding,
                b.stride,
                b.divisor,
                if b.divisor != 0 { "INST" } else { "VTX" }
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    log::warn!(
        "[drawtrace] op={} #{} rt={} va={:#x} keys=[{}] {}x{} topo={} first={} v={} i={} indexed={} pos={} \
         vp_en={} vp={:?} scale=({:.3},{:.3},{:.3}) trans=({:.3},{:.3},{:.3}) clip=({},{} {}x{}) scissor={}:({},{})->({},{}) origin={:#x} ll={} fy={} sw={:#x}/{} \
         depth={}/{} clamp={} vclip={:#x}/{} func={:#x} zeta={} cull={} ff={:#x} \
         tex={:?} tics=[{}] sampled={:?} \
         blend={} per={} rgb=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         a=({:#x},{:#x},{:#x})->({:?},{:?},{:?}) \
         vb={:#x} attrs={} {} cbuf={:#x}/{} masks={:#x}/{:#x} {} vs={:#x} fs={:#x} inst={}/{} binds=[{}]",
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
        draw.scissor.enabled,
        draw.scissor.min_x,
        draw.scissor.min_y,
        draw.scissor.max_x,
        draw.scissor.max_y,
        draw.window_origin.raw,
        draw.window_origin.lower_left(),
        draw.window_origin.triangle_rast_flip(),
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
        vs_addr,
        draw.fs_shader_gpu_va,
        draw.instance_count,
        draw.first_instance,
        binds,
    );
    let attachments = color_rt_keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let location = color_rt_locations.get(index).copied().unwrap_or(index);
            let slot = rt_control_target(draw.rt_control, location);
            let mask_index = if color_mask_common {
                0
            } else {
                location.min(7)
            };
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
    do_stencil: bool,
) {
    if !clear_trace_enabled() {
        return;
    }
    use std::sync::OnceLock;
    static TARGETS: OnceLock<Vec<u64>> = OnceLock::new();
    let targets = TARGETS.get_or_init(|| parse_env_u64_list("NEXIUM_CLEAR_TRACE_RT_VA"));
    if !targets.is_empty() && !targets.contains(&rt_gpu_va) {
        return;
    }
    let clip = draw.surface_clip.effective(rt.width, rt.height);
    log::warn!(
        "[cleartrace] op={} rt={} va={:#x} {}x{} mask={:#x} color={} depth={} stencil={} clear_depth={:.6} clear_stencil={:#x} \
         ctrl={:#x} scissor_en={} rect={:?} clip=({},{} {}x{}) origin={:#x} ll={} fy={} zeta={}",
        op_seq,
        nvmap_id,
        rt_gpu_va,
        rt.width,
        rt.height,
        draw.clear_mask,
        want_color_clear,
        do_depth,
        do_stencil,
        draw.clear_depth,
        draw.clear_stencil,
        draw.clear_control,
        draw.scissor.enabled,
        clear_scissor,
        clip.x,
        clip.y,
        clip.width,
        clip.height,
        draw.window_origin.raw,
        draw.window_origin.lower_left(),
        draw.window_origin.triangle_rast_flip(),
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

fn cbuf_read_origin_label(read: CbufRead) -> String {
    match read.index_origin {
        CbufIndexOrigin::Static => "static".to_string(),
        CbufIndexOrigin::Constant(index) => format!("indexed:constant({index:#010x})"),
        CbufIndexOrigin::Gpr(register) => format!("indexed:gpr(R{register})"),
        CbufIndexOrigin::Instruction(value) => format!("indexed:value(v{value})"),
    }
}

fn format_cbuf_read(
    data: &[u8],
    read: CbufRead,
    cbuf_binds: &[[(u64, u32); 16]; 5],
) -> String {
    let logical_slot = u32::from(read.logical_slot);
    let (base, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot);
    let origin = cbuf_read_origin_label(read);
    let Some(effective) = read.effective_byte_offset() else {
        return format!(
            "slot={} base={:#x} size={} origin={} effective=dynamic word=dynamic in_range=unknown",
            logical_slot, base, size, origin
        );
    };
    let word_offset = effective & !3;
    let raw = packed_cbuf_word(data, logical_slot as usize, word_offset as usize);
    match raw {
        Some(raw) => format!(
            "slot={} base={:#x} size={} origin={} effective={:#010x} word={:#010x} in_range=true value={:.3}/{:#010x}",
            logical_slot,
            base,
            size,
            origin,
            effective,
            word_offset,
            f32::from_bits(raw),
            raw
        ),
        None => format!(
            "slot={} base={:#x} size={} origin={} effective={:#010x} word={:#010x} in_range=false",
            logical_slot, base, size, origin, effective, word_offset
        ),
    }
}

fn cbuf_sample(
    cbuf_data: Option<&[u8]>,
    used_mask: u32,
    cbuf_reads: &[CbufRead],
    cbuf_binds: &[[(u64, u32); 16]; 5],
) -> String {
    let Some(data) = cbuf_data else {
        return "cbuf_sample=none".to_string();
    };
    let full = std::env::var_os("NEXIUM_DRAW_TRACE_CBUF_FULL").is_some();
    if !cbuf_reads.is_empty() {
        let max_reads = if full {
            cbuf_reads.len()
        } else {
            std::env::var("NEXIUM_DRAW_TRACE_CBUF_READS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(12)
                .clamp(1, 64)
        };
        let vals = cbuf_reads
            .iter()
            .take(max_reads)
            .copied()
            .map(|read| format_cbuf_read(data, read, cbuf_binds))
            .collect::<Vec<_>>();
        let shown = vals.len();
        return format!(
            "cbuf_reads shown={}/{} truncated={} [{}]",
            shown,
            cbuf_reads.len(),
            shown < cbuf_reads.len(),
            vals.join(" | ")
        );
    }
    let mut slots = Vec::new();
    for logical_slot in 0..PACKED_CBUF_SLOTS {
        if (used_mask & (1u32 << logical_slot)) == 0 {
            continue;
        }
        let Some(slot_data) = packed_cbuf_slot(data, logical_slot) else {
            continue;
        };
        let (addr, size) = cbuf_bind_for_slot(cbuf_binds, logical_slot as u32);
        let dump_bytes = if full {
            let max = std::env::var("NEXIUM_DRAW_TRACE_CBUF_MAX")
                .ok()
                .and_then(|v| {
                    let v = v.trim();
                    usize::from_str_radix(v.trim_start_matches("0x"), 16)
                        .ok()
                        .or_else(|| v.parse().ok())
                })
                .unwrap_or(256);
            slot_data.len().min(max)
        } else {
            slot_data.len().min(16)
        };
        let mut vals = Vec::new();
        for (i, c) in slot_data[..dump_bytes].chunks_exact(4).enumerate() {
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
        if !full && slots.len() >= 4 {
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
                        .min(nexium_spirv::GFX_CBUF_MAX_SIZE as u64)
                        as usize;
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
    graphics_cbuf_reads: &[CbufRead],
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
        let Some(slot_data) = packed_cbuf_slot(data, region.slot) else {
            continue;
        };
        let (addr, size) = cbuf_bind_for_slot(cbuf_binds, region.slot as u32);
        let bind_len = size as usize;
        if bind_len <= region.offset || slot_data.len() <= region.offset {
            continue;
        }
        let len = region
            .len
            .min(slot_data.len() - region.offset)
            .min(bind_len - region.offset);
        if len == 0 {
            continue;
        }
        let bytes = slot_data[region.offset..region.offset + len].to_vec();
        if !bind_filter.is_empty() && !bind_filter.contains(&addr) {
            continue;
        }
        let seen_key = (fs_addr, region.slot, region.offset, len, addr);
        let value_key = (region.slot, region.offset, len, addr);
        let first = {
            let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
            seen.lock()
                .map(|mut seen| seen.insert(seen_key))
                .unwrap_or(false)
        };
        let changed = {
            let last = LAST.get_or_init(|| Mutex::new(HashMap::new()));
            match last.lock() {
                Ok(mut last) => {
                    let changed = last
                        .get(&value_key)
                        .map(|old| old != &bytes)
                        .unwrap_or(true);
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
        let read_hits = graphics_cbuf_reads
            .iter()
            .filter_map(|read| {
                if read.logical_slot as usize != region.slot {
                    return None;
                }
                let origin = cbuf_read_origin_label(*read);
                match read.effective_byte_offset().map(|offset| offset as usize) {
                    Some(effective)
                        if effective >= region.offset && effective < region.offset + len =>
                    {
                        Some(format!(
                            "{} base_offset={:#x} effective={:#x}",
                            origin, read.byte_offset, effective
                        ))
                    }
                    Some(_) => None,
                    None => Some(format!(
                        "{} base_offset={:#x} effective=dynamic",
                        origin, read.byte_offset
                    )),
                }
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

fn texture_cbuf_origin(
    shader_id: u32,
    fallback_cb_index: usize,
    fallback_trusted: bool,
) -> (usize, u32, Option<u32>, bool) {
    match nexium_shader::decode_bindless_texture_id(shader_id) {
        Some((binding, word_offset, secondary_word_offset)) => {
            (binding as usize, word_offset, secondary_word_offset, true)
        }
        None => (fallback_cb_index, shader_id, None, fallback_trusted),
    }
}

#[cfg(test)]
fn cfg_uses_texture_descriptors(cfg: &nexium_shader::Cfg) -> bool {
    cfg.blocks.iter().any(|block| {
        block.program.instructions.iter().any(|inst| {
            matches!(
                &inst.op,
                nexium_shader::IrOp::SampleTex { .. }
                    | nexium_shader::IrOp::TexelFetch { .. }
                    | nexium_shader::IrOp::GatherTex { .. }
            )
        })
    })
}

fn fragment_texture_numeric_metadata(
    cfg: &nexium_shader::Cfg,
) -> Result<FragmentTextureNumericMetadata, String> {
    let constant_facts = nexium_shader::IrConstantFacts::analyze(cfg);
    let mut texel_fetches = std::collections::BTreeMap::new();
    let mut buffer_candidates = std::collections::BTreeSet::new();
    let mut sampled_ids = std::collections::BTreeSet::new();
    let mut depth_compare_2d_ids = std::collections::BTreeSet::new();
    let mut depth_compare_cube_ids = std::collections::BTreeSet::new();
    let mut depth_compare_cube_array_ids = std::collections::BTreeSet::new();
    let mut image_kinds = std::collections::BTreeMap::new();
    let sampler_arrayed = cfg.blocks.iter().any(|block| {
        block.program.instructions.iter().any(|instruction| {
            matches!(
                &instruction.op,
                nexium_shader::IrOp::SampleTex {
                    array: Some(_),
                    volume: None,
                    cube: None,
                    ..
                }
            )
        })
    });
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            match &inst.op {
                nexium_shader::IrOp::TexelFetch {
                    cbuf_binding,
                    cbuf_word_offset,
                    cbuf_secondary_word_offset,
                    y,
                    z,
                    component,
                    ..
                } => {
                    let shader_id = nexium_shader::bindless_texture_id_pair(
                        *cbuf_binding,
                        *cbuf_word_offset,
                        *cbuf_secondary_word_offset,
                    );
                    *texel_fetches.entry(shader_id).or_insert(0u8) |= 1u8 << (*component).min(3);
                    if constant_facts
                        .texel_fetch_buffer_coordinates_compatible(y.as_ref(), z.as_ref())
                    {
                        buffer_candidates.insert(shader_id);
                    }
                    let image_kind = if z.is_some() {
                        GraphicsTextureImageKind::D3
                    } else if sampler_arrayed {
                        GraphicsTextureImageKind::D2Array
                    } else {
                        GraphicsTextureImageKind::D2
                    };
                    record_runtime_texture_image_kind(
                        &mut image_kinds,
                        shader_id,
                        image_kind,
                    )?;
                }
                nexium_shader::IrOp::SampleTex {
                    tex_id,
                    array,
                    volume,
                    cube,
                    dref,
                    ..
                } => {
                    sampled_ids.insert(*tex_id);
                    let image_kind = if cube.is_some() {
                        if array.is_some() {
                            GraphicsTextureImageKind::CubeArray
                        } else {
                            GraphicsTextureImageKind::Cube
                        }
                    } else if volume.is_some() {
                        GraphicsTextureImageKind::D3
                    } else if sampler_arrayed {
                        GraphicsTextureImageKind::D2Array
                    } else {
                        GraphicsTextureImageKind::D2
                    };
                    record_runtime_texture_image_kind(&mut image_kinds, *tex_id, image_kind)?;
                    if dref.is_some() && volume.is_none() {
                        if cube.is_some() && array.is_some() {
                            depth_compare_cube_array_ids.insert(*tex_id);
                        } else if cube.is_some() {
                            depth_compare_cube_ids.insert(*tex_id);
                        } else {
                            depth_compare_2d_ids.insert(*tex_id);
                        }
                    }
                }
                nexium_shader::IrOp::GatherTex { tex_id, .. } => {
                    sampled_ids.insert(*tex_id);
                    record_runtime_texture_image_kind(
                        &mut image_kinds,
                        *tex_id,
                        if sampler_arrayed {
                            GraphicsTextureImageKind::D2Array
                        } else {
                            GraphicsTextureImageKind::D2
                        },
                    )?;
                }
                _ => {}
            }
        }
    }
    let texture_ids = nexium_shader::texture_ids(cfg);
    Ok(FragmentTextureNumericMetadata {
        texel_fetches: (!texel_fetches.is_empty()).then(|| texel_fetches.into_iter().collect()),
        descriptor_ids: texture_ids.clone(),
        texture_ids,
        sampled_ids: sampled_ids.into_iter().collect(),
        buffer_candidates: buffer_candidates.into_iter().collect(),
        image_kinds: image_kinds.into_iter().collect(),
        sampler_arrayed,
        depth_compare_2d_ids: depth_compare_2d_ids.into_iter().collect(),
        depth_compare_cube_ids: depth_compare_cube_ids.into_iter().collect(),
        depth_compare_cube_array_ids: depth_compare_cube_array_ids.into_iter().collect(),
        or_partners: cfg.bindless_or_partners.clone(),
    })
}

fn record_runtime_texture_image_kind(
    image_kinds: &mut std::collections::BTreeMap<u32, GraphicsTextureImageKind>,
    shader_id: u32,
    image_kind: GraphicsTextureImageKind,
) -> Result<(), String> {
    if let Some(previous) = image_kinds.insert(shader_id, image_kind) {
        if previous != image_kind {
            return Err(format!(
                "graphics shader texture {shader_id:#x} is used with incompatible {previous:?} and {image_kind:?} image families"
            ));
        }
    }
    Ok(())
}

fn inferred_texture_image_kind(
    metadata: &FragmentTextureNumericMetadata,
    shader_id: u32,
) -> GraphicsTextureImageKind {
    metadata
        .image_kinds
        .binary_search_by_key(&shader_id, |(id, _)| *id)
        .ok()
        .map(|index| metadata.image_kinds[index].1)
        .unwrap_or(GraphicsTextureImageKind::D2)
}

fn texture_descriptor_mask(
    metadata: &FragmentTextureNumericMetadata,
    descriptor_slot_base: u32,
    shader_ids: &[u32],
) -> u32 {
    shader_ids.iter().fold(0u32, |mask, shader_id| {
        let Some(local_slot) = metadata
            .descriptor_ids
            .iter()
            .position(|candidate| candidate == shader_id)
        else {
            return mask;
        };
        let slot = descriptor_slot_base + local_slot as u32;
        if slot < u32::BITS {
            mask | (1u32 << slot)
        } else {
            mask
        }
    })
}

fn append_walked_texture_ids(
    metadata: &mut FragmentTextureNumericMetadata,
    walked: impl IntoIterator<Item = nexium_shader::FsTexId>,
) -> (usize, usize) {
    let before = metadata.descriptor_ids.len();
    let mut bindless = 0;
    for id in walked {
        match id {
            nexium_shader::FsTexId::ImmediateTic(id) => {
                if !metadata.descriptor_ids.contains(&id) {
                    metadata.descriptor_ids.push(id);
                }
            }
            nexium_shader::FsTexId::BindlessCbufOffset(_) => bindless += 1,
        }
    }
    (metadata.descriptor_ids.len() - before, bindless)
}

fn resolved_tic(
    tic_id: u32,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<(nexium_gpu::texture::TicEntry, u64)> {
    if tic_pool_gpu_va == 0 || tic_id == u32::MAX || tic_id > tic_pool_limit {
        return None;
    }
    let tic_addr = tic_pool_gpu_va.wrapping_add((tic_id as u64).saturating_mul(32));
    let cpu = mappings.cpu_address_for(tic_addr)?;
    let mut raw = [0u8; 32];
    if !mem_read(cpu, &mut raw) {
        return None;
    }
    Some((nexium_gpu::texture::TicEntry::parse(&raw)?, tic_addr))
}

#[derive(Clone, Copy, Debug)]
struct ResolvedTextureNumericUse {
    stage_name: &'static str,
    shader_id: u32,
    descriptor_slot: u32,
    tic_id: Option<u32>,
    numeric_type: nexium_spirv::TextureNumericType,
    image_kind: GraphicsTextureImageKind,
}

#[allow(clippy::too_many_arguments)]
fn stage_texture_numeric_bindings(
    stage_name: &'static str,
    metadata: &FragmentTextureNumericMetadata,
    descriptor_slot_base: u32,
    cbuf_binds: &[(u64, u32); 16],
    bindless_slot: u32,
    tex_cb_slot: u32,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<
    (
        Vec<TextureNumericBinding>,
        u32,
        Vec<ResolvedTextureNumericUse>,
    ),
    String,
> {
    let max_slots = nexium_gpu::descriptor::MAX_TEXTURE_DESCRIPTORS as usize;
    if descriptor_slot_base as usize + metadata.descriptor_ids.len() > max_slots {
        return Err(format!(
            "{stage_name} texture descriptors exceed the {max_slots}-slot graphics ABI: base={} count={}",
            descriptor_slot_base,
            metadata.descriptor_ids.len()
        ));
    }

    let shader_ids = &metadata.descriptor_ids;
    let mut tic_ids = shader_ids.clone();
    let mut sampler_ids = vec![0; tic_ids.len()];
    let mut remap_log = Vec::new();
    if tic_pool_gpu_va != 0 && !tic_ids.is_empty() {
        remap_texture_ids_for_stage(
            stage_name,
            cbuf_binds,
            bindless_slot,
            tex_cb_slot,
            &mut tic_ids,
            &mut sampler_ids,
            descriptor_slot_base as usize,
            &mut remap_log,
            tic_pool_gpu_va,
            tic_pool_limit,
            via_header_index,
            mappings,
            mem_read,
            &metadata.or_partners,
        );
    }

    let mut bindings = Vec::with_capacity(shader_ids.len());
    let mut uses = Vec::with_capacity(shader_ids.len());
    let mut mask = 0u32;
    for (local_slot, (&shader_id, &tic_id)) in shader_ids.iter().zip(&tic_ids).enumerate() {
        let descriptor_slot = descriptor_slot_base + local_slot as u32;
        let sampled = metadata.sampled_ids.binary_search(&shader_id).is_ok();
        let referenced_components = metadata.texel_fetches.as_ref().and_then(|fetches| {
            fetches
                .binary_search_by_key(&shader_id, |(id, _)| *id)
                .ok()
                .map(|index| fetches[index].1)
        });
        let resolved = resolved_tic(
            tic_id,
            tic_pool_gpu_va,
            tic_pool_limit,
            mappings,
            mem_read,
        );
        let numeric_type = referenced_components
            .and_then(|components| {
                resolved
                    .as_ref()
                    .map(|(tic, _)| tic_numeric_type(tic, components))
            })
            .unwrap_or(nexium_spirv::TextureNumericType::Float);
        let buffer_backed = referenced_components.is_some()
            && metadata
                .buffer_candidates
                .binary_search(&shader_id)
                .is_ok()
            && resolved
                .as_ref()
                .is_some_and(|(tic, _)| tic.is_buffer() && supported_texel_buffer(tic, numeric_type));
        if sampled && numeric_type != nexium_spirv::TextureNumericType::Float {
            let tic = resolved.as_ref().map(|(tic, _)| tic);
            return Err(format!(
                "{stage_name} texture numeric conflict: shader_id={shader_id:#x} slot={descriptor_slot} tic_id={tic_id:#x} is used by Sample/Gather (Float) and TexelFetch ({numeric_type:?}); tic_format={:?} component_types={:?}",
                tic.map(|tic| tic.format),
                tic.map(|tic| tic.component_types),
            ));
        }

        let inferred_image_kind = inferred_texture_image_kind(metadata, shader_id);
        if sampled && buffer_backed {
            return Err(format!(
                "{stage_name} texture image-kind conflict: shader_id={shader_id:#x} slot={descriptor_slot} tic_id={tic_id:#x} is used by Sample/Gather ({inferred_image_kind:?}) and TexelFetch (Buffer)"
            ));
        }
        let image_kind = if buffer_backed {
            GraphicsTextureImageKind::Buffer
        } else {
            inferred_image_kind
        };

        if image_kind == GraphicsTextureImageKind::Buffer {
            mask |= 1 << descriptor_slot;
        }

        bindings.push(TextureNumericBinding::new(
            shader_id,
            descriptor_slot,
            numeric_type,
        )
        .with_image_kind(image_kind));
        uses.push(ResolvedTextureNumericUse {
            stage_name,
            shader_id,
            descriptor_slot,
            tic_id: (tic_pool_gpu_va != 0
                && tic_id != u32::MAX
                && tic_id <= tic_pool_limit)
                .then_some(tic_id),
            numeric_type,
            image_kind,
        });
    }
    Ok((bindings, mask, uses))
}

#[allow(clippy::too_many_arguments)]
fn graphics_texture_layout_from_metadata(
    fs_metadata: &FragmentTextureNumericMetadata,
    vs_metadata: &FragmentTextureNumericMetadata,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    bindless_slot: u32,
    tex_cb_slot: u32,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<GraphicsTextureLayout, String> {
    let vs_tex_base = fs_metadata.descriptor_ids.len() as u32;
    let vs_tex_count = vs_metadata.descriptor_ids.len() as u32;
    if vs_tex_base.saturating_add(vs_tex_count)
        > nexium_gpu::descriptor::MAX_TEXTURE_DESCRIPTORS
    {
        return Err(format!(
            "graphics texture descriptors exceed the {}-slot ABI: fs={} vs={}",
            nexium_gpu::descriptor::MAX_TEXTURE_DESCRIPTORS,
            vs_tex_base,
            vs_tex_count
        ));
    }

    let (mut bindings, fs_texel_buffer_mask, mut uses) = stage_texture_numeric_bindings(
        "fs",
        fs_metadata,
        0,
        &cbuf_binds[4],
        bindless_slot,
        tex_cb_slot,
        tic_pool_gpu_va,
        tic_pool_limit,
        via_header_index,
        mappings,
        mem_read,
    )?;
    let (vs_bindings, vs_texel_buffer_mask, vs_uses) = stage_texture_numeric_bindings(
        "vs",
        vs_metadata,
        vs_tex_base,
        &cbuf_binds[0],
        bindless_slot,
        tex_cb_slot,
        tic_pool_gpu_va,
        tic_pool_limit,
        via_header_index,
        mappings,
        mem_read,
    )?;
    bindings.extend(vs_bindings);
    uses.extend(vs_uses);

    let mut resources = std::collections::HashMap::<u32, ResolvedTextureNumericUse>::new();
    for usage in uses {
        let Some(tic_id) = usage.tic_id else {
            continue;
        };
        if let Some(previous) = resources.insert(tic_id, usage) {
            if previous.numeric_type != usage.numeric_type {
                return Err(format!(
                    "graphics texture numeric conflict: TIC {tic_id:#x} is used by {} shader_id={:#x} slot={} as {:?} and {} shader_id={:#x} slot={} as {:?}",
                    previous.stage_name,
                    previous.shader_id,
                    previous.descriptor_slot,
                    previous.numeric_type,
                    usage.stage_name,
                    usage.shader_id,
                    usage.descriptor_slot,
                    usage.numeric_type,
                ));
            }
            if previous.image_kind != usage.image_kind {
                return Err(format!(
                    "graphics texture image-kind conflict: TIC {tic_id:#x} is used by {} shader_id={:#x} slot={} as {:?} and {} shader_id={:#x} slot={} as {:?}",
                    previous.stage_name,
                    previous.shader_id,
                    previous.descriptor_slot,
                    previous.image_kind,
                    usage.stage_name,
                    usage.shader_id,
                    usage.descriptor_slot,
                    usage.image_kind,
                ));
            }
        }
    }

    let manifest = normalize_texture_numeric_manifest(bindings)
        .map_err(|error| format!("invalid graphics texture numeric manifest: {error}"))?;
    let mut fs_ids = fs_metadata.descriptor_ids.clone();
    fs_ids.extend(vs_metadata.descriptor_ids.iter().copied());
    let depth_compare_2d_mask = texture_descriptor_mask(
        fs_metadata,
        0,
        &fs_metadata.depth_compare_2d_ids,
    ) | texture_descriptor_mask(
        vs_metadata,
        vs_tex_base,
        &vs_metadata.depth_compare_2d_ids,
    );
    let depth_compare_cube_mask = texture_descriptor_mask(
        fs_metadata,
        0,
        &fs_metadata.depth_compare_cube_ids,
    ) | texture_descriptor_mask(
        vs_metadata,
        vs_tex_base,
        &vs_metadata.depth_compare_cube_ids,
    );
    let depth_compare_cube_array_mask = texture_descriptor_mask(
        fs_metadata,
        0,
        &fs_metadata.depth_compare_cube_array_ids,
    ) | texture_descriptor_mask(
        vs_metadata,
        vs_tex_base,
        &vs_metadata.depth_compare_cube_array_ids,
    );
    Ok(GraphicsTextureLayout {
        fs_ids,
        vs_tex_base,
        vs_tex_count,
        manifest,
        fs_texel_buffer_mask,
        vs_texel_buffer_mask,
        fs_sampler_arrayed: fs_metadata.sampler_arrayed,
        vs_sampler_arrayed: vs_metadata.sampler_arrayed,
        depth_compare_2d_mask,
        depth_compare_cube_mask,
        depth_compare_cube_array_mask,
    })
}

fn supported_texel_buffer(
    tic: &nexium_gpu::texture::TicEntry,
    numeric_type: nexium_spirv::TextureNumericType,
) -> bool {
    nexium_gpu::renderer::texel_buffer_format(tic, numeric_type).is_some()
}

#[cfg(test)]
fn texel_buffer_numeric_type(
    tic: &nexium_gpu::texture::TicEntry,
) -> Option<nexium_spirv::TextureNumericType> {
    use nexium_spirv::TextureNumericType::{Float, Sint, Uint};

    let mut supported = [Float, Uint, Sint]
        .into_iter()
        .filter(|numeric_type| supported_texel_buffer(tic, *numeric_type));
    let numeric_type = supported.next()?;
    supported.next().is_none().then_some(numeric_type)
}

fn tic_numeric_type(
    tic: &nexium_gpu::texture::TicEntry,
    referenced_components: u8,
) -> nexium_spirv::TextureNumericType {
    use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
    use nexium_spirv::TextureNumericType;

    let is_unorm = |component_type| {
        matches!(
            component_type,
            ComponentType::Unorm | ComponentType::UnormForceFp16
        )
    };
    let any_r = tic.swizzle.contains(&SwizzleSource::R);
    match tic.format {
        TicFormat::G24R8
            if tic.component_types[0] == ComponentType::Uint
                && is_unorm(tic.component_types[1]) =>
        {
            return if any_r {
                TextureNumericType::Uint
            } else {
                TextureNumericType::Float
            };
        }
        TicFormat::Z24S8 | TicFormat::X8Z24 | TicFormat::S8Z24 | TicFormat::Z32 => {
            return TextureNumericType::Float;
        }
        _ => {}
    }

    let mut component_types = Vec::new();
    for output_component in 0..4 {
        if referenced_components & (1 << output_component) == 0 {
            continue;
        }
        let source_component = match tic.swizzle[output_component] {
            SwizzleSource::R => 0,
            SwizzleSource::G => 1,
            SwizzleSource::B => 2,
            SwizzleSource::A => 3,
            SwizzleSource::Zero | SwizzleSource::One => continue,
            SwizzleSource::Unknown(_) => return TextureNumericType::Float,
        };
        component_types.push(tic.component_types[source_component]);
    }
    consistent_tic_numeric_type(component_types)
}

fn consistent_tic_numeric_type(
    component_types: impl IntoIterator<Item = nexium_gpu::texture::ComponentType>,
) -> nexium_spirv::TextureNumericType {
    let mut resolved_type = None;
    for component_type in component_types {
        let current = match component_type {
            nexium_gpu::texture::ComponentType::Uint => nexium_spirv::TextureNumericType::Uint,
            nexium_gpu::texture::ComponentType::Sint => nexium_spirv::TextureNumericType::Sint,
            _ => return nexium_spirv::TextureNumericType::Float,
        };
        match resolved_type {
            Some(previous) if previous != current => {
                return nexium_spirv::TextureNumericType::Float;
            }
            None => resolved_type = Some(current),
            _ => {}
        }
    }
    resolved_type.unwrap_or(nexium_spirv::TextureNumericType::Float)
}

fn remap_texture_ids_for_stage(
    stage_name: &str,
    cbuf_binds: &[(u64, u32); 16],
    bindless_slot: u32,
    tex_cb_slot: u32,
    tex_ids: &mut [u32],
    sampler_ids: &mut [u32],
    slot_offset: usize,
    tex_remap: &mut Vec<String>,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    or_partners: &std::collections::HashMap<u32, u32>,
) {
    let designated = (tex_cb_slot as usize).min(15);
    let trusted = tex_cb_slot != 0 && cbuf_binds[designated].0 != 0 && cbuf_binds[designated].1 > 0;
    let tex_cb_index = if trusted {
        designated
    } else {
        choose_texture_cb_index(
            cbuf_binds,
            bindless_slot,
            tex_cb_slot,
            tex_ids,
            tic_pool_gpu_va,
            tic_pool_limit,
            via_header_index,
            stage_name == "fs",
            mappings,
            mem_read,
        )
    };
    let (tcb_addr, tcb_size) = cbuf_binds[tex_cb_index.min(15)];
    'texture_slot: for (local_i, unit_slot) in tex_ids.iter_mut().enumerate() {
        let i = slot_offset + local_i;
        let shader_id = *unit_slot;
        let exact_origin = nexium_shader::decode_bindless_texture_id(shader_id);
        let (entry_cb_index, shader_word_offset, secondary_word_offset, entry_trusted) =
            texture_cbuf_origin(shader_id, tex_cb_index, trusted);
        let (entry_tcb_addr, entry_tcb_size) = if exact_origin.is_some() {
            cbuf_binds[entry_cb_index.min(15)]
        } else {
            (tcb_addr, tcb_size)
        };
        let off = (shader_word_offset as u64).saturating_mul(4);
        let secondary_off = secondary_word_offset.map(|word| (word as u64).saturating_mul(4));
        let offsets = secondary_off.map_or_else(
            || format!("{off:#x}"),
            |secondary| format!("{off:#x}|{secondary:#x}"),
        );
        let mut remap = format!(
            "s{}:{} stage={} cb{} base={:#x}/{} off={}",
            i, shader_id, stage_name, entry_cb_index, entry_tcb_addr, entry_tcb_size, offsets
        );
        if entry_tcb_addr == 0 {
            remap.push_str(" no-tcb");
            tex_remap.push(remap);
            continue;
        }
        let mut handle = 0u32;
        for word_offset in [Some(shader_word_offset), secondary_word_offset]
            .into_iter()
            .flatten()
        {
            let word_off = (word_offset as u64).saturating_mul(4);
            if word_off + 4 > entry_tcb_size as u64 {
                remap.push_str(" out-of-range");
                tex_remap.push(remap);
                continue 'texture_slot;
            }
            let Some(cpu) = mappings.cpu_address_for(entry_tcb_addr.wrapping_add(word_off)) else {
                remap.push_str(" unmapped");
                tex_remap.push(remap);
                continue 'texture_slot;
            };
            let mut bytes = [0u8; 4];
            if !mem_read(cpu, &mut bytes) {
                remap.push_str(" read-fail");
                tex_remap.push(remap);
                continue 'texture_slot;
            }
            handle |= u32::from_le_bytes(bytes);
        }
        if let Some(&partner_word) = or_partners.get(&shader_id) {
            let poff = (partner_word as u64).saturating_mul(4);
            if poff + 4 <= entry_tcb_size as u64 {
                if let Some(pcpu) = mappings.cpu_address_for(entry_tcb_addr.wrapping_add(poff)) {
                    let mut pbytes = [0u8; 4];
                    if mem_read(pcpu, &mut pbytes) {
                        handle |= u32::from_le_bytes(pbytes);
                    }
                }
            }
        }
        let (tic, tsc) = split_texture_handle(handle, via_header_index);
        let accept = if entry_trusted {
            tic <= tic_pool_limit
        } else if handle != 0 {
            texture_tic_readable(tic, tic_pool_gpu_va, tic_pool_limit, mappings, mem_read)
        } else {
            texture_tic_readable(0, tic_pool_gpu_va, tic_pool_limit, mappings, mem_read)
                && !texture_tic_readable(
                    shader_word_offset,
                    tic_pool_gpu_va,
                    tic_pool_limit,
                    mappings,
                    mem_read,
                )
        };
        if accept {
            if std::env::var_os("NEXIUM_TEX_BIND_LOG").is_some() {
                log::warn!(
                    "tex_handle: stage={} cb{} off={} handle={:#x} -> TIC {} TSC {}",
                    stage_name,
                    entry_cb_index,
                    offsets,
                    handle,
                    tic,
                    tsc
                );
            }
            *unit_slot = tic;
            sampler_ids[local_i] = tsc;
            remap.push_str(&format!(" handle={:#x}->tic{} tsc{}", handle, tic, tsc));
        } else {
            remap.push_str(&format!(
                " handle={:#x} invalid tic{} tsc{}",
                handle, tic, tsc
            ));
        }
        tex_remap.push(remap);
    }
}

fn trace_vs_tex_remap(
    vs_addr: u64,
    fs_addr: u64,
    vs_tex_base: u32,
    vs_tex_count: u32,
    shader_tex_ids: &[u32],
    final_tex_ids: &[u32],
    sampler_ids: &[u32],
    tex_remap: &[String],
    bindless_slot: u32,
    tex_cb_slot: u32,
    via_header_index: bool,
    split_vs_stage: bool,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
) {
    if vs_tex_count == 0 || !vs_tex_bind_fs_trace(fs_addr) {
        return;
    }
    log::warn!(
        "[vs-tex-remap] vs={:#x} fs={:#x} base={} count={} shader_tex={:?} final_tex={:?} samplers={:?} remap=[{}] bindless_slot={} tex_cb_slot={} via_header={} split_vs_stage={} tic_pool={:#x} limit={}",
        vs_addr,
        fs_addr,
        vs_tex_base,
        vs_tex_count,
        shader_tex_ids,
        final_tex_ids,
        sampler_ids,
        tex_remap.join(" | "),
        bindless_slot,
        tex_cb_slot,
        via_header_index,
        split_vs_stage,
        tic_pool_gpu_va,
        tic_pool_limit
    );
}

fn fs_remap_trace(fs_addr: u64, fs_hash: u64) -> bool {
    use std::sync::OnceLock;
    static FILTER: OnceLock<(bool, Vec<u64>, Vec<u64>)> = OnceLock::new();
    let (all, addresses, hashes) = FILTER.get_or_init(|| {
        let value = std::env::var("NEXIUM_BIND_TRACE_FS").unwrap_or_default();
        let all = value
            .split(',')
            .map(str::trim)
            .any(|part| matches!(part.to_ascii_lowercase().as_str(), "1" | "true" | "all"));
        let mut addresses = Vec::new();
        let mut hashes = Vec::new();
        for part in value.split(',').map(str::trim).filter(|part| !part.is_empty()) {
            if let Some(hash) = part.strip_prefix("hash:") {
                let hash = hash
                    .strip_prefix("0x")
                    .or_else(|| hash.strip_prefix("0X"))
                    .unwrap_or(hash);
                if let Ok(hash) = u64::from_str_radix(hash, 16) {
                    hashes.push(hash);
                }
            } else if let Some(address) = parse_env_u64(part) {
                addresses.push(address);
            }
        }
        (all, addresses, hashes)
    });
    *all || addresses.contains(&fs_addr) || hashes.contains(&fs_hash)
}

fn trace_matching_tic_pool_entries(
    fs_addr: u64,
    fs_hash: u64,
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    if std::env::var_os("NEXIUM_TIC_POOL_SCAN").is_none()
        || !fs_remap_trace(fs_addr, fs_hash)
        || tic_pool_gpu_va == 0
    {
        return;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static SCANNED: AtomicBool = AtomicBool::new(false);
    if SCANNED.swap(true, Ordering::Relaxed) {
        return;
    }

    let configured_limit = std::env::var("NEXIUM_TIC_POOL_SCAN_LIMIT")
        .ok()
        .as_deref()
        .and_then(parse_env_u64)
        .unwrap_or(8192) as usize;
    let entry_count = (tic_pool_limit as usize)
        .saturating_add(1)
        .min(configured_limit.max(1));
    let mut readable = 0usize;
    let mut parsed = 0usize;
    let mut cube_arrays = 0usize;
    let mut candidates = Vec::new();
    for tic_id in 0..entry_count {
        let tic_addr = tic_pool_gpu_va.wrapping_add((tic_id as u64) * 32);
        let Some(cpu_addr) = mappings.cpu_address_for(tic_addr) else {
            continue;
        };
        let mut raw = [0u8; 32];
        if !mem_read(cpu_addr, &mut raw) {
            continue;
        }
        readable += 1;
        let Some(tic) = nexium_gpu::texture::TicEntry::parse(&raw) else {
            continue;
        };
        parsed += 1;
        if tic.texture_type == 8 {
            cube_arrays += 1;
        }
        let ambient_shape = tic.width == 128
            && tic.height == 128
            && tic.format == nexium_gpu::texture::TicFormat::R16G16B16A16
            && matches!(tic.texture_type, 3 | 8);
        if tic.texture_type == 8 || ambient_shape {
            let words = raw
                .chunks_exact(4)
                .map(|word| format!("{:08x}", u32::from_le_bytes(word.try_into().unwrap())))
                .collect::<Vec<_>>()
                .join(":");
            candidates.push(format!(
                "tic{} addr={:#x} va={:#x} fmt={:?} {}x{}x{} type={} base={} mips={}..{} raw={}",
                tic_id,
                tic_addr,
                tic.gpu_va,
                tic.format,
                tic.width,
                tic.height,
                tic.depth,
                tic.texture_type,
                tic.base_layer,
                tic.res_min_mip_level,
                tic.res_max_mip_level,
                words,
            ));
        }
    }
    log::warn!(
        "[tic-pool-scan] fs={:#x} hash={:016x} pool={:#x} scanned={} readable={} parsed={} cube_arrays={} candidates=[{}]",
        fs_addr,
        fs_hash,
        tic_pool_gpu_va,
        entry_count,
        readable,
        parsed,
        cube_arrays,
        candidates.join(" | "),
    );
}

fn vs_tex_bind_fs_trace(fs_addr: u64) -> bool {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<(bool, Vec<u64>)> = OnceLock::new();
    let (all, list) = CONFIG.get_or_init(|| {
        (
            std::env::var_os("NEXIUM_VS_TEX_BIND").is_some(),
            parse_env_u64_list("NEXIUM_VS_TEX_BIND_FS"),
        )
    });
    if !list.is_empty() {
        list.contains(&fs_addr)
    } else {
        *all
    }
}

fn tex_cb_sticky(stage_is_fs: bool) -> &'static std::sync::atomic::AtomicUsize {
    use std::sync::atomic::AtomicUsize;
    static FS: AtomicUsize = AtomicUsize::new(usize::MAX);
    static VS: AtomicUsize = AtomicUsize::new(usize::MAX);
    if stage_is_fs {
        &FS
    } else {
        &VS
    }
}

fn tex_cb_sticky_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("NEXIUM_TEX_CB_STICKY")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn choose_texture_cb_index(
    cbuf_binds: &[(u64, u32); 16],
    bindless_slot: u32,
    tex_cb_slot: u32,
    shader_ids: &[u32],
    tic_pool_gpu_va: u64,
    tic_pool_limit: u32,
    via_header_index: bool,
    stage_is_fs: bool,
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
    if tex_cb_sticky_enabled() {
        use std::sync::atomic::Ordering;
        let sticky = tex_cb_sticky(stage_is_fs);
        let prev = sticky.load(Ordering::Relaxed);
        if prev < 16 && prev != best {
            let prev_score = texture_cb_score(
                cbuf_binds,
                prev,
                shader_ids,
                tic_pool_gpu_va,
                tic_pool_limit,
                via_header_index,
                mappings,
                mem_read,
            );
            if prev_score > 0 && prev_score >= best_score {
                best = prev;
                best_score = prev_score;
            }
        }
        if best_score > 0 {
            sticky.store(best, Ordering::Relaxed);
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
            if texture_tic_readable(0, tic_pool_gpu_va, tic_pool_limit, mappings, mem_read)
                && !texture_tic_readable(
                    *shader_id,
                    tic_pool_gpu_va,
                    tic_pool_limit,
                    mappings,
                    mem_read,
                )
            {
                score = score.saturating_add(1);
            }
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

fn sass_read_diag(
    stage: &str,
    addr: u64,
    program_region: u64,
    offset: u32,
    mappings: &GpuMappings,
) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 32 {
        return;
    }
    let start = addr.wrapping_add(SPH_SIZE as u64);
    let range = mappings.cpu_range_for(start);
    log::warn!(
        "[sassfail] {} addr={:#x} region={:#x} off={:#x} start={:#x} range={:?} {}",
        stage,
        addr,
        program_region,
        offset,
        start,
        range,
        mappings.describe_around(addr)
    );
}

fn fetch_sass(
    gpu_va: u64,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<Vec<u8>> {
    let start = gpu_va.wrapping_add(SPH_SIZE as u64);
    let mut buf = vec![0u8; MAX_SASS_BYTES];
    let mut filled = 0usize;
    while filled < MAX_SASS_BYTES {
        let va = start.wrapping_add(filled as u64);
        let cpu = match mappings.cpu_address_for(va) {
            Some(c) => c,
            None => break,
        };
        let page_end = (va & !0xFFF).wrapping_add(0x1000);
        let chunk = std::cmp::min((page_end - va) as usize, MAX_SASS_BYTES - filled);
        if !mem_read(cpu, &mut buf[filled..filled + chunk]) {
            break;
        }
        filled += chunk;
    }
    if filled == 0 {
        return None;
    }
    buf.truncate(filled);
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

fn fs_solid_probe_matches(fs_addr: u64) -> bool {
    use std::sync::OnceLock;
    static CFG: OnceLock<(bool, bool, Vec<u64>)> = OnceLock::new();
    let (enabled, all, list) = CFG.get_or_init(|| {
        let Ok(raw) = std::env::var("NEXIUM_FS_SOLID") else {
            return (false, false, Vec::new());
        };
        let all = raw.trim().eq_ignore_ascii_case("all");
        let list = if all {
            Vec::new()
        } else {
            raw.split(',')
                .filter_map(|part| parse_env_u64(part.trim()))
                .collect()
        };
        (true, all, list)
    });
    *enabled && (*all || list.contains(&fs_addr))
}

fn solid_red_fs_spirv() -> Vec<u32> {
    vec![
        0x0723_0203,
        0x0001_0000,
        0,
        12,
        0,
        (2 << 16) | 17,
        1,
        (3 << 16) | 14,
        0,
        1,
        (6 << 16) | 15,
        4,
        10,
        0x6e69_616d,
        0,
        6,
        (3 << 16) | 16,
        10,
        7,
        (4 << 16) | 71,
        6,
        30,
        0,
        (2 << 16) | 19,
        1,
        (3 << 16) | 33,
        2,
        1,
        (3 << 16) | 22,
        3,
        32,
        (4 << 16) | 23,
        4,
        3,
        4,
        (4 << 16) | 32,
        5,
        3,
        4,
        (4 << 16) | 59,
        5,
        6,
        3,
        (4 << 16) | 43,
        3,
        7,
        0x3F80_0000,
        (4 << 16) | 43,
        3,
        8,
        0,
        (7 << 16) | 44,
        4,
        9,
        8,
        7,
        8,
        7,
        (5 << 16) | 54,
        1,
        10,
        0,
        2,
        (2 << 16) | 248,
        11,
        (3 << 16) | 62,
        6,
        9,
        (1 << 16) | 253,
        (1 << 16) | 56,
    ]
}

fn build_vertex_layout(draw: &DrawCall, used_locations: &[u32]) -> Result<VertexLayout, String> {
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
        if attrib.format == 0 || used_locations.binary_search(&(loc as u32)).is_err() {
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
            let divisor = if instancing_disabled() {
                0
            } else {
                draw.vertex_stream_instances
                    .get(binding as usize)
                    .copied()
                    .filter(|v| *v != 0)
                    .and_then(|_| draw.vertex_buffers.get(binding as usize))
                    .map(|vb| vb.frequency.max(1))
                    .unwrap_or(0)
            };
            if stride == 0 {
                let mut packed = 0u32;
                for (packed_loc, a) in draw.vertex_attribs.iter().enumerate() {
                    if a.format != 0
                        && !a.constant
                        && a.buffer == binding
                        && used_locations.binary_search(&(packed_loc as u32)).is_ok()
                    {
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
            bindings.push(VertexBinding {
                binding,
                stride,
                divisor,
            });
            seen_bindings.insert(binding);
        }
        attrs.push(VertexAttr {
            location: loc as u32,
            binding,
            format,
            offset: attrib.offset,
        });
    }

    if need_white {
        bindings.push(VertexBinding {
            binding: WHITE_BINDING,
            stride: 0,
            divisor: 0,
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
) -> Vec<u8> {
    let used = vs_mask | fs_mask;
    let mut out = vec![0u8; nexium_spirv::GFX_CBUF_MIN_SIZE as usize];
    for logical_slot in 0..PACKED_CBUF_SLOTS {
        let directory = logical_slot * 8;
        out[directory..directory + 4]
            .copy_from_slice(&nexium_spirv::GFX_CBUF_ZERO_WORD.to_le_bytes());
    }
    static CBUF_STASH: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, [u8; 64])>>> =
        std::sync::OnceLock::new();
    let recheck = {
        use std::sync::OnceLock;
        static R: OnceLock<bool> = OnceLock::new();
        *R.get_or_init(|| std::env::var_os("NEXIUM_CBUF_RECHECK").is_some())
    };
    if recheck {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LATE_LOGS: AtomicU64 = AtomicU64::new(0);
        let stash = CBUF_STASH.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let mut guard = stash.lock().unwrap();
        for (cpu, old) in guard.iter() {
            let mut cur = [0u8; 64];
            if mem_read(*cpu, &mut cur) && cur != *old {
                let n = LATE_LOGS.fetch_add(1, Ordering::Relaxed);
                if n < 64 {
                    let w = |b: &[u8; 64], i: usize| {
                        u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
                    };
                    log::warn!(
                        "[cbuf-late-write] cpu={:#x} old={:08x} {:08x} {:08x} {:08x} new={:08x} {:08x} {:08x} {:08x}",
                        cpu,
                        w(old, 0),
                        w(old, 1),
                        w(old, 2),
                        w(old, 3),
                        w(&cur, 0),
                        w(&cur, 1),
                        w(&cur, 2),
                        w(&cur, 3)
                    );
                }
            }
        }
        guard.clear();
    }
    let prefer_vertex_b = {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var_os("NEXIUM_CBUF_VS_STAGE1").is_some())
    };
    for logical_slot in 0..PACKED_CBUF_SLOTS {
        if (used & (1u32 << logical_slot)) == 0 {
            continue;
        }
        let stage = if logical_slot < 16 { 0 } else { 4 };
        let binding = logical_slot & 15;
        let (addr, size) = if prefer_vertex_b && logical_slot < 16 && cbuf_binds[1][binding].0 != 0
        {
            cbuf_binds[1][binding]
        } else {
            cbuf_binds[stage][binding]
        };
        if addr == 0 || size == 0 {
            continue;
        }
        let len = (size as usize).min(nexium_spirv::GFX_CBUF_MAX_SIZE as usize);
        let word_count = len.div_ceil(4);
        if word_count == 0 {
            continue;
        }
        let aligned_len = (out.len() + 15) & !15;
        out.resize(aligned_len, 0);
        let base_word = out.len() / 4;
        let off = out.len();
        out.resize(off + word_count * 4, 0);

        let directory = logical_slot * 8;
        out[directory..directory + 4].copy_from_slice(&(base_word as u32).to_le_bytes());
        out[directory + 4..directory + 8]
            .copy_from_slice(&(word_count as u32).to_le_bytes());

        let readable = copy_mapped_cbuf_bytes(
            mappings,
            addr,
            &mut out[off..off + len],
            mem_read,
        );
        if let Some(cpu) = mappings.cpu_address_for(addr) {
            if readable && recheck && logical_slot == 3 && size == 2560 {
                let mut first = [0u8; 64];
                let take = len.min(64);
                first[..take].copy_from_slice(&out[off..off + take]);
                let stash = CBUF_STASH.get_or_init(|| std::sync::Mutex::new(Vec::new()));
                let mut guard = stash.lock().unwrap();
                if guard.len() < 8 {
                    guard.push((cpu, first));
                }
            }
        }
    }
    out
}

fn read_stage_cbuf_u32(
    stage: usize,
    binding: u8,
    offset: u32,
    cbuf_binds: &[[(u64, u32); 16]; 5],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<u32> {
    let &(address, size) = cbuf_binds.get(stage)?.get(binding as usize)?;
    if address == 0 || offset.checked_add(4)? > size {
        if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
            log::warn!(
                "[brx-cbuf] stage={} binding={} offset={:#x} bind={:#x}/{} unavailable",
                stage,
                binding,
                offset,
                address,
                size
            );
        }
        return None;
    }
    let gpu_address = address.checked_add(offset as u64)?;
    let Some(cpu_address) = mappings.cpu_address_for(gpu_address) else {
        if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
            log::warn!(
                "[brx-cbuf] stage={} binding={} gpu={:#x} has no CPU mapping",
                stage,
                binding,
                gpu_address
            );
        }
        return None;
    };
    let mut bytes = [0u8; 4];
    if !mem_read(cpu_address, &mut bytes) {
        if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
            log::warn!(
                "[brx-cbuf] stage={} binding={} gpu={:#x} cpu={:#x} read failed",
                stage,
                binding,
                gpu_address,
                cpu_address
            );
        }
        return None;
    }
    let value = u32::from_le_bytes(bytes);
    if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
        log::warn!(
            "[brx-cbuf] stage={} binding={} gpu={:#x} cpu={:#x} value={:#010x}",
            stage,
            binding,
            gpu_address,
            cpu_address,
            value
        );
    }
    Some(value)
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
            if !vb.enabled {
                return None;
            }
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
                divisor: b.divisor,
                size,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::{
        build_vertex_layout, cfg_uses_texture_descriptors, collect_cbuf_reads,
        collect_graphics_cbuf_reads,
        consistent_tic_numeric_type,
        effective_depth_states, font_tic_layer_count, fragment_output_numeric_masks,
        format_cbuf_read, fragment_texture_numeric_metadata, graphics_ring_chunk_ranges_for_costs,
        graphics_texture_layout_from_metadata,
        has_unimplemented_brx, map_stencil_op, map_zeta_format, pack_cbuf_data,
        maxwell_draw_orientation,
        aliased_guest_ranges, guest_write_alias_ranges, normalize_guest_ranges,
        packed_cbuf_slot, packed_cbuf_word, plan_guest_write_chunks, read_gpu_strict,
        register_small_rt_after_prior_work,
        shader_numeric_key, shader_resource_fingerprint, small_rt_registry,
        snapshot_texture_once, small_rt_starts_in_ranges,
        spirv_texture_manifest_for_stage,
        split_texture_handle, supported_texel_buffer, texel_buffer_numeric_type,
        texture_cbuf_origin, texture_view_metadata_fingerprint,
        tic_can_alias_render_target, tic_can_alias_render_target_view, tic_numeric_type,
        tic_snapshot_layer_count, vertex_buffer_bindings, write_guest_strict, y_direction_key,
        FragmentTextureNumericMetadata, GuestRange, GuestWriteChunk,
    };

    #[test]
    fn triangle_rast_flip_changes_winding_not_present_row_order() {
        use crate::gpu::engines::maxwell3d::WindowOrigin;

        assert_eq!(
            maxwell_draw_orientation(WindowOrigin { raw: 0x10 }, 0x0900),
            (0x0901, false)
        );
        assert_eq!(
            maxwell_draw_orientation(WindowOrigin { raw: 0 }, 0x0900),
            (0x0900, false)
        );
    }

    #[test]
    fn small_rt_registration_happens_after_prior_batch_work() {
        let key = nexium_gpu::rt_cache::RtKey::new(0xffff_ff01, 17, 19, 0x1234_5678_9000);
        small_rt_registry().lock().unwrap().remove(&key);

        register_small_rt_after_prior_work(key, Some(0x20), || {
            assert!(!small_rt_registry().lock().unwrap().contains_key(&key));
        });

        assert_eq!(small_rt_registry().lock().unwrap().remove(&key), Some(0x20));
    }

    #[test]
    fn cube_dependency_ranges_follow_cpu_backing_aliases() {
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 0x100, 0x50_000, 1);
        mappings.add(0x8000, 0x100, 0x50_000, 2);

        let ranges = aliased_guest_ranges(
            &mappings,
            GuestRange {
                start: 0x1020,
                end: 0x1060,
            },
        );
        assert!(ranges.contains(&GuestRange {
            start: 0x1020,
            end: 0x1060,
        }));
        assert!(ranges.contains(&GuestRange {
            start: 0x8020,
            end: 0x8060,
        }));
    }

    #[test]
    fn small_rt_guest_write_crosses_mappings_and_retains_partial_failure() {
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 4, 0x20_000, 1);
        mappings.add(0x1004, 4, 0x30_000, 1);
        mappings.add(0x9000, 4, 0x20_000, 2);
        let expected_chunks = vec![
            GuestWriteChunk {
                gpu_va: 0x1002,
                cpu_addr: 0x20_002,
                data_offset: 0,
                len: 2,
            },
            GuestWriteChunk {
                gpu_va: 0x1004,
                cpu_addr: 0x30_000,
                data_offset: 2,
                len: 4,
            },
        ];
        assert_eq!(
            plan_guest_write_chunks(&mappings, 0x1002, 6),
            Some(expected_chunks.clone())
        );

        let writes = std::sync::Mutex::new(Vec::new());
        let complete = write_guest_strict(
            &mappings,
            &|cpu, bytes| {
                writes.lock().unwrap().push((cpu, bytes.to_vec()));
                true
            },
            0x1002,
            &[1, 2, 3, 4, 5, 6],
        );
        assert!(complete.complete);
        assert_eq!(complete.written, expected_chunks);
        assert_eq!(
            *writes.lock().unwrap(),
            vec![(0x20_002, vec![1, 2]), (0x30_000, vec![3, 4, 5, 6])]
        );
        assert!(guest_write_alias_ranges(&mappings, &complete.written).contains(
            &GuestRange {
                start: 0x9002,
                end: 0x9004,
            }
        ));

        let failed = write_guest_strict(
            &mappings,
            &|cpu, _| cpu != 0x30_000,
            0x1002,
            &[1, 2, 3, 4, 5, 6],
        );
        assert!(!failed.complete);
        assert_eq!(failed.written, expected_chunks[..1]);
    }

    #[test]
    fn cube_dependency_ranges_cover_every_face_and_mip_without_touching_next_cube() {
        let base = 0x5689_d0000u64;
        let layer_stride = 0x2c000u64;
        let ranges = normalize_guest_ranges(vec![
            GuestRange {
                start: base + layer_stride,
                end: base + layer_stride * 4,
            },
            GuestRange {
                start: base,
                end: base + layer_stride * 2,
            },
            GuestRange {
                start: base + layer_stride * 4,
                end: base + layer_stride * 6,
            },
        ]);
        assert_eq!(
            ranges,
            vec![GuestRange {
                start: base,
                end: base + layer_stride * 6,
            }]
        );

        for face in 0..6u64 {
            let face_key = nexium_gpu::rt_cache::RtKey::new(
                168,
                128,
                128,
                base + face * layer_stride,
            );
            assert!(small_rt_starts_in_ranges(face_key, &ranges));

            let mip_key = nexium_gpu::rt_cache::RtKey::new(
                168,
                32,
                32,
                base + face * layer_stride + 0x28000,
            );
            assert!(small_rt_starts_in_ranges(mip_key, &ranges));
        }

        let next_cube = nexium_gpu::rt_cache::RtKey::new(
            169,
            128,
            128,
            base + layer_stride * 6,
        );
        assert!(!small_rt_starts_in_ranges(next_cube, &ranges));
    }

    #[test]
    fn graphics_ring_chunks_never_exceed_the_safe_multi_draw_budget() {
        let mib = 1024 * 1024u64;
        let safe = nexium_gpu::renderer::GRAPHICS_RING_SAFE_BATCH_BYTES;
        let capacity = nexium_gpu::renderer::GRAPHICS_RING_CAPACITY_BYTES;
        let costs = [3 * mib, 3 * mib, 3 * mib, 7 * mib, 1 * mib];
        let ranges = graphics_ring_chunk_ranges_for_costs(&costs, safe);

        assert_eq!(ranges, vec![(0, 2), (2, 3), (3, 5)]);
        let mut next = 0usize;
        for (start, end) in ranges {
            assert_eq!(start, next, "chunks must preserve draw order");
            assert!(start < end);
            let bytes = costs[start..end].iter().sum::<u64>();
            assert!(bytes <= safe);
            assert!(bytes <= capacity);
            next = end;
        }
        assert_eq!(next, costs.len());
    }

    #[test]
    fn cbuf_read_metadata_retains_static_indexed_and_more_than_thirty_two_entries() {
        let mut program = nexium_shader::IrProgram::new();
        for offset in (0..40u32).map(|word| word * 4) {
            program.emit(
                nexium_shader::IrOp::LoadCbuf {
                    binding: 6,
                    byte_offset: offset,
                },
                None,
            );
        }
        program.emit(
            nexium_shader::IrOp::LoadCbufIndexed {
                binding: 6,
                byte_offset: 0x2aa0,
                index: nexium_shader::IrValue::GprIn(37),
                address_mode: nexium_shader::CbufAddressMode::Default,
            },
            None,
        );
        let cfg = nexium_shader::Cfg {
            blocks: vec![nexium_shader::BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 8,
                branch: nexium_shader::BranchKind::Exit,
                program,
                reg_exit: std::collections::HashMap::new(),
                pred_exit: std::collections::HashMap::new(),
                pred_phis: Vec::new(),
            }],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let reads = collect_cbuf_reads(&cfg, 16);
        assert_eq!(reads.len(), 41);
        assert!(reads.iter().any(|read| {
            read.logical_slot == 22 && read.byte_offset == 0 && !read.is_indexed()
        }));
        assert!(reads.iter().any(|read| {
            read.logical_slot == 22
                && read.byte_offset == 0x2aa0
                && read.index_origin == nexium_gpu::bundle_cache::CbufIndexOrigin::Gpr(37)
        }));
    }

    #[test]
    fn graphics_cbuf_metadata_combines_stages_and_records_both_storage_base_words() {
        fn cfg(program: nexium_shader::IrProgram) -> nexium_shader::Cfg {
            nexium_shader::Cfg {
                blocks: vec![nexium_shader::BasicBlock {
                    id: 0,
                    start_offset: 0,
                    end_offset: 8,
                    branch: nexium_shader::BranchKind::Exit,
                    program,
                    reg_exit: std::collections::HashMap::new(),
                    pred_exit: std::collections::HashMap::new(),
                    pred_phis: Vec::new(),
                }],
                unimplemented: 0,
                bindless_or_partners: Default::default(),
            }
        }

        let mut vs_program = nexium_shader::IrProgram::new();
        vs_program.emit(
            nexium_shader::IrOp::LoadStorage {
                buffer_index: 0,
                addr_lo: nexium_shader::IrValue::Zero,
                imm: 0,
                cbuf_binding: 4,
                cbuf_offset: 0x80,
                align: 16,
            },
            None,
        );
        let constant_index = vs_program.emit(
            nexium_shader::IrOp::Mov(nexium_shader::IrValue::ImmU32(0xffff_fff0)),
            None,
        );
        vs_program.emit(
            nexium_shader::IrOp::LoadCbufIndexed {
                binding: 5,
                byte_offset: 0x10,
                index: nexium_shader::IrValue::Inst(constant_index),
                address_mode: nexium_shader::CbufAddressMode::Default,
            },
            None,
        );

        let mut fs_program = nexium_shader::IrProgram::new();
        fs_program.emit(
            nexium_shader::IrOp::LoadCbuf {
                binding: 6,
                byte_offset: 0x1550,
            },
            None,
        );
        fs_program.emit(
            nexium_shader::IrOp::LoadCbufIndexed {
                binding: 7,
                byte_offset: 0x2aa0,
                index: nexium_shader::IrValue::Inst(nexium_shader::ValueId(73)),
                address_mode: nexium_shader::CbufAddressMode::Default,
            },
            None,
        );

        let reads = collect_graphics_cbuf_reads(&cfg(vs_program), &cfg(fs_program));
        assert_eq!(
            reads,
            vec![
                nexium_gpu::bundle_cache::CbufRead {
                    logical_slot: 4,
                    byte_offset: 0x80,
                    index_origin: nexium_gpu::bundle_cache::CbufIndexOrigin::Static,
                },
                nexium_gpu::bundle_cache::CbufRead {
                    logical_slot: 4,
                    byte_offset: 0x84,
                    index_origin: nexium_gpu::bundle_cache::CbufIndexOrigin::Static,
                },
                nexium_gpu::bundle_cache::CbufRead {
                    logical_slot: 5,
                    byte_offset: 0x10,
                    index_origin: nexium_gpu::bundle_cache::CbufIndexOrigin::Constant(
                        0xffff_fff0,
                    ),
                },
                nexium_gpu::bundle_cache::CbufRead {
                    logical_slot: 22,
                    byte_offset: 0x1550,
                    index_origin: nexium_gpu::bundle_cache::CbufIndexOrigin::Static,
                },
                nexium_gpu::bundle_cache::CbufRead {
                    logical_slot: 23,
                    byte_offset: 0x2aa0,
                    index_origin: nexium_gpu::bundle_cache::CbufIndexOrigin::Instruction(73),
                },
            ]
        );
        assert_eq!(reads[2].effective_byte_offset(), Some(0));
        assert_eq!(reads[4].effective_byte_offset(), None);
    }

    #[test]
    fn cbuf_read_trace_samples_only_exact_known_effective_addresses() {
        use nexium_gpu::bundle_cache::{CbufIndexOrigin, CbufRead};

        let payload_word = nexium_spirv::GFX_CBUF_PAYLOAD_WORD as usize;
        let mut packed = vec![0u8; (payload_word + 2) * 4];
        let directory = 22usize * 8;
        packed[directory..directory + 4]
            .copy_from_slice(&(payload_word as u32).to_le_bytes());
        packed[directory + 4..directory + 8].copy_from_slice(&2u32.to_le_bytes());
        packed[payload_word * 4..payload_word * 4 + 4]
            .copy_from_slice(&0x3f80_0000u32.to_le_bytes());
        packed[payload_word * 4 + 4..payload_word * 4 + 8]
            .copy_from_slice(&0x4000_0000u32.to_le_bytes());
        let mut binds = [[(0u64, 0u32); 16]; 5];
        binds[4][6] = (0x1234_0000, 8);

        let known = format_cbuf_read(
            &packed,
            CbufRead {
                logical_slot: 22,
                byte_offset: 0x10,
                index_origin: CbufIndexOrigin::Constant(0xffff_fff3),
            },
            &binds,
        );
        assert!(known.contains("slot=22 base=0x12340000 size=8"), "{known}");
        assert!(known.contains("effective=0x00000003"), "{known}");
        assert!(known.contains("word=0x00000000"), "{known}");
        assert!(known.contains("in_range=true"), "{known}");
        assert!(known.contains("value=1.000/0x3f800000"), "{known}");

        let dynamic = format_cbuf_read(
            &packed,
            CbufRead {
                logical_slot: 22,
                byte_offset: 0,
                index_origin: CbufIndexOrigin::Gpr(37),
            },
            &binds,
        );
        assert!(dynamic.contains("origin=indexed:gpr(R37)"), "{dynamic}");
        assert!(dynamic.contains("effective=dynamic"), "{dynamic}");
        assert!(dynamic.contains("in_range=unknown"), "{dynamic}");
        assert!(!dynamic.contains("value="), "{dynamic}");

        let out = format_cbuf_read(
            &packed,
            CbufRead {
                logical_slot: 22,
                byte_offset: 8,
                index_origin: CbufIndexOrigin::Static,
            },
            &binds,
        );
        assert!(out.contains("in_range=false"), "{out}");
        assert!(!out.contains("value="), "{out}");
    }

    #[test]
    fn graphics_cbuf_packer_preserves_full_distinct_stage_payloads() {
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 0x1_0000, 0x10_0000, 1);
        mappings.add(0x20_000, 0x1_0000, 0x20_0000, 2);
        mappings.add(0x40_000, 0x100, 0x30_0000, 3);

        let mut vs = vec![0u8; 0x1_0000];
        let mut fs = vec![0u8; 0x1_0000];
        vs[0x2a0..0x2a4].copy_from_slice(&0x1111_02a0u32.to_le_bytes());
        for (offset, value) in [
            (0x2a0usize, 0x2222_02a0u32),
            (0x550, 0x2222_0550),
            (0x1550, 0x2222_1550),
            (0x2aa0, 0x2222_2aa0),
            (0x4000, 0x2222_4000),
        ] {
            fs[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let read = |cpu: u64, dst: &mut [u8]| {
            let copy = |base: u64, source: &[u8], dst: &mut [u8]| {
                let Some(offset) = cpu.checked_sub(base).map(|offset| offset as usize) else {
                    return false;
                };
                let Some(end) = offset.checked_add(dst.len()) else {
                    return false;
                };
                let Some(bytes) = source.get(offset..end) else {
                    return false;
                };
                dst.copy_from_slice(bytes);
                true
            };
            copy(0x10_0000, &vs, dst) || copy(0x20_0000, &fs, dst)
        };

        let mut binds = [[(0u64, 0u32); 16]; 5];
        binds[0][6] = (0x1000, 0x1_0000);
        binds[4][6] = (0x20_000, 0x1_0000);
        binds[0][7] = (0x40_000, 0x24);
        let packed = pack_cbuf_data(
            &binds,
            (1 << 6) | (1 << 7),
            1 << 22,
            &mappings,
            &read,
        );

        assert_eq!(packed_cbuf_slot(&packed, 6).unwrap().len(), 0x1_0000);
        assert_eq!(packed_cbuf_slot(&packed, 22).unwrap().len(), 0x1_0000);
        assert_eq!(packed_cbuf_word(&packed, 6, 0x2a0), Some(0x1111_02a0));
        for (offset, value) in [
            (0x2a0usize, 0x2222_02a0u32),
            (0x550, 0x2222_0550),
            (0x1550, 0x2222_1550),
            (0x2aa0, 0x2222_2aa0),
            (0x4000, 0x2222_4000),
        ] {
            assert_eq!(packed_cbuf_word(&packed, 22, offset), Some(value));
        }
        assert_eq!(packed_cbuf_slot(&packed, 7).unwrap(), &[0u8; 0x24]);
        assert!(packed_cbuf_slot(&packed, 12).unwrap().is_empty());
        let absent_base = u32::from_le_bytes(packed[12 * 8..12 * 8 + 4].try_into().unwrap());
        assert_eq!(absent_base, nexium_spirv::GFX_CBUF_ZERO_WORD);
        for slot in [6usize, 7, 22] {
            let base = u32::from_le_bytes(packed[slot * 8..slot * 8 + 4].try_into().unwrap());
            assert_eq!(base % 4, 0, "slot {slot} payload must be 16-byte aligned");
        }
    }

    #[test]
    fn graphics_cbuf_packer_keeps_readable_prefix_and_zero_fills_unmapped_tail() {
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x5000, 0x10, 0x40_0000, 1);
        let source = (0..0x10u8).collect::<Vec<_>>();
        let read = |cpu: u64, dst: &mut [u8]| {
            let Some(offset) = cpu.checked_sub(0x40_0000).map(|offset| offset as usize) else {
                return false;
            };
            let Some(bytes) = source.get(offset..offset.saturating_add(dst.len())) else {
                return false;
            };
            dst.copy_from_slice(bytes);
            true
        };

        let mut binds = [[(0u64, 0u32); 16]; 5];
        binds[0][8] = (0x5000, 0x24);
        let packed = pack_cbuf_data(&binds, 1 << 8, 0, &mappings, &read);
        let slot = packed_cbuf_slot(&packed, 8).expect("packed VS cbuf 8");

        assert_eq!(slot.len(), 0x24);
        assert_eq!(&slot[..0x10], source.as_slice());
        assert_eq!(&slot[0x10..], &[0u8; 0x14]);
    }

    fn cfg_with_unimplemented(opcode: nexium_shader::Opcode) -> nexium_shader::Cfg {
        let mut program = nexium_shader::IrProgram::new();
        program.emit_void(nexium_shader::IrOp::Unimplemented { opcode, raw: 0 });
        nexium_shader::Cfg {
            blocks: vec![nexium_shader::BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 8,
                branch: nexium_shader::BranchKind::Exit,
                program,
                reg_exit: std::collections::HashMap::new(),
                pred_phis: Vec::new(),
                pred_exit: std::collections::HashMap::new(),
            }],
            unimplemented: 1,
            bindless_or_partners: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn strict_gpu_read_follows_adjacent_noncontiguous_mappings() {
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 4, 0x2000, 1);
        mappings.add(0x1004, 4, 0x3000, 2);
        let read = |cpu: u64, out: &mut [u8]| {
            let (base, bytes): (u64, &[u8]) = if (0x2000..0x2004).contains(&cpu) {
                (0x2000, &[1, 2, 3, 4])
            } else if (0x3000..0x3004).contains(&cpu) {
                (0x3000, &[5, 6, 7, 8])
            } else {
                return false;
            };
            let offset = (cpu - base) as usize;
            let Some(source) = bytes.get(offset..offset + out.len()) else {
                return false;
            };
            out.copy_from_slice(source);
            true
        };
        assert_eq!(
            read_gpu_strict(&mappings, &read, 0x1002, 4),
            Some(vec![3, 4, 5, 6])
        );
        assert!(read_gpu_strict(&mappings, &read, 0x1006, 4).is_none());
    }

    #[test]
    fn texture_snapshot_does_not_shrink_existing_vertex_range() {
        let mut snapshot = std::collections::HashMap::new();
        snapshot.insert(0x4000, vec![0x5a; 0x100]);
        let mut generations = std::collections::HashMap::new();
        let reads = std::cell::Cell::new(0usize);
        let read = |_: u64, len: usize| {
            reads.set(reads.get() + 1);
            Some(vec![0xa5; len])
        };

        assert_eq!(
            snapshot_texture_once(&mut snapshot, &mut generations, &read, 0x4000, 0x10),
            0
        );
        assert_eq!(reads.get(), 0);
        assert_eq!(snapshot[&0x4000], vec![0x5a; 0x100]);
        assert!(generations.contains_key(&(0x4000, 0x10)));
    }

    #[test]
    fn vertex_layout_ignores_unused_and_disabled_streams() {
        let mut engine = crate::gpu::engines::Maxwell3D::new();
        engine.dispatch_method(0x700, 0x1000 | 0x10, true);
        engine.dispatch_method(0x701, 4, true);
        engine.dispatch_method(0x702, 0x2000, true);
        engine.dispatch_method(0x704, 0x10, true);
        engine.dispatch_method(0x705, 0, true);
        engine.dispatch_method(0x706, 0x1000, true);

        let r32g32b32a32_float = (1 << 21) | (7 << 27);
        engine.dispatch_method(0x458, r32g32b32a32_float, true);
        engine.dispatch_method(0x459, r32g32b32a32_float | 1, true);
        engine.dispatch_method(0x35e, 3, true);
        let draw = engine.pending_draws.last().unwrap();

        let layout = build_vertex_layout(draw, &[0]).unwrap();
        assert_eq!(layout.attrs.len(), 1);
        assert_eq!(layout.attrs[0].location, 0);
        assert_eq!(layout.bindings.len(), 1);
        assert_eq!(layout.bindings[0].binding, 0);

        let bindings = vertex_buffer_bindings(&draw.vertex_buffers, &layout);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].addr, 0x4_0000_2000);

        let layout_with_disabled = build_vertex_layout(draw, &[0, 1]).unwrap();
        let bindings = vertex_buffer_bindings(&draw.vertex_buffers, &layout_with_disabled);
        assert_eq!(bindings.len(), 1);
        assert!(bindings.iter().all(|binding| binding.addr != 0x1000));

        let empty_layout = build_vertex_layout(draw, &[]).unwrap();
        assert!(empty_layout.attrs.is_empty());
        assert!(empty_layout.bindings.is_empty());
    }

    #[test]
    fn rejects_only_unimplemented_brx() {
        assert!(has_unimplemented_brx(&cfg_with_unimplemented(
            nexium_shader::Opcode::BRX
        )));
        assert!(!has_unimplemented_brx(&cfg_with_unimplemented(
            nexium_shader::Opcode::TEX_b
        )));
    }

    #[test]
    fn z24s8_can_alias_native_depth_render_target() {
        assert!(tic_can_alias_render_target(
            nexium_gpu::texture::TicFormat::Z24S8
        ));
    }

    #[test]
    fn tagged_texel_fetch_id_uses_exact_cbuf_origin() {
        let tagged = nexium_shader::bindless_texture_id(2, 0x68);
        assert_eq!(
            texture_cbuf_origin(tagged, 15, false),
            (2, 0x68, None, true)
        );
        assert_eq!(
            texture_cbuf_origin(0x68, 15, false),
            (15, 0x68, None, false)
        );
        assert_ne!(
            tagged,
            nexium_shader::bindless_texture_id(3, 0x68),
            "equal word offsets in different cbuf bindings need distinct descriptor slots"
        );
    }

    #[test]
    fn tagged_texel_fetch_pair_reconstructs_exact_texture_handle() {
        for (primary, secondary) in [(0x6c, 0x162), (0x5a, 0x15a)] {
            let tagged = nexium_shader::bindless_texture_id_pair(2, primary, Some(secondary));
            assert_eq!(
                texture_cbuf_origin(tagged, 15, false),
                (2, primary, Some(secondary), true)
            );
        }
        let handle = 0x0001_2345 | 0xabc0_0000;
        assert_eq!(split_texture_handle(handle, false), (0x1_2345, 0xabc));
    }

    #[test]
    fn fragment_output_numeric_masks_follow_guest_locations() {
        let key = nexium_gpu::rt_cache::RtKey::new(1, 16, 16, 0x1000);
        let rts = [
            (0, key, vk::Format::R16_UINT),
            (1, key, vk::Format::R8G8B8A8_UNORM),
            (3, key, vk::Format::R32_SINT),
        ];
        assert_eq!(fragment_output_numeric_masks(&rts), (1, 1 << 3));
    }

    #[test]
    fn mixed_or_missing_tic_numeric_evidence_falls_back_to_float() {
        use nexium_gpu::texture::ComponentType;
        use nexium_spirv::TextureNumericType;

        assert_eq!(
            consistent_tic_numeric_type([ComponentType::Uint, ComponentType::Uint]),
            TextureNumericType::Uint
        );
        assert_eq!(
            consistent_tic_numeric_type([ComponentType::Sint, ComponentType::Sint]),
            TextureNumericType::Sint
        );
        assert_eq!(
            consistent_tic_numeric_type([ComponentType::Uint, ComponentType::Sint]),
            TextureNumericType::Float
        );
        assert_eq!(
            consistent_tic_numeric_type([ComponentType::Uint, ComponentType::Unorm]),
            TextureNumericType::Float
        );
        assert_eq!(
            consistent_tic_numeric_type(std::iter::empty()),
            TextureNumericType::Float
        );
    }

    #[test]
    fn shader_keys_distinguish_outputs_and_texture_manifests() {
        use nexium_gpu::texture_manifest::{
            normalize_texture_numeric_manifest, texture_numeric_manifest_fingerprint,
            TextureNumericBinding,
        };
        use nexium_spirv::TextureNumericType;

        let base = shader_numeric_key(0, 0);
        let uint_output = shader_numeric_key(1, 0);
        let sint_output = shader_numeric_key(0, 1);
        let lower_left = base | y_direction_key(true);
        let keys = [base, uint_output, sint_output, lower_left]
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
        assert_eq!(keys.len(), 4);

        let float_manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(0x44, 3, TextureNumericType::Float),
        ])
        .unwrap();
        let uint_manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(0x44, 3, TextureNumericType::Uint),
        ])
        .unwrap();
        let cube_array_manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(0x44, 3, TextureNumericType::Float)
                .with_image_kind(
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::CubeArray,
                ),
        ])
        .unwrap();
        let float_fingerprint = texture_numeric_manifest_fingerprint(&float_manifest);
        let uint_fingerprint = texture_numeric_manifest_fingerprint(&uint_manifest);
        let cube_array_fingerprint =
            texture_numeric_manifest_fingerprint(&cube_array_manifest);
        assert_ne!(float_fingerprint, uint_fingerprint);
        assert_ne!(float_fingerprint, cube_array_fingerprint);
        assert_ne!(
            shader_resource_fingerprint(0, 1 << 8, 0, float_fingerprint, 0),
            shader_resource_fingerprint(0, 1 << 8, 0, uint_fingerprint, 0)
        );
        assert_ne!(
            shader_resource_fingerprint(0, 0, 0, float_fingerprint, 0),
            shader_resource_fingerprint(0, 0, 0, float_fingerprint, 1)
        );
    }

    #[test]
    fn canonical_texture_manifest_converts_to_stage_spirv_resources() {
        use nexium_gpu::texture_manifest::{
            normalize_texture_numeric_manifest, TextureNumericBinding,
        };
        use nexium_spirv::{GraphicsTextureResource, TextureNumericType};

        let manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(0x88, 3, TextureNumericType::Sint),
            TextureNumericBinding::new(0x44, 0, TextureNumericType::Float),
            TextureNumericBinding::new(0xfe, 1, TextureNumericType::Float),
            TextureNumericBinding::new(0x66, 2, TextureNumericType::Uint),
        ])
        .unwrap();
        assert_eq!(
            spirv_texture_manifest_for_stage(&manifest, 0, 1),
            vec![GraphicsTextureResource::new(
                0x44,
                0,
                TextureNumericType::Float,
            )],
            "walker-only FS descriptors appended after translated IDs must not reach the emitter"
        );
        assert_eq!(
            spirv_texture_manifest_for_stage(&manifest, 2, 1),
            vec![GraphicsTextureResource::new(
                0x66,
                2,
                TextureNumericType::Uint,
            )],
            "walker-only VS descriptors appended after translated IDs must not reach the emitter"
        );
    }

    #[test]
    fn fragment_texture_metadata_preserves_tld_ids_in_mixed_shaders() {
        let texel_fetch = |component| nexium_shader::IrOp::TexelFetch {
            cbuf_binding: 2,
            cbuf_word_offset: 0x6c,
            cbuf_secondary_word_offset: Some(0x162),
            x: nexium_shader::IrValue::ImmF32(0.0),
            y: None,
            z: None,
            component,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit(texel_fetch(1), Some(0));
        program.emit(texel_fetch(3), Some(1));
        let block = |program| nexium_shader::BasicBlock {
            id: 0,
            start_offset: 0,
            end_offset: 8,
            branch: nexium_shader::BranchKind::Exit,
            program,
            reg_exit: std::collections::HashMap::new(),
            pred_phis: Vec::new(),
            pred_exit: std::collections::HashMap::new(),
        };
        let tld_cfg = nexium_shader::Cfg {
            blocks: vec![block(program)],
            unimplemented: 0,
            bindless_or_partners: std::collections::HashMap::new(),
        };
        let tld_metadata = fragment_texture_numeric_metadata(&tld_cfg).unwrap();
        let bindless_id = nexium_shader::bindless_texture_id_pair(2, 0x6c, Some(0x162));
        assert_eq!(
            tld_metadata.texel_fetches,
            Some(vec![(bindless_id, 0b1010,)])
        );
        assert_eq!(tld_metadata.texture_ids, vec![bindless_id]);
        assert_eq!(tld_metadata.descriptor_ids, vec![bindless_id]);
        assert!(tld_metadata.sampled_ids.is_empty());
        assert_eq!(tld_metadata.buffer_candidates, vec![bindless_id]);
        assert_eq!(
            tld_metadata.image_kinds,
            vec![(
                bindless_id,
                nexium_gpu::texture_manifest::GraphicsTextureImageKind::D2
            )]
        );
        assert!(cfg_uses_texture_descriptors(&tld_cfg));

        let d2_fetch = |y| nexium_shader::IrOp::TexelFetch {
            cbuf_binding: 2,
            cbuf_word_offset: 0x5e,
            cbuf_secondary_word_offset: Some(0x15e),
            x: nexium_shader::IrValue::GprIn(8),
            y: Some(y),
            z: None,
            component: 0,
        };
        let d2_bindless_id = nexium_shader::bindless_texture_id_pair(2, 0x5e, Some(0x15e));
        let mut zero_y_program = nexium_shader::IrProgram::new();
        let zero_y = zero_y_program.emit(
            nexium_shader::IrOp::Mov(nexium_shader::IrValue::Zero),
            Some(9),
        );
        zero_y_program.emit(
            d2_fetch(nexium_shader::IrValue::Inst(zero_y)),
            Some(0),
        );
        let zero_y_cfg = nexium_shader::Cfg {
            blocks: vec![block(zero_y_program)],
            unimplemented: 0,
            bindless_or_partners: std::collections::HashMap::new(),
        };
        let zero_y_metadata = fragment_texture_numeric_metadata(&zero_y_cfg).unwrap();
        assert_eq!(zero_y_metadata.buffer_candidates, vec![d2_bindless_id]);
        assert_eq!(
            zero_y_metadata.image_kinds,
            vec![(
                d2_bindless_id,
                nexium_gpu::texture_manifest::GraphicsTextureImageKind::D2
            )],
            "the resolved TIC selects Buffer; the instruction's fallback family remains D2"
        );

        for y in [
            nexium_shader::IrValue::ImmU32(1),
            nexium_shader::IrValue::GprIn(9),
        ] {
            let mut program = nexium_shader::IrProgram::new();
            program.emit(d2_fetch(y), Some(0));
            let cfg = nexium_shader::Cfg {
                blocks: vec![block(program)],
                unimplemented: 0,
                bindless_or_partners: std::collections::HashMap::new(),
            };
            assert!(fragment_texture_numeric_metadata(&cfg)
                .unwrap()
                .buffer_candidates
                .is_empty());
        }

        let mut mixed_program = nexium_shader::IrProgram::new();
        mixed_program.emit(texel_fetch(0), Some(0));
        mixed_program.emit(
            nexium_shader::IrOp::SampleTex {
                tex_id: 0,
                u: nexium_shader::IrValue::ImmF32(0.0),
                v: nexium_shader::IrValue::ImmF32(0.0),
                array: None,
                volume: None,
                cube: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                dref: None,
                component: 0,
            },
            Some(1),
        );
        let mixed_cfg = nexium_shader::Cfg {
            blocks: vec![block(mixed_program)],
            unimplemented: 0,
            bindless_or_partners: std::collections::HashMap::new(),
        };
        let mixed_metadata = fragment_texture_numeric_metadata(&mixed_cfg).unwrap();
        assert_eq!(mixed_metadata.texel_fetches, Some(vec![(bindless_id, 1)]));
        assert_eq!(mixed_metadata.texture_ids, vec![0, bindless_id]);
        assert_eq!(mixed_metadata.descriptor_ids, vec![0, bindless_id]);
        assert_eq!(mixed_metadata.sampled_ids, vec![0]);
        assert_eq!(mixed_metadata.buffer_candidates, vec![bindless_id]);
        assert_eq!(
            mixed_metadata.image_kinds,
            vec![
                (
                    0,
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::D2
                ),
                (
                    bindless_id,
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::D2
                ),
            ]
        );
    }

    #[test]
    fn pps_captured_vertex_zero_y_ssa_marks_slot1_and_slot10_as_buffer_candidates() {
        for (primary, expected_shader_id) in [(0x5a, 0x9016_815b), (0x5e, 0x9017_815b)] {
            let mut program = nexium_shader::IrProgram::new();
            let zero_y = program.emit(
                nexium_shader::IrOp::Mov(nexium_shader::IrValue::Zero),
                Some(9),
            );
            program.emit(
                nexium_shader::IrOp::TexelFetch {
                    cbuf_binding: 2,
                    cbuf_word_offset: primary,
                    cbuf_secondary_word_offset: Some(0x15a),
                    x: nexium_shader::IrValue::GprIn(8),
                    y: Some(nexium_shader::IrValue::Inst(zero_y)),
                    z: None,
                    component: 0,
                },
                Some(0),
            );
            let cfg = nexium_shader::Cfg {
                blocks: vec![nexium_shader::BasicBlock {
                    id: 0,
                    start_offset: 0,
                    end_offset: 8,
                    branch: nexium_shader::BranchKind::Exit,
                    program,
                    reg_exit: Default::default(),
                    pred_phis: Vec::new(),
                    pred_exit: Default::default(),
                }],
                unimplemented: 0,
                bindless_or_partners: Default::default(),
            };
            let metadata = fragment_texture_numeric_metadata(&cfg).unwrap();

            assert_eq!(
                nexium_shader::bindless_texture_id_pair(2, primary, Some(0x15a)),
                expected_shader_id
            );
            assert_eq!(metadata.buffer_candidates, vec![expected_shader_id]);
        }
    }

    #[test]
    fn graphics_texture_view_metadata_keeps_stage_shape_and_depth_slots() {
        let sample = |tex_id: u32, array: bool, cube: bool, dref: bool| {
            nexium_shader::IrOp::SampleTex {
                tex_id,
                u: nexium_shader::IrValue::ImmF32(0.0),
                v: nexium_shader::IrValue::ImmF32(0.0),
                array: array.then_some(nexium_shader::IrValue::ImmF32(0.0)),
                volume: None,
                cube: cube.then_some(nexium_shader::IrValue::ImmF32(1.0)),
                dref: dref.then_some(nexium_shader::IrValue::ImmF32(0.5)),
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                component: 0,
            }
        };
        let cfg = |ops: Vec<nexium_shader::IrOp>| {
            let mut program = nexium_shader::IrProgram::new();
            for (destination, op) in ops.into_iter().enumerate() {
                program.emit(op, Some(destination as u8));
            }
            nexium_shader::Cfg {
                blocks: vec![nexium_shader::BasicBlock {
                    id: 0,
                    start_offset: 0,
                    end_offset: 8,
                    branch: nexium_shader::BranchKind::Exit,
                    program,
                    reg_exit: std::collections::HashMap::new(),
                    pred_phis: Vec::new(),
                    pred_exit: std::collections::HashMap::new(),
                }],
                unimplemented: 0,
                bindless_or_partners: std::collections::HashMap::new(),
            }
        };

        let fs_metadata =
            fragment_texture_numeric_metadata(&cfg(vec![sample(5, false, false, true)]))
                .unwrap();
        let vs_metadata = fragment_texture_numeric_metadata(&cfg(vec![
            sample(9, true, false, false),
            sample(10, false, true, true),
            sample(11, true, true, true),
        ]))
        .unwrap();
        assert!(!fs_metadata.sampler_arrayed);
        assert!(vs_metadata.sampler_arrayed);
        assert_eq!(fs_metadata.depth_compare_2d_ids, vec![5]);
        assert_eq!(vs_metadata.depth_compare_cube_ids, vec![10]);
        assert_eq!(vs_metadata.depth_compare_cube_array_ids, vec![11]);
        assert_eq!(
            vs_metadata.image_kinds,
            vec![
                (
                    9,
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::D2Array
                ),
                (
                    10,
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::Cube
                ),
                (
                    11,
                    nexium_gpu::texture_manifest::GraphicsTextureImageKind::CubeArray
                ),
            ]
        );

        let layout = graphics_texture_layout_from_metadata(
            &fs_metadata,
            &vs_metadata,
            &[[(0, 0); 16]; 5],
            0,
            0,
            0,
            0,
            false,
            &crate::gpu::GpuMappings::new(),
            &|_, _| false,
        )
        .unwrap();
        assert!(!layout.fs_sampler_arrayed);
        assert!(layout.vs_sampler_arrayed);
        assert_eq!(layout.vs_tex_base, 1);
        assert_eq!(layout.depth_compare_2d_mask, 1 << 0);
        assert_eq!(layout.depth_compare_cube_mask, 1 << 2);
        assert_eq!(layout.depth_compare_cube_array_mask, 1 << 3);
        assert_eq!(
            nexium_gpu::texture_manifest::texture_image_kind_for_slot(&layout.manifest, 3),
            nexium_gpu::texture_manifest::GraphicsTextureImageKind::CubeArray
        );
        assert_ne!(texture_view_metadata_fingerprint(&layout), 0);

        let conflict = fragment_texture_numeric_metadata(&cfg(vec![
            sample(12, false, true, false),
            sample(12, true, true, false),
        ]))
        .unwrap_err();
        assert!(conflict.contains("texture 0xc"), "{conflict}");
        assert!(conflict.contains("Cube"), "{conflict}");
        assert!(conflict.contains("CubeArray"), "{conflict}");
    }

    fn numeric_tic(
        format: nexium_gpu::texture::TicFormat,
        component_types: [nexium_gpu::texture::ComponentType; 4],
        swizzle: [nexium_gpu::texture::SwizzleSource; 4],
    ) -> nexium_gpu::texture::TicEntry {
        nexium_gpu::texture::TicEntry {
            format,
            component_types,
            swizzle,
            gpu_va: 1,
            width: 1,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: false,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        }
    }

    #[test]
    fn cubemap_snapshot_starts_at_tic_address_and_reads_six_faces() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};

        let mut tic = numeric_tic(
            TicFormat::R8G8B8A8,
            [ComponentType::Unorm; 4],
            [SwizzleSource::R; 4],
        );
        tic.texture_type = 3;
        tic.base_layer = 4;

        assert_eq!(tic_snapshot_layer_count(&tic), 6);
        assert!(!tic_can_alias_render_target_view(&tic));
        assert!(tic_can_alias_render_target(tic.format));
    }

    #[test]
    fn cube_array_snapshot_ignores_preceding_base_layers() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};

        let mut tic = numeric_tic(
            TicFormat::R8G8B8A8,
            [ComponentType::Unorm; 4],
            [SwizzleSource::R; 4],
        );
        tic.texture_type = 8;
        tic.base_layer = 6;
        tic.depth = 3;

        assert_eq!(tic_snapshot_layer_count(&tic), 18);
        assert!(!tic_can_alias_render_target_view(&tic));
        assert!(tic_can_alias_render_target(tic.format));
    }

    #[test]
    fn texel_buffer_support_matches_shader_numeric_type_and_view_format() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
        use nexium_spirv::TextureNumericType;

        let mut tic = numeric_tic(
            TicFormat::R16,
            [ComponentType::Uint; 4],
            [
                SwizzleSource::R,
                SwizzleSource::Zero,
                SwizzleSource::Zero,
                SwizzleSource::One,
            ],
        );
        tic.texture_type = 6;
        assert!(supported_texel_buffer(&tic, TextureNumericType::Uint));

        tic.format = TicFormat::R8;
        assert!(!supported_texel_buffer(&tic, TextureNumericType::Uint));
        tic.format = TicFormat::R32G32B32A32;
        tic.component_types = [ComponentType::Float; 4];
        tic.swizzle = [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ];
        assert!(supported_texel_buffer(&tic, TextureNumericType::Float));
        assert!(!supported_texel_buffer(&tic, TextureNumericType::Uint));
    }

    #[test]
    fn pps_mixed_shader_keeps_float_sample_and_r16_uint_fetch_independent() {
        use nexium_gpu::texture::{TicEntry, TicFormat};
        use nexium_gpu::texture_manifest::texture_numeric_type_for_slot;
        use nexium_spirv::TextureNumericType;

        let r16_uint = [
            0x12, 0x92, 0x14, 0x60, 0x00, 0x00, 0x6d, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x7f, 0xbb, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let tic = TicEntry::parse(&r16_uint).unwrap();
        assert!(tic.is_buffer());
        assert_eq!(tic.format, TicFormat::R16);
        assert_eq!(tic_numeric_type(&tic, 1), TextureNumericType::Uint);
        assert_eq!(
            texel_buffer_numeric_type(&tic),
            Some(TextureNumericType::Uint)
        );

        let fs_metadata = FragmentTextureNumericMetadata {
            texel_fetches: Some(vec![(0, 1)]),
            texture_ids: vec![0, 1],
            descriptor_ids: vec![0, 1],
            sampled_ids: vec![1],
            buffer_candidates: vec![0],
            or_partners: Default::default(),
            ..Default::default()
        };
        let vs_metadata = FragmentTextureNumericMetadata::default();
        let cbuf_binds = [[(0, 0); 16]; 5];
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 64, 0x2000, 1);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x2000 || out.len() != r16_uint.len() {
                return false;
            }
            out.copy_from_slice(&r16_uint);
            true
        };
        let layout = graphics_texture_layout_from_metadata(
            &fs_metadata,
            &vs_metadata,
            &cbuf_binds,
            0,
            0,
            0x1000,
            1,
            false,
            &mappings,
            &read,
        )
        .unwrap();
        assert_eq!(layout.fs_ids, vec![0, 1]);
        assert_eq!(layout.fs_texel_buffer_mask, 1);
        assert_eq!(
            texture_numeric_type_for_slot(&layout.manifest, 0),
            TextureNumericType::Uint
        );
        assert_eq!(
            texture_numeric_type_for_slot(&layout.manifest, 1),
            TextureNumericType::Float
        );
    }

    #[test]
    fn d2_tld_with_zero_y_and_buffer_tic_normalizes_manifest_to_buffer() {
        use nexium_gpu::texture::TicEntry;
        use nexium_gpu::texture_manifest::{
            texture_image_kind_for_slot, GraphicsTextureImageKind,
        };
        use nexium_spirv::TextureNumericType;

        let r16_uint = [
            0x12, 0x92, 0x14, 0x60, 0x00, 0x00, 0x6d, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x7f, 0xbb, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        assert!(TicEntry::parse(&r16_uint).unwrap().is_buffer());

        let shader_id = nexium_shader::bindless_texture_id_pair(2, 0x5e, Some(0x15a));
        let mut program = nexium_shader::IrProgram::new();
        let zero_y = program.emit(
            nexium_shader::IrOp::Mov(nexium_shader::IrValue::Zero),
            Some(9),
        );
        program.emit(
            nexium_shader::IrOp::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x5e,
                cbuf_secondary_word_offset: Some(0x15a),
                x: nexium_shader::IrValue::GprIn(8),
                y: Some(nexium_shader::IrValue::Inst(zero_y)),
                z: None,
                component: 0,
            },
            Some(0),
        );
        let fs_metadata = fragment_texture_numeric_metadata(&nexium_shader::Cfg {
            blocks: vec![nexium_shader::BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 8,
                branch: nexium_shader::BranchKind::Exit,
                program,
                reg_exit: std::collections::HashMap::new(),
                pred_phis: Vec::new(),
                pred_exit: std::collections::HashMap::new(),
            }],
            unimplemented: 0,
            bindless_or_partners: std::collections::HashMap::new(),
        })
        .unwrap();

        let mut cbuf_binds = [[(0, 0); 16]; 5];
        cbuf_binds[4][2] = (0x3000, 0x1000);
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 32, 0x2000, 1);
        mappings.add(0x3000, 0x1000, 0x4000, 2);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu == 0x2000 && out.len() == r16_uint.len() {
                out.copy_from_slice(&r16_uint);
                return true;
            }
            if matches!(cpu, 0x4178 | 0x4568) && out.len() == 4 {
                out.fill(0);
                return true;
            }
            false
        };
        let layout = graphics_texture_layout_from_metadata(
            &fs_metadata,
            &FragmentTextureNumericMetadata::default(),
            &cbuf_binds,
            0,
            2,
            0x1000,
            0,
            false,
            &mappings,
            &read,
        )
        .unwrap();

        assert_eq!(layout.fs_ids, vec![shader_id]);
        assert_eq!(layout.fs_texel_buffer_mask, 1);
        assert_eq!(
            texture_image_kind_for_slot(&layout.manifest, 0),
            GraphicsTextureImageKind::Buffer
        );
        assert_eq!(layout.manifest[0].spirv_type(), TextureNumericType::Uint);
    }

    #[test]
    fn graphics_texture_manifest_rejects_same_resource_float_integer_conflict() {
        let r16_uint = [
            0x12, 0x92, 0x14, 0x60, 0x00, 0x00, 0x6d, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x7f, 0xbb, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let fs_metadata = FragmentTextureNumericMetadata {
            texel_fetches: Some(vec![(0, 1)]),
            texture_ids: vec![0],
            descriptor_ids: vec![0],
            sampled_ids: vec![0],
            buffer_candidates: vec![0],
            or_partners: Default::default(),
            ..Default::default()
        };
        let mut mappings = crate::gpu::GpuMappings::new();
        mappings.add(0x1000, 32, 0x2000, 1);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x2000 || out.len() != r16_uint.len() {
                return false;
            }
            out.copy_from_slice(&r16_uint);
            true
        };
        let error = graphics_texture_layout_from_metadata(
            &fs_metadata,
            &FragmentTextureNumericMetadata::default(),
            &[[(0, 0); 16]; 5],
            0,
            0,
            0x1000,
            0,
            false,
            &mappings,
            &read,
        )
        .unwrap_err();
        assert!(error.contains("Sample/Gather (Float) and TexelFetch (Uint)"));
        assert!(error.contains("shader_id=0x0 slot=0 tic_id=0x0"));
    }

    #[test]
    fn graphics_texture_manifest_assigns_fs_then_vs_slots() {
        use nexium_gpu::texture_manifest::texture_numeric_type_for_slot;
        use nexium_spirv::TextureNumericType;

        let fs_metadata = FragmentTextureNumericMetadata {
            texture_ids: vec![5, 9],
            descriptor_ids: vec![5, 9],
            sampled_ids: vec![5, 9],
            ..Default::default()
        };
        let vs_metadata = FragmentTextureNumericMetadata {
            texture_ids: vec![2],
            descriptor_ids: vec![2],
            sampled_ids: vec![2],
            ..Default::default()
        };
        let layout = graphics_texture_layout_from_metadata(
            &fs_metadata,
            &vs_metadata,
            &[[(0, 0); 16]; 5],
            0,
            0,
            0,
            0,
            false,
            &crate::gpu::GpuMappings::new(),
            &|_, _| false,
        )
        .unwrap();
        assert_eq!(layout.fs_ids, vec![5, 9, 2]);
        assert_eq!(layout.vs_tex_base, 2);
        assert_eq!(layout.vs_tex_count, 1);
        assert_eq!(
            layout
                .manifest
                .iter()
                .map(|binding| (binding.shader_id, binding.descriptor_slot))
                .collect::<Vec<_>>(),
            vec![(5, 0), (9, 1), (2, 2)]
        );
        assert_eq!(
            texture_numeric_type_for_slot(&layout.manifest, 2),
            TextureNumericType::Float
        );
    }

    #[test]
    fn font_probe_array_read_starts_at_tic_address() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};

        let mut tic = numeric_tic(
            TicFormat::R8G8B8A8,
            [ComponentType::Unorm; 4],
            [SwizzleSource::R; 4],
        );
        tic.texture_type = 5;
        tic.base_layer = 9;
        tic.depth = 4;
        assert_eq!(font_tic_layer_count(&tic), 4);
    }

    #[test]
    fn g24r8_numeric_type_follows_full_view_aspect() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
        use nexium_spirv::TextureNumericType;

        let component_types = [
            ComponentType::Uint,
            ComponentType::Unorm,
            ComponentType::Unorm,
            ComponentType::Unorm,
        ];
        let target = numeric_tic(TicFormat::G24R8, component_types, [SwizzleSource::R; 4]);
        assert_eq!(tic_numeric_type(&target, 1 << 1), TextureNumericType::Uint);

        let identity = numeric_tic(
            TicFormat::G24R8,
            component_types,
            [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
        );
        assert_eq!(
            tic_numeric_type(&identity, 1 << 1),
            TextureNumericType::Uint
        );

        let depth = numeric_tic(TicFormat::G24R8, component_types, [SwizzleSource::G; 4]);
        assert_eq!(tic_numeric_type(&depth, 1 << 1), TextureNumericType::Float);
    }

    #[test]
    fn ordinary_numeric_type_uses_referenced_swizzled_components() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
        use nexium_spirv::TextureNumericType;

        let tic = numeric_tic(
            TicFormat::R8G8B8A8,
            [
                ComponentType::Uint,
                ComponentType::Unorm,
                ComponentType::Uint,
                ComponentType::Uint,
            ],
            [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::One,
            ],
        );
        assert_eq!(tic_numeric_type(&tic, 1 << 0), TextureNumericType::Uint);
        assert_eq!(tic_numeric_type(&tic, 1 << 1), TextureNumericType::Float);
        assert_eq!(
            tic_numeric_type(&tic, (1 << 0) | (1 << 1)),
            TextureNumericType::Float
        );
        assert_eq!(tic_numeric_type(&tic, 1 << 3), TextureNumericType::Float);
    }

    #[test]
    fn integer_numeric_type_covers_all_native_integer_upload_formats() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
        use nexium_spirv::TextureNumericType;

        for format in [
            TicFormat::R8,
            TicFormat::R8G8,
            TicFormat::R8G8B8A8,
            TicFormat::A8B8G8R8,
            TicFormat::R16,
            TicFormat::R16G16,
            TicFormat::R16G16B16A16,
            TicFormat::R32,
            TicFormat::R32G32,
            TicFormat::R32G32B32A32,
        ] {
            let uint = numeric_tic(
                format,
                [ComponentType::Uint; 4],
                [
                    SwizzleSource::R,
                    SwizzleSource::G,
                    SwizzleSource::B,
                    SwizzleSource::A,
                ],
            );
            assert_eq!(
                tic_numeric_type(&uint, 1 << 0),
                TextureNumericType::Uint,
                "{format:?} must retain unsigned integer texels"
            );
            let sint = numeric_tic(
                format,
                [ComponentType::Sint; 4],
                [
                    SwizzleSource::R,
                    SwizzleSource::G,
                    SwizzleSource::B,
                    SwizzleSource::A,
                ],
            );
            assert_eq!(
                tic_numeric_type(&sint, 1 << 0),
                TextureNumericType::Sint,
                "{format:?} must retain signed integer texels"
            );
        }
    }

    #[test]
    fn packed_depth_stencil_formats_stay_float_without_exact_uploads() {
        use nexium_gpu::texture::{ComponentType, SwizzleSource, TicFormat};
        use nexium_spirv::TextureNumericType;

        let z24s8 = numeric_tic(
            TicFormat::Z24S8,
            [
                ComponentType::Uint,
                ComponentType::Unorm,
                ComponentType::Unorm,
                ComponentType::Unorm,
            ],
            [SwizzleSource::R; 4],
        );
        assert_eq!(tic_numeric_type(&z24s8, 1 << 0), TextureNumericType::Float);

        let s8z24 = numeric_tic(
            TicFormat::S8Z24,
            [
                ComponentType::Unorm,
                ComponentType::Uint,
                ComponentType::Unorm,
                ComponentType::Unorm,
            ],
            [SwizzleSource::G; 4],
        );
        assert_eq!(tic_numeric_type(&s8z24, 1 << 0), TextureNumericType::Float);
    }

    #[test]
    fn pps_zeta_format_maps_to_combined_d24s8() {
        let (format, aspects) = map_zeta_format(0x14);
        assert_eq!(format, vk::Format::D24_UNORM_S8_UINT);
        assert_eq!(
            aspects,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn depth_write_without_depth_test_does_not_activate_zeta() {
        assert_eq!(
            effective_depth_states(false, true, false, true, false),
            (false, false, false)
        );
        assert_eq!(
            effective_depth_states(false, true, true, true, false),
            (true, true, false)
        );
        assert_eq!(
            effective_depth_states(false, true, false, true, true),
            (false, false, true)
        );
        assert_eq!(
            effective_depth_states(true, true, true, true, true),
            (false, false, false)
        );
    }

    #[test]
    fn maxwell_stencil_ops_match_yuzu_gl_and_d3d_encodings() {
        for (raw, expected) in [
            (1, vk::StencilOp::KEEP),
            (0x1e00, vk::StencilOp::KEEP),
            (2, vk::StencilOp::ZERO),
            (0, vk::StencilOp::ZERO),
            (3, vk::StencilOp::REPLACE),
            (0x1e01, vk::StencilOp::REPLACE),
            (4, vk::StencilOp::INCREMENT_AND_CLAMP),
            (0x1e02, vk::StencilOp::INCREMENT_AND_CLAMP),
            (5, vk::StencilOp::DECREMENT_AND_CLAMP),
            (0x1e03, vk::StencilOp::DECREMENT_AND_CLAMP),
            (6, vk::StencilOp::INVERT),
            (0x150a, vk::StencilOp::INVERT),
            (7, vk::StencilOp::INCREMENT_AND_WRAP),
            (0x8507, vk::StencilOp::INCREMENT_AND_WRAP),
            (8, vk::StencilOp::DECREMENT_AND_WRAP),
            (0x8508, vk::StencilOp::DECREMENT_AND_WRAP),
        ] {
            assert_eq!(map_stencil_op(raw), expected, "raw={raw:#x}");
        }
    }
}
