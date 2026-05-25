use std::sync::Arc;
use ash::vk;

use super::engines::Maxwell3D;
use super::engines::maxwell3d::{DrawCall, VertexBuffer};
use super::GpuMappings;

use nexium_gpu::draw::{Maxwell3dDrawCall, VertexAttr, VertexBinding, VertexLayout, DrawState, BlendState, DepthState};
use nexium_gpu::rt_cache::RtKey;

const SPH_SIZE: usize = 48;
const MAX_SASS_BYTES: usize = 16 * 1024;

pub fn try_vulkan_draws(
    draws: &[DrawCall],
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    renderer: &Arc<nexium_gpu::Renderer>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<(), String> {
    for draw in draws {
        if draw.draw_texture.is_some() {
            return Err("DrawTexture path not handled in vk_dispatch".to_string());
        }
        execute_one(draw, mappings, maxwell, renderer, mem_read)?;
    }
    Ok(())
}

fn execute_one(
    draw: &DrawCall,
    mappings: &GpuMappings,
    maxwell: &Maxwell3D,
    renderer: &Arc<nexium_gpu::Renderer>,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<(), String> {
    let rt = &draw.rt[0];
    if rt.width == 0 || rt.height == 0 {
        return Err("RT[0] has zero extent".to_string());
    }
    let rt_gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
    let nvmap_id = mappings
        .nvmap_id_for(rt_gpu_va)
        .ok_or_else(|| format!("RT gpu_va={:#x} not mapped", rt_gpu_va))?;
    let rt_key = RtKey { nvmap_id, width: rt.width, height: rt.height };

    if draw.is_clear {
        return renderer.clear_target(
            nvmap_id,
            rt.width,
            rt.height,
            [draw.clear_color.r, draw.clear_color.g, draw.clear_color.b, draw.clear_color.a],
        );
    }

    let program_region = ((maxwell.regs.program_region_va_hi as u64) << 32)
        | maxwell.regs.program_region_va_lo as u64;

    let vs_prog = &maxwell.regs.shader_programs[1];
    let fs_prog = &maxwell.regs.shader_programs[5];
    if !vs_prog.enabled || !fs_prog.enabled {
        return Err("VS or FS program disabled".to_string());
    }

    let vs_sass = fetch_sass(program_region.wrapping_add(vs_prog.address_lo as u64), mappings, mem_read)
        .ok_or_else(|| "VS SASS read failed".to_string())?;
    let fs_sass = fetch_sass(program_region.wrapping_add(fs_prog.address_lo as u64), mappings, mem_read)
        .ok_or_else(|| "FS SASS read failed".to_string())?;

    let vs_cfg = nexium_shader::build_cfg(&vs_sass);
    let fs_cfg = nexium_shader::build_cfg(&fs_sass);

    let (fs_spirv, fs_cbuf_mask, _fs_tex_ids) = nexium_spirv::emit_fragment_full(&fs_cfg);
    let required_outputs = nexium_spirv::scan_input_locations(&fs_spirv);
    let (vs_spirv, vs_cbuf_mask) = nexium_spirv::emit_vertex_with_bindings_opts(
        &vs_cfg,
        &required_outputs,
        nexium_spirv::VertexOptions::default(),
    );

    let layout = build_vertex_layout(draw)?;
    let topology = map_topology(draw.topology)
        .ok_or_else(|| format!("unsupported topology {}", draw.topology))?;

    let (cbuf_addr, cbuf_size) = resolve_cbuf(draw, &maxwell.regs.cbuf_binds);

    let vertex_addr = first_vertex_buffer_address(&draw.vertex_buffers, &layout)
        .ok_or_else(|| "no vertex buffer bound".to_string())?;

    let call = Maxwell3dDrawCall {
        vs_spirv,
        fs_spirv,
        vs_cbuf_mask,
        fs_cbuf_mask,
        fs_tex_ids: Vec::new(),
        vertex_layout: layout,
        cbuf_addr,
        cbuf_size,
        vertex_addr,
        vertex_count: draw.vertex_count,
        index_addr: None,
        index_count: None,
        index_type: vk::IndexType::UINT16,
        rt_key,
        rt_format: vk::Format::R8G8B8A8_UNORM,
        state: DrawState {
            topology,
            vertex_count: draw.vertex_count,
            index_count: 0,
            indexed: false,
        },
        blend: BlendState {
            enabled: false,
            src_factor: vk::BlendFactor::ONE,
            dst_factor: vk::BlendFactor::ZERO,
            op: vk::BlendOp::ADD,
        },
        depth: DepthState {
            test_enabled: false,
            write_enabled: false,
            compare_op: vk::CompareOp::ALWAYS,
        },
        clear: false,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };

    renderer.execute_draw(&call, |gpu_va, len| {
        let cpu = mappings.cpu_address_for(gpu_va)?;
        let mut buf = vec![0u8; len];
        if mem_read(cpu, &mut buf) { Some(buf) } else { None }
    })
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

    for (loc, attrib) in draw.vertex_attribs.iter().enumerate() {
        if attrib.format == 0 {
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
    Ok(VertexLayout { bindings, attrs })
}

fn map_attrib_format(format: u32) -> Option<vk::Format> {
    let size = format & 0x3F;
    let type_ = format >> 6;
    match (size, type_) {
        (0x01, 7) | (0x12, 7) => Some(vk::Format::R32G32B32A32_SFLOAT),
        (0x04, 7) => Some(vk::Format::R32G32_SFLOAT),
        (0x05, 7) => Some(vk::Format::R32_SFLOAT),
        (0x03, 7) => Some(vk::Format::R32G32B32_SFLOAT),
        (0x0A, 2) => Some(vk::Format::R8G8B8A8_UNORM),
        (0x09, 7) => Some(vk::Format::R16G16_SNORM),
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

fn first_vertex_buffer_address(
    vertex_buffers: &[VertexBuffer; 32],
    layout: &VertexLayout,
) -> Option<u64> {
    let first_binding = layout.bindings.first()?.binding as usize;
    let vb = vertex_buffers.get(first_binding)?;
    let va = ((vb.address_hi as u64) << 32) | vb.address_lo as u64;
    if va == 0 { None } else { Some(va) }
}
