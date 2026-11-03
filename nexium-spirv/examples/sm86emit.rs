use std::panic::{catch_unwind, AssertUnwindSafe};

use nexium_shader::ImageDimension;
use nexium_spirv::{
    emit_compute, ComputeEmitError, ComputeImageResource, ComputeOptions, ComputeResourceKind, ComputeTexelFormat,
    TextureNumericType, COMPUTE_CBUF_SLOTS,
};

fn jump_table(bytes: &[u8]) -> impl FnMut(u8, u32) -> Option<u32> {
    let brx = bytes[0x80..].chunks_exact(16).position(|word| u16::from_le_bytes([word[0], word[1]]) & 0xFFF == 0x949);
    move |_, offset| brx.map(|at| ((at + 1 + offset as usize / 4 % 3) * 16) as u32)
}

fn emit(bytes: &[u8], lowered: &[u8]) -> Option<Vec<u32>> {
    match u32::from_le_bytes(bytes[..4].try_into().unwrap()) >> 10 & 0xF {
        1 => Some(nexium_spirv::emit_vertex(&nexium_shader::build_cfg_with_cbuf(lowered, jump_table(bytes)))),
        5 => Some(nexium_spirv::emit_fragment(&nexium_shader::build_fragment_cfg_with_cbuf(lowered, jump_table(bytes)))),
        4 => None,
        _ => {
            let mut cfg = nexium_shader::build_compute_cfg_with_cbuf(lowered, jump_table(bytes));
            let buffers = nexium_shader::collect_storage_buffers(&mut cfg);
            let mut options = ComputeOptions {
                local_size: [256, 1, 1],
                vertex_memory_store: false,
                local_memory_low_size: 0,
                local_memory_high_size: 0,
                local_memory_crs_size: 0,
                shared_memory_size: 0x8000,
                texture_bound_cbuf: 0,
                cbuf_sizes: [65536; COMPUTE_CBUF_SLOTS],
                num_storage_buffers: buffers.len() as u32,
                resources: Vec::new(),
                big_warp: std::env::var_os("BIG_WARP").is_some(),
            };
            for _ in 0..32 {
                match emit_compute(&cfg, &options) {
                    Ok(module) => return Some(module.words),
                    Err(ComputeEmitError::MissingResource { handle, kind }) => {
                        let buffer = matches!(kind, ComputeResourceKind::StorageTexelBuffer | ComputeResourceKind::UniformTexelBuffer);
                        options.resources.push(ComputeImageResource {
                            handle,
                            binding: 16 + options.resources.len() as u32,
                            kind,
                            dimension: if buffer { ImageDimension::Buffer } else { ImageDimension::D2 },
                            numeric_type: TextureNumericType::Float,
                            texel_format: matches!(kind, ComputeResourceKind::StorageTexelBuffer).then_some(ComputeTexelFormat::R32Float),
                        });
                    }
                    Err(ComputeEmitError::DimensionMismatch { handle, instruction, .. }) => {
                        for resource in options.resources.iter_mut().filter(|resource| resource.handle == handle) {
                            resource.dimension = instruction;
                        }
                    }
                    Err(ComputeEmitError::InvalidResourceDimension { handle, kind: ComputeResourceKind::StorageTexelBuffer, .. }) => {
                        for resource in options.resources.iter_mut().filter(|resource| resource.handle == handle) {
                            resource.kind = ComputeResourceKind::StorageImage;
                        }
                    }
                    Err(ComputeEmitError::UnsupportedOperation(message)) if message.contains("instead of Uint") && options.resources.iter().any(|resource| resource.numeric_type == TextureNumericType::Float) => {
                        for resource in &mut options.resources {
                            resource.numeric_type = TextureNumericType::Uint;
                            resource.texel_format = resource.texel_format.map(|_| ComputeTexelFormat::R32Uint);
                        }
                    }
                    Err(error) => {
                        eprintln!("compute error: {error:?}");
                        return None;
                    }
                }
            }
            None
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: sm86emit <dir with .sm86 files> <output dir>");
    let output = std::path::PathBuf::from(args.next().expect("output dir"));
    std::fs::create_dir_all(&output).unwrap();
    let (mut emitted, mut skipped, mut panicked) = (0, 0, 0);
    let mut paths: Vec<_> = std::fs::read_dir(&input)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "sm86"))
        .collect();
    paths.sort();
    for path in paths {
        let bytes = std::fs::read(&path).unwrap();
        if bytes.len() <= 0x80 {
            continue;
        }
        let Ok(lowered) = nexium_shader::ampere::translate_cached(0, &bytes) else {
            continue;
        };
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        match catch_unwind(AssertUnwindSafe(|| emit(&bytes, &lowered.code))) {
            Ok(Some(words)) => {
                emitted += 1;
                std::fs::write(output.join(format!("{name}.spv")), bytemuck::cast_slice(&words)).unwrap();
            }
            Ok(None) => skipped += 1,
            Err(_) => {
                panicked += 1;
                println!("panic {name}");
            }
        }
    }
    println!("emitted {emitted}, skipped {skipped}, panicked {panicked}");
}
