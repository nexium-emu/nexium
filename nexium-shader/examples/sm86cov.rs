use std::collections::BTreeMap;

use nexium_shader::ir::ShaderStage;

fn stage(header: &[u8]) -> ShaderStage {
    match u32::from_le_bytes(header[..4].try_into().unwrap()) >> 10 & 0xF {
        1 => ShaderStage::Vertex,
        4 => ShaderStage::Geometry,
        5 => ShaderStage::Fragment,
        _ => ShaderStage::Compute,
    }
}

fn jump_table(bytes: &[u8]) -> impl FnMut(u8, u32) -> Option<u32> {
    let brx = bytes[0x80..].chunks_exact(16).position(|word| u16::from_le_bytes([word[0], word[1]]) & 0xFFF == 0x949);
    move |_, offset| brx.map(|at| ((at + 1 + offset as usize / 4 % 3) * 16) as u32)
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: sm86cov <dir with .sm86 files>");
    let verbose = std::env::var_os("SM86COV_VERBOSE").is_some();
    let mut total = 0;
    let mut lowering_failed = 0;
    let mut native = 0;
    let mut native_ops = 0usize;
    let mut lowered_ops = 0usize;
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
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
        let Ok(lowered) = nexium_shader::ampere::translate(&bytes) else {
            lowering_failed += 1;
            continue;
        };
        total += 1;
        match nexium_shader::cfg::sm86_native_probe_with_cbuf(&bytes[0x80..], stage(&bytes), &mut jump_table(&bytes)) {
            Ok(cfg) => {
                native += 1;
                let maxwell = nexium_shader::build_cfg(&lowered.code);
                let count = |cfg: &nexium_shader::Cfg| cfg.blocks.iter().map(|block| block.program.instructions.len()).sum::<usize>();
                native_ops += count(&cfg);
                lowered_ops += count(&maxwell);
            }
            Err(reason) => {
                if verbose {
                    println!("{} {reason}", path.display());
                }
                let key = reason.split(" (").next().unwrap_or(&reason).to_string();
                *failures.entry(key).or_default() += 1;
            }
        }
    }
    println!("native {native}/{total} (lowering also fails on {lowering_failed})");
    println!("IR instructions on natively translated shaders: native {native_ops}, via Maxwell {lowered_ops} ({:.1}% fewer)", 100.0 * (1.0 - native_ops as f64 / lowered_ops.max(1) as f64));
    let mut sorted: Vec<_> = failures.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, count) in sorted {
        println!("{count:6} {reason}");
    }
}
