use nexium_shader::cfg::{build_compute_cfg, collect_storage_buffers};

fn main() {
    let path = std::env::args().nth(1).expect("usage: rawdis <dumpfile>");
    let bytes = std::fs::read(&path).expect("read dump");
    let mut cfg = build_compute_cfg(&bytes);
    let buffers = collect_storage_buffers(&mut cfg);
    println!("unimplemented={}", cfg.unimplemented);
    for (index, buffer) in buffers.iter().enumerate() {
        println!("ssbo{index}: {buffer:?}");
    }
    for block in &cfg.blocks {
        println!(
            "block {} [{:#x}..{:#x}] branch={:?}",
            block.id, block.start_offset, block.end_offset, block.branch
        );
        for inst in &block.program.instructions {
            let dest = inst
                .result
                .map(|value| format!("v{} = ", value.0))
                .unwrap_or_default();
            let pred = inst
                .pred
                .map(|p| format!("@{}p{} ", if p.negate { "!" } else { "" }, p.idx))
                .unwrap_or_default();
            println!("  {pred}{dest}{:?}", inst.op);
        }
    }
}
