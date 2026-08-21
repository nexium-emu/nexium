use nexium_spirv::{emit_compute, ComputeOptions, COMPUTE_CBUF_SLOTS};

fn main() {
    let path = std::env::args().nth(1).expect("usage: csdis <dumpfile>");
    let bytes = std::fs::read(&path).expect("read dump");
    let mut cfg = nexium_shader::cfg::build_compute_cfg(&bytes);
    let buffers = nexium_shader::cfg::collect_storage_buffers(&mut cfg);
    let options = ComputeOptions {
        local_size: [1, 1, 1],
        local_memory_low_size: 0,
        local_memory_high_size: 0,
        local_memory_crs_size: 0,
        shared_memory_size: 0,
        texture_bound_cbuf: 0,
        cbuf_sizes: [65536; COMPUTE_CBUF_SLOTS],
        num_storage_buffers: buffers.len() as u32,
        resources: Vec::new(),
        big_warp: false,
    };
    let module = emit_compute(&cfg, &options).expect("emit compute");
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(&module.words, &mut loader).expect("parse words");
    use rspirv::binary::Disassemble;
    println!("{}", loader.module().disassemble());
}
