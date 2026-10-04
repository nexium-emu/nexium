fn main() {
    let path = "src/probe.wgsl";
    println!("cargo:rerun-if-changed={path}");
    let source = std::fs::read_to_string(path).unwrap();
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    for (entry, stage) in [
        ("vs_main", naga::ShaderStage::Vertex),
        ("fs_main", naga::ShaderStage::Fragment),
        ("cs_main", naga::ShaderStage::Compute),
    ] {
        let options = naga::back::spv::Options { flags: naga::back::spv::WriterFlags::empty(), ..Default::default() };
        let pipeline = naga::back::spv::PipelineOptions { shader_stage: stage, entry_point: entry.into() };
        let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).unwrap();
        let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        std::fs::write(output.join(format!("probe_{entry}.spv")), bytes).unwrap();
    }
}
