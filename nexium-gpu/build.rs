fn main() {
    compile(
        "present",
        &[
            ("vertex", naga::ShaderStage::Vertex),
            ("fragment", naga::ShaderStage::Fragment),
        ],
    );
    compile("depth_pack", &[("main", naga::ShaderStage::Compute)]);
}

fn compile(shader: &str, entries: &[(&str, naga::ShaderStage)]) {
    let path = format!("src/{shader}.wgsl");
    println!("cargo:rerun-if-changed={path}");
    let mut source = std::fs::read_to_string(&path).unwrap();
    if shader == "present" {
        println!("cargo:rerun-if-changed=src/scale.wgsl");
        source.push_str(&std::fs::read_to_string("src/scale.wgsl").unwrap());
    }
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for &(name, shader_stage) in entries {
        let options = naga::back::spv::Options {
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };
        let pipeline = naga::back::spv::PipelineOptions {
            shader_stage,
            entry_point: name.into(),
        };
        let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).unwrap();
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        std::fs::write(output.join(format!("{shader}_{name}.spv")), bytes).unwrap();
    }
}
