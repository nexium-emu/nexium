use nexium_spirv::{emit_geometry, GeometryOptions, VertexOptions};

#[test]
fn geometry_shader_emits_triangles_with_arrayed_inputs() {
    let code = [
        0u64, 0xefd800800707ff00, 0xeff000000707ff00, 0xfbe000800ff70000,
        0, 0xfbe001000ff70000, 0xe30000000007000f,
    ].into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
    let code = std::env::var_os("NEXIUM_TEST_GS_DUMP")
        .map(|path| std::fs::read(path).unwrap()).unwrap_or(code);
    let cfg = nexium_shader::cfg::build_geometry_cfg(&code);
    let unsupported: Vec<_> = cfg.blocks.iter().flat_map(|b| &b.program.instructions)
        .filter(|inst| matches!(inst.op, nexium_shader::IrOp::Unimplemented { .. })).collect();
    assert!(unsupported.is_empty(), "{unsupported:?}");
    assert_eq!(cfg.unimplemented, 0);
    let (words, _) = emit_geometry(&cfg, &[], VertexOptions::default(), GeometryOptions::default());
    let module = rspirv::dr::load_words(&words).unwrap();
    assert!(module.entry_points.iter().any(|entry| entry.operands[0] ==
        rspirv::dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Geometry)));
    let emitted = module.all_inst_iter().filter(|i| i.class.opcode == rspirv::spirv::Op::EmitVertex).count();
    assert!(emitted > 0);
    let file = std::env::temp_dir().join(format!("nexium-gs-{}.spv", std::process::id()));
    std::fs::write(&file, words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let result = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&file)
        .output();
    std::fs::remove_file(file).unwrap();
    match result {
        Ok(output) => assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("spirv-val: {error}"),
    }
}
