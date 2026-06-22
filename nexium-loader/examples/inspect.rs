use nexium_loader::{LoadedProgram, Loader};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: inspect <file.dxci|file.dnsp|file.nro>");
            std::process::exit(2);
        }
    };

    match Loader::load_any(&path) {
        Ok(LoadedProgram::Nro(nro)) => {
            println!(
                "NRO: image_size={:#x} romfs={}",
                nro.total_memory_size(),
                nro.romfs_range().is_some()
            );
        }
        Ok(LoadedProgram::Application(app)) => {
            println!("APPLICATION title_id={:#018x}", app.title_id);
            println!(
                "  address_space={:?} 64bit={} stack={:#x} sysres={:#x}",
                app.npdm.address_space,
                app.npdm.is_64bit,
                app.npdm.main_stack_size,
                app.npdm.system_resource_size
            );
            println!(
                "  code_size={:#x} modules={}",
                app.total_code_size,
                app.modules.len()
            );
            let needle = b"__nnDetailInitLibc0";
            for m in &app.modules {
                let img = &m.nso.module_image;
                let b0 = u32::from_le_bytes([img[0], img[1], img[2], img[3]]);
                let mod0_off = u32::from_le_bytes([img[4], img[5], img[6], img[7]]) as usize;
                let mod0_ok = img.get(mod0_off..mod0_off + 4) == Some(b"MOD0");
                let has_sym = img.windows(needle.len()).any(|w| w == needle);
                println!("    {:<8} +{:#010x} image={:#x} text={:#x} ro={:#x} data={:#x} bss={:#x} | entry_word={:#010x} mod0@{:#x}={} sym={}",
                    m.name, m.load_offset, m.nso.image_size, m.nso.text.decompressed_size,
                    m.nso.ro.decompressed_size, m.nso.data.decompressed_size, m.nso.bss_size,
                    b0, mod0_off, mod0_ok, has_sym);
            }
            match &app.romfs {
                Some(r) => {
                    let head = r.as_slice();
                    let hdr_size = if head.len() >= 4 {
                        u32::from_le_bytes([head[0], head[1], head[2], head[3]])
                    } else {
                        0
                    };
                    println!(
                        "  romfs {} bytes, header_size={:#x} (expect 0x50)",
                        r.len(),
                        hdr_size
                    );
                }
                None => println!("  romfs: none"),
            }
        }
        Err(e) => {
            eprintln!("load failed: {}", e);
            std::process::exit(1);
        }
    }
}
