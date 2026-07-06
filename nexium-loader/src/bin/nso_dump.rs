use nexium_loader::application::Application;
use std::io::Write;
use std::path::PathBuf;

const CODE_BASE: u64 = 0x8000000;
const MOD0_MAGIC: u32 = 0x30444F4D;

fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

struct Sym {
    abs: u64,
    value: u64,
    size: u64,
    name: String,
}

fn parse_syms(base: u64, img: &[u8]) -> Vec<Sym> {
    let mut out = Vec::new();
    if img.len() < 8 {
        return out;
    }
    let mut mod0 = u32at(img, 4) as usize;
    if mod0 + 4 > img.len() || u32at(img, mod0) != MOD0_MAGIC {
        mod0 = usize::MAX;
        let scan = img.len().min(0x2000);
        for o in (0..scan).step_by(4) {
            if u32at(img, o) == MOD0_MAGIC {
                mod0 = o;
                break;
            }
        }
        if mod0 == usize::MAX {
            return out;
        }
    }
    let dyn_off = (mod0 as i64 + u32at(img, mod0 + 4) as i32 as i64) as usize;
    let (mut symtab, mut strtab, mut hash) = (0usize, 0usize, 0usize);
    let mut p = dyn_off;
    while p + 16 <= img.len() {
        let tag = u64at(img, p);
        let val = u64at(img, p + 8);
        p += 16;
        match tag {
            0 => break,
            4 => hash = val as usize,
            5 => strtab = val as usize,
            6 => symtab = val as usize,
            _ => {}
        }
    }
    if symtab == 0 || strtab == 0 {
        return out;
    }
    let count = if hash != 0 && hash + 8 <= img.len() {
        u32at(img, hash + 4) as usize
    } else if strtab > symtab {
        (strtab - symtab) / 24
    } else {
        0
    };
    for i in 0..count {
        let s = symtab + i * 24;
        if s + 24 > img.len() {
            break;
        }
        let st_name = u32at(img, s) as usize;
        let st_shndx = u16::from_le_bytes([img[s + 6], img[s + 7]]);
        let value = u64at(img, s + 8);
        let size = u64at(img, s + 16);
        if value == 0 || st_shndx == 0 {
            continue;
        }
        let n0 = strtab + st_name;
        if n0 >= img.len() {
            continue;
        }
        let end = img[n0..]
            .iter()
            .position(|&c| c == 0)
            .map(|e| n0 + e)
            .unwrap_or(img.len());
        let name = String::from_utf8_lossy(&img[n0..end]).into_owned();
        if !name.is_empty() {
            out.push(Sym {
                abs: base + value,
                value,
                size,
                name,
            });
        }
    }
    out.sort_by_key(|s| s.value);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: nso_dump <app.dxci> <module-name> <out-dir>");
        std::process::exit(2);
    }
    let app = Application::load(&args[1]).expect("load application");
    let module_name = &args[2];
    let out_dir = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let m = app
        .modules
        .iter()
        .find(|m| m.name == *module_name)
        .expect("module not found");
    let base = CODE_BASE + m.load_offset;
    let image = &m.nso.module_image;
    let image_path = out_dir.join(format!("{}.image.bin", m.name));
    let text_path = out_dir.join(format!("{}.text.bin", m.name));
    let syms_path = out_dir.join(format!("{}.syms.txt", m.name));
    std::fs::write(&image_path, image).expect("write image");
    let text_start = m.nso.text.mem_offset as usize;
    let text_end = text_start + m.nso.text.decompressed_size as usize;
    std::fs::write(&text_path, &image[text_start..text_end]).expect("write text");
    let mut syms_file = std::fs::File::create(&syms_path).expect("create syms");
    for s in parse_syms(base, image) {
        writeln!(
            syms_file,
            "{:#010x} value={:#x} size={:#x} {}",
            s.abs, s.value, s.size, s.name
        )
        .expect("write sym");
    }
    println!(
        "{} base={:#x} image={} text={} text_va={:#x}..{:#x} syms={}",
        m.name,
        base,
        image_path.display(),
        text_path.display(),
        base + text_start as u64,
        base + text_end as u64,
        syms_path.display()
    );
}
