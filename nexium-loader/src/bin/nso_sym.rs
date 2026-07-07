use nexium_loader::application::Application;

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
    value: u64,
    size: u64,
    name: String,
}

struct Module {
    name: String,
    base: u64,
    image_len: u64,
    syms: Vec<Sym>,
}

fn parse_syms(img: &[u8]) -> Vec<Sym> {
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

    let (mut symtab, mut strtab, mut strsz, mut hash) = (0usize, 0usize, 0usize, 0usize);
    let mut p = dyn_off;
    while p + 16 <= img.len() {
        let tag = u64at(img, p);
        let val = u64at(img, p + 8);
        p += 16;
        match tag {
            0 => break,
            4 => hash = val as usize,   // DT_HASH
            5 => strtab = val as usize, // DT_STRTAB
            6 => symtab = val as usize, // DT_SYMTAB
            10 => strsz = val as usize, // DT_STRSZ
            _ => {}
        }
    }
    if symtab == 0 || strtab == 0 {
        return out;
    }
    let _ = strsz;

    let count = if hash != 0 && hash + 8 <= img.len() {
        u32at(img, hash + 4) as usize // nchain == symbol count
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
            continue; // undefined / imported
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
        if name.is_empty() {
            continue;
        }
        out.push(Sym { value, size, name });
    }
    out.sort_by_key(|s| s.value);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nso_sym <app.dxci> <addr> [addr...]   (addr in hex)");
        std::process::exit(2);
    }
    let app = Application::load(&args[1]).expect("load application");

    let mods: Vec<Module> = app
        .modules
        .iter()
        .map(|m| {
            let base = CODE_BASE + m.load_offset;
            let syms = parse_syms(&m.nso.module_image);
            println!(
                "module {:<8} base={:#x} image={:#x} syms={}",
                m.name,
                base,
                m.nso.image_size,
                syms.len()
            );
            Module {
                name: m.name.clone(),
                base,
                image_len: m.nso.image_size as u64,
                syms,
            }
        })
        .collect();

    println!("----");
    for a in &args[2..] {
        let addr = u64::from_str_radix(a.trim_start_matches("0x"), 16).unwrap_or(0);
        let m = mods
            .iter()
            .find(|m| addr >= m.base && addr < m.base + m.image_len);
        match m {
            None => println!("{:#x}  <no module>", addr),
            Some(m) => {
                let off = addr - m.base;
                let best = m.syms.iter().rev().find(|s| s.value <= off);
                match best {
                    Some(s) => {
                        let delta = off - s.value;
                        let within = s.size == 0 || delta < s.size;
                        println!(
                            "{:#x}  {}+{:#x}  {}{}",
                            addr,
                            m.name,
                            off,
                            s.name,
                            if within {
                                format!("+{:#x}", delta)
                            } else {
                                format!(" (+{:#x}, past sym size {:#x})", delta, s.size)
                            }
                        );
                    }
                    None => println!("{:#x}  {}+{:#x}  <no symbol>", addr, m.name, off),
                }
            }
        }
    }
}
