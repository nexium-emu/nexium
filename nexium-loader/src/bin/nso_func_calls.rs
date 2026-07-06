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

#[derive(Clone)]
struct Sym {
    abs: u64,
    value: u64,
    size: u64,
    name: String,
}

struct Module {
    name: String,
    base: u64,
    text_start: usize,
    text_end: usize,
    image: Vec<u8>,
    syms: Vec<Sym>,
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

fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

fn branch_target(pc: u64, op: u32) -> Option<(&'static str, u64)> {
    if op & 0xfc00_0000 == 0x9400_0000 {
        let imm = sign_extend((op & 0x03ff_ffff) as u64, 26) << 2;
        return Some(("bl", pc.wrapping_add(imm as u64)));
    }
    if op & 0xfc00_0000 == 0x1400_0000 {
        let imm = sign_extend((op & 0x03ff_ffff) as u64, 26) << 2;
        return Some(("b", pc.wrapping_add(imm as u64)));
    }
    if op & 0xff00_0010 == 0x5400_0000 {
        let imm = sign_extend(((op >> 5) & 0x7ffff) as u64, 19) << 2;
        return Some(("b.cond", pc.wrapping_add(imm as u64)));
    }
    if op & 0x7f00_0000 == 0x3400_0000 {
        let imm = sign_extend(((op >> 5) & 0x7ffff) as u64, 19) << 2;
        return Some(("cbz/cbnz", pc.wrapping_add(imm as u64)));
    }
    if op & 0x7f00_0000 == 0x3600_0000 {
        let imm = sign_extend(((op >> 5) & 0x3fff) as u64, 14) << 2;
        return Some(("tbz/tbnz", pc.wrapping_add(imm as u64)));
    }
    None
}

fn find_sym(syms: &[Sym], abs: u64) -> Option<usize> {
    let pos = syms.partition_point(|s| s.abs <= abs);
    if pos == 0 {
        return None;
    }
    let idx = pos - 1;
    let s = &syms[idx];
    let end = if s.size == 0 { s.abs + 4 } else { s.abs + s.size };
    if abs >= s.abs && abs < end {
        Some(idx)
    } else {
        None
    }
}

fn hex_arg(s: &str) -> Option<u64> {
    let s = s.trim_start_matches("0x");
    u64::from_str_radix(s, 16).ok()
}

fn name_for(modules: &[Module], abs: u64) -> String {
    for m in modules {
        if let Some(idx) = find_sym(&m.syms, abs) {
            let s = &m.syms[idx];
            return format!("{}+{:#x}", s.name, abs - s.abs);
        }
    }
    "<no-symbol>".to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nso_func_calls <app.dxci> <symbol-substring-or-hex> [...]");
        std::process::exit(2);
    }
    let app = Application::load(&args[1]).expect("load application");
    let mut modules = Vec::new();
    for m in &app.modules {
        let base = CODE_BASE + m.load_offset;
        let image = m.nso.module_image.clone();
        let syms = parse_syms(base, &image);
        modules.push(Module {
            name: m.name.clone(),
            base,
            text_start: m.nso.text.mem_offset as usize,
            text_end: m.nso.text.mem_offset as usize + m.nso.text.decompressed_size as usize,
            image,
            syms,
        });
    }
    for needle in &args[2..] {
        let lower = needle.to_ascii_lowercase();
        let mut targets = Vec::new();
        for (mi, m) in modules.iter().enumerate() {
            if let Some(addr) = hex_arg(needle) {
                if let Some(si) = find_sym(&m.syms, addr) {
                    targets.push((mi, si));
                }
            } else {
                for (si, s) in m.syms.iter().enumerate() {
                    if s.name.to_ascii_lowercase().contains(&lower) {
                        targets.push((mi, si));
                    }
                }
            }
        }
        for (mi, si) in targets {
            let m = &modules[mi];
            let s = &m.syms[si];
            println!(
                "function {:#010x} {}+{:#x} size={:#x} {}",
                s.abs, m.name, s.value, s.size, s.name
            );
            let mut off = s.value as usize;
            let end = (s.value + s.size).min(m.text_end as u64) as usize;
            while off + 4 <= end && off + 4 <= m.image.len() {
                if off >= m.text_start {
                    let pc = m.base + off as u64;
                    let op = u32at(&m.image, off);
                    if let Some((kind, dst)) = branch_target(pc, op) {
                        if kind == "bl" || (dst >= s.abs && dst < s.abs + s.size) {
                            println!(
                                "  {:#010x} {:<8} -> {:#010x} {}",
                                pc,
                                kind,
                                dst,
                                name_for(&modules, dst)
                            );
                        }
                    }
                }
                off += 4;
            }
        }
    }
}
