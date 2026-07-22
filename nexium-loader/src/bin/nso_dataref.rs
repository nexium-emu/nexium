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
    abs: u64,
    name: String,
    size: u64,
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
        for o in (0..img.len().min(0x2000)).step_by(4) {
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
                name,
                size,
            });
        }
    }
    out.sort_by_key(|s| s.abs);
    out
}

fn find_sym(syms: &[Sym], abs: u64) -> Option<String> {
    let pos = syms.partition_point(|s| s.abs <= abs);
    if pos == 0 {
        return None;
    }
    let s = &syms[pos - 1];
    let end = if s.size == 0 {
        s.abs + 4
    } else {
        s.abs + s.size
    };
    if abs >= s.abs && abs < end {
        Some(format!("{}+{:#x}", s.name, abs - s.abs))
    } else {
        None
    }
}

fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

fn find_import_slots(m: &Module, needle: &str) -> Vec<(u64, String)> {
    let img = &m.image;
    let mut out = Vec::new();
    if img.len() < 8 {
        return out;
    }
    let mod0 = u32at(img, 4) as usize;
    if mod0 + 4 > img.len() || u32at(img, mod0) != MOD0_MAGIC {
        return out;
    }
    let dyn_off = (mod0 as i64 + u32at(img, mod0 + 4) as i32 as i64) as usize;
    let (mut symtab, mut strtab) = (0usize, 0usize);
    let (mut rela, mut relasz, mut jmprel, mut pltrelsz) = (0usize, 0usize, 0usize, 0usize);
    let mut p = dyn_off;
    while p + 16 <= img.len() {
        let tag = u64at(img, p);
        let val = u64at(img, p + 8);
        p += 16;
        match tag {
            0 => break,
            5 => strtab = val as usize,
            6 => symtab = val as usize,
            7 => rela = val as usize,
            8 => relasz = val as usize,
            23 => jmprel = val as usize,
            2 => pltrelsz = val as usize,
            _ => {}
        }
    }
    if symtab == 0 || strtab == 0 {
        return out;
    }
    let lower = needle.to_ascii_lowercase();
    for (tab, sz) in [(rela, relasz), (jmprel, pltrelsz)] {
        if tab == 0 || sz == 0 {
            continue;
        }
        let mut q = tab;
        while q + 24 <= tab + sz && q + 24 <= img.len() {
            let r_offset = u64at(img, q);
            let r_info = u64at(img, q + 8);
            q += 24;
            let rtype = (r_info & 0xffff_ffff) as u32;
            if rtype != 1025 && rtype != 1026 && rtype != 257 {
                continue;
            }
            let symidx = (r_info >> 32) as usize;
            let s = symtab + symidx * 24;
            if s + 24 > img.len() {
                continue;
            }
            let n0 = strtab + u32at(img, s) as usize;
            if n0 >= img.len() {
                continue;
            }
            let end = img[n0..]
                .iter()
                .position(|&c| c == 0)
                .map(|e| n0 + e)
                .unwrap_or(img.len());
            let name = String::from_utf8_lossy(&img[n0..end]).into_owned();
            if name.to_ascii_lowercase().contains(&lower) {
                out.push((m.base + r_offset, name));
            }
        }
    }
    out
}

fn find_strings(m: &Module, needle: &str) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    let ascii = needle.as_bytes();
    let mut wide = Vec::with_capacity(ascii.len() * 2);
    for &b in ascii {
        wide.push(b);
        wide.push(0);
    }
    for (pat, kind) in [(ascii.to_vec(), "ascii"), (wide, "utf16")] {
        let mut start = 0usize;
        while start + pat.len() <= m.image.len() {
            match m.image[start..]
                .windows(pat.len())
                .position(|w| w == &pat[..])
            {
                Some(rel) => {
                    let off = start + rel;
                    out.push((m.base + off as u64, kind.to_string()));
                    start = off + 1;
                }
                None => break,
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nso_dataref <app.dxci> <hex-va[+len] | str:needle> [...]");
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

    let mut targets: Vec<(u64, u64, String)> = Vec::new();
    let mut bl_targets: Vec<u64> = Vec::new();
    for arg in &args[2..] {
        if let Some(addr) = arg.strip_prefix("bl:") {
            bl_targets
                .push(u64::from_str_radix(addr.trim_start_matches("0x"), 16).expect("hex va"));
        } else if let Some(needle) = arg.strip_prefix("imp:") {
            for m in &modules {
                for (va, name) in find_import_slots(m, needle) {
                    println!("import-slot {:#010x} {} {}", va, m.name, name);
                    targets.push((va, 8, format!("imp:{}", name)));
                }
            }
        } else if let Some(needle) = arg.strip_prefix("str:") {
            for m in &modules {
                for (va, kind) in find_strings(m, needle) {
                    println!("string {:#010x} {} {} \"{}\"", va, m.name, kind, needle);
                    let len = if kind == "utf16" {
                        needle.len() as u64 * 2
                    } else {
                        needle.len() as u64
                    };
                    targets.push((va, len, format!("{}({})", needle, kind)));
                }
            }
        } else {
            let (addr_s, len) = match arg.split_once('+') {
                Some((a, l)) => (
                    a,
                    u64::from_str_radix(l.trim_start_matches("0x"), 16).unwrap_or(4),
                ),
                None => (arg.as_str(), 4u64),
            };
            let addr = u64::from_str_radix(addr_s.trim_start_matches("0x"), 16).expect("hex va");
            targets.push((addr, len, format!("{:#x}", addr)));
        }
    }
    println!("----");

    for m in &modules {
        let end = m.text_end.min(m.image.len());
        let mut off = m.text_start;
        while off + 4 <= end {
            let op = u32at(&m.image, off);
            if !bl_targets.is_empty()
                && ((op & 0xfc00_0000) == 0x9400_0000 || (op & 0xfc00_0000) == 0x1400_0000)
            {
                let pc = m.base + off as u64;
                let imm = sign_extend((op & 0x03ff_ffff) as u64, 26) << 2;
                let dst = pc.wrapping_add(imm as u64);
                if bl_targets.contains(&dst) {
                    let kind = if (op & 0xfc00_0000) == 0x9400_0000 {
                        "bl"
                    } else {
                        "b"
                    };
                    let caller = find_sym(&m.syms, pc).unwrap_or_else(|| "<no-symbol>".to_string());
                    println!(
                        "{:#010x} {:<3} {} {} -> {:#010x}",
                        pc, kind, m.name, caller, dst
                    );
                }
            }
            if (op & 0x9f00_0000) == 0x9000_0000 {
                let pc = m.base + off as u64;
                let rd = (op & 0x1f) as u32;
                let immlo = ((op >> 29) & 3) as u64;
                let immhi = ((op >> 5) & 0x7ffff) as u64;
                let imm = sign_extend((immhi << 2) | immlo, 21) << 12;
                let page = (pc & !0xfff).wrapping_add(imm as u64);
                let lookahead = 12.min((end - off) / 4);
                for k in 1..lookahead {
                    let op2 = u32at(&m.image, off + k * 4);
                    let rn = ((op2 >> 5) & 0x1f) as u32;
                    let abs = if (op2 & 0xff80_0000) == 0x9100_0000 && rn == rd {
                        let mut imm12 = ((op2 >> 10) & 0xfff) as u64;
                        if (op2 >> 22) & 3 == 1 {
                            imm12 <<= 12;
                        }
                        Some(page.wrapping_add(imm12))
                    } else if (op2 & 0xffc0_0000) == 0xf940_0000 && rn == rd {
                        Some(page.wrapping_add((((op2 >> 10) & 0xfff) as u64) * 8))
                    } else if (op2 & 0xffc0_0000) == 0xb940_0000 && rn == rd {
                        Some(page.wrapping_add((((op2 >> 10) & 0xfff) as u64) * 4))
                    } else {
                        if (op2 & 0x1f) == rd
                            && (op2 & 0x9f00_0000) != 0x9000_0000
                            && ((op2 & 0xff80_0000) == 0x9100_0000
                                || (op2 & 0xffc0_0000) == 0xf940_0000)
                        {
                            break;
                        }
                        None
                    };
                    if let Some(abs) = abs {
                        for (ta, tl, tn) in &targets {
                            if abs >= *ta && abs < ta + tl.max(&1) {
                                let caller = find_sym(&m.syms, pc)
                                    .unwrap_or_else(|| "<no-symbol>".to_string());
                                println!(
                                    "{:#010x} {:<8} {} -> {:#010x} {}",
                                    pc, m.name, caller, abs, tn
                                );
                            }
                        }
                        break;
                    }
                }
            }
            off += 4;
        }
    }
}
