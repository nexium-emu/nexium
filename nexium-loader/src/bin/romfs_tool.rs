use nexium_loader::application::Application;
use std::fs;

fn u16be(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|v| u16::from_be_bytes([v[0], v[1]]))
}

fn u16le(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|v| u16::from_le_bytes([v[0], v[1]]))
}

fn u32be(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4)
        .map(|v| u32::from_be_bytes([v[0], v[1], v[2], v[3]]))
}

fn u32le(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4)
        .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
}

fn u64le(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o + 8)
        .map(|v| u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]))
}

#[derive(Clone, Copy)]
struct RomfsHeader {
    dir_meta_off: usize,
    file_meta_off: usize,
    file_data_off: usize,
}

fn romfs_header(romfs: &[u8]) -> Option<RomfsHeader> {
    if u64le(romfs, 0)? != 0x50 {
        return None;
    }
    Some(RomfsHeader {
        dir_meta_off: u64le(romfs, 0x18)? as usize,
        file_meta_off: u64le(romfs, 0x38)? as usize,
        file_data_off: u64le(romfs, 0x48)? as usize,
    })
}

fn romfs_name(
    romfs: &[u8],
    entry_abs: usize,
    name_off: usize,
    name_len_off: usize,
) -> Option<&str> {
    let name_len = u32le(romfs, entry_abs + name_len_off)? as usize;
    let start = entry_abs + name_off;
    let end = start.checked_add(name_len)?;
    std::str::from_utf8(romfs.get(start..end)?).ok()
}

fn find_child_dir(romfs: &[u8], hdr: RomfsHeader, dir_off: u32, name: &str) -> Option<u32> {
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut child = u32le(romfs, dir_abs + 0x08)?;
    while child != 0xffff_ffff {
        let abs = hdr.dir_meta_off + child as usize;
        if romfs_name(romfs, abs, 0x18, 0x14)? == name {
            return Some(child);
        }
        child = u32le(romfs, abs + 0x04)?;
    }
    None
}

fn find_child_file(
    romfs: &[u8],
    hdr: RomfsHeader,
    dir_off: u32,
    name: &str,
) -> Option<(usize, usize)> {
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut child = u32le(romfs, dir_abs + 0x0c)?;
    while child != 0xffff_ffff {
        let abs = hdr.file_meta_off + child as usize;
        if romfs_name(romfs, abs, 0x20, 0x1c)? == name {
            let rel = u64le(romfs, abs + 0x08)? as usize;
            let size = u64le(romfs, abs + 0x10)? as usize;
            return Some((hdr.file_data_off + rel, size));
        }
        child = u32le(romfs, abs + 0x04)?;
    }
    None
}

fn romfs_file<'a>(romfs: &'a [u8], path: &str) -> Option<&'a [u8]> {
    let hdr = romfs_header(romfs)?;
    let parts = path
        .trim_start_matches('/')
        .split('/')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>();
    let (file, dirs) = parts.split_last()?;
    let mut dir = 0u32;
    for part in dirs {
        dir = find_child_dir(romfs, hdr, dir, part)?;
    }
    let (off, size) = find_child_file(romfs, hdr, dir, file)?;
    romfs.get(off..off.checked_add(size)?)
}

fn list_romfs_dir(romfs: &[u8], hdr: RomfsHeader, dir_off: u32, prefix: &str, needle: &str) {
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut file = match u32le(romfs, dir_abs + 0x0c) {
        Some(v) => v,
        None => return,
    };
    while file != 0xffff_ffff {
        let abs = hdr.file_meta_off + file as usize;
        if let Some(name) = romfs_name(romfs, abs, 0x20, 0x1c) {
            let path = format!("{}/{}", prefix, name);
            if needle.is_empty() || path.to_ascii_lowercase().contains(needle) {
                let size = u64le(romfs, abs + 0x10).unwrap_or(0);
                println!("{} {}", size, path);
            }
        }
        file = match u32le(romfs, abs + 0x04) {
            Some(v) => v,
            None => break,
        };
    }
    let mut child = match u32le(romfs, dir_abs + 0x08) {
        Some(v) => v,
        None => return,
    };
    while child != 0xffff_ffff {
        let abs = hdr.dir_meta_off + child as usize;
        if let Some(name) = romfs_name(romfs, abs, 0x18, 0x14) {
            let path = format!("{}/{}", prefix, name);
            list_romfs_dir(romfs, hdr, child, &path, needle);
        }
        child = match u32le(romfs, abs + 0x04) {
            Some(v) => v,
            None => break,
        };
    }
}

fn yaz0(data: &[u8]) -> Option<Vec<u8>> {
    if data.get(0..4)? != b"Yaz0" {
        return Some(data.to_vec());
    }
    let out_size = u32be(data, 4)? as usize;
    let mut out = Vec::with_capacity(out_size);
    let mut src = 0x10usize;
    let mut code = 0u8;
    let mut bits = 0u8;
    while out.len() < out_size {
        if bits == 0 {
            code = *data.get(src)?;
            src += 1;
            bits = 8;
        }
        if (code & 0x80) != 0 {
            out.push(*data.get(src)?);
            src += 1;
        } else {
            let b1 = *data.get(src)? as usize;
            let b2 = *data.get(src + 1)? as usize;
            src += 2;
            let dist = ((b1 & 0x0f) << 8) | b2;
            let mut len = b1 >> 4;
            if len == 0 {
                len = *data.get(src)? as usize + 0x12;
                src += 1;
            } else {
                len += 2;
            }
            let copy_src = out.len().checked_sub(dist + 1)?;
            for i in 0..len {
                let v = *out.get(copy_src + i)?;
                out.push(v);
                if out.len() >= out_size {
                    break;
                }
            }
        }
        code <<= 1;
        bits -= 1;
    }
    Some(out)
}

fn sarc_entries(data: &[u8]) -> Option<Vec<(String, usize, usize)>> {
    if data.get(0..4)? != b"SARC" {
        return None;
    }
    let little = match data.get(0x06..0x08)? {
        [0xff, 0xfe] => true,
        [0xfe, 0xff] => false,
        _ => return None,
    };
    let rd16 = |o| {
        if little {
            u16le(data, o)
        } else {
            u16be(data, o)
        }
    };
    let rd32 = |o| {
        if little {
            u32le(data, o)
        } else {
            u32be(data, o)
        }
    };
    let data_off = rd16(0x0c)? as usize;
    if data.get(0x14..0x18)? != b"SFAT" {
        return None;
    }
    let count = rd16(0x1a)? as usize;
    let sfnt = 0x14 + 0x0c + count * 0x10;
    if data.get(sfnt..sfnt + 4)? != b"SFNT" {
        return None;
    }
    let str_base = sfnt + 0x08;
    let mut out = Vec::new();
    for i in 0..count {
        let n = 0x20 + i * 0x10;
        let name_flags = rd32(n + 4)?;
        let start = data_off + rd32(n + 8)? as usize;
        let end = data_off + rd32(n + 12)? as usize;
        let name = if (name_flags & 0x0100_0000) != 0 {
            let rel = ((name_flags & 0x00ff_ffff) as usize) * 4;
            let s = str_base + rel;
            let e = data[s..]
                .iter()
                .position(|b| *b == 0)
                .map(|p| s + p)
                .unwrap_or(data.len());
            String::from_utf8_lossy(&data[s..e]).into_owned()
        } else {
            format!("#{:08x}", rd32(n)?)
        };
        out.push((name, start, end.saturating_sub(start)));
    }
    Some(out)
}

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  romfs_tool <app.dxci> list [needle]");
    eprintln!("  romfs_tool <app.dxci> dump <romfs-path> <out>");
    eprintln!("  romfs_tool <app.dxci> sarc-list <romfs-path> [needle]");
    eprintln!("  romfs_tool <app.dxci> sarc-dump <romfs-path> <sarc-name> <out>");
    std::process::exit(2);
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        usage();
    }
    let app = Application::load(&args[1]).expect("load application");
    let romfs = app
        .romfs
        .as_ref()
        .expect("application has no RomFS")
        .as_slice();
    match args[2].as_str() {
        "list" => {
            let needle = args
                .get(3)
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let hdr = romfs_header(romfs).expect("bad RomFS");
            list_romfs_dir(romfs, hdr, 0, "", &needle);
        }
        "dump" => {
            let path = args.get(3).unwrap_or_else(|| usage());
            let out = args.get(4).unwrap_or_else(|| usage());
            let data = romfs_file(romfs, path).expect("RomFS path not found");
            fs::write(out, data).expect("write output");
        }
        "sarc-list" => {
            let path = args.get(3).unwrap_or_else(|| usage());
            let needle = args
                .get(4)
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let data = romfs_file(romfs, path).expect("RomFS path not found");
            let data = yaz0(data).expect("Yaz0 decode failed");
            for (name, _, size) in sarc_entries(&data).expect("not a SARC") {
                if needle.is_empty() || name.to_ascii_lowercase().contains(&needle) {
                    println!("{} {}", size, name);
                }
            }
        }
        "sarc-dump" => {
            let path = args.get(3).unwrap_or_else(|| usage());
            let name = args.get(4).unwrap_or_else(|| usage());
            let out = args.get(5).unwrap_or_else(|| usage());
            let data = romfs_file(romfs, path).expect("RomFS path not found");
            let data = yaz0(data).expect("Yaz0 decode failed");
            let entries = sarc_entries(&data).expect("not a SARC");
            let (_, off, size) = entries
                .iter()
                .find(|(n, _, _)| n == name)
                .unwrap_or_else(|| panic!("SARC entry not found: {}", name));
            fs::write(out, &data[*off..off + size]).expect("write output");
        }
        _ => usage(),
    }
}
