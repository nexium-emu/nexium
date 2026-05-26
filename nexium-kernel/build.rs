use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ipc_dir = manifest_dir.join("ipc");

    println!("cargo:rerun-if-changed=ipc");
    println!("cargo:rerun-if-changed=build.rs");

    let mut services: Vec<Service> = Vec::new();
    if let Ok(rd) = fs::read_dir(&ipc_dir) {
        let mut paths: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("swipc"))
            .collect();
        paths.sort();
        for path in paths {
            println!("cargo:rerun-if-changed={}", path.display());
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("swipc: failed to read {}: {}", path.display(), e));
            let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
            services.extend(parse(&text, &file_name));
        }
    }

    let code = generate(&services);
    let out_file = out_dir.join("swipc_generated.rs");
    fs::write(&out_file, code).expect("swipc: write OUT_DIR file");
}

#[derive(Debug)]
struct Service {
    module: String,
    ports: Vec<String>,
    commands: Vec<Command>,
    source_file: String,
}

#[derive(Debug)]
struct Command {
    id: u32,
    name: String,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
}

#[derive(Debug)]
enum Input {
    Pid,
    Prim { ty: Prim, name: String },
    InBuffer { name: String },
    OutBuffer { name: String },
    InHandle { name: String },
}

#[derive(Debug, Clone, Copy)]
enum Prim {
    U8, U16, U32, U64, I8, I16, I32, I64, Bool,
}

impl Prim {
    fn size(self) -> usize {
        match self {
            Prim::U8 | Prim::I8 | Prim::Bool => 1,
            Prim::U16 | Prim::I16 => 2,
            Prim::U32 | Prim::I32 => 4,
            Prim::U64 | Prim::I64 => 8,
        }
    }
    fn align(self) -> usize { self.size() }
    fn rust(self) -> &'static str {
        match self {
            Prim::U8 => "u8", Prim::I8 => "i8", Prim::Bool => "bool",
            Prim::U16 => "u16", Prim::I16 => "i16",
            Prim::U32 => "u32", Prim::I32 => "i32",
            Prim::U64 => "u64", Prim::I64 => "i64",
        }
    }
}

#[derive(Debug)]
enum Output {
    Prim(Prim),
    Subsession(String),
    OutHandle,
    ZeroBytes(usize),
    HandlerBytes(usize),
}

fn parse(text: &str, source_file: &str) -> Vec<Service> {
    let mut lex = Lexer::new(text);
    let mut services = Vec::new();
    while !lex.eof() {
        let kw = lex.ident();
        if kw.as_deref() != Some("service") {
            lex.fail(&format!("expected `service` keyword, got {:?}", kw));
        }
        let first = lex.ident().unwrap_or_else(|| lex.fail("expected port name"));
        let mut ports = vec![first];
        while lex.peek_punct(',') {
            lex.consume_punct(',');
            ports.push(lex.ident().unwrap_or_else(|| lex.fail("expected port alias")));
        }
        let module = if lex.peek_ident("as") {
            lex.consume_ident("as");
            lex.ident().unwrap_or_else(|| lex.fail("expected module identifier after `as`"))
        } else {
            sanitize_module(&ports[0])
        };
        lex.consume_punct('{');
        let mut commands = Vec::new();
        while !lex.peek_punct('}') {
            commands.push(parse_command(&mut lex));
        }
        lex.consume_punct('}');
        services.push(Service { module, ports, commands, source_file: source_file.to_string() });
    }
    services
}

fn sanitize_module(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' { out.push(c); }
        else { out.push('_'); }
    }
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        out.insert(0, '_');
    }
    out
}

fn parse_command(lex: &mut Lexer) -> Command {
    lex.consume_punct('[');
    let id = lex.number();
    lex.consume_punct(']');
    let name = lex.ident().unwrap_or_else(|| lex.fail("expected command name"));
    lex.consume_punct('(');
    let mut inputs: Vec<Input> = Vec::new();
    if !lex.peek_punct(')') {
        loop {
            inputs.push(parse_input(lex));
            if !lex.peek_punct(',') { break; }
            lex.consume_punct(',');
        }
    }
    lex.consume_punct(')');
    let outputs = if lex.peek_punct('-') {
        lex.consume_punct('-');
        lex.consume_punct('>');
        parse_outputs(lex)
    } else { Vec::new() };
    lex.consume_punct(';');
    Command { id, name, inputs, outputs }
}

fn parse_input(lex: &mut Lexer) -> Input {
    let kw = lex.ident().unwrap_or_else(|| lex.fail("expected arg type or `pid`"));
    if kw == "pid" { return Input::Pid; }
    if kw == "in_buffer" || kw == "out_buffer" {
        if lex.peek_punct('<') {
            lex.consume_punct('<');
            let _ = lex.ident();
            while lex.peek_punct(',') { lex.consume_punct(','); let _ = lex.ident(); }
            lex.consume_punct('>');
        }
        let name = lex.ident().unwrap_or_else(|| lex.fail("expected arg name after buffer type"));
        return if kw == "in_buffer" { Input::InBuffer { name } } else { Input::OutBuffer { name } };
    }
    if kw == "in_handle" {
        let name = lex.ident().unwrap_or_else(|| lex.fail("expected arg name after in_handle"));
        return Input::InHandle { name };
    }
    let ty = parse_prim(&kw).unwrap_or_else(|| lex.fail(&format!("unknown input type `{}`", kw)));
    let name = lex.ident().unwrap_or_else(|| lex.fail("expected arg name"));
    Input::Prim { ty, name }
}

fn parse_outputs(lex: &mut Lexer) -> Vec<Output> {
    if lex.peek_punct('(') {
        lex.consume_punct('(');
        let mut outs = Vec::new();
        if !lex.peek_punct(')') {
            loop {
                outs.push(parse_one_output(lex));
                if !lex.peek_punct(',') { break; }
                lex.consume_punct(',');
            }
        }
        lex.consume_punct(')');
        outs
    } else {
        vec![parse_one_output(lex)]
    }
}

fn parse_one_output(lex: &mut Lexer) -> Output {
    let kw = lex.ident().unwrap_or_else(|| lex.fail("expected return type"));
    match kw.as_str() {
        "session" => {
            lex.consume_punct('<');
            let iface = lex.ident().unwrap_or_else(|| lex.fail("expected interface name"));
            lex.consume_punct('>');
            Output::Subsession(iface)
        }
        "out_handle" => Output::OutHandle,
        "zeros" => {
            lex.consume_punct('(');
            let n = lex.number();
            lex.consume_punct(')');
            Output::ZeroBytes(n as usize)
        }
        "bytes" => {
            lex.consume_punct('(');
            let n = lex.number();
            lex.consume_punct(')');
            Output::HandlerBytes(n as usize)
        }
        other => parse_prim(other).map(Output::Prim).unwrap_or_else(|| lex.fail(&format!("unknown return type `{}`", other))),
    }
}

fn parse_prim(s: &str) -> Option<Prim> {
    Some(match s {
        "u8" => Prim::U8, "i8" => Prim::I8, "bool" => Prim::Bool,
        "u16" => Prim::U16, "i16" => Prim::I16,
        "u32" => Prim::U32, "i32" => Prim::I32,
        "u64" => Prim::U64, "i64" => Prim::I64,
        _ => return None,
    })
}

struct Lexer<'a> { src: &'a str, pos: usize, line: usize }

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self { Self { src, pos: 0, line: 1 } }
    fn fail(&self, msg: &str) -> ! {
        panic!("swipc parse error at line {}: {} (near {:?})", self.line, msg, self.peek_window());
    }
    fn peek_window(&self) -> String {
        let end = (self.pos + 32).min(self.src.len());
        self.src[self.pos..end].to_string()
    }
    fn eof(&mut self) -> bool { self.skip_ws(); self.pos >= self.src.len() }
    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.src.len() {
                let c = self.src.as_bytes()[self.pos];
                if c == b'\n' { self.line += 1; self.pos += 1; continue; }
                if c == b' ' || c == b'\t' || c == b'\r' { self.pos += 1; continue; }
                break;
            }
            if self.pos < self.src.len() && self.src.as_bytes()[self.pos] == b'#' {
                while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'\n' { self.pos += 1; }
                continue;
            }
            if self.pos + 1 < self.src.len()
                && self.src.as_bytes()[self.pos] == b'/'
                && self.src.as_bytes()[self.pos + 1] == b'/'
            {
                while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'\n' { self.pos += 1; }
                continue;
            }
            break;
        }
    }
    fn ident(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src.as_bytes()[self.pos];
            let is_ident = c.is_ascii_alphanumeric() || c == b'_' || c == b':' || c == b'-';
            if !is_ident { break; }
            self.pos += 1;
        }
        if self.pos == start { return None; }
        if self.src.as_bytes()[start].is_ascii_digit() {
            self.pos = start;
            return None;
        }
        Some(self.src[start..self.pos].to_string())
    }
    fn peek_ident(&mut self, kw: &str) -> bool {
        self.skip_ws();
        let save = self.pos;
        let save_line = self.line;
        let id = self.ident();
        let matched = id.as_deref() == Some(kw);
        self.pos = save;
        self.line = save_line;
        matched
    }
    fn consume_ident(&mut self, kw: &str) {
        self.skip_ws();
        let save = self.pos;
        let id = self.ident();
        if id.as_deref() != Some(kw) {
            self.pos = save;
            self.fail(&format!("expected `{}`", kw));
        }
    }
    fn number(&mut self) -> u32 {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.src.as_bytes();
        let (radix, skip) = if bytes.len() > start + 1 && bytes[start] == b'0' && (bytes[start + 1] == b'x' || bytes[start + 1] == b'X') {
            (16u32, 2usize)
        } else { (10u32, 0usize) };
        self.pos += skip;
        let num_start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src.as_bytes()[self.pos];
            let ok = match radix { 16 => c.is_ascii_hexdigit(), _ => c.is_ascii_digit() };
            if !ok { break; }
            self.pos += 1;
        }
        if self.pos == num_start { self.fail("expected number"); }
        u32::from_str_radix(&self.src[num_start..self.pos], radix).unwrap_or_else(|e| self.fail(&format!("bad number: {}", e)))
    }
    fn peek_punct(&mut self, c: char) -> bool {
        self.skip_ws();
        self.pos < self.src.len() && self.src.as_bytes()[self.pos] as char == c
    }
    fn consume_punct(&mut self, c: char) {
        self.skip_ws();
        if !self.peek_punct(c) { self.fail(&format!("expected '{}'", c)); }
        self.pos += 1;
    }
}

fn generate(services: &[Service]) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED FILE — do not edit. See nexium-kernel/build.rs and ipc/*.swipc.\n");
    out.push_str("use nexium_ipc as ipc;\n");
    out.push_str("use crate::kernel::Kernel;\n\n");
    out.push_str("#[inline] fn read_u8(buf: &[u8], o: usize) -> u8 { buf.get(o).copied().unwrap_or(0) }\n");
    out.push_str("#[inline] fn read_u16(buf: &[u8], o: usize) -> u16 { if o+2<=buf.len() { u16::from_le_bytes([buf[o],buf[o+1]]) } else { 0 } }\n");
    out.push_str("#[inline] fn read_u32(buf: &[u8], o: usize) -> u32 { if o+4<=buf.len() { u32::from_le_bytes([buf[o],buf[o+1],buf[o+2],buf[o+3]]) } else { 0 } }\n");
    out.push_str("#[inline] fn read_u64(buf: &[u8], o: usize) -> u64 { if o+8<=buf.len() { u64::from_le_bytes([buf[o],buf[o+1],buf[o+2],buf[o+3],buf[o+4],buf[o+5],buf[o+6],buf[o+7]]) } else { 0 } }\n");
    out.push_str("\n");
    out.push_str("fn read_in_buffer(kernel: &mut Kernel, ctx: &ipc::IpcCtx, index: usize) -> Vec<u8> {\n");
    out.push_str("    let mut all: Vec<ipc::IpcBuffer> = Vec::new();\n");
    out.push_str("    all.extend(ctx.send_buffers.iter().copied());\n");
    out.push_str("    all.extend(ctx.send_statics.iter().copied());\n");
    out.push_str("    let Some(b) = all.iter().filter(|b| b.size > 0 && b.addr != 0).nth(index).copied() else { return Vec::new(); };\n");
    out.push_str("    let mut data = vec![0u8; b.size as usize];\n");
    out.push_str("    if kernel.address_space.read(b.addr, &mut data).is_err() { return Vec::new(); }\n");
    out.push_str("    data\n");
    out.push_str("}\n");
    out.push_str("fn write_out_buffer(kernel: &mut Kernel, ctx: &ipc::IpcCtx, index: usize, data: &[u8]) {\n");
    out.push_str("    let mut all: Vec<ipc::IpcBuffer> = Vec::new();\n");
    out.push_str("    all.extend(ctx.recv_buffers.iter().copied());\n");
    out.push_str("    all.extend(ctx.recv_statics.iter().copied());\n");
    out.push_str("    let Some(b) = all.iter().filter(|b| b.size > 0 && b.addr != 0).nth(index).copied() else { return; };\n");
    out.push_str("    let n = data.len().min(b.size as usize);\n");
    out.push_str("    let _ = kernel.address_space.write(b.addr, &data[..n]);\n");
    out.push_str("}\n\n");

    for svc in services {
        emit_service(&mut out, svc);
    }

    let mut ports_map: BTreeMap<&str, &Service> = BTreeMap::new();
    for svc in services {
        for p in &svc.ports {
            ports_map.insert(p.as_str(), svc);
        }
    }
    out.push_str("\npub fn dispatch_generated(\n");
    out.push_str("    kernel: &mut Kernel,\n");
    out.push_str("    port_name: &str,\n");
    out.push_str("    ctx: &mut ipc::IpcCtx,\n");
    out.push_str("    session_handle: u32,\n");
    out.push_str(") -> Option<Vec<u8>> {\n");
    out.push_str("    match port_name {\n");
    for (port, svc) in &ports_map {
        out.push_str(&format!("        \"{}\" => dispatch_{}(kernel, ctx, session_handle),\n", port, svc.module));
    }
    out.push_str("        _ => None,\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

fn emit_service(out: &mut String, svc: &Service) {
    out.push_str(&format!("\n// ───── from {} ─── service `{}` (ports: {}) ─────\n", svc.source_file, svc.module, svc.ports.join(", ")));
    out.push_str(&format!("fn dispatch_{}(kernel: &mut Kernel, ctx: &mut ipc::IpcCtx, session_handle: u32) -> Option<Vec<u8>> {{\n", svc.module));
    out.push_str("    let cmd_id = ctx.cmif_in.cmd_id;\n");
    out.push_str("    let in_off = ctx.cmif_in_data_off;\n");
    out.push_str("    let buf = ctx.buf.clone();\n");
    out.push_str("    match cmd_id {\n");
    for cmd in &svc.commands {
        emit_command(out, &svc.module, cmd);
    }
    out.push_str("        _ => return None,\n");
    out.push_str("    }\n");
    out.push_str("}\n");
}

fn emit_command(out: &mut String, module: &str, cmd: &Command) {
    out.push_str(&format!("        {} => {{ // {}\n", cmd.id, cmd.name));
    let mut off: usize = 0;
    let mut call_args: Vec<String> = vec!["kernel".to_string(), "ctx".to_string(), "session_handle".to_string()];
    let mut in_buf_idx: usize = 0;
    let mut out_buf_idx: usize = 0;
    let mut in_handle_idx: usize = 0;
    let mut post_writes: Vec<(String, usize)> = Vec::new();
    for inp in &cmd.inputs {
        match inp {
            Input::Pid => {}
            Input::Prim { ty, name } => {
                let a = ty.align();
                if off % a != 0 { off += a - (off % a); }
                let reader = match ty {
                    Prim::U8 => format!("read_u8(&buf, in_off + {})", off),
                    Prim::I8 => format!("read_u8(&buf, in_off + {}) as i8", off),
                    Prim::Bool => format!("read_u8(&buf, in_off + {}) != 0", off),
                    Prim::U16 => format!("read_u16(&buf, in_off + {})", off),
                    Prim::I16 => format!("read_u16(&buf, in_off + {}) as i16", off),
                    Prim::U32 => format!("read_u32(&buf, in_off + {})", off),
                    Prim::I32 => format!("read_u32(&buf, in_off + {}) as i32", off),
                    Prim::U64 => format!("read_u64(&buf, in_off + {})", off),
                    Prim::I64 => format!("read_u64(&buf, in_off + {}) as i64", off),
                };
                out.push_str(&format!("            let {}: {} = {};\n", name, ty.rust(), reader));
                call_args.push(name.clone());
                off += ty.size();
            }
            Input::InBuffer { name } => {
                out.push_str(&format!("            let {}: Vec<u8> = read_in_buffer(kernel, ctx, {});\n", name, in_buf_idx));
                call_args.push(format!("&{}", name));
                in_buf_idx += 1;
            }
            Input::OutBuffer { name } => {
                out.push_str(&format!("            let mut {}: Vec<u8> = Vec::new();\n", name));
                call_args.push(format!("&mut {}", name));
                post_writes.push((name.clone(), out_buf_idx));
                out_buf_idx += 1;
            }
            Input::InHandle { name } => {
                out.push_str(&format!("            let {}: u32 = ctx.copy_handles.get({}).copied().unwrap_or_else(|| ctx.move_handles.get({}).copied().unwrap_or(0));\n", name, in_handle_idx, in_handle_idx));
                call_args.push(name.clone());
                in_handle_idx += 1;
            }
        }
    }
    let handler = to_snake(&cmd.name);
    let call = format!("crate::services::{}::handlers::{}({})", module, handler, call_args.join(", "));
    let post_write_block: String = post_writes.iter().map(|(n, i)| {
        format!("            write_out_buffer(kernel, ctx, {}, &{});\n", i, n)
    }).collect();

    if cmd.outputs.is_empty() {
        out.push_str(&format!("            let _: () = {};\n", call));
        out.push_str(&post_write_block);
        out.push_str("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &[], &[]))\n");
        out.push_str("        }\n");
        return;
    }

    if cmd.outputs.len() == 1 {
        match &cmd.outputs[0] {
            Output::Prim(p) => {
                out.push_str(&format!("            let __ret: {} = {};\n", p.rust(), call));
                out.push_str(&post_write_block);
                let bytes_expr = match p {
                    Prim::Bool => "[(__ret as u8)]".to_string(),
                    _ => "__ret.to_le_bytes()".to_string(),
                };
                out.push_str(&format!("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &{}, &[]))\n", bytes_expr));
            }
            Output::Subsession(iface) => {
                out.push_str(&format!("            let _: () = {};\n", call));
                out.push_str(&post_write_block);
                out.push_str(&format!("            Some(crate::kernel::svc::return_subsession(kernel, ctx, session_handle, \"{}\"))\n", iface));
            }
            Output::OutHandle => {
                out.push_str(&format!("            let __h: u32 = {};\n", call));
                out.push_str(&post_write_block);
                out.push_str("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &[], &[__h]))\n");
            }
            Output::ZeroBytes(n) => {
                out.push_str(&format!("            let _: () = {};\n", call));
                out.push_str(&post_write_block);
                out.push_str(&format!("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &[0u8; {}], &[]))\n", n));
            }
            Output::HandlerBytes(n) => {
                out.push_str(&format!("            let __ret: Vec<u8> = {};\n", call));
                out.push_str(&post_write_block);
                out.push_str(&format!("            let mut __resp = [0u8; {}];\n", n));
                out.push_str(&format!("            let __n = __ret.len().min({});\n", n));
                out.push_str("            __resp[..__n].copy_from_slice(&__ret[..__n]);\n");
                out.push_str("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &__resp, &[]))\n");
            }
        }
        out.push_str("        }\n");
        return;
    }

    out.push_str(&format!("            let __tup = {};\n", call));
    out.push_str(&post_write_block);
    out.push_str("            let mut __data: Vec<u8> = Vec::new();\n");
    out.push_str("            let mut __handles: Vec<u32> = Vec::new();\n");
    for (i, o) in cmd.outputs.iter().enumerate() {
        match o {
            Output::Prim(p) => {
                let elem = format!("__tup.{}", i);
                let bytes_expr = match p {
                    Prim::Bool => format!("[{} as u8]", elem),
                    _ => format!("{}.to_le_bytes()", elem),
                };
                out.push_str(&format!("            __data.extend_from_slice(&{});\n", bytes_expr));
            }
            Output::OutHandle => {
                out.push_str(&format!("            __handles.push(__tup.{});\n", i));
            }
            Output::ZeroBytes(_) | Output::HandlerBytes(_) | Output::Subsession(_) => {
                panic!("tuple returns may only contain primitives and out_handle");
            }
        }
    }
    out.push_str("            Some(crate::kernel::svc::build_ipc_response(ctx, 0, &__data, &__handles))\n");
    out.push_str("        }\n");
}

fn to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let bytes = s.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                let prev = bytes[i - 1];
                let next = bytes.get(i + 1).copied().unwrap_or(0);
                if prev.is_ascii_lowercase() || prev.is_ascii_digit()
                    || (prev.is_ascii_uppercase() && next.is_ascii_lowercase())
                {
                    out.push('_');
                }
            }
            out.push(c.to_ascii_lowercase() as char);
        } else if c == b':' || c == b'-' {
            out.push('_');
        } else {
            out.push(c as char);
        }
    }
    out
}
