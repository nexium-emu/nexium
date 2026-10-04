use nexium_shader::disassemble;

fn main() {
    let path = std::env::args().nth(1).expect("usage: sassdis <dumpfile>");
    let bytes = std::fs::read(&path).expect("read dump");
    for line in disassemble(&bytes) {
        println!("{}", line.to_string_compact());
    }
}
