use std::fs;

fn main() {
    let data = fs::read("space-nx-master/spacenx.nro").unwrap();
    let first_insn = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    
    println!("First instruction (little-endian): 0x{:08x}", first_insn);
    
    // ARM64 B instruction format: bits [25-0] = signed offset
    // The offset is in units of 4 bytes
    if (first_insn & 0xFC000000) == 0x14000000 {
        let mut offset_units = (first_insn & 0x03FFFFFF) as i32;
        // Sign extend if negative
        if offset_units & 0x02000000 != 0 {
            offset_units -= 0x04000000;
        }
        let offset_bytes = offset_units * 4;
        println!("This is a B (branch) instruction with offset {} units ({:#x} bytes)", offset_units, offset_bytes as u32);
        println!("Branch target: PC + {:#x} = 0x{:x}", offset_bytes as u32, offset_bytes as u32);
    } else {
        println!("Not a B instruction: opcode = {:#x}", (first_insn >> 26) & 0x3F);
    }
}
