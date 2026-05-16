use std::fs;

fn main() {
    let data = fs::read("space-nx-master/spacenx.nro").unwrap();
    
    println!("Total file size: {:#x} ({}MB)", data.len(), data.len() / (1024*1024));
    
    // Parse header
    let text_off = u32::from_le_bytes([data[0x14], data[0x15], data[0x16], data[0x17]]);
    let text_size = u32::from_le_bytes([data[0x18], data[0x19], data[0x1a], data[0x1b]]);
    let ro_off = u32::from_le_bytes([data[0x1c], data[0x1d], data[0x1e], data[0x1f]]);
    let ro_size = u32::from_le_bytes([data[0x20], data[0x21], data[0x22], data[0x23]]);
    let data_off = u32::from_le_bytes([data[0x24], data[0x25], data[0x26], data[0x27]]);
    let data_size = u32::from_le_bytes([data[0x28], data[0x29], data[0x2a], data[0x2b]]);
    let bss_size = u32::from_le_bytes([data[0x2c], data[0x2d], data[0x2e], data[0x2f]]);
    
    println!("Text:  off={:#x}, size={:#x}", text_off, text_size);
    println!("RO:    off={:#x}, size={:#x}", ro_off, ro_size);
    println!("Data:  off={:#x}, size={:#x}", data_off, data_size);
    println!("BSS:   size={:#x}", bss_size);
    
    // First 16 bytes of data segment
    println!("\nFirst 16 bytes of data segment (at file offset {:#x}):", data_off);
    for i in 0..16 {
        print!("{:02x} ", data[data_off as usize + i]);
    }
    println!();
    
    // Check RELRO segment (assuming RO ends with RELRO)
    let relro_start = data_off;
    let relro_size = 0x3e000;  // From the svcSetMemoryPermission call
    let relro_file_offset = data_off;
    println!("\nRELRO segment (assuming it's in data):");
    println!("  File offset: {:#x}, Size: {:#x}", relro_file_offset, relro_size);
    println!("  First 16 bytes:");
    for i in 0..16.min(relro_size as usize) {
        print!("{:02x} ", data[relro_file_offset as usize + i]);
    }
    println!();
}
