use nexium_core::boot::{BootConfig, BootContext};
use nexium_core::cpu::CpuEvent;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NeXium SVC Execution Test");
    println!("================================");

    let config = BootConfig::new("C:\\Users\\Mythrax\\Downloads\\NRO\\spacenx.nro");
    let mut boot_ctx = BootContext::new(config)?;

    println!("Testing SVC dispatch mechanism...\n");

    if let Some(cpu) = &mut boot_ctx.kernel.cpu {
        println!("[TEST 1] Testing svcOutputDebugString");
        println!("----- ");

        let test_message = "Hello from SVC test!";
        let msg_ptr: u64 = 0x9000_0000_0000;
        let msg_len = test_message.len() as u64;

        boot_ctx.kernel.address_space.write(msg_ptr, test_message.as_bytes())
            .expect("Failed to write test message");

        cpu.set_register(0, msg_ptr);
        cpu.set_register(1, msg_len);

        cpu.inject_svc(0x1b);

        if let CpuEvent::Svc(imm) = cpu.run(100) {
            println!("Got SVC {:#04x}", imm);
            let result = boot_ctx.kernel.dispatch_svc(imm);
            println!("SVC returned: {:#x}\n", result);
        }
    }

    if let Some(cpu) = &mut boot_ctx.kernel.cpu {
        println!("[TEST 2] Testing svcExitProcess");
        println!("----- ");

        cpu.inject_svc(0x07);

        if let CpuEvent::Svc(imm) = cpu.run(100) {
            println!("Got SVC {:#04x}", imm);
            let result = boot_ctx.kernel.dispatch_svc(imm);
            println!("SVC returned: {:#x}\n", result);
        }
    }

    println!("================================");
    println!("All SVC tests completed successfully!");
    Ok(())
}
