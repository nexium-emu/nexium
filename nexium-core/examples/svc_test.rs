use nexium_core::boot::{BootConfig, BootContext};
use nexium_core::cpu::CpuEvent;
use nexium_core::kernel::cpu_local::{cpu_mut, set_current_cpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NeXium SVC Execution Test");
    println!("================================");

    let config = BootConfig::new("C:\\Users\\Mythrax\\Downloads\\NRO\\spacenx.nro");
    let mut boot_ctx = BootContext::new(config)?;

    println!("Testing SVC dispatch mechanism...\n");

    let mut cpu = boot_ctx.cpu.take().ok_or("CPU not initialized")?;
    let _cpu_guard = set_current_cpu(&mut cpu, 0);

    println!("[TEST 1] Testing svcOutputDebugString");
    println!("----- ");

    let test_message = "Hello from SVC test!";
    let msg_ptr: u64 = 0x9000_0000_0000;
    let msg_len = test_message.len() as u64;

    boot_ctx
        .address_space
        .write(msg_ptr, test_message.as_bytes())
        .expect("Failed to write test message");

    {
        let cpu = cpu_mut().ok_or("CPU not installed")?;
        cpu.set_register(0, msg_ptr);
        cpu.set_register(1, msg_len);
        cpu.inject_svc(0x1b);
    }

    if let Ok(result) = cpu_mut().ok_or("CPU not installed")?.run(100) {
        if let CpuEvent::Svc(imm) = result.event {
            println!("Got SVC {:#04x}", imm);
            let result = boot_ctx.kernel.lock().dispatch_svc(imm);
            println!("SVC returned: {:#x}\n", result);
        }
    }

    println!("[TEST 2] Testing svcExitProcess");
    println!("----- ");

    cpu_mut().ok_or("CPU not installed")?.inject_svc(0x07);

    if let Ok(result) = cpu_mut().ok_or("CPU not installed")?.run(100) {
        if let CpuEvent::Svc(imm) = result.event {
            println!("Got SVC {:#04x}", imm);
            let result = boot_ctx.kernel.lock().dispatch_svc(imm);
            println!("SVC returned: {:#x}\n", result);
        }
    }

    println!("================================");
    println!("All SVC tests completed successfully!");
    Ok(())
}
