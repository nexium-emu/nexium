use nexium_core::boot::{BootConfig, BootContext};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("debug,wgpu_core=warn,naga=warn")
    ).init();

    let nro_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "space-nx-master/spacenx.nro".to_string());

    let log_file = std::fs::File::create("debug_continue.log")?;
    let mut log = std::io::BufWriter::new(log_file);

    writeln!(log, "=== NeXium Debug Continue (ignore Break) ===")?;
    writeln!(log, "NRO: {}", nro_path)?;
    log.flush()?;

    if !Path::new(&nro_path).exists() {
        writeln!(log, "ERROR: NRO file not found: {}", nro_path)?;
        return Err(format!("NRO file not found: {}", nro_path).into());
    }

    let config = BootConfig::new(&nro_path);
    let mut boot_ctx = BootContext::new(config)?;
    let mut cpu = boot_ctx.cpu.take().expect("BootContext CPU not initialized");
    let _cpu_guard = nexium_kernel::kernel::cpu_local::set_current_cpu(&mut cpu, 0);
    use nexium_kernel::kernel::cpu_local::cpu_mut;

    writeln!(log, "Boot context initialized, starting emulation...")?;
    log.flush()?;

    let max_cycles = 500_000_000u64;
    let mut cycle_count = 0u64;
    let mut svc_count = 0u32;
    let mut break_count = 0u32;

    loop {
        let mut guard = boot_ctx.kernel.lock();
        if let Some(cpu) = cpu_mut() {
            let pc_before = cpu.get_pc();
            let event = cpu.run(100_000);
            cycle_count += 100_000;
            guard.cycle_count += 100_000;

            if guard.cycle_count >= guard.next_vsync_cycle {
                guard.next_vsync_cycle += 16_666_667;
                guard.display_ready = true;
            }

            match event {
                nexium_core::cpu::CpuEvent::Running => {
                    if cycle_count % 100_000_000 == 0 {
                        writeln!(log, "[{}] Running... PC={:#x}", cycle_count, pc_before)?;
                        log.flush()?;
                    }
                }
                nexium_core::cpu::CpuEvent::Svc(imm) => {
                    svc_count += 1;
                    writeln!(log, "[{}] SVC {:#04x} @ {:#x} (count: {})",
                        cycle_count, imm, pc_before, svc_count)?;
                    log.flush()?;

                    let result = guard.dispatch_svc(imm);

                    if let Some(cpu) = cpu_mut() {
                        cpu.set_register(0, result as u64);
                    }

                    // IMPORTANT: Don't exit on Break, just log it and continue
                    if imm == 0x26 {
                        break_count += 1;
                        writeln!(log, "[{}] Break #{} encountered, continuing execution...", cycle_count, break_count)?;
                        guard.process_exited = false;
                        log.flush()?;
                    }

                    for f in guard.drain_frames() {
                        writeln!(log, "[{}] Frame: {}x{}", cycle_count, f.width, f.height)?;
                    }
                }
                nexium_core::cpu::CpuEvent::Stalled => {
                    writeln!(log, "[{}] CPU stalled at {:#x}", cycle_count, pc_before)?;
                    break;
                }
                nexium_core::cpu::CpuEvent::Interrupted => {
                    writeln!(log, "[{}] CPU interrupted", cycle_count)?;
                    break;
                }
                nexium_core::cpu::CpuEvent::Exception(code) => {
                    writeln!(log, "[{}] CPU exception {:#x}", cycle_count, code)?;
                    break;
                }
            }

            if cycle_count > max_cycles {
                writeln!(log, "Max cycles exceeded")?;
                break;
            }
        } else {
            writeln!(log, "ERROR: CPU not initialized")?;
            break;
        }
    }

    writeln!(log, "\n=== Execution Complete ===")?;
    writeln!(log, "Total cycles: {}", cycle_count)?;
    writeln!(log, "Total SVCs: {}", svc_count)?;
    writeln!(log, "Break calls: {}", break_count)?;
    log.flush()?;

    println!("Debug continue completed. Check debug_continue.log for details.");
    Ok(())
}
