use nexium_core::boot::{BootConfig, BootContext};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("debug,wgpu_core=warn,naga=warn")
    ).init();
    let nro_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "space-nx-master/space-nx.nro".to_string());

    let log_file = std::fs::File::create("debug_boot.log")?;
    let mut log = std::io::BufWriter::new(log_file);

    writeln!(log, "=== NeXium Debug Boot ===")?;
    writeln!(log, "NRO: {}", nro_path)?;
    writeln!(log, "Starting boot context...")?;
    log.flush()?;

    if !Path::new(&nro_path).exists() {
        writeln!(log, "ERROR: NRO file not found: {}", nro_path)?;
        return Err(format!("NRO file not found: {}", nro_path).into());
    }

    let config = BootConfig::new(&nro_path);
    let mut boot_ctx = BootContext::new(config)?;

    writeln!(log, "Boot context initialized")?;
    writeln!(log, "Starting emulation loop")?;
    log.flush()?;

    let max_cycles = 200_000_000u64;
    let mut cycle_count = 0u64;
    let mut svc_count = 0u32;
    let mut pc_check_count = 0u32;
    let mut stuck_pc: Option<u64> = None;
    let mut stuck_count = 0u32;
    let mut last_svc_cycle = 0u64;

    if let Some(cpu) = &boot_ctx.kernel.cpu {
        writeln!(log, "Initial PC: {:#x}", cpu.get_pc())?;
        writeln!(log, "Initial SP: {:#x}", cpu.get_register(31))?;
    }
    log.flush()?;

    loop {
        if let Some(cpu) = &mut boot_ctx.kernel.cpu {
            let pc_before = cpu.get_pc();
            let event = cpu.run(100_000);
            let pc_after = cpu.get_pc();
            cycle_count += 100_000;
            boot_ctx.kernel.cycle_count += 100_000;

            if boot_ctx.kernel.cycle_count >= boot_ctx.kernel.next_vsync_cycle && !boot_ctx.kernel.display_ready {
                boot_ctx.kernel.display_ready = true;
            }

            if pc_check_count < 20 {
                writeln!(log, "[{}] CPU: {:#x} → {:#x} event={:?}",
                    cycle_count, pc_before, pc_after, event)?;
                pc_check_count += 1;
            }

            // Log every SVC call in detail
            if matches!(event, nexium_core::cpu::CpuEvent::Svc(_)) || cycle_count < 500_000 {
                if cycle_count % 50_000 == 0 {
                    writeln!(log, "[{}] >> PC {:#x}", cycle_count, cpu.get_pc())?;
                }
            }

            if pc_before == pc_after && matches!(event, nexium_core::cpu::CpuEvent::Running) {
                if stuck_pc == Some(pc_before) {
                    stuck_count += 1;
                    if stuck_count == 100 {
                        writeln!(log, "[{}] STUCK: PC {:#x} looping 10M+ cycles", cycle_count, pc_before)?;
                    }
                } else {
                    stuck_pc = Some(pc_before);
                    stuck_count = 1;
                }
            } else {
                stuck_pc = None;
                stuck_count = 0;
            }

            match event {
                nexium_core::cpu::CpuEvent::Running => {
                    if cycle_count % 50_000_000 == 0 {
                        writeln!(log, "[{}] Still running...", cycle_count)?;
                        log.flush()?;
                    }
                }
                nexium_core::cpu::CpuEvent::Svc(imm) => {
                    svc_count += 1;
                    last_svc_cycle = cycle_count;
                    writeln!(log, "[{}] SVC {:#04x} @ {:#x} (count: {})",
                        cycle_count, imm, pc_before, svc_count)?;
                    log.flush()?;

                    let result = boot_ctx.kernel.dispatch_svc(imm);

                    if let Some(cpu) = &mut boot_ctx.kernel.cpu {
                        cpu.set_register(0, result as u64);
                    }

                    for f in boot_ctx.kernel.drain_frames() {
                        writeln!(log, "[{}] Frame: {}x{}", cycle_count, f.width, f.height)?;
                    }

                    if boot_ctx.kernel.process_exited {
                        writeln!(log, "[{}] Process exited via svcBreak", cycle_count)?;
                        break;
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
                writeln!(log, "Max cycles ({}) exceeded", max_cycles)?;
                break;
            }

            if svc_count > 0 && (cycle_count - last_svc_cycle) > 100_000_000 {
                writeln!(log, "[{}] STUCK: No SVCs for 100M+ cycles. Last PC {:#x}", cycle_count, pc_before)?;
                break;
            }
        } else {
            writeln!(log, "ERROR: CPU not initialized")?;
            break;
        }
    }

    writeln!(log, "\n=== Emulation Complete ===")?;
    writeln!(log, "Total cycles: {}", cycle_count)?;
    writeln!(log, "Total SVCs: {}", svc_count)?;
    log.flush()?;

    println!("Debug boot completed. Check debug_boot.log for details.");
    Ok(())
}
