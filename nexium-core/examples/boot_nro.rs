use nexium_core::boot::{BootConfig, BootContext};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NeXium Boot Utility");

    let nro_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "spacenx.nro".to_string());

    println!("Loading: {}", nro_path);

    match BootConfig::new(&nro_path) {
        config => {
            match BootContext::new(config) {
                Ok(mut boot_ctx) => {
                    println!("Boot context created successfully");
                    println!("Starting execution...");

                    match boot_ctx.run() {
                        Ok(_) => println!("Execution completed"),
                        Err(e) => {
                            eprintln!("Execution error: {}", e);
                            return Err(e.into());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create boot context: {}", e);
                    return Err(e.into());
                }
            }
        }
    }

    Ok(())
}
