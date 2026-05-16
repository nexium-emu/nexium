fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();

    log::info!("HorizonRust - Nintendo Switch Emulator");
    log::info!("Phase 4 - GUI Integration");

    println!("HorizonRust GUI - Phase 4 stub");
    println!("To boot an NRO, provide the path as an argument");
    println!("Usage: horizonrust path/to/game.nro");
}
