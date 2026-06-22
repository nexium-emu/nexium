pub fn init_logger() {
    let _ = env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .try_init();
}

pub fn init_logger_with_filter(default_filter: &str) {
    let _ =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
            .format_timestamp_millis()
            .try_init();
}
