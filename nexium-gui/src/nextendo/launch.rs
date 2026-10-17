use super::{bcat, npln};
use nexium_common::nextendo::{self, OnlinePatch};
use std::sync::Arc;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(10);
const USER_AGENT: &str = concat!("NeXium/", env!("CARGO_PKG_VERSION"));

pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .http_status_as_error(false)
        .user_agent(USER_AGENT)
        .build()
        .into()
}

pub fn install_hook() {
    nextendo::set_launch_hook(Arc::new(prepare));
}

fn prepare(title_id: u64, build: &str) -> Option<OnlinePatch> {
    if !nextendo::redirect().enabled {
        return None;
    }
    let agent = agent();
    if nextendo::bcat_package_title(title_id).is_some() {
        match bcat::sync(&agent, title_id) {
            Ok(true) => log::info!("nextendo: BCAT content for {:016X} is up to date", title_id),
            Ok(false) => log::debug!("nextendo: BCAT content for {:016X} unchanged", title_id),
            Err(error) => log::warn!("nextendo: BCAT update for {:016X} failed; keeping local content: {error}", title_id),
        }
    }
    npln::patch_for(&agent, title_id, build)
}
