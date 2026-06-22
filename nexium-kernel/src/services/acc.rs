pub struct AccountService;

impl AccountService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("acc cmd: {}", cmd_id);
        0
    }
}

impl Default for AccountService {
    fn default() -> Self {
        Self::new()
    }
}
