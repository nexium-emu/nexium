pub struct FriendsService;

impl FriendsService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("friends cmd: {}", cmd_id);
        0
    }
}

impl Default for FriendsService {
    fn default() -> Self {
        Self::new()
    }
}
