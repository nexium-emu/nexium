pub struct FileSystemService;

impl FileSystemService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("fs cmd: {}", cmd_id);
        0
    }
}

impl Default for FileSystemService {
    fn default() -> Self {
        Self::new()
    }
}
