use crate::common::result::SUCCESS;

pub struct FileSystemService;

impl FileSystemService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("fs cmd: {}", cmd_id);
        match cmd_id {
            0 => self.cmd_open_sd_card(),
            1 => self.cmd_format_sd_card(),
            2 => self.cmd_delete_save_data(),
            3 => self.cmd_create_save_data(),
            _ => {
                log::warn!("unknown fs command: {}", cmd_id);
                1
            }
        }
    }

    fn cmd_open_sd_card(&self) -> u32 {
        log::debug!("FS::OpenSdCard");
        SUCCESS
    }

    fn cmd_format_sd_card(&self) -> u32 {
        log::debug!("FS::FormatSdCard");
        SUCCESS
    }

    fn cmd_delete_save_data(&self) -> u32 {
        log::debug!("FS::DeleteSaveData");
        SUCCESS
    }

    fn cmd_create_save_data(&self) -> u32 {
        log::debug!("FS::CreateSaveData");
        SUCCESS
    }
}

impl Default for FileSystemService {
    fn default() -> Self {
        Self::new()
    }
}
