pub const RESULT_SUCCESS: u32 = 0;

pub const KERNEL_NOT_IMPLEMENTED: u32 = 1 | (33 << 9);
pub const KERNEL_INVALID_HANDLE: u32 = 1 | (114 << 9);
pub const KERNEL_INVALID_ADDRESS: u32 = 3 | (33 << 9);
pub const KERNEL_INVALID_SIZE: u32 = 4 | (33 << 9);
pub const KERNEL_INVALID_STATE: u32 = 5 | (33 << 9);
pub const KERNEL_INVALID_POINTER: u32 = 7 | (33 << 9);
pub const KERNEL_TIMEOUT: u32 = 1 | (117 << 9);
pub const KERNEL_CANCELLED: u32 = 1 | (118 << 9);
pub const KERNEL_PORT_NOT_FOUND: u32 = 131 | (33 << 9);

pub const FS_NOT_FOUND: u32 = 1 | (2 << 9) | (2 << 21);
pub const FS_PATH_NOT_FOUND: u32 = 2 | (2 << 9) | (2 << 21);

pub const AM_SESSION_NOT_FOUND: u32 = 1 | (150 << 9);

pub const VI_NOT_FOUND: u32 = 1 | (3 << 9) | (1 << 21);

pub const AUDIO_NOT_FOUND: u32 = 1 | (13 << 9);

pub const HID_NOT_FOUND: u32 = 1 | (107 << 9);

pub const TIME_NOT_FOUND: u32 = 1 | (116 << 9);
pub const TIME_INVALID_CLOCK: u32 = 0 | (116 << 9);

pub const SETTINGS_NOT_FOUND: u32 = 1 | (105 << 9);

pub const NVDRV_NOT_IMPLEMENTED: u32 = 1 | (2 << 9);

pub use self::RESULT_SUCCESS as SUCCESS;
