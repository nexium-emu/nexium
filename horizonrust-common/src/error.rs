#[derive(Debug, thiserror::Error)]
pub enum HorizonError {
    #[error("CPU: {0}")]
    Cpu(String),
    #[error("Memory: {0}")]
    Memory(String),
    #[error("Loader: {0}")]
    Loader(String),
    #[error("Kernel: {0}")]
    Kernel(String),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error("IPC: {0}")]
    Ipc(String),
    #[error("Service: {0}")]
    Service(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HorizonError>;
