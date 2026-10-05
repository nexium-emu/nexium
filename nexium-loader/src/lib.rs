pub mod application;
mod bktr;
pub mod bin_read;
pub mod cnmt;
pub mod container;
pub mod control;
pub mod content;
pub mod env;
pub mod firmware;
pub mod layered;
pub mod mods;
pub mod patch;
pub mod nca;
pub mod npdm;
pub mod nro;
pub mod nso;
pub mod romfs;

pub use application::{detect, read_application_title_id, Application, ContainerKind, LazyRomfs, LoadedModule};
pub use layered::{find_overlay_dir, AppRomfs, LayeredRomfs};
pub use control::read_container_metadata;
pub use env::EnvBlockBuilder;
pub use nro::{read_nro_metadata, Nro, NroMetadata};

pub enum LoadedProgram {
    Nro(Nro),
    Application(Application),
}

pub struct Loader;

impl Loader {
    pub fn load_nro(path: &str) -> Result<Nro, String> {
        Nro::load_from_file(path)
    }

    pub fn load_application(path: &str) -> Result<Application, String> {
        Application::load(path)
    }

    pub fn load_any(path: &str) -> Result<LoadedProgram, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {}: {}", path, e))?;
        match application::detect(path, &mmap) {
            ContainerKind::Nro => Ok(LoadedProgram::Nro(Nro::parse_mmap(std::sync::Arc::new(
                mmap,
            ))?)),
            ContainerKind::Unknown => Err(format!("unrecognized file format: {}", path)),
            _ => Ok(LoadedProgram::Application(Application::load(path)?)),
        }
    }
}

pub fn load_any(path: &str) -> Result<LoadedProgram, String> {
    Loader::load_any(path)
}
