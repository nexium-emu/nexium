pub mod nro;
pub mod env;

pub use nro::Nro;
pub use env::EnvBlockBuilder;

pub struct Loader;

impl Loader {
    pub fn load_nro(path: &str) -> Result<Nro, String> {
        Nro::load_from_file(path)
    }
}
