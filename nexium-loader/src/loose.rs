use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::application::{place_module, Application, MODULE_ORDER};
use crate::npdm::Npdm;
use crate::nro::{parse_nacp, NroMetadata};

const EXEFS_MAIN: &str = "main";
const EXEFS_NPDM: &str = "main.npdm";
const PRODUCT_TABLE_PREFIX: &str = "ResourceSizeTable.Product.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LooseTitle {
    pub exefs: PathBuf,
    pub romfs: Option<PathBuf>,
}

impl LooseTitle {
    pub fn resolve(path: &Path) -> Option<Self> {
        let exefs = if path.is_dir() {
            if is_exefs_dir(path) {
                path.to_path_buf()
            } else {
                child(path, "exefs").filter(|dir| is_exefs_dir(dir))?
            }
        } else if path.file_name().is_some_and(|name| name == EXEFS_MAIN) {
            path.parent().filter(|dir| is_exefs_dir(dir))?.to_path_buf()
        } else {
            return None;
        };
        let romfs = exefs
            .parent()
            .and_then(|parent| child(parent, "romfs"))
            .filter(|dir| dir.is_dir());
        Some(Self { exefs, romfs })
    }

    pub fn root(&self) -> &Path {
        match self.exefs.parent() {
            Some(parent) if self.exefs.file_name().is_some_and(|name| name.eq_ignore_ascii_case("exefs")) => parent,
            _ => &self.exefs,
        }
    }

    pub fn name(&self) -> String {
        self.root()
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Unknown".into())
    }

    pub fn display_name(&self) -> String {
        let name = self.name();
        let mut title = name.trim();
        while let Some(stripped) = title
            .strip_suffix(']')
            .and_then(|rest| rest.rfind('[').map(|open| &rest[..open]))
            .or_else(|| title.strip_suffix(')').and_then(|rest| rest.rfind('(').map(|open| &rest[..open])))
        {
            title = stripped.trim_end();
        }
        if title.is_empty() { name } else { title.to_string() }
    }

    pub fn module(&self, name: &str) -> Option<PathBuf> {
        Some(self.exefs.join(name)).filter(|path| path.is_file())
    }

    pub fn npdm(&self) -> Result<Npdm, String> {
        let path = self.exefs.join(EXEFS_NPDM);
        let bytes = std::fs::read(&path).map_err(|error| format!("Read {}: {error}", path.display()))?;
        Npdm::parse(&bytes).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn is_switch2(&self) -> bool {
        self.npdm().is_ok_and(|npdm| npdm.is_switch2())
    }

    pub fn product_version(&self) -> Option<String> {
        let resources = child(&child(self.romfs.as_deref()?, "System")?, "Resource")?;
        let best = std::fs::read_dir(resources)
            .ok()?
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let digits = name.strip_prefix(PRODUCT_TABLE_PREFIX)?.get(..4)?.strip_suffix('.')?.to_string();
                digits.bytes().all(|byte| byte.is_ascii_digit()).then_some(digits)
            })
            .max()?;
        let digits = best.as_bytes();
        Some(format!("{}.{}.{}", digits[0] as char, digits[1] as char, digits[2] as char))
    }

    pub fn control(&self) -> Option<Vec<u8>> {
        self.control_dirs()
            .into_iter()
            .filter_map(|dir| child(&dir, "control.nacp"))
            .find_map(|path| std::fs::read(path).ok().filter(|bytes| bytes.len() >= 0x4000))
    }

    pub fn icon(&self) -> Option<Vec<u8>> {
        self.control_dirs().into_iter().find_map(|dir| {
            let mut entries = std::fs::read_dir(dir).ok()?.flatten().collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.file_name());
            entries.into_iter().find_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let icon = (name.starts_with("icon") && name.ends_with(".dat"))
                    || name.ends_with(".jpg")
                    || name.ends_with(".jpeg")
                    || name.ends_with(".png");
                icon.then(|| std::fs::read(entry.path()).ok()).flatten().filter(|bytes| !bytes.is_empty())
            })
        })
    }

    pub fn metadata(&self) -> NroMetadata {
        let (title, author, version) = self.control().as_deref().map(parse_nacp).unwrap_or_default();
        NroMetadata {
            title: if title.is_empty() { self.display_name() } else { title },
            author,
            version: if version.is_empty() { self.product_version().unwrap_or_default() } else { version },
            icon_jpeg: self.icon(),
        }
    }

    pub fn content_size(&self) -> u64 {
        [Some(&self.exefs), self.romfs.as_ref()].into_iter().flatten().map(|dir| tree_size(dir)).sum()
    }

    fn control_dirs(&self) -> Vec<PathBuf> {
        let root = self.root().to_path_buf();
        let mut dirs = vec![self.exefs.clone()];
        if root != self.exefs {
            dirs.push(root.clone());
        }
        dirs.extend(child(&root, "control").filter(|dir| dir.is_dir()));
        dirs
    }
}

impl Application {
    pub fn load_loose(title: &LooseTitle) -> Result<Self, String> {
        let npdm_path = title.exefs.join(EXEFS_NPDM);
        let mmap = map_file(&npdm_path)?;
        let npdm = Npdm::parse(&mmap).map_err(|error| format!("{}: {error}", npdm_path.display()))?;
        log::info!(
            "loose ExeFS {} (npdm acid={:?} title_id={:#018x}) romfs={}",
            title.exefs.display(),
            npdm.acid_layout(),
            npdm.title_id,
            title.romfs.as_deref().map_or("none".into(), |romfs| romfs.display().to_string())
        );

        let mut modules = Vec::new();
        let mut total_code_size = 0;
        for name in MODULE_ORDER {
            if let Some(path) = title.module(name) {
                place_module(&mut modules, &mut total_code_size, name, &map_file(&path)?)?;
            }
        }
        if modules.iter().all(|module| module.name != EXEFS_MAIN) {
            return Err(format!("{} has no loadable main module", title.exefs.display()));
        }

        let application_control = title.control();
        let display_version = application_control
            .as_deref()
            .map(|nacp| parse_nacp(nacp).2)
            .filter(|version| !version.is_empty())
            .or_else(|| title.product_version())
            .unwrap_or_default();
        log::info!(
            "application title_id={:#018x} addr_space={:?} stack={:#x} code_size={:#x} modules={} version={:?}",
            npdm.title_id,
            npdm.address_space,
            npdm.main_stack_size,
            total_code_size,
            modules.len(),
            display_version
        );

        Ok(Self {
            mmap,
            modules,
            total_code_size,
            title_id: npdm.title_id,
            npdm,
            romfs: None,
            system_romfs: HashMap::new(),
            content_version: 0,
            display_version,
            application_control,
            patch_romfs: None,
            add_on_content: Default::default(),
            exefs_files: Vec::new(),
            loose: Some(title.clone()),
        })
    }
}

fn is_exefs_dir(dir: &Path) -> bool {
    dir.join(EXEFS_MAIN).is_file() && dir.join(EXEFS_NPDM).is_file()
}

fn child(dir: &Path, name: &str) -> Option<PathBuf> {
    let exact = dir.join(name);
    if exact.exists() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|entry| entry.file_name().to_str().is_some_and(|entry| entry.eq_ignore_ascii_case(name)))
        .map(|entry| entry.path())
}

fn tree_size(root: &Path) -> u64 {
    let mut total = 0;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => pending.push(entry.path()),
                Ok(_) => total += entry.metadata().map_or(0, |metadata| metadata.len()),
                Err(_) => {}
            }
        }
    }
    total
}

fn map_file(path: &Path) -> Result<Arc<Mmap>, String> {
    let file = std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    unsafe { Mmap::map(&file) }
        .map(Arc::new)
        .map_err(|error| format!("mmap {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("nexium-loose-{}-{id}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, path: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn nso(fill: u8) -> Vec<u8> {
        let mut bytes = vec![0; 0x130];
        bytes[..4].copy_from_slice(b"NSO0");
        for index in 0..3usize {
            let header = 0x10 + index * 0x10;
            let file = 0x100 + index * 0x10;
            bytes[header..header + 4].copy_from_slice(&(file as u32).to_le_bytes());
            bytes[header + 4..header + 8].copy_from_slice(&(index as u32 * 0x1000).to_le_bytes());
            bytes[header + 8..header + 12].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[0x60 + index * 4..0x64 + index * 4].copy_from_slice(&0x10u32.to_le_bytes());
            bytes[file..file + 0x10].fill(fill);
        }
        bytes
    }

    fn switch2_npdm(program_id: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 0x800];
        bytes[..4].copy_from_slice(b"META");
        bytes[0x0C] = 0x37;
        bytes[0x0E] = 0x2C;
        bytes[0x1C..0x20].copy_from_slice(&0x10_0000u32.to_le_bytes());
        for (offset, value) in [(0x70u32, 0x600u32), (0x74, 0x40), (0x78, 0x80), (0x7C, 0x340)] {
            bytes[offset as usize..offset as usize + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[0x380..0x384].copy_from_slice(b"ACID");
        bytes[0x600..0x604].copy_from_slice(b"ACI0");
        bytes[0x610..0x618].copy_from_slice(&program_id.to_le_bytes());
        bytes
    }

    fn title(dir: &TestDir) -> PathBuf {
        dir.write("Game/exefs/main", &nso(1));
        dir.write("Game/exefs/rtld", &nso(2));
        dir.write("Game/exefs/sdk", &nso(3));
        dir.write("Game/exefs/subsdk0", &nso(4));
        dir.write("Game/exefs/main.npdm", &switch2_npdm(0x0400_C3F0_0006_E000));
        dir.write("Game/romfs/System/Resource/ResourceSizeTable.Product.100.Nin.rsizetable.zs", b"t");
        dir.write("Game/romfs/System/Resource/ResourceSizeTable.Product.180.Nin.rsizetable.zs", b"t");
        dir.write("Game/romfs/System/Resource/ResourceSizeTable.Pilot.200.Nin.rsizetable.zs", b"t");
        dir.write("Game/romfs/Data/file.bin", b"payload");
        dir.0.join("Game")
    }

    #[test]
    fn title_folder_exefs_folder_and_main_resolve_to_the_same_title() {
        let dir = TestDir::new();
        let root = title(&dir);
        let expected = LooseTitle { exefs: root.join("exefs"), romfs: Some(root.join("romfs")) };
        for path in [root.clone(), root.join("exefs"), root.join("exefs/main")] {
            assert_eq!(LooseTitle::resolve(&path).as_ref(), Some(&expected), "{}", path.display());
        }
        assert_eq!(expected.root(), root);
        assert_eq!(expected.name(), "Game");
        assert!(LooseTitle::resolve(&root.join("romfs")).is_none());
        assert!(LooseTitle::resolve(&root.join("exefs/sdk")).is_none());
        assert!(LooseTitle::resolve(&root.join("exefs/main.npdm")).is_none());
    }

    #[test]
    fn exefs_without_npdm_or_romfs_is_handled() {
        let dir = TestDir::new();
        dir.write("NoNpdm/exefs/main", &nso(1));
        assert!(LooseTitle::resolve(&dir.0.join("NoNpdm")).is_none());
        dir.write("CodeOnly/main", &nso(1));
        dir.write("CodeOnly/main.npdm", &switch2_npdm(1));
        let code_only = LooseTitle::resolve(&dir.0.join("CodeOnly")).unwrap();
        assert_eq!(code_only.romfs, None);
        assert_eq!(code_only.root(), dir.0.join("CodeOnly"));
    }

    #[test]
    fn upper_case_title_folders_resolve() {
        let dir = TestDir::new();
        dir.write("Upper/ExeFS/main", &nso(1));
        dir.write("Upper/ExeFS/main.npdm", &switch2_npdm(1));
        std::fs::create_dir_all(dir.0.join("Upper/RomFS")).unwrap();
        let title = LooseTitle::resolve(&dir.0.join("Upper")).unwrap();
        assert_eq!(title.exefs, dir.0.join("Upper/ExeFS"));
        assert_eq!(title.romfs, Some(dir.0.join("Upper/RomFS")));
        assert_eq!(title.root(), dir.0.join("Upper"));
    }

    #[test]
    fn loads_every_module_in_layout_order_with_the_sibling_romfs() {
        let dir = TestDir::new();
        let root = title(&dir);
        let title = LooseTitle::resolve(&root).unwrap();
        assert!(title.is_switch2());
        assert_eq!(title.product_version().as_deref(), Some("1.8.0"));
        let app = Application::load_loose(&title).unwrap();
        assert_eq!(app.title_id, 0x0400_C3F0_0006_E000);
        assert!(app.npdm.is_switch2());
        assert_eq!(
            app.modules.iter().map(|module| (module.name.as_str(), module.load_offset)).collect::<Vec<_>>(),
            [("rtld", 0), ("main", 0x3000), ("subsdk0", 0x6000), ("sdk", 0x9000)]
        );
        assert_eq!(app.total_code_size, 0xC000);
        assert_eq!(app.display_version, "1.8.0");
        assert_eq!(app.loose.as_ref().and_then(|title| title.romfs.clone()), Some(root.join("romfs")));
        assert!(app.romfs.is_none());
        let modules = 4 * nso(0).len() as u64 + switch2_npdm(0).len() as u64;
        assert_eq!(title.content_size(), modules + 3 + 7);

        let romfs = crate::mods::build_mod_romfs(&app, &[]).unwrap().unwrap();
        let image = romfs.read(0, romfs.len() as usize).unwrap();
        assert_eq!(crate::romfs::romfs_file(&image, "/Data/file.bin").unwrap(), b"payload");
    }

    #[test]
    fn display_name_drops_dump_tags() {
        let dir = TestDir::new();
        for (folder, expected) in [
            ("Mario Kart World [0400C3F00006E000][v0]", "Mario Kart World"),
            ("Game (Rev 1) [0400000000001000] [v65536]", "Game"),
            ("Plain Name", "Plain Name"),
            ("[0400000000002000]", "[0400000000002000]"),
        ] {
            dir.write(&format!("{folder}/exefs/main"), &nso(1));
            dir.write(&format!("{folder}/exefs/main.npdm"), &switch2_npdm(1));
            let title = LooseTitle::resolve(&dir.0.join(folder)).unwrap();
            assert_eq!(title.display_name(), expected, "{folder}");
            assert_eq!(title.metadata().title, expected, "{folder}");
        }
    }

    #[test]
    fn png_art_next_to_the_dump_becomes_the_icon() {
        let dir = TestDir::new();
        let root = title(&dir);
        dir.write("Game/icon.png", b"\x89PNG");
        let title = LooseTitle::resolve(&root).unwrap();
        assert_eq!(title.icon().as_deref(), Some(&b"\x89PNG"[..]));
    }

    #[test]
    fn metadata_falls_back_to_the_folder_name_and_romfs_version() {
        let dir = TestDir::new();
        let title = LooseTitle::resolve(&title(&dir)).unwrap();
        let metadata = title.metadata();
        assert_eq!(metadata.title, "Game");
        assert_eq!(metadata.version, "1.8.0");
        assert!(metadata.author.is_empty() && metadata.icon_jpeg.is_none());

        let mut nacp = vec![0u8; 0x4000];
        nacp[..9].copy_from_slice(b"Real Name");
        nacp[0x200..0x208].copy_from_slice(b"Nintendo");
        nacp[0x3060..0x3065].copy_from_slice(b"1.2.3");
        dir.write("Game/control/control.nacp", &nacp);
        dir.write("Game/control/icon_AmericanEnglish.dat", b"\xFF\xD8jpeg");
        let metadata = title.metadata();
        assert_eq!((metadata.title.as_str(), metadata.author.as_str()), ("Real Name", "Nintendo"));
        assert_eq!(metadata.version, "1.2.3");
        assert_eq!(metadata.icon_jpeg.as_deref(), Some(&b"\xFF\xD8jpeg"[..]));
        assert_eq!(Application::load_loose(&title).unwrap().display_version, "1.2.3");
    }
}
