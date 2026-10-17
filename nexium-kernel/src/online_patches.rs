mod data;

use nexium_common::nextendo::{self, OnlinePatch, PatchGuard};
use nexium_loader::{Application, LoadedModule};

const NSO_HEADER: usize = 0x100;
const PEACE_WALKER_BUILD: &str = "BCA1A793E41A4836EF3C03286B18E816EEC62338";
const PEACE_WALKER_US_LANGUAGE_OFFSET: u32 = 0x5F998;
const PEACE_WALKER_US_LANGUAGE_ORIGINAL: [u8; 4] = [0xD4, 0x00, 0x80, 0x52];
const PEACE_WALKER_US_LANGUAGE_PATCHED: [u8; 4] = [0x14, 0x00, 0x80, 0x52];
const DUNGEONS_2_BUILD: &str = "9E9D887D59F7F7DB176B35E3379258B9";
const PVZ_BUILDS: [(&str, u32); 2] = [
    ("D89E062DA1D854AC9D98E2D664DA43D4", 0x86E2C78),
    ("5AD255D6667B6EB5BDCF74B74BCD2602", 0x86E2DD8),
];

const PATCH_DEPENDENT_TITLES: &[u64] = &[
    nextendo::SPLATOON_3,
    0x010015100B514000,
    0x0100C9A00ECE6000,
    0x0100C6F01C4F8000,
    nextendo::PEACE_WALKER_US,
    nextendo::CRASH_TEAM_RACING,
    0x01006FD0080B2000,
    nextendo::POKEMON_SCARLET,
    nextendo::POKEMON_VIOLET,
    nextendo::POKEMON_LEGENDS_ZA,
    0x0100453019AA8000,
    0x0100965017338000,
    0x0100B04011742000,
    nextendo::MINECRAFT_DUNGEONS_2,
    nextendo::PVZ_BATTLE_FOR_NEIGHBORVILLE,
];

struct Builtin {
    build: &'static str,
    title: Option<u64>,
    label: &'static str,
    patches: &'static [&'static [u8]],
}

const BUILTINS: &[Builtin] = &[
    Builtin {
        build: "6830B3A12406CB4716FEC5ADDC35D3E2DC92D212",
        title: None,
        label: "Splatoon 3",
        patches: &[&data::SPLATOON_3_CERTIFICATE, &data::SPLATOON_3_PEER_NAME],
    },
    Builtin {
        build: "726D2B882DD9EF10F4A9D73EED088740630FB6C8",
        title: None,
        label: "Splatoon 3",
        patches: &[&data::SPLATOON_3_CERTIFICATE],
    },
    Builtin {
        build: "28C4287AEE36F7499DA60F3E68B54C70DA382D75",
        title: None,
        label: "Splatoon 3 11.3.0",
        patches: &[&data::SPLATOON_3_CERTIFICATE, &data::SPLATOON_3_PEER_NAME],
    },
    Builtin {
        build: "FF773E90972D544EB79406EAA65396D53C43EFB9",
        title: None,
        label: "Super Mario Bros. Wonder 1.2.1",
        patches: &[&data::WONDER_121_CERTIFICATE, &data::WONDER_121_PEER_NAME],
    },
    Builtin {
        build: "44EFA84EE9C32B466EA27215D3D91DFD0EBAE625",
        title: None,
        label: "Nintendo 64 4.2.0",
        patches: &[&data::N64_420_CERTIFICATE, &data::N64_420_PEER_NAME],
    },
    Builtin {
        build: PEACE_WALKER_BUILD,
        title: None,
        label: "Metal Gear Solid: Peace Walker 1.3.0",
        patches: &[&data::PEACE_WALKER_CERTIFICATE, &data::PEACE_WALKER_PEER_NAME],
    },
    Builtin {
        build: "1C689518406930512C13DDF4217E7676",
        title: None,
        label: "Crash Team Racing Nitro-Fueled 1.0.15",
        patches: &[&data::CTR_SERVER_TRUST, &data::CTR_FRIEND_LOOKUP],
    },
    Builtin {
        build: "616640F27B9362502D1CE10BF01EB9A6",
        title: None,
        label: "Overcooked! 2 1.0.19",
        patches: &[&data::OVERCOOKED_2],
    },
    Builtin {
        build: "709BFD66115298640155FCC4979DBA151C7CC79A",
        title: None,
        label: "Pokémon Violet 4.0.0",
        patches: &[&data::POKEMON_SV],
    },
    Builtin {
        build: "421C5411B487EB4D049DD065FEC9547773E8E598",
        title: None,
        label: "Pokémon Scarlet 4.0.0",
        patches: &[&data::POKEMON_SV],
    },
    Builtin {
        build: "B1F12FD919EAE86AB8A978317677E64BCE443D1F",
        title: None,
        label: "Pokémon Legends: Z-A 2.0.2",
        patches: &[&data::POKEMON_ZA],
    },
    Builtin {
        build: "EC4ED46BBC4A9EBBE7CE0AF7A4A8EBBC09E5724A",
        title: None,
        label: "Xenoblade Chronicles X: Definitive Edition 2.0.0",
        patches: &[&data::XENOBLADE_X],
    },
    Builtin {
        build: "D70CE1726C979E45ED4200E9C2B23B2A258FDC18",
        title: None,
        label: "Super Mario Party Jamboree 2.3.0",
        patches: &[&data::JAMBOREE_230_CERTIFICATE],
    },
    Builtin {
        build: "C9A3DD7702075ECD90101F45553895822804D38F",
        title: None,
        label: "Monster Hunter Rise 16.0.2",
        patches: &[&data::MONSTER_HUNTER_RISE],
    },
    Builtin {
        build: DUNGEONS_2_BUILD,
        title: Some(nextendo::MINECRAFT_DUNGEONS_2),
        label: "Minecraft Dungeons II 1.1.1.0 listen host",
        patches: &[&data::DUNGEONS_2_HOST],
    },
];

pub fn full_build_id(build_id: &[u8; 32]) -> String {
    build_id.iter().map(|byte| format!("{byte:02X}")).collect()
}

pub fn trimmed_build_id(build_id: &[u8; 32]) -> String {
    full_build_id(build_id).trim_end_matches('0').to_string()
}

pub fn ips32(records: &[(u32, &[u8])]) -> Vec<u8> {
    let mut out = b"IPS32".to_vec();
    for (offset, bytes) in records {
        out.extend_from_slice(&offset.to_be_bytes());
        out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        out.extend_from_slice(bytes);
    }
    out.extend_from_slice(b"EEOF");
    out
}

pub fn guards_hold(image: &[u8], guards: &[PatchGuard]) -> bool {
    guards.iter().all(|guard| {
        let Some(start) = (guard.offset as usize).checked_sub(NSO_HEADER) else {
            return false;
        };
        let Some(window) = image.get(start..start + guard.original.len()) else {
            return false;
        };
        window == guard.original.as_slice() || window == guard.patched.as_slice()
    })
}

fn apply_to_module(module: &mut LoadedModule, ips: &[u8]) -> Result<usize, String> {
    nexium_loader::patch::apply_ips(&mut module.nso.module_image, ips, NSO_HEADER)
}

fn guarded(offset: u32, original: &[u8], patched: &[u8]) -> (Vec<u8>, Vec<PatchGuard>) {
    (
        ips32(&[(offset, patched)]),
        vec![PatchGuard {
            offset,
            original: original.to_vec(),
            patched: patched.to_vec(),
        }],
    )
}

fn apply_guarded(module: &mut LoadedModule, label: &str, ips: &[u8], guards: &[PatchGuard]) -> bool {
    if !guards_hold(&module.nso.module_image, guards) {
        log::error!("[nextendo] {label}: original instructions do not match; patch skipped");
        return false;
    }
    match apply_to_module(module, ips) {
        Ok(bytes) => {
            log::info!("[nextendo] {label}: applied ({bytes} bytes)");
            true
        }
        Err(error) => {
            log::error!("[nextendo] {label}: failed to apply: {error}");
            false
        }
    }
}

fn apply_startup_fixes(app: &mut Application, build: &str) {
    if app.title_id != nextendo::PEACE_WALKER_US || build != PEACE_WALKER_BUILD {
        return;
    }
    let (ips, guards) = guarded(
        PEACE_WALKER_US_LANGUAGE_OFFSET,
        &PEACE_WALKER_US_LANGUAGE_ORIGINAL,
        &PEACE_WALKER_US_LANGUAGE_PATCHED,
    );
    if let Some(module) = app.modules.iter_mut().find(|module| module.name == "main") {
        apply_guarded(module, "Peace Walker US direct-launch language fallback", &ips, &guards);
    }
}

fn apply_catalog(app: &mut Application, patch: &OnlinePatch) -> bool {
    if !patch.enabled {
        log::warn!("[nextendo] NPLN patch {} withdrawn by the signed catalog", patch.id);
        return false;
    }
    let Some(module) = app.modules.iter_mut().find(|module| module.name == "main") else {
        return false;
    };
    apply_guarded(
        module,
        &format!("NPLN catalog patch {} v{}", patch.id, patch.version),
        &patch.ips,
        &patch.guards,
    )
}

fn apply_pvz(app: &mut Application, build: &str) -> bool {
    if app.title_id != nextendo::PVZ_BATTLE_FOR_NEIGHBORVILLE {
        return false;
    }
    let Some((_, modulus_offset)) = PVZ_BUILDS.iter().find(|(known, _)| *known == build) else {
        return false;
    };
    let offset = NSO_HEADER as u32 + modulus_offset;
    let ips = ips32(&[(offset, &data::PVZ_NEXTENDO_MODULUS)]);
    let guards = vec![PatchGuard {
        offset,
        original: data::PVZ_EA_MODULUS_PREFIX.to_vec(),
        patched: data::PVZ_NEXTENDO_MODULUS[..data::PVZ_EA_MODULUS_PREFIX.len()].to_vec(),
    }];
    let Some(module) = app.modules.iter_mut().find(|module| module.name == "main") else {
        return false;
    };
    apply_guarded(module, "Plants vs. Zombies: Battle for Neighborville CA trust", &ips, &guards)
}

fn apply_builtin(app: &mut Application, build: &str) -> bool {
    let mut applied = apply_pvz(app, build);
    let title_id = app.title_id;
    for entry in BUILTINS {
        if entry.build != build || entry.title.is_some_and(|title| title != title_id) {
            continue;
        }
        for module in app.modules.iter_mut() {
            if trimmed_build_id(&module.nso.build_id) != build {
                continue;
            }
            for (index, ips) in entry.patches.iter().enumerate() {
                match apply_to_module(module, ips) {
                    Ok(bytes) => {
                        log::info!(
                            "[nextendo] {}: built-in patch #{} applied to {} ({bytes} bytes)",
                            entry.label,
                            index + 1,
                            module.name
                        );
                        applied = true;
                    }
                    Err(error) => log::error!(
                        "[nextendo] {}: built-in patch #{} failed to apply: {error}",
                        entry.label,
                        index + 1
                    ),
                }
            }
        }
    }
    applied
}

pub fn prepare_and_apply(app: &mut Application) -> Result<(), String> {
    let Some(main) = app.modules.iter().find(|module| module.name == "main") else {
        return Ok(());
    };
    let full = full_build_id(&main.nso.build_id);
    let build = trimmed_build_id(&main.nso.build_id);
    let catalog = nextendo::prepare_launch(app.title_id, &full);
    apply_startup_fixes(app, &build);
    if !nextendo::redirect().enabled {
        return Ok(());
    }
    let applied = match &catalog {
        Some(patch) => apply_catalog(app, patch),
        None => apply_builtin(app, &build),
    };
    if !applied && catalog.is_none() && PATCH_DEPENDENT_TITLES.contains(&app.title_id) {
        log::error!(
            "[nextendo] no online patch is known for build {build} of {:016X}; the game keeps its certificate pinning and online play will fail. Check that the supported update is installed and enabled.",
            app.title_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ids_follow_the_forks_trimming_rule() {
        let mut id = [0u8; 32];
        id[..4].copy_from_slice(&[0xAB, 0xCD, 0xE0, 0x10]);
        assert_eq!(trimmed_build_id(&id), "ABCDE01");
        assert_eq!(full_build_id(&id).len(), 64);
        assert!(full_build_id(&id).ends_with("0000"));
    }

    #[test]
    fn ips32_records_round_trip_through_the_loader() {
        let ips = ips32(&[(0x104, &[1, 2, 3]), (0x110, &[9])]);
        assert_eq!(&ips[..5], b"IPS32");
        assert_eq!(&ips[ips.len() - 4..], b"EEOF");
        let mut image = vec![0u8; 0x20];
        assert_eq!(nexium_loader::patch::apply_ips(&mut image, &ips, NSO_HEADER).unwrap(), 4);
        assert_eq!(&image[4..7], &[1, 2, 3]);
        assert_eq!(image[0x10], 9);
    }

    #[test]
    fn guards_accept_original_or_already_patched_bytes() {
        let guards = vec![PatchGuard {
            offset: 0x108,
            original: vec![0xAA, 0xBB],
            patched: vec![0x11, 0x22],
        }];
        let mut image = vec![0u8; 0x10];
        assert!(!guards_hold(&image, &guards));
        image[8..10].copy_from_slice(&[0xAA, 0xBB]);
        assert!(guards_hold(&image, &guards));
        image[8..10].copy_from_slice(&[0x11, 0x22]);
        assert!(guards_hold(&image, &guards));
        assert!(!guards_hold(&image[..9], &guards));
        let below_header = vec![PatchGuard {
            offset: 0x10,
            original: vec![0],
            patched: vec![0],
        }];
        assert!(!guards_hold(&image, &below_header));
    }

    #[test]
    fn every_built_in_patch_is_well_formed() {
        let mut image = vec![0u8; 0x0800_0000];
        for entry in BUILTINS {
            for ips in entry.patches {
                let written = nexium_loader::patch::apply_ips(&mut image, ips, NSO_HEADER)
                    .unwrap_or_else(|error| panic!("{}: {error}", entry.label));
                assert!(written > 0, "{}", entry.label);
            }
        }
        assert_eq!(data::PVZ_NEXTENDO_MODULUS.len(), 256);
        assert_eq!(&data::DUNGEONS_2_HOST[..5], b"IPS32");
        assert_eq!(&data::CTR_SERVER_TRUST[5..9], &[0x04, 0x0B, 0x4E, 0x40]);
    }
}
