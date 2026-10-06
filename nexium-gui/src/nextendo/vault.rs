use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedSession {
    pub refresh_token: String,
    pub pid: u64,
    pub username: String,
    pub friend_code: String,
    #[serde(default)]
    pub scope: Option<String>,
}

fn path() -> PathBuf {
    nexium_common::paths::root()
        .join("nextendo")
        .join("session.dat")
}

pub fn load() -> Option<SavedSession> {
    let sealed = std::fs::read(path()).ok()?;
    let plain = unseal(&sealed)?;
    serde_json::from_slice(&plain).ok()
}

pub fn save(session: &SavedSession) -> std::io::Result<()> {
    let plain = serde_json::to_vec(session)?;
    let sealed = seal(&plain).ok_or_else(|| std::io::Error::other("could not protect the session"))?;
    let target = path();
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let staging = target.with_extension("tmp");
    write_private(&staging, &sealed)?;
    std::fs::rename(&staging, &target)
}

pub fn clear() {
    let _ = std::fs::remove_file(path());
}

#[cfg(unix)]
fn write_private(target: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(target)?;
    file.write_all(bytes)
}

#[cfg(not(unix))]
fn write_private(target: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(target, bytes)
}

#[cfg(windows)]
const ENTROPY: &[u8] = b"NeXium.Nextendo.Session";

#[cfg(windows)]
fn seal(plain: &[u8]) -> Option<Vec<u8>> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: plain.len() as u32,
        pbData: plain.as_ptr() as *mut u8,
    };
    let entropy = CRYPT_INTEGER_BLOB {
        cbData: ENTROPY.len() as u32,
        pbData: ENTROPY.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let ok = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            &entropy,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if ok == 0 || output.pbData.is_null() {
        return None;
    }
    let sealed = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    unsafe {
        LocalFree(output.pbData as _);
    }
    Some(sealed)
}

#[cfg(windows)]
fn unseal(sealed: &[u8]) -> Option<Vec<u8>> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: sealed.len() as u32,
        pbData: sealed.as_ptr() as *mut u8,
    };
    let entropy = CRYPT_INTEGER_BLOB {
        cbData: ENTROPY.len() as u32,
        pbData: ENTROPY.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let ok = unsafe {
        CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            &entropy,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if ok == 0 || output.pbData.is_null() {
        return None;
    }
    let plain = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    unsafe {
        LocalFree(output.pbData as _);
    }
    Some(plain)
}

#[cfg(not(windows))]
fn seal(plain: &[u8]) -> Option<Vec<u8>> {
    Some(plain.to_vec())
}

#[cfg(not(windows))]
fn unseal(sealed: &[u8]) -> Option<Vec<u8>> {
    Some(sealed.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealing_round_trips() {
        let secret = br#"{"refresh_token":"nxr_example"}"#;
        let sealed = seal(secret).unwrap();
        #[cfg(windows)]
        assert!(!sealed.windows(secret.len()).any(|window| window == secret));
        assert_eq!(unseal(&sealed).unwrap(), secret);
    }

    #[test]
    fn sessions_serialize_without_optional_fields() {
        let session: SavedSession =
            serde_json::from_str(r#"{"refresh_token":"r","pid":5,"username":"u","friend_code":"f"}"#)
                .unwrap();
        assert_eq!(session.pid, 5);
        assert_eq!(session.scope, None);
    }
}
