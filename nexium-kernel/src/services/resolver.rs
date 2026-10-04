const GAI_NO_DATA: i32 = 7;
const NETDB_HOST_NOT_FOUND: i32 = 1;

pub fn sfdnsres_lookup_reply(cmd_id: u32) -> Option<Vec<u8>> {
    let words: &[i32] = match cmd_id {
        2 | 6 => &[NETDB_HOST_NOT_FOUND, GAI_NO_DATA, 0],
        10 | 12 => &[0, GAI_NO_DATA, NETDB_HOST_NOT_FOUND, 0],
        _ => return None,
    };
    Some(words.iter().flat_map(|word| word.to_le_bytes()).collect())
}

pub fn nsd_resolve(fqdn: &str) -> String {
    fqdn.replace('%', "lp1")
}

pub fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_report_host_not_found_in_both_layouts() {
        let words = |cmd| {
            sfdnsres_lookup_reply(cmd)
                .unwrap()
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(words(2), vec![NETDB_HOST_NOT_FOUND, GAI_NO_DATA, 0]);
        assert_eq!(words(12), vec![0, GAI_NO_DATA, NETDB_HOST_NOT_FOUND, 0]);
        assert!(sfdnsres_lookup_reply(14).is_none());
    }

    #[test]
    fn nsd_substitutes_the_environment() {
        assert_eq!(nsd_resolve("api.%.example.com"), "api.lp1.example.com");
        assert_eq!(c_string(b"host\0junk"), "host");
    }
}
