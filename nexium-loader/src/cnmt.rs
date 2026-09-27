use crate::bin_read::{u16at, u32at, u48at, u64at, u8at};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentType {
    Meta,
    Program,
    Data,
    Control,
    HtmlDocument,
    LegalInformation,
    DeltaFragment,
    Unknown,
}

impl ContentType {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ContentType::Meta,
            1 => ContentType::Program,
            2 => ContentType::Data,
            3 => ContentType::Control,
            4 => ContentType::HtmlDocument,
            5 => ContentType::LegalInformation,
            6 => ContentType::DeltaFragment,
            _ => ContentType::Unknown,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContentRecord {
    pub nca_id: [u8; 16],
    pub size: u64,
    pub content_type: ContentType,
    pub id_offset: u8,
}

impl ContentRecord {
    pub fn nca_filename(&self) -> String {
        let mut s = String::with_capacity(36);
        for b in &self.nca_id {
            s.push_str(&format!("{:02x}", b));
        }
        s.push_str(".nca");
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentMetaType {
    Application,
    Patch,
    AddOnContent,
    Other(u8),
}

#[derive(Clone, Debug)]
pub struct Cnmt {
    pub title_id: u64,
    pub version: u32,
    pub meta_type: ContentMetaType,
    pub application_id: u64,
    pub required_application_version: u32,
    pub records: Vec<ContentRecord>,
}

impl Cnmt {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let title_id = u64at(bytes, 0)?;
        let version = u32at(bytes, 8)?;
        let meta_type = match u8at(bytes, 0x0C)? {
            0x80 => ContentMetaType::Application,
            0x81 => ContentMetaType::Patch,
            0x82 => ContentMetaType::AddOnContent,
            value => ContentMetaType::Other(value),
        };
        let table_offset = u16at(bytes, 0x0E)? as usize;
        let minimum_header = match meta_type {
            ContentMetaType::Application | ContentMetaType::AddOnContent => 0x10,
            ContentMetaType::Patch => 0x18,
            ContentMetaType::Other(_) => 0,
        };
        if table_offset < minimum_header {
            return Err(format!("CNMT {title_id:016X} extended header is truncated"));
        }
        let application_id = match meta_type {
            ContentMetaType::Patch | ContentMetaType::AddOnContent => u64at(bytes, 0x20)?,
            _ => title_id,
        };
        let required_application_version = match meta_type {
            ContentMetaType::Application => u32at(bytes, 0x2C)?,
            ContentMetaType::AddOnContent => u32at(bytes, 0x28)?,
            _ => 0,
        };
        let num_entries = u16at(bytes, 0x10)? as usize;

        let table = 0x20 + table_offset;
        crate::bin_read::slice(bytes, 0..table + num_entries * 0x38)?;
        let mut records = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let r = table + i * 0x38;
            let mut nca_id = [0u8; 16];
            let id_slice = crate::bin_read::slice(bytes, r + 0x20..r + 0x30)?;
            nca_id.copy_from_slice(id_slice);
            let size = u48at(bytes, r + 0x30)?;
            let content_type = ContentType::from_u8(u8at(bytes, r + 0x36)?);
            records.push(ContentRecord {
                nca_id,
                size,
                content_type,
                id_offset: u8at(bytes, r + 0x37)?,
            });
        }

        Ok(Self {
            title_id,
            version,
            meta_type,
            application_id,
            required_application_version,
            records,
        })
    }

    pub fn find(&self, ct: ContentType) -> Option<&ContentRecord> {
        self.records.iter().filter(|r| r.content_type == ct).min_by_key(|r| r.id_offset)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_metadata_preserves_version_application_and_content_index() {
        let mut bytes = vec![0; 0x38 + 2 * 0x38];
        bytes[..8].copy_from_slice(&0x0100_1234_5678_0800u64.to_le_bytes());
        bytes[8..12].copy_from_slice(&196608u32.to_le_bytes());
        bytes[0xC] = 0x81;
        bytes[0xE..0x10].copy_from_slice(&0x18u16.to_le_bytes());
        bytes[0x10..0x12].copy_from_slice(&2u16.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&0x0100_1234_5678_0000u64.to_le_bytes());
        bytes[0x38 + 0x36] = 1;
        bytes[0x38 + 0x37] = 3;
        bytes[0x70 + 0x36] = 1;
        bytes[0x70 + 0x37] = 1;
        let parsed = Cnmt::parse(&bytes).unwrap();
        assert_eq!(parsed.meta_type, ContentMetaType::Patch);
        assert_eq!(parsed.version, 196608);
        assert_eq!(parsed.application_id, 0x0100_1234_5678_0000);
        assert_eq!(parsed.find(ContentType::Program).unwrap().id_offset, 1);
        bytes.truncate(bytes.len() - 1);
        assert!(Cnmt::parse(&bytes).is_err());
    }

    #[test]
    fn truncated_extended_header_is_rejected_before_reading_records() {
        let mut bytes = vec![0; 0x40];
        bytes[0xC] = 0x82;
        bytes[0xE] = 8;
        assert!(Cnmt::parse(&bytes).unwrap_err().contains("extended header"));
    }
}
