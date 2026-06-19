use crate::bin_read::{u16at, u48at, u64at, u8at};

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

pub struct Cnmt {
    pub title_id: u64,
    pub records: Vec<ContentRecord>,
}

impl Cnmt {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let title_id = u64at(bytes, 0)?;
        let table_offset = u16at(bytes, 0x0E)? as usize;
        let num_entries = u16at(bytes, 0x10)? as usize;

        let table = 0x20 + table_offset;
        let mut records = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let r = table + i * 0x38;
            let mut nca_id = [0u8; 16];
            let id_slice = crate::bin_read::slice(bytes, r + 0x20..r + 0x30)?;
            nca_id.copy_from_slice(id_slice);
            let size = u48at(bytes, r + 0x30)?;
            let content_type = ContentType::from_u8(u8at(bytes, r + 0x36)?);
            records.push(ContentRecord { nca_id, size, content_type });
        }

        Ok(Self { title_id, records })
    }

    pub fn find(&self, ct: ContentType) -> Option<&ContentRecord> {
        self.records.iter().find(|r| r.content_type == ct)
    }
}
