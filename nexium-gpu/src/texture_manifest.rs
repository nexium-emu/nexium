use nexium_spirv::TextureNumericType;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(u8)]
pub enum GraphicsTextureImageKind {
    #[default]
    D2 = 0,
    D2Array = 1,
    D3 = 2,
    Cube = 3,
    CubeArray = 4,
    Buffer = 5,
}

impl GraphicsTextureImageKind {
    pub const fn spirv_kind(self) -> nexium_spirv::GraphicsImageKind {
        match self {
            Self::D2 => nexium_spirv::GraphicsImageKind::D2,
            Self::D2Array => nexium_spirv::GraphicsImageKind::D2Array,
            Self::D3 => nexium_spirv::GraphicsImageKind::D3,
            Self::Cube => nexium_spirv::GraphicsImageKind::Cube,
            Self::CubeArray => nexium_spirv::GraphicsImageKind::CubeArray,
            Self::Buffer => nexium_spirv::GraphicsImageKind::Buffer,
        }
    }
}

impl From<nexium_spirv::GraphicsImageKind> for GraphicsTextureImageKind {
    fn from(value: nexium_spirv::GraphicsImageKind) -> Self {
        match value {
            nexium_spirv::GraphicsImageKind::D2 => Self::D2,
            nexium_spirv::GraphicsImageKind::D2Array => Self::D2Array,
            nexium_spirv::GraphicsImageKind::D3 => Self::D3,
            nexium_spirv::GraphicsImageKind::Cube => Self::Cube,
            nexium_spirv::GraphicsImageKind::CubeArray => Self::CubeArray,
            nexium_spirv::GraphicsImageKind::Buffer => Self::Buffer,
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(u8)]
pub enum GraphicsTextureNumericType {
    #[default]
    Float = 0,
    Uint = 1,
    Sint = 2,
}

impl GraphicsTextureNumericType {
    pub const fn spirv_type(self) -> TextureNumericType {
        match self {
            Self::Float => TextureNumericType::Float,
            Self::Uint => TextureNumericType::Uint,
            Self::Sint => TextureNumericType::Sint,
        }
    }
}

impl From<TextureNumericType> for GraphicsTextureNumericType {
    fn from(value: TextureNumericType) -> Self {
        match value {
            TextureNumericType::Float => Self::Float,
            TextureNumericType::Uint => Self::Uint,
            TextureNumericType::Sint => Self::Sint,
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct TextureNumericBinding {
    pub shader_id: u32,
    pub descriptor_slot: u32,
    pub numeric_type: GraphicsTextureNumericType,
    pub image_kind: GraphicsTextureImageKind,
}

impl TextureNumericBinding {
    pub const fn new(
        shader_id: u32,
        descriptor_slot: u32,
        numeric_type: TextureNumericType,
    ) -> Self {
        let numeric_type = match numeric_type {
            TextureNumericType::Float => GraphicsTextureNumericType::Float,
            TextureNumericType::Uint => GraphicsTextureNumericType::Uint,
            TextureNumericType::Sint => GraphicsTextureNumericType::Sint,
        };
        Self {
            shader_id,
            descriptor_slot,
            numeric_type,
            image_kind: GraphicsTextureImageKind::D2,
        }
    }

    pub const fn with_image_kind(mut self, image_kind: GraphicsTextureImageKind) -> Self {
        self.image_kind = image_kind;
        self
    }

    pub const fn spirv_type(self) -> TextureNumericType {
        self.numeric_type.spirv_type()
    }

    pub const fn spirv_image_kind(self) -> nexium_spirv::GraphicsImageKind {
        self.image_kind.spirv_kind()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TextureNumericManifestError {
    #[error("graphics texture descriptor slot {0} is outside the 32-entry ABI")]
    InvalidSlot(u32),
    #[error(
        "graphics texture descriptor slot {slot} is assigned to both shader texture IDs {first_shader_id:#x} and {second_shader_id:#x}"
    )]
    SlotConflict {
        slot: u32,
        first_shader_id: u32,
        second_shader_id: u32,
    },
    #[error(
        "shader texture ID {shader_id:#x} at descriptor slot {slot} requires incompatible {first:?} and {second:?} numeric families"
    )]
    NumericConflict {
        shader_id: u32,
        slot: u32,
        first: GraphicsTextureNumericType,
        second: GraphicsTextureNumericType,
    },
    #[error(
        "shader texture ID {shader_id:#x} at descriptor slot {slot} requires incompatible {first:?} and {second:?} image families"
    )]
    ImageKindConflict {
        shader_id: u32,
        slot: u32,
        first: GraphicsTextureImageKind,
        second: GraphicsTextureImageKind,
    },
}

pub fn normalize_texture_numeric_manifest(
    mut bindings: Vec<TextureNumericBinding>,
) -> Result<Vec<TextureNumericBinding>, TextureNumericManifestError> {
    bindings.sort_unstable_by_key(|binding| {
        (
            binding.descriptor_slot,
            binding.shader_id,
            binding.numeric_type,
            binding.image_kind,
        )
    });

    let mut normalized: Vec<TextureNumericBinding> = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if binding.descriptor_slot >= crate::descriptor::MAX_TEXTURE_DESCRIPTORS {
            return Err(TextureNumericManifestError::InvalidSlot(
                binding.descriptor_slot,
            ));
        }
        if let Some(previous) = normalized.last().copied() {
            if previous.descriptor_slot == binding.descriptor_slot {
                if previous.shader_id != binding.shader_id {
                    return Err(TextureNumericManifestError::SlotConflict {
                        slot: binding.descriptor_slot,
                        first_shader_id: previous.shader_id,
                        second_shader_id: binding.shader_id,
                    });
                }
                if previous.numeric_type != binding.numeric_type {
                    return Err(TextureNumericManifestError::NumericConflict {
                        shader_id: binding.shader_id,
                        slot: binding.descriptor_slot,
                        first: previous.numeric_type,
                        second: binding.numeric_type,
                    });
                }
                if previous.image_kind != binding.image_kind {
                    return Err(TextureNumericManifestError::ImageKindConflict {
                        shader_id: binding.shader_id,
                        slot: binding.descriptor_slot,
                        first: previous.image_kind,
                        second: binding.image_kind,
                    });
                }
                continue;
            }
        }
        normalized.push(binding);
    }
    Ok(normalized)
}

pub fn texture_numeric_type_for_slot(
    manifest: &[TextureNumericBinding],
    descriptor_slot: usize,
) -> TextureNumericType {
    manifest
        .binary_search_by_key(&(descriptor_slot as u32), |binding| binding.descriptor_slot)
        .ok()
        .map(|index| manifest[index].spirv_type())
        .unwrap_or(TextureNumericType::Float)
}

pub fn texture_image_kind_for_slot(
    manifest: &[TextureNumericBinding],
    descriptor_slot: usize,
) -> GraphicsTextureImageKind {
    manifest
        .binary_search_by_key(&(descriptor_slot as u32), |binding| binding.descriptor_slot)
        .ok()
        .map(|index| manifest[index].image_kind)
        .unwrap_or(GraphicsTextureImageKind::D2)
}

pub fn texture_numeric_manifest_fingerprint(manifest: &[TextureNumericBinding]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&(manifest.len() as u64).to_le_bytes());
    for binding in manifest {
        eat(&binding.shader_id.to_le_bytes());
        eat(&binding.descriptor_slot.to_le_bytes());
        eat(&[binding.numeric_type as u8]);
        eat(&[binding.image_kind as u8]);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_normalization_is_deterministic_and_deduplicates_exact_entries() {
        let uint = TextureNumericBinding::new(0x28, 3, TextureNumericType::Uint);
        let float = TextureNumericBinding::new(0x10, 1, TextureNumericType::Float);
        let manifest = normalize_texture_numeric_manifest(vec![uint, float, uint]).unwrap();
        assert_eq!(manifest, vec![float, uint]);
        assert_eq!(
            texture_numeric_type_for_slot(&manifest, 1),
            TextureNumericType::Float
        );
        assert_eq!(
            texture_numeric_type_for_slot(&manifest, 3),
            TextureNumericType::Uint
        );
        assert_eq!(
            texture_numeric_type_for_slot(&manifest, 2),
            TextureNumericType::Float
        );
    }

    #[test]
    fn manifest_rejects_slot_and_numeric_conflicts() {
        let slot_conflict = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(1, 7, TextureNumericType::Float),
            TextureNumericBinding::new(2, 7, TextureNumericType::Float),
        ]);
        assert!(matches!(
            slot_conflict,
            Err(TextureNumericManifestError::SlotConflict { slot: 7, .. })
        ));

        let numeric_conflict = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(1, 7, TextureNumericType::Float),
            TextureNumericBinding::new(1, 7, TextureNumericType::Uint),
        ]);
        assert!(matches!(
            numeric_conflict,
            Err(TextureNumericManifestError::NumericConflict { slot: 7, .. })
        ));

        let image_kind_conflict = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(1, 7, TextureNumericType::Float)
                .with_image_kind(GraphicsTextureImageKind::Cube),
            TextureNumericBinding::new(1, 7, TextureNumericType::Float)
                .with_image_kind(GraphicsTextureImageKind::CubeArray),
        ]);
        assert!(matches!(
            image_kind_conflict,
            Err(TextureNumericManifestError::ImageKindConflict { slot: 7, .. })
        ));

        assert!(matches!(
            normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
                1,
                crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                TextureNumericType::Float,
            )]),
            Err(TextureNumericManifestError::InvalidSlot(32))
        ));
    }

    #[test]
    fn manifest_fingerprint_includes_slot_id_and_numeric_family() {
        let base = normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
            0x44,
            2,
            TextureNumericType::Float,
        )])
        .unwrap();
        let changed_id = normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
            0x45,
            2,
            TextureNumericType::Float,
        )])
        .unwrap();
        let changed_slot = normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
            0x44,
            3,
            TextureNumericType::Float,
        )])
        .unwrap();
        let changed_type = normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
            0x44,
            2,
            TextureNumericType::Uint,
        )])
        .unwrap();
        let changed_kind = normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(
            0x44,
            2,
            TextureNumericType::Float,
        )
        .with_image_kind(GraphicsTextureImageKind::CubeArray)])
        .unwrap();

        let fingerprint = texture_numeric_manifest_fingerprint(&base);
        assert_ne!(
            fingerprint,
            texture_numeric_manifest_fingerprint(&changed_id)
        );
        assert_ne!(
            fingerprint,
            texture_numeric_manifest_fingerprint(&changed_slot)
        );
        assert_ne!(
            fingerprint,
            texture_numeric_manifest_fingerprint(&changed_type)
        );
        assert_ne!(
            fingerprint,
            texture_numeric_manifest_fingerprint(&changed_kind)
        );
    }
}
