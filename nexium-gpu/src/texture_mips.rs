use ash::vk::{self, Handle};

use crate::rt_cache::{RtCache, RtKey};
use crate::texture::{block_linear_mip_layout, TicEntry, TicFormat};

#[derive(Clone, Copy, Debug)]
pub struct TextureRtMip {
    pub level: u32,
    pub layer: u32,
    pub width: u32,
    pub height: u32,
    pub key: RtKey,
    pub image: vk::Image,
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub stamp: u64,
    pub src_layer: u32,
}

impl TextureRtMip {
    pub fn copy_region(&self) -> vk::ImageCopy {
        vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: self.src_layer,
                layer_count: 1,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: self.level,
                base_array_layer: self.layer,
                layer_count: 1,
            },
            extent: vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            ..Default::default()
        }
    }
}

pub fn texture_rt_mip_layers(tic: &TicEntry) -> u32 {
    match tic.texture_type {
        5 => tic.depth,
        3 => 6,
        8 => tic.depth.max(1).saturating_mul(6),
        _ => 1,
    }
}

pub fn texture_rt_mip_candidate(tic: &TicEntry) -> bool {
    ((tic.texture_type == 1 && tic.depth == 1)
        || (tic.texture_type == 5 && tic.depth > 0)
        || tic.texture_type == 3
        || (tic.texture_type == 8 && tic.depth > 0))
        && tic.base_layer == 0
        && tic.view_base_mip() == 0
        && (tic.view_mip_levels() > 1 || matches!(tic.texture_type, 3 | 5 | 8))
        && tic.is_block_linear
        && !tic.is_sparse
        && tic.pitch_bytes == 0
        && tic.block_width_log2 == 0
        && tic.block_depth_log2 == 0
        && tic.tile_width_spacing == 0
        && tic.sample_count() == Some(1)
        && tic.format.block_extent() == (1, 1)
        && !matches!(
            tic.format,
            TicFormat::Unknown(_)
                | TicFormat::G24R8
                | TicFormat::Z24S8
                | TicFormat::X8Z24
                | TicFormat::S8Z24
                | TicFormat::Z16
                | TicFormat::Z32
        )
}

pub fn find_texture_rt_mips(
    rt_cache: &RtCache,
    tic: &TicEntry,
    base_key: RtKey,
    format: vk::Format,
) -> Vec<TextureRtMip> {
    if !texture_rt_mip_candidate(tic)
        || !rt_cache.has_color_for_nvmap(base_key.nvmap_id)
        || base_key.gpu_va != tic.gpu_va
        || base_key.width != tic.width
        || base_key.height != tic.height
        || base_key.cpu_addr == 0
        || base_key.mapping_epoch == 0
        || base_key.is_3d
        || base_key.depth != 1
        || base_key.base_layer != 0
        || base_key.sample_grid() != (1, 1)
        || crate::renderer::exact_rt_copy_format_bpp(format) != Some(tic.format.src_bpp())
    {
        return Vec::new();
    }
    let Some(layout) = block_linear_mip_layout(tic) else {
        return Vec::new();
    };
    let layers = texture_rt_mip_layers(tic);
    if base_key.guest_size_bytes < layout.guest_size_bytes(layers) as u64 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for layer in 0..layers {
        for mip in layout.levels.iter().take(tic.view_mip_levels() as usize) {
            let offset = layer as u64 * layout.layer_stride as u64 + mip.guest_offset as u64;
            let Some(gpu_va) = tic.gpu_va.checked_add(offset) else {
                continue;
            };
            let Some(cpu_addr) = base_key.cpu_addr.checked_add(offset) else {
                continue;
            };
            let padded_width =
                ((mip.width as usize * tic.format.src_bpp() + 63) & !63) / tic.format.src_bpp();
            let want = RtKey::with_cpu(base_key.nvmap_id, mip.width, mip.height, gpu_va, cpu_addr)
                .with_mapping_epoch(base_key.mapping_epoch)
                .with_guest_size_bytes(mip.guest_size as u64)
                .with_block_linear_layout(0, mip.block_height_log2, 0, 0);
            let padded = RtKey {
                width: padded_width as u32,
                ..want
            };
            let mut src_layer = 0;
            let mut source = rt_cache
                .find_drawn_color_for_exact_alias(want)
                .or_else(|| rt_cache.find_drawn_color_for_exact_alias(padded));
            if source.is_none() && layers > 1 {
                let level_va = tic.gpu_va.checked_add(mip.guest_offset as u64);
                let level_cpu = base_key.cpu_addr.checked_add(mip.guest_offset as u64);
                if let Some((level_va, level_cpu)) = level_va.zip(level_cpu) {
                    let layered = RtKey::with_cpu(
                        base_key.nvmap_id,
                        mip.width,
                        mip.height,
                        level_va,
                        level_cpu,
                    )
                    .with_mapping_epoch(base_key.mapping_epoch)
                    .with_guest_size_bytes(mip.guest_size as u64)
                    .with_block_linear_layout(0, mip.block_height_log2, 0, 0)
                    .with_array_layers(layers, layout.layer_stride as u64);
                    let layered_padded = RtKey {
                        width: padded_width as u32,
                        ..layered
                    };
                    source = rt_cache
                        .find_drawn_color_for_exact_alias(layered)
                        .or_else(|| rt_cache.find_drawn_color_for_exact_alias(layered_padded));
                    src_layer = layer;
                }
            }
            let Some((key, image, image_layout, source_format, stamp)) = source else {
                continue;
            };
            if !crate::renderer::rt_copy_formats_compatible(source_format, format)
                || stamp == 0
                || image == vk::Image::null()
                || image_layout == vk::ImageLayout::UNDEFINED
                || rt_cache.color_extent(key)
                    != Some(vk::Extent2D {
                        width: key.width,
                        height: key.height,
                    })
            {
                continue;
            }
            out.push(TextureRtMip {
                level: mip.level,
                layer,
                src_layer,
                width: mip.width,
                height: mip.height,
                key,
                image,
                layout: image_layout,
                format,
                stamp,
            });
        }
    }
    if layers > 1 {
        for mip in layout.levels.iter().take(tic.view_mip_levels() as usize) {
            let found = out.iter().filter(|entry| entry.level == mip.level).count() as u32;
            if found != 0 && found != layers {
                static PARTIAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if PARTIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    log::warn!(
                        "[rt-mips] partial layered level {} type={} level={} found={} layers={}",
                        base_key.label(),
                        tic.texture_type,
                        mip.level,
                        found,
                        layers
                    );
                }
            }
        }
    }
    out
}

pub(crate) fn resolved_rt_mip_memo_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_RESOLVED_RT_MIP_MEMO").is_some())
}

fn resolved_mip_profile(hit: bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_RESOLVED_MIP_PROFILE").is_some()) { return; }
    static HITS: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);
    let count = if hit { HITS.fetch_add(1, Ordering::Relaxed) + 1 + MISSES.load(Ordering::Relaxed) }
        else { MISSES.fetch_add(1, Ordering::Relaxed) + 1 + HITS.load(Ordering::Relaxed) };
    if count % 32768 == 0 {
        log::info!("[resolved-mip-reuse] hits={} misses={}", HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed));
    }
}

#[derive(Default)]
pub(crate) struct ResolvedTextureRtMipMemo {
    entries: nexium_common::fast_hash::FastMap<(TicEntry, RtKey, vk::Format), (RtKey, (u64, u64), Vec<TextureRtMip>)>,
    hits: u64,
    misses: u64,
}

impl ResolvedTextureRtMipMemo {
    pub(crate) fn find(
        &mut self,
        rt_cache: &RtCache,
        tic: &TicEntry,
        base_key: RtKey,
        format: vk::Format,
    ) -> Vec<TextureRtMip> {
        if !texture_rt_mip_candidate(tic) {
            return Vec::new();
        }
        if !cfg!(test) && !resolved_rt_mip_memo_enabled() {
            return find_texture_rt_mips(rt_cache, tic, base_key, format);
        }
        let generation = rt_cache.color_sampling_generation(base_key.nvmap_id);
        let key = (*tic, base_key, format);
        if let Some((stored, previous, mips)) = self.entries.get(&key) {
            if *previous == generation && stored.same_live_identity(base_key) {
                self.hits += 1;
                resolved_mip_profile(true);
                return mips.clone();
            }
        }
        self.misses += 1;
        resolved_mip_profile(false);
        let mips = find_texture_rt_mips(rt_cache, tic, base_key, format);
        if self.entries.len() >= 4096 {
            self.entries.clear();
        }
        self.entries.insert(key, (base_key, generation, mips.clone()));
        mips
    }
}

fn rt_mip_memo_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_RT_MIP_MEMO").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

#[derive(Clone)]
struct TextureRtMipMemoEntry {
    structure_generation: u64,
    found: Vec<(u32, u32, u32, u32, u32, RtKey)>,
}

#[derive(Default)]
pub struct TextureRtMipMemo {
    entries: std::collections::HashMap<(u64, u64, u32, u32, u32), TextureRtMipMemoEntry>,
    hits: u64,
    misses: u64,
}

const TEXTURE_RT_MIP_MEMO_CAPACITY: usize = 4096;

fn rt_mip_memo() -> &'static std::sync::Mutex<TextureRtMipMemo> {
    static MEMO: std::sync::OnceLock<std::sync::Mutex<TextureRtMipMemo>> =
        std::sync::OnceLock::new();
    MEMO.get_or_init(|| std::sync::Mutex::new(TextureRtMipMemo::default()))
}

fn resolve_memo_mip(
    rt_cache: &RtCache,
    format: vk::Format,
    entry: (u32, u32, u32, u32, u32, RtKey),
) -> Option<TextureRtMip> {
    let (level, layer, src_layer, width, height, key) = entry;
    let (image, layout, source_format, extent, stamp) = rt_cache.resolve_drawn_color_exact(key)?;
    if stamp == 0
        || image == vk::Image::null()
        || layout == vk::ImageLayout::UNDEFINED
        || extent.width != key.width
        || extent.height != key.height
        || !crate::renderer::rt_copy_formats_compatible(source_format, format)
    {
        return None;
    }
    Some(TextureRtMip {
        level,
        layer,
        src_layer,
        width,
        height,
        key,
        image,
        layout,
        format,
        stamp,
    })
}

pub fn rt_mip_memo_stats() -> (u64, u64, usize) {
    let mut memo = rt_mip_memo().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let stats = (memo.hits, memo.misses, memo.entries.len());
    memo.hits = 0;
    memo.misses = 0;
    stats
}

pub fn find_texture_rt_mips_memo(
    rt_cache: &RtCache,
    texture_identity: impl FnOnce() -> u64,
    tic: &TicEntry,
    base_key: RtKey,
    format: vk::Format,
) -> Vec<TextureRtMip> {
    if !rt_mip_memo_enabled() || !texture_rt_mip_candidate(tic) {
        return find_texture_rt_mips(rt_cache, tic, base_key, format);
    }
    let memo_key = (
        texture_identity(),
        base_key.cpu_addr ^ (base_key.mapping_epoch.rotate_left(32)),
        base_key.nvmap_id,
        format.as_raw() as u32,
        tic.view_mip_levels(),
    );
    let generation = rt_cache.structure_generation();
    let mut memo = rt_mip_memo().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(entry) = memo.entries.get(&memo_key) {
        if entry.structure_generation == generation {
            let resolved: Option<Vec<TextureRtMip>> = entry
                .found
                .iter()
                .map(|found| resolve_memo_mip(rt_cache, format, *found))
                .collect();
            if let Some(resolved) = resolved {
                memo.hits += 1;
                return resolved;
            }
        }
    }
    memo.misses += 1;
    let out = find_texture_rt_mips(rt_cache, tic, base_key, format);
    if memo.entries.len() >= TEXTURE_RT_MIP_MEMO_CAPACITY {
        memo.entries.clear();
    }
    memo.entries.insert(
        memo_key,
        TextureRtMipMemoEntry {
            structure_generation: generation,
            found: out
                .iter()
                .map(|mip| (mip.level, mip.layer, mip.src_layer, mip.width, mip.height, mip.key))
                .collect(),
        },
    );
    if (memo.hits + memo.misses) % 65536 == 0 && crate::renderer::record_stage_profile_enabled() {
        log::warn!(
            "[rt-mip-memo] hits={} misses={} entries={}",
            memo.hits,
            memo.misses,
            memo.entries.len()
        );
    }
    out
}

pub fn texture_rt_mip_hash(mut hash: u64, mips: &[TextureRtMip]) -> u64 {
    for value in std::iter::once(mips.len() as u64).chain(mips.iter().flat_map(|mip| {
        [
            u64::from(mip.level),
            u64::from(mip.layer),
            u64::from(mip.width),
            u64::from(mip.height),
            mip.image.as_raw(),
            mip.stamp,
            mip.format.as_raw() as u64,
            u64::from(mip.key.nvmap_id),
            mip.key.gpu_va,
            mip.key.cpu_addr,
            mip.key.mapping_epoch,
            mip.key.guest_size_bytes,
            mip.key.layout_signature,
            u64::from(mip.key.width),
            u64::from(mip.key.height),
        ]
    })) {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn height_texture() -> (TicEntry, RtKey) {
        let words = [
            0x1bu32,
            0xa4970000,
            5 | (3 << 21),
            (4 << 3) | (7 << 28),
            511 | (1 << 23),
            511 | (1 << 31),
            0,
            0x70,
        ];
        let raw = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let tic = TicEntry::parse(&raw).unwrap();
        let layout = block_linear_mip_layout(&tic).unwrap();
        let key = RtKey::with_cpu(642, 512, 512, tic.gpu_va, 0x1122330000)
            .with_mapping_epoch(26)
            .with_guest_size_bytes(layout.layer_size as u64)
            .with_block_linear_layout(0, 4, 0, 0);
        (tic, key)
    }

    fn mip_key(tic: &TicEntry, base: RtKey, level: usize) -> RtKey {
        let mip = block_linear_mip_layout(tic).unwrap().levels[level];
        RtKey::with_cpu(
            base.nvmap_id,
            mip.width,
            mip.height,
            base.gpu_va + mip.guest_offset as u64,
            base.cpu_addr + mip.guest_offset as u64,
        )
        .with_mapping_epoch(base.mapping_epoch)
        .with_guest_size_bytes(mip.guest_size as u64)
        .with_block_linear_layout(0, mip.block_height_log2, 0, 0)
    }

    fn insert_source(cache: &mut RtCache, key: RtKey, format: vk::Format, extent: vk::Extent2D) {
        assert!(cache.adopt_external_color(
            key,
            vk::Image::from_raw(key.gpu_va),
            vk::ImageView::null(),
            vk::DeviceMemory::null(),
            format,
            extent,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ));
    }

    fn assert_resolved_memo_matches(
        memo: &mut ResolvedTextureRtMipMemo,
        cache: &RtCache,
        tic: &TicEntry,
        base: RtKey,
        format: vk::Format,
    ) -> Vec<TextureRtMip> {
        let expected = find_texture_rt_mips(cache, tic, base, format);
        let actual = memo.find(cache, tic, base, format);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!((actual.level, actual.layer, actual.src_layer), (expected.level, expected.layer, expected.src_layer));
            assert_eq!((actual.width, actual.height), (expected.width, expected.height));
            assert!(actual.key.same_live_identity(expected.key));
            assert_eq!((actual.image, actual.layout, actual.format, actual.stamp), (expected.image, expected.layout, expected.format, expected.stamp));
        }
        actual
    }

    #[test]
    fn resolved_memo_reuses_across_unrelated_allocations_but_tracks_new_source_mips() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        let mut memo = ResolvedTextureRtMipMemo::default();
        let format = vk::Format::R16_SFLOAT;
        let first = mip_key(&tic, base, 0);
        insert_source(&mut cache, first, format, vk::Extent2D { width: first.width, height: first.height });
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 1);
        let misses = memo.misses;
        let unrelated = RtKey { nvmap_id: first.nvmap_id + 1, gpu_va: first.gpu_va + 0x10000000, cpu_addr: first.cpu_addr + 0x10000000, ..first };
        insert_source(&mut cache, unrelated, format, vk::Extent2D { width: first.width, height: first.height });
        cache.mark_drawn(unrelated);
        cache.mark_cleared(unrelated, true);
        cache.mark_guest_written_range(unrelated.cpu_addr, 16, &[]);
        assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format);
        assert_eq!(memo.misses, misses);
        assert_eq!(memo.hits, 1);
        let second = mip_key(&tic, base, 1);
        insert_source(&mut cache, second, format, vk::Extent2D { width: second.width, height: second.height });
        cache.mark_cleared(second, true);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 1);
        cache.mark_drawn(second);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 2);
        assert!(memo.misses > misses);
    }

    #[test]
    fn resolved_memo_tracks_first_draw_clears_guest_writes_and_image_changes() {
        let (tic, base) = height_texture();
        let key = mip_key(&tic, base, 0);
        let mut cache = RtCache::new();
        let mut memo = ResolvedTextureRtMipMemo::default();
        let format = vk::Format::R16_SFLOAT;
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        insert_source(&mut cache, key, format, vk::Extent2D { width: key.width, height: key.height });
        let initial = assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format);
        assert_eq!(initial.len(), 1);
        assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format);
        cache.mark_cleared(key, true);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        cache.mark_drawn(key);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format)[0].stamp > initial[0].stamp);
        cache.mark_cleared(key, false);
        assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format);
        cache.mark_guest_written(key);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        cache.mark_synced_sample(key);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 1);
        cache.mark_guest_uploaded(key);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        cache.mark_synced_sample_from(key, 100);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format)[0].stamp, 100);
        cache.set_color_layout(key, vk::ImageLayout::UNDEFINED);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        cache.set_color_layout(key, vk::ImageLayout::GENERAL);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format)[0].layout, vk::ImageLayout::GENERAL);
        cache.get_existing(key).unwrap().image = vk::Image::from_raw(1234);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format)[0].image, vk::Image::from_raw(1234));
        cache.mark_guest_written_range(key.cpu_addr, 16, &[]);
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
        cache.mark_drawn(key);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 1);
        cache.mark_all_guest_written();
        assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).is_empty());
    }

    #[test]
    fn resolved_memo_distinguishes_mapping_layout_footprint_and_texture_view() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        let mut memo = ResolvedTextureRtMipMemo::default();
        let format = vk::Format::R16_SFLOAT;
        for level in 0..2 {
            let key = mip_key(&tic, base, level);
            insert_source(&mut cache, key, format, vk::Extent2D { width: key.width, height: key.height });
        }
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 2);
        for changed in [
            RtKey { mapping_epoch: base.mapping_epoch + 1, ..base },
            RtKey { cpu_addr: base.cpu_addr + 4096, ..base },
            RtKey { guest_size_bytes: 16, ..base },
            RtKey { base_layer: 1, ..base },
            RtKey { is_3d: true, ..base },
        ] {
            assert!(assert_resolved_memo_matches(&mut memo, &cache, &tic, changed, format).is_empty());
            assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 2);
        }
        let changed = TicEntry { res_max_mip_level: 0, ..tic };
        assert_resolved_memo_matches(&mut memo, &cache, &changed, base, format);
        assert_resolved_memo_matches(&mut memo, &cache, &tic, base, vk::Format::R8_UNORM);
        assert_eq!(assert_resolved_memo_matches(&mut memo, &cache, &tic, base, format).len(), 2);
    }

    #[test]
    fn gathers_each_live_mip_with_guest_offsets_and_padded_small_rows() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        for level in 0..8 {
            let mut key = mip_key(&tic, base, level);
            key.width = key.width.max(32);
            insert_source(
                &mut cache,
                key,
                vk::Format::R16_SFLOAT,
                vk::Extent2D {
                    width: key.width,
                    height: key.height,
                },
            );
        }
        let mips = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_eq!(mips.len(), 8);
        let expected = [
            0, 0x80000, 0xa0000, 0xa8000, 0xaa000, 0xaa800, 0xaac00, 0xaae00,
        ];
        for (level, mip) in mips.iter().enumerate() {
            assert_eq!(mip.level, level as u32);
            assert_eq!(mip.key.gpu_va, tic.gpu_va + expected[level]);
            assert_eq!(mip.key.cpu_addr, base.cpu_addr + expected[level]);
            assert_eq!(mip.width, 512 >> level);
            assert_eq!(mip.height, 512 >> level);
            let copy = mip.copy_region();
            assert_eq!(copy.src_subresource.mip_level, 0);
            assert_eq!(copy.dst_subresource.mip_level, level as u32);
            assert_eq!(copy.extent.width, 512 >> level);
            assert_eq!(copy.extent.height, 512 >> level);
            assert_eq!(copy.extent.depth, 1);
        }
        assert_eq!(mips[7].key.width, 32);
        assert_eq!(mips[7].copy_region().extent.width, 4);
    }

    #[test]
    fn rejects_wrong_mapping_layout_format_extent_and_multisample_sources() {
        let (tic, base) = height_texture();
        let key = mip_key(&tic, base, 0);
        let extent = vk::Extent2D {
            width: 512,
            height: 512,
        };
        for (candidate, format, image_extent) in [
            (
                RtKey {
                    mapping_epoch: 27,
                    ..key
                },
                vk::Format::R16_SFLOAT,
                extent,
            ),
            (
                RtKey {
                    cpu_addr: key.cpu_addr + 4096,
                    ..key
                },
                vk::Format::R16_SFLOAT,
                extent,
            ),
            (
                RtKey {
                    nvmap_id: 643,
                    ..key
                },
                vk::Format::R16_SFLOAT,
                extent,
            ),
            (
                RtKey {
                    layout_signature: 1,
                    ..key
                },
                vk::Format::R16_SFLOAT,
                extent,
            ),
            (key.with_sample_grid(2, 1), vk::Format::R16_SFLOAT, extent),
            (key, vk::Format::R8G8B8A8_UNORM, extent),
            (
                key,
                vk::Format::R16_SFLOAT,
                vk::Extent2D {
                    width: 1024,
                    height: 1024,
                },
            ),
            (
                RtKey { width: 1024, ..key },
                vk::Format::R16_SFLOAT,
                vk::Extent2D {
                    width: 1024,
                    height: 512,
                },
            ),
        ] {
            let mut cache = RtCache::new();
            insert_source(&mut cache, candidate, format, image_extent);
            assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT).is_empty());
        }
    }

    #[test]
    fn excludes_stale_or_undefined_sources_and_changes_hash_after_gpu_writes() {
        let (tic, base) = height_texture();
        let key = mip_key(&tic, base, 0);
        let mut cache = RtCache::new();
        insert_source(
            &mut cache,
            key,
            vk::Format::R16_SFLOAT,
            vk::Extent2D {
                width: 512,
                height: 512,
            },
        );
        let initial = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_eq!(initial.len(), 1);
        let initial_hash = texture_rt_mip_hash(123, &initial);
        cache.mark_drawn(key);
        let rewritten = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_ne!(initial_hash, texture_rt_mip_hash(123, &rewritten));
        cache.set_color_layout(key, vk::ImageLayout::UNDEFINED);
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT).is_empty());
        cache.set_color_layout(key, vk::ImageLayout::GENERAL);
        cache.mark_guest_written_range(key.cpu_addr, 2, &[]);
        let stale = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert!(stale.is_empty());
        assert_ne!(initial_hash, texture_rt_mip_hash(123, &stale));
    }

    #[test]
    fn content_hash_tracks_image_mapping_and_mip_identity() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        let key = mip_key(&tic, base, 0);
        insert_source(
            &mut cache,
            key,
            vk::Format::R16_SFLOAT,
            vk::Extent2D {
                width: 512,
                height: 512,
            },
        );
        let mip = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT)[0];
        let original = texture_rt_mip_hash(123, &[mip]);
        for changed in [
            TextureRtMip {
                image: vk::Image::from_raw(999),
                ..mip
            },
            TextureRtMip { level: 1, ..mip },
            TextureRtMip {
                key: RtKey {
                    mapping_epoch: 27,
                    ..mip.key
                },
                ..mip
            },
            TextureRtMip {
                key: RtKey {
                    cpu_addr: 0x55667700,
                    ..mip.key
                },
                ..mip
            },
        ] {
            assert_ne!(original, texture_rt_mip_hash(123, &[changed]));
        }
    }

    #[test]
    fn skips_absent_allocations_but_preserves_higher_mip_only_sources() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        assert!(!cache.has_color_for_nvmap(base.nvmap_id));
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT).is_empty());
        let source = mip_key(&tic, base, 1);
        let extent = vk::Extent2D {
            width: source.width,
            height: source.height,
        };
        insert_source(
            &mut cache,
            RtKey {
                nvmap_id: base.nvmap_id + 1,
                ..source
            },
            vk::Format::R16_SFLOAT,
            extent,
        );
        assert!(!cache.has_color_for_nvmap(base.nvmap_id));
        assert!(cache.has_color_for_nvmap(base.nvmap_id + 1));
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT).is_empty());
        insert_source(&mut cache, source, vk::Format::R16_SFLOAT, extent);
        assert!(cache.has_color_for_nvmap(base.nvmap_id));
        let mips = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_eq!(mips.len(), 1);
        assert_eq!(mips[0].level, 1);
        assert_eq!(mips[0].key.nvmap_id, base.nvmap_id);
        assert_eq!(mips[0].key.gpu_va, tic.gpu_va + 0x80000);
    }

    #[test]
    fn gathers_array_layers_using_full_mip_chain_stride() {
        let (mut tic, mut base) = height_texture();
        tic.texture_type = 5;
        tic.depth = 3;
        let layout = block_linear_mip_layout(&tic).unwrap();
        base.guest_size_bytes = layout.guest_size_bytes(tic.depth) as u64;
        let mut cache = RtCache::new();
        for layer in 0..tic.depth {
            for level in [0, 2] {
                let mut key = mip_key(&tic, base, level);
                let offset = layer as u64 * layout.layer_stride as u64;
                key.gpu_va += offset;
                key.cpu_addr += offset;
                insert_source(&mut cache, key, vk::Format::R16_SFLOAT,
                    vk::Extent2D { width: key.width, height: key.height });
            }
        }
        let mips = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_eq!(mips.len(), 6);
        for (index, mip) in mips.iter().enumerate() {
            let layer = index as u32 / 2;
            let level = if index % 2 == 0 { 0 } else { 2 };
            assert_eq!(mip.layer, layer);
            assert_eq!(mip.level, level as u32);
            assert_eq!(mip.key.gpu_va, tic.gpu_va + layer as u64 * layout.layer_stride as u64
                + layout.levels[level].guest_offset as u64);
            assert_eq!(mip.copy_region().dst_subresource.base_array_layer, layer);
            assert_eq!(mip.copy_region().src_subresource.base_array_layer, 0);
        }
        let changed_layer = TextureRtMip { layer: 1, ..mips[0] };
        assert_ne!(texture_rt_mip_hash(0, &[mips[0]]), texture_rt_mip_hash(0, &[changed_layer]));
        base.guest_size_bytes = layout.layer_size as u64;
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT).is_empty());
    }

    #[test]
    fn gathers_sources_whose_format_only_differs_in_texel_encoding() {
        let (tic, base) = height_texture();
        let mut cache = RtCache::new();
        for level in [0, 2] {
            let key = mip_key(&tic, base, level);
            insert_source(&mut cache, key, vk::Format::R16_UNORM,
                vk::Extent2D { width: key.width, height: key.height });
        }
        let mips = find_texture_rt_mips(&cache, &tic, base, vk::Format::R16_SFLOAT);
        assert_eq!(mips.iter().map(|mip| mip.level).collect::<Vec<_>>(), vec![0, 2]);
        assert!(mips.iter().all(|mip| mip.format == vk::Format::R16_SFLOAT));
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R8G8B8A8_UNORM).is_empty());
    }

    #[test]
    fn only_accepts_native_uncompressed_two_dimensional_mip_views() {
        let (tic, base) = height_texture();
        assert!(texture_rt_mip_candidate(&tic));
        for rejected in [
            TicEntry {
                texture_type: 2,
                ..tic
            },
            TicEntry { depth: 2, ..tic },
            TicEntry {
                base_layer: 1,
                ..tic
            },
            TicEntry {
                res_min_mip_level: 1,
                ..tic
            },
            TicEntry {
                res_max_mip_level: 0,
                ..tic
            },
            TicEntry {
                msaa_mode: 2,
                ..tic
            },
            TicEntry {
                is_sparse: true,
                ..tic
            },
            TicEntry {
                is_block_linear: false,
                ..tic
            },
            TicEntry {
                block_depth_log2: 1,
                ..tic
            },
            TicEntry {
                tile_width_spacing: 1,
                ..tic
            },
            TicEntry {
                format: TicFormat::BC1,
                ..tic
            },
            TicEntry {
                format: TicFormat::Z32,
                ..tic
            },
        ] {
            assert!(!texture_rt_mip_candidate(&rejected));
        }
        let mut cache = RtCache::new();
        let key = mip_key(&tic, base, 0);
        insert_source(
            &mut cache,
            key,
            vk::Format::R16_SFLOAT,
            vk::Extent2D {
                width: 512,
                height: 512,
            },
        );
        assert!(find_texture_rt_mips(
            &cache,
            &tic,
            RtKey {
                guest_size_bytes: 1,
                ..base
            },
            vk::Format::R16_SFLOAT
        )
        .is_empty());
        assert!(find_texture_rt_mips(&cache, &tic, base, vk::Format::R32_SFLOAT).is_empty());
    }
}
