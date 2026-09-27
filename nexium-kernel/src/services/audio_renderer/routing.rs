const CHANNELS: usize = 24;
const MIX_SIZE: usize = 0x930;
const UNUSED_MIX_ID: usize = i32::MAX as usize;
const UNUSED_SPLITTER_ID: usize = u32::MAX as usize;
type StereoMap = [[f32; 2]; CHANNELS];

#[derive(Clone, Debug)]
struct Mix {
    volume: f32,
    count: usize,
    used: bool,
    destination: usize,
    splitter: usize,
    matrix: [[f32; CHANNELS]; CHANNELS],
}

#[derive(Clone, Debug)]
struct SplitterDestination {
    mix_id: usize,
    used: bool,
    volumes: [f32; CHANNELS],
}

#[derive(Clone, Debug, Default)]
pub struct AudioRouting {
    mixes: Vec<Option<Mix>>,
    channels: Vec<Option<[f32; CHANNELS]>>,
    splitters: Vec<Option<Vec<usize>>>,
    splitter_destinations: Vec<Option<SplitterDestination>>,
    outputs: Vec<StereoMap>,
}

fn uint(bytes: &[u8], offset: usize) -> u32 {
    bytes
        .get(offset..offset + 4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .unwrap_or(u32::MAX)
}

fn scalar(bytes: &[u8], offset: usize) -> f32 {
    let value = f32::from_bits(uint(bytes, offset));
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

impl AudioRouting {
    pub fn update_splitters(&mut self, data: &[u8], revision_num: u32) {
        const HEADER_MAGIC: u32 = 0x4844_4e53;
        const INFO_MAGIC: u32 = 0x4944_4e53;
        const DEST_MAGIC: u32 = 0x4444_4e53;
        if data.len() < 0x20 || uint(data, 0) != HEADER_MAGIC {
            return;
        }
        let info_count = uint(data, 4) as usize;
        let destination_count = uint(data, 8) as usize;
        if info_count > 256 || destination_count > 1024 {
            return;
        }
        let mut offset = 0x20usize;
        let mut updated_infos = Vec::with_capacity(info_count);
        for _ in 0..info_count {
            if data.get(offset..offset + 0x10).is_none() || uint(data, offset) != INFO_MAGIC {
                return;
            }
            let id = uint(data, offset + 4) as usize;
            let count = uint(data, offset + 0xC) as usize;
            let Some(end) = count.checked_mul(4).and_then(|size| offset.checked_add(0x1C + size)) else {
                return;
            };
            if id >= 256 || count > 1024 || end > data.len() {
                return;
            }
            let ids = (0..count)
                .map(|index| uint(data, offset + 0x10 + index * 4) as usize)
                .collect();
            updated_infos.push((id, ids));
            offset = end;
        }
        let (stride, used_offset) = if revision_num >= 15 {
            (0xA8, 0x9C)
        } else if revision_num >= 12 {
            (0x90, 0x84)
        } else {
            (0x70, 0x6C)
        };
        let mut updated_destinations = Vec::with_capacity(destination_count);
        for _ in 0..destination_count {
            if data.get(offset..offset + stride).is_none() || uint(data, offset) != DEST_MAGIC {
                return;
            }
            let id = uint(data, offset + 4) as usize;
            if id >= 1024 {
                return;
            }
            updated_destinations.push((id, SplitterDestination {
                mix_id: uint(data, offset + 0x68) as usize,
                used: data[offset + used_offset] != 0,
                volumes: std::array::from_fn(|index| scalar(data, offset + 8 + index * 4)),
            }));
            offset += stride;
        }
        for (id, ids) in updated_infos {
            self.splitters.resize_with(self.splitters.len().max(id + 1), || None);
            self.splitters[id] = Some(ids);
        }
        for (id, destination) in updated_destinations {
            self.splitter_destinations
                .resize_with(self.splitter_destinations.len().max(id + 1), || None);
            self.splitter_destinations[id] = Some(destination);
        }
    }

    pub fn update(&mut self, channels: &[u8], mixes: &[u8], sinks: &[u8], revision_num: u32) {
        let behavior = super::behavior::BehaviorInfo::from_user_revision(revision_num);
        self.channels.clear();
        self.channels.resize(channels.len() / 0x70, None);
        for channel in channels.chunks_exact(0x70) {
            let id = uint(channel, 0) as usize;
            if id >= self.channels.len() || channel[0x64] == 0 {
                continue;
            }
            self.channels[id] = Some(std::array::from_fn(|i| scalar(channel, 4 + i * 4)));
        }
        let records = if revision_num >= 7 {
            let count = uint(mixes, 4) as usize;
            mixes
                .get(0x20..)
                .and_then(|data| data.get(..count.checked_mul(MIX_SIZE)?))
                .unwrap_or(&[])
        } else {
            self.mixes.clear();
            mixes
        };
        for (index, bytes) in records.chunks_exact(MIX_SIZE).enumerate() {
            let id = if revision_num >= 7 {
                uint(bytes, 0x10) as usize
            } else {
                index
            };
            if id >= 256 {
                continue;
            }
            self.mixes
                .resize_with(self.mixes.len().max(id + 1), || None);
            self.mixes[id] = Some(Mix {
                volume: scalar(bytes, 0),
                count: (uint(bytes, 8) as usize).min(CHANNELS),
                used: bytes[12] != 0,
                destination: uint(bytes, 0x924) as usize,
                splitter: if behavior.is_splitter_supported() {
                    uint(bytes, 0x928) as usize
                } else {
                    UNUSED_SPLITTER_ID
                },
                matrix: std::array::from_fn(|i| {
                    std::array::from_fn(|j| scalar(bytes, 0x24 + (i * CHANNELS + j) * 4))
                }),
            });
        }
        let mut final_map = [[0.0; 2]; CHANNELS];
        for sink in sinks.chunks_exact(0x140) {
            if sink[0] != 1 || sink[1] == 0 {
                continue;
            }
            let count = uint(sink, 0x120) as usize;
            if !matches!(count, 1 | 2 | 6) {
                continue;
            }
            let coefficients = if sink[0x12b] != 0 {
                std::array::from_fn(|i| scalar(sink, 0x12c + i * 4))
            } else {
                [1.0, 0.596, 0.354, 0.707]
            };
            for channel in 0..count {
                let input = sink[0x124 + channel] as usize;
                if input >= CHANNELS {
                    continue;
                }
                let gain = match (count, channel) {
                    (1, _) => [1.0, 1.0],
                    (2, 0) => [1.0, 0.0],
                    (2, _) => [0.0, 1.0],
                    (_, 0) => [coefficients[0], 0.0],
                    (_, 1) => [0.0, coefficients[0]],
                    (_, 2) => [coefficients[1]; 2],
                    (_, 3) => [coefficients[2]; 2],
                    (_, 4) => [coefficients[3], 0.0],
                    _ => [0.0, coefficients[3]],
                };
                for side in 0..2 {
                    final_map[input][side] += gain[side];
                }
            }
        }
        let mut cache = vec![None; self.mixes.len()];
        let mut visiting = vec![false; self.mixes.len()];
        for id in 0..self.mixes.len() {
            Self::resolve(
                id,
                &self.mixes,
                &self.splitters,
                &self.splitter_destinations,
                &final_map,
                &mut cache,
                &mut visiting,
            );
        }
        self.outputs = cache
            .into_iter()
            .map(|map| map.unwrap_or([[0.0; 2]; CHANNELS]))
            .collect();
    }

    fn resolve(
        id: usize,
        mixes: &[Option<Mix>],
        splitters: &[Option<Vec<usize>>],
        destinations: &[Option<SplitterDestination>],
        final_map: &StereoMap,
        cache: &mut [Option<StereoMap>],
        visiting: &mut [bool],
    ) -> StereoMap {
        let zero = [[0.0; 2]; CHANNELS];
        let Some(Some(mix)) = mixes.get(id) else {
            return zero;
        };
        if !mix.used || visiting[id] {
            return zero;
        }
        if let Some(map) = cache[id] {
            return map;
        }
        visiting[id] = true;
        let mut map = zero;
        if id == 0 {
            for i in 0..mix.count {
                for side in 0..2 {
                    map[i][side] = final_map[i][side] * mix.volume;
                }
            }
        } else if mix.splitter != UNUSED_SPLITTER_ID {
            if let Some(Some(links)) = splitters.get(mix.splitter) {
                for (position, destination_id) in links.iter().copied().enumerate() {
                    let Some(Some(destination)) = destinations.get(destination_id) else {
                        continue;
                    };
                    if !destination.used || destination.mix_id == UNUSED_MIX_ID {
                        continue;
                    }
                    let next = Self::resolve(
                        destination.mix_id, mixes, splitters, destinations,
                        final_map, cache, visiting,
                    );
                    if mix.count == 0 {
                        continue;
                    }
                    let source = position % mix.count;
                    if source >= CHANNELS {
                        continue;
                    }
                    for output in 0..CHANNELS {
                        for side in 0..2 {
                            map[source][side] +=
                                mix.volume * destination.volumes[output] * next[output][side];
                        }
                    }
                }
            }
        } else {
            let next = Self::resolve(
                mix.destination, mixes, splitters, destinations,
                final_map, cache, visiting,
            );
            for i in 0..mix.count {
                for j in 0..CHANNELS {
                    for side in 0..2 {
                        map[i][side] += mix.volume * mix.matrix[i][j] * next[j][side];
                    }
                }
            }
        }
        visiting[id] = false;
        cache[id] = Some(map);
        map
    }

    pub fn voice_gains(&self, voice: &[u8], channel_count: usize) -> [[f32; 2]; 2] {
        let mut gains = [[0.0; 2]; 2];
        let mix_id = uint(voice, 0x58) as usize;
        if mix_id == UNUSED_MIX_ID {
            let splitter_id = uint(voice, 0x5C) as usize;
            let Some(Some(links)) = self.splitters.get(splitter_id) else {
                return gains;
            };
            for channel in 0..channel_count.min(2) {
                for destination_id in links.iter().copied().skip(channel).step_by(channel_count) {
                    let Some(Some(destination)) = self.splitter_destinations.get(destination_id) else {
                        continue;
                    };
                    if !destination.used {
                        continue;
                    }
                    let Some(map) = self.outputs.get(destination.mix_id) else {
                        continue;
                    };
                    for output in 0..CHANNELS {
                        for side in 0..2 {
                            gains[channel][side] += destination.volumes[output] * map[output][side];
                        }
                    }
                }
            }
            return gains;
        }
        let Some(map) = self.outputs.get(mix_id) else {
            return gains;
        };
        for channel in 0..channel_count.min(2) {
            let id = uint(voice, 0x140 + channel * 4) as usize;
            let Some(Some(volumes)) = self.channels.get(id) else {
                continue;
            };
            for i in 0..CHANNELS {
                for side in 0..2 {
                    gains[channel][side] += volumes[i] * map[i][side];
                }
            }
        }
        gains
    }

    pub fn describe_voice_path(&self, voice: &[u8], channel_count: usize) -> String {
        let mix_id = uint(voice, 0x58) as usize;
        let splitter_id = uint(voice, 0x5C) as usize;
        let mix_state = |id: usize| -> Option<(bool, f32, usize, usize)> {
            self.mixes
                .get(id)
                .and_then(Option::as_ref)
                .map(|mix| (mix.used, mix.volume, mix.count, mix.destination))
        };
        let output_entries = self
            .outputs
            .get(mix_id)
            .map(|map| {
                map.iter()
                    .filter(|pair| pair[0].abs() > 0.001 || pair[1].abs() > 0.001)
                    .count()
            })
            .unwrap_or(0);
        let channels: Vec<_> = (0..channel_count.min(2))
            .map(|channel| {
                let id = uint(voice, 0x140 + channel * 4) as usize;
                let state = self.channels.get(id).and_then(Option::as_ref);
                let nonzero = state
                    .map(|volumes| volumes.iter().filter(|v| v.abs() > 0.001).count())
                    .unwrap_or(0);
                let leading: Vec<_> = state
                    .into_iter()
                    .flat_map(|volumes| volumes.iter().copied().take(8))
                    .collect();
                (id, state.is_some(), nonzero, leading)
            })
            .collect();
        let linked_destinations: Vec<_> = self.splitters
            .get(splitter_id)
            .and_then(Option::as_ref)
            .into_iter()
            .flat_map(|ids| ids.iter().copied().take(12))
            .map(|id| {
                let destination = self.splitter_destinations.get(id).and_then(Option::as_ref);
                (id, destination.map(|d| (d.used, d.mix_id, d.volumes[..6].to_vec())))
            })
            .collect();
        format!(
            "mix_id={mix_id} mix={:?} mix0={:?} mixes={} output_entries={output_entries} splitter_id={splitter_id} links={linked_destinations:?} channels={channels:?}",
            mix_state(mix_id),
            mix_state(0),
            self.mixes.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn float(bytes: &mut [u8], offset: usize, value: f32) {
        put(bytes, offset, value.to_bits());
    }

    fn setup() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut channels = vec![0; 0x70 * 2];
        for i in 0..2 {
            put(&mut channels, i * 0x70, i as u32);
            channels[i * 0x70 + 0x64] = 1;
        }
        let mut mixes = vec![0; MIX_SIZE * 3];
        for i in 0..3 {
            let off = i * MIX_SIZE;
            float(&mut mixes, off, 1.0);
            put(&mut mixes, off + 8, 6);
            mixes[off + 12] = 1;
            put(&mut mixes, off + 0x10, i as u32);
            put(&mut mixes, off + 0x924, i.saturating_sub(1) as u32);
            put(&mut mixes, off + 0x928, u32::MAX);
            for j in 0..6 {
                float(&mut mixes, off + 0x24 + (j * 24 + j) * 4, 1.0);
            }
        }
        let mut sink = vec![0; 0x140];
        sink[0] = 1;
        sink[1] = 1;
        put(&mut sink, 0x120, 2);
        sink[0x125] = 1;
        let mut voice = vec![0; 0x170];
        put(&mut voice, 0x58, 2);
        put(&mut voice, 0x144, 1);
        (channels, mixes, sink, voice)
    }

    #[test]
    fn legacy_mix_ignores_reserved_splitter_field() {
        let (mut channels, mut mixes, sink, voice) = setup();
        float(&mut channels, 4, 1.0);
        for i in 0..3 {
            put(&mut mixes, i * MIX_SIZE + 0x928, 0);
        }
        let mut routing = AudioRouting::default();
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1)[0], [1.0, 0.0]);
        routing.update(&channels, &mixes, &sink, 2);
        assert_eq!(routing.voice_gains(&voice, 1)[0], [0.0, 0.0]);
    }

    #[test]
    fn muted_ambience_has_no_output_and_panning_is_preserved() {
        let (mut channels, mixes, sink, voice) = setup();
        let mut routing = AudioRouting::default();
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.0; 2]; 2]);
        float(&mut channels, 4, 0.7);
        float(&mut channels, 0x70 + 8, 0.4);
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 2), [[0.7, 0.0], [0.0, 0.4]]);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.7, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn chained_mix_volumes_and_dirty_mutes_are_applied() {
        let (mut channels, mut mixes, sink, voice) = setup();
        float(&mut channels, 4, 1.0);
        float(&mut mixes, 0, 0.5);
        float(&mut mixes, MIX_SIZE, 0.25);
        let mut routing = AudioRouting::default();
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1)[0], [0.125, 0.0]);
        let mut dirty = vec![0; 0x20];
        put(&mut dirty, 4, 1);
        dirty.extend_from_slice(&mixes[MIX_SIZE..MIX_SIZE * 2]);
        float(&mut dirty, 0x20, 0.0);
        routing.update(&channels, &dirty, &sink, 7);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.0; 2]; 2]);
    }

    #[test]
    fn sink_selection_surround_and_disabled_sink() {
        let (mut channels, mixes, mut sink, voice) = setup();
        float(&mut channels, 4 + 4 * 4, 1.0);
        put(&mut sink, 0x120, 6);
        sink[0x124..0x12a].copy_from_slice(&[0, 1, 4, 5, 2, 3]);
        let mut routing = AudioRouting::default();
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1)[0], [0.596; 2]);
        sink[1] = 0;
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.0; 2]; 2]);
    }

    #[test]
    fn disconnected_and_cyclic_routes_stay_silent() {
        let (mut channels, mut mixes, sink, mut voice) = setup();
        float(&mut channels, 4, 1.0);
        put(&mut mixes, MIX_SIZE + 0x924, 2);
        let mut routing = AudioRouting::default();
        routing.update(&channels, &mixes, &sink, 1);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.0; 2]; 2]);
        put(&mut voice, 0x58, u32::MAX);
        assert_eq!(routing.voice_gains(&voice, 1), [[0.0; 2]; 2]);
        routing.update(&[], &[], &[], 1);
        assert_eq!(routing.voice_gains(&voice, 2), [[0.0; 2]; 2]);
    }
}
