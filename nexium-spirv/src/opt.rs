use std::collections::HashMap;

#[derive(Clone)]
struct Inst {
    opcode: u16,
    words: Vec<u32>,
}

impl Inst {
    #[allow(dead_code)]
    fn result_id(&self) -> Option<u32> {
        let (has_rtype, has_rid) = opcode_meta(self.opcode);
        if has_rid {
            let idx = if has_rtype { 2 } else { 1 };
            self.words.get(idx).copied()
        } else {
            None
        }
    }
}

fn opcode_meta(op: u16) -> (bool, bool) {
    match op {
        19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 32 | 33 => (false, true),
        41 | 42 | 43 | 44 | 46 | 48 | 49 | 50 => (true, true),
        1 => (true, true),
        _ => (false, false),
    }
}

fn parse(words: &[u32]) -> Option<(Vec<u32>, Vec<Inst>)> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return None;
    }
    let header = words[..5].to_vec();
    let mut insts = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let w0 = words[i];
        let wc = (w0 >> 16) as usize;
        let op = (w0 & 0xFFFF) as u16;
        if wc == 0 || i + wc > words.len() {
            return None;
        }
        insts.push(Inst {
            opcode: op,
            words: words[i..i + wc].to_vec(),
        });
        i += wc;
    }
    Some((header, insts))
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct CanonKey(Vec<u32>);

fn make_canon_key(inst: &Inst, remap: &HashMap<u32, u32>) -> Option<CanonKey> {
    let op = inst.opcode;
    let is_type = matches!(op, 19..=33);
    let is_const = matches!(op, 41..=50 | 1);
    if !is_type && !is_const {
        return None;
    }

    let mut key = vec![op as u32];

    let (has_rtype, has_rid) = opcode_meta(op);
    let skip_start = if has_rtype { 1 } else { 0 } + if has_rid { 1 } else { 0 };
    let operand_start = 1 + skip_start;

    if has_rtype && inst.words.len() > 1 {
        let raw_type = inst.words[1];
        key.push(*remap.get(&raw_type).unwrap_or(&raw_type));
    }

    for &w in &inst.words[operand_start..] {
        key.push(*remap.get(&w).unwrap_or(&w));
    }

    Some(CanonKey(key))
}

pub fn dedup_constants(words: Vec<u32>) -> Vec<u32> {
    let Some((header, mut insts)) = parse(&words) else {
        return words;
    };

    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut dead: std::collections::HashSet<u32> = std::collections::HashSet::new();

    let mut canon: HashMap<CanonKey, u32> = HashMap::new();

    for _round in 0..3 {
        for inst in &insts {
            let (has_rtype, has_rid) = opcode_meta(inst.opcode);
            if !has_rid {
                continue;
            }
            let raw_id = if has_rtype && inst.words.len() > 2 {
                inst.words[2]
            } else if !has_rtype && inst.words.len() > 1 {
                inst.words[1]
            } else {
                continue;
            };

            if dead.contains(&raw_id) {
                continue;
            }

            let Some(key) = make_canon_key(inst, &remap) else {
                continue;
            };

            match canon.get(&key) {
                Some(&survivor) if survivor != raw_id => {
                    remap.insert(raw_id, survivor);
                    dead.insert(raw_id);
                }
                Some(_) => {}
                None => {
                    canon.insert(key, raw_id);
                }
            }
        }
    }

    if remap.is_empty() {
        return words;
    }

    let resolve = |id: u32| -> u32 { *remap.get(&id).unwrap_or(&id) };

    let mut out_insts: Vec<Inst> = Vec::with_capacity(insts.len());
    for mut inst in insts.drain(..) {
        let (has_rtype, has_rid) = opcode_meta(inst.opcode);

        let result_word_idx = if has_rtype && has_rid {
            Some(2usize)
        } else if !has_rtype && has_rid {
            Some(1usize)
        } else {
            None
        };

        if let Some(ri) = result_word_idx {
            if ri < inst.words.len() {
                let rid = inst.words[ri];
                if dead.contains(&rid) {
                    continue;
                }
            }
        }

        let literal_start = match inst.opcode {
            43 | 50 => Some(3usize),
            41 | 42 | 48 | 49 => Some(2usize),
            44 => None,
            _ => None,
        };

        for j in 1..inst.words.len() {
            if let Some(ls) = literal_start {
                if j >= ls {
                    break;
                }
            }
            inst.words[j] = resolve(inst.words[j]);
        }

        let wc = inst.words.len() as u32;
        inst.words[0] = (wc << 16) | inst.opcode as u32;

        out_insts.push(inst);
    }

    let bound = header[3];
    let mut out = header.clone();
    out[3] = bound;
    for inst in &out_insts {
        out.extend_from_slice(&inst.words);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spv_header(bound: u32) -> Vec<u32> {
        vec![0x07230203, 0x00010000, 0, bound, 0]
    }

    fn op_type_float(result_id: u32) -> Vec<u32> {
        vec![(3 << 16) | 22, result_id, 32]
    }

    fn op_constant_f32(type_id: u32, result_id: u32, bits: u32) -> Vec<u32> {
        vec![(4 << 16) | 43, type_id, result_id, bits]
    }

    #[test]
    fn dedup_removes_identical_type() {
        let mut words = spv_header(4);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_type_float(2));
        words.extend_from_slice(&[(1 << 16) | 253]);

        let out = dedup_constants(words);
        let count = count_opcode(&out, 22);
        assert_eq!(count, 1);
    }

    #[test]
    fn dedup_removes_identical_constant() {
        let mut words = spv_header(5);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_constant_f32(1, 2, 0));
        words.extend_from_slice(&op_constant_f32(1, 3, 0));
        words.extend_from_slice(&[(1 << 16) | 253]);

        let out = dedup_constants(words);
        let count = count_opcode(&out, 43);
        assert_eq!(count, 1);
    }

    #[test]
    fn distinct_constants_are_kept() {
        let mut words = spv_header(5);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_constant_f32(1, 2, 0));
        words.extend_from_slice(&op_constant_f32(1, 3, 1065353216));
        words.extend_from_slice(&[(1 << 16) | 253]);

        let out = dedup_constants(words);
        let count = count_opcode(&out, 43);
        assert_eq!(count, 2);
    }

    #[test]
    fn passthrough_on_non_spirv() {
        let garbage = vec![0xDEAD_BEEFu32; 4];
        let out = dedup_constants(garbage.clone());
        assert_eq!(out, garbage);
    }

    fn count_opcode(words: &[u32], opcode: u16) -> usize {
        let mut i = 5;
        let mut count = 0;
        while i < words.len() {
            let w0 = words[i];
            let wc = (w0 >> 16) as usize;
            let op = (w0 & 0xFFFF) as u16;
            if wc == 0 {
                break;
            }
            if op == opcode {
                count += 1;
            }
            i += wc;
        }
        count
    }
}
