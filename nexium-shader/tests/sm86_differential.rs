use std::collections::HashMap;

use nexium_shader::ir::FMods;
use nexium_shader::{
    BoolOp, BranchKind, Cfg, FComp, HalfMerge, HalfSwizzle, ICmp, IrOp, IrValue, LogicOp, MufuFunc,
    Predicate, ShaderStage, VoteMode,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, limit: u64) -> u64 {
        self.next() % limit
    }
}

fn set(word: &mut u128, start: u32, len: u32, value: u64) {
    let mask = ((1u128 << len) - 1) << start;
    *word = (*word & !mask) | ((u128::from(value) << start) & mask);
}

fn program(instruction: u128) -> Vec<u8> {
    let mut exit = 0u128;
    set(&mut exit, 0, 12, 0x94d);
    set(&mut exit, 12, 3, 7);
    set(&mut exit, 87, 3, 7);
    let mut end = 0u128;
    set(&mut end, 0, 12, 0x947);
    set(&mut end, 12, 3, 7);
    set(&mut end, 87, 3, 7);
    set(&mut end, 34, 48, (-4i64) as u64);
    let mut bytes = vec![0u8; 0x80];
    for word in [instruction, exit, end] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn hash(parts: &[u64]) -> u32 {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        state ^= part;
        state = state.wrapping_mul(0x0100_0000_01b3);
        state ^= state >> 29;
    }
    (state ^ (state >> 32)) as u32
}

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn mods(value: u32, neg: bool, abs: bool) -> f32 {
    let mut x = f(value);
    if abs {
        x = x.abs();
    }
    if neg {
        x = -x;
    }
    x
}

fn saturate(x: f32, sat: bool) -> f32 {
    if !sat {
        x
    } else if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

fn fcmp(cmp: FComp, a: f32, b: f32) -> bool {
    let unordered = a.is_nan() || b.is_nan();
    match cmp {
        FComp::F => false,
        FComp::Lt => a < b,
        FComp::Eq => a == b,
        FComp::Le => a <= b,
        FComp::Gt => a > b,
        FComp::Ne => !unordered && a != b,
        FComp::Ge => a >= b,
        FComp::Num => !unordered,
        FComp::Nan => unordered,
        FComp::Ltu => unordered || a < b,
        FComp::Equ => unordered || a == b,
        FComp::Leu => unordered || a <= b,
        FComp::Gtu => unordered || a > b,
        FComp::Neu => unordered || a != b,
        FComp::Geu => unordered || a >= b,
        FComp::T => true,
    }
}

fn icmp(cmp: ICmp, signed: bool, a: u32, b: u32) -> bool {
    let ordering = if signed { (a as i32).cmp(&(b as i32)) } else { a.cmp(&b) };
    match cmp {
        ICmp::F => false,
        ICmp::Lt => ordering.is_lt(),
        ICmp::Eq => ordering.is_eq(),
        ICmp::Le => ordering.is_le(),
        ICmp::Gt => ordering.is_gt(),
        ICmp::Ne => ordering.is_ne(),
        ICmp::Ge => ordering.is_ge(),
        ICmp::T => true,
    }
}

fn combine(bop: BoolOp, a: bool, b: bool) -> bool {
    match bop {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a ^ b,
    }
}

fn to_half(x: f32) -> u16 {
    half_from_f32(x)
}

fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00;
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - unbiased) as u32;
        let half = full >> shift;
        let rest = full & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if rest > halfway || (rest == halfway && half & 1 != 0) { half + 1 } else { half };
        return sign | rounded as u16;
    }
    let half = ((unbiased as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1FFF;
    let rounded = if rest > 0x1000 || (rest == 0x1000 && half & 1 != 0) { half + 1 } else { half };
    sign | rounded as u16
}

fn from_half(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exponent = u32::from((h >> 10) & 0x1F);
    let mantissa = u32::from(h & 0x3FF);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, m) => {
            let value = m as f32 / 1024.0 / 16384.0;
            return if sign != 0 { -value } else { value };
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, m) => sign | 0x7FC0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn halves(value: u32, swizzle: HalfSwizzle) -> [f32; 2] {
    let low = from_half(value as u16);
    let high = from_half((value >> 16) as u16);
    match swizzle {
        HalfSwizzle::H1H0 => [low, high],
        HalfSwizzle::F32 => [f(value), f(value)],
        HalfSwizzle::H0H0 => [low, low],
        HalfSwizzle::H1H1 => [high, high],
    }
}

struct State {
    regs: [u32; 256],
    preds: [bool; 32],
}

fn random_state(rng: &mut Rng) -> State {
    let mut regs = [0u32; 256];
    for reg in regs.iter_mut() {
        *reg = match rng.below(6) {
            0 => rng.next() as u32,
            1 => (rng.below(9) as f32 - 4.0).to_bits(),
            2 => (rng.below(2000) as f32 / 7.0 - 100.0).to_bits(),
            3 => rng.below(64) as u32,
            4 => (rng.next() as u32) & 0x3C00_3C00,
            _ => (f32::from_bits((rng.next() as u32 & 0x007F_FFFF) | 0x3F80_0000) - 1.5).to_bits(),
        };
    }
    let mut preds = [false; 32];
    for pred in preds.iter_mut() {
        *pred = rng.below(2) == 0;
    }
    preds[7] = true;
    regs[239] = (0..7).fold(0, |bits, index| bits | u32::from(preds[16 + index]) << index);
    State { regs, preds }
}

struct Outcome {
    regs: Vec<u32>,
    preds: Vec<bool>,
}

fn run(cfg: &Cfg, input: &State) -> Option<Outcome> {
    let mut values: HashMap<u32, u32> = HashMap::new();
    let mut preds = input.preds;
    let value = |values: &HashMap<u32, u32>, v: &IrValue| -> Option<u32> {
        Some(match v {
            IrValue::Inst(id) => *values.get(&id.0)?,
            IrValue::GprIn(r) => if *r == 255 { 0 } else { input.regs[*r as usize] },
            IrValue::ImmU32(bits) => *bits,
            IrValue::ImmF32(x) => x.to_bits(),
            IrValue::Zero => 0,
        })
    };
    let mut block = 0usize;
    let mut previous: Option<u32> = None;
    for _ in 0..64 {
        let current = &cfg.blocks[block];
        for inst in &current.program.instructions {
            let pred_value = |preds: &[bool; 32], p: Predicate| -> bool {
                let v = if p.idx == 7 { true } else { preds[p.idx as usize] };
                v ^ p.negate
            };
            let guard = inst.pred.map_or(true, |p| pred_value(&preds, p));
            let get = |v: &IrValue| value(&values, v);
            let src_pred = |preds: &[bool; 32], idx: u8, inv: bool| pred_value(preds, Predicate { idx, negate: inv });
            let mut set_pred = |preds: &mut [bool; 32], p: u8, v: bool| {
                if guard && p != 7 {
                    preds[p as usize] = v;
                }
            };
            let result: Option<u32> = match &inst.op {
                IrOp::Mov(a) => get(a),
                IrOp::FMul { a, b, mods: m } => {
                    let mut x = mods(get(a)?, m.neg_a, m.abs_a);
                    let y = mods(get(b)?, m.neg_b, m.abs_b);
                    x *= match m.scale {
                        1 => 0.5,
                        2 => 0.25,
                        3 => 0.125,
                        4 => 8.0,
                        5 => 4.0,
                        6 => 2.0,
                        _ => 1.0,
                    };
                    Some(saturate(x * y, m.sat).to_bits())
                }
                IrOp::FAdd { a, b, mods: m } => Some(
                    saturate(mods(get(a)?, m.neg_a, m.abs_a) + mods(get(b)?, m.neg_b, m.abs_b), m.sat).to_bits(),
                ),
                IrOp::FFma { a, b, c, mods: m } => {
                    let x = mods(get(a)?, m.neg_a, m.abs_a);
                    let y = mods(get(b)?, m.neg_b, m.abs_b);
                    let z = mods(get(c)?, m.neg_c, false);
                    Some(saturate(x.mul_add(y, z), m.sat).to_bits())
                }
                IrOp::FMin { a, b, mods: m } | IrOp::FMax { a, b, mods: m } => {
                    let x = mods(get(a)?, m.neg_a, m.abs_a);
                    let y = mods(get(b)?, m.neg_b, m.abs_b);
                    let r = if matches!(inst.op, IrOp::FMin { .. }) { x.min(y) } else { x.max(y) };
                    Some(r.to_bits())
                }
                IrOp::FMinMaxPred { a, b, mods: m, pred, neg_pred } => {
                    let x = mods(get(a)?, m.neg_a, m.abs_a);
                    let y = mods(get(b)?, m.neg_b, m.abs_b);
                    let min = src_pred(&preds, *pred, *neg_pred);
                    Some(if min { x.min(y) } else { x.max(y) }.to_bits())
                }
                IrOp::MultiFunc { src, func, mods: m } => {
                    let x = mods(get(src)?, m.neg_a, m.abs_a);
                    let r = match func {
                        MufuFunc::Cos => x.cos(),
                        MufuFunc::Sin => x.sin(),
                        MufuFunc::Ex2 => x.exp2(),
                        MufuFunc::Lg2 => x.log2(),
                        MufuFunc::Rcp => 1.0 / x,
                        MufuFunc::Rsq => 1.0 / x.sqrt(),
                        MufuFunc::Sqrt => x.sqrt(),
                        _ => return None,
                    };
                    Some(saturate(r, m.sat).to_bits())
                }
                IrOp::LoadCbuf { binding, byte_offset } => Some(hash(&[1, u64::from(*binding), u64::from(*byte_offset)])),
                IrOp::LoadCbufIndexed { binding, byte_offset, index, .. } => {
                    let index = get(index)?;
                    Some(hash(&[1, u64::from(*binding), u64::from(byte_offset.wrapping_add(index))]))
                }
                IrOp::F2F { src, neg, abs, sat, round } => {
                    let x = mods(get(src)?, *neg, *abs);
                    let r = match round {
                        1 => x.round_ties_even(),
                        2 => x.floor(),
                        3 => x.ceil(),
                        4 => x.trunc(),
                        _ => x,
                    };
                    Some(saturate(r, *sat).to_bits())
                }
                IrOp::I2F { src, signed, neg, abs, int_format, selector } => {
                    let bits = [8, 16, 32][usize::from(*int_format).min(2)];
                    let raw = get(src)? >> (u32::from(*selector) * 8);
                    let raw = if bits == 32 { raw } else { raw & ((1 << bits) - 1) };
                    let mut x = if *signed {
                        (((raw << (32 - bits)) as i32) >> (32 - bits)) as f32
                    } else {
                        raw as f32
                    };
                    if *abs {
                        x = x.abs();
                    }
                    if *neg {
                        x = -x;
                    }
                    Some(x.to_bits())
                }
                IrOp::F2I { src, signed, round } => {
                    let x = f(get(src)?);
                    let r = match round {
                        1 => x.floor(),
                        2 => x.ceil(),
                        3 => x.trunc(),
                        _ => x.round_ties_even(),
                    };
                    Some(if x.is_nan() { 0 } else if *signed { r as i32 as u32 } else { r as u32 })
                }
                IrOp::FSet { cmp, bop, src_a, src_b, neg_a, abs_a, neg_b, abs_b, bf, src_pred: sp, src_pred_inv } => {
                    let r = fcmp(*cmp, mods(get(src_a)?, *neg_a, *abs_a), mods(get(src_b)?, *neg_b, *abs_b));
                    let r = combine(*bop, r, src_pred(&preds, *sp, *src_pred_inv));
                    Some(if !r { 0 } else if *bf { 1.0f32.to_bits() } else { u32::MAX })
                }
                IrOp::FSetPred { cmp, bop, src_a, src_b, neg_a, abs_a, neg_b, abs_b, src_pred: sp, src_pred_inv, dest_p, dest_np } => {
                    let r = fcmp(*cmp, mods(get(src_a)?, *neg_a, *abs_a), mods(get(src_b)?, *neg_b, *abs_b));
                    let s = src_pred(&preds, *sp, *src_pred_inv);
                    let (p, np) = (combine(*bop, r, s), combine(*bop, !r, s));
                    set_pred(&mut preds, *dest_p, p);
                    set_pred(&mut preds, *dest_np, np);
                    None
                }
                IrOp::ISet { cmp, signed, a, b, bool_float } => {
                    let r = icmp(*cmp, *signed, get(a)?, get(b)?);
                    Some(if !r { 0 } else if *bool_float { 1.0f32.to_bits() } else { u32::MAX })
                }
                IrOp::ISetPred { cmp, signed, bop, src_a, src_b, src_pred: sp, src_pred_inv, dest_p, dest_np } => {
                    let r = icmp(*cmp, *signed, get(src_a)?, get(src_b)?);
                    let s = src_pred(&preds, *sp, *src_pred_inv);
                    let (p, np) = (combine(*bop, r, s), combine(*bop, !r, s));
                    set_pred(&mut preds, *dest_p, p);
                    set_pred(&mut preds, *dest_np, np);
                    None
                }
                IrOp::PSet { pred_a, neg_pred_a, pred_b, neg_pred_b, pred_c, neg_pred_c, bop_1, bop_2, bool_float } => {
                    let ab = combine(*bop_1, src_pred(&preds, *pred_a, *neg_pred_a), src_pred(&preds, *pred_b, *neg_pred_b));
                    let r = combine(*bop_2, ab, src_pred(&preds, *pred_c, *neg_pred_c));
                    Some(if !r { 0 } else if *bool_float { 1.0f32.to_bits() } else { u32::MAX })
                }
                IrOp::PSetPred { dest_p, dest_np, pred_a, neg_pred_a, pred_b, neg_pred_b, pred_c, neg_pred_c, bop_1, bop_2 } => {
                    let ab = combine(*bop_1, src_pred(&preds, *pred_a, *neg_pred_a), src_pred(&preds, *pred_b, *neg_pred_b));
                    let c = src_pred(&preds, *pred_c, *neg_pred_c);
                    let (p, np) = (combine(*bop_2, ab, c), combine(*bop_2, !ab, c));
                    set_pred(&mut preds, *dest_p, p);
                    set_pred(&mut preds, *dest_np, np);
                    None
                }
                IrOp::IAdd { a, b, neg_a, neg_b } => {
                    let x = if *neg_a { get(a)?.wrapping_neg() } else { get(a)? };
                    let y = if *neg_b { get(b)?.wrapping_neg() } else { get(b)? };
                    Some(x.wrapping_add(y))
                }
                IrOp::IMul { a, b } => Some(get(a)?.wrapping_mul(get(b)?)),
                IrOp::IMulHigh { a, b, signed_a, signed_b } => {
                    let x = if *signed_a { i64::from(get(a)? as i32) } else { i64::from(get(a)?) };
                    let y = if *signed_b { i64::from(get(b)? as i32) } else { i64::from(get(b)?) };
                    Some((x.wrapping_mul(y) >> 32) as u32)
                }
                IrOp::IMinMaxPred { a, b, signed, pred, neg_pred } => {
                    let (x, y) = (get(a)?, get(b)?);
                    let min = src_pred(&preds, *pred, *neg_pred);
                    let less = if *signed { (x as i32) < (y as i32) } else { x < y };
                    Some(if less == min { x } else { y })
                }
                IrOp::IScAdd { a, b, shift, neg_a, neg_b } => {
                    let x = if *neg_a { get(a)?.wrapping_neg() } else { get(a)? };
                    let y = if *neg_b { get(b)?.wrapping_neg() } else { get(b)? };
                    Some((x << shift).wrapping_add(y))
                }
                IrOp::ILop { a, b, op, not_a, not_b } => {
                    let x = if *not_a { !get(a)? } else { get(a)? };
                    let y = if *not_b { !get(b)? } else { get(b)? };
                    Some(match op {
                        LogicOp::And => x & y,
                        LogicOp::Or => x | y,
                        LogicOp::Xor => x ^ y,
                        LogicOp::PassB => y,
                    })
                }
                IrOp::ILop3 { a, b, c, lut } => {
                    let (x, y, z) = (get(a)?, get(b)?, get(c)?);
                    let mut r = 0u32;
                    for bit in 0..32 {
                        let index = ((x >> bit) & 1) << 2 | ((y >> bit) & 1) << 1 | ((z >> bit) & 1);
                        r |= u32::from((lut >> index) & 1) << bit;
                    }
                    Some(r)
                }
                IrOp::FindUMsb { value: v } => {
                    let x = get(v)?;
                    Some(if x == 0 { u32::MAX } else { 31 - x.leading_zeros() })
                }
                IrOp::BitCount { value: v } => Some(get(v)?.count_ones()),
                IrOp::BitReverse { value: v } => Some(get(v)?.reverse_bits()),
                IrOp::IShl { a, b } => Some(get(a)? << (get(b)? & 31)),
                IrOp::IShr { a, b, signed } => {
                    let s = get(b)? & 31;
                    let x = get(a)?;
                    Some(if *signed { ((x as i32) >> s) as u32 } else { x >> s })
                }
                IrOp::Bfe { a, b, signed } => {
                    let control = get(b)?;
                    let (pos, len) = (control & 0xFF, (control >> 8) & 0xFF);
                    let x = get(a)?;
                    if len == 0 {
                        Some(0)
                    } else {
                        let len = len.min(32);
                        let wide = u64::from(x) >> pos.min(32);
                        let field = (wide & ((1u64 << len) - 1)) as u32;
                        Some(if *signed && len < 32 && field >> (len - 1) & 1 != 0 { field | !((1u32 << len) - 1) } else { field })
                    }
                }
                IrOp::Bfi { base, insert, control } => {
                    let control = get(control)?;
                    let (pos, len) = (control & 0xFF, (control >> 8) & 0xFF);
                    let mask = if len >= 32 { u32::MAX } else { (1u32 << len) - 1 };
                    let mask = if pos >= 32 { 0 } else { mask << pos };
                    let shifted = if pos >= 32 { 0 } else { get(insert)? << pos };
                    Some((get(base)? & !mask) | (shifted & mask))
                }
                IrOp::SelectPred { pred, if_true, if_false } => {
                    if pred_value(&preds, *pred) { get(if_true) } else { get(if_false) }
                }
                IrOp::Phi { sources } => {
                    let from = previous?;
                    let (_, source) = sources.iter().find(|(block, _)| *block == from)?;
                    get(source)
                }
                IrOp::PackHalf2 { lo, hi } => Some(u32::from(to_half(f(get(lo)?))) | (u32::from(to_half(f(get(hi)?))) << 16)),
                IrOp::HAdd { a, b, old, merge, swizzle_a, swizzle_b, abs_a, neg_a, abs_b, neg_b, sat, .. } => {
                    let x = halves(get(a)?, *swizzle_a);
                    let y = halves(get(b)?, *swizzle_b);
                    let lane = |i: usize| {
                        let mut p = x[i];
                        let mut q = y[i];
                        if *abs_a { p = p.abs(); }
                        if *neg_a { p = -p; }
                        if *abs_b { q = q.abs(); }
                        if *neg_b { q = -q; }
                        saturate(p + q, *sat)
                    };
                    let (lo, hi) = (to_half(lane(0)), to_half(lane(1)));
                    let old = get(old)?;
                    Some(match merge {
                        HalfMerge::H1H0 => u32::from(lo) | (u32::from(hi) << 16),
                        HalfMerge::F32 => from_half(lo).to_bits(),
                        HalfMerge::MrgH0 => (old & 0xFFFF_0000) | u32::from(lo),
                        HalfMerge::MrgH1 => (old & 0xFFFF) | (u32::from(lo) << 16),
                    })
                }
                IrOp::SubgroupVote { source_pred, mode, pred_dest, .. } => {
                    let p = pred_value(&preds, *source_pred);
                    let r = match mode {
                        VoteMode::All | VoteMode::Any => p,
                        VoteMode::Equal => true,
                    };
                    set_pred(&mut preds, *pred_dest, r);
                    Some(u32::from(p))
                }
                IrOp::Exit => None,
                _ => return None,
            };
            if let (Some(id), Some(v)) = (inst.result, result) {
                values.insert(id.0, v);
            }
        }
        previous = Some(current.id);
        block = match current.branch {
            BranchKind::FallThrough => block + 1,
            BranchKind::Unconditional { target } => target as usize,
            BranchKind::Conditional { target, pred } => {
                let v = if pred.idx == 7 { true } else { preds[pred.idx as usize] } ^ pred.negate;
                if v { target as usize } else { block + 1 }
            }
            BranchKind::Exit => {
                let regs = (0..=255u8)
                    .map(|r| current.reg_exit.get(&r).map_or(Some(input.regs[r as usize]), |v| value(&values, v)))
                    .collect::<Option<Vec<_>>>()?;
                return Some(Outcome { regs, preds: preds.to_vec() });
            }
            BranchKind::Indirect { .. } => return None,
        };
    }
    None
}

fn same(a: u32, b: u32) -> bool {
    a == b || (f(a).is_nan() && f(b).is_nan()) || (a | b) == 0x8000_0000
}

#[test]
#[ignore = "set SM86_DUMP=<hi>_<lo> to print both IR forms of one instruction"]
fn dump_one_instruction() {
    let spec = std::env::var("SM86_DUMP").unwrap();
    let (hi, lo) = spec.split_once('_').unwrap();
    let word = (u128::from(u64::from_str_radix(hi, 16).unwrap()) << 64) | u128::from(u64::from_str_radix(lo, 16).unwrap());
    let bytes = program(word);
    let lowered = nexium_shader::ampere::translate(&bytes).unwrap();
    for line in nexium_shader::disassemble(&lowered.code) {
        println!("sass {}", line.to_string_compact());
    }
    let maxwell = nexium_shader::build_compute_cfg(&lowered.code);
    for block in &maxwell.blocks {
        println!("lowered block {} {:?}\n{}", block.id, block.branch, block.program);
    }
    let native = nexium_shader::cfg::sm86_native_probe(&bytes[0x80..], ShaderStage::Compute).unwrap();
    for block in &native.blocks {
        println!("native block {} {:?}\n{}", block.id, block.branch, block.program);
    }
}

#[test]
fn native_alu_matches_the_lowering() {
    let opcodes: &[u16] = &[
        0x002, 0x007, 0x008, 0x009, 0x00a, 0x00b, 0x00c, 0x010, 0x011, 0x012, 0x013, 0x016, 0x017, 0x019, 0x01a,
        0x01c, 0x020, 0x021, 0x023, 0x024, 0x045, 0x106, 0x105, 0x107, 0x108, 0x100, 0x101, 0x109, 0x003, 0x004,
        0x030, 0x034, 0x03e,
    ];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut compared = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    for &alu in opcodes {
        for form in [1u16, 2, 4, 5] {
            for _ in 0..1500 {
                let mut word = (u128::from(rng.next()) << 64) | u128::from(rng.next());
                set(&mut word, 0, 12, u64::from(alu | (form << 9)));
                let guard = if rng.below(4) == 0 { rng.below(3) } else { 7 };
                set(&mut word, 12, 3, guard);
                set(&mut word, 15, 1, 0);
                for start in [16u32, 24, 32, 64] {
                    set(&mut word, start, 8, rng.below(16));
                }
                if form == 5 {
                    set(&mut word, 54, 5, rng.below(4));
                }
                set(&mut word, 91, 1, 0);
                set(&mut word, 105, 23, 0);
                if guard != 7 && [81u32, 84].iter().any(|&start| (word >> start) as u64 & 7 == guard) {
                    continue;
                }
                let bytes = program(word);
                let Ok(lowered) = nexium_shader::ampere::translate(&bytes) else {
                    continue;
                };
                let maxwell = nexium_shader::build_compute_cfg(&lowered.code);
                if maxwell.unimplemented != 0 {
                    continue;
                }
                let Ok(native) = nexium_shader::cfg::sm86_native_probe(&bytes[0x80..], ShaderStage::Compute) else {
                    continue;
                };
                for _ in 0..4 {
                    let input = random_state(&mut rng);
                    let (Some(expected), Some(actual)) = (run(&maxwell, &input), run(&native, &input)) else {
                        continue;
                    };
                    compared += 1;
                    let regs = expected.regs[..32].iter().zip(&actual.regs[..32]).enumerate().find(|(_, (a, b))| !same(**a, **b));
                    let preds = expected.preds[..7] != actual.preds[..7];
                    if regs.is_some() || preds {
                        let detail = match regs {
                            Some((r, (a, b))) => format!("R{r}: lowered {a:#010x} native {b:#010x}"),
                            None => format!("preds lowered {:?} native {:?}", &expected.preds[..7], &actual.preds[..7]),
                        };
                        mismatches.push(format!(
                            "{:#05x} form {form}: {:016x}_{:016x} {detail}",
                            alu,
                            (word >> 64) as u64,
                            word as u64
                        ));
                        break;
                    }
                }
            }
        }
    }
    println!("compared {compared} evaluations, {} mismatching encodings", mismatches.len());
    let mut by_opcode: HashMap<String, usize> = HashMap::new();
    for mismatch in &mismatches {
        *by_opcode.entry(mismatch[..14].to_string()).or_default() += 1;
    }
    let mut summary: Vec<_> = by_opcode.into_iter().collect();
    summary.sort();
    for (key, count) in &summary {
        println!("{count:5} {key}");
    }
    let mut shown: HashMap<String, usize> = HashMap::new();
    for mismatch in &mismatches {
        let count = shown.entry(mismatch[..14].to_string()).or_default();
        if *count < 4 {
            println!("{mismatch}");
        }
        *count += 1;
    }
    assert!(compared > 10_000, "too few comparable encodings: {compared}");
    assert!(mismatches.is_empty(), "{} native/lowered mismatches", mismatches.len());
}

#[test]
fn native_uniform_alu_matches_the_lowering() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut compared = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    for alu in (0x080u16..0x0c0).filter(|&alu| alu != 0x086) {
        for form in [1u16, 2, 4] {
            for _ in 0..600 {
                let mut word = (u128::from(rng.next()) << 64) | u128::from(rng.next());
                set(&mut word, 0, 12, u64::from(alu | (form << 9)));
                let guard = if rng.below(4) == 0 { rng.below(3) } else { 7 };
                set(&mut word, 12, 3, guard);
                set(&mut word, 15, 1, 0);
                let fields: &[u32] = if form == 1 { &[16, 24, 32, 64] } else { &[16, 24, 64] };
                for &start in fields {
                    set(&mut word, start, 8, if rng.below(8) == 0 { 63 } else { rng.below(16) });
                }
                set(&mut word, 91, 1, 0);
                set(&mut word, 105, 23, 0);
                if guard != 7 && [81u32, 84].iter().any(|&start| (word >> start) as u64 & 7 == guard) {
                    continue;
                }
                if matches!(alu, 0x0b9 | 0x0bb) {
                    set(&mut word, 53, 1, 0);
                }
                let Some(highest) = fields.iter().map(|&start| (word >> start) as u8).filter(|&index| index < 63).max() else {
                    continue;
                };
                let lowered_base = (239 - (highest + 4).min(63)) & !3;
                let native_base = (254 - (highest + 3).min(62)) & !3;
                let bytes = program(word);
                let Ok(lowered) = nexium_shader::ampere::translate(&bytes) else {
                    continue;
                };
                let maxwell = nexium_shader::build_compute_cfg(&lowered.code);
                if maxwell.unimplemented != 0 {
                    continue;
                }
                let Ok(native) = nexium_shader::cfg::sm86_native_probe(&bytes[0x80..], ShaderStage::Compute) else {
                    continue;
                };
                let urs = highest + 2;
                for _ in 0..4 {
                    let mut lowered_input = random_state(&mut rng);
                    let mut native_input = State { regs: lowered_input.regs, preds: lowered_input.preds };
                    for index in 0..urs {
                        let value = native_input.regs[index as usize];
                        lowered_input.regs[(lowered_base + index) as usize] = value;
                        native_input.regs[(native_base + index) as usize] = value;
                    }
                    let (Some(expected), Some(actual)) = (run(&maxwell, &lowered_input), run(&native, &native_input)) else {
                        continue;
                    };
                    compared += 1;
                    let ur = (0..urs).find(|&index| {
                        !same(expected.regs[(lowered_base + index) as usize], actual.regs[(native_base + index) as usize])
                    });
                    let ups: Vec<bool> = (0..7).map(|index| expected.regs[239] >> index & 1 != 0).collect();
                    if ur.is_some() || ups != actual.preds[16..23] || expected.preds[..7] != actual.preds[..7] {
                        let detail = match ur {
                            Some(index) => format!(
                                "UR{index}: lowered {:#010x} native {:#010x}",
                                expected.regs[(lowered_base + index) as usize],
                                actual.regs[(native_base + index) as usize]
                            ),
                            None => format!("UP lowered {ups:?} native {:?}", &actual.preds[16..23]),
                        };
                        mismatches.push(format!("{alu:#05x} form {form}: {:016x}_{:016x} {detail}", (word >> 64) as u64, word as u64));
                        break;
                    }
                }
            }
        }
    }
    println!("compared {compared} evaluations, {} mismatching encodings", mismatches.len());
    let mut shown: HashMap<String, usize> = HashMap::new();
    for mismatch in &mismatches {
        let count = shown.entry(mismatch[..14].to_string()).or_default();
        if *count < 4 {
            println!("{mismatch}");
        }
        *count += 1;
    }
    let mut summary: Vec<_> = shown.into_iter().collect();
    summary.sort();
    for (key, count) in &summary {
        println!("{count:5} {key}");
    }
    assert!(compared > 5_000, "too few comparable encodings: {compared}");
    assert!(mismatches.is_empty(), "{} native/lowered mismatches", mismatches.len());
}
