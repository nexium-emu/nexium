use super::*;
use crate::decode::decode_one;
use crate::opcodes::Opcode;
use crate::operand;

const EXIT: (u64, u64) = (0x0000_0000_0000_794d, 0x000f_ea00_0380_0000);
const END: (u64, u64) = (0xffff_fff0_0000_7947, 0x000f_c000_0383_ffff);
const VERTEX: u32 = 0x020c_0481;
const FRAGMENT: u32 = 0x000c_5482;

fn binary(header: u32, instructions: &[(u64, u64)]) -> Vec<u8> {
    let mut bytes = vec![0u8; HEADER_SIZE];
    bytes[..4].copy_from_slice(&header.to_le_bytes());
    for (lo, hi) in instructions {
        bytes.extend_from_slice(&lo.to_le_bytes());
        bytes.extend_from_slice(&hi.to_le_bytes());
    }
    bytes
}

fn lowered(binary: &[u8]) -> Vec<u64> {
    let program = translate(binary).expect("translates");
    program
        .code
        .chunks_exact(8)
        .enumerate()
        .filter(|(index, _)| index % 4 != 0)
        .map(|(_, chunk)| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn opcodes(words: &[u64]) -> Vec<Opcode> {
    words
        .iter()
        .map(|word| decode_one(*word).expect("decodes").opcode)
        .collect()
}

#[test]
fn lowers_vertex_passthrough() {
    let binary = binary(
        VERTEX,
        &[
            (0x0000_80ff_ff00_7321, 0x000e_2200_0000_0400),
            (0x00c0_2400_0002_7a02, 0x000f_e200_0000_0f00),
            (0x4000_0000_0000_7820, 0x001f_e200_0041_0000),
            (0x4000_0000_0101_7820, 0x000f_ca00_0041_0000),
            (0x0000_7000_ff00_7322, 0x000f_e200_0000_08ff),
            EXIT,
            END,
        ],
    );
    let header = translate(&binary).unwrap().header;
    assert_eq!((u32::from_le_bytes(header[..4].try_into().unwrap()) >> 5) & 0x1F, 3);
    let words = lowered(&binary);
    assert_eq!(
        &opcodes(&words)[..7],
        &[
            Opcode::ALD,
            Opcode::MOV_cbuf,
            Opcode::FMUL_imm,
            Opcode::FMUL_imm,
            Opcode::AST,
            Opcode::EXIT,
            Opcode::BRA
        ]
    );
    assert_eq!((operand::attr_slot_ald(words[0]), operand::ald_num_elements(words[0])), (0x80, 2));
    let cbuf = operand::cbuf(words[1]);
    assert_eq!((operand::reg_dest(words[1]), cbuf.binding, cbuf.byte_offset), (2, 3, 0x90));
    assert_eq!((operand::reg_dest(words[3]), operand::reg_a(words[3])), (1, 1));
    assert_eq!(operand::float_imm20(words[3]), 2.0);
    assert_eq!(
        (operand::reg_dest(words[4]), operand::attr_slot_ald(words[4]), operand::ald_num_elements(words[4])),
        (0, 0x70, 3)
    );
    assert_eq!(words[6], 0xE240_0FFF_FF87_000F);
}

#[test]
fn lowers_textured_fragment() {
    let binary = binary(
        FRAGMENT,
        &[
            (0x0000_00ff_0006_7326, 0x040e_0200_000e_001f),
            (0x0000_00ff_0001_7326, 0x000e_6200_000e_0020),
            (0x0000_00ff_0005_7326, 0x000e_a200_000e_0021),
            (0x0000_0006_0006_7308, 0x001e_6400_0000_1000),
            (0x0000_0001_0604_7220, 0x042f_e200_0041_c000),
            (0x0000_0005_0605_7220, 0x004f_c800_0041_c000),
            (0x3080_0805_0400_7b60, 0x100f_c200_041e_0f02),
            EXIT,
            END,
        ],
    );
    let words = lowered(&binary);
    assert_eq!(
        &opcodes(&words)[..10],
        &[
            Opcode::IPA,
            Opcode::IPA,
            Opcode::IPA,
            Opcode::MUFU,
            Opcode::FMUL_reg,
            Opcode::FMUL_reg,
            Opcode::MOV_cbuf,
            Opcode::TEX_b,
            Opcode::EXIT,
            Opcode::BRA
        ]
    );
    assert_eq!((operand::attr_slot_ipa(words[0]), operand::ipa_interpolation_mode(words[0])), (0x7c, 0));
    assert_eq!((operand::reg_dest(words[2]), operand::attr_slot_ipa(words[2])), (5, 0x84));
    assert_eq!(operand::mufu_func_bits(words[3]), 4);
    assert_eq!((operand::reg_a(words[4]), operand::reg_b(words[4])), (6, 1));
    let handle = operand::cbuf(words[6]);
    assert_eq!((operand::reg_dest(words[6]), handle.binding, handle.byte_offset), (240, 2, 0x20));
    let tex = words[7];
    assert_eq!((operand::reg_dest(tex), operand::reg_a(tex), operand::reg_b(tex)), (0, 4, 240));
    assert_eq!(((tex >> 28) & 7, (tex >> 31) & 0xF, (tex >> 37) & 7), (2, 0xF, 0));
}

#[test]
fn surface_reductions_lower_to_a_discarded_surface_atomic() {
    let binary = binary(FRAGMENT, &[(0x6000_0c00_0400_759f, 0x000f_e200_0010_e000), EXIT, END]);
    let words = lowered(&binary);
    let ops = opcodes(&words);
    let at = ops.iter().position(|op| *op == Opcode::SUATOM).expect("lowers to SUATOM");
    assert_eq!(ops[at - 1], Opcode::MOV_cbuf);
    let atom = words[at];
    assert_eq!((atom & 0xff, (atom >> 8) & 0xff, (atom >> 20) & 0xff), (0xff, 4, 0));
    assert_eq!(((atom >> 29) & 0xf, (atom >> 33) & 7, (atom >> 54) & 1), (0, 3, 1));
    assert_eq!((atom >> 39) & 0xff, u64::from(operand::reg_dest(words[at - 1])));
}

#[test]
fn half_texture_results_pack_two_channels_per_register() {
    let binary = binary(
        FRAGMENT,
        &[
            (0x0000_00ff_0006_7326, 0x040e_0200_000e_001f),
            (0x0000_00ff_0001_7326, 0x000e_6200_000e_0020),
            (0x0000_00ff_0005_7326, 0x000e_a200_000e_0021),
            (0x0000_0006_0006_7308, 0x001e_6400_0000_1000),
            (0x0000_0001_0604_7220, 0x042f_e200_0041_c000),
            (0x0000_0005_0605_7220, 0x004f_c800_0041_c000),
            (0x3080_0805_0400_7b60, 0x100f_c200_041e_8f02),
            EXIT,
            END,
        ],
    );
    let words = lowered(&binary);
    assert_eq!(
        &opcodes(&words)[7..15],
        &[
            Opcode::TEX_b,
            Opcode::F2F_reg,
            Opcode::F2F_reg,
            Opcode::LEA_lo_reg,
            Opcode::F2F_reg,
            Opcode::F2F_reg,
            Opcode::LEA_lo_reg,
            Opcode::EXIT
        ]
    );
    let staging = operand::reg_dest(words[7]);
    assert_eq!((operand::reg_b(words[8]), operand::reg_b(words[9])), (staging + 1, staging));
    assert_eq!((operand::reg_dest(words[10]), operand::reg_dest(words[13])), (0, 2));
    assert_eq!((operand::reg_b(words[11]), operand::reg_b(words[12])), (staging + 3, staging + 2));
}

#[test]
fn splits_scalar_texture_sources() {
    let binary = binary(
        FRAGMENT,
        &[
            (0x5080_0a02_0600_7b60, 0x000f_c200_041e_0702),
            EXIT,
            END,
        ],
    );
    let words = lowered(&binary);
    assert_eq!(
        &opcodes(&words)[..6],
        &[
            Opcode::MOV_reg,
            Opcode::MOV_reg,
            Opcode::MOV_reg,
            Opcode::MOV_cbuf,
            Opcode::TEX_b,
            Opcode::EXIT
        ]
    );
    assert_eq!(
        words[..3].iter().map(|word| (operand::reg_dest(*word), operand::reg_b(*word))).collect::<Vec<_>>(),
        vec![(240, 6), (241, 7), (242, 2)]
    );
    let tex = words[4];
    assert_eq!((operand::reg_dest(tex), operand::reg_a(tex), operand::reg_b(tex)), (0, 240, 243));
    assert_eq!(((tex >> 28) & 7, (tex >> 31) & 0xF), (4, 7));
}

#[test]
fn branches_land_on_expanded_instructions() {
    let binary = binary(
        VERTEX,
        &[
            (0x947 | (8 << 34), 7 << 23),
            (0x423 | (7 << 12) | (1 << 16) | (2 << 24) | (0x3f80_0001u64 << 32), 3),
            (0x202 | (7 << 12) | (4 << 16) | (5 << 32), 0xF << 8),
            EXIT,
            END,
        ],
    );
    let code = translate(&binary).unwrap().code;
    let words = lowered(&binary);
    assert_eq!(
        &opcodes(&words)[..6],
        &[
            Opcode::BRA,
            Opcode::MOV32I,
            Opcode::FFMA_reg,
            Opcode::MOV_reg,
            Opcode::EXIT,
            Opcode::BRA
        ]
    );
    assert_eq!((operand::pred_idx(words[0]), operand::pred_negate(words[0])), (0, false));
    let offset = ((((words[0] >> 20) & 0xFF_FFFF) as i64) << 40) >> 40;
    let target = (8 + 8 + offset) as usize;
    let word = u64::from_le_bytes(code[target..target + 8].try_into().unwrap());
    assert_eq!(decode_one(word).unwrap().opcode, Opcode::EXIT);
    assert_eq!((operand::reg_b(words[2]), operand::reg_c(words[2])), (3, 240));
}

#[test]
fn swaps_logic_table_for_immediate_third_source() {
    let lo = 0x412 | (7 << 12) | (1 << 16) | (2 << 24) | (0xF0u64 << 32);
    let hi = 3 | (0xC0 << 8) | (7 << 17) | (7 << 23) | (1 << 26);
    let words = lowered(&binary(VERTEX, &[(lo, hi), EXIT, END]));
    assert_eq!(opcodes(&words)[0], Opcode::LOP3_imm);
    assert_eq!(((words[0] >> 48) & 0xFF, operand::reg_c(words[0]), operand::imm20(words[0])), (0xA0, 3, 0xF0));
}

#[test]
fn scales_sine_input_to_radians() {
    let words = lowered(&binary(
        FRAGMENT,
        &[(0x308 | (7 << 12) | (1 << 16) | (2 << 32), 1 << 10), EXIT, END],
    ));
    assert_eq!(&opcodes(&words)[..2], &[Opcode::FMUL32I, Opcode::MUFU]);
    assert_eq!((operand::reg_a(words[0]), operand::imm32(words[0])), (2, 0x40C9_0FDB));
    assert_eq!(
        (operand::reg_a(words[1]), operand::mufu_func_bits(words[1])),
        (operand::reg_dest(words[0]), 1)
    );
}

#[test]
fn negated_multiply_move_becomes_add() {
    let words = lowered(&binary(
        VERTEX,
        &[(0x0000_00ff_ff00_8224, 0x000f_e200_078e_0a02), EXIT, END],
    ));
    assert_eq!(opcodes(&words)[0], Opcode::IADD3_reg);
    assert_eq!(
        (operand::reg_dest(words[0]), operand::reg_b(words[0]), operand::reg_c(words[0])),
        (0, 0xFF, 2)
    );
    assert!(operand::iadd3_neg_c(words[0]));
    assert_eq!((operand::pred_idx(words[0]), operand::pred_negate(words[0])), (0, true));
}

#[test]
fn lowers_uniform_registers_into_reserved_gprs() {
    let words = lowered(&binary(
        VERTEX,
        &[
            (0x0001_9c00_0006_7ab9, 0x000f_c800_0000_0a00),
            (0x0000_0004_0003_7c20, 0x000f_e200_0841_0000),
            (0x0000_0004_0505_7290, 0x000f_c800_0fff_e03f),
            EXIT,
            END,
        ],
    ));
    assert_eq!(&opcodes(&words)[..3], &[Opcode::LDC, Opcode::FMUL_reg, Opcode::IADD3_reg]);
    assert_eq!((operand::reg_dest(words[0]), (words[0] >> 48) & 7, (words[0] >> 20) & 0xFFFF), (234, 5, 0x670));
    assert_eq!(operand::reg_dest(words[0]) % 2, 0, "64-bit constant loads need an even register pair");
    assert_eq!((operand::reg_dest(words[1]), operand::reg_a(words[1]), operand::reg_b(words[1])), (3, 0, 232));
    assert_eq!(
        (operand::reg_dest(words[2]), operand::reg_a(words[2]), operand::reg_b(words[2]), operand::reg_c(words[2])),
        (233, 233, 232, 0xFF)
    );
}

#[test]
fn uniform_predicates_guard_uniform_loads() {
    let words = lowered(&binary(
        VERTEX,
        &[
            (0x0001_9100_0005_7ab9, 0x000f_e400_0000_0800),
            (0x0000_003f_0400_728c, 0x000f_e400_0bfa_2270),
            (0x0384_3000_0005_5ab9, 0x000f_e200_0000_0800),
            EXIT,
            END,
        ],
    ));
    assert_eq!(
        &opcodes(&words)[..11],
        &[
            Opcode::LDC,
            Opcode::ISET_reg,
            Opcode::MOV32I,
            Opcode::LOP3_reg,
            Opcode::LOP3_imm,
            Opcode::P2R_imm,
            Opcode::SHL_imm,
            Opcode::R2P_imm,
            Opcode::LDC,
            Opcode::R2P_imm,
            Opcode::EXIT
        ]
    );
    assert_eq!((operand::reg_dest(words[4]), operand::imm20(words[4]), (words[4] >> 48) & 0xFF), (239, 1 << 5, 0xE2));
    assert_eq!((operand::reg_a(words[6]), operand::imm20(words[6])), (239, 1));
    assert_eq!((operand::pred_idx(words[8]), operand::pred_negate(words[8])), (6, false));
    assert_eq!((operand::pred_idx(words[9]), operand::imm20(words[9])), (7, 0x40));
}

#[test]
fn branch_with_second_predicate_skips_on_the_guard() {
    let (lo, hi) = sm70_branch_offset(0, 0x20);
    let words = lowered(&binary(
        VERTEX,
        &[
            (0x947 | (9 << 12) | lo, hi | (2 << 23)),
            (0x802 | (7 << 12) | (0x3f80_0000u64 << 32), 0xF << 8),
            EXIT,
            END,
        ],
    ));
    assert_eq!(&opcodes(&words)[..4], &[Opcode::BRA, Opcode::BRA, Opcode::MOV32I, Opcode::EXIT]);
    assert_eq!((operand::pred_idx(words[0]), operand::pred_negate(words[0])), (1, false));
    assert_eq!((operand::pred_idx(words[1]), operand::pred_negate(words[1])), (2, false));
}

#[test]
fn the_single_call_runs_its_body_before_the_caller_continues() {
    let (lo, hi) = sm70_branch_offset(0, 0x20);
    let binary = binary(
        FRAGMENT,
        &[
            (0x944 | (7 << 12) | lo, hi | 0x03c0_0000),
            EXIT,
            (0x802 | (7 << 12) | (0x3f80_0000u64 << 32), 0xF << 8),
            (0x950 | (7 << 12) | (4 << 24), 0x03c0_0000),
            END,
        ],
    );
    let codes = opcodes(&lowered(&binary));
    let body = codes.iter().position(|opcode| *opcode == Opcode::MOV32I).unwrap();
    let exit = codes.iter().position(|opcode| *opcode == Opcode::EXIT).unwrap();
    assert!(body < exit);
}

#[test]
fn calls_from_several_sites_expand_the_callee_inline() {
    let (first_lo, first_hi) = sm70_branch_offset(0, 0x40);
    let (second_lo, second_hi) = sm70_branch_offset(0x20, 0x40);
    let one = (0x802 | (7 << 12) | (0x3f80_0000u64 << 32), 0xF << 8);
    let binary = binary(
        FRAGMENT,
        &[
            (0x944 | (7 << 12) | first_lo, first_hi | 0x03c0_0000),
            (0x802 | (7 << 12) | (1 << 16) | (0x4000_0000u64 << 32), 0xF << 8),
            (0x944 | (7 << 12) | second_lo, second_hi | 0x03c0_0000),
            EXIT,
            one,
            (0x950 | (7 << 12) | (4 << 24), 0x03c0_0000),
            END,
        ],
    );
    let words = lowered(&binary);
    let ones = words
        .iter()
        .filter(|word| decode_one(**word).unwrap().opcode == Opcode::MOV32I && (*word >> 20) & 0xFFFF_FFFF == 0x3f80_0000)
        .count();
    assert_eq!(ones, 3);
    assert_eq!(opcodes(&words).iter().filter(|opcode| **opcode == Opcode::EXIT).count(), 2);
}

#[test]
fn power_of_two_multiplies_lower_to_shifts() {
    let scale = (0x824 | (7 << 12) | (6 << 16) | (6 << 24) | (4u64 << 32), 0x000f_c800_078e_00ff);
    let codes = opcodes(&lowered(&binary(VERTEX, &[scale, EXIT, END])));
    assert_eq!(codes[0], Opcode::SHL_imm);
    assert!(!codes.contains(&Opcode::XMAD_imm) && !codes.contains(&Opcode::IMUL32I));
}

#[test]
fn swizzle_adds_reverse_their_lane_table_for_maxwell() {
    let horizontal = (0x822 | (7 << 12) | (5 << 16) | (6 << 24) | (0x66u64 << 32), 0x000f_e200_0000_0007);
    let words = lowered(&binary(FRAGMENT, &[horizontal, EXIT, END]));
    let word = words[0];
    assert_eq!(decode_one(word).unwrap().opcode, Opcode::FSWZADD);
    assert_eq!((word & 0xff, (word >> 8) & 0xff, (word >> 20) & 0xff), (5, 6, 7));
    assert_eq!((word >> 28) & 0xff, 0x99);
}

#[test]
fn shifted_adds_with_a_negated_addend_subtract_it() {
    let scale = (0x211 | (7 << 12) | (4 << 16) | (10 << 24) | (10u64 << 32) | (1 << 63), 0x001f_c800_078e_40ff);
    let words = lowered(&binary(VERTEX, &[scale, EXIT, END]));
    let codes = opcodes(&words);
    assert_eq!(&codes[..2], &[Opcode::IADD_reg, Opcode::LEA_lo_reg]);
    assert_eq!((words[0] >> 48) & 1, 1);
    assert_eq!((words[1] >> 20) & 0xff, words[0] & 0xff);
    assert_eq!(((words[1] >> 39) & 0x1f, words[1] & 0xff, (words[1] >> 8) & 0xff), (8, 4, 10));
}

#[test]
fn half_compares_and_conversions_read_constant_buffer_halves() {
    let convert = (0x21c0_3c00_ff0a_7630, 0x000f_e200_0000_4100);
    let compare = (0x31c0_3f00_ff00_7634, 0x000f_ca00_04f2_4800);
    let words = lowered(&binary(FRAGMENT, &[convert, compare, EXIT, END]));
    assert_eq!(&opcodes(&words)[..4], &[Opcode::MOV_cbuf, Opcode::F2F_reg, Opcode::MOV_cbuf, Opcode::HSETP2_reg]);
    assert_eq!((operand::reg_dest(words[1]), operand::reg_b(words[1])), (10, operand::reg_dest(words[0])));
    let field = |at: u32, len: u32| (words[3] >> at) & ((1 << len) - 1);
    assert_eq!((field(3, 3), field(0, 3), field(35, 4), field(39, 3), field(42, 1)), (1, 7, 4, 1, 1));
    assert_eq!((field(28, 2), field(47, 2), field(8, 8), field(20, 8)), (3, 2, 0xff, operand::reg_dest(words[2]) as u64));
}

#[test]
fn a_subroutine_called_once_expands_inline() {
    let (call_lo, call_hi) = sm70_branch_offset(0, 0x30);
    let one = (0x802 | (7 << 12) | (0x3f80_0000u64 << 32), 0xF << 8);
    let binary = binary(
        FRAGMENT,
        &[
            (0x944 | (7 << 12) | call_lo, call_hi | 0x03c0_0000),
            (0x802 | (7 << 12) | (1 << 16) | (0x4000_0000u64 << 32), 0xF << 8),
            EXIT,
            one,
            (0x950 | (7 << 12) | (4 << 24), 0x03c0_0000),
            END,
        ],
    );
    let words = lowered(&binary);
    let ones = words
        .iter()
        .filter(|word| decode_one(**word).unwrap().opcode == Opcode::MOV32I && (*word >> 20) & 0xFFFF_FFFF == 0x3f80_0000)
        .count();
    assert_eq!(ones, 2);
    assert_eq!(opcodes(&words).iter().filter(|opcode| **opcode == Opcode::EXIT).count(), 2);
}

#[test]
fn carry_chain_adds_the_low_word_carry() {
    let pt_off = |start: u32| 0xF_u64 << (start - 64);
    let low = (
        0x210 | (7 << 12) | (4 << 16) | (2 << 24) | (6 << 32),
        0xff | pt_off(77) | (7 << 20) | pt_off(87),
    );
    let high = (
        0x210 | (7 << 12) | (5 << 16) | (3 << 24) | (7 << 32),
        0xff | (1 << 10) | pt_off(77) | (7 << 17) | (7 << 20),
    );
    let words = lowered(&binary(VERTEX, &[low, high, EXIT, END]));
    assert_eq!(
        &opcodes(&words)[..6],
        &[
            Opcode::IADD3_reg,
            Opcode::ISETP_reg,
            Opcode::MOV_reg,
            Opcode::SEL_imm,
            Opcode::IADD3_reg,
            Opcode::IADD3_reg
        ]
    );
    assert_eq!((operand::isetp_dest_p(words[1]), operand::isetp_cmp(words[1])), (0, 1));
    assert_eq!((operand::reg_a(words[1]), operand::reg_b(words[1])), (operand::reg_dest(words[0]), 2));
    assert_eq!((operand::reg_dest(words[5]), operand::reg_a(words[5])), (5, operand::reg_dest(words[4])));
}

#[test]
fn global_atomics_lower_to_maxwell_atom_and_red() {
    let words = lowered(&binary(
        VERTEX,
        &[
            (0x0000_0001_ff00_79a8, 0x0001_6600_081e_e144),
            (0x0000_0419_ff00_798e, 0x0001_e400_0d10_e106),
            (0x0000_0003_0e00_098e, 0x0021_e400_0010_e100),
            EXIT,
            END,
        ],
    ));
    assert_eq!(&opcodes(&words)[..3], &[Opcode::ATOM, Opcode::RED, Opcode::RED]);
    let fields = |word: u64| (operand::reg_dest(word), operand::reg_a(word), (word >> 28) & 0xF_FFFF, (word >> 48) & 1);
    assert_eq!(fields(words[0]), (0, 232, 0, 1));
    assert_eq!(operand::reg_b(words[0]), 1);
    assert_eq!((fields(words[1]), (words[1] >> 23) & 7), ((25, 234, 4, 1), 2));
    assert_eq!((fields(words[2]), (words[2] >> 23) & 7), ((3, 14, 0, 1), 0));
}

#[test]
fn uniform_wide_multiply_carries_into_the_high_word() {
    let words = lowered(&binary(VERTEX, &[(0x0000_000a_0608_72a5, 0x000f_e200_0f8e_0008), EXIT, END]));
    assert_eq!(
        &opcodes(&words)[..4],
        &[Opcode::IMUL_reg, Opcode::IMUL_reg, Opcode::IADD_reg, Opcode::IADD_reg]
    );
    assert_eq!((operand::reg_a(words[0]), operand::reg_b(words[0]), (words[1] >> 53) & 1), (230, 234, 1));
    assert_eq!((operand::reg_dest(words[2]), operand::reg_b(words[2]), (words[2] >> 47) & 1), (232, 232, 1));
    assert_eq!((operand::reg_dest(words[3]), operand::reg_b(words[3]), (words[3] >> 43) & 1), (233, 233, 1));
}

#[test]
fn warp_sum_accumulates_every_active_lane() {
    let words = lowered(&binary(VERTEX, &[(0x0000_0000_0305_73c4, 0x000e_2200_0000_c000), EXIT, END]));
    let ops = opcodes(&words);
    assert_eq!(&ops[..3], &[Opcode::VOTE, Opcode::MOV32I, Opcode::SHFL]);
    assert_eq!(ops.iter().filter(|op| **op == Opcode::SHFL).count(), 32);
    assert_eq!((operand::reg_dest(words[1]), operand::reg_a(words[2])), (233, 3));
    assert_eq!((ops[194 - 1], operand::reg_dest(words[194 - 1])), (Opcode::IADD_reg, 233));
}

#[test]
fn match_all_compares_against_the_lowest_active_lane() {
    let words = lowered(&binary(VERTEX, &[(0x0000_0000_16ff_73a1, 0x000e_2200_0000_0200), EXIT, END]));
    assert_eq!(
        &opcodes(&words)[..11],
        &[
            Opcode::P2R_imm,
            Opcode::VOTE,
            Opcode::IADD_reg,
            Opcode::LOP3_reg,
            Opcode::FLO_reg,
            Opcode::SHFL,
            Opcode::ISETP_reg,
            Opcode::SHFL,
            Opcode::ISETP_reg,
            Opcode::VOTE,
            Opcode::R2P_imm
        ]
    );
    assert_eq!((operand::reg_a(words[5]), operand::reg_b(words[5])), (22, operand::reg_dest(words[4])));
    assert_eq!((operand::reg_a(words[8]), (words[8] >> 39) & 7), (23, 6));
    assert_eq!(((words[9] >> 39) & 7, (words[9] >> 45) & 7), (6, 0));
}

#[test]
fn predicate_save_lowers_to_maxwell_p2r() {
    let words = lowered(&binary(VERTEX, &[(0x0000_0008_ff16_7803, 0x000f_c400_0000_0000), EXIT, END]));
    assert_eq!(opcodes(&words)[0], Opcode::P2R_imm);
    assert_eq!((operand::reg_dest(words[0]), operand::reg_a(words[0]), (words[0] >> 20) & 0x7F), (22, 0xFF, 8));
}

#[test]
fn half_adds_lower_to_hadd2_or_a_half_to_float_conversion() {
    let words = lowered(&binary(
        FRAGMENT,
        &[
            (0x2000_0004_ff08_7230, 0x020f_ca00_0000_4100),
            (0x230 | (7 << 12) | (5 << 16) | (6 << 24) | (7 << 32) | (3 << 60), 1 << 10),
            EXIT,
            END,
        ],
    ));
    assert_eq!(&opcodes(&words)[..2], &[Opcode::F2F_reg, Opcode::HADD2_reg]);
    assert_eq!((operand::reg_dest(words[0]), operand::reg_b(words[0])), (8, 4));
    assert_eq!(((words[0] >> 8) & 3, (words[0] >> 10) & 3, (words[0] >> 41) & 1), (2, 1, 0));
    assert_eq!((operand::reg_a(words[1]), operand::reg_b(words[1]), (words[1] >> 28) & 3, (words[1] >> 47) & 3), (6, 7, 3, 1));
}

#[test]
fn rejects_programs_without_end() {
    assert_eq!(translate(&binary(VERTEX, &[EXIT])), Err(Error::MissingEnd));
}

fn sm70_branch_offset(offset: u64, base: i64) -> (u64, u64) {
    let relative = ((base - offset as i64 - 16) / 4) as u64 & 0xFFFF_FFFF_FFFF;
    ((relative & 0x3FFF_FFFF) << 34, relative >> 30)
}

#[test]
fn jump_tables_reach_lowered_cases() {
    let (brx_lo, brx_hi) = sm70_branch_offset(0x30, 0);
    let binary = binary(
        VERTEX,
        &[
            (0x817 | (7 << 12) | (6 << 16) | (5 << 24) | (1 << 32), 7 << 23),
            (0x819 | (7 << 12) | (12 << 16) | (6 << 24) | (2 << 32), 0xFF | (3 << 9)),
            (0xb82 | (7 << 12) | (6 << 16) | (12 << 24) | (1 << 54), 4 << 9),
            (0x949 | (7 << 12) | (6 << 24) | brx_lo, brx_hi | (7 << 23)),
            (0x802 | (7 << 12) | (0x3f80_0000u64 << 32), 0xF << 8),
            EXIT,
            (0x802 | (7 << 12) | (0x4000_0000u64 << 32), 0xF << 8),
            EXIT,
            END,
        ],
    );
    let program = translate(&binary).unwrap();
    let cfg = crate::build_cfg_with_cbuf(&program.code, |binding, offset| {
        (binding == 1).then(|| [0x40u32, 0x60].get(offset as usize / 4).copied()).flatten()
    });
    let (count, targets) = cfg
        .blocks
        .iter()
        .find_map(|block| match block.branch {
            crate::BranchKind::Indirect { count, targets, .. } => Some((count, targets)),
            _ => None,
        })
        .expect("indirect branch resolved");
    assert_eq!(count, 2);
    let constant = |entry: usize| {
        let mut block = &cfg.blocks[targets[entry].target as usize];
        while let crate::BranchKind::Unconditional { target } = block.branch {
            block = &cfg.blocks[target as usize];
        }
        block.program.instructions.iter().find_map(|inst| match inst.op {
            crate::IrOp::Mov(crate::IrValue::ImmU32(value)) => Some(value),
            _ => None,
        })
    };
    assert_eq!((constant(0), constant(1)), (Some(0x3f80_0000), Some(0x4000_0000)));
}

#[test]
#[ignore]
fn survey_dumped_programs() {
    use std::collections::{BTreeMap, BTreeSet};
    let Some(directory) = std::env::var_os("NEXIUM_AMPERE_SURVEY") else {
        return;
    };
    let mut by_opcode: BTreeMap<u16, (BTreeSet<String>, usize, BTreeSet<(u64, u64)>)> = BTreeMap::new();
    let mut other: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_shader: BTreeMap<String, BTreeSet<u16>> = BTreeMap::new();
    for entry in std::fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("sm86") || path.with_extension("sm50").exists() {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut binary = std::fs::read(&path).unwrap();
        for _ in 0..4096 {
            let (offset, lo, hi) = match translate(&binary) {
                Ok(_) => break,
                Err(lower::Error::Unsupported { offset, lo, hi }) => (offset, lo, hi),
                Err(lower::Error::Register { offset, register }) => {
                    *other.entry(format!("register R{register}")).or_default() += 1;
                    let at = HEADER_SIZE + offset;
                    (offset, u64::from_le_bytes(binary[at..at + 8].try_into().unwrap()), u64::MAX)
                }
                Err(error) => {
                    *other.entry(error.to_string()).or_default() += 1;
                    break;
                }
            };
            per_shader.entry(name.clone()).or_default().insert(if hi == u64::MAX { 0xFFF } else { (lo & 0xFFF) as u16 });
            if hi != u64::MAX {
                let slot = by_opcode.entry((lo & 0xFFF) as u16).or_default();
                slot.0.insert(name.clone());
                slot.1 += 1;
                if slot.2.len() < 4 {
                    slot.2.insert((hi, lo));
                }
            }
            let at = HEADER_SIZE + offset;
            binary[at..at + 8].copy_from_slice(&0x7918u64.to_le_bytes());
        }
    }
    let mut rows: Vec<_> = by_opcode.into_iter().collect();
    rows.sort_by_key(|(_, (shaders, _, _))| std::cmp::Reverse(shaders.len()));
    for (opcode, (shaders, count, samples)) in rows {
        let samples: Vec<String> = samples.iter().map(|(hi, lo)| format!("{hi:016x}_{lo:016x}")).collect();
        println!("{opcode:#05x} shaders={} instances={count} {}", shaders.len(), samples.join(" "));
    }
    for (error, count) in other {
        println!("other {error}: {count}");
    }
    for (name, set) in per_shader {
        let set: Vec<String> = set.iter().map(|op| format!("{op:03x}")).collect();
        println!("shader {name} {}", set.join(","));
    }
}
