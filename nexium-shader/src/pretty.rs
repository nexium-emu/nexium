

use super::operand::{cbuf, fmt_cbuf, fmt_reg, imm20, imm32, ldc_ref, ldc_src_reg, reg_a, reg_b, reg_c, reg_dest};
use super::opcodes::Opcode;

pub fn pretty_operands(opcode: Opcode, insn: u64) -> Option<String> {
    use Opcode::*;
    match opcode {

        FMUL_reg | FADD_reg | IADD_reg | IMUL_reg | LOP_reg | SHL_reg | SHR_reg => Some(format!(
            "{}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_reg(reg_b(insn))
        )),
        FMUL_cbuf | FADD_cbuf | IADD_cbuf | IMUL_cbuf | LOP_cbuf => Some(format!(
            "{}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_cbuf(cbuf(insn))
        )),
        FMUL_imm | FADD_imm | IADD_imm => Some(format!(
            "{}, {}, {:#x}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            imm20(insn)
        )),
        FMUL32I | IMUL32I => Some(format!(
            "{}, {}, {:#010x}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            imm32(insn)
        )),

        FFMA_reg => Some(format!(
            "{}, {}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_reg(reg_b(insn)),
            fmt_reg(reg_c(insn))
        )),
        FFMA_cr => Some(format!(
            "{}, {}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_cbuf(cbuf(insn)),
            fmt_reg(reg_c(insn))
        )),
        FFMA_rc => Some(format!(
            "{}, {}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_reg(reg_b(insn)),
            fmt_cbuf(cbuf(insn))
        )),
        FFMA_imm => Some(format!(
            "{}, {}, {:#x}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            imm20(insn),
            fmt_reg(reg_c(insn))
        )),

        MOV_reg => Some(format!("{}, {}", fmt_reg(reg_dest(insn)), fmt_reg(reg_b(insn)))),
        MOV_cbuf => Some(format!("{}, {}", fmt_reg(reg_dest(insn)), fmt_cbuf(cbuf(insn)))),
        MOV_imm => Some(format!("{}, {:#x}", fmt_reg(reg_dest(insn)), imm20(insn))),
        MOV32I => Some(format!(
            "{}, {:#010x}",
            fmt_reg(reg_dest(insn)),
            imm32(insn)
        )),

        MUFU => {
            let func = (insn >> 20) & 0xF;
            let func_name = match func {
                0 => "cos",
                1 => "sin",
                2 => "ex2",
                3 => "lg2",
                4 => "rcp",
                5 => "rsq",
                6 => "rcp_64h",
                7 => "rsq_64h",
                8 => "sqrt",
                _ => "?",
            };
            Some(format!(
                "{}, {}, {}",
                fmt_reg(reg_dest(insn)),
                fmt_reg(reg_a(insn)),
                func_name
            ))
        }

        ALD => Some(format!(
            "{}, a[{:#x}]",
            fmt_reg(reg_dest(insn)),
            (insn >> 20) & 0x3FF
        )),
        AST => Some(format!(
            "a[{:#x}], {}",
            (insn >> 20) & 0x3FF,
            fmt_reg(reg_dest(insn))
        )),
        IPA => Some(format!(
            "{}, a[{:#x}], {}",
            fmt_reg(reg_dest(insn)),
            ((insn >> 30) & 0xFF) * 4,
            fmt_reg(reg_b(insn))
        )),

        TEXS | TLD4S | TLDS => Some(format!(
            "{}, {}, {}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            fmt_reg(reg_b(insn))
        )),

        LDC => {
            let r = ldc_ref(insn);
            let src = ldc_src_reg(insn);
            if src == super::operand::RZ {
                Some(format!(
                    "{}, c[{:#x}]:{:#x}",
                    fmt_reg(reg_dest(insn)),
                    r.binding,
                    r.byte_offset
                ))
            } else {
                Some(format!(
                    "{}, c[{:#x}]:{}+{}",
                    fmt_reg(reg_dest(insn)),
                    r.binding,
                    fmt_reg(src),
                    r.byte_offset
                ))
            }
        }
        LDG | LDL | LDS | STG | STL | STS => Some(format!(
            "{}, [{}+{:#x}]",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            imm20(insn)
        )),

        EXIT => Some(String::new()),
        BRA | JMP | RET => Some(format!("{:#x}", imm20(insn))),

        SHL_imm | SHR_imm => Some(format!(
            "{}, {}, {:#x}",
            fmt_reg(reg_dest(insn)),
            fmt_reg(reg_a(insn)),
            imm20(insn)
        )),
        ISETP_reg | FSETP_reg => Some(format!(
            "P{}|P{}, {}, {}",
            (insn >> 3) & 0x7,
            insn & 0x7,
            fmt_reg(reg_a(insn)),
            fmt_reg(reg_b(insn))
        )),

        _ => None,
    }
}
