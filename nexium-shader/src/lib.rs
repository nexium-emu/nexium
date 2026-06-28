pub mod cfg;
mod decode;
mod disasm;
pub mod ir;
mod opcodes;
mod operand;
mod pretty;
mod translate;
mod walk;

pub use cfg::{build_cfg, BasicBlock, BlockId, BranchKind, Cfg};
pub use decode::{decode_one, Decoded};
pub use disasm::{disassemble, DisasmKind, DisasmLine};
pub use ir::{
    BoolOp, FComp, ICmp, Inst as IrInst, LogicOp, MufuFunc, Op as IrOp, Predicate,
    Program as IrProgram, Value as IrValue, ValueId,
};
pub use opcodes::{Opcode, OPCODE_TABLE};
pub use operand::{cbuf, fmt_cbuf, fmt_reg, imm20, imm32, reg_a, reg_b, reg_c, reg_dest, CbufRef};
pub use pretty::pretty_operands;
pub use translate::{translate_shader, Translator};
pub use walk::{extract_fs_tex_ids, walk_instructions, FsTexId, Instruction};
