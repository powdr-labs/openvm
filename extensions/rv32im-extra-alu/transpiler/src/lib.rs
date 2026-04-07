use std::cell::Cell;

use openvm_instructions::VmOpcode;
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::{BaseAluOpcode, Rv32ITranspilerExtension};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};

/// Number of opcodes in the BaseAlu family (ADD, SUB, XOR, OR, AND).
const BASE_ALU_COUNT: usize = 5;

/// Starting opcode offset for extra BaseAlu chips.
/// Chip `k` (k = 1..N-1) uses opcodes at
/// `BASE_EXTRA_ALU_OFFSET + (k - 1) * BASE_ALU_COUNT`.
pub const BASE_EXTRA_ALU_OFFSET: usize = 0x900;

/// A transpiler extension that wraps `Rv32ITranspilerExtension` and distributes BaseAlu
/// instructions (ADD/SUB/XOR/OR/AND) round-robin across `num_total_alu` chips.
///
/// Chip 0 keeps the original opcodes (`BaseAluOpcode::CLASS_OFFSET` = 0x200–0x204).
/// Chips k = 1..N-1 use opcodes at `BASE_EXTRA_ALU_OFFSET + (k-1)*5` through `+4`.
///
/// `num_total_alu` must be a power of two >= 2.
///
/// **Register this extension instead of** (not alongside) `Rv32ITranspilerExtension`.
/// `Rv32MTranspilerExtension` and `Rv32IoTranspilerExtension` should be registered separately.
pub struct Rv32ExtraAluTranspilerExtension {
    /// Total number of BaseAlu chips (must be a power of two, >= 2).
    pub num_total_alu: usize,
    counter: Cell<usize>,
}

impl Rv32ExtraAluTranspilerExtension {
    pub fn new(num_total_alu: usize) -> Self {
        assert!(
            num_total_alu.is_power_of_two() && num_total_alu >= 2,
            "num_total_alu must be a power of two >= 2, got {num_total_alu}",
        );
        Self {
            num_total_alu,
            counter: Cell::new(0),
        }
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv32ExtraAluTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        let output = Rv32ITranspilerExtension.process_custom(instruction_stream)?;

        // Check whether the (single) transpiled instruction is a BaseAlu opcode.
        if let [Some(insn)] = output.instructions.as_slice() {
            let opcode_val = insn.opcode.as_usize();
            if opcode_val >= <BaseAluOpcode as LocalOpcode>::CLASS_OFFSET
                && opcode_val < <BaseAluOpcode as LocalOpcode>::CLASS_OFFSET + BASE_ALU_COUNT
            {
                let local_op = opcode_val - <BaseAluOpcode as LocalOpcode>::CLASS_OFFSET;
                let cnt = self.counter.get();
                self.counter.set(cnt.wrapping_add(1));
                let chip_idx = cnt % self.num_total_alu;

                if chip_idx > 0 {
                    let new_opcode = VmOpcode::from_usize(
                        BASE_EXTRA_ALU_OFFSET + (chip_idx - 1) * BASE_ALU_COUNT + local_op,
                    );
                    let mut new_insn = insn.clone();
                    new_insn.opcode = new_opcode;
                    return Some(TranspilerOutput {
                        instructions: vec![Some(new_insn)],
                        used_u32s: output.used_u32s,
                    });
                }
            }
        }

        Some(output)
    }
}
