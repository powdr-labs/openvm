use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, ImmInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{AirBuilder, BaseAir, BaseAirWithPublicValues},
};
use serde::{Deserialize, Serialize};

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, FlattenFields)]
pub struct Rv32AuipcCoreCols<T> {
    pub is_valid: T,
    pub imm_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub pc_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32AuipcCoreAir {
    fn width(&self) -> usize {
        Rv32AuipcCoreCols::<F>::width()
    }

    fn columns(&self) -> Vec<String> {
        Rv32AuipcCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcCoreAir {
    fn columns(&self) -> Vec<String> {
        Rv32AuipcCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl Rv32AuipcCoreAir {
    pub fn columns<F: Field>(&self) -> Vec<String> {
        Rv32AuipcCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB, I> VmCoreAir<AB, I> for Rv32AuipcCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32AuipcCoreCols<AB::Var> = (*local_core).borrow();

        let Rv32AuipcCoreCols {
            is_valid,
            imm_limbs,
            pc_limbs,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);
        let intermed_val = pc_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });
        let imm = imm_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });

        builder
            .when(cols.is_valid)
            .assert_eq(rd_data[0], from_pc - intermed_val);

        let mut carry: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << RV32_CELL_BITS).inverse();

        for i in 1..RV32_REGISTER_NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (pc_limbs[i - 1] + imm_limbs[i - 1] - rd_data[i] + carry[i - 1].clone());
            builder.when(is_valid).assert_bool(carry[i].clone());
        }

        // Range checking to 8 bits
        for i in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
            self.bus
                .send_range(rd_data[i * 2], rd_data[i * 2 + 1])
                .eval(builder, is_valid);
        }
        let limbs = [imm_limbs, pc_limbs].concat();
        for i in 0..(RV32_REGISTER_NUM_LIMBS - 2) {
            self.bus
                .send_range(limbs[i * 2], limbs[i * 2 + 1])
                .eval(builder, is_valid);
        }

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [rd_data.map(|x| x.into())].into(),
            instruction: ImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32AuipcOpcode::CLASS_OFFSET
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32AuipcCoreRecord<F> {
    pub imm_limbs: [F; RV32_REGISTER_NUM_LIMBS - 1],
    pub pc_limbs: [F; RV32_REGISTER_NUM_LIMBS - 1],
    pub rd_data: [F; RV32_REGISTER_NUM_LIMBS],
}

pub struct Rv32AuipcCoreChip {
    pub air: Rv32AuipcCoreAir,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl Rv32AuipcCoreChip {
    pub fn new(bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>) -> Self {
        Self {
            air: Rv32AuipcCoreAir {
                bus: bitwise_lookup_chip.bus(),
            },
            bitwise_lookup_chip,
        }
    }

    pub fn columns<F: Field>(&self) -> Vec<String> {
        self.air.columns::<F>()
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32AuipcCoreChip
where
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = Rv32AuipcCoreRecord<F>;
    type Air = Rv32AuipcCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode = Rv32AuipcOpcode::from_usize(
            instruction
                .opcode
                .local_opcode_idx(Rv32AuipcOpcode::CLASS_OFFSET),
        );
        let imm = instruction.c.as_canonical_u32();
        let rd_data = run_auipc(local_opcode, from_pc, imm);
        let rd_data_field = rd_data.map(F::from_canonical_u32);

        let output = AdapterRuntimeContext::without_pc([rd_data_field]);

        let imm_limbs = array::from_fn(|i| (imm >> (i * RV32_CELL_BITS)) & RV32_LIMB_MAX);
        let pc_limbs = array::from_fn(|i| (from_pc >> ((i + 1) * RV32_CELL_BITS)) & RV32_LIMB_MAX);

        for i in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
            self.bitwise_lookup_chip
                .request_range(rd_data[i * 2], rd_data[i * 2 + 1]);
        }

        let limbs: Vec<u32> = [imm_limbs, pc_limbs].concat();
        for i in 0..(RV32_REGISTER_NUM_LIMBS - 2) {
            self.bitwise_lookup_chip
                .request_range(limbs[i * 2], limbs[i * 2 + 1]);
        }

        Ok((
            output,
            Self::Record {
                imm_limbs: imm_limbs.map(F::from_canonical_u32),
                pc_limbs: pc_limbs.map(F::from_canonical_u32),
                rd_data: rd_data.map(F::from_canonical_u32),
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32AuipcOpcode::from_usize(opcode - Rv32AuipcOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32AuipcCoreCols<F> = row_slice.borrow_mut();
        core_cols.imm_limbs = record.imm_limbs;
        core_cols.pc_limbs = record.pc_limbs;
        core_cols.rd_data = record.rd_data;
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns rd_data
pub(super) fn run_auipc(
    _opcode: Rv32AuipcOpcode,
    pc: u32,
    imm: u32,
) -> [u32; RV32_REGISTER_NUM_LIMBS] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    array::from_fn(|i| (rd >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX)
}
