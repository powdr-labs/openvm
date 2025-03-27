use std::borrow::{Borrow, BorrowMut};

use itertools::izip;
use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::FieldArithmeticOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(AlignedBorrow, FlattenFields)]
pub struct FieldArithmeticCoreCols<T> {
    pub a: T,
    pub b: T,
    pub c: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,
}

#[derive(Copy, Clone, Debug)]
pub struct FieldArithmeticCoreAir {}

impl<F: Field> BaseAir<F> for FieldArithmeticCoreAir {
    fn width(&self) -> usize {
        FieldArithmeticCoreCols::<F>::width()
    }

    fn columns(&self) -> Vec<String> {
        FieldArithmeticCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl FieldArithmeticCoreAir {
    pub fn columns<F: Field>(&self) -> Vec<String> {
        FieldArithmeticCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldArithmeticCoreAir {
    fn columns(&self) -> Vec<String> {
        FieldArithmeticCoreCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB, I> VmCoreAir<AB, I> for FieldArithmeticCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 2]>,
    I::Writes: From<[[AB::Expr; 1]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FieldArithmeticCoreCols<_> = local_core.borrow();

        let a = cols.a;
        let b = cols.b;
        let c = cols.c;

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [ADD, SUB, MUL, DIV];
        let results = [b + c, b - c, b * c, b * cols.divisor_inv];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative inverse of `io.y`.

        let mut is_valid = AB::Expr::ZERO;
        let mut expected_opcode = AB::Expr::ZERO;
        let mut expected_result = AB::Expr::ZERO;
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        builder.assert_eq(a, expected_result);
        builder.assert_bool(is_valid.clone());
        builder.assert_eq(cols.is_div, c * cols.divisor_inv);

        AdapterAirContext {
            to_pc: None,
            reads: [[cols.b.into()], [cols.c.into()]].into(),
            writes: [[cols.a.into()]].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        FieldArithmeticOpcode::CLASS_OFFSET
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FieldArithmeticRecord<F> {
    pub opcode: FieldArithmeticOpcode,
    pub a: F,
    pub b: F,
    pub c: F,
}

pub struct FieldArithmeticCoreChip {
    pub air: FieldArithmeticCoreAir,
}

impl FieldArithmeticCoreChip {
    pub fn new() -> Self {
        Self {
            air: FieldArithmeticCoreAir {},
        }
    }

    pub fn columns<F: Field>(&self) -> Vec<String> {
        self.air.columns::<F>()
    }
}

impl Default for FieldArithmeticCoreChip {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for FieldArithmeticCoreChip
where
    I::Reads: Into<[[F; 1]; 2]>,
    I::Writes: From<[[F; 1]; 1]>,
{
    type Record = FieldArithmeticRecord<F>;
    type Air = FieldArithmeticCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let local_opcode = FieldArithmeticOpcode::from_usize(
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET),
        );

        let data: [[F; 1]; 2] = reads.into();
        let b = data[0][0];
        let c = data[1][0];
        let a = FieldArithmetic::run_field_arithmetic(local_opcode, b, c).unwrap();

        let output: AdapterRuntimeContext<F, I> = AdapterRuntimeContext {
            to_pc: None,
            writes: [[a]].into(),
        };

        let record = Self::Record {
            opcode: local_opcode,
            a,
            b,
            c,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - FieldArithmeticOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let FieldArithmeticRecord { opcode, a, b, c } = record;
        let row_slice: &mut FieldArithmeticCoreCols<_> = row_slice.borrow_mut();
        row_slice.a = a;
        row_slice.b = b;
        row_slice.c = c;

        row_slice.is_add = F::from_bool(opcode == FieldArithmeticOpcode::ADD);
        row_slice.is_sub = F::from_bool(opcode == FieldArithmeticOpcode::SUB);
        row_slice.is_mul = F::from_bool(opcode == FieldArithmeticOpcode::MUL);
        row_slice.is_div = F::from_bool(opcode == FieldArithmeticOpcode::DIV);
        row_slice.divisor_inv = if opcode == FieldArithmeticOpcode::DIV {
            c.inverse()
        } else {
            F::ZERO
        };
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub struct FieldArithmetic;
impl FieldArithmetic {
    pub(super) fn run_field_arithmetic<F: Field>(
        opcode: FieldArithmeticOpcode,
        b: F,
        c: F,
    ) -> Option<F> {
        match opcode {
            FieldArithmeticOpcode::ADD => Some(b + c),
            FieldArithmeticOpcode::SUB => Some(b - c),
            FieldArithmeticOpcode::MUL => Some(b * c),
            FieldArithmeticOpcode::DIV => {
                if c.is_zero() {
                    None
                } else {
                    Some(b * c.inverse())
                }
            }
        }
    }
}
