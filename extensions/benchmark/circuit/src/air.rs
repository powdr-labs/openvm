use openvm_circuit::arch::{ExecutionBridge, ExecutionState, PcIncOrSet};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir,
};

/// Column indices for the structural columns in BenchmarkAir.
pub const COL_IS_VALID: usize = 0;
pub const COL_IS_FIRST: usize = 1;
pub const COL_IS_LAST: usize = 2;
pub const COL_SECTION_IDX: usize = 3;
pub const COL_PC: usize = 4;
pub const COL_TIMESTAMP: usize = 5;
pub const NUM_STRUCTURAL_COLS: usize = 6;

/// A configurable benchmark AIR with runtime-determined width.
///
/// Implements the DeferralOutput-style multi-row pattern:
/// each precompile invocation spans `rows_per_invocation` consecutive rows.
/// Only the last row of each section interacts with the execution bridge.
///
/// Payload columns (indices `NUM_STRUCTURAL_COLS..num_columns`) are zero-filled
/// and constrained with boolean constraints and range check interactions.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkAir {
    pub execution_bridge: ExecutionBridge,
    pub bitwise_bus: BitwiseOperationLookupBus,
    pub num_columns: usize,
    pub num_constraints: usize,
    pub num_interactions: usize,
    pub rows_per_invocation: usize,
    pub opcode: usize,
}

impl<F> BaseAir<F> for BenchmarkAir {
    fn width(&self) -> usize {
        self.num_columns
    }
}
impl<F> BaseAirWithPublicValues<F> for BenchmarkAir {}
impl<F> PartitionedBaseAir<F> for BenchmarkAir {}
impl<F> ColumnsAir<F> for BenchmarkAir {}

impl<AB> Air<AB> for BenchmarkAir
where
    AB: InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let next = main.row_slice(1).expect("window should have two elements");

        let local_is_valid = local[COL_IS_VALID];
        let local_is_first = local[COL_IS_FIRST];
        let local_section_idx = local[COL_SECTION_IDX];
        let local_pc = local[COL_PC];
        let local_timestamp = local[COL_TIMESTAMP];

        let next_is_valid = next[COL_IS_VALID];
        let next_is_first = next[COL_IS_FIRST];
        let next_section_idx = next[COL_SECTION_IDX];
        let next_pc = next[COL_PC];
        let next_timestamp = next[COL_TIMESTAMP];

        // is_transition = next row is a continuation (valid but not first)
        let is_transition: AB::Expr = next_is_valid.into() - next_is_first.into();
        // is_last = current row is the last of a section
        let local_is_last: AB::Expr = local_is_valid.into() - is_transition.clone();

        // -- Section structure constraints (following DeferralOutputAir) --

        // is_valid and is_first are boolean
        builder.assert_bool(local_is_valid);
        builder.assert_bool(local_is_first);

        // Valid rows are contiguous at the top of the trace
        builder
            .when_transition()
            .assert_bool(AB::Expr::from(local_is_valid) - AB::Expr::from(next_is_valid));

        // First valid row must be is_first
        builder
            .when_first_row()
            .when(local_is_valid)
            .assert_one(local_is_first);

        // Constrain is_last column
        builder.assert_eq(local[COL_IS_LAST], local_is_last.clone());

        // Invalid rows have zero flags
        builder
            .when(AB::Expr::ONE - AB::Expr::from(local_is_valid))
            .assert_zero(local_is_first);
        builder
            .when(AB::Expr::ONE - AB::Expr::from(local_is_valid))
            .assert_zero(local_section_idx);

        // section_idx resets on is_first, increments by 1 on transition
        builder.when(local_is_first).assert_zero(local_section_idx);
        builder
            .when(is_transition.clone())
            .assert_one(AB::Expr::from(next_section_idx) - AB::Expr::from(local_section_idx));

        // State columns constant within a section (when next.section_idx != 0)
        let mut when_section_transition = builder.when(next_section_idx);
        when_section_transition.assert_eq(local_pc, next_pc);
        when_section_transition.assert_eq(local_timestamp, next_timestamp);

        // -- Boolean constraints on payload columns --
        let num_payload = self.num_columns.saturating_sub(NUM_STRUCTURAL_COLS);
        let num_bool_constraints = self.num_constraints.min(num_payload);
        for i in 0..num_bool_constraints {
            let col = local[NUM_STRUCTURAL_COLS + i];
            builder
                .when(local_is_valid)
                .assert_zero(AB::Expr::from(col) * (AB::Expr::from(col) - AB::Expr::ONE));
        }

        // -- Range check interactions on payload columns --
        let num_range_interactions = self.num_interactions.min(num_payload);
        for i in 0..num_range_interactions {
            let col = local[NUM_STRUCTURAL_COLS + i];
            self.bitwise_bus
                .send_range(col.into(), AB::Expr::ZERO)
                .eval(builder, local_is_valid);
        }

        // -- Execution bridge on is_last row --
        // Operands are all zero (our benchmark instructions have no meaningful operands)
        // timestamp_change = 1 (minimal)
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                AB::Expr::from_usize(self.opcode),
                [AB::Expr::ZERO; 5],
                ExecutionState::<AB::Expr> {
                    pc: local_pc.into(),
                    timestamp: local_timestamp.into(),
                },
                AB::Expr::ONE,
                PcIncOrSet::Inc(AB::Expr::from_u32(DEFAULT_PC_STEP)),
            )
            .eval(builder, local_is_last);
    }
}
