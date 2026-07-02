use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::LocalOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir, StarkProtocolConfig, Val,
};
use serde::{Deserialize, Serialize};

use crate::{
    arch::{instructions::SystemOpcode::TERMINATE, ExecutionBus, ExecutionState, EXTRA_EXEC_REGS},
    primitives::Chip,
    system::program::ProgramBus,
};

#[cfg(test)]
mod tests;

/// When a program hasn't terminated. There is no constraints on the exit code.
/// But we will use this value when generating the proof.
pub const DEFAULT_SUSPEND_EXIT_CODE: u32 = 42;

#[derive(Debug, Clone, Copy)]
pub struct VmConnectorAir {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
    pub range_bus: VariableRangeCheckerBus,
    /// The final timestamp will be constrained to be in the range [0, 2^timestamp_max_bits).
    timestamp_max_bits: usize,
}

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct VmConnectorPvs<F> {
    /// The initial PC of this segment.
    pub initial_pc: F,
    /// The final PC of this segment.
    pub final_pc: F,
    /// The initial extra ISA-specific registers of this segment (empty unless the `fp` feature is
    /// on; see [`EXTRA_EXEC_REGS`]).
    pub initial_extra_regs: [F; EXTRA_EXEC_REGS],
    /// The final extra ISA-specific registers of this segment.
    pub final_extra_regs: [F; EXTRA_EXEC_REGS],
    /// The exit code of the whole program. 0 means exited normally. This is only meaningful when
    /// `is_terminate` is 1.
    pub exit_code: F,
    /// Whether the whole program is terminated. 0 means not terminated. 1 means terminated.
    /// Only the last segment of an execution can have `is_terminate` = 1.
    pub is_terminate: F,
}

impl<F: PrimeField32> VmConnectorPvs<F> {
    pub fn is_terminate(&self) -> bool {
        self.is_terminate == F::from_bool(true)
    }

    pub fn exit_code(&self) -> Option<u32> {
        if self.is_terminate() && self.exit_code == F::ZERO {
            Some(self.exit_code.as_canonical_u32())
        } else {
            None
        }
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for VmConnectorAir {
    fn num_public_values(&self) -> usize {
        VmConnectorPvs::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for VmConnectorAir {}
impl<F: Field> ColumnsAir<F> for VmConnectorAir {
    fn columns(&self) -> Option<Vec<String>> {
        <ConnectorCols<F> as openvm_circuit_primitives::StructReflectionHelper>::struct_reflection()
    }
}
impl<F: Field> BaseAir<F> for VmConnectorAir {
    fn width(&self) -> usize {
        ConnectorCols::<F>::width()
    }
}

impl VmConnectorAir {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        range_bus: VariableRangeCheckerBus,
        timestamp_max_bits: usize,
    ) -> Self {
        assert!(
            range_bus.range_max_bits * 2 >= timestamp_max_bits,
            "Range checker not large enough: range_max_bits={}, timestamp_max_bits={}",
            range_bus.range_max_bits,
            timestamp_max_bits
        );
        Self {
            execution_bus,
            program_bus,
            range_bus,
            timestamp_max_bits,
        }
    }

    /// Returns (low_bits, high_bits) to range check.
    fn timestamp_limb_bits(&self) -> (usize, usize) {
        let range_max_bits = self.range_bus.range_max_bits;
        if self.timestamp_max_bits <= range_max_bits {
            (self.timestamp_max_bits, 0)
        } else {
            (range_max_bits, self.timestamp_max_bits - range_max_bits)
        }
    }
}

#[derive(Debug, Copy, Clone, AlignedBorrow, StructReflection, Serialize, Deserialize)]
#[repr(C)]
pub struct ConnectorCols<T> {
    pub pc: T,
    pub extra_regs: [T; EXTRA_EXEC_REGS],
    pub timestamp: T,
    pub is_terminate: T,
    pub exit_code: T,
    /// Lowest `range_bus.range_max_bits` bits of the timestamp
    timestamp_low_limb: T,
    /// Equals 1 if this is the first row of the segment, 0 if this is the second row of the
    /// segment. Used to enforce that the trace has exactly two rows.
    is_begin: T,
}

impl<T: Copy> ConnectorCols<T> {
    fn map<F>(self, f: impl Fn(T) -> F) -> ConnectorCols<F> {
        ConnectorCols {
            pc: f(self.pc),
            extra_regs: self.extra_regs.map(&f),
            timestamp: f(self.timestamp),
            is_terminate: f(self.is_terminate),
            exit_code: f(self.exit_code),
            timestamp_low_limb: f(self.timestamp_low_limb),
            is_begin: f(self.is_begin),
        }
    }

    /// Flattened in `repr(C)` field order, matching [`ConnectorCols`]'s memory layout so the trace
    /// row can be borrowed back as a `ConnectorCols`.
    fn flatten(&self) -> Vec<T> {
        let mut cols = Vec::with_capacity(ConnectorCols::<T>::width());
        cols.push(self.pc);
        cols.extend(self.extra_regs.iter().copied());
        cols.push(self.timestamp);
        cols.push(self.is_terminate);
        cols.push(self.exit_code);
        cols.push(self.timestamp_low_limb);
        cols.push(self.is_begin);
        cols
    }
}

impl<AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues> Air<AB> for VmConnectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &ConnectorCols<AB::Var> = (*local).borrow();
        let next: &ConnectorCols<AB::Var> = (*next).borrow();

        let &VmConnectorPvs {
            initial_pc,
            final_pc,
            initial_extra_regs,
            final_extra_regs,
            exit_code,
            is_terminate,
        } = builder.public_values().borrow();

        builder.when_transition().assert_eq(local.pc, initial_pc);
        builder.when_transition().assert_eq(next.pc, final_pc);
        for i in 0..EXTRA_EXEC_REGS {
            builder
                .when_transition()
                .assert_eq(local.extra_regs[i], initial_extra_regs[i]);
            builder
                .when_transition()
                .assert_eq(next.extra_regs[i], final_extra_regs[i]);
        }
        builder
            .when_transition()
            .when(next.is_terminate)
            .assert_eq(next.exit_code, exit_code);
        builder
            .when_transition()
            .assert_eq(next.is_terminate, is_terminate);

        builder.when_transition().assert_one(local.timestamp);

        // We force the first row to have is_begin = 1 and the last row to have is_begin = 0.
        // Additionally, we enforce that the is_begin column decreases by exactly 1 per row.
        // The only way to satisfy this is to have exactly two rows: one with is_begin = 1 and
        // one with is_begin = 0 (assuming max height < field characteristic)
        builder.when_first_row().assert_one(local.is_begin);
        builder
            .when_transition()
            .assert_eq(next.is_begin + AB::Expr::ONE, local.is_begin);
        builder.when_last_row().assert_zero(local.is_begin);

        self.execution_bus.execute(
            builder,
            local.is_begin, // 1 only if these are [0th, 1st] and not [1st, 0th]
            ExecutionState {
                pc: next.pc,
                timestamp: next.timestamp,
                extra_regs: next.extra_regs,
            },
            ExecutionState {
                pc: local.pc,
                timestamp: local.timestamp,
                extra_regs: local.extra_regs,
            },
        );
        self.program_bus.lookup_instruction(
            builder,
            next.pc,
            AB::Expr::from_usize(TERMINATE.global_opcode().as_usize()),
            [AB::Expr::ZERO, AB::Expr::ZERO, next.exit_code.into()],
            local.is_begin * next.is_terminate,
        );

        // We decompose and range check `local.timestamp` as `timestamp_low_limb,
        // timestamp_high_limb` where `timestamp = timestamp_low_limb + timestamp_high_limb
        // * 2^range_max_bits`.
        let (low_bits, high_bits) = self.timestamp_limb_bits();
        let high_limb = (local.timestamp - local.timestamp_low_limb)
            * AB::F::ONE.div_2exp_u64(self.range_bus.range_max_bits as u64);
        self.range_bus
            .range_check(local.timestamp_low_limb, low_bits)
            .eval(builder, AB::Expr::ONE);
        self.range_bus
            .range_check(high_limb, high_bits)
            .eval(builder, AB::Expr::ONE);
    }
}

pub struct VmConnectorChip<F> {
    pub range_checker: SharedVariableRangeCheckerChip,
    pub boundary_states: [Option<ConnectorCols<u32>>; 2],
    timestamp_max_bits: usize,
    _marker: PhantomData<F>,
}

impl<F> VmConnectorChip<F> {
    pub fn new(range_checker: SharedVariableRangeCheckerChip, timestamp_max_bits: usize) -> Self {
        let range_bus = range_checker.bus();
        assert!(
            range_bus.range_max_bits * 2 >= timestamp_max_bits,
            "Range checker not large enough: range_max_bits={}, timestamp_max_bits={}",
            range_bus.range_max_bits,
            timestamp_max_bits
        );
        Self {
            range_checker,
            boundary_states: [None, None],
            timestamp_max_bits,
            _marker: PhantomData,
        }
    }

    pub fn begin(&mut self, state: ExecutionState<u32, EXTRA_EXEC_REGS>) {
        self.boundary_states[0] = Some(ConnectorCols {
            pc: state.pc,
            extra_regs: state.extra_regs,
            timestamp: state.timestamp,
            is_terminate: 0,
            exit_code: 0,
            timestamp_low_limb: 0, // will be computed during tracegen
            is_begin: 1,
        });
    }

    pub fn end(&mut self, state: ExecutionState<u32, EXTRA_EXEC_REGS>, exit_code: Option<u32>) {
        self.boundary_states[1] = Some(ConnectorCols {
            pc: state.pc,
            extra_regs: state.extra_regs,
            timestamp: state.timestamp,
            is_terminate: exit_code.is_some() as u32,
            exit_code: exit_code.unwrap_or(DEFAULT_SUSPEND_EXIT_CODE),
            timestamp_low_limb: 0, // will be computed during tracegen
            is_begin: 0,
        });
    }

    fn timestamp_limb_bits(&self) -> (usize, usize) {
        let range_max_bits = self.range_checker.bus().range_max_bits;
        if self.timestamp_max_bits <= range_max_bits {
            (self.timestamp_max_bits, 0)
        } else {
            (range_max_bits, self.timestamp_max_bits - range_max_bits)
        }
    }
}

impl<RA, SC> Chip<RA, CpuBackend<SC>> for VmConnectorChip<Val<SC>>
where
    SC: StarkProtocolConfig,
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let [initial_state, final_state] = self.boundary_states.map(|state| {
            let mut state = state.unwrap();
            // Decompose and range check timestamp
            let range_max_bits = self.range_checker.range_max_bits();
            let timestamp_low_limb = state.timestamp & ((1u32 << range_max_bits) - 1);
            state.timestamp_low_limb = timestamp_low_limb;
            let (low_bits, high_bits) = self.timestamp_limb_bits();
            self.range_checker.add_count(timestamp_low_limb, low_bits);
            self.range_checker
                .add_count(state.timestamp >> range_max_bits, high_bits);

            state.map(Val::<SC>::from_u32)
        });

        let trace = RowMajorMatrix::new(
            [initial_state.flatten(), final_state.flatten()].concat(),
            ConnectorCols::<Val<SC>>::width(),
        );

        let mut public_values = Val::<SC>::zero_vec(VmConnectorPvs::<Val<SC>>::width());
        *public_values.as_mut_slice().borrow_mut() = VmConnectorPvs {
            initial_pc: initial_state.pc,
            final_pc: final_state.pc,
            initial_extra_regs: initial_state.extra_regs,
            final_extra_regs: final_state.extra_regs,
            exit_code: final_state.exit_code,
            is_terminate: final_state.is_terminate,
        };
        AirProvingContext::simple(trace, public_values)
    }
}
