use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::Field,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::ProgramBus;

#[derive(Copy, Clone, Debug, AlignedBorrow, PartialEq, Eq, FlattenFields)]
#[repr(C)]
pub struct ProgramCols<T> {
    pub exec: ProgramExecutionCols<T>,
    pub exec_freq: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow, PartialEq, Eq, FlattenFields)]
#[repr(C)]
pub struct ProgramExecutionCols<T> {
    pub pc: T,

    pub opcode: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    pub g: T,
}

#[derive(Clone, Copy, Debug)]
pub struct ProgramAir {
    pub bus: ProgramBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for ProgramAir {
    fn columns(&self) -> Vec<String> {
        ProgramCols::<F>::flatten_fields().unwrap()
    }
}
impl<F: Field> PartitionedBaseAir<F> for ProgramAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![ProgramExecutionCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        1
    }
}
impl<F: Field> BaseAir<F> for ProgramAir {
    fn width(&self) -> usize {
        ProgramCols::<F>::width()
    }

    fn columns(&self) -> Vec<String> {
        ProgramCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for ProgramAir {
    fn eval(&self, builder: &mut AB) {
        let common_trace = builder.common_main();
        let cached_trace = &builder.cached_mains()[0];

        let exec_freq = common_trace.row_slice(0)[0];
        let exec_cols = cached_trace.row_slice(0).to_vec();

        builder.push_receive(self.bus.0, exec_cols, exec_freq);
    }
}
