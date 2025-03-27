//! Range check for a fixed bit size via preprocessed trace.
//!
//! Caution: We almost always prefer to use the [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead of this chip.
// Adapted from Valida

use core::mem::size_of;
use std::{
    borrow::{Borrow, BorrowMut},
    sync::atomic::AtomicU32,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir, PairBuilder},
    p3_field::Field,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

mod bus;

#[cfg(test)]
pub mod tests;

pub use bus::*;

#[derive(Default, AlignedBorrow, Copy, Clone, FlattenFields)]
#[repr(C)]
pub struct RangeCols<T> {
    pub mult: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone, FlattenFields)]
#[repr(C)]
pub struct RangePreprocessedCols<T> {
    pub counter: T,
}

pub const NUM_RANGE_COLS: usize = size_of::<RangeCols<u8>>();
pub const NUM_RANGE_PREPROCESSED_COLS: usize = size_of::<RangePreprocessedCols<u8>>();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RangeCheckerAir {
    pub bus: RangeCheckBus,
}

impl RangeCheckerAir {
    pub fn range_max(&self) -> u32 {
        self.bus.range_max
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RangeCheckerAir {
    fn columns(&self) -> Vec<String> {
        RangeCols::<F>::flatten_fields().unwrap()
    }
}
impl<F: Field> PartitionedBaseAir<F> for RangeCheckerAir {}
impl<F: Field> BaseAir<F> for RangeCheckerAir {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..self.range_max()).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl RangeCheckerAir {
    pub fn columns<F: Field>(&self) -> Vec<String> {
        RangeCols::<F>::flatten_fields().unwrap()
            
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for RangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &RangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &RangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.counter)
            .eval(builder, local.mult);
    }
}

pub struct RangeCheckerChip {
    pub air: RangeCheckerAir,
    count: Vec<AtomicU32>,
}

impl RangeCheckerChip {
    pub fn new(bus: RangeCheckBus) -> Self {
        let mut count = vec![];
        for _ in 0..bus.range_max {
            count.push(AtomicU32::new(0));
        }

        Self {
            air: RangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> RangeCheckBus {
        self.air.bus
    }

    pub fn range_max(&self) -> u32 {
        self.air.range_max()
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(self.air.range_max() as usize * NUM_RANGE_COLS);
        for (n, row) in rows.chunks_exact_mut(NUM_RANGE_COLS).enumerate() {
            let cols: &mut RangeCols<F> = (*row).borrow_mut();
            cols.mult =
                F::from_canonical_u32(self.count[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_RANGE_COLS)
    }

    pub fn columns<F: Field>(&self) -> Vec<String> {
        self.air.columns::<F>()
    }
}
