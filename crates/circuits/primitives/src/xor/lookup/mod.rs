use std::{
    borrow::Borrow,
    mem::size_of,
    sync::{
        atomic::{self, AtomicU32},
        Arc,
    },
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_field::Field,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{get_air_name, Air, BaseAir, BaseAirWithPublicValues, PairBuilder, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};

use super::bus::XorBus;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, FlattenFields)]
pub struct XorLookupCols<T> {
    pub mult: T,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, FlattenFields)]
pub struct XorLookupPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub const NUM_XOR_LOOKUP_COLS: usize = size_of::<XorLookupCols<u8>>();
pub const NUM_XOR_LOOKUP_PREPROCESSED_COLS: usize = size_of::<XorLookupPreprocessedCols<u8>>();

/// Xor via preprocessed lookup table. Can only be used if inputs have less than approximately 10-bits.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct XorLookupAir<const M: usize> {
    pub bus: XorBus,
}

impl<F: Field, const M: usize> BaseAirWithPublicValues<F> for XorLookupAir<M> {
    fn columns(&self) -> Vec<String> {
        XorLookupCols::<F>::flatten_fields().unwrap()
    }
}
impl<F: Field, const M: usize> PartitionedBaseAir<F> for XorLookupAir<M> {}
impl<F: Field, const M: usize> BaseAir<F> for XorLookupAir<M> {
    fn width(&self) -> usize {
        NUM_XOR_LOOKUP_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<_> = (0..(1 << M) * (1 << M))
            .flat_map(|i| {
                let x = i / (1 << M);
                let y = i % (1 << M);
                let z = x ^ y;
                [x, y, z].map(F::from_canonical_u32)
            })
            .collect();

        Some(RowMajorMatrix::new(rows, NUM_XOR_LOOKUP_PREPROCESSED_COLS))
    }

    fn columns(&self) -> Vec<String> {
        XorLookupCols::<F>::flatten_fields().unwrap()
    }
}

impl<const M: usize> XorLookupAir<M> {
    pub fn columns<F: Field>(&self) -> Vec<String> {
        XorLookupCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB, const M: usize> Air<AB> for XorLookupAir<M>
where
    AB: InteractionBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let prep_local = preprocessed.row_slice(0);
        let prep_local: &XorLookupPreprocessedCols<AB::Var> = (*prep_local).borrow();
        let local = main.row_slice(0);
        let local: &XorLookupCols<AB::Var> = (*local).borrow();

        self.bus
            .receive(prep_local.x, prep_local.y, prep_local.z)
            .eval(builder, local.mult);
    }
}

/// This chip gets requests to compute the xor of two numbers x and y of at most M bits.
/// It generates a preprocessed table with a row for each possible triple (x, y, x^y)
/// and keeps count of the number of times each triple is requested for the single main trace column.
#[derive(Debug)]
pub struct XorLookupChip<const M: usize> {
    pub air: XorLookupAir<M>,
    pub count: Vec<Vec<AtomicU32>>,
}

impl<const M: usize> XorLookupChip<M> {
    pub fn new(bus: usize) -> Self {
        let mut count = vec![];
        for _ in 0..(1 << M) {
            let mut row = vec![];
            for _ in 0..(1 << M) {
                row.push(AtomicU32::new(0));
            }
            count.push(row);
        }
        Self {
            air: XorLookupAir::new(XorBus(bus)),
            count,
        }
    }

    /// The xor bus this chip interacts with
    pub fn bus(&self) -> XorBus {
        self.air.bus
    }

    fn calc_xor(&self, x: u32, y: u32) -> u32 {
        x ^ y
    }

    pub fn request(&self, x: u32, y: u32) -> u32 {
        let val_atomic = &self.count[x as usize][y as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        self.calc_xor(x, y)
    }

    pub fn clear(&self) {
        for i in 0..(1 << M) {
            for j in 0..(1 << M) {
                self.count[i][j].store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        debug_assert_eq!(self.count.len(), 1 << M);
        let multiplicities: Vec<_> = self
            .count
            .iter()
            .flat_map(|count_x| {
                debug_assert_eq!(count_x.len(), 1 << M);
                count_x
                    .iter()
                    .map(|count_xy| F::from_canonical_u32(count_xy.load(atomic::Ordering::SeqCst)))
            })
            .collect();

        RowMajorMatrix::new_col(multiplicities)
    }

    pub fn columns<F: Field>(&self) -> Vec<String> {
        self.air.columns::<F>()
    }
}

impl<SC: StarkGenericConfig, const M: usize> Chip<SC> for XorLookupChip<M> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProofInput::simple_no_pis(trace)
    }
}

impl<const M: usize> ChipUsageGetter for XorLookupChip<M> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        1 << (2 * M)
    }

    fn trace_width(&self) -> usize {
        NUM_XOR_LOOKUP_COLS
    }
}
