use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use derive_new::new;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};

use super::*;
use crate::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};

// We only create an Air for testing purposes

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, new)]
pub struct AssertLessThanCols<T, const AUX_LEN: usize> {
    pub x: T,
    pub y: T,
    pub aux: LessThanAuxCols<T, AUX_LEN>,
}

#[derive(Clone, Copy)]
pub struct AssertLtTestAir<const AUX_LEN: usize>(pub AssertLtSubAir);

impl<F: Field, const AUX_LEN: usize> BaseAirWithPublicValues<F> for AssertLtTestAir<AUX_LEN> {
    fn columns(&self) -> Vec<String> {
        todo!()
    }
}
impl<F: Field, const AUX_LEN: usize> PartitionedBaseAir<F> for AssertLtTestAir<AUX_LEN> {}
impl<F: Field, const AUX_LEN: usize> BaseAir<F> for AssertLtTestAir<AUX_LEN> {
    fn width(&self) -> usize {
        AssertLessThanCols::<F, AUX_LEN>::width()
    }
}
impl<AB: InteractionBuilder, const AUX_LEN: usize> Air<AB> for AssertLtTestAir<AUX_LEN> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &AssertLessThanCols<_, AUX_LEN> = (*local).borrow();

        let io = AssertLessThanIo::new(local.x, local.y, AB::F::ONE);
        self.0.eval(builder, (io, &local.aux.lower_decomp));
    }
}

pub struct AssertLessThanChip<const AUX_LEN: usize> {
    pub air: AssertLtTestAir<AUX_LEN>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
    pub pairs: Vec<(u32, u32)>,
}

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn new(max_bits: usize, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let bus = range_checker.bus();
        Self {
            air: AssertLtTestAir(AssertLtSubAir::new(bus, max_bits)),
            range_checker,
            pairs: vec![],
        }
    }

    pub fn generate_trace<F: Field>(self) -> RowMajorMatrix<F> {
        assert!(self.pairs.len().is_power_of_two());
        let width: usize = AssertLessThanCols::<F, AUX_LEN>::width();

        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut AssertLessThanCols<F, AUX_LEN> = row.borrow_mut();
                row.x = F::from_canonical_u32(x);
                row.y = F::from_canonical_u32(y);
                self.air
                    .0
                    .generate_subrow((&self.range_checker, x, y), &mut row.aux.lower_decomp);
            });

        RowMajorMatrix::new(rows, width)
    }
}

#[test]
fn test_borrow_mut_roundtrip() {
    const AUX_LEN: usize = 2; // number of auxiliary columns is two

    let num_cols = AssertLessThanCols::<usize, AUX_LEN>::width();
    let mut all_cols = (0..num_cols).collect::<Vec<usize>>();

    let lt_cols: &mut AssertLessThanCols<_, AUX_LEN> = all_cols[..].borrow_mut();

    lt_cols.x = 2;
    lt_cols.y = 8;
    lt_cols.aux.lower_decomp[0] = 1;
    lt_cols.aux.lower_decomp[1] = 0;

    assert_eq!(all_cols[0], 2);
    assert_eq!(all_cols[1], 8);
    assert_eq!(all_cols[2], 1);
    assert_eq!(all_cols[3], 0);
}

#[test]
fn test_assert_less_than_chip_lt() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut chip = AssertLessThanChip::<AUX_LEN>::new(max_bits, range_checker.clone());
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(14321, 26883), (0, 1), (28, 120), (337, 456)];
    let trace = chip.generate_trace();
    let range_trace: DenseMatrix<BabyBear> = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut chip = AssertLessThanChip::<AUX_LEN>::new(max_bits, range_checker.clone());
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(14321, 26883), (0, 1), (28, 120), (337, 456)];
    let trace = chip.generate_trace();
    let range_trace: DenseMatrix<BabyBear> = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_assert_less_than_negative_1() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut chip = AssertLessThanChip::<AUX_LEN>::new(max_bits, range_checker.clone());
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(28, 29)];
    let mut trace = chip.generate_trace();
    let range_trace = range_checker.generate_trace();

    // Make the trace invalid
    trace.values.swap(0, 1);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace]).err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_assert_less_than_negative_2() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));
    let mut chip = AssertLessThanChip::<AUX_LEN>::new(max_bits, range_checker.clone());
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(28, 29)];
    let mut trace = chip.generate_trace();
    let range_trace = range_checker.generate_trace();

    // Make the trace invalid
    trace.values[2] = FieldAlgebra::from_canonical_u64(1 << decomp as u64);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace],).err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
