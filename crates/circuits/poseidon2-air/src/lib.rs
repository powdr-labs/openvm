//! This is a wrapper around the Plonky3 [p3_poseidon2_air] used only for integration convenience to
//! get around some complications with field-specific generics associated with Poseidon2.
//! Currently it is only intended for use in OpenVM with BabyBear.
//!
//! We do not recommend external use of this crate, and suggest using the [p3_poseidon2_air] crate directly.

use core::mem::MaybeUninit;
use std::sync::Arc;

use openvm_stark_backend::{
    p3_field::{Field, PrimeField},
    p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut},
};
pub use openvm_stark_sdk::p3_baby_bear;
pub use p3_poseidon2;
use p3_poseidon2::{ExternalLayerConstants, GenericPoseidon2LinearLayers, Poseidon2};
pub use p3_poseidon2_air::{self, Poseidon2Air};
pub use p3_symmetric::{self, Permutation};
use tracing::instrument;

mod air;
mod babybear;
mod config;
mod permute;

pub use air::*;
pub use babybear::*;
pub use config::*;
pub use permute::*;

#[cfg(test)]
mod tests;

pub const POSEIDON2_WIDTH: usize = 16;
// NOTE: these constants are for BabyBear only.
pub const BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS: usize = 4;
pub const BABY_BEAR_POSEIDON2_FULL_ROUNDS: usize = 8;
pub const BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS: usize = 13;

// Currently we only support SBOX_DEGREE = 7
pub const BABY_BEAR_POSEIDON2_SBOX_DEGREE: u64 = 7;

/// `SBOX_REGISTERS` affects the max constraint degree of the AIR. See [p3_poseidon2_air] for more details.
#[derive(Debug)]
pub struct Poseidon2SubChip<F: Field, const SBOX_REGISTERS: usize> {
    // This is Arc purely because Poseidon2Air cannot derive Clone
    pub air: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub(crate) executor: Poseidon2Executor<F>,
    pub(crate) constants: Plonky3RoundConstants<F>,
}

impl<F: PrimeField, const SBOX_REGISTERS: usize> Poseidon2SubChip<F, SBOX_REGISTERS> {
    pub fn new(constants: Poseidon2Constants<F>) -> Self {
        let (external_constants, internal_constants) = constants.to_external_internal_constants();
        Self {
            air: Arc::new(Poseidon2SubAir::new(constants.into())),
            executor: Poseidon2Executor::new(external_constants, internal_constants),
            constants: constants.into(),
        }
    }

    pub fn permute(&self, input_state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute(input_state),
        }
    }

    pub fn permute_mut(&self, input_state: &mut [F; POSEIDON2_WIDTH]) {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute_mut(input_state),
        };
    }

    pub fn generate_trace(&self, inputs: Vec<[F; POSEIDON2_WIDTH]>) -> RowMajorMatrix<F>
    where
        F: PrimeField,
    {
        match self.air.as_ref() {
            Poseidon2SubAir::BabyBearMds(_) => generate_trace_rows::<
                F,
                BabyBearPoseidon2LinearLayers,
                POSEIDON2_WIDTH,
                BABY_BEAR_POSEIDON2_SBOX_DEGREE,
                SBOX_REGISTERS,
                BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
                BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
            >(inputs, &self.constants),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Poseidon2Executor<F: Field> {
    BabyBearMds(Plonky3Poseidon2Executor<F, BabyBearPoseidon2LinearLayers>),
}

impl<F: PrimeField> Poseidon2Executor<F> {
    pub fn new(
        external_constants: ExternalLayerConstants<F, POSEIDON2_WIDTH>,
        internal_constants: Vec<F>,
    ) -> Self {
        Self::BabyBearMds(Plonky3Poseidon2Executor::new(
            external_constants,
            internal_constants,
        ))
    }
}

pub type Plonky3Poseidon2Executor<F, LinearLayers> = Poseidon2<
    <F as Field>::Packing,
    Poseidon2ExternalLayer<F, LinearLayers, POSEIDON2_WIDTH>,
    Poseidon2InternalLayer<F, LinearLayers>,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
>;

// TODO: Take generic iterable
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut vec = Vec::with_capacity(n * ncols * 2);
    let trace: &mut [MaybeUninit<F>] = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace: RowMajorMatrixViewMut<MaybeUninit<F>> = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(perm, input, constants);
    });

    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    mut state: [F; WIDTH],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    perm.export.write(F::ONE);
    perm.inputs
        .iter_mut()
        .zip(state.iter())
        .for_each(|(input, &x)| {
            input.write(x);
        });

    LinearLayers::external_linear_layer(&mut state);

    for (full_round, constants) in perm
        .beginning_full_rounds
        .iter_mut()
        .zip(&constants.beginning_full_round_constants)
    {
        generate_full_round::<F, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }

    for (partial_round, constant) in perm
        .partial_rounds
        .iter_mut()
        .zip(&constants.partial_round_constants)
    {
        generate_partial_round::<F, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            partial_round,
            *constant,
        );
    }

    for (full_round, constants) in perm
        .ending_full_rounds
        .iter_mut()
        .zip(&constants.ending_full_round_constants)
    {
        generate_full_round::<F, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }
}

#[inline]
fn generate_full_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
) {
    for (state_i, const_i) in state.iter_mut().zip(round_constants) {
        *state_i += *const_i;
    }
    for (state_i, sbox_i) in state.iter_mut().zip(full_round.sbox.iter_mut()) {
        generate_sbox(sbox_i, state_i);
    }
    LinearLayers::external_linear_layer(state);
    full_round
        .post
        .iter_mut()
        .zip(*state)
        .for_each(|(post, x)| {
            post.write(x);
        });
}

#[inline]
fn generate_partial_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: F,
) {
    state[0] += round_constant;
    generate_sbox(&mut partial_round.sbox, &mut state[0]);
    partial_round.post_sbox.write(state[0]);
    LinearLayers::internal_linear_layer(state);
}

#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &mut SBox<MaybeUninit<F>, DEGREE, REGISTERS>,
    x: &mut F,
) {
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            sbox.0[0].write(x3);
            x3 * x2
        }
        (7, 1) => {
            let x3 = x.cube();
            sbox.0[0].write(x3);
            x3 * x3 * *x
        }
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            let x9 = x3.cube();
            sbox.0[0].write(x3);
            sbox.0[1].write(x9);
            x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}

pub trait IntoParallelIterator {
    type Iter: Iterator<Item = Self::Item>;
    type Item: Send;

    fn into_par_iter(self) -> Self::Iter;
}
impl<T: IntoIterator> IntoParallelIterator for T
where
    T::Item: Send,
{
    type Iter = T::IntoIter;
    type Item = T::Item;

    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

pub trait IntoParallelRefIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: Send + 'data;

    fn par_iter(&'data self) -> Self::Iter;
}

impl<'data, I: 'data + ?Sized> IntoParallelRefIterator<'data> for I
where
    &'data I: IntoParallelIterator,
{
    type Iter = <&'data I as IntoParallelIterator>::Iter;
    type Item = <&'data I as IntoParallelIterator>::Item;

    fn par_iter(&'data self) -> Self::Iter {
        self.into_par_iter()
    }
}

pub trait IntoParallelRefMutIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: Send + 'data;

    fn par_iter_mut(&'data mut self) -> Self::Iter;
}

impl<'data, I: 'data + ?Sized> IntoParallelRefMutIterator<'data> for I
where
    &'data mut I: IntoParallelIterator,
{
    type Iter = <&'data mut I as IntoParallelIterator>::Iter;
    type Item = <&'data mut I as IntoParallelIterator>::Item;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.into_par_iter()
    }
}
