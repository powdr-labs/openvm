use core::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::size_of,
};
use std::{string, vec};

use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_poseidon2::GenericPoseidon2LinearLayers;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use super::{
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE, POSEIDON2_WIDTH,
};
use crate::{BabyBearPoseidon2LinearLayers, Plonky3RoundConstants};

// Round constants for Poseidon2, in a format that's convenient for the AIR.
#[derive(Debug, Clone)]
pub struct RoundConstants<
    F: Field,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub(crate) partial_round_constants: [F; PARTIAL_ROUNDS],
    pub(crate) ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
}

impl<F: Field, const WIDTH: usize, const HALF_FULL_ROUNDS: usize, const PARTIAL_ROUNDS: usize>
    RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    pub fn new(
        beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        partial_round_constants: [F; PARTIAL_ROUNDS],
        ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    ) -> Self {
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
        }
    }

    pub fn from_rng<R: Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let beginning_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        let partial_round_constants = rng
            .sample_iter(Standard)
            .take(PARTIAL_ROUNDS)
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let ending_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
        }
    }
}

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct Poseidon2Air<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    _phantom: PhantomData<LinearLayers>,
}

impl<
        F: Field,
        LinearLayers,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    pub fn new(constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>) -> Self {
        Self {
            constants,
            _phantom: PhantomData,
        }
    }
}

impl<
        F: Field,
        LinearLayers: Sync,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAir<F>
    for Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }

    fn columns(&self) -> vec::Vec<string::String> {
        todo!()
    }
}

pub(crate) fn eval<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &Poseidon2Air<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    builder: &mut AB,
    local: &Poseidon2Cols<
        AB::Var,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
) {
    let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

    LinearLayers::external_linear_layer(&mut state);

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.constants.beginning_full_round_constants[round],
            builder,
        );
    }

    for round in 0..PARTIAL_ROUNDS {
        eval_partial_round::<AB, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.partial_rounds[round],
            &air.constants.partial_round_constants[round],
            builder,
        );
    }

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.ending_full_rounds[round],
            &air.constants.ending_full_round_constants[round],
            builder,
        );
    }
}

impl<
        AB: AirBuilder,
        LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Air<AB>
    for Poseidon2Air<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();

        eval::<
            AB,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(self, builder, local);
    }
}

#[inline]
fn eval_full_round<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::F; WIDTH],
    builder: &mut AB,
) {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s = s.clone() + *r;
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    LinearLayers::external_linear_layer(state);
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

#[inline]
fn eval_partial_round<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: &AB::F,
    builder: &mut AB,
) {
    state[0] = state[0].clone() + *round_constant;
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    builder.assert_eq(state[0].clone(), partial_round.post_sbox);
    state[0] = partial_round.post_sbox.into();

    LinearLayers::internal_linear_layer(state);
}

/// Evaluates the S-box over a degree-1 expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-box. The supported degrees are
/// `3`, `5`, `7`, and `11`.
#[inline]
fn eval_sbox<AB, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let committed_x3 = sbox.0[0].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            committed_x3 * x2
        }
        (7, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.clone(), x.cube());
            committed_x3.square() * x.clone()
        }
        (11, 2) => {
            let committed_x3 = sbox.0[0].into();
            let committed_x9 = sbox.0[1].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            builder.assert_eq(committed_x9.clone(), committed_x3.cube());
            committed_x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}

/// Columns for a Poseidon2 AIR which computes one permutation per row.
///
/// The columns of the STARK are divided into the three different round sections of the Poseidon2
/// Permutation: beginning full rounds, partial rounds, and ending full rounds. For the full
/// rounds we store an [`SBox`] columnset for each state variable, and for the partial rounds we
/// store only for the first state variable. Because the matrix multiplications are linear
/// functions, we need only keep auxiliary columns for the S-box computations.
#[repr(C)]
pub struct Poseidon2Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub export: T,

    pub inputs: [T; WIDTH],

    /// Beginning Full Rounds
    pub beginning_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],

    /// Partial Rounds
    pub partial_rounds: [PartialRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; PARTIAL_ROUNDS],

    /// Ending Full Rounds
    pub ending_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],
}

/// Full round columns.
#[repr(C)]
pub struct FullRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize> {
    /// Possible intermediate results within each S-box.
    pub sbox: [SBox<T, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH],
    /// The post-state, i.e. the entire layer after this full round.
    pub post: [T; WIDTH],
}

/// Partial round columns.
#[repr(C)]
pub struct PartialRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize>
{
    /// Possible intermediate results within the S-box.
    pub sbox: SBox<T, SBOX_DEGREE, SBOX_REGISTERS>,
    /// The output of the S-box.
    pub post_sbox: T,
}

/// Possible intermediate results within an S-box.
///
/// Use this column-set for an S-box that can be computed with `REGISTERS`-many intermediate results
/// (not counting the final output). The S-box is checked to ensure that `REGISTERS` is the optimal
/// number of registers for the given `DEGREE` for the degrees given in the Poseidon2 paper:
/// `3`, `5`, `7`, and `11`. See `eval_sbox` for more information.
#[repr(C)]
pub struct SBox<T, const DEGREE: u64, const REGISTERS: usize>(pub [T; REGISTERS]);

pub const fn num_cols<
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> usize {
    size_of::<Poseidon2Cols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>(
    )
}

pub const fn make_col_map<
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> Poseidon2Cols<usize, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> {
    todo!()
    // let indices_arr = indices_arr::<
    //     { num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>() },
    // >();
    // unsafe {
    //     transmute::<
    //         [usize;
    //             num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()],
    //         Poseidon2Cols<
    //             usize,
    //             WIDTH,
    //             SBOX_DEGREE,
    //             SBOX_REGISTERS,
    //             HALF_FULL_ROUNDS,
    //             PARTIAL_ROUNDS,
    //         >,
    //     >(indices_arr)
    // }
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Borrow<Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>
    for [T]
{
    fn borrow(
        &self,
    ) -> &Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<Poseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    BorrowMut<
        Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<Poseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

pub type Poseidon2SubCols<F, const SBOX_REGISTERS: usize> = Poseidon2Cols<
    F,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
>;

pub type Plonky3Poseidon2Air<F, LinearLayers, const SBOX_REGISTERS: usize> = Poseidon2Air<
    F,
    LinearLayers,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
>;

#[derive(Debug)]
pub enum Poseidon2SubAir<F: Field, const SBOX_REGISTERS: usize> {
    BabyBearMds(Plonky3Poseidon2Air<F, BabyBearPoseidon2LinearLayers, SBOX_REGISTERS>),
}

impl<F: Field, const SBOX_REGISTERS: usize> Poseidon2SubAir<F, SBOX_REGISTERS> {
    pub fn new(constants: Plonky3RoundConstants<F>) -> Self {
        Self::BabyBearMds(Plonky3Poseidon2Air::new(constants))
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2SubAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        match self {
            Self::BabyBearMds(air) => air.width(),
        }
    }

    fn columns(&self) -> Vec<String> {
        unimplemented!()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2SubAir<F, SBOX_REGISTERS>
{
    fn columns(&self) -> Vec<String> {
        // TODO: fix this
        unimplemented!()
        //match self {
        //    Self::BabyBearMds(air) => air.columns(),
        //}
    }
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2SubAir<F, SBOX_REGISTERS>
{
}

impl<AB: AirBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2SubAir<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::BabyBearMds(air) => air.eval(builder),
        }
    }
}

impl<
        F,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > FlattenFieldsHelper
    for Poseidon2Cols<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn flatten_fields() -> Option<Vec<String>> {
        let mut fields = Vec::new();

        // Generate field names using exact parameter values
        fields.push("export".to_string());

        // Use actual parameters for array sizes
        for i in 0..WIDTH {
            fields.push(format!("inputs__{}", i));
        }

        // Other fields with their array sizes
        for i in 0..HALF_FULL_ROUNDS {
            fields.push(format!("beginning_full_rounds__{}", i));
        }

        for i in 0..PARTIAL_ROUNDS {
            fields.push(format!("partial_rounds__{}", i));
        }

        for i in 0..HALF_FULL_ROUNDS {
            fields.push(format!("ending_full_rounds__{}", i));
        }

        Some(fields)
    }
}
