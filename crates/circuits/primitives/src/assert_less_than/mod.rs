use derive_new::new;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, FieldAlgebra},
    rap::AirBuilder,
};

use crate::{
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

/// The IO is typically provided with `T = AB::Expr` as external context.
// This does not derive AlignedBorrow because it is usually **not** going to be
// direct columns in an AIR.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AssertLessThanIo<T> {
    pub x: T,
    pub y: T,
    /// Will only apply constraints when `count != 0`.
    /// Range checks are done with multiplicity `count`.
    /// If `count == 0` then no range checks are done.
    /// In practice `count` is always boolean, although this is not enforced
    /// by the subair.
    ///
    /// N.B.: in fact range checks could always be done, if the aux
    /// subrow values are set to 0 when `count == 0`. This would slightly
    /// simplify the range check interactions, although usually doesn't change
    /// the overall constraint degree. It however leads to the annoyance that
    /// you must update the RangeChecker's multiplicities even on dummy padding
    /// rows. To improve quality of life,
    /// we currently use this more complex constraint.
    pub count: T,
}
impl<T> AssertLessThanIo<T> {
    pub fn new(x: impl Into<T>, y: impl Into<T>, count: impl Into<T>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            count: count.into(),
        }
    }
}

/// These columns are owned by the SubAir. Typically used with `T = AB::Var`.
/// `AUX_LEN` is the number of AUX columns
/// we have that AUX_LEN = max_bits.div_ceil(bus.range_max_bits)
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, new, FlattenFields)]
pub struct LessThanAuxCols<T, const AUX_LEN: usize> {
    // lower_decomp consists of lower decomposed into limbs of size bus.range_max_bits
    // note: the final limb might have less than bus.range_max_bits bits
    pub lower_decomp: [T; AUX_LEN],
}

/// This is intended for use as a **SubAir**, not as a standalone Air.
///
/// This SubAir constrains that `x < y` when `count != 0`, assuming
/// the two numbers both have a max number of bits, given by `max_bits`.
/// The SubAir decomposes `y - x - 1` into limbs of
/// size `bus.range_max_bits`, and interacts with a
/// `VariableRangeCheckerBus` to range check the decompositions.
///
/// The SubAir will own auxiliary columns to store the decomposed limbs.
/// The number of limbs is `max_bits.div_ceil(bus.range_max_bits)`.
///
/// The expected max constraint degree of `eval` is
///     deg(count) + max(1, deg(x), deg(y))
#[derive(Copy, Clone, Debug)]
pub struct AssertLtSubAir {
    /// The bus for sends to range chip
    pub bus: VariableRangeCheckerBus,
    /// The maximum number of bits for the numbers to compare
    /// Soundness requirement: max_bits <= 29
    ///     max_bits > 29 doesn't work: the approach is to check that y-x-1 is non-negative.
    ///     For a field with prime modular, this is equivalent to checking that y-x-1 is in
    ///     the range [0, 2^max_bits - 1]. However, for max_bits > 29, if y is small enough
    ///     and x is large enough, then y-x-1 is negative but can still be in the range due
    ///     to the field size not being big enough.
    pub max_bits: usize,
    /// `decomp_limbs = max_bits.div_ceil(bus.range_max_bits)` is the
    /// number of limbs that `y - x - 1` will be decomposed into.
    pub decomp_limbs: usize,
}

impl AssertLtSubAir {
    pub fn new(bus: VariableRangeCheckerBus, max_bits: usize) -> Self {
        let decomp_limbs = max_bits.div_ceil(bus.range_max_bits);
        Self {
            bus,
            max_bits,
            decomp_limbs,
        }
    }

    pub fn when_transition(self) -> AssertLtWhenTransitionAir {
        AssertLtWhenTransitionAir(self)
    }

    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }

    /// FOR INTERNAL USE ONLY.
    /// This AIR is only sound if interactions are enabled
    ///
    /// Constraints between `io` and `aux` are only enforced when `count != 0`.
    /// This means `aux` can be all zero independent on what `io` is by setting `count = 0`.
    #[inline(always)]
    fn eval_without_range_checks<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: AssertLessThanIo<AB::Expr>,
        lower_decomp: &[AB::Var],
    ) {
        assert_eq!(lower_decomp.len(), self.decomp_limbs);
        // this is the desired intermediate value (i.e. y - x - 1)
        // deg(intermed_val) = deg(io)
        let intermed_val = io.y - io.x - AB::Expr::ONE;

        // Construct lower from lower_decomp:
        // - each limb of lower_decomp will be range checked
        // deg(lower) = 1
        let lower = lower_decomp
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_usize(1 << (i * self.range_max_bits()))
            });

        // constrain that y - x - 1 is equal to the constructed lower value.
        // this enforces that the intermediate value is in the range [0, 2^max_bits - 1], which is equivalent to x < y
        builder.when(io.count).assert_eq(intermed_val, lower);
        // the degree of this constraint is expected to be deg(count) + max(deg(intermed_val), deg(lower))
        // since we are constraining count * intermed_val == count * lower
    }

    #[inline(always)]
    fn eval_range_checks<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        lower_decomp: &[AB::Var],
        count: impl Into<AB::Expr>,
    ) {
        let count = count.into();
        let mut bits_remaining = self.max_bits;
        // we range check the limbs of the lower_decomp so that we know each element
        // of lower_decomp has the correct number of bits
        for limb in lower_decomp {
            // the last limb might have fewer than `bus.range_max_bits` bits
            let range_bits = bits_remaining.min(self.range_max_bits());
            self.bus
                .range_check(*limb, range_bits)
                .eval(builder, count.clone());
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }
    }

    pub fn columns<F: Field, const AUX_LEN: usize>(&self) -> Vec<String> {
        LessThanAuxCols::<F, AUX_LEN>::flatten_fields().unwrap()
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for AssertLtSubAir {
    type AirContext<'a>
        = (AssertLessThanIo<AB::Expr>, &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    // constrain that x < y
    // warning: send for range check must be included for the constraints to be sound
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, lower_decomp): (AssertLessThanIo<AB::Expr>, &'a [AB::Var]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        // Note: every AIR that uses this sub-AIR must include the range checks for soundness
        self.eval_range_checks(builder, lower_decomp, io.count.clone());
        self.eval_without_range_checks(builder, io, lower_decomp);
    }
}

/// The same subair as [AssertLtSubAir] except that non-range check
/// constraints are not imposed on the last row.
/// Intended use case is for asserting less than between entries in
/// adjacent rows.
#[derive(Clone, Copy, Debug)]
pub struct AssertLtWhenTransitionAir(pub AssertLtSubAir);

impl<AB: InteractionBuilder> SubAir<AB> for AssertLtWhenTransitionAir {
    type AirContext<'a>
        = (AssertLessThanIo<AB::Expr>, &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    /// Imposes the non-interaction constraints on all except the last row. This is
    /// intended for use when the comparators `x, y` are on adjacent rows.
    ///
    /// This function does also enable the interaction constraints _on every row_.
    /// The `eval_interactions` performs range checks on `lower_decomp` on every row, even
    /// though in this AIR the lower_decomp is not used on the last row.
    /// This simply means the trace generation must fill in the last row with numbers in
    /// range (e.g., with zeros)
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, lower_decomp): (AssertLessThanIo<AB::Expr>, &'a [AB::Var]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        self.0
            .eval_range_checks(builder, lower_decomp, io.count.clone());
        self.0
            .eval_without_range_checks(&mut builder.when_transition(), io, lower_decomp);
    }
}

impl<F: Field> TraceSubRowGenerator<F> for AssertLtSubAir {
    /// (range_checker, x, y)
    // x, y are u32 because memory records are storing u32 and there would be needless conversions. It also prevents a F: PrimeField32 trait bound.
    type TraceContext<'a> = (&'a VariableRangeCheckerChip, u32, u32);
    /// lower_decomp
    type ColsMut<'a> = &'a mut [F];

    /// Should only be used when `io.count != 0`.
    #[inline(always)]
    fn generate_subrow<'a>(
        &'a self,
        (range_checker, x, y): (&'a VariableRangeCheckerChip, u32, u32),
        lower_decomp: &'a mut [F],
    ) {
        debug_assert!(x < y, "assert {x} < {y} failed");
        debug_assert_eq!(lower_decomp.len(), self.decomp_limbs);
        debug_assert!(
            x < (1 << self.max_bits),
            "{x} has more than {} bits",
            self.max_bits
        );
        debug_assert!(
            y < (1 << self.max_bits),
            "{y} has more than {} bits",
            self.max_bits
        );

        // Note: if x < y then y - x - 1 should already have <= max_bits bits
        range_checker.decompose(y - x - 1, self.max_bits, lower_decomp);
    }
}
