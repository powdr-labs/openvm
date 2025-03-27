use openvm_columns::FlattenFields;
use openvm_columns_core::FlattenFieldsHelper;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, FieldAlgebra},
};

use super::{utils::range_check, OverflowInt};
use crate::SubAir;

#[derive(FlattenFields)]
pub struct CheckCarryToZeroCols<T> {
    pub carries: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct CheckCarryToZeroSubAir {
    // The number of bits for each limb (not overflowed). Example: 10.
    pub limb_bits: usize,

    pub range_checker_bus: usize,
    // The range checker decomp bits.
    pub decomp: usize,
}

// max_overflow_bits: number of bits needed to represent the limbs of an expr / OverflowInt.
// limb_bits: number of bits for each limb for a canonical representation, typically 8.
pub fn get_carry_max_abs_and_bits(max_overflow_bits: usize, limb_bits: usize) -> (usize, usize) {
    let carry_bits = max_overflow_bits - limb_bits;
    let carry_min_value_abs = 1 << carry_bits;
    let carry_abs_bits = carry_bits + 1;
    (carry_min_value_abs, carry_abs_bits)
}

impl CheckCarryToZeroSubAir {
    pub fn new(limb_bits: usize, range_checker_bus: usize, decomp: usize) -> Self {
        Self {
            limb_bits,
            range_checker_bus,
            decomp,
        }
    }

    pub fn columns<F: Field>(&self) -> Vec<String> {
        CheckCarryToZeroCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for CheckCarryToZeroSubAir {
    /// `(expr, cols, is_valid)`
    type AirContext<'a>
        = (
        OverflowInt<AB::Expr>,
        CheckCarryToZeroCols<AB::Var>,
        AB::Expr,
    )
    where
        AB::Var: 'a,
        AB::Expr: 'a,
        AB: 'a;

    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (expr, cols, is_valid): (
            OverflowInt<AB::Expr>,
            CheckCarryToZeroCols<AB::Var>,
            AB::Expr,
        ),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        assert_eq!(expr.limbs.len(), cols.carries.len());
        builder.assert_bool(is_valid.clone());
        let (carry_min_value_abs, carry_abs_bits) =
            get_carry_max_abs_and_bits(expr.max_overflow_bits, self.limb_bits);
        // 1. Constrain the limbs size of carries.
        for &carry in cols.carries.iter() {
            range_check(
                builder,
                self.range_checker_bus,
                self.decomp,
                carry_abs_bits,
                carry + AB::F::from_canonical_usize(carry_min_value_abs),
                is_valid.clone(),
            );
        }

        // 2. Constrain the carries and expr.
        let mut previous_carry = AB::Expr::ZERO;
        for (i, limb) in expr.limbs.iter().enumerate() {
            builder.assert_eq(
                limb.clone() + previous_carry.clone(),
                cols.carries[i] * AB::F::from_canonical_usize(1 << self.limb_bits),
            );
            previous_carry = cols.carries[i].into();
        }
        // The last (highest) carry should be zero.
        builder.assert_eq(previous_carry, AB::Expr::ZERO);
    }
}
