use std::iter;

use openvm_stark_backend::interaction::{Interaction, InteractionBuilder};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
//use sp1_stark::air::{AirInteraction, InteractionScope, MessageBuilder};
use tracing::instrument;

// Mostly copied from Plonky3!
use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::{Entry, SymbolicVariable};

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree =
        get_max_constraint_degree(air, preprocessed_width, num_public_values).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_symbolic_constraints(air, preprocessed_width, num_public_values)
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(preprocessed_width, air.width(), num_public_values);
    air.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    pub constraints: Vec<SymbolicExpression<F>>,
    pub bus_sends: Vec<Interaction<SymbolicExpression<F>>>,
    pub bus_receives: Vec<Interaction<SymbolicExpression<F>>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub fn new(preprocessed_width: usize, width: usize, num_public_values: usize) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            constraints: vec![],
            bus_sends: vec![],
            bus_receives: vec![],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicAirBuilder<F> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> PairBuilder for SymbolicAirBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> InteractionBuilder for SymbolicAirBuilder<F> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: openvm_stark_backend::interaction::BusIndex,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        count_weight: u32,
    ) {
        todo!()
    }

    fn num_interactions(&self) -> usize {
        todo!()
    }

    fn all_interactions(&self) -> &[openvm_stark_backend::interaction::Interaction<Self::Expr>] {
        todo!()
    }
}

// impl<F: Field> MessageBuilder<AirInteraction<SymbolicExpression<F>>> for SymbolicAirBuilder<F> {
//     fn send(&mut self, message: AirInteraction<SymbolicExpression<F>>, _scope: InteractionScope) {
//         let interaction_kind =
//             SymbolicExpression::Constant(F::from_canonical_u64(message.kind as u64));
//         self.bus_sends
//             .push((interaction_kind, message.values, message.multiplicity));
//     }

//     fn receive(
//         &mut self,
//         message: AirInteraction<SymbolicExpression<F>>,
//         _scope: InteractionScope,
//     ) {
//         let interaction_kind =
//             SymbolicExpression::Constant(F::from_canonical_u64(message.kind as u64));
//         self.bus_receives
//             .push((interaction_kind, message.values, message.multiplicity));
//     }
// }
