use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::Field,
    p3_matrix::Matrix,
    rap::{Air, BaseAir, BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::ProgramTester;
use crate::system::program::ProgramBus;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ProgramDummyAir {
    pub bus: ProgramBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for ProgramDummyAir {
    fn columns(&self) -> Vec<String> {
        todo!()
    }
}
impl<F: Field> PartitionedBaseAir<F> for ProgramDummyAir {}
impl<F: Field> BaseAir<F> for ProgramDummyAir {
    fn width(&self) -> usize {
        ProgramTester::<F>::width()
    }

    fn columns(&self) -> Vec<String> {
        todo!()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ProgramDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = local.iter().map(|x| (*x).into()).collect::<Vec<AB::Expr>>();
        builder.push_receive(
            self.bus.0,
            local[..local.len() - 1].iter().cloned(),
            local[local.len() - 1].clone(),
        );
    }
}
