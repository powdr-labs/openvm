use openvm_native_compiler::ir::Witness;
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2_root::{BabyBearPoseidon2RootConfig, BabyBearPoseidon2RootEngine},
        setup_tracing_with_log_level, FriParameters,
    },
    engine::StarkFriEngine,
    utils::ProofInputForTest,
};
use tracing::Level;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::Halo2Prover,
    stark::outer::build_circuit_verify_operations,
    tests::{fibonacci_test_proof_input, interaction_test_proof_input},
    witness::Witnessable,
};

#[test]
fn test_fibonacci() {
    run_recursive_test(fibonacci_test_proof_input::<BabyBearPoseidon2RootConfig>(
        16,
    ))
}

#[test]
fn test_interactions() {
    run_recursive_test(interaction_test_proof_input::<BabyBearPoseidon2RootConfig>())
}

fn run_recursive_test(mut test_proof_input: ProofInputForTest<BabyBearPoseidon2RootConfig>) {
    setup_tracing_with_log_level(Level::INFO);
    test_proof_input.sort_chips();
    let vparams = test_proof_input
        .run_test(&BabyBearPoseidon2RootEngine::new(
            FriParameters::standard_fast(),
        ))
        .unwrap();
    let advice = new_from_outer_multi_vk(&vparams.data.vk);
    let proof = vparams.data.proof;

    let mut witness = Witness::default();
    proof.write(&mut witness);
    let operations = build_circuit_verify_operations(advice, &vparams.fri_params, &proof);
    Halo2Prover::mock(20, operations, witness);
}
