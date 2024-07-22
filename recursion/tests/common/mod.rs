use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use afs_compiler::util::execute_program;
use afs_recursion::hints::{Hintable, InnerVal};
use afs_recursion::stark::{AxiomVerifier, DynRapForRecursion};
use afs_recursion::types::{AxiomMemoryLayout, InnerConfig, MultiStarkVerificationAdvice};
use afs_stark_backend::keygen::types::MultiStarkPartialVerifyingKey;
use afs_stark_backend::prover::trace::TraceCommitmentBuilder;
use afs_stark_backend::prover::types::Proof;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::MultiTraceStarkVerifier;
use afs_test_utils::config::baby_bear_poseidon2::{
    default_engine, BabyBearPoseidon2Config, BabyBearPoseidon2Engine,
};
use afs_test_utils::engine::StarkEngine;

pub fn run_recursive_test(
    // TODO: find way to de-duplicate parameters
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) {
    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();
    let log_degree = log2_strict_usize(trace_heights.clone().into_iter().max().unwrap());

    let engine = default_engine(log_degree);

    let mut keygen_builder = engine.keygen_builder();
    for (&rap, &num_pv) in any_raps.iter().zip(num_pvs.iter()) {
        keygen_builder.add_air(rap, num_pv);
    }

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&partial_vk, any_raps.clone());

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pvs);

    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(
            &mut engine.new_challenger(),
            &partial_vk,
            any_raps,
            &proof,
            &pvs,
        )
        .expect("afs proof should verify");

    run_verification_program(
        rec_raps,
        pvs,
        &engine,
        &partial_vk,
        proof,
    );
}

pub fn run_verification_program(
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    pvs: Vec<Vec<InnerVal>>,
    engine: &BabyBearPoseidon2Engine,
    partial_vk: &MultiStarkPartialVerifyingKey<BabyBearPoseidon2Config>,
    proof: Proof<BabyBearPoseidon2Config>,
) {
    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();

    let advice = MultiStarkVerificationAdvice::new_from_multi_vk(&partial_vk);

    let program = AxiomVerifier::build(rec_raps, advice, &engine.fri_params);

    let input = AxiomMemoryLayout {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness_stream = Vec::new();
    witness_stream.extend(input.write());

    execute_program::<1, _>(program, witness_stream);
}
