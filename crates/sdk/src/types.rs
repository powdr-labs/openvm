use std::{
    io::Cursor,
    iter::{once, repeat},
};

use eyre::Result;
use itertools::Itertools;
use openvm_continuations::{verifier::internal::types::VmStarkProof, SC};
use openvm_native_recursion::halo2::{wrapper::EvmVerifierByteCode, Fr, RawEvmProof};
use openvm_stark_backend::proof::Proof;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use thiserror::Error;

use crate::{
    codec::{decode_vec, encode_slice, Decode, Encode},
    commit::{AppExecutionCommit, CommitBytes},
};

/// Number of bytes in a Bn254Fr.
pub(crate) const BN254_BYTES: usize = 32;
/// Number of Bn254Fr in `accumulator` field.
pub const NUM_BN254_ACCUMULATOR: usize = 12;
/// Number of Bn254Fr in `proof` field for a circuit with only 1 advice column.
const NUM_BN254_PROOF: usize = 43;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmHalo2Verifier {
    pub halo2_verifier_code: String,
    pub openvm_verifier_code: String,
    pub openvm_verifier_interface: String,
    pub artifact: EvmVerifierByteCode,
}

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProofData {
    #[serde_as(as = "serde_with::hex::Hex")]
    /// KZG accumulator.
    pub accumulator: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// Bn254Fr proof in little-endian bytes. The circuit only has 1 advice column, so the proof is
    /// of length `NUM_BN254_PROOF * BN254_BYTES`.
    pub proof: Vec<u8>,
}

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvmProof {
    #[serde(flatten)]
    /// Bn254Fr public value app commits.
    pub app_commit: AppExecutionCommit,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// User public values packed into bytes.
    pub user_public_values: Vec<u8>,
    /// The concatenation of `accumulator` and `proof`.
    pub proof_data: ProofData,
}

#[derive(Debug, Error)]
pub enum EvmProofConversionError {
    #[error("Invalid length of proof")]
    InvalidLengthProof,
    #[error("Invalid length of instances")]
    InvalidLengthInstances,
    #[error("Invalid length of user public values")]
    InvalidUserPublicValuesLength,
    #[error("Invalid length of accumulator")]
    InvalidLengthAccumulator,
}

impl EvmProof {
    #[cfg(feature = "evm-verify")]
    /// Return bytes calldata to be passed to the verifier contract.
    pub fn verifier_calldata(self) -> Vec<u8> {
        use alloy_sol_types::SolCall;

        use crate::IOpenVmHalo2Verifier;

        let EvmProof {
            user_public_values,
            app_commit,
            proof_data,
        } = self;

        let ProofData { accumulator, proof } = proof_data;

        let mut proof_data = accumulator;
        proof_data.extend(proof);

        IOpenVmHalo2Verifier::verifyCall {
            publicValues: user_public_values.into(),
            proofData: proof_data.into(),
            appExeCommit: app_commit.app_exe_commit.as_slice().into(),
            appVmCommit: app_commit.app_vm_commit.as_slice().into(),
        }
        .abi_encode()
    }

    #[cfg(feature = "evm-verify")]
    pub fn fallback_calldata(&self) -> Vec<u8> {
        let evm_proof: RawEvmProof = self.clone().try_into().unwrap();
        evm_proof.verifier_calldata()
    }
}

impl TryFrom<RawEvmProof> for EvmProof {
    type Error = EvmProofConversionError;

    fn try_from(evm_proof: RawEvmProof) -> Result<Self, Self::Error> {
        let RawEvmProof { instances, proof } = evm_proof;
        if NUM_BN254_ACCUMULATOR + 2 >= instances.len() {
            return Err(EvmProofConversionError::InvalidLengthInstances);
        }
        if proof.len() != NUM_BN254_PROOF * BN254_BYTES {
            return Err(EvmProofConversionError::InvalidLengthProof);
        }
        let accumulator = instances[0..NUM_BN254_ACCUMULATOR]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>();
        let mut app_exe_commit = instances[NUM_BN254_ACCUMULATOR].to_bytes();
        let mut app_vm_commit = instances[NUM_BN254_ACCUMULATOR + 1].to_bytes();
        app_exe_commit.reverse();
        app_vm_commit.reverse();

        let mut evm_accumulator: Vec<u8> = Vec::with_capacity(accumulator.len());
        accumulator
            .chunks(32)
            .for_each(|chunk| evm_accumulator.extend(chunk.iter().rev().cloned()));

        let user_public_values = instances[NUM_BN254_ACCUMULATOR + 2..].iter().fold(
            Vec::<u8>::new(),
            |mut acc: Vec<u8>, chunk| {
                // We only care about the first byte, everything else should be 0-bytes
                acc.push(*chunk.to_bytes().first().unwrap());
                acc
            },
        );
        let app_commit = AppExecutionCommit {
            app_exe_commit: CommitBytes::new(app_exe_commit),
            app_vm_commit: CommitBytes::new(app_vm_commit),
        };

        Ok(Self {
            app_commit,
            user_public_values,
            proof_data: ProofData {
                accumulator: evm_accumulator,
                proof,
            },
        })
    }
}

impl TryFrom<EvmProof> for RawEvmProof {
    type Error = EvmProofConversionError;
    fn try_from(evm_openvm_proof: EvmProof) -> Result<Self, Self::Error> {
        let EvmProof {
            mut app_commit,
            user_public_values,
            proof_data,
        } = evm_openvm_proof;

        app_commit.app_exe_commit.reverse();
        app_commit.app_vm_commit.reverse();

        let ProofData { accumulator, proof } = proof_data;

        if proof.len() != NUM_BN254_PROOF * BN254_BYTES {
            return Err(EvmProofConversionError::InvalidLengthProof);
        }
        let instances = {
            if accumulator.len() != NUM_BN254_ACCUMULATOR * BN254_BYTES {
                return Err(EvmProofConversionError::InvalidLengthAccumulator);
            }

            let mut reversed_accumulator: Vec<u8> = Vec::with_capacity(accumulator.len());
            accumulator
                .chunks(32)
                .for_each(|chunk| reversed_accumulator.extend(chunk.iter().rev().cloned()));

            if user_public_values.is_empty() {
                return Err(EvmProofConversionError::InvalidUserPublicValuesLength);
            }

            let user_public_values = user_public_values
                .into_iter()
                .flat_map(|byte| once(byte).chain(repeat(0).take(31)))
                .collect::<Vec<_>>();

            let mut ret = Vec::new();
            for chunk in &reversed_accumulator
                .iter()
                .chain(app_commit.app_exe_commit.as_slice())
                .chain(app_commit.app_vm_commit.as_slice())
                .chain(&user_public_values)
                .chunks(BN254_BYTES)
            {
                let c = chunk.copied().collect::<Vec<_>>().try_into().unwrap();
                ret.push(Fr::from_bytes(&c).unwrap());
            }
            ret
        };
        Ok(RawEvmProof { instances, proof })
    }
}

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VmStarkProofBytes {
    #[serde(flatten)]
    pub app_commit: AppExecutionCommit,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub user_public_values: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub proof: Vec<u8>,
}

impl VmStarkProofBytes {
    pub fn new(app_commit: AppExecutionCommit, proof: VmStarkProof<SC>) -> Result<Self> {
        let mut user_public_values = Vec::new();
        encode_slice(&proof.user_public_values, &mut user_public_values)?;
        Ok(Self {
            app_commit,
            user_public_values,
            proof: proof.proof.encode_to_vec()?,
        })
    }
}

impl TryFrom<VmStarkProofBytes> for VmStarkProof<SC> {
    type Error = std::io::Error;
    fn try_from(proof: VmStarkProofBytes) -> Result<Self, std::io::Error> {
        let VmStarkProofBytes {
            proof,
            user_public_values,
            ..
        } = proof;
        let mut reader = Cursor::new(user_public_values);
        let user_public_values = decode_vec(&mut reader)?;
        Ok(Self {
            user_public_values,
            proof: Proof::decode_from_bytes(&proof)?,
        })
    }
}
