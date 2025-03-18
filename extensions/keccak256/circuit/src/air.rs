use std::{array::from_fn, borrow::Borrow, iter::zip, string};

use itertools::{izip, Itertools};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    utils::{assert_array_eq, not, select},
};
use openvm_columns_core::FlattenFieldsHelper;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_rv32im_circuit::adapters::abstract_compose;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    p3_air::utils::{andn, xor, xor3},
    p3_field::{FieldAlgebra, PrimeField64},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_keccak_air::{
    generate_trace_rows, KeccakCols, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS, U64_LIMBS,
};
use rand::random;

use super::{
    columns::{KeccakVmCols, NUM_KECCAK_VM_COLS},
    KECCAK_ABSORB_READS, KECCAK_DIGEST_BYTES, KECCAK_DIGEST_WRITES, KECCAK_RATE_BYTES,
    KECCAK_RATE_U16S, KECCAK_REGISTER_READS, KECCAK_WIDTH_U16S, KECCAK_WORD_SIZE,
    NUM_ABSORB_ROUNDS,
};
// ________________________________________________
#[derive(Debug)]
pub struct KeccakAir {}
const BITS_PER_LIMB: usize = 16;

impl KeccakAir {
    pub fn generate_trace_rows<F: PrimeField64>(&self, num_hashes: usize) -> RowMajorMatrix<F> {
        let inputs = (0..num_hashes).map(|_| random()).collect::<Vec<_>>();
        generate_trace_rows(inputs)
    }
}

impl<F> BaseAir<F> for KeccakAir {
    fn width(&self) -> usize {
        NUM_KECCAK_PERM_COLS
    }

    fn columns(&self) -> Vec<string::String> {
        todo!()
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        eval_round_flags(builder);

        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &KeccakCols<AB::Var> = (*local).borrow();
        let next: &KeccakCols<AB::Var> = (*next).borrow();

        let first_step = local.step_flags[0];
        let final_step = local.step_flags[NUM_ROUNDS - 1];
        let not_final_step = AB::Expr::ONE - final_step;

        // If this is the first step, the input A must match the preimage.
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    builder
                        .when(first_step)
                        .assert_eq(local.preimage[y][x][limb], local.a[y][x][limb]);
                }
            }
        }

        // The export flag must be 0 or 1.
        builder.assert_bool(local.export);

        // If this is not the final step, the export flag must be off.
        builder
            .when(not_final_step.clone())
            .assert_zero(local.export);

        // If this is not the final step, the local and next preimages must match.
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    builder
                        .when(not_final_step.clone())
                        .when_transition()
                        .assert_eq(local.preimage[y][x][limb], next.preimage[y][x][limb]);
                }
            }
        }

        // C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
        for x in 0..5 {
            for z in 0..64 {
                builder.assert_bool(local.c[x][z]);
                let xor = xor3::<AB::Expr>(
                    local.c[x][z].into(),
                    local.c[(x + 4) % 5][z].into(),
                    local.c[(x + 1) % 5][(z + 63) % 64].into(),
                );
                let c_prime = local.c_prime[x][z];
                builder.assert_eq(c_prime, xor);
            }
        }

        // Check that the input limbs are consistent with A' and D.
        // A[x, y, z] = xor(A'[x, y, z], D[x, y, z])
        //            = xor(A'[x, y, z], C[x - 1, z], C[x + 1, z - 1])
        //            = xor(A'[x, y, z], C[x, z], C'[x, z]).
        // The last step is valid based on the identity we checked above.
        // It isn't required, but makes this check a bit cleaner.
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z| {
                    let a_prime: AB::Var = local.a_prime[y][x][z];
                    let c: AB::Var = local.c[x][z];
                    let c_prime: AB::Var = local.c_prime[x][z];
                    xor3::<AB::Expr>(a_prime.into(), c.into(), c_prime.into())
                };

                for limb in 0..U64_LIMBS {
                    let a_limb = local.a[y][x][limb];
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::ZERO, |acc, z| {
                            builder.assert_bool(local.a_prime[y][x][z]);
                            acc.double() + get_bit(z)
                        });
                    builder.assert_eq(computed_limb, a_limb);
                }
            }
        }

        // xor_{i=0}^4 A'[x, i, z] = C'[x, z], so for each x, z,
        // diff * (diff - 2) * (diff - 4) = 0, where
        // diff = sum_{i=0}^4 A'[x, i, z] - C'[x, z]
        for x in 0..5 {
            for z in 0..64 {
                let sum: AB::Expr = (0..5).map(|y| local.a_prime[y][x][z].into()).sum();
                let diff = sum - local.c_prime[x][z];
                let four = AB::Expr::from_canonical_u8(4);
                builder.assert_zero(diff.clone() * (diff.clone() - AB::Expr::TWO) * (diff - four));
            }
        }

        // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z| {
                    let andn = andn::<AB::Expr>(
                        local.b((x + 1) % 5, y, z).into(),
                        local.b((x + 2) % 5, y, z).into(),
                    );
                    xor::<AB::Expr>(local.b(x, y, z).into(), andn)
                };

                for limb in 0..U64_LIMBS {
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::ZERO, |acc, z| acc.double() + get_bit(z));
                    builder.assert_eq(computed_limb, local.a_prime_prime[y][x][limb]);
                }
            }
        }

        // A'''[0, 0] = A''[0, 0] XOR RC
        for limb in 0..U64_LIMBS {
            let computed_a_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::ZERO, |acc, z| {
                    builder.assert_bool(local.a_prime_prime_0_0_bits[z]);
                    acc.double() + local.a_prime_prime_0_0_bits[z]
                });
            let a_prime_prime_0_0_limb = local.a_prime_prime[0][0][limb];
            builder.assert_eq(computed_a_prime_prime_0_0_limb, a_prime_prime_0_0_limb);
        }

        let get_xored_bit = |i| {
            let mut rc_bit_i = AB::Expr::ZERO;
            for r in 0..NUM_ROUNDS {
                let this_round = local.step_flags[r];
                let this_round_constant = AB::Expr::from_canonical_u8(rc_value_bit(r, i));
                rc_bit_i += this_round * this_round_constant;
            }

            xor::<AB::Expr>(local.a_prime_prime_0_0_bits[i].into(), rc_bit_i)
        };

        for limb in 0..U64_LIMBS {
            let a_prime_prime_prime_0_0_limb = local.a_prime_prime_prime_0_0_limbs[limb];
            let computed_a_prime_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::ZERO, |acc, z| acc.double() + get_xored_bit(z));
            builder.assert_eq(
                computed_a_prime_prime_prime_0_0_limb,
                a_prime_prime_prime_0_0_limb,
            );
        }

        // Enforce that this round's output equals the next round's input.
        for x in 0..5 {
            for y in 0..5 {
                for limb in 0..U64_LIMBS {
                    let output = local.a_prime_prime_prime(y, x, limb);
                    let input = next.a[y][x][limb];
                    builder
                        .when_transition()
                        .when(not_final_step.clone())
                        .assert_eq(output, input);
                }
            }
        }
    }
}

#[inline]
pub(crate) fn eval_round_flags<AB: AirBuilder>(builder: &mut AB) {
    let main = builder.main();
    let (local, next) = (main.row_slice(0), main.row_slice(1));
    let local: &KeccakCols<AB::Var> = (*local).borrow();
    let next: &KeccakCols<AB::Var> = (*next).borrow();

    // Initially, the first step flag should be 1 while the others should be 0.
    builder.when_first_row().assert_one(local.step_flags[0]);
    for i in 1..NUM_ROUNDS {
        builder.when_first_row().assert_zero(local.step_flags[i]);
    }

    for i in 0..NUM_ROUNDS {
        let current_round_flag = local.step_flags[i];
        let next_round_flag = next.step_flags[(i + 1) % NUM_ROUNDS];
        builder
            .when_transition()
            .assert_eq(next_round_flag, current_round_flag);
    }
}

pub const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

const RC_BITS: [[u8; 64]; 24] = [
    [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
    [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    [
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1,
    ],
];

pub(crate) const fn rc_value_bit(round: usize, bit_index: usize) -> u8 {
    RC_BITS[round][bit_index]
}

// ________________________________________________

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send 8-bit XOR requests to.
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Maximum number of bits allowed for an address pointer
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakVmAir {
    fn columns(&self) -> Vec<String> {
        KeccakVmCols::<F>::flatten_fields().unwrap()
    }
}
impl<F> PartitionedBaseAir<F> for KeccakVmAir {}
impl<F> BaseAir<F> for KeccakVmAir {
    fn width(&self) -> usize {
        NUM_KECCAK_VM_COLS
    }

    fn columns(&self) -> Vec<String> {
        KeccakVmCols::<F>::flatten_fields().unwrap()
    }
}

impl KeccakVmAir {
    pub fn columns<F>(&self) -> Vec<String> {
        KeccakVmCols::<F>::flatten_fields().unwrap()
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &KeccakVmCols<AB::Var> = (*local).borrow();
        let next: &KeccakVmCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.sponge.is_new_start);
        builder.assert_eq(
            local.sponge.is_new_start,
            local.sponge.is_new_start * local.is_first_round(),
        );
        builder.assert_eq(
            local.instruction.is_enabled_first_round,
            local.instruction.is_enabled * local.is_first_round(),
        );
        // Not strictly necessary:
        builder
            .when_first_row()
            .assert_one(local.sponge.is_new_start);

        self.eval_keccak_f(builder);
        self.constrain_padding(builder, local, next);
        self.constrain_consistency_across_rounds(builder, local, next);

        let mem = &local.mem_oc;
        // Interactions:
        self.constrain_absorb(builder, local, next);
        let start_read_timestamp = self.eval_instruction(builder, local, &mem.register_aux);
        let start_write_timestamp =
            self.constrain_input_read(builder, local, start_read_timestamp, &mem.absorb_reads);
        self.constrain_output_write(
            builder,
            local,
            start_write_timestamp.clone(),
            &mem.digest_writes,
        );

        self.constrain_block_transition(builder, local, next, start_write_timestamp);
    }
}

impl KeccakVmAir {
    /// Evaluate the keccak-f permutation constraints.
    ///
    /// WARNING: The keccak-f AIR columns **must** be the first columns in the main AIR.
    #[inline]
    pub fn eval_keccak_f<AB: AirBuilder>(&self, builder: &mut AB) {
        let keccak_f_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_PERM_COLS);
        keccak_f_air.eval(&mut sub_builder);
    }

    /// Many columns are expected to be the same between rounds and only change per-block.
    pub fn constrain_consistency_across_rounds<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let mut transition_builder = builder.when_transition();
        let mut round_builder = transition_builder.when(not(local.is_last_round()));
        // Instruction columns
        local
            .instruction
            .assert_eq(&mut round_builder, next.instruction);
    }

    pub fn constrain_block_transition<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
    ) {
        // When we transition between blocks, if the next block isn't a new block
        // (this means it's not receiving a new opcode or starting a dummy block)
        // then we want _parts_ of opcode instruction to stay the same
        // between blocks.
        let mut block_transition = builder.when(local.is_last_round() * not(next.is_new_start()));
        block_transition.assert_eq(local.instruction.is_enabled, next.instruction.is_enabled);
        // dst is only going to be used for writes in the last input block
        assert_array_eq(
            &mut block_transition,
            local.instruction.dst,
            next.instruction.dst,
        );
        // these are not used and hence not necessary, but putting for safety until performance becomes an issue:
        block_transition.assert_eq(local.instruction.dst_ptr, next.instruction.dst_ptr);
        block_transition.assert_eq(local.instruction.src_ptr, next.instruction.src_ptr);
        block_transition.assert_eq(local.instruction.len_ptr, next.instruction.len_ptr);
        // no constraint on `instruction.len` because we use `remaining_len` instead

        // Move the src pointer over based on the number of bytes read.
        // This should always be RATE_BYTES since it's a non-final block.
        block_transition.assert_eq(
            next.instruction.src,
            local.instruction.src + AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
        );
        // Advance timestamp by the number of memory accesses from reading
        // `dst, src, len` and block input bytes.
        block_transition.assert_eq(next.instruction.start_timestamp, start_write_timestamp);
        block_transition.assert_eq(
            next.instruction.remaining_len,
            local.instruction.remaining_len - AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
        );
        // Padding transition is constrained in `constrain_padding`.
    }

    /// Keccak follows the 10*1 padding rule.
    /// See Section 5.1 of <https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf>
    /// Note this is the ONLY difference between Keccak and SHA-3
    ///
    /// Constrains padding constraints and length between rounds and
    /// between blocks. Padding logic is tied to constraints on `is_new_start`.
    pub fn constrain_padding<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let is_padding_byte = local.sponge.is_padding_byte;
        let block_bytes = &local.sponge.block_bytes;
        let remaining_len = local.remaining_len();

        // is_padding_byte should all be boolean
        for &is_padding_byte in is_padding_byte.iter() {
            builder.assert_bool(is_padding_byte);
        }
        // is_padding_byte should transition from 0 to 1 only once and then stay 1
        for i in 1..KECCAK_RATE_BYTES {
            builder
                .when(is_padding_byte[i - 1])
                .assert_one(is_padding_byte[i]);
        }
        // is_padding_byte must stay the same on all rounds in a block
        // we use next instead of local.step_flags.last() because the last row of the trace overall may not
        // end on a last round
        let is_last_round = next.inner.step_flags[0];
        let is_not_last_round = not(is_last_round);
        for i in 0..KECCAK_RATE_BYTES {
            builder.when(is_not_last_round.clone()).assert_eq(
                local.sponge.is_padding_byte[i],
                next.sponge.is_padding_byte[i],
            );
        }

        let num_padding_bytes = local
            .sponge
            .is_padding_byte
            .iter()
            .fold(AB::Expr::ZERO, |a, &b| a + b);

        // If final rate block of input, then last byte must be padding
        let is_final_block = is_padding_byte[KECCAK_RATE_BYTES - 1];

        // is_padding_byte must be consistent with remaining_len
        builder.when(is_final_block).assert_eq(
            remaining_len,
            AB::Expr::from_canonical_usize(KECCAK_RATE_BYTES) - num_padding_bytes,
        );
        // If this block is not final, when transitioning to next block, remaining len
        // must decrease by `KECCAK_RATE_BYTES`.
        builder
            .when(is_last_round)
            .when(not(is_final_block))
            .assert_eq(
                remaining_len - AB::F::from_canonical_usize(KECCAK_RATE_BYTES),
                next.remaining_len(),
            );
        // To enforce that is_padding_byte must be set appropriately for an input, we require
        // the block before a new start to have padding
        builder
            .when(is_last_round)
            .when(next.is_new_start())
            .assert_one(is_final_block);
        // Make sure there are not repeated padding blocks
        builder
            .when(is_last_round)
            .when(is_final_block)
            .assert_one(next.is_new_start());
        // The chain above enforces that for an input, the remaining length must decrease by RATE
        // block-by-block until it reaches a final block with padding.

        // ====== Constrain the block_bytes are padded according to is_padding_byte =====

        // If the first padding byte is at the end of the block, then the block has a
        // single padding byte
        let has_single_padding_byte: AB::Expr =
            is_padding_byte[KECCAK_RATE_BYTES - 1] - is_padding_byte[KECCAK_RATE_BYTES - 2];

        // If the row has a single padding byte, then it must be the last byte with
        // value 0b10000001
        builder.when(has_single_padding_byte.clone()).assert_eq(
            block_bytes[KECCAK_RATE_BYTES - 1],
            AB::F::from_canonical_u8(0b10000001),
        );

        let has_multiple_padding_bytes: AB::Expr = not(has_single_padding_byte.clone());
        for i in 0..KECCAK_RATE_BYTES - 1 {
            let is_first_padding_byte: AB::Expr = {
                if i > 0 {
                    is_padding_byte[i] - is_padding_byte[i - 1]
                } else {
                    is_padding_byte[i].into()
                }
            };
            // If the row has multiple padding bytes, the first padding byte must be 0x01
            // because the padding 1*0 is *little-endian*
            builder
                .when(has_multiple_padding_bytes.clone())
                .when(is_first_padding_byte.clone())
                .assert_eq(block_bytes[i], AB::F::from_canonical_u8(0x01));
            // If the row has multiple padding bytes, the other padding bytes
            // except the last one must be 0
            builder
                .when(is_padding_byte[i])
                .when(not::<AB::Expr>(is_first_padding_byte)) // hence never when single padding byte
                .assert_zero(block_bytes[i]);
        }

        // If the row has multiple padding bytes, then the last byte must be 0x80
        // because the padding *01 is *little-endian*
        builder
            .when(is_final_block)
            .when(has_multiple_padding_bytes)
            .assert_eq(
                block_bytes[KECCAK_RATE_BYTES - 1],
                AB::F::from_canonical_u8(0x80),
            );
    }

    /// Constrain state transition between keccak-f permutations is valid absorb of input bytes.
    /// The end-state in last round is given by `a_prime_prime_prime()` in `u16` limbs.
    /// The pre-state is given by `preimage` also in `u16` limbs.
    /// The input `block_bytes` will be given as **bytes**.
    ///
    /// We will XOR `block_bytes` with `a_prime_prime_prime()` and constrain to be `next.preimage`.
    /// This will be done using 8-bit XOR lookup in a separate AIR via interactions.
    /// This will require decomposing `u16` into bytes.
    /// Note that the XOR lookup automatically range checks its inputs to be bytes.
    ///
    /// We use the following trick to keep `u16` limbs and avoid changing
    /// the `keccak-f` AIR itself:
    /// if we already have a 16-bit limb `x` and we also provide a 8-bit limb
    /// `hi = x >> 8`, assuming `x` and `hi` have been range checked,
    /// we can use the expression `lo = x - hi * 256` for the low byte.
    /// If `lo` is range checked to `8`-bits, this constrains a valid byte
    ///  decomposition of `x` into `hi, lo`.
    /// This means in terms of trace cells, it is equivalent to provide
    /// `x, hi` versus `hi, lo`.
    pub fn constrain_absorb<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        next: &KeccakVmCols<AB::Var>,
    ) {
        let updated_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = local.postimage(y, x, limb);
                let hi = local.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                // Conversion from bytes to u64 is little-endian
                [lo, hi.into()]
            })
        });

        let post_absorb_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = next.inner.preimage[y][x][limb];
                let hi = next.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                [lo, hi.into()]
            })
        });

        // We xor on last round of each block, even if it is a final block,
        // because we use xor to range check the output bytes (= updated_state_bytes)
        let is_final_block = *local.sponge.is_padding_byte.last().unwrap();
        for (input, prev, post) in izip!(
            next.sponge.block_bytes,
            updated_state_bytes,
            post_absorb_state_bytes
        ) {
            // Add new send interaction to lookup (x, y, x ^ y) where x, y, z
            // will all be range checked to be 8-bits (assuming the bus is
            // received by an 8-bit xor chip).

            // When absorb, input ^ prev = post
            // Otherwise, 0 ^ prev = prev
            // The interaction fields are degree 2, leading to degree 3 constraint
            self.bitwise_lookup_bus
                .send_xor(
                    input * not(is_final_block),
                    prev.clone(),
                    select(is_final_block, prev, post),
                )
                .eval(
                    builder,
                    local.is_last_round() * local.instruction.is_enabled,
                );
        }

        // We separately constrain that when(local.is_new_start), the preimage (u16s) equals the block bytes
        let local_preimage_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = local.inner.preimage[y][x][limb];
                let hi = local.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                [lo, hi.into()]
            })
        });
        let mut when_is_new_start =
            builder.when(local.is_new_start() * local.instruction.is_enabled);
        for (preimage_byte, block_byte) in zip(local_preimage_bytes, local.sponge.block_bytes) {
            when_is_new_start.assert_eq(preimage_byte, block_byte);
        }

        // constrain transition on the state outside rate
        let mut reset_builder = builder.when(local.is_new_start());
        for i in KECCAK_RATE_U16S..KECCAK_WIDTH_U16S {
            let y = i / U64_LIMBS / 5;
            let x = (i / U64_LIMBS) % 5;
            let limb = i % U64_LIMBS;
            reset_builder.assert_zero(local.inner.preimage[y][x][limb]);
        }
        let mut absorb_builder = builder.when(local.is_last_round() * not(next.is_new_start()));
        for i in KECCAK_RATE_U16S..KECCAK_WIDTH_U16S {
            let y = i / U64_LIMBS / 5;
            let x = (i / U64_LIMBS) % 5;
            let limb = i % U64_LIMBS;
            absorb_builder.assert_eq(local.postimage(y, x, limb), next.inner.preimage[y][x][limb]);
        }
    }

    /// Receive the instruction itself on program bus. Send+receive on execution bus.
    /// Then does memory read in addr space 1 to get `dst, src, len` from memory.
    ///
    /// Adds range check interactions for the most significant limbs of the register values
    /// using BitwiseOperationLookupBus.
    ///
    /// Returns `start_read_timestamp` which is only relevant when `local.instruction.is_enabled`.
    /// Note that `start_read_timestamp` is a linear expression.
    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        register_aux: &[MemoryReadAuxCols<AB::Var>; KECCAK_REGISTER_READS],
    ) -> AB::Expr {
        let instruction = local.instruction;
        // Only receive opcode if:
        // - enabled row (not dummy row)
        // - first round of block
        // - is_new_start
        // Note this is degree 3, which results in quotient degree 2 if used
        // as `count` in interaction
        let should_receive = local.instruction.is_enabled * local.sponge.is_new_start;

        let [dst_ptr, src_ptr, len_ptr] = [
            instruction.dst_ptr,
            instruction.src_ptr,
            instruction.len_ptr,
        ];
        let reg_addr_sp = AB::F::ONE;
        let timestamp_change: AB::Expr = Self::timestamp_change(instruction.remaining_len);
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(Rv32KeccakOpcode::KECCAK256 as usize + self.offset),
                [
                    dst_ptr.into(),
                    src_ptr.into(),
                    len_ptr.into(),
                    reg_addr_sp.into(),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change,
            )
            .eval(builder, should_receive.clone());

        let mut timestamp: AB::Expr = instruction.start_timestamp.into();
        let recover_limbs = |limbs: [AB::Var; RV32_REGISTER_NUM_LIMBS - 1],
                             val: AB::Var|
         -> [AB::Expr; RV32_REGISTER_NUM_LIMBS] {
            from_fn(|i| {
                if i == 0 {
                    limbs
                        .into_iter()
                        .enumerate()
                        .fold(val.into(), |acc, (j, limb)| {
                            acc - limb
                                * AB::Expr::from_canonical_usize(1 << ((j + 1) * RV32_CELL_BITS))
                        })
                } else {
                    limbs[i - 1].into()
                }
            })
        };
        // Only when it is an input do we want to do memory read for
        // dst <- word[a]_d, src <- word[b]_d
        let dst_data = instruction.dst.map(Into::into);
        let src_data = recover_limbs(instruction.src_limbs, instruction.src);
        let len_data = recover_limbs(instruction.len_limbs, instruction.remaining_len);
        for (ptr, value, aux) in izip!(
            [dst_ptr, src_ptr, len_ptr],
            [dst_data, src_data, len_data],
            register_aux,
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(reg_addr_sp, ptr),
                    value,
                    timestamp.clone(),
                    aux,
                )
                .eval(builder, should_receive.clone());

            timestamp += AB::Expr::ONE;
        }
        // See Rv32VecHeapAdapterAir
        // repeat len for even number
        // We range check `len` to `max_ptr_bits` to ensure `remaining_len` doesn't overflow.
        // We could range check it to some other size, but `max_ptr_bits` is convenient.
        let need_range_check = [
            *instruction.dst.last().unwrap(),
            *instruction.src_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
        ];
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, should_receive.clone());
        }

        timestamp
    }

    /// Constrain reading the input as `block_bytes` from memory.
    /// Reads input based on `is_padding_byte`.
    /// Constrains timestamp transitions between blocks if input crosses blocks.
    ///
    /// Expects `start_read_timestamp` to be a linear expression.
    /// Returns the `start_write_timestamp` which is the timestamp to start from
    /// for writing digest to memory.
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
        mem_aux: &[MemoryReadAuxCols<AB::Var>; KECCAK_ABSORB_READS],
    ) -> AB::Expr {
        let partial_block = &local.mem_oc.partial_block;
        // Only read input from memory when it is an opcode-related row
        // and only on the first round of block
        let is_input = local.instruction.is_enabled_first_round;

        let mut timestamp = start_read_timestamp;
        // read `state` into `word[src + ...]_e`
        // iterator of state as u16:
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.block_bytes.chunks_exact(KECCAK_WORD_SIZE),
            local.sponge.is_padding_byte.chunks_exact(KECCAK_WORD_SIZE),
            mem_aux
        )
        .enumerate()
        {
            let ptr = local.instruction.src + AB::F::from_canonical_usize(i * KECCAK_WORD_SIZE);
            // Only read block i if it is not entirely padding bytes
            // count is degree 2
            let count = is_input * not(is_padding[0]);
            // The memory block read is partial if first byte is not padding but the last byte is padding. Since `count` is only 1 when first byte isn't padding, use check just if last byte is padding.
            let is_partial_read = *is_padding.last().unwrap();
            // word is degree 2
            let word: [_; KECCAK_WORD_SIZE] = from_fn(|i| {
                if i == 0 {
                    // first byte is always ok
                    input[0].into()
                } else {
                    // use `partial_block` if this is a partial read, otherwise use the normal input block
                    select(is_partial_read, partial_block[i - 1], input[i])
                }
            });
            for i in 1..KECCAK_WORD_SIZE {
                let not_padding: AB::Expr = not(is_padding[i]);
                // When not a padding byte, the word byte and input byte must be equal
                // This is constraint degree 3
                builder.assert_eq(
                    not_padding.clone() * word[i].clone(),
                    not_padding.clone() * input[i],
                );
            }

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    word, // degree 2
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, count);

            timestamp += AB::Expr::ONE;
        }
        timestamp
    }

    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
        mem_aux: &[MemoryWriteAuxCols<AB::Var, KECCAK_WORD_SIZE>; KECCAK_DIGEST_WRITES],
    ) {
        let instruction = local.instruction;

        let is_final_block = *local.sponge.is_padding_byte.last().unwrap();
        // since keccak-f AIR has this column, we might as well use it
        builder.assert_eq(
            local.inner.export,
            instruction.is_enabled * is_final_block * local.is_last_round(),
        );
        // See `constrain_absorb` on how we derive the postimage bytes from u16 limbs
        // **SAFETY:** we always XOR the final state with 0 in `constrain_absorb`,
        // so the output bytes **are** range checked.
        let updated_state_bytes = (0..NUM_ABSORB_ROUNDS).flat_map(|i| {
            let y = i / 5;
            let x = i % 5;
            (0..U64_LIMBS).flat_map(move |limb| {
                let state_limb = local.postimage(y, x, limb);
                let hi = local.sponge.state_hi[i * U64_LIMBS + limb];
                let lo = state_limb - hi * AB::F::from_canonical_u64(1 << 8);
                // Conversion from bytes to u64 is little-endian
                [lo, hi.into()]
            })
        });
        let dst = abstract_compose::<AB::Expr, _>(instruction.dst);
        for (i, digest_bytes) in updated_state_bytes
            .take(KECCAK_DIGEST_BYTES)
            .chunks(KECCAK_WORD_SIZE)
            .into_iter()
            .enumerate()
        {
            let digest_bytes = digest_bytes.collect_vec();
            let timestamp = start_write_timestamp.clone() + AB::Expr::from_canonical_usize(i);
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        dst.clone() + AB::F::from_canonical_usize(i * KECCAK_WORD_SIZE),
                    ),
                    digest_bytes.try_into().unwrap(),
                    timestamp,
                    &mem_aux[i],
                )
                .eval(builder, local.inner.export)
        }
    }

    /// Amount to advance timestamp by after execution of one opcode instruction.
    /// This is an upper bound dependent on the length `len` operand, which is unbounded.
    pub fn timestamp_change<T: FieldAlgebra>(len: impl Into<T>) -> T {
        // actual number is ceil(len / 136) * (3 + 17) + KECCAK_DIGEST_WRITES
        // digest writes only done on last row of multi-block
        // add another KECCAK_ABSORB_READS to round up so we don't deal with padding
        len.into()
            + T::from_canonical_usize(
                KECCAK_REGISTER_READS + KECCAK_ABSORB_READS + KECCAK_DIGEST_WRITES,
            )
    }
}
