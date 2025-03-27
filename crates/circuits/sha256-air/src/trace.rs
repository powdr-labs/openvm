use std::{array, borrow::BorrowMut, ops::Range};

use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
    rap::BaseAir,
};
use sha2::{compress256, digest::generic_array::GenericArray};

use super::{
    air::Sha256Air, big_sig0_field, big_sig1_field, ch_field, columns::Sha256RoundCols, compose,
    get_flag_pt_array, maj_field, small_sig0_field, small_sig1_field, SHA256_BLOCK_WORDS,
    SHA256_DIGEST_WIDTH, SHA256_HASH_WORDS, SHA256_ROUND_WIDTH,
};
use crate::{
    big_sig0, big_sig1, ch, columns::Sha256DigestCols, limbs_into_u32, maj, small_sig0, small_sig1,
    u32_into_limbs, SHA256_BLOCK_U8S, SHA256_BUFFER_SIZE, SHA256_H, SHA256_INVALID_CARRY_A,
    SHA256_INVALID_CARRY_E, SHA256_K, SHA256_ROUNDS_PER_ROW, SHA256_ROWS_PER_BLOCK,
    SHA256_WORD_BITS, SHA256_WORD_U16S, SHA256_WORD_U8S,
};

/// The trace generation of SHA256 should be done in two passes.
/// The first pass should do `get_block_trace` for every block and generate the invalid rows through `get_default_row`
/// The second pass should go through all the blocks and call `generate_missing_values`
impl Sha256Air {
    /// This function takes the input_message (padding not handled), the previous hash,
    /// and returns the new hash after processing the block input
    pub fn get_block_hash(
        prev_hash: &[u32; SHA256_HASH_WORDS],
        input: [u8; SHA256_BLOCK_U8S],
    ) -> [u32; SHA256_HASH_WORDS] {
        let mut new_hash = *prev_hash;
        let input_array = [GenericArray::from(input)];
        compress256(&mut new_hash, &input_array);
        new_hash
    }

    /// This function takes a 512-bit chunk of the input message (padding not handled), the previous hash,
    /// a flag indicating if it's the last block, the global block index, the local block index,
    /// and the buffer values that will be put in rows 0..4.
    /// Will populate the given `trace` with the trace of the block, where the width of the trace is `trace_width`
    /// and the starting column for the `Sha256Air` is `trace_start_col`.
    /// **Note**: this function only generates some of the required trace. Another pass is required, refer to [`Self::generate_missing_cells`] for details.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_block_trace<F: PrimeField32>(
        &self,
        trace: &mut [F],
        trace_width: usize,
        trace_start_col: usize,
        input: &[u32; SHA256_BLOCK_WORDS],
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        prev_hash: &[u32; SHA256_HASH_WORDS],
        is_last_block: bool,
        global_block_idx: u32,
        local_block_idx: u32,
        buffer_vals: &[[F; SHA256_BUFFER_SIZE]; 4],
    ) {
        #[cfg(debug_assertions)]
        {
            assert!(trace.len() == trace_width * SHA256_ROWS_PER_BLOCK);
            assert!(trace_start_col + super::SHA256_WIDTH <= trace_width);
            assert!(self.bitwise_lookup_bus == bitwise_lookup_chip.bus());
            if local_block_idx == 0 {
                assert!(*prev_hash == SHA256_H);
            }
        }
        let get_range = |start: usize, len: usize| -> Range<usize> { start..start + len };
        let mut message_schedule = [0u32; 64];
        message_schedule[..input.len()].copy_from_slice(input);
        let mut work_vars = *prev_hash;
        for (i, row) in trace.chunks_exact_mut(trace_width).enumerate() {
            // doing the 64 rounds in 16 rows
            if i < 16 {
                let cols: &mut Sha256RoundCols<F> =
                    row[get_range(trace_start_col, SHA256_ROUND_WIDTH)].borrow_mut();
                cols.flags.is_round_row = F::ONE;
                cols.flags.is_first_4_rows = if i < 4 { F::ONE } else { F::ZERO };
                cols.flags.is_digest_row = F::ZERO;
                cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags.row_idx =
                    get_flag_pt_array(&self.row_idx_encoder, i).map(F::from_canonical_u32);
                cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);
                cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);

                // W_idx = M_idx
                if i < SHA256_ROWS_PER_BLOCK / SHA256_ROUNDS_PER_ROW {
                    for j in 0..SHA256_ROUNDS_PER_ROW {
                        cols.message_schedule.w[j] = u32_into_limbs::<SHA256_WORD_BITS>(
                            input[i * SHA256_ROUNDS_PER_ROW + j],
                        )
                        .map(F::from_canonical_u32);
                        cols.message_schedule.carry_or_buffer[j] =
                            array::from_fn(|k| buffer_vals[i][j * SHA256_WORD_U16S * 2 + k]);
                    }
                }
                // W_idx = SIG1(W_{idx-2}) + W_{idx-7} + SIG0(W_{idx-15}) + W_{idx-16}
                else {
                    for j in 0..SHA256_ROUNDS_PER_ROW {
                        let idx = i * SHA256_ROUNDS_PER_ROW + j;
                        let nums: [u32; 4] = [
                            small_sig1(message_schedule[idx - 2]),
                            message_schedule[idx - 7],
                            small_sig0(message_schedule[idx - 15]),
                            message_schedule[idx - 16],
                        ];
                        let w: u32 = nums.iter().fold(0, |acc, &num| acc.wrapping_add(num));
                        cols.message_schedule.w[j] =
                            u32_into_limbs::<SHA256_WORD_BITS>(w).map(F::from_canonical_u32);

                        let nums_limbs = nums
                            .iter()
                            .map(|x| u32_into_limbs::<SHA256_WORD_U16S>(*x))
                            .collect::<Vec<_>>();
                        let w_limbs = u32_into_limbs::<SHA256_WORD_U16S>(w);

                        // fill in the carrys
                        for k in 0..SHA256_WORD_U16S {
                            let mut sum = nums_limbs.iter().fold(0, |acc, num| acc + num[k]);
                            if k > 0 {
                                sum += (cols.message_schedule.carry_or_buffer[j][k * 2 - 2]
                                    + F::TWO * cols.message_schedule.carry_or_buffer[j][k * 2 - 1])
                                    .as_canonical_u32();
                            }
                            let carry = (sum - w_limbs[k]) >> 16;
                            cols.message_schedule.carry_or_buffer[j][k * 2] =
                                F::from_canonical_u32(carry & 1);
                            cols.message_schedule.carry_or_buffer[j][k * 2 + 1] =
                                F::from_canonical_u32(carry >> 1);
                        }
                        // update the message schedule
                        message_schedule[idx] = w;
                    }
                }
                // fill in the work variables
                for j in 0..SHA256_ROUNDS_PER_ROW {
                    // t1 = h + SIG1(e) + ch(e, f, g) + K_idx + W_idx
                    let t1 = [
                        work_vars[7],
                        big_sig1(work_vars[4]),
                        ch(work_vars[4], work_vars[5], work_vars[6]),
                        SHA256_K[i * SHA256_ROUNDS_PER_ROW + j],
                        limbs_into_u32(cols.message_schedule.w[j].map(|f| f.as_canonical_u32())),
                    ];
                    let t1_sum: u32 = t1.iter().fold(0, |acc, &num| acc.wrapping_add(num));

                    // t2 = SIG0(a) + maj(a, b, c)
                    let t2 = [
                        big_sig0(work_vars[0]),
                        maj(work_vars[0], work_vars[1], work_vars[2]),
                    ];

                    let t2_sum: u32 = t2.iter().fold(0, |acc, &num| acc.wrapping_add(num));

                    // e = d + t1
                    let e = work_vars[3].wrapping_add(t1_sum);
                    cols.work_vars.e[j] =
                        u32_into_limbs::<SHA256_WORD_BITS>(e).map(F::from_canonical_u32);
                    let e_limbs = u32_into_limbs::<SHA256_WORD_U16S>(e);
                    // a = t1 + t2
                    let a = t1_sum.wrapping_add(t2_sum);
                    cols.work_vars.a[j] =
                        u32_into_limbs::<SHA256_WORD_BITS>(a).map(F::from_canonical_u32);
                    let a_limbs = u32_into_limbs::<SHA256_WORD_U16S>(a);
                    // fill in the carrys
                    for k in 0..SHA256_WORD_U16S {
                        let t1_limb = t1.iter().fold(0, |acc, &num| {
                            acc + u32_into_limbs::<SHA256_WORD_U16S>(num)[k]
                        });
                        let t2_limb = t2.iter().fold(0, |acc, &num| {
                            acc + u32_into_limbs::<SHA256_WORD_U16S>(num)[k]
                        });

                        let mut e_limb =
                            t1_limb + u32_into_limbs::<SHA256_WORD_U16S>(work_vars[3])[k];
                        let mut a_limb = t1_limb + t2_limb;
                        if k > 0 {
                            a_limb += cols.work_vars.carry_a[j][k - 1].as_canonical_u32();
                            e_limb += cols.work_vars.carry_e[j][k - 1].as_canonical_u32();
                        }
                        let carry_a = (a_limb - a_limbs[k]) >> 16;
                        let carry_e = (e_limb - e_limbs[k]) >> 16;
                        cols.work_vars.carry_a[j][k] = F::from_canonical_u32(carry_a);
                        cols.work_vars.carry_e[j][k] = F::from_canonical_u32(carry_e);
                        bitwise_lookup_chip.request_range(carry_a, carry_e);
                    }

                    // update working variables
                    work_vars[7] = work_vars[6];
                    work_vars[6] = work_vars[5];
                    work_vars[5] = work_vars[4];
                    work_vars[4] = e;
                    work_vars[3] = work_vars[2];
                    work_vars[2] = work_vars[1];
                    work_vars[1] = work_vars[0];
                    work_vars[0] = a;
                }

                // filling w_3 and intermed_4 here and the rest later
                if i > 0 {
                    for j in 0..SHA256_ROUNDS_PER_ROW {
                        let idx = i * SHA256_ROUNDS_PER_ROW + j;
                        let w_4 = u32_into_limbs::<SHA256_WORD_U16S>(message_schedule[idx - 4]);
                        let sig_0_w_3 = u32_into_limbs::<SHA256_WORD_U16S>(small_sig0(
                            message_schedule[idx - 3],
                        ));
                        cols.schedule_helper.intermed_4[j] =
                            array::from_fn(|k| F::from_canonical_u32(w_4[k] + sig_0_w_3[k]));
                        if j < SHA256_ROUNDS_PER_ROW - 1 {
                            let w_3 = message_schedule[idx - 3];
                            cols.schedule_helper.w_3[j] =
                                u32_into_limbs::<SHA256_WORD_U16S>(w_3).map(F::from_canonical_u32);
                        }
                    }
                }
            }
            // generate the digest row
            else {
                let cols: &mut Sha256DigestCols<F> =
                    row[get_range(trace_start_col, SHA256_DIGEST_WIDTH)].borrow_mut();
                for j in 0..SHA256_ROUNDS_PER_ROW - 1 {
                    let w_3 = message_schedule[i * SHA256_ROUNDS_PER_ROW + j - 3];
                    cols.schedule_helper.w_3[j] =
                        u32_into_limbs::<SHA256_WORD_U16S>(w_3).map(F::from_canonical_u32);
                }
                cols.flags.is_round_row = F::ZERO;
                cols.flags.is_first_4_rows = F::ZERO;
                cols.flags.is_digest_row = F::ONE;
                cols.flags.is_last_block = F::from_bool(is_last_block);
                cols.flags.row_idx =
                    get_flag_pt_array(&self.row_idx_encoder, 16).map(F::from_canonical_u32);
                cols.flags.global_block_idx = F::from_canonical_u32(global_block_idx);

                cols.flags.local_block_idx = F::from_canonical_u32(local_block_idx);
                let final_hash: [u32; SHA256_HASH_WORDS] =
                    array::from_fn(|i| work_vars[i].wrapping_add(prev_hash[i]));
                cols.final_hash = array::from_fn(|i| {
                    u32_into_limbs::<SHA256_WORD_U8S>(final_hash[i]).map(F::from_canonical_u32)
                });
                cols.prev_hash = prev_hash
                    .map(|f| u32_into_limbs::<SHA256_WORD_U16S>(f).map(F::from_canonical_u32));
                let hash = if is_last_block {
                    SHA256_H.map(u32_into_limbs::<SHA256_WORD_BITS>)
                } else {
                    cols.final_hash
                        .map(|f| limbs_into_u32(f.map(|x| x.as_canonical_u32())))
                        .map(u32_into_limbs::<SHA256_WORD_BITS>)
                }
                .map(|x| x.map(F::from_canonical_u32));

                for i in 0..SHA256_ROUNDS_PER_ROW {
                    cols.hash.a[i] = hash[SHA256_ROUNDS_PER_ROW - i - 1];
                    cols.hash.e[i] = hash[SHA256_ROUNDS_PER_ROW - i + 3];
                }
            }
        }

        for i in 0..SHA256_ROWS_PER_BLOCK - 1 {
            let rows = &mut trace[i * trace_width..(i + 2) * trace_width];
            let (local, next) = rows.split_at_mut(trace_width);
            let local_cols: &mut Sha256RoundCols<F> =
                local[get_range(trace_start_col, SHA256_ROUND_WIDTH)].borrow_mut();
            let next_cols: &mut Sha256RoundCols<F> =
                next[get_range(trace_start_col, SHA256_ROUND_WIDTH)].borrow_mut();
            if i > 0 {
                for j in 0..SHA256_ROUNDS_PER_ROW {
                    next_cols.schedule_helper.intermed_8[j] =
                        local_cols.schedule_helper.intermed_4[j];
                    if (2..SHA256_ROWS_PER_BLOCK - 3).contains(&i) {
                        next_cols.schedule_helper.intermed_12[j] =
                            local_cols.schedule_helper.intermed_8[j];
                    }
                }
            }
            if i == SHA256_ROWS_PER_BLOCK - 2 {
                Self::generate_carry_ae(local_cols, next_cols);
                Self::generate_intermed_4(local_cols, next_cols);
            }
            if i <= 2 {
                Self::generate_intermed_12(local_cols, next_cols);
            }
        }
    }

    /// This function will fill in the cells that we couldn't do during the first pass.
    /// This function should be called only after `generate_block_trace` was called for all blocks
    /// And [`Self::generate_default_row`] is called for all invalid rows
    /// Will populate the missing values of `trace`, where the width of the trace is `trace_width`
    /// and the starting column for the `Sha256Air` is `trace_start_col`.
    /// Note: `trace` needs to be the rows 1..17 of a block and the first row of the next block
    pub fn generate_missing_cells<F: PrimeField32>(
        &self,
        trace: &mut [F],
        trace_width: usize,
        trace_start_col: usize,
    ) {
        // Here row_17 = next blocks row 0
        let rows_15_17 = &mut trace[14 * trace_width..17 * trace_width];
        let (row_15, row_16_17) = rows_15_17.split_at_mut(trace_width);
        let (row_16, row_17) = row_16_17.split_at_mut(trace_width);
        let cols_15: &mut Sha256RoundCols<F> =
            row_15[trace_start_col..trace_start_col + SHA256_ROUND_WIDTH].borrow_mut();
        let cols_16: &mut Sha256RoundCols<F> =
            row_16[trace_start_col..trace_start_col + SHA256_ROUND_WIDTH].borrow_mut();
        let cols_17: &mut Sha256RoundCols<F> =
            row_17[trace_start_col..trace_start_col + SHA256_ROUND_WIDTH].borrow_mut();
        Self::generate_intermed_12(cols_15, cols_16);
        Self::generate_intermed_12(cols_16, cols_17);
        Self::generate_intermed_4(cols_16, cols_17);
    }

    /// Fills the `cols` as a padding row
    /// Note: we still need to correctly fill in the hash values, carries and intermeds
    pub fn generate_default_row<F: PrimeField32>(self: &Sha256Air, cols: &mut Sha256RoundCols<F>) {
        cols.flags.is_round_row = F::ZERO;
        cols.flags.is_first_4_rows = F::ZERO;
        cols.flags.is_digest_row = F::ZERO;

        cols.flags.is_last_block = F::ZERO;
        cols.flags.global_block_idx = F::ZERO;
        cols.flags.row_idx =
            get_flag_pt_array(&self.row_idx_encoder, 17).map(F::from_canonical_u32);
        cols.flags.local_block_idx = F::ZERO;

        cols.message_schedule.w = [[F::ZERO; SHA256_WORD_BITS]; SHA256_ROUNDS_PER_ROW];
        cols.message_schedule.carry_or_buffer =
            [[F::ZERO; SHA256_WORD_U16S * 2]; SHA256_ROUNDS_PER_ROW];

        let hash = SHA256_H
            .map(u32_into_limbs::<SHA256_WORD_BITS>)
            .map(|x| x.map(F::from_canonical_u32));

        for i in 0..SHA256_ROUNDS_PER_ROW {
            cols.work_vars.a[i] = hash[SHA256_ROUNDS_PER_ROW - i - 1];
            cols.work_vars.e[i] = hash[SHA256_ROUNDS_PER_ROW - i + 3];
        }

        cols.work_vars.carry_a = array::from_fn(|i| {
            array::from_fn(|j| F::from_canonical_u32(SHA256_INVALID_CARRY_A[i][j]))
        });
        cols.work_vars.carry_e = array::from_fn(|i| {
            array::from_fn(|j| F::from_canonical_u32(SHA256_INVALID_CARRY_E[i][j]))
        });
    }

    /// The following functions do the calculations in native field since they will be called on padding rows
    /// which can overflow and we need to make sure it matches the AIR constraints
    /// Puts the correct carrys in the `next_row`, the resulting carrys can be out of bound
    fn generate_carry_ae<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        let a = [local_cols.work_vars.a, next_cols.work_vars.a].concat();
        let e = [local_cols.work_vars.e, next_cols.work_vars.e].concat();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let cur_a = a[i + 4];
            let sig_a = big_sig0_field::<F>(&a[i + 3]);
            let maj_abc = maj_field::<F>(&a[i + 3], &a[i + 2], &a[i + 1]);
            let d = a[i];
            let cur_e = e[i + 4];
            let sig_e = big_sig1_field::<F>(&e[i + 3]);
            let ch_efg = ch_field::<F>(&e[i + 3], &e[i + 2], &e[i + 1]);
            let h = e[i];

            let t1 = [h, sig_e, ch_efg];
            let t2 = [sig_a, maj_abc];
            for j in 0..SHA256_WORD_U16S {
                let t1_limb_sum = t1.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let t2_limb_sum = t2.iter().fold(F::ZERO, |acc, x| {
                    acc + compose::<F>(&x[j * 16..(j + 1) * 16], 1)
                });
                let d_limb = compose::<F>(&d[j * 16..(j + 1) * 16], 1);
                let cur_a_limb = compose::<F>(&cur_a[j * 16..(j + 1) * 16], 1);
                let cur_e_limb = compose::<F>(&cur_e[j * 16..(j + 1) * 16], 1);
                let sum = d_limb
                    + t1_limb_sum
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_e[i][j - 1]
                    }
                    - cur_e_limb;
                let carry_e = sum * (F::from_canonical_u32(1 << 16).inverse());

                let sum = t1_limb_sum
                    + t2_limb_sum
                    + if j == 0 {
                        F::ZERO
                    } else {
                        next_cols.work_vars.carry_a[i][j - 1]
                    }
                    - cur_a_limb;
                let carry_a = sum * (F::from_canonical_u32(1 << 16).inverse());
                next_cols.work_vars.carry_e[i][j] = carry_e;
                next_cols.work_vars.carry_a[i][j] = carry_a;
            }
        }
    }

    /// Puts the correct intermed_4 in the `next_row`
    fn generate_intermed_4<F: PrimeField32>(
        local_cols: &Sha256RoundCols<F>,
        next_cols: &mut Sha256RoundCols<F>,
    ) {
        let w = [local_cols.message_schedule.w, next_cols.message_schedule.w].concat();
        let w_limbs: Vec<[F; SHA256_WORD_U16S]> = w
            .iter()
            .map(|x| array::from_fn(|i| compose::<F>(&x[i * 16..(i + 1) * 16], 1)))
            .collect();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            let sig_w = small_sig0_field::<F>(&w[i + 1]);
            let sig_w_limbs: [F; SHA256_WORD_U16S] =
                array::from_fn(|j| compose::<F>(&sig_w[j * 16..(j + 1) * 16], 1));
            for (j, sig_w_limb) in sig_w_limbs.iter().enumerate() {
                next_cols.schedule_helper.intermed_4[i][j] = w_limbs[i][j] + *sig_w_limb;
            }
        }
    }

    /// Puts the needed intermed_12 in the `local_row`
    fn generate_intermed_12<F: PrimeField32>(
        local_cols: &mut Sha256RoundCols<F>,
        next_cols: &Sha256RoundCols<F>,
    ) {
        let w = [local_cols.message_schedule.w, next_cols.message_schedule.w].concat();
        let w_limbs: Vec<[F; SHA256_WORD_U16S]> = w
            .iter()
            .map(|x| array::from_fn(|i| compose::<F>(&x[i * 16..(i + 1) * 16], 1)))
            .collect();
        for i in 0..SHA256_ROUNDS_PER_ROW {
            // sig_1(w_{t-2})
            let sig_w_2: [F; SHA256_WORD_U16S] = array::from_fn(|j| {
                compose::<F>(&small_sig1_field::<F>(&w[i + 2])[j * 16..(j + 1) * 16], 1)
            });
            // w_{t-7}
            let w_7 = if i < 3 {
                local_cols.schedule_helper.w_3[i]
            } else {
                w_limbs[i - 3]
            };
            // w_t
            let w_cur = w_limbs[i + 4];
            for j in 0..SHA256_WORD_U16S {
                let carry = next_cols.message_schedule.carry_or_buffer[i][j * 2]
                    + F::TWO * next_cols.message_schedule.carry_or_buffer[i][j * 2 + 1];
                let sum = sig_w_2[j] + w_7[j] - carry * F::from_canonical_u32(1 << 16) - w_cur[j]
                    + if j > 0 {
                        next_cols.message_schedule.carry_or_buffer[i][j * 2 - 2]
                            + F::from_canonical_u32(2)
                                * next_cols.message_schedule.carry_or_buffer[i][j * 2 - 1]
                    } else {
                        F::ZERO
                    };
                local_cols.schedule_helper.intermed_12[i][j] = -sum;
            }
        }
    }
}

/// `records` consists of pairs of `(input_block, is_last_block)`.
pub fn generate_trace<F: PrimeField32>(
    sub_air: &Sha256Air,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    records: Vec<([u8; SHA256_BLOCK_U8S], bool)>,
) -> RowMajorMatrix<F> {
    let non_padded_height = records.len() * SHA256_ROWS_PER_BLOCK;
    let height = next_power_of_two_or_zero(non_padded_height);
    let width = <Sha256Air as BaseAir<F>>::width(sub_air);
    let mut values = F::zero_vec(height * width);

    struct BlockContext {
        prev_hash: [u32; 8],
        local_block_idx: u32,
        global_block_idx: u32,
        input: [u8; SHA256_BLOCK_U8S],
        is_last_block: bool,
    }
    let mut block_ctx: Vec<BlockContext> = Vec::with_capacity(records.len());
    let mut prev_hash = SHA256_H;
    let mut local_block_idx = 0;
    let mut global_block_idx = 1;
    for (input, is_last_block) in records {
        block_ctx.push(BlockContext {
            prev_hash,
            local_block_idx,
            global_block_idx,
            input,
            is_last_block,
        });
        global_block_idx += 1;
        if is_last_block {
            local_block_idx = 0;
            prev_hash = SHA256_H;
        } else {
            local_block_idx += 1;
            prev_hash = Sha256Air::get_block_hash(&prev_hash, input);
        }
    }
    // first pass
    values
        .par_chunks_exact_mut(width * SHA256_ROWS_PER_BLOCK)
        .zip(block_ctx)
        .for_each(|(block, ctx)| {
            let BlockContext {
                prev_hash,
                local_block_idx,
                global_block_idx,
                input,
                is_last_block,
            } = ctx;
            let input_words = array::from_fn(|i| {
                limbs_into_u32::<SHA256_WORD_U8S>(array::from_fn(|j| {
                    input[i * SHA256_WORD_U8S + j] as u32
                }))
            });
            sub_air.generate_block_trace(
                block,
                width,
                0,
                &input_words,
                bitwise_lookup_chip.clone(),
                &prev_hash,
                is_last_block,
                global_block_idx,
                local_block_idx,
                &[[F::ZERO; 16]; 4],
            );
        });
    // second pass: padding rows
    values[width * non_padded_height..]
        .par_chunks_mut(width)
        .for_each(|row| {
            let cols: &mut Sha256RoundCols<F> = row.borrow_mut();
            sub_air.generate_default_row(cols);
        });
    // second pass: non-padding rows
    values[width..]
        .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
        .take(non_padded_height / SHA256_ROWS_PER_BLOCK)
        .for_each(|chunk| {
            sub_air.generate_missing_cells(chunk, width, 0);
        });
    RowMajorMatrix::new(values, width)
}
