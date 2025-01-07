use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32im_circuit::adapters::compose;
use openvm_sha256_air::{
    get_flag_pt_array, limbs_into_u32, Sha256Air, SHA256_BLOCK_WORDS, SHA256_BUFFER_SIZE, SHA256_H,
    SHA256_HASH_WORDS, SHA256_ROWS_PER_BLOCK, SHA256_WORD_U8S,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{AbstractField, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
    },
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};

use super::{
    Sha256VmChip, Sha256VmDigestCols, Sha256VmRoundCols, SHA256VM_CONTROL_WIDTH,
    SHA256VM_DIGEST_WIDTH, SHA256VM_ROUND_WIDTH,
};
use crate::{
    sha256_chip::{PaddingFlags, SHA256_READ_SIZE},
    SHA256_BLOCK_CELLS,
};

impl<SC: StarkGenericConfig> Chip<SC> for Sha256VmChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let non_padded_height = self.current_trace_height();
        let height = next_power_of_two_or_zero(non_padded_height);
        let width = self.trace_width();
        let mut values = Val::<SC>::zero_vec(height * width);
        if height == 0 {
            return AirProofInput::simple(air, RowMajorMatrix::new(values, width), vec![]);
        }
        let records = self.records;
        let offline_memory = self.offline_memory.lock().unwrap();
        let memory_aux_cols_factory = offline_memory.aux_cols_factory();

        let mem_ptr_shift: u32 =
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.air.ptr_max_bits);

        let mut states = Vec::with_capacity(height.div_ceil(SHA256_ROWS_PER_BLOCK));
        let mut global_block_idx = 0;
        for (record_idx, record) in records.iter().enumerate() {
            let dst_read = offline_memory.record_by_id(record.dst_read);
            let src_read = offline_memory.record_by_id(record.src_read);
            let len_read = offline_memory.record_by_id(record.len_read);

            self.bitwise_lookup_chip.request_range(
                dst_read.data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() * mem_ptr_shift,
                src_read.data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() * mem_ptr_shift,
            );
            let len = compose(len_read.data.clone().try_into().unwrap());
            let mut state = &None;
            for (i, input_message) in record.input_message.iter().enumerate() {
                let input_message = input_message
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                states.push(Some(Self::generate_state(
                    state,
                    input_message,
                    record_idx,
                    len,
                    i == record.input_records.len() - 1,
                )));
                state = &states[global_block_idx];
                global_block_idx += 1;
            }
        }
        states.extend(
            std::iter::repeat(None)
                .take((height - non_padded_height).div_ceil(SHA256_ROWS_PER_BLOCK)),
        );

        // During the first pass we will fill out most of the matrix
        // But there are some cells that can't be generated by the first pass so we will do a second pass over the matrix
        values
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .zip(states.into_par_iter().enumerate())
            .for_each(|(block, (global_block_idx, state))| {
                // Fill in a valid block
                if let Some(state) = state {
                    let mut has_padding_occurred =
                        state.local_block_idx * SHA256_BLOCK_CELLS > state.message_len as usize;
                    let message_left = if has_padding_occurred {
                        0
                    } else {
                        state.message_len as usize - state.local_block_idx * SHA256_BLOCK_CELLS
                    };
                    let is_last_block = state.is_last_block;
                    let buffer: [[Val<SC>; SHA256_BUFFER_SIZE]; 4] = array::from_fn(|j| {
                        array::from_fn(|k| {
                            Val::<SC>::from_canonical_u8(
                                state.block_input_message[j * SHA256_BUFFER_SIZE + k],
                            )
                        })
                    });

                    let padded_message: [u32; SHA256_BLOCK_WORDS] = array::from_fn(|j| {
                        limbs_into_u32::<RV32_REGISTER_NUM_LIMBS>(array::from_fn(|k| {
                            state.block_padded_message[(j + 1) * SHA256_WORD_U8S - k - 1] as u32
                        }))
                    });

                    self.air.sha256_subair.generate_block_trace::<Val<SC>>(
                        block,
                        width,
                        SHA256VM_CONTROL_WIDTH,
                        &padded_message,
                        self.bitwise_lookup_chip.as_ref(),
                        &state.hash,
                        is_last_block,
                        global_block_idx as u32 + 1,
                        state.local_block_idx as u32,
                        &buffer,
                    );

                    let block_reads = records[state.message_idx].input_records
                        [state.local_block_idx]
                        .map(|record_id| offline_memory.record_by_id(record_id));

                    let mut read_ptr = block_reads[0].pointer;
                    let mut cur_timestamp = Val::<SC>::from_canonical_u32(block_reads[0].timestamp);

                    let read_size = Val::<SC>::from_canonical_usize(SHA256_READ_SIZE);
                    for row in 0..SHA256_ROWS_PER_BLOCK {
                        let row_slice = &mut block[row * width..(row + 1) * width];
                        if row < 16 {
                            let cols: &mut Sha256VmRoundCols<Val<SC>> =
                                row_slice[..SHA256VM_ROUND_WIDTH].borrow_mut();
                            cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                            cols.control.read_ptr = read_ptr;
                            cols.control.cur_timestamp = cur_timestamp;
                            if row < 4 {
                                read_ptr += read_size;
                                cur_timestamp += Val::<SC>::ONE;
                                cols.read_aux =
                                    memory_aux_cols_factory.make_read_aux_cols(block_reads[row]);

                                if (row + 1) * SHA256_READ_SIZE <= message_left {
                                    cols.control.pad_flags = get_flag_pt_array(
                                        &self.air.padding_encoder,
                                        PaddingFlags::NotPadding as usize,
                                    )
                                    .map(Val::<SC>::from_canonical_u32);
                                } else if !has_padding_occurred {
                                    has_padding_occurred = true;
                                    let len = message_left - row * SHA256_READ_SIZE;
                                    cols.control.pad_flags = get_flag_pt_array(
                                        &self.air.padding_encoder,
                                        if row == 3 && is_last_block {
                                            PaddingFlags::FirstPadding0_LastRow
                                        } else {
                                            PaddingFlags::FirstPadding0
                                        } as usize
                                            + len,
                                    )
                                    .map(Val::<SC>::from_canonical_u32);
                                } else {
                                    cols.control.pad_flags = get_flag_pt_array(
                                        &self.air.padding_encoder,
                                        if row == 3 && is_last_block {
                                            PaddingFlags::EntirePaddingLastRow
                                        } else {
                                            PaddingFlags::EntirePadding
                                        } as usize,
                                    )
                                    .map(Val::<SC>::from_canonical_u32);
                                }
                            } else {
                                cols.control.pad_flags = get_flag_pt_array(
                                    &self.air.padding_encoder,
                                    PaddingFlags::NotConsidered as usize,
                                )
                                .map(Val::<SC>::from_canonical_u32);
                            }
                            cols.control.padding_occurred =
                                Val::<SC>::from_bool(has_padding_occurred);
                        } else {
                            if is_last_block {
                                has_padding_occurred = false;
                            }
                            let cols: &mut Sha256VmDigestCols<Val<SC>> =
                                row_slice[..SHA256VM_DIGEST_WIDTH].borrow_mut();
                            cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                            cols.control.read_ptr = read_ptr;
                            cols.control.cur_timestamp = cur_timestamp;
                            cols.control.pad_flags = get_flag_pt_array(
                                &self.air.padding_encoder,
                                PaddingFlags::NotConsidered as usize,
                            )
                            .map(Val::<SC>::from_canonical_u32);
                            if is_last_block {
                                let record = &records[state.message_idx];
                                let dst_read = offline_memory.record_by_id(record.dst_read);
                                let src_read = offline_memory.record_by_id(record.src_read);
                                let len_read = offline_memory.record_by_id(record.len_read);
                                let digest_write = offline_memory.record_by_id(record.digest_write);
                                cols.from_state = record.from_state;
                                cols.rd_ptr = dst_read.pointer;
                                cols.rs1_ptr = src_read.pointer;
                                cols.rs2_ptr = len_read.pointer;
                                cols.dst_ptr = dst_read.data.clone().try_into().unwrap();
                                cols.src_ptr = src_read.data.clone().try_into().unwrap();
                                cols.len_data = len_read.data.clone().try_into().unwrap();
                                cols.register_reads_aux = [
                                    memory_aux_cols_factory.make_read_aux_cols(dst_read),
                                    memory_aux_cols_factory.make_read_aux_cols(src_read),
                                    memory_aux_cols_factory.make_read_aux_cols(len_read),
                                ];
                                cols.writes_aux =
                                    memory_aux_cols_factory.make_write_aux_cols(digest_write);
                            }
                            cols.control.padding_occurred =
                                Val::<SC>::from_bool(has_padding_occurred);
                        }
                    }
                }
                // Fill in the invalid rows
                else {
                    block.par_chunks_mut(width).for_each(|row| {
                        let cols: &mut Sha256VmRoundCols<Val<SC>> = row.borrow_mut();
                        self.air.sha256_subair.generate_default_row(&mut cols.inner);
                    })
                }
            });

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        values[width..]
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .take(non_padded_height / SHA256_ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.air
                    .sha256_subair
                    .generate_missing_cells(chunk, width, SHA256VM_CONTROL_WIDTH);
            });

        AirProofInput::simple(air, RowMajorMatrix::new(values, width), vec![])
    }
}

impl<F: PrimeField32> ChipUsageGetter for Sha256VmChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.iter().fold(0, |acc, record| {
            acc + record.input_records.len() * SHA256_ROWS_PER_BLOCK
        })
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

/// This is the state information that a block will use to generate its trace
#[derive(Debug, Clone)]
struct Sha256State {
    hash: [u32; SHA256_HASH_WORDS],
    local_block_idx: usize,
    message_len: u32,
    block_input_message: [u8; SHA256_BLOCK_CELLS],
    block_padded_message: [u8; SHA256_BLOCK_CELLS],
    message_idx: usize,
    is_last_block: bool,
}

impl<F: PrimeField32> Sha256VmChip<F> {
    fn generate_state(
        prev_state: &Option<Sha256State>,
        block_input_message: [u8; SHA256_BLOCK_CELLS],
        message_idx: usize,
        message_len: u32,
        is_last_block: bool,
    ) -> Sha256State {
        let local_block_idx = if let Some(prev_state) = prev_state {
            prev_state.local_block_idx + 1
        } else {
            0
        };
        let has_padding_occurred = local_block_idx * SHA256_BLOCK_CELLS > message_len as usize;
        let message_left = if has_padding_occurred {
            0
        } else {
            message_len as usize - local_block_idx * SHA256_BLOCK_CELLS
        };

        let padded_message_bytes: [u8; SHA256_BLOCK_CELLS] = array::from_fn(|j| {
            if j < message_left {
                block_input_message[j]
            } else if j == message_left && !has_padding_occurred {
                1 << (RV32_CELL_BITS - 1)
            } else if !is_last_block || j < SHA256_BLOCK_CELLS - 4 {
                0u8
            } else {
                let shift_amount = (SHA256_BLOCK_CELLS - j - 1) * RV32_CELL_BITS;
                ((message_len * RV32_CELL_BITS as u32)
                    .checked_shr(shift_amount as u32)
                    .unwrap_or(0)
                    & ((1 << RV32_CELL_BITS) - 1)) as u8
            }
        });

        if let Some(prev_state) = prev_state {
            Sha256State {
                hash: Sha256Air::get_block_hash(&prev_state.hash, prev_state.block_padded_message),
                local_block_idx,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        } else {
            Sha256State {
                hash: SHA256_H,
                local_block_idx: 0,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        }
    }
}
