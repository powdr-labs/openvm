#pragma once

#include "offline_checker.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

struct MemoryAuxColsFactory {
    VariableRangeChecker range_checker;
    uint32_t timestamp_max_bits;

    __device__ MemoryAuxColsFactory(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : range_checker(range_checker), timestamp_max_bits(timestamp_max_bits) {}

    __device__ void fill(RowSlice row, uint32_t prev_timestamp, uint32_t timestamp) {
        AssertLessThan::generate_subrow(
            range_checker,
            timestamp_max_bits,
            prev_timestamp,
            timestamp,
            AUX_LEN,
            row.slice_from(COL_INDEX(MemoryBaseAuxCols, timestamp_lt_aux))
        );
        COL_WRITE_VALUE(row, MemoryBaseAuxCols, prev_timestamp, prev_timestamp);
    }

    __device__ void fill_new(RowSlice row, uint32_t prev_timestamp, uint32_t timestamp, uint32_t *sub) {
        AssertLessThan::generate_subrow_new(
            range_checker,
            timestamp_max_bits,
            prev_timestamp,
            timestamp,
            AUX_LEN,
            row.slice_from(COL_INDEX(MemoryBaseAuxCols, timestamp_lt_aux)) - number_of_gaps_in(sub, sizeof(MemoryBaseAuxCols<uint8_t>)),
            sub
        );
        COL_WRITE_VALUE_NEW(row, MemoryBaseAuxCols, prev_timestamp, prev_timestamp, sub);
    }

    __device__ void fill_zero(RowSlice row) {
        row.fill_zero(0, sizeof(MemoryBaseAuxCols<uint8_t>));
    }
};
