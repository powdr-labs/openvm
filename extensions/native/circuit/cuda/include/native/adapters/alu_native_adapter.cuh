#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace native;

template <typename T> struct AluNativeAdapterCols {
    ExecutionState<T> from_state;
    T a_pointer;
    T b_pointer;
    T c_pointer;
    T e_as;
    T f_as;
    MemoryReadOrImmediateAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, 1> write_aux;
};

struct AluNativeAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t a_ptr;
    uint32_t b;
    uint32_t c;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteAuxRecord<Fp, 1> write_aux;
};

struct AluNativeAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ AluNativeAdapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, AluNativeAdapterRecord const &rec) {
        COL_WRITE_VALUE(row, AluNativeAdapterCols, from_state.pc, Fp(rec.from_pc));
        COL_WRITE_VALUE(row, AluNativeAdapterCols, from_state.timestamp, Fp(rec.from_timestamp));
        COL_WRITE_VALUE(row, AluNativeAdapterCols, a_pointer, Fp::fromRaw(rec.a_ptr));
        COL_WRITE_VALUE(row, AluNativeAdapterCols, b_pointer, Fp::fromRaw(rec.b));
        COL_WRITE_VALUE(row, AluNativeAdapterCols, c_pointer, Fp::fromRaw(rec.c));

        // Fill read auxiliary columns for two operands (b and c)
        const Fp native_as = Fp(AS_NATIVE);
        for (int i = 0; i < 2; i++) {
            const uint32_t prev_timestamp = rec.reads_aux[i].prev_timestamp;
            const uint32_t current_timestamp = rec.from_timestamp + i;
            RowSlice aux_slice = row.slice_from(COL_INDEX(AluNativeAdapterCols, reads_aux[i]));

            if (prev_timestamp == UINT32_MAX) {
                // Immediate
                mem_helper.fill(aux_slice, 0, current_timestamp);
                COL_WRITE_VALUE(row, AluNativeAdapterCols, reads_aux[i].is_zero_aux, Fp::zero());
                COL_WRITE_VALUE(row, AluNativeAdapterCols, reads_aux[i].is_immediate, Fp::one());
                if (i == 0) {
                    COL_WRITE_VALUE(row, AluNativeAdapterCols, e_as, Fp(AS_IMMEDIATE));
                } else {
                    COL_WRITE_VALUE(row, AluNativeAdapterCols, f_as, Fp(AS_IMMEDIATE));
                }
            } else {
                // Memory
                mem_helper.fill(aux_slice, prev_timestamp, current_timestamp);
                COL_WRITE_VALUE(
                    row, AluNativeAdapterCols, reads_aux[i].is_zero_aux, inv(native_as)
                );
                COL_WRITE_VALUE(row, AluNativeAdapterCols, reads_aux[i].is_immediate, Fp::zero());
                if (i == 0) {
                    COL_WRITE_VALUE(row, AluNativeAdapterCols, e_as, native_as);
                } else {
                    COL_WRITE_VALUE(row, AluNativeAdapterCols, f_as, native_as);
                }
            }
        }

        COL_WRITE_ARRAY(row, AluNativeAdapterCols, write_aux.prev_data, rec.write_aux.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(AluNativeAdapterCols, write_aux)),
            rec.write_aux.prev_timestamp,
            rec.from_timestamp + 2
        );
    }

    __device__ void fill_trace_row_new(RowSlice row, AluNativeAdapterRecord const &rec, uint32_t *sub) {
        // if !apc_row.is_null() {
        //     COL_WRITE_VALUE_APC(apc_row, AluNativeAdapterCols, from_state.timestamp, Fp(rec.from_timestamp), sub, offset);
        //     COL_WRITE_VALUE_APC(row, AluNativeAdapterCols, a_pointer, Fp::fromRaw(rec.a_ptr), sub, offset);
        //     COL_WRITE_VALUE_APC(row, AluNativeAdapterCols, b_pointer, Fp::fromRaw(rec.b), sub, offset);
        //     COL_WRITE_VALUE_APC(row, AluNativeAdapterCols, c_pointer, Fp::fromRaw(rec.c), sub, offset);
            
        //     // TODO: adapt the rest similar to above
        // } else {
        COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, from_state.timestamp, Fp(rec.from_timestamp), sub);
        COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, a_pointer, Fp::fromRaw(rec.a_ptr), sub);
        COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, b_pointer, Fp::fromRaw(rec.b), sub);
        COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, c_pointer, Fp::fromRaw(rec.c), sub);

        // Fill read auxiliary columns for two operands (b and c)
        const Fp native_as = Fp(AS_NATIVE);
        for (int i = 0; i < 2; i++) {
            const uint32_t prev_timestamp = rec.reads_aux[i].prev_timestamp;
            const uint32_t current_timestamp = rec.from_timestamp + i;
            RowSlice aux_slice = row.slice_from(COL_INDEX(AluNativeAdapterCols, reads_aux[i]) - number_of_gaps_in(sub, sizeof(AluNativeAdapterCols<uint8_t>)));

            if (prev_timestamp == UINT32_MAX) {
                // Immediate
                mem_helper.fill_new(aux_slice, 0, current_timestamp, sub);
                COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, reads_aux[i].is_zero_aux, Fp::zero(), sub);
                COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, reads_aux[i].is_immediate, Fp::one(), sub);
                if (i == 0) {
                    COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, e_as, Fp(AS_IMMEDIATE), sub);
                } else {
                    COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, f_as, Fp(AS_IMMEDIATE), sub);
                }
            } else {
                // Memory
                mem_helper.fill_new(aux_slice, prev_timestamp, current_timestamp, sub);
                COL_WRITE_VALUE_NEW(
                    row, AluNativeAdapterCols, reads_aux[i].is_zero_aux, inv(native_as), sub
                );
                COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, reads_aux[i].is_immediate, Fp::zero(), sub);
                if (i == 0) {
                    COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, e_as, native_as, sub);
                } else {
                    COL_WRITE_VALUE_NEW(row, AluNativeAdapterCols, f_as, native_as, sub);
                }
            }
        }

        COL_WRITE_ARRAY_NEW(row, AluNativeAdapterCols, write_aux.prev_data, rec.write_aux.prev_data, sub);
        mem_helper.fill_new(
            row.slice_from(COL_INDEX(AluNativeAdapterCols, write_aux)),
            rec.write_aux.prev_timestamp,
            rec.from_timestamp + 2,
            sub
        );
        // }
        
    }
};