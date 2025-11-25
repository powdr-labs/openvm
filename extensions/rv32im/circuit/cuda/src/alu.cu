#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "rv32im/adapters/alu.cuh"
#include "rv32im/cores/alu.cuh"

#include <cstdio>

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BaseAluCoreRecord = BaseAluCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32BaseAluCore = BaseAluCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using Rv32BaseAluCoreCols = BaseAluCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv32BaseAluCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32BaseAluCoreCols<T> core;
};

struct Rv32BaseAluRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32BaseAluCoreRecord core;
};

__global__ void alu_tracegen(
    Fp *d_trace, // can be apc trace
    size_t height, // can be apc height
    DeviceBufferConstView<Rv32BaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    // Fp *d_apc_trace,
    uint32_t *subs, // same length as dummy width
    uint32_t *d_pre_opt_widths,
    uint32_t *d_post_opt_widths,
    uint32_t calls_per_apc_row, // 1 for non-apc
    size_t width // dummy width or apc width
    // uint32_t *apc_row_index, // dummy row mapping to apc row same length as d_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSliceNew row(d_trace + idx / calls_per_apc_row + d_post_opt_widths[idx % calls_per_apc_row] * height, height, d_post_opt_widths[idx % calls_per_apc_row], d_pre_opt_widths[idx % calls_per_apc_row], subs); // we need to slice to the correct APC row, but if non-APC it's dividing by 1 and therefore the same idx

    // Print a single APC row via a buffer to avoid noisy, interleaved output.
    if (calls_per_apc_row > 1) {
        uint32_t apc_row_idx = idx / calls_per_apc_row;
        uint32_t dummy_row_in_apc = idx % calls_per_apc_row;
        if (apc_row_idx == 0 && dummy_row_in_apc == 1) {
            RowPrintBuffer buffer;
            buffer.reset();
            buffer.append_literal("apc row ");
            buffer.append_uint(apc_row_idx);
            buffer.append_literal(" | ");
            buffer.append_literal("dummy row ");
            buffer.append_uint(dummy_row_in_apc);
            buffer.append_literal(" | ");
            for (uint32_t c = 0; c < 41; ++c) { // hardcoded APC width for now
                const Fp val = row.ptr[c * row.stride]; // RowSliceNew::operator[] multiplies by stride for you
                buffer.append_literal(" ");
                buffer.append_uint(static_cast<unsigned long long>(val));
            }
            buffer.append_literal("\n");
            buffer.flush();
        }
    }

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];
        // RowSlice apc_row(d_apc_trace + apc_row_index[idx], height);
        // auto const sub = subs[idx * width]; // offset the subs to the corresponding dummy row
        // uint32_t *sub = &subs[(idx % calls_per_apc_row) * width]; // dummy width

        Rv32BaseAluAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        if (idx == 1) {
            adapter.fill_trace_row_new(row, rec.adapter, subs, idx);
        }

        // Print a single APC row via a buffer to avoid noisy, interleaved output.
        if (calls_per_apc_row > 1) {
            uint32_t apc_row_idx = idx / calls_per_apc_row;
            uint32_t dummy_row_in_apc = idx % calls_per_apc_row;
            if (apc_row_idx == 0 && dummy_row_in_apc == 1) {
                RowPrintBuffer buffer;
                buffer.reset();
                buffer.append_literal("apc row ");
                buffer.append_uint(apc_row_idx);
                buffer.append_literal(" | ");
                buffer.append_literal("dummy row ");
                buffer.append_uint(dummy_row_in_apc);
                buffer.append_literal(" | ");
                for (uint32_t c = 0; c < 41; ++c) { // hardcoded APC width for now
                    const Fp val = row.ptr[c * row.stride]; // RowSliceNew::operator[] multiplies by stride for you
                    buffer.append_literal(" ");
                    buffer.append_uint(static_cast<unsigned long long>(val));
                }
                buffer.append_literal("\n");
                buffer.flush();
            }
        }

        Rv32BaseAluCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        
        if (idx == 1) {
            core.fill_trace_row_new(row.slice_from(COL_INDEX(Rv32BaseAluCols, core)), rec.core, subs);
        }

        // Print a single APC row via a buffer to avoid noisy, interleaved output.
        if (calls_per_apc_row > 1) {
            uint32_t apc_row_idx = idx / calls_per_apc_row;
            uint32_t dummy_row_in_apc = idx % calls_per_apc_row;
            if (apc_row_idx == 0 
                && dummy_row_in_apc == 1
            ) {
                RowPrintBuffer buffer;
                buffer.reset();
                buffer.append_literal("apc row ");
                buffer.append_uint(apc_row_idx);
                buffer.append_literal(" | ");
                buffer.append_literal("dummy row ");
                buffer.append_uint(dummy_row_in_apc);
                buffer.append_literal(" | ");
                for (uint32_t c = 0; c < 41; ++c) { // hardcoded APC width for now
                    const Fp val = row.ptr[c * row.stride]; // RowSliceNew::operator[] multiplies by stride for you
                    buffer.append_literal(" ");
                    buffer.append_uint(static_cast<unsigned long long>(val));
                }
                buffer.append_literal("\n");
                buffer.flush();
            }
        }
    } else {
        // TODO: use APC width if APC
        // this is now a hack because calls_per_apc_row can still be 1 even if we are in an APC
        if (calls_per_apc_row == 1) {
            row.fill_zero(0, sizeof(Rv32BaseAluCols<uint8_t>));
        } else {
            row.fill_zero(0, 41); // hardcoded APC width for now
        }
    }

    // Print a single APC row via a buffer to avoid noisy, interleaved output.
    if (calls_per_apc_row > 1) {
        uint32_t apc_row_idx = idx / calls_per_apc_row;
        uint32_t dummy_row_in_apc = idx % calls_per_apc_row;
        if (apc_row_idx == 0 && dummy_row_in_apc == 1) {
            RowPrintBuffer buffer;
            buffer.reset();
            buffer.append_literal("apc row ");
            buffer.append_uint(apc_row_idx);
            buffer.append_literal(" | ");
            buffer.append_literal("dummy row ");
            buffer.append_uint(dummy_row_in_apc);
            buffer.append_literal(" | ");
            for (uint32_t c = 0; c < 41; ++c) { // hardcoded APC width for now
                const Fp val = row.ptr[c * row.stride]; // RowSliceNew::operator[] multiplies by stride for you
                buffer.append_literal(" ");
                buffer.append_uint(static_cast<unsigned long long>(val));
            }
            buffer.append_literal("\n");
            buffer.flush();
        }
    }
}

extern "C" int _alu_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32BaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    // Fp *d_apc_trace,
    uint32_t *subs, // same length as dummy width
    uint32_t *d_pre_opt_widths,
    uint32_t *d_post_opt_widths,
    uint32_t calls_per_apc_row // 1 for non-apc
    // uint32_t *apc_row_index, // dummy row mapping to apc row same length as d_records
) {
    assert((height & (height - 1)) == 0);
    // assert(height >= d_records.len()); // might be false for apc case
    // assert(width == sizeof(Rv32BaseAluCols<uint8_t>)); // this is no longer true for APC
    auto [grid, block] = kernel_launch_params(height);
    alu_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits,
        // Fp *d_apc_trace,
        subs, // same length as dummy width
        d_pre_opt_widths,
        d_post_opt_widths,
        calls_per_apc_row, // 1 for non-apc
        width // dummy width or apc width
        // uint32_t *apc_row_index, // dummy row mapping to apc row same length as d_records
    );
    // print all rows here
    for (size_t i = 0; i < height; i++) {
        printf("Row %zu: ", i);
        for (size_t j = 0; j < width; j++) {
            Fp val = d_trace[j * height + i];
            printf("%llu ", static_cast<unsigned long long>(val));
        }
        printf("\n");
    }
    printf("\n");

    return CHECK_KERNEL();
}
