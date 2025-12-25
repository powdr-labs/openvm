#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/alu.cuh"
#include "rv32im/cores/shift.cuh"

#include <cstdio>

using namespace riscv;
using namespace program;

// Concrete type aliases for 32-bit
using Rv32ShiftCoreRecord = ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32ShiftCore = ShiftCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using Rv32ShiftCoreCols = ShiftCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct ShiftCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32ShiftCoreCols<T> core;
};

struct ShiftRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32ShiftCoreRecord core;
};

__global__ void rv32_shift_tracegen(
    Fp *d_trace, // can be apc trace
    size_t height,
    DeviceBufferConstView<ShiftRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_opt_widths,
    uint32_t *d_post_opt_offsets,
    size_t apc_width, // 0 for non-apc
    uint32_t calls_per_apc_row // 1 for non-apc
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // d_post_opt_offsets is always 0 for non APC case
    bool is_apc = apc_width != 0;
    RowSlice row(
        is_apc ? d_trace + idx / calls_per_apc_row + d_post_opt_offsets[idx % calls_per_apc_row] * height : d_trace + idx,
        height,
        is_apc ? d_post_opt_offsets[idx % calls_per_apc_row] : 0,
        is_apc ? sizeof(ShiftCols<uint8_t>) * (idx % calls_per_apc_row): 0, // this way we don't need to pass over d_pre_opt_offsets
        subs,
        is_apc
    ); // we need to slice to the correct APC row, but if non-APC it's dividing by 1 and therefore the same idx

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv32BaseAluAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv32ShiftCore core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftCols, core)), rec.core);
    } else {
        if (!is_apc) {
            // non-apc case
            row.fill_zero(0, sizeof(ShiftCols<uint8_t>));
        } else if (idx < height * calls_per_apc_row) {
            // apc case, but we need to limit idx to smaller than the # of dummy instruction runs
            // because `kernel_launch_params` rounds to the next MAX_THREADS number of runs
            // which can write beyond what we desire
            row.fill_zero_no_offset(0, d_opt_widths[idx % calls_per_apc_row]);
        }
    }
}

extern "C" int _rv32_shift_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_opt_widths,
    uint32_t *d_post_opt_offsets,
    size_t apc_height, // 0 for non-apc
    size_t apc_width, // 0 for non-apc
    uint32_t calls_per_apc_row // 1 for non-apc
) {
    assert((height & (height - 1)) == 0);
    assert((apc_height & (apc_height - 1)) == 0);
    assert(height >= d_records.len());
    bool is_apc = apc_width != 0;
    if (!is_apc) { // only check for non-apc
        assert(width == sizeof(ShiftCols<uint8_t>));
    }
    size_t threads = is_apc ? (apc_height * calls_per_apc_row) : height;
    auto [grid, block] = kernel_launch_params(threads);
    rv32_shift_tracegen<<<grid, block>>>(
        d_trace,
        is_apc ? apc_height : height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits,
        subs,
        d_opt_widths,
        d_post_opt_offsets,
        apc_width, // 0 for non-apc
        calls_per_apc_row // 1 for non-apc
    );

    return CHECK_KERNEL();
}