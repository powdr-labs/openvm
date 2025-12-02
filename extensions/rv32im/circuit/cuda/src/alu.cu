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
    size_t height,
    DeviceBufferConstView<Rv32BaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_pre_opt_widths,
    uint32_t *d_post_opt_widths,
    size_t apc_width, // 0 for non-apc
    uint32_t calls_per_apc_row // 1 for non-apc
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // d_post_opt_widths is always 0 for non APC case
    bool is_apc = apc_width != 0;
    RowSliceNew row(
        d_trace + idx / calls_per_apc_row + d_post_opt_widths[idx % calls_per_apc_row] * height, 
        height, 
        d_post_opt_widths[idx % calls_per_apc_row], 
        d_pre_opt_widths[idx % calls_per_apc_row], 
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
        adapter.fill_trace_row_new(row, rec.adapter);

        Rv32BaseAluCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row_new(row.slice_from(COL_INDEX(Rv32BaseAluCols, core)), rec.core);
    } else {
        if (apc_width == 0) {
            // non-apc case
            row.fill_zero(0, sizeof(Rv32BaseAluCols<uint8_t>));
        } else if (idx % calls_per_apc_row == 0) {
            // apc case: only fill if it's the first original instruction
            // because otherwise the APC row doesn't start with 0 offset
            row.fill_zero(0, apc_width);
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
    uint32_t *subs,
    uint32_t *d_pre_opt_widths,
    uint32_t *d_post_opt_widths,
    size_t apc_height, // 0 for non-apc
    size_t apc_width, // 0 for non-apc
    uint32_t calls_per_apc_row // 1 for non-apc
) {
    assert((height & (height - 1)) == 0);
    assert((apc_height & (apc_height - 1)) == 0);
    assert(height >= d_records.len());
    if (apc_width == 0) { // only check for non-apc
        assert(width == sizeof(Rv32BaseAluCols<uint8_t>));
    }
    auto [grid, block] = kernel_launch_params(height);
    alu_tracegen<<<grid, block>>>(
        d_trace,
        apc_width == 0 ? height : apc_height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits,
        subs,
        d_pre_opt_widths,
        d_post_opt_widths,
        apc_width, // 0 for non-apc
        calls_per_apc_row // 1 for non-apc
    );

    return CHECK_KERNEL();
}
