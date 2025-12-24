#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "rv32im/cores/blt.cuh"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BranchLessThanCoreRecord = BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32BranchLessThanCore = BranchLessThanCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32BranchLessThanCoreCols = BranchLessThanCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct BranchLessThanCols {
    Rv32BranchAdapterCols<T> adapter;
    Rv32BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv32BranchAdapterRecord adapter;
    Rv32BranchLessThanCoreRecord core;
};

__global__ void blt_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<BranchLessThanRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_opt_widths,
    uint32_t *d_post_opt_offsets,
    size_t apc_width,
    uint32_t calls_per_apc_row
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_apc = apc_width != 0;
    RowSliceNew row(
        is_apc ? d_trace + idx / calls_per_apc_row + d_post_opt_offsets[idx % calls_per_apc_row] * height : d_trace + idx,
        height,
        is_apc ? d_post_opt_offsets[idx % calls_per_apc_row] : 0,
        is_apc ? sizeof(BranchLessThanCols<uint8_t>) * (idx % calls_per_apc_row) : 0,
        subs,
        is_apc
    );

    if (idx < d_records.len()) {
        auto const &full_record = d_records[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits);
        adapter.fill_trace_row_new(row, full_record.adapter);

        Rv32BranchLessThanCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row_new(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full_record.core);
    } else {
        if (!is_apc) {
            row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));
        } else if (idx < height * calls_per_apc_row) {
            row.fill_zero(0, d_opt_widths[idx % calls_per_apc_row]);
        }
    }
}

extern "C" int _blt_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BranchLessThanRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_opt_widths,
    uint32_t *d_post_opt_offsets,
    size_t apc_height,
    size_t apc_width,
    uint32_t calls_per_apc_row
) {
    assert((height & (height - 1)) == 0);
    assert((apc_height & (apc_height - 1)) == 0);
    assert(height >= d_records.len());
    bool is_apc = apc_width != 0;
    if (!is_apc) {
        assert(width == sizeof(BranchLessThanCols<uint8_t>));
    }
    size_t threads = is_apc ? (apc_height * calls_per_apc_row) : height;
    auto [grid, block] = kernel_launch_params(threads);

    blt_tracegen<<<grid, block>>>(
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
        apc_width,
        calls_per_apc_row
    );
    return CHECK_KERNEL();
}
