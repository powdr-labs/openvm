#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "rv32im/cores/beq.cuh"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BranchEqualCore = BranchEqualCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32BranchEqualCoreCols = BranchEqualCoreCols<T, RV32_REGISTER_NUM_LIMBS>;
using Rv32BranchEqualCoreRecord = BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct BranchEqualCols {
    Rv32BranchAdapterCols<T> adapter;
    Rv32BranchEqualCoreCols<T> core;
};

struct BranchEqualRecord {
    Rv32BranchAdapterRecord adapter;
    Rv32BranchEqualCoreRecord core;
};

__global__ void beq_tracegen(
    Fp *d_trace, // can be apc trace
    size_t height,
    DeviceBufferConstView<BranchEqualRecord> d_records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    uint32_t *subs,
    uint32_t *d_opt_widths,
    uint32_t *d_post_opt_offsets,
    size_t apc_width, // 0 for non-apc
    uint32_t calls_per_apc_row // 1 for non-apc
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_apc = apc_width != 0;
    RowSliceNew row(
        is_apc ? d_trace + idx / calls_per_apc_row + d_post_opt_offsets[idx % calls_per_apc_row] * height : d_trace + idx,
        height,
        is_apc ? d_post_opt_offsets[idx % calls_per_apc_row] : 0,
        is_apc ? sizeof(BranchEqualCols<uint8_t>) * (idx % calls_per_apc_row) : 0,
        subs,
        is_apc
    );

    if (idx < d_records.len()) {
        auto const &full = d_records[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row_new(row, full.adapter);

        Rv32BranchEqualCore core;
        core.fill_trace_row_new(row.slice_from(COL_INDEX(BranchEqualCols, core)), full.core);
    } else {
        if (!is_apc) {
            // non-apc case
            row.fill_zero(0, sizeof(BranchEqualCols<uint8_t>));
        } else if (idx < height * calls_per_apc_row) {
            // apc case, but we need to limit idx to smaller than the # of dummy instruction runs
            row.fill_zero(0, d_opt_widths[idx % calls_per_apc_row]);
        }
    }
}

extern "C" int _beq_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BranchEqualRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
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
    bool is_apc = apc_width != 0;
    if (!is_apc) { // only check for non-apc
        assert(width == sizeof(BranchEqualCols<uint8_t>));
    }
    size_t threads = is_apc ? (apc_height * calls_per_apc_row) : height;
    auto [grid, block] = kernel_launch_params(threads);

    beq_tracegen<<<grid, block>>>(
        d_trace,
        is_apc ? apc_height : height,
        d_records,
        d_rc,
        rc_bins,
        timestamp_max_bits,
        subs,
        d_opt_widths,
        d_post_opt_offsets,
        apc_width, // 0 for non-apc
        calls_per_apc_row // 1 for non-apc
    );
    return CHECK_KERNEL();
}
