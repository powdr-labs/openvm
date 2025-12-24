#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/alu.cuh"
#include "rv32im/cores/less_than.cuh"

using namespace riscv;
using namespace program;

// Concrete type aliases for 32-bit
using Rv32LessThanCoreRecord = LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32LessThanCore = LessThanCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using Rv32LessThanCoreCols = LessThanCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct LessThanCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32LessThanCoreRecord core;
};

__global__ void rv32_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<LessThanRecord> d_records,
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
        is_apc ? sizeof(LessThanCols<uint8_t>) * (idx % calls_per_apc_row) : 0,
        subs,
        is_apc
    );

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv32BaseAluAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row_new(row, rec.adapter);

        Rv32LessThanCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row_new(row.slice_from(COL_INDEX(LessThanCols, core)), rec.core);
    } else {
        if (!is_apc) {
            row.fill_zero(0, sizeof(LessThanCols<uint8_t>));
        } else if (idx < height * calls_per_apc_row) {
            row.fill_zero(0, d_opt_widths[idx % calls_per_apc_row]);
        }
    }
}

extern "C" int _rv32_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanRecord> d_records,
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
        assert(width == sizeof(LessThanCols<uint8_t>));
    }
    size_t threads = is_apc ? (apc_height * calls_per_apc_row) : height;
    auto [grid, block] = kernel_launch_params(threads);

    rv32_less_than_tracegen<<<grid, block>>>(
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